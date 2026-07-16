"""MEMOIR: Contrastive Behavioral Memory for Preference Evolution in Recommendation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_encoder import LLMMemoryEncoder, EvolutionAwareAggregator
from .item_encoder import ItemEncoder, RandomItemEncoder
from .evolution_contrastive import EvolutionContrastive
from .trajectory import TrajectoryPredictor, ExtrapolationLoss
from .retriever import ANNRetriever
from .reasoning import ReasoningLayer


class MEMOIRModel(nn.Module):
    """Full MEMOIR model combining all components."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        embed_dim = model_cfg["embedding_dim"]

        self.memory_encoder = LLMMemoryEncoder(
            model_name=model_cfg["llm_name"],
            output_dim=embed_dim,
            load_in_4bit=model_cfg["llm_load_in_4bit"],
            freeze_llm=model_cfg.get("freeze_llm", False),
        )

        item_enc_cfg = model_cfg["item_encoder"]
        self.use_random_item_encoder = item_enc_cfg.get("type") == "random"
        if self.use_random_item_encoder:
            self.item_encoder = RandomItemEncoder(
                num_items=item_enc_cfg["num_items"],
                output_dim=embed_dim,
            )
        else:
            self.item_encoder = ItemEncoder(
                pretrained=item_enc_cfg["pretrained"],
                output_dim=embed_dim,
            )

        self.evolution_contrastive = EvolutionContrastive(
            temperature=model_cfg["temperature"],
            alpha_direction=config["training"]["alpha_direction"],
            margin=config["training"]["margin"],
        )

        self.trajectory_predictor = TrajectoryPredictor(
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            num_layers=model_cfg.get("trajectory_gru_layers", 1),
            dropout=model_cfg["dropout"],
        )

        self.evo_aggregator = EvolutionAwareAggregator(embed_dim=embed_dim)

        self.extrap_loss_fn = ExtrapolationLoss(
            temperature=model_cfg["temperature"],
        )

        self.retriever = ANNRetriever(top_k=model_cfg["retrieval_top_k"])

        self.reasoning = ReasoningLayer(
            embed_dim=embed_dim,
            dropout=model_cfg["dropout"],
        )

        self.embed_dim = embed_dim

    def _build_aggregate(
        self,
        enc_output: dict,
        window_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Build evolution-aware aggregate memory from encoder output.

        Args:
            enc_output: dict from LLMMemoryEncoder.forward()
            window_masks: [B, W]

        Returns:
            aggregate_memory: [B, D]
        """
        traj = self.trajectory_predictor(enc_output["memory_grid"], window_masks)
        return self.evo_aggregator(
            enc_output["base_aggregate"],
            traj["direction"],
            traj["predicted_next"],
        )

    def forward(self, batch: dict) -> dict:
        """Full forward pass for training.

        Returns dict with:
            - rec_scores: [B] recommendation scores
            - user_memory: [B, D] evolution-aware aggregate memory
            - item_embeds: [B, D]
            - evo_loss_dict: contrastive loss components
            - consistency_loss: scalar
            - extrap_loss: scalar (trajectory extrapolation)
        """
        # 1. Encode user behavioral memory
        enc_output = self.memory_encoder(
            batch["window_texts"],
            batch["window_masks"],
        )
        aggregate_memory = self._build_aggregate(enc_output, batch["window_masks"])

        # 2. Encode target items
        if self.use_random_item_encoder:
            item_embeds = self.item_encoder(batch["target_item_indices"])  # [B, D]
        else:
            item_embeds = self.item_encoder(batch["target_titles"])  # [B, D]

        # 3. Recommendation score
        rec_scores = (F.normalize(aggregate_memory, dim=-1) *
                      F.normalize(item_embeds, dim=-1)).sum(dim=-1)  # [B]

        # 4. Evolution contrastive loss
        evo_loss_dict = {"loss": torch.tensor(0.0, device=rec_scores.device),
                         "smoothness_loss": torch.tensor(0.0, device=rec_scores.device),
                         "direction_loss": torch.tensor(0.0, device=rec_scores.device)}
        if batch.get("pos_anchors") and len(batch["pos_anchors"]) > 0:
            anchor_embeds = self.memory_encoder.encode_text(batch["pos_anchors"])
            positive_embeds = self.memory_encoder.encode_text(batch["pos_positives"])

            prev_anchor_embeds = None
            if batch.get("prev_anchors") and len(batch["prev_anchors"]) > 0:
                prev_anchor_embeds = self.memory_encoder.encode_text(batch["prev_anchors"])

            flat_negs = [t for neg_list in batch["neg_texts"] for t in neg_list]
            if flat_negs:
                neg_embeds = self.memory_encoder.encode_text(flat_negs)
                K = len(batch["neg_texts"][0]) if batch["neg_texts"] else 0
                N = len(batch["pos_anchors"])
                if K > 0:
                    neg_embeds = neg_embeds[:N * K].view(N, K, -1)
                    evo_loss_dict = self.evolution_contrastive(
                        anchor_embeds, positive_embeds, neg_embeds,
                        prev_anchor_embeds=prev_anchor_embeds,
                    )

        # 5. Behavioral consistency loss
        consistency_loss = torch.tensor(0.0, device=rec_scores.device)
        if item_embeds.shape[0] == aggregate_memory.shape[0]:
            consistency_loss = F.mse_loss(
                F.normalize(aggregate_memory, dim=-1),
                F.normalize(item_embeds, dim=-1).detach(),
            )

        # 6. Trajectory extrapolation loss (hold-out-last-window supervision)
        extrap_loss = torch.tensor(0.0, device=rec_scores.device)
        memory_grid = enc_output["memory_grid"]
        masks = batch["window_masks"]
        lengths = masks.sum(dim=1).long()
        has_holdout = (lengths >= 3)

        if has_holdout.any():
            ho_idx = has_holdout.nonzero(as_tuple=True)[0]
            ho_grid = memory_grid[ho_idx]
            ho_lengths = lengths[ho_idx]

            ho_batch_idx = torch.arange(ho_grid.shape[0], device=ho_grid.device)
            actual_last = ho_grid[ho_batch_idx, ho_lengths - 1]  # [N_ho, D]

            trimmed_masks = masks[ho_idx].clone()
            for i, l in enumerate(ho_lengths):
                trimmed_masks[i, l - 1] = False

            traj_ho = self.trajectory_predictor(ho_grid, trimmed_masks)
            extrap_loss = self.extrap_loss_fn(traj_ho["predicted_next"], actual_last)

        return {
            "rec_scores": rec_scores,
            "user_memory": aggregate_memory,
            "item_embeds": item_embeds,
            "evo_loss_dict": evo_loss_dict,
            "consistency_loss": consistency_loss,
            "extrap_loss": extrap_loss,
        }

    def retrieve_and_rank(
        self,
        window_texts: list[list[str]],
        window_masks: torch.Tensor,
        all_item_titles: list[str],
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference: retrieve candidates and re-rank with reasoning layer."""
        with torch.no_grad():
            enc_output = self.memory_encoder(window_texts, window_masks)
            aggregate_memory = self._build_aggregate(enc_output, window_masks)
            all_item_embeds = self.item_encoder(all_item_titles)

            _, retrieval_indices = self.retriever(aggregate_memory, all_item_embeds)

            B, K = retrieval_indices.shape
            candidate_embeds = all_item_embeds[retrieval_indices.view(-1)].view(B, K, -1)

            max_w = window_masks.shape[1]
            D = aggregate_memory.shape[-1]
            memory_tensor = torch.zeros(B, max_w, D, device=aggregate_memory.device)
            for i, pw in enumerate(enc_output["per_window"]):
                w = pw.shape[0]
                memory_tensor[i, :w] = pw

            reasoning_scores = self.reasoning(memory_tensor, candidate_embeds, window_masks)

            final_scores, final_idx = torch.topk(reasoning_scores, k=min(top_k, K), dim=-1)
            final_indices = retrieval_indices[torch.arange(retrieval_indices.shape[0], device=retrieval_indices.device).unsqueeze(1), final_idx]

        return final_scores, final_indices
