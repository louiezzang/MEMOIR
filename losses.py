"""MEMOIR loss functions: L = L_rec + λ1 * L_evo + λ2 * L_consistency + λ3 * L_extrap."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MEMOIRLoss(nn.Module):
    """Combined loss for MEMOIR training."""

    def __init__(
        self,
        lambda_evo: float = 0.5,
        lambda_consistency: float = 0.3,
        lambda_extrap: float = 0.3,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_evo = lambda_evo
        self.lambda_consistency = lambda_consistency
        self.lambda_extrap = lambda_extrap
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def recommendation_loss(
        self,
        user_embeds: torch.Tensor,
        pos_item_embeds: torch.Tensor | None = None,
        neg_item_embeds: torch.Tensor | None = None,
        catalog_item_embeds: torch.Tensor | None = None,
        target_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recommendation loss.

        If catalog_item_embeds/target_idx are given, scores the user against the
        *entire* item catalog and computes softmax cross-entropy against the true
        item's index — the same training signal every baseline in this repo uses
        (SASRec, UniSRec, etc. all score against the full catalog via
        get_scores() + F.cross_entropy). This replaces the much weaker in-batch
        InfoNCE fallback below, which only ever sees batch_size-1 negatives.

        Args:
            user_embeds: [B, D]
            pos_item_embeds: [B, D], required for the in-batch fallback
            neg_item_embeds: [B, K, D] or None (in-batch fallback only)
            catalog_item_embeds: [N, D], full catalog of item embeddings
            target_idx: [B], index of each sample's true item within the catalog
        """
        user = F.normalize(user_embeds, dim=-1)

        if catalog_item_embeds is not None and target_idx is not None:
            catalog = F.normalize(catalog_item_embeds, dim=-1)
            logits = torch.matmul(user, catalog.T) / self.temperature  # [B, N]
            # Full-catalog softmax is a ~245k-way classification — without label
            # smoothing the model can drive the true class's logit arbitrarily high
            # relative to all others, which overfits fast on a small item pool
            # sample per batch. Smoothing caps how confident it's rewarded for being.
            return F.cross_entropy(logits, target_idx, label_smoothing=self.label_smoothing)

        pos = F.normalize(pos_item_embeds, dim=-1)
        pos_score = (user * pos).sum(dim=-1) / self.temperature  # [B]

        if neg_item_embeds is not None:
            neg = F.normalize(neg_item_embeds, dim=-1)
            neg_scores = torch.bmm(neg, user.unsqueeze(-1)).squeeze(-1) / self.temperature  # [B, K]
        else:
            neg_scores = torch.matmul(user, pos.T) / self.temperature  # [B, B]

        logits = torch.cat([pos_score.unsqueeze(-1), neg_scores], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(self, model_output: dict) -> dict[str, torch.Tensor]:
        """Compute total loss from model output.

        Args:
            model_output: dict from MEMOIRModel.forward()

        Returns:
            dict with 'total', 'rec', 'evo', 'consistency', 'extrap' loss values
        """
        rec_loss = self.recommendation_loss(
            model_output["user_memory"],
            pos_item_embeds=model_output["item_embeds"],
            catalog_item_embeds=model_output.get("catalog_item_embeds"),
            target_idx=model_output.get("target_catalog_idx"),
        )

        evo_loss = model_output["evo_loss_dict"]["loss"]
        consistency_loss = model_output["consistency_loss"]
        extrap_loss = model_output["extrap_loss"]

        total = (
            rec_loss
            + self.lambda_evo * evo_loss
            + self.lambda_consistency * consistency_loss
            + self.lambda_extrap * extrap_loss
        )

        return {
            "total": total,
            "rec": rec_loss,
            "evo": evo_loss,
            "evo_smoothness": model_output["evo_loss_dict"].get(
                "smoothness_loss", torch.tensor(0.0)
            ),
            "evo_direction": model_output["evo_loss_dict"].get(
                "direction_loss", torch.tensor(0.0)
            ),
            "consistency": consistency_loss,
            "extrap": extrap_loss,
        }
