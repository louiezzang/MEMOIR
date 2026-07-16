"""CLLMRec: Contrastive Learning with LLMs-based View Augmentation for Sequential Rec (IJCAI 2025).

Reimplemented from paper since official code is not publicly available.
Reference: Lu et al., "CLLMRec: Contrastive Learning with LLMs-based View
Augmentation for Sequential Recommendation", IJCAI 2025.

Key idea: Use BERT hidden states + self-attention weights to compute importance
scores per position, then apply bidirectional threshold pruning to create
semantically meaningful positive/negative augmented views.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


# ---------------------------------------------------------------------------
# LRURec: Linear Recurrent Unit encoder (Yue et al., WSDM 2024)
# Used as the base sequential recommender in CLLMRec
# ---------------------------------------------------------------------------

class LRULayer(nn.Module):
    """Single Linear Recurrent Unit layer."""

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable complex-valued recurrence parameters
        nu_log = torch.randn(embed_dim) * 0.01
        theta_log = torch.linspace(0, math.pi, embed_dim)
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)

        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D]"""
        B, L, D = x.shape

        nu = torch.sigmoid(self.nu_log)
        theta = self.theta_log
        # Complex eigenvalues: lambda = nu * exp(i*theta)
        lambda_real = nu * torch.cos(theta)  # [D]
        lambda_imag = nu * torch.sin(theta)  # [D]

        u = self.input_proj(x)  # [B, L, D]

        # Parallel scan (simplified linear recurrence)
        h_real = torch.zeros(B, D, device=x.device)
        h_imag = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(L):
            h_real_new = lambda_real * h_real - lambda_imag * h_imag + u[:, t]
            h_imag_new = lambda_imag * h_real + lambda_real * h_imag
            h_real, h_imag = h_real_new, h_imag_new
            outputs.append(h_real)

        out = torch.stack(outputs, dim=1)  # [B, L, D]
        out = self.output_proj(self.dropout(out))
        return self.norm(out + x)


class LRURecEncoder(nn.Module):
    """LRURec: sequential encoder using stacked LRU layers."""

    def __init__(self, embed_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LRULayer(embed_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# View Augmentation Module: BERT-based importance scoring + pruning
# ---------------------------------------------------------------------------

class ViewAugmentationModule(nn.Module):
    """Uses BERT hidden states and self-attention weights to compute
    per-position importance scores, then performs bidirectional pruning."""

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        bert_hidden: int = 256,
        bert_heads: int = 4,
        bert_layers: int = 2,
        lambda_a: float = 0.5,
        prune_ratio: float = 0.25,
    ):
        super().__init__()
        self.lambda_a = lambda_a
        self.prune_ratio = prune_ratio

        # Lightweight BERT for importance scoring
        config = BertConfig(
            vocab_size=num_items + 2,
            hidden_size=bert_hidden,
            num_hidden_layers=bert_layers,
            num_attention_heads=bert_heads,
            intermediate_size=bert_hidden * 4,
            max_position_embeddings=512,
        )
        self.bert = BertModel(config)

        # Project item embeddings to BERT hidden size if different
        self.input_proj = nn.Linear(embed_dim, bert_hidden) if embed_dim != bert_hidden else nn.Identity()

    def compute_importance(
        self,
        item_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance score for each position (Eq. 3 in paper).

        gamma_i = (1/d) * sum(|h_ij|) + lambda_a * (1/H) * sum(A_ii^(h))

        Args:
            item_embeds: [B, L, D] item embeddings
            attention_mask: [B, L] binary mask

        Returns:
            importance: [B, L] per-position importance scores
        """
        projected = self.input_proj(item_embeds)

        outputs = self.bert(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        hidden_states = outputs.last_hidden_state  # [B, L, bert_hidden]
        attentions = outputs.attentions  # list of [B, H, L, L]

        # Term 1: mean absolute hidden state magnitude
        hidden_importance = hidden_states.abs().mean(dim=-1)  # [B, L]

        # Term 2: mean diagonal self-attention across heads and layers
        attn_importance = torch.zeros_like(hidden_importance)
        for attn in attentions:
            # attn: [B, H, L, L] → diagonal: [B, H, L]
            diag = torch.diagonal(attn, dim1=-2, dim2=-1)  # [B, H, L]
            attn_importance += diag.mean(dim=1)  # [B, L]
        attn_importance /= len(attentions)

        importance = hidden_importance + self.lambda_a * attn_importance
        importance = importance * attention_mask.float()

        return importance

    def prune(
        self,
        item_seq: torch.Tensor,
        importance: torch.Tensor,
        seq_len: torch.Tensor,
        prune_high: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional threshold pruning.

        prune_high=False: remove LOW importance positions → positive view
        prune_high=True:  remove HIGH importance positions → negative view
        """
        B, L = item_seq.shape
        num_prune = (seq_len.float() * self.prune_ratio).long().clamp(min=1)

        pruned_seqs = []
        pruned_lens = []

        for i in range(B):
            valid_len = seq_len[i].item()
            scores = importance[i, :valid_len]
            k = min(num_prune[i].item(), valid_len - 1)

            if prune_high:
                _, indices_to_remove = torch.topk(scores, k)
            else:
                _, indices_to_remove = torch.topk(scores, k, largest=False)

            mask = torch.ones(valid_len, dtype=torch.bool, device=item_seq.device)
            mask[indices_to_remove] = False

            kept = item_seq[i, :valid_len][mask]
            new_len = kept.shape[0]
            padded = F.pad(kept, (0, L - new_len), value=0)
            pruned_seqs.append(padded)
            pruned_lens.append(new_len)

        return torch.stack(pruned_seqs), torch.tensor(pruned_lens, device=item_seq.device)


# ---------------------------------------------------------------------------
# Contrastive Learning Module: Weighted InfoNCE
# ---------------------------------------------------------------------------

class WeightedInfoNCE(nn.Module):
    """Weighted InfoNCE loss (Eq. 7 in paper)."""

    def __init__(self, temperature: float = 0.1, neg_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.neg_weight = neg_weight

    def forward(
        self,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pos_embeds: [B, D] embeddings from positive (low-pruned) view
            neg_embeds: [B, D] embeddings from negative (high-pruned) view
        """
        pos = F.normalize(pos_embeds, dim=-1)
        neg = F.normalize(neg_embeds, dim=-1)

        B = pos.shape[0]

        # Positive: sim(pos_i, neg_i) -- same user's two views
        pos_sim = (pos * neg).sum(dim=-1) / self.temperature  # [B]

        # Negatives: sim(pos_i, neg_j) for j != i
        neg_sim = torch.matmul(pos, neg.T) / self.temperature  # [B, B]

        # Apply weight to negative samples
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=pos.device)
        weighted_neg = neg_sim * self.neg_weight * neg_mask.float()

        # InfoNCE
        numerator = torch.exp(pos_sim)
        denominator = numerator + (torch.exp(weighted_neg) * neg_mask.float()).sum(dim=-1)

        loss = -torch.log(numerator / denominator.clamp(min=1e-9))
        return loss.mean()


# ---------------------------------------------------------------------------
# CLLMRec: Full model
# ---------------------------------------------------------------------------

class CLLMRec(nn.Module):
    """CLLMRec: Contrastive Learning with LLM-based View Augmentation.

    Architecture:
    1. View Augmentation: BERT importance scoring → bidirectional pruning
    2. Contrastive Learning: SASRec encoder + Weighted InfoNCE
    3. Sequential Recommendation: LRURec encoder + CE loss
    4. Joint training with 3-stage strategy
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        cl_temperature: float = 0.1,
        cl_weight: float = 0.5,
        seq_weight: float = 0.5,
        neg_weight: float = 0.5,
        lambda_a: float = 0.5,
        prune_ratio: float = 0.25,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.cl_weight = cl_weight
        self.seq_weight = seq_weight
        self.pad_idx = pad_idx

        # Shared item embedding
        self.item_embedding = nn.Embedding(num_items + 2, embed_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Module 1: View Augmentation (BERT-based)
        self.view_aug = ViewAugmentationModule(
            num_items=num_items,
            embed_dim=embed_dim,
            bert_hidden=embed_dim,
            bert_heads=num_heads,
            bert_layers=max(1, num_layers // 2),
            lambda_a=lambda_a,
            prune_ratio=prune_ratio,
        )

        # Module 2: Contrastive Learning (SASRec encoder)
        cl_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.cl_encoder = nn.TransformerEncoder(cl_encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.cl_loss_fn = WeightedInfoNCE(temperature=cl_temperature, neg_weight=neg_weight)

        # Module 3: Sequential Recommendation (LRURec)
        self.seq_encoder = LRURecEncoder(embed_dim, num_layers=num_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def _get_embeddings(self, item_seq: torch.Tensor) -> torch.Tensor:
        B, L = item_seq.shape
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        return self.layer_norm(self.dropout(
            self.item_embedding(item_seq) + self.position_embedding(positions)
        ))

    def _encode_cl(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Encode with SASRec (for contrastive learning)."""
        x = self._get_embeddings(item_seq)
        B, L, _ = x.shape
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        padding_mask = item_seq == self.pad_idx
        x = self.cl_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        last_idx = (seq_len - 1).clamp(min=0).long()
        return x[torch.arange(B, device=x.device), last_idx]

    def _encode_seq(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Encode with LRURec (for sequential recommendation)."""
        x = self._get_embeddings(item_seq)
        x = self.seq_encoder(x)
        B = x.shape[0]
        last_idx = (seq_len - 1).clamp(min=0).long()
        return x[torch.arange(B, device=x.device), last_idx]

    def forward(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Standard forward: returns user representation for inference."""
        return self._encode_seq(item_seq, seq_len)

    def get_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
        return torch.matmul(user_repr, self.item_embedding.weight[:self.num_items + 1].T)

    def compute_loss(
        self,
        item_seq: torch.Tensor,
        seq_len: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Joint loss: L = lambda_1 * L_WCL + lambda_2 * L_seq."""
        attention_mask = (item_seq != self.pad_idx).float()
        item_embeds = self.item_embedding(item_seq)

        # --- Module 1: View Augmentation ---
        importance = self.view_aug.compute_importance(item_embeds, attention_mask)

        # Positive view: prune low-importance positions (keep important ones)
        pos_seq, pos_len = self.view_aug.prune(item_seq, importance, seq_len, prune_high=False)
        # Negative view: prune high-importance positions (remove important ones)
        neg_seq, neg_len = self.view_aug.prune(item_seq, importance, seq_len, prune_high=True)

        # --- Module 2: Contrastive Learning ---
        pos_embed = self._encode_cl(pos_seq, pos_len)
        neg_embed = self._encode_cl(neg_seq, neg_len)
        cl_loss = self.cl_loss_fn(pos_embed, neg_embed)

        # --- Module 3: Sequential Recommendation ---
        user_repr = self._encode_seq(item_seq, seq_len)
        scores = self.get_scores(user_repr)
        seq_loss = F.cross_entropy(scores, target)

        # --- Joint Loss ---
        return self.cl_weight * cl_loss + self.seq_weight * seq_loss
