"""DuoRec: Contrastive Learning with Dual Augmentations for Sequential Rec (WSDM 2022)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuoRec(nn.Module):
    """Model-level contrastive learning for sequential recommendation.

    Reference: Qiu et al., "Contrastive Learning for Representation Degeneration
    Problem in Sequential Recommendation", WSDM 2022.

    Key idea: Uses supervised contrastive learning with sequences sharing the
    same target item as positive pairs (model-level augmentation).
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        cl_temperature: float = 1.0,
        cl_weight: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.cl_temperature = cl_temperature
        self.cl_weight = cl_weight
        self.pad_idx = pad_idx

        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Separate dropout for model-level augmentation
        self.aug_dropout = nn.Dropout(dropout * 2)

    def _encode(self, item_seq: torch.Tensor, seq_len: torch.Tensor, use_aug_dropout: bool = False) -> torch.Tensor:
        B, L = item_seq.shape
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)

        x = self.item_embedding(item_seq) + self.position_embedding(positions)
        drop = self.aug_dropout if use_aug_dropout else self.dropout_layer
        x = self.layer_norm(drop(x))

        causal_mask = torch.triu(
            torch.ones(L, L, device=item_seq.device, dtype=torch.bool), diagonal=1,
        )
        padding_mask = item_seq == self.pad_idx
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        last_idx = (seq_len - 1).clamp(min=0).long()
        return x[torch.arange(B, device=x.device), last_idx]

    def forward(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        return self._encode(item_seq, seq_len)

    def get_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
        return torch.matmul(user_repr, self.item_embedding.weight.T)

    def supervised_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised contrastive loss: sequences with same target are positive pairs."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        B = z1.shape[0]
        sim = torch.matmul(z1, z2.T) / self.cl_temperature  # [B, B]

        # Positive mask: same target item
        target_match = (targets.unsqueeze(0) == targets.unsqueeze(1))  # [B, B]
        target_match.fill_diagonal_(False)

        if target_match.sum() == 0:
            # Fallback to self-contrastive (diagonal as positives)
            labels = torch.arange(B, device=sim.device)
            return F.cross_entropy(sim, labels)

        # InfoNCE with supervised positives
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * target_match.float()).sum(dim=1)
        neg_sum = exp_sim.sum(dim=1) - exp_sim.diag()

        loss = -torch.log(pos_sum / neg_sum.clamp(min=1e-9) + 1e-9)
        valid = target_match.sum(dim=1) > 0
        return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=sim.device)

    def compute_loss(
        self,
        item_seq: torch.Tensor,
        seq_len: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Recommendation loss
        user_repr = self._encode(item_seq, seq_len)
        scores = self.get_scores(user_repr)
        rec_loss = F.cross_entropy(scores, target)

        # Model-level contrastive: two forward passes with different dropout
        z1 = self._encode(item_seq, seq_len, use_aug_dropout=False)
        z2 = self._encode(item_seq, seq_len, use_aug_dropout=True)
        cl_loss = self.supervised_contrastive_loss(z1, z2, target)

        return rec_loss + self.cl_weight * cl_loss
