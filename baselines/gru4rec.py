"""GRU4Rec: Session-based Recommendations with Recurrent Neural Networks (ICLR 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4Rec(nn.Module):
    """GRU-based sequential recommendation model.

    Reference: Hidasi et al., "Session-based Recommendations with
    Recurrent Neural Networks", ICLR 2016.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_seq: [B, L] item ID sequence (padded)
            seq_len: [B] actual sequence lengths

        Returns:
            user_repr: [B, embed_dim] user representation from last GRU state
        """
        x = self.dropout(self.item_embedding(item_seq))  # [B, L, D]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_len.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.gru(packed)  # hidden: [num_layers, B, H]
        user_repr = self.output_proj(hidden[-1])  # [B, D]
        return user_repr

    def get_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
        """Compute scores for all items. Returns [B, num_items+1]."""
        return torch.matmul(user_repr, self.item_embedding.weight.T)

    def compute_loss(
        self,
        item_seq: torch.Tensor,
        seq_len: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        user_repr = self.forward(item_seq, seq_len)
        scores = self.get_scores(user_repr)
        return F.cross_entropy(scores, target)
