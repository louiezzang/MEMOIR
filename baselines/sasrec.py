"""SASRec: Self-Attentive Sequential Recommendation (ICDM 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    """Transformer-based sequential recommendation model.

    Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
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

    def forward(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_seq: [B, L] item ID sequence (padded)
            seq_len: [B] actual sequence lengths

        Returns:
            user_repr: [B, embed_dim]
        """
        B, L = item_seq.shape
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(item_seq) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout(x))

        # Causal mask: each position can only attend to itself and previous positions
        causal_mask = torch.triu(
            torch.ones(L, L, device=item_seq.device, dtype=torch.bool), diagonal=1,
        )
        # src_key_padding_mask is not supported on MPS; causal mask is sufficient
        # since we only use the last valid position's representation
        if item_seq.device.type == "mps":
            x = self.encoder(x, mask=causal_mask)
        else:
            x = self.encoder(x, mask=causal_mask, src_key_padding_mask=(item_seq == 0))

        # Gather last valid position's representation
        last_idx = (seq_len - 1).clamp(min=0).long()  # [B]
        user_repr = x[torch.arange(B, device=x.device), last_idx]  # [B, D]
        return user_repr

    def get_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
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
