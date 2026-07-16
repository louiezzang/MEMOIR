"""BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations (CIKM 2019)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT4Rec(nn.Module):
    """BERT-style masked sequential recommendation model.

    Reference: Sun et al., "BERT4Rec: Sequential Recommendation with
    Bidirectional Encoder Representations from Transformers", CIKM 2019.
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        mask_prob: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1
        self.pad_idx = pad_idx

        self.item_embedding = nn.Embedding(num_items + 2, embed_dim, padding_idx=pad_idx)
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
        self.output_proj = nn.Linear(embed_dim, num_items + 2)

    def mask_sequence(self, item_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking for training."""
        masked_seq = item_seq.clone()
        labels = torch.full_like(item_seq, -100)  # ignore index

        for i in range(item_seq.shape[0]):
            for j in range(item_seq.shape[1]):
                if item_seq[i, j] == self.pad_idx:
                    continue
                if torch.rand(1).item() < self.mask_prob:
                    labels[i, j] = item_seq[i, j]
                    masked_seq[i, j] = self.mask_token

        return masked_seq, labels

    def forward(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Encode sequence (for inference, last position is masked).

        Returns:
            user_repr: [B, embed_dim] from the last valid position
        """
        B, L = item_seq.shape
        # For inference: mask last position
        masked_seq = item_seq.clone()
        last_idx = (seq_len - 1).clamp(min=0).long()
        masked_seq[torch.arange(B), last_idx] = self.mask_token

        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(masked_seq) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout(x))

        padding_mask = item_seq == self.pad_idx
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        user_repr = x[torch.arange(B, device=x.device), last_idx]
        return user_repr

    def get_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
        return torch.matmul(user_repr, self.item_embedding.weight[:self.num_items + 1].T)

    def compute_loss(
        self,
        item_seq: torch.Tensor,
        seq_len: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Masked language model loss for training."""
        B, L = item_seq.shape
        masked_seq, labels = self.mask_sequence(item_seq)

        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(masked_seq) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout(x))

        padding_mask = item_seq == self.pad_idx
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        logits = self.output_proj(x)  # [B, L, V]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        # Also add next-item prediction loss on last position
        last_idx = (seq_len - 1).clamp(min=0).long()
        last_hidden = x[torch.arange(B, device=x.device), last_idx]
        next_scores = self.get_scores(last_hidden)
        loss += F.cross_entropy(next_scores, target)

        return loss
