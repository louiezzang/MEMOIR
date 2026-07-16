"""LLM Reasoning Layer: re-ranks candidates using user memory context."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasoningLayer(nn.Module):
    """Cross-attention based reasoning layer that re-ranks candidates using user memory.

    Instead of calling LLM inference at ranking time (expensive),
    we use a learned cross-attention mechanism between user memory and item embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        user_memories: torch.Tensor,
        candidate_items: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Re-rank candidate items using user memory context.

        Args:
            user_memories: [B, W, D] per-window memory embeddings
            candidate_items: [B, K, D] candidate item embeddings
            memory_mask: [B, W] bool mask for valid memory windows

        Returns:
            scores: [B, K] ranking scores for each candidate
        """
        if memory_mask is not None:
            # Convert to attention mask format (True = ignore)
            memory_key_padding_mask = ~memory_mask
        else:
            memory_key_padding_mask = None

        # Cross-attend: items attend to user memories
        reasoned = self.decoder(
            tgt=candidate_items,
            memory=user_memories,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, K, D]

        scores = self.score_head(reasoned).squeeze(-1)  # [B, K]
        return scores
