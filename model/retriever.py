"""ANN-based retrieval module for candidate generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANNRetriever(nn.Module):
    """Approximate nearest neighbor retrieval using dot product."""

    def __init__(self, top_k: int = 50):
        super().__init__()
        self.top_k = top_k
        self._item_index: torch.Tensor | None = None
        self._item_ids: list[str] | None = None

    def build_index(self, item_embeddings: torch.Tensor, item_ids: list[str]):
        """Pre-compute item index for fast retrieval."""
        self._item_index = F.normalize(item_embeddings, dim=-1).detach()
        self._item_ids = item_ids

    def forward(
        self,
        user_memory: torch.Tensor,
        item_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-K items for each user.

        Args:
            user_memory: [B, D] user memory embeddings
            item_embeddings: [M, D] item embeddings (uses pre-built index if None)

        Returns:
            scores: [B, K] retrieval scores
            indices: [B, K] item indices
        """
        if item_embeddings is None:
            if self._item_index is None:
                raise RuntimeError("Call build_index() first or pass item_embeddings")
            item_embeddings = self._item_index.to(user_memory.device)

        user_norm = F.normalize(user_memory, dim=-1)
        item_norm = F.normalize(item_embeddings, dim=-1)

        scores = torch.matmul(user_norm, item_norm.T)  # [B, M]
        top_scores, top_indices = torch.topk(scores, k=min(self.top_k, scores.shape[1]), dim=-1)

        return top_scores, top_indices
