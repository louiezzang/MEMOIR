"""Evolution-Preserving Contrastive Learning module (core novelty of MEMOIR).

Two components:
1. Temporal smoothness: adjacent windows of the same user should be close (InfoNCE)
2. Directional consistency: evolution *direction vectors* should be aligned (cosine)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolutionContrastive(nn.Module):
    """Contrastive learning over temporal behavioral memory windows."""

    def __init__(
        self,
        temperature: float = 0.07,
        alpha_direction: float = 0.2,
        margin: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha_direction
        self.margin = margin

    def forward(
        self,
        anchor_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        prev_anchor_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute evolution-preserving contrastive loss.

        Args:
            anchor_embeds: [N, D] - memory at time t
            positive_embeds: [N, D] - memory at time t+1 (same user)
            negative_embeds: [N, K, D] - memories from different users
            prev_anchor_embeds: [N, D] - memory at time t-1 (for directional loss)

        Returns:
            dict with 'loss', 'smoothness_loss', 'direction_loss' keys
        """
        anchor = F.normalize(anchor_embeds, dim=-1)
        positive = F.normalize(positive_embeds, dim=-1)
        negative = F.normalize(negative_embeds, dim=-1)

        # (1) Temporal smoothness loss (InfoNCE)
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # [N]
        neg_sim = torch.bmm(negative, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # [N, K]
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [N, 1+K]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        smoothness_loss = F.cross_entropy(logits, labels)

        # (2) Directional consistency loss — direction vector cosine similarity
        direction_loss = torch.tensor(0.0, device=anchor.device)
        if prev_anchor_embeds is not None:
            prev_anchor = F.normalize(prev_anchor_embeds, dim=-1)

            d_prev = anchor - prev_anchor          # direction t-1 → t
            d_curr = positive - anchor              # direction t → t+1

            # Guard against near-zero-length direction vectors: as smoothness_loss
            # pulls anchor/positive/prev_anchor together, these differences shrink
            # toward zero, and F.normalize on a near-zero vector blows up the
            # backward gradient (scales ~1/||x||). Skip transitions too small to
            # give a numerically stable direction instead of amplifying noise.
            min_norm = 1e-3
            valid = (d_prev.norm(dim=-1) > min_norm) & (d_curr.norm(dim=-1) > min_norm)

            if valid.any():
                d_prev_norm = F.normalize(d_prev[valid], dim=-1)
                d_curr_norm = F.normalize(d_curr[valid], dim=-1)

                cosine_sim = (d_prev_norm * d_curr_norm).sum(dim=-1)  # [N_valid]
                direction_loss = F.relu(self.margin - cosine_sim).mean()

        loss = smoothness_loss + self.alpha * direction_loss

        return {
            "loss": loss,
            "smoothness_loss": smoothness_loss,
            "direction_loss": direction_loss,
        }
