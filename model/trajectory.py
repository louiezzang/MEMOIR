"""Trajectory Predictor: GRU-based module that learns evolution direction and predicts future memory states."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryPredictor(nn.Module):
    """Predicts next-step memory embedding from a sequence of window memories.

    Uses a GRU to model the evolution trajectory, then projects the hidden
    state to predict the next window's memory embedding.  Also exposes the
    current evolution direction vector for the evolution-aware aggregate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        memory_sequence: torch.Tensor,
        window_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run trajectory prediction.

        Args:
            memory_sequence: [B, W, D] per-window memory embeddings (chronological)
            window_masks: [B, W] bool mask (True = valid window)

        Returns:
            dict with:
                predicted_next: [B, D] predicted next-step memory
                direction: [B, D] current evolution direction (last hidden delta)
                hidden: [B, H] final GRU hidden state
        """
        B, W, D = memory_sequence.shape
        lengths = window_masks.sum(dim=1).clamp(min=1).long()  # [B]

        gru_out, _ = self.gru(memory_sequence)

        batch_idx = torch.arange(B, device=memory_sequence.device)
        last_pos = (lengths - 1)
        last_hidden = gru_out[batch_idx, last_pos]  # [B, H]

        predicted_next = self.predictor(last_hidden)  # [B, D]

        # Direction: difference between last two valid memory embeddings
        second_last_pos = (lengths - 2).clamp(min=0)

        last_mem = memory_sequence[batch_idx, last_pos]            # [B, D]
        second_last_mem = memory_sequence[batch_idx, second_last_pos]  # [B, D]

        has_two = (lengths >= 2).float().unsqueeze(-1).to(memory_sequence.device)
        direction = (last_mem - second_last_mem) * has_two  # zero if only 1 window

        return {
            "predicted_next": predicted_next,
            "direction": direction,
            "hidden": last_hidden,
        }


class ExtrapolationLoss(nn.Module):
    """Hold-out-last-window supervision: predicted next ≈ actual next."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE between predicted next-step and actual held-out window.

        Args:
            predicted: [B, D]
            actual: [B, D]
        """
        predicted = F.normalize(predicted, dim=-1)
        actual = F.normalize(actual, dim=-1)

        sim_matrix = torch.matmul(predicted, actual.T) / self.temperature  # [B, B]
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        return F.cross_entropy(sim_matrix, labels)
