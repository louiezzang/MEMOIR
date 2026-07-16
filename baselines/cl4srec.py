"""CL4SRec: Contrastive Learning for Sequential Recommendation (ICDE 2022)."""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class CL4SRec(nn.Module):
    """SASRec backbone + contrastive learning with sequence augmentations.

    Reference: Xie et al., "Contrastive Learning for Sequential Recommendation", ICDE 2022.

    Augmentation strategies:
    - Item crop: randomly crop a contiguous sub-sequence
    - Item mask: randomly mask items in the sequence
    - Item reorder: randomly shuffle a contiguous sub-sequence
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
        aug_ratio: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.cl_temperature = cl_temperature
        self.cl_weight = cl_weight
        self.aug_ratio = aug_ratio
        self.pad_idx = pad_idx
        self.mask_token = num_items + 1

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
        self.cl_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _encode(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        B, L = item_seq.shape
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(item_seq) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout(x))

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
        return torch.matmul(user_repr, self.item_embedding.weight[:self.num_items + 1].T)

    # --- Augmentation strategies ---

    def _augment_crop(self, seq: torch.Tensor, length: int) -> tuple[torch.Tensor, int]:
        crop_len = max(1, int(length * (1 - self.aug_ratio)))
        start = random.randint(0, length - crop_len)
        cropped = seq[start:start + crop_len]
        padded = F.pad(cropped, (0, self.max_len - crop_len), value=self.pad_idx)
        return padded, crop_len

    def _augment_mask(self, seq: torch.Tensor, length: int) -> torch.Tensor:
        masked = seq.clone()
        num_mask = max(1, int(length * self.aug_ratio))
        positions = random.sample(range(length), num_mask)
        for p in positions:
            masked[p] = self.mask_token
        return masked

    def _augment_reorder(self, seq: torch.Tensor, length: int) -> torch.Tensor:
        reordered = seq.clone()
        sub_len = max(2, int(length * self.aug_ratio))
        start = random.randint(0, max(0, length - sub_len))
        indices = list(range(start, min(start + sub_len, length)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            reordered[start + i] = seq[idx]
        return reordered

    def augment(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a random augmentation to each sequence in the batch."""
        B, L = item_seq.shape
        aug_seqs = []
        aug_lens = []

        for i in range(B):
            length = seq_len[i].item()
            s = item_seq[i]
            aug_type = random.choice(["crop", "mask", "reorder"])

            if aug_type == "crop":
                aug_s, aug_l = self._augment_crop(s, length)
                aug_seqs.append(aug_s)
                aug_lens.append(aug_l)
            elif aug_type == "mask":
                aug_seqs.append(self._augment_mask(s, length))
                aug_lens.append(length)
            else:
                aug_seqs.append(self._augment_reorder(s, length))
                aug_lens.append(length)

        return torch.stack(aug_seqs), torch.tensor(aug_lens, device=item_seq.device)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss between two augmented views."""
        z1 = F.normalize(self.cl_projector(z1), dim=-1)
        z2 = F.normalize(self.cl_projector(z2), dim=-1)

        B = z1.shape[0]
        sim = torch.matmul(z1, z2.T) / self.cl_temperature  # [B, B]
        labels = torch.arange(B, device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

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

        # Contrastive loss: two augmented views
        aug_seq1, aug_len1 = self.augment(item_seq, seq_len)
        aug_seq2, aug_len2 = self.augment(item_seq, seq_len)
        z1 = self._encode(aug_seq1, aug_len1)
        z2 = self._encode(aug_seq2, aug_len2)
        cl_loss = self.contrastive_loss(z1, z2)

        return rec_loss + self.cl_weight * cl_loss
