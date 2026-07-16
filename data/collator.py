"""Data collator for MEMOIR: handles variable-length window texts and creates contrastive pairs."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class MEMOIRCollator:
    """Collates samples into batches with contrastive pairs for evolution learning."""

    max_windows: int = 6
    num_negatives: int = 4

    def __call__(self, batch: list[dict]) -> dict:
        user_ids = [s["user_id"] for s in batch]
        window_texts_batch = [s["window_texts"] for s in batch]
        target_titles = [s["target_title"] for s in batch]
        target_item_ids = [s["target_item_id"] for s in batch]

        # Pad window texts to max_windows
        padded_windows = []
        window_masks = []
        for texts in window_texts_batch:
            n = len(texts)
            padded = texts[-self.max_windows:] if n > self.max_windows else texts
            mask = [1] * len(padded) + [0] * (self.max_windows - len(padded))
            padded = padded + [""] * (self.max_windows - len(padded))
            padded_windows.append(padded)
            window_masks.append(mask)

        # Build temporal contrastive triplets (t-1, t, t+1) from same user.
        # Requires >= 3 windows so all three positions are valid.
        # prev_anchors, pos_anchors, pos_positives are always the same length N,
        # enabling the directional consistency loss in EvolutionContrastive.
        prev_anchor_texts = []
        pos_anchor_texts = []
        pos_positive_texts = []
        neg_texts = []
        triplet_user_indices = []

        for i, texts in enumerate(window_texts_batch):
            if len(texts) < 3:
                continue
            t = random.randint(1, len(texts) - 2)  # ensures t-1 and t+1 both exist
            prev_anchor_texts.append(texts[t - 1])
            pos_anchor_texts.append(texts[t])
            pos_positive_texts.append(texts[t + 1])
            triplet_user_indices.append(i)

        # Negatives for each triplet user — sampled from other users in the batch
        for i in triplet_user_indices:
            negs = []
            candidates = [j for j in range(len(batch)) if j != i and len(window_texts_batch[j]) > 0]
            for j in random.sample(candidates, min(self.num_negatives, len(candidates))):
                neg_t = random.randint(0, len(window_texts_batch[j]) - 1)
                negs.append(window_texts_batch[j][neg_t])
            neg_texts.append(negs)

        return {
            "user_ids": user_ids,
            "window_texts": padded_windows,       # [B, max_windows] list of strings
            "window_masks": torch.tensor(window_masks, dtype=torch.bool),  # [B, max_windows]
            "target_item_ids": target_item_ids,
            "target_titles": target_titles,
            # Contrastive triplets (all same length N = number of 3+-window users)
            "prev_anchors": prev_anchor_texts,    # texts[t-1]
            "pos_anchors": pos_anchor_texts,      # texts[t]
            "pos_positives": pos_positive_texts,  # texts[t+1]
            "neg_texts": neg_texts,               # list[list[str]]
        }
