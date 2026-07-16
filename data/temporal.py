"""Temporal behavior segmentation: splits user interaction history into time windows."""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass


@dataclass
class BehaviorWindow:
    user_id: str
    window_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    interactions: list[dict]  # [{item_id, category, title, rating, timestamp}, ...]
    text: str  # serialized natural language description


class TemporalBehaviorSegmenter:
    """Segments user interaction history into time windows and serializes each to text."""

    WINDOW_FREQ = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
    }

    def __init__(self, window_type: str = "monthly", min_interactions: int = 2):
        if window_type not in self.WINDOW_FREQ:
            raise ValueError(f"window_type must be one of {list(self.WINDOW_FREQ)}")
        self.freq = self.WINDOW_FREQ[window_type]
        self.min_interactions = min_interactions

    def segment(self, user_id: str, interactions: pd.DataFrame) -> list[BehaviorWindow]:
        """Split a single user's interactions into temporal windows.

        Args:
            user_id: user identifier
            interactions: DataFrame with columns [item_id, category, title, rating, timestamp]
                          sorted by timestamp ascending
        """
        if interactions.empty:
            return []

        interactions = interactions.sort_values("timestamp")
        interactions["period"] = interactions["timestamp"].dt.to_period(self.freq)

        windows = []
        for idx, (period, group) in enumerate(interactions.groupby("period")):
            if len(group) < self.min_interactions:
                continue
            records = group.to_dict("records")
            text = self._serialize(records)
            windows.append(BehaviorWindow(
                user_id=user_id,
                window_idx=idx,
                start_time=period.start_time,
                end_time=period.end_time,
                interactions=records,
                text=text,
            ))
        return windows

    def _serialize(self, records: list[dict]) -> str:
        """Convert a window's interactions to natural language."""
        lines = []
        for r in records:
            action = self._rating_to_action(r.get("rating"))
            title = r.get("title", r.get("item_id", "unknown item"))
            category = r.get("category", "")
            cat_str = f" in {category}" if category else ""
            lines.append(f"{action} \"{title}\"{cat_str}")

        return "User " + "; ".join(lines) + "."

    @staticmethod
    def _rating_to_action(rating) -> str:
        if rating is None:
            return "interacted with"
        rating = float(rating)
        if rating >= 4.0:
            return "highly rated"
        if rating >= 3.0:
            return "viewed"
        return "disliked"
