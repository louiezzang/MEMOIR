"""MIND (Microsoft News Dataset) loading and preprocessing with temporal segmentation."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from .temporal import TemporalBehaviorSegmenter, BehaviorWindow


class MINDDataset(Dataset):
    """MIND dataset with temporal behavior windows.

    Download from: https://msnews.github.io/
    Expected structure:
        data_dir/
            MINDlarge_train/ or MINDsmall_train/
                behaviors.tsv
                news.tsv
    """

    def __init__(
        self,
        data_dir: str = "./data/raw/mind",
        processed_dir: str = "./data/processed/mind",
        size: str = "small",  # small or large
        window_type: str = "weekly",
        min_interactions: int = 5,
        max_history_len: int = 50,
        num_windows: int = 6,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.size = size
        self.min_interactions = min_interactions
        self.max_history_len = max_history_len
        self.num_windows = num_windows
        self.split = split
        self.segmenter = TemporalBehaviorSegmenter(window_type, min_interactions=2)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.processed_dir / f"{split}_samples.parquet"

        if cache_path.exists():
            self.samples = pd.read_parquet(cache_path)
            self.user_windows = self._load_windows()
        else:
            self._process()

    def _process(self):
        prefix = f"MIND{self.size}"
        news = self._load_news(prefix)
        behaviors = self._load_behaviors(prefix, news)

        user_counts = behaviors.groupby("user_id").size()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        behaviors = behaviors[behaviors["user_id"].isin(valid_users)]

        self.user_windows: dict[str, list[BehaviorWindow]] = {}
        for user_id, group in behaviors.groupby("user_id"):
            group = group.tail(self.max_history_len)
            windows = self.segmenter.segment(user_id, group)
            if len(windows) >= 2:
                self.user_windows[user_id] = windows[-self.num_windows:]

        self._split_and_save()

    def _load_news(self, prefix: str) -> dict[str, dict]:
        news_path = self.data_dir / f"{prefix}_train" / "news.tsv"
        if not news_path.exists():
            raise FileNotFoundError(
                f"{news_path} not found. Download MIND dataset from https://msnews.github.io/"
            )

        news = {}
        with open(news_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    news_id = parts[0]
                    news[news_id] = {
                        "category": parts[1],
                        "subcategory": parts[2],
                        "title": parts[3],
                    }
        return news

    def _load_behaviors(self, prefix: str, news: dict) -> pd.DataFrame:
        split_map = {"train": "train", "val": "dev", "test": "dev"}
        bhv_path = self.data_dir / f"{prefix}_{split_map.get(self.split, 'train')}" / "behaviors.tsv"
        if not bhv_path.exists():
            bhv_path = self.data_dir / f"{prefix}_train" / "behaviors.tsv"

        records = []
        with open(bhv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                user_id = parts[1]
                timestamp = pd.Timestamp(parts[2])
                history_ids = parts[3].split() if parts[3] else []

                for nid in history_ids:
                    info = news.get(nid, {})
                    records.append({
                        "user_id": user_id,
                        "item_id": nid,
                        "title": info.get("title", ""),
                        "category": info.get("category", ""),
                        "rating": None,
                        "timestamp": timestamp,
                    })

        return pd.DataFrame(records).sort_values(["user_id", "timestamp"])

    def _split_and_save(self):
        users = list(self.user_windows.keys())
        np.random.seed(42)
        np.random.shuffle(users)

        n = len(users)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        splits = {
            "train": users[:train_end],
            "val": users[train_end:val_end],
            "test": users[val_end:],
        }

        for split_name, split_users in splits.items():
            samples = []
            for uid in split_users:
                windows = self.user_windows[uid]
                last_window = windows[-1]
                for interaction in last_window.interactions:
                    samples.append({
                        "user_id": uid,
                        "target_item_id": interaction["item_id"],
                        "target_title": interaction.get("title", ""),
                        "target_rating": interaction.get("rating"),
                        "num_windows": len(windows),
                    })
            pd.DataFrame(samples).to_parquet(self.processed_dir / f"{split_name}_samples.parquet")

        self._save_windows()
        self.samples = pd.read_parquet(self.processed_dir / f"{self.split}_samples.parquet")

    def _save_windows(self):
        records = []
        for uid, windows in self.user_windows.items():
            for w in windows:
                records.append({
                    "user_id": uid,
                    "window_idx": w.window_idx,
                    "start_time": str(w.start_time),
                    "end_time": str(w.end_time),
                    "text": w.text,
                    "num_interactions": len(w.interactions),
                })
        pd.DataFrame(records).to_parquet(self.processed_dir / "windows.parquet")

    def _load_windows(self) -> dict[str, list[BehaviorWindow]]:
        path = self.processed_dir / "windows.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        windows = {}
        for uid, group in df.groupby("user_id"):
            wins = []
            for _, row in group.sort_values("window_idx").iterrows():
                wins.append(BehaviorWindow(
                    user_id=uid,
                    window_idx=row["window_idx"],
                    start_time=pd.Timestamp(row["start_time"]),
                    end_time=pd.Timestamp(row["end_time"]),
                    interactions=[],
                    text=row["text"],
                ))
            windows[uid] = wins
        return windows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        uid = row["user_id"]
        windows = self.user_windows.get(uid, [])
        window_texts = [w.text for w in windows]

        return {
            "user_id": uid,
            "window_texts": window_texts,
            "target_item_id": row["target_item_id"],
            "target_title": row["target_title"],
            "target_rating": row.get("target_rating"),
        }
