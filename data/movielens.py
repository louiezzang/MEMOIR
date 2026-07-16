"""MovieLens dataset loading and preprocessing with temporal segmentation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from .temporal import TemporalBehaviorSegmenter, BehaviorWindow


class MovieLensDataset(Dataset):
    """MovieLens 25M dataset with temporal behavior windows.

    Download from: https://grouplens.org/datasets/movielens/25m/
    Expected structure:
        data_dir/
            ml-25m/
                ratings.csv
                movies.csv
    """

    def __init__(
        self,
        data_dir: str = "./data/raw/movielens",
        processed_dir: str = "./data/processed/movielens",
        window_type: str = "monthly",
        min_interactions: int = 20,
        max_history_len: int = 50,
        num_windows: int = 6,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
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
        ratings, movie_info = self._load_raw_data()

        ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
        ratings["title"] = ratings["item_id"].map(lambda x: movie_info.get(x, ("unknown", ""))[0])
        ratings["category"] = ratings["item_id"].map(lambda x: movie_info.get(x, ("", "unknown"))[1])
        ratings = ratings.sort_values(["user_id", "timestamp"])

        user_counts = ratings.groupby("user_id").size()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        ratings = ratings[ratings["user_id"].isin(valid_users)]

        # Subsample users for tractability (25M is large)
        all_users = ratings["user_id"].unique()
        np.random.seed(42)
        if len(all_users) > 50000:
            sampled = np.random.choice(all_users, 50000, replace=False)
            ratings = ratings[ratings["user_id"].isin(sampled)]

        self.user_windows: dict[str, list[BehaviorWindow]] = {}
        for user_id, group in ratings.groupby("user_id"):
            group = group.tail(self.max_history_len)
            windows = self.segmenter.segment(str(user_id), group)
            if len(windows) >= 2:
                self.user_windows[str(user_id)] = windows[-self.num_windows:]

        self._split_and_save()

    def _load_raw_data(self) -> tuple[pd.DataFrame, dict]:
        """Load ratings and movie info, auto-detecting ML-25M or ML-100K."""
        search_dirs = [self.data_dir, self.data_dir.parent]
        ml25m = None
        ml100k = None
        for d in search_dirs:
            if (d / "ml-25m" / "ratings.csv").exists():
                ml25m = d / "ml-25m"
                break
            if (d / "ml-100k" / "u.data").exists():
                ml100k = d / "ml-100k"
                break

        if ml25m is not None:
            movies = pd.read_csv(ml25m / "movies.csv")
            movie_info = dict(zip(movies["movieId"], zip(movies["title"], movies["genres"])))
            ratings = pd.read_csv(ml25m / "ratings.csv")
            ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
            return ratings, movie_info

        if ml100k is not None:
            base = ml100k
            ratings = pd.read_csv(
                base / "u.data", sep="\t", header=None,
                names=["user_id", "item_id", "rating", "timestamp"],
            )
            genres = []
            with open(base / "u.genre", "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        name, _ = line.split("|")
                        genres.append(name)

            movie_info = {}
            with open(base / "u.item", "r", encoding="latin-1") as f:
                for line in f:
                    parts = line.strip().split("|")
                    mid = int(parts[0])
                    title = parts[1]
                    genre_flags = [int(g) for g in parts[5:]]
                    cats = [genres[i] for i, flag in enumerate(genre_flags) if flag and i < len(genres)]
                    movie_info[mid] = (title, "|".join(cats) if cats else "unknown")
            return ratings, movie_info

        raise FileNotFoundError(
            f"No MovieLens data found in {self.data_dir}. "
            f"Expected ml-25m/ratings.csv or ml-100k/u.data"
        )

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
                        "target_item_id": str(interaction["item_id"]),
                        "target_title": interaction.get("title", ""),
                        "target_rating": interaction.get("rating", 0.0),
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
                    user_id=str(uid),
                    window_idx=row["window_idx"],
                    start_time=pd.Timestamp(row["start_time"]),
                    end_time=pd.Timestamp(row["end_time"]),
                    interactions=[],
                    text=row["text"],
                ))
            windows[str(uid)] = wins
        return windows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        uid = row["user_id"]
        windows = self.user_windows.get(str(uid), [])
        window_texts = [w.text for w in windows]

        return {
            "user_id": str(uid),
            "window_texts": window_texts,
            "target_item_id": row["target_item_id"],
            "target_title": row["target_title"],
            "target_rating": row["target_rating"],
        }
