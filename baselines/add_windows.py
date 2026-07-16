"""Generate temporal windows from preprocessed MovieLens samples for MEMOIR."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


class BehaviorWindow:
    """Represents a time window of user behavior."""

    def __init__(self, user_id: str, window_idx: int, start_time, end_time, interactions: list, text: str):
        self.user_id = user_id
        self.window_idx = window_idx
        self.start_time = start_time
        self.end_time = end_time
        self.interactions = interactions
        self.text = text


def generate_windows(data_dir: str, processed_dir: str, window_type: str = "monthly"):
    """Generate temporal windows from preprocessed samples.

    Args:
        data_dir: Raw MovieLens directory (for original ratings)
        processed_dir: Preprocessed samples directory
        window_type: Window type - monthly, weekly, quarterly
    """
    raw_path = Path(data_dir)
    output_path = Path(processed_dir)

    # Find the raw ratings file
    ratings_path = None
    for subdir in ["ml-25m", "ml-100k"]:
        path = raw_path / subdir / "ratings.csv"
        if path.exists():
            ratings_path = path
            break
        path = raw_path / subdir / "u.data"
        if path.exists():
            ratings_path = path
            break

    if not ratings_path:
        raise FileNotFoundError(f"No ratings file found in {raw_path}")

    print(f"Loading raw ratings from {ratings_path}")
    if ratings_path.name == "ratings.csv":
        ratings = pd.read_csv(ratings_path)
        ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    else:
        ratings = pd.read_csv(
            ratings_path, sep="\t", header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
        )

    # Convert timestamp
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    print(f"Loaded {len(ratings)} interactions from {ratings['user_id'].nunique()} users")

    # Load preprocessed samples to get the user set
    splits = ["train", "val", "test"]
    all_users = set()
    for split in splits:
        samples_path = output_path / f"{split}_samples.parquet"
        if samples_path.exists():
            df = pd.read_parquet(samples_path)
            all_users.update(df["user_id"].unique())

    print(f"Processing {len(all_users)} users from preprocessed data")

    # Filter ratings to only include users in our dataset
    ratings = ratings[ratings["user_id"].isin(all_users)]

    # Load item titles for text generation
    movies_path = raw_path / "ml-25m" / "movies.csv"
    if movies_path.exists():
        movies = pd.read_csv(movies_path)
        movie_titles = dict(zip(movies["movieId"], movies["title"]))
    else:
        movie_titles = {}

    # Generate windows per user
    window_records = []
    all_windows: dict[str, list[BehaviorWindow]] = {}

    from datetime import timedelta

    for i, (user_id, group) in enumerate(ratings.groupby("user_id")):
        if i % 10000 == 0:
            print(f"  Processed {i} users...")

        # Sort by timestamp
        group = group.sort_values("timestamp")

        # Create time windows based on window_type
        min_time = group["timestamp"].min()
        max_time = group["timestamp"].max()

        if window_type == "monthly":
            window_duration = pd.Timedelta(days=30)
        elif window_type == "weekly":
            window_duration = pd.Timedelta(weeks=1)
        elif window_type == "quarterly":
            window_duration = pd.Timedelta(days=90)
        else:
            window_duration = pd.Timedelta(days=30)  # default to monthly

        windows = []
        current_start = min_time
        window_idx = 0

        while current_start <= max_time:
            current_end = current_start + window_duration

            # Get interactions in this window
            mask = (group["timestamp"] >= current_start) & (group["timestamp"] < current_end)
            window_ratings = group[mask]

            if len(window_ratings) > 0:
                # Build text description of window
                items_in_window = window_ratings["item_id"].head(5).tolist()
                item_names = [movie_titles.get(itemId, f"Item {itemId}") for itemId in items_in_window]

                if len(item_names) <= 3:
                    items_str = " and ".join(item_names)
                elif len(item_names) > 3:
                    items_str = ", ".join(item_names[:2]) + " and others"

                window_text = f"User viewed {len(window_ratings)} movies including {items_str}"

                windows.append(BehaviorWindow(
                    user_id=str(user_id),
                    window_idx=window_idx,
                    start_time=current_start,
                    end_time=current_end,
                    interactions=[
                        {"item_id": r["item_id"], "rating": r["rating"]}
                        for _, r in window_ratings.iterrows()
                    ],
                    text=window_text
                ))

            window_idx += 1
            current_start = current_end

        if windows:
            all_windows[str(user_id)] = windows

    print(f"\nGenerated {len(all_windows)} users with windows")

    # Save windows.parquet
    records = []
    for uid, windows in all_windows.items():
        for w in windows:
            records.append({
                "user_id": int(uid) if uid.isdigit() else uid,
                "window_idx": w.window_idx,
                "start_time": str(w.start_time),
                "end_time": str(w.end_time),
                "text": w.text,
                "num_interactions": len(w.interactions),
            })

    windows_df = pd.DataFrame(records)
    output_path.mkdir(parents=True, exist_ok=True)
    windows_path = output_path / "windows.parquet"
    windows_df.to_parquet(windows_path)
    print(f"Saved {len(windows_df)} window records to {windows_path}")

    # Update samples with num_windows column
    for split in splits:
        samples_path = output_path / f"{split}_samples.parquet"
        if not samples_path.exists():
            continue

        df = pd.read_parquet(samples_path)

        # Add num_windows column
        df["num_windows"] = df["user_id"].map(lambda x: len(all_windows.get(str(x), [])))

        output_file = output_path / f"{split}_samples.parquet"
        df.to_parquet(output_file)
        print(f"Updated {split} samples with num_windows")

    print("\nDone! Windows generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate temporal windows for MEMOIR")
    parser.add_argument("--data-dir", type=str, default="./data/raw/movielens",
                        help="Raw MovieLens directory")
    parser.add_argument("--processed-dir", type=str, default="./data/processed/movielens",
                        help="Preprocessed samples directory")
    parser.add_argument("--window-type", type=str, choices=["monthly", "weekly", "quarterly"],
                        default="monthly", help="Window type")

    args = parser.parse_args()

    generate_windows(args.data_dir, args.processed_dir, args.window_type)
