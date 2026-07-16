"""Preprocess data for baseline sequential recommendation models.

Supports multiple dataset formats:
- Amazon: JSONL files (user_id, parent_asin, timestamp)
- MovieLens: CSV files (ratings.csv with userId, movieId, timestamp)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np


def preprocess_amazon(
    data_dir: str = "./data/raw/amazon",
    processed_dir: str = "./data/processed/amazon",
    categories: list[str] | None = None,
    min_interactions: int = 5,
):
    """Preprocess Amazon Reviews data for baseline models.

    Args:
        data_dir: Path to raw JSONL files
        processed_dir: Output directory for Parquet files
        categories: List of category names (default: Electronics, Clothing_Shoes_and_Jewelry)
        min_interactions: Minimum interactions per user
    """
    categories = categories or ["Electronics", "Clothing_Shoes_and_Jewelry"]
    data_path = Path(data_dir)
    output_path = Path(processed_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all categories
    dfs = []
    for cat in categories:
        path = data_path / f"{cat}.jsonl"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        print(f"Loading {cat}...")
        records = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                records.append({
                    "user_id": r["user_id"],
                    "item_id": r["parent_asin"],
                    "timestamp": int(r.get("timestamp", 0)),
                })

        df = pd.DataFrame(records)
        print(f"  {len(df)} interactions from {df['user_id'].nunique()} users")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No data found in {data_path}")

    # Combine all categories
    reviews = pd.concat(dfs, ignore_index=True)
    reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], unit="ms")
    reviews = reviews.sort_values(["user_id", "timestamp"])

    print(f"\nTotal: {len(reviews)} interactions from {reviews['user_id'].nunique()} users")

    # Filter users with minimum interactions
    user_counts = reviews.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_interactions].index
    reviews = reviews[reviews["user_id"].isin(valid_users)]

    print(f"After filtering ({min_interactions}+ interactions): {len(reviews)} interactions from {reviews['user_id'].nunique()} users")

    # Build item vocabulary and mapping
    all_items = set()
    for split_name in ["train", "val", "test"]:
        split_path = output_path / f"{split_name}_samples.parquet"
        if split_path.exists():
            df = pd.read_parquet(split_path)
            all_items.update(df["target_item_id"].unique())

    # If no splits exist yet, use all items
    if not all_items:
        all_items = set(reviews["item_id"].unique())

    item2idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
    reviews["item_idx"] = reviews["item_id"].map(item2idx)

    # Split data (80/10/10) by user
    users = list(reviews["user_id"].unique())
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

    # For baselines, we need per-user sequences
    # Save as samples format expected by train_baseline.py
    for split_name, split_users in splits.items():
        print(f"\nSaving {split_name} split...")
        user_data = reviews[reviews["user_id"].isin(split_users)].copy()

        # Create samples: each interaction is a target, with context from previous interactions
        samples = []
        for user_id, group in user_data.groupby("user_id"):
            group = group.sort_values("timestamp")
            item_ids = group["item_idx"].tolist()
            timestamps = group["timestamp"].tolist()

            # Each position after the first can be a target
            for i in range(1, len(item_ids)):
                samples.append({
                    "user_id": user_id,
                    "target_item_id": item_ids[i],
                    "target_timestamp": int(timestamps[i].timestamp()),
                    "num_interactions": len(group),
                })

        df = pd.DataFrame(samples)
        output_file = output_path / f"{split_name}_samples.parquet"
        df.to_parquet(output_file)
        print(f"  {len(df)} samples saved to {output_file}")

    # Save item mapping
    pd.DataFrame([
        {"item_id": item, "item_idx": idx}
        for item, idx in item2idx.items()
    ]).to_parquet(output_path / "item_mapping.parquet")

    print(f"\nPreprocessing complete!")
    print(f"  Items: {len(item2idx)}")
    print(f"  Output: {output_path}")


def preprocess_movielens(
    data_dir: str = "./data/raw/movielens",
    processed_dir: str = "./data/processed/movielens",
    min_interactions: int = 5,
):
    """Preprocess MovieLens data for baseline models.

    Args:
        data_dir: Path to MovieLens directory (containing ratings.csv)
        processed_dir: Output directory for Parquet files
        min_interactions: Minimum interactions per user
    """
    data_path = Path(data_dir)
    output_path = Path(processed_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load ratings file
    ratings_path = data_path / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"No ratings.csv found in {data_path}")

    print(f"Loading ratings...")
    df = pd.read_csv(ratings_path)
    print(f"  {len(df)} interactions from {df['userId'].nunique()} users")

    # Convert to expected format
    reviews = df.rename(columns={
        "userId": "user_id",
        "movieId": "item_id",
    }).copy()
    reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], unit="s")
    reviews = reviews.sort_values(["user_id", "timestamp"])

    print(f"\nTotal: {len(reviews)} interactions from {reviews['user_id'].nunique()} users")

    # Filter users with minimum interactions
    user_counts = reviews.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_interactions].index
    reviews = reviews[reviews["user_id"].isin(valid_users)]

    print(f"After filtering ({min_interactions}+ interactions): {len(reviews)} interactions from {reviews['user_id'].nunique()} users")

    # Build item vocabulary
    all_items = set()
    for split_name in ["train", "val", "test"]:
        split_path = output_path / f"{split_name}_samples.parquet"
        if split_path.exists():
            d = pd.read_parquet(split_path)
            all_items.update(d["target_item_id"].unique())

    if not all_items:
        all_items = set(reviews["item_id"].unique())

    item2idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
    reviews["item_idx"] = reviews["item_id"].map(item2idx)

    # Split data (80/10/10) by user
    users = list(reviews["user_id"].unique())
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

    # Save samples
    for split_name, split_users in splits.items():
        print(f"\nSaving {split_name} split...")
        user_data = reviews[reviews["user_id"].isin(split_users)].copy()

        samples = []
        for user_id, group in user_data.groupby("user_id"):
            group = group.sort_values("timestamp")
            item_ids = group["item_idx"].tolist()
            timestamps = group["timestamp"].tolist()

            for i in range(1, len(item_ids)):
                samples.append({
                    "user_id": user_id,
                    "target_item_id": item_ids[i],
                    "target_timestamp": int(timestamps[i].timestamp()),
                    "num_interactions": len(group),
                })

        d = pd.DataFrame(samples)
        output_file = output_path / f"{split_name}_samples.parquet"
        d.to_parquet(output_file)
        print(f"  {len(d)} samples saved to {output_file}")

    # Save item mapping
    pd.DataFrame([
        {"item_id": item, "item_idx": idx}
        for item, idx in item2idx.items()
    ]).to_parquet(output_path / "item_mapping.parquet")

    print(f"\nPreprocessing complete!")
    print(f"  Items: {len(item2idx)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for baselines")
    parser.add_argument("--data-dir", type=str, default="./data/raw/amazon",
                        help="Path to raw data (JSONL for Amazon, ratings.csv for MovieLens)")
    parser.add_argument("--processed-dir", type=str, default="./data/processed/amazon",
                        help="Output directory")
    parser.add_argument("--dataset-type", type=str, choices=["amazon", "movielens"], default="amazon",
                        help="Dataset type: amazon (JSONL) or movielens (CSV)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Categories to process (for Amazon only)")
    parser.add_argument("--min-interactions", type=int, default=5,
                        help="Minimum interactions per user")

    args = parser.parse_args()

    if args.dataset_type == "amazon":
        preprocess_amazon(
            data_dir=args.data_dir,
            processed_dir=args.processed_dir,
            categories=args.categories,
            min_interactions=args.min_interactions,
        )
    else:
        preprocess_movielens(
            data_dir=args.data_dir,
            processed_dir=args.processed_dir,
            min_interactions=args.min_interactions,
        )
