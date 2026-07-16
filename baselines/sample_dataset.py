"""Sample dataset to reduce number of users for faster training."""

import argparse
from pathlib import Path

import pandas as pd


def sample_dataset(
    data_dir: str,
    processed_dir: str,
    num_users: int = 100000,
    seed: int = 42,
):
    """Sample a dataset by keeping only N users.

    Args:
        data_dir: Path to preprocessed data directory
        processed_dir: Output directory for sampled data
        num_users: Number of users to keep (most frequent)
        seed: Random seed for reproducibility
    """
    import numpy as np

    data_path = Path(data_dir)
    output_path = Path(processed_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all splits and find top users
    splits = ["train", "val", "test"]
    dfs = {}
    for split in splits:
        path = data_path / f"{split}_samples.parquet"
        if path.exists():
            dfs[split] = pd.read_parquet(path)
            print(f"Loaded {split}: {len(dfs[split])} samples, {dfs[split]['user_id'].nunique()} users")

    if not dfs:
        raise ValueError("No data files found")

    # Find top users by frequency across all splits
    all_users = pd.concat([df['user_id'] for df in dfs.values()])
    user_counts = all_users.value_counts()
    print(f"\nTotal unique users: {len(user_counts)}")
    print(f"Top user has {user_counts.iloc[0]} samples")

    # Get top N users
    top_users = set(user_counts.head(num_users).index.tolist())
    print(f"Keeping {len(top_users)} most frequent users")

    # Sample each split
    for split in splits:
        if split not in dfs:
            continue

        df = dfs[split]
        sampled = df[df['user_id'].isin(top_users)].copy()
        original_len = len(df)
        sampled_len = len(sampled)

        print(f"\n{split}: {original_len} -> {sampled_len} samples ({100*sampled_len/original_len:.1f}%)")

        # Save
        output_file = output_path / f"{split}_samples.parquet"
        sampled.to_parquet(output_file)
        print(f"  Saved to {output_file}")

    # Copy item mapping (unchanged since we kept same items)
    item_mapping = data_path / "item_mapping.parquet"
    if item_mapping.exists():
        import shutil
        shutil.copy(item_mapping, output_path / "item_mapping.parquet")
        print("\nCopied item mapping")

    # Copy windows.parquet if it exists (for MEMOIR compatibility)
    windows_path = data_path / "windows.parquet"
    if windows_path.exists():
        shutil.copy(windows_path, output_path / "windows.parquet")
        print("Copied windows.parquet (preserved for MEMOIR)")

    print(f"\nSampling complete!")
    print(f"Output: {output_path}")
    print(f"Users: {len(top_users)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample dataset by user count")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to preprocessed data directory")
    parser.add_argument("--processed-dir", type=str, required=True,
                        help="Output directory for sampled data")
    parser.add_argument("--num-users", type=int, default=100000,
                        help="Number of users to keep (default: 100k)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    sample_dataset(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        num_users=args.num_users,
        seed=args.seed,
    )
