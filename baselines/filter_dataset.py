"""Filter dataset to top-K most frequent items for efficiency."""

import argparse
from pathlib import Path

import pandas as pd


def filter_dataset(
    data_dir: str,
    processed_dir: str,
    top_k: int = 100000,
):
    """Filter a preprocessed dataset to top-K most frequent items.

    Args:
        data_dir: Path to raw data directory
        processed_dir: Output directory for filtered data
        top_k: Number of most frequent items to keep
    """
    data_path = Path(data_dir)
    output_path = Path(processed_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all splits
    splits = ["train", "val", "test"]
    dfs = {}
    for split in splits:
        path = data_path / f"{split}_samples.parquet"
        if path.exists():
            dfs[split] = pd.read_parquet(path)
            print(f"Loaded {split}: {len(dfs[split])} samples")
        else:
            print(f"Warning: {path} not found")

    if not dfs:
        raise ValueError("No data files found")

    # Combine to find top items across all splits
    all_samples = pd.concat(list(dfs.values()), ignore_index=True)
    item_counts = all_samples['target_item_id'].value_counts()

    print(f"\nTotal unique items: {len(item_counts)}")
    print(f"Top {top_k}th item count: {item_counts.iloc[top_k-1] if top_k <= len(item_counts) else 'N/A'}")

    # Get top K items
    top_items = set(item_counts.head(top_k).index.tolist())
    print(f"Filtering to top {len(top_items)} items")

    # Filter each split
    for split, df in dfs.items():
        filtered = df[df['target_item_id'].isin(top_items)].copy()
        print(f"{split}: {len(df)} -> {len(filtered)} samples after filtering")

        # Save filtered data
        output_file = output_path / f"{split}_samples.parquet"
        filtered.to_parquet(output_file)
        print(f"  Saved to {output_file}")

    # Build new item mapping (1-indexed, consecutive)
    items_sorted = sorted(top_items)
    item2idx = {item: idx + 1 for idx, item in enumerate(items_sorted)}

    # Remap item IDs
    for split in splits:
        if split in dfs:
            df = pd.read_parquet(output_path / f"{split}_samples.parquet")
            df['target_item_id'] = df['target_item_id'].map(item2idx)
            df.to_parquet(output_path / f"{split}_samples.parquet")

    # Save item mapping
    mapping_df = pd.DataFrame([
        {"item_id": item, "item_idx": idx}
        for item, idx in item2idx.items()
    ])
    mapping_df.to_parquet(output_path / "item_mapping.parquet")
    print(f"\nSaved item mapping: {len(item2idx)} items")

    # Copy windows.parquet if it exists (for MEMOIR compatibility)
    windows_path = data_path / "windows.parquet"
    if windows_path.exists():
        import shutil
        shutil.copy(windows_path, output_path / "windows.parquet")
        print("Copied windows.parquet (preserved for MEMOIR)")

    print("\nFiltering complete!")
    print(f"Output: {output_path}")

    return len(item2idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset to top-K items")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to preprocessed data directory")
    parser.add_argument("--processed-dir", type=str, required=True,
                        help="Output directory for filtered data")
    parser.add_argument("--top-k", type=int, default=100000,
                        help="Number of most frequent items to keep (default: 100k)")

    args = parser.parse_args()

    filter_dataset(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        top_k=args.top_k,
    )
