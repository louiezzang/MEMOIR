"""Amazon Reviews dataset loading and preprocessing with temporal segmentation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .temporal import TemporalBehaviorSegmenter, BehaviorWindow


def _process_user_windows(args):
    """Process a single user's windows - for multiprocessing."""
    user_id, group, segmenter, max_history_len, num_windows = args
    if len(group) < 2:
        return None
    group = group.tail(max_history_len)
    windows = segmenter.segment(user_id, group)
    if len(windows) >= 2:
        return (user_id, windows[-num_windows:])
    return None


def _process_split_user(args):
    """Process a single user for split saving - for multiprocessing."""
    uid, windows = args
    last_window = windows[-1]

    # Vectorize: convert interactions list to DataFrame then to records
    if not last_window.interactions:
        return []

    df = pd.DataFrame(last_window.interactions)
    df["user_id"] = uid
    df["num_windows"] = len(windows)
    df = df.rename(columns={
        "item_id": "target_item_id",
        "title": "target_title",
        "rating": "target_rating"
    })
    # Reorder columns and return as list of dicts
    return df[["user_id", "target_item_id", "target_title", "target_rating", "num_windows"]].to_dict("records")


def _save_user_windows(args):
    """Save a single user's windows - for multiprocessing."""
    uid, windows = args
    records = []
    for w in windows:
        records.append({
            "user_id": uid,
            "window_idx": w.window_idx,
            "start_time": str(w.start_time),
            "end_time": str(w.end_time),
            "text": w.text,
            "num_interactions": len(w.interactions),
            "interactions": json.dumps(w.interactions, default=str),
        })
    return records


class AmazonReviewDataset(Dataset):
    """Amazon Reviews dataset with temporal behavior windows.

    Expected raw data format: JSONL files from Amazon Reviews 2023
    (https://amazon-reviews-2023.github.io/)
    """

    CATEGORIES = [
        "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
        "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
        "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
        "Digital_Music", "Gift_Cards", "Grocery_and_Gourmet_Food",
        "Handmade_Products", "Health_and_Household", "Home_and_Kitchen",
        "Industrial_and_Scientific", "Kindle_Store", "Magazine_Subscriptions",
        "Movies_and_TV", "Musical_Instruments", "Office_Products",
        "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors",
        "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games",
        "Video_Games",
    ]

    def __init__(
        self,
        data_dir: str = "./data/raw/amazon",
        processed_dir: str = "./data/processed/amazon",
        categories: list[str] | None = None,
        window_type: str = "monthly",
        min_interactions: int = 5,
        max_history_len: int = 50,
        num_windows: int = 6,
        max_users: int = 100000,
        split: str = "train",
        force_regenerate: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.categories = categories or ["Electronics", "Clothing_Shoes_and_Jewelry"]
        self.min_interactions = min_interactions
        self.max_history_len = max_history_len
        self.num_windows = num_windows
        self.max_users = max_users  # For limiting users during preprocessing
        self.split = split
        self.segmenter = TemporalBehaviorSegmenter(window_type, min_interactions=2)
        self.force_regenerate = force_regenerate

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load from cache or process
        self._load_or_process()

    def _get_cache_info(self) -> dict[str, Path]:
        """Get paths to all cache files."""
        return {
            "train_samples": self.processed_dir / "train_samples.parquet",
            "val_samples": self.processed_dir / "val_samples.parquet",
            "test_samples": self.processed_dir / "test_samples.parquet",
            "windows": self.processed_dir / "windows.parquet",
            "metadata": self.processed_dir / "_metadata.json",
        }

    def _check_cache_valid(self) -> tuple[bool, list[str]]:
        """Check if cache is valid and return missing/invalid caches.

        Returns:
            Tuple of (is_valid, list_of_missing_or_invalid_caches)
        """
        cache_info = self._get_cache_info()
        metadata_path = cache_info["metadata"]

        # If force_regenerate, skip cache entirely
        if self.force_regenerate:
            return False, ["all"]

        # Check if metadata exists and is valid
        if not metadata_path.exists():
            # No metadata means legacy cache or no preprocessing done
            # Check if we have all required files (including windows.parquet)
            if not cache_info["windows"].exists():
                # No windows.parquet - need to generate everything
                return False, ["all"]

            # We have windows.parquet but no metadata - this is legacy cache
            # All splits should exist from previous runs; use them directly
            missing_caches = []
            for split in ["train", "val", "test"]:
                if not cache_info[f"{split}_samples"].exists():
                    missing_caches.append(f"{split}_samples")

            if len(missing_caches) == 0:
                return True, []  # All files exist, use them
            else:
                # Some split samples missing - need to regenerate those only
                return False, missing_caches

        with open(metadata_path) as f:
            stored_metadata = json.load(f)

        # Verify parameter hash matches current config
        current_params = {
            "window_type": self.segmenter.freq,
            "min_interactions": self.min_interactions,
            "max_history_len": self.max_history_len,
            "num_windows": self.num_windows,
            "categories": sorted(self.categories),
        }
        current_hash = hashlib.md5(json.dumps(current_params, sort_keys=True).encode()).hexdigest()

        if stored_metadata.get("param_hash") != current_hash:
            # Parameters changed - need to regenerate everything
            return False, ["windows", "train_samples", "val_samples", "test_samples"]

        # Check which sample caches exist
        missing_caches = []
        for split in ["train", "val", "test"]:
            if not cache_info[f"{split}_samples"].exists():
                missing_caches.append(f"{split}_samples")

        return len(missing_caches) == 0, missing_caches

    def _save_metadata(self):
        """Save processing metadata for cache validation."""
        cache_info = self._get_cache_info()
        metadata_path = cache_info["metadata"]

        current_params = {
            "window_type": self.segmenter.freq,
            "min_interactions": self.min_interactions,
            "max_history_len": self.max_history_len,
            "num_windows": self.num_windows,
            "categories": sorted(self.categories),
        }
        param_hash = hashlib.md5(json.dumps(current_params, sort_keys=True).encode()).hexdigest()

        metadata = {
            "param_hash": param_hash,
            "processed_at": pd.Timestamp.now().isoformat(),
            "user_counts": len(getattr(self, "user_windows", {})),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_or_process(self):
        """Load from cache if valid, otherwise process data."""
        is_valid, missing_caches = self._check_cache_valid()

        if is_valid:
            print("  All cache files found and valid, loading...")
            self.user_windows = self._load_windows()
            self.samples = pd.read_parquet(
                self.processed_dir / f"{self.split}_samples.parquet"
            )
            return

        # Detect CPU count and set workers based on platform
        n_workers = cpu_count()
        print(f"n_cpus = {n_workers}")
        import platform
        n_workers = min(n_workers, 16)
        self.n_workers = n_workers
        print(f"{n_workers} workers for multiprocessing...")

        self.TQDM_AVAILABLE = tqdm is not None

        # Determine processing strategy based on what's missing
        windows_cache = self._get_cache_info()["windows"]

        any_samples_missing = any(c.endswith("_samples") for c in missing_caches)
        if windows_cache.exists() and any_samples_missing:
            # Check the cache has interaction data (older caches omitted it)
            import pyarrow.parquet as pq
            cached_columns = pq.read_schema(windows_cache).names
            if "interactions" not in cached_columns:
                print("  Windows cache missing interaction data, reprocessing from scratch...")
                self._process()
            else:
                # Windows exist and have interactions, only regenerate split samples
                print("  Windows cache found, regenerating split samples only...")
                self.user_windows = self._load_windows()
                print(f"  Loaded {len(self.user_windows)} users with windows")
                self._regenerate_split_samples(missing_caches)
        else:
            # Full processing needed
            self._process()

    def _process(self):
        """Load raw data, filter users, create temporal windows, and split."""
        # tqdm is already imported at module level (may be None)
        self.TQDM_AVAILABLE = tqdm is not None

        print("Loading Amazon dataset...")
        dfs = []
        for cat in self.categories:
            path = self.data_dir / f"{cat}.jsonl"
            if not path.exists():
                print(f"  Warning: {path} not found, skipping")
                continue
            print(f"  Loading {cat}...")
            df = self._load_category(path, cat)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"No data found in {self.data_dir}. Download Amazon Reviews 2023 first.\n"
                f"See: https://amazon-reviews-2023.github.io/"
            )

        reviews = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(reviews)} total interactions")

        reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], unit="ms")
        reviews = reviews.sort_values(["user_id", "timestamp"])

        user_counts = reviews.groupby("user_id").size()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        reviews = reviews[reviews["user_id"].isin(valid_users)]
        print(f"  After filtering ({self.min_interactions}+ interactions): {len(reviews)} interactions from {valid_users.nunique()} users")

        # Filter to top N most frequent users for faster processing (default: max_users)
        if len(valid_users) > self.max_users:
            print(f"  Filtering to top {self.max_users} most frequent users for faster processing...")
            top_user_counts = user_counts.nlargest(self.max_users)
            valid_users = top_user_counts.index
            reviews = reviews[reviews["user_id"].isin(valid_users)]
            print(f"    Now working with {len(reviews)} interactions from {valid_users.nunique()} users")

        # Use multiprocessing to create windows in parallel
        print(f"  Creating temporal windows using {self.n_workers} processes...")

        # Prepare arguments for each user
        args_list = []
        for user_id in valid_users:
            group = reviews[reviews["user_id"] == user_id]
            args_list.append((user_id, group, self.segmenter, self.max_history_len, self.num_windows))

        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(pool.imap(_process_user_windows, args_list), total=len(args_list),
                              desc="Creating temporal windows") if self.TQDM_AVAILABLE else
                        pool.imap(_process_user_windows, args_list))

        # Filter None results and build user_windows dict
        self.user_windows: dict[str, list[BehaviorWindow]] = {}
        for result in results:
            if result is not None and isinstance(result, tuple) and len(result) == 2:
                user_id, windows = result
                self.user_windows[user_id] = windows

        print(f"  Created windows for {len(self.user_windows)} users (>= 2 windows)")

        # Save windows first
        self._save_windows()
        print("  Saved windows.parquet")

        # Then save splits
        self._split_and_save(reviews)

        # Save metadata after successful processing
        self._save_metadata()

    def _regenerate_split_samples(self, missing_caches: list[str]):
        """Regenerate only split samples using existing windows."""
        print("  Regenerating split samples...")

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
            # Skip if this split exists and is not in missing_caches
            if f"{split_name}_samples" not in missing_caches:
                print(f"    Skipping {split_name} (already exists)")
                continue

            print(f"  Processing {split_name} split ({len(split_users)} users)...")

            # Prepare args for multiprocessing
            args_list = [(uid, self.user_windows[uid]) for uid in split_users]

            # Use larger chunksize to reduce IPC overhead
            chunksize = max(1, len(args_list) // (self.n_workers * 10))
            with Pool(processes=self.n_workers) as pool:
                results = list(tqdm(pool.imap(_process_split_user, args_list, chunksize=chunksize),
                                  total=len(args_list), desc=f"Processing {split_name}")
                              if self.TQDM_AVAILABLE else
                            pool.imap(_process_split_user, args_list, chunksize=chunksize))

            # Flatten results
            samples = []
            for user_samples in results:
                samples.extend(user_samples)

            df = pd.DataFrame(samples)
            output_path = self.processed_dir / f"{split_name}_samples.parquet"
            df.to_parquet(output_path)
            print(f"    Saved {len(df)} samples to {output_path.name}")

        # Load back for this dataset instance
        self.user_windows = self._load_windows()
        self.samples = pd.read_parquet(self.processed_dir / f"{self.split}_samples.parquet")

    def _split_and_save(self, reviews: pd.DataFrame):
        """Legacy: kept for compatibility with full processing path."""
        # This is now called only from _process() after window creation
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
            print(f"  Saving {split_name} split ({len(split_users)} users)...")

            # Prepare args for multiprocessing
            args_list = [(uid, self.user_windows[uid]) for uid in split_users]

            chunksize = max(1, len(args_list) // (self.n_workers * 10))
            with Pool(processes=self.n_workers) as pool:
                results = list(tqdm(pool.imap(_process_split_user, args_list, chunksize=chunksize),
                                  total=len(args_list), desc=f"Processing {split_name}")
                              if self.TQDM_AVAILABLE else
                            pool.imap(_process_split_user, args_list, chunksize=chunksize))

            samples = []
            for user_samples in results:
                samples.extend(user_samples)

            df = pd.DataFrame(samples)
            df.to_parquet(self.processed_dir / f"{split_name}_samples.parquet")
            print(f"    Saved {len(df)} samples")

        self._save_windows()

        # Load back for this dataset instance
        self.user_windows = self._load_windows()
        self.samples = pd.read_parquet(self.processed_dir / f"{self.split}_samples.parquet")

    def _load_category(self, path: Path, category: str) -> pd.DataFrame:
        meta_map = self._load_meta_categories(category)
        records = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                asin = r["parent_asin"]
                meta = meta_map.get(asin, {})
                records.append({
                    "user_id": r["user_id"],
                    "item_id": asin,
                    "rating": r.get("rating", 0.0),
                    # Product title from item metadata, NOT the review's own
                    # headline (r["title"] here is reviewer-written text like
                    # "Five Stars" or "Didn't work", not the product's name).
                    "title": meta.get("title", ""),
                    "category": meta.get("category", category),
                    "timestamp": r.get("timestamp", 0),
                })
        return pd.DataFrame(records)

    def _load_meta_categories(self, category: str) -> dict[str, dict]:
        """Maps asin -> {"category": ..., "title": ...} from item metadata.

        Uses the *item's* title from meta_{category}.jsonl, not the review's
        own headline text (which is what the per-review jsonl's "title" field
        actually contains).
        """
        meta_path = self.data_dir / f"raw/meta_categories/meta_{category}.jsonl"
        json_path = self.data_dir / "asin_categories.json"
        if json_path.exists():
            with open(json_path) as f:
                cached = json.load(f)
            # Only trust this cache if it already has the {category, title} shape;
            # an older category-only cache would silently drop titles otherwise.
            sample = next(iter(cached.values()), None)
            if isinstance(sample, dict) and "title" in sample:
                return cached
        if not meta_path.exists():
            return {}
        mapping = {}
        with open(meta_path) as f:
            for line in f:
                r = json.loads(line)
                cats = r.get("categories", [])
                cat = cats[-1] if len(cats) > 1 else (cats[0] if cats else category)
                mapping[r["parent_asin"]] = {"category": cat, "title": r.get("title", "")}
        return mapping

    def _save_windows(self):
        """Save windows with multiprocessing."""
        from itertools import chain

        # Prepare args for multiprocessing: each (uid, windows) pair
        args_list = [(uid, self.user_windows[uid]) for uid in self.user_windows.keys()]

        with Pool(processes=self.n_workers) as pool:
            results = list(pool.imap(_save_user_windows, args_list))

        # Flatten and save
        records = list(chain.from_iterable(results))
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
                interactions = json.loads(row["interactions"]) if "interactions" in row.index else []
                wins.append(BehaviorWindow(
                    user_id=uid,
                    window_idx=row["window_idx"],
                    start_time=pd.Timestamp(row["start_time"]),
                    end_time=pd.Timestamp(row["end_time"]),
                    interactions=interactions,
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
        # Exclude last window: targets are sampled from windows[-1],
        # so including it leaks the target item into the input.
        history_windows = windows[:-1] if len(windows) > 1 else windows
        window_texts = [w.text for w in history_windows]

        return {
            "user_id": uid,
            "window_texts": window_texts,
            "target_item_id": row["target_item_id"],
            "target_title": row.get("target_title", ""),
            "target_rating": row.get("target_rating", 0.0),
        }
