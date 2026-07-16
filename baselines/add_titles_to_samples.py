"""Add movie titles to preprocessed samples for MEMOIR compatibility."""

import argparse
from pathlib import Path

import pandas as pd


def add_titles(data_dir: str, processed_dir: str):
    """Add movie titles to existing preprocessed samples.

    Args:
        data_dir: Raw MovieLens directory (contains ml-25m/movies.csv)
        processed_dir: Preprocessed samples directory
    """
    raw_path = Path(data_dir)
    output_path = Path(processed_dir)

    # Find movies.csv in ml-25m or ml-100k
    movies_path = None
    for subdir in ["ml-25m", "ml-100k"]:
        path = raw_path / subdir / "movies.csv"
        if path.exists():
            movies_path = path
            break
        # For ml-100k, get u.item
        path = raw_path / subdir / "u.item"
        if path.exists():
            movies_path = path
            break

    if not movies_path:
        raise FileNotFoundError(f"No movie data found in {raw_path}")

    print(f"Loading movie data from {movies_path}")

    # Load movie info based on format
    if movies_path.name == "movies.csv":
        movies = pd.read_csv(movies_path)
        movie_info = dict(zip(
            movies["movieId"],
            zip(movies["title"], movies.get("genres", ""))
        ))
    else:
        # ml-100k u.item format
        genres = []
        with open(raw_path / "ml-100k" / "u.genre", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    name, _ = line.split("|")
                    genres.append(name)

        movie_info = {}
        with open(movies_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                mid = int(parts[0])
                title = parts[1]
                genre_flags = [int(g) for g in parts[5:]]
                cats = [genres[i] for i, flag in enumerate(genre_flags) if flag and i < len(genres)]
                movie_info[mid] = (title, "|".join(cats) if cats else "unknown")

    print(f"Loaded {len(movie_info)} movies")

    # Load and update each split
    splits = ["train", "val", "test"]
    for split in splits:
        samples_path = output_path / f"{split}_samples.parquet"
        if not samples_path.exists():
            print(f"Skipping {split} - not found")
            continue

        print(f"\nProcessing {split}...")
        df = pd.read_parquet(samples_path)
        print(f"  Original columns: {list(df.columns)}")

        # Add titles from item_mapping if available
        mapping_path = output_path / "item_mapping.parquet"
        if mapping_path.exists():
            mapping = pd.read_parquet(mapping_path)
            # Create a mapping from item_id to item_idx
            item_to_idx = dict(zip(mapping["item_id"], mapping["item_idx"]))

            # Map idx back to original item_id for lookup
            idx_to_item = {v: k for k, v in item_to_idx.items()}

            # Get titles for target_item_id (assuming it's the original item_id)
            df["target_title"] = df["target_item_id"].map(lambda x: movie_info.get(x, ("", ""))[0])
        else:
            # If no mapping, try direct lookup
            df["target_title"] = df["target_item_id"].map(lambda x: movie_info.get(x, ("", ""))[0])

        print(f"  Missing titles: {df['target_title'].isna().sum()}")
        df["target_title"] = df["target_title"].fillna("")

        # Save with new column
        output_file = output_path / f"{split}_samples.parquet"
        df.to_parquet(output_file)
        print(f"  Saved to {output_file}")

    print("\nDone! Added target_title column to all samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add titles to preprocessed samples")
    parser.add_argument("--data-dir", type=str, default="./data/raw/movielens",
                        help="Raw MovieLens directory")
    parser.add_argument("--processed-dir", type=str, default="./data/processed/movielens",
                        help="Preprocessed samples directory")

    args = parser.parse_args()

    add_titles(args.data_dir, args.processed_dir)
