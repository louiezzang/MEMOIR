#!/usr/bin/env python3
"""Preprocess Amazon dataset for MEMOIR (creates windows.parquet)."""

import argparse
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.amazon import AmazonReviewDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Amazon dataset")
    parser.add_argument("--max-users", type=int, default=100000,
                        help="Maximum users to process (default: 100k)")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Force regenerate all files, bypassing cache")
    args = parser.parse_args()

    # Create dataset with max_users passed during initialization
    dataset = AmazonReviewDataset(
        data_dir='./data/raw/amazon',
        processed_dir='./data/processed/amazon',
        categories=['Electronics', 'Clothing_Shoes_and_Jewelry'],
        window_type='monthly',
        min_interactions=5,
        max_history_len=50,
        num_windows=4,
        max_users=args.max_users,
        force_regenerate=args.force_regenerate
    )

    print(f'Created dataset with {len(dataset)} samples')
