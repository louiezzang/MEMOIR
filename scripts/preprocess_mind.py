#!/usr/bin/env python3
"""Preprocess MIND dataset for MEMOIR (creates windows.parquet)."""

from data.mind import MINDDataset

if __name__ == "__main__":
    dataset = MINDDataset(
        data_dir='./data/raw/mind',
        processed_dir='./data/processed/mind',
        size='small',  # 'small' or 'large'
        window_type='weekly',
        min_interactions=5,
        max_history_len=50,
        num_windows=4
    )
    print(f'Created dataset with {len(dataset)} samples')
