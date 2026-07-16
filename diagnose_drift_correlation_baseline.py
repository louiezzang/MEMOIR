"""Same per-sample drift-decile diagnostic as diagnose_drift_correlation.py,
but for ID-based / text-aware baselines, to check whether the decile-0
(extreme low-novelty) NDCG spike seen for MEMOIR is dataset-inherent (i.e.
also present for baselines) or MEMOIR-specific.

Usage:
    PYTHONPATH=. uv run python diagnose_drift_correlation_baseline.py --model sasrec
    PYTHONPATH=. uv run python diagnose_drift_correlation_baseline.py --model sasrec_text
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from baselines.train_baseline import SequentialDataset, collate_fn
from baselines import get_baseline


def evaluate_baseline_per_sample(
    model,
    dataloader,
    samples_df: pd.DataFrame,
    k: int = 10,
    num_neg: int = 999,
    eval_seed: int = 42,
) -> list[dict]:
    """Same protocol as evaluate_baseline() in train_baseline.py, but returns
    one record per sample (with user_id) instead of aggregating."""
    model.eval()
    num_items = model.num_items
    device = next(model.parameters()).device

    rng = torch.Generator()
    rng.manual_seed(eval_seed)

    records: list[dict] = []
    ptr = 0
    with torch.no_grad():
        for batch in dataloader:
            item_seq = batch["item_seq"].to(device)
            seq_len = batch["seq_len"].to(device)
            target = batch["target"].to(device)
            B = target.shape[0]
            batch_user_ids = samples_df.iloc[ptr:ptr + B]["user_id"].tolist()
            ptr += B

            user_repr = model(item_seq, seq_len)
            all_scores = model.get_scores(user_repr)

            for i in range(B):
                t = target[i].item()
                if t == 0:
                    continue
                if torch.isnan(all_scores[i, t]):
                    continue
                neg_items = []
                while len(neg_items) < num_neg:
                    cands = torch.randint(1, num_items + 1, (num_neg * 2,), generator=rng)
                    cands = cands[cands != t].tolist()
                    neg_items.extend(cands)
                neg_items = neg_items[:num_neg]

                test_items = [t] + neg_items
                scores = all_scores[i, test_items].cpu().numpy()
                rank = int((scores > scores[0]).sum()) + 1

                records.append({
                    "user_id": batch_user_ids[i],
                    f"ndcg@{k}": 1.0 / math.log2(rank + 1) if rank <= k else 0.0,
                    f"hr@{k}": 1.0 if rank <= k else 0.0,
                    "rank": rank,
                })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="./configs/default.yaml")
    parser.add_argument("--processed-dir", type=str, default="./data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    data_cfg["dataset"] = args.dataset
    processed_dir = os.path.join(args.processed_dir, args.dataset)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    test_ds = SequentialDataset(processed_dir, split="test", max_len=data_cfg["max_history_len"])
    num_items = test_ds.num_items

    model_cls = get_baseline(args.model)
    model_kwargs = {
        "num_items": num_items,
        "embed_dim": config["model"]["embedding_dim"],
        "dropout": config["model"]["dropout"],
    }
    if args.model in ("sasrec", "bert4rec", "cl4srec", "duorec", "cllmrec", "sasrec_text", "unisrec"):
        model_kwargs["max_len"] = data_cfg["max_history_len"]
        model_kwargs["num_heads"] = 2
        model_kwargs["num_layers"] = 2
    if args.model == "gru4rec":
        model_kwargs["hidden_dim"] = 128
        model_kwargs["num_layers"] = 1
    if args.model in ("cl4srec", "duorec"):
        model_kwargs["cl_temperature"] = config["model"]["temperature"]
        model_kwargs["cl_weight"] = 0.1

    model = model_cls(**model_kwargs).to(device)
    ckpt = Path(args.checkpoint_dir) / f"{args.model}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn,
    )

    records = evaluate_baseline_per_sample(
        model, test_loader, test_ds.samples, k=args.k,
        eval_seed=config["eval"].get("eval_seed", 42),
    )
    per_sample_df = pd.DataFrame(records)
    print(f"\n[{args.model}] Evaluated {len(per_sample_df)} test samples")

    drift_df = pd.read_parquet(Path(processed_dir) / "user_drift_analysis.parquet")
    merged = per_sample_df.merge(drift_df, on="user_id", how="inner")
    print(f"[{args.model}] Merged {len(merged)} samples with drift metadata")

    ndcg_col = f"ndcg@{args.k}"
    spearman = merged[ndcg_col].corr(merged["composite_drift_score"], method="spearman")
    print(f"[{args.model}] Spearman correlation (NDCG@{args.k} vs composite_drift_score): {spearman:.4f}")

    merged["decile"] = pd.qcut(merged["composite_drift_score"], 10, labels=False, duplicates="drop")
    decile_table = merged.groupby("decile").agg(
        n=("user_id", "count"),
        mean_drift=("composite_drift_score", "mean"),
        mean_ndcg=(ndcg_col, "mean"),
        mean_hr=(f"hr@{args.k}", "mean"),
        perfect_rank_frac=("rank", lambda r: (r == 1).mean()),
    )
    print(f"\n[{args.model}] Decile breakdown (decile 0 = lowest drift, 9 = highest drift):")
    print(decile_table.to_string(float_format=lambda x: f"{x:.4f}"))

    out_path = Path(processed_dir) / f"drift_correlation_per_sample_{args.model}.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\n[{args.model}] Saved to {out_path}")


if __name__ == "__main__":
    main()
