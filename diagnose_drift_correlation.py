"""Diagnostic: correlate per-user NDCG@10 against the *continuous* composite
drift score, instead of only comparing high/medium/low tercile averages.

Purpose: check whether MEMOIR's low-drift > high-drift NDCG@10 gap (observed
via analyze_drift.py --evaluate) is a smooth, real trend across the full drift
range, or an artifact concentrated at the tercile boundaries / a small subset
of outlier users.

Usage:
    PYTHONPATH=. uv run python diagnose_drift_correlation.py --dataset amazon
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_dataset, MEMOIRCollator
from model import MEMOIRModel
from eval import build_item_base_embeddings, apply_item_projection, ndcg, hit_rate


def evaluate_per_sample(
    model,
    dataloader,
    k: int = 10,
    num_neg: int = 999,
    precomputed: tuple[list[str], torch.Tensor] | None = None,
    eval_seed: int = 42,
) -> list[dict]:
    """Same protocol as eval.evaluate(), but returns one record per sample
    instead of aggregating, so results can be joined against per-user metadata."""
    model.eval()
    device = next(model.parameters()).device
    unique_titles, item_embeds_cpu = precomputed
    title_to_idx = {t: i for i, t in enumerate(unique_titles)}
    N_items = len(unique_titles)

    rng = torch.Generator()
    rng.manual_seed(eval_seed)
    item_embeds = item_embeds_cpu.to(device)

    records: list[dict] = []
    for batch in tqdm(dataloader, desc="Evaluating per-sample"):
        batch_on_device = {
            kk: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for kk, v in batch.items()
        }
        with torch.no_grad():
            output = model(batch_on_device)
            user_repr = F.normalize(output["user_memory"], dim=-1)

        B = user_repr.shape[0]
        target_idxs = [title_to_idx.get(batch["target_titles"][i]) for i in range(B)]

        cand_list, valid_mask = [], []
        for i in range(B):
            t = target_idxs[i]
            if t is None:
                valid_mask.append(False)
                cand_list.append(torch.zeros(1 + num_neg, dtype=torch.long))
                continue
            negs = torch.randint(0, N_items, (num_neg + 32,), generator=rng)
            negs = negs[negs != t][:num_neg]
            cand_list.append(torch.cat([torch.tensor([t]), negs]))
            valid_mask.append(True)

        cand_indices = torch.stack(cand_list).to(device)
        cand_embeds = item_embeds[cand_indices]
        with torch.no_grad():
            scores = torch.bmm(
                user_repr.unsqueeze(1), cand_embeds.transpose(1, 2)
            ).squeeze(1)
        ranked = torch.argsort(scores, dim=1, descending=True).cpu().numpy()

        gt = {0}
        for i, r in enumerate(ranked):
            if not valid_mask[i]:
                continue
            records.append({
                "user_id": batch["user_ids"][i],
                f"ndcg@{k}": ndcg(r, gt, k),
                f"hr@{k}": hit_rate(r, gt, k),
                "rank": int(np.where(r == 0)[0][0]) + 1,
            })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--config", type=str, default="./configs/default.yaml")
    parser.add_argument("--processed-dir", type=str, default="./data/processed")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["data"]["dataset"] = args.dataset
    config["data"]["processed_dir"] = args.processed_dir

    data_cfg = config["data"]
    model_cfg = config["model"]

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ckpt_state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    inferred_dim = ckpt_state["memory_encoder.projection.2.weight"].shape[0]
    if inferred_dim != model_cfg.get("embedding_dim"):
        print(f"checkpoint embedding_dim={inferred_dim}, overriding config value {model_cfg.get('embedding_dim')}")
        config = {**config, "model": {**model_cfg, "embedding_dim": inferred_dim}}
        model_cfg = config["model"]

    ds_kwargs = dict(
        data_dir=os.path.join(data_cfg["data_dir"], data_cfg["dataset"]),
        processed_dir=os.path.join(data_cfg["processed_dir"], data_cfg["dataset"]),
        window_type=data_cfg["time_window"],
        min_interactions=data_cfg["min_interactions"],
        max_history_len=data_cfg["max_history_len"],
        num_windows=model_cfg["num_memory_windows"],
    )
    test_ds = build_dataset(data_cfg["dataset"], split="test", **ds_kwargs)
    collator = MEMOIRCollator(max_windows=model_cfg["num_memory_windows"], num_negatives=4)
    test_loader = DataLoader(
        test_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, collate_fn=collator,
    )

    model = MEMOIRModel(config).to(device)
    model.load_state_dict(ckpt_state)

    if model_cfg.get("freeze_llm", False):
        llm_tag = model_cfg["llm_name"].replace("/", "_")
        cache_path = (
            Path(data_cfg["processed_dir"]) / data_cfg["dataset"]
            / f"pooled_cache_{llm_tag}_d{model_cfg['embedding_dim']}.pt"
        )
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            model.memory_encoder.set_text_cache(cache)

    processed_dir = Path(data_cfg["processed_dir"]) / data_cfg["dataset"]
    all_titles: set[str] = set()
    for split in ("train", "val", "test"):
        p = processed_dir / f"{split}_samples.parquet"
        if p.exists():
            all_titles.update(pd.read_parquet(p, columns=["target_title"])["target_title"].dropna())
    item_titles, base_embeds = build_item_base_embeddings(model, sorted(all_titles))
    item_embeds = apply_item_projection(model, base_embeds)

    records = evaluate_per_sample(
        model, test_loader, k=args.k,
        precomputed=(item_titles, item_embeds),
        eval_seed=config["eval"].get("eval_seed", 42),
    )
    per_sample_df = pd.DataFrame(records)
    print(f"\nEvaluated {len(per_sample_df)} test samples")

    drift_df = pd.read_parquet(processed_dir / "user_drift_analysis.parquet")
    merged = per_sample_df.merge(drift_df, on="user_id", how="inner")
    print(f"Merged {len(merged)} samples with drift metadata")

    ndcg_col = f"ndcg@{args.k}"
    spearman = merged[ndcg_col].corr(merged["composite_drift_score"], method="spearman")
    pearson = merged[ndcg_col].corr(merged["composite_drift_score"], method="pearson")
    print(f"\nSpearman correlation (NDCG@{args.k} vs composite_drift_score): {spearman:.4f}")
    print(f"Pearson correlation:  {pearson:.4f}")

    print("\nDecile breakdown (decile 0 = lowest drift, 9 = highest drift):")
    merged["decile"] = pd.qcut(merged["composite_drift_score"], 10, labels=False, duplicates="drop")
    decile_table = merged.groupby("decile").agg(
        n=("user_id", "count"),
        mean_drift=("composite_drift_score", "mean"),
        mean_ndcg=(ndcg_col, "mean"),
        mean_hr=(f"hr@{args.k}", "mean"),
        mean_rank=("rank", "mean"),
    )
    print(decile_table.to_string(float_format=lambda x: f"{x:.4f}"))

    out_path = processed_dir / "drift_correlation_per_sample.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nPer-sample results saved to {out_path}")


if __name__ == "__main__":
    main()
