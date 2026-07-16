"""Preference drift analysis: quantify how much each user's preferences change over time."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

INTERACTION_RE = re.compile(
    r"(highly rated|liked|neutral about|disliked|strongly disliked|"
    r"purchased|viewed|clicked on|read|interacted with)"
    r'\s+"([^"]+)"'
    r"(?:\s+in\s+([^;.]+))?"
)

ACTION_TO_RATING = {
    "highly rated": 5.0,
    "liked": 4.0,
    "viewed": 3.0,
    "neutral about": 3.0,
    "disliked": 1.0,
    "strongly disliked": 1.0,
    "purchased": None,
    "clicked on": None,
    "read": None,
    "interacted with": None,
}


def parse_window_text(text: str) -> list[dict]:
    results = []
    for action, title, raw_cat in INTERACTION_RE.findall(text):
        categories = [c.strip() for c in raw_cat.split("|")] if raw_cat else []
        results.append({
            "action": action,
            "title": title,
            "categories": categories,
            "rating": ACTION_TO_RATING.get(action),
        })
    return results


def compute_category_distribution(interactions: list[dict]) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for item in interactions:
        for cat in item["categories"]:
            counts[cat] = counts.get(cat, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {cat: cnt / total for cat, cnt in counts.items()}


def compute_category_jsd(dist1: dict[str, float], dist2: dict[str, float]) -> float:
    if not dist1 or not dist2:
        return float("nan")
    all_cats = sorted(set(dist1) | set(dist2))
    p = np.array([dist1.get(c, 0.0) for c in all_cats])
    q = np.array([dist2.get(c, 0.0) for c in all_cats])
    return float(jensenshannon(p, q))


def compute_user_metrics(user_windows: pd.DataFrame) -> dict:
    user_windows = user_windows.sort_values("window_idx")
    texts = user_windows["text"].tolist()

    if len(texts) < 2:
        return {
            "category_jsd_mean": float("nan"),
            "category_jsd_max": float("nan"),
            "rating_drift": float("nan"),
            "novelty_rate": float("nan"),
        }

    parsed_windows = [parse_window_text(t) for t in texts]

    # 1. Category JSD
    cat_dists = [compute_category_distribution(pw) for pw in parsed_windows]
    jsd_values = []
    for i in range(len(cat_dists) - 1):
        jsd = compute_category_jsd(cat_dists[i], cat_dists[i + 1])
        if not np.isnan(jsd):
            jsd_values.append(jsd)

    category_jsd_mean = float(np.mean(jsd_values)) if jsd_values else float("nan")
    category_jsd_max = float(np.max(jsd_values)) if jsd_values else float("nan")

    # 2. Rating drift
    window_mean_ratings = []
    for pw in parsed_windows:
        ratings = [item["rating"] for item in pw if item["rating"] is not None]
        if ratings:
            window_mean_ratings.append(np.mean(ratings))

    rating_drift = float(np.var(window_mean_ratings)) if len(window_mean_ratings) >= 2 else float("nan")

    # 3. Novelty rate
    seen_titles: set[str] = set()
    novelty_values = []
    for i, pw in enumerate(parsed_windows):
        titles_in_window = {item["title"] for item in pw}
        if i > 0 and titles_in_window:
            new_titles = titles_in_window - seen_titles
            novelty_values.append(len(new_titles) / len(titles_in_window))
        seen_titles.update(titles_in_window)

    novelty_rate = float(np.mean(novelty_values)) if novelty_values else float("nan")

    return {
        "category_jsd_mean": category_jsd_mean,
        "category_jsd_max": category_jsd_max,
        "rating_drift": rating_drift,
        "novelty_rate": novelty_rate,
    }


def normalize_and_composite(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["category_jsd_mean", "rating_drift", "novelty_rate"]
    weights = {"category_jsd_mean": 0.4, "rating_drift": 0.3, "novelty_rate": 0.3}

    normalized = {}
    for col in metric_cols:
        vals = df[col]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            normalized[col] = (vals - vmin) / (vmax - vmin)
        else:
            normalized[col] = pd.Series(0.0, index=df.index)

    scores = []
    for idx in df.index:
        available_w = {}
        for col in metric_cols:
            if not np.isnan(normalized[col].loc[idx]):
                available_w[col] = weights[col]
        if not available_w:
            scores.append(float("nan"))
            continue
        w_total = sum(available_w.values())
        score = sum(
            (available_w[col] / w_total) * normalized[col].loc[idx]
            for col in available_w
        )
        scores.append(score)

    df["composite_drift_score"] = scores
    return df


def classify_drift_groups(df: pd.DataFrame, high_pct: int = 75, low_pct: int = 25) -> pd.DataFrame:
    valid = df["composite_drift_score"].dropna()
    if len(valid) == 0:
        df["drift_group"] = "unknown"
        return df

    high_thresh = np.percentile(valid, high_pct)
    low_thresh = np.percentile(valid, low_pct)

    def _classify(score):
        if np.isnan(score):
            return "unknown"
        if score >= high_thresh:
            return "high"
        if score <= low_thresh:
            return "low"
        return "medium"

    df["drift_group"] = df["composite_drift_score"].apply(_classify)
    return df


def print_summary(df: pd.DataFrame, dataset: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Preference Drift Analysis: {dataset}")
    print(f"{'=' * 60}")
    print(f"  Total users: {len(df)}")

    valid = df[df["drift_group"] != "unknown"]
    print(f"  Users with valid drift scores: {len(valid)}")
    print()

    metric_cols = ["category_jsd_mean", "category_jsd_max", "rating_drift", "novelty_rate", "composite_drift_score"]
    print("  Metric Statistics:")
    print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-' * 73}")
    for col in metric_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            print(f"  {col:<25} {'N/A':>8}")
            continue
        print(
            f"  {col:<25} {vals.mean():8.4f} {vals.median():8.4f} "
            f"{vals.std():8.4f} {vals.min():8.4f} {vals.max():8.4f}"
        )

    print()
    print("  Drift Group Distribution:")
    print(f"  {'Group':<10} {'Count':>8} {'Pct':>8} {'Mean Score':>12}")
    print(f"  {'-' * 42}")
    for group in ["high", "medium", "low", "unknown"]:
        subset = df[df["drift_group"] == group]
        if len(subset) == 0:
            continue
        pct = 100.0 * len(subset) / len(df)
        mean_score = subset["composite_drift_score"].mean()
        print(f"  {group:<10} {len(subset):8d} {pct:7.1f}% {mean_score:12.4f}")

    print(f"{'=' * 60}\n")


def plot_drift_distributions(df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Preference Drift Analysis: {dataset}", fontsize=14)

    metrics = [
        ("category_jsd_mean", "Category JSD (Mean)"),
        ("rating_drift", "Rating Drift (Variance)"),
        ("novelty_rate", "Item Novelty Rate"),
        ("composite_drift_score", "Composite Drift Score"),
    ]

    colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71", "unknown": "#95a5a6"}

    for ax, (col, title) in zip(axes.flat, metrics):
        valid = df[col].dropna()
        if len(valid) == 0:
            ax.set_title(f"{title} (no data)")
            continue

        for group in ["low", "medium", "high"]:
            subset = df[(df["drift_group"] == group) & df[col].notna()][col]
            if len(subset) > 0:
                ax.hist(subset, bins=20, alpha=0.6, label=group, color=colors[group])

        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    out_path = output_dir / f"{dataset}_drift_distributions.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {out_path}")


def plot_drift_summary(df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    """Single-figure paper plot: composite drift score KDE with group boundaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}
    labels = {"high": f"High drift (top 25%, n={len(df[df.drift_group=='high']):,})",
              "medium": f"Medium drift (middle 50%, n={len(df[df.drift_group=='medium']):,})",
              "low": f"Low drift (bottom 25%, n={len(df[df.drift_group=='low']):,})"}

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.linspace(df["composite_drift_score"].min() - 0.05,
                    df["composite_drift_score"].max() + 0.05, 500)

    for group in ["low", "medium", "high"]:
        subset = df[df["drift_group"] == group]["composite_drift_score"].dropna().values
        if len(subset) < 2:
            continue
        kde = gaussian_kde(subset, bw_method=0.15)
        density = kde(x)
        ax.fill_between(x, density, alpha=0.4, color=colors[group])
        ax.plot(x, density, color=colors[group], linewidth=1.8, label=labels[group])

    # Group boundaries
    low_thresh = df[df["drift_group"] == "low"]["composite_drift_score"].max()
    high_thresh = df[df["drift_group"] == "high"]["composite_drift_score"].min()
    ymax = ax.get_ylim()[1]
    for thresh in [low_thresh, high_thresh]:
        ax.axvline(thresh, color="#555", linestyle="--", linewidth=1.2)

    median = df["composite_drift_score"].median()
    ax.axvline(median, color="#222", linestyle=":", linewidth=1.5, label=f"Median = {median:.2f}")

    ax.set_xlabel("Composite Drift Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("User Preference Drift Distribution", fontsize=12)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / f"{dataset}_drift_summary.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Summary plot saved: {out_path}")


def _evaluate_memoir_by_group(
    config: dict,
    drift_df: pd.DataFrame,
    ks: list[int],
    checkpoint_dir: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate MEMOIR test-set metrics broken down by drift group."""
    import sys
    import os
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(Path(__file__).parent))
    from data import build_dataset, MEMOIRCollator
    from model import MEMOIRModel
    from eval import evaluate, build_item_base_embeddings, apply_item_projection

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ckpt = checkpoint_dir / "best_model.pt"
    if not ckpt.exists():
        print("  [memoir] best_model.pt not found — skipping")
        return {}

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["eval"]

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

    # Infer embedding_dim from checkpoint to handle config/checkpoint mismatches
    ckpt_state = torch.load(ckpt, map_location="cpu", weights_only=True)
    inferred_dim = ckpt_state["memory_encoder.projection.2.weight"].shape[0]
    if inferred_dim != model_cfg.get("embedding_dim"):
        print(f"  [memoir] checkpoint embedding_dim={inferred_dim}, overriding config value {model_cfg.get('embedding_dim')}")
        config = {**config, "model": {**model_cfg, "embedding_dim": inferred_dim}}

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

    # Build full item pool and pre-compute base embeddings once
    processed_dir = Path(data_cfg["processed_dir"]) / data_cfg["dataset"]
    all_titles: set[str] = set()
    for split in ("train", "val", "test"):
        p = processed_dir / f"{split}_samples.parquet"
        if p.exists():
            all_titles.update(
                pd.read_parquet(p, columns=["target_title"])["target_title"].dropna()
            )
    item_titles, base_embeds = build_item_base_embeddings(model, sorted(all_titles))
    item_embeds = apply_item_projection(model, base_embeds)
    eval_seed = eval_cfg.get("eval_seed", 42)

    drift_map = dict(zip(drift_df["user_id"], drift_df["drift_group"]))
    group_results: dict[str, dict[str, float]] = {}

    for group in ("high", "medium", "low"):
        group_users = {uid for uid, g in drift_map.items() if g == group}
        indices = [
            i for i, s in enumerate(test_ds.samples.itertuples())
            if s.user_id in group_users
        ]
        if not indices:
            continue
        subset_loader = DataLoader(
            Subset(test_ds, indices),
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            collate_fn=collator,
        )
        print(f"  [memoir] evaluating {group}-drift ({len(indices)} samples)...")
        metrics = evaluate(
            model, subset_loader, ks,
            precomputed=(item_titles, item_embeds), eval_seed=eval_seed,
        )
        group_results[group] = metrics

    return group_results


def _evaluate_baseline_by_group(
    model_name: str,
    config: dict,
    drift_df: pd.DataFrame,
    ks: list[int],
    checkpoint_dir: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate a baseline model's test-set metrics broken down by drift group."""
    import sys
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(Path(__file__).parent))
    from baselines.train_baseline import SequentialDataset, collate_fn, evaluate_baseline
    from baselines import get_baseline, SEQUENTIAL_BASELINES

    if model_name not in SEQUENTIAL_BASELINES:
        print(f"  [{model_name}] not a sequential baseline — skipping")
        return {}

    ckpt = checkpoint_dir / f"{model_name}_best.pt"
    if not ckpt.exists():
        print(f"  [{model_name}] checkpoint not found — skipping")
        return {}

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    data_cfg = config["data"]
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    import os
    processed_dir = os.path.join(data_cfg["processed_dir"], data_cfg["dataset"])

    test_ds = SequentialDataset(processed_dir, split="test", max_len=data_cfg["max_history_len"])
    num_items = test_ds.num_items

    model_cls = get_baseline(model_name)
    model_kwargs = {
        "num_items": num_items,
        "embed_dim": config["model"]["embedding_dim"],
        "dropout": config["model"]["dropout"],
    }
    if model_name in ("sasrec", "bert4rec", "cl4srec", "duorec", "cllmrec"):
        model_kwargs["max_len"] = data_cfg["max_history_len"]
        model_kwargs["num_heads"] = 2
        model_kwargs["num_layers"] = 2
    if model_name == "gru4rec":
        model_kwargs["hidden_dim"] = 128
        model_kwargs["num_layers"] = 1
    if model_name in ("cl4srec", "duorec"):
        model_kwargs["cl_temperature"] = config["model"]["temperature"]
        model_kwargs["cl_weight"] = 0.1
    if model_name == "sracl":
        model_kwargs["num_heads"] = 2
        model_kwargs["num_layers"] = 2
        model_kwargs["alpha"] = 0.1
        model_kwargs["beta"] = 0.1
        model_kwargs["k_neighbors"] = 10
        model_kwargs["mlm_probability"] = 0.2
        model_kwargs["temperature"] = config["model"].get("temperature", 1.0)

    model = model_cls(**model_kwargs).to(device)

    if model_name == "sracl":
        # SRA-CL's user_neighbors/user_semantic_emb/all_seqs_tensor/all_lens_tensor
        # buffers are only allocated (at the correct, full-user-count shape) once
        # set_semantic_data() runs -- load_state_dict() can't load into them until
        # then, since it requires matching shapes. This re-encodes item/user texts
        # via SentenceTransformer just to get the buffer shapes right; the actual
        # trained values are overwritten by load_state_dict() immediately after.
        import pandas as pd
        import json as _json

        id_to_title: dict[str, str] = {}
        for split_name in ["train", "val", "test"]:
            p = os.path.join(processed_dir, f"{split_name}_samples.parquet")
            if os.path.exists(p):
                sdf = pd.read_parquet(p, columns=["target_item_id", "target_title"])
                for _, row in sdf.drop_duplicates("target_item_id").iterrows():
                    if row["target_item_id"] not in id_to_title and pd.notna(row.get("target_title")):
                        id_to_title[row["target_item_id"]] = row["target_title"]

        item_texts = [""] * (num_items + 1)
        for item_str, idx in test_ds.item2idx.items():
            item_texts[idx] = id_to_title.get(item_str, item_str)

        all_user_ids = sorted(test_ds.user_sequences.keys())
        user2idx = {uid: i for i, uid in enumerate(all_user_ids)}

        windows_path = os.path.join(processed_dir, "windows.parquet")
        user_texts: dict = {}
        if os.path.exists(windows_path):
            windows = pd.read_parquet(windows_path)
            for uid, group in windows.groupby("user_id"):
                user_texts[uid] = " ".join(group.sort_values("window_idx")["text"].tolist())

        model.set_semantic_data(
            item_texts=item_texts,
            user_texts_dict=user_texts,
            user_sequences_dict=test_ds.user_sequences,
            user2idx=user2idx,
        )

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    drift_map = dict(zip(drift_df["user_id"], drift_df["drift_group"]))
    group_results: dict[str, dict[str, float]] = {}

    for group in ("high", "medium", "low"):
        group_users = {uid for uid, g in drift_map.items() if g == group}
        indices = [
            i for i, row in enumerate(test_ds.samples.itertuples())
            if row.user_id in group_users
        ]
        if not indices:
            continue
        subset_loader = DataLoader(
            Subset(test_ds, indices),
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        print(f"  [{model_name}] evaluating {group}-drift ({len(indices)} samples)...")
        metrics = evaluate_baseline(
            model, subset_loader, ks,
            eval_seed=eval_cfg.get("eval_seed", 42),
        )
        group_results[group] = metrics

    return group_results


def print_drift_eval_table(
    all_results: dict[str, dict[str, dict[str, float]]],
    ks: list[int],
) -> None:
    """Print a comparison table of per-drift-group metrics across models."""
    groups = ["high", "medium", "low"]
    col_w = 10
    model_w = 14

    header = f"{'Model':<{model_w}}  {'Group':<8}"
    for k in ks:
        header += f"  {'HR@'+str(k):>{col_w}}  {'NDCG@'+str(k):>{col_w}}"
    header += f"  {'MRR':>{col_w}}"

    print(f"\n{'=' * len(header)}")
    print("  Drift-Stratified Evaluation (Test Set)")
    print(f"{'=' * len(header)}")
    print(f"  {header}")
    print(f"  {'-' * (len(header) - 2)}")

    for model_name, group_metrics in all_results.items():
        for gi, group in enumerate(groups):
            if group not in group_metrics:
                continue
            m = group_metrics[group]
            row = f"  {model_name if gi == 0 else '':<{model_w}}  {group:<8}"
            for k in ks:
                row += f"  {m.get(f'hr@{k}', float('nan')):>{col_w}.4f}"
                row += f"  {m.get(f'ndcg@{k}', float('nan')):>{col_w}.4f}"
            row += f"  {m.get('mrr', float('nan')):>{col_w}.4f}"
            print(row)
        print(f"  {'-' * (len(header) - 2)}")

    print(f"{'=' * len(header)}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze preference drift in user interaction data")
    parser.add_argument("--dataset", type=str, default="amazon", choices=["movielens", "amazon", "mind"])
    parser.add_argument("--processed-dir", type=str, default="./data/processed")
    parser.add_argument("--high-pct", type=int, default=75)
    parser.add_argument("--low-pct", type=int, default=25)
    parser.add_argument("--plot", action="store_true")
    # Drift-stratified evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate trained models by drift group (requires checkpoints)")
    parser.add_argument("--config", type=str, default="./configs/default.yaml",
                        help="Config file (used for --evaluate)")
    parser.add_argument("--models", type=str,
                        default="memoir,gru4rec,sasrec,cl4srec,duorec,sracl,sasrec_text,unisrec",
                        help="Comma-separated models to evaluate (used with --evaluate)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory containing best model checkpoints")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip models already present in drift_eval_results.parquet (used with --evaluate)")
    args = parser.parse_args()

    data_dir = Path(args.processed_dir) / args.dataset
    windows_path = data_dir / "windows.parquet"

    if not windows_path.exists():
        print(f"Error: {windows_path} not found. Run data preprocessing first.")
        return

    # Load or (re)compute drift analysis
    drift_path = data_dir / "user_drift_analysis.parquet"
    if drift_path.exists() and not args.evaluate:
        # Fast path: just load if we only need drift stats
        df = pd.read_parquet(drift_path)
        print(f"Loaded cached drift analysis for {len(df)} users")
    else:
        windows = pd.read_parquet(windows_path)
        print(f"Loaded {len(windows)} windows for {windows['user_id'].nunique()} users")

        rows = []
        for user_id, group in windows.groupby("user_id"):
            metrics = compute_user_metrics(group)
            metrics["user_id"] = user_id
            metrics["num_windows"] = len(group)
            rows.append(metrics)

        df = pd.DataFrame(rows)
        df = normalize_and_composite(df)
        df = classify_drift_groups(df, args.high_pct, args.low_pct)

        col_order = [
            "user_id", "num_windows",
            "category_jsd_mean", "category_jsd_max",
            "rating_drift", "novelty_rate",
            "composite_drift_score", "drift_group",
        ]
        df = df[col_order]
        df.to_parquet(drift_path, index=False)
        print(f"Saved: {drift_path}")

    print_summary(df, args.dataset)

    if args.plot:
        plot_drift_distributions(df, args.dataset, data_dir)
        plot_drift_summary(df, args.dataset, data_dir)

    if args.evaluate:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config["data"]["dataset"] = args.dataset
        config["data"]["processed_dir"] = args.processed_dir

        ks = config["eval"]["ks"]
        checkpoint_dir = Path(args.checkpoint_dir)
        model_list = [m.strip() for m in args.models.split(",")]
        out_path = data_dir / "drift_eval_results.parquet"

        # Resume from any previously saved results so a crash partway through
        # (or --skip-done) doesn't force re-evaluating already-completed models.
        all_results: dict[str, dict] = {}
        if out_path.exists():
            prev_df = pd.read_parquet(out_path)
            metric_cols = [c for c in prev_df.columns if c not in ("model", "drift_group")]
            for model_name, group_df in prev_df.groupby("model"):
                all_results[model_name] = {
                    row["drift_group"]: {c: row[c] for c in metric_cols}
                    for _, row in group_df.iterrows()
                }
            print(f"Loaded {len(all_results)} previously evaluated model(s) from {out_path}")

        def _save_results():
            rows_out = []
            for model_name, group_metrics in all_results.items():
                for group, metrics in group_metrics.items():
                    row = {"model": model_name, "drift_group": group}
                    row.update(metrics)
                    rows_out.append(row)
            out_df = pd.DataFrame(rows_out)
            out_df.to_parquet(out_path, index=False)

        for model_name in model_list:
            if args.skip_done and model_name in all_results:
                print(f"\nSkipping [{model_name}] (already in {out_path.name})")
                continue

            print(f"\nEvaluating [{model_name}] by drift group...")
            if model_name == "memoir":
                results = _evaluate_memoir_by_group(config, df, ks, checkpoint_dir)
            else:
                results = _evaluate_baseline_by_group(
                    model_name, config, df, ks, checkpoint_dir,
                )
            if results:
                all_results[model_name] = results
                _save_results()
                print(f"  [{model_name}] results saved to {out_path}")

        if all_results:
            print_drift_eval_table(all_results, ks)
            print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
