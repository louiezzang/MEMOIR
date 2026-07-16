"""Unified training script for all sequential baselines."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baselines import get_baseline, SEQUENTIAL_BASELINES, BASELINE_NAMES


class SequentialDataset(Dataset):
    """Converts MEMOIR-style data into ID-based sequences for traditional baselines."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_len: int = 50,
    ):
        import pandas as pd
        from pathlib import Path

        self.max_len = max_len
        processed = Path(data_dir)

        samples_path = processed / f"{split}_samples.parquet"
        if not samples_path.exists():
            raise FileNotFoundError(f"{samples_path} not found. Run MEMOIR data preprocessing first.")

        self.samples = pd.read_parquet(samples_path)

        # Build item vocabulary from target IDs only (keeps vocab manageable)
        all_items = set()
        for split_name in ["train", "val", "test"]:
            p = processed / f"{split_name}_samples.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                all_items.update(df["target_item_id"].unique())

        self.item2idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
        self.num_items = len(self.item2idx)

        # Build per-user item sequences from raw interactions (chronological)
        self.user_sequences: dict[str, list[int]] = {}
        raw_dir = processed.parent.parent / "raw" / "amazon_5core" / "benchmark" / "5core" / "rating_only"
        if raw_dir.exists():
            all_users = set()
            for split_name in ["train", "val", "test"]:
                p = processed / f"{split_name}_samples.parquet"
                if p.exists():
                    all_users.update(pd.read_parquet(p)["user_id"])

            raw_frames = []
            for csv_path in raw_dir.glob("*.csv"):
                raw_frames.append(pd.read_csv(csv_path))
            raw = pd.concat(raw_frames)
            user_raw = raw[raw["user_id"].isin(all_users)].sort_values("timestamp")
            for uid, group in user_raw.groupby("user_id"):
                seq = [self.item2idx[a] for a in group["parent_asin"] if a in self.item2idx]
                if seq:
                    self.user_sequences[uid] = seq
            print(f"[debug] Built sequences from raw CSVs: {len(self.user_sequences)} users")
        else:
            import json as _json
            windows_path = processed / "windows.parquet"
            print(f"[debug] Raw dir not found: {raw_dir}")
            print(f"[debug] Falling back to windows.parquet: {windows_path} (exists={windows_path.exists()})")
            if windows_path.exists():
                windows_df = pd.read_parquet(windows_path)
                print(f"[debug] windows.parquet: {len(windows_df)} rows, columns={windows_df.columns.tolist()}")
                if 'interactions' in windows_df.columns:
                    sample = _json.loads(windows_df['interactions'].iloc[0])
                    print(f"[debug] Sample interaction: {sample[0] if sample else 'EMPTY'}")
                for uid, group in windows_df.groupby("user_id"):
                    items = []
                    for _, row in group.sort_values("window_idx").iterrows():
                        interactions = _json.loads(row["interactions"])
                        for inter in sorted(interactions, key=lambda x: x.get("timestamp", 0)):
                            iid = inter.get("item_id")
                            if iid and iid in self.item2idx:
                                items.append(self.item2idx[iid])
                    if items:
                        self.user_sequences[uid] = items
                print(f"[debug] Built sequences from windows.parquet: {len(self.user_sequences)} users")
            else:
                print("[debug] WARNING: No windows.parquet found! All sequences will be empty.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        target_idx = self.item2idx.get(row["target_item_id"], 0)
        uid = row["user_id"]

        history = self.user_sequences.get(uid, [])
        # Remove target from history to avoid leakage
        history = [h for h in history if h != target_idx]
        history = history[-self.max_len:]

        seq = torch.zeros(self.max_len, dtype=torch.long)
        seq_len = min(len(history), self.max_len)
        if seq_len > 0:
            seq[:seq_len] = torch.tensor(history[-seq_len:], dtype=torch.long)

        result = {
            "item_seq": seq,
            "seq_len": torch.tensor(max(seq_len, 1), dtype=torch.long),
            "target": torch.tensor(target_idx, dtype=torch.long),
        }
        if hasattr(self, "user2idx") and self.user2idx is not None:
            user_int = self.user2idx.get(uid, -1)
            result["user_idx"] = torch.tensor(user_int, dtype=torch.long)

            # Pre-fetch neighbor sequences for SRA-CL cross-sequence CL
            if (
                hasattr(self, "sracl_user_neighbors")
                and self.sracl_user_neighbors is not None
                and 0 <= user_int < self.sracl_user_neighbors.size(0)
            ):
                neighbor_ids = self.sracl_user_neighbors[user_int]  # [k]
                neighbor_ids_clamp = neighbor_ids.clamp(max=self.sracl_all_seqs.size(0) - 1)
                result["neighbor_seqs"] = self.sracl_all_seqs[neighbor_ids_clamp]  # [k, L]

        return result


def collate_fn(batch):
    result = {
        "item_seq": torch.stack([b["item_seq"] for b in batch]),
        "seq_len": torch.stack([b["seq_len"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }
    if "user_idx" in batch[0]:
        result["user_idx"] = torch.stack([b["user_idx"] for b in batch])
    if "neighbor_seqs" in batch[0]:
        result["neighbor_seqs"] = torch.stack([b["neighbor_seqs"] for b in batch])
    return result


def evaluate_baseline(model, dataloader, ks=(5, 10, 20), num_neg=999, eval_seed=42):
    """Sampled evaluation: rank target against num_neg random negatives (standard protocol)."""
    model.eval()
    num_items = model.num_items
    results = {f"{m}@{k}": [] for k in ks for m in ["hr", "ndcg", "recall"]}
    results["mrr"] = []

    rng = torch.Generator()
    rng.manual_seed(eval_seed)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            item_seq = batch["item_seq"].to(next(model.parameters()).device)
            seq_len = batch["seq_len"].to(item_seq.device)
            target = batch["target"].to(item_seq.device)

            user_repr = model(item_seq, seq_len)
            all_scores = model.get_scores(user_repr)

            B = target.shape[0]
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
                rank = (scores > scores[0]).sum() + 1

                for k in ks:
                    results[f"hr@{k}"].append(1.0 if rank <= k else 0.0)
                    results[f"ndcg@{k}"].append(1.0 / math.log2(rank + 1) if rank <= k else 0.0)
                    results[f"recall@{k}"].append(1.0 if rank <= k else 0.0)
                results["mrr"].append(1.0 / rank)

    return {k: float(np.mean(v)) for k, v in results.items() if v}


def train_baseline(config: dict, model_name: str):
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    data_cfg = config["data"]
    train_cfg = config["training"]
    eval_cfg = config["eval"]

    import os
    processed_dir = os.path.join(data_cfg["processed_dir"], data_cfg["dataset"])

    train_ds = SequentialDataset(processed_dir, split="train", max_len=data_cfg["max_history_len"])
    val_ds = SequentialDataset(processed_dir, split="val", max_len=data_cfg["max_history_len"])

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    num_items = train_ds.num_items

    # Build model
    model_cls = get_baseline(model_name)

    # Model-specific kwargs
    model_kwargs = {
        "num_items": num_items,
        "embed_dim": config["model"]["embedding_dim"],
        "dropout": config["model"]["dropout"],
    }

    if model_name == "gru4rec":
        model_kwargs["hidden_dim"] = 128  # GRU hidden size
        model_kwargs["num_layers"] = 1  # GRU layers
    elif model_name in ("sasrec", "bert4rec", "cl4srec", "duorec", "cllmrec", "sasrec_text", "unisrec"):
        model_kwargs["max_len"] = data_cfg["max_history_len"]
        model_kwargs["num_heads"] = 2
        model_kwargs["num_layers"] = 2
    if model_name in ("cl4srec", "duorec"):
        model_kwargs["cl_temperature"] = config["model"]["temperature"]
        model_kwargs["cl_weight"] = 0.1
    if model_name == "cllmrec":
        model_kwargs["cl_temperature"] = 0.1
        model_kwargs["cl_weight"] = 0.5
        model_kwargs["seq_weight"] = 0.5
        model_kwargs["neg_weight"] = 0.5
        model_kwargs["lambda_a"] = 0.5
        model_kwargs["prune_ratio"] = 0.25
    if model_name == "sracl":
        model_kwargs["num_heads"] = 2
        model_kwargs["num_layers"] = 2
        model_kwargs["alpha"] = 0.1
        model_kwargs["beta"] = 0.1
        model_kwargs["k_neighbors"] = 10
        model_kwargs["mlm_probability"] = 0.2
        model_kwargs["temperature"] = config["model"].get("temperature", 1.0)
    if model_name == "unisrec":
        model_kwargs["num_experts"] = config["model"].get("num_experts", 2)

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils import get_device
    from logger import ExperimentLogger
    device = get_device()
    model = model_cls(**model_kwargs).to(device)

    logger = ExperimentLogger(config, run_name=f"{model_name}_{data_cfg['dataset']}")
    logger.log_debug(f"Model: {model_name}, device: {device}, num_items: {num_items}")
    logger.log_debug(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    logger.log_debug(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Text-aware baselines: load item texts ---
    if model_name in ("sracl", "sasrec_text", "unisrec"):
        import pandas as pd

        id_to_title: dict[str, str] = {}
        for split_name in ["train", "val", "test"]:
            p = os.path.join(processed_dir, f"{split_name}_samples.parquet")
            if os.path.exists(p):
                df = pd.read_parquet(p, columns=["target_item_id", "target_title"])
                for _, row in df.drop_duplicates("target_item_id").iterrows():
                    if row["target_item_id"] not in id_to_title and pd.notna(row.get("target_title")):
                        id_to_title[row["target_item_id"]] = row["target_title"]

        item_texts = [""] * (num_items + 1)
        for item_str, idx in train_ds.item2idx.items():
            item_texts[idx] = id_to_title.get(item_str, item_str)

        if model_name == "sracl":
            all_user_ids = sorted(train_ds.user_sequences.keys())
            user2idx = {uid: i for i, uid in enumerate(all_user_ids)}

            windows_path = os.path.join(processed_dir, "windows.parquet")
            user_texts: dict = {}
            if os.path.exists(windows_path):
                windows = pd.read_parquet(windows_path)
                for uid, group in windows.groupby("user_id"):
                    user_texts[uid] = " ".join(group.sort_values("window_idx")["text"].tolist())

            logger.log_debug(f"[sracl] Initializing semantic data ({len(item_texts)} items, {len(user_texts)} users)...")
            model.set_semantic_data(
                item_texts=item_texts,
                user_texts_dict=user_texts,
                user_sequences_dict=train_ds.user_sequences,
                user2idx=user2idx,
                logger=logger,
            )
            train_ds.user2idx = user2idx
            val_ds.user2idx = user2idx
            train_ds.sracl_user_neighbors = model.user_neighbors.cpu()
            train_ds.sracl_all_seqs = model.all_seqs_tensor.cpu()
            logger.log_debug("[sracl] Semantic data ready.")
        else:
            logger.log_debug(f"[{model_name}] Initializing text data ({len(item_texts)} items)...")
            model.set_text_data(item_texts=item_texts, logger=logger)
            logger.log_debug(f"[{model_name}] Text data ready.")

    baseline_lr = train_cfg.get("baseline_lr", 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=baseline_lr, weight_decay=train_cfg["weight_decay"])

    # Enable AMP for faster training on supported devices
    use_amp = device.type in ("cuda", "mps")
    scaler = None
    autocast_device = None

    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_device = "cuda"
    elif device.type == "mps":
        # MPS supports autocast but not GradScaler
        autocast_device = "mps"

    best_ndcg = 0.0
    patience = train_cfg.get("early_stopping_patience", 0)
    epochs_no_improve = 0

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0

        # Enable gradient accumulation
        accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{train_cfg['epochs']}")
        for i, batch in enumerate(pbar):
            item_seq = batch["item_seq"].to(device)
            seq_len = batch["seq_len"].to(device)
            target = batch["target"].to(device)

            kwargs = {}
            if model_name == "sracl" and "user_idx" in batch:
                kwargs["user_idx"] = batch["user_idx"].to(device)
                if "neighbor_seqs" in batch:
                    kwargs["neighbor_seqs"] = batch["neighbor_seqs"].to(device)

            # Mixed precision training
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = model.compute_loss(item_seq, seq_len, target, **kwargs)

            # Scale loss for gradient accumulation
            loss = loss / accum_steps

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimization step every accum_steps
            if (i + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.log_debug(f"Epoch {epoch+1}/{train_cfg['epochs']} avg_loss={avg_loss:.4f}")
        logger.log_train(epoch + 1, {"avg_loss": avg_loss})

        if (epoch + 1) % eval_cfg["eval_every"] == 0:
            metrics = evaluate_baseline(model, val_loader, eval_cfg["ks"], eval_seed=eval_cfg.get("eval_seed", 42))
            logger.log_eval(epoch + 1, metrics)
            logger.log_debug(f"Epoch {epoch+1} val: {metrics}")

            ndcg_key = f"ndcg@{eval_cfg['ks'][1]}"
            if ndcg_key in metrics and metrics[ndcg_key] > best_ndcg:
                best_ndcg = metrics[ndcg_key]
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pt")
                logger.log_debug(f"New best! {ndcg_key}={best_ndcg:.4f}")
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    logger.log_debug(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    print(f"[{model_name}] Early stopping at epoch {epoch+1}")
                    break

    # --- Test set evaluation with best checkpoint ---
    best_ckpt = f"checkpoints/{model_name}_best.pt"
    if os.path.exists(best_ckpt):
        print(f"\n[{model_name}] Evaluating best checkpoint on test set...")
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        test_ds = SequentialDataset(processed_dir, split="test", max_len=data_cfg["max_history_len"])
        test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn)
        test_metrics = evaluate_baseline(
            model, test_loader, eval_cfg["ks"],
            eval_seed=eval_cfg.get("eval_seed", 42),
        )
        print(f"[{model_name}] Test metrics (best checkpoint):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        logger.log_eval("test", test_metrics)
        logger.log_debug(f"Test results: {test_metrics}")

    logger.finish()
    print(f"[{model_name}] Done. Best val NDCG: {best_ndcg:.4f}")
    return best_ndcg


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, required=True, choices=BASELINE_NAMES)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config["data"]["dataset"] = args.dataset

    if args.model in SEQUENTIAL_BASELINES:
        train_baseline(config, args.model)
    else:
        print(f"{args.model} is an LLM baseline - use separate evaluation script.")
        print("For TALLRec: requires LoRA fine-tuning (see tallrec.py)")
        print("For LLMRanker: zero-shot, no training needed (see llm_ranker.py)")


if __name__ == "__main__":
    main()
