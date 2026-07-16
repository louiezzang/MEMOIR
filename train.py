"""MEMOIR training script."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_dataset, MEMOIRCollator
from model import MEMOIRModel
from losses import MEMOIRLoss
from eval import evaluate, build_item_base_embeddings, apply_item_projection, build_random_item_embeddings
from logger import ExperimentLogger
from utils import get_device, get_autocast_ctx, supports_grad_scaler


def _mean_pairwise_cosine_sim(x: torch.Tensor) -> float:
    """Mean off-diagonal pairwise cosine similarity within a batch.

    Cheap collapse detector: values creeping toward 1.0 mean the batch's
    embeddings are becoming indistinguishable from each other.
    """
    b = x.shape[0]
    if b < 2:
        return float("nan")
    normed = F.normalize(x.float(), dim=-1)
    sim = normed @ normed.T
    off_diag_sum = sim.sum() - sim.diagonal().sum()
    return (off_diag_sum / (b * (b - 1))).item()


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _build_text_cache(
    model: "MEMOIRModel",
    datasets: list,
    device: torch.device,
    cache_path: Path,
) -> None:
    """Pre-compute and cache pooled (pre-projection) window text embeddings.

    Used when freeze_llm=True. Caches the frozen LLM's pooled hidden state only —
    the trainable projection is applied fresh on every forward pass (see
    LLMMemoryEncoder.encode_text), so this caching does not block its gradients.
    Saves to disk on first run; loads from disk on subsequent runs.
    """
    all_texts: list[str] = sorted({
        w.text
        for ds in datasets
        for windows in ds.user_windows.values()
        for w in windows
    })

    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path.name}...")
        cache: dict[str, torch.Tensor] = torch.load(cache_path, map_location="cpu", weights_only=True)
        if len(cache) == len(all_texts):
            model.memory_encoder.set_text_cache(cache)
            print(f"  Loaded {len(cache)} cached embeddings.")
            return
        print(f"  Cache incomplete ({len(cache)}/{len(all_texts)}), recomputing...")

    print(f"  Pre-computing embeddings for {len(all_texts)} unique window texts...")
    model.memory_encoder.eval()
    cache = {}
    encode_bs = 256  # large batch is fine without gradients
    with torch.no_grad():
        for i in tqdm(range(0, len(all_texts), encode_bs), desc="Caching embeddings"):
            chunk = all_texts[i : i + encode_bs]
            embeds = model.memory_encoder.encode_pooled(chunk)
            for text, emb in zip(chunk, embeds.cpu()):
                cache[text] = emb

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    model.memory_encoder.set_text_cache(cache)
    print(f"  Done. Saved {len(cache)} embeddings to {cache_path.name}.")


def train(config: dict):
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    data_cfg = config["data"]
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    log_cfg = config["logging"]
    model_cfg = config["model"]

    # Datasets
    ds_kwargs = dict(
        data_dir=os.path.join(data_cfg["data_dir"], data_cfg["dataset"]),
        processed_dir=os.path.join(data_cfg["processed_dir"], data_cfg["dataset"]),
        window_type=data_cfg["time_window"],
        min_interactions=data_cfg["min_interactions"],
        max_history_len=data_cfg["max_history_len"],
        num_windows=config["model"]["num_memory_windows"],
    )

    train_ds = build_dataset(data_cfg["dataset"], split="train", **ds_kwargs)
    val_ds = build_dataset(data_cfg["dataset"], split="val", **ds_kwargs)

    # Build full item pool from all splits for fair evaluation (matches baseline protocol)
    processed_dir = Path(data_cfg["processed_dir"]) / data_cfg["dataset"]
    _all_titles: set[str] = set()
    _all_item_ids: set[str] = set()
    for split in ("train", "val", "test"):
        p = processed_dir / f"{split}_samples.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["target_item_id", "target_title"])
            _all_titles.update(df["target_title"].dropna())
            _all_item_ids.update(df["target_item_id"].dropna())
    item_pool: list[str] = sorted(_all_titles)
    print(f"Item pool: {len(item_pool)} unique titles across all splits")

    use_random_items = model_cfg["item_encoder"].get("type") == "random"
    item2idx: dict[str, int] | None = None
    if use_random_items:
        item2idx = {item_id: idx + 1 for idx, item_id in enumerate(sorted(_all_item_ids))}
        model_cfg["item_encoder"]["num_items"] = len(item2idx)
        print(f"Random item encoder: {len(item2idx)} items")

    collator = MEMOIRCollator(
        max_windows=config["model"]["num_memory_windows"],
        num_negatives=4,
    )

    pin = device.type == "cuda"
    workers = 4 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=pin,
    )

    # Initialize logger after datasets are loaded
    run_suffix = log_cfg.get("run_suffix", "")
    logger = ExperimentLogger(config, run_name=f"memoir_{config['data']['dataset']}{run_suffix}")
    logger.log_debug(f"Model: MEMOIR, device: {device}, num_items: {train_ds.num_items if hasattr(train_ds, 'num_items') else 'unknown'}")
    logger.log_debug(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model
    model = MEMOIRModel(config)
    model.to(device)

    if model_cfg.get("freeze_llm", False):
        print("freeze_llm=True: pre-computing window embeddings...")
        llm_tag = model_cfg["llm_name"].replace("/", "_")
        cache_path = Path(data_cfg["processed_dir"]) / data_cfg["dataset"] / f"pooled_cache_{llm_tag}_d{model_cfg['embedding_dim']}.pt"
        _build_text_cache(model, [train_ds, val_ds], device, cache_path)
        model.train()

    eval_item_titles, eval_base_embeds = None, None
    title_to_idx, catalog_base_embeds_gpu = None, None
    if use_random_items:
        print("Using random item encoder — skipping MiniLM pre-computation.")
    else:
        print("Pre-computing item base embeddings (MiniLM, frozen)...")
        eval_item_titles, eval_base_embeds = build_item_base_embeddings(model, item_pool)
        print("Fitting whitening transform on item base embeddings...")
        model.item_encoder.set_whitening(eval_base_embeds)
        title_to_idx = {t: i for i, t in enumerate(eval_item_titles)}
        catalog_base_embeds_gpu = eval_base_embeds.to(device)
    eval_seed = eval_cfg.get("eval_seed", 42)

    def _attach_catalog(output: dict, batch: dict) -> dict:
        """Score against the full item catalog for rec_loss (see losses.py) —
        matches every baseline's training signal instead of in-batch negatives.
        """
        if use_random_items:
            output["target_catalog_idx"] = batch["target_item_indices"]
            output["catalog_item_embeds"] = model.item_encoder.catalog_embeddings()
        else:
            output["target_catalog_idx"] = torch.tensor(
                [title_to_idx[t] for t in batch["target_titles"]],
                dtype=torch.long, device=device,
            )
            output["catalog_item_embeds"] = model.item_encoder.catalog_embeddings(catalog_base_embeds_gpu)
        return output

    criterion = MEMOIRLoss(
        lambda_evo=train_cfg["lambda_evo"],
        lambda_consistency=train_cfg["lambda_consistency"],
        lambda_extrap=train_cfg["lambda_extrap"],
        temperature=config["model"]["temperature"],
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )

    # Optimizer: different LR for LLM vs projection vs other params.
    # The projection MLP only just started receiving real gradients (previously
    # dead under the embedding-cache bug), so its AdamW state is cold — give it
    # its own, lower LR rather than lumping it in with the already-stable
    # GRU/aggregator/trajectory_predictor, which have been training fine all along.
    llm_params = list(model.memory_encoder.llm.parameters())
    llm_param_ids = {id(p) for p in llm_params}
    trainable_llm = [p for p in llm_params if p.requires_grad]

    projection_params = list(model.memory_encoder.projection.parameters())
    projection_param_ids = {id(p) for p in projection_params}

    other_params = [
        p for p in model.parameters()
        if id(p) not in llm_param_ids and id(p) not in projection_param_ids and p.requires_grad
    ]

    projection_lr = train_cfg.get("projection_lr", train_cfg["lr"] * 0.1)
    param_groups = [
        {"params": other_params, "lr": train_cfg["lr"]},
        {"params": projection_params, "lr": projection_lr},
    ]
    if trainable_llm:
        param_groups.append({"params": trainable_llm, "lr": train_cfg["llm_lr"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg["weight_decay"])

    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    num_training_steps = optimizer_steps_per_epoch * train_cfg["epochs"]
    warmup_steps = min(train_cfg.get("warmup_steps", 0), max(num_training_steps - 1, 0))

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps - warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps,
        )

    use_scaler = train_cfg.get("fp16", False) and supports_grad_scaler(device)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    save_dir = Path(log_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    run_suffix = log_cfg.get("run_suffix", "")
    logger = ExperimentLogger(config, run_name=f"memoir_{data_cfg['dataset']}{run_suffix}")
    best_ndcg = 0.0
    patience = train_cfg.get("early_stopping_patience", 0)
    epochs_no_improve = 0

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        loss_components = {"rec": 0.0, "evo": 0.0, "consistency": 0.0, "extrap": 0.0}
        collapse_diag = {"item_sim": 0.0, "user_sim": 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")
        for step, batch in enumerate(pbar):
            if use_random_items:
                indices = [item2idx.get(iid, 0) for iid in batch["target_item_ids"]]
                batch["target_item_indices"] = torch.tensor(indices, dtype=torch.long)
            batch = _batch_to_device(batch, device)
            if scaler:
                with get_autocast_ctx(device):
                    output = model(batch)
                    output = _attach_catalog(output, batch)
                    losses = criterion(output)

                scaler.scale(losses["total"]).backward()

                if (step + 1) % train_cfg.get("gradient_accumulation_steps", 1) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                output = model(batch)
                output = _attach_catalog(output, batch)
                losses = criterion(output)
                losses["total"].backward()

                if (step + 1) % train_cfg.get("gradient_accumulation_steps", 1) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            total_loss += losses["total"].item()
            for k in loss_components:
                loss_components[k] += losses[k].item()

            with torch.no_grad():
                collapse_diag["item_sim"] += _mean_pairwise_cosine_sim(output["item_embeds"])
                collapse_diag["user_sim"] += _mean_pairwise_cosine_sim(output["user_memory"])

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "rec": f"{losses['rec'].item():.4f}",
                "evo": f"{losses['evo'].item():.4f}",
            })

        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        avg_collapse_diag = {k: v / len(train_loader) for k, v in collapse_diag.items()}
        print(f"\nEpoch {epoch+1} | avg_loss={avg_loss:.4f}")
        for k, v in avg_components.items():
            print(f"  {k}: {v:.4f}")
        print(f"  [collapse check] item_sim: {avg_collapse_diag['item_sim']:.4f}, user_sim: {avg_collapse_diag['user_sim']:.4f}  (mean pairwise cosine sim; near 1.0 = collapsed)")

        logger.log_train(epoch + 1, {"avg_loss": avg_loss, **avg_components, **avg_collapse_diag})
        logger.log_debug(
            f"Epoch {epoch+1}/{train_cfg['epochs']} completed - avg_loss={avg_loss:.4f}, "
            f"rec={avg_components['rec']:.4f}, evo={avg_components['evo']:.4f}, "
            f"item_sim={avg_collapse_diag['item_sim']:.4f}, user_sim={avg_collapse_diag['user_sim']:.4f}"
        )

        # Evaluation
        if (epoch + 1) % eval_cfg["eval_every"] == 0:
            if use_random_items:
                metrics = evaluate(
                    model, val_loader, eval_cfg["ks"],
                    item2idx=item2idx, eval_seed=eval_seed,
                )
            else:
                item_embeds = apply_item_projection(model, eval_base_embeds)
                metrics = evaluate(
                    model, val_loader, eval_cfg["ks"],
                    precomputed=(eval_item_titles, item_embeds), eval_seed=eval_seed,
                )
            logger.log_eval(epoch + 1, metrics)
            print(f"  Validation metrics:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")

            ndcg_key = f"ndcg@{eval_cfg['ks'][1]}"
            if ndcg_key in metrics and metrics[ndcg_key] > best_ndcg:
                best_ndcg = metrics[ndcg_key]
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_dir / "best_model.pt")
                print(f"  New best model saved (NDCG@{eval_cfg['ks'][1]}={best_ndcg:.4f})")
                logger.log_debug(f"New best! {ndcg_key}={best_ndcg:.4f}")
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    logger.log_debug(f"Early stopping at epoch {epoch+1}")
                    break

        # Periodic save
        if (epoch + 1) % log_cfg["save_every"] == 0:
            torch.save(model.state_dict(), save_dir / f"model_epoch{epoch+1}.pt")

    # --- Test set evaluation with best checkpoint ---
    best_ckpt = save_dir / "best_model.pt"
    if best_ckpt.exists():
        print("\nEvaluating best model on test set...")
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        if model_cfg.get("freeze_llm", False):
            model.memory_encoder.set_text_cache(
                torch.load(
                    Path(data_cfg["processed_dir"]) / data_cfg["dataset"] / f"pooled_cache_{model_cfg['llm_name'].replace('/', '_')}_d{model_cfg['embedding_dim']}.pt",
                    map_location="cpu", weights_only=True,
                )
            )
        test_ds = build_dataset(data_cfg["dataset"], split="test", **ds_kwargs)
        test_loader = DataLoader(
            test_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=workers,
            pin_memory=pin,
        )
        if use_random_items:
            test_metrics = evaluate(
                model, test_loader, eval_cfg["ks"],
                item2idx=item2idx, eval_seed=eval_seed,
            )
        else:
            item_embeds = apply_item_projection(model, eval_base_embeds)
            test_metrics = evaluate(
                model, test_loader, eval_cfg["ks"],
                precomputed=(eval_item_titles, item_embeds), eval_seed=eval_seed,
            )
        print("  Test metrics (best checkpoint):")
        for k, v in test_metrics.items():
            print(f"    {k}: {v:.4f}")
        logger.log_eval("test", test_metrics)
        logger.log_debug(f"Test results: {test_metrics}")

    logger.finish()
    print(f"\nTraining complete. Best val NDCG@{eval_cfg['ks'][1]}: {best_ndcg:.4f}")
    logger.log_debug(f"Training completed. Best NDCG@{eval_cfg['ks'][1]}={best_ndcg:.4f}")


def _set_nested(d: dict, key_path: str, value: str):
    """Set a dot-separated key in a nested dict, coercing value to float/int/bool if possible."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d[k]
    leaf = keys[-1]
    existing = d.get(leaf)
    try:
        if isinstance(existing, bool):
            d[leaf] = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            d[leaf] = int(value)
        elif isinstance(existing, float):
            d[leaf] = float(value)
        else:
            d[leaf] = value
    except (ValueError, TypeError):
        d[leaf] = value


def main():
    parser = argparse.ArgumentParser(description="Train MEMOIR model")
    parser.add_argument("--config", type=str, default="configs/memoir.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--log-suffix", type=str, default="", help="Suffix appended to the run log directory")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. training.lambda_evo=0.1")
    parser.add_argument("--no-evo-cl", action="store_true", help="Ablation: disable evolution contrastive loss")
    parser.add_argument("--no-dir-loss", action="store_true", help="Ablation: disable directional consistency loss")
    parser.add_argument("--no-temporal", action="store_true", help="Ablation: collapse all windows into one (num_windows=1)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config["data"]["dataset"] = args.dataset

    for kv in args.override:
        if "=" not in kv:
            raise ValueError(f"--override expects KEY=VALUE, got: {kv!r}")
        key, val = kv.split("=", 1)
        _set_nested(config, key, val)

    if args.no_evo_cl:
        config["training"]["lambda_evo"] = 0.0
    if args.no_dir_loss:
        config["training"]["alpha_direction"] = 0.0
    if args.no_temporal:
        config["model"]["num_memory_windows"] = 1

    if args.log_suffix:
        config["logging"]["save_dir"] = config["logging"]["save_dir"].rstrip("/") + args.log_suffix
        config["logging"]["run_suffix"] = args.log_suffix

    train(config)


if __name__ == "__main__":
    main()
