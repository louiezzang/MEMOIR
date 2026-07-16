"""MEMOIR evaluation: HR@K, NDCG@K, MRR, Recall@K."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def hit_rate(ranked_list: np.ndarray, ground_truth: set, k: int) -> float:
    return 1.0 if any(item in ground_truth for item in ranked_list[:k]) else 0.0


def ndcg(ranked_list: np.ndarray, ground_truth: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_list: np.ndarray, ground_truth: set) -> float:
    for i, item in enumerate(ranked_list):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def recall(ranked_list: np.ndarray, ground_truth: set, k: int) -> float:
    hits = sum(1 for item in ranked_list[:k] if item in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0


@torch.no_grad()
def build_item_base_embeddings(
    model,
    item_pool: list[str],
    encode_bs: int = 64,
) -> tuple[list[str], torch.Tensor]:
    """Pre-compute frozen MiniLM base embeddings for the item pool.

    MiniLM is frozen throughout training so these never change — call once
    before the training loop. The trainable projection is applied separately
    at each evaluation via apply_item_projection().

    Returns:
        unique_titles: deduplicated item title list
        base_embeds_cpu: [N, base_dim] float32 tensor on CPU (pre-projection)
    """
    unique_titles = list(dict.fromkeys(item_pool))
    N = len(unique_titles)
    base_embeds = []
    for i in tqdm(range(0, N, encode_bs), desc="Pre-encoding item pool (MiniLM)"):
        chunk = unique_titles[i : i + encode_bs]
        emb = model.item_encoder.encoder.encode(
            chunk, convert_to_tensor=True, show_progress_bar=False,
        ).float().cpu()
        base_embeds.append(emb)
    return unique_titles, torch.cat(base_embeds, dim=0)  # [N, base_dim]


@torch.no_grad()
def apply_item_projection(
    model,
    base_embeds_cpu: torch.Tensor,
    encode_bs: int = 512,
) -> torch.Tensor:
    """Apply the trainable projection to get final normalized item embeddings.

    Call this at the start of each evaluation epoch since the projection
    weights change during training.

    Returns:
        item_embeds_cpu: [N, output_dim] normalized float32 tensor on CPU
    """
    device = next(model.parameters()).device
    projected = []
    for i in range(0, len(base_embeds_cpu), encode_bs):
        chunk = base_embeds_cpu[i : i + encode_bs].to(device)
        proj = model.item_encoder.catalog_embeddings(chunk)
        projected.append(proj.cpu())
    return F.normalize(torch.cat(projected, dim=0), dim=-1)  # [N, output_dim]


@torch.no_grad()
def build_random_item_embeddings(model, num_items: int) -> torch.Tensor:
    """Get normalized item embeddings from RandomItemEncoder.

    Returns:
        item_embeds_cpu: [num_items+1, output_dim] normalized float32 tensor on CPU
    """
    embeds = model.item_encoder.embedding.weight.data.clone().cpu()
    return F.normalize(embeds, dim=-1)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    ks: list[int] = (5, 10, 20),
    num_neg: int = 999,
    item_pool: list[str] | None = None,
    precomputed: tuple[list[str], torch.Tensor] | None = None,
    item2idx: dict[str, int] | None = None,
    eval_seed: int = 42,
) -> dict[str, float]:
    """Evaluate MEMOIR with sampled-negatives protocol (same as baselines).

    Args:
        model: MEMOIRModel
        dataloader: val or test DataLoader
        ks: cutoff values for HR/NDCG/Recall
        num_neg: number of random negative items to rank against (default 999)
        item_pool: list of all item title strings (all splits). Ignored if
                   precomputed is provided or model uses random item encoder.
        precomputed: tuple (unique_titles, item_embeds_cpu) from
                     build_item_embeddings(). Pass this to avoid re-encoding
                     every epoch.
        item2idx: mapping from item_id string to embedding index (required for
                  random item encoder mode).
        eval_seed: fixed seed for reproducible negative sampling
    """
    model.eval()
    device = next(model.parameters()).device
    use_random = getattr(model, "use_random_item_encoder", False)

    if use_random:
        num_items = model.item_encoder.embedding.num_embeddings - 1
        item_embeds_cpu = build_random_item_embeddings(model, num_items)
        N_items = num_items + 1
    elif precomputed is not None:
        unique_titles, item_embeds_cpu = precomputed
        title_to_idx = {t: i for i, t in enumerate(unique_titles)}
        N_items = len(unique_titles)
    elif item_pool is not None:
        titles, base = build_item_base_embeddings(model, item_pool)
        unique_titles = titles
        item_embeds_cpu = apply_item_projection(model, base)
        title_to_idx = {t: i for i, t in enumerate(unique_titles)}
        N_items = len(unique_titles)
    else:
        all_titles: list[str] = []
        for batch in dataloader:
            all_titles.extend(batch["target_titles"])
        titles, base = build_item_base_embeddings(model, all_titles)
        unique_titles = titles
        item_embeds_cpu = apply_item_projection(model, base)
        title_to_idx = {t: i for i, t in enumerate(unique_titles)}
        N_items = len(unique_titles)

    results: dict[str, list[float]] = {
        f"{metric}@{k}": [] for k in ks for metric in ["hr", "ndcg", "recall"]
    }
    results["mrr"] = []

    rng = torch.Generator()
    rng.manual_seed(eval_seed)

    # Move item embeddings to GPU once — avoids repeated CPU→GPU transfers
    item_embeds = item_embeds_cpu.to(device)

    for batch in tqdm(dataloader, desc="Evaluating"):
        if use_random and item2idx is not None:
            indices = [item2idx.get(iid, 0) for iid in batch["target_item_ids"]]
            batch["target_item_indices"] = torch.tensor(indices, dtype=torch.long)
        batch_on_device = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        output = model(batch_on_device)
        user_repr = F.normalize(output["user_memory"], dim=-1)  # [B, D] on GPU

        B = user_repr.shape[0]

        # Resolve target indices for the whole batch
        target_idxs = []
        for i in range(B):
            if use_random:
                item_id = batch["target_item_ids"][i]
                target_idxs.append(item2idx.get(item_id) if item2idx else None)
            else:
                target_idxs.append(title_to_idx.get(batch["target_titles"][i]))

        # Sample negatives with randint (much faster than randperm over 260k items)
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

        cand_indices = torch.stack(cand_list).to(device)          # [B, 1+num_neg]
        cand_embeds = item_embeds[cand_indices]                    # [B, 1+num_neg, D]
        scores = torch.bmm(
            user_repr.unsqueeze(1), cand_embeds.transpose(1, 2)
        ).squeeze(1)                                               # [B, 1+num_neg]
        ranked = torch.argsort(scores, dim=1, descending=True).cpu().numpy()

        gt = {0}
        for i, r in enumerate(ranked):
            if not valid_mask[i]:
                continue
            for k in ks:
                results[f"hr@{k}"].append(hit_rate(r, gt, k))
                results[f"ndcg@{k}"].append(ndcg(r, gt, k))
                results[f"recall@{k}"].append(recall(r, gt, k))
            results["mrr"].append(mrr(r, gt))

    return {k: float(np.mean(v)) for k, v in results.items() if v}
