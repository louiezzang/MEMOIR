"""SRA-CL: Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation.

Reference: Cui et al., NeurIPS 2025. (arXiv 2503.04162)
Official repo: https://github.com/ziqiangcui/SRA-CL-NeurIPS25

Two contrastive objectives on top of a SASRec backbone:
  L_IS (intra-sequence): augment item sequences by replacing ~20% of items with
       semantically similar neighbors, then InfoNCE between two augmented views.
  L_CS (cross-sequence): for each user, retrieve top-k semantically similar users,
       encode their sequences through the backbone, weight via a learnable GAT-style
       attention adapter, then InfoNCE between anchor and weighted neighbor repr.

Adaptation notes:
  - Uses all-MiniLM-L6-v2 (384d) instead of SimCSE-RoBERTa (1024d) for semantic
    embeddings, since it is already a project dependency.
  - Item titles and user window texts substitute for the offline LLM summary pipeline.
  - Neighbor sequences are encoded with gradients detached to limit memory usage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRACL(nn.Module):

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        temperature: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.1,
        k_neighbors: int = 10,
        mlm_probability: float = 0.2,
        semantic_dim: int = 384,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.k_neighbors = k_neighbors
        self.mlm_probability = mlm_probability
        self.semantic_dim = semantic_dim
        self.pad_idx = pad_idx

        # ---- SASRec backbone ----
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # ---- GAT-style learnable sample synthesizer (cross-sequence CL) ----
        self.W = nn.Parameter(torch.empty(semantic_dim, embed_dim))
        self.a = nn.Parameter(torch.empty(2 * embed_dim, 1))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # ---- Buffers populated by set_semantic_data() ----
        self._semantic_ready = False
        self.register_buffer("item_neighbors", torch.zeros(num_items + 1, k_neighbors, dtype=torch.long))
        self.register_buffer("user_neighbors", torch.zeros(1, k_neighbors, dtype=torch.long))
        self.register_buffer("user_semantic_emb", torch.zeros(1, semantic_dim))
        self.register_buffer("all_seqs_tensor", torch.zeros(1, max_len, dtype=torch.long))
        self.register_buffer("all_lens_tensor", torch.ones(1, dtype=torch.long))

    # ------------------------------------------------------------------
    # Semantic data initialization (called once after construction)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_semantic_data(
        self,
        item_texts: list[str],
        user_texts_dict: dict,
        user_sequences_dict: dict,
        user2idx: dict,
        logger=None,
    ) -> None:
        """Encode item/user texts, build neighbor indices, store user sequences."""
        from sentence_transformers import SentenceTransformer
        import time

        def _log(msg: str):
            print(msg, flush=True)
            if logger is not None:
                logger.log_debug(msg)

        device = self.item_embedding.weight.device
        import numpy as np
        import gc

        # --- Encode all texts first, then free SentenceTransformer ---
        _log("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        N = len(item_texts)
        _log(f"Encoding {N} item texts...")
        t0 = time.time()
        item_emb = st_model.encode(item_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
        item_emb = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8)
        item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        _log(f"Item encoding done in {time.time() - t0:.1f}s, shape={item_emb.shape}")

        ordered_uids = sorted(user2idx, key=user2idx.get)
        user_texts_ordered = [user_texts_dict.get(uid, "") for uid in ordered_uids]
        M = len(user_texts_ordered)
        _log(f"Encoding {M} user texts...")
        t0 = time.time()
        user_emb = st_model.encode(user_texts_ordered, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
        user_emb = user_emb / (np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-8)
        user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
        _log(f"User encoding done in {time.time() - t0:.1f}s, shape={user_emb.shape}")

        del st_model
        gc.collect()
        _log("SentenceTransformer freed, starting neighbor search...")

        def _chunked_neighbors(emb: np.ndarray, k_neighbors: int, label: str) -> torch.Tensor:
            """Chunked cosine-sim neighbor search (numpy, no FAISS)."""
            N_emb = emb.shape[0]
            _log(f"Computing {label} neighbors for {N_emb} vectors (k={k_neighbors})...")
            t1 = time.time()
            result = torch.zeros(N_emb, k_neighbors, dtype=torch.long)
            chunk_size = 2048
            for start in range(0, N_emb, chunk_size):
                end = min(start + chunk_size, N_emb)
                sims = emb[start:end] @ emb.T  # [chunk, N]
                for i in range(end - start):
                    sims[i, start + i] = -1.0  # exclude self
                top_idx = np.argpartition(-sims, k_neighbors, axis=1)[:, :k_neighbors]
                for i in range(end - start):
                    row_top = top_idx[i]
                    row_sims = sims[i, row_top]
                    sorted_order = np.argsort(-row_sims)
                    result[start + i] = torch.from_numpy(row_top[sorted_order].copy())
                if (end // chunk_size) % 10 == 0 or end == N_emb:
                    _log(f"  {label} {end}/{N_emb} ({time.time() - t1:.1f}s)")
            _log(f"  {label} neighbors done in {time.time() - t1:.1f}s")
            return result

        # --- Item neighbors ---
        t0 = time.time()
        self.item_neighbors = _chunked_neighbors(item_emb, self.k_neighbors, "items").to(device)
        del item_emb
        gc.collect()
        _log(f"Item neighbors total: {time.time() - t0:.1f}s")

        # --- User neighbors ---
        t0 = time.time()
        self.user_neighbors = _chunked_neighbors(user_emb, self.k_neighbors, "users").to(device)
        self.user_semantic_emb = torch.from_numpy(user_emb).float().to(device)
        del user_emb
        gc.collect()
        _log(f"User neighbors total: {time.time() - t0:.1f}s")

        # --- Store user sequences as tensors for efficient neighbor lookup ---
        num_users = len(user2idx)
        seqs_t = torch.zeros(num_users, self.max_len, dtype=torch.long)
        lens_t = torch.ones(num_users, dtype=torch.long)
        for uid, idx in user2idx.items():
            seq = user_sequences_dict.get(uid, [])
            seq = seq[-self.max_len:]
            slen = min(len(seq), self.max_len)
            if slen > 0:
                seqs_t[idx, :slen] = torch.tensor(seq[-slen:], dtype=torch.long)
                lens_t[idx] = slen
        self.all_seqs_tensor = seqs_t.to(device)
        self.all_lens_tensor = lens_t.to(device)
        _log(f"Stored {num_users} user sequences as tensors")

        self._semantic_ready = True

    # ------------------------------------------------------------------
    # SASRec backbone
    # ------------------------------------------------------------------

    def _encode(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        B, L = item_seq.shape
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(item_seq) + self.position_embedding(positions)
        x = self.layer_norm(self.emb_dropout(x))

        causal_mask = torch.triu(torch.ones(L, L, device=item_seq.device, dtype=torch.bool), diagonal=1)
        padding_mask = item_seq == self.pad_idx
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        last_idx = (seq_len - 1).clamp(min=0).long()
        return x[torch.arange(B, device=x.device), last_idx]

    def forward(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        return self._encode(item_seq, seq_len)

    def get_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
        return user_repr @ self.item_embedding.weight[: self.num_items + 1].T

    # ------------------------------------------------------------------
    # InfoNCE (matches official SRA-CL: 2N concatenation with correlated mask)
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_correlated_samples(batch_size: int) -> torch.Tensor:
        N = 2 * batch_size
        mask = torch.ones(N, N, dtype=torch.bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE with 2N concatenation and correlated-sample masking."""
        B = z1.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z1.device)
        N = 2 * B
        z = torch.cat([F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        pos_ij = torch.diag(sim, B)
        pos_ji = torch.diag(sim, -B)
        positives = torch.cat([pos_ij, pos_ji]).reshape(N, 1)
        mask = self._mask_correlated_samples(B).to(z1.device)
        negatives = sim[mask].reshape(N, -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=z1.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Intra-sequence CL: semantic item substitution
    # ------------------------------------------------------------------

    def _augment_items(self, item_seq: torch.Tensor) -> torch.Tensor:
        if not self._semantic_ready:
            return item_seq
        aug = item_seq.clone()
        mask = (torch.rand_like(item_seq.float()) < self.mlm_probability) & (item_seq != self.pad_idx)
        if not mask.any():
            return aug

        items_to_replace = item_seq[mask]
        valid = items_to_replace < self.item_neighbors.size(0)
        if not valid.any():
            return aug

        neighbor_rows = self.item_neighbors[items_to_replace[valid]]
        rand_col = torch.randint(0, self.k_neighbors, (neighbor_rows.size(0),), device=item_seq.device)
        replacements = neighbor_rows[torch.arange(neighbor_rows.size(0), device=item_seq.device), rand_col]

        full_replacements = items_to_replace.clone()
        full_replacements[valid] = replacements
        aug[mask] = full_replacements
        return aug

    # ------------------------------------------------------------------
    # Cross-sequence CL: user-level contrastive with GAT attention
    # ------------------------------------------------------------------

    def _gat_attention(self, anchor_sem: torch.Tensor, neighbor_sem: torch.Tensor) -> torch.Tensor:
        """GAT-style attention over neighbor semantic embeddings.

        Args:
            anchor_sem: [B, semantic_dim]
            neighbor_sem: [B, k, semantic_dim]
        Returns:
            weights: [B, k] softmax attention weights
        """
        k = neighbor_sem.size(1)
        anchor_proj = (anchor_sem @ self.W).unsqueeze(1).expand(-1, k, -1)  # [B, k, H]
        neighbor_proj = neighbor_sem @ self.W  # [B, k, H]
        concat = torch.cat([anchor_proj, neighbor_proj], dim=-1)  # [B, k, 2H]
        attn = self.leaky_relu((concat @ self.a).squeeze(-1))  # [B, k]
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.5, training=self.training)
        return attn

    def _compute_cross_sequence_cl(
        self,
        item_seq: torch.Tensor,
        seq_len: torch.Tensor,
        user_idx: torch.Tensor,
        neighbor_seqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._semantic_ready:
            return torch.tensor(0.0, device=item_seq.device)

        B = item_seq.size(0)
        device = item_seq.device

        valid_mask = (user_idx >= 0) & (user_idx < self.user_neighbors.size(0))
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        v_idx = user_idx[valid_mask]
        v_seq = item_seq[valid_mask]
        v_len = seq_len[valid_mask]
        Bv = v_idx.size(0)

        seq_output = self._encode(v_seq, v_len)  # [Bv, H]

        # Get neighbor sequences: pre-fetched or look up from tensor buffer
        if neighbor_seqs is not None:
            n_seqs_3d = neighbor_seqs[valid_mask]  # [Bv, k, L]
        else:
            neighbor_ids = self.user_neighbors[v_idx]  # [Bv, k]
            neighbor_ids_clamp = neighbor_ids.clamp(max=self.all_seqs_tensor.size(0) - 1)
            n_seqs_3d = self.all_seqs_tensor[neighbor_ids_clamp]  # [Bv, k, L]

        k = n_seqs_3d.size(1)
        n_seqs_flat = n_seqs_3d.reshape(Bv * k, -1)  # [Bv*k, L]
        n_lens_flat = (n_seqs_flat != self.pad_idx).sum(dim=-1).clamp(min=1)

        # Encode neighbors WITH gradients (matching official implementation)
        n_output = self._encode(n_seqs_flat, n_lens_flat)  # [Bv*k, H]
        n_output = n_output.view(Bv, k, -1)  # [Bv, k, H]

        anchor_sem = self.user_semantic_emb[v_idx]  # [Bv, sem_dim]
        neighbor_ids_for_sem = self.user_neighbors[v_idx]  # [Bv, k]
        neighbor_sem = self.user_semantic_emb[neighbor_ids_for_sem]  # [Bv, k, sem_dim]
        attn = self._gat_attention(anchor_sem, neighbor_sem)  # [Bv, k]

        weighted_neighbor = (attn.unsqueeze(-1) * n_output).sum(dim=1)  # [Bv, H]

        return self._info_nce(seq_output, weighted_neighbor)

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        item_seq: torch.Tensor,
        seq_len: torch.Tensor,
        target: torch.Tensor,
        user_idx: torch.Tensor | None = None,
        neighbor_seqs: torch.Tensor | None = None,
        aug_seq1: torch.Tensor | None = None,
        aug_seq2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_repr = self._encode(item_seq, seq_len)
        scores = self.get_scores(user_repr)
        rec_loss = F.cross_entropy(scores, target)

        # Intra-sequence CL: use pre-augmented views if provided, else augment on the fly
        if aug_seq1 is not None and aug_seq2 is not None:
            z1 = self._encode(aug_seq1, seq_len)
            z2 = self._encode(aug_seq2, seq_len)
        else:
            aug1 = self._augment_items(item_seq)
            aug2 = self._augment_items(item_seq)
            z1 = self._encode(aug1, seq_len)
            z2 = self._encode(aug2, seq_len)
        is_loss = self._info_nce(z1, z2)

        cs_loss = torch.tensor(0.0, device=item_seq.device)
        if user_idx is not None:
            cs_loss = self._compute_cross_sequence_cl(
                item_seq, seq_len, user_idx, neighbor_seqs=neighbor_seqs
            )

        return rec_loss + self.alpha * cs_loss + self.beta * is_loss
