"""UniSRec: Universal Sequence Representation Learning for Recommendation (KDD 2022).

Reference: Hou et al., "Towards Universal Sequence Representation Learning for
Recommender Systems", KDD 2022.

Key idea: Use pretrained text encoder (MiniLM) to encode item descriptions,
apply whitening to remove anisotropy, then learn a Mixture-of-Experts adaptor
to project text embeddings into the sequential recommendation space.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEAdaptor(nn.Module):
    """Mixture-of-Experts adaptor that transforms pretrained text embeddings."""

    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [..., num_experts]
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-2,
        )  # [..., num_experts, output_dim]
        return (gate_scores.unsqueeze(-1) * expert_outputs).sum(dim=-2)


class UniSRec(nn.Module):
    """UniSRec: Whitening + MoE adaptor + SASRec backbone on text embeddings.

    Reference: Hou et al., KDD 2022.
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        semantic_dim: int = 384,
        num_experts: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.semantic_dim = semantic_dim

        self.register_buffer("whitened_embeddings", torch.zeros(num_items + 1, semantic_dim))
        self.moe_adaptor = MoEAdaptor(semantic_dim, embed_dim, num_experts)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
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
        self._text_ready = False

    @torch.no_grad()
    def set_text_data(self, item_texts: list[str], logger=None) -> None:
        """Encode item texts, apply whitening, store as buffer."""
        from sentence_transformers import SentenceTransformer
        import time

        def _log(msg: str):
            print(msg, flush=True)
            if logger is not None:
                logger.log_debug(msg)

        _log("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        N = len(item_texts)
        _log(f"Encoding {N} item texts...")
        t0 = time.time()
        item_emb = st_model.encode(item_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
        item_emb = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8)
        item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        _log(f"Item encoding done in {time.time() - t0:.1f}s, shape={item_emb.shape}")

        del st_model
        gc.collect()

        # Whitening: mean-center + PCA rotation to remove anisotropy
        _log("Applying whitening transform...")
        mean = item_emb.mean(axis=0)
        centered = item_emb - mean
        cov = centered.T @ centered / max(N - 1, 1)
        U, S, _ = np.linalg.svd(cov)
        W = U @ np.diag(1.0 / np.sqrt(S + 1e-5))
        whitened = centered @ W
        whitened = whitened.astype(np.float32)
        _log(f"Whitening done, shape={whitened.shape}")

        device = self.whitened_embeddings.device
        self.whitened_embeddings.copy_(torch.from_numpy(whitened).to(device))
        self._text_ready = True
        _log("UniSRec: text data ready.")

    def _get_item_embeddings(self, item_indices: torch.Tensor) -> torch.Tensor:
        raw = self.whitened_embeddings[item_indices]
        return self.moe_adaptor(raw)

    def forward(self, item_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        B, L = item_seq.shape
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self._get_item_embeddings(item_seq) + self.position_embedding(positions)
        x = self.layer_norm(self.dropout_layer(x))

        causal_mask = torch.triu(
            torch.ones(L, L, device=item_seq.device, dtype=torch.bool), diagonal=1,
        )
        if item_seq.device.type == "mps":
            x = self.encoder(x, mask=causal_mask)
        else:
            x = self.encoder(x, mask=causal_mask, src_key_padding_mask=(item_seq == 0))

        last_idx = (seq_len - 1).clamp(min=0).long()
        user_repr = x[torch.arange(B, device=x.device), last_idx]
        return user_repr

    def get_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
        all_adapted = self.moe_adaptor(self.whitened_embeddings)
        return torch.matmul(user_repr, all_adapted.T)

    def compute_loss(
        self,
        item_seq: torch.Tensor,
        seq_len: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        user_repr = self.forward(item_seq, seq_len)
        scores = self.get_scores(user_repr)
        return F.cross_entropy(scores, target)
