"""SASRec-Text: SASRec with pretrained MiniLM text embeddings (instead of random ID embeddings)."""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRecText(nn.Module):
    """SASRec backbone where item embeddings come from frozen MiniLM text encoder
    with a trainable linear projection.

    Reference backbone: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
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
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.semantic_dim = semantic_dim

        self.register_buffer("text_embeddings", torch.zeros(num_items + 1, semantic_dim))
        self.text_projection = nn.Linear(semantic_dim, embed_dim)
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
        """Encode item texts with frozen MiniLM and store as buffer."""
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

        device = self.text_embeddings.device
        self.text_embeddings.copy_(torch.from_numpy(item_emb).to(device))
        self._text_ready = True
        _log("SASRec-Text: text data ready.")

    def _get_item_embeddings(self, item_indices: torch.Tensor) -> torch.Tensor:
        raw = self.text_embeddings[item_indices]
        return self.text_projection(raw)

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
        all_projected = self.text_projection(self.text_embeddings)
        return torch.matmul(user_repr, all_projected.T)

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
