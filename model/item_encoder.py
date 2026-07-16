"""Item encoder: encodes item text/metadata into embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class MoEAdaptor(nn.Module):
    """Mixture-of-experts adaptor: gated combination of linear experts.

    Same structure as UniSRec's adaptor (Hou et al., KDD 2022) — gives the item
    projection more capacity to reshape frozen text embeddings than a plain
    linear/MLP projection.
    """

    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = torch.softmax(self.gate(x), dim=-1)  # [..., num_experts]
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-2,
        )  # [..., num_experts, output_dim]
        return (gate_scores.unsqueeze(-1) * expert_outputs).sum(dim=-2)


class ItemEncoder(nn.Module):
    """Encodes item titles into embeddings via a frozen pretrained text encoder.

    Raw sentence-transformer embeddings are whitened (mean-centered + decorrelated)
    to correct their anisotropy before being passed through a trainable
    mixture-of-experts adaptor — matching UniSRec's item representation pipeline
    (Hou et al., KDD 2022) so the two share the same architectural ingredients.
    """

    def __init__(
        self,
        pretrained: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 256,
        freeze_base: bool = True,
        num_experts: int = 2,
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(pretrained)
        base_dim = self.encoder.get_embedding_dimension()

        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.register_buffer("whitening_mean", torch.zeros(base_dim))
        self.register_buffer("whitening_matrix", torch.eye(base_dim))

        self.projection = MoEAdaptor(base_dim, output_dim, num_experts=num_experts)

    @torch.no_grad()
    def set_whitening(self, base_embeds: torch.Tensor) -> None:
        """Fit a parametric whitening transform from a sample of frozen base
        embeddings: mean-center, then rotate/rescale by the inverse sqrt of the
        covariance eigenvalues. Corrects the anisotropy that raw
        sentence-transformer embeddings otherwise have. Call once before
        training, using the full item pool's base embeddings.
        """
        embeds = base_embeds.double()
        mean = embeds.mean(dim=0)
        centered = embeds - mean
        cov = centered.T @ centered / max(embeds.shape[0] - 1, 1)
        U, S, _ = torch.linalg.svd(cov)
        W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-5))
        self.whitening_mean.copy_(mean.float())
        self.whitening_matrix.copy_(W.float())

    def _whiten(self, base_embeds: torch.Tensor) -> torch.Tensor:
        mean = self.whitening_mean.to(base_embeds.dtype)
        matrix = self.whitening_matrix.to(base_embeds.dtype)
        return (base_embeds - mean) @ matrix

    def forward(self, titles: list[str]) -> torch.Tensor:
        """Encode item titles. Returns [len(titles), output_dim]."""
        with torch.no_grad():
            base_embeds = self.encoder.encode(
                titles, convert_to_tensor=True, show_progress_bar=False,
            ).clone().float()
        return self.catalog_embeddings(base_embeds)

    def catalog_embeddings(self, base_embeds: torch.Tensor) -> torch.Tensor:
        """Apply whitening + the trainable MoE adaptor to precomputed frozen base
        embeddings. Used both by forward() for single-batch encoding and for
        full-catalog scoring, where base_embeds (MiniLM output, frozen) are
        precomputed once and re-projected fresh on every call so gradients reach
        the adaptor regardless of caching.

        Always whitens, even before set_whitening() has been called — the
        buffers default to a no-op transform (mean=0, matrix=identity), so this
        is safe pre-fit and correct after loading a trained checkpoint, where
        the buffer values are restored via state_dict but a plain instance flag
        would not be.
        """
        return self.projection(self._whiten(base_embeds))


class RandomItemEncoder(nn.Module):
    """Random-init ID-based item encoder (ablation: no pretrained text representations)."""

    def __init__(self, num_items: int, output_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, output_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight[1:])

    def forward(self, item_indices: torch.Tensor) -> torch.Tensor:
        """Lookup item embeddings by index. Returns [B, output_dim]."""
        return self.embedding(item_indices)

    def catalog_embeddings(self, base_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Full catalog is just the trainable embedding table itself."""
        return self.embedding.weight
