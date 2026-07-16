"""LLM-based Behavioral Memory Encoder: converts window text into semantic memory embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

from utils import get_device, get_autocast_ctx


class EvolutionAwareAggregator(nn.Module):
    """Replaces simple recency-weighted mean with trajectory-aware aggregation.

    Concatenates [current_memory, direction_vector, predicted_next] and
    projects down to the embedding dimension.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self,
        current_memory: torch.Tensor,
        direction: torch.Tensor,
        predicted_next: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse current state + direction + predicted future.

        Args:
            current_memory: [B, D] recency-weighted aggregate (baseline)
            direction: [B, D] evolution direction from TrajectoryPredictor
            predicted_next: [B, D] predicted next-step memory

        Returns:
            [B, D] evolution-aware aggregate memory
        """
        fused = torch.cat([current_memory, direction, predicted_next], dim=-1)
        return self.fuse(fused)


class LLMMemoryEncoder(nn.Module):
    """Encodes behavior window texts into semantic memory embeddings using an LLM."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        output_dim: int = 256,
        load_in_4bit: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        max_length: int = 128,
        encode_batch_size: int = 16,
        freeze_llm: bool = False,
    ):
        super().__init__()
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size
        self.freeze_llm = freeze_llm
        self._text_cache: dict | None = None
        self.device = get_device()

        load_kwargs: dict = {"dtype": torch.float16}

        if load_in_4bit and self.device.type == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        elif self.device.type == "mps":
            load_kwargs["dtype"] = torch.float16
        else:
            load_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModel.from_pretrained(model_name, **load_kwargs)

        # Move to device if not using device_map="auto"
        if "device_map" not in load_kwargs:
            self.llm = self.llm.to(self.device)

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        else:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.enable_input_require_grads()
            self.llm.gradient_checkpointing_enable()

        hidden_size = self.llm.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, output_dim),
        )

    def set_text_cache(self, cache: dict[str, torch.Tensor]) -> None:
        """Store a pre-computed text→pooled-hidden-state mapping (CPU tensors, pre-projection)."""
        self._text_cache = cache

    def _pool_chunk(self, texts: list[str]) -> torch.Tensor:
        """Run the LLM and mean-pool hidden states for one chunk. Returns [len(texts), hidden_size].

        This is the expensive part that is frozen when freeze_llm=True, so it is safe
        to cache. The trainable projection is applied separately in encode_text() so
        gradients still reach it even when this output comes from the cache.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.llm.device)

        with get_autocast_ctx(self.llm.device):
            outputs = self.llm(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = outputs.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return pooled.float()

    def encode_pooled(self, texts: list[str]) -> torch.Tensor:
        """Mean-pooled LLM hidden states before projection. Returns [len(texts), hidden_size]."""
        if self._text_cache is not None:
            device = next(self.parameters()).device
            cached = [self._text_cache.get(t) for t in texts]
            if all(e is not None for e in cached):
                return torch.stack([e.to(device) for e in cached])

        chunks = [
            texts[i : i + self.encode_batch_size]
            for i in range(0, len(texts), self.encode_batch_size)
        ]
        return torch.cat([self._pool_chunk(chunk) for chunk in chunks], dim=0)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of texts into embeddings. Returns [len(texts), output_dim].

        Always applies the trainable projection fresh, even when the pooled
        hidden state came from the cache, so the projection keeps receiving
        gradients under freeze_llm=True.
        """
        return self.projection(self.encode_pooled(texts))

    def forward(
        self,
        window_texts: list[list[str]],
        window_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Encode all windows for a batch of users.

        Args:
            window_texts: [B, max_windows] list of window text strings
            window_masks: [B, max_windows] bool tensor

        Returns:
            dict with:
                base_aggregate: [B, D] recency-weighted aggregate (baseline)
                memory_grid: [B, W, D] full grid of per-window embeddings
                per_window: list of [num_valid_windows, D] per user
        """
        B, W = len(window_texts), len(window_texts[0])

        flat_texts = []
        indices = []
        for i in range(B):
            for j in range(W):
                if window_masks[i, j]:
                    flat_texts.append(window_texts[i][j])
                    indices.append((i, j))

        if not flat_texts:
            device = next(self.parameters()).device
            dim = self.projection[-1].out_features
            dummy = torch.zeros(B, dim, device=device)
            grid = torch.zeros(B, W, dim, device=device)
            return {
                "base_aggregate": dummy,
                "memory_grid": grid,
                "per_window": [dummy[i:i+1] for i in range(B)],
            }

        all_embeds = self.encode_text(flat_texts)

        dim = all_embeds.shape[-1]
        device = all_embeds.device
        memory_grid = torch.zeros(B, W, dim, device=device)
        for idx, (i, j) in enumerate(indices):
            memory_grid[i, j] = all_embeds[idx]

        weights = torch.arange(1, W + 1, dtype=torch.float32, device=device).unsqueeze(0)
        weights = weights * window_masks.float().to(device)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)

        base_aggregate = (memory_grid * weights.unsqueeze(-1)).sum(dim=1)

        per_window = []
        for i in range(B):
            valid = window_masks[i].sum().item()
            per_window.append(memory_grid[i, :valid])

        return {
            "base_aggregate": base_aggregate,
            "memory_grid": memory_grid,
            "per_window": per_window,
        }
