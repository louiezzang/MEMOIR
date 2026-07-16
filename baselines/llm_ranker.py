"""LLMRanker: Prompt-based LLM recommendation without fine-tuning (zero-shot baseline)."""

from __future__ import annotations

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_device, get_autocast_ctx


class LLMRanker(nn.Module):
    """Zero/few-shot LLM-based ranking via prompting.

    Uses LLM to directly rank candidate items given user history.
    No training required - serves as a strong LLM baseline.
    """

    RANKING_PROMPT = (
        "You are a recommendation system. Based on the user's interaction history, "
        "rank the candidate items from most to least relevant.\n\n"
        "User history:\n{history}\n\n"
        "Candidate items:\n{candidates}\n\n"
        "Rank the items by relevance (most relevant first). "
        "Output ONLY the item numbers in order, separated by commas.\n"
        "Ranking:"
    )

    POINTWISE_PROMPT = (
        "You are a recommendation system. Rate how relevant this item is for the user "
        "on a scale of 1-10.\n\n"
        "User history:\n{history}\n\n"
        "Item: {item}\n\n"
        "Relevance score (1-10):"
    )

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_4bit: bool = False,
        max_length: int = 1024,
        mode: str = "pointwise",
    ):
        super().__init__()
        self.max_length = max_length
        self.mode = mode
        self.device = get_device()

        load_kwargs: dict = {"torch_dtype": torch.float16}

        if load_in_4bit and self.device.type == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        elif self.device.type == "mps":
            load_kwargs["torch_dtype"] = torch.float32
        else:
            load_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def pointwise_score(
        self,
        history_texts: list[str],
        item_titles: list[str],
    ) -> torch.Tensor:
        """Score each (user, item) pair independently. Returns [B] scores."""
        prompts = [
            self.POINTWISE_PROMPT.format(history=h, item=t)
            for h, t in zip(history_texts, item_titles)
        ]

        inputs = self.tokenizer(
            prompts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.model.device)

        with get_autocast_ctx(self.model.device):
            outputs = self.model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        scores = []
        for i in range(len(prompts)):
            generated = self.tokenizer.decode(
                outputs[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()
            score = self._parse_score(generated)
            scores.append(score)

        return torch.tensor(scores, dtype=torch.float32)

    @torch.no_grad()
    def listwise_rank(
        self,
        history_text: str,
        candidate_titles: list[str],
    ) -> list[int]:
        """Rank a list of candidates for a single user. Returns ranking indices."""
        candidates_str = "\n".join(
            f"{i+1}. {title}" for i, title in enumerate(candidate_titles)
        )
        prompt = self.RANKING_PROMPT.format(
            history=history_text, candidates=candidates_str,
        )

        inputs = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, return_tensors="pt",
        ).to(self.model.device)

        with get_autocast_ctx(self.model.device):
            outputs = self.model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()

        return self._parse_ranking(generated, len(candidate_titles))

    @staticmethod
    def _parse_score(text: str) -> float:
        numbers = re.findall(r'\d+', text)
        if numbers:
            score = float(numbers[0])
            return min(max(score, 1.0), 10.0) / 10.0
        return 0.5

    @staticmethod
    def _parse_ranking(text: str, num_items: int) -> list[int]:
        numbers = re.findall(r'\d+', text)
        ranking = []
        seen = set()
        for n in numbers:
            idx = int(n) - 1
            if 0 <= idx < num_items and idx not in seen:
                ranking.append(idx)
                seen.add(idx)
        for i in range(num_items):
            if i not in seen:
                ranking.append(i)
        return ranking

    def forward(self, **kwargs):
        raise NotImplementedError(
            "LLMRanker is inference-only. Use pointwise_score() or listwise_rank()."
        )
