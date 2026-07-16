"""TALLRec: An Effective and Efficient Framework for LLM-based Recommendation (RecSys 2023)."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from utils import get_device, get_autocast_ctx


class TALLRec(nn.Module):
    """LoRA-tuned LLM for sequential recommendation via instruction tuning.

    Reference: Bao et al., "TALLRec: An Effective and Efficient Framework
    for Text Alignment for LLM-based Recommendation", RecSys 2023.
    """

    PROMPT_TEMPLATE = (
        "A user has interacted with the following items in order:\n"
        "{history}\n\n"
        "Based on the user's history, will the user enjoy \"{candidate}\"?\n"
        "Answer:"
    )

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_4bit: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_length: int = 512,
    ):
        super().__init__()
        self.max_length = max_length
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

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
        self.model = get_peft_model(self.model, lora_config)

        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

    def _build_prompts(self, history_texts: list[str], candidates: list[str]) -> list[str]:
        prompts = []
        for hist, cand in zip(history_texts, candidates):
            prompts.append(self.PROMPT_TEMPLATE.format(history=hist, candidate=cand))
        return prompts

    def forward(
        self,
        history_texts: list[str],
        candidates: list[str],
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prompts = self._build_prompts(history_texts, candidates)

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.model.device)

        with get_autocast_ctx(self.model.device):
            outputs = self.model(**inputs)
            logits = outputs.logits

        seq_lens = inputs["attention_mask"].sum(dim=1) - 1
        B = logits.shape[0]
        last_logits = logits[torch.arange(B, device=logits.device), seq_lens.long()]

        yes_no_logits = torch.stack([
            last_logits[:, self.no_token_id],
            last_logits[:, self.yes_token_id],
        ], dim=-1)

        scores = torch.softmax(yes_no_logits.float(), dim=-1)[:, 1]

        result = {"scores": scores}
        if labels is not None:
            result["loss"] = nn.functional.binary_cross_entropy(scores, labels.float())

        return result
