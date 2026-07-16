"""Shared utilities for MEMOIR."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_ctx(device: torch.device):
    """Return appropriate autocast context manager for the device."""
    import contextlib
    if device.type in ("cuda", "mps"):
        return torch.amp.autocast(device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def supports_grad_scaler(device: torch.device) -> bool:
    """GradScaler only works with CUDA."""
    return device.type == "cuda"
