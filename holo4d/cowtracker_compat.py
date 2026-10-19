"""CoWTracker constructor compatibility for the released raw-video runner."""
from __future__ import annotations

import functools
import importlib
import inspect
import os
from pathlib import Path
import sys


def _init_without_custom_fa3(self, original_init, *args, **kwargs):
    parameters = inspect.signature(original_init).parameters
    if 'use_flash_attention3' in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        kwargs["use_flash_attention3"] = False
    return original_init(self, *args, **kwargs)


def configure_cowtracker_attention(repo: Path) -> None:
    """Use timm's native SDPA attention unless custom upstream FA3 is requested.

    Older CoWTracker FA3 wrappers lack timm's is_causal argument and their
    xFormers kernels do not support every released GPU. This preserves the
    model weights while selecting the validated PyTorch Flash-SDPA path.
    """
    disabled = os.environ.get("COWTRACKER_DISABLE_FUSED_ATTN", "1").lower()
    if disabled not in {"1", "true", "yes", "on"}:
        return
    backend = os.environ.get("COWTRACKER_SDPA_BACKEND", "auto").lower()
    if backend not in {"auto", "flash", "math"}:
        raise ValueError(f"Unknown COWTRACKER_SDPA_BACKEND: {backend}")
    path = str(repo)
    if path not in sys.path:
        sys.path.insert(0, path)
    video_module = importlib.import_module("cowtracker.layers.video_transformer")
    cls = video_module.VisionTransformerVideo
    if not getattr(cls, "_holo4d_native_sdpa", False):
        cls.__init__ = functools.partialmethod(_init_without_custom_fa3, cls.__init__)
        cls._holo4d_native_sdpa = True
    import torch

    torch.backends.cuda.enable_flash_sdp(backend in {"auto", "flash"})
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(backend in {"auto", "math"})
    print(f"[cow-batch] custom FA3 disabled; PyTorch {backend} SDPA", flush=True)
