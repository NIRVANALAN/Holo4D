"""Lightweight package bootstrap for the public Holo4D model stack.

The CUDA 13 environment can expose two FlashAttention-3 shared libraries
(standalone and xformers-bundled). They register the same Torch namespace and
can segfault during import on SM80. H100/H200 (SM90) keep FA3; A100 (SM80)
automatically uses the tested FA2 path. Set ``HOLO4D_FLASH_ATTN3=1`` to force
FA3 or ``DISABLE_FLASH_ATTN_3=1``/``HOLO4D_FLASH_ATTN3=0`` to force FA2.
"""

from __future__ import annotations

import importlib.util
import os


def _a100_should_disable_fa3() -> bool:
    override = os.environ.get("HOLO4D_FLASH_ATTN3", "auto").strip().lower()
    if override in {"0", "false", "off", "no"}:
        return True
    if override in {"1", "true", "on", "yes"}:
        return False
    if os.environ.get("DISABLE_FLASH_ATTN_3", "0") == "1":
        return True
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9)
    except Exception:
        # CPU-only imports should remain dependency-light; the inference
        # launcher sets DISABLE_FLASH_ATTN_3 explicitly on A100.
        return False


if _a100_should_disable_fa3() and not getattr(importlib.util, "_holo4d_fa3_guard", False):
    _holo4d_original_find_spec = importlib.util.find_spec

    def _holo4d_find_spec_without_fa3(fullname, package=None):
        if fullname == "flash_attn_3" or fullname.startswith("flash_attn_3.") or ".flash_attn_3" in fullname:
            return None
        return _holo4d_original_find_spec(fullname, package)

    importlib.util.find_spec = _holo4d_find_spec_without_fa3
    importlib.util._holo4d_fa3_guard = True
