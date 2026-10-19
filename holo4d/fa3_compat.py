"""Select one FlashAttention-3 provider before xFormers is imported.

A shared cluster environment may contain two Hopper FA3 extensions: Holo4D's
standalone ``flash_attn_interface`` and xFormers' bundled
``flash_attn_3._C``.  Loading both aborts the interpreter because both define
the same Torch operator namespace.  Holo4D uses the standalone interface, so
we hide only the optional ``flash_attn_3`` package from dependency discovery;
``flash_attn_interface`` remains importable and is used by WanVideoDiT.

Call :func:`configure_flash_attention_imports` before importing diffusers,
transformers, xformers, or any VideoX-Fun model module.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Callable


def _resolved_name(fullname: str, package: str | None) -> str:
    if fullname.startswith(".") and package:
        try:
            return importlib.util.resolve_name(fullname, package)
        except (ImportError, ValueError):
            return fullname
    return fullname


def _is_duplicate_fa3_package(fullname: str, package: str | None) -> bool:
    resolved = _resolved_name(fullname, package)
    return (
        resolved == "flash_attn_3"
        or resolved.startswith("flash_attn_3.")
        or resolved == "xformers.flash_attn_3"
        or resolved.startswith("xformers.flash_attn_3.")
    )


def configure_flash_attention_imports() -> str:
    """Install the FA3 namespace guard and return the requested backend.

    ``HOLO4D_FLASH_ATTN3=1`` selects the standalone Hopper kernel while hiding
    the duplicate xFormers/package extension.  ``DISABLE_FLASH_ATTN_3=1``
    hides the same optional package and lets the model select FA2.  The patch
    is process-local, idempotent, and never changes the conda environment.
    """

    use_fa3 = os.environ.get("HOLO4D_FLASH_ATTN3", "0") == "1"
    disable_fa3 = os.environ.get("DISABLE_FLASH_ATTN_3", "0") == "1"
    if not (use_fa3 or disable_fa3):
        return "auto"

    current: Callable = importlib.util.find_spec
    if not getattr(current, "__holo4d_fa3_namespace_guard__", False):
        original = current

        def find_spec_without_duplicate_fa3(fullname, package=None):
            if _is_duplicate_fa3_package(fullname, package):
                return None
            return original(fullname, package)

        find_spec_without_duplicate_fa3.__holo4d_fa3_namespace_guard__ = True
        find_spec_without_duplicate_fa3.__holo4d_original_find_spec__ = original
        importlib.util.find_spec = find_spec_without_duplicate_fa3

    # Diffusers caches package availability at import time. Import it only
    # after installing the guard, then make the choice explicit.
    try:
        from diffusers.utils import import_utils as diffusers_import_utils

        diffusers_import_utils._flash_attn_3_available = False
    except ImportError:
        pass

    return "fa3-standalone" if use_fa3 and not disable_fa3 else "fa2"
