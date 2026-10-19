"""Low-overhead opt-in torch profiler spans for Track4D-360.

The context manager is a no-op unless TRACK4D360_PROFILE_SPANS=1. This keeps
normal training free of Python profiler context overhead while giving torch
profiler traces readable names when debugging forward-time jitter.
"""

import os
from typing import Iterable
from contextlib import nullcontext


def track4d_profile_spans_enabled() -> bool:
    value = str(os.environ.get("TRACK4D360_PROFILE_SPANS", "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def track4d_record_function(name: str):
    if not track4d_profile_spans_enabled():
        return nullcontext()
    try:
        from torch.profiler import record_function
    except Exception:
        return nullcontext()
    return record_function(name)


def _track4d_env_flag(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _truncate_profile_name(name: str, max_len: int) -> str:
    if len(name) <= max_len:
        return name
    keep = max(8, max_len - 12)
    return f"{name[:keep]}...{abs(hash(name)) & 0xfffffff:07x}"


def enable_track4d_conv3d_module_profile_spans(
    roots: object | Iterable[object],
    *,
    prefix: str = "pipe",
    max_name_len: int = 220,
) -> int:
    """Wrap ``nn.Conv3d`` forwards with profiler spans when explicitly enabled.

    This is diagnostic-only and opt-in via
    ``TRACK4D360_PROFILE_CONV3D_MODULES=1``. It is meant for cases where
    ``aten::slow_conv_dilated3d`` appears outside existing Track4D spans: the
    parent span will name the concrete module path that issued the Conv3D.
    """
    if not _track4d_env_flag("TRACK4D360_PROFILE_CONV3D_MODULES"):
        return 0

    try:
        import torch.nn as nn
    except Exception:
        return 0

    if roots is None:
        return 0
    if hasattr(roots, "named_modules"):
        root_items = [(prefix, roots)]
    else:
        root_items = []
        for idx, root in enumerate(roots):
            if root is not None and hasattr(root, "named_modules"):
                root_items.append((f"{prefix}{idx}", root))

    wrapped = 0
    for root_prefix, root in root_items:
        for name, module in root.named_modules():
            if not isinstance(module, nn.Conv3d):
                continue
            if getattr(module, "_track4d_conv3d_profile_wrapped", False):
                continue
            module_name = f"{root_prefix}.{name}" if name else root_prefix
            label = _truncate_profile_name(
                f"track4d360/conv3d_module/{module_name}",
                max_name_len,
            )
            orig_forward = module.forward

            def wrapped_forward(*args, __orig_forward=orig_forward, __label=label, **kwargs):
                with track4d_record_function(__label):
                    return __orig_forward(*args, **kwargs)

            module.forward = wrapped_forward
            module._track4d_conv3d_profile_wrapped = True
            module._track4d_conv3d_profile_label = label
            wrapped += 1
    return wrapped
