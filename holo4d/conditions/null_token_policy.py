"""Centralized policy helpers for the opt-in hard-zero null token and the
occupancy-channel checkpoint migration.

Why this module exists (2026-07-23 Codex review findings F1/F3):

F1 — constructor-time ``requires_grad_(False)`` on ``null_token`` is defeated
by ``freeze_except()`` (which re-enables grads on whole trainable modules) and
by resume-checkpoint loads (which restore a drifted nonzero null). The frozen
flag must therefore be re-enforced AFTER training-mode switch and AFTER every
checkpoint load, via one shared helper that zeroes AND freezes.

F3 — ``load_track_checkpoint`` filters shape-mismatched keys and loads with
``strict=False``; flipping ``TRACK4D360_TRACK_OCCUPANCY_CHANNEL`` between
train and eval therefore silently skipped the injector/projector input layer
(random init!) instead of failing loudly. This module provides an explicit
D → D+1 zero-column migration (legacy ckpt → occupancy model) and hard errors
for the reverse/incompatible directions.
"""

from __future__ import annotations

import os

import torch


def null_token_frozen_enabled() -> bool:
    return os.environ.get("TRACK4D360_TRACK_NULL_TOKEN_FROZEN", "0") == "1"


def enforce_frozen_null_tokens(*owners, log_prefix: str = "[Track4D-360]") -> int:
    """Zero AND freeze every ``null_token``-like parameter on the given owners.

    Idempotent; call after ``switch_pipe_to_training_mode`` and after every
    checkpoint load. ``owners`` may be modules with a ``null_token`` attribute
    or bare ``nn.Parameter`` objects; ``None`` entries are skipped. Returns
    the number of parameters enforced. No-op unless
    TRACK4D360_TRACK_NULL_TOKEN_FROZEN=1.
    """
    if not null_token_frozen_enabled():
        return 0
    enforced = 0
    for owner in owners:
        if owner is None:
            continue
        param = owner if isinstance(owner, torch.nn.Parameter) else getattr(owner, "null_token", None)
        if param is None:
            continue
        with torch.no_grad():
            param.zero_()
        param.requires_grad_(False)
        enforced += 1
    if enforced:
        print(f"{log_prefix} hard-zero null enforced on {enforced} null token(s) "
              "(post train-mode-switch / checkpoint-load)")
    return enforced


def enforce_frozen_null_tokens_on_pipe(pipe, log_prefix: str = "[Track4D-360]") -> int:
    """Convenience wrapper covering every null-token owner on the pipeline."""
    return enforce_frozen_null_tokens(
        getattr(pipe, "track_adapter", None),
        getattr(pipe, "dense_target_control_encoder", None),
        getattr(pipe, "track_null_token", None),
        getattr(pipe, "dense_null_token", None),
        log_prefix=log_prefix,
    )


def migrate_state_for_occupancy_channel(
    module_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    *,
    occupancy_enabled: bool,
    module_name: str,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Reconcile injector/projector input dims across the occupancy flag.

    For each 2D weight where model ``in_features == ckpt in_features + 1`` and
    ``occupancy_enabled``: zero-pad ONE trailing input column (the occupancy
    channel is always appended LAST, after the video/track concat, so the
    legacy columns map 1:1 and the new column starts at exact zero — a strict
    no-op until trained). Any other input-dim mismatch on these modules, and
    ANY mismatch when the flag state does not explain it (occupancy ckpt
    loaded without the flag, or vice versa), raises instead of silently
    skipping.

    Returns (possibly-migrated state, list of migrated key names).
    """
    migrated: list[str] = []
    out: dict[str, torch.Tensor] = {}
    for name, value in module_state.items():
        current = current_state.get(name)
        if current is None or tuple(current.shape) == tuple(value.shape):
            out[name] = value
            continue
        same_but_last = (
            value.ndim == current.ndim
            and value.ndim >= 2
            and tuple(value.shape[:-1]) == tuple(current.shape[:-1])
        )
        if same_but_last and occupancy_enabled and current.shape[-1] == value.shape[-1] + 1:
            padded = value.new_zeros(current.shape)
            padded[..., : value.shape[-1]] = value
            out[name] = padded
            migrated.append(name)
            continue
        if same_but_last and not occupancy_enabled and current.shape[-1] + 1 == value.shape[-1]:
            raise RuntimeError(
                f"{module_name}.{name}: checkpoint carries an occupancy-channel "
                f"input dim {value.shape[-1]} but TRACK4D360_TRACK_OCCUPANCY_CHANNEL "
                "is not set. Set the flag to evaluate this checkpoint."
            )
        raise RuntimeError(
            f"{module_name}.{name}: incompatible shape ckpt{tuple(value.shape)} vs "
            f"model{tuple(current.shape)} (occupancy_enabled={occupancy_enabled}). "
            "Refusing to silently skip an injector/projector weight."
        )
    return out, migrated
