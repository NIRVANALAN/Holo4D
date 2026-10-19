"""Deterministic train-time variable trajectory-count policy.

The active count may vary per optimizer/micro step while tensors are padded to
one fixed maximum width.  Padding keeps the FSDP graph/shape contract stable;
invalid rows contribute no track features or occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class VariableIdentityCount:
    active_count: int
    padded_count: int
    minimum: int
    maximum: int
    enabled: bool


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw or default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _mix_seed(seed: int) -> int:
    """SplitMix64 finalizer: deterministic without consuming Torch/Python RNG."""
    mask = (1 << 64) - 1
    value = (int(seed) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def resolve_variable_identity_count(
    padded_count: int,
    *,
    sampling_seed: Optional[int],
    training: bool,
) -> VariableIdentityCount:
    """Resolve one uniformly sampled active N and a shape-stable padded N.

    ``HOLO4D_TRACK_IDENTITY_RANDOM_N_MIN/MAX`` are train-only and default
    to disabled.  A dedicated step/rank/dataset seed is mandatory so sampling
    never advances the diffusion timestep/noise RNG stream.
    """
    padded_count = int(padded_count)
    minimum = _env_int("HOLO4D_TRACK_IDENTITY_RANDOM_N_MIN", 0)
    maximum = _env_int("HOLO4D_TRACK_IDENTITY_RANDOM_N_MAX", 0)
    configured = minimum != 0 or maximum != 0
    if not configured or not training:
        return VariableIdentityCount(
            active_count=padded_count,
            padded_count=padded_count,
            minimum=minimum,
            maximum=maximum,
            enabled=False,
        )
    if padded_count <= 0:
        raise ValueError(
            "Variable identity count requires HOLO4D_TRACK_IDENTITY_FIXED_N > 0"
        )
    if minimum <= 0 or maximum <= 0:
        raise ValueError(
            "HOLO4D_TRACK_IDENTITY_RANDOM_N_MIN and MAX must both be > 0"
        )
    if minimum > maximum:
        raise ValueError(
            "HOLO4D_TRACK_IDENTITY_RANDOM_N_MIN must be <= MAX, got "
            f"{minimum} > {maximum}"
        )
    if maximum > padded_count:
        raise ValueError(
            "HOLO4D_TRACK_IDENTITY_RANDOM_N_MAX cannot exceed the fixed padded "
            f"width: {maximum} > {padded_count}"
        )
    if sampling_seed is None:
        raise ValueError(
            "Variable identity count requires track_identity_sampling_seed. Set "
            "HOLO4D_DEDICATED_IDENTITY_SAMPLING_SEED=1 in the training config."
        )
    span = maximum - minimum + 1
    active_count = minimum + (_mix_seed(int(sampling_seed)) % span)
    return VariableIdentityCount(
        active_count=int(active_count),
        padded_count=padded_count,
        minimum=minimum,
        maximum=maximum,
        enabled=True,
    )
