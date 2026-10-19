"""Torch-only primitives for Wan 2.2 TI2V-native I2V conditioning."""

from __future__ import annotations

import torch


def validate_ti2v_native_config(
    *,
    enabled: bool,
    legacy_i2v_enabled: bool,
    concat_mode: str,
) -> None:
    """Fail loudly on combinations with ambiguous conditioning semantics."""
    if not bool(enabled):
        return
    if bool(legacy_i2v_enabled):
        raise ValueError(
            "--ti2v_native_i2v and --i2v_target_condition are mutually exclusive"
        )
    if str(concat_mode) != "width":
        raise ValueError("--ti2v_native_i2v requires --v2v_concat_mode width")


def target_domain_width(full_latents: torch.Tensor, *, num_domains: int) -> int:
    """Return one width-concatenated domain width after validating the layout."""
    if full_latents.ndim != 5:
        raise ValueError(
            f"full_latents must be BCTHW, got shape={tuple(full_latents.shape)}"
        )
    num_domains = int(num_domains)
    if num_domains < 2:
        raise ValueError(f"num_domains must be >= 2, got {num_domains}")
    full_width = int(full_latents.shape[-1])
    if full_width % num_domains != 0:
        raise ValueError(
            f"latent width {full_width} is not divisible by num_domains={num_domains}"
        )
    return full_width // num_domains


def extract_target_first_frame_latents(
    full_latents: torch.Tensor,
    *,
    num_domains: int,
    target_domain: int = 1,
) -> torch.Tensor:
    """Extract clean target-domain latent frame 0 from a width-concat tensor."""
    domain_width = target_domain_width(full_latents, num_domains=num_domains)
    target_domain = int(target_domain)
    if not 0 <= target_domain < int(num_domains):
        raise ValueError(
            f"target_domain must be in [0, {int(num_domains) - 1}], got {target_domain}"
        )
    start = target_domain * domain_width
    return full_latents[:, :, 0:1, :, start : start + domain_width]


def inject_target_first_frame_latents(
    latents: torch.Tensor,
    first_frame_latents: torch.Tensor,
    *,
    num_domains: int,
    target_domain: int = 1,
) -> torch.Tensor:
    """Out-of-place target-domain frame-0 overwrite used by train and inference."""
    domain_width = target_domain_width(latents, num_domains=num_domains)
    expected = (
        int(latents.shape[0]),
        int(latents.shape[1]),
        1,
        int(latents.shape[3]),
        domain_width,
    )
    if tuple(first_frame_latents.shape) != expected:
        raise ValueError(
            "first_frame_latents must contain one target-domain latent frame; "
            f"expected {expected}, got {tuple(first_frame_latents.shape)}"
        )
    target_domain = int(target_domain)
    if not 0 <= target_domain < int(num_domains):
        raise ValueError(
            f"target_domain must be in [0, {int(num_domains) - 1}], got {target_domain}"
        )
    start = target_domain * domain_width
    conditioned = latents.clone()
    conditioned[:, :, 0:1, :, start : start + domain_width] = first_frame_latents.to(
        device=latents.device, dtype=latents.dtype
    )
    return conditioned


def apply_target_frame0_loss_mask(
    loss_weights: torch.Tensor,
    *,
    num_domains: int,
) -> torch.Tensor:
    """Zero loss for target-RGB latent frame 0 in a target-only prediction slice.

    ``loss_weights`` covers domains 1..N because the source domain has already
    been sliced away. Target RGB is therefore the left-most remaining domain.
    Other target modalities, when present, retain frame-0 supervision.
    """
    if loss_weights.ndim != 5:
        raise ValueError(
            f"loss_weights must be BCTHW, got shape={tuple(loss_weights.shape)}"
        )
    num_domains = int(num_domains)
    if num_domains < 2:
        raise ValueError(f"num_domains must be >= 2, got {num_domains}")
    target_domains = num_domains - 1
    target_width = int(loss_weights.shape[-1])
    if target_width % target_domains != 0:
        raise ValueError(
            f"target width {target_width} is not divisible by {target_domains} domains"
        )
    rgb_width = target_width // target_domains
    masked = loss_weights.clone()
    masked[:, :, 0:1, :, :rgb_width] = 0
    return masked


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """MSE normalized by active element weights."""
    if pred.shape != target.shape or pred.shape != weights.shape:
        raise ValueError(
            "pred, target, and weights must share a shape; got "
            f"{tuple(pred.shape)}, {tuple(target.shape)}, {tuple(weights.shape)}"
        )
    sq_error = (pred.float() - target.float()) ** 2
    weights = weights.to(device=sq_error.device, dtype=sq_error.dtype)
    return sq_error.mul(weights).sum() / weights.sum().clamp_min(1.0)
