"""Future-prediction (temporal extrapolation) primitives for Track4D-360.

Torch-only helpers, importable without the heavy diffusion/transformers stack so
they can be unit-tested on CPU. Used by the V2V loss (and later the pipeline) to
mark the trailing latent frames as UNOBSERVED and replace their source domain
with a learned null, while the target is supervised on all frames.

All entry points are no-ops unless future prediction is explicitly enabled, so
the default V2V NVS path is byte-identical.
"""

from __future__ import annotations

import torch


def _as_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.flatten()[0].item()) if value.numel() else False
    return bool(value)


def resolve_observed_mask(inputs, *, latent_t: int, device, batch: int):
    """Resolve the per-(batch, latent-frame) observed mask for future prediction.

    Returns a ``(batch, latent_t)`` bool tensor (True=observed) or ``None`` when
    future prediction is off. When on but no explicit ``observed_mask`` is
    supplied, the trailing ``future_num_latent_frames`` latent frames are marked
    unobserved.
    """
    if not _as_bool(inputs.get("future_prediction", False)):
        return None
    mask = inputs.get("observed_mask")
    if mask is not None:
        mask = mask.to(device=device).bool()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch, -1)
        if int(mask.shape[1]) != int(latent_t):
            raise ValueError(
                f"observed_mask latent length {int(mask.shape[1])} != latent_t {int(latent_t)}"
            )
        return mask
    n_future = int(inputs.get("future_num_latent_frames", 0))
    if n_future <= 0 or n_future >= int(latent_t):
        raise ValueError(
            f"future_prediction requires 0 < future_num_latent_frames < latent_t; "
            f"got future_num_latent_frames={n_future}, latent_t={latent_t}"
        )
    mask = torch.ones(batch, int(latent_t), dtype=torch.bool, device=device)
    mask[:, int(latent_t) - n_future:] = False
    return mask


def apply_future_source_null(noisy_latents, split: int, observed_mask, future_source_null=None):
    """Overwrite the source (left-of-width) domain of FUTURE latent frames with
    the learned ``future_source_null`` (or zeros if None). Width-concat only.

    No-op when ``observed_mask`` is None (future prediction off) → byte-identical
    default path. Observed frames and the target (right-of-width) domain are
    never touched.
    """
    if observed_mask is None:
        return noisy_latents
    B, C, T, H, W = noisy_latents.shape
    future = (~observed_mask.bool()).view(B, 1, T, 1, 1).to(noisy_latents.dtype)
    if future_source_null is not None:
        null = future_source_null.to(device=noisy_latents.device, dtype=noisy_latents.dtype).view(1, C, 1, 1, 1)
    else:
        null = torch.zeros(1, C, 1, 1, 1, device=noisy_latents.device, dtype=noisy_latents.dtype)
    src = noisy_latents[..., :split]
    noisy_latents[..., :split] = src * (1.0 - future) + null * future
    return noisy_latents


def future_weighted_mse(pred, gt, observed_mask, future_loss_weight):
    """Per-latent-frame weighted MSE: future (predicted) frames are weighted by
    ``future_loss_weight``, observed frames keep weight 1.

    ``pred``/``gt`` are ``(B, C, T, H, W)``; ``observed_mask`` is ``(B, T)`` bool.
    The weight tensor is built at FULL pred shape so the normalizer
    (``w.sum()``) matches the weighted numerator — a per-frame ``(B,1,T,1,1)``
    weight would undercount the denominator by C*H*W. Returns a scalar that
    equals ``mse(pred, gt)`` when the mask is None or the weight is 1.0.
    """
    sq = (pred.float() - gt.float()) ** 2
    if observed_mask is None or float(future_loss_weight) == 1.0:
        return sq.mean()
    B, C, T, H, W = sq.shape
    w = torch.ones_like(sq)
    future = (~observed_mask.bool()).view(B, 1, T, 1, 1).expand_as(w)
    w = w.masked_fill(future, float(future_loss_weight))
    return sq.mul(w).sum() / w.sum().clamp_min(1.0)


def future_pixel_mask(observed_mask, num_video_frames: int):
    """Map a latent-frame observed mask to a per-VIDEO-frame future mask.

    Wan VAE temporal compression is causal: video frame 0 → latent 0; video
    frame p≥1 → latent (p-1)//4 + 1. Returns ``(B, num_video_frames)`` bool where
    True == future/unobserved, for nulling future-frame track visibility.
    """
    if observed_mask is None:
        return None
    device = observed_mask.device
    p = torch.arange(int(num_video_frames), device=device)
    latent_of_pixel = torch.where(p == 0, torch.zeros_like(p), (p - 1) // 4 + 1)
    obs = observed_mask.bool()  # (B, latent_t)
    B = obs.shape[0]
    # Gather observed flag for each video frame's latent index, then invert.
    idx = latent_of_pixel.clamp_max(obs.shape[1] - 1).view(1, -1).expand(B, -1)
    pixel_observed = torch.gather(obs, 1, idx)
    return ~pixel_observed
