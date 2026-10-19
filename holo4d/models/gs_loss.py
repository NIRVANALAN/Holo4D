"""Gaussian Splatting loss for Track4D-360.

Current GS supervision follows a video-timeline reconstruction objective:

- image MSE
- LPIPS perceptual loss (VGG backbone)
- scale-invariant depth loss
- opacity L1 regularization

The GS loss is computed in ``FlowMatchV2VLoss`` alongside the standard
diffusion loss.
"""

import math
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_LPIPS_VGG = None


def _get_lpips_vgg(device: torch.device):
    global _LPIPS_VGG
    if _LPIPS_VGG is None:
        if os.environ.get("TRACK4D_DEBUG_QUIET_AUX", "0") == "1":
            with open(os.devnull, "w") as devnull, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    _LPIPS_VGG = lpips.LPIPS(net="vgg").eval()
        else:
            _LPIPS_VGG = lpips.LPIPS(net="vgg").eval()
        for p in _LPIPS_VGG.parameters():
            p.requires_grad_(False)
    return _LPIPS_VGG.to(device=device, dtype=torch.float32)


def _absolute_log_depth_l1_foreground(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Foreground-masked absolute log-depth L1.

    ``L = mean_valid |log(pred + eps) - log(gt + eps)|``

    Complements ``_scale_invariant_depth_loss``: the SI term aligns
    relative depth structure modulo a per-frame scale/shift, but lets
    the absolute scale drift. This term ties the absolute scale down
    by directly penalising log-depth offset, masked to foreground
    via the dataset's ``target_depth_valid_mask`` (which excludes sky
    / invalid pixels on MVS-Synth, DL3DV, etc.).

    Most useful for v4 (`gs_dynamic_mode=static_with_track`) on
    track-less batches (MVS-Synth, DL3DV) where the persistent-track
    loss can't anchor absolute scale. With exp-parameterised depth
    (`gs_lyra_depth_param=exp`), this term keeps the predicted depth
    in absolute common_scale-normalized units instead of merely
    shape-correct. It is not raw Lyra-1 metric-scale supervision.
    """
    valid = valid_mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
    valid = valid & (pred_depth > 0) & (gt_depth > 0)
    if not valid.any():
        return pred_depth.new_zeros(())
    return (
        torch.log(pred_depth[valid] + eps) - torch.log(gt_depth[valid] + eps)
    ).abs().mean()


def _scale_invariant_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 0.15,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Scale-invariant log-depth loss (Long-LRM / AdaBins SigLoss formulation).

    ``L = sqrt( Var(g) + alpha * mean(g)^2 )``

    where ``g = log(pred + eps) - log(gt + eps)``, masked to valid pixels.
    The ``alpha * mean(g)^2`` term penalises global scale / shift mismatch on
    top of the pure-variance term, and the ``sqrt`` keeps gradient magnitude
    well-scaled.  With ``alpha=0.15`` this matches Long-LRM
    (``model/depth_anything/.../sigloss.py``) and Lyra's depth supervision.
    """
    valid = valid_mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
    valid = valid & (pred_depth > 0) & (gt_depth > 0)
    if not valid.any():
        return pred_depth.new_zeros(())

    g = torch.log(pred_depth[valid] + eps) - torch.log(gt_depth[valid] + eps)
    Dg = torch.var(g) + alpha * torch.pow(g.mean(), 2)
    return torch.sqrt(Dg)


def _masked_depth_range_stats(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Scalar diagnostics for checking whether predicted depth covers GT."""
    valid = valid_mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
    valid = valid & (pred_depth > 0) & (gt_depth > 0)
    zero = pred_depth.new_zeros(())
    if not valid.any():
        return {
            "gs_depth_valid_frac": zero,
            "gs_pred_depth_min": zero,
            "gs_pred_depth_p05": zero,
            "gs_pred_depth_p50": zero,
            "gs_pred_depth_p95": zero,
            "gs_pred_depth_max": zero,
            "gs_gt_depth_min": zero,
            "gs_gt_depth_p05": zero,
            "gs_gt_depth_p50": zero,
            "gs_gt_depth_p95": zero,
            "gs_gt_depth_max": zero,
            "gs_depth_gt_in_pred_range_frac": zero,
            "gs_depth_pred_covers_gt_range": zero,
            "gs_depth_log_median_ratio": zero,
        }

    pred = pred_depth[valid].detach().float()
    gt = gt_depth[valid].detach().float()
    pred_q = torch.quantile(pred, pred.new_tensor([0.05, 0.50, 0.95]))
    gt_q = torch.quantile(gt, gt.new_tensor([0.05, 0.50, 0.95]))
    pred_min = pred.min()
    pred_max = pred.max()
    gt_min = gt.min()
    gt_max = gt.max()
    gt_in_pred_range = ((gt >= pred_min) & (gt <= pred_max)).float().mean()
    pred_covers_gt = ((pred_min <= gt_min) & (pred_max >= gt_max)).to(pred.dtype)
    return {
        "gs_depth_valid_frac": valid.float().mean().detach(),
        "gs_pred_depth_min": pred_min,
        "gs_pred_depth_p05": pred_q[0],
        "gs_pred_depth_p50": pred_q[1],
        "gs_pred_depth_p95": pred_q[2],
        "gs_pred_depth_max": pred_max,
        "gs_gt_depth_min": gt_min,
        "gs_gt_depth_p05": gt_q[0],
        "gs_gt_depth_p50": gt_q[1],
        "gs_gt_depth_p95": gt_q[2],
        "gs_gt_depth_max": gt_max,
        "gs_depth_gt_in_pred_range_frac": gt_in_pred_range,
        "gs_depth_pred_covers_gt_range": pred_covers_gt,
        "gs_depth_log_median_ratio": (
            torch.log(pred_q[1].clamp_min(1e-6)) -
            torch.log(gt_q[1].clamp_min(1e-6))
        ),
    }


def apply_opacity_pruning(
    gaussian_params: torch.Tensor,
    keep_frac: float,
) -> torch.Tensor:
    """Optionally zero opacity outside the top opacity fraction per frame.

    This is kept only as an explicit legacy-compatibility knob. Default
    training should render the same cloud it predicts.
    """
    keep_frac = float(keep_frac)
    if keep_frac >= 1.0:
        return gaussian_params
    if keep_frac <= 0.0:
        raise ValueError(f"opacity prune keep_frac must be > 0, got {keep_frac}")
    if gaussian_params.dim() != 6:
        raise ValueError(
            f"Expected sequence gaussian params for opacity pruning, got "
            f"{tuple(gaussian_params.shape)}"
        )

    B, T_video, npp, gh, gw, D = gaussian_params.shape
    flat_params = gaussian_params.reshape(B * T_video, npp * gh * gw, D).clone()
    opacity_flat = flat_params[..., 3]
    keep_k = max(1, int(math.ceil(opacity_flat.shape[1] * keep_frac)))
    keep_k = min(keep_k, opacity_flat.shape[1])
    with torch.no_grad():
        thresh = torch.topk(opacity_flat, k=keep_k, dim=1).values[:, -1:]
        keep_mask = opacity_flat >= thresh
    flat_params[..., 3] = flat_params[..., 3] * keep_mask.to(flat_params.dtype)
    return flat_params.reshape(B, T_video, npp, gh, gw, D)


def _coerce_rgb_video_tensor(gt_target_rgb: Any) -> torch.Tensor:
    """Accept canonical tensors or a list of image frames and return (C,T,H,W) / (B,C,T,H,W).

    The static-scene DL3DV loader feeds ``target_video`` as a Python list of PIL
    frames. The diffusion path handles that elsewhere, but the GS branch needs an
    explicit tensor in ``[-1, 1]`` for render supervision.
    """
    if torch.is_tensor(gt_target_rgb):
        return gt_target_rgb

    if isinstance(gt_target_rgb, np.ndarray):
        gt_target_rgb = torch.from_numpy(gt_target_rgb)
        if gt_target_rgb.dtype == torch.uint8:
            gt_target_rgb = gt_target_rgb.float().div_(127.5).sub_(1.0)
        return gt_target_rgb

    if not isinstance(gt_target_rgb, (list, tuple)):
        raise TypeError(
            "gt_target_rgb must be a tensor, ndarray, or a list/tuple of frames; "
            f"got {type(gt_target_rgb)!r}"
        )
    if len(gt_target_rgb) == 0:
        raise ValueError("gt_target_rgb frame list is empty")

    frames = []
    for frame in gt_target_rgb:
        if isinstance(frame, Image.Image):
            arr = np.array(frame, copy=True)
        elif torch.is_tensor(frame):
            frame_t = frame.detach().cpu().float()
            if frame_t.dim() != 3:
                raise ValueError(f"RGB frame tensor must be 3D, got {tuple(frame_t.shape)}")
            if frame_t.shape[0] in (1, 3, 4):
                if frame_t.shape[0] == 4:
                    frame_t = frame_t[:3]
            elif frame_t.shape[-1] in (1, 3, 4):
                frame_t = frame_t.permute(2, 0, 1)
                if frame_t.shape[0] == 4:
                    frame_t = frame_t[:3]
            else:
                raise ValueError(f"Unrecognized RGB frame tensor shape: {tuple(frame_t.shape)}")

            if frame_t.max() > 1.5:
                frame_t = frame_t.div(127.5).sub_(1.0)
            elif frame_t.min() >= 0.0:
                frame_t = frame_t.mul(2.0).sub_(1.0)
            frames.append(frame_t)
            continue
        else:
            arr = np.array(frame, copy=True)

        if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            raise ValueError(f"RGB frame array must be HxWxC with C in (1,3,4), got {arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        frame_t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        frame_t = frame_t.div_(127.5).sub_(1.0)
        frames.append(frame_t)

    return torch.stack(frames, dim=1)


def compute_gs_splatting_loss(
    gs_decoder,
    gs_features_target: torch.Tensor,
    latents_pred_target: torch.Tensor,
    target_cameras: torch.Tensor,
    gt_target_rgb: torch.Tensor,
    gt_target_depth: torch.Tensor | None = None,
    depth_valid_mask: torch.Tensor | None = None,
    image_height: int = 288,
    image_width: int = 512,
    mse_loss_weight: float = 1.0,
    lpips_loss_weight: float = 0.5,
    depth_loss_weight: float = 0.05,
    abs_depth_loss_weight: float = 0.0,
    opacity_loss_weight: float = 0.1,
    detach_predicted_latents: bool = True,
    gaussian_params: torch.Tensor | None = None,
    opacity_prune_keep_frac: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute GS splatting loss.

    Args:
        gs_decoder: GaussianSplattingDecoder instance.
        gs_features_target: (B, feat_dim, T', H', W') GS features for target domain.
        latents_pred_target: (B, z_dim, T', H', W') predicted clean latents.
        detach_predicted_latents: if True (default), detach before ``gs_decoder`` so GS loss
            does not train ``dit.head``; if False, gradients flow through latents.
        target_cameras: (B, T, 11) FlashWorld-format target cameras.
        gt_target_rgb: (B, 3, T, H, W) GT target RGB in [-1, 1].
        gt_target_depth: optional (B, 1, T, H, W) GT target depth.
        depth_valid_mask: optional (B, 1, T, H, W) valid mask for depth.
        image_height, image_width: render resolution.
        mse_loss_weight: weight for image-space MSE render loss.
        lpips_loss_weight: weight for LPIPS(VGG) render loss.
        depth_loss_weight: weight for scale-invariant depth loss.
        opacity_loss_weight: weight for opacity L1 regularization.
        detach_predicted_latents: whether to ``.detach()`` latents before the decoder.
    Returns:
        dict with per-term losses plus the weighted ``gs_loss``.
    """
    gt_target_rgb = _coerce_rgb_video_tensor(gt_target_rgb)

    if target_cameras.dim() == 2:
        target_cameras = target_cameras.unsqueeze(0)
    if gt_target_rgb.dim() == 4:
        gt_target_rgb = gt_target_rgb.unsqueeze(0)
    if gt_target_depth is not None and gt_target_depth.dim() == 4:
        gt_target_depth = gt_target_depth.unsqueeze(0)
    if depth_valid_mask is not None and depth_valid_mask.dim() == 4:
        depth_valid_mask = depth_valid_mask.unsqueeze(0)

    B, C_feat, T_lat, H_lat, W_lat = gs_features_target.shape
    feats = gs_features_target
    z = (
        latents_pred_target.detach()
        if detach_predicted_latents
        else latents_pred_target
    )
    target_device = feats.device

    target_cameras = target_cameras.to(device=target_device)
    gt_target_rgb = gt_target_rgb.to(device=target_device)
    if gt_target_depth is not None:
        gt_target_depth = gt_target_depth.to(device=target_device)
    if depth_valid_mask is not None:
        depth_valid_mask = depth_valid_mask.to(device=target_device)

    if target_cameras.dim() != 3:
        raise ValueError(f"target_cameras must be (B, T, 11), got {tuple(target_cameras.shape)}")
    T_video = target_cameras.shape[1]

    # GS decoder forward over the latent video sequence. The decoder runs the
    # Wan VAE body in per-frame autoregressive mode with feat_cache, so
    # ``T_lat -> T_video = (T_lat - 1) * 4 + 1`` temporal upsampling fires and
    # the rendered timeline matches ``target_cameras.shape[1]`` natively.
    # v4 path supplies pre-computed gaussian_params (decoder forward already
    # ran upstream to also produce track_delta_xyz); skip the redundant pass.
    if gaussian_params is None:
        gaussian_params = gs_decoder(feats, z, target_cameras)
    # gaussian_params: (B, T_video, npp, H_full, W_full, D)

    if gaussian_params.dim() != 6:
        raise ValueError(f"Expected sequence gaussian params, got {tuple(gaussian_params.shape)}")

    gaussian_params = apply_opacity_pruning(
        gaussian_params,
        keep_frac=opacity_prune_keep_frac,
    )

    rendered_rgb, rendered_depth = gs_decoder.render(
        gaussian_params,
        target_cameras,
        height=image_height,
        width=image_width,
        bg_mode="white",
    )
    # rendered_rgb: (B, T, 3, H, W) in [-1, 1]
    # rendered_depth: (B, T, 1, H, W)

    if gt_target_rgb.shape[2] != T_video:
        raise ValueError(
            f"GT RGB timeline {gt_target_rgb.shape[2]} does not match rendered timeline {T_video}"
        )

    gt_rgb_seq = gt_target_rgb.permute(0, 2, 1, 3, 4)
    gt_rgb_flat = gt_rgb_seq.reshape(B * T_video, 3, image_height, image_width)
    rendered_rgb_flat = rendered_rgb.reshape(B * T_video, 3, image_height, image_width)
    # SceneTok uses LPIPS(VGG) with normalize=True on [0,1] images.
    # Our renderer/targets are in [-1,1], so map them before image-space losses.
    rendered_rgb_01 = rendered_rgb_flat.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
    gt_rgb_flat_01 = gt_rgb_flat.float().add(1.0).mul(0.5).clamp(0.0, 1.0)

    gs_mse_loss = F.mse_loss(rendered_rgb_01, gt_rgb_flat_01)
    gs_psnr = -10.0 * torch.log10(gs_mse_loss.clamp_min(1e-10))

    if lpips_loss_weight > 0:
        gs_lpips_loss = _get_lpips_vgg(rendered_rgb.device)(
            rendered_rgb_01,
            gt_rgb_flat_01,
            normalize=True,
        ).mean()
    else:
        gs_lpips_loss = rendered_rgb.new_zeros(())

    # Depth loss (optional). Foreground supervision is enforced via
    # `depth_valid_mask` (= dataset's `target_depth_valid_mask`), which
    # excludes sky / invalid pixels on every dataset that emits it
    # (MVS-Synth, DL3DV, RE10K, TartanAir, Kubric). When the dataset
    # does not provide the mask we fall back to (gt_depth > 1e-6) which
    # also acts as a foreground filter (sky often serializes to 0).
    gs_depth_loss = rendered_rgb_flat.new_zeros(())
    gs_abs_depth_loss = rendered_rgb_flat.new_zeros(())
    gs_depth_stats = {}
    if (depth_loss_weight > 0 or abs_depth_loss_weight > 0) and gt_target_depth is not None:
        if gt_target_depth.shape[2] != T_video:
            raise ValueError(
                f"GT depth timeline {gt_target_depth.shape[2]} does not match rendered timeline {T_video}"
            )
        gt_depth_flat = gt_target_depth.permute(0, 2, 1, 3, 4).reshape(B * T_video, 1, image_height, image_width)
        rendered_depth_flat = rendered_depth.reshape(B * T_video, 1, image_height, image_width)

        if depth_valid_mask is not None:
            if depth_valid_mask.shape[2] != T_video:
                raise ValueError(
                    f"Depth-valid timeline {depth_valid_mask.shape[2]} does not match rendered timeline {T_video}"
                )
            mask_flat = depth_valid_mask.permute(0, 2, 1, 3, 4).reshape(B * T_video, 1, image_height, image_width)
            valid = mask_flat > 0.5
        else:
            valid = gt_depth_flat > 1e-6
        gs_depth_stats = _masked_depth_range_stats(
            pred_depth=rendered_depth_flat.squeeze(1),
            gt_depth=gt_depth_flat.squeeze(1),
            valid_mask=valid.squeeze(1),
        )
        if depth_loss_weight > 0:
            gs_depth_loss = _scale_invariant_depth_loss(
                pred_depth=rendered_depth_flat.squeeze(1),
                gt_depth=gt_depth_flat.squeeze(1),
                valid_mask=valid.squeeze(1),
            )
        if abs_depth_loss_weight > 0:
            # Absolute log-depth L1 on foreground only — anchors the
            # common_scale-normalized absolute scale of exp-parameterised
            # depth predictions on batches without 3D-track supervision
            # (MVS-Synth, DL3DV).
            gs_abs_depth_loss = _absolute_log_depth_l1_foreground(
                pred_depth=rendered_depth_flat.squeeze(1),
                gt_depth=gt_depth_flat.squeeze(1),
                valid_mask=valid.squeeze(1),
            )

    gs_opacity_loss = gaussian_params[..., 3].abs().mean()

    gs_loss = (
        mse_loss_weight * gs_mse_loss
        + lpips_loss_weight * gs_lpips_loss
        + depth_loss_weight * gs_depth_loss
        + abs_depth_loss_weight * gs_abs_depth_loss
        + opacity_loss_weight * gs_opacity_loss
    )

    return {
        "gs_loss": gs_loss,
        "gs_rgb_loss": gs_mse_loss,
        "gs_mse_loss": gs_mse_loss,
        "gs_psnr": gs_psnr,
        "gs_abs_depth_loss": gs_abs_depth_loss,
        "gs_lpips_loss": gs_lpips_loss,
        "gs_depth_loss": gs_depth_loss,
        "gs_opacity_loss": gs_opacity_loss,
        **gs_depth_stats,
    }
