"""
Dynamic Gaussian Splatting loss for Track4D-360.

Companion to :mod:`holo4d.models.gs_loss` (the static GS loss). This
module implements the **canonical first-frame Gaussians + per-frame deltas**
loss path used when ``--gs_dynamic_mode deltas`` is set on the training
script. It is intentionally written as a sibling to ``gs_loss.py`` and does
not import from it; the static loss path stays byte-identical.

Loss formula
------------

::

    total_gs_loss = w_rgb     * Σ_t L1(render_rgb_t,  gt_target_rgb[t])
                  + w_pos     * Σ_t masked_L1(render_pos_t,
                                              target_track_video[t],
                                              target_track_valid_mask[t])
                  + w_depth   * Σ_t masked_L1(render_depth_t,
                                              target_depth_video[t],
                                              target_depth_valid_mask[t])
                  + w_opa_tv  * Σ_t L1(opacity_t, opacity_{t-1})
                  + w_xyz_tv  * Σ_t L1(delta_xyz_t, delta_xyz_{t-1})

The position term is the SoM-style upgrade: it uses gsplat to render
per-Gaussian xyz as a 3-channel feature image, indexed by current-frame
target-view pixels — exactly the indexing of ``target_track_video[t]``.

Full-timeline supervision (post 2026-04-09 refactor)
----------------------------------------------------

Historically the dynamic loss downsampled every GT stream to ``T_lat``
and only ran the render loop for ``T_lat`` steps. That meant a 49-frame
clip (``T_lat=13``) got supervision at only 13 frames — contradicting
the "static + dynamic paths both render + supervise on the full input
timeline" contract documented in ``README.md`` and
``doc/track4d-360/gs_output_integration.md``.

The fix is to mirror the static path exactly:

- ``target_cameras`` already arrive at ``T_video`` rate from the dataset.
- ``gt_target_rgb``, ``target_track_video``, ``target_depth_video`` all
  arrive at ``T_video`` rate as well (the loss used to check with
  ``_subsample_time``; now we assert instead of resampling).
- The dynamic decoder predicts up to ``max_t_video - 1`` deltas in one
  shot from frame-0 features, sliced down to the actual ``T_video - 1``.
- The render loop iterates ``t ∈ [0, T_video)`` so every target-video
  frame gets its own rendered RGB/position/depth and its own L1.

The dynamic path is still cheaper than the static path because
``DynamicGaussianSplattingDecoder.forward`` only decodes a single latent
frame (``T_lat=1`` slice) rather than the full autoregressive
``T_lat → T_video`` loop — Wan-DiT's spatio-temporal attention lets
frame-0 features stand in for the whole sequence on the dynamic side.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gs_camera_utils import quaternion_to_matrix
from .gs_decoder_dynamic import DynamicGaussianSplattingDecoder
from .gs_renderer import gaussian_render  # read-only — same call the static loss makes
from .gs_renderer_features import gaussian_render_features


# ---------------------------------------------------------------------------
# Helpers (kept local — do NOT depend on gs_loss.py)
# ---------------------------------------------------------------------------


def _flashworld_to_c2w_intr(
    cameras_flat: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (BT, 11) FlashWorld cameras → (BT, 1, 4, 4) c2w + (BT, 1, 4) intrinsics.

    Mirrors the conversion in :func:`gs_loss.compute_gs_splatting_loss`,
    but kept local so this module does not depend on the static loss file.

    The returned ``c2ws`` is in the FlashWorld/OpenGL convention (camera looks
    along −Z). For gsplat rendering this is the expected convention. For
    standard-CV pixel projection, flip Y/Z columns of the rotation — see
    :func:`_flashworld_c2w_to_opencv`.
    """
    BT = cameras_flat.shape[0]
    cameras_flat = cameras_flat.to(torch.float32)
    c2ws = torch.eye(4, device=cameras_flat.device, dtype=torch.float32)[None, None].repeat(BT, 1, 1, 1)
    c2ws[:, 0, :3, :3] = quaternion_to_matrix(cameras_flat[:, :4])
    c2ws[:, 0, :3, 3] = cameras_flat[:, 4:7]

    fx_px = cameras_flat[:, 7:8] * image_width
    fy_px = cameras_flat[:, 8:9] * image_height
    cx_px = cameras_flat[:, 9:10] * image_width
    cy_px = cameras_flat[:, 10:11] * image_height
    intrinsics = torch.cat([fx_px, fy_px, cx_px, cy_px], dim=-1).unsqueeze(1)  # (BT, 1, 4)
    return c2ws, intrinsics


def masked_l1_loss_quantile(
    diff: torch.Tensor,
    mask: torch.Tensor,
    quantile: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """SoM-style masked L1 with quantile outlier suppression.

    Port of ``shape-of-motion/flow3d/loss_utils.py:50`` adapted to our
    broadcast convention: ``diff`` is the pre-computed ``|pred - gt|``
    tensor (any shape), ``mask`` is a broadcastable float/bool selector of
    same shape, ``quantile`` in ``(0, 1]`` clips values above its kthvalue
    to zero so outliers do not dominate gradients.

    Gradient-safe: the threshold is computed under ``no_grad`` and only
    used to build a detached selector; gradients flow normally through
    the kept (in-quantile) entries.
    """
    mask_f = mask.to(diff.dtype)
    if quantile < 1.0:
        with torch.no_grad():
            vals = (diff * mask_f).reshape(-1)
            pos = mask_f.reshape(-1) > 0
            kept = vals[pos]
            if kept.numel() > 1:
                k = max(1, int(kept.numel() * quantile))
                k = min(k, kept.numel())
                threshold = kept.kthvalue(k).values
                mask_f = mask_f * (diff <= threshold).to(diff.dtype)
    weighted = diff * mask_f
    if normalize:
        denom = mask_f.sum().clamp_min(1.0)
        return weighted.sum() / denom
    return weighted.mean()


def _persistent_tracks_to_grid(
    target_tracks_world: torch.Tensor,
    target_tracks_world_valid_mask: Optional[torch.Tensor],
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape persistent ``(B, N, T, 3)`` tracks to ``(B, T, H, W, 3)`` grid.

    Assumes row-major ``y·W + x`` indexing (the Kubric `_compute_tracks`
    convention at ``holo4d_runtime/data/stream3r/dust3r/datasets_cut3r/
    track4d360_mixed_nvs.py:720``). Raises if ``N != H·W``.
    """
    if target_tracks_world.dim() == 3:
        target_tracks_world = target_tracks_world.unsqueeze(0)
    B, N, T, _ = target_tracks_world.shape
    if N != H * W:
        raise ValueError(
            f"target_tracks_world has N={N} tracks but grid is H*W={H * W}. "
            f"Objective A requires dense row-major persistent tracks."
        )
    tw_grid = target_tracks_world.reshape(B, H, W, T, 3).permute(0, 3, 1, 2, 4).contiguous()
    if target_tracks_world_valid_mask is None:
        tw_valid = (tw_grid.abs().sum(dim=-1) > 1e-8)
    else:
        if target_tracks_world_valid_mask.dim() == 2:
            target_tracks_world_valid_mask = target_tracks_world_valid_mask.unsqueeze(0)
        tw_valid = (
            target_tracks_world_valid_mask.reshape(B, H, W, T).permute(0, 3, 1, 2).contiguous()
            > 0.5
        )
    return tw_grid, tw_valid


def _build_per_frame_cameras_for_projection(
    target_cameras: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, T, 11)`` FlashWorld → ``(B, T, 4, 4)`` OpenCV c2w + ``(B, T, 4)`` intr_pixel."""
    B, T, _ = target_cameras.shape
    tc_flat = target_cameras.reshape(B * T, 11)
    c2w_gl, intr = _flashworld_to_c2w_intr(tc_flat, image_height, image_width)
    c2w_cv = _flashworld_c2w_to_opencv(c2w_gl.squeeze(1)).reshape(B, T, 4, 4)
    intr_bt = intr.squeeze(1).reshape(B, T, 4)
    return c2w_cv, intr_bt


def _compute_assoc_persistent_track_loss(
    pred_full: torch.Tensor,
    tw_grid: torch.Tensor,
    tw_valid: torch.Tensor,
    target_cameras: torch.Tensor,
    image_height: int,
    image_width: int,
    quantile: float,
) -> torch.Tensor:
    """Objective A — associated trajectory vs persistent GT, in 2D pixel units.

    pred_full: (B, T, H, W, 3) — associated Gaussian trajectory.
    tw_grid:   (B, T, H, W, 3) — persistent `target_tracks_world` reshaped.
    tw_valid:  (B, T, H, W).
    Returns scalar L1 pixel distance normalized by max(H, W), quantile-clipped.
    """
    c2w_cv, intr = _build_per_frame_cameras_for_projection(target_cameras, image_height, image_width)
    uv_pred, valid_pred = _project_world_to_image_torch(pred_full, c2w_cv, intr)
    uv_gt,   valid_gt   = _project_world_to_image_torch(tw_grid,  c2w_cv, intr)
    mask = tw_valid & valid_pred & valid_gt          # (B, T, H, W) bool
    diff = (uv_pred - uv_gt).abs().sum(dim=-1)        # (B, T, H, W) L1 over (u, v)
    pixel_denom = float(max(image_height, image_width))
    return masked_l1_loss_quantile(diff, mask, quantile=quantile, normalize=True) / pixel_denom


def _compute_accel_loss(
    pred_full: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Second-derivative L1 on the associated trajectory (world units).

    L = mean_valid |2·x[t] − x[t-1] − x[t+1]|  for t ∈ [1, T-2].

    Zero trajectory acceleration for static/constant-velocity motion.
    Masked on pixels valid at t-1, t, t+1.
    """
    B, T, H, W, _ = pred_full.shape
    if T < 3:
        return pred_full.new_tensor(0.0)
    accel = 2 * pred_full[:, 1:-1] - pred_full[:, :-2] - pred_full[:, 2:]    # (B, T-2, H, W, 3)
    mask = (valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]).to(accel.dtype)
    diff = accel.abs().sum(dim=-1)                                            # (B, T-2, H, W)
    denom = mask.sum().clamp_min(1.0)
    return (diff * mask).sum() / denom


@torch.no_grad()
def _compute_assoc_persistent_track_metrics(
    pred_full: torch.Tensor,
    tw_grid: torch.Tensor,
    tw_valid: torch.Tensor,
    target_cameras: torch.Tensor,
    image_height: int,
    image_width: int,
) -> dict[str, torch.Tensor]:
    """Persistent-track diagnostics (the correct-target counterparts of
    `_compute_track_metrics`). Everything detached; never contributes to loss.
    """
    c2w_cv, intr = _build_per_frame_cameras_for_projection(target_cameras, image_height, image_width)
    uv_pred, valid_pred = _project_world_to_image_torch(pred_full, c2w_cv, intr)
    uv_gt,   valid_gt   = _project_world_to_image_torch(tw_grid,  c2w_cv, intr)
    px_mask = tw_valid & valid_pred & valid_gt
    diff_2d = (uv_pred - uv_gt).abs().sum(dim=-1)
    vals = diff_2d[px_mask]
    out: dict[str, torch.Tensor] = {}

    if vals.numel() > 0:
        out["gs_assoc_persistent_2d_px_l1_mean"] = vals.mean()
        p50, p90, p95 = _percentiles(vals, (0.50, 0.90, 0.95))
        out["gs_assoc_persistent_2d_px_l1_p50"] = p50
        out["gs_assoc_persistent_2d_px_l1_p90"] = p90
        out["gs_assoc_persistent_2d_px_l1_p95"] = p95
    else:
        for k in (
            "gs_assoc_persistent_2d_px_l1_mean",
            "gs_assoc_persistent_2d_px_l1_p50",
            "gs_assoc_persistent_2d_px_l1_p90",
            "gs_assoc_persistent_2d_px_l1_p95",
        ):
            out[k] = diff_2d.new_tensor(0.0)

    B, T = uv_pred.shape[:2]
    if T > 1:
        vel_pred = uv_pred[:, 1:] - uv_pred[:, :-1]
        vel_gt   = uv_gt[:, 1:]   - uv_gt[:, :-1]
        vel_mask = px_mask[:, 1:] & px_mask[:, :-1]
        vel_diff = (vel_pred - vel_gt).abs().sum(dim=-1)
        vel_vals = vel_diff[vel_mask]
        out["gs_assoc_persistent_2d_px_vel_l1"] = (
            vel_vals.mean() if vel_vals.numel() > 0 else vel_diff.new_tensor(0.0)
        )
    else:
        out["gs_assoc_persistent_2d_px_vel_l1"] = uv_pred.new_tensor(0.0)

    # 3D scene-scale normalized L1 (VGGT-style mean Euclidean norm over valid).
    gt_norm = tw_grid.norm(dim=-1)
    valid_f = tw_valid.to(torch.float32)
    scene_scale = (gt_norm * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    scene_scale = scene_scale.clamp_min(1e-6)
    diff_3d = (pred_full - tw_grid).norm(dim=-1)
    persistent_3d_l1 = (diff_3d * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    out["gs_persistent_scene_scale"] = scene_scale
    out["gs_persistent_3d_norm_l1"] = persistent_3d_l1 / scene_scale

    return out


def _flashworld_c2w_to_opencv(c2w_gl: torch.Tensor) -> torch.Tensor:
    """Flip Y/Z columns of rotation to convert FlashWorld/OpenGL c2w → OpenCV c2w.

    This is the convention used in :func:`examples/wan2.1_fun/eval_gs_head_from_cache.py`
    for 2D pixel projection. Accepts any batched shape ending in ``(4, 4)``.
    """
    c2w_cv = c2w_gl.clone()
    c2w_cv[..., :3, 1:3] = -c2w_cv[..., :3, 1:3]
    return c2w_cv


def _project_world_to_image_torch(
    pts_world: torch.Tensor,
    c2w_cv: torch.Tensor,
    intrinsics_4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched world→pixel projection under OpenCV convention.

    Args:
        pts_world: ``(B, T, H, W, 3)`` 3D points in world coordinates.
        c2w_cv:    ``(B, T, 4, 4)`` OpenCV-convention camera-to-world.
        intrinsics_4: ``(B, T, 4)`` ``[fx, fy, cx, cy]`` in pixels.

    Returns:
        uv: ``(B, T, H, W, 2)`` pixel coordinates. Entries for invalid
            (behind-camera) points are clamped-safe but should be masked via
            the returned ``valid`` tensor.
        valid: ``(B, T, H, W)`` True where projection is meaningful (z > 1e-6).
    """
    w2c = torch.linalg.inv(c2w_cv.to(torch.float32))                # (B, T, 4, 4)
    ones = torch.ones_like(pts_world[..., :1])
    pts_h = torch.cat([pts_world, ones], dim=-1)                    # (B, T, H, W, 4)
    # w2c[b,t] @ pts_h[b,t,h,w] → (B, T, H, W, 4)
    pts_cam = torch.einsum("btij,bthwj->bthwi", w2c, pts_h)[..., :3]
    z = pts_cam[..., 2]
    valid = z > 1e-6
    z_safe = z.clamp_min(1e-6)
    fx = intrinsics_4[..., 0][..., None, None]
    fy = intrinsics_4[..., 1][..., None, None]
    cx = intrinsics_4[..., 2][..., None, None]
    cy = intrinsics_4[..., 3][..., None, None]
    u = fx * pts_cam[..., 0] / z_safe + cx
    v = fy * pts_cam[..., 1] / z_safe + cy
    return torch.stack([u, v], dim=-1), valid


def _extract_associated_trajectory(
    canonical_params: torch.Tensor,
    delta_params: torch.Tensor,
    assoc: torch.Tensor,
    predict_rotation: bool,
    disable_delta_opacity: bool,
    t_video: int,
) -> torch.Tensor:
    """Return ``(B, T, H, W, 3)`` — the per-frame xyz of the associated Gaussian.

    Shared helper used by both the dense track loss and the Phase-1 metrics.
    Using a single extractor guarantees training and metric numbers refer to
    the same quantity the eval "GS 2D Tracks" panel reads (§15 of
    CLAUDE-gs-head.md).
    """
    B, npp, H, W, D = canonical_params.shape
    assoc_exp = assoc.unsqueeze(1).unsqueeze(-1).expand(B, 1, H, W, 3)
    canonical_xyz = canonical_params[..., 0:3]
    canonical_assoc = torch.gather(canonical_xyz, dim=1, index=assoc_exp).squeeze(1)
    traj = [canonical_assoc]
    for t_idx in range(1, t_video):
        full_t, _, _ = _compose_frame_gaussians(
            canonical_params, delta_params, predict_rotation, disable_delta_opacity, t_idx,
        )
        xyz_t = full_t[..., 0:3]
        xyz_assoc = torch.gather(xyz_t, dim=1, index=assoc_exp).squeeze(1)
        traj.append(xyz_assoc)
    return torch.stack(traj, dim=1)


def _percentiles(values: torch.Tensor, qs: tuple[float, ...]) -> list[torch.Tensor]:
    """Return percentile scalars for a 1-D tensor. Empty input yields zeros."""
    if values.numel() == 0:
        return [values.new_tensor(0.0) for _ in qs]
    # torch.quantile needs contiguous float input.
    v = values.detach().to(torch.float32).reshape(-1)
    return [torch.quantile(v, q) for q in qs]


@torch.no_grad()
def _compute_track_metrics(
    canonical_params: torch.Tensor,
    delta_params: torch.Tensor,
    gt_track_ds: torch.Tensor,
    valid_ds: torch.Tensor,
    assoc: torch.Tensor,
    target_cameras: torch.Tensor,
    image_height: int,
    image_width: int,
    predict_rotation: bool,
    disable_delta_opacity: bool,
    final_pos_loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Phase-1 diagnostics: scene-scale 3D, 2D pixel error tail, velocity, assoc-vs-contribution.

    All metrics are detached — they never reach the optimizer. Shapes:
      gt_track_ds: ``(B, T, H, W, 3)``   (already resized to GS-head grid)
      valid_ds:    ``(B, T, H, W)`` bool
      assoc:       ``(B, H, W)``   long in [0, npp)
    """
    B, T, H, W, _ = gt_track_ds.shape
    device = canonical_params.device
    metrics: dict[str, torch.Tensor] = {}

    # ---- (2) Scene-scale normalization (VGGT-style: mean Euclidean norm) ----
    gt_norm = gt_track_ds.norm(dim=-1)                              # (B, T, H, W)
    valid_f = valid_ds.to(torch.float32)
    denom = valid_f.sum().clamp_min(1.0)
    scene_scale = (gt_norm * valid_f).sum() / denom
    scene_scale = scene_scale.clamp_min(1e-6)
    metrics["gs_track_scene_scale"] = scene_scale
    metrics["gs_track_3d_norm_l1"] = final_pos_loss.detach() / scene_scale

    # Build the associated trajectory once and re-use for 2D metrics.
    assoc_traj = _extract_associated_trajectory(
        canonical_params, delta_params, assoc,
        predict_rotation, disable_delta_opacity, T,
    )                                                               # (B, T, H, W, 3)

    # ---- (3) + (4) 2D pixel error (mean/p50/p90/p95 + velocity residual) ---
    # target_cameras at (B, T, 11); build per-frame OpenCV c2w and intrinsics.
    tc_flat = target_cameras.reshape(B * T, 11)
    c2w_gl, intr = _flashworld_to_c2w_intr(tc_flat, image_height, image_width)
    c2w_cv = _flashworld_c2w_to_opencv(c2w_gl.squeeze(1)).reshape(B, T, 4, 4)
    intr_bt = intr.squeeze(1).reshape(B, T, 4)

    uv_pred, valid_pred = _project_world_to_image_torch(assoc_traj, c2w_cv, intr_bt)
    uv_gt,   valid_gt   = _project_world_to_image_torch(gt_track_ds,  c2w_cv, intr_bt)

    # Pixel-L1 mask: pixel must be GT-valid for this frame AND assoc must exist
    # (assoc is only meaningful where frame-0 was valid).
    valid0 = valid_ds[:, 0:1].expand(B, T, H, W)
    px_mask = valid_ds & valid0 & valid_pred & valid_gt             # (B, T, H, W)

    diff = (uv_pred - uv_gt).abs().sum(dim=-1)                      # L1 on (u, v) → (B, T, H, W)
    px_values = diff[px_mask]                                       # flat valid entries

    mean_px = px_values.mean() if px_values.numel() > 0 else diff.new_tensor(0.0)
    p50, p90, p95 = _percentiles(px_values, (0.50, 0.90, 0.95))
    metrics["gs_track_2d_px_l1_mean"] = mean_px
    metrics["gs_track_2d_px_l1_p50"]  = p50
    metrics["gs_track_2d_px_l1_p90"]  = p90
    metrics["gs_track_2d_px_l1_p95"]  = p95

    # 2D velocity residual: |(Δu_pred) − (Δu_gt)| L1, t >= 1.
    if T > 1:
        vel_pred = uv_pred[:, 1:] - uv_pred[:, :-1]                 # (B, T-1, H, W, 2)
        vel_gt   = uv_gt[:, 1:]   - uv_gt[:, :-1]
        vel_mask = px_mask[:, 1:] & px_mask[:, :-1]
        vel_diff = (vel_pred - vel_gt).abs().sum(dim=-1)
        vel_values = vel_diff[vel_mask]
        metrics["gs_track_2d_px_vel_l1"] = (
            vel_values.mean() if vel_values.numel() > 0 else vel_diff.new_tensor(0.0)
        )
    else:
        metrics["gs_track_2d_px_vel_l1"] = diff.new_tensor(0.0)

    # ---- (5) assoc-vs-contribution disagreement (frame-0 opacity proxy) ----
    # Contribution-top proxy: argmax of canonical opacity at each pixel.
    # True α·T requires the rendered transmittance; this proxy is a free
    # lower-bound on disagreement (opacity != contribution-top under
    # occlusion). Reported as a flag to interpret.
    canonical_opa = canonical_params[..., 3].detach()               # (B, npp, H, W)
    contrib_top = canonical_opa.argmax(dim=1)                       # (B, H, W)
    disagree = (assoc != contrib_top) & valid_ds[:, 0]              # (B, H, W)
    n_valid0 = valid_ds[:, 0].float().sum().clamp_min(1.0)
    metrics["gs_track_assoc_vs_contribution_disagreement"] = (
        disagree.float().sum() / n_valid0
    )

    return metrics


def _masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean L1 over valid entries. ``mask`` broadcasts over channels."""
    if mask is None:
        return F.l1_loss(pred, gt)
    mask = mask.to(pred.dtype)
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)  # add channel dim
    diff = (pred - gt).abs()
    weighted = diff * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-3]  # mean over channels too
    return weighted.sum() / denom


def _masked_l1_vec(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean L1 for vector-valued tensors whose last dim is the channel dim."""
    if mask is None:
        return F.l1_loss(pred, gt)
    mask = mask.to(pred.dtype)
    diff = (pred - gt).abs() * mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
    return diff.sum() / denom


# ---------------------------------------------------------------------------
# Per-frame Gaussian composition
# ---------------------------------------------------------------------------


def _compose_frame_gaussians(
    canonical_params: torch.Tensor,
    delta_params: torch.Tensor,
    predict_rotation: bool,
    disable_delta_opacity: bool,
    t_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-frame Gaussian state at target-video index ``t_idx``.

    Args:
        canonical_params: ``(B, npp, H, W, D)`` canonical parameters from
            ``DynamicGaussianSplattingDecoder.forward``.
        delta_params: ``(B, T_video - 1, npp, H, W, ch_per_g)`` per-frame
            deltas.
        predict_rotation: whether the trailing 4 channels of delta are
            quaternion updates.
        disable_delta_opacity: if true, ignore ``Δopacity_logit`` and keep
            opacity fixed to the canonical frame-0 value.
        t_idx: target-video frame index in ``[0, T_video)``. Frame 0 returns
            the canonical state with no delta applied.

    Returns:
        Tuple ``(full_params, opacity_t, delta_xyz_t)`` where:
            full_params: ``(B, npp, H, W, D)`` Gaussian params at frame t,
                ready to be reshaped + handed to the renderer.
            opacity_t: ``(B, npp, H, W)`` activated opacity at frame t.
            delta_xyz_t: ``(B, npp, H, W, 3)`` delta xyz at frame t (zero
                for t=0). Returned for the xyz-TV loss.
    """
    B, npp, H, W, D = canonical_params.shape

    # Canonical channel layout from PixelAligned3DGS:
    #   xyz(3) | opacity(1) | scale(3) | rot(4) | features(...)
    canonical_xyz = canonical_params[..., 0:3]
    canonical_opa = canonical_params[..., 3:4]   # already sigmoid'd by gs_head
    canonical_scale = canonical_params[..., 4:7]
    canonical_rot = canonical_params[..., 7:11]
    canonical_feat = canonical_params[..., 11:]

    if t_idx == 0:
        full_params = canonical_params
        opacity_t = canonical_opa.squeeze(-1)
        delta_xyz_t = torch.zeros_like(canonical_xyz)
        return full_params, opacity_t, delta_xyz_t

    # delta_params time index: t_idx - 1
    delta_t = delta_params[:, t_idx - 1]  # (B, npp, H, W, ch_per_g)
    if predict_rotation:
        delta_xyz, delta_opa_logit, delta_quat = delta_t.split([3, 1, 4], dim=-1)
    else:
        delta_xyz, delta_opa_logit = delta_t.split([3, 1], dim=-1)
        delta_quat = None

    xyz_t = canonical_xyz + delta_xyz

    # Opacity is the post-sigmoid value coming out of the static head, but
    # we want to apply an additive logit delta and re-activate. Recover the
    # logit via clamped logit() to keep gradients sane.
    if disable_delta_opacity:
        opa_t = canonical_opa
    else:
        canonical_opa_clamped = canonical_opa.clamp(1e-4, 1 - 1e-4)
        canonical_opa_logit = torch.log(canonical_opa_clamped) - torch.log1p(-canonical_opa_clamped)
        opa_t_logit = canonical_opa_logit + delta_opa_logit
        opa_t = torch.sigmoid(opa_t_logit)

    if predict_rotation:
        # Right-multiply by a (small) quaternion update, then renormalise.
        # Quat layout in canonical_rot is real-first (r, i, j, k) per
        # PixelAligned3DGS / FlashWorld convention.
        #
        # NOTE: at init the DynamicGSHead bias sets delta_quat to the
        # identity quaternion (1, 0, 0, 0), not zero, so
        # F.normalize(delta_quat) stays well-defined and
        # _quat_mul(canonical_rot, identity) == canonical_rot exactly.
        # See DynamicGSHead.__init__ for the bias pattern.
        rot_t = _quat_mul(canonical_rot, F.normalize(delta_quat, dim=-1))
        rot_t = F.normalize(rot_t, dim=-1)
    else:
        rot_t = canonical_rot

    full_params = torch.cat(
        [xyz_t, opa_t, canonical_scale, rot_t, canonical_feat],
        dim=-1,
    )
    return full_params, opa_t.squeeze(-1), delta_xyz


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two real-first quaternions ``(..., 4)``."""
    r1, i1, j1, k1 = torch.unbind(q1, dim=-1)
    r2, i2, j2, k2 = torch.unbind(q2, dim=-1)
    return torch.stack(
        [
            r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2,
            r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2,
            r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2,
            r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2,
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_frame0_association(
    canonical_params: torch.Tensor,
    gt_track_frame0: torch.Tensor,
    valid_mask_frame0: torch.Tensor,
) -> torch.Tensor:
    """Build a fixed per-pixel Gaussian association at frame 0.

    For each valid foreground pixel, find the Gaussian (among npp candidates)
    whose canonical xyz is nearest to the GT track position at frame 0.

    Args:
        canonical_params: ``(B, npp, H, W, D)`` canonical Gaussian params.
        gt_track_frame0: ``(B, H_img, W_img, 3)`` GT 3D track positions at t=0.
        valid_mask_frame0: ``(B, H_img, W_img)`` foreground validity mask at t=0.

    Returns:
        assoc: ``(B, H, W)`` long tensor, index into the npp dim for the
            associated Gaussian per pixel. Invalid pixels get index 0 (arbitrary).
    """
    B, npp, H, W, D = canonical_params.shape
    canonical_xyz = canonical_params[..., 0:3]  # (B, npp, H, W, 3)

    # GT tracks are at image resolution; canonical xyz is at GS-head resolution
    # (typically smaller). Resize GT to match.
    H_img, W_img = gt_track_frame0.shape[1], gt_track_frame0.shape[2]
    if H_img != H or W_img != W:
        gt_resized = F.interpolate(
            gt_track_frame0.permute(0, 3, 1, 2),  # (B, 3, H_img, W_img)
            size=(H, W), mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)  # (B, H, W, 3)
        mask_resized = F.interpolate(
            valid_mask_frame0.unsqueeze(1).float(),
            size=(H, W), mode="nearest",
        ).squeeze(1) > 0.5
    else:
        gt_resized = gt_track_frame0
        mask_resized = valid_mask_frame0

    # Distance: (B, npp, H, W) — L2 between each Gaussian center and GT.
    dist = (canonical_xyz - gt_resized.unsqueeze(1)).norm(dim=-1)  # (B, npp, H, W)
    # Set invalid pixels to large distance so argmin picks index 0 (arbitrary).
    invalid = ~mask_resized.bool()  # (B, H, W)
    dist.masked_fill_(invalid.unsqueeze(1).expand_as(dist), 1e8)
    assoc = dist.argmin(dim=1)  # (B, H, W) — index into npp
    return assoc


def _expand_batch_tensor(t: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if t.shape[0] == batch_size:
        return t
    if t.shape[0] == 1:
        return t.expand(batch_size, *t.shape[1:])
    raise ValueError(f"{name} batch {t.shape[0]} does not match B={batch_size}")


def _gather_query_points_spatial(
    values: torch.Tensor,
    query_xy: torch.Tensor,
) -> torch.Tensor:
    """Gather `(x, y)` query points from a flattened spatial dimension.

    Args:
        values:
          - `(B, npp, H, W, C)` -> returns `(B, npp, N, C)`
          - `(B, T, npp, H, W, C)` -> returns `(B, T, npp, N, C)`
        query_xy: `(B, N, 2)` or `(1, N, 2)` int64 with `(x, y)` order.
    """
    if values.dim() == 5:
        B, npp, H, W, C = values.shape
        query_xy = _expand_batch_tensor(query_xy, B, "target_tracks_query_xy")
        lin_idx = (query_xy[..., 1] * W + query_xy[..., 0]).to(torch.long)  # (B, N)
        flat = values.reshape(B, npp, H * W, C)
        gather_idx = lin_idx[:, None, :, None].expand(B, npp, lin_idx.shape[1], C)
        return torch.gather(flat, dim=2, index=gather_idx)
    if values.dim() == 6:
        B, T, npp, H, W, C = values.shape
        query_xy = _expand_batch_tensor(query_xy, B, "target_tracks_query_xy")
        lin_idx = (query_xy[..., 1] * W + query_xy[..., 0]).to(torch.long)  # (B, N)
        flat = values.reshape(B, T, npp, H * W, C)
        gather_idx = lin_idx[:, None, None, :, None].expand(B, T, npp, lin_idx.shape[1], C)
        return torch.gather(flat, dim=3, index=gather_idx)
    raise ValueError(f"Unsupported values rank for query gather: {values.dim()}")


def _resize_dense_track_targets(
    target_track_video: torch.Tensor,
    target_track_valid_mask: torch.Tensor,
    h_out: int,
    w_out: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize dense GT 3D tracks + validity mask to the GS-head spatial grid."""
    B, T, H, W, _ = target_track_video.shape
    if H == h_out and W == w_out:
        return target_track_video, target_track_valid_mask > 0.5

    gt_track_ds = F.interpolate(
        target_track_video.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W),
        size=(h_out, w_out),
        mode="bilinear",
        align_corners=False,
    ).reshape(B, T, 3, h_out, w_out).permute(0, 1, 3, 4, 2).contiguous()

    valid_ds = F.interpolate(
        target_track_valid_mask.reshape(B * T, 1, H, W).float(),
        size=(h_out, w_out),
        mode="nearest",
    ).reshape(B, T, h_out, w_out) > 0.5
    return gt_track_ds, valid_ds


def _compute_dense_associated_track_losses(
    canonical_params: torch.Tensor,
    delta_params: torch.Tensor,
    target_track_video: torch.Tensor,
    target_track_valid_mask: torch.Tensor,
    target_cameras: torch.Tensor,
    image_height: int,
    image_width: int,
    predict_rotation: bool,
    disable_delta_opacity: bool,
    track_canonical_loss_weight: float,
    track_delta_loss_weight: float,
    track_final_pos_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Dense frame-0-associated supervision using GT target-view 3D tracks.

    Returns ``(canonical_loss, delta_loss, final_pos_loss, metrics)`` where
    ``metrics`` carries Phase-1 diagnostics (scene-scale normalized 3D error,
    2D pixel error mean/p50/p90/p95, 2D velocity residual, assoc-vs-contribution
    disagreement). Metrics are detached; they never contribute to ``gs_loss``.
    """
    B, npp, H, W, _ = canonical_params.shape
    gt_track_ds, valid_ds = _resize_dense_track_targets(
        target_track_video.to(device=canonical_params.device, dtype=torch.float32),
        target_track_valid_mask.to(device=canonical_params.device, dtype=torch.float32),
        H,
        W,
    )

    gt_track_t0 = gt_track_ds[:, 0]
    valid_t0 = valid_ds[:, 0]
    assoc = _build_frame0_association(canonical_params, gt_track_t0, valid_t0)

    # Shared associated trajectory — same quantity the eval "GS 2D Tracks"
    # panel reads. Used by both the loss terms and the Phase-1 metrics.
    pred_full = _extract_associated_trajectory(
        canonical_params, delta_params, assoc,
        predict_rotation, disable_delta_opacity, gt_track_ds.shape[1],
    )
    canonical_assoc = pred_full[:, 0]

    canonical_loss = canonical_params.new_tensor(0.0)
    delta_loss = canonical_params.new_tensor(0.0)
    final_pos_loss = canonical_params.new_tensor(0.0)

    if track_canonical_loss_weight > 0.0:
        canonical_loss = _masked_l1_vec(canonical_assoc, gt_track_t0, valid_t0.float())

    if track_delta_loss_weight > 0.0 and pred_full.shape[1] > 1:
        pred_delta = pred_full[:, 1:] - pred_full[:, :1]
        gt_delta = gt_track_ds[:, 1:] - gt_track_ds[:, :1]
        valid_future = valid_ds[:, 1:] & valid_t0.unsqueeze(1)
        delta_loss = _masked_l1_vec(pred_delta, gt_delta, valid_future.float())

    if track_final_pos_loss_weight > 0.0:
        final_pos_loss = _masked_l1_vec(pred_full, gt_track_ds, valid_ds.float())

    # Phase-1 metrics. Always computed when dense GT is available.
    metrics = _compute_track_metrics(
        canonical_params=canonical_params,
        delta_params=delta_params,
        gt_track_ds=gt_track_ds,
        valid_ds=valid_ds,
        assoc=assoc,
        target_cameras=target_cameras,
        image_height=image_height,
        image_width=image_width,
        predict_rotation=predict_rotation,
        disable_delta_opacity=disable_delta_opacity,
        final_pos_loss=(
            final_pos_loss
            if track_final_pos_loss_weight > 0.0
            else _masked_l1_vec(pred_full, gt_track_ds, valid_ds.float())
        ),
    )

    return canonical_loss, delta_loss, final_pos_loss, metrics


def _compute_persistent_track_losses(
    canonical_params: torch.Tensor,
    delta_params: torch.Tensor,
    target_tracks_world: torch.Tensor,
    target_tracks_world_valid_mask: Optional[torch.Tensor],
    target_tracks_query_xy: torch.Tensor,
    target_track_video: Optional[torch.Tensor],
    target_track_valid_mask: Optional[torch.Tensor],
    target_cameras: torch.Tensor,
    image_height: int,
    image_width: int,
    predict_rotation: bool,
    disable_delta_opacity: bool,
    track_canonical_loss_weight: float,
    track_delta_loss_weight: float,
    track_final_pos_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Direct persistent-track supervision on frame-0-associated Gaussians.

    Returns a fourth element ``metrics``: Phase-1 diagnostics on the dense
    path (empty dict for the sparse fallback path).
    """
    if target_track_video is not None and target_track_valid_mask is not None:
        return _compute_dense_associated_track_losses(
            canonical_params=canonical_params,
            delta_params=delta_params,
            target_track_video=target_track_video,
            target_track_valid_mask=target_track_valid_mask,
            target_cameras=target_cameras,
            image_height=image_height,
            image_width=image_width,
            predict_rotation=predict_rotation,
            disable_delta_opacity=disable_delta_opacity,
            track_canonical_loss_weight=track_canonical_loss_weight,
            track_delta_loss_weight=track_delta_loss_weight,
            track_final_pos_loss_weight=track_final_pos_loss_weight,
        )

    B, npp, H, W, _ = canonical_params.shape
    target_tracks_world = target_tracks_world.to(device=canonical_params.device, dtype=torch.float32)
    if target_tracks_world.dim() == 3:
        target_tracks_world = target_tracks_world.unsqueeze(0)
    target_tracks_world = _expand_batch_tensor(target_tracks_world, B, "target_tracks_world")
    if target_tracks_world_valid_mask is None:
        valid_mask = (target_tracks_world.abs().sum(dim=-1) > 1e-8).to(torch.float32)
    else:
        if target_tracks_world_valid_mask.dim() == 2:
            target_tracks_world_valid_mask = target_tracks_world_valid_mask.unsqueeze(0)
        valid_mask = _expand_batch_tensor(
            target_tracks_world_valid_mask.to(device=canonical_params.device, dtype=torch.float32),
            B,
            "target_tracks_world_valid_mask",
        )
    if target_tracks_query_xy.dim() == 2:
        target_tracks_query_xy = target_tracks_query_xy.unsqueeze(0)
    query_xy = _expand_batch_tensor(
        target_tracks_query_xy.to(device=canonical_params.device, dtype=torch.long),
        B,
        "target_tracks_query_xy",
    )

    canonical_xyz = canonical_params[..., 0:3]  # (B, npp, H, W, 3)
    canonical_query = _gather_query_points_spatial(canonical_xyz, query_xy)  # (B, npp, N, 3)
    gt_anchor = target_tracks_world[:, :, 0, :]  # (B, N, 3)
    valid_anchor = valid_mask[:, :, 0] > 0.5

    dist = (canonical_query - gt_anchor.unsqueeze(1)).norm(dim=-1)  # (B, npp, N)
    dist = dist.masked_fill(~valid_anchor.unsqueeze(1), 1e8)
    assoc = dist.argmin(dim=1)  # (B, N)
    assoc_exp = assoc.unsqueeze(1).unsqueeze(-1).expand(B, 1, assoc.shape[1], 3)
    canonical_assoc = torch.gather(canonical_query, dim=1, index=assoc_exp).squeeze(1)  # (B, N, 3)

    canonical_loss = canonical_params.new_tensor(0.0)
    delta_loss = canonical_params.new_tensor(0.0)
    final_pos_loss = canonical_params.new_tensor(0.0)

    if track_canonical_loss_weight > 0.0:
        canonical_loss = _masked_l1_vec(canonical_assoc, gt_anchor, valid_anchor.float())

    if delta_params.shape[1] > 0 and (track_delta_loss_weight > 0.0 or track_final_pos_loss_weight > 0.0):
        delta_xyz = delta_params[..., 0:3]  # (B, T-1, npp, H, W, 3)
        delta_query = _gather_query_points_spatial(delta_xyz, query_xy)  # (B, T-1, npp, N, 3)
        assoc_delta_exp = assoc[:, None, None, :, None].expand(
            B, delta_query.shape[1], 1, assoc.shape[1], 3
        )
        delta_assoc = torch.gather(delta_query, dim=2, index=assoc_delta_exp).squeeze(2)  # (B, T-1, N, 3)
        pred_delta = delta_assoc.permute(0, 2, 1, 3).contiguous()  # (B, N, T-1, 3)
        gt_delta = target_tracks_world[:, :, 1:, :] - gt_anchor.unsqueeze(2)
        valid_future = valid_mask[:, :, 1:]

        if track_delta_loss_weight > 0.0:
            delta_loss = _masked_l1_vec(pred_delta, gt_delta, valid_future)

        if track_final_pos_loss_weight > 0.0:
            pred_full = torch.cat(
                [canonical_assoc.unsqueeze(2), canonical_assoc.unsqueeze(2) + pred_delta],
                dim=2,
            )
            final_pos_loss = _masked_l1_vec(pred_full, target_tracks_world, valid_mask)
    elif track_final_pos_loss_weight > 0.0:
        final_pos_loss = _masked_l1_vec(
            canonical_assoc.unsqueeze(2),
            target_tracks_world[:, :, :1, :],
            valid_mask[:, :, :1],
        )

    # Sparse fallback path: no Phase-1 metrics (they require dense GT grid).
    return canonical_loss, delta_loss, final_pos_loss, {}


def compute_gs_splatting_loss_dynamic(
    gs_decoder_dynamic: DynamicGaussianSplattingDecoder,
    gs_features_target: torch.Tensor,
    latents_pred_target: torch.Tensor,
    target_cameras: torch.Tensor,
    gt_target_rgb: torch.Tensor,
    target_track_video: Optional[torch.Tensor],
    target_track_valid_mask: Optional[torch.Tensor],
    image_height: int,
    image_width: int,
    target_tracks_world: Optional[torch.Tensor] = None,
    target_tracks_world_valid_mask: Optional[torch.Tensor] = None,
    target_tracks_query_xy: Optional[torch.Tensor] = None,
    target_depth_video: Optional[torch.Tensor] = None,
    target_depth_valid_mask: Optional[torch.Tensor] = None,
    rgb_loss_weight: float = 1.0,
    pos_loss_weight: float = 1.0,
    depth_loss_weight: float = 0.5,
    opa_tv_weight: float = 0.01,
    xyz_tv_weight: float = 0.01,
    direct_pos_loss_weight: float = 0.0,
    gs_track_supervision_mode: str = "pointmap",
    track_canonical_loss_weight: float = 0.0,
    track_delta_loss_weight: float = 0.0,
    track_final_pos_loss_weight: float = 0.0,
    # Objective A (2026-04-22 spec §1-bis): associated trajectory vs persistent GT in 2D pixels
    assoc_track_2d_loss_weight: float = 0.0,
    # Associated-trajectory acceleration smoothness (world-units second derivative)
    track_accel_loss_weight: float = 0.0,
    track_loss_quantile: float = 0.98,
    predict_rotation: bool = False,
    disable_delta_opacity: bool = False,
    detach_predicted_latents: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute the dynamic GS loss (canonical + deltas) over the full
    target-video timeline.

    Per target-video frame ``t`` this:
      1. Composes ``(canonical + delta_{t-1})`` Gaussians (or canonical
         alone for ``t == 0``).
      2. Renders RGB + depth through ``target_cameras[:, t]`` via
         :func:`gaussian_render`.
      3. Renders a per-pixel 3D-position map via
         :func:`gaussian_render_features` with per-Gaussian xyz as "color".
      4. L1s each render against the corresponding GT frame, masked where
         GT masks are available.

    Args:
        gs_decoder_dynamic: :class:`DynamicGaussianSplattingDecoder` instance
            wrapping the static decoder.
        gs_features_target: ``(B, feat_dim, T_lat, H', W')`` GS features for
            the target domain.
        latents_pred_target: ``(B, z_dim, T_lat, H', W')`` predicted clean latents.
        detach_predicted_latents: if True (default), detach before the dynamic decoder.
        target_cameras: ``(B, T_video, 11)`` target FlashWorld cameras at
            target-video rate. **No subsampling is performed** — the
            caller must provide the full timeline.
        gt_target_rgb: ``(B, 3, T_video, H, W)`` GT target RGB in ``[-1, 1]``.
        target_track_video: ``(B, T_video, H, W, 3)`` GT target-view dense
            3D tracks in cam0-normalised world space, or ``None`` to skip
            the position loss term entirely.
        target_track_valid_mask: ``(B, T_video, H, W)`` valid mask for
            ``target_track_video``, or ``None``.
        image_height, image_width: render resolution (full image, not
            latent).
        target_tracks_world: ``(B, N, T_video, 3)`` persistent GT frame-0-query
            tracks, or ``None``.
        target_tracks_world_valid_mask: ``(B, N, T_video)`` validity mask for
            ``target_tracks_world``, or ``None``.
        target_tracks_query_xy: ``(B, N, 2)`` or ``(1, N, 2)`` query pixel
            coordinates in ``(x, y)`` order matching ``target_tracks_world``.
        target_depth_video: ``(B, T_video, H, W)`` GT target-view depth
            (cam0 common-scale units), or ``None`` to skip the depth term.
        target_depth_valid_mask: ``(B, T_video, H, W)`` valid mask for
            ``target_depth_video``, or ``None``.
        rgb_loss_weight, pos_loss_weight, depth_loss_weight, opa_tv_weight,
            xyz_tv_weight: per-term weights.
        gs_track_supervision_mode: ``pointmap`` or ``persistent_tracks``. The
            persistent-track mode adds direct canonical/delta/final-position
            supervision but does not disable the pointmap loss automatically.
        predict_rotation: whether ``delta_params`` carries a 4-channel
            rotation update tail.
        disable_delta_opacity: whether to ignore the per-frame
            ``Δopacity_logit`` and keep opacity canonical across time.

    Returns:
        Dict with ``gs_loss`` (the weighted sum) and per-term diagnostics
        ``gs_rgb_loss``, ``gs_pos_loss``, ``gs_depth_loss``,
        ``gs_opa_tv_loss``, ``gs_xyz_tv_loss``.
    """
    # ---- Shape normalisation -------------------------------------------------
    if target_cameras.dim() == 2:
        target_cameras = target_cameras.unsqueeze(0)
    if gt_target_rgb.dim() == 4:
        gt_target_rgb = gt_target_rgb.unsqueeze(0)
    if target_track_video is not None and target_track_video.dim() == 4:
        target_track_video = target_track_video.unsqueeze(0)
    if target_track_valid_mask is not None and target_track_valid_mask.dim() == 3:
        target_track_valid_mask = target_track_valid_mask.unsqueeze(0)
    if target_tracks_world is not None and target_tracks_world.dim() == 3:
        target_tracks_world = target_tracks_world.unsqueeze(0)
    if target_tracks_world_valid_mask is not None and target_tracks_world_valid_mask.dim() == 2:
        target_tracks_world_valid_mask = target_tracks_world_valid_mask.unsqueeze(0)
    if target_tracks_query_xy is not None and target_tracks_query_xy.dim() == 2:
        target_tracks_query_xy = target_tracks_query_xy.unsqueeze(0)
    if target_depth_video is not None and target_depth_video.dim() == 3:
        target_depth_video = target_depth_video.unsqueeze(0)
    if target_depth_valid_mask is not None and target_depth_valid_mask.dim() == 3:
        target_depth_valid_mask = target_depth_valid_mask.unsqueeze(0)

    if target_cameras.dim() != 3:
        raise ValueError(
            f"target_cameras must be (B, T_video, 11), got {tuple(target_cameras.shape)}"
        )

    B = gs_features_target.shape[0]
    T_video = int(target_cameras.shape[1])
    target_device = gs_features_target.device

    # ---- Full-timeline GT asserts (no subsampling, matching the static path) -
    if gt_target_rgb.shape[0] != B or gt_target_rgb.shape[2] != T_video:
        raise ValueError(
            f"gt_target_rgb timeline {tuple(gt_target_rgb.shape)} does not match "
            f"(B={B}, T_video={T_video})"
        )
    if target_track_video is not None and (
        target_track_video.shape[0] != B or target_track_video.shape[1] != T_video
    ):
        raise ValueError(
            f"target_track_video timeline {tuple(target_track_video.shape)} does "
            f"not match (B={B}, T_video={T_video})"
        )
    if target_track_valid_mask is not None and (
        target_track_valid_mask.shape[0] != B or target_track_valid_mask.shape[1] != T_video
    ):
        raise ValueError(
            f"target_track_valid_mask timeline {tuple(target_track_valid_mask.shape)} "
            f"does not match (B={B}, T_video={T_video})"
        )
    if target_depth_video is not None and (
        target_depth_video.shape[0] != B or target_depth_video.shape[1] != T_video
    ):
        raise ValueError(
            f"target_depth_video timeline {tuple(target_depth_video.shape)} does "
            f"not match (B={B}, T_video={T_video})"
        )
    if target_depth_valid_mask is not None and (
        target_depth_valid_mask.shape[0] != B or target_depth_valid_mask.shape[1] != T_video
    ):
        raise ValueError(
            f"target_depth_valid_mask timeline {tuple(target_depth_valid_mask.shape)} "
            f"does not match (B={B}, T_video={T_video})"
        )

    target_cameras = target_cameras.to(device=target_device, dtype=torch.float32)
    gt_target_rgb = gt_target_rgb.to(device=target_device)
    if target_track_video is not None:
        target_track_video = target_track_video.to(device=target_device)
    if target_track_valid_mask is not None:
        target_track_valid_mask = target_track_valid_mask.to(device=target_device)
    if target_tracks_world is not None:
        target_tracks_world = target_tracks_world.to(device=target_device)
    if target_tracks_world_valid_mask is not None:
        target_tracks_world_valid_mask = target_tracks_world_valid_mask.to(device=target_device)
    if target_tracks_query_xy is not None:
        target_tracks_query_xy = target_tracks_query_xy.to(device=target_device, dtype=torch.long)
    if target_depth_video is not None:
        target_depth_video = target_depth_video.to(device=target_device)
    if target_depth_valid_mask is not None:
        target_depth_valid_mask = target_depth_valid_mask.to(device=target_device)
    canonical_camera = target_cameras[:, 0]  # (B, 11)

    # ---- Decode canonical + per-target-video-frame deltas -------------------
    z_dyn = (
        latents_pred_target.detach()
        if detach_predicted_latents
        else latents_pred_target
    )
    canonical_params, delta_params = gs_decoder_dynamic(
        gs_features_target,
        z_dyn,
        canonical_camera,
        t_video=T_video,
    )
    # canonical_params: (B, npp, H, W, D)
    # delta_params:     (B, T_video - 1, npp, H, W, ch_per_g)
    npp = canonical_params.shape[1]
    H_full = canonical_params.shape[2]
    W_full = canonical_params.shape[3]
    D = canonical_params.shape[4]
    sh_degree = gs_decoder_dynamic.sh_degree

    # ---- Permute GT to per-frame indexing layouts ---------------------------
    # gt_target_rgb: (B, 3, T_video, H, W) → (B, T_video, 3, H, W)
    gt_rgb_per_frame = gt_target_rgb.permute(0, 2, 1, 3, 4).contiguous()

    if target_track_video is not None:
        # (B, T_video, H, W, 3) → (B, T_video, 3, H, W)
        gt_pos_per_frame = target_track_video.permute(0, 1, 4, 2, 3).contiguous().to(torch.float32)
    else:
        gt_pos_per_frame = None

    if target_track_valid_mask is not None:
        gt_pos_mask_per_frame = target_track_valid_mask.to(torch.float32)
    else:
        gt_pos_mask_per_frame = None

    if target_depth_video is not None:
        gt_depth_per_frame = target_depth_video.to(torch.float32)
    else:
        gt_depth_per_frame = None

    if target_depth_valid_mask is not None:
        gt_depth_mask_per_frame = target_depth_valid_mask.to(torch.float32)
    else:
        gt_depth_mask_per_frame = None

    # ---- Per-frame render + loss accumulation over the FULL T_video ---------
    rgb_losses = []
    mse_losses = []
    pos_losses = []
    depth_losses = []
    opacity_history: list[torch.Tensor] = []
    delta_xyz_history: list[torch.Tensor] = []

    for t_idx in range(T_video):
        full_params, opa_t, delta_xyz_t = _compose_frame_gaussians(
            canonical_params,
            delta_params,
            predict_rotation,
            disable_delta_opacity,
            t_idx,
        )
        # Reshape to renderer-friendly (B, V=1, npp*H*W, D).
        flat = full_params.reshape(B, npp * H_full * W_full, D).unsqueeze(1)

        cams_t = target_cameras[:, t_idx]  # (B, 11)
        c2ws_t, intr_t = _flashworld_to_c2w_intr(cams_t, image_height, image_width)
        # _flashworld_to_c2w_intr returns (B, 1, 4, 4) and (B, 1, 4)

        rendered_rgb, rendered_depth = gaussian_render(
            flat, c2ws_t, intr_t,
            image_width, image_height,
            sh_degree=sh_degree,
            bg_mode="white",
            use_checkpoint=gs_decoder_dynamic.render_checkpointing,
            patch_size=getattr(gs_decoder_dynamic, "render_patch_size", None),
        )
        # rendered_rgb: (B, 1, 3, H, W) → (B, 3, H, W)
        # rendered_depth: (B, 1, 1, H, W) → (B, 1, H, W)
        rendered_rgb = rendered_rgb.squeeze(1)
        rendered_depth = rendered_depth.squeeze(1)

        # Lyra-1 §4.2: L_rgb = λmse · L_mse (squared error). Prior revisions
        # used F.l1_loss here — that was an implementation choice carried
        # over from the original dynamic-mode refactor, not the Lyra-1
        # recipe. Switched to F.mse_loss 2026-04-24 to match the paper
        # recipe the rest of the GS head is calibrated against. MSE is
        # also what the cache overfit (35 dB reference) optimises.
        _rgb_mse_t = F.mse_loss(rendered_rgb, gt_rgb_per_frame[:, t_idx])
        rgb_losses.append(_rgb_mse_t)
        mse_losses.append(_rgb_mse_t.detach())

        if gt_depth_per_frame is not None:
            depth_mask_t = (
                gt_depth_mask_per_frame[:, t_idx] if gt_depth_mask_per_frame is not None else None
            )
            # gt depth: (B, H, W) → (B, 1, H, W) for broadcasting with rendered.
            gt_depth_t = gt_depth_per_frame[:, t_idx].unsqueeze(1)
            depth_losses.append(_masked_l1(rendered_depth, gt_depth_t, depth_mask_t))

        if gt_pos_per_frame is not None:
            # Build per-Gaussian feature = current xyz, then render as features.
            xyz_t = full_params[..., 0:3]  # (B, npp, H, W, 3)
            opa_t_for_render = full_params[..., 3:4].squeeze(-1)
            scale_t = full_params[..., 4:7]
            rot_t = full_params[..., 7:11]

            # (B, V=1, N_pts, *)
            xyz_flat = xyz_t.reshape(B, 1, npp * H_full * W_full, 3)
            feat_flat = xyz_flat  # render xyz as colors → position map
            scale_flat = scale_t.reshape(B, 1, npp * H_full * W_full, 3)
            rot_flat = rot_t.reshape(B, 1, npp * H_full * W_full, 4)
            opa_flat = opa_t_for_render.reshape(B, 1, npp * H_full * W_full)

            rendered_pos, _ = gaussian_render_features(
                xyz_flat, feat_flat, scale_flat, rot_flat, opa_flat,
                c2ws_t, intr_t,
                image_width, image_height,
                use_checkpoint=gs_decoder_dynamic.render_checkpointing,
                bg_value=0.0,
            )
            rendered_pos = rendered_pos.squeeze(1)  # (B, 3, H, W)

            mask_t = gt_pos_mask_per_frame[:, t_idx] if gt_pos_mask_per_frame is not None else None
            pos_losses.append(_masked_l1(rendered_pos, gt_pos_per_frame[:, t_idx], mask_t))

        opacity_history.append(opa_t)
        delta_xyz_history.append(delta_xyz_t)

    # ---- Direct-center loss (SoM-style, optional) --------------------------
    # Fixed frame-0 association: for each foreground pixel, pick the Gaussian
    # (from npp) whose canonical xyz is nearest to the GT track at t=0. Then
    # supervise that Gaussian's xyz at every frame directly (no rendering).
    direct_pos_loss = canonical_params.new_tensor(0.0)
    if direct_pos_loss_weight > 0.0 and target_track_video is not None and target_track_valid_mask is not None:
        gt_track_t0 = target_track_video[:, 0]      # (B, H_img, W_img, 3)
        valid_t0 = target_track_valid_mask[:, 0]     # (B, H_img, W_img)
        assoc = _build_frame0_association(canonical_params, gt_track_t0, valid_t0)  # (B, H, W)

        # Build a downsampled valid mask at the GS-head resolution.
        H_img_local = int(gt_track_t0.shape[1])
        W_img_local = int(gt_track_t0.shape[2])
        if H_img_local != H_full or W_img_local != W_full:
            valid_ds = F.interpolate(
                target_track_valid_mask.float().unsqueeze(2),  # (B, T, 1, H, W)
                size=(H_full, W_full), mode="nearest",
            ).squeeze(2) > 0.5  # (B, T, H, W)
            gt_track_ds = F.interpolate(
                target_track_video.permute(0, 1, 4, 2, 3).reshape(B * T_video, 3, H_img_local, W_img_local),
                size=(H_full, W_full), mode="bilinear", align_corners=False,
            ).reshape(B, T_video, 3, H_full, W_full).permute(0, 1, 3, 4, 2)  # (B, T, H, W, 3)
        else:
            valid_ds = target_track_valid_mask > 0.5
            gt_track_ds = target_track_video

        direct_pos_losses = []
        for t_idx in range(T_video):
            full_t, _, _ = _compose_frame_gaussians(
                canonical_params,
                delta_params,
                predict_rotation,
                disable_delta_opacity,
                t_idx,
            )
            xyz_t = full_t[..., 0:3]  # (B, npp, H, W, 3)

            # Gather the associated Gaussian's xyz per pixel.
            # assoc: (B, H, W) → expand to (B, 1, H, W, 3) for gather on dim=1 (npp).
            assoc_exp = assoc.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, -1, 3)  # (B, 1, H, W, 3)
            xyz_assoc = torch.gather(xyz_t, dim=1, index=assoc_exp).squeeze(1)  # (B, H, W, 3)

            gt_t = gt_track_ds[:, t_idx]     # (B, H, W, 3)
            mask_t = valid_ds[:, t_idx]       # (B, H, W)

            diff = (xyz_assoc - gt_t).abs()   # (B, H, W, 3)
            if mask_t.any():
                masked_diff = diff * mask_t.unsqueeze(-1).float()
                direct_pos_losses.append(
                    masked_diff.sum() / (mask_t.float().sum() * 3 + 1e-8)
                )

        if direct_pos_losses:
            direct_pos_loss = torch.stack(direct_pos_losses).mean()

    track_canonical_loss = canonical_params.new_tensor(0.0)
    track_delta_loss = canonical_params.new_tensor(0.0)
    track_final_pos_loss = canonical_params.new_tensor(0.0)
    track_metrics: dict[str, torch.Tensor] = {}
    if gs_track_supervision_mode not in {"pointmap", "persistent_tracks"}:
        raise ValueError(f"Unknown gs_track_supervision_mode={gs_track_supervision_mode}")
    if (
        gs_track_supervision_mode == "persistent_tracks"
        and max(track_canonical_loss_weight, track_delta_loss_weight, track_final_pos_loss_weight) > 0.0
        and target_tracks_world is not None
        and target_tracks_query_xy is not None
    ):
        track_canonical_loss, track_delta_loss, track_final_pos_loss, track_metrics = _compute_persistent_track_losses(
            canonical_params=canonical_params,
            delta_params=delta_params,
            target_tracks_world=target_tracks_world,
            target_tracks_world_valid_mask=target_tracks_world_valid_mask,
            target_tracks_query_xy=target_tracks_query_xy,
            target_track_video=target_track_video,
            target_track_valid_mask=target_track_valid_mask,
            target_cameras=target_cameras,
            image_height=image_height,
            image_width=image_width,
            predict_rotation=predict_rotation,
            disable_delta_opacity=disable_delta_opacity,
            track_canonical_loss_weight=track_canonical_loss_weight,
            track_delta_loss_weight=track_delta_loss_weight,
            track_final_pos_loss_weight=track_final_pos_loss_weight,
        )

    # ---- Objective A + accel + persistent-track metrics (2026-04-22 spec §1-bis) ---
    # These run whenever target_tracks_world is available with dense row-major
    # indexing AND persistent_tracks supervision mode is active. They always
    # log the persistent metrics (detached); they contribute to gs_loss only
    # when their weights > 0.
    assoc_track_2d_loss = canonical_params.new_tensor(0.0)
    track_accel_loss = canonical_params.new_tensor(0.0)
    persistent_metrics: dict[str, torch.Tensor] = {}
    if (
        gs_track_supervision_mode == "persistent_tracks"
        and target_tracks_world is not None
        and target_track_video is not None  # needed only to build assoc at frame 0
        and target_track_valid_mask is not None
    ):
        try:
            H_canon = canonical_params.shape[2]
            W_canon = canonical_params.shape[3]
            tw_grid, tw_valid = _persistent_tracks_to_grid(
                target_tracks_world,
                target_tracks_world_valid_mask,
                H_canon,
                W_canon,
            )
            # Build the same frame-0 association used by the existing dense loss.
            gt_t0 = target_track_video.to(device=canonical_params.device, dtype=torch.float32)
            valid_t0 = target_track_valid_mask.to(device=canonical_params.device, dtype=torch.float32)
            if gt_t0.shape[2] != H_canon or gt_t0.shape[3] != W_canon:
                gt_t0_resized, valid_t0_resized = _resize_dense_track_targets(
                    gt_t0, valid_t0, H_canon, W_canon,
                )
            else:
                gt_t0_resized, valid_t0_resized = gt_t0, valid_t0 > 0.5
            assoc = _build_frame0_association(
                canonical_params,
                gt_t0_resized[:, 0],
                valid_t0_resized[:, 0].to(torch.float32),
            )
            pred_full = _extract_associated_trajectory(
                canonical_params, delta_params, assoc,
                predict_rotation, disable_delta_opacity, tw_grid.shape[1],
            )

            # ---- Objective A loss ----
            if assoc_track_2d_loss_weight > 0.0:
                assoc_track_2d_loss = _compute_assoc_persistent_track_loss(
                    pred_full=pred_full,
                    tw_grid=tw_grid,
                    tw_valid=tw_valid,
                    target_cameras=target_cameras,
                    image_height=image_height,
                    image_width=image_width,
                    quantile=track_loss_quantile,
                )

            # ---- Accel loss ----
            if track_accel_loss_weight > 0.0:
                track_accel_loss = _compute_accel_loss(pred_full, tw_valid)

            # ---- Persistent metrics (always) ----
            persistent_metrics = _compute_assoc_persistent_track_metrics(
                pred_full=pred_full,
                tw_grid=tw_grid,
                tw_valid=tw_valid,
                target_cameras=target_cameras,
                image_height=image_height,
                image_width=image_width,
            )
        except ValueError as exc:
            # Row-major assumption failed (e.g. sparse target_tracks_world).
            # Skip Objective A silently — fall through to the pointmap-only path.
            persistent_metrics = {
                "gs_assoc_persistent_track_path_skipped": canonical_params.new_tensor(1.0),
            }
            if assoc_track_2d_loss_weight > 0.0 or track_accel_loss_weight > 0.0:
                raise RuntimeError(
                    f"Objective A / accel loss requested (weights > 0) but persistent-track "
                    f"grid reshape failed: {exc}"
                )

    # Number of rendered frames must equal T_video — this is the regression
    # guardrail for Codex finding #1 (dynamic path was previously stuck at
    # T_lat). If a future refactor re-introduces subsampling, this invariant
    # will blow up before any wandb metric can silently drop frames.
    assert len(rgb_losses) == T_video, (
        f"dynamic loss ran {len(rgb_losses)} frames, expected T_video={T_video}"
    )

    rgb_loss = torch.stack(rgb_losses).mean()
    rgb_mse = (
        torch.stack(mse_losses).mean()
        if len(mse_losses) > 0
        else rgb_loss.new_tensor(0.0)
    )

    if pos_losses:
        pos_loss = torch.stack(pos_losses).mean()
    else:
        pos_loss = rgb_loss.new_tensor(0.0)

    if depth_losses:
        depth_loss = torch.stack(depth_losses).mean()
    else:
        depth_loss = rgb_loss.new_tensor(0.0)

    # ---- Temporal smoothness terms ------------------------------------------
    if len(opacity_history) > 1:
        opa_tv = torch.stack(
            [F.l1_loss(opacity_history[t], opacity_history[t - 1])
             for t in range(1, len(opacity_history))],
        ).mean()
    else:
        opa_tv = rgb_loss.new_tensor(0.0)

    if len(delta_xyz_history) > 1:
        xyz_tv = torch.stack(
            [F.l1_loss(delta_xyz_history[t], delta_xyz_history[t - 1])
             for t in range(1, len(delta_xyz_history))],
        ).mean()
    else:
        xyz_tv = rgb_loss.new_tensor(0.0)

    gs_loss = (
        rgb_loss_weight * rgb_loss
        + pos_loss_weight * pos_loss
        + depth_loss_weight * depth_loss
        + opa_tv_weight * opa_tv
        + xyz_tv_weight * xyz_tv
        + direct_pos_loss_weight * direct_pos_loss
        + track_canonical_loss_weight * track_canonical_loss
        + track_delta_loss_weight * track_delta_loss
        + track_final_pos_loss_weight * track_final_pos_loss
        + assoc_track_2d_loss_weight * assoc_track_2d_loss
        + track_accel_loss_weight * track_accel_loss
    )

    # PSNR from MSE on RGB in [-1, 1] (peak² = 4). Matches the cache
    # trainer convention at examples/wan2.1_fun/train_gs_head_from_cache.py
    # so dynamic-mode wandb curves are directly comparable to the 35 dB
    # cache overfit reference. Detached — diagnostic only.
    gs_psnr = 10.0 * torch.log10(
        rgb_mse.detach().clamp_min(1e-12).reciprocal() * 4.0
    )
    result = {
        "gs_loss": gs_loss,
        "gs_rgb_loss": rgb_loss,
        "gs_rgb_mse": rgb_mse,
        "gs_psnr": gs_psnr,
        "gs_pos_loss": pos_loss,
        "gs_direct_pos_loss": direct_pos_loss,
        "gs_track_canonical_loss": track_canonical_loss,
        "gs_track_delta_loss": track_delta_loss,
        "gs_track_final_pos_loss": track_final_pos_loss,
        "gs_assoc_track_2d_loss": assoc_track_2d_loss,
        "gs_track_accel_loss": track_accel_loss,
        "gs_depth_loss": depth_loss,
        "gs_opa_tv_loss": opa_tv,
        "gs_xyz_tv_loss": xyz_tv,
    }
    # Phase-1 track diagnostics (pointmap-target, from target_track_video).
    # Useful for Objective B / surface-pointmap monitoring; NOT the overlay
    # scoreboard. See CLAUDE-gs-head.md §14.4 and the 2026-04-22 spec §1-bis.
    result.update(track_metrics)
    # Objective A diagnostics (persistent-target, from target_tracks_world).
    # THIS is the "GS 2D tracks match GT overlay" scoreboard.
    result.update(persistent_metrics)
    return result
