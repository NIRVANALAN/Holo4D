"""v4 track-only loss: per-frame 3D delta supervision decoupled from rendering.

Architecture (see doc/track4d-360/2026-04-26-v4-per-frame-track-head.md):

  - Render path: per-frame independent pixel-aligned Gaussians from
    ``LyraPixelAligned3DGS`` (existing static head with exp depth).
    Photometric / depth / opacity losses fire there, computed by
    ``holo4d.models.gs_loss.compute_gs_splatting_loss``.
  - Track path (this module): per-pixel-per-frame 3D delta from a parallel
    ``track_delta_proj`` head (``LyraPixelAligned3DGS.predict_track_delta_xyz``).
    Supervised by ``target_tracks_world`` only — never enters rendering.

Loss family mirrors the Kubric cache-v3 dynamic-mode recipe (canonical /
delta / final_pos / assoc_2d / accel) but computes everything from
(canonical_xyz, per_frame_delta_xyz) where:

  - canonical_xyz: gaussian_params[:, 0, :, :, :, 0:3]  (frame-0 of static head)
  - per_frame_delta_xyz: track_delta_xyz                (B, T_video, npp, H, W, 3)

When ``target_tracks_world`` is None (e.g. MVS-Synth, DL3DV) every loss
returns 0; the head still receives gradient through the photometric path
via the shared backbone, so v4 degrades cleanly to v3 on track-less data.
"""

from __future__ import annotations

from typing import Optional

import torch

from .gs_loss_dynamic import (
    _build_frame0_association,
    _masked_l1_vec,
    _persistent_tracks_to_grid,
    _resize_dense_track_targets,
    _compute_assoc_persistent_track_loss,
    _compute_accel_loss,
)


def _zero(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_tensor(0.0)


def _associated_canonical(canonical_params: torch.Tensor, assoc: torch.Tensor) -> torch.Tensor:
    """Pick the per-pixel canonical xyz of the associated Gaussian.

    canonical_params: (B, npp, H, W, D) — frame-0 static head output.
    assoc:            (B, H, W) long in [0, npp).
    Returns:          (B, H, W, 3).
    """
    B, npp, H, W, _D = canonical_params.shape
    canonical_xyz = canonical_params[..., 0:3]  # (B, npp, H, W, 3)
    idx = assoc.unsqueeze(1).unsqueeze(-1).expand(B, 1, H, W, 3)
    return torch.gather(canonical_xyz, dim=1, index=idx).squeeze(1)  # (B, H, W, 3)


def _associated_delta(track_delta_xyz: torch.Tensor, assoc: torch.Tensor) -> torch.Tensor:
    """Pick the per-pixel-per-frame delta_xyz of the associated Gaussian.

    track_delta_xyz: (B, T_video, npp, H, W, 3).
    assoc:           (B, H, W).
    Returns:         (B, T_video, H, W, 3).
    """
    B, T, npp, H, W, _ = track_delta_xyz.shape
    idx = (
        assoc.unsqueeze(1).unsqueeze(2).unsqueeze(-1)  # (B, 1, 1, H, W, 1)
             .expand(B, T, 1, H, W, 3)
    )
    return torch.gather(track_delta_xyz, dim=2, index=idx).squeeze(2)


def build_joint_gaussian_params(
    gaussian_params: torch.Tensor,
    track_delta_xyz: torch.Tensor,
) -> torch.Tensor:
    """Concatenate per-frame Gaussians with canonical-warped Gaussians along npp axis.

    Per-frame Gaussians (set A): ``gaussian_params[:, t]`` — current v4 behavior.
    Canonical warped (set B): canonical = ``gaussian_params[:, 0]`` with xyz
    replaced by ``canonical_xyz + Δxyz[t]``. Other attrs (color / scale /
    opacity / rotation / offset) inherited from t=0. At t=0 the two sets are
    near-identical (Δxyz ≈ 0 from zero-init), so rendering both is wasted; the
    caller may special-case t=0 to skip set B if compute matters.

    The 3DGS renderer alpha-blends the union via z-buffer in one call. Both
    sets receive photometric gradient → `track_delta_proj` gets render gradient
    (not only track-GT gradient), and `gs_proj`'s frame-0 attrs get gradient
    from both their own t=0 render AND every t>0 warped-canonical render.

    Shapes:
      gaussian_params:  (B, T, npp, H, W, D)
      track_delta_xyz:  (B, T, npp, H, W, 3)
      returns:          (B, T, 2*npp, H, W, D)
    """
    B, T, npp, H, W, D = gaussian_params.shape
    if D < 3:
        raise ValueError(f"gaussian_params last dim must be >=3 (xyz), got D={D}")
    if track_delta_xyz.shape[:5] != (B, T, npp, H, W) or track_delta_xyz.shape[5] != 3:
        raise ValueError(
            f"track_delta_xyz shape {tuple(track_delta_xyz.shape)} must equal "
            f"({B}, {T}, {npp}, {H}, {W}, 3)"
        )

    canonical = gaussian_params[:, 0]                    # (B, npp, H, W, D)
    canonical_xyz = canonical[..., 0:3]                  # (B, npp, H, W, 3)
    canonical_other = canonical[..., 3:]                 # (B, npp, H, W, D-3)
    warped_xyz = canonical_xyz.unsqueeze(1) + track_delta_xyz   # (B, T, npp, H, W, 3)
    warped_other = canonical_other.unsqueeze(1).expand(B, T, npp, H, W, D - 3)
    warped_canonical = torch.cat([warped_xyz, warped_other], dim=-1)   # (B, T, npp, H, W, D)
    return torch.cat([gaussian_params, warped_canonical], dim=2)       # (B, T, 2*npp, H, W, D)


def compute_v4_track_loss(
    *,
    canonical_params: torch.Tensor,           # (B, npp, H, W, D) — frame-0 static gaussian_params
    track_delta_xyz: torch.Tensor,            # (B, T_video, npp, H, W, 3) — per-frame delta from frame 0
    target_cameras: torch.Tensor,             # (B, T_video, 11) FlashWorld cams
    image_height: int,
    image_width: int,
    target_tracks_world: Optional[torch.Tensor] = None,           # (B, N=H*W, T, 3) row-major
    target_tracks_world_valid_mask: Optional[torch.Tensor] = None,
    target_track_video: Optional[torch.Tensor] = None,            # dense pointmap fallback (B, T, H, W, 3)
    target_track_valid_mask: Optional[torch.Tensor] = None,
    track_canonical_loss_weight: float = 1.0,
    track_delta_loss_weight: float = 1.0,
    track_final_pos_loss_weight: float = 1.0,
    assoc_track_2d_loss_weight: float = 5.0,
    track_accel_loss_weight: float = 0.5,
    track_loss_quantile: float = 0.98,
    canonical_detach: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute v4 track losses on (canonical_xyz, per_frame_delta_xyz).

    ``canonical_detach`` (default False): when True, detaches
    ``gaussian_params[:, 0]`` before computing track losses so the render
    head's gs_proj receives no track-GT gradient (matches doc claim "track
    head never enters rendering" literally). Default False preserves cache-v3
    behavior: track GT in world space directly anchors gs_proj's frame-0 xyz,
    which photometric cannot do alone (RGB renders are depth-ambiguous on
    monocular dynamic content). If the world-unit canonical term swamps RGB
    MSE early on, prefer halving ``track_canonical_loss_weight`` over flipping
    this to True. Diagnose with the [v4 mag] log on the first batch.
    """
    if canonical_detach:
        canonical_params = canonical_params.detach()

    out: dict[str, torch.Tensor] = {
        "gs_track_canonical_loss": _zero(canonical_params),
        "gs_track_delta_loss": _zero(canonical_params),
        "gs_track_final_pos_loss": _zero(canonical_params),
        "gs_assoc_track_2d_loss": _zero(canonical_params),
        "gs_track_accel_loss": _zero(canonical_params),
    }

    # No GT 3D tracks → nothing to supervise. Photometric path drives all
    # gradients via the shared backbone.
    #
    # DDP nuance: we still need `track_delta_xyz` to reach the loss graph,
    # otherwise `track_delta_proj.{weight,bias}` will be flagged as unused
    # parameters by `_rebuild_buckets()` on track-less batches and crash with
    # "Expected to have finished reduction in the prior iteration before
    # starting a new one." Tying with a `.sum() * 0.0` term keeps the
    # autograd graph intact while contributing 0 to the optimizer step.
    has_persistent = target_tracks_world is not None
    has_pointmap = target_track_video is not None and target_track_valid_mask is not None
    if not has_persistent and not has_pointmap:
        out["gs_track_total_loss"] = track_delta_xyz.sum() * 0.0
        return out

    B, npp, H, W, _D = canonical_params.shape
    T_video = int(track_delta_xyz.shape[1])

    # ---- Reduce GT tracks to the GS-head spatial grid (B, T, H, W, 3) -------
    # Two routing paths:
    #   1. persistent_tracks: target_tracks_world is the SoM-style row-major
    #      per-pixel 3D position at each frame. Reshape to grid via
    #      `_persistent_tracks_to_grid`. Requires N == H * W.
    #   2. pointmap: target_track_video is already (B, T, H_full, W_full, 3);
    #      bilinear-resize to (H, W) via `_resize_dense_track_targets`.
    if has_persistent:
        try:
            gt_track_ds, valid_ds = _persistent_tracks_to_grid(
                target_tracks_world.to(canonical_params.device, dtype=torch.float32),
                target_tracks_world_valid_mask,
                H,
                W,
            )
        except (ValueError, RuntimeError) as exc:
            # Persistent grid reshape failed (likely N != H*W). Fall back to
            # pointmap if available; otherwise zero out with a diagnostic flag.
            if has_pointmap:
                gt_track_ds, valid_ds = _resize_dense_track_targets(
                    target_track_video.to(canonical_params.device, dtype=torch.float32),
                    target_track_valid_mask.to(canonical_params.device, dtype=torch.float32),
                    H,
                    W,
                )
            else:
                out["gs_track_persistent_grid_failed"] = canonical_params.new_tensor(1.0)
                # Same DDP unused-param fix: tie track_delta_xyz into the
                # graph even when no GT path could be used.
                out["gs_track_total_loss"] = track_delta_xyz.sum() * 0.0
                return out
    else:
        gt_track_ds, valid_ds = _resize_dense_track_targets(
            target_track_video.to(canonical_params.device, dtype=torch.float32),
            target_track_valid_mask.to(canonical_params.device, dtype=torch.float32),
            H,
            W,
        )

    # Truncate to common timeline.
    T_gt = int(gt_track_ds.shape[1])
    T_use = min(T_gt, T_video)
    gt_track_ds = gt_track_ds[:, :T_use]
    valid_ds = valid_ds[:, :T_use]

    # ---- Frame-0 association (per-pixel: which Gaussian among npp matches GT?)
    gt_track_t0 = gt_track_ds[:, 0]            # (B, H, W, 3)
    valid_t0 = valid_ds[:, 0]                  # (B, H, W) bool
    assoc = _build_frame0_association(canonical_params, gt_track_t0, valid_t0)

    canonical_assoc = _associated_canonical(canonical_params, assoc)        # (B, H, W, 3)
    delta_assoc = _associated_delta(track_delta_xyz[:, :T_use], assoc)      # (B, T_use, H, W, 3)

    # ---- Loss 1, 2, 3: canonical / delta / final_pos -----------------------
    if track_canonical_loss_weight > 0.0:
        out["gs_track_canonical_loss"] = _masked_l1_vec(
            canonical_assoc, gt_track_t0, valid_t0.float(),
        )

    if track_delta_loss_weight > 0.0 and T_use > 1:
        # Predicted delta_assoc IS the delta from frame-0 by construction.
        # GT delta = tracks[t] - tracks[0] for t >= 1.
        pred_delta = delta_assoc[:, 1:]                    # (B, T-1, H, W, 3)
        gt_delta = gt_track_ds[:, 1:] - gt_track_ds[:, :1]
        valid_future = valid_ds[:, 1:] & valid_t0.unsqueeze(1)
        out["gs_track_delta_loss"] = _masked_l1_vec(
            pred_delta, gt_delta, valid_future.float(),
        )

    pred_traj_world = None
    if track_final_pos_loss_weight > 0.0 or assoc_track_2d_loss_weight > 0.0:
        # final = canonical + delta vs GT. canonical broadcasts across time.
        pred_traj_world = canonical_assoc.unsqueeze(1) + delta_assoc        # (B, T, H, W, 3)

    if track_final_pos_loss_weight > 0.0:
        out["gs_track_final_pos_loss"] = _masked_l1_vec(
            pred_traj_world, gt_track_ds, valid_ds.float(),
        )

    # ---- Loss 4: assoc_track_2d (project predicted trajectory → 2D pixel L1)
    if assoc_track_2d_loss_weight > 0.0 and pred_traj_world is not None:
        out["gs_assoc_track_2d_loss"] = _compute_assoc_persistent_track_loss(
            pred_traj_world,
            gt_track_ds,
            valid_ds,
            target_cameras[:, :T_use],
            image_height,
            image_width,
            track_loss_quantile,
        )

    # ---- Loss 5: trajectory acceleration smoothness ------------------------
    if track_accel_loss_weight > 0.0 and T_use >= 3:
        # Acceleration on absolute position == acceleration on delta
        # (canonical is constant in t). Reuse delta_assoc directly.
        out["gs_track_accel_loss"] = _compute_accel_loss(
            delta_assoc, valid_ds,
        )

    out["gs_track_total_loss"] = (
        track_canonical_loss_weight * out["gs_track_canonical_loss"]
        + track_delta_loss_weight * out["gs_track_delta_loss"]
        + track_final_pos_loss_weight * out["gs_track_final_pos_loss"]
        + assoc_track_2d_loss_weight * out["gs_assoc_track_2d_loss"]
        + track_accel_loss_weight * out["gs_track_accel_loss"]
        # DDP unused-param guard: even if all five terms above are zero
        # (all weights set to 0, or no sub-conditions fired), keep
        # track_delta_xyz in the autograd graph so DDP doesn't flag
        # track_delta_proj.{weight,bias} as unused on this rank.
        # Magnitude 0, so no contribution to the optimizer step.
        + track_delta_xyz.sum() * 0.0
    )

    # First-batch magnitude breadcrumb. Catches the failure mode where one
    # term (typically canonical, in world units) dwarfs RGB MSE and freezes
    # photometric learning. Prints once per process; subsequent calls no-op.
    if not getattr(compute_v4_track_loss, "_logged_mag", False):
        try:
            import os as _os
            if int(_os.environ.get("LOCAL_RANK", "0")) == 0:
                print(
                    f"[v4 mag] canonical={float(out['gs_track_canonical_loss']):.4f} "
                    f"delta={float(out['gs_track_delta_loss']):.4f} "
                    f"final_pos={float(out['gs_track_final_pos_loss']):.4f} "
                    f"assoc2d={float(out['gs_assoc_track_2d_loss']):.4f} "
                    f"accel={float(out['gs_track_accel_loss']):.4f} "
                    f"weighted_total={float(out['gs_track_total_loss']):.4f} "
                    f"detach={canonical_detach} npp={int(canonical_params.shape[1])} "
                    f"H={int(canonical_params.shape[2])} W={int(canonical_params.shape[3])}"
                )
            compute_v4_track_loss._logged_mag = True
        except Exception:
            compute_v4_track_loss._logged_mag = True
    return out
