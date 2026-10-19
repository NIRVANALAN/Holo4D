"""
warped_appearance_unit.py — Lyra-2-inspired warped RGB appearance conditioning.

Pipeline unit that:
  1. Soft-splat warps source RGB into the target viewpoint
  2. VAE-encodes the warped RGB into latent space
  3. Outputs warped_rgb_latent for shallow fusion before DiT blocks

Only active when warped_condition_mode == "latent_fuse".
When mode == "off" (default), this unit is a no-op and existing behavior is unchanged.
"""

from __future__ import annotations

import torch
from holo4d_runtime.diffusion.base_pipeline import PipelineUnit

from .soft_splat_warp import (
    build_soft_splat_warped_rgb,
    build_hybrid_warped_rgb,
    compute_static_mask_from_tracks_world,
)
from .dense_target_control_unit import WanVideoUnit_DenseTargetControl


class WanVideoUnit_WarpedAppearance(PipelineUnit):
    """
    Warped RGB appearance conditioning unit (Lyra-2-inspired).

    Produces a VAE-encoded warped source appearance latent that can be
    fused with the DiT input via a shallow learned operator.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                # Source inputs (reused from dense branch)
                "source_rgb_video",
                "source_depth_video",
                "source_depth_valid",
                "source_cam2world",
                "source_intrinsics",
                "source_intrinsics_4rc",
                "source_intrinsics_gt",
                "target_cam2world",
                "target_intrinsics",
                # Per-pixel 3D tracks (Kubric_NVS / Syn4D) — drives the
                # static-mask filter when multi-frame mode is on. Optional;
                # absent for static-scene datasets (RE10K / DL3DV / etc.)
                # which are pure static and need no filter.
                "tracks_world",
                # Shape
                "height",
                "width",
                "num_frames",
                # Mode gate
                "warped_condition_mode",
                # Multi-frame mode (Lyra-2 forward_warp_multiframes style):
                # when True, fuse all T source frames into each target view
                # for the static region and fall back to per-frame for
                # dynamic. When False, classic per-frame s=t→t splat.
                "multi_frame_warped_appearance",
                "multi_frame_motion_threshold",
                # Tiling (for VAE encode)
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=(
                "warped_rgb_latent",
                "warped_appearance_confidence",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        source_rgb_video=None,
        source_depth_video=None,
        source_depth_valid=None,
        source_cam2world=None,
        source_intrinsics=None,
        source_intrinsics_4rc=None,
        source_intrinsics_gt=None,
        target_cam2world=None,
        target_intrinsics=None,
        tracks_world=None,
        height=None,
        width=None,
        num_frames=None,
        warped_condition_mode="off",
        multi_frame_warped_appearance=False,
        multi_frame_motion_threshold=0.01,
        tiled=False,
        tile_size=None,
        tile_stride=None,
    ):
        # Mode gate: only run in latent_fuse mode
        if warped_condition_mode is None or warped_condition_mode == "off":
            return {}
        if warped_condition_mode != "latent_fuse":
            raise ValueError(
                f"warped_condition_mode must be 'off' or 'latent_fuse', got {warped_condition_mode!r}"
            )

        # Need source+target cameras
        if target_cam2world is None or source_rgb_video is None:
            return {}
        if source_depth_video is None or source_cam2world is None:
            return {}

        device = pipe.device
        H = int(height)
        W = int(width)
        T = int(num_frames)

        # Determine batch size
        tgt_c2w_t = target_cam2world if torch.is_tensor(target_cam2world) else torch.as_tensor(target_cam2world)
        B = tgt_c2w_t.shape[0] if tgt_c2w_t.ndim == 4 else 1

        # Reuse coercion helpers from the dense target control unit
        _coerce = WanVideoUnit_DenseTargetControl
        src_rgb = _coerce._coerce_source_rgb_video(source_rgb_video, B, T, H, W, device)
        src_depth = _coerce._coerce_source_depth_video(source_depth_video, B, T, H, W, device)
        src_valid = _coerce._coerce_source_depth_valid(source_depth_valid, src_depth)
        src_c2w = _coerce._coerce_cam2world(source_cam2world, B, T, device)
        tgt_c2w = _coerce._coerce_cam2world(target_cam2world, B, T, device)

        # Source RGB/depth are sampled on the cropped GT image grid. Use GT K
        # for the appearance warp; 4RC-predicted K can differ enough to break
        # identity-like source->target cases.
        K_unproj_in = source_intrinsics_gt if source_intrinsics_gt is not None else source_intrinsics
        if K_unproj_in is None:
            K_unproj_in = source_intrinsics_4rc if source_intrinsics_4rc is not None else target_intrinsics
        K_proj_in = source_intrinsics_gt if source_intrinsics_gt is not None else target_intrinsics
        if K_proj_in is None:
            K_proj_in = source_intrinsics
        if K_unproj_in is None or K_proj_in is None:
            return {}
        K = _coerce._coerce_intrinsics(K_unproj_in, B, T, device)
        K_tgt = _coerce._coerce_intrinsics(K_proj_in, B, T, device) \
            if K_proj_in is not K_unproj_in else K

        # Step 1: Soft-splat warp source RGB to target viewpoint.
        # `multi_frame_warped_appearance` toggles between:
        #   - per-frame (default, classic Lyra-2 Path A): for each t, warp
        #     source[t] -> target[t]; the only contributor is the s=t source.
        #   - hybrid: at static target pixels, fuse all T source frames
        #     (Lyra-2 forward_warp_multiframes); at dynamic target pixels,
        #     fall back to the per-frame warp. Static / dynamic separation
        #     comes from the per-pixel motion range of `tracks_world`
        #     (Kubric_NVS / Syn4D); when tracks are absent the dataset is
        #     treated as pure static and the multi-frame fill covers
        #     everywhere.
        if bool(multi_frame_warped_appearance):
            static_mask = compute_static_mask_from_tracks_world(
                tracks_world=tracks_world,
                image_h=H,
                image_w=W,
                motion_threshold=float(multi_frame_motion_threshold),
            )
            warped_rgb, warp_valid, warp_weights = build_hybrid_warped_rgb(
                source_rgb=src_rgb,
                source_depth=src_depth,
                source_depth_valid=src_valid,
                src_cam2world=src_c2w,
                tgt_cam2world=tgt_c2w,
                intrinsics=K,
                image_h=H,
                image_w=W,
                target_intrinsics=K_tgt,
                static_mask=static_mask,
            )
        else:
            warped_rgb, warp_valid, warp_weights = build_soft_splat_warped_rgb(
                source_rgb=src_rgb,
                source_depth=src_depth,
                source_depth_valid=src_valid,
                src_cam2world=src_c2w,
                tgt_cam2world=tgt_c2w,
                intrinsics=K,  # already (B, T, 3, 3) after coercion
                image_h=H,
                image_w=W,
                target_intrinsics=K_tgt,
            )
        # warped_rgb: (B, T, 3, H, W) float32 [0, 1]
        # warp_valid: (B, T, 1, H, W) float32
        # warp_weights: (B, T, 1, H, W) float32 — confidence proxy

        # Step 2: VAE-encode warped RGB into latent space
        # Permute to (B, 3, T, H, W) for VAE and scale [0,1] → [-1,1]
        warped_rgb_bctHW = warped_rgb.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, T, H, W)
        warped_rgb_bctHW = warped_rgb_bctHW * 2.0 - 1.0

        pipe.load_models_to_device(self.onload_model_names)
        with torch.no_grad():
            warped_rgb_latent = pipe.vae.encode(
                warped_rgb_bctHW.to(device=device, dtype=pipe.torch_dtype),
                device=device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        # warped_rgb_latent: (B, latent_ch, T_lat, H_lat, W_lat)

        # Step 3: Build confidence at latent resolution from warp_weights.
        # Match the ACTUAL latent shape from VAE output rather than computing
        # it ourselves — avoids hard-coding VAE compression factors.
        _, _, latent_t, latent_h, latent_w = warped_rgb_latent.shape

        # Pool warp_weights: (B, T, 1, H, W) → (B, 1, latent_t, latent_h, latent_w)
        ww = warp_weights.permute(0, 2, 1, 3, 4).contiguous()  # (B, 1, T, H, W)
        # Use adaptive_avg_pool3d to downsample temporal+spatial in one shot
        confidence = torch.nn.functional.adaptive_avg_pool3d(
            ww.float(), (latent_t, latent_h, latent_w),
        )  # (B, 1, latent_t, latent_h, latent_w)

        # Normalize confidence to [0, 1] range
        c_max = confidence.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
        confidence = confidence / c_max

        return {
            "warped_rgb_latent": warped_rgb_latent,
            "warped_appearance_confidence": confidence,
        }
