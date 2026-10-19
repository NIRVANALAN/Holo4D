"""Vista4D-style source / point-cloud latent stream conditioning.

This unit is opt-in and does not alter the existing dense-control branch unless
``use_vista4d_latent_streams`` is passed through the pipeline inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from holo4d.conditions.dense_target_control_unit import (
    PipelineUnit,
    WanVideoUnit_DenseTargetControl,
    _maybe_remove_source_mask_boundary_depth,
)


def _pack_vista4d_masks(mask_pair: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Pack ``(B, 2, T, H, W)`` masks to Vista4D mask-latent channels."""
    if mask_pair.ndim != 5 or int(mask_pair.shape[1]) != 2:
        raise ValueError(f"Expected mask_pair shape (B,2,T,H,W), got {tuple(mask_pair.shape)}")
    b, c, t, h, w = mask_pair.shape
    masks = F.interpolate(
        mask_pair.float(),
        size=(int(t), int(height), int(width)),
        mode="trilinear",
        align_corners=False,
    )
    masks = torch.cat(
        [torch.repeat_interleave(masks[:, :, 0:1], repeats=4, dim=2), masks[:, :, 1:]],
        dim=2,
    )
    if int(masks.shape[2]) % 4 != 0:
        raise ValueError(f"Packed temporal mask length must be divisible by 4, got {int(masks.shape[2])}")
    return rearrange(masks, "b c (f sf) (h sh) (w sw) -> b (c sf sh sw) f h w", sf=4, sh=8, sw=8).contiguous()


def _coerce_video_mask(mask, B: int, device: torch.device) -> torch.Tensor:
    mask_t = mask.to(device=device).bool()
    if mask_t.ndim == 3:
        mask_t = mask_t.unsqueeze(0).expand(B, -1, -1, -1)
    if mask_t.ndim != 4:
        raise ValueError(f"Expected mask shape (T,H,W) or (B,T,H,W), got {tuple(mask_t.shape)}")
    return mask_t


def _reproject_source_motion_mask_to_target(
    source_dynamic_mask: torch.Tensor,
    source_depth: torch.Tensor,
    source_depth_valid: torch.Tensor,
    src_cam2world: torch.Tensor,
    tgt_cam2world: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    image_h: int,
    image_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render the source dynamic mask into target view, matching Vista4D.

    Vista4D does not read a separate target segmentation. It unprojects the
    source dynamic/static masks with the source RGBD point cloud, then renders
    those labels into the target camera. This helper mirrors that contract with
    the same softsplat reprojection used by the dense branch.
    """
    from holo4d.conditions.dense_target_reprojection_builder import (
        build_dense_target_reprojection_softsplat,
    )

    source_alpha = torch.ones_like(source_depth_valid, dtype=torch.float32, device=source_depth.device)
    source_motion = source_dynamic_mask.to(device=source_depth.device).float()
    source_mask_rgb = torch.stack(
        [source_alpha, source_motion, torch.zeros_like(source_alpha)],
        dim=-1,
    ).contiguous()
    warped_source_mask_rgb, _, _ = build_dense_target_reprojection_softsplat(
        source_rgb=source_mask_rgb,
        source_depth=source_depth,
        source_depth_valid=source_depth_valid,
        src_cam2world=src_cam2world,
        tgt_cam2world=tgt_cam2world,
        intrinsics=source_intrinsics,
        image_h=image_h,
        image_w=image_w,
        target_intrinsics=target_intrinsics,
    )
    target_motion = warped_source_mask_rgb[..., 1].clamp(0.0, 1.0)
    return target_motion, target_motion > 0.5


class WanVideoUnit_Vista4DStreams(PipelineUnit):
    """Build Vista4D-style latent streams from Track4D source/reprojection data."""

    def __init__(self):
        super().__init__(
            input_params=(
                "use_vista4d_latent_streams",
                "use_gen3c_warp_latent_channels",
                "source_rgb_video",
                "source_depth_video",
                "source_depth_valid",
                "source_cam2world",
                "source_intrinsics",
                "source_intrinsics_4rc",
                "source_intrinsics_gt",
                "target_cam2world",
                "target_intrinsics",
                "height",
                "width",
                "num_frames",
                "dense_warp_mode",
                "dense_warp_motion_threshold",
                "dense_hybrid_disable_dynamic_masks",
                "tracks_world",
                "tracks_world_valid_mask",
                "tracks_world_dense_valid_mask",
                "source_dynamic_mask",
                "target_dynamic_mask",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=(
                "vista4d_source_video_latents",
                "vista4d_pointcloud_video_latents",
                "vista4d_source_mask_latents",
                "vista4d_pointcloud_mask_latents",
                "gen3c_warp_condition_latents",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        use_vista4d_latent_streams=False,
        use_gen3c_warp_latent_channels=False,
        source_rgb_video=None,
        source_depth_video=None,
        source_depth_valid=None,
        source_cam2world=None,
        source_intrinsics=None,
        source_intrinsics_4rc=None,
        source_intrinsics_gt=None,
        target_cam2world=None,
        target_intrinsics=None,
        height=None,
        width=None,
        num_frames=None,
        dense_warp_mode="hybrid_softsplat",
        dense_warp_motion_threshold=0.005,
        dense_hybrid_disable_dynamic_masks=False,
        tracks_world=None,
        tracks_world_valid_mask=None,
        tracks_world_dense_valid_mask=None,
        source_dynamic_mask=None,
        target_dynamic_mask=None,
        tiled=False,
        tile_size=None,
        tile_stride=None,
    ):
        build_vista4d_streams = bool(use_vista4d_latent_streams)
        build_gen3c_warp = bool(use_gen3c_warp_latent_channels)
        if not (build_vista4d_streams or build_gen3c_warp):
            return {}
        if source_rgb_video is None or source_depth_video is None or source_cam2world is None or target_cam2world is None:
            return {}

        device = pipe.device
        dtype = pipe.torch_dtype
        H = int(height)
        W = int(width)
        T = int(num_frames)

        tgt_c2w_t = target_cam2world if torch.is_tensor(target_cam2world) else torch.as_tensor(target_cam2world)
        B = tgt_c2w_t.shape[0] if tgt_c2w_t.ndim == 4 else 1

        coerce = WanVideoUnit_DenseTargetControl
        src_rgb = coerce._coerce_source_rgb_video(source_rgb_video, B, T, H, W, device)
        src_depth = coerce._coerce_source_depth_video(source_depth_video, B, T, H, W, device)
        src_valid = coerce._coerce_source_depth_valid(source_depth_valid, src_depth)
        src_c2w = coerce._coerce_cam2world(source_cam2world, B, T, device)
        tgt_c2w = coerce._coerce_cam2world(target_cam2world, B, T, device)

        K_unproj_in = source_intrinsics_gt if source_intrinsics_gt is not None else source_intrinsics
        if K_unproj_in is None:
            K_unproj_in = source_intrinsics_4rc if source_intrinsics_4rc is not None else target_intrinsics
        K_proj_in = source_intrinsics_gt if source_intrinsics_gt is not None else target_intrinsics
        if K_proj_in is None:
            K_proj_in = source_intrinsics
        if K_unproj_in is None or K_proj_in is None:
            return {}
        K = coerce._coerce_intrinsics(K_unproj_in, B, T, device)
        K_tgt = coerce._coerce_intrinsics(K_proj_in, B, T, device) if K_proj_in is not K_unproj_in else K

        dense_warp_mode_l = str(dense_warp_mode).lower()
        source_dynamic_mask_t = None
        target_dynamic_mask_t = None
        pointcloud_motion_t = None
        if dense_warp_mode_l == "hybrid_softsplat":
            from holo4d.conditions.dense_target_reprojection_builder import (
                build_dense_target_reprojection_hybrid,
            )
            from holo4d.conditions.soft_splat_warp import (
                compute_projected_dynamic_mask_from_tracks_world,
                compute_target_dynamic_mask_from_tracks_world,
            )

            disable_dynamic_masks = bool(dense_hybrid_disable_dynamic_masks)
            if disable_dynamic_masks:
                source_dynamic_mask_t = None
                target_dynamic_mask_t = None
            elif source_dynamic_mask is not None:
                source_dynamic_mask_t = _coerce_video_mask(source_dynamic_mask, B, device)
            else:
                source_dynamic_mask_t = compute_projected_dynamic_mask_from_tracks_world(
                    tracks_world=tracks_world,
                    cam2world=src_c2w,
                    intrinsics=K_tgt,
                    image_h=H,
                    image_w=W,
                    motion_threshold=float(dense_warp_motion_threshold),
                )
            if source_dynamic_mask_t is not None:
                source_dynamic_mask_t = source_dynamic_mask_t.to(device=device).bool()
            src_valid = _maybe_remove_source_mask_boundary_depth(
                src_valid,
                source_dynamic_mask_t,
                prob=float(getattr(pipe, "source_mask_boundary_depth_dropout_prob", 0.0)),
                kernel_size=int(getattr(pipe, "source_mask_boundary_kernel_size", 5)),
                training=bool(getattr(getattr(pipe, "scheduler", None), "training", False)),
            )
            if not disable_dynamic_masks:
                if source_dynamic_mask_t is not None:
                    pointcloud_motion_t, target_dynamic_mask_t = _reproject_source_motion_mask_to_target(
                        source_dynamic_mask=source_dynamic_mask_t,
                        source_depth=src_depth,
                        source_depth_valid=src_valid,
                        src_cam2world=src_c2w,
                        tgt_cam2world=tgt_c2w,
                        source_intrinsics=K,
                        target_intrinsics=K_tgt,
                        image_h=H,
                        image_w=W,
                    )
                elif target_dynamic_mask is not None:
                    target_dynamic_mask_t = _coerce_video_mask(target_dynamic_mask, B, device)
                    pointcloud_motion_t = target_dynamic_mask_t.to(device=device).float()
                else:
                    target_dynamic_mask_t = compute_target_dynamic_mask_from_tracks_world(
                        tracks_world=tracks_world,
                        tgt_cam2world=tgt_c2w,
                        tgt_intrinsics=K_tgt,
                        image_h=H,
                        image_w=W,
                        motion_threshold=float(dense_warp_motion_threshold),
                    )
                    pointcloud_motion_t = target_dynamic_mask_t.to(device=device).float()
            if target_dynamic_mask_t is not None:
                target_dynamic_mask_t = target_dynamic_mask_t.to(device=device).bool()
            warped_rgb, _, warp_valid = build_dense_target_reprojection_hybrid(
                source_rgb=src_rgb,
                source_depth=src_depth,
                source_depth_valid=src_valid,
                src_cam2world=src_c2w,
                tgt_cam2world=tgt_c2w,
                intrinsics=K,
                image_h=H,
                image_w=W,
                source_static_mask=None if source_dynamic_mask_t is None else ~source_dynamic_mask_t,
                target_dynamic_mask=target_dynamic_mask_t,
                dynamic_tracks_world=tracks_world,
                dynamic_tracks_valid_mask=(
                    tracks_world_dense_valid_mask
                    if tracks_world_dense_valid_mask is not None
                    else tracks_world_valid_mask
                ),
                target_intrinsics=K_tgt,
            )
        else:
            from holo4d.conditions.dense_target_reprojection_builder import (
                build_dense_target_reprojection_softsplat,
            )
            warped_rgb, _, warp_valid = build_dense_target_reprojection_softsplat(
                source_rgb=src_rgb,
                source_depth=src_depth,
                source_depth_valid=src_valid,
                src_cam2world=src_c2w,
                tgt_cam2world=tgt_c2w,
                intrinsics=K,
                image_h=H,
                image_w=W,
                target_intrinsics=K_tgt,
            )
            if source_dynamic_mask is not None:
                source_dynamic_mask_t = _coerce_video_mask(source_dynamic_mask, B, device)
                pointcloud_motion_t, target_dynamic_mask_t = _reproject_source_motion_mask_to_target(
                    source_dynamic_mask=source_dynamic_mask_t,
                    source_depth=src_depth,
                    source_depth_valid=src_valid,
                    src_cam2world=src_c2w,
                    tgt_cam2world=tgt_c2w,
                    source_intrinsics=K,
                    target_intrinsics=K_tgt,
                    image_h=H,
                    image_w=W,
                )
            elif target_dynamic_mask is not None:
                target_dynamic_mask_t = _coerce_video_mask(target_dynamic_mask, B, device)
                pointcloud_motion_t = target_dynamic_mask_t.to(device=device).float()
            if target_dynamic_mask_t is not None:
                target_dynamic_mask_t = target_dynamic_mask_t.to(device=device).bool()

        pipe.load_models_to_device(self.onload_model_names)

        def encode_rgb_video(rgb_bthwc: torch.Tensor) -> torch.Tensor:
            video = rgb_bthwc.permute(0, 4, 1, 2, 3).contiguous()
            video = (video * 2.0 - 1.0).clamp(-1.0, 1.0).to(device=device, dtype=dtype)
            return pipe.vae.encode(
                video,
                device=device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(device=device, dtype=dtype)

        # RGB-space appearance noise on the reprojected (point-cloud) and source
        # streams — Vista4D-style robustness, same scale as the source-RGB noise.
        # Training only; per the env knob TRACK4D360_VISTA4D_PC_NOISE_STD.
        import os as _os
        _pc_noise_std = float(_os.environ.get("TRACK4D360_VISTA4D_PC_NOISE_STD", "0.0"))
        _train_mode = bool(getattr(getattr(pipe, "dit", None), "training", False))

        def _maybe_noise_rgb(rgb):
            if not _train_mode or _pc_noise_std <= 0.0:
                return rgb
            return (rgb + torch.randn_like(rgb) * _pc_noise_std).clamp(0.0, 1.0)

        pointcloud_video_latents = encode_rgb_video(_maybe_noise_rgb(warped_rgb.clamp(0.0, 1.0)))

        source_alpha = torch.ones_like(src_valid, dtype=torch.float32, device=device)
        if source_dynamic_mask_t is None:
            source_motion = torch.zeros_like(src_valid, dtype=torch.float32, device=device)
        else:
            source_motion = source_dynamic_mask_t.to(device=device).float()
        pointcloud_alpha = warp_valid.to(device=device).float()
        if pointcloud_motion_t is None:
            pointcloud_motion = torch.zeros_like(pointcloud_alpha)
        else:
            pointcloud_motion = pointcloud_motion_t.to(device=device).float()

        output = {}
        if build_vista4d_streams:
            source_video_latents = encode_rgb_video(_maybe_noise_rgb(src_rgb))
            source_masks = torch.stack([source_alpha, source_motion], dim=1)
            pointcloud_masks = torch.stack([pointcloud_alpha, pointcloud_motion], dim=1)
            source_mask_latents = _pack_vista4d_masks(source_masks, H, W).to(device=device, dtype=dtype)
            pointcloud_mask_latents = _pack_vista4d_masks(pointcloud_masks, H, W).to(device=device, dtype=dtype)
            output.update({
                "vista4d_source_video_latents": source_video_latents,
                "vista4d_pointcloud_video_latents": pointcloud_video_latents,
                "vista4d_source_mask_latents": source_mask_latents,
                "vista4d_pointcloud_mask_latents": pointcloud_mask_latents,
            })

        if build_gen3c_warp:
            pointcloud_mask_video = pointcloud_alpha.unsqueeze(-1).expand(-1, -1, -1, -1, 3).contiguous()
            pointcloud_mask_latents_vae = encode_rgb_video(pointcloud_mask_video)
            output["gen3c_warp_condition_latents"] = torch.cat(
                [pointcloud_video_latents, pointcloud_mask_latents_vae],
                dim=1,
            ).contiguous()

        return output


__all__ = ["WanVideoUnit_Vista4DStreams"]
