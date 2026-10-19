from .base_pipeline import BasePipeline
import os
import random
import torch
from torch.utils.checkpoint import checkpoint
from ..utils.geometry import compute_signed_log_quantiles, denormalize_track_signed_log
from holo4d.ar.rolling_forcing import (
    allocate_kv_cache,
    build_latent_windows,
    resolve_ar_timesteps,
    select_window_timestep,
    slice_latent_frames,
    slice_tokens_by_latent_frames,
)
from holo4d.ar.diffusion_forcing import (
    expand_latent_timesteps_to_token_timesteps,
    flow_add_noise_per_frame,
    flow_training_target_per_frame,
    parse_fixed_timestep_values,
    sample_random_independent_timesteps,
)


# ---------------------------------------------------------------------------
# GS replay cache (overfit accelerator)
# ---------------------------------------------------------------------------
# When `gs_config["replay_cache_size"] > 0`, the FlowMatchV2VLoss path snapshots
# the post-DiT GS-loss inputs (gs_features_target, latents_pred_target, GT
# rgb/depth/cameras/tracks) on the first N steps and stashes them on CPU.
# Once the cache is full, subsequent steps SKIP the DiT forward entirely and
# replay a random cached entry through the GS loss, ~halving step time for
# GS-only overfit training. Only valid when `skip_diffusion_loss=True` (DiT
# frozen): the replay path produces no gradient w.r.t. the DiT.
_GS_REPLAY_CACHE: list[dict] = []
_GS_REPLAY_LOG_DONE = False


def _gs_cache_clear() -> None:
    """Drop any cached entries (test/debug only)."""
    _GS_REPLAY_CACHE.clear()
    global _GS_REPLAY_LOG_DONE
    _GS_REPLAY_LOG_DONE = False


def _gs_cache_snapshot(snap: dict, max_size: int) -> bool:
    """Detach + move-to-CPU and append, up to max_size. Returns True if added."""
    if len(_GS_REPLAY_CACHE) >= max_size:
        return False
    cpu_snap: dict = {}
    for k, v in snap.items():
        if torch.is_tensor(v):
            cpu_snap[k] = v.detach().to(device="cpu", copy=True)
        else:
            cpu_snap[k] = v
    _GS_REPLAY_CACHE.append(cpu_snap)
    return True


def _gs_cache_pick_random(device: torch.device) -> dict:
    """Pick a random cached entry, move tensors back to `device`."""
    entry = random.choice(_GS_REPLAY_CACHE)
    out: dict = {}
    for k, v in entry.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _replay_gs_only_loss(pipe, gs_config: dict):
    """Replay path: pull a random cached GS-input snapshot and run the GS
    splatting loss directly. No DiT forward, no diffusion loss. Mirrors the
    extra_metrics shape returned by the live FlowMatchV2VLoss path so the
    accelerate trainer's logging stays identical."""
    global _GS_REPLAY_LOG_DONE
    if not _GS_REPLAY_LOG_DONE:
        print(f"[gs-replay] cache filled ({len(_GS_REPLAY_CACHE)} entries); "
              f"DiT forward will be skipped from this step onward.")
        _GS_REPLAY_LOG_DONE = True

    device = pipe.device
    snap = _gs_cache_pick_random(device)

    latents_pred_target = snap["latents_pred_target"]
    target_cameras = snap.get("target_cameras")
    gt_target_rgb = snap.get("gt_target_rgb")
    gt_target_depth = snap.get("gt_target_depth")
    depth_valid_mask = snap.get("depth_valid_mask")

    # Re-run gs_feature_head on the cached pre-head input so its
    # parameters see real gradient on every replay step (instead of
    # caching post-head features and skipping the head — which
    # would freeze gs_feature_head and trip DDP unused-param checks).
    # Falls back to the cached gs_features_target only when the
    # snapshot doesn't have the pre-head input (legacy / non-track-aware
    # caches).
    gs_pre_head_input = snap.get("gs_pre_head_input")
    gs_pre_head_t = snap.get("gs_pre_head_t")
    gs_unpatchify_shape = snap.get("gs_unpatchify_shape")
    _v2v_concat_mode = snap.get("v2v_concat_mode", "width")
    _split = snap.get("split", None)
    if (
        gs_pre_head_input is not None
        and gs_pre_head_t is not None
        and gs_unpatchify_shape is not None
        and getattr(pipe.dit, "gs_feature_head", None) is not None
    ):
        gs_features = pipe.dit.gs_feature_head(gs_pre_head_input, gs_pre_head_t)
        gs_features = pipe.dit.unpatchify(gs_features, gs_unpatchify_shape)
        if _v2v_concat_mode == "width" and _split is not None:
            gs_features_target = gs_features[:, :, :, :, int(_split):]
        elif _v2v_concat_mode == "frame" and _split is not None:
            gs_features_target = gs_features[:, :, int(_split):]
        else:
            gs_features_target = gs_features
        if _as_bool(gs_config.get("zero_features", False)):
            gs_features_target = torch.zeros_like(gs_features_target)
    else:
        # Legacy cache without pre-head input — fall back to cached
        # post-head features. gs_feature_head won't see gradient on
        # these snapshots; only relevant if the cache was filled by
        # a pre-fix run.
        gs_features_target = snap["gs_features_target"]

    _gs_detach_latents = bool(gs_config.get("detach_predicted_latents", True))
    _dynamic_mode = str(gs_config.get("dynamic_mode", "static")).lower()

    if (
        _dynamic_mode == "static_with_track"
        and getattr(pipe.gs_decoder, "gs_head", None) is not None
        and getattr(pipe.gs_decoder.gs_head, "track_delta_proj", None) is not None
    ):
        from holo4d.models.gs_loss import compute_gs_splatting_loss
        from holo4d.models.gs_loss_v4_track import compute_v4_track_loss

        _z_in = (
            latents_pred_target.detach()
            if _gs_detach_latents else latents_pred_target
        )
        gaussian_params_v4, track_delta_xyz = pipe.gs_decoder(
            gs_features_target,
            _z_in,
            target_cameras,
            return_track_delta=True,
        )
        # v4 joint render: at every t, also render canonical warped by Δxyz[t]
        # alongside the per-frame Gaussians. Track-head's track_delta_proj then
        # receives photometric gradient, not just track-GT gradient.
        # See doc/track4d-360/2026-04-26-v4-per-frame-track-head.md (joint-render).
        _joint_render = bool(gs_config.get("track_joint_render", False))
        if _joint_render:
            from holo4d.models.gs_loss_v4_track import build_joint_gaussian_params
            gaussian_params_for_render = build_joint_gaussian_params(
                gaussian_params_v4, track_delta_xyz,
            )
        else:
            gaussian_params_for_render = gaussian_params_v4
        gs_losses_photo = compute_gs_splatting_loss(
            gs_decoder=pipe.gs_decoder,
            gs_features_target=gs_features_target,
            latents_pred_target=latents_pred_target,
            target_cameras=target_cameras,
            gt_target_rgb=gt_target_rgb,
            gt_target_depth=gt_target_depth,
            depth_valid_mask=depth_valid_mask,
            image_height=int(gs_config.get("image_height", 288)),
            image_width=int(gs_config.get("image_width", 512)),
            mse_loss_weight=float(gs_config.get("mse_loss_weight", gs_config.get("rgb_loss_weight", 1.0))),
            lpips_loss_weight=float(gs_config.get("lpips_loss_weight", 0.5)),
            depth_loss_weight=float(gs_config.get("depth_loss_weight", 0.0)),
            abs_depth_loss_weight=float(gs_config.get("abs_depth_loss_weight", 0.0)),
            opacity_loss_weight=float(gs_config.get("opacity_loss_weight", 0.1)),
            detach_predicted_latents=_gs_detach_latents,
            gaussian_params=gaussian_params_for_render,
            opacity_prune_keep_frac=float(gs_config.get("opacity_prune_keep_frac", 1.0)),
        )
        gs_losses_track = compute_v4_track_loss(
            canonical_params=gaussian_params_v4[:, 0],
            track_delta_xyz=track_delta_xyz,
            target_cameras=target_cameras,
            image_height=int(gs_config.get("image_height", 288)),
            image_width=int(gs_config.get("image_width", 512)),
            target_tracks_world=snap.get("target_tracks_world"),
            target_tracks_world_valid_mask=snap.get("target_tracks_world_valid_mask"),
            target_track_video=snap.get("target_track_video"),
            target_track_valid_mask=snap.get("target_track_valid_mask"),
            track_canonical_loss_weight=float(gs_config.get("track_canonical_loss_weight", 1.0)),
            track_delta_loss_weight=float(gs_config.get("track_delta_loss_weight", 1.0)),
            track_final_pos_loss_weight=float(gs_config.get("track_final_pos_loss_weight", 1.0)),
            assoc_track_2d_loss_weight=float(gs_config.get("assoc_track_2d_loss_weight", 5.0)),
            track_accel_loss_weight=float(gs_config.get("track_accel_loss_weight", 0.5)),
            track_loss_quantile=float(gs_config.get("track_loss_quantile", 0.98)),
            canonical_detach=bool(gs_config.get("track_canonical_detach", True)),
        )
        gs_losses = {**gs_losses_photo, **gs_losses_track}
        gs_losses["gs_loss"] = (
            gs_losses_photo["gs_loss"] + gs_losses_track["gs_track_total_loss"]
        )
    elif _dynamic_mode == "deltas" and getattr(pipe.gs_decoder, "dynamic", None) is not None:
        from holo4d.models.gs_loss_dynamic import compute_gs_splatting_loss_dynamic
        gs_losses = compute_gs_splatting_loss_dynamic(
            gs_decoder_dynamic=pipe.gs_decoder.dynamic,
            gs_features_target=gs_features_target,
            latents_pred_target=latents_pred_target,
            target_cameras=target_cameras,
            gt_target_rgb=gt_target_rgb,
            target_track_video=snap.get("target_track_video"),
            target_track_valid_mask=snap.get("target_track_valid_mask"),
            target_tracks_world=snap.get("target_tracks_world"),
            target_tracks_world_valid_mask=snap.get("target_tracks_world_valid_mask"),
            target_tracks_query_xy=snap.get("target_tracks_query_xy"),
            target_depth_video=snap.get("target_depth_video"),
            target_depth_valid_mask=snap.get("target_depth_valid_mask"),
            image_height=int(gs_config.get("image_height", 288)),
            image_width=int(gs_config.get("image_width", 512)),
            rgb_loss_weight=float(gs_config.get("mse_loss_weight", gs_config.get("rgb_loss_weight", 1.0))),
            pos_loss_weight=float(gs_config.get("pos_loss_weight", 1.0)),
            depth_loss_weight=float(gs_config.get("dyn_depth_loss_weight", 0.5)),
            opa_tv_weight=float(gs_config.get("opa_tv_weight", 0.01)),
            xyz_tv_weight=float(gs_config.get("xyz_tv_weight", 0.01)),
            gs_track_supervision_mode=str(gs_config.get("gs_track_supervision_mode", "pointmap")),
            track_canonical_loss_weight=float(gs_config.get("track_canonical_loss_weight", 0.0)),
            track_delta_loss_weight=float(gs_config.get("track_delta_loss_weight", 0.0)),
            track_final_pos_loss_weight=float(gs_config.get("track_final_pos_loss_weight", 0.0)),
            assoc_track_2d_loss_weight=float(gs_config.get("assoc_track_2d_loss_weight", 0.0)),
            track_accel_loss_weight=float(gs_config.get("track_accel_loss_weight", 0.0)),
            track_loss_quantile=float(gs_config.get("track_loss_quantile", 0.98)),
            predict_rotation=bool(gs_config.get("predict_rotation", False)),
            detach_predicted_latents=_gs_detach_latents,
        )
    else:
        from holo4d.models.gs_loss import compute_gs_splatting_loss
        gs_losses = compute_gs_splatting_loss(
            gs_decoder=pipe.gs_decoder,
            gs_features_target=gs_features_target,
            latents_pred_target=latents_pred_target,
            target_cameras=target_cameras,
            gt_target_rgb=gt_target_rgb,
            gt_target_depth=gt_target_depth,
            depth_valid_mask=depth_valid_mask,
            image_height=int(gs_config.get("image_height", 288)),
            image_width=int(gs_config.get("image_width", 512)),
            mse_loss_weight=float(gs_config.get("mse_loss_weight", gs_config.get("rgb_loss_weight", 1.0))),
            lpips_loss_weight=float(gs_config.get("lpips_loss_weight", 0.5)),
            depth_loss_weight=float(gs_config.get("depth_loss_weight", 0.0)),
            abs_depth_loss_weight=float(gs_config.get("abs_depth_loss_weight", 0.0)),
            opacity_loss_weight=float(gs_config.get("opacity_loss_weight", 0.1)),
            detach_predicted_latents=_gs_detach_latents,
            opacity_prune_keep_frac=float(gs_config.get("opacity_prune_keep_frac", 1.0)),
        )

    gs_loss_weight = float(gs_config.get("gs_loss_weight", 0.1))
    weighted = gs_loss_weight * gs_losses["gs_loss"]
    zero = weighted.detach() * 0.0
    total_loss = weighted + zero

    extra_metrics = {
        "loss_diffusion": zero,
        "loss_gs_weighted": weighted.detach(),
        "gs_loss_weight": torch.tensor(float(gs_loss_weight), device=total_loss.device),
        "gs_replay_active": torch.tensor(1.0, device=total_loss.device),
    }
    for k, v in gs_losses.items():
        extra_metrics[k] = v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=total_loss.device)
    extra_metrics["loss_total"] = total_loss.detach()
    return total_loss, extra_metrics


def _as_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor for bool conversion, got shape={tuple(value.shape)}")
        return bool(value.detach().cpu().item())
    return bool(value)


def _as_float(value, default=0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor for float conversion, got shape={tuple(value.shape)}")
        return float(value.detach().cpu().item())
    return float(value)


def _coerce_sample_weight_vector(
    value,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float = 1.0,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch,), float(default), device=device, dtype=dtype)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device, dtype=dtype)
    else:
        value = value.to(device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.reshape(1).expand(batch)
    else:
        value = value.reshape(-1)
        if int(value.numel()) == 1 and batch > 1:
            value = value.expand(batch)
    if int(value.numel()) != int(batch):
        raise ValueError(f"Expected sample-weight vector with B={batch}, got shape={tuple(value.shape)}")
    return value


def _coerce_bcthw(x: torch.Tensor | None, *, name: str, expected_batch: int | None = None) -> torch.Tensor | None:
    if x is None:
        return None
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")
    if x.ndim == 4:  # (C,T,H,W)
        x = x.unsqueeze(0)  # (1,C,T,H,W)
    elif x.ndim != 5:
        raise ValueError(f"{name} must be (C,T,H,W) or (B,C,T,H,W), got shape={tuple(x.shape)}")
    if expected_batch is not None and int(x.shape[0]) == 1 and int(expected_batch) > 1:
        x = x.expand(int(expected_batch), -1, -1, -1, -1)
    if expected_batch is not None and int(x.shape[0]) != int(expected_batch):
        raise ValueError(f"{name} batch mismatch: expected B={expected_batch}, got shape={tuple(x.shape)}")
    return x


def _coerce_mask_b1thw(mask: torch.Tensor | None, *, batch: int, t: int, h: int, w: int) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, 1, t, h, w), dtype=torch.bool)
    if not torch.is_tensor(mask):
        raise TypeError(f"track valid mask must be torch.Tensor, got {type(mask)}")

    # Accepted:
    # - (T,H,W)
    # - (1,T,H,W)
    # - (B,T,H,W)
    # - (B,1,T,H,W)
    # - (1,1,T,H,W)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4:
        if int(mask.shape[0]) == batch:
            mask = mask.unsqueeze(1)  # (B,1,T,H,W)
        else:
            mask = mask.unsqueeze(0)  # (1,1,T,H,W) for (1,T,H,W)
    elif mask.ndim == 5:
        pass
    else:
        raise ValueError(f"Invalid valid-mask shape: {tuple(mask.shape)}")

    if int(mask.shape[1]) != 1:
        raise ValueError(f"Expected valid-mask channel dim=1, got shape={tuple(mask.shape)}")
    if int(mask.shape[0]) == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1, -1, -1)
    if tuple(mask.shape[2:]) != (t, h, w):
        raise ValueError(
            f"Valid-mask spatial/temporal mismatch: expected {(t, h, w)}, got {tuple(mask.shape[2:])}"
        )
    if int(mask.shape[0]) != batch:
        raise ValueError(f"Valid-mask batch mismatch: expected B={batch}, got shape={tuple(mask.shape)}")
    return mask.to(dtype=torch.bool)


def _motion_target_slice(*, v2v_concat_mode: str, input_latents: torch.Tensor, num_domains: int):
    if v2v_concat_mode == "channel":
        return (slice(None), slice(None), slice(None), slice(None), slice(None))
    if v2v_concat_mode == "frame":
        split = input_latents.shape[2] // num_domains
        if num_domains == 3:
            return (slice(None), slice(None), slice(split, 2 * split), slice(None), slice(None))
        return (slice(None), slice(None), slice(split, None), slice(None), slice(None))
    if v2v_concat_mode == "width":
        split = input_latents.shape[4] // num_domains
        if num_domains == 3:
            return (slice(None), slice(None), slice(None), slice(None), slice(split, 2 * split))
        return (slice(None), slice(None), slice(None), slice(None), slice(split, None))
    raise ValueError(f"Unsupported v2v_concat_mode={v2v_concat_mode!r}")


def _denormalize_pred_motion_to_scaled(
    pred_norm_bcthw: torch.Tensor,
    stats_source_bcthw: torch.Tensor,
    stats_valid_b1thw: torch.Tensor,
    *,
    track_xyz_normalization: str,
    track_xyz_quantile_low: float,
    track_xyz_quantile_high: float,
    zero_center: bool,
    residual_linear_normalization: bool,
    residual_log_scale_factor: float,
) -> torch.Tensor:
    """
    Invert track normalization to recover predictions in /common_scale 3D space.
    """
    mode = str(track_xyz_normalization or "minmax")
    b, c, t, h, w = pred_norm_bcthw.shape
    if c != 3:
        raise ValueError(f"Explicit motion loss expects decoded C=3 track tensor, got shape={tuple(pred_norm_bcthw.shape)}")

    if bool(residual_linear_normalization):
        k = float(residual_log_scale_factor)
        if not (k > 0.0):
            k = 1.0
        return pred_norm_bcthw.float() / k

    if mode == "minmax":
        pred_scaled = torch.zeros_like(pred_norm_bcthw, dtype=torch.float32)
        for bi in range(b):
            valid_thw = stats_valid_b1thw[bi, 0]
            for ch in range(c):
                vals = stats_source_bcthw[bi, ch][valid_thw]
                if vals.numel() == 0:
                    lo = torch.tensor(0.0, device=pred_norm_bcthw.device, dtype=torch.float32)
                    hi = torch.tensor(1.0, device=pred_norm_bcthw.device, dtype=torch.float32)
                else:
                    lo = vals.min().float()
                    hi = vals.max().float()
                if (hi - lo).abs() < 1e-6:
                    hi = lo + 1e-6
                if bool(zero_center):
                    scale = torch.maximum(lo.abs(), hi.abs()).clamp_min(1e-6)
                    pred_scaled[bi, ch] = pred_norm_bcthw[bi, ch].float() * scale
                else:
                    pred_scaled[bi, ch] = ((pred_norm_bcthw[bi, ch].float() + 1.0) * 0.5) * (hi - lo) + lo
        return pred_scaled

    if mode in ("signed_log_p2_p98", "signed_log_p0_p100"):
        pred_scaled = torch.zeros_like(pred_norm_bcthw, dtype=torch.float32)
        q_low = 0.0 if mode == "signed_log_p0_p100" else float(track_xyz_quantile_low)
        q_high = 1.0 if mode == "signed_log_p0_p100" else float(track_xyz_quantile_high)
        for bi in range(b):
            if bool(zero_center):
                ql = torch.tensor([-1.0, -1.0, -1.0], device=pred_norm_bcthw.device, dtype=torch.float32)
                qh = torch.tensor([1.0, 1.0, 1.0], device=pred_norm_bcthw.device, dtype=torch.float32)
            else:
                stats_thwc = stats_source_bcthw[bi].permute(1, 2, 3, 0).contiguous().float()
                valid_thw = stats_valid_b1thw[bi, 0]
                ql, qh = compute_signed_log_quantiles(
                    stats_thwc,
                    valid_thw,
                    q_low=q_low,
                    q_high=q_high,
                )
                ql = ql.to(device=pred_norm_bcthw.device, dtype=torch.float32)
                qh = qh.to(device=pred_norm_bcthw.device, dtype=torch.float32)

            params = {
                "ql": ql.detach().cpu().tolist(),
                "qh": qh.detach().cpu().tolist(),
                "q_low": q_low,
                "q_high": q_high,
                "zero_center": bool(zero_center),
                "post_scale": float(residual_log_scale_factor) if bool(zero_center) else 1.0,
            }
            pred_thwc = pred_norm_bcthw[bi].permute(1, 2, 3, 0).contiguous().float()
            pred_scaled_thwc = denormalize_track_signed_log(pred_thwc, params)
            pred_scaled[bi] = pred_scaled_thwc.permute(3, 0, 1, 2).contiguous()
        return pred_scaled

    raise ValueError(f"Unsupported track_xyz_normalization={mode!r} for explicit motion loss")


def _compute_explicit_motion_l2_loss(
    pipe: BasePipeline,
    pred_motion_latents: torch.Tensor,
    inputs: dict,
) -> torch.Tensor:
    """
    Decode x0_hat target motion latents, denormalize to 3D track/residual space, and compare to GT via L2.

    Stream3R adapter already divides GT track/residual by `common_scale`, so this loss is
    computed directly in that shared /common_scale space (no extra common_scale multiply).
    """
    gt_track_scaled = _coerce_bcthw(inputs.get("track"), name="track")
    if gt_track_scaled is None:
        raise ValueError("explicit_motion_loss_weight>0 requires `track` in inputs (add `track` to --extra_inputs)")
    gt_track_scaled = gt_track_scaled.float().to(device=pred_motion_latents.device)

    # Decode denoised target x0_hat to normalized 3-channel motion prediction.
    # For training stability/memory, explicit-motion branch defaults to activation checkpointing.
    pred_motion_norm = _decode_motion_latents_for_explicit_loss(
        pipe=pipe,
        pred_motion_latents=pred_motion_latents,
        inputs=inputs,
    ).float()

    b, c, t, h, w = pred_motion_norm.shape
    if c != 3:
        raise ValueError(
            f"explicit motion loss expects decoded prediction channels=3, got shape={tuple(pred_motion_norm.shape)}"
        )
    if gt_track_scaled.shape != pred_motion_norm.shape:
        raise ValueError(
            f"GT track shape {tuple(gt_track_scaled.shape)} does not match decoded prediction {tuple(pred_motion_norm.shape)}"
        )

    gt_valid = _coerce_mask_b1thw(
        inputs.get("track_valid_mask"),
        batch=b,
        t=t,
        h=h,
        w=w,
    ).to(device=pred_motion_norm.device)

    use_xyz_stats = _as_bool(inputs.get("normalize_track_with_xyz_cond_stats"), False) and (inputs.get("xyz_video_cond") is not None)
    if use_xyz_stats:
        stats_source = _coerce_bcthw(inputs.get("xyz_video_cond"), name="xyz_video_cond", expected_batch=b)
        stats_source = stats_source.float().to(device=pred_motion_norm.device)
        stats_valid = _coerce_mask_b1thw(
            inputs.get("xyz_video_valid_mask"),
            batch=b,
            t=t,
            h=h,
            w=w,
        ).to(device=pred_motion_norm.device)
    else:
        stats_source = gt_track_scaled
        stats_valid = gt_valid

    pred_scaled = _denormalize_pred_motion_to_scaled(
        pred_motion_norm,
        stats_source,
        stats_valid,
        track_xyz_normalization=str(inputs.get("track_xyz_normalization", "minmax")),
        track_xyz_quantile_low=_as_float(inputs.get("track_xyz_quantile_low"), 0.02),
        track_xyz_quantile_high=_as_float(inputs.get("track_xyz_quantile_high"), 0.98),
        zero_center=_as_bool(inputs.get("residual_zero_center_normalization"), False),
        residual_linear_normalization=_as_bool(inputs.get("residual_linear_normalization"), False),
        residual_log_scale_factor=_as_float(inputs.get("residual_log_scale_factor"), 1.0),
    )

    valid_bc = gt_valid.expand(-1, pred_scaled.shape[1], -1, -1, -1)
    sq_err = (pred_scaled - gt_track_scaled).pow(2)
    sq_err = torch.where(valid_bc, sq_err, torch.zeros_like(sq_err))
    denom = valid_bc.sum().clamp_min(1).to(dtype=sq_err.dtype)
    return sq_err.sum() / denom


def _decode_motion_latents_for_explicit_loss(
    *,
    pipe: BasePipeline,
    pred_motion_latents: torch.Tensor,
    inputs: dict,
) -> torch.Tensor:
    """
    Decode motion latents with memory-aware defaults for explicit motion supervision.

    Priority path:
    - decode with `pipe.vae.model.decode` in micro-batches,
    - apply activation checkpointing (default on) to reduce peak VRAM.
    Fallback path:
    - use `pipe.vae.decode` with optional tiled decode flags.
    """
    decode_chunk_cfg = int(_as_float(inputs.get("explicit_motion_loss_decode_chunk_size"), 0))
    use_decode_checkpoint = _as_bool(inputs.get("explicit_motion_loss_decode_checkpoint"), False)

    # Default behavior: preserve original decode path exactly.
    # Optimized path only activates when explicitly requested.
    if (not use_decode_checkpoint) and decode_chunk_cfg <= 0:
        tiled = _as_bool(inputs.get("tiled"), False)
        tile_size = tuple(int(x) for x in (inputs.get("tile_size") or (30, 52)))
        tile_stride = tuple(int(x) for x in (inputs.get("tile_stride") or (15, 26)))
        return pipe.vae.decode(
            pred_motion_latents,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    vae_model = getattr(pipe.vae, "model", None)
    vae_scale = getattr(pipe.vae, "scale", None)
    if vae_model is not None and vae_scale is not None and hasattr(vae_model, "decode"):
        chunks = []
        batch = int(pred_motion_latents.shape[0])
        decode_chunk = batch if decode_chunk_cfg <= 0 else max(1, decode_chunk_cfg)
        for start in range(0, batch, decode_chunk):
            end = min(batch, start + decode_chunk)
            latents_chunk = pred_motion_latents[start:end].to(device=pipe.device)
            if use_decode_checkpoint and latents_chunk.requires_grad:
                def _decode_fn(z):
                    return vae_model.decode(z, vae_scale)
                decoded_chunk = checkpoint(_decode_fn, latents_chunk, use_reentrant=False)
            else:
                decoded_chunk = vae_model.decode(latents_chunk, vae_scale)
            chunks.append(decoded_chunk.clamp(-1, 1))
        return torch.cat(chunks, dim=0)

    # Backward-compatible fallback for VAE wrappers without direct `.model.decode`.
    tiled = _as_bool(inputs.get("tiled"), False)
    tile_size = tuple(int(x) for x in (inputs.get("tile_size") or (30, 52)))
    tile_stride = tuple(int(x) for x in (inputs.get("tile_stride") or (15, 26)))
    return pipe.vae.decode(
        pred_motion_latents,
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


def _apply_explicit_motion_loss_if_enabled(
    diffusion_loss: torch.Tensor,
    *,
    pipe: BasePipeline,
    inputs: dict,
    timestep_id: torch.Tensor,
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    motion_slice,
):
    explicit_motion_loss_weight = _as_float(inputs.get("explicit_motion_loss_weight"), 0.0)
    if explicit_motion_loss_weight <= 0.0:
        return diffusion_loss

    sigma = pipe.scheduler.sigmas[timestep_id].to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    sigma = sigma.reshape(1, 1, 1, 1, 1)
    x0_hat = noisy_latents - sigma * noise_pred
    pred_motion_latents = x0_hat[motion_slice]

    motion_loss_raw = _compute_explicit_motion_l2_loss(
        pipe=pipe,
        pred_motion_latents=pred_motion_latents,
        inputs=inputs,
    )

    # SNR for FM parameterization:
    # x_t = (1 - sigma) * x0 + sigma * eps  => alpha = 1 - sigma
    # snr = alpha^2 / sigma^2.
    sigma_scalar = pipe.scheduler.sigmas[timestep_id].to(device=noisy_latents.device, dtype=torch.float32)
    timestep_scalar = pipe.scheduler.timesteps[timestep_id].to(device=noisy_latents.device, dtype=torch.float32)
    alpha_scalar = (1.0 - sigma_scalar).clamp_min(1e-8)
    snr = (alpha_scalar * alpha_scalar) / (sigma_scalar * sigma_scalar + 1e-8)
    log_snr = torch.log(snr + 1e-12)
    snr_sigmoid_a = _as_float(inputs.get("explicit_motion_loss_snr_sigmoid_a"), 1.0)
    snr_sigmoid_b = _as_float(inputs.get("explicit_motion_loss_snr_sigmoid_b"), -5.0)
    use_snr_weight = _as_bool(inputs.get("explicit_motion_loss_snr_weight"), True)
    if use_snr_weight:
        use_sigmoid_snr = _as_bool(inputs.get("explicit_motion_loss_snr_sigmoid"), True)
        if use_sigmoid_snr:
            snr_weight = torch.sigmoid(snr_sigmoid_a * (log_snr - snr_sigmoid_b))
        else:
            # Legacy weighting for ablation/backward-compat.
            snr_weight = snr / (snr + 1.0)
    else:
        snr_weight = torch.ones_like(snr)

    motion_loss_weighted = motion_loss_raw * snr_weight
    total_loss = diffusion_loss + (explicit_motion_loss_weight * motion_loss_weighted)

    metrics = {
        "loss_diffusion": diffusion_loss.detach(),
        "loss_motion_l2_raw": motion_loss_raw.detach(),
        "loss_motion_l2_weighted": motion_loss_weighted.detach(),
        "loss_motion_weight": torch.tensor(float(explicit_motion_loss_weight), device=diffusion_loss.device),
        "motion_snr": snr.detach(),
        "motion_log_snr": log_snr.detach(),
        "motion_snr_weight": snr_weight.detach(),
        "motion_sigma": sigma_scalar.detach(),
        "motion_timestep": timestep_scalar.detach(),
        "motion_snr_sigmoid_a": torch.tensor(float(snr_sigmoid_a), device=diffusion_loss.device),
        "motion_snr_sigmoid_b": torch.tensor(float(snr_sigmoid_b), device=diffusion_loss.device),
        "loss_total": total_loss.detach(),
    }
    return total_loss, metrics


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def _v2v_num_domains(inputs: dict) -> int:
    if bool(inputs.get("channel_concat", False)):
        return 2
    if bool(inputs.get("multimodality_track_depth", False)):
        return 4
    if bool(inputs.get("multimodality_nvs", False)) or bool(inputs.get("multimodality_tracking", False)):
        return 3
    return 2


def _slice_ar_condition_payload(pipe: BasePipeline, inputs: dict, window_start: int, window_end: int) -> dict:
    """Slice flattened target-condition tokens and rebuild Track4D hooks."""

    def _infer_token_frame_seq_length(tokens, default_frame_seq_length: int) -> int:
        if not torch.is_tensor(tokens):
            return int(default_frame_seq_length)
        full_frames = int(out.get("_ar_full_num_latent_frames") or 0)
        if full_frames > 0 and tokens.ndim == 3 and int(tokens.shape[1]) % full_frames == 0:
            return int(tokens.shape[1]) // full_frames
        patch_size = getattr(getattr(pipe, "dit", None), "patch_size", (1, 1, 1))
        if len(patch_size) >= 3:
            patch_area = max(1, int(patch_size[1]) * int(patch_size[2]))
            if tokens.ndim == 3 and int(default_frame_seq_length) % patch_area == 0:
                return int(default_frame_seq_length) // patch_area
        return int(default_frame_seq_length)

    def _ensure_window_token_span(tokens, key: str, token_frame_seq_length: int):
        if not torch.is_tensor(tokens) or tokens.ndim < 3:
            return tokens
        expected_tokens = int(window_end - window_start) * int(token_frame_seq_length)
        if expected_tokens <= 0:
            return tokens
        token_count = int(tokens[:, ..., 0].numel() // max(1, int(tokens.shape[0])))
        if token_count == expected_tokens:
            return tokens
        flat = tokens.reshape(int(tokens.shape[0]), -1, int(tokens.shape[-1]))
        token_start = int(window_start) * int(token_frame_seq_length)
        token_end = int(window_end) * int(token_frame_seq_length)
        if token_end > int(flat.shape[1]):
            raise ValueError(
                f"{key} AR slice [{token_start}:{token_end}] exceeds token length "
                f"{int(flat.shape[1])}; original_shape={tuple(tokens.shape)}, "
                f"frame_seq_length={token_frame_seq_length}"
            )
        return flat[:, token_start:token_end]

    out = dict(inputs)
    latents = out["latents"]
    debug_ar_shapes = os.environ.get("TRACK4D360_AR_DEBUG_SHAPES", "0").strip().lower() in {"1", "true", "yes", "on"}
    v2v_concat_mode = out.get("v2v_concat_mode", "width")
    num_domains = _v2v_num_domains(out)
    if v2v_concat_mode == "width":
        frame_seq_length = int(latents.shape[3]) * (int(latents.shape[4]) // int(num_domains))
    elif v2v_concat_mode == "frame":
        frame_seq_length = int(latents.shape[3]) * int(latents.shape[4])
    else:
        frame_seq_length = int(latents.shape[3]) * int(latents.shape[4])

    for key in ("track_point_tokens", "dense_ctrl_tokens"):
        _before = out.get(key)
        token_frame_seq_length = _infer_token_frame_seq_length(_before, frame_seq_length)
        out[key] = slice_tokens_by_latent_frames(
            out.get(key),
            start=window_start,
            end=window_end,
            frame_seq_length=token_frame_seq_length,
        )
        out[key] = _ensure_window_token_span(out.get(key), key, token_frame_seq_length)
        if debug_ar_shapes and torch.is_tensor(_before):
            print(
                f"[Track4D-360 AR debug] {key}: window={window_start}:{window_end} "
                f"frame_seq_length={token_frame_seq_length} before={tuple(_before.shape)} "
                f"after={tuple(out[key].shape) if torch.is_tensor(out[key]) else type(out[key]).__name__}",
                flush=True,
            )
    for key in (
        "control_latents_input",
        "control_xyz_latents_input",
        "control_camera_latents_input",
        "control_camera_pose_input",
        "control_camera_plucker_input",
        "vista4d_source_video_latents",
        "vista4d_pointcloud_video_latents",
        "vista4d_source_mask_latents",
        "vista4d_pointcloud_mask_latents",
        "gen3c_warp_condition_latents",
    ):
        out[key] = slice_latent_frames(out.get(key), window_start, window_end)

    track_tokens = out.get("track_point_tokens")
    adapter = getattr(pipe, "track_adapter", None)
    if track_tokens is None or adapter is None:
        return out

    from holo4d.conditions.dense_target_control_unit import _align_track_to_sequence

    injection_mode = getattr(pipe, "track_injection_mode", "single")
    if injection_mode == "per_block":
        block_injector = out.get("_track_block_injector") or getattr(pipe, "track_block_injector", None)
        if block_injector is None:
            return out

        def align_fn(x, f, h, w, num_domains=2, v2v_concat_mode="width"):
            return _align_track_to_sequence(track_tokens, adapter, x, f, h, w, num_domains, v2v_concat_mode)

        out["_track_align_fn"] = align_fn
        out["_track_block_injector"] = block_injector
    else:
        projector = getattr(pipe, "track_projector", None)
        if projector is None:
            return out

        def hook(x, f, h, w, num_domains=2, v2v_concat_mode="width"):
            track_full = _align_track_to_sequence(track_tokens, adapter, x, f, h, w, num_domains, v2v_concat_mode)
            return projector(x, track_full)

        out["_pre_block_hook"] = hook
    return out


def _drop_ar_forward_kwargs(inputs: dict) -> dict:
    """Remove AR-control kwargs that are passed explicitly to model_fn."""

    out = dict(inputs)
    for key in (
        "kv_cache",
        "current_start",
        "cache_start",
        "updating_cache",
        "ar_num_frame_per_block",
        "ar_cache_max_latent_frames",
        "ar_cache_keep_first_latent",
        "_ar_full_num_latent_frames",
    ):
        out.pop(key, None)
    return out


def _target_slice_for_window(inputs: dict, latents: torch.Tensor):
    v2v_concat_mode = inputs.get("v2v_concat_mode", "width")
    if v2v_concat_mode is None:
        return (slice(None), slice(None), slice(None), slice(None), slice(None))
    num_domains = _v2v_num_domains(inputs)
    if v2v_concat_mode == "frame":
        if latents.shape[2] % num_domains != 0:
            raise ValueError(
                f"AR V2V frame mode with {num_domains} domains expects T divisible by domains, got {latents.shape[2]}"
            )
        split = latents.shape[2] // num_domains
        return (slice(None), slice(None), slice(split, None), Ellipsis)
    if v2v_concat_mode != "width":
        raise NotImplementedError("Track4D-360 AR loss currently supports width/frame V2V layouts.")
    if latents.shape[4] % num_domains != 0:
        raise ValueError(
            f"AR V2V width mode with {num_domains} domains expects W divisible by domains, got {latents.shape[4]}"
        )
    split = latents.shape[4] // num_domains
    return (slice(None), slice(None), slice(None), slice(None), slice(split, None))


def _ar_sigma_from_timestep(timestep, num_train_timesteps: int = 1000):
    """Rectified-flow sigma for a Wan timestep (x_sigma = x0 + sigma·v)."""
    return timestep.float().reshape(-1)[0] / float(num_train_timesteps)


def _optional_env_int(name: str):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _optional_env_float(name: str):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _randn_like_ar(ref: torch.Tensor, *, seed: int | None = None, offset: int = 0) -> torch.Tensor:
    if seed is None:
        return torch.randn_like(ref)
    generator = torch.Generator(device=ref.device)
    generator.manual_seed(int(seed) + int(offset))
    return torch.randn(
        ref.shape,
        dtype=ref.dtype,
        device=ref.device,
        layout=ref.layout,
        generator=generator,
    )


def FlowMatchTrack4D360RollingForcingLoss(pipe: BasePipeline, **inputs):
    """Self-forcing AR FM training (no distillation). GPU-VALIDATION-REQUIRED.

    The model learns to denoise the next block GIVEN its OWN self-generated past:
    context blocks 0..exit-1 are rolled out (denoised from noise) under no_grad and
    written into a CLEAN, DETACHED KV cache (built in-order, one forward sweep — so
    no positional cache-overwrite and no per-frame timestep are needed); then a
    single random EXIT block (GT, independently noised) is denoised WITH gradient
    conditioned on that self-generated cache, supervised by a plain flow-matching
    MSE. This keeps RollingForcing's self-rollout + rolling cache but drops the DMD
    distillation. See doc/track4d-360/2026-06-15-rolling-forcing-training-rewrite.md.

    Rectified-flow identities (Wan): sigma=timestep/1000, x0 = x − sigma·v,
    deterministic step x_{s} = x_t + (sigma_s − sigma_t)·v.
    """
    if _as_bool(inputs.get("use_gs_output", False)):
        raise NotImplementedError("rolling_forcing with use_gs_output is not supported.")
    if inputs.get("v2v_concat_mode", "width") != "width":
        raise NotImplementedError("rolling_forcing currently supports v2v_concat_mode='width' only.")

    input_latents = inputs["input_latents"]
    num_latent_frames = int(input_latents.shape[2])
    frames_per_block = int(inputs.get("ar_num_frame_per_block", 3))
    windows = build_latent_windows(num_latent_frames, frames_per_block)
    num_blocks = len(windows)
    denoising_steps = resolve_ar_timesteps(
        pipe.scheduler, inputs.get("ar_denoising_step_list", "1000,800,600,400,200"),
        device=pipe.device, dtype=pipe.torch_dtype,
    )
    context_noise_step = int(inputs.get("ar_context_noise_step", 0))
    num_domains = _v2v_num_domains(inputs)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    assert_exit_only_grad = os.environ.get(
        "TRACK4D360_AR_ASSERT_EXIT_ONLY_GRAD", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    fixed_exit_block = _optional_env_int("TRACK4D360_AR_FIXED_EXIT_BLOCK")
    fixed_timestep_index = _optional_env_int("TRACK4D360_AR_FIXED_TIMESTEP_INDEX")
    fixed_timestep_value = _optional_env_float("TRACK4D360_AR_FIXED_TIMESTEP_VALUE")
    fixed_noise_seed = _optional_env_int("TRACK4D360_AR_NOISE_SEED")
    if fixed_timestep_index is not None and fixed_timestep_value is not None:
        raise ValueError(
            "Set only one of TRACK4D360_AR_FIXED_TIMESTEP_INDEX and "
            "TRACK4D360_AR_FIXED_TIMESTEP_VALUE."
        )

    def _window_model_inputs(window, clean_source_window, latents_window):
        wi = dict(inputs)
        wi["_ar_full_num_latent_frames"] = num_latent_frames
        wi["input_latents"] = clean_source_window  # source-left conditioning for this window
        wi["latents"] = latents_window
        return _drop_ar_forward_kwargs(_slice_ar_condition_payload(pipe, wi, window.start, window.end))

    def _keep_source_clean(x, clean_source_window):
        split = x.shape[4] // num_domains
        x[:, :, :, :, :split] = clean_source_window[:, :, :, :, :split]
        return x

    def _forward_v(window, clean_source_window, x, timestep, kv_cache, updating_cache):
        out = pipe.model_fn(
            **models, **_window_model_inputs(window, clean_source_window, x),
            timestep=timestep, kv_cache=kv_cache, current_start=window.start,
            updating_cache=updating_cache, ar_num_frame_per_block=frames_per_block,
            ar_cache_max_latent_frames=int(inputs.get("ar_cache_max_latent_frames", 21)),
            ar_cache_keep_first_latent=_as_bool(inputs.get("ar_cache_keep_first_latent", False)),
        )
        return out["noise_pred"] if isinstance(out, dict) else out

    # Random exit block: the only block that carries gradient (self-forcing).
    # Env overrides are for deterministic one-sample overfit validation only.
    if fixed_exit_block is None:
        exit_block = int(torch.randint(num_blocks, size=(), device="cpu").item())
    else:
        if fixed_exit_block < 0 or fixed_exit_block >= num_blocks:
            raise ValueError(
                f"TRACK4D360_AR_FIXED_EXIT_BLOCK={fixed_exit_block} is outside "
                f"[0, {num_blocks})"
            )
        exit_block = int(fixed_exit_block)
    kv_cache = allocate_kv_cache(len(getattr(pipe.dit, "blocks", [])))

    # 1. Self-rollout the context blocks 0..exit-1 under no_grad → clean detached cache.
    with torch.no_grad():
        for j in range(exit_block):
            w = windows[j]
            src = slice_latent_frames(input_latents, w.start, w.end)
            x = _randn_like_ar(src, seed=fixed_noise_seed, offset=j)
            x = _keep_source_clean(x, src)
            x0 = x
            for step_idx in range(len(denoising_steps)):
                t = denoising_steps[step_idx].reshape(1)
                sigma = _ar_sigma_from_timestep(t)
                v = _forward_v(w, src, x, t, kv_cache=None, updating_cache=False)
                x0 = x - sigma * v
                sigma_next = (_ar_sigma_from_timestep(denoising_steps[step_idx + 1].reshape(1))
                              if step_idx + 1 < len(denoising_steps) else x0.new_zeros(()))
                x = x + (sigma_next - sigma) * v
                x = _keep_source_clean(x, src)
            # Write the clean denoised block into the rolling cache (≈zero noise).
            ctx_t = torch.full((1,), float(context_noise_step), device=pipe.device, dtype=pipe.torch_dtype)
            _forward_v(w, src, _keep_source_clean(x0.clone(), src), ctx_t, kv_cache=kv_cache, updating_cache=True)
            for _entry in kv_cache:
                for _ck in ("k", "v"):
                    if torch.is_tensor(_entry.get(_ck)):
                        _entry[_ck] = _entry[_ck].detach()
            if assert_exit_only_grad:
                if x.grad_fn is not None or x0.grad_fn is not None:
                    raise RuntimeError(
                        "TRACK4D360_AR_ASSERT_EXIT_ONLY_GRAD failed: rolling_forcing "
                        "context rollout kept autograd history."
                    )
                for block_idx, _entry in enumerate(kv_cache):
                    for _ck in ("k", "v"):
                        _tensor = _entry.get(_ck)
                        if torch.is_tensor(_tensor) and (_tensor.grad_fn is not None or _tensor.requires_grad):
                            raise RuntimeError(
                                "TRACK4D360_AR_ASSERT_EXIT_ONLY_GRAD failed: clean KV "
                                f"cache block {block_idx} tensor {_ck!r} is not detached."
                            )

    # 2. Train on the GT exit block, conditioned on the self-generated cache (with grad).
    w = windows[exit_block]
    src = slice_latent_frames(input_latents, w.start, w.end)
    # Independent random noise level for this block.
    if fixed_timestep_value is not None:
        timesteps_float = pipe.scheduler.timesteps.float().to(device="cpu")
        t_id = torch.argmin(torch.abs(timesteps_float - float(fixed_timestep_value))).reshape(1).long()
    elif fixed_timestep_index is None:
        t_id = torch.randint(0, len(pipe.scheduler.timesteps), (1,))
    else:
        if fixed_timestep_index < 0 or fixed_timestep_index >= len(pipe.scheduler.timesteps):
            raise ValueError(
                f"TRACK4D360_AR_FIXED_TIMESTEP_INDEX={fixed_timestep_index} is outside "
                f"[0, {len(pipe.scheduler.timesteps)})"
            )
        t_id = torch.tensor([fixed_timestep_index], dtype=torch.long)
    timestep = pipe.scheduler.timesteps[t_id].to(device=pipe.device, dtype=pipe.torch_dtype)
    noise = _randn_like_ar(src, seed=fixed_noise_seed, offset=10000 + exit_block)
    noisy = pipe.scheduler.add_noise(src, noise, timestep)
    noisy = _keep_source_clean(noisy, src)
    target = pipe.scheduler.training_target(src, noise, timestep)
    v = _forward_v(w, src, noisy, timestep, kv_cache=kv_cache, updating_cache=False)
    if assert_exit_only_grad and v.grad_fn is None:
        raise RuntimeError(
            "TRACK4D360_AR_ASSERT_EXIT_ONLY_GRAD failed: exit-block prediction "
            "has no autograd history."
        )
    window_inputs = _window_model_inputs(w, src, noisy)
    pred_slice = _target_slice_for_window(window_inputs, noisy)
    loss = torch.nn.functional.mse_loss(v[pred_slice].float(), target[pred_slice].float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    metrics = {
        "loss_ar_selfforcing": loss.detach(),
        "ar_exit_block": torch.tensor(float(exit_block), device=loss.device),
        "ar_context_blocks": torch.tensor(float(exit_block), device=loss.device),
        "ar_timestep_index": torch.tensor(float(int(t_id.reshape(-1)[0].item())), device=loss.device),
        "ar_timestep_value": timestep.detach().float().reshape(-1)[0],
        "loss_total": loss.detach(),
    }
    return loss, metrics


def _ar_frame_seq_length_for_latents(pipe: BasePipeline, latents: torch.Tensor) -> int:
    patch_size = tuple(getattr(getattr(pipe, "dit", None), "patch_size", (1, 2, 2)))
    patch_h = int(patch_size[1]) if len(patch_size) >= 2 else 1
    patch_w = int(patch_size[2]) if len(patch_size) >= 3 else 1
    return int(latents.shape[3] * latents.shape[4] // max(1, patch_h * patch_w))


def _ar_keep_source_clean(inputs: dict, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    out = noisy.clone()
    num_domains = _v2v_num_domains(inputs)
    if inputs.get("v2v_concat_mode", "width") == "width":
        split = out.shape[4] // num_domains
        out[:, :, :, :, :split] = clean[:, :, :, :, :split]
    elif inputs.get("v2v_concat_mode", "width") == "frame":
        split = out.shape[2] // num_domains
        out[:, :, :split] = clean[:, :, :split]
    else:
        raise NotImplementedError("diffusion_forcing currently supports width/frame V2V layouts.")
    return out


def _ar_training_weight_bcthw(pipe: BasePipeline, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(timesteps) and timesteps.numel() > 1:
        flat_weights = [
            pipe.scheduler.training_weight(t.reshape(1)).to(device=like.device, dtype=like.dtype)
            for t in timesteps.reshape(-1)
        ]
        weight = torch.stack([w.reshape(()) for w in flat_weights]).reshape(timesteps.shape)
    else:
        weight = pipe.scheduler.training_weight(timesteps)
    if not torch.is_tensor(weight):
        weight = torch.tensor(weight, device=like.device, dtype=like.dtype)
    weight = weight.to(device=like.device, dtype=like.dtype)
    if weight.ndim == 0:
        return weight
    if weight.ndim == 1:
        if weight.numel() == 1:
            return weight.reshape(())
        weight = weight.reshape(1, 1, int(weight.numel()), 1, 1)
    elif weight.ndim == 2:
        weight = weight.reshape(int(weight.shape[0]), 1, int(weight.shape[1]), 1, 1)
    return weight


def FlowMatchTrack4D360DiffusionForcingLoss(pipe: BasePipeline, **inputs):
    """Diffusion Forcing-style Wan FM loss: independent timestep per latent frame.

    This is the non-DMD path: one full-sequence forward with a token-level
    timestep tensor. It preserves the existing scalar-timestep path unless
    explicitly selected via ``--ar_training_algorithm diffusion_forcing``.
    """

    if _as_bool(inputs.get("use_gs_output", False)):
        raise NotImplementedError("diffusion_forcing with use_gs_output is not supported.")
    if inputs.get("v2v_concat_mode", "width") == "channel":
        raise NotImplementedError("diffusion_forcing currently supports width/frame V2V layouts, not channel-concat.")

    input_latents = inputs["input_latents"]
    batch_size = int(input_latents.shape[0])
    num_latent_frames = int(input_latents.shape[2])
    fixed_values = os.environ.get("TRACK4D360_DF_TIMESTEP_VALUES")
    if fixed_values:
        latent_timesteps = parse_fixed_timestep_values(
            fixed_values,
            num_latent_frames=num_latent_frames,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        ).expand(batch_size, num_latent_frames)
    else:
        latent_timesteps = sample_random_independent_timesteps(
            pipe.scheduler,
            batch_size=batch_size,
            num_latent_frames=num_latent_frames,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )

    noise = torch.randn_like(input_latents)
    noisy_latents = flow_add_noise_per_frame(input_latents, noise, latent_timesteps)
    noisy_latents = _ar_keep_source_clean(inputs, noisy_latents, input_latents)
    target = flow_training_target_per_frame(input_latents, noise)

    frame_seq_length = _ar_frame_seq_length_for_latents(pipe, noisy_latents)
    token_timesteps = expand_latent_timesteps_to_token_timesteps(
        latent_timesteps.to(device=pipe.device, dtype=pipe.torch_dtype),
        frame_seq_length=frame_seq_length,
    )
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    forward_inputs = dict(inputs)
    forward_inputs["latents"] = noisy_latents
    forward_inputs = _drop_ar_forward_kwargs(forward_inputs)
    model_output = pipe.model_fn(
        **models,
        **forward_inputs,
        timestep=token_timesteps,
    )
    noise_pred = model_output["noise_pred"] if isinstance(model_output, dict) else model_output

    pred_slice = _target_slice_for_window(forward_inputs, noisy_latents)
    per_element = torch.nn.functional.mse_loss(
        noise_pred[pred_slice].float(),
        target[pred_slice].float(),
        reduction="none",
    )
    loss = per_element * _ar_training_weight_bcthw(pipe, latent_timesteps, per_element)
    loss = loss.mean()
    metrics = {
        "loss_ar_diffusion_forcing": loss.detach(),
        "df_mean_timestep": latent_timesteps.detach().float().mean(),
        "df_min_timestep": latent_timesteps.detach().float().min(),
        "df_max_timestep": latent_timesteps.detach().float().max(),
        "loss_total": loss.detach(),
    }
    return loss, metrics


def FlowMatchTrack4D360CausalTeacherForcingLoss(pipe: BasePipeline, **inputs):
    """Causal-Forcing stage-1 style Wan FM loss.

    This is the non-DMD first stage used to validate causal teacher forcing on
    raw/no-control Wan: clean-context tokens and noisy tokens are concatenated in
    one forward, and a teacher-forcing causal attention mask prevents noisy
    tokens from seeing their own clean frame. The current implementation uses a
    dense mask, so it is intentionally a low-resolution smoke/overfit path; the
    480p/81f path needs the reference-style flex/block mask backend.
    """

    if _as_bool(inputs.get("use_gs_output", False)):
        raise NotImplementedError("causal_teacher_forcing with use_gs_output is not supported.")
    if inputs.get("v2v_concat_mode", None) is not None:
        raise NotImplementedError(
            "causal_teacher_forcing is target-only raw T2V: expected v2v_concat_mode=None. "
            "Use --raw_t2v_target_only with the provided launchers."
        )
    if _as_bool(inputs.get("use_domain_embedding", True)):
        raise NotImplementedError("causal_teacher_forcing first cut requires --disable_domain_embedding.")

    input_latents = inputs["input_latents"]
    batch_size = int(input_latents.shape[0])
    num_latent_frames = int(input_latents.shape[2])
    frames_per_block = int(inputs.get("ar_num_frame_per_block", 1))
    fixed_values = os.environ.get("TRACK4D360_DF_TIMESTEP_VALUES")
    if fixed_values:
        latent_timesteps = parse_fixed_timestep_values(
            fixed_values,
            num_latent_frames=num_latent_frames,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        ).expand(batch_size, num_latent_frames)
    else:
        latent_timesteps = sample_random_independent_timesteps(
            pipe.scheduler,
            batch_size=batch_size,
            num_latent_frames=num_latent_frames,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
    if frames_per_block > 1:
        for start in range(0, num_latent_frames, frames_per_block):
            end = min(start + frames_per_block, num_latent_frames)
            latent_timesteps[:, start:end] = latent_timesteps[:, start:start + 1]

    noise = torch.randn_like(input_latents)
    noisy_latents = flow_add_noise_per_frame(input_latents, noise, latent_timesteps)
    target = flow_training_target_per_frame(input_latents, noise)

    frame_seq_length = _ar_frame_seq_length_for_latents(pipe, noisy_latents)
    token_timesteps = expand_latent_timesteps_to_token_timesteps(
        latent_timesteps.to(device=pipe.device, dtype=pipe.torch_dtype),
        frame_seq_length=frame_seq_length,
    )

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    forward_inputs = dict(inputs)
    forward_inputs["latents"] = noisy_latents
    forward_inputs = _drop_ar_forward_kwargs(forward_inputs)
    model_output = pipe.model_fn(
        **models,
        **forward_inputs,
        timestep=token_timesteps,
        causal_teacher_clean_latents=input_latents,
        causal_teacher_num_frame_per_block=frames_per_block,
    )
    noise_pred = model_output["noise_pred"] if isinstance(model_output, dict) else model_output

    pred_slice = _target_slice_for_window(forward_inputs, noisy_latents)
    per_element = torch.nn.functional.mse_loss(
        noise_pred[pred_slice].float(),
        target[pred_slice].float(),
        reduction="none",
    )
    loss = per_element * _ar_training_weight_bcthw(pipe, latent_timesteps, per_element)
    loss = loss.mean()
    metrics = {
        "loss_ar_causal_teacher_forcing": loss.detach(),
        "ctf_mean_timestep": latent_timesteps.detach().float().mean(),
        "ctf_min_timestep": latent_timesteps.detach().float().min(),
        "ctf_max_timestep": latent_timesteps.detach().float().max(),
        "ctf_num_frame_per_block": torch.tensor(
            float(inputs.get("ar_num_frame_per_block", 1)),
            device=loss.device,
            dtype=loss.dtype,
        ),
        "loss_total": loss.detach(),
    }
    return loss, metrics


def FlowMatchTrack4D360ARLoss(pipe: BasePipeline, **inputs):
    """Windowed teacher-forcing AR V2V loss (the default `windowed_teacher_forcing`
    algorithm). Noises GT latent windows with a causal KV cache — this is NOT the
    self-forcing RollingForcing objective (see FlowMatchTrack4D360RollingForcingLoss
    and the rewrite doc). The model sees GT/teacher context, not its own rollout."""

    if _as_bool(inputs.get("use_gs_output", False)):
        raise NotImplementedError("ar_training=True with use_gs_output=True is intentionally disabled.")
    if inputs.get("v2v_concat_mode", "width") == "channel":
        raise NotImplementedError("Track4D-360 AR loss does not support channel-concat mode yet.")

    input_latents = inputs["input_latents"]
    num_latent_frames = int(input_latents.shape[2])
    frames_per_block = int(inputs.get("ar_num_frame_per_block", 3))
    windows = build_latent_windows(num_latent_frames, frames_per_block)
    ar_timesteps = resolve_ar_timesteps(
        pipe.scheduler,
        inputs.get("ar_denoising_step_list", "1000,800,600,400,200"),
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    same_step = _as_bool(inputs.get("ar_same_step_across_blocks", False))
    cache_max_frames = int(inputs.get("ar_cache_max_latent_frames", 21))
    cache_keep_first_latent = _as_bool(inputs.get("ar_cache_keep_first_latent", False))
    loss_last_frames = int(inputs.get("ar_loss_last_latent_frames", 21))
    # Teacher-forcing AR: treat the cached past windows as FIXED context (detach),
    # so each window only trains "predict this window given the given past". The
    # previous grad-attached cache backprop'd through how the past would have been
    # produced (window-coupled, memory-heavy). Detaching is the standard AR/teacher-
    # forcing behavior and is a no-op at inference (eval rollout runs under no_grad).
    detach_context_cache = _as_bool(inputs.get("ar_detach_context_cache", True))

    noise = torch.randn_like(input_latents)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    kv_cache = allocate_kv_cache(len(getattr(pipe.dit, "blocks", [])))
    losses = []
    used_windows = 0

    for window_idx, window in enumerate(windows):
        timestep = select_window_timestep(
            ar_timesteps,
            window_idx,
            same_step_across_blocks=same_step,
        ).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_window = slice_latent_frames(input_latents, window.start, window.end)
        noise_window = slice_latent_frames(noise, window.start, window.end)
        noisy_window = pipe.scheduler.add_noise(input_window, noise_window, timestep)
        target_window = pipe.scheduler.training_target(input_window, noise_window, timestep)

        # Keep source domain clean exactly like FlowMatchV2VLoss.
        num_domains = _v2v_num_domains(inputs)
        if inputs.get("v2v_concat_mode", "width") == "width":
            split = noisy_window.shape[4] // num_domains
            noisy_window[:, :, :, :, :split] = input_window[:, :, :, :, :split]
        else:
            split = noisy_window.shape[2] // num_domains
            noisy_window[:, :, :split] = input_window[:, :, :split]

        window_inputs = dict(inputs)
        window_inputs["_ar_full_num_latent_frames"] = num_latent_frames
        window_inputs["input_latents"] = input_window
        window_inputs["latents"] = noisy_window
        window_inputs = _drop_ar_forward_kwargs(
            _slice_ar_condition_payload(pipe, window_inputs, window.start, window.end)
        )
        model_output = pipe.model_fn(
            **models,
            **window_inputs,
            timestep=timestep,
            kv_cache=kv_cache,
            current_start=window.start,
            cache_start=max(0, window.end - cache_max_frames),
            updating_cache=True,
            ar_num_frame_per_block=frames_per_block,
            ar_cache_max_latent_frames=cache_max_frames,
            ar_cache_keep_first_latent=cache_keep_first_latent,
        )
        if isinstance(model_output, dict):
            noise_pred = model_output["noise_pred"]
        else:
            noise_pred = model_output

        # Detach the accumulated cache so the NEXT window sees fixed past context.
        if detach_context_cache:
            for _entry in kv_cache:
                for _ck in ("k", "v"):
                    if torch.is_tensor(_entry.get(_ck)):
                        _entry[_ck] = _entry[_ck].detach()

        # By default, only supervise the tail latent frames in the full sequence.
        if window.end <= max(0, num_latent_frames - loss_last_frames):
            continue
        pred_slice = _target_slice_for_window(window_inputs, noisy_window)
        losses.append(torch.nn.functional.mse_loss(noise_pred[pred_slice].float(), target_window[pred_slice].float()))
        used_windows += 1

    if not losses:
        raise RuntimeError("AR loss selected zero windows; increase ar_loss_last_latent_frames.")
    loss = torch.stack(losses).mean()
    loss = loss * pipe.scheduler.training_weight(ar_timesteps[-1].reshape(1).to(device=pipe.device, dtype=pipe.torch_dtype))
    metrics = {
        "loss_ar_diffusion": loss.detach(),
        "ar_windows": torch.tensor(float(len(windows)), device=loss.device),
        "ar_loss_windows": torch.tensor(float(used_windows), device=loss.device),
        "ar_cache_max_latent_frames": torch.tensor(float(cache_max_frames), device=loss.device),
        "ar_cache_keep_first_latent": torch.tensor(float(cache_keep_first_latent), device=loss.device),
        "loss_total": loss.detach(),
    }
    return loss, metrics


# Future-prediction primitives live in a torch-only module so they remain
# unit-testable without the heavy diffusion/transformers import stack. Both are
# no-ops unless future prediction is enabled → default V2V path unchanged.
from holo4d.future_prediction import (
    resolve_observed_mask as _future_observed_mask,
    apply_future_source_null as _apply_future_source_null,
    future_weighted_mse as _future_weighted_mse,
)


def FlowMatchV2VLoss(pipe: BasePipeline, **inputs):
    # GS replay-cache fast path (overfit accelerator). Once the cache has
    # `replay_cache_size` snapshots, every subsequent step skips the DiT
    # forward and replays a random cached entry. Only safe when
    # skip_diffusion_loss is True (DiT frozen) — otherwise the cached
    # gs_features_target / latents_pred_target would reflect a stale DiT.
    _gs_cfg_top = inputs.get("gs_config", {}) or {}
    _replay_cap = int(_gs_cfg_top.get("replay_cache_size", 0))
    if (
        _replay_cap > 0
        and len(_GS_REPLAY_CACHE) >= _replay_cap
        and _as_bool(inputs.get("skip_diffusion_loss", False))
    ):
        return _replay_gs_only_loss(pipe, {
            **_gs_cfg_top,
            "image_height": int(_gs_cfg_top.get("image_height", 288)),
            "image_width": int(_gs_cfg_top.get("image_width", 512)),
        })

    # Sigma-pinned path (GS-only training): resolve the requested noise level
    # to the nearest scheduler index and freeze it for this step. Done BEFORE
    # the index-fraction path so step-boundary semantics stay unchanged when
    # fixed_sigma is not set. At shift=5 the sigma↔index map is highly
    # nonlinear (sigma=0.25 ≈ index 937/1000), so index-fraction pinning
    # cannot stand in for sigma pinning; they are different knobs.
    fixed_sigma = inputs.get("fixed_sigma")
    if fixed_sigma is not None:
        _sigmas_cpu = pipe.scheduler.sigmas.detach().cpu().float()
        timestep_id = torch.argmin((_sigmas_cpu - float(fixed_sigma)).abs()).reshape(1)
        timestep = pipe.scheduler.timesteps[timestep_id].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
    else:
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents = inputs["input_latents"]
    v2v_concat_mode = inputs.get("v2v_concat_mode", "frame")
    channel_concat = _as_bool(inputs.get("channel_concat", False))
    channel_modalities = inputs.get("channel_modalities") or []
    multimodality_nvs = inputs.get("multimodality_nvs", False)
    multimodality_tracking = inputs.get("multimodality_tracking", False)
    multimodality_track_depth = inputs.get("multimodality_track_depth", False)

    active_multi_modes = int(bool(multimodality_nvs)) + int(bool(multimodality_tracking)) + int(bool(multimodality_track_depth))
    if active_multi_modes > 1:
        raise ValueError("Only one of multimodality_nvs, multimodality_tracking, or multimodality_track_depth can be enabled.")
    if (bool(multimodality_tracking) or bool(multimodality_track_depth)) and v2v_concat_mode == "frame":
        raise NotImplementedError(
            "multimodality tracking modes with frame-wise concat are not implemented. Use v2v_concat_mode='width'."
        )

    if v2v_concat_mode not in ("frame", "width", "channel"):
        raise ValueError(f"Invalid v2v_concat_mode: {v2v_concat_mode}")

    if v2v_concat_mode == "channel":
        cond_latents = inputs.get("cond_latents")
        if cond_latents is None:
            raise ValueError("Channel concat mode requires `cond_latents` (clean conditioning RGB latents).")
        if cond_latents.shape != input_latents.shape:
            raise ValueError(
                f"Channel concat expects cond_latents and input_latents to have identical shapes. "
                f"Got cond_latents={tuple(cond_latents.shape)}, input_latents={tuple(input_latents.shape)}."
            )

        noise = torch.randn_like(input_latents)
        noisy_latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
        training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

        inputs["latents"] = noisy_latents
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

        diffusion_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        diffusion_loss = diffusion_loss * pipe.scheduler.training_weight(timestep)
        motion_slice = _motion_target_slice(
            v2v_concat_mode=v2v_concat_mode,
            input_latents=input_latents,
            num_domains=1,
        )
        return _apply_explicit_motion_loss_if_enabled(
            diffusion_loss,
            pipe=pipe,
            inputs=inputs,
            timestep_id=timestep_id,
            noisy_latents=noisy_latents,
            noise_pred=noise_pred,
            motion_slice=motion_slice,
        )

    noise = torch.randn_like(input_latents)
    noisy_latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    # Determine number of domains based on multi-modality mode
    if channel_concat:
        num_domains = 2
    else:
        num_domains = 4 if bool(multimodality_track_depth) else (3 if (multimodality_nvs or multimodality_tracking) else 2)

    # Future prediction (temporal extrapolation): resolve the observed/future
    # latent-frame mask. None when future prediction is off → default path.
    _future_observed = _future_observed_mask(
        inputs, latent_t=int(input_latents.shape[2]),
        device=input_latents.device, batch=int(input_latents.shape[0]),
    )
    if _future_observed is not None and v2v_concat_mode != "width":
        raise NotImplementedError("future_prediction currently supports v2v_concat_mode='width' only.")

    # Keep conditioning (source RGB) clean and compute loss on targets
    if v2v_concat_mode == "frame":
        if noisy_latents.shape[2] % num_domains != 0:
            raise ValueError(f"V2V frame mode with {num_domains} domains expects frame length divisible by {num_domains}, got T={noisy_latents.shape[2]}.")
        split = noisy_latents.shape[2] // num_domains
        # Keep source RGB clean (first 1/num_domains)
        noisy_latents[:, :, :split] = input_latents[:, :, :split]
        # Compute loss on target domains (remaining)
        pred_slice = (slice(None), slice(None), slice(split, None), Ellipsis)
    else:  # width mode
        if noisy_latents.shape[4] % num_domains != 0:
            raise ValueError(f"V2V width mode with {num_domains} domains expects width divisible by {num_domains}, got W={noisy_latents.shape[4]}.")
        split = noisy_latents.shape[4] // num_domains
        # Keep source RGB clean (first 1/num_domains)
        noisy_latents[:, :, :, :, :split] = input_latents[:, :, :, :, :split]
        # Future prediction: future latent frames have no observed source → replace
        # their source domain with the learned null (observed frames keep the clean
        # source). No-op when future prediction is off.
        noisy_latents = _apply_future_source_null(
            noisy_latents, split, _future_observed,
            getattr(pipe.dit, "future_source_null", None),
        )
        # Compute loss on target domains (remaining)
        if num_domains == 3:
            # 3 domains: [source | target_1 | target_2]
            # Loss on both target domains (middle 1/3 + right 1/3)
            pred_slice = (slice(None), slice(None), slice(None), slice(None), slice(split, None))
        else:
            # 2 domains: [source_rgb | target_rgb]
            # Loss on target_rgb only (right half)
            pred_slice = (slice(None), slice(None), slice(None), slice(None), slice(split, None))

    inputs["latents"] = noisy_latents
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # --- FlashWorld-style 2-forward GS feedback training ---
    # Forward 1 (bootstrap): run without feedback to produce gs_features + x0_hat,
    # then GS decode + render + VAE encode → z_3d. These become the feedback
    # conditioning channels for Forward 2.
    use_gs_feedback_training = _as_bool(inputs.get("use_gs_feedback_training", False))
    feedback_input = None
    if (
        use_gs_feedback_training
        and hasattr(pipe, "gs_decoder") and pipe.gs_decoder is not None
        and getattr(pipe.dit, "_feedback_channels", 0) > 0
    ):
        with torch.no_grad():
            model_out_1 = pipe.model_fn(**models, **inputs, timestep=timestep, feedback_input=None)
        if isinstance(model_out_1, dict):
            gs_features_1 = model_out_1.get("gs_features")
            noise_pred_1 = model_out_1["noise_pred"]
        else:
            gs_features_1 = None
            noise_pred_1 = model_out_1

        if gs_features_1 is not None:
            # Compute x0_hat from forward 1
            _sigma = timestep / 1000.0
            if v2v_concat_mode == "width":
                gs_feats_tgt_1 = gs_features_1[:, :, :, :, split:]
                np_tgt_1 = noise_pred_1[:, :, :, :, split:]
                noisy_tgt_1 = noisy_latents[:, :, :, :, split:]
                x0_hat_tgt_1 = noisy_tgt_1 - _sigma.view(-1, 1, 1, 1, 1) * np_tgt_1
            else:
                gs_feats_tgt_1 = gs_features_1[:, :, split:]
                np_tgt_1 = noise_pred_1[:, :, split:]
                noisy_tgt_1 = noisy_latents[:, :, split:]
                x0_hat_tgt_1 = noisy_tgt_1 - _sigma.view(-1, 1, 1, 1) * np_tgt_1

            # GS decode + render + VAE encode → z_3d (3D-consistent latent)
            target_cameras = inputs.get("target_cameras_flashworld")
            gs_config = inputs.get("gs_config", {})
            if target_cameras is not None:
                if torch.is_tensor(target_cameras) and target_cameras.dim() == 2:
                    target_cameras = target_cameras.unsqueeze(0)
                H_img = int(gs_config.get("image_height", 288))
                W_img = int(gs_config.get("image_width", 512))
                with torch.no_grad():
                    gs_params = pipe.gs_decoder(gs_feats_tgt_1, x0_hat_tgt_1, target_cameras)
                    rendered_rgb, _ = pipe.gs_decoder.render(
                        gs_params, target_cameras, height=H_img, width=W_img, bg_mode="white",
                    )
                    # rendered_rgb: (B, T_video, 3, H, W) in [-1, 1]
                    rgb_seq = rendered_rgb.permute(0, 2, 1, 3, 4).to(
                        device=noisy_latents.device,
                        dtype=noisy_latents.dtype,
                    )  # (B, 3, T, H, W)
                    z_3d = pipe.vae.encode(
                        rgb_seq, device=noisy_latents.device, tiled=True,
                        tile_size=(30, 52), tile_stride=(15, 26),
                    )
                    if z_3d.dim() == 4:
                        z_3d = z_3d.unsqueeze(0)
                    z_3d = z_3d.to(dtype=noisy_latents.dtype, device=noisy_latents.device)

                # Build feedback: cat([prev_feats, prev_latents_pred], dim=1)
                # with zeros for source half, signal for target half
                prev_feats = torch.zeros_like(gs_features_1)
                prev_latents_pred_full = torch.zeros_like(noisy_latents)
                if v2v_concat_mode == "width":
                    prev_feats[:, :, :, :, split:] = gs_feats_tgt_1.detach()
                    prev_latents_pred_full[:, :, :, :, split:] = z_3d.detach()
                else:
                    prev_feats[:, :, split:] = gs_feats_tgt_1.detach()
                    prev_latents_pred_full[:, :, split:] = z_3d.detach()
                feedback_input = torch.cat([prev_feats, prev_latents_pred_full], dim=1)

    model_output = pipe.model_fn(**models, **inputs, timestep=timestep, feedback_input=feedback_input)

    # model_fn returns dict when GS feature head is active, plain tensor otherwise.
    # gs_pre_head_input / gs_pre_head_t / gs_unpatchify_shape are surfaced so
    # the GS replay cache can stash them and re-run gs_feature_head on every
    # replay step (instead of caching post-head features and skipping the head,
    # which would leave gs_feature_head's params without gradient).
    if isinstance(model_output, dict):
        noise_pred = model_output["noise_pred"]
        gs_features = model_output.get("gs_features")
        gs_pre_head_input = model_output.get("gs_pre_head_input")
        gs_pre_head_t = model_output.get("gs_pre_head_t")
        gs_unpatchify_shape = model_output.get("gs_unpatchify_shape")
    else:
        noise_pred = model_output
        gs_features = None
        gs_pre_head_input = None
        gs_pre_head_t = None
        gs_unpatchify_shape = None

    pred_target = noise_pred[pred_slice].float()
    gt_target = training_target[pred_slice].float()
    if channel_concat and v2v_concat_mode == "width":
        num_modalities = int(inputs.get("num_channel_modalities", 0) or len(channel_modalities))
        if num_modalities <= 0:
            raise ValueError("channel_concat=True requires num_channel_modalities or channel_modalities.")
        if len(channel_modalities) < num_modalities:
            raise ValueError(
                f"channel_modalities has {len(channel_modalities)} entries but "
                f"num_channel_modalities={num_modalities}. "
                "Depth/track channels would receive no mask weighting."
            )
        if int(pred_target.shape[1]) % num_modalities != 0:
            raise ValueError(
                f"Channel-concat target channels={int(pred_target.shape[1])} must be divisible by "
                f"num_channel_modalities={num_modalities}."
            )
        depth_loss_mask = _coerce_sample_weight_vector(
            inputs.get("depth_loss_mask"),
            batch=int(pred_target.shape[0]),
            device=pred_target.device,
            dtype=pred_target.dtype,
            default=1.0,
        )
        track_video_loss_mask = _coerce_sample_weight_vector(
            inputs.get("track_video_loss_mask"),
            batch=int(pred_target.shape[0]),
            device=pred_target.device,
            dtype=pred_target.dtype,
            default=1.0,
        )
        loss_weights = torch.ones_like(pred_target)
        c_per_modality = int(pred_target.shape[1]) // num_modalities
        for mod_idx, modality in enumerate(channel_modalities[:num_modalities]):
            start = mod_idx * c_per_modality
            end = start + c_per_modality
            if modality in ("depth", "target_depth"):
                loss_weights[:, start:end] = depth_loss_mask.view(-1, 1, 1, 1, 1)
            elif modality in ("track", "target_track", "residual_track"):
                loss_weights[:, start:end] = track_video_loss_mask.view(-1, 1, 1, 1, 1)
        sq_err = (pred_target - gt_target) ** 2
        diffusion_loss = sq_err.mul(loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
    elif (bool(multimodality_tracking) or bool(multimodality_track_depth)) and v2v_concat_mode == "width" and num_domains >= 3:
        track_video_loss_mask = _coerce_sample_weight_vector(
            inputs.get("track_video_loss_mask"),
            batch=int(pred_target.shape[0]),
            device=pred_target.device,
            dtype=pred_target.dtype,
            default=1.0,
        )
        track_loss_mask = _coerce_sample_weight_vector(
            inputs.get("track_loss_mask"),
            batch=int(pred_target.shape[0]),
            device=pred_target.device,
            dtype=pred_target.dtype,
            default=1.0,
        )
        loss_weights = torch.ones_like(pred_target)
        # pred_target covers domains 1..N (source excluded).
        # For 3-domain: [target_rgb | target_track] → split at W/2
        # For 4-domain: [target_rgb | target_track | target_depth] → split at W/3 boundaries
        if num_domains == 3:
            loss_weights[..., :split] = track_loss_mask.view(-1, 1, 1, 1, 1)
            loss_weights[..., split:] = track_video_loss_mask.view(-1, 1, 1, 1, 1)
        elif num_domains == 4:
            # 4-domain target: [target_rgb | target_track | target_depth]
            # Each domain is split-wide in the width dimension
            w_total = pred_target.shape[-1]
            w_third = w_total // 3
            # Domain 1 (target_rgb): always supervised → weight=1
            # Domain 2 (target_track): use track_video_loss_mask
            loss_weights[..., w_third:2*w_third] = track_video_loss_mask.view(-1, 1, 1, 1, 1)
            # Domain 3 (target_depth): use depth_loss_mask (has_depth flag)
            depth_loss_mask = _coerce_sample_weight_vector(
                inputs.get("depth_loss_mask"),
                batch=int(pred_target.shape[0]),
                device=pred_target.device,
                dtype=pred_target.dtype,
                default=1.0,
            )
            loss_weights[..., 2*w_third:] = depth_loss_mask.view(-1, 1, 1, 1, 1)
        sq_err = (pred_target - gt_target) ** 2
        diffusion_loss = sq_err.mul(loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
    else:
        # Per-latent-frame future reweight (full-shape weight so the normalizer
        # matches the numerator). Equals mse(pred, gt) when future prediction is
        # off or future_loss_weight == 1.0 → default path byte-identical.
        diffusion_loss = _future_weighted_mse(
            pred_target, gt_target, _future_observed,
            float(inputs.get("future_loss_weight", 1.0)),
        )
    diffusion_loss = diffusion_loss * pipe.scheduler.training_weight(timestep)

    # GS-only training path: zero out the FM MSE so it neither contributes to
    # the optimizer step nor to the logged total. DiT forward still ran above
    # so gs_features is populated for the splatting loss below.
    if _as_bool(inputs.get("skip_diffusion_loss", False)):
        diffusion_loss = diffusion_loss.detach() * 0.0

    # --- GS splatting loss (optional) ---
    use_gs_output = _as_bool(inputs.get("use_gs_output", False))
    extra_metrics = None
    if use_gs_output and gs_features is not None and hasattr(pipe, "gs_decoder") and pipe.gs_decoder is not None:
        from holo4d.models.gs_loss import compute_gs_splatting_loss

        # Extract target-domain GS features
        if v2v_concat_mode == "width":
            gs_features_target = gs_features[:, :, :, :, split:]
        else:  # frame
            gs_features_target = gs_features[:, :, split:]

        # Zero the feature stream going into the GS decoder when
        # gs_zero_features is set (via gs_config["zero_features"]).
        # This replicates the cache overfit's --feature_mode x0_mid
        # semantics (train_gs_head_from_cache.py:697-700 substitutes
        # zero features). Decoder output then depends purely on z
        # (x0-hat from latents_pred_target); extra_conv_in(0) stays 0
        # as long as extra_conv_in's weights receive no gradient, which
        # they don't because zero-input conv has zero input-gradient.
        _gs_config_preview = inputs.get("gs_config", {})
        if _as_bool(_gs_config_preview.get("zero_features", False)):
            gs_features_target = torch.zeros_like(gs_features_target)

        # Predict clean x0 from v_pred: x0 = noisy - sigma * v_pred
        # Wan flow-matching: timesteps = sigmas * 1000 (after shift), so sigma = t/1000 is exact.
        _sigma = timestep / 1000.0
        if v2v_concat_mode == "width":
            noise_pred_target = noise_pred[:, :, :, :, split:]
            noisy_target = noisy_latents[:, :, :, :, split:]
            latents_pred_target = noisy_target - _sigma.view(-1, 1, 1, 1, 1) * noise_pred_target
        else:
            noise_pred_target = noise_pred[:, :, split:]
            noisy_target = noisy_latents[:, :, split:]
            latents_pred_target = noisy_target - _sigma.view(-1, 1, 1, 1) * noise_pred_target

        gs_config = inputs.get("gs_config", {})
        _gs_detach_latents = bool(gs_config.get("detach_predicted_latents", True))
        target_cameras = inputs.get("target_cameras_flashworld")
        gt_target_rgb = inputs.get("gt_target_rgb")
        gt_target_depth = inputs.get("gt_target_depth")
        if gt_target_depth is None:
            gt_target_depth = inputs.get("target_depth_video")
        depth_valid_mask = inputs.get("depth_valid_mask")
        if depth_valid_mask is None:
            depth_valid_mask = inputs.get("target_depth_valid_mask")

        # Batch-size-1 training commonly provides unbatched tensors:
        #   target_cameras: (T, 11)
        #   gt_target_rgb: (3, T, H, W)
        # The GS loss path expects batch-aware shapes so it can align the
        # latent-time supervision with the target-camera timeline. Without this
        # coercion, the dim()==2 camera case skips temporal subsampling and can
        # pass the wrong number of cameras into the GS decoder.
        if target_cameras is not None and torch.is_tensor(target_cameras) and target_cameras.dim() == 2:
            target_cameras = target_cameras.unsqueeze(0)
        if gt_target_rgb is not None and torch.is_tensor(gt_target_rgb) and gt_target_rgb.dim() == 4:
            gt_target_rgb = gt_target_rgb.unsqueeze(0)
        # Depth comes from the dataset as (T, H, W) (no channel/batch).
        # The static GS loss path (gs_loss.py:304) reads shape[2] expecting
        # T, so we coerce up to (1, 1, T, H, W). dim==4 is the legacy path
        # where someone pre-batched but kept 4D layout (1, T, H, W).
        if gt_target_depth is not None and torch.is_tensor(gt_target_depth):
            if gt_target_depth.dim() == 3:
                gt_target_depth = gt_target_depth.unsqueeze(0).unsqueeze(0)
            elif gt_target_depth.dim() == 4:
                gt_target_depth = gt_target_depth.unsqueeze(0)
        if depth_valid_mask is not None and torch.is_tensor(depth_valid_mask):
            if depth_valid_mask.dim() == 3:
                depth_valid_mask = depth_valid_mask.unsqueeze(0).unsqueeze(0)
            elif depth_valid_mask.dim() == 4:
                depth_valid_mask = depth_valid_mask.unsqueeze(0)

        if target_cameras is not None and gt_target_rgb is not None:
            # Dispatch on dynamic_mode. Static is the legacy path; deltas
            # routes through gs_loss_dynamic.compute_gs_splatting_loss_dynamic
            # using the canonical+deltas wrapper attached to pipe.gs_decoder.
            _dynamic_mode = str(gs_config.get("dynamic_mode", "static")).lower()
            if _dynamic_mode == "static_with_track" and getattr(pipe.gs_decoder, "gs_head", None) is not None and getattr(pipe.gs_decoder.gs_head, "track_delta_proj", None) is not None:
                # v4: per-frame static rendering + per-pixel-per-frame
                # 3D delta supervised by tracks (NOT used in rendering).
                # Single decoder forward producing both gaussian_params
                # and track_delta_xyz, then split into photometric +
                # track loss paths. See gs_loss_v4_track.py and the
                # 2026-04-26-v4 doc for the full architecture rationale.
                from holo4d.models.gs_loss import compute_gs_splatting_loss
                from holo4d.models.gs_loss_v4_track import compute_v4_track_loss

                # Decoder forward with track_delta enabled.
                _z_in = (
                    latents_pred_target.detach()
                    if _gs_detach_latents else latents_pred_target
                )
                gaussian_params_v4, track_delta_xyz = pipe.gs_decoder(
                    gs_features_target, _z_in, target_cameras,
                    return_track_delta=True,
                )
                # Joint render path (mirrors the train-mode dispatch above).
                _joint_render = bool(gs_config.get("track_joint_render", False))
                if _joint_render:
                    from holo4d.models.gs_loss_v4_track import build_joint_gaussian_params
                    gaussian_params_for_render = build_joint_gaussian_params(
                        gaussian_params_v4, track_delta_xyz,
                    )
                else:
                    gaussian_params_for_render = gaussian_params_v4
                # Photometric / depth / opacity (reuse static path with
                # the pre-computed gaussian_params to avoid a redundant
                # decoder forward).
                gs_losses_photo = compute_gs_splatting_loss(
                    gs_decoder=pipe.gs_decoder,
                    gs_features_target=gs_features_target,
                    latents_pred_target=latents_pred_target,
                    target_cameras=target_cameras,
                    gt_target_rgb=gt_target_rgb,
                    gt_target_depth=gt_target_depth,
                    depth_valid_mask=depth_valid_mask,
                    image_height=int(gs_config.get("image_height", 288)),
                    image_width=int(gs_config.get("image_width", 512)),
                    mse_loss_weight=float(gs_config.get("mse_loss_weight", gs_config.get("rgb_loss_weight", 1.0))),
                    lpips_loss_weight=float(gs_config.get("lpips_loss_weight", 0.5)),
                    depth_loss_weight=float(gs_config.get("depth_loss_weight", 0.0)),
                    abs_depth_loss_weight=float(gs_config.get("abs_depth_loss_weight", 0.0)),
                    opacity_loss_weight=float(gs_config.get("opacity_loss_weight", 0.1)),
                    detach_predicted_latents=_gs_detach_latents,
                    gaussian_params=gaussian_params_for_render,
                    opacity_prune_keep_frac=float(gs_config.get("opacity_prune_keep_frac", 1.0)),
                )
                # Track loss (zero when GT tracks absent).
                gs_losses_track = compute_v4_track_loss(
                    canonical_params=gaussian_params_v4[:, 0],
                    track_delta_xyz=track_delta_xyz,
                    target_cameras=target_cameras,
                    image_height=int(gs_config.get("image_height", 288)),
                    image_width=int(gs_config.get("image_width", 512)),
                    target_tracks_world=inputs.get("target_tracks_world"),
                    target_tracks_world_valid_mask=inputs.get("target_tracks_world_valid_mask"),
                    target_track_video=inputs.get("target_track_video"),
                    target_track_valid_mask=inputs.get("target_track_valid_mask"),
                    track_canonical_loss_weight=float(gs_config.get("track_canonical_loss_weight", 1.0)),
                    track_delta_loss_weight=float(gs_config.get("track_delta_loss_weight", 1.0)),
                    track_final_pos_loss_weight=float(gs_config.get("track_final_pos_loss_weight", 1.0)),
                    assoc_track_2d_loss_weight=float(gs_config.get("assoc_track_2d_loss_weight", 5.0)),
                    track_accel_loss_weight=float(gs_config.get("track_accel_loss_weight", 0.5)),
                    track_loss_quantile=float(gs_config.get("track_loss_quantile", 0.98)),
                )
                gs_losses = {
                    **gs_losses_photo,
                    **gs_losses_track,
                }
                gs_losses["gs_loss"] = (
                    gs_losses_photo["gs_loss"] + gs_losses_track["gs_track_total_loss"]
                )
            elif _dynamic_mode == "deltas" and getattr(pipe.gs_decoder, "dynamic", None) is not None:
                from holo4d.models.gs_loss_dynamic import compute_gs_splatting_loss_dynamic

                target_track_video = inputs.get("target_track_video")
                target_track_valid_mask = inputs.get("target_track_valid_mask")
                # GT depth: prefer the dataset's target_depth_video / mask
                # (these are the same keys the static path's depth fallback
                # already uses, see lines 657-661).
                target_depth_video_dyn = inputs.get("target_depth_video")
                if target_depth_video_dyn is None:
                    target_depth_video_dyn = inputs.get("gt_target_depth")
                target_depth_valid_mask_dyn = inputs.get("target_depth_valid_mask")
                if target_depth_valid_mask_dyn is None:
                    target_depth_valid_mask_dyn = inputs.get("depth_valid_mask")
                if target_depth_video_dyn is not None and torch.is_tensor(target_depth_video_dyn) and target_depth_video_dyn.dim() == 3:
                    target_depth_video_dyn = target_depth_video_dyn.unsqueeze(0)
                if target_depth_valid_mask_dyn is not None and torch.is_tensor(target_depth_valid_mask_dyn) and target_depth_valid_mask_dyn.dim() == 3:
                    target_depth_valid_mask_dyn = target_depth_valid_mask_dyn.unsqueeze(0)
                # Persistent-track supervision (cache v3 recipe) — pass
                # target_tracks_world + query idx + track3d weights so
                # `persistent_tracks` mode is actually active when
                # gs_config["gs_track_supervision_mode"] == "persistent_tracks".
                # Defaults preserve the old pointmap-only behaviour.
                target_tracks_world_dyn = inputs.get("target_tracks_world")
                target_tracks_world_valid_dyn = inputs.get("target_tracks_world_valid_mask")
                target_tracks_query_xy_dyn = inputs.get("track_query_idx_ds")
                # Pull dynamic-loss weights out of gs_config (set by training_module).
                gs_losses = compute_gs_splatting_loss_dynamic(
                    gs_decoder_dynamic=pipe.gs_decoder.dynamic,
                    gs_features_target=gs_features_target,
                    latents_pred_target=latents_pred_target,
                    target_cameras=target_cameras,
                    gt_target_rgb=gt_target_rgb,
                    target_track_video=target_track_video,
                    target_track_valid_mask=target_track_valid_mask,
                    target_tracks_world=target_tracks_world_dyn,
                    target_tracks_world_valid_mask=target_tracks_world_valid_dyn,
                    target_tracks_query_xy=target_tracks_query_xy_dyn,
                    target_depth_video=target_depth_video_dyn,
                    target_depth_valid_mask=target_depth_valid_mask_dyn,
                    image_height=int(gs_config.get("image_height", 288)),
                    image_width=int(gs_config.get("image_width", 512)),
                    rgb_loss_weight=float(gs_config.get("mse_loss_weight", gs_config.get("rgb_loss_weight", 1.0))),
                    pos_loss_weight=float(gs_config.get("pos_loss_weight", 1.0)),
                    depth_loss_weight=float(gs_config.get("dyn_depth_loss_weight", 0.5)),
                    opa_tv_weight=float(gs_config.get("opa_tv_weight", 0.01)),
                    xyz_tv_weight=float(gs_config.get("xyz_tv_weight", 0.01)),
                    gs_track_supervision_mode=str(gs_config.get("gs_track_supervision_mode", "pointmap")),
                    track_canonical_loss_weight=float(gs_config.get("track_canonical_loss_weight", 0.0)),
                    track_delta_loss_weight=float(gs_config.get("track_delta_loss_weight", 0.0)),
                    track_final_pos_loss_weight=float(gs_config.get("track_final_pos_loss_weight", 0.0)),
                    assoc_track_2d_loss_weight=float(gs_config.get("assoc_track_2d_loss_weight", 0.0)),
                    track_accel_loss_weight=float(gs_config.get("track_accel_loss_weight", 0.0)),
                    track_loss_quantile=float(gs_config.get("track_loss_quantile", 0.98)),
                    predict_rotation=bool(gs_config.get("predict_rotation", False)),
                    detach_predicted_latents=_gs_detach_latents,
                )
            else:
                gs_losses = compute_gs_splatting_loss(
                    gs_decoder=pipe.gs_decoder,
                    gs_features_target=gs_features_target,
                    latents_pred_target=latents_pred_target,
                    target_cameras=target_cameras,
                    gt_target_rgb=gt_target_rgb,
                    gt_target_depth=gt_target_depth,
                    depth_valid_mask=depth_valid_mask,
                    image_height=int(gs_config.get("image_height", 288)),
                    image_width=int(gs_config.get("image_width", 512)),
                    mse_loss_weight=float(gs_config.get("mse_loss_weight", gs_config.get("rgb_loss_weight", 1.0))),
                    lpips_loss_weight=float(gs_config.get("lpips_loss_weight", 0.5)),
                    depth_loss_weight=float(gs_config.get("depth_loss_weight", 0.0)),
                    abs_depth_loss_weight=float(gs_config.get("abs_depth_loss_weight", 0.0)),
                    opacity_loss_weight=float(gs_config.get("opacity_loss_weight", 0.1)),
                    detach_predicted_latents=_gs_detach_latents,
                    opacity_prune_keep_frac=float(gs_config.get("opacity_prune_keep_frac", 1.0)),
                )
            gs_loss_weight = float(gs_config.get("gs_loss_weight", 0.1))
            weighted_gs_loss = gs_loss_weight * gs_losses["gs_loss"]
            extra_metrics = {
                "loss_diffusion": diffusion_loss.detach(),
                "loss_gs_weighted": weighted_gs_loss.detach(),
                "gs_loss_weight": torch.tensor(float(gs_loss_weight), device=diffusion_loss.device),
            }
            for key, value in gs_losses.items():
                if torch.is_tensor(value):
                    extra_metrics[key] = value.detach()
                else:
                    extra_metrics[key] = torch.tensor(float(value), device=diffusion_loss.device)
            diffusion_loss = diffusion_loss + gs_loss_weight * gs_losses["gs_loss"]

            # GS replay-cache snapshot (overfit accelerator). Stash a CPU
            # copy of the GS-loss inputs from this step so future steps can
            # skip the DiT forward and replay them. Once the cache hits
            # `replay_cache_size`, FlowMatchV2VLoss short-circuits at the
            # top into _replay_gs_only_loss. No-op when replay_cache_size=0
            # or the cache is already full.
            _replay_cap_save = int(gs_config.get("replay_cache_size", 0))
            if _replay_cap_save > 0 and len(_GS_REPLAY_CACHE) < _replay_cap_save:
                # Stash the GS-loss inputs PLUS the pre-head DiT hidden state
                # (gs_pre_head_input/t/shape). On replay we re-run
                # `dit.gs_feature_head(cached_input, cached_t)` and
                # unpatchify to regenerate gs_features, so gs_feature_head's
                # parameters see real gradient on every step instead of
                # staying frozen behind a cached post-head tensor.
                # gs_pre_head_input is `x.detach()` when the DiT backbone is
                # frozen (the common case here), so caching is safe.
                _snap = {
                    "gs_features_target": gs_features_target,  # fallback / shape ref
                    "latents_pred_target": latents_pred_target,
                    "target_cameras": target_cameras,
                    "gt_target_rgb": gt_target_rgb,
                    "gt_target_depth": gt_target_depth,
                    "depth_valid_mask": depth_valid_mask,
                    "target_track_video": inputs.get("target_track_video"),
                    "target_track_valid_mask": inputs.get("target_track_valid_mask"),
                    "target_tracks_world": inputs.get("target_tracks_world"),
                    "target_tracks_world_valid_mask": inputs.get("target_tracks_world_valid_mask"),
                    "target_tracks_query_xy": inputs.get("track_query_idx_ds"),
                    "target_depth_video": inputs.get("target_depth_video"),
                    "target_depth_valid_mask": inputs.get("target_depth_valid_mask"),
                    "gs_pre_head_input": gs_pre_head_input,
                    "gs_pre_head_t": gs_pre_head_t,
                    "gs_unpatchify_shape": gs_unpatchify_shape,
                    "v2v_concat_mode": str(v2v_concat_mode),
                    "split": int(split),
                }
                _gs_cache_snapshot(_snap, _replay_cap_save)
                extra_metrics["gs_replay_cache_size"] = torch.tensor(
                    float(len(_GS_REPLAY_CACHE)), device=diffusion_loss.device,
                )

    motion_slice = _motion_target_slice(
        v2v_concat_mode=v2v_concat_mode,
        input_latents=input_latents,
        num_domains=num_domains,
    )
    maybe_loss = _apply_explicit_motion_loss_if_enabled(
        diffusion_loss,
        pipe=pipe,
        inputs=inputs,
        timestep_id=timestep_id,
        noisy_latents=noisy_latents,
        noise_pred=noise_pred,
        motion_slice=motion_slice,
    )
    if isinstance(maybe_loss, tuple):
        total_loss, motion_metrics = maybe_loss
        if extra_metrics is not None:
            extra_metrics = {**extra_metrics, **motion_metrics}
            extra_metrics["loss_total"] = total_loss.detach()
            return total_loss, extra_metrics
        return total_loss, motion_metrics
    if extra_metrics is not None:
        extra_metrics["loss_total"] = maybe_loss.detach()
        return maybe_loss, extra_metrics
    return maybe_loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
