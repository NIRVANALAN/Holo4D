from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import imageio
import cv2


@dataclass(frozen=True)
class V2VSlices:
    cond_latents: torch.Tensor
    target_latents: torch.Tensor
    cond_slice: Tuple[Any, ...]
    target_slice: Tuple[Any, ...]
    # For 3-way split (multi-modality):
    # - NVS:      target_rgb_latents=target_rgb, target_depth_latents=target_depth
    # - Tracking: target_rgb_latents=target_track, target_depth_latents=target_track_video
    target_rgb_latents: Optional[torch.Tensor] = None
    target_depth_latents: Optional[torch.Tensor] = None
    target_rgb_slice: Optional[Tuple[Any, ...]] = None
    target_depth_slice: Optional[Tuple[Any, ...]] = None
    num_domains: int = 2


def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_mp4(frames, save_path: str, fps: int = 16, quality: int = 5) -> None:
    writer = imageio.get_writer(save_path, fps=int(fps), quality=int(quality))
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def _maybe_truncate_frames(frames, max_frames: Optional[int]):
    if max_frames is None:
        return frames
    max_frames = int(max_frames)
    if max_frames <= 0:
        return frames[:0]
    return frames[:max_frames]


def _extract_noise_pred(model_out: Any) -> torch.Tensor:
    """Normalize model_fn outputs for debug visualizations.

    GS-enabled V2V runs return ``{"noise_pred": ..., "gs_features": ...}``,
    while the legacy path returns the tensor directly.
    """
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, dict):
        noise_pred = model_out.get("noise_pred")
        if torch.is_tensor(noise_pred):
            return noise_pred
        raise TypeError(
            "model_fn returned a dict without tensor 'noise_pred': "
            f"{sorted(model_out.keys())}"
        )
    raise TypeError(f"Unsupported model_fn output type for debug latents: {type(model_out)!r}")


def _build_column_labels(
    *,
    v2v_concat_mode: str,
    multimodality_nvs: bool,
    multimodality_tracking: bool,
    has_xyz_cond: bool,
    has_query_impulse: bool,
    has_absolute_track: bool,
    has_depth_secondary: bool = False,
) -> List[str]:
    if v2v_concat_mode == "channel":
        labels = ["x0_gt"]
        if has_xyz_cond:
            labels.append("xyz_cond")
        if has_query_impulse:
            labels.append("query_impulse")
        labels.extend(["xt", "x0_hat"])
        return labels

    labels = ["src_rgb"]
    if has_xyz_cond:
        labels.append("xyz_cond")
    if has_query_impulse:
        labels.append("query_impulse")

    if multimodality_nvs:
        labels.extend(
            [
                "tgt_rgb_gt",
                "tgt_depth_gt",
                "tgt_rgb_noisy",
                "tgt_depth_noisy",
                "tgt_rgb_denoised",
                "tgt_depth_denoised",
            ]
        )
        return labels

    if multimodality_tracking:
        sec_gt = "depth_gt" if has_depth_secondary else "tgt_track_video_gt"
        sec_noisy = "depth_noisy" if has_depth_secondary else "tgt_track_video_noisy"
        sec_denoised = "depth_denoised" if has_depth_secondary else "tgt_track_video_denoised"
        if has_absolute_track:
            labels.extend(
                [
                    "tgt_track_residual_gt",
                    "tgt_track_residual_noisy",
                    "tgt_track_residual_denoised",
                    sec_gt,
                    sec_noisy,
                    sec_denoised,
                    "tgt_track_absolute_gt",
                ]
            )
        else:
            labels.extend(
                [
                    "tgt_track_gt",
                    "tgt_track_noisy",
                    "tgt_track_denoised",
                    sec_gt,
                    sec_noisy,
                    sec_denoised,
                ]
            )
        return labels

    if has_absolute_track:
        labels.extend(
            [
                "tgt_residual_gt",
                "tgt_residual_noisy",
                "tgt_residual_denoised",
                "tgt_absolute_gt",
            ]
        )
    else:
        labels.extend(["tgt_gt", "tgt_noisy", "tgt_denoised"])
    return labels


def _draw_column_captions(frame: Image.Image, labels: List[str]) -> Image.Image:
    if not labels:
        return frame
    w, h = frame.size
    n = len(labels)
    if n <= 1 or w < n:
        return frame

    footer_h = max(20, min(40, h // 9))
    out = Image.new("RGB", (w, h + footer_h), color=(0, 0, 0))
    out.paste(frame, (0, 0))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    col_w = w / float(n)
    def _text_size(text: str) -> tuple[int, int]:
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        return draw.textsize(text, font=font)

    for i, raw in enumerate(labels):
        x0 = int(round(i * col_w))
        x1 = int(round((i + 1) * col_w))
        if i > 0:
            draw.line([(x0, h), (x0, h + footer_h)], fill=(80, 80, 80), width=1)
        text = str(raw).replace("_", " ")
        max_text_w = max(12, (x1 - x0) - 8)
        while len(text) > 3:
            tw, _th = _text_size(text)
            if tw <= max_text_w:
                break
            text = text[:-4] + "..."
        tw, th = _text_size(text)
        tx = x0 + max(2, ((x1 - x0) - tw) // 2)
        ty = h + max(1, (footer_h - th) // 2)
        draw.text((tx, ty), text, fill=(230, 230, 230), font=font)
    return out


def _colorize_depth_frame(arr: np.ndarray) -> np.ndarray:
    """
    Apply Spectral_r colormap to a single decoded depth frame (H, W, 3).
    Since depth is stored as 3-channel replicated, average channels first.
    """
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    depth = arr.astype(np.float32).mean(axis=2)  # (H, W)
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)
    return (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)


def _colorize_depth_frames(frames: List[Image.Image]) -> List[Image.Image]:
    """Apply Spectral_r depth colorization with shared global min/max across frames.

    Per-frame normalization causes jitter when scene depth range varies — a
    background pixel mapped to red in frame t becomes orange in frame t+1.
    Compute one (d_min, d_max) over all frames so colors are temporally stable.
    """
    import matplotlib
    if not frames:
        return []
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    arrs = [np.asarray(f, dtype=np.float32).mean(axis=2) for f in frames]
    stacked = np.stack(arrs, axis=0)  # (T, H, W)
    d_min = float(stacked.min())
    d_max = float(stacked.max())
    out: List[Image.Image] = []
    for d in arrs:
        if d_max > d_min:
            n = (d - d_min) / (d_max - d_min)
        else:
            n = np.zeros_like(d)
        rgb = (cmap(n)[:, :, :3] * 255).astype(np.uint8)
        out.append(Image.fromarray(rgb))
    return out


def _colorize_depth_tensor(depth: torch.Tensor) -> List[np.ndarray]:
    """Colorize a (T, 1, H, W) or (T, H, W) depth tensor to Spectral_r RGB frames."""
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    if depth.dim() == 4:
        depth = depth[:, 0]
    depth_np = depth.detach().float().cpu().numpy()
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        depth_norm = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_np)
    frames = []
    for t in range(depth_norm.shape[0]):
        rgb = (cmap(depth_norm[t])[:, :, :3] * 255).astype(np.uint8)
        frames.append(rgb)
    return frames


def _colorize_depth_tensor_masked(depth: torch.Tensor, valid_mask: torch.Tensor) -> List[np.ndarray]:
    """Colorize depth and zero the background where the mask is invalid."""
    depth = torch.as_tensor(depth)
    valid_mask = torch.as_tensor(valid_mask)
    frames = _colorize_depth_tensor(depth)
    if valid_mask.dim() == 4:
        valid_mask = valid_mask[:, 0]
    mask_np = valid_mask.detach().bool().cpu().numpy()
    out = []
    for frame, mask in zip(frames, mask_np):
        frame = frame.copy()
        frame[~mask] = 0
        out.append(frame)
    return out


def _opacity_to_grayscale(opacity: torch.Tensor) -> List[np.ndarray]:
    """Convert (T, npp, H, W) or (T, H, W) opacity to grayscale RGB frames."""
    if opacity.dim() == 4:
        opacity = opacity.mean(dim=1)
    opa_np = opacity.detach().float().cpu().clamp(0, 1).numpy()
    frames = []
    for t in range(opa_np.shape[0]):
        gray = (opa_np[t] * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        frames.append(rgb)
    return frames


def _as_depth_video(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x.detach().cpu()
    else:
        t = torch.as_tensor(np.asarray(x))
    if t.dim() == 5:
        if t.shape[1] == 1:
            t = t[:, 0]
        elif t.shape[-1] == 1:
            t = t[..., 0]
        else:
            t = t[:, 0]
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3:
        return None
    return t.float()


def _as_mask_video(x: Any) -> Optional[torch.Tensor]:
    t = _as_depth_video(x)
    if t is None:
        return None
    return t > 0.5


def _as_camera_sequence(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        arr = x.detach().cpu().float().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
        return None
    return arr.astype(np.float32, copy=False)


def _as_intrinsics(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        arr = x.detach().cpu().float().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape != (3, 3):
        return None
    return arr.astype(np.float32, copy=False)


def _colorize_depth_with_valid(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from holo4d.visualization.hybrid_panels import colorize_depth

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth frame, got {depth.shape}")
    if valid.shape != depth.shape:
        raise ValueError(f"Depth/valid shape mismatch: {depth.shape} vs {valid.shape}")
    return colorize_depth(depth, valid)


@torch.no_grad()
def _build_training_depth_panels(inputs_shared: dict) -> dict[str, List[np.ndarray]]:
    target_depth = _as_depth_video(
        inputs_shared.get("target_depth_video", inputs_shared.get("gt_target_depth"))
    )
    if target_depth is None:
        # Diagnostic so missing-panel bugs are visible in the train log.
        # Lists the depth-related keys we tried so a stale routing path
        # is obvious. Cheap (one print per debug-latents step).
        depth_keys_present = [
            k for k in (
                "target_depth_video", "target_depth_valid_mask",
                "gt_target_depth", "depth_valid_mask",
            ) if k in inputs_shared
        ]
        print(
            "[GS debug] tgt_depth panel skipped: target_depth_video / "
            f"gt_target_depth not in inputs_shared. Present depth keys: "
            f"{depth_keys_present or '(none)'}"
        )
        return {}

    target_valid = _as_mask_video(
        inputs_shared.get("target_depth_valid_mask", inputs_shared.get("depth_valid_mask"))
    )
    if target_valid is None or target_valid.shape != target_depth.shape:
        target_valid = torch.isfinite(target_depth) & (target_depth > 1e-6)

    target_depth_frames = [
        _colorize_depth_with_valid(target_depth[t].numpy(), target_valid[t].numpy())
        for t in range(target_depth.shape[0])
    ]

    source_depth = _as_depth_video(inputs_shared.get("source_depth_video"))
    source_cam2world = _as_camera_sequence(inputs_shared.get("source_cam2world"))
    source_intrinsics = _as_intrinsics(inputs_shared.get("source_intrinsics"))
    target_cam2world = _as_camera_sequence(inputs_shared.get("target_cam2world"))
    target_intrinsics = _as_intrinsics(inputs_shared.get("target_intrinsics"))

    result = {"target depth": target_depth_frames}
    if (
        source_depth is None
        or source_cam2world is None
        or source_intrinsics is None
        or target_cam2world is None
        or target_intrinsics is None
    ):
        return result

    from holo4d.visualization.track_panels import forward_warp_rgbd_to_target

    T = min(
        int(target_depth.shape[0]),
        int(source_depth.shape[0]),
        int(source_cam2world.shape[0]),
        int(target_cam2world.shape[0]),
    )
    projected_depth_frames: List[np.ndarray] = []
    for t in range(T):
        h, w = source_depth[t].shape[-2:]
        dummy_rgb = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        _warped_rgb, warped_depth, warped_valid = forward_warp_rgbd_to_target(
            src_rgb=dummy_rgb,
            src_depth=source_depth[t].numpy(),
            src_cam2world=source_cam2world[t],
            src_intrinsics=source_intrinsics,
            tgt_cam2world=target_cam2world[t],
            tgt_intrinsics=target_intrinsics,
            src_cam0_pose=None,
            common_scale=1.0,
        )
        projected_depth_frames.append(
            _colorize_depth_with_valid(warped_depth, warped_valid)
        )
    if projected_depth_frames:
        result["projected depth"] = projected_depth_frames
    return result


@torch.no_grad()
def _build_dense_warp_panels(inputs_shared: dict) -> dict[str, List[np.ndarray]]:
    """Build dense forward-warped RGB / depth / valid visualization panels.

    Replicates the bottom row of the eval ``comparison.mp4`` so the same
    geometry can be inspected during training without running inference.

    Uses ``source_rgb_video``, ``source_depth_video``, ``source_cam2world``,
    ``source_intrinsics``, ``target_cam2world``, and ``target_intrinsics``
    from ``inputs_shared``.  All of these are present whenever
    ``WanVideoUnit_DenseTargetControl`` is active in the pipeline (i.e. when
    ``--use_dense_branch`` is set and the dataset emits the corresponding
    fields).

    Returns a dict mapping panel name → list of ``(H, W, 3)`` uint8 numpy
    frames.  Returns an empty dict when any required input is absent or when
    the warp fails.
    """
    source_depth = _as_depth_video(inputs_shared.get("source_depth_video"))
    source_c2w = _as_camera_sequence(inputs_shared.get("source_cam2world"))
    source_K = _as_intrinsics(inputs_shared.get("source_intrinsics"))
    target_c2w = _as_camera_sequence(inputs_shared.get("target_cam2world"))
    target_K = _as_intrinsics(inputs_shared.get("target_intrinsics"))

    if any(x is None for x in (source_depth, source_c2w, source_K, target_c2w, target_K)):
        return {}

    src_rgb_raw = inputs_shared.get("source_rgb_video")
    if src_rgb_raw is None:
        return {}

    try:
        if isinstance(src_rgb_raw, (list, tuple)):
            # List of PIL images or numpy arrays
            src_rgb_np = np.stack(
                [np.asarray(f, dtype=np.float32) for f in src_rgb_raw], axis=0
            )  # (T, H, W, 3)
        elif torch.is_tensor(src_rgb_raw):
            t = src_rgb_raw.detach().cpu().float()
            if t.ndim == 5 and t.shape[1] == 3:     # (B, 3, T, H, W)
                t = t[0].permute(1, 2, 3, 0)
            elif t.ndim == 5:                         # (B, T, H, W, 3)
                t = t[0]
            # t is now (T, H, W, 3)
            if float(t.max()) > 2.0:
                t = t / 255.0
            src_rgb_np = t.clamp(0.0, 1.0).numpy()
        else:
            src_rgb_np = np.asarray(src_rgb_raw, dtype=np.float32)
            if src_rgb_np.ndim == 5:                  # (B, T, H, W, 3) or (B, 3, T, H, W)
                if src_rgb_np.shape[-1] == 3:
                    src_rgb_np = src_rgb_np[0]
                else:
                    src_rgb_np = src_rgb_np[0].transpose(1, 2, 3, 0)
            if float(src_rgb_np.max()) > 2.0:
                src_rgb_np = src_rgb_np / 255.0
    except Exception:
        return {}

    from holo4d.visualization.track_panels import forward_warp_rgbd_to_target
    from holo4d.visualization.hybrid_panels import colorize_depth, colorize_mask

    T = min(
        int(src_rgb_np.shape[0]),
        int(source_depth.shape[0]),
        int(source_c2w.shape[0]),
        int(target_c2w.shape[0]),
    )

    # Honor inputs_shared["dense_warp_mode"] so the panel reflects the same
    # backend the trainer feeds into the dense encoder. "hybrid_softsplat"
    # → multi-frame Lyra-2 / GEN3C soft-splat (the default for new runs).
    # "zbuffer" or anything else → fall back to the legacy per-frame path.
    dense_warp_mode = str(inputs_shared.get("dense_warp_mode", "zbuffer")).lower()
    label_suffix = f" ({dense_warp_mode})"

    if dense_warp_mode == "hybrid_softsplat":
        try:
            from holo4d.conditions.dense_target_reprojection_builder import (
                build_dense_target_reprojection_hybrid,
            )
        except Exception as e:
            print(f"[debug_latents] hybrid_softsplat backend unavailable, falling back to per-frame: {e}")
            dense_warp_mode = "zbuffer"

    if dense_warp_mode == "hybrid_softsplat":
        # Pack inputs as (B=1, T, H, W, ...) tensors. The hybrid builder does
        # the multi-frame fusion internally — we just consume its outputs.
        H = int(src_rgb_np.shape[1])
        W = int(src_rgb_np.shape[2])
        src_rgb_t = torch.from_numpy(src_rgb_np[:T]).unsqueeze(0).float()      # (1, T, H, W, 3) [0,1]
        src_depth_t = source_depth[:T].unsqueeze(0).float()                    # (1, T, H, W)
        src_valid_t = (torch.isfinite(src_depth_t) & (src_depth_t > 1e-6)).bool()
        src_c2w_t = torch.from_numpy(np.asarray(source_c2w[:T])).unsqueeze(0).float()  # (1, T, 4, 4)
        tgt_c2w_t = torch.from_numpy(np.asarray(target_c2w[:T])).unsqueeze(0).float()
        K_t = torch.from_numpy(np.asarray(source_K)).float()                   # (3, 3) — broadcast inside builder

        # The hybrid builder accepts source_static_mask / target_dynamic_mask.
        # We don't have tracks_world here at viz time, so we pass None — the
        # builder treats all pixels as static, which mirrors the per-frame
        # zbuffer fallback for static datasets and is exactly the behavior
        # `visualize_multiframe_warp.py` shows when tracks aren't provided.
        try:
            warped_rgb_BTHWC, warped_depth_BTHW, warp_valid_BTHW = build_dense_target_reprojection_hybrid(
                source_rgb=src_rgb_t,
                source_depth=src_depth_t,
                source_depth_valid=src_valid_t,
                src_cam2world=src_c2w_t,
                tgt_cam2world=tgt_c2w_t,
                intrinsics=K_t,
                image_h=H,
                image_w=W,
                source_static_mask=None,
                target_dynamic_mask=None,
            )
        except Exception as e:
            print(f"[debug_latents] hybrid_softsplat call failed, falling back to per-frame: {e}")
            dense_warp_mode = "zbuffer"
            label_suffix = " (zbuffer-fallback)"

    if dense_warp_mode == "hybrid_softsplat":
        # warped_rgb_BTHWC: (1, T, H, W, 3) float32 [0, 1]
        # warped_depth_BTHW: (1, T, H, W) float32
        # warp_valid_BTHW:   (1, T, H, W) bool
        warped_rgb_np = (warped_rgb_BTHWC[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        warped_depth_np = warped_depth_BTHW[0].cpu().numpy()
        warped_valid_np = warp_valid_BTHW[0].cpu().numpy().astype(bool)
        rgb_frames = [warped_rgb_np[t] for t in range(T)]
        depth_frames = [colorize_depth(warped_depth_np[t], warped_valid_np[t]) for t in range(T)]
        valid_frames = [colorize_mask(warped_valid_np[t]) for t in range(T)]
    else:
        rgb_frames: List[np.ndarray] = []
        depth_frames: List[np.ndarray] = []
        valid_frames: List[np.ndarray] = []
        for t in range(T):
            src_rgb_t_uint8 = (src_rgb_np[t] * 255.0).clip(0.0, 255.0).astype(np.uint8)
            warped_rgb, warped_depth, warped_valid = forward_warp_rgbd_to_target(
                src_rgb=src_rgb_t_uint8,
                src_depth=source_depth[t].numpy(),
                src_cam2world=source_c2w[t],
                src_intrinsics=source_K,
                tgt_cam2world=target_c2w[t],
                tgt_intrinsics=target_K,
                src_cam0_pose=None,
                common_scale=1.0,
            )
            rgb_frames.append(warped_rgb)
            depth_frames.append(colorize_depth(warped_depth, warped_valid))
            valid_frames.append(colorize_mask(warped_valid))

    if not rgb_frames:
        return {}

    return {
        f"dense warped RGB{label_suffix}": rgb_frames,
        f"dense warped depth{label_suffix}": depth_frames,
        f"dense valid{label_suffix}": valid_frames,
    }


def _to_numpy_rgb_frames(frames: Any) -> List[np.ndarray]:
    out = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            out.append(np.asarray(frame.convert("RGB")))
        else:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            out.append(arr.astype(np.uint8, copy=False))
    return out


def _compute_track_colors(tracks_world: np.ndarray) -> np.ndarray:
    pts0 = tracks_world[:, 0, :]
    valid0 = np.any(pts0 != 0, axis=-1)
    colors = np.full((tracks_world.shape[0], 3), 255, dtype=np.uint8)
    if valid0.any():
        vmin = np.percentile(pts0[valid0], 2, axis=0)
        vmax = np.percentile(pts0[valid0], 98, axis=0)
        rng = np.clip(vmax - vmin, 1e-6, None)
        colors = np.clip((pts0 - vmin) / rng, 0, 1)
        colors = (colors * 255).astype(np.uint8)
    return colors


def _project_world_to_image(
    pts_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.any(pts_world != 0, axis=-1)
    if not valid.any():
        return np.zeros((pts_world.shape[0], 2), dtype=np.float32), np.zeros(pts_world.shape[0], dtype=bool)
    w2c = np.linalg.inv(c2w.astype(np.float32))
    pts_h = np.concatenate(
        [pts_world.astype(np.float32), np.ones((pts_world.shape[0], 1), dtype=np.float32)],
        axis=-1,
    )
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid_proj = valid & (z > 1e-6)
    uv = np.zeros((pts_world.shape[0], 2), dtype=np.float32)
    uv[valid_proj, 0] = K[0, 0] * pts_cam[valid_proj, 0] / z[valid_proj] + K[0, 2]
    uv[valid_proj, 1] = K[1, 1] * pts_cam[valid_proj, 1] / z[valid_proj] + K[1, 2]
    return uv, valid_proj


def _render_track_overlay_video(
    tracks_world: Any,
    target_cam2world: Any,
    target_intrinsics: Any,
    bg_frames: Any,
    track_window: int = 20,
) -> List[np.ndarray]:
    """Project 3D tracks to each target frame and draw short trajectories."""
    tracks_world = np.asarray(torch.as_tensor(tracks_world).detach().cpu(), dtype=np.float32)
    target_cam2world = np.asarray(torch.as_tensor(target_cam2world).detach().cpu(), dtype=np.float32)
    target_intrinsics = np.asarray(torch.as_tensor(target_intrinsics).detach().cpu(), dtype=np.float32)
    bg_frames = _to_numpy_rgb_frames(bg_frames)

    if tracks_world.ndim != 3 or target_cam2world.ndim != 3:
        return []

    H, W = bg_frames[0].shape[:2]
    colors = _compute_track_colors(tracks_world)
    overlays = []
    for t in range(min(len(bg_frames), tracks_world.shape[1], target_cam2world.shape[0])):
        canvas = bg_frames[t].copy()
        t0 = max(0, t - int(track_window) + 1)
        for n in range(tracks_world.shape[0]):
            seg = tracks_world[n, t0:t + 1]
            uv, valid = _project_world_to_image(seg, target_cam2world[t], target_intrinsics)
            prev = None
            color = tuple(int(c) for c in colors[n].tolist())
            for p, is_valid in zip(uv, valid):
                if not is_valid:
                    prev = None
                    continue
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
                if 0 <= x < W and 0 <= y < H:
                    if prev is not None:
                        cv2.line(canvas, prev, (x, y), color, 1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
                    prev = (x, y)
                else:
                    prev = None
        overlays.append(canvas)
    return overlays


@torch.no_grad()
def _extract_dynamic_gs_tracks_world(
    pipe,
    gs_features: torch.Tensor,
    x0_hat: torch.Tensor,
    inputs_shared: dict,
    v2v_concat_mode: str,
    fg_mask_frame0: Optional[np.ndarray] = None,
    cell_stride: int = 2,
    opacity_thresh: float = 0.05,
) -> Optional[np.ndarray]:
    """Sample a sparse set of dynamic-GS trajectories for 2D overlay debugging."""
    gs_decoder = getattr(pipe, "gs_decoder", None)
    if gs_decoder is None or getattr(gs_decoder, "dynamic", None) is None:
        return None

    from holo4d.models.gs_loss_dynamic import _compose_frame_gaussians

    if v2v_concat_mode == "width":
        split = gs_features.shape[-1] // 2
        gs_features_target = gs_features[:, :, :, :, split:]
        latents_target = x0_hat[:, :, :, :, split:]
    else:
        split = gs_features.shape[2] // 2
        gs_features_target = gs_features[:, :, split:]
        latents_target = x0_hat[:, :, split:]

    target_cameras = inputs_shared.get("target_cameras_flashworld")
    if target_cameras is None:
        return None
    if target_cameras.dim() == 2:
        target_cameras = target_cameras.unsqueeze(0)
    t_video = int(target_cameras.shape[1])
    gs_cfg = inputs_shared.get("gs_config", {}) or {}
    disable_delta_opacity = bool(gs_cfg.get("disable_delta_opacity", False))

    canonical_params, delta_params = gs_decoder.dynamic(
        gs_features_target,
        latents_target.detach(),
        target_cameras[:, 0],
        t_video,
    )
    full0, opa0, _ = _compose_frame_gaussians(
        canonical_params,
        delta_params,
        gs_decoder.dynamic.predict_rotation,
        disable_delta_opacity,
        0,
    )
    opa0 = opa0[0]
    best_idx = opa0.argmax(dim=0)
    best_opa = opa0.max(dim=0).values
    H, W = best_idx.shape

    fg_small = np.ones((H, W), dtype=bool)
    if fg_mask_frame0 is not None:
        fg_small = np.array(
            Image.fromarray(fg_mask_frame0.astype(np.uint8) * 255).resize((W, H), resample=Image.NEAREST)
        ) > 0

    ys = np.arange(0, H, max(int(cell_stride), 1))
    xs = np.arange(0, W, max(int(cell_stride), 1))
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    keep = fg_small[grid_y, grid_x] & (best_opa.detach().cpu().numpy()[grid_y, grid_x] > float(opacity_thresh))
    if not np.any(keep):
        return None
    grid_y = grid_y[keep]
    grid_x = grid_x[keep]
    grid_i = best_idx.detach().cpu().numpy()[grid_y, grid_x]

    tracks = []
    for t in range(t_video):
        full_t, _opa_t, _ = _compose_frame_gaussians(
            canonical_params,
            delta_params,
            gs_decoder.dynamic.predict_rotation,
            disable_delta_opacity,
            t,
        )
        xyz_t = full_t[0, ..., 0:3]
        sel = xyz_t[grid_i, grid_y, grid_x]
        tracks.append(sel.detach().cpu().numpy())
    return np.stack(tracks, axis=1).astype(np.float32, copy=False)


@torch.no_grad()
def _render_gs_debug_panels(
    pipe,
    gs_features: torch.Tensor,
    x0_hat: torch.Tensor,
    inputs_shared: dict,
    v2v_concat_mode: str,
    image_height: int,
    image_width: int,
) -> Optional[List[List[np.ndarray]]]:
    """Render GS debug panels (rendered RGB, depth, opacity) from predicted x0_hat.

    Returns a list of per-frame panel lists: [[rgb, depth, opacity], ...] or None
    if the GS decoder is not available or rendering fails.
    """
    gs_decoder = getattr(pipe, "gs_decoder", None)
    if gs_decoder is None:
        return None

    try:
        # Extract target-half features (width concat: right half)
        if v2v_concat_mode == "width":
            split = gs_features.shape[-1] // 2
            gs_features_target = gs_features[:, :, :, :, split:]
            latents_target = x0_hat[:, :, :, :, split:]
        else:
            split = gs_features.shape[2] // 2
            gs_features_target = gs_features[:, :, split:]
            latents_target = x0_hat[:, :, split:]

        target_cameras = inputs_shared.get("target_cameras_flashworld")
        if target_cameras is None:
            return None
        if target_cameras.dim() == 2:
            target_cameras = target_cameras.unsqueeze(0)

        # Run GS decoder: feats + latents → Gaussian params
        gaussian_params = gs_decoder(
            gs_features_target,
            latents_target.detach(),
            target_cameras,
        )
        # gaussian_params: (B, T_video, npp, H, W, D)
        gs_cfg = inputs_shared.get("gs_config", {}) or {}
        keep_frac = float(gs_cfg.get("opacity_prune_keep_frac", 1.0))
        if keep_frac < 1.0:
            from holo4d.models.gs_loss import apply_opacity_pruning
            gaussian_params = apply_opacity_pruning(
                gaussian_params,
                keep_frac=keep_frac,
            )

        # Render RGB + depth
        rendered_rgb, rendered_depth = gs_decoder.render(
            gaussian_params,
            target_cameras,
            height=image_height,
            width=image_width,
            bg_mode="white",
        )
        # rendered_rgb: (B, T, 3, H, W) in [-1, 1]
        # rendered_depth: (B, T, 1, H, W)

        # Extract opacity from gaussian params (channel index 3)
        opacity = gaussian_params[0, :, :, :, :, 3]  # (T, npp, H, W)

        # Convert to numpy panels per frame
        rgb_01 = rendered_rgb[0].float().add(1).mul(0.5).clamp(0, 1)  # (T, 3, H, W)
        rgb_frames = []
        for t in range(rgb_01.shape[0]):
            frame = (rgb_01[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb_frames.append(frame)

        depth_frames = _colorize_depth_tensor(rendered_depth[0])
        opa_frames = _opacity_to_grayscale(opacity)

        panels_per_frame = []
        n = min(len(rgb_frames), len(depth_frames), len(opa_frames))
        for t in range(n):
            panels_per_frame.append([rgb_frames[t], depth_frames[t], opa_frames[t]])
        return panels_per_frame

    except Exception as e:
        print(f"  [GS debug] render failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _build_two_row_frame(
    row1_panels: List[np.ndarray],
    row1_labels: List[str],
    row2_panels: List[np.ndarray],
    row2_labels: List[str],
) -> Image.Image:
    """
    Build a 2-row debug frame.
    Row 1 (reference/inputs) panels are scaled to match the total width of Row 2 (predictions).
    Each row gets its own column labels drawn at the bottom.
    """
    h, w = row2_panels[0].shape[:2]
    row2_total_w = len(row2_panels) * w
    row1_panel_w = row2_total_w // len(row1_panels)

    row1_resized = [
        np.array(Image.fromarray(p).resize((row1_panel_w, h), resample=Image.BILINEAR))
        for p in row1_panels
    ]
    row1_combined = np.concatenate(row1_resized, axis=1)
    # Pad to exact width in case of integer rounding
    if row1_combined.shape[1] < row2_total_w:
        pad_w = row2_total_w - row1_combined.shape[1]
        row1_combined = np.concatenate(
            [row1_combined, np.zeros((h, pad_w, 3), dtype=np.uint8)], axis=1
        )

    row2_combined = np.concatenate(row2_panels, axis=1)

    row1_img = _draw_column_captions(Image.fromarray(row1_combined), row1_labels)
    row2_img = _draw_column_captions(Image.fromarray(row2_combined), row2_labels)

    final = np.concatenate([np.array(row1_img), np.array(row2_img)], axis=0)
    return Image.fromarray(final)


def _split_v2v_latents(
    input_latents: torch.Tensor,
    v2v_concat_mode: str,
    multimodality_nvs: bool = False,
    multimodality_tracking: bool = False,
) -> V2VSlices:
    """
    Split concatenated V2V latents into source and target(s).

    Args:
        input_latents: Concatenated latents
        v2v_concat_mode: 'width' or 'frame'
        multimodality_nvs: If True, expects 3-way split [source | target_rgb | target_depth]
        multimodality_tracking: If True, expects 3-way split [source | target_track | target_track_video]
        If both False, expects 2-way split [source | target]

    Returns:
        V2VSlices with cond_latents and target_latents, plus optional RGB/depth splits
    """
    num_domains = 3 if (multimodality_nvs or multimodality_tracking) else 2

    if v2v_concat_mode == "width":
        if input_latents.shape[4] % num_domains != 0:
            raise ValueError(f"V2V width mode with {num_domains} domains expects width divisible by {num_domains}, got W={input_latents.shape[4]}.")
        split = input_latents.shape[4] // num_domains

        cond_slice = (slice(None), slice(None), slice(None), slice(None), slice(0, split))
        cond_latents = input_latents[cond_slice]

        if multimodality_nvs or multimodality_tracking:
            # 3-way split: [source_rgb | target_rgb | target_depth]
            target_rgb_slice = (slice(None), slice(None), slice(None), slice(None), slice(split, 2*split))
            target_depth_slice = (slice(None), slice(None), slice(None), slice(None), slice(2*split, None))
            target_slice = (slice(None), slice(None), slice(None), slice(None), slice(split, None))  # Both targets

            return V2VSlices(
                cond_latents=cond_latents,
                target_latents=input_latents[target_slice],
                cond_slice=cond_slice,
                target_slice=target_slice,
                target_rgb_latents=input_latents[target_rgb_slice],
                target_depth_latents=input_latents[target_depth_slice],
                target_rgb_slice=target_rgb_slice,
                target_depth_slice=target_depth_slice,
                num_domains=3,
            )
        else:
            # 2-way split: [source_rgb | target_rgb]
            target_slice = (slice(None), slice(None), slice(None), slice(None), slice(split, None))
            return V2VSlices(
                cond_latents=cond_latents,
                target_latents=input_latents[target_slice],
                cond_slice=cond_slice,
                target_slice=target_slice,
                num_domains=2,
            )

    if v2v_concat_mode == "frame":
        if input_latents.shape[2] % num_domains != 0:
            raise ValueError(f"V2V frame mode with {num_domains} domains expects frame length divisible by {num_domains}, got T={input_latents.shape[2]}.")
        split = input_latents.shape[2] // num_domains

        cond_slice = (slice(None), slice(None), slice(0, split), Ellipsis)
        cond_latents = input_latents[cond_slice]

        if multimodality_nvs or multimodality_tracking:
            # 3-way split: [source_rgb | target_rgb | target_depth]
            target_rgb_slice = (slice(None), slice(None), slice(split, 2*split), Ellipsis)
            target_depth_slice = (slice(None), slice(None), slice(2*split, None), Ellipsis)
            target_slice = (slice(None), slice(None), slice(split, None), Ellipsis)  # Both targets

            return V2VSlices(
                cond_latents=cond_latents,
                target_latents=input_latents[target_slice],
                cond_slice=cond_slice,
                target_slice=target_slice,
                target_rgb_latents=input_latents[target_rgb_slice],
                target_depth_latents=input_latents[target_depth_slice],
                target_rgb_slice=target_rgb_slice,
                target_depth_slice=target_depth_slice,
                num_domains=3,
            )
        else:
            # 2-way split: [source_rgb | target_rgb]
            target_slice = (slice(None), slice(None), slice(split, None), Ellipsis)
            return V2VSlices(
                cond_latents=cond_latents,
                target_latents=input_latents[target_slice],
                cond_slice=cond_slice,
                target_slice=target_slice,
                num_domains=2,
            )

    raise ValueError(f"Unsupported v2v_concat_mode for splitting: {v2v_concat_mode}")


@torch.no_grad()
def decode_latents_to_video_frames(
    pipe,
    latents: torch.Tensor,
    *,
    tiled: bool = True,
    tile_size: Tuple[int, int] = (30, 52),
    tile_stride: Tuple[int, int] = (15, 26),
):
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=bool(tiled),
        tile_size=tuple(tile_size),
        tile_stride=tuple(tile_stride),
    )
    return pipe.vae_output_to_video(video)


@torch.no_grad()
def decode_latents_to_tensor(
    pipe,
    latents: torch.Tensor,
    *,
    tiled: bool = True,
    tile_size: Tuple[int, int] = (30, 52),
    tile_stride: Tuple[int, int] = (15, 26),
) -> torch.Tensor:
    """
    Decode latents to tensor (without converting to PIL frames).

    Returns:
        Decoded tensor in (B, C, T, H, W) format
    """
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=bool(tiled),
        tile_size=tuple(tile_size),
        tile_stride=tuple(tile_stride),
    )
    return video


def _extract_pointmap_stats(batch_data: dict) -> Optional[dict]:
    """
    Extract pointmap normalization stats from batch data.

    Returns:
        dict with keys: pointmap_q_low, pointmap_q_high, common_scale
        or None if pointmap stats are not available
    """
    if "pointmap_q_low" not in batch_data or "pointmap_q_high" not in batch_data:
        return None

    # Stream3R adapter returns these as float/list (not tensors)
    pointmap_q_low = batch_data.get("pointmap_q_low")
    pointmap_q_high = batch_data.get("pointmap_q_high")
    common_scale = batch_data.get("common_scale", 1.0)

    # Convert to numpy arrays if needed
    if isinstance(pointmap_q_low, (int, float)):
        pointmap_q_low = np.array([pointmap_q_low, pointmap_q_low, pointmap_q_low])
    elif torch.is_tensor(pointmap_q_low):
        pointmap_q_low = pointmap_q_low.cpu().numpy()
    else:
        pointmap_q_low = np.array(pointmap_q_low)

    if isinstance(pointmap_q_high, (int, float)):
        pointmap_q_high = np.array([pointmap_q_high, pointmap_q_high, pointmap_q_high])
    elif torch.is_tensor(pointmap_q_high):
        pointmap_q_high = pointmap_q_high.cpu().numpy()
    else:
        pointmap_q_high = np.array(pointmap_q_high)

    if torch.is_tensor(common_scale):
        common_scale = float(common_scale.cpu().item())
    else:
        common_scale = float(common_scale)

    return {
        "pointmap_q_low": pointmap_q_low,
        "pointmap_q_high": pointmap_q_high,
        "common_scale": common_scale,
    }


def _extract_z_channel_frames(
    pointmap_tensor: torch.Tensor,
) -> List[Image.Image]:
    """
    Extract z-channel from pointmap and convert to grayscale video frames.

    Matches the approach in Wan2.2-TI2V-5B-v2v_inference_stream3r.py:723-743

    Args:
        pointmap_tensor: (B, C, T, H, W) pointmap in normalized space [-1, 1]

    Returns:
        List of PIL Images with z-channel as grayscale (brighter = larger z)
    """
    # Extract z channel only: (B, C, T, H, W) -> (1, T, H, W)
    z_channel = pointmap_tensor[0, 2:3, :, :, :]  # (1, T, H, W) - z channel only
    z_channel_rgb = z_channel.repeat(3, 1, 1, 1)  # (3, T, H, W) - repeat to RGB

    # Convert to numpy: (3, T, H, W)
    z_channel_rgb_np = z_channel_rgb.cpu().float().numpy()
    z_channel_rgb_np = z_channel_rgb_np.transpose(1, 0, 2, 3)  # (T, 3, H, W)

    # Normalize from [-1, 1] to [0, 1] for visualization
    z_channel_rgb_np = np.clip((z_channel_rgb_np + 1.0) / 2.0, 0.0, 1.0)

    # Convert to PIL images
    z_frames = []
    for t in range(z_channel_rgb_np.shape[0]):
        frame = (z_channel_rgb_np[t].transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
        z_frames.append(Image.fromarray(frame))

    return z_frames


@torch.no_grad()
def log_flowmatch_v2v_latents(
    training_module,
    batch_data: dict,
    *,
    output_root: str,
    global_step: int,
    fps: int = 16,
    quality: int = 5,
    max_frames: Optional[int] = None,
    tiled_decode: bool = True,
    tile_size: Tuple[int, int] = (30, 52),
    tile_stride: Tuple[int, int] = (15, 26),
) -> dict:
    """
    Logs (to disk) V2V latents decoded back to RGB space for debugging:
    - conditioning (RGB) latents (decoded)
    - clean x0 (ground truth) latents (decoded)
    - noisy xt at a random training timestep (decoded)
    - predicted x0_hat inferred from flow-matching output (decoded)
    """
    pipe = training_module.pipe
    pipe.scheduler.training = True

    step_dir = _mkdir(os.path.join(output_root, f"step-{int(global_step):06d}"))

    # Run the same preprocessing graph as training to obtain latents + conditioning.
    inputs = training_module.get_pipeline_inputs(batch_data)
    inputs = training_module.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
    for unit in pipe.units:
        inputs = pipe.unit_runner(unit, pipe, *inputs)
    inputs_shared, inputs_posi, _inputs_nega = inputs

    v2v_concat_mode = inputs_shared.get("v2v_concat_mode") or getattr(training_module, "v2v_concat_mode", None)
    if v2v_concat_mode is None:
        raise ValueError("V2V debug requested but `v2v_concat_mode` is missing.")

    # Detect multi-modality mode
    multimodality_nvs = inputs_shared.get("multimodality_nvs", False)
    multimodality_tracking = inputs_shared.get("multimodality_tracking", False)
    # When depth_video_raw=True the secondary modality is depth (3-ch replicated) not trackvid
    depth_secondary = bool(inputs_shared.get("depth_video_raw", False))

    # Honor training's fixed_sigma pin so debug viz matches the actual
    # training regime. Without this the viz samples a different σ every
    # time and the saved `debug_t{t}_...mp4` filenames mislead diagnostics.
    # Always compute boundaries first — downstream logging at :2024
    # references them unconditionally.
    max_timestep_boundary = int(inputs_shared.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs_shared.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    _fixed_sigma = inputs_shared.get("fixed_sigma")
    if _fixed_sigma is not None:
        _sigmas_cpu = pipe.scheduler.sigmas.detach().cpu().float()
        timestep_id = torch.argmin((_sigmas_cpu - float(_fixed_sigma)).abs()).reshape(1)
    else:
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device).reshape(1, 1, 1, 1, 1)

    # Compute xt and x0_hat in the same parameterization used by FlowMatchScheduler:
    # xt = x0 + sigma * v, with v = noise - x0 (training_target).
    if v2v_concat_mode == "channel":
        cond_latents = inputs_shared.get("cond_latents", None)
        if cond_latents is None:
            raise ValueError("Channel concat mode requires `cond_latents` for debugging.")

        x0 = inputs_shared["input_latents"]
        noise = torch.randn_like(x0)
        xt = pipe.scheduler.add_noise(x0, noise, timestep)

        inputs_for_model = dict(inputs_shared)
        inputs_for_model["latents"] = xt
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        v_hat = _extract_noise_pred(
            pipe.model_fn(**models, **inputs_for_model, **inputs_posi, timestep=timestep)
        )
        x0_hat = xt - sigma * v_hat

        # For channel mode, only decode the full concat latents
        frames_x0_concat = decode_latents_to_video_frames(pipe, x0, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)
        frames_xt_concat = decode_latents_to_video_frames(pipe, xt, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)
        frames_x0_hat_concat = decode_latents_to_video_frames(pipe, x0_hat, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)
    else:
        input_latents = inputs_shared["input_latents"]
        slices = _split_v2v_latents(
            input_latents,
            v2v_concat_mode,
            multimodality_nvs=multimodality_nvs,
            multimodality_tracking=multimodality_tracking,
        )
        x0 = input_latents
        noise = torch.randn_like(x0)
        xt = pipe.scheduler.add_noise(x0, noise, timestep)
        xt[slices.cond_slice] = x0[slices.cond_slice]

        inputs_for_model = dict(inputs_shared)
        inputs_for_model["latents"] = xt
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        model_out_raw = pipe.model_fn(**models, **inputs_for_model, **inputs_posi, timestep=timestep)
        v_hat = _extract_noise_pred(model_out_raw)
        # Capture GS features if the model returned them (GS-enabled forward)
        gs_features_debug = model_out_raw.get("gs_features") if isinstance(model_out_raw, dict) else None
        x0_hat = xt - sigma * v_hat
        x0_hat[slices.cond_slice] = x0[slices.cond_slice]

        # Decode strategy:
        # - width/channel concat: decoding the full tensor is fine (layout is spatial/channel-wise)
        # - frame concat: decoding the full tensor can be misleading because Wan's temporal packing
        #   does not compose cleanly across concatenated latent sequences; decode per-domain instead.
        frames_x0_concat = None
        frames_xt_concat = None
        frames_x0_hat_concat = None
        if v2v_concat_mode != "frame":
            frames_x0_concat = decode_latents_to_video_frames(pipe, x0, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)
            frames_xt_concat = decode_latents_to_video_frames(pipe, xt, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)
            frames_x0_hat_concat = decode_latents_to_video_frames(pipe, x0_hat, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)

    # Save to disk: combined video with optimal layout
    t_float = float(timestep.detach().float().cpu().item())
    if frames_x0_concat is not None:
        frames_x0_concat = _maybe_truncate_frames(frames_x0_concat, max_frames)
    if frames_xt_concat is not None:
        frames_xt_concat = _maybe_truncate_frames(frames_xt_concat, max_frames)
    if frames_x0_hat_concat is not None:
        frames_x0_hat_concat = _maybe_truncate_frames(frames_x0_hat_concat, max_frames)

    # Optional: decode and save XYZ conditioning video (if enabled in preprocessing).
    frames_xyz_cond: Optional[List[Image.Image]] = None
    control_xyz_latents_input = inputs_shared.get("control_xyz_latents_input", None)
    xyz_control_latents_input = control_xyz_latents_input
    # Query frame index (if provided by dataset/adapter). Used for labeling and query-impulse visualization.
    try:
        track_query_idx = int(batch_data.get("track_query_idx", inputs_shared.get("track_query_idx", 0)))
    except Exception:
        track_query_idx = 0
    use_query_frame_impulse = bool(
        batch_data.get("use_query_frame_impulse_condition", False)
        or inputs_shared.get("use_query_frame_impulse_condition", False)
    )

    def _impulse_col(h: int, w: int, *, is_query: bool) -> np.ndarray:
        val = 255 if bool(is_query) else 0
        return np.full((int(h), int(w), 3), val, dtype=np.uint8)
    if control_xyz_latents_input is not None:
        # If control latents were channel-concatenated (e.g., [xyz | query-impulse]), keep only xyz for visualization.
        try:
            vae_z_dim = int(getattr(getattr(pipe, "vae", None).model, "z_dim", int(control_xyz_latents_input.shape[1])))
            if int(control_xyz_latents_input.shape[1]) > vae_z_dim:
                xyz_control_latents_input = control_xyz_latents_input[:, :vae_z_dim]
        except Exception:
            xyz_control_latents_input = control_xyz_latents_input

    if xyz_control_latents_input is not None:
        try:
            frames_xyz_cond = decode_latents_to_video_frames(
                pipe,
                xyz_control_latents_input,
                tiled=tiled_decode,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            frames_xyz_cond = _maybe_truncate_frames(frames_xyz_cond, max_frames)
        except Exception as e:
            print(f"  ⚠ Failed to decode xyz condition latents: {e}")
            frames_xyz_cond = None

    # RESIDUAL MODE: If predicting residuals, decode absolute track = residual + xyz_cond[query_frame]
    # This provides visual reference to understand what the model is learning
    # IMPORTANT: Since residual = track - xyz[query], we only add query frame's xyz back
    frames_absolute_track: Optional[List[Image.Image]] = None
    predict_residual = batch_data.get("predict_track_residual", False) or inputs_shared.get("predict_track_residual", False)

    if predict_residual and xyz_control_latents_input is not None:
        try:
            # In residual mode, we need to add xyz_cond_latents (query frame only, in latent space) to residual_latents
            # to get absolute_track_latents, then decode to pixel space.
            # IMPORTANT: Addition must happen in LATENT space, not pixel space!

            # Get query frame index (default to 0 if not provided)
            track_query_idx = int(track_query_idx)

            # Extract query frame from xyz_cond latents (B, C, T, H, W) -> (B, C, 1, H, W)
            xyz_query_latents = xyz_control_latents_input[:, :, track_query_idx:track_query_idx+1, :, :]

            if v2v_concat_mode == "channel":
                # Channel mode: x0/xt/x0_hat are full tensors (RGB+residual concatenated)
                # In this case, target is the residual track
                x0_absolute_latents = x0 + xyz_query_latents
                frames_absolute_track = decode_latents_to_video_frames(
                    pipe, x0_absolute_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
                )
                frames_absolute_track = _maybe_truncate_frames(frames_absolute_track, max_frames)
            else:
                # Width/frame mode: only the target portion contains residual latents
                slices = _split_v2v_latents(
                    x0,
                    v2v_concat_mode,
                    multimodality_nvs=multimodality_nvs,
                    multimodality_tracking=multimodality_tracking,
                )
                # For 3-way multimodality layouts, residual is defined on the primary target only
                # (NVS: target_rgb, Tracking: target_track), not on the secondary modality.
                if (multimodality_nvs or multimodality_tracking) and slices.target_rgb_slice is not None:
                    residual_latents_x0 = x0[slices.target_rgb_slice]
                else:
                    residual_latents_x0 = x0[slices.target_slice]

                # Add xyz_cond latents (query frame only, in latent space) to residual latents
                # Broadcasting will expand query frame to all frames
                absolute_latents_x0 = residual_latents_x0 + xyz_query_latents

                frames_absolute_track = decode_latents_to_video_frames(
                    pipe, absolute_latents_x0, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
                )
                frames_absolute_track = _maybe_truncate_frames(frames_absolute_track, max_frames)

            print(f"  ✓ Decoded absolute track (residual + xyz_cond[query_frame={track_query_idx}]) for visualization")
        except Exception as e:
            print(f"  ⚠ Failed to decode absolute track in residual mode: {e}")
            frames_absolute_track = None

    # For width/frame concat modes: avoid duplicating RGB 3 times
    # Layout: RGB (cond) | target_x0 | target_xt | target_x0_hat
    # For channel mode: use full videos (no split needed)
    from PIL import Image
    frames_combined = []
    _is_two_row = False  # Set True for multimodality_tracking+abs_gt layouts; skips global caption pass

    if v2v_concat_mode == "channel":
        assert frames_x0_concat is not None and frames_xt_concat is not None and frames_x0_hat_concat is not None
        # Channel mode: show full x0, xt, x0_hat side-by-side
        n_frames = min(len(frames_x0_concat), len(frames_xt_concat), len(frames_x0_hat_concat))
        q_vis = max(0, min(int(track_query_idx), int(n_frames) - 1)) if n_frames > 0 else 0
        for i, (f_x0, f_xt, f_x0_hat) in enumerate(zip(frames_x0_concat, frames_xt_concat, frames_x0_hat_concat)):
            arr_x0 = np.array(f_x0)
            arr_xt = np.array(f_xt)
            arr_x0_hat = np.array(f_x0_hat)
            parts = [arr_x0]
            if frames_xyz_cond is not None and i < len(frames_xyz_cond):
                arr_xyz = np.array(frames_xyz_cond[i])
                if arr_xyz.shape[:2] != arr_x0.shape[:2]:
                    arr_xyz = np.array(
                        Image.fromarray(arr_xyz).resize((arr_x0.shape[1], arr_x0.shape[0]), resample=Image.BILINEAR)
                    )
                parts.append(arr_xyz)
            if use_query_frame_impulse:
                parts.append(_impulse_col(arr_x0.shape[0], arr_x0.shape[1], is_query=(int(i) == int(q_vis))))
            parts.extend([arr_xt, arr_x0_hat])
            combined = np.concatenate(parts, axis=1)
            frames_combined.append(Image.fromarray(combined))
    elif v2v_concat_mode == "frame":
        # Frame mode: decode source/target latents separately to avoid temporal packing artifacts.
        slices_x0 = _split_v2v_latents(
            x0,
            v2v_concat_mode,
            multimodality_nvs=multimodality_nvs,
            multimodality_tracking=multimodality_tracking,
        )
        slices_xt = _split_v2v_latents(
            xt,
            v2v_concat_mode,
            multimodality_nvs=multimodality_nvs,
            multimodality_tracking=multimodality_tracking,
        )
        slices_x0_hat = _split_v2v_latents(
            x0_hat,
            v2v_concat_mode,
            multimodality_nvs=multimodality_nvs,
            multimodality_tracking=multimodality_tracking,
        )

        frames_src = decode_latents_to_video_frames(
            pipe, slices_x0.cond_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
        )
        frames_src = _maybe_truncate_frames(frames_src, max_frames)

        if multimodality_nvs:
            assert slices_x0.target_rgb_latents is not None and slices_x0.target_depth_latents is not None
            assert slices_xt.target_rgb_latents is not None and slices_xt.target_depth_latents is not None
            assert slices_x0_hat.target_rgb_latents is not None and slices_x0_hat.target_depth_latents is not None

            frames_tgt_rgb_x0 = decode_latents_to_video_frames(
                pipe, slices_x0.target_rgb_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_depth_x0 = decode_latents_to_video_frames(
                pipe, slices_x0.target_depth_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_rgb_xt = decode_latents_to_video_frames(
                pipe, slices_xt.target_rgb_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_depth_xt = decode_latents_to_video_frames(
                pipe, slices_xt.target_depth_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_rgb_x0_hat = decode_latents_to_video_frames(
                pipe, slices_x0_hat.target_rgb_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_depth_x0_hat = decode_latents_to_video_frames(
                pipe, slices_x0_hat.target_depth_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )

            frames_tgt_rgb_x0 = _maybe_truncate_frames(frames_tgt_rgb_x0, max_frames)
            frames_tgt_depth_x0 = _maybe_truncate_frames(frames_tgt_depth_x0, max_frames)
            frames_tgt_rgb_xt = _maybe_truncate_frames(frames_tgt_rgb_xt, max_frames)
            frames_tgt_depth_xt = _maybe_truncate_frames(frames_tgt_depth_xt, max_frames)
            frames_tgt_rgb_x0_hat = _maybe_truncate_frames(frames_tgt_rgb_x0_hat, max_frames)
            frames_tgt_depth_x0_hat = _maybe_truncate_frames(frames_tgt_depth_x0_hat, max_frames)

            n_frames = min(
                    len(frames_src),
                    len(frames_tgt_rgb_x0),
                    len(frames_tgt_depth_x0),
                    len(frames_tgt_rgb_xt),
                    len(frames_tgt_depth_xt),
                    len(frames_tgt_rgb_x0_hat),
                    len(frames_tgt_depth_x0_hat),
            )
            q_vis = max(0, min(int(track_query_idx), int(n_frames) - 1)) if n_frames > 0 else 0
            for i in range(n_frames):
                src_rgb = np.array(frames_src[i])
                parts = [src_rgb]
                if frames_xyz_cond is not None and i < len(frames_xyz_cond):
                    arr_xyz = np.array(frames_xyz_cond[i])
                    if arr_xyz.shape[:2] != src_rgb.shape[:2]:
                        arr_xyz = np.array(
                            Image.fromarray(arr_xyz).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                        )
                    parts.append(arr_xyz)
                if use_query_frame_impulse:
                    parts.append(_impulse_col(src_rgb.shape[0], src_rgb.shape[1], is_query=(int(i) == int(q_vis))))

                parts.extend(
                    [
                        np.array(frames_tgt_rgb_x0[i]),
                        np.array(frames_tgt_depth_x0[i]),
                        np.array(frames_tgt_rgb_xt[i]),
                        np.array(frames_tgt_depth_xt[i]),
                        np.array(frames_tgt_rgb_x0_hat[i]),
                        np.array(frames_tgt_depth_x0_hat[i]),
                    ]
                )
                frames_combined.append(Image.fromarray(np.concatenate(parts, axis=1)))
        elif multimodality_tracking:
            assert slices_x0.target_rgb_latents is not None and slices_x0.target_depth_latents is not None
            assert slices_xt.target_rgb_latents is not None and slices_xt.target_depth_latents is not None
            assert slices_x0_hat.target_rgb_latents is not None and slices_x0_hat.target_depth_latents is not None

            # Primary target: track residual/track
            # Secondary target: track_video
            frames_tgt_track_x0 = decode_latents_to_video_frames(
                pipe, slices_x0.target_rgb_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_track_xt = decode_latents_to_video_frames(
                pipe, slices_xt.target_rgb_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_track_x0_hat = decode_latents_to_video_frames(
                pipe, slices_x0_hat.target_rgb_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )

            # Secondary target: show GT + noisy + denoised
            frames_tgt_trackvid_x0 = decode_latents_to_video_frames(
                pipe, slices_x0.target_depth_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_trackvid_xt = decode_latents_to_video_frames(
                pipe, slices_xt.target_depth_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_trackvid_x0_hat = decode_latents_to_video_frames(
                pipe, slices_x0_hat.target_depth_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )

            frames_tgt_track_x0 = _maybe_truncate_frames(frames_tgt_track_x0, max_frames)
            frames_tgt_track_xt = _maybe_truncate_frames(frames_tgt_track_xt, max_frames)
            frames_tgt_track_x0_hat = _maybe_truncate_frames(frames_tgt_track_x0_hat, max_frames)
            frames_tgt_trackvid_x0 = _maybe_truncate_frames(frames_tgt_trackvid_x0, max_frames)
            frames_tgt_trackvid_xt = _maybe_truncate_frames(frames_tgt_trackvid_xt, max_frames)
            frames_tgt_trackvid_x0_hat = _maybe_truncate_frames(frames_tgt_trackvid_x0_hat, max_frames)

            # Colorize secondary modality when it carries depth (3-ch replicated → Spectral_r).
            # Share one (d_min, d_max) across x0 / xt / x0_hat AND across all frames so the
            # three columns are directly comparable and individual frames don't flicker.
            if depth_secondary:
                import matplotlib as _mpl
                _cmap = _mpl.colormaps.get_cmap("Spectral_r")

                def _stack(fs):
                    return np.stack(
                        [np.asarray(f, dtype=np.float32).mean(axis=2) for f in fs], axis=0
                    ) if fs else np.zeros((0,))

                _all = np.concatenate([
                    _stack(frames_tgt_trackvid_x0).reshape(-1),
                    _stack(frames_tgt_trackvid_xt).reshape(-1),
                    _stack(frames_tgt_trackvid_x0_hat).reshape(-1),
                ], axis=0) if frames_tgt_trackvid_x0 else np.zeros((0,))
                if _all.size > 0:
                    _dmin, _dmax = float(_all.min()), float(_all.max())
                else:
                    _dmin, _dmax = 0.0, 1.0

                def _cz(fs):
                    out = []
                    for f in fs:
                        d = np.asarray(f, dtype=np.float32).mean(axis=2)
                        if _dmax > _dmin:
                            n = (d - _dmin) / (_dmax - _dmin)
                        else:
                            n = np.zeros_like(d)
                        rgb = (_cmap(n)[:, :, :3] * 255).astype(np.uint8)
                        out.append(Image.fromarray(rgb))
                    return out

                frames_tgt_trackvid_x0 = _cz(frames_tgt_trackvid_x0)
                frames_tgt_trackvid_xt = _cz(frames_tgt_trackvid_xt)
                frames_tgt_trackvid_x0_hat = _cz(frames_tgt_trackvid_x0_hat)

            use_two_row = frames_absolute_track is not None
            if use_two_row:
                _is_two_row = True
                sec_gt_lbl = "depth_gt" if depth_secondary else "tgt_track_video_gt"
                sec_noisy_lbl = "depth_noisy" if depth_secondary else "tgt_track_video_noisy"
                sec_dn_lbl = "depth_denoised" if depth_secondary else "tgt_track_video_denoised"
                row1_labels = (
                    ["src_rgb"]
                    + (["xyz_cond"] if frames_xyz_cond is not None else [])
                    + ["tgt_track_absolute_gt"]
                )
                row2_labels = [
                    "tgt_track_residual_gt", "tgt_track_residual_noisy", "tgt_track_residual_denoised",
                    sec_gt_lbl, sec_noisy_lbl, sec_dn_lbl,
                ]

            n_frames = min(
                len(frames_src),
                len(frames_tgt_track_x0),
                len(frames_tgt_track_xt),
                len(frames_tgt_track_x0_hat),
                len(frames_tgt_trackvid_x0),
                len(frames_tgt_trackvid_xt),
                len(frames_tgt_trackvid_x0_hat),
            )
            q_vis = max(0, min(int(track_query_idx), int(n_frames) - 1)) if n_frames > 0 else 0
            for i in range(n_frames):
                src_rgb = np.array(frames_src[i])
                arr_xyz = None
                if frames_xyz_cond is not None and i < len(frames_xyz_cond):
                    arr_xyz = np.array(frames_xyz_cond[i])
                    if arr_xyz.shape[:2] != src_rgb.shape[:2]:
                        arr_xyz = np.array(
                            Image.fromarray(arr_xyz).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                        )

                tgt_track_x0_arr = np.array(frames_tgt_track_x0[i])
                tgt_track_xt_arr = np.array(frames_tgt_track_xt[i])
                tgt_track_x0_hat_arr = np.array(frames_tgt_track_x0_hat[i])
                tgt_depth_x0_arr = np.array(frames_tgt_trackvid_x0[i])
                tgt_depth_xt_arr = np.array(frames_tgt_trackvid_xt[i])
                tgt_depth_x0_hat_arr = np.array(frames_tgt_trackvid_x0_hat[i])

                if use_two_row and frames_absolute_track is not None and i < len(frames_absolute_track):
                    arr_abs = np.array(frames_absolute_track[i])
                    if arr_abs.shape[:2] != src_rgb.shape[:2]:
                        arr_abs = np.array(
                            Image.fromarray(arr_abs).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                        )
                    row1_panels = [src_rgb] + ([arr_xyz] if arr_xyz is not None else []) + [arr_abs]
                    row2_panels = [
                        tgt_track_x0_arr, tgt_track_xt_arr, tgt_track_x0_hat_arr,
                        tgt_depth_x0_arr, tgt_depth_xt_arr, tgt_depth_x0_hat_arr,
                    ]
                    frames_combined.append(_build_two_row_frame(row1_panels, row1_labels, row2_panels, row2_labels))
                else:
                    parts = [src_rgb]
                    if arr_xyz is not None:
                        parts.append(arr_xyz)
                    if use_query_frame_impulse:
                        parts.append(_impulse_col(src_rgb.shape[0], src_rgb.shape[1], is_query=(int(i) == int(q_vis))))
                    parts.extend([
                        tgt_track_x0_arr, tgt_track_xt_arr, tgt_track_x0_hat_arr,
                        tgt_depth_x0_arr, tgt_depth_xt_arr, tgt_depth_x0_hat_arr,
                    ])
                    if frames_absolute_track is not None and i < len(frames_absolute_track):
                        arr_abs = np.array(frames_absolute_track[i])
                        if arr_abs.shape[:2] != src_rgb.shape[:2]:
                            arr_abs = np.array(
                                Image.fromarray(arr_abs).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                            )
                        parts.append(arr_abs)
                    frames_combined.append(Image.fromarray(np.concatenate(parts, axis=1)))
        else:
            frames_tgt_x0 = decode_latents_to_video_frames(
                pipe, slices_x0.target_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_xt = decode_latents_to_video_frames(
                pipe, slices_xt.target_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )
            frames_tgt_x0_hat = decode_latents_to_video_frames(
                pipe, slices_x0_hat.target_latents, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride
            )

            frames_tgt_x0 = _maybe_truncate_frames(frames_tgt_x0, max_frames)
            frames_tgt_xt = _maybe_truncate_frames(frames_tgt_xt, max_frames)
            frames_tgt_x0_hat = _maybe_truncate_frames(frames_tgt_x0_hat, max_frames)

            n_frames = min(len(frames_src), len(frames_tgt_x0), len(frames_tgt_xt), len(frames_tgt_x0_hat))
            q_vis = max(0, min(int(track_query_idx), int(n_frames) - 1)) if n_frames > 0 else 0
            for i in range(n_frames):
                src_rgb = np.array(frames_src[i])
                parts = [src_rgb]
                if frames_xyz_cond is not None and i < len(frames_xyz_cond):
                    arr_xyz = np.array(frames_xyz_cond[i])
                    if arr_xyz.shape[:2] != src_rgb.shape[:2]:
                        arr_xyz = np.array(
                            Image.fromarray(arr_xyz).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                        )
                    parts.append(arr_xyz)
                if use_query_frame_impulse:
                    parts.append(_impulse_col(src_rgb.shape[0], src_rgb.shape[1], is_query=(int(i) == int(q_vis))))

                tgt_x0 = np.array(frames_tgt_x0[i])
                tgt_xt = np.array(frames_tgt_xt[i])
                tgt_x0_hat = np.array(frames_tgt_x0_hat[i])
                parts.extend([tgt_x0, tgt_xt, tgt_x0_hat])

                # Add absolute track if in residual mode
                if frames_absolute_track is not None and i < len(frames_absolute_track):
                    arr_abs = np.array(frames_absolute_track[i])
                    if arr_abs.shape[:2] != src_rgb.shape[:2]:
                        arr_abs = np.array(
                            Image.fromarray(arr_abs).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                        )
                    parts.append(arr_abs)

                frames_combined.append(Image.fromarray(np.concatenate(parts, axis=1)))
    else:
        assert frames_x0_concat is not None and frames_xt_concat is not None and frames_x0_hat_concat is not None
        # Width/Frame mode: extract RGB once, then show target portions
        n_frames = min(len(frames_x0_concat), len(frames_xt_concat), len(frames_x0_hat_concat))
        q_vis = max(0, min(int(track_query_idx), int(n_frames) - 1)) if n_frames > 0 else 0
        for i, (f_x0, f_xt, f_x0_hat) in enumerate(zip(frames_x0_concat, frames_xt_concat, frames_x0_hat_concat)):
            arr_x0 = np.array(f_x0)
            arr_xt = np.array(f_xt)
            arr_x0_hat = np.array(f_x0_hat)
            arr_xyz = None
            if frames_xyz_cond is not None and i < len(frames_xyz_cond):
                arr_xyz = np.array(frames_xyz_cond[i])

            # Split each frame to extract RGB and target portions
            if v2v_concat_mode == "width":
                # Split horizontally (width dimension)
                num_domains = 3 if (multimodality_nvs or multimodality_tracking) else 2
                split_w = arr_x0.shape[1] // num_domains

                # Always extract source RGB (from x0)
                src_rgb = arr_x0[:, :split_w, :]

                if multimodality_nvs:
                    # 3-way split: [source_rgb | target_rgb | target_depth]
                    # Extract target RGB and target depth separately
                    tgt_rgb_x0 = arr_x0[:, split_w:2*split_w, :]
                    tgt_depth_x0 = arr_x0[:, 2*split_w:, :]

                    tgt_rgb_xt = arr_xt[:, split_w:2*split_w, :]
                    tgt_depth_xt = arr_xt[:, 2*split_w:, :]

                    tgt_rgb_x0_hat = arr_x0_hat[:, split_w:2*split_w, :]
                    tgt_depth_x0_hat = arr_x0_hat[:, 2*split_w:, :]

                    parts = [src_rgb]
                    if arr_xyz is not None:
                        if arr_xyz.shape[:2] != src_rgb.shape[:2]:
                            arr_xyz = np.array(
                                Image.fromarray(arr_xyz).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                            )
                        parts.append(arr_xyz)
                    if use_query_frame_impulse:
                        parts.append(_impulse_col(src_rgb.shape[0], src_rgb.shape[1], is_query=(int(i) == int(q_vis))))
                    parts.extend([tgt_rgb_x0, tgt_depth_x0, tgt_rgb_xt, tgt_depth_xt, tgt_rgb_x0_hat, tgt_depth_x0_hat])
                    combined = np.concatenate(parts, axis=1)
                elif multimodality_tracking:
                    # 3-way split: [source_rgb | target_track | target_track_video/depth]
                    tgt_track_x0 = arr_x0[:, split_w:2 * split_w, :]
                    tgt_track_xt = arr_xt[:, split_w:2 * split_w, :]
                    tgt_track_x0_hat = arr_x0_hat[:, split_w:2 * split_w, :]

                    tgt_trackvid_x0 = arr_x0[:, 2 * split_w:, :]
                    tgt_trackvid_xt = arr_xt[:, 2 * split_w:, :]
                    tgt_trackvid_x0_hat = arr_x0_hat[:, 2 * split_w:, :]

                    # Colorize secondary modality if it carries depth
                    if depth_secondary:
                        tgt_trackvid_x0 = _colorize_depth_frame(tgt_trackvid_x0)
                        tgt_trackvid_xt = _colorize_depth_frame(tgt_trackvid_xt)
                        tgt_trackvid_x0_hat = _colorize_depth_frame(tgt_trackvid_x0_hat)

                    if frames_absolute_track is not None and i < len(frames_absolute_track):
                        # 2-row layout: Row 1 = inputs/GT, Row 2 = predictions
                        arr_abs = np.array(frames_absolute_track[i])
                        if arr_abs.shape[:2] != src_rgb.shape[:2]:
                            arr_abs = np.array(
                                Image.fromarray(arr_abs).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                            )
                        if arr_xyz is not None and arr_xyz.shape[:2] != src_rgb.shape[:2]:
                            arr_xyz = np.array(
                                Image.fromarray(arr_xyz).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                            )
                        _is_two_row = True
                        sec_gt_lbl = "depth_gt" if depth_secondary else "tgt_track_video_gt"
                        sec_noisy_lbl = "depth_noisy" if depth_secondary else "tgt_track_video_noisy"
                        sec_dn_lbl = "depth_denoised" if depth_secondary else "tgt_track_video_denoised"
                        row1_panels = [src_rgb] + ([arr_xyz] if arr_xyz is not None else []) + [arr_abs]
                        row1_lbls = (
                            ["src_rgb"]
                            + (["xyz_cond"] if arr_xyz is not None else [])
                            + ["tgt_track_absolute_gt"]
                        )
                        row2_panels = [
                            tgt_track_x0, tgt_track_xt, tgt_track_x0_hat,
                            tgt_trackvid_x0, tgt_trackvid_xt, tgt_trackvid_x0_hat,
                        ]
                        row2_lbls = [
                            "tgt_track_residual_gt", "tgt_track_residual_noisy", "tgt_track_residual_denoised",
                            sec_gt_lbl, sec_noisy_lbl, sec_dn_lbl,
                        ]
                        combined = np.array(_build_two_row_frame(row1_panels, row1_lbls, row2_panels, row2_lbls))
                    else:
                        parts = [src_rgb]
                        if arr_xyz is not None:
                            if arr_xyz.shape[:2] != src_rgb.shape[:2]:
                                arr_xyz = np.array(
                                    Image.fromarray(arr_xyz).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                                )
                            parts.append(arr_xyz)
                        if use_query_frame_impulse:
                            parts.append(_impulse_col(src_rgb.shape[0], src_rgb.shape[1], is_query=(int(i) == int(q_vis))))
                        parts.extend([tgt_track_x0, tgt_track_xt, tgt_track_x0_hat,
                                       tgt_trackvid_x0, tgt_trackvid_xt, tgt_trackvid_x0_hat])
                        combined = np.concatenate(parts, axis=1)
                else:
                    # 2-way split: [source_rgb | target_rgb]
                    tgt_rgb_x0 = arr_x0[:, split_w:, :]
                    tgt_rgb_xt = arr_xt[:, split_w:, :]
                    tgt_rgb_x0_hat = arr_x0_hat[:, split_w:, :]

                    # Clean layout: src_rgb | tgt_gt | tgt_noisy | tgt_denoised
                    # (xyz_cond and query_impulse columns removed — query frame
                    # info is shown in the filename "q{N}" suffix instead)
                    parts = [src_rgb, tgt_rgb_x0, tgt_rgb_xt, tgt_rgb_x0_hat]

                    # Add absolute track if in residual mode
                    if frames_absolute_track is not None and i < len(frames_absolute_track):
                        arr_abs = np.array(frames_absolute_track[i])
                        if arr_abs.shape[:2] != src_rgb.shape[:2]:
                            arr_abs = np.array(
                                Image.fromarray(arr_abs).resize((src_rgb.shape[1], src_rgb.shape[0]), resample=Image.BILINEAR)
                            )
                        parts.append(arr_abs)

                    combined = np.concatenate(parts, axis=1)
            frames_combined.append(Image.fromarray(combined))

    has_query_idx = ("track_query_idx" in batch_data) or bool(use_query_frame_impulse) or bool(predict_residual)
    # For the 2-way width non-multimodality path we removed the xyz_cond
    # and query_impulse columns from the frame assembly (query info is in
    # the filename "q{N}" suffix), so suppress those labels too.
    _gs_active = locals().get("gs_features_debug", None) is not None
    _is_plain_2way = (
        v2v_concat_mode == "width"
        and not multimodality_nvs
        and not multimodality_tracking
    )
    column_labels = _build_column_labels(
        v2v_concat_mode=str(v2v_concat_mode),
        multimodality_nvs=bool(multimodality_nvs),
        multimodality_tracking=bool(multimodality_tracking),
        has_xyz_cond=(frames_xyz_cond is not None) and not _is_plain_2way,
        has_query_impulse=bool(use_query_frame_impulse) and not _is_plain_2way,
        has_absolute_track=frames_absolute_track is not None,
        has_depth_secondary=bool(depth_secondary),
    )
    # 2-row frames already have per-row captions drawn inline; skip global pass for those
    if frames_combined and not _is_two_row:
        frames_combined = [_draw_column_captions(f, column_labels) for f in frames_combined]

    # --- 3DGS debug row: rendered RGB + depth + opacity from the GS branch ---
    gs_features_debug_local = locals().get("gs_features_debug", None)
    if gs_features_debug_local is not None and frames_combined:
        depth_debug_panels = _build_training_depth_panels(inputs_shared)
        # Determine image resolution from the first combined frame
        first_frame_arr = np.array(frames_combined[0])
        # For width-concat 2-way split, the per-panel width is frame_w / num_panels.
        # The GS branch renders at the target resolution (one panel width × height).
        gs_config = inputs_shared.get("gs_config", {})
        img_h = int(gs_config.get("image_height", first_frame_arr.shape[0]))
        img_w = int(gs_config.get("image_width", first_frame_arr.shape[0]))
        # Fallback: derive from the training data if gs_config isn't set
        if "video" in batch_data and hasattr(batch_data["video"][0], "size"):
            img_w, img_h = batch_data["video"][0].size

        gs_panels = _render_gs_debug_panels(
            pipe=pipe,
            gs_features=gs_features_debug_local,
            x0_hat=x0_hat,
            inputs_shared=inputs_shared,
            v2v_concat_mode=str(v2v_concat_mode),
            image_height=img_h,
            image_width=img_w,
        )
        if gs_panels is not None:
            n_gs = min(len(gs_panels), len(frames_combined))
            new_combined = []
            for i in range(len(frames_combined)):
                diffusion_arr = np.array(frames_combined[i])
                if i < n_gs:
                    total_w = diffusion_arr.shape[1]
                    gs_row_items = [
                        ("GS rendered RGB", gs_panels[i][0]),
                        ("GS rendered depth", gs_panels[i][1]),
                        ("GS opacity", gs_panels[i][2]),
                    ]
                    for label, panel_frames in depth_debug_panels.items():
                        if i < len(panel_frames):
                            gs_row_items.append((label, panel_frames[i]))
                    # Keep native aspect ratio: scale each GS panel so the
                    # three panels together fill the diffusion row width, with
                    # height derived from the native aspect ratio of the GS
                    # render (image_height / image_width).
                    native_h, native_w = gs_row_items[0][1].shape[:2]
                    n_panels = len(gs_row_items)
                    panel_w = total_w // n_panels
                    panel_h = max(1, int(panel_w * native_h / max(1, native_w)))
                    gs_row_panels = [
                        np.array(Image.fromarray(panel).resize((panel_w, panel_h), resample=Image.BILINEAR))
                        for _, panel in gs_row_items
                    ]
                    gs_row = np.concatenate(gs_row_panels, axis=1)
                    # Pad width to exact match (integer rounding)
                    if gs_row.shape[1] < total_w:
                        gs_row = np.concatenate(
                            [gs_row, np.zeros((gs_row.shape[0], total_w - gs_row.shape[1], 3), dtype=np.uint8)],
                            axis=1,
                        )
                    elif gs_row.shape[1] > total_w:
                        gs_row = gs_row[:, :total_w]
                    gs_row_labels = [label for label, _ in gs_row_items]
                    gs_row_captioned = np.array(_draw_column_captions(Image.fromarray(gs_row), gs_row_labels))
                    final_frame = np.concatenate([diffusion_arr, gs_row_captioned], axis=0)
                    new_combined.append(Image.fromarray(final_frame))
                else:
                    new_combined.append(frames_combined[i])
            frames_combined = new_combined
            extra_labels = list(depth_debug_panels.keys())
            if extra_labels:
                print(
                    f"  [GS debug] appended 3DGS row ({n_gs} frames): rendered RGB + depth + opacity + "
                    + " + ".join(extra_labels)
                )
            else:
                print(f"  [GS debug] appended 3DGS row ({n_gs} frames): rendered RGB + depth + opacity")

    # --- Dense warp row: forward-warped source RGB / depth / valid panels ---
    # Mirrors the bottom row of eval comparison.mp4 so the same geometry is
    # visible during training without running inference.  Active whenever
    # ``use_dense_branch`` is on and the dataset emits source_rgb_video +
    # source_depth_video + camera data.
    if frames_combined:
        try:
            dense_warp_panels = _build_dense_warp_panels(inputs_shared)
        except Exception as _dw_exc:
            print(f"  [dense warp] panel build failed: {type(_dw_exc).__name__}: {_dw_exc}")
            dense_warp_panels = {}
        if dense_warp_panels:
            warp_items = list(dense_warp_panels.items())  # [(label, [frames]), ...]
            n_warp = min(len(warp_items[0][1]), len(frames_combined))
            new_combined = []
            for i in range(len(frames_combined)):
                diffusion_arr = np.array(frames_combined[i])
                if i < n_warp:
                    # Keep warp panels at native resolution (1:1 with the source
                    # data — bilinear-upscaling 144×256 to 512×288 just to fill
                    # 1/3 of the diffusion-grid width makes them look blocky
                    # without adding signal). Match the diffusion-grid per-cell
                    # width so warp panels look as sharp as src_rgb / tgt_gt.
                    total_w = diffusion_arr.shape[1]
                    native_h, native_w = warp_items[0][1][i].shape[:2]
                    n_panels = len(warp_items)
                    n_diffusion_cols = max(1, total_w // max(1, native_w))
                    cell_w = total_w // n_diffusion_cols  # = native_w when grid aligns
                    panel_w = cell_w
                    panel_h = native_h
                    warp_row_panels = [
                        np.asarray(frames_list[i], dtype=np.uint8)
                        if (frames_list[i].shape[0] == panel_h
                            and frames_list[i].shape[1] == panel_w)
                        else np.array(
                            Image.fromarray(frames_list[i]).resize(
                                (panel_w, panel_h), resample=Image.BILINEAR
                            )
                        )
                        for _label, frames_list in warp_items
                    ]
                    warp_row = np.concatenate(warp_row_panels, axis=1)
                    if warp_row.shape[1] < total_w:
                        warp_row = np.concatenate(
                            [warp_row, np.zeros((warp_row.shape[0], total_w - warp_row.shape[1], 3), dtype=np.uint8)],
                            axis=1,
                        )
                    elif warp_row.shape[1] > total_w:
                        warp_row = warp_row[:, :total_w]
                    warp_labels = [label for label, _ in warp_items]
                    warp_row_captioned = np.array(
                        _draw_column_captions(Image.fromarray(warp_row), warp_labels)
                    )
                    final_frame = np.concatenate([diffusion_arr, warp_row_captioned], axis=0)
                    new_combined.append(Image.fromarray(final_frame))
                else:
                    new_combined.append(frames_combined[i])
            frames_combined = new_combined
            print(
                f"  [dense warp] appended dense warp row ({n_warp} frames): "
                + ", ".join(dense_warp_panels.keys())
            )

    if has_query_idx:
        combined_name = f"debug_t{t_float:.3f}_q{int(track_query_idx)}.mp4"
    else:
        combined_name = f"debug_t{t_float:.3f}.mp4"
    combined_path = os.path.join(step_dir, combined_name)
    _write_mp4(frames_combined, combined_path, fps=fps, quality=quality)

    paths = {"combined_mp4": combined_path}

    if frames_xyz_cond is not None:
        xyz_path = os.path.join(step_dir, f"xyz_condition_t{t_float:.3f}.mp4")
        _write_mp4(frames_xyz_cond, xyz_path, fps=fps, quality=quality)
        paths["xyz_condition_mp4"] = xyz_path

    # Extra Kubric / Track4D supervision artifacts.
    target_depth_video = batch_data.get("target_depth_video", batch_data.get("target_depth"))
    target_depth_valid_mask = batch_data.get(
        "target_depth_valid_mask",
        batch_data.get("target_depth_valid"),
    )
    target_tracks_world = batch_data.get("target_tracks_world")
    target_cam2world = batch_data.get("target_cam2world")
    target_intrinsics = batch_data.get("target_intrinsics")
    target_video_frames = batch_data.get("target_video")

    target_bg_frames = None
    if isinstance(target_video_frames, list) and target_video_frames:
        target_bg_frames = _to_numpy_rgb_frames(_maybe_truncate_frames(target_video_frames, max_frames))

    if (
        target_depth_video is not None
        and target_depth_valid_mask is not None
        and target_bg_frames is not None
    ):
        depth_tensor = torch.as_tensor(target_depth_video)
        valid_tensor = torch.as_tensor(target_depth_valid_mask)
        gt_depth_masked_frames = _colorize_depth_tensor_masked(depth_tensor, valid_tensor)
        gt_depth_path = os.path.join(step_dir, "gt_depth_masked.mp4")
        _write_mp4(gt_depth_masked_frames, gt_depth_path, fps=fps, quality=quality)
        paths["gt_depth_masked_mp4"] = gt_depth_path

    if (
        target_tracks_world is not None
        and target_cam2world is not None
        and target_intrinsics is not None
        and target_bg_frames is not None
    ):
        gt_tracks_frames = _render_track_overlay_video(
            tracks_world=target_tracks_world,
            target_cam2world=target_cam2world,
            target_intrinsics=target_intrinsics,
            bg_frames=target_bg_frames,
        )
        if gt_tracks_frames:
            gt_tracks_path = os.path.join(step_dir, "gt_tracks_2d.mp4")
            _write_mp4(gt_tracks_frames, gt_tracks_path, fps=fps, quality=quality)
            paths["gt_tracks_2d_mp4"] = gt_tracks_path

    if (
        gs_features_debug_local is not None
        and target_cam2world is not None
        and target_intrinsics is not None
        and target_bg_frames is not None
    ):
        fg_mask_frame0 = None
        if target_depth_valid_mask is not None:
            fg_mask = torch.as_tensor(target_depth_valid_mask).detach().cpu().bool().numpy()
            fg_mask_frame0 = fg_mask[0]
        pred_tracks_world = _extract_dynamic_gs_tracks_world(
            pipe=pipe,
            gs_features=gs_features_debug_local,
            x0_hat=x0_hat,
            inputs_shared=inputs_shared,
            v2v_concat_mode=str(v2v_concat_mode),
            fg_mask_frame0=fg_mask_frame0,
        )
        if pred_tracks_world is not None:
            pred_tracks_frames = _render_track_overlay_video(
                tracks_world=pred_tracks_world,
                target_cam2world=target_cam2world,
                target_intrinsics=target_intrinsics,
                bg_frames=target_bg_frames,
            )
            if pred_tracks_frames:
                pred_tracks_path = os.path.join(step_dir, "pred_gs_tracks_2d.mp4")
                _write_mp4(pred_tracks_frames, pred_tracks_path, fps=fps, quality=quality)
                paths["pred_gs_tracks_2d_mp4"] = pred_tracks_path

    # === Pointmap Z-Channel Visualization ===
    # If predicting pointmaps, extract and save z-channel as grayscale videos
    # (matches the approach in Wan2.2-TI2V-5B-v2v_inference_stream3r.py:717-743)
    pointmap_stats = _extract_pointmap_stats(batch_data)
    if pointmap_stats is not None:
        try:
            # For frame/width concat modes, we need to extract only the pointmap latents
            # For channel mode, x0/xt/x0_hat are already pointmap latents
            if v2v_concat_mode == "channel":
                pointmap_latents_x0 = x0
                pointmap_latents_xt = xt
                pointmap_latents_x0_hat = x0_hat
            else:
                # Split concatenated latents to get pointmap portion
                slices = _split_v2v_latents(
                    x0,
                    v2v_concat_mode,
                    multimodality_nvs=multimodality_nvs,
                    multimodality_tracking=multimodality_tracking,
                )
                if (multimodality_nvs or multimodality_tracking) and slices.target_rgb_slice is not None:
                    # Pointmap visualization should use the primary target branch.
                    pointmap_latents_x0 = x0[slices.target_rgb_slice]
                    pointmap_latents_xt = xt[slices.target_rgb_slice]
                    pointmap_latents_x0_hat = x0_hat[slices.target_rgb_slice]
                else:
                    pointmap_latents_x0 = x0[slices.target_slice]
                    pointmap_latents_xt = xt[slices.target_slice]
                    pointmap_latents_x0_hat = x0_hat[slices.target_slice]

            # Decode pointmap latents to get pointmap tensors (B, 3, T, H, W)
            pointmap_x0 = decode_latents_to_tensor(pipe, pointmap_latents_x0, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)
            pointmap_xt = decode_latents_to_tensor(pipe, pointmap_latents_xt, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)
            pointmap_x0_hat = decode_latents_to_tensor(pipe, pointmap_latents_x0_hat, tiled=tiled_decode, tile_size=tile_size, tile_stride=tile_stride)

            # Extract z-channel (grayscale visualization, brighter = larger z)
            # Input is decoded normalized pointmap in [-1, 1]
            z_frames_x0 = _extract_z_channel_frames(pointmap_x0)
            z_frames_xt = _extract_z_channel_frames(pointmap_xt)
            z_frames_x0_hat = _extract_z_channel_frames(pointmap_x0_hat)

            # Truncate frames if needed
            z_frames_x0 = _maybe_truncate_frames(z_frames_x0, max_frames)
            z_frames_xt = _maybe_truncate_frames(z_frames_xt, max_frames)
            z_frames_x0_hat = _maybe_truncate_frames(z_frames_x0_hat, max_frames)

            # Save individual z-channel videos
            z_x0_path = os.path.join(step_dir, f"pointmap_z_x0_t{t_float:.3f}.mp4")
            z_xt_path = os.path.join(step_dir, f"pointmap_z_xt_t{t_float:.3f}.mp4")
            z_x0_hat_path = os.path.join(step_dir, f"pointmap_z_x0_hat_t{t_float:.3f}.mp4")

            _write_mp4(z_frames_x0, z_x0_path, fps=fps, quality=quality)
            _write_mp4(z_frames_xt, z_xt_path, fps=fps, quality=quality)
            _write_mp4(z_frames_x0_hat, z_x0_hat_path, fps=fps, quality=quality)

            # Save combined z-channel video (x0 | xt | x0_hat side-by-side)
            z_frames_combined = []
            for z_x0, z_xt, z_x0_hat in zip(z_frames_x0, z_frames_xt, z_frames_x0_hat):
                arr_z_x0 = np.array(z_x0)
                arr_z_xt = np.array(z_xt)
                arr_z_x0_hat = np.array(z_x0_hat)
                combined_z = np.concatenate([arr_z_x0, arr_z_xt, arr_z_x0_hat], axis=1)
                z_frames_combined.append(_draw_column_captions(Image.fromarray(combined_z), ["z_x0", "z_xt", "z_x0_hat"]))

            z_combined_path = os.path.join(step_dir, f"pointmap_z_combined_t{t_float:.3f}.mp4")
            _write_mp4(z_frames_combined, z_combined_path, fps=fps, quality=quality)

            paths.update({
                "pointmap_z_x0": z_x0_path,
                "pointmap_z_xt": z_xt_path,
                "pointmap_z_x0_hat": z_x0_hat_path,
                "pointmap_z_combined": z_combined_path,
            })

            print(f"  ✓ Saved pointmap z-channel visualizations (t={t_float:.3f})")

        except Exception as e:
            print(f"  ⚠ Failed to save pointmap z-channel visualizations: {e}")

    # Determine layout description based on concat mode and multi-modality
    if v2v_concat_mode == "channel":
        layout_desc = "x0_gt | xt | x0_hat (horizontal, full frames)"
        if frames_xyz_cond is not None:
            layout_desc = "x0_gt | xyz_cond | xt | x0_hat (horizontal, full frames)"
    elif v2v_concat_mode == "width":
        if multimodality_nvs:
            layout_desc = "src_rgb | tgt_rgb_gt | tgt_depth_gt | tgt_rgb_noisy | tgt_depth_noisy | tgt_rgb_denoised | tgt_depth_denoised (7 components, multi-modality)"
            if frames_xyz_cond is not None:
                layout_desc = "src_rgb | xyz_cond | tgt_rgb_gt | tgt_depth_gt | tgt_rgb_noisy | tgt_depth_noisy | tgt_rgb_denoised | tgt_depth_denoised (8 components, multi-modality)"
        elif multimodality_tracking:
            layout_desc = "src_rgb | tgt_track_gt | tgt_track_noisy | tgt_track_denoised | tgt_track_video_gt | tgt_track_video_noisy | tgt_track_video_denoised (7 components, multimodality tracking)"
            if frames_xyz_cond is not None:
                layout_desc = "src_rgb | xyz_cond | tgt_track_gt | tgt_track_noisy | tgt_track_denoised | tgt_track_video_gt | tgt_track_video_noisy | tgt_track_video_denoised (8 components, multimodality tracking)"
            if frames_absolute_track is not None:
                if frames_xyz_cond is not None:
                    layout_desc = "src_rgb | xyz_cond | tgt_track_residual_gt | tgt_track_residual_noisy | tgt_track_residual_denoised | tgt_track_video_gt | tgt_track_video_noisy | tgt_track_video_denoised | tgt_track_absolute_gt (9 components, multimodality tracking residual)"
                else:
                    layout_desc = "src_rgb | tgt_track_residual_gt | tgt_track_residual_noisy | tgt_track_residual_denoised | tgt_track_video_gt | tgt_track_video_noisy | tgt_track_video_denoised | tgt_track_absolute_gt (8 components, multimodality tracking residual)"
        else:
            layout_desc = "src_rgb | tgt_gt | tgt_noisy | tgt_denoised (4 components, RGB-only)"
            if frames_xyz_cond is not None:
                layout_desc = "src_rgb | xyz_cond | tgt_gt | tgt_noisy | tgt_denoised (5 components, RGB-only)"
            if frames_absolute_track is not None:
                # In residual mode: tgt_gt is residual, add absolute track column
                if frames_xyz_cond is not None:
                    layout_desc = "src_rgb | xyz_cond | tgt_residual_gt | tgt_residual_noisy | tgt_residual_denoised | tgt_absolute_gt (6 components, residual mode)"
                else:
                    layout_desc = "src_rgb | tgt_residual_gt | tgt_residual_noisy | tgt_residual_denoised | tgt_absolute_gt (5 components, residual mode)"
    else:  # frame mode
        if multimodality_nvs:
            layout_desc = "src_rgb | tgt_rgb_gt | tgt_depth_gt | tgt_rgb_noisy | tgt_depth_noisy | tgt_rgb_denoised | tgt_depth_denoised (7 components, multi-modality)"
            if frames_xyz_cond is not None:
                layout_desc = "src_rgb | xyz_cond | tgt_rgb_gt | tgt_depth_gt | tgt_rgb_noisy | tgt_depth_noisy | tgt_rgb_denoised | tgt_depth_denoised (8 components, multi-modality)"
        elif multimodality_tracking:
            layout_desc = "src_rgb | tgt_track_gt | tgt_track_noisy | tgt_track_denoised | tgt_track_video_gt | tgt_track_video_noisy | tgt_track_video_denoised (7 components, multimodality tracking)"
            if frames_xyz_cond is not None:
                layout_desc = "src_rgb | xyz_cond | tgt_track_gt | tgt_track_noisy | tgt_track_denoised | tgt_track_video_gt | tgt_track_video_noisy | tgt_track_video_denoised (8 components, multimodality tracking)"
            if frames_absolute_track is not None:
                if frames_xyz_cond is not None:
                    layout_desc = "src_rgb | xyz_cond | tgt_track_residual_gt | tgt_track_residual_noisy | tgt_track_residual_denoised | tgt_track_video_gt | tgt_track_video_noisy | tgt_track_video_denoised | tgt_track_absolute_gt (9 components, multimodality tracking residual)"
                else:
                    layout_desc = "src_rgb | tgt_track_residual_gt | tgt_track_residual_noisy | tgt_track_residual_denoised | tgt_track_video_gt | tgt_track_video_noisy | tgt_track_video_denoised | tgt_track_absolute_gt (8 components, multimodality tracking residual)"
        else:
            layout_desc = "src_rgb | tgt_gt | tgt_noisy | tgt_denoised (4 components, frame concat)"
            if frames_xyz_cond is not None:
                layout_desc = "src_rgb | xyz_cond | tgt_gt | tgt_noisy | tgt_denoised (5 components, frame concat)"
            if frames_absolute_track is not None:
                # In residual mode: tgt_gt is residual, add absolute track column
                if frames_xyz_cond is not None:
                    layout_desc = "src_rgb | xyz_cond | tgt_residual_gt | tgt_residual_noisy | tgt_residual_denoised | tgt_absolute_gt (6 components, residual mode)"
                else:
                    layout_desc = "src_rgb | tgt_residual_gt | tgt_residual_noisy | tgt_residual_denoised | tgt_absolute_gt (5 components, residual mode)"
    if use_query_frame_impulse and "query_impulse" not in layout_desc:
        if "xyz_cond | " in layout_desc:
            layout_desc = layout_desc.replace("xyz_cond | ", "xyz_cond | query_impulse | ")
        elif "src_rgb | " in layout_desc:
            layout_desc = layout_desc.replace("src_rgb | ", "src_rgb | query_impulse | ")
        elif "x0_gt | " in layout_desc:
            layout_desc = layout_desc.replace("x0_gt | ", "x0_gt | query_impulse | ")

    track_video_prenorm_flag = inputs_shared.get("track_video_prenormalized", False)
    if torch.is_tensor(track_video_prenorm_flag):
        track_video_prenorm_flag = bool(track_video_prenorm_flag.detach().cpu().bool().any().item())
    else:
        track_video_prenorm_flag = bool(track_video_prenorm_flag)

    meta = {
        "global_step": int(global_step),
        "v2v_concat_mode": str(v2v_concat_mode),
        "multimodality_nvs": bool(multimodality_nvs),
        "multimodality_tracking": bool(multimodality_tracking),
        "track_video_prenormalized": bool(track_video_prenorm_flag),
        "track_query_idx": int(track_query_idx),
        "query_impulse_present": bool(use_query_frame_impulse),
        "xyz_condition_present": frames_xyz_cond is not None,
        "predict_residual": bool(predict_residual),
        "absolute_track_present": frames_absolute_track is not None,
        "timestep_id": int(timestep_id.item()),
        "timestep": float(timestep.detach().float().cpu().item()),
        "sigma": float(sigma.detach().float().cpu().item()),
        "min_timestep_boundary": int(min_timestep_boundary),
        "max_timestep_boundary": int(max_timestep_boundary),
        "paths": {k: os.path.relpath(v, output_root) for k, v in paths.items()},
        "shapes": {
            "x0": tuple(int(x) for x in x0.shape),
            "xt": tuple(int(x) for x in xt.shape),
            "x0_hat": tuple(int(x) for x in x0_hat.shape),
        },
        "layout": layout_desc,
        "pointmap_mode": pointmap_stats is not None,
    }

    # Add pointmap-specific metadata if available
    if pointmap_stats is not None:
        meta["pointmap_stats"] = {
            "pointmap_q_low": pointmap_stats["pointmap_q_low"].tolist() if isinstance(pointmap_stats["pointmap_q_low"], np.ndarray) else pointmap_stats["pointmap_q_low"],
            "pointmap_q_high": pointmap_stats["pointmap_q_high"].tolist() if isinstance(pointmap_stats["pointmap_q_high"], np.ndarray) else pointmap_stats["pointmap_q_high"],
            "common_scale": pointmap_stats["common_scale"],
        }
    with open(os.path.join(step_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Best-effort restore training compute models (esp. under VRAM management).
    pipe.load_models_to_device(list(pipe.in_iteration_models))

    return meta
