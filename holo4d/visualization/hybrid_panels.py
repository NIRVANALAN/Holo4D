"""Shared rendering helpers for hybrid Track4D-360 evaluation visualization.

This module provides the panel-rendering primitives used by:

- ``examples/wan2.1_fun/eval_track4d360_hybrid_dense_syn4d.py``
  (inline ``comparison.mp4`` written next to ``pred_rgb.mp4`` during eval)

- ``bash_scripts/holo4d/tools/visualize_eval_pair.py``
  (standalone retroactive viz of an existing eval pair output)

The main entry point is :func:`write_eval_comparison_mp4`, which renders a
2x4/2x5 diagnostic grid:

    top:  source | target GT | pred | |pred - gt|
    bot:  dense warped RGB | dense depth | dense valid | track conf/valid | tracks on pred

All cameras / depths in the input ``sample`` dict are assumed to be in the
``Syn4D_Multi_NVS`` cam0-normalized + ``common_scale``-rescaled frame, which
is what ``Syn4D_Multi_NVS.load_pair`` returns.
"""

from pathlib import Path
import os

import cv2
import imageio
import numpy as np

from holo4d.visualization.artifact_io import save_mp4
from holo4d.visualization.track_panels import (
    draw_label,
    forward_warp_rgbd_to_target,
)
from holo4d.visualization.vis_track_condition import draw_tracks_on_frame


# ---------------------------------------------------------------------------
# Color / mask panels
# ---------------------------------------------------------------------------

def colorize_depth(
    depth: np.ndarray,
    valid: np.ndarray,
    depth_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Turbo colormap of a depth frame, masked by ``valid``.

    ``depth_range`` should be a sequence-level range when rendering video;
    otherwise each frame normalizes independently and the color scale jitters.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    H, W = depth.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    if not valid.any():
        return out
    if depth_range is None:
        vals = depth[valid]
        lo = float(vals.min())
        hi = float(vals.max())
    else:
        lo, hi = float(depth_range[0]), float(depth_range[1])
    denom = max(hi - lo, 1e-6)
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / denom, 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[valid] = heat[valid]
    return out


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Binary mask → grayscale visualization."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    out[mask] = np.array([240, 240, 240], dtype=np.uint8)
    return out


def colorize_confidence(conf: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Turbo colormap for a [0,1]-ish confidence map with invalid pixels black."""
    conf = np.asarray(conf, dtype=np.float32)
    if valid is None:
        valid_arr = np.isfinite(conf)
    else:
        valid_arr = np.asarray(valid, dtype=bool) & np.isfinite(conf)
    out = np.zeros(conf.shape + (3,), dtype=np.uint8)
    if not valid_arr.any():
        return out
    vals = conf[valid_arr]
    lo = float(np.nanpercentile(vals, 2.0))
    hi = float(np.nanpercentile(vals, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    denom = max(hi - lo, 1e-6)
    norm = np.zeros_like(conf, dtype=np.float32)
    norm[valid_arr] = np.clip((conf[valid_arr] - lo) / denom, 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[valid_arr] = heat[valid_arr]
    return out


def _infer_track_grid_hw(num_tracks: int, image_h: int, image_w: int) -> tuple[int, int] | None:
    """Infer the source query grid for dense tracker identities."""
    n = int(num_tracks)
    if n <= 0:
        return None
    if n == int(image_h) * int(image_w):
        return int(image_h), int(image_w)
    target_aspect = float(image_h) / max(float(image_w), 1.0)
    best = None
    for h in range(1, int(np.sqrt(n)) + 1):
        if n % h != 0:
            continue
        for cand_h, cand_w in ((h, n // h), (n // h, h)):
            score = abs((float(cand_h) / float(cand_w)) - target_aspect)
            if best is None or score < best[0]:
                best = (score, cand_h, cand_w)
    if best is None:
        return None
    return int(best[1]), int(best[2])


def _track_scalar_grid_panel(
    values: np.ndarray | None,
    *,
    frame_idx: int,
    out_h: int,
    out_w: int,
    valid: np.ndarray | None = None,
    mode: str = "confidence",
) -> np.ndarray | None:
    """Render a per-track scalar/valid matrix as an image-grid panel."""
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim != 2 or frame_idx >= arr.shape[1]:
        return None
    grid_hw = _infer_track_grid_hw(arr.shape[0], out_h, out_w)
    if grid_hw is None:
        return None
    gh, gw = grid_hw
    if gh * gw != arr.shape[0]:
        return None
    frame = arr[:, frame_idx].reshape(gh, gw)
    valid_frame = None
    if valid is not None:
        valid_arr = np.asarray(valid)
        if valid_arr.shape == arr.shape and frame_idx < valid_arr.shape[1]:
            valid_frame = valid_arr[:, frame_idx].reshape(gh, gw).astype(bool)
    if mode == "mask":
        panel = colorize_mask(frame.astype(bool))
    else:
        panel = colorize_confidence(frame.astype(np.float32), valid=valid_frame)
    if (gh, gw) != (out_h, out_w):
        panel = cv2.resize(panel, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return panel


def _camera_frame(matrix: np.ndarray, frame_idx: int) -> np.ndarray | None:
    """Return a rank-2 camera matrix from a static or per-frame input."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] > 0:
        return arr[min(int(frame_idx), arr.shape[0] - 1)]
    return None


def _track_scalar_source_projection_panel(
    source_frame: np.ndarray,
    tracks_world: np.ndarray,
    source_cam2world: np.ndarray,
    source_intrinsics: np.ndarray,
    values: np.ndarray | None,
    *,
    frame_idx: int,
    valid: np.ndarray | None,
    mode: str,
) -> np.ndarray | None:
    """Overlay per-identity scalars at their actual source-camera positions.

    A q1 dense tracker has raster-order identities, but q2 raw-union selection
    intentionally shuffles identities before fixed-budget compaction. Reshaping
    that vector into HxW invents spatial structure. Project each 3D identity
    instead, which is valid for both ordered q1 grids and shuffled q>1 unions.
    """
    if values is None:
        return None
    scalar = np.asarray(values)
    tracks = np.asarray(tracks_world, dtype=np.float32)
    if (
        scalar.ndim != 2
        or tracks.ndim != 3
        or scalar.shape[:2] != tracks.shape[:2]
        or frame_idx >= scalar.shape[1]
    ):
        return None
    c2w = _camera_frame(source_cam2world, frame_idx)
    intrinsics = _camera_frame(source_intrinsics, frame_idx)
    if c2w is None or intrinsics is None:
        return None

    canvas = np.asarray(source_frame, dtype=np.uint8).copy()
    height, width = canvas.shape[:2]
    uv, visible = project_track_frame(
        tracks[:, frame_idx, :], c2w, intrinsics, height, width
    )
    scalar_frame = np.asarray(scalar[:, frame_idx], dtype=np.float32)
    visible &= np.isfinite(scalar_frame)
    if valid is not None:
        valid_arr = np.asarray(valid)
        if valid_arr.shape == scalar.shape:
            visible &= valid_arr[:, frame_idx].astype(bool)
    if not visible.any():
        return canvas

    colors = np.zeros((scalar.shape[0], 3), dtype=np.uint8)
    if mode == "mask":
        colors[visible] = np.array([240, 240, 240], dtype=np.uint8)
    else:
        vals = scalar_frame[visible]
        lo, hi = np.nanpercentile(vals, [2.0, 98.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        norm = np.zeros_like(scalar_frame, dtype=np.float32)
        norm[visible] = np.clip((scalar_frame[visible] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        colors = cv2.applyColorMap(
            (norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
        ).reshape(-1, 3)

    visible_indices = np.flatnonzero(visible)
    u = np.rint(uv[visible_indices, 0]).astype(np.int32)
    v = np.rint(uv[visible_indices, 1]).astype(np.int32)
    pixel_in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not pixel_in_bounds.any():
        return canvas
    visible_indices = visible_indices[pixel_in_bounds]
    u = u[pixel_in_bounds]
    v = v[pixel_in_bounds]
    overlay = np.zeros_like(canvas)
    overlay[v, u] = colors[visible_indices]
    marker_mask = overlay.any(axis=2)
    # Keep the underlying RGB legible even for dense q1/q2 track sets.
    canvas[marker_mask] = (
        0.55 * canvas[marker_mask].astype(np.float32)
        + 0.45 * overlay[marker_mask].astype(np.float32)
    ).astype(np.uint8)
    return canvas


def sequence_depth_range(
    depths,
    valids,
    *,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[float, float] | None:
    """Stable depth range for video panels.

    A per-frame range makes the colormap flicker even when geometry is stable.
    Use one robust range across the whole rendered sequence instead.
    """
    vals_list = [
        np.asarray(depth, dtype=np.float32)[np.asarray(valid, dtype=bool)]
        for depth, valid in zip(depths, valids)
        if np.asarray(valid, dtype=bool).any()
    ]
    vals_list = [v[np.isfinite(v)] for v in vals_list if v.size]
    if not vals_list:
        return None
    vals = np.concatenate(vals_list)
    if vals.size == 0:
        return None
    lo, hi = np.percentile(vals, percentiles)
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return lo, hi


def select_dynamic_track_valid_mask(sample: dict):
    """Match the dense-hybrid training mask selection for viz precompute."""
    sparse = sample.get("tracks_world_valid_mask")
    dense = sample.get("tracks_world_dense_valid_mask")
    mode = os.environ.get("TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_VALID_MODE", "dense")
    mode = str(mode).strip().lower().replace("-", "_")
    if mode in {"sparse", "adapter", "current", "tracks_world_valid_mask"}:
        return sparse if sparse is not None else dense
    if mode in {"none", "off", "false", "0"}:
        return None
    return dense if dense is not None else sparse


def coerce_rgb_frame_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB frame in either uint8 [0,255] or float [0,1] to uint8.

    Dataset loaders use both conventions. The ReCamMaster 4RC eval path keeps
    source frames as PIL/uint8, while most training dataset tensors expose
    normalized float RGB. Visualization must preserve both without clipping the
    uint8 path to white.
    """
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) <= 2.0:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def render_pred_gt_diff(
    pred_frame: np.ndarray, gt_frame: np.ndarray, max_diff: float = 60.0
) -> np.ndarray:
    """``|pred - gt|`` per-pixel L1 (averaged over channels) → turbo heatmap.

    ``max_diff`` saturates the heatmap at ``max_diff / 255`` of full pixel
    range; lower values make subtle artifacts more visible.
    """
    pred_f = np.asarray(pred_frame, dtype=np.float32)
    gt_f = np.asarray(gt_frame, dtype=np.float32)
    if pred_f.shape != gt_f.shape:
        H = min(pred_f.shape[0], gt_f.shape[0])
        W = min(pred_f.shape[1], gt_f.shape[1])
        pred_f = pred_f[:H, :W]
        gt_f = gt_f[:H, :W]
    diff = np.abs(pred_f - gt_f).mean(axis=-1)  # (H, W)
    norm = np.clip(diff / max(max_diff, 1e-6), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


# ---------------------------------------------------------------------------
# Track projection
# ---------------------------------------------------------------------------

def project_track_frame(
    points_t: np.ndarray,
    cam2world_t: np.ndarray,
    intrinsics: np.ndarray,
    H: int,
    W: int,
):
    """Project ``(N, 3)`` world points to a single target frame's pixel UV.

    Returns ``(uv (N, 2), valid (N,))``. ``valid`` is ``True`` only when the
    point is in front of the camera AND inside the image bounds.
    """
    pts = np.asarray(points_t, dtype=np.float32)
    c2w = np.asarray(cam2world_t, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)

    w2c = np.linalg.inv(c2w).astype(np.float32)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ pts.T).T + t[None, :]

    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
    if not valid.any():
        return uv, valid

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z_safe = np.where(valid, z, 1.0)
    uv[:, 0] = pts_cam[:, 0] / z_safe * fx + cx
    uv[:, 1] = pts_cam[:, 1] / z_safe * fy + cy

    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    valid &= in_bounds
    return uv, valid


def identity_colors_from_tracks(tracks_world: np.ndarray) -> np.ndarray:
    """Per-track ``(N, 3)`` uint8 color from frame-0 normalized 3D position."""
    pts0 = np.asarray(tracks_world, dtype=np.float32)[:, 0, :]
    lo = np.percentile(pts0, 2, axis=0)
    hi = np.percentile(pts0, 98, axis=0)
    rng = np.maximum(hi - lo, 1e-6)
    return (np.clip((pts0 - lo) / rng, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def pil_list_to_np(frames, width: int, height: int):
    """Convert PIL or NumPy frames to ``(H, W, 3)`` uint8 arrays."""
    result = []
    for frame in frames:
        arr = coerce_rgb_frame_uint8(frame)
        if arr.shape[:2] != (height, width):
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
        result.append(np.ascontiguousarray(arr))
    return result


def compose_2x4_grid_frame(
    top_panels,
    bot_panels,
    top_labels,
    bot_labels,
    caption: str = "",
):
    """Compose one 2-row grid frame with labels and an optional caption bar.

    Top and bottom rows must have the same number of panels; all panels must
    share the same ``(H, W)`` shape.
    """
    if len(top_panels) != len(bot_panels):
        raise ValueError("top_panels and bot_panels must have the same length")
    if len(top_panels) == 0:
        raise ValueError("at least one panel is required")
    H, W = top_panels[0].shape[:2]
    n_cols = len(top_panels)
    top_row = np.concatenate([p.copy() for p in top_panels], axis=1)
    bot_row = np.concatenate([p.copy() for p in bot_panels], axis=1)

    for i, label in enumerate(top_labels):
        draw_label(top_row, label, pos=(i * W + 4, 16))
    for i, label in enumerate(bot_labels):
        draw_label(bot_row, label, pos=(i * W + 4, 16))

    raw_h = 2 * H
    caption_h = 16 * ((raw_h + 48) // 16 + 1) - raw_h
    if caption_h < 32:
        caption_h += 16

    caption_bar = np.zeros((caption_h, W * n_cols, 3), dtype=np.uint8)
    if caption:
        max_chars = (W * n_cols) // 7
        lines = []
        text = caption
        while text and len(lines) < 2:
            lines.append(text[:max_chars])
            text = text[max_chars:]
        y = 14
        for line in lines:
            cv2.putText(
                caption_bar,
                line,
                (6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            y += 16

    return np.concatenate([top_row, bot_row, caption_bar], axis=0)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def read_video_frames(path, target_size=None):
    """Read an mp4 to a list of ``(H, W, 3)`` uint8 RGB arrays.

    Args:
        path: file path.
        target_size: optional ``(H, W)`` to resize each frame.
    """
    reader = imageio.get_reader(str(path))
    frames = []
    for frame in reader:
        f = np.asarray(frame, dtype=np.uint8)
        if f.ndim == 2:
            f = np.stack([f, f, f], axis=-1)
        elif f.shape[-1] == 4:
            f = f[..., :3]
        if target_size is not None:
            tH, tW = target_size
            if f.shape[:2] != (tH, tW):
                f = cv2.resize(f, (tW, tH), interpolation=cv2.INTER_AREA)
        frames.append(f)
    reader.close()
    return frames


# ---------------------------------------------------------------------------
# Main entry: write_eval_comparison_mp4
# ---------------------------------------------------------------------------

def write_eval_comparison_mp4(
    out_path,
    sample,
    pred_frames,
    *,
    fps: int = 12,
    caption_extra: str = "",
    dense_warp_mode: str = "zbuffer",
    dense_warp_motion_threshold: float = 0.005,
    dense_depth_panel_source: str = "per_frame",
):
    """Write a 2x4 diagnostic ``comparison.mp4`` for one eval pair.

    Layout::

        ┌──────────┬──────────┬──────────┬──────────┐
        │  source  │target GT │   pred   │|pred - gt|│
        ├──────────┼──────────┼──────────┼──────────┤
        │dense  RGB│dense dpth│dense vld │tracks pred│
        └──────────┴──────────┴──────────┴──────────┘

    Args:
        out_path: where to write the .mp4
        sample: dict from ``Syn4D_Multi_NVS.load_pair`` with at minimum
            ``video``, ``target_video``, ``tracks_world``,
            ``target_cam2world``, ``target_intrinsics``, ``prompt``.
            Dense panels also require ``source_rgb_video``,
            ``source_depth_video``, ``source_cam2world``,
            ``source_intrinsics`` (constructor flags
            ``return_source_rgb_video=True`` /
            ``return_source_cameras=True`` / ``return_source_depth_video=True``).
            All cameras / depths must be in the cam0-normalized +
            common_scale-rescaled frame returned by ``load_pair``.
        pred_frames: list of ``(H, W, 3)`` uint8 RGB model output frames.
        fps: output video fps.
        caption_extra: optional prefix appended to the caption bar.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_pil = sample["video"]
    tgt_pil = sample["target_video"]
    H, W = np.asarray(src_pil[0]).shape[:2]

    src_np = pil_list_to_np(src_pil, W, H)
    tgt_np = pil_list_to_np(tgt_pil, W, H)

    pred_np = []
    for f in pred_frames:
        f_arr = np.asarray(f, dtype=np.uint8)
        if f_arr.ndim == 2:
            f_arr = np.stack([f_arr, f_arr, f_arr], axis=-1)
        elif f_arr.shape[-1] == 4:
            f_arr = f_arr[..., :3]
        if f_arr.shape[:2] != (H, W):
            f_arr = cv2.resize(f_arr, (W, H), interpolation=cv2.INTER_AREA)
        pred_np.append(f_arr)

    T = min(len(src_np), len(tgt_np), len(pred_np))
    if T == 0:
        raise ValueError(
            f"empty inputs: src={len(src_np)} tgt={len(tgt_np)} pred={len(pred_np)}"
        )

    tracks_world_raw = sample.get("tracks_world")
    has_tracks = tracks_world_raw is not None
    tracks_world = (
        np.asarray(tracks_world_raw, dtype=np.float32)
        if has_tracks else np.empty((0, T, 3), dtype=np.float32)
    )
    has_tracks = (
        has_tracks
        and tracks_world.ndim == 3
        and tracks_world.shape[0] > 0
        and tracks_world.shape[1] >= T
    )
    tgt_c2w = np.asarray(sample["target_cam2world"], dtype=np.float32)
    tgt_K = np.asarray(sample["target_intrinsics"], dtype=np.float32)
    # Per-frame target K (optional). If absent, broadcast tgt_K to (T,3,3) so
    # downstream re-warp + track-projection can index uniformly. For Vista4D
    # zoom trajectories this carries fx 5049→1999 evolution; for static-K
    # trajectories the broadcast is a no-op.
    tgt_K_pf_raw = sample.get("target_intrinsics_per_frame")
    if tgt_K_pf_raw is not None:
        tgt_K_per_frame = np.asarray(tgt_K_pf_raw, dtype=np.float32)
        if tgt_K_per_frame.ndim == 2:
            tgt_K_per_frame = np.broadcast_to(tgt_K_per_frame, (len(tgt_c2w), 3, 3)).copy()
    else:
        tgt_K_per_frame = np.broadcast_to(tgt_K, (len(tgt_c2w), 3, 3)).copy()

    src_rgb_video = sample.get("source_rgb_video")
    src_depth_video = sample.get("source_depth_video")
    src_c2w = sample.get("source_cam2world")
    src_K = sample.get("source_intrinsics")
    has_dense = all(
        x is not None for x in (src_rgb_video, src_depth_video, src_c2w, src_K)
    )
    # Track-only runs deliberately do not request source_rgb_video because the
    # dense branch is disabled. Their comparison should not advertise three
    # black dense-warp panels; source video/depth remain valid diagnostics.
    track_only_source_depth = None
    track_only_source_depth_valid = None
    track_only_source_depth_range = None
    if not has_dense and src_depth_video is not None:
        depth_arr = np.asarray(src_depth_video, dtype=np.float32)
        if depth_arr.ndim == 3 and depth_arr.shape[0] >= T:
            track_only_source_depth = depth_arr[:T]
            track_only_source_depth_valid = (
                np.isfinite(track_only_source_depth) & (track_only_source_depth > 0)
            )
            track_only_source_depth_range = sequence_depth_range(
                track_only_source_depth, track_only_source_depth_valid
            )

    # If the model ran with --dense_warp_mode hybrid_softsplat, the panel
    # labelled "dense warped RGB" must reflect what the *model actually saw*
    # — i.e. the multi-frame fused soft-splat output, not the legacy per-frame
    # zbuffer scatter produced by forward_warp_rgbd_to_target. Pre-compute the
    # full-sequence multiframe stack once via the same builder the
    # DenseTargetControlUnit uses (build_dense_target_reprojection_hybrid),
    # then index per-frame inside the writer loop.
    multiframe_dense = None  # (T, H, W, 3) uint8 / (T, H, W) float32 / (T, H, W) bool
    if has_dense and str(dense_warp_mode).lower() == "hybrid_softsplat":
        try:
            import torch
            from holo4d.conditions.dense_target_reprojection_builder import (
                build_dense_target_reprojection_hybrid,
            )
            from holo4d.conditions.soft_splat_warp import (
                compute_projected_dynamic_mask_from_tracks_world,
                compute_target_dynamic_mask_from_tracks_world,
            )

            T_full = min(len(src_rgb_video), len(src_depth_video))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            src_rgb_t = torch.from_numpy(
                np.stack([
                    np.asarray(coerce_rgb_frame_uint8(src_rgb_video[t]), dtype=np.float32) / 255.0
                    for t in range(T_full)
                ], axis=0)
            ).unsqueeze(0).to(device=device)  # (1, T, H, W, 3)
            src_depth_t = torch.from_numpy(
                np.stack([np.asarray(src_depth_video[t], dtype=np.float32)
                          for t in range(T_full)], axis=0)
            ).unsqueeze(0).to(device=device)  # (1, T, H, W)
            src_valid_t = (src_depth_t > 0).to(device=device)
            src_c2w_t = torch.from_numpy(np.asarray(src_c2w, dtype=np.float32)
                                          ).unsqueeze(0).to(device=device)
            tgt_c2w_t = torch.from_numpy(np.asarray(tgt_c2w, dtype=np.float32)
                                          ).unsqueeze(0).to(device=device)
            K_t = torch.from_numpy(np.asarray(src_K, dtype=np.float32)
                                    ).to(device=device)
            # Keep a single 3x3 intrinsic as rank-2. The hybrid builder
            # expands (3, 3) to (B, T, 3, 3); passing (1, 3, 3) is
            # interpreted as T=1 and later fails when reshaping to B*T.
            if K_t.ndim == 3 and K_t.shape[0] == 1 and T_full > 1:
                K_t = K_t.expand(T_full, -1, -1)

            def _sample_mask_to_bthw(key):
                value = sample.get(key)
                if value is None:
                    return None
                arr = np.asarray(value).astype(bool)
                if arr.ndim != 3:
                    return None
                arr = arr[:T_full]
                if arr.shape[1:] != (H, W):
                    resized = [
                        cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                        for m in arr
                    ]
                    arr = np.stack(resized, axis=0)
                return torch.from_numpy(arr).unsqueeze(0).to(device=device).bool()

            # Static / dynamic mask: use explicit dataset GT masks first
            # (Syn4D/Kubric/ReCam). Only fall back to 3D motion-threshold masks
            # for datasets that do not provide 2D masks.
            source_static_mask = None
            source_dynamic_mask = _sample_mask_to_bthw("source_dynamic_mask")
            target_dynamic_mask = _sample_mask_to_bthw("target_dynamic_mask")
            if source_dynamic_mask is not None:
                source_static_mask = ~source_dynamic_mask
            tw_t = None
            if has_tracks:
                tw_t = torch.from_numpy(tracks_world).to(device=device)
                if tw_t.ndim == 3:
                    tw_t = tw_t.unsqueeze(0)
            if has_tracks and (source_dynamic_mask is None or target_dynamic_mask is None):
                if source_dynamic_mask is None:
                    src_dyn = compute_projected_dynamic_mask_from_tracks_world(
                        tracks_world=tw_t,
                        cam2world=src_c2w_t,
                        intrinsics=K_t,
                        image_h=H,
                        image_w=W,
                        motion_threshold=float(dense_warp_motion_threshold),
                    )
                    source_static_mask = None if src_dyn is None else ~src_dyn
                if target_dynamic_mask is None:
                    target_dynamic_mask = compute_target_dynamic_mask_from_tracks_world(
                        tracks_world=tw_t,
                        tgt_cam2world=tgt_c2w_t,
                        tgt_intrinsics=K_t,
                        image_h=H,
                        image_w=W,
                        motion_threshold=float(dense_warp_motion_threshold),
                    )

            warped_rgb, warped_depth, warp_valid = build_dense_target_reprojection_hybrid(
                source_rgb=src_rgb_t,
                source_depth=src_depth_t,
                source_depth_valid=src_valid_t,
                src_cam2world=src_c2w_t,
                tgt_cam2world=tgt_c2w_t,
                intrinsics=K_t,
                image_h=H,
                image_w=W,
                source_static_mask=source_static_mask,
                target_dynamic_mask=target_dynamic_mask,
                dynamic_tracks_world=tw_t if has_tracks else None,
                dynamic_tracks_valid_mask=(
                    torch.from_numpy(
                        np.asarray(select_dynamic_track_valid_mask(sample)).astype(bool)
                    ).unsqueeze(0).to(device=device)
                    if has_tracks and select_dynamic_track_valid_mask(sample) is not None
                    else None
                ),
                dynamic_tracks_query_count=sample.get("track_query_count"),
                target_intrinsics=torch.from_numpy(np.asarray(tgt_K_per_frame, dtype=np.float32)
                                                   ).unsqueeze(0).to(device=device),
            )
            mf_rgb = (warped_rgb[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            mf_depth = warped_depth[0].cpu().numpy().astype(np.float32)
            mf_valid = warp_valid[0].cpu().numpy().astype(bool)
            multiframe_dense = (mf_rgb, mf_depth, mf_valid)
        except Exception as _e:
            print(f"[viz] hybrid_softsplat panel pre-compute failed, "
                  f"falling back to zbuffer panel: {_e}")
            multiframe_dense = None

    # Per-frame hard-zbuffer warp is ALWAYS the depth source for the depth panel,
    # even when multiframe_dense is available. Reverts d2cd603's "use mf_depth"
    # choice for the viz layer only — at 384x672 (and higher) that choice produced
    # a near-uniform-blue depth panel because the multiframe builder's channel-4
    # is a depth-weighted average with depth_weight_scale=50:
    #     exp(z_norm * 50) → ratio ~1e21 between near/far weights
    #     out_z ≈ z_min(contributing source pts)
    # so any sub-pixel splat that lets a foreground source pixel leak into a
    # background target pixel pulls that pixel's mf_depth down to foreground z.
    # The leakage compounds at higher resolutions because each source pixel
    # bilinear-scatters into 4 neighbouring target pixels, and the target grid
    # has ~target_area / source_area pixels per source point. Result: at
    # 384x672 vs 280x504 4RC labels the foreground "wins" the softmax across
    # most pixels and the depth panel collapses to ~min depth everywhere.
    # The per-frame zbuffer warp uses hard min-z (no softmax bias) and produces
    # an informative depth panel that matches the colormap range over the
    # actual scene depth distribution.
    # RGB / valid panels still come from multiframe_dense when available, so
    # they continue to reflect what the model saw (the d2cd603 fix's original
    # intent is preserved for those panels).
    # See doc/track4d-360/bugs/2026-05-11-comparison-mp4-depth-panel-blue-at-high-res.md.
    per_frame_warp = None  # list of (rgb, depth, valid) length T
    if has_dense:
        per_frame_warp = []
        for _t in range(T):
            _pf_rgb, _pf_depth, _pf_valid = forward_warp_rgbd_to_target(
                src_rgb=coerce_rgb_frame_uint8(src_rgb_video[_t]),
                src_depth=np.asarray(src_depth_video[_t], dtype=np.float32),
                src_cam2world=np.asarray(src_c2w[_t], dtype=np.float32),
                src_intrinsics=np.asarray(src_K, dtype=np.float32),
                tgt_cam2world=tgt_c2w[_t],
                tgt_intrinsics=tgt_K_per_frame[_t],
                src_cam0_pose=None,
                common_scale=1.0,
            )
            per_frame_warp.append(
                (_pf_rgb, _pf_depth, np.asarray(_pf_valid, dtype=bool))
            )

    depth_range = None
    if per_frame_warp:
        depth_range = sequence_depth_range(
            [d for _, d, _v in per_frame_warp],
            [v for _, _d, v in per_frame_warp],
        )
    multiframe_depth_range = None
    if multiframe_dense is not None:
        _mf_rgb, _mf_depth, _mf_valid = multiframe_dense
        multiframe_depth_range = sequence_depth_range(_mf_depth, _mf_valid)

    identity_colors = (
        identity_colors_from_tracks(tracks_world)
        if has_tracks else np.empty((0, 3), dtype=np.uint8)
    )
    track_valid_mask_raw = sample.get("tracks_world_valid_mask")
    track_valid_mask = None
    track_valid_label = "track valid | source projection"
    if has_tracks and track_valid_mask_raw is not None:
        track_valid_mask = np.asarray(track_valid_mask_raw).astype(bool)
        if track_valid_mask.shape[:2] != tracks_world.shape[:2]:
            track_valid_mask = None
    if has_tracks and track_valid_mask is None:
        # Legacy q1 Syn4D intentionally omits an explicit per-frame visibility
        # field. The inference path derives usable identities from finite,
        # nonzero geometry; use that same conservative diagnostic fallback
        # instead of rendering a misleading all-black "track valid" panel.
        track_valid_mask = np.isfinite(tracks_world).all(axis=-1)
        track_valid_mask &= np.linalg.norm(tracks_world, axis=-1) > 1e-8
        track_valid_label = "track geometry valid (derived) | source projection"
    track_confidence_raw = sample.get("tracks_world_confidence")
    track_confidence = None
    if has_tracks and track_confidence_raw is not None:
        track_confidence = np.asarray(track_confidence_raw, dtype=np.float32)
        if track_confidence.shape[:2] != tracks_world.shape[:2]:
            track_confidence = None
    track_dynamic_mask_raw = sample.get("source_dynamic_mask_at_tracks")
    track_dynamic_mask = None
    if has_tracks and track_dynamic_mask_raw is not None:
        track_dynamic_mask = np.asarray(track_dynamic_mask_raw).astype(bool)
        if track_dynamic_mask.shape[:2] != tracks_world.shape[:2]:
            track_dynamic_mask = None

    prompt = str(sample.get("prompt", ""))
    caption = caption_extra.strip()
    if prompt:
        caption = f"{caption} | {prompt}" if caption else prompt

    frames_out = []
    for t in range(T):
        if has_tracks:
            uv_t, vis_t = project_track_frame(
                tracks_world[:, t, :], tgt_c2w[t], tgt_K_per_frame[t], H, W
            )
            if track_valid_mask is not None and t < track_valid_mask.shape[1]:
                if track_dynamic_mask is not None:
                    dynamic_ids = track_dynamic_mask[:, 0]
                    vis_t[dynamic_ids] &= track_valid_mask[dynamic_ids, t]
                else:
                    vis_t = vis_t & track_valid_mask[:, t]
            tracks_on_pred = draw_tracks_on_frame(
                pred_np[t].copy(), uv_t, vis_t, colors=identity_colors, radius=1
            )
        else:
            tracks_on_pred = pred_np[t].copy()
        diff_panel = render_pred_gt_diff(pred_np[t], tgt_np[t])

        if has_dense and multiframe_dense is not None:
            # RGB + valid: from the multiframe soft-splat builder so they
            # reflect the multi-frame fusion the model actually saw.
            # Depth: from per-frame hard-zbuffer warp (per_frame_warp[t][1])
            # instead of the multiframe builder's softmax-blended channel-4.
            # The softmax bias (depth_weight_scale=50, weight = exp(z_norm*50))
            # collapses mf_depth to ≈min_z over contributing source pts,
            # which renders as near-uniform foreground depth at high target
            # resolutions because sub-pixel splat leakage spreads foreground
            # pixels across more neighbours. See doc bug note 2026-05-11.
            mf_rgb, mf_depth, mf_valid = multiframe_dense
            pf_rgb, pf_depth, pf_valid = per_frame_warp[t]
            dense_rgb = mf_rgb[t]
            if str(dense_depth_panel_source).lower() in {"multiframe", "hybrid", "model"}:
                dense_depth_vis = colorize_depth(mf_depth[t], mf_valid[t], multiframe_depth_range)
            else:
                dense_depth_vis = colorize_depth(pf_depth, pf_valid, depth_range)
            dense_valid = mf_valid[t]
            dense_valid_vis = colorize_mask(dense_valid)
        elif has_dense:
            pf_rgb, pf_depth, pf_valid = per_frame_warp[t]
            dense_rgb = pf_rgb
            dense_depth_vis = colorize_depth(pf_depth, pf_valid, depth_range)
            dense_valid_vis = colorize_mask(pf_valid)
        else:
            dense_rgb = src_np[t].copy()
            if track_only_source_depth is not None:
                dense_depth_vis = colorize_depth(
                    track_only_source_depth[t],
                    track_only_source_depth_valid[t],
                    track_only_source_depth_range,
                )
                dense_valid_vis = colorize_mask(track_only_source_depth_valid[t])
            else:
                dense_depth_vis = np.zeros((H, W, 3), dtype=np.uint8)
                dense_valid_vis = np.zeros((H, W, 3), dtype=np.uint8)

        track_valid_panel = _track_scalar_source_projection_panel(
            src_np[t], tracks_world, src_c2w, src_K, track_valid_mask,
            frame_idx=t, valid=None, mode="mask",
        )
        if track_valid_panel is None:
            track_valid_panel = _track_scalar_grid_panel(
                track_valid_mask,
                frame_idx=t, out_h=H, out_w=W, mode="mask",
            )
        if track_valid_panel is None:
            track_valid_panel = np.zeros((H, W, 3), dtype=np.uint8)
        track_quality_panel = _track_scalar_source_projection_panel(
            src_np[t], tracks_world, src_c2w, src_K, track_confidence,
            frame_idx=t, valid=track_valid_mask, mode="confidence",
        )
        track_quality_label = "track confidence | source projection"
        if track_quality_panel is None:
            track_quality_panel = _track_scalar_grid_panel(
                track_confidence,
                frame_idx=t, out_h=H, out_w=W, valid=track_valid_mask,
                mode="confidence",
            )
            track_quality_label = "track confidence grid (ordered q1 only)"
        if track_quality_panel is None:
            track_quality_panel = track_valid_panel
            track_quality_label = track_valid_label

        frame = compose_2x4_grid_frame(
            top_panels=[src_np[t], tgt_np[t], pred_np[t], diff_panel, track_quality_panel],
            bot_panels=[
                dense_rgb,
                dense_depth_vis,
                dense_valid_vis,
                track_valid_panel,
                tracks_on_pred,
            ],
            top_labels=["source", "target GT", "pred", "|pred - gt|", track_quality_label],
            bot_labels=[
                (
                    f"dense warped RGB ({'multiframe' if multiframe_dense is not None else 'zbuffer'})"
                    if has_dense else "source RGB (track-only condition)"
                ),
                (
                    "dense warped depth (multiframe)"
                    if has_dense and multiframe_dense is not None
                    and str(dense_depth_panel_source).lower() in {"multiframe", "hybrid", "model"}
                    else ("source depth (track-only condition)" if not has_dense else "dense warped depth")
                ),
                "source depth valid (track-only condition)" if not has_dense else "dense valid",
                track_valid_label,
                "tracks on pred",
            ],
            caption=caption,
        )
        frames_out.append(np.asarray(frame, dtype=np.uint8))

    save_mp4(out_path, np.stack(frames_out, axis=0), fps=fps)

    return out_path
