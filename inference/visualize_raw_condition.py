#!/usr/bin/env python3
"""Render a compact diagnostic video from a portable Holo4D condition bundle.

The first output shows source RGB/depth, source-view dense tracks, a dense
source-depth reprojection into the requested target camera, and the camera path.
If a generated target video is supplied, the second output overlays the same
condition tracks on it so the two geometry views cannot be confused.

This renderer is intentionally NumPy/OpenCV-only; it does not import the model
or any external reconstructor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def factor_grid(count: int, target_aspect: float) -> tuple[int, int]:
    """Infer the query grid dimensions from its count and the image aspect."""
    pairs = [(h, count // h) for h in range(1, int(np.sqrt(count)) + 1) if count % h == 0]
    candidates = [shape for h, w in pairs for shape in ((h, w), (w, h))]
    return min(candidates, key=lambda hw: abs(hw[0] / hw[1] - target_aspect))


def _colors(anchor_uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """CoWTracker-style Bremm colors anchored at each track's source pixel."""
    xy = np.asarray(anchor_uv, dtype=np.float32)
    nx = np.clip(xy[:, 0] / max(width - 1, 1), 0.0, 1.0)
    ny = np.clip(xy[:, 1] / max(height - 1, 1), 0.0, 1.0)
    hsv = np.stack(
        [(np.mod(nx * 0.8 + ny * 0.2, 1.0) * 179).astype(np.uint8),
         ((0.6 + 0.4 * (1.0 - ny)) * 255).astype(np.uint8),
         ((0.7 + 0.3 * nx) * 255).astype(np.uint8)], axis=1
    ).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return bgr[:, ::-1].copy()


def _frame_intrinsics(K: np.ndarray, frame: int) -> np.ndarray:
    return K[frame] if K.ndim == 3 else K


def _project(points_world: np.ndarray, cam2world: np.ndarray, K: np.ndarray):
    cam = (points_world - cam2world[:3, 3]) @ cam2world[:3, :3]
    z = cam[:, 2]
    hom = cam @ K.T
    uv = hom[:, :2] / np.maximum(hom[:, 2:3], 1e-8)
    return uv, z


def _draw_trails(rgb: np.ndarray, uv: np.ndarray, visible: np.ndarray, colors: np.ndarray,
                 radius: int = 1) -> np.ndarray:
    canvas = np.clip(np.asarray(rgb, dtype=np.float32) * 0.4, 0, 255).astype(np.uint8)
    for i in range(uv.shape[0]):
        valid_t = np.flatnonzero(visible[i])
        if valid_t.size < 1:
            continue
        color = tuple(int(c) for c in colors[i])
        for a, b in zip(valid_t[:-1], valid_t[1:]):
            if b != a + 1:
                continue
            p0 = tuple(np.rint(uv[i, a]).astype(int))
            p1 = tuple(np.rint(uv[i, b]).astype(int))
            cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        t = int(valid_t[-1])
        x, y = np.rint(uv[i, t]).astype(int)
        cv2.circle(canvas, (x, y), radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
    return canvas


def _depth_rgb(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    bgr = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 360), 34), (18, 22, 28), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (245, 245, 245), 1, cv2.LINE_AA)
    return out


def _camera_panel(source_c2w: np.ndarray, target_c2w: np.ndarray, t: int,
                  height: int, width: int) -> np.ndarray:
    """Top-down path relative to source camera, with current target pose marked."""
    panel = np.full((height, width, 3), 247, dtype=np.uint8)
    src = source_c2w[0]
    relative = np.stack([(c[:3, 3] - src[:3, 3]) @ src[:3, :3] for c in target_c2w])
    # x-z is the horizontal plane in this camera convention; y is height.
    xy = relative[:, [0, 2]]
    lo, hi = np.percentile(xy, [2, 98], axis=0)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 0.12)
    lo, hi = center - 0.5 * span, center + 0.5 * span
    margin = 48

    def pixel(point):
        q = (point - lo) / span
        return int(margin + q[0] * (width - 2 * margin)), int(height - margin - q[1] * (height - 2 * margin))

    pts = [pixel(p) for p in xy]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(panel, a, b, (58, 111, 177), 3, cv2.LINE_AA)
    cv2.circle(panel, pixel(np.zeros(2, dtype=np.float32)), 8, (195, 72, 64), -1, cv2.LINE_AA)
    for p in pts[: t + 1]:
        cv2.circle(panel, p, 3, (80, 132, 92), -1, cv2.LINE_AA)
    cv2.circle(panel, pts[t], 9, (236, 160, 62), -1, cv2.LINE_AA)
    cv2.putText(panel, "source", (pixel(np.zeros(2, dtype=np.float32))[0] + 8,
                                   pixel(np.zeros(2, dtype=np.float32))[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(panel, f"target camera  {t + 1}/{len(target_c2w)}", (12, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def _warp_source_to_target(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray,
                           src_c2w: np.ndarray, tgt_c2w: np.ndarray, K: np.ndarray,
                           *, source_K: np.ndarray | None = None):
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3).astype(np.float32)
    rays = pixels @ np.linalg.inv(K if source_K is None else source_K).T
    z = depth.reshape(-1)
    ok = valid.reshape(-1) & np.isfinite(z) & (z > 1e-6)
    source_xyz = rays[ok] * z[ok, None]
    world = source_xyz @ src_c2w[:3, :3].T + src_c2w[:3, 3]
    target_xyz = (world - tgt_c2w[:3, 3]) @ tgt_c2w[:3, :3]
    target_z = target_xyz[:, 2]
    projected = target_xyz @ K.T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
    finite_inside = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(target_z)
        & (target_z > 1e-6)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    uv = uv[finite_inside]
    target_z = target_z[finite_inside]
    src_ids = np.flatnonzero(ok)[finite_inside]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    inside_rounded = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = u[inside_rounded], v[inside_rounded]
    src_ids, target_z = src_ids[inside_rounded], target_z[inside_rounded]
    flat = v * w + u
    zvals = target_z
    out = np.zeros((h * w, 3), dtype=np.uint8)
    dep = np.zeros((h * w,), dtype=np.float32)
    if flat.size:
        order = np.lexsort((zvals, flat))
        flat, src_ids, zvals = flat[order], src_ids[order], zvals[order]
        first = np.r_[True, flat[1:] != flat[:-1]]
        flat, src_ids, zvals = flat[first], src_ids[first], zvals[first]
        out[flat] = rgb.reshape(-1, 3)[src_ids]
        dep[flat] = zvals
    mask = dep.reshape(h, w) > 0
    return out.reshape(h, w, 3), dep.reshape(h, w), mask


def _write(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                               macro_block_size=1, pixelformat="yuv420p")
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def render(condition_path: Path, out_dir: Path, generated_path: Path | None,
           max_tracks: int, fps: int) -> dict:
    with np.load(condition_path, allow_pickle=False) as data:
        required = {"source_video", "source_depth_video", "tracks_world",
                    "tracks_world_valid_mask", "source_cam2world", "source_intrinsics",
                    "target_cam2world", "target_intrinsics"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"condition is missing fields: {sorted(missing)}")
        source = data["source_video"].astype(np.uint8)
        depth = data["source_depth_video"].astype(np.float32)
        tracks = data["tracks_world"].astype(np.float32)
        valid_tracks = data["tracks_world_valid_mask"].astype(bool)
        src_c2w = data["source_cam2world"].astype(np.float32)
        tgt_c2w = data["target_cam2world"].astype(np.float32)
        src_K = data["source_intrinsics"].astype(np.float32)
        tgt_K = data["target_intrinsics"].astype(np.float32)
        backend = str(data["source_backend"].item()) if "source_backend" in data.files else "unknown"

    t_count, height, width = source.shape[:3]
    grid_h, grid_w = factor_grid(tracks.shape[0], height / width)
    if grid_h * grid_w != tracks.shape[0]:
        raise ValueError("cannot infer dense track grid")
    stride = max(1, int(np.ceil(np.sqrt(tracks.shape[0] / max_tracks))))
    ids = np.arange(tracks.shape[0]).reshape(grid_h, grid_w)[::stride, ::stride].reshape(-1)
    if ids.size > max_tracks:
        ids = ids[np.linspace(0, ids.size - 1, max_tracks, dtype=np.int64)]
    sampled_tracks = tracks[ids]
    sampled_valid = valid_tracks[ids]
    source_uv = np.zeros((len(ids), t_count, 2), dtype=np.float32)
    target_uv = np.zeros_like(source_uv)
    source_vis = sampled_valid.copy()
    target_vis = sampled_valid.copy()
    for t in range(t_count):
        source_uv[:, t], z0 = _project(sampled_tracks[:, t], src_c2w[t], _frame_intrinsics(src_K, t))
        target_uv[:, t], z1 = _project(sampled_tracks[:, t], tgt_c2w[t], _frame_intrinsics(tgt_K, t))
        source_vis[:, t] &= (z0 > 1e-6) & (source_uv[:, t, 0] >= 0) & (source_uv[:, t, 0] < width) & (source_uv[:, t, 1] >= 0) & (source_uv[:, t, 1] < height)
        target_vis[:, t] &= (z1 > 1e-6) & (target_uv[:, t, 0] >= 0) & (target_uv[:, t, 0] < width) & (target_uv[:, t, 1] >= 0) & (target_uv[:, t, 1] < height)

    colors = _colors(source_uv[:, 0], width, height)

    depth_valid = np.isfinite(depth) & (depth > 1e-6)
    dvals = depth[depth_valid]
    dlo, dhi = np.percentile(dvals, [2, 98]) if dvals.size else (0.0, 1.0)
    gen_frames = None
    if generated_path is not None:
        gen_frames = imageio.mimread(str(generated_path))
        if len(gen_frames) != t_count:
            raise ValueError(f"generated video has {len(gen_frames)} frames; condition has {t_count}")
        gen_frames = [cv2.resize(np.asarray(frame)[..., :3], (width, height), interpolation=cv2.INTER_AREA)
                      if np.asarray(frame).shape[:2] != (height, width) else np.asarray(frame)[..., :3]
                      for frame in gen_frames]

    review, postgen = [], []
    for t in range(t_count):
        trail_start = max(0, t - 13)
        target_rgb, target_depth, target_depth_valid = _warp_source_to_target(
            source[t], depth[t], depth_valid[t], src_c2w[t], tgt_c2w[t],
            _frame_intrinsics(tgt_K, t), source_K=_frame_intrinsics(src_K, t)
        )
        target_depth_rgb = _depth_rgb(target_depth, target_depth_valid, dlo, dhi)
        source_track = _draw_trails(source[t], source_uv[:, trail_start : t + 1],
                                    source_vis[:, trail_start : t + 1], colors)
        target_track = _draw_trails(target_rgb, target_uv[:, trail_start : t + 1],
                                    target_vis[:, trail_start : t + 1], colors)
        panels = [
            _label(source[t], "source RGB"),
            _label(_depth_rgb(depth[t], depth_valid[t], dlo, dhi), f"source depth | backend {backend.upper()}"),
            _label(source_track, f"tracks in source camera | backend {backend.upper()}"),
            _label(target_rgb, f"{backend.upper()} depth reprojection | target camera"),
            _label(target_depth_rgb, "reprojected target depth"),
            _label(_camera_panel(src_c2w, tgt_c2w, t, height, width), "relative camera path"),
        ]
        review.append(np.concatenate([np.concatenate(panels[:3], axis=1),
                                     np.concatenate(panels[3:], axis=1)], axis=0))
        if gen_frames is not None:
            postgen.append(_label(_draw_trails(gen_frames[t], target_uv[:, trail_start : t + 1],
                                                target_vis[:, trail_start : t + 1], colors),
                                  f"projected source tracks | {backend.upper()} | visualization"))

    out_dir.mkdir(parents=True, exist_ok=True)
    backend_tag = backend if backend in {"4rc", "cow_vggt"} else "unknown"
    condition_video = out_dir / f"backend_{backend_tag}_condition_review_2x3.mp4"
    _write(condition_video, review, fps)
    result = {"backend_condition_review": str(condition_video), "backend": backend,
              "frames": t_count, "height": height, "width": width,
              "displayed_tracks": int(len(ids)), "track_grid": [grid_h, grid_w]}
    if postgen:
        overlay_video = out_dir / f"postgen_{backend_tag}_condition_track_overlay.mp4"
        _write(overlay_video, postgen, fps)
        result["post_generation_overlay"] = str(overlay_video)
    (out_dir / "visualization_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-video", type=Path)
    parser.add_argument("--max-tracks", type=int, default=256)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(render(args.condition, args.output_dir, args.generated_video,
                            args.max_tracks, args.fps), indent=2), flush=True)


if __name__ == "__main__":
    main()
