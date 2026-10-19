#!/usr/bin/env python3
"""Visualize CowTracker 2D tracks lifted with VGGT-Omega depth/cameras."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def pixel_center_resize_xy(
    xy: np.ndarray,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    out = np.asarray(xy, dtype=np.float32).copy()
    out[..., 0] = (out[..., 0] + 0.5) * (float(dst_w) / float(src_w)) - 0.5
    out[..., 1] = (out[..., 1] + 0.5) * (float(dst_h) / float(src_h)) - 0.5
    return out


def center_crop_box_for_aspect(raw_hw: tuple[int, int], target_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return left, top, width, height for center crop before resize."""
    raw_h, raw_w = raw_hw
    target_h, target_w = target_hw
    raw_aspect = float(raw_w) / float(raw_h)
    target_aspect = float(target_w) / float(target_h)
    if raw_aspect > target_aspect:
        crop_h = float(raw_h)
        crop_w = crop_h * target_aspect
    else:
        crop_w = float(raw_w)
        crop_h = crop_w / target_aspect
    left = (float(raw_w) - crop_w) * 0.5
    top = (float(raw_h) - crop_h) * 0.5
    return left, top, crop_w, crop_h


def map_xy_between_center_crops(
    xy: np.ndarray,
    *,
    raw_hw: tuple[int, int],
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> np.ndarray:
    """Map pixel centers between two center-crop-resize views of one raw frame."""
    xy = np.asarray(xy, dtype=np.float32)
    src_left, src_top, src_w, src_h = center_crop_box_for_aspect(raw_hw, src_hw)
    dst_left, dst_top, dst_w, dst_h = center_crop_box_for_aspect(raw_hw, dst_hw)
    src_h_out, src_w_out = src_hw
    dst_h_out, dst_w_out = dst_hw

    raw_x = (xy[..., 0] + 0.5) * (src_w / float(src_w_out)) + src_left - 0.5
    raw_y = (xy[..., 1] + 0.5) * (src_h / float(src_h_out)) + src_top - 0.5

    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (raw_x - dst_left + 0.5) * (float(dst_w_out) / dst_w) - 0.5
    out[..., 1] = (raw_y - dst_top + 0.5) * (float(dst_h_out) / dst_h) - 0.5
    return out


def map_xy_between_preprocessed_views(
    xy: np.ndarray,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    raw_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Map xy between preprocessed views, using crop metadata when available.

    ReCamMaster CoWTracker currently runs at 560x336 while VGGT-Omega runs at
    512x288. Those aspect ratios differ, so a simple resize is wrong; the
    coordinates must pass through the raw-frame center-crop boxes. Older
    staged datasets may not carry raw source shape metadata, so those keep the
    historical pixel-center resize fallback.
    """
    if raw_hw is None or raw_hw[0] <= 0 or raw_hw[1] <= 0:
        return pixel_center_resize_xy(xy, src_hw=src_hw, dst_hw=dst_hw)
    return map_xy_between_center_crops(xy, raw_hw=raw_hw, src_hw=src_hw, dst_hw=dst_hw)


def unproject_depth_to_world(
    xy: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    cam2world: np.ndarray,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    c2w = np.asarray(cam2world, dtype=np.float32)
    z = depth
    x = (xy[..., 0] - K[0, 2]) / K[0, 0] * z
    y = (xy[..., 1] - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], axis=-1)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32)


def project_world_to_image(
    world: np.ndarray,
    world2cam: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world = np.asarray(world, dtype=np.float32)
    w2c = np.asarray(world2cam, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[..., 2]
    z_safe = np.maximum(z, 1e-7)
    x = K[0, 0] * cam[..., 0] / z_safe + K[0, 2]
    y = K[1, 1] * cam[..., 1] / z_safe + K[1, 2]
    return np.stack([x, y], axis=-1).astype(np.float32), z.astype(np.float32)


def assign_shard(sequences: list[str], num_shards: int, shard_index: int) -> list[str]:
    return [seq for idx, seq in enumerate(sequences) if idx % num_shards == shard_index]


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros(depth.shape, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return turbo_rgb(out)


def turbo_rgb(values: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    import matplotlib

    x = (np.asarray(values, dtype=np.float32) - float(vmin)) / max(float(vmax) - float(vmin), 1e-6)
    x = np.clip(x, 0.0, 1.0)
    colormaps = getattr(matplotlib, "colormaps", None)
    if colormaps is not None:
        cmap = colormaps.get_cmap("turbo")
    else:
        from matplotlib import pyplot as plt
        cmap = plt.get_cmap("turbo")
    rgb = cmap(x)[..., :3]
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def colorize_values(values: np.ndarray, vmax: float) -> np.ndarray:
    return turbo_rgb(values, vmin=0.0, vmax=float(vmax))


def make_colors(query_xy: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    x = np.clip(query_xy[:, 0] / max(w - 1, 1), 0.0, 1.0)
    y = np.clip(query_xy[:, 1] / max(h - 1, 1), 0.0, 1.0)
    return turbo_rgb(0.65 * x + 0.35 * y)


def resize_rgb(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def load_source_rgb(frame_paths: np.ndarray, hw: tuple[int, int], T: int) -> np.ndarray:
    frames = []
    for path in frame_paths[:T]:
        frame = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.uint8)
        frames.append(resize_rgb(frame, hw))
    return np.stack(frames, axis=0)


def bilinear_sample(map_hw: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = map_hw.shape[:2]
    x = xy[:, 0]
    y = xy[:, 1]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
    x0 = np.floor(np.clip(x, 0, w - 1)).astype(np.int32)
    y0 = np.floor(np.clip(y, 0, h - 1)).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    v00 = map_hw[y0, x0]
    v01 = map_hw[y0, x1]
    v10 = map_hw[y1, x0]
    v11 = map_hw[y1, x1]
    out = (
        v00 * (1 - wx) * (1 - wy)
        + v01 * wx * (1 - wy)
        + v10 * (1 - wx) * wy
        + v11 * wx * wy
    )
    return out.astype(np.float32), valid


def draw_tracks(
    background: np.ndarray,
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    colors: np.ndarray,
    t: int,
    *,
    history: int,
    radius: int,
) -> np.ndarray:
    img = Image.fromarray(background.copy())
    draw = ImageDraw.Draw(img)
    t0 = max(0, t - int(history))
    for i in range(tracks_xy.shape[1]):
        pts = []
        for tt in range(t0, t + 1):
            if valid[tt, i]:
                pts.append((float(tracks_xy[tt, i, 0]), float(tracks_xy[tt, i, 1])))
        if len(pts) >= 2:
            draw.line(pts, fill=tuple(int(v) for v in colors[i]), width=1)
        if pts:
            x, y = pts[-1]
            r = max(1, radius)
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=tuple(int(v) for v in colors[i]),
            )
    return np.asarray(img, dtype=np.uint8)


def draw_scalar_points(
    background: np.ndarray,
    xy_t: np.ndarray,
    valid_t: np.ndarray,
    values_t: np.ndarray,
    *,
    vmax: float,
    radius: int,
) -> np.ndarray:
    img = Image.fromarray(background.copy())
    draw = ImageDraw.Draw(img)
    colors = colorize_values(values_t, vmax=vmax)
    for (x, y), ok, color in zip(xy_t, valid_t, colors):
        if not ok:
            continue
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=tuple(int(v) for v in color),
        )
    return np.asarray(img, dtype=np.uint8)


def draw_label(frame: np.ndarray, label: str) -> None:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((4, 4, 4 + 7 * len(label) + 8, 22), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    frame[:] = np.asarray(img, dtype=np.uint8)


def compose_grid(panels: list[np.ndarray], labels: list[str], caption: str) -> np.ndarray:
    labeled = []
    for panel, label in zip(panels, labels):
        p = panel.copy()
        draw_label(p, label)
        labeled.append(p)
    top = np.concatenate(labeled[:3], axis=1)
    bot = np.concatenate(labeled[3:], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    cap_h = 28
    out = np.zeros((grid.shape[0] + cap_h, grid.shape[1], 3), dtype=np.uint8)
    out[:grid.shape[0]] = grid
    img = Image.fromarray(out)
    ImageDraw.Draw(img).text((8, grid.shape[0] + 8), caption, fill=(255, 255, 255), font=ImageFont.load_default())
    return np.asarray(img, dtype=np.uint8)


def ensure_c2w_from_w2c(w2c: np.ndarray) -> np.ndarray:
    w2c4 = np.eye(4, dtype=np.float32)
    w2c4[:3, :4] = w2c.astype(np.float32)
    return np.linalg.inv(w2c4).astype(np.float32)


def render_sequence(args: argparse.Namespace, seq: str) -> bool:
    cow_npz = Path(args.cow_root) / f"{seq}_pred_trackvideo.npz"
    pose_npz = Path(args.vggt_root) / seq / "pose" / "vggt_omega.npz"
    depth_npz = Path(args.vggt_root) / seq / "geometry" / "vggt_omega_depth.npz"
    out_dir = Path(args.output_root) / seq
    out_mp4 = out_dir / "cow_vggt_dense3d_comparison.mp4"
    out_npz = out_dir / "cow_vggt_dense3d_sampled.npz"
    if args.skip_existing and out_mp4.is_file() and out_npz.is_file():
        print(f"[cow-vggt-viz] skip existing {seq}", flush=True)
        return True
    if not cow_npz.is_file() or not pose_npz.is_file() or not depth_npz.is_file():
        print(f"[cow-vggt-viz] missing inputs for {seq}", flush=True)
        return False

    cow = np.load(cow_npz, allow_pickle=True)
    pose = np.load(pose_npz, allow_pickle=True)
    geom = np.load(depth_npz, allow_pickle=True)
    cow_xy = np.asarray(cow["tracks_xy"], dtype=np.float32)
    cow_vis = np.asarray(cow["visibility"], dtype=bool)
    cow_conf = np.asarray(cow["confidence"], dtype=np.float32)
    raw_hw = None
    if "source_video_shape_thwc" in cow:
        raw_shape = np.asarray(cow["source_video_shape_thwc"], dtype=np.int32)
        if raw_shape.size >= 3:
            raw_hw = (int(raw_shape[1]), int(raw_shape[2]))
    depth = np.asarray(geom["depth"], dtype=np.float32)
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth_conf = np.asarray(geom["depth_conf"], dtype=np.float32)
    if depth_conf.ndim == 4 and depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]
    intr = np.asarray(geom["intrinsics"], dtype=np.float32)
    w2c = np.asarray(geom["extrinsics_w2c"], dtype=np.float32)
    frame_paths = pose["frame_paths"]

    T = min(cow_xy.shape[0], depth.shape[0], intr.shape[0], w2c.shape[0], len(frame_paths))
    cow_xy = cow_xy[:T]
    cow_vis = cow_vis[:T]
    cow_conf = cow_conf[:T]
    depth = depth[:T]
    depth_conf = depth_conf[:T]
    intr = intr[:T]
    w2c = w2c[:T]
    cow_h, cow_w = cow_xy.shape[1:3]
    vggt_h, vggt_w = depth.shape[1:3]

    yy, xx = np.mgrid[0:cow_h:args.track_stride, 0:cow_w:args.track_stride]
    qy = yy.reshape(-1)
    qx = xx.reshape(-1)
    query_xy = np.stack([qx, qy], axis=-1).astype(np.float32)
    sampled_cow_xy = cow_xy[:, qy, qx]
    sampled_cow_vis = cow_vis[:, qy, qx]
    sampled_cow_conf = cow_conf[:, qy, qx]
    sampled_vggt_xy = map_xy_between_preprocessed_views(
        sampled_cow_xy,
        src_hw=(cow_h, cow_w),
        dst_hw=(vggt_h, vggt_w),
        raw_hw=raw_hw,
    )

    world = np.zeros((T, query_xy.shape[0], 3), dtype=np.float32)
    lifted_valid = np.zeros((T, query_xy.shape[0]), dtype=bool)
    reproj_xy_vggt = np.zeros_like(sampled_vggt_xy)
    reproj_error = np.zeros((T, query_xy.shape[0]), dtype=np.float32)
    for t in range(T):
        d, in_bounds = bilinear_sample(depth[t], sampled_vggt_xy[t])
        dc, conf_bounds = bilinear_sample(depth_conf[t], sampled_vggt_xy[t])
        valid = (
            sampled_cow_vis[t]
            & (sampled_cow_conf[t] >= float(args.cow_conf_threshold))
            & in_bounds
            & conf_bounds
            & np.isfinite(d)
            & (d > 1e-6)
            & np.isfinite(dc)
            & (dc >= float(args.depth_conf_threshold))
        )
        c2w = ensure_c2w_from_w2c(w2c[t])
        world[t] = unproject_depth_to_world(sampled_vggt_xy[t], d, intr[t], c2w)
        reproj_xy_vggt[t], reproj_z = project_world_to_image(world[t], w2c[t], intr[t])
        reproj_ok = reproj_z > 1e-6
        err = np.linalg.norm(reproj_xy_vggt[t] - sampled_vggt_xy[t], axis=-1)
        lifted_valid[t] = valid & reproj_ok & np.isfinite(err)
        reproj_error[t] = err

    reproj_xy_cow = map_xy_between_preprocessed_views(
        reproj_xy_vggt,
        src_hw=(vggt_h, vggt_w),
        dst_hw=(cow_h, cow_w),
        raw_hw=raw_hw,
    )
    cow_track_valid = (
        sampled_cow_vis
        & np.isfinite(sampled_cow_xy).all(axis=-1)
        & (sampled_cow_xy[..., 0] >= 0)
        & (sampled_cow_xy[..., 1] >= 0)
        & (sampled_cow_xy[..., 0] <= cow_w - 1)
        & (sampled_cow_xy[..., 1] <= cow_h - 1)
    )
    motion = np.linalg.norm(world - world[:1], axis=-1)
    motion[~lifted_valid] = 0.0
    colors = make_colors(query_xy, (cow_h, cow_w))
    source_rgb = load_source_rgb(frame_paths, (cow_h, cow_w), T)
    depth_vis = np.stack([resize_rgb(normalize_depth(depth[t]), (cow_h, cow_w)) for t in range(T)], axis=0)

    frames = []
    err_vmax = float(args.error_vmax)
    motion_vmax = float(args.motion_vmax)
    for t in range(T):
        cow_panel = draw_tracks(
            (source_rgb[t].astype(np.float32) * 0.55).astype(np.uint8),
            sampled_cow_xy,
            cow_track_valid,
            colors,
            t,
            history=args.history,
            radius=args.radius,
        )
        lifted_panel = draw_tracks(
            (source_rgb[t].astype(np.float32) * 0.55).astype(np.uint8),
            reproj_xy_cow,
            lifted_valid,
            colors,
            t,
            history=args.history,
            radius=args.radius,
        )
        motion_panel = draw_scalar_points(
            (source_rgb[t].astype(np.float32) * 0.45).astype(np.uint8),
            sampled_cow_xy[t],
            lifted_valid[t],
            motion[t],
            vmax=motion_vmax,
            radius=max(1, args.radius),
        )
        err_panel = draw_scalar_points(
            np.zeros_like(source_rgb[t]),
            sampled_cow_xy[t],
            lifted_valid[t],
            reproj_error[t],
            vmax=err_vmax,
            radius=max(1, args.radius),
        )
        valid_panel = np.zeros_like(source_rgb[t])
        valid_panel[lifted_valid[t].sum() * 0: lifted_valid[t].sum() * 0 + 1, :1] = 0
        valid_panel = draw_scalar_points(
            valid_panel,
            sampled_cow_xy[t],
            lifted_valid[t],
            np.ones_like(motion[t]),
            vmax=1.0,
            radius=max(1, args.radius),
        )
        caption = (
            f"{args.dataset} {seq}  CowTracker 2D + VGGT-Omega depth/camera -> sampled 3D tracks  "
            f"stride={args.track_stride} conf>={args.depth_conf_threshold:g} "
            f"xy_map={'center_crop' if raw_hw is not None else 'resize'}"
        )
        frames.append(
            compose_grid(
                [
                    source_rgb[t],
                    cow_panel,
                    lifted_panel,
                    depth_vis[t],
                    motion_panel,
                    err_panel,
                ],
                [
                    "source RGB",
                    "CowTracker 2D tracks",
                    "lifted 3D reprojected",
                    "VGGT-Omega depth",
                    "3D motion magnitude",
                    "reprojection error",
                ],
                caption,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_mp4),
        fps=int(args.fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    np.savez_compressed(
        out_npz,
        query_xy_cow=query_xy.astype(np.float32),
        cow_tracks_xy_sampled=sampled_cow_xy.astype(np.float32),
        vggt_tracks_xy_sampled=sampled_vggt_xy.astype(np.float32),
        lifted_tracks_world_sampled=world.astype(np.float32),
        reprojected_xy_cow_sampled=reproj_xy_cow.astype(np.float32),
        lifted_valid=lifted_valid.astype(bool),
        reprojection_error_vggt_px=reproj_error.astype(np.float32),
        motion_world=motion.astype(np.float32),
        cow_hw=np.asarray([cow_h, cow_w], dtype=np.int32),
        vggt_hw=np.asarray([vggt_h, vggt_w], dtype=np.int32),
        raw_hw=np.asarray(raw_hw if raw_hw is not None else [-1, -1], dtype=np.int32),
        xy_mapping=np.asarray("center_crop" if raw_hw is not None else "resize"),
        track_stride=np.int32(args.track_stride),
    )
    print(f"[cow-vggt-viz] wrote {out_mp4}", flush=True)
    return True


def discover_sequences(cow_root: Path, seqs: str) -> list[str]:
    if seqs != "all":
        return [s.strip() for s in seqs.split(",") if s.strip()]
    return sorted(p.name.removesuffix("_pred_trackvideo.npz") for p in cow_root.glob("*_pred_trackvideo.npz"))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--cow_root", required=True)
    p.add_argument("--vggt_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--seqs", default="all")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--track_stride", type=int, default=8)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--cow_conf_threshold", type=float, default=0.1)
    p.add_argument("--depth_conf_threshold", type=float, default=0.0)
    p.add_argument("--motion_vmax", type=float, default=0.25)
    p.add_argument("--error_vmax", type=float, default=2.0)
    p.add_argument("--skip_existing", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seqs = discover_sequences(Path(args.cow_root), args.seqs)
    shard = assign_shard(seqs, args.num_shards, args.shard_index)
    print(
        f"[cow-vggt-viz] dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"seqs={len(shard)} cow={args.cow_root} vggt={args.vggt_root} out={args.output_root}",
        flush=True,
    )
    ok = 0
    for seq in shard:
        ok += int(render_sequence(args, seq))
    print(f"[cow-vggt-viz] done ok={ok}/{len(shard)}", flush=True)


if __name__ == "__main__":
    main()
