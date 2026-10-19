"""
Shared rendering helpers for Track4D-360 2×3 visualization panels.

Used by:
- Training debug (train_wan_track4d360.py: log_debug_visualization)
- Eval comparison (eval_wan_track4d360_syn4d.py: comparison.mp4)
- Standalone viz (scripts/visualize_track4d360_pair.py)

All render_* functions return lists of (H, W, 3) uint8 numpy arrays.
"""

import cv2
import numpy as np
import torch

from holo4d.visualization.artifact_io import save_mp4


# ---------- Utility ----------

def _to_np(x, dtype=np.float64):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


def _pil_to_np(img):
    return np.array(img)


def infer_track_stride(N, H, W):
    """Recover spatial stride from track count and image dimensions."""
    for s in [1, 2, 4, 8, 16]:
        hg = len(range(0, H, s))
        wg = len(range(0, W, s))
        if hg * wg == N:
            return s
    return 4


def draw_label(frame, text, pos=(4, 16), scale=0.45):
    """Draw label with black outline + white fill."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def save_video(frames, path, fps=16):
    """Save list of (H, W, 3) uint8 numpy arrays to mp4."""
    save_mp4(path, np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0), fps=fps)


def forward_warp_rgbd_to_target(
    src_rgb,
    src_depth,
    src_cam2world,
    src_intrinsics,
    tgt_cam2world,
    tgt_intrinsics,
    src_cam0_pose=None,
    common_scale=1.0,
):
    """Forward-warp dense source RGB+depth into the target camera with z-buffering.

    Args:
        src_rgb: (H, W, 3) uint8 or float source RGB frame.
        src_depth: (H, W) float32 source depth in source-camera metric space.
        src_cam2world: (4, 4) source camera pose in world space.
        src_intrinsics: (3, 3) source intrinsics.
        tgt_cam2world: (4, 4) target camera pose in cam0-normalized space.
        tgt_intrinsics: (3, 3) target intrinsics.
        src_cam0_pose: (4, 4) source frame-0 pose in world space. If provided,
            points are transformed into frame-0 camera space before projection.
            If None, points stay in world space.
        common_scale: scalar used to normalize world positions into cam0 space.

    Returns:
        warped_rgb: (H, W, 3) uint8
        warped_depth: (H, W) float32 target-camera depth
        valid_mask: (H, W) bool
    """
    rgb = np.asarray(src_rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    depth = np.asarray(src_depth, dtype=np.float32)
    src_c2w = np.asarray(src_cam2world, dtype=np.float32)
    src_K = np.asarray(src_intrinsics, dtype=np.float32)
    tgt_c2w = np.asarray(tgt_cam2world, dtype=np.float32)
    tgt_K = np.asarray(tgt_intrinsics, dtype=np.float32)
    common_scale = float(max(common_scale, 1e-6))

    H, W = depth.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    flat_depth = depth.reshape(-1)
    flat_rgb = rgb.reshape(-1, 3)

    valid = np.isfinite(flat_depth) & (flat_depth > 1e-6)
    if not np.any(valid):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

    fx_s, fy_s = src_K[0, 0], src_K[1, 1]
    cx_s, cy_s = src_K[0, 2], src_K[1, 2]

    x_cam = (xx.reshape(-1) - cx_s) * flat_depth / max(fx_s, 1e-6)
    y_cam = (yy.reshape(-1) - cy_s) * flat_depth / max(fy_s, 1e-6)
    pts_src_cam = np.stack([x_cam, y_cam, flat_depth], axis=-1)

    pts_src_cam = pts_src_cam[valid]
    rgb_valid = flat_rgb[valid]

    R_src = src_c2w[:3, :3]
    t_src = src_c2w[:3, 3]
    pts_world = (R_src @ pts_src_cam.T).T + t_src[None, :]

    if src_cam0_pose is not None:
        src_cam0_pose = np.asarray(src_cam0_pose, dtype=np.float32)
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)
        R_w2c0 = world_to_cam0[:3, :3]
        t_w2c0 = world_to_cam0[:3, 3]
        pts_model = (R_w2c0 @ pts_world.T).T + t_w2c0[None, :]
        pts_model = pts_model / common_scale
    else:
        pts_model = pts_world / common_scale

    tgt_w2c = np.linalg.inv(tgt_c2w).astype(np.float32)
    R_tgt = tgt_w2c[:3, :3]
    t_tgt = tgt_w2c[:3, 3]
    pts_tgt = (R_tgt @ pts_model.T).T + t_tgt[None, :]
    z_tgt = pts_tgt[:, 2]

    valid_tgt = np.isfinite(z_tgt) & (z_tgt > 1e-6)
    if not np.any(valid_tgt):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

    pts_tgt = pts_tgt[valid_tgt]
    z_tgt = z_tgt[valid_tgt]
    rgb_valid = rgb_valid[valid_tgt]

    fx_t, fy_t = tgt_K[0, 0], tgt_K[1, 1]
    cx_t, cy_t = tgt_K[0, 2], tgt_K[1, 2]
    u = np.rint(fx_t * pts_tgt[:, 0] / z_tgt + cx_t).astype(np.int32)
    v = np.rint(fy_t * pts_tgt[:, 1] / z_tgt + cy_t).astype(np.int32)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

    u = u[in_bounds]
    v = v[in_bounds]
    z_tgt = z_tgt[in_bounds]
    rgb_valid = rgb_valid[in_bounds]

    pix = v * W + u
    order = np.lexsort((z_tgt, pix))
    pix_sorted = pix[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = pix_sorted[1:] != pix_sorted[:-1]
    keep = order[first]

    pix_keep = pix[keep]
    u_keep = u[keep]
    v_keep = v[keep]
    z_keep = z_tgt[keep]
    rgb_keep = rgb_valid[keep]

    warped_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    warped_depth = np.zeros((H, W), dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)

    warped_rgb[v_keep, u_keep] = rgb_keep
    warped_depth[v_keep, u_keep] = z_keep.astype(np.float32)
    valid_mask[v_keep, u_keep] = True

    return warped_rgb, warped_depth, valid_mask


# ---------- Track Residual ----------

def render_track_residual(tracks_world, H, W, stride=None):
    """Dense track residual (motion from frame 0). Grey=static, color=motion.

    Args:
        tracks_world: (N, T, 3) — dense grid tracks.
        H, W: full image size.
        stride: spatial stride (auto-inferred if None).

    Returns:
        list of (H, W, 3) uint8 numpy arrays.
    """
    pts = _to_np(tracks_world, np.float32)
    N, T, _ = pts.shape
    if stride is None:
        stride = infer_track_stride(N, H, W)
    Hs, Ws = H // stride, W // stride

    track_map = pts.reshape(Hs, Ws, T, 3).transpose(2, 0, 1, 3)
    residual = track_map - track_map[0:1]

    max_abs = np.maximum(np.percentile(np.abs(residual.reshape(-1, 3)), 98, axis=0), 1e-6)

    frames = []
    for t in range(T):
        rgb = np.clip(residual[t] / (2 * max_abs) + 0.5, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        if stride > 1:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)
        frames.append(rgb)
    return frames


# ---------- Source Visibility Map ----------

def render_source_vis_map(tracks_world, cam2world, intrinsics, H, W, stride=None,
                          src_frames_np=None):
    """Source-view grid: green=visible, red=behind camera, orange=OOB in target.

    Args:
        tracks_world: (N, T, 3).
        cam2world: (T, 4, 4) target camera extrinsics.
        intrinsics: (3, 3) target camera intrinsics.
        H, W: target image size.
        stride: spatial stride (auto-inferred if None).
        src_frames_np: optional list of (H, W, 3) uint8 np arrays for blending.

    Returns:
        list of (H, W, 3) uint8 numpy arrays.
    """
    pts = _to_np(tracks_world, np.float64)
    c2w = _to_np(cam2world, np.float64)
    K = _to_np(intrinsics, np.float64)
    N, T, _ = pts.shape
    if stride is None:
        stride = infer_track_stride(N, H, W)
    Hs, Ws = H // stride, W // stride

    frames = []
    for t in range(T):
        w2c = np.linalg.inv(c2w[t])
        pts_cam = (w2c[:3, :3] @ pts[:, t, :].T + w2c[:3, 3:4]).T

        in_front = pts_cam[:, 2] > 0.01
        u = np.full(N, -1.0)
        v = np.full(N, -1.0)
        u[in_front] = K[0, 0] * pts_cam[in_front, 0] / pts_cam[in_front, 2] + K[0, 2]
        v[in_front] = K[1, 1] * pts_cam[in_front, 1] / pts_cam[in_front, 2] + K[1, 2]
        in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        colors = np.zeros((N, 3), dtype=np.uint8)
        colors[in_bounds] = [0, 200, 0]
        colors[~in_bounds & in_front] = [200, 100, 0]
        colors[~in_front] = [200, 0, 0]

        grid = colors.reshape(Hs, Ws, 3)
        if stride > 1:
            grid = cv2.resize(grid, (W, H), interpolation=cv2.INTER_NEAREST)

        if src_frames_np is not None and t < len(src_frames_np):
            blended = (0.4 * src_frames_np[t].astype(np.float32)
                       + 0.6 * grid.astype(np.float32)).astype(np.uint8)
        else:
            blended = grid

        n_vis = int(in_bounds.sum())
        vis_pct = 100.0 * n_vis / N if N > 0 else 0.0
        draw_label(blended, f"vis: {vis_pct:.0f}%", pos=(W - 80, H - 8), scale=0.35)
        frames.append(blended)
    return frames


# ---------- Projected UV on Target ----------

def _project_to_target(pts, c2w, K, H, W, t):
    """Project 3D tracks at frame t to target camera. Returns u, v, in_bounds mask."""
    w2c = np.linalg.inv(c2w[t])
    pts_cam = (w2c[:3, :3] @ pts[:, t, :].T + w2c[:3, 3:4]).T
    N = pts.shape[0]
    in_front = pts_cam[:, 2] > 0.01
    u = np.full(N, np.nan)
    v = np.full(N, np.nan)
    u[in_front] = K[0, 0] * pts_cam[in_front, 0] / pts_cam[in_front, 2] + K[0, 2]
    v[in_front] = K[1, 1] * pts_cam[in_front, 1] / pts_cam[in_front, 2] + K[1, 2]
    in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, in_front, in_bounds


def _identity_colors(pts, T):
    """Frame-0 identity colors: normalize 3D position to RGB."""
    pts0 = pts[:, 0, :]
    lo = np.percentile(pts0, 2, axis=0)
    hi = np.percentile(pts0, 98, axis=0)
    rng = np.maximum(hi - lo, 1e-6)
    return (np.clip((pts0 - lo) / rng, 0, 1) * 255).astype(np.uint8)


def _paint_markers(canvas, u_int, v_int, colors, H, W, marker_size):
    """Vectorized marker painting: paint single pixels then dilate."""
    # Paint single-pixel dots (vectorized)
    dot_layer = np.zeros((H, W, 3), dtype=np.uint8)
    dot_layer[v_int, u_int] = colors
    if marker_size <= 1:
        mask = dot_layer.any(axis=2)
        canvas[mask] = (
            0.4 * canvas[mask].astype(np.float32)
            + 0.6 * dot_layer[mask].astype(np.float32)
        ).astype(np.uint8)
    else:
        # Dilate each channel to expand dots, then blend
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * marker_size + 1, 2 * marker_size + 1))
        dilated = cv2.dilate(dot_layer, kernel, iterations=1)
        mask = dilated.any(axis=2)
        canvas[mask] = (
            0.3 * canvas[mask].astype(np.float32)
            + 0.7 * dilated[mask].astype(np.float32)
        ).astype(np.uint8)


def render_proj_uv_overlay(tracks_world, cam2world, intrinsics, H, W,
                           target_frames_np, marker_size=1):
    """Project tracks to target camera, overlay identity-colored dots on target.

    Args:
        tracks_world: (N, T, 3).
        cam2world: (T, 4, 4).
        intrinsics: (3, 3).
        H, W: target image size.
        target_frames_np: list of (H, W, 3) uint8 np arrays.
        marker_size: radius of markers (1=single pixel, 2+=dilated square).

    Returns:
        list of (H, W, 3) uint8 numpy arrays.
    """
    pts = _to_np(tracks_world, np.float64)
    c2w = _to_np(cam2world, np.float64)
    K = _to_np(intrinsics, np.float64)
    N, T, _ = pts.shape
    id_colors = _identity_colors(pts, T)

    frames = []
    for t in range(T):
        u, v, in_front, in_bounds = _project_to_target(pts, c2w, K, H, W, t)
        tgt = target_frames_np[t].copy() if t < len(target_frames_np) else np.zeros((H, W, 3), dtype=np.uint8)
        overlay = tgt.copy()

        _paint_markers(overlay, u[in_bounds].astype(np.int32),
                       v[in_bounds].astype(np.int32),
                       id_colors[in_bounds], H, W, marker_size)

        # Red for OOB but in front
        oob = in_front & ~in_bounds
        if oob.any():
            u_oob = np.clip(u[oob], 0, W - 1).astype(np.int32)
            v_oob = np.clip(v[oob], 0, H - 1).astype(np.int32)
            overlay[v_oob, u_oob] = (
                0.3 * tgt[v_oob, u_oob].astype(np.float32)
                + 0.7 * np.array([255, 50, 50], dtype=np.float32)
            ).astype(np.uint8)

        frames.append(overlay)
    return frames


# ---------- Block Density Heatmap ----------

def render_block_density(tracks_world, cam2world, intrinsics, H, W, T,
                         target_frames_np=None):
    """Block density heatmap: how many track observations per latent grid cell.

    Args:
        tracks_world: (N, T, 3).
        cam2world: (T, 4, 4).
        intrinsics: (3, 3).
        H, W, T: image/video dimensions.
        target_frames_np: optional list of (H, W, 3) uint8 np arrays for blending.

    Returns:
        list of (H, W, 3) uint8 numpy arrays.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm_mpl

    pts = _to_np(tracks_world, np.float64)
    c2w = _to_np(cam2world, np.float64)
    K = _to_np(intrinsics, np.float64)
    N, T_tracks, _ = pts.shape

    # Latent grid dimensions (Wan: spatial_div=16, temporal_div=4)
    latent_t = (T - 1) // 4 + 1
    latent_h = H // 16
    latent_w = W // 16

    # Frame-to-temporal-block mapping
    frame_to_lt = np.array([
        int(np.clip(np.floor(t / max(T - 1, 1) * max(latent_t - 1, 1)), 0, latent_t - 1))
        for t in range(T_tracks)
    ])
    lt_to_frames = {}
    for t in range(T_tracks):
        lt_to_frames.setdefault(frame_to_lt[t], []).append(t)

    # Pre-compute density per temporal block
    lt_density = {}
    for lt, frame_list in lt_to_frames.items():
        counts = np.zeros((latent_h, latent_w), dtype=np.int32)
        for t in frame_list:
            w2c = np.linalg.inv(c2w[t])
            pts_cam = (w2c[:3, :3] @ pts[:, t, :].T + w2c[:3, 3:4]).T
            valid = pts_cam[:, 2] > 0.01
            u_f = K[0, 0] * pts_cam[valid, 0] / pts_cam[valid, 2] + K[0, 2]
            v_f = K[1, 1] * pts_cam[valid, 1] / pts_cam[valid, 2] + K[1, 2]
            u = u_f.astype(np.int32)
            v = v_f.astype(np.int32)
            ib = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            lh = np.clip((v[ib].astype(np.float64) / H * latent_h).astype(int), 0, latent_h - 1)
            lw = np.clip((u[ib].astype(np.float64) / W * latent_w).astype(int), 0, latent_w - 1)
            np.add.at(counts, (lh, lw), 1)
        lt_density[lt] = counts

    global_max = max((c.max() for c in lt_density.values()), default=1)
    global_max = max(global_max, 1)

    frames = []
    for t in range(T_tracks):
        lt = frame_to_lt[t]
        counts = lt_density[lt]
        n_in_block = len(lt_to_frames[lt])

        heat = counts.astype(np.float32) / global_max
        heat_rgb = (cm_mpl.inferno(heat)[:, :, :3] * 255).astype(np.uint8)
        heat_full = cv2.resize(heat_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

        if target_frames_np is not None and t < len(target_frames_np):
            blended = cv2.addWeighted(target_frames_np[t], 0.3, heat_full, 0.7, 0)
        else:
            blended = heat_full

        # Grid lines
        for lh_i in range(latent_h + 1):
            y = min(int(lh_i / latent_h * H), H - 1)
            cv2.line(blended, (0, y), (W - 1, y), (150, 150, 150), 1)
        for lw_i in range(latent_w + 1):
            x = min(int(lw_i / latent_w * W), W - 1)
            cv2.line(blended, (x, 0), (x, H - 1), (150, 150, 150), 1)

        draw_label(blended, f"lt={lt} ({n_in_block}f)", pos=(W - 120, H - 8), scale=0.35)
        frames.append(blended)
    return frames


# ---------- Merged: Proj UV + Block Density ----------

def render_proj_uv_density_merged(tracks_world, cam2world, intrinsics, H, W, T,
                                  target_frames_np, marker_size=2):
    """Merged panel: block density heatmap + projected UV identity-colored markers.

    Background: target frame (20%) + density heatmap (50%).
    Foreground: identity-colored marker squares at projected UV positions.

    Returns:
        list of (H, W, 3) uint8 numpy arrays.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm_mpl

    pts = _to_np(tracks_world, np.float64)
    c2w = _to_np(cam2world, np.float64)
    K = _to_np(intrinsics, np.float64)
    N, T_tracks, _ = pts.shape
    id_colors = _identity_colors(pts, T_tracks)

    latent_t = (T - 1) // 4 + 1
    latent_h = H // 16
    latent_w = W // 16

    frame_to_lt = np.array([
        int(np.clip(np.floor(t / max(T - 1, 1) * max(latent_t - 1, 1)), 0, latent_t - 1))
        for t in range(T_tracks)
    ])
    lt_to_frames = {}
    for t in range(T_tracks):
        lt_to_frames.setdefault(frame_to_lt[t], []).append(t)

    lt_density = {}
    for lt, frame_list in lt_to_frames.items():
        counts = np.zeros((latent_h, latent_w), dtype=np.int32)
        for t in frame_list:
            w2c = np.linalg.inv(c2w[t])
            pts_cam = (w2c[:3, :3] @ pts[:, t, :].T + w2c[:3, 3:4]).T
            valid = pts_cam[:, 2] > 0.01
            u_f = K[0, 0] * pts_cam[valid, 0] / pts_cam[valid, 2] + K[0, 2]
            v_f = K[1, 1] * pts_cam[valid, 1] / pts_cam[valid, 2] + K[1, 2]
            u_i = u_f.astype(np.int32)
            v_i = v_f.astype(np.int32)
            ib = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
            lh = np.clip((v_i[ib].astype(np.float64) / H * latent_h).astype(int), 0, latent_h - 1)
            lw = np.clip((u_i[ib].astype(np.float64) / W * latent_w).astype(int), 0, latent_w - 1)
            np.add.at(counts, (lh, lw), 1)
        lt_density[lt] = counts

    global_max = max((c.max() for c in lt_density.values()), default=1)
    global_max = max(global_max, 1)

    frames = []
    for t in range(T_tracks):
        lt = frame_to_lt[t]
        counts = lt_density[lt]

        # Background: density heatmap blended with target
        heat = counts.astype(np.float32) / global_max
        heat_rgb = (cm_mpl.inferno(heat)[:, :, :3] * 255).astype(np.uint8)
        heat_full = cv2.resize(heat_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

        tgt = target_frames_np[t] if t < len(target_frames_np) else np.zeros((H, W, 3), dtype=np.uint8)
        canvas = (0.2 * tgt.astype(np.float32) + 0.5 * heat_full.astype(np.float32)).astype(np.uint8)

        # Latent grid lines (subtle)
        for lh_i in range(latent_h + 1):
            y = min(int(lh_i / latent_h * H), H - 1)
            cv2.line(canvas, (0, y), (W - 1, y), (100, 100, 100), 1)
        for lw_i in range(latent_w + 1):
            x = min(int(lw_i / latent_w * W), W - 1)
            cv2.line(canvas, (x, 0), (x, H - 1), (100, 100, 100), 1)

        # Foreground: projected UV identity dots (vectorized + dilated)
        u, v, in_front, in_bounds = _project_to_target(pts, c2w, K, H, W, t)
        dot_layer = np.zeros((H, W, 3), dtype=np.uint8)
        dot_layer[v[in_bounds].astype(np.int32), u[in_bounds].astype(np.int32)] = id_colors[in_bounds]
        if marker_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (2 * marker_size + 1, 2 * marker_size + 1))
            dot_layer = cv2.dilate(dot_layer, kernel, iterations=1)
        mask = dot_layer.any(axis=2)
        canvas[mask] = dot_layer[mask]

        frames.append(canvas)
    return frames


# ---------- 2×3 Grid Compositor ----------

def compose_2x3_grid(top_panels, bot_panels, top_labels, bot_labels, caption=""):
    """Compose 2×3 grid frames with labels and optional caption bar.

    Args:
        top_panels: list of 3 lists, each with T frames of (H, W, 3) uint8.
        bot_panels: list of 3 lists, each with T frames of (H, W, 3) uint8.
        top_labels: list of 3 strings.
        bot_labels: list of 3 strings.
        caption: optional caption text for bottom bar.

    Returns:
        list of (H_total, W_total, 3) uint8 numpy arrays.
    """
    T = len(top_panels[0])
    H, W = top_panels[0][0].shape[:2]

    # Caption bar (ensure total height divisible by 16 for video codec)
    raw_h = 2 * H
    caption_h = 16 * ((raw_h + 48) // 16 + 1) - raw_h
    if caption_h < 32:
        caption_h += 16

    caption_bar = np.zeros((caption_h, W * 3, 3), dtype=np.uint8)
    if caption:
        max_chars = (W * 3) // 7
        lines = []
        while caption and len(lines) < 2:
            lines.append(caption[:max_chars])
            caption = caption[max_chars:]
        y = 14
        for line in lines:
            cv2.putText(caption_bar, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (220, 220, 220), 1, cv2.LINE_AA)
            y += 16

    combined = []
    for t in range(T):
        top_row = np.concatenate([p[t].copy() for p in top_panels], axis=1)
        bot_row = np.concatenate([p[t].copy() for p in bot_panels], axis=1)

        # Draw labels
        for i, label in enumerate(top_labels):
            draw_label(top_row, label, pos=(i * W + 4, 16))
        for i, label in enumerate(bot_labels):
            draw_label(bot_row, label, pos=(i * W + 4, 16))

        frame = np.concatenate([top_row, bot_row, caption_bar], axis=0)
        combined.append(frame)
    return combined


def compose_2x4_grid(top_panels, bot_panels, top_labels, bot_labels, caption=""):
    """Compose 2x4 grid frames with labels and optional caption bar."""
    T = len(top_panels[0])
    H, W = top_panels[0][0].shape[:2]

    raw_h = 2 * H
    caption_h = 16 * ((raw_h + 48) // 16 + 1) - raw_h
    if caption_h < 32:
        caption_h += 16

    caption_bar = np.zeros((caption_h, W * 4, 3), dtype=np.uint8)
    if caption:
        max_chars = (W * 4) // 7
        lines = []
        while caption and len(lines) < 2:
            lines.append(caption[:max_chars])
            caption = caption[max_chars:]
        y = 14
        for line in lines:
            cv2.putText(caption_bar, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (220, 220, 220), 1, cv2.LINE_AA)
            y += 16

    combined = []
    for t in range(T):
        top_row = np.concatenate([p[t].copy() for p in top_panels], axis=1)
        bot_row = np.concatenate([p[t].copy() for p in bot_panels], axis=1)

        for i, label in enumerate(top_labels):
            draw_label(top_row, label, pos=(i * W + 4, 16))
        for i, label in enumerate(bot_labels):
            draw_label(bot_row, label, pos=(i * W + 4, 16))

        frame = np.concatenate([top_row, bot_row, caption_bar], axis=0)
        combined.append(frame)
    return combined
