"""OmniDROID-specific novel-view comparison rendering.

The real ext2 video is only a reference for interpolated/borrowed cameras. This
renderer therefore avoids the generic ``target GT`` / ``pred-GT`` labels and
uses one source-track subset for every source/real-target/virtual projection.
"""

from pathlib import Path

import cv2
import imageio
import numpy as np

from holo4d.visualization.hybrid_panels import (
    coerce_rgb_frame_uint8,
    colorize_depth,
    pil_list_to_np,
    project_track_frame,
    sequence_depth_range,
)
from holo4d.visualization.track_panels import (
    draw_label,
    forward_warp_rgbd_to_target,
)


def _as_pose_sequence(value, length):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.broadcast_to(arr, (length, 4, 4)).copy()
    return arr[:length]


def _as_intrinsics_sequence(value, length):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.broadcast_to(arr, (length, 3, 3)).copy()
    return arr[:length]


def _project_sequence(tracks, poses, intrinsics, height, width, valid):
    length, num_tracks = tracks.shape[1], tracks.shape[0]
    uv = np.zeros((length, num_tracks, 2), dtype=np.float32)
    visible = np.zeros((length, num_tracks), dtype=bool)
    for frame_idx in range(length):
        uv_t, vis_t = project_track_frame(
            tracks[:, frame_idx], poses[frame_idx], intrinsics[frame_idx], height, width
        )
        uv[frame_idx] = uv_t
        visible[frame_idx] = vis_t & valid[:, frame_idx]
    return uv, visible


def _gate_large_steps(uv, visible, max_step_px=25.0):
    gated = np.asarray(visible, dtype=bool).copy()
    for track_idx in range(gated.shape[1]):
        last = None
        for frame_idx in range(gated.shape[0]):
            if not gated[frame_idx, track_idx]:
                continue
            if last is not None and np.linalg.norm(uv[frame_idx, track_idx] - uv[last, track_idx]) > max_step_px:
                gated[frame_idx, track_idx] = False
            else:
                last = frame_idx
    return gated


def _bremm_rgb(query_uv, width, height):
    xy = np.asarray(query_uv, dtype=np.float32)
    nx = np.clip(xy[:, 0] / max(width - 1, 1), 0.0, 1.0)
    ny = np.clip(xy[:, 1] / max(height - 1, 1), 0.0, 1.0)
    hue = (nx * 0.8 + ny * 0.2) % 1.0
    sat = 0.6 + 0.4 * (1.0 - ny)
    val = 0.7 + 0.3 * nx
    hsv = np.stack(
        [(hue * 179).astype(np.uint8), (sat * 255).astype(np.uint8), (val * 255).astype(np.uint8)],
        axis=1,
    ).reshape(-1, 1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)


def _uniform_track_selection(visible, max_tracks):
    candidates = np.flatnonzero(np.asarray(visible).any(axis=0))
    if candidates.size <= max_tracks:
        return candidates
    return candidates[np.linspace(0, candidates.size - 1, max_tracks).astype(np.int64)]


def _overlay_track_trails(frames, uv, visible, colors, max_tracks=800, darken=0.5):
    selected = _uniform_track_selection(visible, max_tracks)
    outputs = []
    for frame_idx, frame in enumerate(frames):
        canvas = (np.asarray(frame, dtype=np.float32) * float(darken)).astype(np.uint8)
        for track_idx in selected:
            color = tuple(int(c) for c in colors[track_idx])
            history = np.flatnonzero(visible[: frame_idx + 1, track_idx])
            if history.size >= 2:
                points = np.rint(uv[history, track_idx]).astype(np.int32)
                cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
            if visible[frame_idx, track_idx]:
                point = tuple(np.rint(uv[frame_idx, track_idx]).astype(np.int32))
                cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)
        outputs.append(canvas)
    return outputs


def _depth_video(depth, length, height, width):
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.broadcast_to(arr, (length, height, width)).copy()
    arr = arr[:length]
    if arr.shape[1:] != (height, width):
        arr = np.stack(
            [cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST) for frame in arr]
        )
    return arr


def _colorize_depth_video(depth, valid):
    depth_range = sequence_depth_range(depth, valid)
    return [
        cv2.cvtColor(colorize_depth(frame, mask, depth_range), cv2.COLOR_BGR2RGB)
        for frame, mask in zip(depth, valid)
    ]


def _warp_sequence(src_frames, src_depth, src_poses, src_intrinsics, dst_poses, dst_intrinsics):
    warped_rgb, warped_depth, warped_valid = [], [], []
    for frame_idx in range(len(src_frames)):
        rgb, depth, valid = forward_warp_rgbd_to_target(
            src_rgb=src_frames[frame_idx],
            src_depth=src_depth[frame_idx],
            src_cam2world=src_poses[frame_idx],
            src_intrinsics=src_intrinsics[frame_idx],
            tgt_cam2world=dst_poses[frame_idx],
            tgt_intrinsics=dst_intrinsics[frame_idx],
            src_cam0_pose=None,
            common_scale=1.0,
        )
        warped_rgb.append(np.asarray(rgb, dtype=np.uint8))
        warped_depth.append(np.asarray(depth, dtype=np.float32))
        warped_valid.append(np.asarray(valid, dtype=bool))
    return warped_rgb, np.stack(warped_depth), np.stack(warped_valid)


def _rotation_angle_deg(relative_pose):
    rotation = np.asarray(relative_pose, dtype=np.float64)[:3, :3]
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _camera_info_panel(height, width, sample, src_pose, virtual_pose, alpha):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    relative = np.linalg.inv(src_pose) @ virtual_pose
    same_view = sample.get("omni_droid_novel_view_mode") == "same_view"
    lines = [
        f"mode: {sample.get('omni_droid_novel_view_mode', 'unknown')}",
        f"alpha: {float(alpha):.2f}" if alpha is not None else "alpha: n/a",
        f"baseline: {np.linalg.norm(relative[:3, 3]):.4f} normalized",
        f"rotation: {_rotation_angle_deg(relative):.2f} deg",
        "paired GT: ext1 source RGB" if same_view else "ext2 is GT only when alpha=1.00",
        "target latent init: random noise",
    ]
    y = 34
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y += 30
    return panel


def _compose_3x4(rows, labels, caption):
    height, width = rows[0][0].shape[:2]
    rendered_rows = []
    for row, row_labels in zip(rows, labels):
        rendered = np.concatenate([np.asarray(panel).copy() for panel in row], axis=1)
        for col_idx, label in enumerate(row_labels):
            draw_label(rendered, label, pos=(col_idx * width + 4, 16))
        rendered_rows.append(rendered)
    caption_bar = np.zeros((48, width * 4, 3), dtype=np.uint8)
    if caption:
        cv2.putText(caption_bar, caption[: max(1, width * 4 // 7)], (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate(rendered_rows + [caption_bar], axis=0)


def write_omni_droid_comparison_mp4(
    out_path,
    sample,
    pred_frames,
    *,
    fps=12,
    caption_extra="",
    **_ignored,
):
    """Write an aligned 3x4 diagnostic video for an OmniDROID novel view."""
    src_pil = sample["video"]
    ref_pil = sample["target_video"]
    height, width = np.asarray(src_pil[0]).shape[:2]
    src_frames = pil_list_to_np(src_pil, width, height)
    ref_frames = pil_list_to_np(ref_pil, width, height)
    pred = [coerce_rgb_frame_uint8(frame) for frame in pred_frames]
    length = min(len(src_frames), len(ref_frames), len(pred))
    src_frames, ref_frames, pred = src_frames[:length], ref_frames[:length], pred[:length]

    src_poses = _as_pose_sequence(sample["source_cam2world"], length)
    src_intrinsics = _as_intrinsics_sequence(sample["source_intrinsics"], length)
    virtual_poses = _as_pose_sequence(sample["target_cam2world"], length)
    virtual_intrinsics_value = sample.get("target_intrinsics_per_frame", sample["target_intrinsics"])
    virtual_intrinsics = _as_intrinsics_sequence(virtual_intrinsics_value, length)
    real_poses = _as_pose_sequence(sample["real_target_cam2world"], length)
    real_intrinsics = _as_intrinsics_sequence(sample["real_target_intrinsics"], length)
    src_depth = _depth_video(sample["source_depth_video"], length, height, width)
    real_depth = _depth_video(sample["target_depth"], length, height, width)

    virtual_rgb, virtual_depth, virtual_valid = _warp_sequence(
        src_frames, src_depth, src_poses, src_intrinsics, virtual_poses, virtual_intrinsics
    )
    real_rgb, _real_warp_depth, _real_warp_valid = _warp_sequence(
        src_frames, src_depth, src_poses, src_intrinsics, real_poses, real_intrinsics
    )

    tracks = np.asarray(sample["tracks_world"], dtype=np.float32)[:, :length]
    valid = np.asarray(sample["tracks_world_valid_mask"], dtype=bool)[:, :length]
    origin = np.asarray(sample.get("track_origin_camera", np.zeros(len(tracks))), dtype=np.int8)
    source_track_mask = origin == 0
    if not source_track_mask.any():
        source_track_mask = np.ones(len(tracks), dtype=bool)
    tracks, valid = tracks[source_track_mask], valid[source_track_mask]

    src_uv, src_vis = _project_sequence(
        tracks, src_poses, src_intrinsics, height, width, valid
    )
    real_uv, real_vis = _project_sequence(
        tracks, real_poses, real_intrinsics, height, width, valid
    )
    virtual_uv, virtual_vis = _project_sequence(
        tracks, virtual_poses, virtual_intrinsics, height, width, valid
    )
    real_vis = _gate_large_steps(real_uv, real_vis)
    virtual_vis = _gate_large_steps(virtual_uv, virtual_vis)
    colors = _bremm_rgb(src_uv[0], width, height)

    src_tracks = _overlay_track_trails(src_frames, src_uv, src_vis, colors)
    real_tracks = _overlay_track_trails(real_rgb, real_uv, real_vis, colors)
    virtual_tracks = _overlay_track_trails(virtual_rgb, virtual_uv, virtual_vis, colors)
    pred_tracks = _overlay_track_trails(pred, virtual_uv, virtual_vis, colors)
    src_depth_vis = _colorize_depth_video(src_depth, src_depth > 0)
    real_depth_vis = _colorize_depth_video(real_depth, real_depth > 0)
    virtual_depth_vis = _colorize_depth_video(virtual_depth, virtual_valid)
    alpha_path = sample.get("omni_droid_interp_alphas")
    if alpha_path is None:
        alpha_path = np.full(length, sample.get("omni_droid_interp_alpha", np.nan), np.float32)
    else:
        alpha_path = np.asarray(alpha_path, dtype=np.float32)[:length]
    same_view = sample.get("omni_droid_novel_view_mode") == "same_view"
    if same_view:
        ref_label = "identity target GT (same ext1 RGB)"
        target_depth_label = "identity target depth (same ext1)"
        target_track_label = "identity-camera reproj tracks"
        pred_label = "prediction @ identity camera"
        reproj_label = "ext1->identity RGB reproj"
        virtual_depth_label = "ext1->identity reproj depth"
        virtual_track_label = "ext1->identity reproj tracks"
    else:
        ref_label = (
            "ext2 paired GT"
            if sample.get("target_video_is_ground_truth", False)
            else "ext2 real reference (NOT virtual GT)"
        )
        target_depth_label = "ext2 depth (real reference)"
        target_track_label = "ext1->ext2 reproj tracks"
        pred_label = "prediction @ virtual"
        reproj_label = "ext1->virtual RGB reproj"
        virtual_depth_label = "ext1->virtual reproj depth"
        virtual_track_label = "ext1->virtual reproj tracks"
    caption = caption_extra.strip()
    prompt = str(sample.get("prompt", "")).strip()
    if prompt:
        caption = f"{caption} | {prompt}" if caption else prompt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1, pixelformat="yuv420p",
    )
    try:
        for frame_idx in range(length):
            frame_alpha = float(alpha_path[frame_idx])
            alpha_value = frame_alpha if np.isfinite(frame_alpha) else None
            alpha_text = f" a={frame_alpha:.2f}" if alpha_value is not None else ""
            info = _camera_info_panel(
                height, width, sample, src_poses[frame_idx], virtual_poses[frame_idx], alpha_value
            )
            rows = [
                [src_frames[frame_idx], ref_frames[frame_idx], pred[frame_idx], virtual_rgb[frame_idx]],
                [src_depth_vis[frame_idx], real_depth_vis[frame_idx], virtual_depth_vis[frame_idx], info],
                [src_tracks[frame_idx], real_tracks[frame_idx], virtual_tracks[frame_idx], pred_tracks[frame_idx]],
            ]
            labels = [
                ["ext1 source RGB", ref_label, f"{pred_label}{alpha_text}", reproj_label],
                ["ext1 depth", target_depth_label, virtual_depth_label, "camera request"],
                ["ext1 source flow tracks", target_track_label, virtual_track_label, "requested-camera tracks on pred"],
            ]
            writer.append_data(_compose_3x4(rows, labels, caption))
    finally:
        writer.close()
    return out_path
