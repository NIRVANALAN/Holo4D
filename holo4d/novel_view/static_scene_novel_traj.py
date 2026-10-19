from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image
import torch

from holo4d.data.track_reprojection import project_tracks_to_target_views
from holo4d.novel_view.object_centric_orbit import (
    build_fixed_height_look_at_orbit_sequence,
    build_moving_track_continuation_orbit_sequence,
    build_object_centric_orbit_sequence,
    estimate_camera_world_up,
    estimate_foreground_pivots_world,
    estimate_moving_track_pivots_world,
    estimate_scene_track_pivots_world,
)
from holo4d.visualization.track_panels import forward_warp_rgbd_to_target


DEFAULT_STATIC_SCENE_NOVEL_TRAJECTORIES = (
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
    "translate_left",
    "translate_right",
    "zoom_in",
    "zoom_out",
    "orbit_left",
    "orbit_right",
    # Robot-scene trajectories. These deliberately avoid object-centric orbit
    # and backward dolly because many DROID cameras face a nearby wall.
    "static_left",
    "static_right",
    "static_up",
    "static_left_front",
    "static_right_front",
    "slow_pan_left",
    "slow_pan_right",
    "slow_truck_left",
    "slow_truck_right",
    "slow_dolly_in",
    "slow_crane_up",
    "slow_crane_down",
    "slow_left_front",
    "slow_right_front",
    "slow_crane_dolly_in",
)


def prediction_filename_for_sample(sample: dict, default_name: str = "pred_rgb.mp4") -> str:
    trajectory_name = str(sample.get("trajectory_name", "")).strip()
    if not trajectory_name:
        return default_name
    safe_name = trajectory_name.replace("/", "_")
    return f"{safe_name}.mp4"


def comparison_filename_for_sample(sample: dict, default_name: str = "comparison.mp4") -> str:
    trajectory_name = str(sample.get("trajectory_name", "")).strip()
    if not trajectory_name:
        return default_name
    safe_name = trajectory_name.replace("/", "_")
    return f"{safe_name}_comparison.mp4"


def parse_static_scene_trajectories(trajectories: str | None) -> tuple[str, ...]:
    if trajectories is None:
        return DEFAULT_STATIC_SCENE_NOVEL_TRAJECTORIES
    names = tuple(t.strip() for t in trajectories.split(",") if t.strip())
    if not names:
        return DEFAULT_STATIC_SCENE_NOVEL_TRAJECTORIES
    unknown = [name for name in names if name not in DEFAULT_STATIC_SCENE_NOVEL_TRAJECTORIES]
    if unknown:
        raise ValueError(
            f"Unknown trajectory names {unknown}. Expected subset of "
            f"{DEFAULT_STATIC_SCENE_NOVEL_TRAJECTORIES}."
        )
    return names


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float32,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def _make_local_camera_deltas(
    trajectory_name: str,
    num_frames: int,
    angle_deg: float,
    distance: float,
    orbit_radius: float,
    orbit_angle_deg: float | None = None,
    orbit_start_deg: float = 0.0,
    trajectory_start_progress: float = 0.0,
) -> np.ndarray:
    if trajectory_name not in DEFAULT_STATIC_SCENE_NOVEL_TRAJECTORIES:
        raise ValueError(f"Unsupported trajectory_name={trajectory_name}")
    progress_start = float(trajectory_start_progress)
    progress = np.linspace(
        progress_start,
        progress_start + 1.0,
        int(num_frames),
        dtype=np.float32,
    )
    angle_rad = np.deg2rad(float(angle_deg))
    # orbit_left/right reuse `angle_deg` historically, but at 288x512 with the
    # close-pivot orbit_radius=0.3 calibration the 28° pan sweep produces an
    # excessively aggressive orbit (sweeps a quarter-quadrant around a pivot
    # inside the frustum). orbit_angle_deg lets the caller dial orbit
    # independently while leaving pan_*/translate_*/zoom_* on `angle_deg`.
    orbit_angle_rad = (
        np.deg2rad(float(orbit_angle_deg)) if orbit_angle_deg is not None else angle_rad
    )
    # orbit_start_deg offsets the orbit sweep so chained rounds can continue
    # from a previous round's end angle: frame 0 sits at orbit_start_deg (not
    # identity) and the sweep covers [start, start + orbit_angle]. Only the
    # orbit_* trajectories honor it; pan/translate/zoom always start at
    # identity (orbit_start_deg=0.0 keeps the historical behavior everywhere).
    orbit_start_rad = np.deg2rad(float(orbit_start_deg))
    deltas = np.repeat(np.eye(4, dtype=np.float32)[None], int(num_frames), axis=0)

    if trajectory_name == "pan_left":
        angles = progress * angle_rad
        for i, angle in enumerate(angles):
            deltas[i, :3, :3] = _rotation_y(float(angle))
    elif trajectory_name == "pan_right":
        angles = -progress * angle_rad
        for i, angle in enumerate(angles):
            deltas[i, :3, :3] = _rotation_y(float(angle))
    elif trajectory_name == "pan_up":
        angles = -progress * angle_rad
        for i, angle in enumerate(angles):
            deltas[i, :3, :3] = _rotation_x(float(angle))
    elif trajectory_name == "pan_down":
        angles = progress * angle_rad
        for i, angle in enumerate(angles):
            deltas[i, :3, :3] = _rotation_x(float(angle))
    elif trajectory_name == "translate_left":
        deltas[:, 0, 3] = -progress * float(distance)
    elif trajectory_name == "translate_right":
        deltas[:, 0, 3] = progress * float(distance)
    elif trajectory_name == "zoom_in":
        deltas[:, 2, 3] = progress * float(distance)
    elif trajectory_name == "zoom_out":
        deltas[:, 2, 3] = -progress * float(distance)
    elif trajectory_name == "orbit_left":
        angles = orbit_start_rad + progress * orbit_angle_rad
        for i, angle in enumerate(angles):
            # The pivot is at ``(0, 0, orbit_radius)`` in the source-camera
            # frame.  With cam2world +Z as the optical axis, the camera at
            # ``(r sin(a), 0, r(1-cos(a)))`` must use ``R_y(-a)`` so +Z points
            # back toward the pivot.  The old +a sign produced an in-place
            # yaw-away effect: the translation followed an arc, but the
            # rendered view did not look at the orbit center.
            deltas[i, :3, :3] = _rotation_y(float(-angle))
            deltas[i, 0, 3] = float(orbit_radius) * float(np.sin(angle))
            deltas[i, 2, 3] = float(orbit_radius) * float(1.0 - np.cos(angle))
    elif trajectory_name == "orbit_right":
        angles = -(orbit_start_rad + progress * orbit_angle_rad)
        for i, angle in enumerate(angles):
            # ``angle`` is signed for the mirrored right-hand sweep; use the
            # same look-at rule as orbit_left rather than duplicating signs.
            deltas[i, :3, :3] = _rotation_y(float(-angle))
            deltas[i, 0, 3] = float(orbit_radius) * float(np.sin(angle))
            deltas[i, 2, 3] = float(orbit_radius) * float(1.0 - np.cos(angle))
    elif trajectory_name.startswith("static_"):
        # Five displaced but temporally static views. The requested offset is
        # applied at every frame, so these are NVS views rather than camera
        # motion sequences. Positive local Z moves toward the workspace.
        x_sign = -1.0 if "left" in trajectory_name else 1.0 if "right" in trajectory_name else 0.0
        y_sign = -1.0 if trajectory_name == "static_up" else 0.0
        z_sign = 0.55 if "front" in trajectory_name else 0.0
        deltas[:, 0, 3] = x_sign * float(distance)
        deltas[:, 1, 3] = y_sign * float(distance) * 0.65
        deltas[:, 2, 3] = z_sign * float(distance)
        yaw = -x_sign * angle_rad * 0.45
        pitch = -y_sign * angle_rad * 0.35
        deltas[:, :3, :3] = (_rotation_y(float(yaw)) @ _rotation_x(float(pitch)))[None]
    elif trajectory_name == "slow_pan_left":
        for i, angle in enumerate(progress * angle_rad):
            deltas[i, :3, :3] = _rotation_y(float(angle))
    elif trajectory_name == "slow_pan_right":
        for i, angle in enumerate(-progress * angle_rad):
            deltas[i, :3, :3] = _rotation_y(float(angle))
    elif trajectory_name == "slow_truck_left":
        deltas[:, 0, 3] = -progress * float(distance)
    elif trajectory_name == "slow_truck_right":
        deltas[:, 0, 3] = progress * float(distance)
    elif trajectory_name == "slow_dolly_in":
        deltas[:, 2, 3] = progress * float(distance)
    elif trajectory_name == "slow_crane_up":
        deltas[:, 1, 3] = -progress * float(distance) * 0.65
    elif trajectory_name == "slow_crane_down":
        deltas[:, 1, 3] = progress * float(distance) * 0.65
    elif trajectory_name == "slow_left_front":
        deltas[:, 0, 3] = -progress * float(distance) * 0.75
        deltas[:, 2, 3] = progress * float(distance) * 0.55
    elif trajectory_name == "slow_right_front":
        deltas[:, 0, 3] = progress * float(distance) * 0.75
        deltas[:, 2, 3] = progress * float(distance) * 0.55
    elif trajectory_name == "slow_crane_dolly_in":
        deltas[:, 1, 3] = -progress * float(distance) * 0.45
        deltas[:, 2, 3] = progress * float(distance) * 0.70
    return deltas


def build_target_camera_sequence(
    source_cam2world: np.ndarray,
    trajectory_name: str,
    angle_deg: float = 28.0,
    distance: float = 0.30,
    orbit_radius: float = 0.3,
    orbit_angle_deg: float | None = None,
    orbit_start_deg: float = 0.0,
    trajectory_start_progress: float = 0.0,
    trajectory_anchor_cam2world: np.ndarray | None = None,
) -> np.ndarray:
    """Build a novel-view camera trajectory from the source anchor pose.

    Default magnitudes calibrated for ~50% overlap at frame 49 (T=49) at
    144x256 with MVS-Synth-typical FOV (~81 deg h, ~56 deg v):
      pan_up/down:   50%    orbit_left/right: 50%
      zoom_in:       49%    zoom_out:         59%
      pan_left/right: 65%   translate_l/r:    82%
    """
    source_cam2world = np.asarray(source_cam2world, dtype=np.float32)
    if source_cam2world.ndim != 3 or source_cam2world.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected source_cam2world with shape (T, 4, 4), got {tuple(source_cam2world.shape)}"
        )
    local_deltas = _make_local_camera_deltas(
        trajectory_name=trajectory_name,
        num_frames=int(source_cam2world.shape[0]),
        angle_deg=angle_deg,
        distance=distance,
        orbit_radius=orbit_radius,
        orbit_angle_deg=orbit_angle_deg,
        orbit_start_deg=orbit_start_deg,
        trajectory_start_progress=trajectory_start_progress,
    )
    if trajectory_anchor_cam2world is None:
        anchor = source_cam2world[0]
    else:
        anchor = np.asarray(trajectory_anchor_cam2world, dtype=np.float32)
        if anchor.shape != (4, 4):
            raise ValueError(
                "trajectory_anchor_cam2world must be (4,4), got "
                f"{anchor.shape}"
            )
    source_anchor = np.repeat(anchor[None], int(source_cam2world.shape[0]), axis=0)
    return np.matmul(source_anchor, local_deltas).astype(np.float32)


def _warp_source_to_target_frames(
    source_rgb_video: np.ndarray,
    source_depth_video: np.ndarray,
    source_cam2world: np.ndarray,
    source_intrinsics: np.ndarray,
    target_cam2world: np.ndarray,
    target_intrinsics: np.ndarray | None = None,
):
    target_frames = []
    target_depth = []
    target_depth_valid = []

    T = int(target_cam2world.shape[0])
    T_src = int(source_rgb_video.shape[0])
    # Resolve per-frame target K. Accept (3,3) -> static; (T,3,3) -> per-frame;
    # None -> fall back to source K (legacy behavior used by static-scene novel
    # trajectories where target K = source K by design).
    if target_intrinsics is None:
        tgt_K_perframe = None
    else:
        tgt_K_arr = np.asarray(target_intrinsics, dtype=np.float32)
        if tgt_K_arr.ndim == 2:
            tgt_K_perframe = np.broadcast_to(tgt_K_arr, (T, 3, 3)).copy()
        elif tgt_K_arr.ndim == 3 and tgt_K_arr.shape[0] == T:
            tgt_K_perframe = tgt_K_arr
        else:
            raise ValueError(
                f"target_intrinsics must be (3,3) or ({T},3,3), got {tgt_K_arr.shape}"
            )
    for t in range(T):
        # Use frame t (not always 0) so dynamic scenes show motion in the warp.
        # For static scenes this is equivalent since all source frames are identical.
        src_t = min(t, T_src - 1)
        src_rgb = np.clip(
            np.asarray(source_rgb_video[src_t], dtype=np.float32) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        tgt_K_t = tgt_K_perframe[t] if tgt_K_perframe is not None else np.asarray(source_intrinsics, dtype=np.float32)
        warped_rgb, warped_depth, warped_valid = forward_warp_rgbd_to_target(
            src_rgb=src_rgb,
            src_depth=np.asarray(source_depth_video[src_t], dtype=np.float32),
            src_cam2world=np.asarray(source_cam2world[min(src_t, source_cam2world.shape[0] - 1)], dtype=np.float32),
            src_intrinsics=np.asarray(source_intrinsics, dtype=np.float32),
            tgt_cam2world=np.asarray(target_cam2world[t], dtype=np.float32),
            tgt_intrinsics=tgt_K_t,
            src_cam0_pose=None,
            common_scale=1.0,
        )
        target_frames.append(PIL.Image.fromarray(warped_rgb))
        target_depth.append(warped_depth.astype(np.float32))
        target_depth_valid.append(warped_valid.astype(np.float32))

    return (
        target_frames,
        np.stack(target_depth, axis=0).astype(np.float32),
        np.stack(target_depth_valid, axis=0).astype(np.float32),
    )


def warp_source_dynamic_mask_to_target_views(
    *,
    source_dynamic_mask: np.ndarray,
    source_depth_video: np.ndarray,
    source_cam2world: np.ndarray,
    source_intrinsics: np.ndarray,
    target_cam2world: np.ndarray,
    target_intrinsics: np.ndarray,
    min_pixels: int = 64,
) -> np.ndarray:
    """Dense z-buffer reprojection of a source foreground mask.

    The semantic label still comes from the source GMOS mask, but visibility
    and target coverage come from the dense source RGB-D surface rather than
    sparse reconstructed track identities.  This mirrors the established
    Vista4D source-mask contract and is robust to CoWTracker identity dropout
    between autoregressive chunks.
    """
    mask = np.asarray(source_dynamic_mask, dtype=bool)
    depth = np.asarray(source_depth_video, dtype=np.float32)
    src_c2w = np.asarray(source_cam2world, dtype=np.float32)
    tgt_c2w = np.asarray(target_cam2world, dtype=np.float32)
    if mask.ndim != 3:
        raise ValueError(f"source_dynamic_mask must be (T,H,W), got {mask.shape}")
    if depth.shape != mask.shape:
        raise ValueError(
            f"source_depth_video must match mask shape {mask.shape}, got {depth.shape}"
        )
    num_frames, height, width = mask.shape
    if src_c2w.shape != (num_frames, 4, 4):
        raise ValueError(
            f"source_cam2world must be ({num_frames},4,4), got {src_c2w.shape}"
        )
    if tgt_c2w.shape != (num_frames, 4, 4):
        raise ValueError(
            f"target_cam2world must be ({num_frames},4,4), got {tgt_c2w.shape}"
        )

    def _per_frame_intrinsics(value: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (3, 3):
            return np.repeat(arr[None], num_frames, axis=0)
        if arr.shape == (num_frames, 3, 3):
            return arr
        raise ValueError(
            f"{name} must be (3,3) or ({num_frames},3,3), got {arr.shape}"
        )

    src_K = _per_frame_intrinsics(source_intrinsics, "source_intrinsics")
    tgt_K = _per_frame_intrinsics(target_intrinsics, "target_intrinsics")
    target_masks = []
    for frame_idx in range(num_frames):
        mask_rgb = np.repeat(
            (mask[frame_idx].astype(np.uint8) * 255)[..., None],
            repeats=3,
            axis=-1,
        )
        warped_rgb, _, warped_valid = forward_warp_rgbd_to_target(
            src_rgb=mask_rgb,
            src_depth=depth[frame_idx],
            src_cam2world=src_c2w[frame_idx],
            src_intrinsics=src_K[frame_idx],
            tgt_cam2world=tgt_c2w[frame_idx],
            tgt_intrinsics=tgt_K[frame_idx],
            src_cam0_pose=None,
            common_scale=1.0,
        )
        target_masks.append(
            (np.asarray(warped_rgb)[..., 0] >= 128)
            & np.asarray(warped_valid, dtype=bool)
        )
    result = np.stack(target_masks, axis=0)
    counts = result.reshape(num_frames, -1).sum(axis=1)
    threshold = max(int(min_pixels), 0)
    if threshold > 0 and int(counts.min()) < threshold:
        frame_idx = int(counts.argmin())
        raise ValueError(
            "dense z-buffer target foreground mask has fewer than "
            f"{threshold} pixels at frame {frame_idx}: {int(counts[frame_idx])}"
        )
    return result


def build_static_scene_novel_view_sample(
    base_sample: dict,
    trajectory_name: str,
    angle_deg: float,
    distance: float,
    orbit_radius: float,
    height: int,
    width: int,
    orbit_angle_deg: float | None = None,
    orbit_start_deg: float = 0.0,
    trajectory_start_progress: float = 0.0,
    orbit_pivot_mode: str = "fixed_radius",
    orbit_pivot_min_pixels: int = 64,
    orbit_pivot_smooth_radius: int = 2,
    orbit_track_motion_threshold: float = 0.005,
    orbit_track_min_identities: int = 64,
    orbit_radius_scale: float = 1.0,
    min_track_visibility_ratio: float = 0.0,
    target_cam2world_override: np.ndarray | None = None,
    target_intrinsics_override: np.ndarray | None = None,
) -> dict:
    required = (
        "tracks_world",
        "source_rgb_video",
        "source_depth_video",
        "source_cam2world",
        "source_intrinsics",
    )
    missing = [name for name in required if base_sample.get(name) is None]
    if missing:
        raise ValueError(
            "Novel static-scene trajectory synthesis requires source dense/camera fields; "
            f"missing {missing}."
        )

    pivot_mode = str(orbit_pivot_mode).strip().lower()
    # External candidate cameras are imported after the legacy warp is built.
    # The legacy builder still needs a recognised trajectory name even though
    # its cameras are discarded immediately afterwards.  Use a harmless static
    # displacement as the temporary scaffold; this keeps reviewed candidate
    # names (for example ``vista4d_*``/``sphere_*``) out of the legacy trajectory
    # parser and preserves the candidate camera exactly.
    builder_trajectory_name = (
        trajectory_name if target_cam2world_override is None else "static_left"
    )
    orbit_metadata = {}
    if pivot_mode == "foreground_depth" and trajectory_name.startswith("orbit_"):
        foreground_mask = base_sample.get("source_dynamic_mask")
        if foreground_mask is None:
            raise ValueError(
                "orbit_pivot_mode='foreground_depth' requires an explicit source_dynamic_mask "
                "(GMOS for DAVIS); fixed-radius fallback is disabled"
            )
        source_c2w = np.asarray(base_sample["source_cam2world"], dtype=np.float32)
        pivots_world = estimate_foreground_pivots_world(
            depth_video=np.asarray(base_sample["source_depth_video"], dtype=np.float32),
            foreground_mask=np.asarray(foreground_mask, dtype=bool),
            intrinsics=np.asarray(base_sample["source_intrinsics"], dtype=np.float32),
            source_cam2world=source_c2w,
            min_pixels=int(orbit_pivot_min_pixels),
            temporal_smooth_radius=int(orbit_pivot_smooth_radius),
        )
        sweep = float(orbit_angle_deg) if orbit_angle_deg is not None else float(angle_deg)
        target_cam2world, orbit_angles_deg, orbit_radii = build_object_centric_orbit_sequence(
            source_cam2world=source_c2w,
            pivots_world=pivots_world,
            trajectory_name=trajectory_name,
            orbit_angle_deg=sweep,
            orbit_start_deg=float(orbit_start_deg),
        )
        orbit_metadata = {
            "orbit_pivot_mode": "foreground_depth",
            "orbit_pivots_world": pivots_world,
            "orbit_angles_deg": orbit_angles_deg,
            "orbit_radii": orbit_radii,
        }
    elif pivot_mode == "moving_tracks" and trajectory_name.startswith("orbit_"):
        source_c2w = np.asarray(base_sample["source_cam2world"], dtype=np.float32)
        sweep = float(orbit_angle_deg) if orbit_angle_deg is not None else float(angle_deg)
        (
            target_cam2world,
            orbit_angles_deg,
            orbit_radii,
            pivots_world,
            moving_identity_mask,
            orbit_world_up,
        ) = build_moving_track_continuation_orbit_sequence(
            source_cam2world=source_c2w,
            tracks_world=np.asarray(base_sample["tracks_world"], dtype=np.float32),
            tracks_valid=base_sample.get("tracks_world_valid_mask"),
            trajectory_name=trajectory_name,
            orbit_offset_deg=float(orbit_start_deg),
            orbit_sweep_deg=sweep,
            motion_threshold=float(orbit_track_motion_threshold),
            min_identities=int(orbit_track_min_identities),
            temporal_smooth_radius=int(orbit_pivot_smooth_radius),
        )
        orbit_metadata = {
            "orbit_pivot_mode": "moving_tracks",
            "orbit_pivots_world": pivots_world,
            "orbit_angles_deg": orbit_angles_deg,
            "orbit_radii": orbit_radii,
            "orbit_moving_identity_mask": moving_identity_mask,
            "orbit_world_up": orbit_world_up,
        }
    elif pivot_mode == "moving_tracks_fixed_height" and trajectory_name.startswith("orbit_"):
        source_c2w = np.asarray(base_sample["source_cam2world"], dtype=np.float32)
        sweep = float(orbit_angle_deg) if orbit_angle_deg is not None else float(angle_deg)
        pivots_world, moving_identity_mask = estimate_moving_track_pivots_world(
            tracks_world=np.asarray(base_sample["tracks_world"], dtype=np.float32),
            valid_mask=base_sample.get("tracks_world_valid_mask"),
            motion_threshold=float(orbit_track_motion_threshold),
            min_identities=int(orbit_track_min_identities),
            temporal_smooth_radius=int(orbit_pivot_smooth_radius),
        )
        orbit_world_up = estimate_camera_world_up(source_c2w)
        (
            target_cam2world,
            orbit_angles_deg,
            orbit_radii,
            fixed_pivot_world,
        ) = build_fixed_height_look_at_orbit_sequence(
            source_cam2world=source_c2w,
            pivots_world=pivots_world,
            trajectory_name=trajectory_name,
            orbit_angle_deg=sweep,
            world_up=orbit_world_up,
            radius_scale=float(orbit_radius_scale),
        )
        orbit_metadata = {
            "orbit_pivot_mode": "moving_tracks_fixed_height",
            "orbit_pivots_world": np.repeat(
                fixed_pivot_world[None], source_c2w.shape[0], axis=0
            ),
            "orbit_angles_deg": orbit_angles_deg,
            "orbit_radii": orbit_radii,
            "orbit_moving_identity_mask": moving_identity_mask,
            "orbit_world_up": orbit_world_up,
            "orbit_fixed_pivot_world": fixed_pivot_world,
            "orbit_radius_scale": np.float32(orbit_radius_scale),
        }
    elif pivot_mode == "scene_tracks_fixed_height" and trajectory_name.startswith("orbit_"):
        source_c2w = np.asarray(base_sample["source_cam2world"], dtype=np.float32)
        sweep = float(orbit_angle_deg) if orbit_angle_deg is not None else float(angle_deg)
        pivots_world, scene_identity_mask = estimate_scene_track_pivots_world(
            tracks_world=np.asarray(base_sample["tracks_world"], dtype=np.float32),
            valid_mask=base_sample.get("tracks_world_valid_mask"),
            min_identities=int(orbit_track_min_identities),
            temporal_smooth_radius=int(orbit_pivot_smooth_radius),
        )
        orbit_world_up = estimate_camera_world_up(source_c2w)
        (
            target_cam2world,
            orbit_angles_deg,
            orbit_radii,
            fixed_pivot_world,
        ) = build_fixed_height_look_at_orbit_sequence(
            source_cam2world=source_c2w,
            pivots_world=pivots_world,
            trajectory_name=trajectory_name,
            orbit_angle_deg=sweep,
            world_up=orbit_world_up,
            radius_scale=float(orbit_radius_scale),
        )
        orbit_metadata = {
            "orbit_pivot_mode": "scene_tracks_fixed_height",
            "orbit_pivots_world": np.repeat(
                fixed_pivot_world[None], source_c2w.shape[0], axis=0
            ),
            "orbit_angles_deg": orbit_angles_deg,
            "orbit_radii": orbit_radii,
            "orbit_scene_identity_mask": scene_identity_mask,
            "orbit_world_up": orbit_world_up,
            "orbit_fixed_pivot_world": fixed_pivot_world,
            "orbit_radius_scale": np.float32(orbit_radius_scale),
        }
    elif pivot_mode == "fixed_radius":
        target_cam2world = build_target_camera_sequence(
            source_cam2world=np.asarray(base_sample["source_cam2world"], dtype=np.float32),
            trajectory_name=builder_trajectory_name,
            angle_deg=angle_deg,
            distance=distance,
            orbit_radius=orbit_radius,
            orbit_angle_deg=orbit_angle_deg,
            orbit_start_deg=orbit_start_deg,
            trajectory_start_progress=trajectory_start_progress,
            trajectory_anchor_cam2world=base_sample.get(
                "trajectory_anchor_cam2world"
            ),
        )
        orbit_metadata = {"orbit_pivot_mode": "fixed_radius"}
    elif pivot_mode in {
        "foreground_depth",
        "moving_tracks",
        "moving_tracks_fixed_height",
        "scene_tracks_fixed_height",
    }:
        target_cam2world = build_target_camera_sequence(
            source_cam2world=np.asarray(base_sample["source_cam2world"], dtype=np.float32),
            trajectory_name=builder_trajectory_name,
            angle_deg=angle_deg,
            distance=distance,
            orbit_radius=orbit_radius,
            orbit_angle_deg=orbit_angle_deg,
            orbit_start_deg=orbit_start_deg,
            trajectory_start_progress=trajectory_start_progress,
            trajectory_anchor_cam2world=base_sample.get(
                "trajectory_anchor_cam2world"
            ),
        )
        orbit_metadata = {"orbit_pivot_mode": "not_applicable"}
    else:
        raise ValueError(f"unsupported orbit_pivot_mode={orbit_pivot_mode!r}")
    target_intrinsics = np.asarray(
        base_sample.get("target_intrinsics", base_sample["source_intrinsics"]),
        dtype=np.float32,
    )

    # Import a reviewed external camera path before any RGB-D warp or track
    # projection.  This keeps a candidate trajectory from being silently
    # replaced by the legacy named-trajectory builder.
    if target_cam2world_override is not None:
        target_cam2world = np.asarray(target_cam2world_override, dtype=np.float32)
        expected_shape = (
            int(np.asarray(base_sample["source_cam2world"]).shape[0]),
            4,
            4,
        )
        if target_cam2world.shape != expected_shape:
            raise ValueError(
                "target_cam2world_override must have shape "
                f"{expected_shape}, got {target_cam2world.shape}"
            )
        if not np.isfinite(target_cam2world).all():
            raise ValueError("target_cam2world_override contains non-finite values")
    if target_intrinsics_override is not None:
        target_intrinsics = np.asarray(target_intrinsics_override, dtype=np.float32)
        expected_t = int(np.asarray(base_sample["source_cam2world"]).shape[0])
        if target_intrinsics.shape not in {(3, 3), (expected_t, 3, 3)}:
            raise ValueError(
                "target_intrinsics_override must have shape (3,3) or "
                f"({expected_t},3,3), got {target_intrinsics.shape}"
            )
        if not np.isfinite(target_intrinsics).all():
            raise ValueError("target_intrinsics_override contains non-finite values")

    target_frames, target_depth, target_depth_valid = _warp_source_to_target_frames(
        source_rgb_video=np.asarray(base_sample["source_rgb_video"], dtype=np.float32),
        source_depth_video=np.asarray(base_sample["source_depth_video"], dtype=np.float32),
        source_cam2world=np.asarray(base_sample["source_cam2world"], dtype=np.float32),
        source_intrinsics=np.asarray(base_sample["source_intrinsics"], dtype=np.float32),
        target_cam2world=target_cam2world,
        target_intrinsics=target_intrinsics,
    )

    _, track_vis, track_depth = project_tracks_to_target_views(
        torch.as_tensor(np.asarray(base_sample["tracks_world"]), dtype=torch.float32),
        torch.as_tensor(target_cam2world, dtype=torch.float32),
        torch.as_tensor(target_intrinsics, dtype=torch.float32),
        image_h=int(height),
        image_w=int(width),
    )
    track_vis_np = track_vis[0].detach().cpu().numpy().astype(np.float32)
    track_depth_np = (
        track_depth[0].detach().cpu().numpy().astype(np.float32) * track_vis_np
    )

    # Report visibility relative to the same tracks under the source cameras,
    # not relative to every dense identity. This makes the gate meaningful for
    # partial reconstructions and implements the robot-scene "retain at least
    # 50% of the visible workspace" contract.
    _, source_track_vis, _ = project_tracks_to_target_views(
        torch.as_tensor(np.asarray(base_sample["tracks_world"]), dtype=torch.float32),
        torch.as_tensor(np.asarray(base_sample["source_cam2world"]), dtype=torch.float32),
        torch.as_tensor(np.asarray(base_sample["source_intrinsics"]), dtype=torch.float32),
        image_h=int(height),
        image_w=int(width),
    )
    source_vis_np = source_track_vis[0].detach().cpu().numpy().astype(bool)
    valid_mask = base_sample.get("tracks_world_valid_mask")
    if valid_mask is not None:
        valid_np = np.asarray(valid_mask, dtype=bool)
        if valid_np.shape == source_vis_np.shape:
            source_vis_np &= valid_np
    denom = source_vis_np.sum(axis=0)
    retained = (track_vis_np.astype(bool) & source_vis_np).sum(axis=0)
    visibility_ratio = np.divide(
        retained,
        denom,
        out=np.ones_like(retained, dtype=np.float32),
        where=denom > 0,
    ).astype(np.float32)
    threshold = float(min_track_visibility_ratio)
    if threshold > 0.0 and float(visibility_ratio.min(initial=1.0)) < threshold:
        worst = int(np.argmin(visibility_ratio))
        raise ValueError(
            f"trajectory {trajectory_name} retains only {visibility_ratio[worst]:.3f} "
            f"of source-visible tracks at frame {worst}; required >= {threshold:.3f}"
        )

    novel_sample = dict(base_sample)
    novel_sample["target_video"] = target_frames
    novel_sample["target_cam2world"] = target_cam2world
    novel_sample["target_intrinsics"] = target_intrinsics
    novel_sample["target_depth"] = target_depth
    novel_sample["target_depth_valid"] = target_depth_valid
    novel_sample["reproj_src_depth_at_target"] = track_depth_np
    novel_sample["reproj_src_depth_valid"] = track_vis_np
    novel_sample["trajectory_name"] = trajectory_name
    novel_sample["trajectory_kind"] = "novel_view"
    novel_sample["trajectory_track_visibility_ratio"] = visibility_ratio
    novel_sample.update(orbit_metadata)
    return novel_sample


def save_static_scene_novel_view_payload(sample: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        target_cam2world=np.asarray(sample["target_cam2world"], dtype=np.float32),
        target_intrinsics=np.asarray(sample["target_intrinsics"], dtype=np.float32),
        target_depth=np.asarray(sample["target_depth"], dtype=np.float32),
        target_depth_valid=np.asarray(sample["target_depth_valid"], dtype=np.float32),
        reproj_src_depth_at_target=np.asarray(sample["reproj_src_depth_at_target"], dtype=np.float32),
        reproj_src_depth_valid=np.asarray(sample["reproj_src_depth_valid"], dtype=np.float32),
        tracks_world=np.asarray(sample["tracks_world"], dtype=np.float32),
        tracks_world_valid_mask=np.asarray(sample.get("tracks_world_valid_mask", []), dtype=bool),
        source_cam2world=np.asarray(sample.get("source_cam2world", []), dtype=np.float32),
        source_intrinsics=np.asarray(sample.get("source_intrinsics", []), dtype=np.float32),
        tracks_source_xy=np.asarray(sample.get("tracks_source_xy", []) if sample.get("tracks_source_xy") is not None else [], dtype=np.float32),
        common_scale=np.asarray(sample.get("common_scale", 1.0), dtype=np.float32),
        trajectory_name=np.asarray(sample.get("trajectory_name", ""), dtype=str),
        orbit_pivot_mode=np.asarray(sample.get("orbit_pivot_mode", ""), dtype=str),
        orbit_pivots_world=np.asarray(sample.get("orbit_pivots_world", []), dtype=np.float32),
        orbit_angles_deg=np.asarray(sample.get("orbit_angles_deg", []), dtype=np.float32),
        orbit_radii=np.asarray(sample.get("orbit_radii", []), dtype=np.float32),
        orbit_moving_identity_mask=np.asarray(
            sample.get("orbit_moving_identity_mask", []), dtype=np.uint8
        ),
        orbit_world_up=np.asarray(
            sample.get("orbit_world_up", []), dtype=np.float32
        ),
        source_dynamic_mask=np.asarray(
            sample.get("source_dynamic_mask", []), dtype=np.uint8
        ),
        trajectory_track_visibility_ratio=np.asarray(
            sample.get("trajectory_track_visibility_ratio", []), dtype=np.float32
        ),
        target_dynamic_mask=np.asarray(
            sample.get("target_dynamic_mask", []), dtype=np.uint8
        ),
        syn4d_sequence_idx=np.asarray(sample.get("syn4d_sequence_idx", -1), dtype=np.int32),
        syn4d_source_camera_idx=np.asarray(sample.get("syn4d_source_camera_idx", -1), dtype=np.int32),
        syn4d_target_camera_idx=np.asarray(sample.get("syn4d_target_camera_idx", -1), dtype=np.int32),
        syn4d_start_frame=np.asarray(sample.get("syn4d_start_frame", -1), dtype=np.int32),
        syn4d_temporal_stride=np.asarray(sample.get("syn4d_temporal_stride", -1), dtype=np.int32),
        src_cam_name=np.asarray(sample.get("src_cam_name", ""), dtype=str),
        tgt_cam_name=np.asarray(sample.get("tgt_cam_name", ""), dtype=str),
    )
    return out_path
