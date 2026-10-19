"""Source-view visibility and temporal RGB observations for 3D trajectories.

The functions in this module are dataset-independent. Tracks, source cameras,
depth, and RGB must already share the source-camera-0 normalized gauge used by
the Track4D-360 dataloaders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_NAMES = (
    "invalid_track",
    "behind_camera",
    "out_of_frame",
    "no_depth",
    "occluded",
    "depth_inconsistent_front",
    "visible",
)

STATE_INVALID = 0
STATE_BEHIND = 1
STATE_OUT_OF_FRAME = 2
STATE_NO_DEPTH = 3
STATE_OCCLUDED = 4
STATE_FRONT_INCONSISTENT = 5
STATE_VISIBLE = 6


@dataclass(frozen=True)
class VisibilityResult:
    """Per-identity, per-time source observation classification."""

    state: np.ndarray
    uv: np.ndarray
    camera_depth: np.ndarray
    sampled_depth: np.ndarray
    relative_depth_residual: np.ndarray
    sampled_rgb_all: np.ndarray
    visible_rgb: np.ndarray

    @property
    def visible(self) -> np.ndarray:
        return self.state == STATE_VISIBLE


def _broadcast_time(array: np.ndarray, time_count: int, tail_shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.shape == tail_shape:
        return np.broadcast_to(value, (time_count, *tail_shape)).copy()
    if value.shape == (time_count, *tail_shape):
        return value
    raise ValueError(
        f"expected {tail_shape} or {(time_count, *tail_shape)}, got {value.shape}"
    )


def _bilinear_sample_rgb(frame: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample one HWC RGB frame at known in-frame coordinates."""
    height, width = frame.shape[:2]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (u - x0).astype(np.float32)
    wy = (v - y0).astype(np.float32)
    top = frame[y0, x0] * (1.0 - wx[:, None]) + frame[y0, x1] * wx[:, None]
    bottom = frame[y1, x0] * (1.0 - wx[:, None]) + frame[y1, x1] * wx[:, None]
    return top * (1.0 - wy[:, None]) + bottom * wy[:, None]


def _normalize_rgb_video(rgb_video: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_video)
    if rgb.ndim != 4:
        raise ValueError(f"rgb_video must have four dimensions, got {rgb.shape}")
    if rgb.shape[1] in (1, 3, 4) and rgb.shape[-1] not in (1, 3, 4):
        rgb = np.moveaxis(rgb, 1, -1)
    if rgb.shape[-1] < 3:
        rgb = np.repeat(rgb, 3, axis=-1)
    rgb = rgb[..., :3].astype(np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def classify_source_visibility(
    tracks_world: np.ndarray,
    track_valid: np.ndarray | None,
    source_cam2world: np.ndarray,
    source_intrinsics: np.ndarray,
    source_depth_video: np.ndarray,
    source_rgb_video: np.ndarray,
    *,
    relative_depth_tolerance: float = 0.05,
    min_depth: float = 1e-6,
) -> VisibilityResult:
    """Project trajectories into source frames and classify observability.

    Depth is sampled with nearest-neighbor lookup to avoid blending foreground
    and background at discontinuities. RGB uses bilinear sampling. RGB values
    for non-visible observations remain NaN in ``visible_rgb``; the ungated
    in-frame sample is retained separately in ``sampled_rgb_all``.
    """
    tracks = np.asarray(tracks_world, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"tracks_world must be (N,T,3), got {tracks.shape}")
    identity_count, time_count = tracks.shape[:2]
    rgb = _normalize_rgb_video(source_rgb_video)
    depth = np.asarray(source_depth_video, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.shape != rgb.shape[:3]:
        raise ValueError(f"source depth/RGB shapes disagree: {depth.shape} vs {rgb.shape}")
    if depth.shape[0] != time_count:
        raise ValueError(f"track/RGB time counts disagree: {time_count} vs {depth.shape[0]}")

    cameras = _broadcast_time(source_cam2world, time_count, (4, 4))
    intrinsics = _broadcast_time(source_intrinsics, time_count, (3, 3))
    if track_valid is None:
        support = np.isfinite(tracks).all(axis=-1) & (np.linalg.norm(tracks, axis=-1) > 1e-8)
    else:
        support = np.asarray(track_valid, dtype=bool)
        if support.shape != (identity_count, time_count):
            raise ValueError(
                f"track_valid must be {(identity_count, time_count)}, got {support.shape}"
            )
        support &= np.isfinite(tracks).all(axis=-1)

    state = np.full((identity_count, time_count), STATE_INVALID, dtype=np.uint8)
    uv = np.full((identity_count, time_count, 2), np.nan, dtype=np.float32)
    camera_depth = np.full((identity_count, time_count), np.nan, dtype=np.float32)
    sampled_depth = np.full_like(camera_depth, np.nan)
    residual = np.full_like(camera_depth, np.nan)
    sampled_rgb_all = np.full((identity_count, time_count, 3), np.nan, dtype=np.float32)
    visible_rgb = np.full_like(sampled_rgb_all, np.nan)
    height, width = rgb.shape[1:3]
    tolerance = max(float(relative_depth_tolerance), 0.0)

    for time_idx in range(time_count):
        supported = support[:, time_idx]
        if not np.any(supported):
            continue
        ids = np.flatnonzero(supported)
        w2c = np.linalg.inv(cameras[time_idx]).astype(np.float32)
        points = tracks[ids, time_idx]
        camera = points @ w2c[:3, :3].T + w2c[:3, 3]
        z = camera[:, 2]
        camera_depth[ids, time_idx] = z
        positive = np.isfinite(z) & (z > min_depth)
        state[ids[~positive], time_idx] = STATE_BEHIND
        if not np.any(positive):
            continue

        positive_ids = ids[positive]
        camera_pos = camera[positive]
        z_pos = z[positive]
        projection = camera_pos @ intrinsics[time_idx].T
        u = projection[:, 0] / projection[:, 2]
        v = projection[:, 1] / projection[:, 2]
        uv[positive_ids, time_idx, 0] = u
        uv[positive_ids, time_idx, 1] = v
        in_frame = (
            np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0.0)
            & (u <= width - 1.0)
            & (v >= 0.0)
            & (v <= height - 1.0)
        )
        state[positive_ids[~in_frame], time_idx] = STATE_OUT_OF_FRAME
        if not np.any(in_frame):
            continue

        frame_ids = positive_ids[in_frame]
        frame_u = u[in_frame]
        frame_v = v[in_frame]
        frame_z = z_pos[in_frame]
        sampled_rgb_all[frame_ids, time_idx] = _bilinear_sample_rgb(
            rgb[time_idx], frame_u, frame_v
        )
        ui = np.rint(frame_u).astype(np.int64)
        vi = np.rint(frame_v).astype(np.int64)
        observed_depth = depth[time_idx, vi, ui]
        sampled_depth[frame_ids, time_idx] = observed_depth
        depth_ok = np.isfinite(observed_depth) & (observed_depth > min_depth)
        state[frame_ids[~depth_ok], time_idx] = STATE_NO_DEPTH
        if not np.any(depth_ok):
            continue

        depth_ids = frame_ids[depth_ok]
        relative = (frame_z[depth_ok] - observed_depth[depth_ok]) / np.maximum(
            observed_depth[depth_ok], min_depth
        )
        residual[depth_ids, time_idx] = relative
        occluded = relative > tolerance
        front = relative < -tolerance
        visible = ~(occluded | front)
        state[depth_ids[occluded], time_idx] = STATE_OCCLUDED
        state[depth_ids[front], time_idx] = STATE_FRONT_INCONSISTENT
        state[depth_ids[visible], time_idx] = STATE_VISIBLE
        visible_ids = depth_ids[visible]
        visible_rgb[visible_ids, time_idx] = sampled_rgb_all[visible_ids, time_idx]

    return VisibilityResult(
        state=state,
        uv=uv,
        camera_depth=camera_depth,
        sampled_depth=sampled_depth,
        relative_depth_residual=residual,
        sampled_rgb_all=sampled_rgb_all,
        visible_rgb=visible_rgb,
    )
