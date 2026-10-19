"""Object-centric camera orbits estimated from foreground depth.

The legacy novel-view helper orbits around a fixed point on the source optical
axis.  That is useful for small static-scene diagnostics, but it is not an
object orbit when the subject is farther away or off centre.  This module uses
an explicit foreground mask (GMOS for DAVIS) and reconstructed metric depth
(VGGT-Omega) to estimate a per-frame 3D subject pivot, then rotates both the
camera centre and camera orientation around that pivot.

There is intentionally no silent fallback when the mask/depth contract is not
met.  A missing pivot would turn the experiment back into the invalid fixed-
radius/yaw-like protocol that this path is designed to replace.
"""
from __future__ import annotations

import numpy as np


def _per_frame_intrinsics(intrinsics: np.ndarray, num_frames: int) -> np.ndarray:
    values = np.asarray(intrinsics, dtype=np.float32)
    if values.shape == (3, 3):
        return np.repeat(values[None], int(num_frames), axis=0)
    if values.shape == (int(num_frames), 3, 3):
        return values
    raise ValueError(
        f"intrinsics must be (3,3) or ({int(num_frames)},3,3), got {values.shape}"
    )


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"orbit axis is degenerate: {axis}")
    x, y, z = axis / norm
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return (np.eye(3) * c + (1.0 - c) * np.outer([x, y, z], [x, y, z]) + s * skew).astype(
        np.float32
    )


def estimate_foreground_pivots_world(
    *,
    depth_video: np.ndarray,
    foreground_mask: np.ndarray,
    intrinsics: np.ndarray,
    source_cam2world: np.ndarray,
    min_pixels: int = 64,
    depth_quantiles: tuple[float, float] = (0.05, 0.95),
    temporal_smooth_radius: int = 2,
) -> np.ndarray:
    """Estimate one robust subject centroid per frame in world coordinates.

    Depth outliers inside the segmentation are removed independently per frame
    using the requested quantiles.  The coordinate-wise median is robust to a
    small amount of mask leakage.  A short temporal median stabilises GMOS mask
    boundaries without assuming that the dynamic subject itself is static.
    """
    depth = np.asarray(depth_video, dtype=np.float32)
    mask = np.asarray(foreground_mask, dtype=bool)
    c2w = np.asarray(source_cam2world, dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[None]
    if mask.ndim == 2:
        mask = mask[None]
    if depth.ndim != 3 or mask.shape != depth.shape:
        raise ValueError(f"depth/mask must have identical (T,H,W) shape, got {depth.shape}/{mask.shape}")
    if c2w.shape != (depth.shape[0], 4, 4):
        raise ValueError(f"source_cam2world must be ({depth.shape[0]},4,4), got {c2w.shape}")
    K = _per_frame_intrinsics(intrinsics, depth.shape[0])
    q_low, q_high = map(float, depth_quantiles)
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"invalid depth_quantiles={depth_quantiles}")

    pivots = []
    for frame_idx in range(depth.shape[0]):
        d = depth[frame_idx]
        valid = mask[frame_idx] & np.isfinite(d) & (d > 1e-6)
        values = d[valid]
        if values.size < int(min_pixels):
            raise ValueError(
                f"frame {frame_idx}: only {values.size} valid foreground-depth pixels; "
                f"required {int(min_pixels)}"
            )
        lo, hi = np.quantile(values, [q_low, q_high])
        valid &= (d >= float(lo)) & (d <= float(hi))
        yy, xx = np.nonzero(valid)
        z = d[yy, xx]
        k = K[frame_idx]
        x = (xx.astype(np.float32) - float(k[0, 2])) * z / float(k[0, 0])
        y = (yy.astype(np.float32) - float(k[1, 2])) * z / float(k[1, 1])
        camera_points = np.stack([x, y, z], axis=-1)
        world_points = camera_points @ c2w[frame_idx, :3, :3].T + c2w[frame_idx, :3, 3]
        pivots.append(np.median(world_points, axis=0))

    pivots = np.asarray(pivots, dtype=np.float32)
    radius = max(int(temporal_smooth_radius), 0)
    if radius:
        smoothed = np.empty_like(pivots)
        for frame_idx in range(pivots.shape[0]):
            start = max(0, frame_idx - radius)
            stop = min(pivots.shape[0], frame_idx + radius + 1)
            smoothed[frame_idx] = np.median(pivots[start:stop], axis=0)
        pivots = smoothed
    return pivots


def estimate_moving_track_pivots_world(
    *,
    tracks_world: np.ndarray,
    valid_mask: np.ndarray | None = None,
    motion_threshold: float = 0.01,
    min_identities: int = 64,
    temporal_smooth_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a dynamic-subject pivot directly from GT world-space tracks.

    Syn4D's static environment tracks remain fixed in world coordinates, while
    the animated people/objects move.  Selecting identities by their maximum
    displacement from the query frame therefore isolates the dynamic subject
    without an image-space segmentation or a depth-estimator fallback.

    Returns ``(pivots_world, moving_identity_mask)``.  The function fails when
    too few moving identities are present; silently falling back to all tracks
    would turn the requested object orbit into an environment-centred orbit.
    """
    tracks = np.asarray(tracks_world, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"tracks_world must be (N,T,3), got {tracks.shape}")
    finite = np.isfinite(tracks).all(axis=-1)
    nonzero = np.linalg.norm(tracks, axis=-1) > 1e-8
    valid = finite & nonzero
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != tracks.shape[:2]:
            raise ValueError(
                f"valid_mask must be {tracks.shape[:2]}, got {supplied.shape}"
            )
        valid &= supplied

    query_valid = valid[:, 0]
    displacement = np.linalg.norm(tracks - tracks[:, :1], axis=-1)
    displacement[~valid] = 0.0
    moving = query_valid & (displacement.max(axis=1) >= float(motion_threshold))
    if int(moving.sum()) < int(min_identities):
        raise ValueError(
            f"only {int(moving.sum())} moving GT-track identities exceed "
            f"threshold {float(motion_threshold):.6g}; required {int(min_identities)}"
        )

    pivots = np.empty((tracks.shape[1], 3), dtype=np.float32)
    for frame_idx in range(tracks.shape[1]):
        frame_keep = moving & valid[:, frame_idx]
        if int(frame_keep.sum()) < int(min_identities):
            raise ValueError(
                f"frame {frame_idx}: only {int(frame_keep.sum())} valid moving GT tracks; "
                f"required {int(min_identities)}"
            )
        pivots[frame_idx] = np.median(tracks[frame_keep, frame_idx], axis=0)

    radius = max(int(temporal_smooth_radius), 0)
    if radius:
        smoothed = np.empty_like(pivots)
        for frame_idx in range(pivots.shape[0]):
            start = max(0, frame_idx - radius)
            stop = min(pivots.shape[0], frame_idx + radius + 1)
            smoothed[frame_idx] = np.median(pivots[start:stop], axis=0)
        pivots = smoothed
    return pivots, moving



def estimate_scene_track_pivots_world(
    *,
    tracks_world: np.ndarray,
    valid_mask: np.ndarray | None = None,
    min_identities: int = 64,
    temporal_smooth_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a scene-centre pivot from all valid world-space tracks.

    This explicit fallback is for static-camera robot clips whose 4RC world
    tracks contain too few moving identities. Keeping it separate from
    ``estimate_moving_track_pivots_world`` prevents object-centric protocols
    from silently degrading to a scene-centred arc.
    """
    tracks = np.asarray(tracks_world, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"tracks_world must be (N,T,3), got {tracks.shape}")
    valid = np.isfinite(tracks).all(axis=-1) & (np.linalg.norm(tracks, axis=-1) > 1e-8)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != tracks.shape[:2]:
            raise ValueError(
                f"valid_mask must be {tracks.shape[:2]}, got {supplied.shape}"
            )
        valid &= supplied

    pivots = np.empty((tracks.shape[1], 3), dtype=np.float32)
    for frame_idx in range(tracks.shape[1]):
        frame_keep = valid[:, frame_idx]
        if int(frame_keep.sum()) < int(min_identities):
            raise ValueError(
                f"frame {frame_idx}: only {int(frame_keep.sum())} valid scene tracks; "
                f"required {int(min_identities)}"
            )
        pivots[frame_idx] = np.median(tracks[frame_keep, frame_idx], axis=0)

    radius = max(int(temporal_smooth_radius), 0)
    if radius:
        smoothed = np.empty_like(pivots)
        for frame_idx in range(pivots.shape[0]):
            start = max(0, frame_idx - radius)
            stop = min(pivots.shape[0], frame_idx + radius + 1)
            smoothed[frame_idx] = np.median(pivots[start:stop], axis=0)
        pivots = smoothed
    return pivots, valid[:, 0]


def estimate_camera_world_up(source_cam2world: np.ndarray) -> np.ndarray:
    """Estimate a stable world-up axis from an OpenCV camera sequence.

    OpenCV camera local +Y points down, so the negative second rotation column
    is an up observation. Aligning all observations to the first before taking
    their median prevents an occasional flipped VGGT-Omega pose from cancelling
    the axis. This is the generated-segment counterpart of Syn4D's explicit
    transformed +Z world-up.
    """
    source = np.asarray(source_cam2world, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (4, 4):
        raise ValueError(f"source_cam2world must be (T,4,4), got {source.shape}")
    up = -source[:, :3, 1].astype(np.float64)
    finite = np.isfinite(up).all(axis=1)
    norms = np.linalg.norm(up, axis=1)
    keep = finite & (norms > 1e-8)
    if not bool(keep.any()):
        raise ValueError("camera sequence has no valid world-up observations")
    up = up[keep] / norms[keep, None]
    reference = up[0]
    signs = np.where((up @ reference)[:, None] < 0.0, -1.0, 1.0)
    up = up * signs
    estimate = np.median(up, axis=0)
    norm = float(np.linalg.norm(estimate))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("camera-derived world-up is degenerate")
    return (estimate / norm).astype(np.float32)



def build_object_centric_orbit_sequence(
    *,
    source_cam2world: np.ndarray,
    pivots_world: np.ndarray,
    trajectory_name: str,
    orbit_angle_deg: float,
    orbit_start_deg: float = 0.0,
    orbit_axis_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a dynamic source camera rigidly around per-frame subject pivots.

    Rotating the camera centre and orientation by the same matrix preserves
    ``R.T @ (pivot - centre)``.  Consequently the subject stays at the same
    relative image direction throughout the orbit and frame zero is exactly the
    source pose.  This is the key distinction from an in-place yaw.

    Returns ``(target_cam2world, angles_deg, horizontal_radii)``.
    """
    source = np.asarray(source_cam2world, dtype=np.float32)
    pivots = np.asarray(pivots_world, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (4, 4):
        raise ValueError(f"source_cam2world must be (T,4,4), got {source.shape}")
    if pivots.shape != (source.shape[0], 3):
        raise ValueError(f"pivots_world must be ({source.shape[0]},3), got {pivots.shape}")
    if trajectory_name not in {"orbit_left", "orbit_right"}:
        raise ValueError(f"object-centric orbit requires orbit_left/right, got {trajectory_name}")

    axis = source[0, :3, 1] if orbit_axis_world is None else np.asarray(orbit_axis_world)
    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)
    sign = 1.0 if trajectory_name == "orbit_left" else -1.0
    progress = np.linspace(0.0, 1.0, source.shape[0], dtype=np.float32)
    angles_deg = sign * (float(orbit_start_deg) + progress * float(orbit_angle_deg))
    target = source.copy()
    radii = np.empty((source.shape[0],), dtype=np.float32)
    for frame_idx, angle_deg in enumerate(angles_deg):
        source_center = source[frame_idx, :3, 3]
        offset = source_center - pivots[frame_idx]
        axial = axis * float(np.dot(offset, axis))
        radii[frame_idx] = float(np.linalg.norm(offset - axial))
        if abs(float(angle_deg)) <= 1e-8:
            # Preserve the shared first-frame contract bit for bit.
            target[frame_idx] = source[frame_idx]
            continue
        rotation = _axis_angle_rotation(axis, np.deg2rad(float(angle_deg)))
        target[frame_idx, :3, 3] = pivots[frame_idx] + rotation @ offset
        target[frame_idx, :3, :3] = rotation @ source[frame_idx, :3, :3]
    return target.astype(np.float32), angles_deg.astype(np.float32), radii


def build_moving_track_continuation_orbit_sequence(
    *,
    source_cam2world: np.ndarray,
    tracks_world: np.ndarray,
    tracks_valid: np.ndarray | None,
    trajectory_name: str,
    orbit_offset_deg: float,
    orbit_sweep_deg: float = 0.0,
    motion_threshold: float = 0.005,
    min_identities: int = 64,
    temporal_smooth_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the next arc from refreshed VGGT-Omega + CoWTracker annotations.

    source_cam2world is reconstructed from the previous generated 49-frame
    segment. A constant 90-degree offset maps a 0--90 path to 90--180; using
    another 0--90 sweep would incorrectly compose into 0--180 and break the
    shared boundary frame. orbit_sweep_deg therefore defaults to zero and
    exists only for explicit diagnostics.

    Returns target cameras, applied angles, radii, per-frame track pivots,
    moving-identity mask, and the camera-derived world-up axis.
    """
    pivots, moving = estimate_moving_track_pivots_world(
        tracks_world=tracks_world,
        valid_mask=tracks_valid,
        motion_threshold=float(motion_threshold),
        min_identities=int(min_identities),
        temporal_smooth_radius=int(temporal_smooth_radius),
    )
    world_up = estimate_camera_world_up(source_cam2world)
    target, angles, radii = build_object_centric_orbit_sequence(
        source_cam2world=source_cam2world,
        pivots_world=pivots,
        trajectory_name=trajectory_name,
        orbit_angle_deg=float(orbit_sweep_deg),
        orbit_start_deg=float(orbit_offset_deg),
        orbit_axis_world=world_up,
    )
    return target, angles, radii, pivots, moving, world_up



def _look_at_cam2world(
    *,
    camera_center: np.ndarray,
    look_at: np.ndarray,
    world_up: np.ndarray,
) -> np.ndarray:
    """Build an upright OpenCV cam2world pose looking at ``look_at``.

    Camera local +Z is forward and local +Y is down.  ``world_up`` therefore
    fixes roll while the optical axis follows the orbit pivot.
    """
    center = np.asarray(camera_center, dtype=np.float64)
    pivot = np.asarray(look_at, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)
    up_norm = float(np.linalg.norm(up))
    if not np.isfinite(up_norm) or up_norm <= 1e-8:
        raise ValueError(f"world_up is degenerate: {world_up}")
    up /= up_norm

    forward = pivot - center
    forward_norm = float(np.linalg.norm(forward))
    if not np.isfinite(forward_norm) or forward_norm <= 1e-8:
        raise ValueError("camera center coincides with the orbit pivot")
    forward /= forward_norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if not np.isfinite(right_norm) or right_norm <= 1e-8:
        raise ValueError("look-at direction is parallel to world_up")
    right /= right_norm
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.stack([right, down, forward], axis=1).astype(np.float32)
    pose[:3, 3] = center.astype(np.float32)
    return pose


def build_fixed_height_look_at_orbit_sequence(
    *,
    source_cam2world: np.ndarray,
    pivots_world: np.ndarray,
    trajectory_name: str,
    orbit_angle_deg: float,
    world_up: np.ndarray,
    radius_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a fixed-height arc from the source frame-0 camera.

    Unlike :func:`build_object_centric_orbit_sequence`, this protocol does not
    compose the requested angle with a moving source camera and moving pivot.
    It takes one temporal-median subject pivot, one frame-0 camera center, and
    one explicit world-up axis. Camera height and horizontal radius are then
    constant for all frames, and every pose is rebuilt to look at the pivot.

    Returns ``(target_cam2world, angles_deg, radii, fixed_pivot_world)``.
    """
    source = np.asarray(source_cam2world, dtype=np.float32)
    pivots = np.asarray(pivots_world, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (4, 4):
        raise ValueError(f"source_cam2world must be (T,4,4), got {source.shape}")
    if pivots.shape != (source.shape[0], 3):
        raise ValueError(f"pivots_world must be ({source.shape[0]},3), got {pivots.shape}")
    if trajectory_name not in {"orbit_left", "orbit_right"}:
        raise ValueError(f"fixed-height orbit requires orbit_left/right, got {trajectory_name}")
    if not np.isfinite(radius_scale) or float(radius_scale) <= 0.0:
        raise ValueError(f"radius_scale must be positive, got {radius_scale}")

    axis = np.asarray(world_up, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1e-8:
        raise ValueError(f"world_up is degenerate: {world_up}")
    axis /= axis_norm
    fixed_pivot = np.median(pivots, axis=0).astype(np.float64)
    source_center = source[0, :3, 3].astype(np.float64)
    offset = source_center - fixed_pivot
    axial_offset = axis * float(np.dot(offset, axis))
    horizontal_offset = offset - axial_offset
    source_radius = float(np.linalg.norm(horizontal_offset))
    if not np.isfinite(source_radius) or source_radius <= 1e-8:
        raise ValueError("source frame-0 camera has zero horizontal orbit radius")
    horizontal_offset *= float(radius_scale)
    radius = source_radius * float(radius_scale)

    sign = 1.0 if trajectory_name == "orbit_left" else -1.0
    progress = np.linspace(0.0, 1.0, source.shape[0], dtype=np.float32)
    angles_deg = sign * progress * float(orbit_angle_deg)
    target = np.empty_like(source)
    for frame_idx, angle_deg in enumerate(angles_deg):
        rotation = _axis_angle_rotation(axis, np.deg2rad(float(angle_deg)))
        center = fixed_pivot + axial_offset + rotation @ horizontal_offset
        target[frame_idx] = _look_at_cam2world(
            camera_center=center,
            look_at=fixed_pivot,
            world_up=axis,
        )
    radii = np.full((source.shape[0],), radius, dtype=np.float32)
    return (
        target.astype(np.float32),
        angles_deg.astype(np.float32),
        radii,
        fixed_pivot.astype(np.float32),
    )
