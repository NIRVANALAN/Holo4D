import numpy as np
import pytest

from holo4d.novel_view.object_centric_orbit import (
    build_moving_track_continuation_orbit_sequence,
    build_fixed_height_look_at_orbit_sequence,
    build_object_centric_orbit_sequence,
    estimate_foreground_pivots_world,
    estimate_moving_track_pivots_world,
)


def test_masked_depth_estimates_off_axis_pivot():
    depth = np.full((3, 8, 10), 8.0, dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[:, 2:6, 6:9] = True
    K = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    pivots = estimate_foreground_pivots_world(
        depth_video=depth,
        foreground_mask=mask,
        intrinsics=K,
        source_cam2world=c2w,
        min_pixels=4,
        temporal_smooth_radius=0,
    )
    np.testing.assert_allclose(pivots[:, 2], 8.0, atol=1e-6)
    assert np.all(pivots[:, 0] > 0.0)


def test_missing_foreground_fails_instead_of_using_fixed_radius():
    with pytest.raises(ValueError, match="valid foreground-depth pixels"):
        estimate_foreground_pivots_world(
            depth_video=np.ones((1, 4, 4), dtype=np.float32),
            foreground_mask=np.zeros((1, 4, 4), dtype=bool),
            intrinsics=np.eye(3, dtype=np.float32),
            source_cam2world=np.eye(4, dtype=np.float32)[None],
            min_pixels=1,
        )


def test_orbit_moves_camera_and_preserves_subject_camera_coordinates():
    source = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
    pivot = np.repeat(np.array([[1.0, 0.25, 6.0]], dtype=np.float32), 5, axis=0)
    target, angles, radii = build_object_centric_orbit_sequence(
        source_cam2world=source,
        pivots_world=pivot,
        trajectory_name="orbit_left",
        orbit_angle_deg=180.0,
    )
    np.testing.assert_array_equal(target[0], source[0])
    assert not np.allclose(target[-1, :3, 3], source[-1, :3, 3])
    assert np.all(radii > 5.0)
    for frame_idx in range(source.shape[0]):
        before = source[frame_idx, :3, :3].T @ (pivot[frame_idx] - source[frame_idx, :3, 3])
        after = target[frame_idx, :3, :3].T @ (pivot[frame_idx] - target[frame_idx, :3, 3])
        np.testing.assert_allclose(after, before, atol=1e-5)
    np.testing.assert_allclose(angles, [0.0, 45.0, 90.0, 135.0, 180.0])


def test_moving_gt_tracks_exclude_static_environment():
    static = np.repeat(
        np.array([[[20.0, 0.0, 5.0]], [[-20.0, 0.0, 5.0]]], dtype=np.float32),
        5,
        axis=1,
    )
    dynamic = np.zeros((4, 5, 3), dtype=np.float32)
    dynamic[..., 2] = 6.0
    dynamic[:, :, 0] = np.linspace(0.0, 1.0, 5)[None]
    tracks = np.concatenate([static, dynamic], axis=0)
    pivots, moving = estimate_moving_track_pivots_world(
        tracks_world=tracks,
        motion_threshold=0.1,
        min_identities=4,
        temporal_smooth_radius=0,
    )
    assert moving.tolist() == [False, False, True, True, True, True]
    np.testing.assert_allclose(pivots[:, 0], np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(pivots[:, 2], 6.0)


@pytest.mark.parametrize(
    "trajectory_name,endpoint",
    [("orbit_left", 90.0), ("orbit_right", -90.0)],
)
def test_fixed_height_orbit_ignores_source_drift_and_looks_at_static_center(
    trajectory_name, endpoint
):
    source = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
    source[:, :3, 3] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.3, -0.2],
            [1.0, 0.7, -0.6],
            [1.8, 1.2, -1.0],
            [3.0, 2.0, -1.5],
        ],
        dtype=np.float32,
    )
    pivots = np.repeat(np.array([[0.0, 0.0, 4.0]], dtype=np.float32), 5, axis=0)
    pivots[:, 0] += np.linspace(-0.4, 0.4, 5)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    target, angles, radii, fixed_pivot = build_fixed_height_look_at_orbit_sequence(
        source_cam2world=source,
        pivots_world=pivots,
        trajectory_name=trajectory_name,
        orbit_angle_deg=90.0,
        world_up=world_up,
    )

    np.testing.assert_allclose(fixed_pivot, [0.0, 0.0, 4.0], atol=1e-6)
    np.testing.assert_allclose(angles[-1], endpoint, atol=1e-6)
    np.testing.assert_allclose(radii, radii[0], atol=1e-6)
    heights = target[:, :3, 3] @ world_up
    np.testing.assert_allclose(heights, heights[0], atol=1e-6)
    np.testing.assert_allclose(target[0, :3, 3], source[0, :3, 3], atol=1e-6)
    assert not np.allclose(target[-1, :3, 3], source[-1, :3, 3])
    for pose in target:
        forward = pose[:3, 2]
        to_pivot = fixed_pivot - pose[:3, 3]
        to_pivot /= np.linalg.norm(to_pivot)
        np.testing.assert_allclose(forward, to_pivot, atol=1e-6)
        np.testing.assert_allclose(pose[:3, 1] @ world_up, -1.0, atol=1e-6)

def test_moving_track_continuation_uses_constant_90_degree_offset():
    num_frames = 5
    source_seed = np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0)
    pivot = np.repeat(np.array([[0.0, 0.0, 4.0]], dtype=np.float32), num_frames, axis=0)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    source, _, _, _ = build_fixed_height_look_at_orbit_sequence(
        source_cam2world=source_seed,
        pivots_world=pivot,
        trajectory_name="orbit_left",
        orbit_angle_deg=90.0,
        world_up=world_up,
    )

    offsets = np.array(
        [
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tracks = np.empty((offsets.shape[0], num_frames, 3), dtype=np.float32)
    for frame_idx, scale in enumerate(np.linspace(0.2, 0.4, num_frames)):
        tracks[:, frame_idx] = pivot[frame_idx] + offsets * scale

    target, angles, radii, pivots, moving, inferred_up = (
        build_moving_track_continuation_orbit_sequence(
            source_cam2world=source,
            tracks_world=tracks,
            tracks_valid=np.ones(tracks.shape[:2], dtype=bool),
            trajectory_name="orbit_left",
            orbit_offset_deg=90.0,
            orbit_sweep_deg=0.0,
            motion_threshold=0.01,
            min_identities=8,
            temporal_smooth_radius=0,
        )
    )

    np.testing.assert_allclose(angles, 90.0, atol=1e-6)
    np.testing.assert_allclose(pivots, pivot, atol=1e-6)
    np.testing.assert_array_equal(moving, np.ones(8, dtype=bool))
    np.testing.assert_allclose(inferred_up, world_up, atol=1e-6)
    np.testing.assert_allclose(target[0], source[-1], atol=1e-5)
    np.testing.assert_allclose(radii, radii[0], atol=1e-6)



def test_fixed_height_orbit_radius_scale_moves_camera_outward():
    source = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    pivots = np.repeat(np.array([[0.0, 0.0, 4.0]], dtype=np.float32), 3, axis=0)
    target, _, radii, _ = build_fixed_height_look_at_orbit_sequence(
        source_cam2world=source,
        pivots_world=pivots,
        trajectory_name="orbit_left",
        orbit_angle_deg=90.0,
        world_up=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        radius_scale=1.15,
    )
    np.testing.assert_allclose(radii, 4.6, atol=1e-6)
    np.testing.assert_allclose(target[0, :3, 3], [0.0, 0.0, -0.6], atol=1e-6)
