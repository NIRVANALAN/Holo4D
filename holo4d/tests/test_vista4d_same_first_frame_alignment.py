from pathlib import Path

import numpy as np

from holo4d.benchmark.vista4d_trajectories import (
    anchor_trajectory_to_source_frame0,
)


def _intrinsics(focals):
    out = np.tile(np.eye(3, dtype=np.float32)[None], (len(focals), 1, 1))
    out[:, 0, 0] = focals
    out[:, 1, 1] = focals
    out[:, 0, 2] = 336.0
    out[:, 1, 2] = 192.0
    return out


def test_same_first_frame_uses_focal_ratio_and_keeps_positive_focal():
    target = np.tile(np.eye(4, dtype=np.float32)[None], (3, 1, 1))
    target[:, 0, 3] = [0.0, 0.5, 1.0]
    target_K = _intrinsics(np.array([713.0, 535.0, 357.0], dtype=np.float32))
    source_K = _intrinsics(np.array([748.0], dtype=np.float32))[0]

    aligned_c2w, aligned_K = anchor_trajectory_to_source_frame0(
        target_c2w=target,
        target_intrinsics=target_K,
        source_frame0_c2w=np.eye(4, dtype=np.float32),
        source_intrinsics=source_K,
        translation_scale=0.4,
    )

    assert np.allclose(aligned_c2w[:, 0, 3], [0.0, 0.2, 0.4])
    assert np.isclose(aligned_K[0, 0, 0], 748.0)
    assert np.isclose(aligned_K[-1, 0, 0], 748.0 * 357.0 / 713.0)
    assert np.all(aligned_K[:, 0, 0] > 0.0)
    assert np.all(aligned_K[:, 1, 1] > 0.0)


def test_scale_is_applied_to_relative_not_absolute_camera_centers():
    target = np.tile(np.eye(4, dtype=np.float32)[None], (2, 1, 1))
    target[:, 0, 3] = [10.0, 12.0]
    target_K = _intrinsics(np.array([500.0, 500.0], dtype=np.float32))
    source = np.eye(4, dtype=np.float32)
    source[1, 3] = 3.0

    aligned_c2w, _ = anchor_trajectory_to_source_frame0(
        target_c2w=target,
        target_intrinsics=target_K,
        source_frame0_c2w=source,
        source_intrinsics=target_K[0],
        translation_scale=0.25,
    )

    assert np.allclose(aligned_c2w[0], source)
    assert np.isclose(aligned_c2w[1, 0, 3], 0.5)
    assert np.isclose(aligned_c2w[1, 1, 3], 3.0)


def test_vista4d_driver_defaults_to_depth_scale_and_fsff_for_shared_frame():
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/wan2.1_fun/eval_track4d360_vista4d_official.py"
    ).read_text()
    assert 'vista4d_target_scale_align: str = "depth"' in source
    assert 'prefer_same_first_frame=use_same_first_frame' in source
    assert '"cam_c2w_fsff"' in source
    assert '"intrinsics_fsff"' in source
    assert 'translation_scale=scale_pi3_to_4rc' in source
    assert 'min_fx <= 0.0' in source
