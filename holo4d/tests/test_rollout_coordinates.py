import numpy as np
import pytest

from holo4d.rollout_coordinates import poses_in_rollout_gauge, target_boundary_report, validate_rollout_camera_metadata


def test_distinct_window_gauges_recover_identical_physical_boundary():
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    gauge = dict(scale=2., rotation=rotation, translation=np.array([1., 2., 3.]))
    local = np.eye(4)[None]
    physical = poses_in_rollout_gauge(local, gauge)
    assert not np.allclose(local, physical)
    assert target_boundary_report(physical, poses_in_rollout_gauge(local, gauge),
                                  np.eye(3), np.eye(3))['passed']
    # Camera orientation must remain a rotation: Sim(3) scale applies only to t.
    np.testing.assert_allclose(np.linalg.det(physical[0, :3, :3]), 1.)


def test_real_camera_or_intrinsics_jump_is_rejected():
    old = np.eye(4)[None]
    new = old.copy()
    new[0, 0, 3] = .01
    with pytest.raises(ValueError, match='discontinuity'):
        target_boundary_report(old, new, np.eye(3), np.eye(3))
    with pytest.raises(ValueError, match='discontinuity'):
        target_boundary_report(old, old, np.eye(3), 2*np.eye(3))


@pytest.mark.parametrize('scale', [0, -1, float('nan')])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match='similarity'):
        poses_in_rollout_gauge(np.eye(4)[None],
                              dict(scale=scale, rotation=np.eye(3), translation=np.zeros(3)))


def test_legacy_bundle_is_not_mislabelled_continuity_pass(tmp_path):
    report = validate_rollout_camera_metadata(tmp_path, dict(windows=[{}, {}]))
    assert report['status'] == 'legacy_window_local_gauges_not_certified'


def test_partial_metadata_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Partial'):
        validate_rollout_camera_metadata(tmp_path, dict(windows=[dict(prepared_to_rollout={}), {}]))


def test_bundle_preflight_checks_actual_arrays_without_rewriting(tmp_path):
    poses = np.repeat(np.eye(4)[None], 49, axis=0)
    k = np.eye(3)
    np.savez(tmp_path/'a.npz', target_cam2world=poses, target_intrinsics=k)
    np.savez(tmp_path/'b.npz', target_cam2world=poses, target_intrinsics=np.repeat(k[None], 49, axis=0))
    gauge = dict(scale=1., rotation=np.eye(3).tolist(), translation=[0., 0., 0.])
    entry = dict(windows=[dict(condition='a.npz', chunk_index=0, prepared_to_rollout=gauge),
                          dict(condition='b.npz', chunk_index=1, prepared_to_rollout=gauge)])
    assert validate_rollout_camera_metadata(tmp_path, entry)['status'] == 'passed'
    poses[0, 0, 3] = .1
    np.savez(tmp_path/'b.npz', target_cam2world=poses, target_intrinsics=k)
    with pytest.raises(ValueError, match='discontinuity'):
        validate_rollout_camera_metadata(tmp_path, entry)
