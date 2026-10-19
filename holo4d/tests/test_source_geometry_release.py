"""Released 4RC scale reference uses world-to-camera, not c2w twice."""
import numpy as np
import pytest

from holo4d.benchmark.cow_vggt_run import _load_4rc_query_depth_for_scale


@pytest.mark.parametrize('shape', ['4x4', '3x4', 'batched'])
def test_translated_rotated_camera_reference_depth(tmp_path, shape):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    c2w[:3, 3] = [4, 2, -3]
    camera = np.zeros((2, 3, 3), dtype=np.float32)
    camera[..., 2] = 6
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    extrinsic = c2w[:3] if shape == '3x4' else c2w
    if shape == 'batched':
        extrinsic = extrinsic[None]
    folder = tmp_path / 'fixture'
    folder.mkdir()
    np.savez(folder/'result.npz', pred_0_pts=world[None], pred_0_extrinsic=extrinsic)
    np.testing.assert_allclose(_load_4rc_query_depth_for_scale(str(tmp_path), 'fixture'), 6)


def test_condition_export_preserves_visibility_and_source_camera(tmp_path):
    from holo4d.novel_view.static_scene_novel_traj import save_static_scene_novel_view_payload
    sample = dict(target_cam2world=np.eye(4)[None], target_intrinsics=np.eye(3),
        target_depth=np.ones((1,2,2)), target_depth_valid=np.ones((1,2,2)),
        reproj_src_depth_at_target=np.ones((1,2,2)), reproj_src_depth_valid=np.ones((1,2,2)),
        tracks_world=np.ones((2,1,3)), tracks_world_valid_mask=np.array([[True],[False]]),
        source_cam2world=np.eye(4)[None], source_intrinsics=np.eye(3), tracks_source_xy=np.zeros((2,2)))
    path = save_static_scene_novel_view_payload(sample, tmp_path/'condition.npz')
    with np.load(path, allow_pickle=False) as data:
        for key in ['tracks_world_valid_mask','source_cam2world','source_intrinsics','tracks_source_xy']:
            np.testing.assert_array_equal(data[key], sample[key])
