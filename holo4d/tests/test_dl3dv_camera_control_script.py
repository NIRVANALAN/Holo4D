from pathlib import Path

import numpy as np

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.mixed_nvs_captions import (
    default_caption_dir_for_dataset,
)

from holo4d.benchmark.camera_utils import k4_from_k33, make_relative_c2w


def test_default_caption_dir_for_dl3dv():
    root = Path("/tmp/caption-root")
    assert default_caption_dir_for_dataset("dl3dv", caption_root=root) == root / "dl3dv"


def test_make_relative_c2w_anchors_to_first_pose():
    poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    poses[0, 0, 3] = 2.0
    poses[1, 0, 3] = 5.0
    poses[2, 1, 3] = 4.0

    rel = make_relative_c2w(poses)

    assert np.allclose(rel[0], np.eye(4, dtype=np.float32))
    assert np.isclose(rel[1, 0, 3], 3.0)
    assert np.isclose(rel[2, 0, 3], -2.0)
    assert np.isclose(rel[2, 1, 3], 4.0)


def test_k4_from_k33_extracts_fx_fy_cx_cy():
    K = np.array(
        [[100.0, 0.0, 40.0], [0.0, 120.0, 30.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    k4 = k4_from_k33(K)

    assert np.allclose(k4, np.array([100.0, 120.0, 40.0, 30.0], dtype=np.float32))
