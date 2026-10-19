"""Regression coverage for compact Syn4D tracking_safetensors indexes."""
from __future__ import annotations

import os
import types

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.syn4d_nvs import Syn4D_Multi_NVS, compact_tracking_if_available


def test_older_syn4d_loader_without_compact_attributes_is_supported():
    assert compact_tracking_if_available(types.SimpleNamespace(), 'scene', 'camera') is None
    assert compact_tracking_if_available(types.SimpleNamespace(tracking_format='auto'), 'scene', 'camera') is None


def test_compact_tracking_index_keeps_trackable_nvs_start_without_npz_dir(tmp_path):
    """The NVS wrapper must accept the same compact tracking format as its base loader."""
    samples = []
    for camera in ("arena_seq_000000_0", "arena_seq_000000_1"):
        view_dir = tmp_path / "arena" / "mp4" / camera
        view_dir.mkdir(parents=True)
        samples.append(
            {
                "scene_name": "arena",
                "camera_seq_name": camera,
                "frame_indices_abs": [0],
                "frame_names": [f"{camera}_0000.png"],
                "view_dir": str(view_dir),
            }
        )

    class FakeSyn4D:
        tracking_format = "safetensor"

        def __init__(self):
            self.samples = samples

        def _get_tracking_safetensor(self, scene, camera):
            assert (scene, camera) == ("arena", "arena_seq_000000_0")
            return {
                "index": {
                    "record_key_to_id": {
                        "arena_seq_000000_0_0000/000000": 0,
                    }
                }
            }

        @staticmethod
        def _list_frame_names_from_camera_csv(_path):
            return [
                "arena_seq_000000_0_0000.png",
                "arena_seq_000000_0_0001.png",
            ]

    dataset = Syn4D_Multi_NVS.__new__(Syn4D_Multi_NVS)
    dataset.ROOT = str(tmp_path)
    dataset._syn4d = FakeSyn4D()
    dataset.scene_filter = None
    dataset.exclude_scenes = set()
    dataset._valid_seq_set = None
    dataset.sequence_idx = None

    groups = dataset._build_nvs_groups()

    assert len(groups) == 1
    assert groups[0]["scene_name"] == "arena"
    assert groups[0]["camera_names"] == [
        "arena_seq_000000_0",
        "arena_seq_000000_1",
    ]
    assert groups[0]["common_starts"] == [0]
    assert not os.path.exists(
        tmp_path / "arena" / "npz" / "arena_seq_000000_0" / "arena_seq_000000_0_0000"
    )
