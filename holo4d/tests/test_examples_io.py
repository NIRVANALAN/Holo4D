import json
from pathlib import Path

import numpy as np
import pytest

from holo4d.examples_io import FORMAT, condition_to_sample, file_sha256, load_example, resolve_member, validate_condition


def payload():
    poses = np.broadcast_to(np.eye(4), (49, 4, 4)).copy()
    intrinsics = np.broadcast_to(np.eye(3), (49, 3, 3)).copy()
    intrinsics[:, 0, 0] = np.linspace(10, 20, 49)
    return dict(source_video=np.zeros((49, 16, 16, 3), dtype=np.uint8),
                tracks_world=np.ones((256, 49, 3), dtype=np.float32),
                tracks_world_valid_mask=np.ones((256, 49), dtype=bool),
                source_cam2world=poses, target_cam2world=poses.copy(),
                source_intrinsics=intrinsics, target_intrinsics=intrinsics.copy(),
                source_depth_video=np.ones((49, 16, 16), dtype=np.float32),
                source_rgb_at_tracks=np.ones((256, 49, 3), dtype=np.float32))


def test_dense_count_and_every_frame_K_are_preserved():
    data = payload()
    assert validate_condition(data)["identities"] == 256
    sample = condition_to_sample(data, {"caption": "A moving robot.", "trajectory": "zoom"})
    assert np.array_equal(sample["target_intrinsics_per_frame"], data["target_intrinsics"])
    assert sample["target_intrinsics_per_frame"][0, 0, 0] != sample["target_intrinsics_per_frame"][-1, 0, 0]
    assert sample["target_video"] is None and sample["inference_target_video"] is None


def test_paths_cannot_escape_pack(tmp_path: Path):
    with pytest.raises(ValueError, match="escapes"):
        resolve_member(tmp_path, "../private.npz")


def test_load_checks_caption_hash_and_pickle_free_arrays(tmp_path: Path):
    condition = tmp_path / "condition.npz"
    np.savez_compressed(condition, **payload())
    entry = dict(id="robot", caption="A moving robot.", condition="condition.npz", condition_sha256=file_sha256(condition))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(format=FORMAT, examples=[entry])))
    assert load_example(manifest, "robot")[2]["frames"] == 49
    entry["caption"] = ""
    manifest.write_text(json.dumps(dict(format=FORMAT, examples=[entry])))
    with pytest.raises(ValueError, match="caption"):
        load_example(manifest, "robot")


def test_invalid_tracks_rejected():
    data = payload()
    data["tracks_world"] = data["tracks_world"][:, :48]
    with pytest.raises(ValueError, match="N,49,3"):
        validate_condition(data)


def test_resize_preserves_world_gauge_and_all_intrinsics():
    from holo4d.examples_io import resize_condition

    video = np.zeros((49, 4, 8, 3), np.uint8)
    tracks = np.ones((12, 49, 3), np.float32)
    cameras = np.tile(np.eye(4, dtype=np.float32), (49, 1, 1))
    k = np.tile(np.eye(3, dtype=np.float32), (49, 1, 1))
    k[:, 0, 0] = np.linspace(1, 2, 49)
    data = dict(source_video=video, tracks_world=tracks,
                tracks_world_valid_mask=np.ones((12, 49), bool),
                source_cam2world=cameras, target_cam2world=cameras,
                source_intrinsics=k, target_intrinsics=k,
                source_depth_video=np.ones((49, 4, 8), np.float32),
                reproj_src_depth_at_target=np.ones((12, 49), np.float32),
                reproj_src_depth_valid=np.ones((12, 49), bool),
                source_rgb_at_tracks=np.ones((12, 3), np.float32))
    result = resize_condition(data, 8, 16)
    assert result["source_video"].shape == (49, 8, 16, 3)
    assert np.array_equal(result["tracks_world"], tracks)
    assert np.array_equal(result["target_cam2world"], cameras)
    assert np.allclose(result["target_intrinsics"][:, 0, 0], 2 * k[:, 0, 0])
    assert np.allclose(result["source_depth_video"], 1)
    assert np.array_equal(result["reproj_src_depth_at_target"], data["reproj_src_depth_at_target"])
