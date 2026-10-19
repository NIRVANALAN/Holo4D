import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import imageio.v2 as imageio
import numpy as np
import pytest

from holo4d.data.demo_rollout_bundle import DemoRolloutBundle, FORMAT
from holo4d.visualization.hybrid_panels import pil_list_to_np


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("entrypoint", ["build_dataset", "_build_i2v_continuation_dataset"])
def test_public_driver_routes_demo_npz_before_raw_syn4d_loader(monkeypatch, entrypoint):
    # Execute the real routing function without importing the GPU model stack.
    import holo4d.data.demo_rollout_bundle as bundles

    factory = Mock(return_value=object())
    monkeypatch.setattr(bundles, "DemoRolloutBundle", factory)
    monkeypatch.setenv("HOLO4D_DEMO_BUNDLE_PATH", "/portable/demo.npz")
    source = ROOT / "examples/wan2.1_fun/eval_track4d360_hybrid_dense_syn4d.py"
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entrypoint)
    scope = {"os": os, "Syn4D_Multi_NVS": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    args = SimpleNamespace(num_frames=49)
    positional = (args, "flying_group", 0) if entrypoint == "build_dataset" else (args,)
    assert scope[entrypoint](*positional) is factory.return_value
    factory.assert_called_once_with("/portable/demo.npz", sequence_idx=0, num_frames=49)


def _payload(prefix: str, query: int):
    t, h, w, n = 49, 2, 3, 6
    return {
        f"{prefix}_source_video": np.zeros((t, h, w, 3), dtype=np.uint8),
        f"{prefix}_target_video": np.zeros((t, h, w, 3), dtype=np.uint8),
        f"{prefix}_tracks_world": np.zeros((n, t, 3), dtype=np.float32),
        f"{prefix}_target_cam2world": np.repeat(
            np.eye(4, dtype=np.float32)[None], t, axis=0
        ),
        f"{prefix}_target_intrinsics": np.eye(3, dtype=np.float32),
        f"{prefix}_common_scale": np.asarray(1.0, dtype=np.float32),
        f"{prefix}_prompt": np.asarray("demo prompt"),
        f"{prefix}_target_prompt": np.asarray("demo target prompt"),
        f"{prefix}_reproj_src_depth_at_target": np.ones((n, t), dtype=np.float32),
        f"{prefix}_reproj_src_depth_valid": np.ones((n, t), dtype=bool),
        f"{prefix}_source_rgb_at_tracks": np.zeros((n, t, 3), dtype=np.float32),
        f"{prefix}_source_cam2world": np.repeat(
            np.eye(4, dtype=np.float32)[None], t, axis=0
        ),
        f"{prefix}_source_intrinsics": np.eye(3, dtype=np.float32),
        f"{prefix}_source_depth_video": np.ones((t, h, w), dtype=np.float32),
    }


def test_demo_bundle_exposes_exact_two_window_contract(tmp_path: Path):
    path = tmp_path / "demo.npz"
    payload = {"format": np.asarray(FORMAT)}
    payload.update(_payload("seed", 0))
    payload.update(_payload("continuation", 2))
    np.savez_compressed(path, **payload)
    dataset = DemoRolloutBundle(path)
    seed = dataset.load_pair(0, 1, 5, 0, 0)
    continuation = dataset.load_pair(0, 1, 55, 0, 2, 53)
    assert len(seed["video"]) == len(continuation["video"]) == 49
    assert seed["tracks_world"].shape == continuation["tracks_world"].shape == (6, 49, 3)
    assert seed["track_query_idx"] == 0
    assert continuation["track_query_idx"] == 2
    assert dataset._nvs_groups[0]["common_starts"] == [5, 55]


def test_comparison_frames_accept_non_owning_numpy_arrays():
    backing = np.zeros((4, 6, 3), dtype=np.uint8)
    frame = backing[:, ::2]
    assert not frame.flags.owndata
    converted = pil_list_to_np([frame], width=5, height=4)
    assert converted[0].shape == (4, 5, 3)
    assert converted[0].flags.c_contiguous


def test_checked_demo_videos_match_manifest():
    root = ROOT / "assets" / "demo"
    manifest = json.loads((root / "manifest.json").read_text())
    assert [item["frames"] for item in manifest["outputs"]] == [49, 97]
    for item in manifest["outputs"]:
        path = root / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        reader = imageio.get_reader(path, output_params=["-threads", "1"])
        try:
            assert reader.count_frames() == item["frames"]
            frame = np.asarray(reader.get_data(0))
        finally:
            reader.close()
        assert frame.shape[:2] == (384, 672)
