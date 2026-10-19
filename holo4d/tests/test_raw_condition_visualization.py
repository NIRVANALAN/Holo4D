"""Small CPU regression tests for raw-condition camera/track rendering."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "inference" / "visualize_raw_condition.py"
SPEC = importlib.util.spec_from_file_location("visualize_raw_condition", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
viz = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viz)


def test_factor_grid_uses_image_aspect():
    assert viz.factor_grid(24, 0.5) == (3, 8)


def test_dense_pointmap_reprojection_is_identity_for_same_camera():
    height, width = 4, 6
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack((xx * 20, yy * 30, np.full_like(xx, 80)), axis=-1).astype(np.uint8)
    depth = np.full((height, width), 2.0, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    identity = np.eye(4, dtype=np.float32)
    K = np.array([[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)

    warped_rgb, warped_depth, warped_valid = viz._warp_source_to_target(
        rgb, depth, valid, identity, identity, K
    )

    np.testing.assert_array_equal(warped_rgb, rgb)
    np.testing.assert_allclose(warped_depth, depth, atol=1e-6)
    assert warped_valid.all()


def test_trail_renderer_handles_nan_and_empty_history():
    frame = np.full((8, 8, 3), 200, dtype=np.uint8)
    uv = np.full((2, 1, 2), np.nan, dtype=np.float32)
    visible = np.zeros((2, 1), dtype=bool)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    rendered = viz._draw_trails(frame, uv, visible, colors)
    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
    assert np.all(rendered == 80)


def test_reprojection_uses_different_source_and_target_focal_lengths():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[1, 2] = [20, 80, 140]
    depth = np.ones((4, 6), dtype=np.float32)
    valid = np.zeros((4, 6), dtype=bool)
    valid[1, 2] = True
    source_K = np.diag([2.0, 2.0, 1.0]).astype(np.float32)
    target_K = np.diag([4.0, 4.0, 1.0]).astype(np.float32)
    identity = np.eye(4, dtype=np.float32)
    projected, _, mask = viz._warp_source_to_target(
        rgb, depth, valid, identity, identity, target_K, source_K=source_K)
    assert mask[2, 4]
    assert not mask[1, 2]
    np.testing.assert_array_equal(projected[2, 4], rgb[1, 2])


def test_full_condition_renderer_accepts_per_frame_intrinsics(tmp_path, monkeypatch):
    from unittest.mock import Mock

    frames, height, width = 49, 4, 6
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([xx / 2, yy / 2, np.ones_like(xx)], axis=-1).reshape(-1, 3)
    cameras = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    source_K = np.repeat(np.diag([2, 2, 1]).astype(np.float32)[None], frames, axis=0)
    target_K = source_K.copy()
    target_K[:, 0, 0] = np.linspace(2, 4, frames)
    condition = tmp_path / "condition.npz"
    np.savez(condition, source_video=np.zeros((frames, height, width, 3), dtype=np.uint8),
             source_depth_video=np.ones((frames, height, width), dtype=np.float32),
             tracks_world=np.repeat(points[:, None, :], frames, axis=1).astype(np.float32),
             tracks_world_valid_mask=np.ones((height * width, frames), dtype=bool),
             source_cam2world=cameras, target_cam2world=cameras,
             source_intrinsics=source_K, target_intrinsics=target_K,
             source_backend=np.asarray("cow_vggt"))
    writer = Mock()
    monkeypatch.setattr(viz, "_write", writer)
    result = viz.render(condition, tmp_path / "review", None, 4, 16)
    assert result["frames"] == 49
    assert Path(result["backend_condition_review"]).name == "backend_cow_vggt_condition_review_2x3.mp4"
    assert len(writer.call_args.args[1]) == 49
