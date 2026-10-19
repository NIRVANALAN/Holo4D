"""Tests for de-duplicated I2V continuation visualization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import imageio.v2 as imageio
import numpy as np

from holo4d.tools.i2v_continuation_long_video import build_case, parse_args, read_video


def test_rollout_defaults_match_16fps_prediction_exports(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["rollout", "--run-root", str(tmp_path)])
    assert parse_args().fps == 16
    launcher = Path(__file__).resolve().parents[2] / "inference/sample_mixed_t2v_i2v_syn4d.sh"
    assert '${OUTPUT_FPS:-16}' in launcher.read_text()


def test_rollout_writer_uses_shared_prediction_quality(monkeypatch, tmp_path):
    import holo4d.tools.i2v_continuation_long_video as rollout

    writer = Mock()
    factory = Mock(return_value=writer)
    monkeypatch.setattr(rollout.imageio, "get_writer", factory)
    assert rollout.write_video(tmp_path / "test.mp4", [np.zeros((8, 8, 3), np.uint8)], fps=16) == 1
    assert factory.call_args.kwargs["quality"] == 9
    assert factory.call_args.kwargs["codec"] == "libx264"
    writer.close.assert_called_once()


def write_video(path: Path, values: list[int], height: int = 32, width: int = 48) -> None:
    frames = [np.full((height, width, 3), value, dtype=np.uint8) for value in values]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=12, pixelformat="yuv420p")


def write_comparison(path: Path, values: list[int], height: int = 32, width: int = 48) -> None:
    frames = []
    for value in values:
        frame = np.zeros((2 * height + 16, 5 * width, 3), dtype=np.uint8)
        frame[height : 2 * height, 4 * width : 5 * width] = value
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=12, pixelformat="yuv420p")


def prepare_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case_000_seq_test_src7_tgt3_seed0015_next0018"
    for filename, seed_values, continuation_values in (
        ("source_rgb.mp4", [10, 20, 30, 40], [40, 50, 60, 70]),
        ("gt_rgb.mp4", [80, 90, 100, 110], [110, 120, 130, 140]),
        ("pred_rgb.mp4", [150, 160, 170, 180], [180, 190, 200, 210]),
    ):
        write_video(case_dir / "seed" / filename, seed_values)
        write_video(case_dir / "continuation" / filename, continuation_values)
    write_comparison(case_dir / "seed" / "comparison.mp4", [10, 20, 30, 40])
    write_comparison(
        case_dir / "continuation" / "comparison.mp4", [40, 50, 60, 70]
    )

    condition = np.full((32, 48, 3), 180, dtype=np.uint8)
    gt_boundary = np.full((32, 48, 3), 110, dtype=np.uint8)
    imageio.imwrite(case_dir / "seed" / "last_target.png", condition)
    imageio.imwrite(
        case_dir / "continuation" / "previous_generated_target_frame48.png",
        condition,
    )
    imageio.imwrite(
        case_dir / "continuation" / "previous_target_gt_frame48.png", gt_boundary
    )
    imageio.imwrite(
        case_dir / "continuation" / "current_target_gt_frame0.png", gt_boundary
    )
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "seed_start_frame": 15,
                "continuation_start_frame": 18,
                "src_cam_idx": 7,
                "tgt_cam_idx": 3,
            }
        )
    )
    return case_dir


def test_build_case_deduplicates_shared_boundary(tmp_path: Path) -> None:
    case_dir = prepare_case(tmp_path)
    result = build_case(
        case_dir,
        "long_rollout",
        fps=12,
        overwrite=False,
    )

    assert result["status"] == "written"
    output_dir = case_dir / "long_rollout"
    generated = read_video(output_dir / "target_generated_97f.mp4")
    comparison = read_video(output_dir / "rgb_depth_tracks_2x3_97f.mp4")
    manifest = json.loads((output_dir / "long_rollout.json").read_text())
    assert len(generated) == 7
    assert len(comparison) == 7
    assert comparison[0].shape[:2] == (64, 144)
    assert manifest["long_frames"] == 7
    assert manifest["shared_boundary_absolute_frame"] == 18
    assert manifest["shared_boundary_long_index"] == 3
    assert manifest["long_absolute_frames"] == [15, 21]


def test_build_case_skips_complete_existing_output(tmp_path: Path) -> None:
    case_dir = prepare_case(tmp_path)
    build_case(
        case_dir, "long_rollout", fps=12, overwrite=False
    )
    result = build_case(
        case_dir, "long_rollout", fps=12, overwrite=False
    )
    assert result["status"] == "skipped"


def test_multi_chunk_stitch_trims_partial_final_window(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_000_seq_test_src0_tgt1_seed0005_next0008"
    (case_dir / "case.json").parent.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        json.dumps({"seed_start_frame": 5, "src_cam_idx": 0, "tgt_cam_idx": 1})
    )
    chunks = [
        ("seed", [10, 20, 30, 40], 4, 0, 5),
        ("continuation", [40, 50, 60, 70], 4, 1, 8),
        ("continuation_2", [70, 80, 90, 100], 2, 1, 11),
    ]
    for index, (name, values, valid, first, absolute_start) in enumerate(chunks):
        chunk = case_dir / name
        for filename in ("source_rgb.mp4", "gt_rgb.mp4", "pred_rgb.mp4"):
            write_video(chunk / filename, values)
        write_comparison(chunk / "comparison.mp4", values)
        (chunk / "chunk_manifest.json").write_text(
            json.dumps(
                {
                    "chunk_index": index,
                    "absolute_start_frame": absolute_start,
                    "valid_frames": valid,
                    "append_from": first,
                }
            )
        )

    result = build_case(case_dir, "long_rollout", fps=12, overwrite=False)

    assert result["status"] == "written"
    output = case_dir / "long_rollout"
    generated = read_video(output / "target_generated_8f.mp4")
    comparison = read_video(output / "rgb_depth_tracks_2x3_8f.mp4")
    manifest = json.loads((output / "long_rollout.json").read_text())
    assert len(generated) == 8
    assert len(comparison) == 8
    assert manifest["chunk_valid_frames"] == [4, 4, 2]
    assert manifest["chunk_append_from"] == [0, 1, 1]
    assert manifest["long_frames"] == 8
