"""CPU-only checks for raw-video staging helpers and overwrite safety."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "inference" / "preprocess_raw_video.py"
SPEC = importlib.util.spec_from_file_location("holo4d_raw_preprocess", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BUILDER_SCRIPT = SCRIPT.with_name("build_raw_condition.py")
BUILDER_SPEC = importlib.util.spec_from_file_location("holo4d_raw_condition_builder", BUILDER_SCRIPT)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


def test_sequence_name_is_safe_and_stable():
    assert MODULE.safe_name("robot / demo: 01") == "robot_demo_01"
    with pytest.raises(ValueError, match="no safe"):
        MODULE.safe_name("///")


def test_env_python_command_uses_conda_run_without_absolute_python():
    command = MODULE.python_command(Path("tool.py"), env_name="fourrc", conda_command="conda")
    assert command == ["conda", "run", "--no-capture-output", "-n", "fourrc", "python", "tool.py"]


def test_current_environment_command_uses_active_python():
    command = MODULE.python_command(Path("tool.py"), env_name=None, conda_command="conda")
    assert command[0] == sys.executable
    assert command[-1] == "tool.py"


def test_backend_trajectory_distances_are_explicitly_not_metric_matched():
    assert BUILDER.trajectory_scale_contract('cow_vggt')['cow_vggt_scale_align'] == 'none'
    assert BUILDER.trajectory_scale_contract('4rc')['cross_backend_distance_comparable'] is False


def test_portrait_crop_is_warned_before_staging(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE.subprocess, 'run', lambda *a, **k: type(
        'Probe', (), {'stdout': json.dumps({'streams': [{'width': 384, 'height': 672}]})})())
    with pytest.warns(RuntimeWarning, match='portrait robot'):
        MODULE.warn_aspect_crop(tmp_path/'robot.mp4', 672, 384)


@pytest.mark.parametrize("width,height", [(672, 384), (384, 672), (512, 512)])
def test_cowtracker_canvas_is_patch14_compatible_for_both_aspects(width, height):
    # 320x560 fails DINO patch-14; 322x560 passes DINO but fails ResNet's
    # stride-8 pyramid. The shared compatible grid is lcm(14,8)=56.
    out_h, out_w = MODULE.cowtracker_canvas(width, height)
    assert out_h % 14 == out_w % 14 == 0
    assert out_h % 8 == out_w % 8 == 0
    assert max(out_h, out_w) == 560
    assert abs((out_w / out_h) / (width / height) - 1) < 0.06


def test_frame_reader_requires_exact_49_and_rgb(tmp_path: Path):
    for index in range(49):
        frame = np.full((6, 8, 3), index, dtype=np.uint8)
        Image.fromarray(frame).save(tmp_path / f"frame_{index:05d}.png")
    video = BUILDER.load_video_frames(tmp_path, expected_count=49)
    assert video.shape == (49, 6, 8, 3)
    assert video.dtype == np.uint8
    np.testing.assert_array_equal(video[0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="expected 50"):
        BUILDER.load_video_frames(tmp_path, expected_count=50)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for the staging integration test")
def test_ffmpeg_staging_produces_exact_49_frame_canvas(tmp_path: Path):
    source = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc=size=160x90:rate=16", "-t", "3.1", "-pix_fmt", "yuv420p",
            str(source), "-y",
        ],
        check=True,
    )
    output = tmp_path / "frames"
    assert MODULE.stage_video(source, output, width=96, height=64, fps=16) == 49
    frames = BUILDER.load_video_frames(output, expected_count=49)
    assert frames.shape == (49, 64, 96, 3)
    assert frames.dtype == np.uint8
    portrait_output = tmp_path / "portrait_frames"
    assert MODULE.stage_video(source, portrait_output, width=64, height=96, fps=16) == 49
    assert BUILDER.load_video_frames(portrait_output, expected_count=49).shape == (49, 96, 64, 3)


def test_orchestrator_rejects_unknown_trajectory(tmp_path: Path, monkeypatch):
    input_video = tmp_path / "source.mp4"
    input_video.write_bytes(b"not used because argparse rejects the trajectory first")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--video", str(input_video),
            "--caption", "A test caption.",
            "--output-dir", str(tmp_path / "run"),
            "--trajectory", "not-a-camera-path",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        MODULE.main()
    assert exc.value.code == 2


def test_orchestrator_writes_command_provenance_and_prepares_sampling(tmp_path: Path, monkeypatch):
    input_video = tmp_path / "source.mp4"
    input_video.write_bytes(b"unit-test-input-not-a-video")
    fourrc_repo = tmp_path / "4RC"
    fourrc_repo.mkdir()
    checkpoint = tmp_path / "holo.safetensors"
    checkpoint.write_bytes(b"unit-test-checkpoint")
    output = tmp_path / "run"
    captured = []
    captured_envs = []

    def fake_stage_video(video, frame_dir, *, width, height, fps, log_path=None, command_manifest=None):
        frame_dir.mkdir(parents=True)
        for index in range(49):
            Image.new("RGB", (width, height), (index, 0, 0)).save(frame_dir / f"frame_{index:05d}.png")
        if command_manifest is not None:
            command_manifest.append({"label": "mock ffmpeg staging", "argv": ["ffmpeg", "..."], "cwd": None})
        return 49

    def fake_run_step(label, command, *, cwd=None, env=None, log_path=None, command_manifest=None):
        record = {"label": label, "argv": command, "cwd": str(cwd) if cwd else None}
        captured.append(record)
        captured_envs.append(dict(env or {}))
        if command_manifest is not None:
            command_manifest.append(record)

    monkeypatch.setattr(MODULE, "stage_video", fake_stage_video)
    monkeypatch.setattr(MODULE, "run_step", fake_run_step)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-uuid")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--video", str(input_video),
            "--caption", "A test caption.",
            "--output-dir", str(output),
            "--sequence-name", "test-sequence",
            "--backend", "4rc",
            "--gpu", "1",
            "--fourrc-repo", str(fourrc_repo),
            "--fourrc-env", "test-4rc",
            "--holo-env", "test-holo",
            "--sample",
            "--checkpoint", str(checkpoint),
        ],
    )
    MODULE.main()

    run_record = json.loads((output / "run.json").read_text())
    assert run_record["cuda_visible_devices"] == "GPU-test-uuid"
    assert run_record["input_video_sha256"] == MODULE.sha256(input_video)
    assert run_record["commands"]
    assert run_record["backend_repositories"]["fourrc"]["path"] == str(fourrc_repo)
    assert run_record["checkpoints"]["holo4d"]["sha256"] == MODULE.sha256(checkpoint)
    assert run_record["outputs"]["sampling"] == str(output / "sampling")
    assert captured_envs and all(item["CUDA_VISIBLE_DEVICES"] == "GPU-test-uuid" for item in captured_envs)
    assert all(item["DISABLE_FLASH_ATTN_3"] == "1" for item in captured_envs)
    sample_command = next(item["argv"] for item in run_record["commands"] if item["label"].startswith("Holo4D diffusion"))
    assert any("inference/sample_examples.py" in part for part in sample_command)
    assert "--manifest" in sample_command
    assert any(item["label"].startswith("4RC direct") for item in captured)
    fourrc_command = next(item['argv'] for item in captured if item['label'].startswith('4RC direct'))
    assert fourrc_command[fourrc_command.index('--device') + 1] == 'cuda:1'


def test_reconstruction_pair_is_atomic_and_serializes_numeric_profiling(tmp_path):
    path = SCRIPT.with_name('run_4rc_reconstruction.py')
    spec = importlib.util.spec_from_file_location('fourrc_atomic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path/'completed'
    module.save_reconstruction_outputs(target, {'x': np.arange(3)},
                                        {'profiling': {'elapsed': np.float32(1.5), 'shape': np.array([1, 2])}})
    assert json.loads((target/'metadata.json').read_text())['profiling']['shape'] == [1, 2]
    with np.load(target/'result.npz') as values:
        np.testing.assert_array_equal(values['x'], np.arange(3))
    with pytest.raises(FileExistsError):
        module.save_reconstruction_outputs(target, {}, {})
    rejected = tmp_path/'rejected'
    with pytest.raises(TypeError):
        module.save_reconstruction_outputs(rejected, {}, {('bad', 'key'): 1})
    assert not rejected.exists()
