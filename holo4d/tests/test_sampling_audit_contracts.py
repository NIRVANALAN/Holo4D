"""Exact checks requested by the release sampling audit; no model execution."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from holo4d.conditions.i2v_target_condition import plan_long_horizon_chunks, stitch_long_horizon_chunks
from holo4d.examples_io import file_sha256


@pytest.mark.parametrize('total', [1, 48, 49, 50, 97, 98, 145, 196, 214, 246])
def test_audit_exact_real_frame_plan(total):
    plan = plan_long_horizon_chunks(total, chunk_frames=49)
    frames = []
    for index, window in enumerate(plan):
        assert window.start == 48 * index
        assert window.append_from == (0 if index == 0 else 1)
        assert window.valid_frames == min(49, total-window.start)
        assert len(window.indices) == 49
        assert list(window.indices) == [min(window.start+j, total-1) for j in range(49)]
        frames.append(list(window.indices))
    assert stitch_long_horizon_chunks(frames, plan) == list(range(total))


def fake_model_args(options, entry, stage):
    return SimpleNamespace(checkpoint_path=options.checkpoint)


def unexpected_build(*args, **kwargs):
    raise AssertionError('Matching completed window should not build a GPU pipeline')


def fake_read_video(path):
    return [np.zeros((2, 2, 3), dtype=np.uint8)] * 49


def test_audit_resume_rejects_changed_previous_hash(tmp_path, monkeypatch):
    from holo4d.tools import i2v_continuation_long_video as video_io
    spec = importlib.util.spec_from_file_location('audit_robot_sampler', Path(__file__).resolve().parents[2]/'inference/sample_robot_rollouts.py')
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    monkeypatch.setattr(sampler, 'model_args', fake_model_args)
    monkeypatch.setattr(video_io, 'read_video', fake_read_video)
    monkeypatch.setitem(sys.modules, 'eval_track4d360_hybrid_dense_syn4d', SimpleNamespace(
        build_eval_pipeline=unexpected_build, _run_one_pair=unexpected_build,
        precompute_eval_prompt_embs=unexpected_build, NEGATIVE_PROMPT='negative'))
    checkpoint = tmp_path/'checkpoint.safetensors'
    checkpoint.write_bytes(b'test checkpoint, not model weights')
    previous = tmp_path/'figure/chunk_000/pred_rgb.mp4'
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b'generated target previous version')
    folder = tmp_path/'figure/chunk_001'
    folder.mkdir()
    (folder/'pred_rgb.mp4').write_bytes(b'existing generated target')
    entry = dict(id='figure', caption='robot moves', windows=[{},dict(condition_sha256='frozen-condition')])
    options = SimpleNamespace(output=tmp_path, checkpoint=str(checkpoint), steps=50)
    contract = dict(example='figure', chunk=1, condition_sha256='frozen-condition', checkpoint_sha256=file_sha256(checkpoint),
        backbone='continuation', steps=50, cfg=5, shift=5, seed=42, fps=16, mp4_quality=9,
        previous_generated_sha256=file_sha256(previous))
    (folder/'COMPLETE.json').write_text(json.dumps(contract))
    sampler.run_stage(options, entry, [1])
    entry['windows'][1]['prepared_to_rollout'] = dict(scale=1, rotation=np.eye(3).tolist(), translation=[0, 0, 0])
    with pytest.raises(ValueError, match='incomplete or incompatible'):
        sampler.run_stage(options, entry, [1])
    del entry['windows'][1]['prepared_to_rollout']
    previous.write_bytes(b'new generated target previous version')
    with pytest.raises(ValueError, match='incomplete or incompatible'):
        sampler.run_stage(options, entry, [1])
