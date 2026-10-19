import json

import numpy as np
import pytest

from holo4d.robot_rollouts import FORMAT, attach_generated_boundary, load_rollout


def test_generated_boundary_uses_last_target():
    frames = [np.full((4, 8, 3), i, dtype=np.uint8) for i in range(49)]
    sample = {'video': [np.zeros((4, 8, 3), dtype=np.uint8)]}
    attach_generated_boundary(sample, frames)
    assert sample['use_target_image_condition'] is True
    assert (np.asarray(sample['inference_input_image']) == 48).all()
    with pytest.raises(ValueError):
        attach_generated_boundary(sample, frames[:48])


def test_manifest_rejects_nonoverlap(tmp_path):
    entry = dict(id='tri_static_high', caption='robot', real_frames=81,
                 windows=[dict(chunk_index=0, start=0, valid_frames=49, condition='a.npz'),
                          dict(chunk_index=1, start=49, valid_frames=32, condition='b.npz')])
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(format=FORMAT, examples=[entry])))
    with pytest.raises(ValueError, match='overlap'):
        load_rollout(path, 'tri_static_high')


def test_manifest_accepts_real_tail_and_rejects_path_escape(tmp_path):
    entry = dict(id='figure0_orbit', caption='robot', real_frames=81,
                 windows=[dict(chunk_index=0, start=0, valid_frames=49, condition='a.npz'),
                          dict(chunk_index=1, start=48, valid_frames=33, condition='b.npz')])
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(format=FORMAT, examples=[entry])))
    assert load_rollout(path, 'figure0_orbit')[1]['windows'][1]['valid_frames'] == 33
    entry['windows'][1]['condition'] = '../escape.npz'
    path.write_text(json.dumps(dict(format=FORMAT, examples=[entry])))
    with pytest.raises(ValueError, match='escapes'):
        load_rollout(path, 'figure0_orbit')
