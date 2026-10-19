from types import SimpleNamespace

import pytest

from holo4d.checkpoint_validation import validate_dense_checkpoint


class ShapeModule:
    def __init__(self, names, buffers=None):
        self.parameters = {name: SimpleNamespace(shape=(2, 2)) for name in names}
        self.buffers = buffers or {}

    def named_parameters(self):
        return self.parameters.items()

    def state_dict(self):
        return {**self.parameters, **self.buffers}


def fixture_pipe():
    dit = ShapeModule(['blocks.0.self_attn.q.weight', 'blocks.0.camera_projector.weight',
                       'domain_embedding', 'control_patch_embedding.weight', 'frozen.weight'])
    adapter = ShapeModule(['mlp.weight'], {'sin_enc.freqs': SimpleNamespace(shape=(4,))})
    pipe = SimpleNamespace(dit=dit, track_adapter=adapter)
    delta = {name: value for name, value in dit.parameters.items() if name != 'frozen.weight'}
    raw = {**delta, 'pipe.track_adapter.mlp.weight': adapter.parameters['mlp.weight']}
    return pipe, raw, delta, {'frozen.weight': dit.parameters['frozen.weight']}


def test_frozen_base_and_derived_buffer_are_not_required_in_delta():
    pipe, raw, delta, base = fixture_pipe()
    assert validate_dense_checkpoint(pipe, raw, delta, base)['passed']


@pytest.mark.parametrize('kind', ['camera', 'track', 'unknown', 'shape'])
def test_bad_condition_delta_fails_instead_of_silent_partial_load(kind):
    pipe, raw, delta, base = fixture_pipe()
    if kind == 'camera':
        del delta['blocks.0.camera_projector.weight']
    elif kind == 'track':
        del raw['pipe.track_adapter.mlp.weight']
    elif kind == 'unknown':
        raw['pipe.track_missing.weight'] = SimpleNamespace(shape=(2, 2))
    else:
        raw['pipe.track_adapter.mlp.weight'] = SimpleNamespace(shape=(3, 3))
    with pytest.raises(ValueError, match='coverage failed'):
        validate_dense_checkpoint(pipe, raw, delta, base)


def test_missing_frozen_base_weight_also_fails():
    pipe, raw, delta, base = fixture_pipe()
    with pytest.raises(ValueError, match='frozen.weight'):
        validate_dense_checkpoint(pipe, raw, delta, {})
