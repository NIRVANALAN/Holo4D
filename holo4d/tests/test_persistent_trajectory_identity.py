import os
from unittest.mock import patch

import torch

from holo4d.conditions.dense_target_control_unit import (
    _align_track_to_sequence,
)
from holo4d.conditions.persistent_trajectory_identity import (
    PersistentTrajectoryIdentityAdapter,
    build_persistent_trajectory_inputs,
    compose_source_identity_tokens,
    compose_target_identity_tokens,
    normalize_persistent_identity_mode,
    persistent_identity_only_enabled,
)


def _example_inputs():
    tracks = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
                [[1.0, 1.0, 2.0], [1.0, 1.1, 2.0], [1.0, 1.2, 2.0]],
            ]
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor([[[True, True, True], [True, True, True]]])
    uv = torch.tensor(
        [[[[2.0, 2.0], [3.0, 2.0], [4.0, 2.0]],
          [[12.0, 12.0], [11.0, 12.0], [10.0, 12.0]]]],
        dtype=torch.float32,
    )
    return tracks, valid, uv


def _open_output_branch(module):
    torch.manual_seed(7)
    torch.nn.init.normal_(module.point_mlp[-1].weight, std=0.02)
    torch.nn.init.normal_(module.point_mlp[-1].bias, std=0.02)


def test_mode_aliases_and_invalid_mode():
    assert normalize_persistent_identity_mode(None) == "legacy"
    assert normalize_persistent_identity_mode("id-target") == "target"
    assert normalize_persistent_identity_mode("source_target") == "shared"
    try:
        normalize_persistent_identity_mode("ambiguous")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid persistent identity mode must fail loudly")


def test_identity_only_requires_nonlegacy_identity_mode():
    with patch.dict(os.environ, {"HOLO4D_TRACK_IDENTITY_ONLY": "1"}):
        assert persistent_identity_only_enabled("shared")
        try:
            persistent_identity_only_enabled("legacy")
        except ValueError as exc:
            assert "requires persistent_identity_mode" in str(exc)
        else:
            raise AssertionError("identity-only with legacy mode must fail loudly")


def test_identity_only_replaces_field_on_target_and_source():
    base = torch.tensor([[[1.0, 2.0]]])
    identity = torch.tensor([[[3.0, 4.0]]])
    assert torch.equal(
        compose_target_identity_tokens(base, identity, identity_only=False),
        torch.tensor([[[4.0, 6.0]]]),
    )
    assert torch.equal(
        compose_target_identity_tokens(base, identity, identity_only=True),
        identity,
    )
    assert torch.equal(
        compose_source_identity_tokens(base, identity, identity_only=False),
        torch.tensor([[[4.0, 6.0]]]),
    )
    assert torch.equal(
        compose_source_identity_tokens(base, identity, identity_only=True),
        identity,
    )


def test_new_identity_branch_is_zero_init_noop():
    tracks, valid, uv = _example_inputs()
    sequence, identity_valid = build_persistent_trajectory_inputs(tracks, valid)
    module = PersistentTrajectoryIdentityAdapter(
        out_dim=12, identity_dim=8, hidden_dim=16
    )
    target, mask, source, source_mask = module(
        sequence,
        identity_valid,
        uv,
        valid,
        image_h=16,
        image_w=16,
        latent_t=2,
        latent_h=2,
        latent_w=2,
    )
    assert mask.any()
    assert torch.equal(target, torch.zeros_like(target))
    assert source is None
    assert source_mask is None


def test_shared_source_target_reuses_identical_identity_path():
    tracks, valid, uv = _example_inputs()
    sequence, identity_valid = build_persistent_trajectory_inputs(tracks, valid)
    module = PersistentTrajectoryIdentityAdapter(
        out_dim=12, identity_dim=8, hidden_dim=16
    )
    _open_output_branch(module)
    target, target_mask, source, source_mask = module(
        sequence,
        identity_valid,
        uv,
        valid,
        image_h=16,
        image_w=16,
        latent_t=2,
        latent_h=2,
        latent_w=2,
        source_uv=uv.clone(),
        source_vis=valid.clone(),
    )
    assert torch.equal(target_mask, source_mask)
    assert torch.equal(target, source)


def test_empty_tokens_are_hard_zero_and_row_order_invariant():
    tracks, valid, uv = _example_inputs()
    sequence, identity_valid = build_persistent_trajectory_inputs(tracks, valid)
    module = PersistentTrajectoryIdentityAdapter(
        out_dim=12, identity_dim=8, hidden_dim=16
    )
    _open_output_branch(module)
    target, mask, _, _ = module(
        sequence,
        identity_valid,
        uv,
        valid,
        image_h=16,
        image_w=16,
        latent_t=2,
        latent_h=2,
        latent_w=2,
    )
    assert torch.equal(target[~mask], torch.zeros_like(target[~mask]))

    order = torch.tensor([1, 0])
    reordered, reordered_mask, _, _ = module(
        sequence[:, order],
        identity_valid[:, order],
        uv[:, order],
        valid[:, order],
        image_h=16,
        image_w=16,
        latent_t=2,
        latent_h=2,
        latent_w=2,
    )
    assert torch.equal(mask, reordered_mask)
    assert torch.allclose(target, reordered, atol=0.0, rtol=0.0)


def test_identity_sequence_ignores_invalid_world_rows():
    tracks, valid, _ = _example_inputs()
    tracks[:, 1] = 1e6
    valid[:, 1] = False
    sequence, identity_valid = build_persistent_trajectory_inputs(tracks, valid)
    assert not identity_valid[:, 1].any()
    assert torch.equal(sequence[:, 1, :, 1:], torch.zeros_like(sequence[:, 1, :, 1:]))
    assert sequence[:, 0, :, 1:].min() >= 0
    assert sequence[:, 0, :, 1:].max() <= 1


def test_legacy_alignment_is_unchanged_and_shared_is_explicit():
    class Adapter:
        null_token = torch.tensor([9.0, 10.0])

    adapter = Adapter()
    target = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    source = torch.full_like(target, 3.0)
    x = torch.zeros((1, 8, 4))

    legacy = _align_track_to_sequence(
        target,
        adapter,
        x,
        f=1,
        h=2,
        w=4,
        num_domains=2,
        v2v_concat_mode="width",
    )
    shared = _align_track_to_sequence(
        target,
        adapter,
        x,
        f=1,
        h=2,
        w=4,
        num_domains=2,
        v2v_concat_mode="width",
        source_tpt=source,
    )
    legacy_spatial = legacy.reshape(1, 1, 2, 4, 2)
    shared_spatial = shared.reshape(1, 1, 2, 4, 2)
    assert torch.equal(
        legacy_spatial[:, :, :, :2],
        adapter.null_token.view(1, 1, 1, 1, 2).expand(1, 1, 2, 2, 2),
    )
    assert torch.equal(shared_spatial[:, :, :, :2], source.reshape(1, 1, 2, 2, 2))
    assert torch.equal(legacy_spatial[:, :, :, 2:], target.reshape(1, 1, 2, 2, 2))
    assert torch.equal(shared_spatial[:, :, :, 2:], target.reshape(1, 1, 2, 2, 2))


def test_zero_init_shared_delta_preserves_legacy_null_alignment():
    class Adapter:
        null_token = torch.tensor([9.0, 10.0])

    adapter = Adapter()
    target = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    identity_delta = torch.zeros_like(target)
    legacy_source = adapter.null_token.view(1, 1, 2).expand_as(identity_delta)
    shared_source = legacy_source + identity_delta
    x = torch.zeros((1, 8, 4))
    legacy = _align_track_to_sequence(
        target, adapter, x, 1, 2, 4, 2, "width"
    )
    shared_zero_init = _align_track_to_sequence(
        target,
        adapter,
        x,
        1,
        2,
        4,
        2,
        "width",
        source_tpt=shared_source,
    )
    assert torch.equal(legacy, shared_zero_init)


def test_zero_init_branch_opens_then_trains_identity_encoder():
    tracks, valid, uv = _example_inputs()
    sequence, identity_valid = build_persistent_trajectory_inputs(tracks, valid)
    module = PersistentTrajectoryIdentityAdapter(
        out_dim=12, identity_dim=8, hidden_dim=16
    )
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    target, _, _, _ = module(
        sequence,
        identity_valid,
        uv,
        valid,
        image_h=16,
        image_w=16,
        latent_t=2,
        latent_h=2,
        latent_w=2,
    )
    target.sum().backward()
    assert module.point_mlp[-1].weight.grad.abs().sum() > 0
    first_identity_grad = module.identity_mlp[0].weight.grad
    assert first_identity_grad is None or torch.equal(
        first_identity_grad, torch.zeros_like(first_identity_grad)
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    target, _, _, _ = module(
        sequence,
        identity_valid,
        uv,
        valid,
        image_h=16,
        image_w=16,
        latent_t=2,
        latent_h=2,
        latent_w=2,
    )
    target.square().sum().backward()
    assert module.identity_mlp[0].weight.grad.abs().sum() > 0


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)} persistent identity tests passed")
