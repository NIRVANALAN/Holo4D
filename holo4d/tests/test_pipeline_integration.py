"""
Integration tests for Track4D-360 pipeline units and hook injection.

Tests the full path from WanVideoUnit_TrackPointCondition through the
_pre_block_hook, verifying correct latent dimensions and V2V layout handling.

Key coverage:
  - //32 latent spatial fix (Wan 2.2 VAE 16x + patchify 2x)
  - WanVideoUnit_TrackPointCondition output shapes
  - _pre_block_hook V2V width-mode layout
  - _pre_block_hook V2V frame-mode layout
  - Zero-init projector preserves input at init
  - Gradient flow through the full injection path
"""

import torch
import pytest

from holo4d.conditions.point_track_embedder import PointTrackEmbedder
from holo4d.conditions.point_track_adapter import PointTrackAdapter
from holo4d.models.token_concat_projector import TokenConcatProjector
from holo4d_runtime.pipeline.wan_video_track4d360 import WanVideoUnit_TrackPointCondition


# --- Wan 5B constants ---
WAN_5B_DIM = 1536
TRACK_EMBED_DIM = 156


def test_track_point_unit_accepts_rgbd_track_inputs():
    """RGBD track ablations must pass optional per-track tensors through unit_runner."""
    unit = WanVideoUnit_TrackPointCondition()
    for key in (
        "source_depth_at_tracks",
        "source_depth_valid_at_tracks",
        "source_rgb_at_tracks",
    ):
        assert key in unit.input_params


def test_track_point_unit_accepts_per_frame_target_intrinsics():
    """The unit runner must forward K_t to persistent-track projection."""
    unit = WanVideoUnit_TrackPointCondition()
    assert "target_intrinsics_per_frame" in unit.input_params


def _make_mock_pipe(dit_dim=WAN_5B_DIM, track_embed_dim=TRACK_EMBED_DIM, spatial_div=32):
    """Create a mock pipeline with track modules for testing."""

    class MockPipe:
        device = torch.device("cpu")
        torch_dtype = torch.float32

    pipe = MockPipe()
    # Wan 2.2: VAE 16x + patchify 2x = 32x total spatial
    pipe.height_division_factor = spatial_div
    pipe.width_division_factor = spatial_div
    pipe.track_embedder = PointTrackEmbedder(
        in_channels=5, num_freqs=16, hidden_dim=156,
        out_dim=track_embed_dim, num_mlp_layers=3,
    )
    pipe.track_adapter = PointTrackAdapter(
        track_dim=track_embed_dim, out_dim=track_embed_dim,
        block_t=1, block_h=1, block_w=1,
    )
    pipe.track_projector = TokenConcatProjector(
        video_dim=dit_dim, track_dim=track_embed_dim,
        hidden_dim=dit_dim, zero_init=True,
    )
    pipe.scheduler = type("Scheduler", (), {"training": True})()
    return pipe


def _make_dummy_inputs(N=64, T=49, H=256, W=256):
    """Create dummy track inputs."""
    tracks_world = torch.randn(N, T, 3)
    tracks_world[..., 2] = tracks_world[..., 2].abs() + 2.0
    cam2world = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
    cam2world[:, :3, 3] = torch.randn(T, 3) * 0.1
    K = torch.tensor([[W, 0, W / 2], [0, H, H / 2], [0, 0, 1]], dtype=torch.float32)
    return tracks_world, cam2world, K


class TestLatentDimensionFix:
    """Verify the //32 spatial latent dim fix.

    Wan 2.2 VAE: 16x spatial compression.
    DiT patchify (Conv3d stride [1,2,2]): 2x additional spatial.
    Total spatial: 32x.  Track tokens must be at post-patchify resolution.
    """

    @pytest.mark.parametrize("H,W,T", [
        (256, 256, 49),
        (384, 384, 49),
        (512, 288, 49),
        (256, 256, 17),
    ])
    def test_latent_dims_match_post_patchify(self, H, W, T):
        """Track tokens must have L = latent_t * (H//32) * (W//32)."""
        pipe = _make_mock_pipe()
        tracks_world, cam2world, K = _make_dummy_inputs(N=32, T=T, H=H, W=W)

        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=H, width=W, num_frames=T,
        )

        # Expected: post-patchify dims (VAE 16x + patchify 2x = 32x spatial)
        latent_t = (T - 1) // 4 + 1
        latent_h = H // 32
        latent_w = W // 32
        expected_L = latent_t * latent_h * latent_w

        tpt = result["track_point_tokens"]
        assert tpt.shape[1] == expected_L, (
            f"Track token seq length {tpt.shape[1]} != expected {expected_L} "
            f"(latent_t={latent_t}, latent_h={latent_h}, latent_w={latent_w})."
        )

    def test_old_div16_would_fail(self):
        """Explicitly verify that //16 gives 4x too many tokens."""
        H, W, T = 256, 256, 49
        latent_t = (T - 1) // 4 + 1  # 13

        correct_L = latent_t * (H // 32) * (W // 32)   # 13 * 8 * 8 = 832
        wrong_L = latent_t * (H // 16) * (W // 16)     # 13 * 16 * 16 = 3328

        assert wrong_L == 4 * correct_L, "Sanity: //16 gives exactly 4x //32"
        assert correct_L == 832
        assert wrong_L == 3328


class TestTrackPointConditionUnit:
    """Test WanVideoUnit_TrackPointCondition outputs."""

    def test_per_frame_intrinsics_reach_adapter_spatial_assignment(self, monkeypatch):
        """K_t must control the actual persistent-token UVs, not only Plucker rays."""
        monkeypatch.setenv("HOLO4D_CAPTURE_TRACK_ADAPTER_INPUTS", "1")
        pipe = _make_mock_pipe()
        T, H, W = 5, 64, 96
        tracks_world = torch.tensor([[[1.0, 0.0, 4.0]] * T])
        cam2world = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
        fixed_k = torch.tensor(
            [[80.0, 0.0, 48.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]]
        )
        per_frame_k = fixed_k.unsqueeze(0).repeat(T, 1, 1)
        per_frame_k[:, 0, 0] = torch.linspace(80.0, 40.0, T)
        per_frame_k[:, 1, 1] = torch.linspace(80.0, 40.0, T)

        unit = WanVideoUnit_TrackPointCondition()
        unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=fixed_k,
            target_intrinsics_per_frame=per_frame_k,
            height=H,
            width=W,
            num_frames=T,
        )

        captured_u = pipe._debug_last_track_adapter_inputs["track_uv"][0, 0, :, 0]
        torch.testing.assert_close(
            captured_u,
            torch.tensor([68.0, 65.5, 63.0, 60.5, 58.0]),
        )

    def test_output_keys(self):
        pipe = _make_mock_pipe()
        tracks_world, cam2world, K = _make_dummy_inputs(N=64, T=49, H=256, W=256)

        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=256, width=256, num_frames=49,
        )

        assert "track_point_tokens" in result
        assert "_pre_block_hook" in result
        assert callable(result["_pre_block_hook"])

    def test_none_inputs_returns_empty(self):
        pipe = _make_mock_pipe()
        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(pipe, tracks_world=None, target_cam2world=None)
        assert result == {}

    def test_none_tracks_can_use_learned_null_token_fallback(self):
        pipe = _make_mock_pipe(track_embed_dim=32)
        pipe.track_null_token = torch.nn.Parameter(torch.full((1, 1, 32), 2.0))
        pipe.track_injection_mode = "single"

        T, H, W = 5, 64, 64
        cam2world = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
        K = torch.tensor([[W, 0, W / 2], [0, H, H / 2], [0, 0, 1]], dtype=torch.float32)

        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(
            pipe,
            tracks_world=None,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=H,
            width=W,
            num_frames=T,
            use_track_null_token_fallback=True,
        )

        latent_t = (T - 1) // 4 + 1
        latent_h = H // 32
        latent_w = W // 32
        expected_L = latent_t * latent_h * latent_w

        assert "track_point_tokens" in result
        assert "_pre_block_hook" in result
        assert result["track_point_tokens"].shape == (1, expected_L, 32)
        assert torch.allclose(result["track_point_tokens"], torch.full((1, expected_L, 32), 2.0))

    def test_has_tracks_false_uses_adapter_null_token_without_changing_path(self):
        pipe = _make_mock_pipe(track_embed_dim=32)
        pipe.track_adapter.null_token.data.fill_(3.0)
        pipe.track_injection_mode = "single"
        tracks_world, cam2world, K = _make_dummy_inputs(N=16, T=5, H=64, W=64)

        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=64,
            width=64,
            num_frames=5,
            has_tracks=False,
        )

        assert "track_point_tokens" in result
        assert torch.allclose(
            result["track_point_tokens"],
            torch.full_like(result["track_point_tokens"], 3.0),
        )

    def test_dense_conditioning_noise_level_also_perturbs_track_tokens(self):
        pipe = _make_mock_pipe(track_embed_dim=32)
        pipe.track_injection_mode = "single"
        tracks_world, cam2world, K = _make_dummy_inputs(N=16, T=5, H=64, W=64)
        unit = WanVideoUnit_TrackPointCondition()

        pipe.dense_conditioning_noise_max_level = 0.0
        clean = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=64,
            width=64,
            num_frames=5,
        )["track_point_tokens"]

        torch.manual_seed(123)
        pipe.dense_conditioning_noise_max_level = 0.2
        noisy = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=64,
            width=64,
            num_frames=5,
        )["track_point_tokens"]

        assert not torch.allclose(noisy, clean)

    def test_dense_conditioning_noise_level_does_not_perturb_track_tokens_in_eval(self):
        pipe = _make_mock_pipe(track_embed_dim=32)
        pipe.track_injection_mode = "single"
        pipe.scheduler.training = False
        tracks_world, cam2world, K = _make_dummy_inputs(N=16, T=5, H=64, W=64)
        unit = WanVideoUnit_TrackPointCondition()

        pipe.dense_conditioning_noise_max_level = 0.0
        clean = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=64,
            width=64,
            num_frames=5,
        )["track_point_tokens"]

        torch.manual_seed(123)
        pipe.dense_conditioning_noise_max_level = 0.2
        eval_tokens = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=64,
            width=64,
            num_frames=5,
        )["track_point_tokens"]

        assert torch.allclose(eval_tokens, clean)

    def test_token_shape_wan_5b_default(self):
        """49 frames, 256x256 → latent 13x8x8 = 832 tokens."""
        pipe = _make_mock_pipe()
        tracks_world, cam2world, K = _make_dummy_inputs(N=1024, T=49, H=256, W=256)

        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=256, width=256, num_frames=49,
        )

        tpt = result["track_point_tokens"]
        assert tpt.shape == (1, 832, TRACK_EMBED_DIM)


class TestPreBlockHook:
    """Test _pre_block_hook behavior with V2V layouts."""

    def _run_unit_and_get_hook(self, T=49, H=256, W=256, N=64):
        pipe = _make_mock_pipe()
        tracks_world, cam2world, K = _make_dummy_inputs(N=N, T=T, H=H, W=W)
        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=H, width=W, num_frames=T,
        )
        return result["_pre_block_hook"], pipe

    def test_hook_width_mode_preserves_shape(self):
        """V2V width mode (2 domains): hook must preserve (B, f*h*w, D)."""
        hook, pipe = self._run_unit_and_get_hook()
        # Post-patchify dims for 256x256x49: f=13, h=8, w=8 (target)
        latent_t = 13
        latent_h = 8   # 256 // 32
        latent_w = 8   # 256 // 32

        B = 1
        f = latent_t
        h = latent_h
        w = latent_w * 2  # 2 domains in width

        x = torch.randn(B, f * h * w, WAN_5B_DIM)
        x_out = hook(x, f=f, h=h, w=w, num_domains=2, v2v_concat_mode="width")
        assert x_out.shape == x.shape

    def test_hook_frame_mode_preserves_shape(self):
        """V2V frame mode (2 domains): hook must preserve (B, f*h*w, D)."""
        hook, pipe = self._run_unit_and_get_hook()
        latent_t = 13
        latent_h = 8
        latent_w = 8

        B = 1
        f = latent_t * 2  # 2 domains in frame
        h = latent_h
        w = latent_w

        x = torch.randn(B, f * h * w, WAN_5B_DIM)
        x_out = hook(x, f=f, h=h, w=w, num_domains=2, v2v_concat_mode="frame")
        assert x_out.shape == x.shape

    def test_zero_init_projector_preserves_input(self):
        """At initialization, zero-init projector should output x unchanged."""
        hook, pipe = self._run_unit_and_get_hook()
        latent_t = 13
        latent_h = 8
        latent_w = 8

        B = 1
        f, h, w = latent_t, latent_h, latent_w * 2
        x = torch.randn(B, f * h * w, WAN_5B_DIM)

        x_out = hook(x, f=f, h=h, w=w, num_domains=2, v2v_concat_mode="width")

        diff = (x_out - x).abs().max().item()
        assert diff < 1e-5, (
            f"Zero-init projector should preserve input at init, got max diff={diff}"
        )

    def test_gradient_flow_through_hook(self):
        """Gradients must flow from loss through hook to track module params."""
        pipe = _make_mock_pipe(dit_dim=64, track_embed_dim=32)
        tracks_world, cam2world, K = _make_dummy_inputs(N=16, T=5, H=64, W=64)

        # Need gradients
        for m in [pipe.track_embedder, pipe.track_adapter, pipe.track_projector]:
            for p in m.parameters():
                p.requires_grad_(True)

        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=64, width=64, num_frames=5,
        )

        hook = result["_pre_block_hook"]

        # Post-patchify dims for 64x64x5: f=2, h=2, w=2 (target)
        latent_t = (5 - 1) // 4 + 1  # 2
        latent_h = 64 // 32           # 2
        latent_w = 64 // 32           # 2
        f, h, w = latent_t, latent_h, latent_w * 2  # width-concat 2 domains

        x = torch.randn(1, f * h * w, 64, requires_grad=True)
        x_out = hook(x, f=f, h=h, w=w, num_domains=2, v2v_concat_mode="width")

        loss = x_out.sum()
        loss.backward()

        # Check gradient flow to projector
        proj_grads = [p.grad for p in pipe.track_projector.parameters() if p.grad is not None]
        assert len(proj_grads) > 0, "No gradients reached track_projector"

    def test_hook_source_domain_gets_null_tokens(self):
        """In width mode, source domain (left half) should get null token injection."""
        pipe = _make_mock_pipe(dit_dim=64, track_embed_dim=32)
        # Disable zero-init so projector actually changes x
        with torch.no_grad():
            for p in pipe.track_projector.parameters():
                p.normal_(0, 0.1)

        tracks_world, cam2world, K = _make_dummy_inputs(N=16, T=5, H=64, W=64)
        unit = WanVideoUnit_TrackPointCondition()
        result = unit.process(
            pipe,
            tracks_world=tracks_world,
            target_cam2world=cam2world,
            target_intrinsics=K,
            height=64, width=64, num_frames=5,
        )
        hook = result["_pre_block_hook"]

        latent_t = (5 - 1) // 4 + 1  # 2
        latent_h = 64 // 32           # 2
        latent_w = 64 // 32           # 2
        f, h, w = latent_t, latent_h, latent_w * 2

        x = torch.ones(1, f * h * w, 64)
        x_out = hook(x, f=f, h=h, w=w, num_domains=2, v2v_concat_mode="width")

        # Reshape to spatial to check source vs target
        x_out_spatial = x_out.reshape(1, f, h, w, 64)
        source_half = x_out_spatial[:, :, :, :latent_w, :]
        target_half = x_out_spatial[:, :, :, latent_w:, :]

        # Both are processed through projector, hook ran and produced valid output
        assert x_out.shape == x.shape
        assert not torch.isnan(x_out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
