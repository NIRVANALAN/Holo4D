"""
dense_target_control_unit.py — GEN3C-style dense target-space control branch.

This module deliberately lives under ``holo4d/conditions/`` (NOT under
``holo4d_runtime/pipeline/``) so that importing it does NOT trigger
``holo4d_runtime/pipeline/__init__.py``, which transitively pulls in
``diffusers``/``transformers``/``deepspeed`` and dies in CUDA-less or
toolchain-incomplete environments.

The pipeline file ``holo4d_runtime/pipeline/wan_video_track4d360.py`` re-exports
``WanVideoUnit_DenseTargetControl``, ``_align_track_to_sequence``, and
``_normalize_warped_depth_per_frame`` from here so production import paths
keep working unchanged:

    from holo4d_runtime.pipeline.wan_video_track4d360 import (
        WanVideoUnit_DenseTargetControl,
        _align_track_to_sequence,
    )

…remains valid for production code, while tests can import directly from
``holo4d.conditions.dense_target_control_unit`` and avoid the heavy
pipeline-package init entirely.

The unit:

  * builds ``[warped_rgb | warped_depth_norm | warp_valid]`` in the target
    pixel space via ``build_dense_target_reprojection``;
  * encodes it with ``DenseTargetControlEncoder`` into per-block dense
    control tokens;
  * fuses additively with the persistent ``track_point_tokens`` produced by
    ``WanVideoUnit_TrackPointCondition``::

        target_tokens = dense_ctrl_tokens + alpha_track * persistent_track_tokens

  * rebuilds whichever injection hook the pipeline's ``track_injection_mode``
    needs (``_pre_block_hook`` for ``"single"``, ``_track_align_fn`` for
    ``"per_block"``).

Skip behaviour: graceful no-op when ``use_dense_branch`` is False, when
``pipe.dense_target_control_encoder`` is missing, or when any of the required
input tensors is absent — leaving the persistent path's hook untouched.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from holo4d.conditions.dense_target_reprojection_builder import (
    build_dense_target_reprojection,
)
from holo4d.conditions.dense_target_control_encoder import (
    DenseTargetControlEncoder,  # noqa: F401  (re-exported for callers)
)
from holo4d.conditions.point_track_adapter import (
    expand_block_tokens_to_latent_seq,
)


def _apply_condition_token_noise(
    tokens: torch.Tensor,
    max_level: float,
    *,
    training: bool,
) -> torch.Tensor:
    """Apply Vista4D-style train-time relative Gaussian noise to condition tokens."""
    max_level = float(max_level)
    if max_level <= 0.0 or not training or not torch.is_grad_enabled():
        return tokens
    level = torch.rand((), device=tokens.device, dtype=tokens.dtype) * max_level
    token_std = tokens.detach().std().clamp_min(1e-6)
    noise = torch.randn_like(tokens) * token_std
    return tokens + level * noise


def _maybe_remove_source_mask_boundary_depth(
    source_depth_valid: torch.Tensor,
    source_dynamic_mask: torch.Tensor | None,
    *,
    prob: float,
    kernel_size: int,
    training: bool,
) -> torch.Tensor:
    """Randomly invalidate depth on source foreground/background boundaries.

    The SAM/GMOS-style masks used at inference are imperfect around object
    silhouettes. Removing a narrow band around the mask boundary during
    training reduces reliance on GT or estimated depth exactly where train and
    inference differ most. Backward-compatible no-op when ``prob <= 0`` or no
    source dynamic mask is available.
    """
    if source_dynamic_mask is None or source_depth_valid is None:
        return source_depth_valid
    if float(prob) <= 0.0 or int(kernel_size) <= 1 or not bool(training):
        return source_depth_valid
    if torch.rand((), device=source_depth_valid.device).item() >= float(prob):
        return source_depth_valid

    valid = source_depth_valid.bool()
    mask = source_dynamic_mask.to(device=valid.device).bool()
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).expand(valid.shape[0], -1, -1, -1)
    if mask.shape != valid.shape:
        raise ValueError(
            f"source_dynamic_mask shape {tuple(mask.shape)} != "
            f"source_depth_valid shape {tuple(valid.shape)}"
        )

    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    b, t, h, w = mask.shape
    m = mask.reshape(b * t, 1, h, w).float()
    dilated = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad) > 0.5
    eroded = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=pad) > 0.5
    boundary = (dilated ^ eroded).reshape(b, t, h, w)
    return valid & ~boundary


def _select_dense_hybrid_dynamic_track_valid_mask(
    tracks_world_valid_mask,
    tracks_world_dense_valid_mask,
):
    """Choose which validity mask bounds dynamic-track RGBD fusion.

    Historical behavior uses the dense pre-sparsification mask so dynamic
    fusion sees every available identity. For large multi-query batches this
    makes the dense hybrid cache scale with q * H * W even when the track
    adapter itself is sparse. Set
    ``TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_VALID_MODE=sparse`` to make the
    fusion cache follow the same sparse budget as the adapter.
    """
    mode = os.environ.get("TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_VALID_MODE", "dense")
    mode = str(mode).strip().lower().replace("-", "_")
    if mode in {"sparse", "adapter", "current", "tracks_world_valid_mask"}:
        return tracks_world_valid_mask if tracks_world_valid_mask is not None else tracks_world_dense_valid_mask
    if mode in {"none", "off", "false", "0"}:
        return None
    if mode not in {"dense", "legacy", "tracks_world_dense_valid_mask"}:
        raise ValueError(
            "TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_VALID_MODE must be one of "
            f"'dense', 'sparse', or 'none', got {mode!r}"
        )
    return tracks_world_dense_valid_mask if tracks_world_dense_valid_mask is not None else tracks_world_valid_mask


# ---------------------------------------------------------------------------
# PipelineUnit base — try the real one, fall back to a minimal stub if the
# heavy import chain fails (e.g. CUDA-less envs where deepspeed can't load).
# ---------------------------------------------------------------------------

try:
    from holo4d_runtime.diffusion.base_pipeline import PipelineUnit  # type: ignore
except Exception:  # noqa: BLE001 — anything in the heavy import chain
    class PipelineUnit:  # type: ignore[no-redef]
        """Minimal stand-in for ``holo4d_runtime.diffusion.base_pipeline.PipelineUnit``.

        Captures the same attribute surface used by the unit class
        (``input_params`` / ``output_params``) so the dense unit can be
        instantiated and exercised in test environments where the full
        holo4d_runtime stack isn't importable.
        """

        def __init__(
            self,
            seperate_cfg: bool = False,
            take_over: bool = False,
            input_params=None,
            output_params=None,
            input_params_posi=None,
            input_params_nega=None,
            onload_model_names=None,
        ):
            self.seperate_cfg = seperate_cfg
            self.take_over = take_over
            self.input_params = input_params
            self.output_params = output_params
            self.input_params_posi = input_params_posi
            self.input_params_nega = input_params_nega
            self.onload_model_names = onload_model_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def track_occupancy_channel_enabled() -> bool:
    """Opt-in structural occupancy channel on injected track tokens.

    TRACK4D360_TRACK_OCCUPANCY_CHANNEL=1 appends one extra channel to the
    aligned track-token sequence: 1.0 where a track/dense observation occupies
    the latent block, 0.0 for unobserved blocks and all padding. The channel is
    structural (never produced by a learnable parameter), so — unlike the
    learned ``null_token`` — it cannot drift toward occupied-token statistics
    (see doc/track4d-360/cross-checks/
    2026-07-22-gwtt-empty-token-ambiguity-and-tref-eligibility-verification.md).
    Default 0 keeps every legacy code path byte-identical.
    """
    return os.environ.get("TRACK4D360_TRACK_OCCUPANCY_CHANNEL", "0") == "1"


def adapter_padding_vector(adapter, track_dim: int) -> torch.Tensor:
    """Padding vector used when aligning track tokens to the full sequence.

    Legacy (track_dim == null numel): the adapter's learnable ``null_token``.
    Occupancy mode (track_dim == null numel + 1): the null token with a
    constant appended 0 — padding is always "unobserved" in the structural
    occupancy channel. Any other mismatch fails loudly.
    """
    null_flat = adapter.null_token.reshape(-1)
    if null_flat.numel() == int(track_dim):
        return null_flat
    if track_occupancy_channel_enabled() and null_flat.numel() + 1 == int(track_dim):
        return torch.cat([null_flat, null_flat.new_zeros(1)])
    raise ValueError(
        "adapter_padding_vector: track_dim="
        f"{int(track_dim)} incompatible with null_token numel="
        f"{null_flat.numel()} (occupancy_channel="
        f"{track_occupancy_channel_enabled()})"
    )


def _align_track_to_sequence(
    tpt,
    adapter,
    x,
    f,
    h,
    w,
    num_domains,
    v2v_concat_mode,
    source_tpt=None,
):
    """Expand target-only track tokens to the full V2V latent sequence.

    By default, pads the source domain with the adapter's learnable
    ``null_token``. The opt-in persistent shared-identity path may provide
    ``source_tpt`` with the same shape as ``tpt``; legacy callers omit it and
    remain numerically identical.

    Used by both the persistent track unit (``WanVideoUnit_TrackPointCondition``)
    and the dense target-control unit so they share a single canonical
    implementation.
    """
    B_x = x.shape[0]
    track_dim = tpt.shape[-1]

    def _pad_or_trunc_to_full_seq(track_full):
        # Pad/truncate the per-(output-stream) track tokens to the DiT's full
        # token sequence length x.shape[1]. This is a NO-OP whenever the track
        # tokens already cover the whole sequence (f * h * w == x.shape[1]),
        # i.e. every run WITHOUT Vista4D latent streams — so existing track runs
        # stay bit-identical. When use_vista4d_latent_streams=1 the sequence is
        # [output | point-cloud | source] = 3 * f * h * w; tracks only condition
        # the (first) output stream, so the two clean conditioning streams are
        # padded with the learnable null_token, mirroring how the V2V source
        # domain is already null-padded above.
        L = x.shape[1]
        if track_full.shape[1] < L:
            pad = adapter_padding_vector(adapter, track_dim).view(1, 1, track_dim)
            pad = pad.expand(B_x, L - track_full.shape[1], track_dim)
            track_full = torch.cat([track_full, pad.to(x.device, x.dtype)], dim=1)
        elif track_full.shape[1] > L:
            track_full = track_full[:, :L]
        return track_full

    if v2v_concat_mode == "width" and num_domains >= 2:
        w_target = w // num_domains
        track_spatial = tpt.reshape(B_x, f, h, w_target, track_dim)
        w_source = w - w_target
        if source_tpt is None:
            source_spatial = adapter_padding_vector(adapter, track_dim).view(1, 1, 1, 1, track_dim)
            source_spatial = source_spatial.expand(
                B_x, f, h, w_source, track_dim
            )
        else:
            source_identity_spatial = source_tpt.reshape(
                B_x, f, h, w_target, track_dim
            )
            if w_source < w_target:
                raise ValueError(
                    "Shared source identity domain is wider than the available "
                    f"source sequence: source={w_target}, available={w_source}"
                )
            if w_source == w_target:
                source_spatial = source_identity_spatial
            else:
                remaining_null = adapter_padding_vector(adapter, track_dim).view(
                    1, 1, 1, 1, track_dim
                ).expand(B_x, f, h, w_source - w_target, track_dim)
                source_spatial = torch.cat(
                    [source_identity_spatial, remaining_null], dim=3
                )
        source_spatial = source_spatial.to(device=x.device, dtype=x.dtype)
        track_spatial = track_spatial.to(device=x.device, dtype=x.dtype)
        track_full_spatial = torch.cat([source_spatial, track_spatial], dim=3)
        return _pad_or_trunc_to_full_seq(track_full_spatial.reshape(B_x, f * h * w, track_dim))
    elif v2v_concat_mode == "frame" and num_domains >= 2:
        f_target = f // num_domains
        track_spatial = tpt.reshape(B_x, f_target, h, w, track_dim)
        f_source = f - f_target
        if source_tpt is None:
            source_spatial = adapter_padding_vector(adapter, track_dim).view(1, 1, 1, 1, track_dim)
            source_spatial = source_spatial.expand(
                B_x, f_source, h, w, track_dim
            )
        else:
            source_identity_spatial = source_tpt.reshape(
                B_x, f_target, h, w, track_dim
            )
            if f_source < f_target:
                raise ValueError(
                    "Shared source identity domain has more frames than the "
                    f"available source sequence: source={f_target}, available={f_source}"
                )
            if f_source == f_target:
                source_spatial = source_identity_spatial
            else:
                remaining_null = adapter_padding_vector(adapter, track_dim).view(
                    1, 1, 1, 1, track_dim
                ).expand(B_x, f_source - f_target, h, w, track_dim)
                source_spatial = torch.cat(
                    [source_identity_spatial, remaining_null], dim=1
                )
        source_spatial = source_spatial.to(device=x.device, dtype=x.dtype)
        track_spatial = track_spatial.to(device=x.device, dtype=x.dtype)
        track_full_spatial = torch.cat([source_spatial, track_spatial], dim=1)
        return _pad_or_trunc_to_full_seq(track_full_spatial.reshape(B_x, f * h * w, track_dim))
    else:
        track_full = tpt.to(device=x.device, dtype=x.dtype)
        return _pad_or_trunc_to_full_seq(track_full)


def _normalize_warped_depth_per_frame(
    warped_depth: torch.Tensor,    # (B, T, H, W) float
    warp_valid: torch.Tensor,      # (B, T, H, W) bool
) -> torch.Tensor:
    """Per-(B,T)-frame max-normalisation of warped depth into [0, 1].

    Used so the dense control encoder sees a bounded depth channel without
    needing the global metric scale.  Invalid pixels stay at 0.
    """
    valid_f = warp_valid.float()
    masked = warped_depth * valid_f
    # max over (H, W); shape (B, T, 1, 1)
    d_max = masked.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-3)
    return (masked / d_max) * valid_f


# ---------------------------------------------------------------------------
# Dense target-space control unit
# ---------------------------------------------------------------------------

class WanVideoUnit_DenseTargetControl(PipelineUnit):
    """
    GEN3C-style dense target-space control branch.

    See module docstring for the design overview.  This class lives in
    ``holo4d/conditions/`` (not in ``holo4d_runtime/pipeline/``) so it can
    be imported in environments without diffusers/deepspeed/CUDA — the
    pipeline file re-exports it for production callers.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                # Geometry
                "source_rgb_video",
                "source_depth_video",
                "source_depth_valid",
                "source_cam2world",
                "source_intrinsics",
                "source_intrinsics_4rc",
                "source_intrinsics_gt",
                "target_cam2world",
                "target_intrinsics",
                # Shape
                "height",
                "width",
                "num_frames",
                # Toggle + persistent path output to fuse with
                "use_dense_branch",
                "use_vista4d_latent_streams",
                "use_gen3c_warp_latent_channels",
                "track_point_tokens",
                # Warped appearance mode: when "latent_fuse", dense branch
                # becomes geometry-only (no RGB) if warped_geom_only_dense=True
                "warped_condition_mode",
                "warped_geom_only_dense",
                # Dense reprojection mode: "zbuffer" (classic nearest-z
                # scatter), "softsplat" (GEN3C-style per-frame bilinear
                # soft-splat), or "hybrid_softsplat" (soft-splat plus
                # multi-frame fusion at static pixels and per-frame fallback
                # at dynamic pixels).
                "dense_warp_mode",
                "dense_warp_motion_threshold",
                "dense_hybrid_disable_dynamic_masks",
                # Per-pixel 3D tracks (Kubric_NVS / Syn4D) — drives the
                # static-mask filter when dense_warp_mode=hybrid_softsplat.
                # Optional; absent for static-scene datasets which are
                # treated as pure-static (multi-frame fill covers everywhere).
                "tracks_world",
                "tracks_world_valid_mask",
                "tracks_world_dense_valid_mask",
                "track_query_count",
                "source_dynamic_mask",
                "target_dynamic_mask",
                "use_dense_null_token_fallback",
            ),
            output_params=(
                "dense_ctrl_tokens",
                "_pre_block_hook",
                "_track_align_fn",
                "_track_block_injector",
                "track_point_tokens",
            ),
        )

    @staticmethod
    def _coerce_source_rgb_video(
        src_rgb,
        B: int,
        T: int,
        H: int,
        W: int,
        device,
    ) -> torch.Tensor:
        """Accept (T,H,W,3), (B,T,H,W,3), (B,3,T,H,W), or list of PILs.

        Returns (B, T, H, W, 3) float32 in [0, 1] on ``device``.
        """
        if isinstance(src_rgb, (list, tuple)):
            import numpy as np
            arr = np.stack(
                [np.asarray(im, dtype=np.float32) / 255.0 for im in src_rgb], axis=0
            )  # (T, H, W, 3)
            src_rgb = torch.from_numpy(arr)
        if not torch.is_tensor(src_rgb):
            raise TypeError(f"source_rgb_video must be tensor or list, got {type(src_rgb)}")
        x = src_rgb.to(device=device, dtype=torch.float32)
        if x.ndim == 4:                                # (T, H, W, 3)
            x = x.unsqueeze(0).expand(B, -1, -1, -1, -1)
        elif x.ndim == 5 and x.shape[1] == 3 and x.shape[2] == T:
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, 3, T, H, W) → (B, T, H, W, 3)
        # already (B, T, H, W, 3)
        if x.shape != (B, T, H, W, 3):
            raise ValueError(
                f"source_rgb_video coerced to {tuple(x.shape)}, expected ({B},{T},{H},{W},3)"
            )
        # Heuristic: if dynamic range looks like [0, 255], rescale.
        if x.max() > 2.0:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _coerce_source_depth_video(
        src_depth,
        B: int,
        T: int,
        H: int,
        W: int,
        device,
    ) -> torch.Tensor:
        x = src_depth
        if not torch.is_tensor(x):
            import numpy as np
            x = torch.from_numpy(np.asarray(x))
        x = x.to(device=device, dtype=torch.float32)
        if x.ndim == 3:                       # (T, H, W)
            x = x.unsqueeze(0).expand(B, -1, -1, -1)
        elif x.ndim == 5:                     # (B, 1, T, H, W)
            x = x.squeeze(1)
        if x.shape != (B, T, H, W):
            raise ValueError(
                f"source_depth_video coerced to {tuple(x.shape)}, expected ({B},{T},{H},{W})"
            )
        return x

    @staticmethod
    def _coerce_source_depth_valid(
        src_valid,
        depth: torch.Tensor,    # (B, T, H, W)
    ) -> torch.Tensor:
        if src_valid is None:
            # Default: depth > 0 is valid
            return depth > 0
        x = src_valid
        if not torch.is_tensor(x):
            import numpy as np
            x = torch.from_numpy(np.asarray(x))
        x = x.to(device=depth.device)
        if x.dtype != torch.bool:
            x = x > 0.5
        if x.ndim == 3:
            x = x.unsqueeze(0).expand_as(depth)
        if x.shape != depth.shape:
            raise ValueError(
                f"source_depth_valid shape {tuple(x.shape)} != depth shape {tuple(depth.shape)}"
            )
        return x

    @staticmethod
    def _coerce_cam2world(c2w, B: int, T: int, device) -> torch.Tensor:
        if not torch.is_tensor(c2w):
            import numpy as np
            c2w = torch.from_numpy(np.asarray(c2w))
        x = c2w.to(device=device, dtype=torch.float32)
        if x.ndim == 3:                  # (T, 4, 4)
            x = x.unsqueeze(0).expand(B, -1, -1, -1)
        if x.shape != (B, T, 4, 4):
            raise ValueError(
                f"cam2world coerced to {tuple(x.shape)}, expected ({B},{T},4,4)"
            )
        return x

    @staticmethod
    def _coerce_intrinsics(K, B: int, T: int, device) -> torch.Tensor:
        if not torch.is_tensor(K):
            import numpy as np
            K = torch.from_numpy(np.asarray(K))
        x = K.to(device=device, dtype=torch.float32)
        if x.ndim == 2:                  # (3, 3)
            x = x.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        elif x.ndim == 3:                # (T, 3, 3)
            x = x.unsqueeze(0).expand(B, -1, -1, -1)
        if x.shape != (B, T, 3, 3):
            raise ValueError(
                f"intrinsics coerced to {tuple(x.shape)}, expected ({B},{T},3,3)"
            )
        return x

    def process(
        self,
        pipe,
        source_rgb_video=None,
        source_depth_video=None,
        source_depth_valid=None,
        source_cam2world=None,
        source_intrinsics=None,
        target_cam2world=None,
        target_intrinsics=None,
        # 4RC-labeled path: keep predicted and GT intrinsics available. The
        # appearance/depth warp uses GT K because source RGB/depth are on the
        # cropped GT image grid; the predicted K is retained for diagnostics.
        source_intrinsics_4rc=None,
        source_intrinsics_gt=None,
        height=None,
        width=None,
        num_frames=None,
        use_dense_branch=False,
        use_vista4d_latent_streams=False,
        use_gen3c_warp_latent_channels=False,
        track_point_tokens=None,
        warped_condition_mode="off",
        warped_geom_only_dense=True,
        dense_warp_mode="zbuffer",
        dense_warp_motion_threshold=0.01,
        dense_hybrid_disable_dynamic_masks=False,
        tracks_world=None,
        tracks_world_valid_mask=None,
        tracks_world_dense_valid_mask=None,
        track_query_count=None,
        source_dynamic_mask=None,
        target_dynamic_mask=None,
        # Explicit flag (also listed in input_params above so the unit_runner
        # forwards it): emit a learned dense null token when the dense inputs
        # are missing/dropped instead of returning no dense tokens. No **kwargs
        # catch-all: every input_params entry is a named arg, so input_params /
        # signature drift fails loudly instead of silently dropping an input.
        use_dense_null_token_fallback=False,
    ):
        # ------------------------------------------------------------------
        # Skip conditions: any of these → return inputs untouched
        # ------------------------------------------------------------------
        if not use_dense_branch:
            return {}
        if bool(use_vista4d_latent_streams) or bool(use_gen3c_warp_latent_channels):
            return {}
        if target_cam2world is None:
            return {}

        device = pipe.device
        dtype = pipe.torch_dtype

        H = int(height)
        W = int(width)
        T = int(num_frames)

        # ------------------------------------------------------------------
        # 1. Coerce / move all inputs to (B, ...) on device, float32
        # ------------------------------------------------------------------
        # Determine batch size from cam2world (most reliable)
        tgt_c2w_t = target_cam2world if torch.is_tensor(target_cam2world) else torch.as_tensor(target_cam2world)
        B = tgt_c2w_t.shape[0] if tgt_c2w_t.ndim == 4 else 1
        spatial_div = getattr(pipe, "height_division_factor", 16)
        latent_t = (T - 1) // 4 + 1
        latent_h = H // spatial_div
        latent_w = W // spatial_div

        missing_dense_inputs = (
            source_rgb_video is None or source_depth_video is None or source_cam2world is None
        )
        # Structural occupancy for the dense branch (opt-in, see
        # track_occupancy_channel_enabled). None on the null-fallback path →
        # treated as all-unobserved when the channel is appended.
        dense_occ_seq = None
        if missing_dense_inputs:
            if not use_dense_null_token_fallback:
                return {}
            null_token = getattr(pipe, "dense_null_token", None)
            if null_token is None:
                return {}
            dense_ctrl_tokens = null_token.to(device=device, dtype=dtype).expand(
                B, latent_t * latent_h * latent_w, -1
            )
        else:
            encoder = getattr(pipe, "dense_target_control_encoder", None)
            if encoder is None:
                return {}
            enc_in_channels = int(getattr(encoder, "in_channels", 5))

            src_rgb = self._coerce_source_rgb_video(source_rgb_video, B, T, H, W, device)
            src_depth = self._coerce_source_depth_video(source_depth_video, B, T, H, W, device)
            src_valid = self._coerce_source_depth_valid(source_depth_valid, src_depth)
            src_c2w = self._coerce_cam2world(source_cam2world, B, T, device)
            tgt_c2w = self._coerce_cam2world(target_cam2world, B, T, device)

            # Choose K for source unproject vs target project. For 4RC-labeled
            # ReCam samples, source RGB/depth live on the cropped GT image
            # grid, so the appearance/depth warp must unproject with GT K.
            # The predicted 4RC K is still carried for audit/label geometry,
            # but using it here makes even same-view RGB warps non-identity
            # when 4RC's camera estimate differs from the dataset camera.
            K_unproj_in = source_intrinsics_gt if source_intrinsics_gt is not None else source_intrinsics
            if K_unproj_in is None:
                K_unproj_in = source_intrinsics_4rc if source_intrinsics_4rc is not None else target_intrinsics
            K_proj_in = source_intrinsics_gt if source_intrinsics_gt is not None else target_intrinsics
            if K_proj_in is None:
                K_proj_in = source_intrinsics
            if K_unproj_in is None or K_proj_in is None:
                return {}
            K = self._coerce_intrinsics(K_unproj_in, B, T, device)
            K_tgt = self._coerce_intrinsics(K_proj_in, B, T, device) \
                if K_proj_in is not K_unproj_in else K

            # ------------------------------------------------------------------
            # 2. Forward-warp source RGB+depth into target camera.
            #    Backends, selected via `dense_warp_mode`:
            #      - "zbuffer" (default, fully back-compat): hard nearest-z
            #        scatter via build_dense_target_reprojection.
            #      - "softsplat": GEN3C-style per-frame bilinear soft-splat
            #        with the same [RGB, target-depth, valid] output contract.
            #      - "hybrid_softsplat": Lyra-2 / GEN3C bilinear soft-splat
            #        with multi-frame fusion at static target pixels and
            #        per-frame fallback at dynamic pixels. Static / dynamic
            #        comes from the per-pixel motion range of `tracks_world`
            #        (Kubric_NVS / Syn4D); datasets without tracks are
            #        treated as pure static.
            # ------------------------------------------------------------------
            dense_warp_mode_l = str(dense_warp_mode).lower()
            dense_motion_mask = None
            source_dynamic_mask_t = None
            dense_source_alpha_map = None
            dense_source_motion_map = None
            if dense_warp_mode_l == "hybrid_softsplat":
                from .dense_target_reprojection_builder import (
                    build_dense_target_reprojection_hybrid,
                )
                from .soft_splat_warp import (
                    compute_projected_dynamic_mask_from_tracks_world,
                    compute_target_dynamic_mask_from_tracks_world,
                )
                # Target-side dynamic mask: project the dynamic 3D points at
                # time t into target camera t. The mask gates the hybrid
                # compose so dynamic regions use the per-frame s=t→t warp
                # (carries motion correctly) while static regions enjoy the
                # multi-frame fused fill. NOT a source-side filter — the
                # query-frame-anchored static mask is misaligned with
                # source frames where t != query_idx, so we filter at the
                # target instead.
                disable_dynamic_masks = bool(dense_hybrid_disable_dynamic_masks)
                if disable_dynamic_masks:
                    source_dynamic_mask_t = None
                elif source_dynamic_mask is not None:
                    source_dynamic_mask_t = source_dynamic_mask.to(device=device).bool()
                    if source_dynamic_mask_t.ndim == 3:
                        source_dynamic_mask_t = source_dynamic_mask_t.unsqueeze(0).expand(B, -1, -1, -1)
                else:
                    source_dynamic_mask_t = compute_projected_dynamic_mask_from_tracks_world(
                        tracks_world=tracks_world,
                        cam2world=src_c2w,
                        # Dynamic masks live in RGB image pixel space, not
                        # source-depth ray space. 4RC samples may use a
                        # separate K for depth unprojection, but mask
                        # projection must match the cropped GT image grid.
                        intrinsics=K_tgt,
                        image_h=H,
                        image_w=W,
                        motion_threshold=float(dense_warp_motion_threshold),
                    )
                src_valid = _maybe_remove_source_mask_boundary_depth(
                    src_valid,
                    source_dynamic_mask_t,
                    prob=float(getattr(pipe, "source_mask_boundary_depth_dropout_prob", 0.0)),
                    kernel_size=int(getattr(pipe, "source_mask_boundary_kernel_size", 5)),
                    training=bool(getattr(getattr(pipe, "scheduler", None), "training", False)),
                )
                source_static_mask = None if source_dynamic_mask_t is None else ~source_dynamic_mask_t

                if disable_dynamic_masks:
                    target_dynamic_mask_t = None
                elif target_dynamic_mask is not None:
                    target_dynamic_mask_t = target_dynamic_mask.to(device=device).bool()
                    if target_dynamic_mask_t.ndim == 3:
                        target_dynamic_mask_t = target_dynamic_mask_t.unsqueeze(0).expand(B, -1, -1, -1)
                else:
                    target_dynamic_mask_t = compute_target_dynamic_mask_from_tracks_world(
                        tracks_world=tracks_world,
                        tgt_cam2world=tgt_c2w,
                        tgt_intrinsics=K_tgt,
                        image_h=H,
                        image_w=W,
                        motion_threshold=float(dense_warp_motion_threshold),
                    )
                warped_rgb, warped_depth, warp_valid = build_dense_target_reprojection_hybrid(
                    source_rgb=src_rgb,
                    source_depth=src_depth,
                    source_depth_valid=src_valid,
                    src_cam2world=src_c2w,
                    tgt_cam2world=tgt_c2w,
                    intrinsics=K,                    # source K for unproject
                    image_h=H,
                    image_w=W,
                    source_static_mask=source_static_mask,
                    target_dynamic_mask=target_dynamic_mask_t,
                    dynamic_tracks_world=tracks_world,
                    dynamic_tracks_valid_mask=_select_dense_hybrid_dynamic_track_valid_mask(
                        tracks_world_valid_mask,
                        tracks_world_dense_valid_mask,
                    ),
                    dynamic_tracks_query_count=track_query_count,
                    target_intrinsics=K_tgt,         # target K for projection
                )
                dense_motion_mask = target_dynamic_mask_t
            elif dense_warp_mode_l == "softsplat":
                from .dense_target_reprojection_builder import (
                    build_dense_target_reprojection_softsplat,
                )
                warped_rgb, warped_depth, warp_valid = build_dense_target_reprojection_softsplat(
                    source_rgb=src_rgb,
                    source_depth=src_depth,
                    source_depth_valid=src_valid,
                    src_cam2world=src_c2w,
                    tgt_cam2world=tgt_c2w,
                    intrinsics=K,                    # source K for unproject
                    image_h=H,
                    image_w=W,
                    target_intrinsics=K_tgt,         # target K for projection
                )
                if target_dynamic_mask is not None:
                    dense_motion_mask = target_dynamic_mask.to(device=device).bool()
                    if dense_motion_mask.ndim == 3:
                        dense_motion_mask = dense_motion_mask.unsqueeze(0).expand(B, -1, -1, -1)
            else:
                warped_rgb, warped_depth, warp_valid = build_dense_target_reprojection(
                    source_rgb=src_rgb,
                    source_depth=src_depth,
                    source_depth_valid=src_valid,
                    src_cam2world=src_c2w,
                    tgt_cam2world=tgt_c2w,
                    intrinsics=K,                    # source K for unproject
                    image_h=H,
                    image_w=W,
                    target_intrinsics=K_tgt,         # target K for projection
                )
                if target_dynamic_mask is not None:
                    dense_motion_mask = target_dynamic_mask.to(device=device).bool()
                    if dense_motion_mask.ndim == 3:
                        dense_motion_mask = dense_motion_mask.unsqueeze(0).expand(B, -1, -1, -1)

            if enc_in_channels in {7, 8}:
                from .dense_target_reprojection_builder import (
                    build_dense_target_reprojection_softsplat,
                )
                source_alpha = (
                    torch.ones_like(src_valid, dtype=torch.float32, device=device)
                    if enc_in_channels == 8
                    else torch.zeros_like(src_valid, dtype=torch.float32, device=device)
                )
                source_motion = (
                    torch.zeros_like(src_valid, dtype=torch.float32, device=device)
                    if source_dynamic_mask_t is None
                    else source_dynamic_mask_t.to(device=device).float()
                )
                source_mask_rgb = torch.stack(
                    [source_alpha, source_motion, torch.zeros_like(source_alpha)],
                    dim=-1,
                ).contiguous()
                warped_source_mask_rgb, _, _ = build_dense_target_reprojection_softsplat(
                    source_rgb=source_mask_rgb,
                    source_depth=src_depth,
                    source_depth_valid=src_valid,
                    src_cam2world=src_c2w,
                    tgt_cam2world=tgt_c2w,
                    intrinsics=K,
                    image_h=H,
                    image_w=W,
                    target_intrinsics=K_tgt,
                )
                if enc_in_channels == 8:
                    dense_source_alpha_map = warped_source_mask_rgb[..., 0].clamp(0.0, 1.0)
                dense_source_motion_map = warped_source_mask_rgb[..., 1].clamp(0.0, 1.0)
            # warped_rgb:   (B, T, H, W, 3) float32 [0, 1]
            # warped_depth: (B, T, H, W) float32 in target-camera space
            # warp_valid:   (B, T, H, W) bool

            # ------------------------------------------------------------------
            # 3. Pack into dense map
            # ------------------------------------------------------------------
            warped_depth_norm = _normalize_warped_depth_per_frame(warped_depth, warp_valid)
            warp_valid_f = warp_valid.float()

            depth_chw = warped_depth_norm.unsqueeze(1)                 # (B, 1, T, H, W)
            valid_chw = warp_valid_f.unsqueeze(1)                      # (B, 1, T, H, W)

            # Pack according to the dense encoder's input contract:
            #   8 channels: [RGB, depth, source_alpha, source_motion,
            #               target_motion, valid]. Backward-compatible full
            #               Vista4D 2x2 mask bundle.
            #   7 channels: [RGB, depth, source_motion, target_motion, valid].
            #               Active Vista4D-D ablation layout: omits all-one
            #               source alpha and keeps only nontrivial mask signals.
            #   6 channels: [RGB, depth, target_motion, valid]
            #   5 channels: [RGB, depth, valid]  (standard RGBD)
            #   4 channels: [RGB, valid]         (RGB-only ablation; depth is
            #                                     used only for warp geometry)
            #   2 channels: [depth, valid]       (latent_fuse geom-only path)
            use_geom_only = (
                warped_condition_mode == "latent_fuse" and warped_geom_only_dense
            )
            if use_geom_only or enc_in_channels == 2:
                if enc_in_channels != 2:
                    raise ValueError(
                        "DenseTargetControl: geom-only dense path requires "
                        f"encoder.in_channels=2, got {enc_in_channels}"
                    )
                # Geometry-only: (B, 2, T, H, W) — [depth, valid]
                dense_maps = torch.cat([depth_chw, valid_chw], dim=1)
            elif enc_in_channels == 4:
                # RGB-only: (B, 4, T, H, W) — [RGB, valid]
                rgb_chw = warped_rgb.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, T, H, W)
                dense_maps = torch.cat([rgb_chw, valid_chw], dim=1)
            elif enc_in_channels == 5:
                # Standard: (B, 5, T, H, W) — [RGB, depth, valid]
                rgb_chw = warped_rgb.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, T, H, W)
                dense_maps = torch.cat([rgb_chw, depth_chw, valid_chw], dim=1)  # (B, 5, T, H, W)
            elif enc_in_channels == 6:
                # Vista4D-strict: (B, 6, T, H, W) — [RGB, depth, motion, valid].
                # If a non-hybrid caller asks for this layout without a target
                # motion mask, keep the channel well-defined as all-zero.
                rgb_chw = warped_rgb.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, T, H, W)
                if dense_motion_mask is None:
                    motion_chw = torch.zeros_like(valid_chw)
                else:
                    motion = dense_motion_mask.to(device=device).bool()
                    if motion.shape != warp_valid.shape:
                        raise ValueError(
                            "DenseTargetControl: motion mask shape "
                            f"{tuple(motion.shape)} != warp_valid shape "
                            f"{tuple(warp_valid.shape)}"
                        )
                    motion_chw = motion.float().unsqueeze(1)
                dense_maps = torch.cat(
                    [rgb_chw, depth_chw, motion_chw, valid_chw], dim=1
                )  # (B, 6, T, H, W)
            elif enc_in_channels == 7:
                # Vista4D nontrivial mask information through the dense-control path:
                # [RGB, depth/null, source_motion, target_motion, valid].
                # `valid` is the point-cloud alpha mask after splatting; the
                # source video alpha is all-one for current Track4D datasets.
                rgb_chw = warped_rgb.permute(0, 4, 1, 2, 3).contiguous()
                if dense_source_motion_map is None:
                    source_motion_chw = torch.zeros_like(valid_chw)
                else:
                    source_motion_chw = dense_source_motion_map.float().unsqueeze(1)
                if dense_motion_mask is None:
                    target_motion_chw = torch.zeros_like(valid_chw)
                else:
                    motion = dense_motion_mask.to(device=device).bool()
                    if motion.shape != warp_valid.shape:
                        raise ValueError(
                            "DenseTargetControl: target motion mask shape "
                            f"{tuple(motion.shape)} != warp_valid shape "
                            f"{tuple(warp_valid.shape)}"
                        )
                    target_motion_chw = motion.float().unsqueeze(1)
                dense_maps = torch.cat(
                    [
                        rgb_chw,
                        depth_chw,
                        source_motion_chw,
                        target_motion_chw,
                        valid_chw,
                    ],
                    dim=1,
                )  # (B, 7, T, H, W)
            elif enc_in_channels == 8:
                # Full Vista4D mask bundle through the dense-control path:
                # [RGB, depth/null, source_alpha, source_motion,
                #  target_motion, pointcloud_alpha].
                rgb_chw = warped_rgb.permute(0, 4, 1, 2, 3).contiguous()
                if dense_source_alpha_map is None:
                    source_alpha_chw = torch.ones_like(valid_chw)
                else:
                    source_alpha_chw = dense_source_alpha_map.float().unsqueeze(1)
                if dense_source_motion_map is None:
                    source_motion_chw = torch.zeros_like(valid_chw)
                else:
                    source_motion_chw = dense_source_motion_map.float().unsqueeze(1)
                if dense_motion_mask is None:
                    target_motion_chw = torch.zeros_like(valid_chw)
                else:
                    motion = dense_motion_mask.to(device=device).bool()
                    if motion.shape != warp_valid.shape:
                        raise ValueError(
                            "DenseTargetControl: target motion mask shape "
                            f"{tuple(motion.shape)} != warp_valid shape "
                            f"{tuple(warp_valid.shape)}"
                        )
                    target_motion_chw = motion.float().unsqueeze(1)
                dense_maps = torch.cat(
                    [
                        rgb_chw,
                        depth_chw,
                        source_alpha_chw,
                        source_motion_chw,
                        target_motion_chw,
                        valid_chw,
                    ],
                    dim=1,
                )  # (B, 8, T, H, W)
            else:
                raise ValueError(
                    "DenseTargetControl: unsupported dense encoder channel "
                    f"count {enc_in_channels}; expected 2, 4, 5, 6, 7, or 8."
                )
            # Match encoder dtype (it lives on pipe with pipe.torch_dtype)
            try:
                enc_dtype = next(encoder.parameters()).dtype
            except StopIteration:
                enc_dtype = dtype
            dense_maps = dense_maps.to(dtype=enc_dtype)

            # ------------------------------------------------------------------
            # 5. Encode dense maps → block tokens
            # ------------------------------------------------------------------
            dense_block_tokens, dense_block_mask = encoder(
                dense_maps, latent_t, latent_h, latent_w
            )
            # dense_block_tokens: (B, num_blocks, out_dim)

            # ------------------------------------------------------------------
            # 6. Expand block tokens → full target-domain latent sequence
            # ------------------------------------------------------------------
            # Use the persistent adapter's block-grid layout when available so the
            # two streams share the same expansion (1:1 by default in this repo).
            block_t = getattr(pipe.track_adapter, "block_t", 1) if hasattr(pipe, "track_adapter") else 1
            block_h = getattr(pipe.track_adapter, "block_h", 1) if hasattr(pipe, "track_adapter") else 1
            block_w = getattr(pipe.track_adapter, "block_w", 1) if hasattr(pipe, "track_adapter") else 1
            dense_ctrl_tokens = expand_block_tokens_to_latent_seq(
                dense_block_tokens, latent_t, latent_h, latent_w, block_t, block_h, block_w,
            )
            # (B, latent_t * latent_h * latent_w, out_dim)

            if track_occupancy_channel_enabled():
                dense_occ_seq = expand_block_tokens_to_latent_seq(
                    dense_block_mask.unsqueeze(-1).to(dtype=dense_ctrl_tokens.dtype),
                    latent_t, latent_h, latent_w, block_t, block_h, block_w,
                )
                # (B, L_target, 1) — 1.0 where the dense encoder observed the block

            # --------------------------------------------------------------
            # 6b. Opt-in Vista4D-style continuous noise augmentation on dense
            # conditioning tokens. Backward-compatible: max_level=0.0 (default)
            # is a no-op. When >0 during scheduler training with grads enabled,
            # samples
            # level ~ U(0, max_level) per batch and adds Gaussian noise scaled
            # to match the current token std — forces the DiT to be robust to
            # imperfect dense conditioning (noisy depth, bad soft-splat warps,
            # recon artifacts). Skipped under torch.no_grad() so eval is
            # unchanged.
            # --------------------------------------------------------------
            _noise_max = float(getattr(pipe, "dense_conditioning_noise_max_level", 0.0))
            _training = bool(getattr(getattr(pipe, "scheduler", None), "training", False))
            dense_ctrl_tokens = _apply_condition_token_noise(
                dense_ctrl_tokens,
                _noise_max,
                training=_training,
            )

        # ------------------------------------------------------------------
        # 7. Fuse geometry (dense) tokens with persistent track tokens.
        #
        # Two fusion strategies:
        #   - Default (old): additive:  fused = dense + alpha * track
        #   - latent_fuse mode (new):   fused = concat_proj(cat(dense, track))
        #     Uses a learned GeometryTrackFuser for concat+project.
        # ------------------------------------------------------------------
        use_concat_fuse = (
            warped_condition_mode == "latent_fuse"
            and getattr(pipe, "geometry_track_fuser", None) is not None
        )

        # With the opt-in occupancy channel, persistent tokens from the track
        # unit carry one extra structural channel. Strip it before fusion (the
        # fusers are sized for the legacy dim; occupancy must be OR-combined,
        # never summed), then re-append the combined occupancy afterward.
        occupancy_enabled = track_occupancy_channel_enabled()
        persistent_occ = None
        if track_point_tokens is not None:
            persistent = track_point_tokens.to(
                device=dense_ctrl_tokens.device, dtype=dense_ctrl_tokens.dtype
            )
            if occupancy_enabled:
                if persistent.shape[-1] != dense_ctrl_tokens.shape[-1] + 1:
                    raise ValueError(
                        "TRACK4D360_TRACK_OCCUPANCY_CHANNEL=1: persistent tokens "
                        f"must carry dense_dim+1 channels, got persistent="
                        f"{persistent.shape[-1]} dense={dense_ctrl_tokens.shape[-1]}"
                    )
                persistent_occ = persistent[..., -1:]
                persistent = persistent[..., :-1]
            if persistent.shape[:-1] != dense_ctrl_tokens.shape[:-1]:
                raise ValueError(
                    f"Dense/persistent token spatial mismatch: "
                    f"dense={tuple(dense_ctrl_tokens.shape[:-1])}, "
                    f"persistent={tuple(persistent.shape[:-1])}"
                )

            if use_concat_fuse:
                # New mode: learned concat+project fusion
                fused_tokens = pipe.geometry_track_fuser(dense_ctrl_tokens, persistent)
            else:
                # Old mode: additive fusion
                if persistent.shape != dense_ctrl_tokens.shape:
                    raise ValueError(
                        f"Dense/persistent token shape mismatch (additive requires same dim): "
                        f"dense={tuple(dense_ctrl_tokens.shape)}, "
                        f"persistent={tuple(persistent.shape)}"
                    )
                alpha_track = float(getattr(pipe, "dense_branch_alpha_track", 1.0))
                fused_tokens = dense_ctrl_tokens + alpha_track * persistent
        else:
            fused_tokens = dense_ctrl_tokens

        if occupancy_enabled:
            # OR-combine (max) dense and persistent occupancy; missing streams
            # count as unobserved. The channel stays structural end to end.
            occ_parts = [
                occ for occ in (dense_occ_seq, persistent_occ) if occ is not None
            ]
            if occ_parts:
                combined_occ = occ_parts[0]
                for occ in occ_parts[1:]:
                    combined_occ = torch.maximum(
                        combined_occ, occ.to(device=combined_occ.device, dtype=combined_occ.dtype)
                    )
            else:
                combined_occ = torch.zeros_like(fused_tokens[..., :1])
            fused_tokens = torch.cat(
                [fused_tokens, combined_occ.to(device=fused_tokens.device, dtype=fused_tokens.dtype)],
                dim=-1,
            )

        # ------------------------------------------------------------------
        # 8. Rebuild the track injection hook around ``fused_tokens``.
        #
        # Two injection paths exist:
        #
        #   * "single"    — one projector fires once per forward via
        #                   ``_pre_block_hook``.  ``pipe.track_projector`` is
        #                   a ``TokenConcatProjector`` and ``track_adapter``
        #                   supplies the ``null_token`` used to pad the
        #                   source half of the V2V sequence.
        #
        #   * "per_block" — each DiT block calls ``_track_align_fn`` to
        #                   obtain the aligned track tokens (cached after
        #                   first call) and ``_track_block_injector`` to
        #                   inject them after layernorm+modulate.  In this
        #                   mode ``track_projector is None``.
        #
        # For dense control to actually reach the model in **both** modes we
        # must rebuild whichever hook that mode's forward pass consumes.
        # ------------------------------------------------------------------
        adapter = getattr(pipe, "track_adapter", None)
        injection_mode = getattr(pipe, "track_injection_mode", "single")

        if adapter is None:
            # No adapter at all → can't align tokens into the full V2V
            # sequence.  Expose fused tokens for downstream units that might
            # consume them directly, but don't pretend to wire a hook.
            return {
                "dense_ctrl_tokens": dense_ctrl_tokens,
                "track_point_tokens": fused_tokens,
            }

        if injection_mode == "per_block":
            # ReCamMaster-style: rebuild ``_track_align_fn`` around the fused
            # tokens so the per-block injector sees dense + persistent.
            block_injector = getattr(pipe, "track_block_injector", None)
            if block_injector is None:
                # Odd state: per_block mode without a registered injector.
                # Fall back to simply exposing fused tokens.
                return {
                    "dense_ctrl_tokens": dense_ctrl_tokens,
                    "track_point_tokens": fused_tokens,
                }

            def _make_fused_per_block_align(tpt, adap):
                # Keep grad-enabled and no-grad forwards separate. The GS
                # feedback training loss runs a bootstrap no-grad forward
                # before the real training forward; sharing a single cache
                # would leak a detached aligned-token tensor into the trainable
                # pass and cut gradients to the dense/track encoders.
                _cached = {"grad": None, "nograd": None}

                def align_fn(x, f, h, w, num_domains=2, v2v_concat_mode="width"):
                    cache_key = "grad" if torch.is_grad_enabled() else "nograd"
                    if _cached[cache_key] is None:
                        _cached[cache_key] = _align_track_to_sequence(
                            tpt, adap, x, f, h, w, num_domains, v2v_concat_mode
                        )
                    return _cached[cache_key]

                return align_fn

            fused_align_fn = _make_fused_per_block_align(fused_tokens, adapter)
            return {
                "dense_ctrl_tokens": dense_ctrl_tokens,
                "track_point_tokens": fused_tokens,
                "_track_align_fn": fused_align_fn,
                "_track_block_injector": block_injector,
            }

        # -------- single-projector mode -----------------------------------
        projector = getattr(pipe, "track_projector", None)
        if projector is None:
            # Single mode but no projector registered — same graceful exit.
            return {
                "dense_ctrl_tokens": dense_ctrl_tokens,
                "track_point_tokens": fused_tokens,
            }

        def _make_fused_hook(tpt, proj, adap):
            def hook(x, f, h, w, num_domains=2, v2v_concat_mode="width"):
                if tpt is None:
                    return x
                track_full = _align_track_to_sequence(
                    tpt, adap, x, f, h, w, num_domains, v2v_concat_mode
                )
                return proj(x, track_full)
            return hook

        fused_hook = _make_fused_hook(fused_tokens, projector, adapter)

        return {
            "dense_ctrl_tokens": dense_ctrl_tokens,
            "track_point_tokens": fused_tokens,
            "_pre_block_hook": fused_hook,
        }


__all__ = [
    "WanVideoUnit_DenseTargetControl",
    "_align_track_to_sequence",
    "_normalize_warped_depth_per_frame",
    "PipelineUnit",
    "DenseTargetControlEncoder",
]
