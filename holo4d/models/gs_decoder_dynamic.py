"""
Dynamic 3D Gaussian Splatting decoder for Track4D-360.

Adds a "canonical first-frame Gaussians + per-frame deltas" head on top of the
existing static :class:`holo4d.models.gs_decoder.GaussianSplattingDecoder`
**without modifying it**. The static decoder is composed (not subclassed and
not deep-copied), so its weights are shared between the two paths if both
ever run in one process.

Architecture
------------

::

    DiT → gs_features                  (B, feat_dim, T_lat, H', W')
                  │
                  └─ slice frame-0 → DynamicGaussianSplattingDecoder.forward()
                                │
                                ├─ base.decode(feats[:, :, 0:1], z[:, :, 0:1])
                                │     → x0 (B, last_dim, H, W)  full-res 2D map
                                │       (T=1 slice decoded via the base's
                                │        per-frame autoregressive decoder; the
                                │        first-frame feat_cache init skips the
                                │        time_conv path, so output is T=1 as
                                │        documented in gs_decoder.decode().)
                                │
                                ├─ base.gs_head(x0, target_camera[:, 0])
                                │     → canonical_params (B, npp, H, W, D)
                                │
                                └─ self.dynamic_head(x0)
                                      → delta_params (B, T_video-1, npp, H, W, ch_per_g)

Wan-DiT applies spatio-temporal attention through every block, so by the time
we read ``gs_features[:, :, 0]``, those features already encode information
about all T_lat frames. This is what makes the "frame-0 features only"
shortcut sound: a single (H', W') feature map is enough to predict deltas for
every future target-video frame in one shot.

Full-timeline supervision
-------------------------

The 2026-04-09 refactor to the base static decoder made it emit one cloud
per target-video frame (``T_video = (T_lat - 1) * 4 + 1``), so the static GS
loss now supervises at ``T_video`` frames rather than ``T_lat``. The dynamic
path follows suit: the dynamic head predicts up to ``max_t_video - 1``
deltas, and the caller slices down to ``t_video - 1`` at runtime. For a
49-frame clip (T_lat=13), this means 48 delta slots rather than 12.

Compute cost
------------

The dynamic decoder runs the VAE-style ``base.decode`` loop on a single
latent frame (``T_lat=1``), so the autoregressive time_conv path is skipped
and the whole decode is one ResidualBlock/upsample pass. This is cheaper
than the static GS path, which runs the full ``T_lat → T_video`` loop. The
only added cost is one Conv2d projection (the dynamic head).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gs_decoder import GaussianSplattingDecoder


# ---------------------------------------------------------------------------
# DynamicGSHead — single Conv2d with zero weights + carefully-chosen bias
# ---------------------------------------------------------------------------


class DynamicGSHead(nn.Module):
    """Per-pixel per-target-video-frame delta head.

    Predicts ``(max_t_video - 1) * num_points_per_pixel * channels_per_gaussian``
    output channels per spatial location, in **one shot**, from the
    full-resolution post-decoder feature map ``x0`` of frame 0.

    Channels per Gaussian:
      - ``Δxyz``            (3)
      - ``Δopacity_logit``  (1)
      - ``Δquat``           (4, optional, only if ``predict_rotation=True``)

    Zero-init invariant
    -------------------

    The Conv2d weights are **always** zero-initialised so that, at the very
    first optimiser step, the learned-delta contribution is exactly zero for
    any input ``x``. What the head outputs at init is therefore fully
    determined by the Conv2d bias.

    - ``predict_rotation=False``: bias = 0 → delta is all zero → canonical +
      delta == canonical at init.
    - ``predict_rotation=True``: bias on the ``Δquat`` channels is set to the
      **identity quaternion** ``(1, 0, 0, 0)`` (not zero!) so that at init
      ``F.normalize(delta_quat) == identity`` and the Hamilton product
      ``canonical_quat * identity_quat == canonical_quat``. Setting that bias
      to zero would zero-collapse the rotation at init because
      ``F.normalize((0, 0, 0, 0)) == (0, 0, 0, 0)`` — the fix for
      Codex finding #2.
    """

    def __init__(
        self,
        embed_dim: int,
        num_points_per_pixel: int,
        max_t_video: int,
        predict_rotation: bool = False,
    ):
        super().__init__()
        if max_t_video < 1:
            raise ValueError(f"max_t_video must be >= 1, got {max_t_video}")
        self.embed_dim = int(embed_dim)
        self.num_points_per_pixel = int(num_points_per_pixel)
        self.max_t_video = int(max_t_video)
        self.predict_rotation = bool(predict_rotation)

        self.ch_per_gaussian = 4 + (4 if self.predict_rotation else 0)
        self.t_minus_1 = self.max_t_video - 1

        out_ch = self.t_minus_1 * self.num_points_per_pixel * self.ch_per_gaussian
        self.proj = nn.Conv2d(
            self.embed_dim, out_ch, kernel_size=3, stride=1, padding=1,
        )

        with torch.no_grad():
            self.proj.weight.data.zero_()
            if self.predict_rotation:
                # Bias layout after reshape is (t_minus_1, npp, ch_per_gaussian)
                # flattened row-major into (t_minus_1 * npp * ch_per_gaussian,).
                # Per-Gaussian layout: [Δxyz(3), Δopa_logit(1), Δquat(4)] where
                # Δquat = (r, i, j, k) with r first. Identity quat = (1,0,0,0).
                # So per-gaussian bias = [0, 0, 0, 0, 1, 0, 0, 0].
                bias_per_g = torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32,
                )
                bias_full = bias_per_g.repeat(self.t_minus_1 * self.num_points_per_pixel)
                self.proj.bias.data.copy_(bias_full)
            else:
                self.proj.bias.data.zero_()

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a full-resolution feature map into per-frame deltas.

        Args:
            x: ``(B, embed_dim, H, W)`` post-``decoder.head`` features for
                frame 0. Operates in float32 to keep delta predictions stable
                regardless of the AMP context.

        Returns:
            ``(B, t_minus_1, num_points_per_pixel, H, W, ch_per_gaussian)``
            tensor of per-frame, per-pixel delta predictions.

        Notes:
            - The temporal dimension is fixed to ``t_minus_1 = max_t_video - 1``
              and is sliced down to the actual ``T_video - 1`` by the caller.
            - Output is in float32; the caller is free to cast.
        """
        x = x.to(torch.float32)
        B, _, H, W = x.shape
        # Run conv in float32 regardless of the stored dtype of self.proj.
        # The module is cast to the decoder's dtype (typically bf16) at
        # construction, but this forward is @autocast(enabled=False), so the
        # nn.Conv2d module call would fail with "Input type (float) and bias
        # type (BFloat16) should be the same". Mirrors PixelAligned3DGS's
        # F.conv2d pattern (gs_decoder.py:148-153).
        raw = F.conv2d(
            x,
            self.proj.weight.to(x.dtype),
            self.proj.bias.to(x.dtype) if self.proj.bias is not None else None,
            stride=self.proj.stride,
            padding=self.proj.padding,
            dilation=self.proj.dilation,
            groups=self.proj.groups,
        )  # (B, t_minus_1*npp*ch, H, W)
        # (B, t_minus_1, npp, ch, H, W) → (B, t_minus_1, npp, H, W, ch)
        raw = raw.view(
            B,
            self.t_minus_1,
            self.num_points_per_pixel,
            self.ch_per_gaussian,
            H,
            W,
        )
        raw = raw.permute(0, 1, 2, 4, 5, 3).contiguous()
        return raw


# ---------------------------------------------------------------------------
# DynamicGaussianSplattingDecoder — composes a static GaussianSplattingDecoder
# ---------------------------------------------------------------------------


class DynamicGaussianSplattingDecoder(nn.Module):
    """Wraps a :class:`GaussianSplattingDecoder` and adds a dynamic head.

    The wrapped decoder is **the same instance** the static GS path uses, so
    its parameters are shared between the two paths. Construction does NOT
    deep-copy or modify the base decoder.

    Forward inputs / outputs follow this contract:

    Inputs
    ------
    feats : ``(B, feat_dim, T_lat, H', W')`` per-latent-frame DiT features.
        Same layout as ``gs_features`` from
        ``holo4d_runtime/pipeline/wan_video_v2v.py``.
    z : ``(B, z_dim, T_lat, H', W')`` predicted clean latents (detached by
        the caller, exactly as in the static GS loss path).
    canonical_camera : ``(B, 11)`` FlashWorld 11-param target camera at
        target-video frame 0. Used by ``base.gs_head`` to compute the
        canonical 3D positions via ray-casting.
    t_video : int number of target-video frames the caller needs deltas
        for. Must satisfy ``1 <= t_video <= max_t_video``.

    Outputs
    -------
    canonical_params : ``(B, npp, H, W, D)`` canonical Gaussian parameters in
        cam0 world space (xyz + opacity + scale + rotation + SH). Produced
        from frame 0 only — Wan-DiT's spatio-temporal attention means these
        features already encode information about all target-video frames.
    delta_params : ``(B, t_video - 1, npp, H, W, ch_per_gaussian)`` per-frame
        per-pixel delta predictions for frames ``1 .. t_video - 1``.
        ``ch_per_gaussian`` is 4 (``Δxyz + Δopacity_logit``) by default, or
        8 (``Δxyz + Δopacity_logit + Δquat``) if ``predict_rotation=True``.
    """

    def __init__(
        self,
        base_decoder: GaussianSplattingDecoder,
        max_t_video: int,
        predict_rotation: bool = False,
    ):
        super().__init__()
        # Store as a plain Python ref (NOT a registered submodule) to avoid a
        # circular module tree: gs_decoder.dynamic.base → gs_decoder would
        # cause infinite recursion in .train()/.eval().
        object.__setattr__(self, '_base', base_decoder)
        self.max_t_video = int(max_t_video)
        self.predict_rotation = bool(predict_rotation)

        # Read embed_dim from the base's existing GS projection so we never
        # have to know the VAE dim_mult conventions ourselves.
        embed_dim = int(base_decoder.gs_head.gs_proj.in_channels)
        npp = int(base_decoder.gs_head.num_points_per_pixel)

        self.dynamic_head = DynamicGSHead(
            embed_dim=embed_dim,
            num_points_per_pixel=npp,
            max_t_video=self.max_t_video,
            predict_rotation=self.predict_rotation,
        )

    @property
    def num_points_per_pixel(self) -> int:
        return int(self._base.gs_head.num_points_per_pixel)

    @property
    def sh_degree(self) -> int:
        return int(self._base.gs_head.sh_degree)

    @property
    def render_checkpointing(self) -> bool:
        return bool(self._base.render_checkpointing)

    @property
    def render_patch_size(self) -> "tuple[int, int] | None":
        return getattr(self._base, "render_patch_size", None)

    def forward(
        self,
        feats: torch.Tensor,
        z: torch.Tensor,
        canonical_camera: torch.Tensor,
        t_video: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode canonical Gaussians + per-frame deltas over the target-video
        timeline.

        Args:
            feats: ``(B, feat_dim, T_lat, H', W')`` DiT features.
            z: ``(B, z_dim, T_lat, H', W')`` predicted clean latents.
            canonical_camera: ``(B, 11)`` FlashWorld camera at target-video
                frame 0.
            t_video: actual number of target-video frames in this batch
                (used to slice the dynamic head's output down from
                ``max_t_video - 1`` to ``t_video - 1``). Must satisfy
                ``1 <= t_video <= max_t_video``.

        Returns:
            Tuple ``(canonical_params, delta_params)`` with shapes
            ``(B, npp, H, W, D)`` and
            ``(B, t_video - 1, npp, H, W, ch_per_g)``.
        """
        if feats.dim() != 5 or z.dim() != 5:
            raise ValueError(
                f"feats and z must be 5D (B, C, T_lat, H', W'); "
                f"got feats={tuple(feats.shape)}, z={tuple(z.shape)}"
            )
        if t_video < 1 or t_video > self.max_t_video:
            raise ValueError(
                f"t_video={t_video} out of range [1, {self.max_t_video}]"
            )

        # 1. Slice frame 0 of features and latents — keep T=1 dim for the VAE
        #    decoder which expects 5D (B, C, 1, H', W'). With T_lat=1 the
        #    base decoder's per-frame autoregressive path skips time_conv,
        #    so the output is also T=1 (see gs_decoder.decode() docstring).
        feats0 = feats[:, :, 0:1]
        z0 = z[:, :, 0:1]

        # 2. Run the base decoder body on frame 0 ONLY.
        x0_seq = self._base.decode(feats0, z0)  # (B, last_dim, 1, H, W)
        if x0_seq.dim() != 5 or x0_seq.shape[2] != 1:
            raise ValueError(
                f"Expected base.decode(T_lat=1) to return T=1, got shape "
                f"{tuple(x0_seq.shape)}"
            )
        x0 = x0_seq.squeeze(2)  # (B, last_dim, H, W)

        # 3. Canonical Gaussians via the existing pixel-aligned head.
        canonical_params = self._base.gs_head(x0, canonical_camera)  # (B, npp, H, W, D)

        # 4. Per-frame deltas via the new dynamic head, sliced down to actual
        #    t_video - 1. Frame 0 has an implicit "zero delta" — the loss
        #    uses the canonical cloud directly at t_idx = 0.
        delta_full = self.dynamic_head(x0)  # (B, max_t_video-1, npp, H, W, ch_per_g)
        if t_video == 1:
            delta_params = delta_full[:, :0]  # empty time dim
            # Loss only supervises canonical at t=0, so delta_full would otherwise
            # be disconnected (DDP unused-parameter on dynamic_head).
            canonical_params = canonical_params + (delta_full * 0).sum()
        else:
            delta_params = delta_full[:, : t_video - 1]

        return canonical_params, delta_params

    @staticmethod
    def split_delta(delta_params: torch.Tensor, predict_rotation: bool):
        """Split a delta-params tensor along the channel dim.

        Args:
            delta_params: ``(..., 4)`` or ``(..., 8)`` last-dim layout
                ``[Δxyz(3), Δopacity_logit(1), (Δquat(4) optional)]``.
            predict_rotation: whether the trailing 4 channels are present.

        Returns:
            Tuple ``(delta_xyz, delta_opa_logit, delta_quat_or_none)``.
        """
        if predict_rotation:
            delta_xyz, delta_opa, delta_quat = delta_params.split([3, 1, 4], dim=-1)
            return delta_xyz, delta_opa, delta_quat
        delta_xyz, delta_opa = delta_params.split([3, 1], dim=-1)
        return delta_xyz, delta_opa, None
