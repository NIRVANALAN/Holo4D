"""
3D Gaussian Splatting decoder for Track4D-360.

Adapts FlashWorld's WANDecoderPixelAligned3DGSReconstructionModel
for Wan 2.1's Decoder3d architecture.

Architecture:
  DiT features (feat_dim) + predicted latents (z_dim)
    → copied VAE Decoder3d (extra_conv_in for features, original conv1 for z)
    → spatiotemporal decode back to the target video timeline
    → PixelAligned3DGS head → per-pixel Gaussian parameters
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gs_camera_utils import create_rays, quaternion_to_matrix


# ---------------------------------------------------------------------------
# Truncated exponential (gradient-safe depth activation)
# ---------------------------------------------------------------------------

class _TruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-10, 10))


trunc_exp = _TruncExp.apply


def inverse_sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x / (1 - x))
    return math.log(x / (1 - x))


# ---------------------------------------------------------------------------
# PixelAligned3DGS — predicts Gaussian params per pixel
# ---------------------------------------------------------------------------

class PixelAligned3DGS(nn.Module):
    """Pixel-aligned 3D Gaussian Splatting head.

    Converts feature maps into per-pixel Gaussian parameters using a single
    Conv2d projection, then computes 3D positions via ray-casting with
    predicted depth.

    Per-Gaussian parameters (38 total with sh_degree=2, no mask):
      - SH coefficients: 3 * (sh_degree+1)^2 = 27
      - UV offset: 2 (sub-pixel perturbation)
      - Depth: 1 (log-space, activated via truncated exp)
      - Opacity: 1 (sigmoid)
      - Scales: 3 (pixel-size-relative, sigmoid)
      - Rotations: 4 (quaternion, L2-normalized)
    """

    def __init__(
        self,
        embed_dim: int,
        sh_degree: int = 2,
        use_mask: bool = False,
        scale_range: tuple[float, float] = (0, 16),
        num_points_per_pixel: int = 2,
        scale_cap: "float | None" = None,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.use_mask = use_mask
        self.scale_range = scale_range
        self.num_points_per_pixel = num_points_per_pixel
        self.scale_cap = scale_cap

        # Per-Gaussian channel layout
        self.gaussian_channels = [
            3 * (sh_degree + 1) ** 2,  # SH coefficients
            2,                          # UV offset
            1,                          # depth
            1,                          # opacity
            3,                          # scales
            4,                          # rotations
            (1 if use_mask else 0),     # optional mask
        ]
        total_channels = sum(self.gaussian_channels)

        self.gs_proj = nn.Conv2d(
            embed_dim, num_points_per_pixel * total_channels, 3, stride=1, padding=1,
        )

        # Learning rate multipliers (applied at forward time via weight scaling)
        lrs = (
            [1.0] * 3                                               # SH DC
            + [0.5] * (3 * ((sh_degree + 1) ** 2 - 1))             # SH higher orders
            + [0.01] * 2                                            # UV offset
            + [1.0] * 1                                             # depth
            + [1.0] * 1                                             # opacity
            + [1.0] * 3                                             # scales
            + [1.0] * 4                                             # rotations
            + [0.1] * (1 if use_mask else 0)                        # mask
        )
        lrs_mul = torch.tensor(lrs).repeat(num_points_per_pixel)
        lrs_mul = lrs_mul / lrs_mul.max()
        self.register_buffer("lrs_mul", lrs_mul, persistent=True)

        # Zero-init weights, carefully chosen bias
        with torch.no_grad():
            self.gs_proj.weight.data.zero_()
            bias_single = torch.tensor(
                [0.0] * (3 * (sh_degree + 1) ** 2)                  # SH
                + [0.0] * 2                                          # UV offset
                + [math.log(1)] * 1                                  # depth (exp(0)=1)
                + [inverse_sigmoid(0.1)] * 1                         # opacity (10%)
                + [inverse_sigmoid(
                    (1 - scale_range[0]) / (scale_range[1] - scale_range[0])
                )] * 3                                               # scales (pixel-size)
                + [1.0, 0, 0, 0]                                    # rotation (identity quat)
                + [inverse_sigmoid(0.9)] * (1 if use_mask else 0)    # mask
            )
            self.gs_proj.bias = nn.Parameter(
                bias_single.repeat(num_points_per_pixel) / lrs_mul
            )

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(
        self, x: torch.Tensor, cameras: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (BN, C, H, W) feature map from decoder.
            cameras: (BN, 11) FlashWorld-format cameras.
        Returns:
            gaussian_params: (BN, num_pts, H, W, D) where
                D = 3(xyz) + 1(opacity) + 3(scale) + 4(rot) + SH_channels.
        """
        BN, _, h, w = x.shape

        # Project features → Gaussian params with LR-scaled weights
        raw = F.conv2d(
            x,
            self.gs_proj.weight * self.lrs_mul[:, None, None, None].to(x.dtype),
            self.gs_proj.bias * self.lrs_mul.to(x.dtype),
            stride=1, padding=1,
        ).unflatten(1, (self.num_points_per_pixel, -1))

        raw = raw.to(torch.float32)
        cameras = cameras.to(torch.float32)

        # (BN, npp, C, H, W) → (BN, npp, H, W, C)
        raw = raw.permute(0, 1, 3, 4, 2)

        features, uv_offset, depth, opacity, scales, rotations, mask = raw.split(
            self.gaussian_channels, dim=-1
        )

        # Ray-based 3D positions
        rays_o, rays_d = create_rays(
            cameras[:, None].repeat(1, self.num_points_per_pixel, 1),
            uv_offset=uv_offset, h=h, w=w,
        )
        depth = trunc_exp(depth)
        xyz = rays_o + depth * rays_d

        # Activations
        opacity = torch.sigmoid(opacity)
        if self.use_mask:
            if torch.is_grad_enabled():
                mask_sig = torch.sigmoid(mask)
                hard_mask = (mask_sig > torch.rand_like(mask_sig)).float()
                opacity = opacity * (mask_sig + (hard_mask - mask_sig).detach())
            else:
                hard_mask = (torch.sigmoid(mask) > torch.rand_like(mask)).float()
                opacity = opacity * hard_mask

        # Resolution-adaptive scale normalization (matches FlashWorld reconstruction_model.py:250-252).
        # NOTE: this is NOT the physical pixel footprint (which would be depth/fx_pixel).
        # It's a learned scale factor: cameras[:,7:9] = (fx_pixel/W, fy_pixel/H),
        # divided by (w, h) = feature map resolution. The network learns to compensate
        # via sigmoid(raw_scale). Changing this formula would break FlashWorld bias init.
        fx, fy = cameras[:, 7:9].split([1, 1], dim=-1)
        fx, fy = fx / w, fy / h
        pixel_size = torch.sqrt(fx.pow(2) + fy.pow(2))[:, None, None, None] * depth
        scales = (
            torch.sigmoid(scales) * (self.scale_range[1] - self.scale_range[0])
            + self.scale_range[0]
        ) * pixel_size
        # Lyra-1 hard cap: prevents deep Gaussians from exploding in world units.
        if self.scale_cap is not None:
            scales = torch.clamp(scales, max=self.scale_cap)

        rotations = F.normalize(rotations, dim=-1)

        # Pack: [xyz(3), opacity(1), scale(3), rotation(4), SH(...)]
        gaussian_params = torch.cat([xyz, opacity, scales, rotations, features], dim=-1)
        return gaussian_params


# ---------------------------------------------------------------------------
# GaussianSplattingDecoder — wraps adapted VAE decoder + GS head
# ---------------------------------------------------------------------------

class GaussianSplattingDecoder(nn.Module):
    """Decodes DiT features + predicted latents into 3D Gaussian parameters.

    Deep-copies the Wan 2.1 VAE Decoder3d, then:
    - Adds a zero-initialized extra_conv_in for feat_dim features
    - Keeps temporal upsampling to recover the full target timeline
    - Replaces the output head with PixelAligned3DGS

    Forward: feats (B*T, feat_dim, 1, H', W') + z (B*T, z_dim, 1, H', W')
           → Gaussian params at full resolution (B*T, npp, H, W, D)
    """

    def __init__(
        self,
        vae_decoder: nn.Module,
        feat_dim: int = 1024,
        num_points_per_pixel: int = 2,
        sh_degree: int = 2,
        scale_range: tuple[float, float] = (0, 16),
        use_network_checkpointing: bool = True,
        use_render_checkpointing: bool = True,
        scale_cap: "float | None" = None,
        render_patch_size: "tuple[int, int] | None" = None,
        render_per_frame_gaussians: bool = True,
    ):
        super().__init__()

        # Deep-copy the VAE decoder
        self.decoder = copy.deepcopy(vae_decoder).requires_grad_(True)

        # Determine channel dimensions
        # Decoder3d: dim=128, dim_mult=[1,2,4,4]
        # conv1 takes z_dim → dims[0] = 512
        dims = [self.decoder.dim * u for u in
                [self.decoder.dim_mult[-1]] + self.decoder.dim_mult[::-1]]
        first_dim = dims[0]   # 512
        last_dim = dims[-1]   # 128

        # Extra conv for feature injection (zero-initialized)
        # Import CausalConv3d from the same module as the decoder
        from holo4d_runtime.models.wan_video_vae import CausalConv3d

        self.extra_conv_in = CausalConv3d(feat_dim, first_dim, 3, padding=1)
        with torch.no_grad():
            self.extra_conv_in.weight.data.zero_()
            self.extra_conv_in.bias.data.zero_()
        # Replace output head: keep the pretrained RMS_norm + SiLU, drop only the
        # final CausalConv3d → RGB projection so we reuse the learned VAE head
        # normalization at init time (more reuse-faithful than FlashWorld, which
        # re-initializes norm_out).
        # Original head: Sequential(RMS_norm(last_dim), SiLU(), CausalConv3d(last_dim, 3, 3))
        self.decoder.head = nn.Sequential(
            self.decoder.head[0],  # pretrained RMS_norm
            self.decoder.head[1],  # SiLU (stateless)
        )

        # Attach GS head
        self.gs_head = PixelAligned3DGS(
            embed_dim=last_dim,
            sh_degree=sh_degree,
            num_points_per_pixel=num_points_per_pixel,
            scale_range=scale_range,
            scale_cap=scale_cap,
        )

        self.network_checkpointing = use_network_checkpointing
        self.render_checkpointing = use_render_checkpointing
        self.render_patch_size = render_patch_size
        self.render_per_frame_gaussians = bool(render_per_frame_gaussians)

        # Cast ALL newly-created layers (extra_conv_in, gs_head) to match
        # the deep-copied decoder's dtype (typically bf16). Without this,
        # mixed-precision forward hits "Input type vs bias type" errors.
        decoder_dtype = next(self.decoder.parameters()).dtype
        self.extra_conv_in = self.extra_conv_in.to(dtype=decoder_dtype)
        self.gs_head = self.gs_head.to(dtype=decoder_dtype)

    def _decode_frame(
        self,
        z_i: torch.Tensor,
        feats_i: torch.Tensor,
        feat_cache: list,
        feat_idx: list,
        extra_cache: list,
    ) -> torch.Tensor:
        """Forward one latent frame through the adapted Wan VAE decoder.

        Mirrors ``Decoder3d.forward``'s feat_cache threading for middle +
        upsamples + head, but (a) injects ``extra_conv_in(feats_i)`` at the
        conv1 step and (b) uses its own single-slot cache for
        ``extra_conv_in`` so the main ``feat_cache`` slot layout keeps
        matching ``count_conv3d(self.decoder)``.
        """
        from holo4d_runtime.models.wan_video_vae import (
            CACHE_T,
            CausalConv3d,
            ResidualBlock,
        )

        # --- conv1(z_i) with feat_cache slot (mirrors Decoder3d.forward) ---
        idx = feat_idx[0]
        cache_x_z = z_i[:, :, -CACHE_T:, :, :].clone()
        if cache_x_z.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x_z = torch.cat(
                [
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x_z.device),
                    cache_x_z,
                ],
                dim=2,
            )
        x = self.decoder.conv1(z_i, feat_cache[idx])
        feat_cache[idx] = cache_x_z
        feat_idx[0] += 1

        # --- extra_conv_in(feats_i) with its own single-slot cache ---
        cache_x_f = feats_i[:, :, -CACHE_T:, :, :].clone()
        if cache_x_f.shape[2] < 2 and extra_cache[0] is not None:
            cache_x_f = torch.cat(
                [
                    extra_cache[0][:, :, -1, :, :].unsqueeze(2).to(cache_x_f.device),
                    cache_x_f,
                ],
                dim=2,
            )
        x = x + self.extra_conv_in(feats_i, extra_cache[0])
        extra_cache[0] = cache_x_f

        # --- middle blocks ---
        for layer in self.decoder.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # --- upsample blocks (ResidualBlock + Resample + optional AttentionBlock) ---
        for layer in self.decoder.upsamples:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache, feat_idx)
            else:
                # Resample accepts (x, feat_cache, feat_idx);
                # AttentionBlock.forward(x) only — detect by arg count.
                try:
                    x = layer(x, feat_cache, feat_idx)
                except TypeError:
                    x = layer(x)

        # --- head (replaced: RMS_norm + SiLU; no CausalConv3d, no cache) ---
        for layer in self.decoder.head:
            x = layer(x)

        return x

    def _ddp_sink_skipped_temporal_convs(self, x: torch.Tensor) -> torch.Tensor:
        """Connect ``Resample.time_conv`` params to the autograd graph when they are skipped.

        Wan ``Decoder3d`` uses ``Resample(upsample3d)`` with a stateful ``feat_cache``.
        On the **first** latent frame, the cache-miss branch sets a sentinel and does
        **not** run ``time_conv`` (see ``wan_video_vae.Resample.forward``). Dynamic GS
        decodes with ``T_lat == 1`` only, so every ``time_conv`` in the decoder is
        skipped while still ``requires_grad=True`` — DDP then fails with
        "parameter indices did not receive gradient".

        Adding ``0 * p.sum()`` for each ``time_conv`` parameter keeps them in the
        graph with zero contribution to activations and zero gradient w.r.t. loss.
        """
        if not torch.is_grad_enabled():
            return x
        sink = (x * 0).sum()
        for mod in self.decoder.modules():
            tc = getattr(mod, "time_conv", None)
            if tc is not None:
                for p in tc.parameters():
                    sink = sink + p.sum() * 0
        return x + sink

    def decode(self, feats: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Per-frame autoregressive decode with feat_cache so the Wan VAE's
        temporal upsampling (time_conv in upsample3d Resample layers)
        actually fires.

        Why this is needed: ``Resample.forward``'s temporal-upsample branch
        is guarded by ``if feat_cache is not None:``. When the decoder is
        driven single-shot via sub-module iteration without feat_cache, the
        time_conv path is skipped and the temporal dim stays at T_lat. Full
        temporal upsampling only happens when the decoder is called
        frame-by-frame with a stateful ``feat_cache`` that threads the
        causal-padding context across frames. This mirrors the reference
        pattern in ``WanVAE_.decode`` (``holo4d_runtime/models/wan_video_vae.py``).

        Args:
            feats: ``(B, feat_dim, T_lat, H', W')`` DiT features for the
                target domain.
            z: ``(B, z_dim, T_lat, H', W')`` predicted clean latents.

        Returns:
            ``(B, last_dim, T_video, H, W)`` decoded features at full
            resolution and the full target video timeline, where
            ``T_video = (T_lat - 1) * 4 + 1`` for the standard
            ``temperal_upsample=[False, True, True]`` Wan VAE config. For
            ``T_lat == 1`` (dynamic path's frame-0 slice) the output is
            also T=1 because the first-frame ``feat_cache`` init skips
            time_conv.
        """
        from holo4d_runtime.models.wan_video_vae import count_conv3d

        T_lat = int(z.shape[2])

        # feat_cache slot count = number of CausalConv3d instances in
        # self.decoder. Our ``decoder.head`` replacement dropped the
        # original RGB CausalConv3d, so count_conv3d automatically reflects
        # the reduced count. extra_conv_in is a sibling of self.decoder (a
        # direct child of ``self``) and is NOT counted here — it uses its
        # own single-slot ``extra_cache``.
        n_conv = count_conv3d(self.decoder)

        def _run_loop() -> torch.Tensor:
            feat_cache: list = [None] * n_conv
            extra_cache: list = [None]
            out_chunks = []
            for i in range(T_lat):
                feat_idx = [0]
                z_i = z[:, :, i : i + 1]
                feats_i = feats[:, :, i : i + 1]
                out_i = self._decode_frame(
                    z_i, feats_i, feat_cache, feat_idx, extra_cache,
                )
                out_chunks.append(out_i)
            return torch.cat(out_chunks, dim=2)

        if self.network_checkpointing and torch.is_grad_enabled() and T_lat > 1:
            # Checkpoint the whole per-frame loop as a single unit. We cannot
            # checkpoint individual frames because feat_cache is mutated
            # across frames and backward-time recompute would see the wrong
            # cache state for earlier frames. Whole-loop checkpointing is
            # correct (same (feats, z) in → same output), at the cost of
            # one full-loop recompute during backward.
            return torch.utils.checkpoint.checkpoint(
                _run_loop, use_reentrant=False,
            )
        out = _run_loop()
        if T_lat == 1:
            out = self._ddp_sink_skipped_temporal_convs(out)
        return out

    def forward(
        self,
        feats: torch.Tensor,
        z: torch.Tensor,
        cameras: torch.Tensor,
        return_track_delta: bool = False,
    ) -> torch.Tensor:
        """Decode features into Gaussian parameters.

        Args:
            feats: (B, feat_dim, T_lat, H', W').
            z: (B, z_dim, T_lat, H', W').
            cameras: (B, T_video, 11) FlashWorld-format cameras.
            return_track_delta: v4 — also tap the head's
                ``predict_track_delta_xyz`` projector and return
                ``(gaussian_params, track_delta_xyz)``. Caller must have
                set ``gs_head.predict_track_delta=True`` (typically via
                ``gs_dynamic_mode=static_with_track`` in the trainer).
        Returns:
            gaussian_params: (B, T_video, npp, H, W, D), or
            (gaussian_params, (B, T_video, npp, H, W, 3)) when
            ``return_track_delta=True``.
        """
        if feats.dim() != 5 or z.dim() != 5:
            raise ValueError(
                f"GaussianSplattingDecoder expects 5D sequence tensors, got feats={tuple(feats.shape)}, z={tuple(z.shape)}"
            )
        if cameras.dim() == 2:
            cameras = cameras.unsqueeze(0)

        x = self.decode(feats, z)  # (B, last_dim, T_video, H, W)
        b, c, t_video, h, w = x.shape
        if cameras.shape[0] != b or cameras.shape[1] != t_video:
            raise ValueError(
                f"Decoded video timeline {tuple(x.shape)} does not match camera timeline {tuple(cameras.shape)}"
            )

        x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * t_video, c, h, w)
        cameras_flat = cameras.reshape(b * t_video, cameras.shape[-1])
        gaussian_params = self.gs_head(x_flat, cameras_flat)
        gaussian_params = gaussian_params.unflatten(0, (b, t_video))
        if return_track_delta:
            # v4 track-only delta (per-pixel-per-frame Δxyz). Shares the
            # same per-pixel features as the render head; never enters
            # rendering. Caller must have constructed gs_head with
            # `predict_track_delta=True`.
            track_delta_xyz = self.gs_head.predict_track_delta_xyz(x_flat)
            track_delta_xyz = track_delta_xyz.unflatten(0, (b, t_video))
            return gaussian_params, track_delta_xyz
        return gaussian_params

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def render(
        self,
        gaussian_params: torch.Tensor,
        cameras: torch.Tensor,
        height: int,
        width: int,
        bg_mode: str = "random",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render Gaussians to RGB + depth.

        Args:
            gaussian_params: (B, T, npp, H, W, D) or (B*T, npp, H, W, D).
            cameras: (B, T, 11) or (B*T, 11) FlashWorld-format cameras.
            height, width: render resolution.
            bg_mode: 'random', 'white', or 'black'.
        Returns:
            rgb: (B, T, 3, H, W) in [-1, 1].
            depth: (B, T, 1, H, W).
        """
        from .gs_renderer import gaussian_render

        cameras = cameras.to(torch.float32)

        # Build c2w and intrinsics from FlashWorld camera format
        if cameras.dim() == 2:
            cameras = cameras.unsqueeze(0)  # (N, 11) → (1, N, 11)
        if gaussian_params.dim() == 6:
            # (B, T, npp, H, W, D) → (B, T, npp*H*W, D)
            B, T, npp, gh, gw, D = gaussian_params.shape
            gaussian_params = gaussian_params.reshape(B, T, npp * gh * gw, D)
        elif gaussian_params.dim() == 5:
            # (B*T, npp, H, W, D) → (B*T, 1, npp*H*W, D)
            BT, npp, gh, gw, D = gaussian_params.shape
            gaussian_params = gaussian_params.reshape(BT, npp * gh * gw, D)
            gaussian_params = gaussian_params.unsqueeze(1)

        B, V = cameras.shape[:2]

        test_c2ws = torch.eye(4, device=cameras.device)[None, None].repeat(B, V, 1, 1).float()
        test_c2ws[:, :, :3, :3] = quaternion_to_matrix(cameras[:, :, :4])
        test_c2ws[:, :, :3, 3] = cameras[:, :, 4:7]

        fx, fy, cx, cy = cameras[:, :, 7:11].split([1, 1, 1, 1], dim=-1)
        test_intr = torch.cat([
            fx * width, fy * height, cx * width, cy * height,
        ], dim=-1)

        return gaussian_render(
            gaussian_params, test_c2ws, test_intr, width, height,
            use_checkpoint=self.render_checkpointing,
            sh_degree=self.gs_head.sh_degree,
            bg_mode=bg_mode,
            patch_size=self.render_patch_size,
            # New runs use per-frame video rendering: view t sees
            # gaussian_params[:, t] only. Old v3b checkpoints used the legacy
            # LRM-style union cloud, so this is configurable for reproduction.
            per_frame_gaussians=self.render_per_frame_gaussians,
        )
