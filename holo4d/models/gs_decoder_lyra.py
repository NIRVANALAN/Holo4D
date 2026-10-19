"""Lyra-1-style pixel-aligned 3DGS head for Track4D-360.

Drop-in replacement for ``PixelAligned3DGS`` and ``GaussianSplattingDecoder``
that uses Lyra-1's activation conventions:

  - **Direct RGB** (3ch, ``0.5 * tanh(x) + 0.5``) instead of SH
  - **Sigmoid-lerp depth** (``dnear * (1-w) + dfar * w`` where ``w = sigmoid(d + shift)``)
  - **Exp + hard cap scale** (``min(exp(x - shift), cap)``)
  - **Opacity** with bias shift (``sigmoid(x - 2.0)``)
  - **Optional position offset** (clamped or tanh-gated)

Output layout per Gaussian (14 channels total, ``sh_degree=0``):
  ``xyz(3) | opacity(1) | scale(3) | rotation(4) | rgb(3)``

Compatible with the existing ``DynamicGSHead`` / ``_compose_frame_gaussians``
contract (which only touches xyz + opacity + optional rotation in the delta;
the remaining channels — scale, features/rgb — are copied from canonical).

Also compatible with ``gaussian_render(sh_degree=0)`` — the renderer treats
the last ``(sh_degree+1)^2 * 3 = 3`` channels as colour.

Reference: Lyra-1 ``src/models/recon/model_latent_recon.py:327-351``
and ``configs/training/default.yaml:96-100``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gs_camera_utils import create_rays, quaternion_to_matrix


class LyraPixelAligned3DGS(nn.Module):
    """Lyra-1-style pixel-aligned 3D Gaussian head.

    Per-Gaussian channels (14 total with ``num_points_per_pixel=1``):
      - Distance:  1 (sigmoid-lerp → depth → ray-cast xyz)
      - RGB:       3 (``0.5 * tanh + 0.5``)
      - Scale:     3 (``min(exp(x - shift), cap)``)
      - Rotation:  4 (L2-normalised quaternion)
      - Opacity:   1 (``sigmoid(x - 2.0)``)
      - Offset:    3 (optional, clamped position offset on top of ray-cast)
    """

    def __init__(
        self,
        embed_dim: int,
        num_points_per_pixel: int = 1,
        dnear: float = 0.1,
        dfar: float = 500.0,
        pre_sigmoid_distance_shift: float = -1.65,
        scale_cap: float = 0.3,
        opacity_bias: float = -2.0,
        predict_offset: bool = True,
        offset_range: float = 0.5,
        depth_param: str = "sigmoid_lerp",
        log_depth_shift: float = 0.0,
        # v4: optional per-pixel-per-frame track-delta head (track-only,
        # NOT used in rendering). When True, exposes
        # `self.track_delta_proj` and the helper method
        # `predict_track_delta(features)`. The 3-channel output is the
        # predicted 3D delta of this pixel's content from frame-0 to the
        # current latent frame, supervised by GT persistent tracks.
        # Per-frame Gaussians from `gs_proj` keep handling rendering
        # photometrically; the delta head adds geometric trajectory
        # supervision without coupling to the render path.
        predict_track_delta: bool = False,
    ):
        super().__init__()
        self.num_points_per_pixel = int(num_points_per_pixel)
        self.dnear = float(dnear)
        self.dfar = float(dfar)
        self.pre_sigmoid_distance_shift = float(pre_sigmoid_distance_shift)
        self.scale_cap = float(scale_cap)
        self.scale_shift = 1.0 - math.log(scale_cap)
        self.opacity_bias = float(opacity_bias)
        self.predict_offset = bool(predict_offset)
        self.offset_range = float(offset_range)
        # Depth parameterization: 'sigmoid_lerp' (Lyra-1 native, bounded
        # to [dnear, dfar]) or 'exp' (DAV3 / FlashWorld style, unbounded,
        # log-parameterized via torch.exp(distance + log_depth_shift)).
        # log_depth_shift=0.0 → exp(0)=1 → initial depth ≈ 1 (unit-mean
        # after common_scale normalization).
        depth_param = str(depth_param).lower()
        if depth_param not in ("sigmoid_lerp", "exp"):
            raise ValueError(
                f"depth_param must be 'sigmoid_lerp' or 'exp'; got {depth_param!r}"
            )
        self.depth_param = depth_param
        self.log_depth_shift = float(log_depth_shift)
        self.sh_degree = 0  # direct RGB — renderer sees 3 colour channels

        # Channel layout: distance(1) + rgb(3) + scale(3) + rot(4) + opa(1) + [offset(3)]
        self.ch_per_gaussian = 12 + (3 if self.predict_offset else 0)
        total_out = self.num_points_per_pixel * self.ch_per_gaussian

        self.gs_proj = nn.Conv2d(embed_dim, total_out, 3, stride=1, padding=1)

        # Zero-init weight so the head starts neutral; bias carries the
        # init-time prior (identity quat, moderate depth, etc.).
        # Distance-channel bias depends on depth_param:
        #   'sigmoid_lerp' → pre_sigmoid_distance_shift (~-1.65, biases
        #                     initial depth near dnear side of the range)
        #   'exp'          → log_depth_shift (0.0 → trunc_exp(0)=1, i.e.
        #                     unit-mean initial depth)
        _distance_bias = (
            self.log_depth_shift if self.depth_param == "exp"
            else self.pre_sigmoid_distance_shift
        )
        with torch.no_grad():
            self.gs_proj.weight.data.zero_()
            bias_single = torch.tensor(
                [_distance_bias]                                # distance (param-specific init)
                + [0.0, 0.0, 0.0]                              # rgb (tanh(0)=0 → 0.5 gray)
                + [0.0, 0.0, 0.0]                              # scale (exp(0-shift)=exp(-shift) small)
                + [1.0, 0.0, 0.0, 0.0]                        # rotation (identity quat)
                + [0.0]                                         # opacity (sigmoid(0-2)≈0.12)
                + ([0.0, 0.0, 0.0] if self.predict_offset else [])
            )
            self.gs_proj.bias.data.copy_(bias_single.repeat(self.num_points_per_pixel))

        # v4 track-delta head: 3 channels per Gaussian per pixel for the
        # predicted (Δx, Δy, Δz) from frame 0 to the current latent frame.
        # Zero-init so init delta = 0 (no motion at t=0 by construction).
        # Stays None unless `predict_track_delta=True`.
        self.predict_track_delta = bool(predict_track_delta)
        if self.predict_track_delta:
            self.track_delta_proj = nn.Conv2d(
                embed_dim, 3 * self.num_points_per_pixel, 3, stride=1, padding=1,
            )
            with torch.no_grad():
                self.track_delta_proj.weight.data.zero_()
                self.track_delta_proj.bias.data.zero_()
        else:
            self.track_delta_proj = None

    def predict_track_delta_xyz(self, x: torch.Tensor) -> torch.Tensor:
        """v4: predict per-pixel-per-Gaussian 3D delta from frame 0.

        Args:
            x: (BN, C, H, W) feature map (same input as ``forward``).
        Returns:
            delta_xyz: (BN, npp, H, W, 3) — world-space delta vectors,
            supervised against (target_tracks_world[t] - target_tracks_world[0])
            in v4's track-only loss. Independent of rendering: never added
            to the rendered Gaussian xyz.

        Raises:
            RuntimeError if the head was constructed without
            ``predict_track_delta=True``.
        """
        if self.track_delta_proj is None:
            raise RuntimeError(
                "predict_track_delta_xyz called on a head built with "
                "predict_track_delta=False. Re-instantiate with the flag set."
            )
        x = x.to(torch.float32)
        BN, _, h, w = x.shape
        raw = self.track_delta_proj(x)  # (BN, npp*3, H, W)
        raw = raw.unflatten(1, (self.num_points_per_pixel, 3))
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()  # (BN, npp, H, W, 3)
        return raw

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor, cameras: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (BN, C, H, W) feature map from decoder.
            cameras: (BN, 11) FlashWorld-format cameras.
        Returns:
            gaussian_params: (BN, npp, H, W, D) where
                D = 3(xyz) + 1(opacity) + 3(scale) + 4(rotation) + 3(rgb) = 14.
        """
        x = x.to(torch.float32)
        cameras = cameras.to(torch.float32)
        BN, _, h, w = x.shape

        raw = self.gs_proj(x)  # (BN, npp*ch, H, W)
        raw = raw.unflatten(1, (self.num_points_per_pixel, self.ch_per_gaussian))
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()  # (BN, npp, H, W, ch)

        if self.predict_offset:
            distance, rgb, scaling, rotation, opacity, offset = raw.split(
                [1, 3, 3, 4, 1, 3], dim=-1
            )
        else:
            distance, rgb, scaling, rotation, opacity = raw.split(
                [1, 3, 3, 4, 1], dim=-1
            )
            offset = None

        # Depth via sigmoid-lerp (Lyra-1 convention, bounded to [dnear, dfar])
        # or trunc_exp (DAV3/FlashWorld convention, unbounded, log-parameterized).
        # Note: for exp mode, log_depth_shift is baked into the distance-channel
        # bias at init (see __init__). Applying it here too would double-apply
        # the shift (Codex review 2026-04-24). So forward is just trunc_exp(distance).
        if self.depth_param == "exp":
            from holo4d.models.gs_decoder import trunc_exp
            depth = trunc_exp(distance)
        else:
            w_depth = torch.sigmoid(distance)
            depth = self.dnear * (1.0 - w_depth) + self.dfar * w_depth

        # Ray-based 3D positions (same ray-casting as FlashWorld).
        rays_o, rays_d = create_rays(
            cameras[:, None].repeat(1, self.num_points_per_pixel, 1),
            uv_offset=None, h=h, w=w,
        )
        xyz = rays_o + depth * rays_d

        # Optional position offset.
        if offset is not None:
            xyz = xyz + torch.clamp(offset, -self.offset_range, self.offset_range)

        # Activations (Lyra-1 style, adapted for gsplat SH DC evaluation).
        # gsplat with sh_degree=0 evaluates color as: C0 * feature + 0.5
        # where C0 ≈ 0.282. So the raw feature (no activation) is the SH DC
        # coefficient — same as FlashWorld. Lyra-1's 0.5*tanh+0.5 was designed
        # for their custom renderer that doesn't apply C0; we skip it here.
        opacity = torch.sigmoid(opacity + self.opacity_bias)
        scale = torch.minimum(
            torch.exp(scaling - self.scale_shift),
            torch.tensor(self.scale_cap, device=scaling.device, dtype=scaling.dtype),
        )
        rotation = F.normalize(rotation, dim=-1)
        rgbs = rgb  # raw SH DC coefficient — gsplat applies C0 internally

        # Pack: [xyz(3), opacity(1), scale(3), rotation(4), rgb(3)] = 14
        gaussian_params = torch.cat([xyz, opacity, scale, rotation, rgbs], dim=-1)
        return gaussian_params


class LyraGaussianSplattingDecoder(nn.Module):
    """Lyra-1-style GS decoder: VAE decoder body + LyraPixelAligned3DGS head.

    Drop-in replacement for ``GaussianSplattingDecoder`` — same ``forward``
    and ``render`` signatures, compatible with ``DynamicGaussianSplattingDecoder``.
    """

    def __init__(
        self,
        vae_decoder: nn.Module,
        feat_dim: int = 1024,
        num_points_per_pixel: int = 1,
        use_network_checkpointing: bool = False,
        use_render_checkpointing: bool = False,
        dnear: float = 0.1,
        dfar: float = 500.0,
        scale_cap: float = 0.3,
        predict_offset: bool = True,
        render_per_frame_gaussians: bool = True,
    ):
        super().__init__()

        import copy
        self.decoder = copy.deepcopy(vae_decoder).requires_grad_(True)

        dims = [self.decoder.dim * u for u in
                [self.decoder.dim_mult[-1]] + self.decoder.dim_mult[::-1]]
        first_dim = dims[0]
        last_dim = dims[-1]

        from holo4d_runtime.models.wan_video_vae import CausalConv3d

        self.extra_conv_in = CausalConv3d(feat_dim, first_dim, 3, padding=1)
        with torch.no_grad():
            self.extra_conv_in.weight.data.zero_()
            self.extra_conv_in.bias.data.zero_()

        self.decoder.head = nn.Sequential(
            self.decoder.head[0],  # pretrained RMS_norm
            self.decoder.head[1],  # SiLU
        )

        self.gs_head = LyraPixelAligned3DGS(
            embed_dim=last_dim,
            num_points_per_pixel=num_points_per_pixel,
            dnear=dnear,
            dfar=dfar,
            scale_cap=scale_cap,
            predict_offset=predict_offset,
        )

        self.network_checkpointing = use_network_checkpointing
        self.render_checkpointing = use_render_checkpointing
        self.render_patch_size = None
        self.render_per_frame_gaussians = bool(render_per_frame_gaussians)

        decoder_dtype = next(self.decoder.parameters()).dtype
        self.extra_conv_in = self.extra_conv_in.to(dtype=decoder_dtype)
        self.gs_head = self.gs_head.to(dtype=decoder_dtype)

    def _decode_frame(self, z_i, feats_i, feat_cache, feat_idx, extra_cache):
        """Same as GaussianSplattingDecoder._decode_frame."""
        from holo4d_runtime.models.wan_video_vae import CACHE_T, CausalConv3d, ResidualBlock

        idx = feat_idx[0]
        cache_x_z = z_i[:, :, -CACHE_T:, :, :].clone()
        if cache_x_z.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x_z = torch.cat(
                [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x_z.device), cache_x_z],
                dim=2,
            )
        x = self.decoder.conv1(z_i, feat_cache[idx])
        feat_cache[idx] = cache_x_z
        feat_idx[0] += 1

        cache_x_f = feats_i[:, :, -CACHE_T:, :, :].clone()
        if cache_x_f.shape[2] < 2 and extra_cache[0] is not None:
            cache_x_f = torch.cat(
                [extra_cache[0][:, :, -1, :, :].unsqueeze(2).to(cache_x_f.device), cache_x_f],
                dim=2,
            )
        x = x + self.extra_conv_in(feats_i, extra_cache[0])
        extra_cache[0] = cache_x_f

        for layer in self.decoder.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.decoder.upsamples:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache, feat_idx)
            else:
                try:
                    x = layer(x, feat_cache, feat_idx)
                except TypeError:
                    x = layer(x)

        for layer in self.decoder.head:
            x = layer(x)

        return x

    def _ddp_sink_skipped_temporal_convs(self, x):
        if not torch.is_grad_enabled():
            return x
        sink = (x * 0).sum()
        for mod in self.decoder.modules():
            tc = getattr(mod, "time_conv", None)
            if tc is not None:
                for p in tc.parameters():
                    sink = sink + p.sum() * 0
        return x + sink

    def decode(self, feats, z):
        from holo4d_runtime.models.wan_video_vae import count_conv3d

        T_lat = int(z.shape[2])
        n_conv = count_conv3d(self.decoder)

        def _run_loop():
            feat_cache = [None] * n_conv
            extra_cache = [None]
            out_chunks = []
            for i in range(T_lat):
                feat_idx = [0]
                out_i = self._decode_frame(
                    z[:, :, i:i+1], feats[:, :, i:i+1],
                    feat_cache, feat_idx, extra_cache,
                )
                out_chunks.append(out_i)
            return torch.cat(out_chunks, dim=2)

        if self.network_checkpointing and torch.is_grad_enabled() and T_lat > 1:
            return torch.utils.checkpoint.checkpoint(_run_loop, use_reentrant=False)
        out = _run_loop()
        if T_lat == 1:
            out = self._ddp_sink_skipped_temporal_convs(out)
        return out

    def forward(self, feats, z, cameras, return_track_delta: bool = False):
        if feats.dim() != 5 or z.dim() != 5:
            raise ValueError(f"Expected 5D, got feats={tuple(feats.shape)}, z={tuple(z.shape)}")
        if cameras.dim() == 2:
            cameras = cameras.unsqueeze(0)
        x = self.decode(feats, z)
        b, c, t_video, h, w = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * t_video, c, h, w)
        cameras_flat = cameras.reshape(b * t_video, cameras.shape[-1])
        gaussian_params = self.gs_head(x_flat, cameras_flat)
        gaussian_params = gaussian_params.unflatten(0, (b, t_video))
        if return_track_delta:
            # v4: tap the same per-pixel features for the track-delta head.
            # Returns (gaussian_params, track_delta_xyz) where track_delta_xyz
            # has shape (B, T_video, npp, h, w, 3). Track-only — never enters
            # rendering. Caller must have constructed the head with
            # `predict_track_delta=True`.
            track_delta_xyz = self.gs_head.predict_track_delta_xyz(x_flat)
            track_delta_xyz = track_delta_xyz.unflatten(0, (b, t_video))
            return gaussian_params, track_delta_xyz
        return gaussian_params

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def render(self, gaussian_params, cameras, height, width, bg_mode="random"):
        from .gs_renderer import gaussian_render
        cameras = cameras.to(torch.float32)
        if cameras.dim() == 2:
            cameras = cameras.unsqueeze(0)
        if gaussian_params.dim() == 6:
            B, T, npp, gh, gw, D = gaussian_params.shape
            gaussian_params = gaussian_params.reshape(B, T, npp * gh * gw, D)
        elif gaussian_params.dim() == 5:
            BT, npp, gh, gw, D = gaussian_params.shape
            gaussian_params = gaussian_params.reshape(BT, npp * gh * gw, D).unsqueeze(1)
        B, V = cameras.shape[:2]
        test_c2ws = torch.eye(4, device=cameras.device)[None, None].repeat(B, V, 1, 1).float()
        test_c2ws[:, :, :3, :3] = quaternion_to_matrix(cameras[:, :, :4])
        test_c2ws[:, :, :3, 3] = cameras[:, :, 4:7]
        fx, fy, cx, cy = cameras[:, :, 7:11].split([1, 1, 1, 1], dim=-1)
        test_intr = torch.cat([fx * width, fy * height, cx * width, cy * height], dim=-1)
        return gaussian_render(
            gaussian_params, test_c2ws, test_intr, width, height,
            use_checkpoint=self.render_checkpointing,
            sh_degree=0,
            bg_mode=bg_mode,
            patch_size=self.render_patch_size,
            # New runs use per-frame video rendering: view t sees
            # gaussian_params[:, t] only. Old v3b checkpoints used the legacy
            # LRM-style union cloud, so this is configurable for reproduction.
            per_frame_gaussians=self.render_per_frame_gaussians,
        )
