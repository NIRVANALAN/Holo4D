"""
Conv3dTrackAdapter: causal 3D patchify encoder for dense grid tracks.

Architecture (Koichi-aligned):
  1. Sinusoidal encoding of (frame_idx, world_x, world_y, world_z, vis)
  2. Per-timestep MLP → (B, N, T, hidden_dim)
  3. proj_down: Linear(hidden_dim → proj_dim)
  4. Target-UV splat: scatter_reduce_('amax') into target grid
       (B, proj_dim, T, H_grid, W_grid)  where H_grid = latent_h * patch_size
  5. First-frame repetition: prepend (ts - 1) copies of frame 0 →
       (B, proj_dim, T + ts - 1, H_grid, W_grid)
  6. Plain Conv3d patchify: kernel=(ts, ps, ps), stride=(ts, ps, ps), padding=0
       → (B, proj_dim, latent_t, latent_h, latent_w)
       Output: latent_t = (T + ts - 1 - ts) // ts + 1 = (T - 1) // ts + 1 ✓
  7. CausalResidualBlock3d × N: feature refinement at latent resolution
  8. proj_up: Linear(proj_dim → out_dim)
  9. block_mask from occupancy max_pool3d (same kernel/stride as patchify)

Key difference from prior implementation:
  - Tracks are scattered directly to the TARGET view's UV grid, not the source
    view grid.  This is correct for NVS (different cameras) where source and
    target grids do not coincide.
  - Temporal context for the first latent frame is provided by explicit frame-0
    repetition rather than zero-padding (CausalConv3d).  This avoids the
    dilution of frame-0 signals that zero-padding causes.

Reference: SimpleAdapter in wan_video_camera_controller.py
Reference: CausalConv3d in wan_video_vae.py
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Optional, Tuple

from holo4d.utils.torch_profile import track4d_record_function


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalConv3d(nn.Conv3d):
    """3D convolution with causal temporal padding (used in residual blocks).

    Temporal: left-pads by (kernel_t - 1), no right padding.
    Spatial: symmetric padding as specified.

    For stride-1 residual convs: output temporal size equals input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kt = self.kernel_size[0]
        ph = self.padding[1]
        pw = self.padding[2]
        # F.pad order: (w_left, w_right, h_left, h_right, t_left, t_right)
        self._causal_pad = (pw, pw, ph, ph, kt - 1, 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        x = F.pad(x, self._causal_pad)
        return super().forward(x)


class CausalResidualBlock3d(nn.Module):
    """3D residual block with causal temporal convolutions.

    Conv3×3×3 → GELU → Conv3×3×3 + skip.
    Spatial padding symmetric, temporal padding causal (left-only).
    """

    def __init__(self, dim):
        super().__init__()
        self.conv1 = CausalConv3d(dim, dim, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.act = nn.GELU()
        self.conv2 = CausalConv3d(dim, dim, kernel_size=(3, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


def _as_3tuple(value, name: str) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must be an int or a length-3 tuple; got {value!r}.")
    return tuple(int(v) for v in value)


class NonOverlappingConv3dPatchifyLinear(nn.Module):
    """Exact Conv3d patchify replacement for stride == kernel and padding == 0.

    The state dict intentionally matches ``nn.Conv3d``: ``weight`` has shape
    ``(out_channels, in_channels, kt, kh, kw)`` and ``bias`` has shape
    ``(out_channels,)``. Existing checkpoints therefore load without a
    conversion step while the forward path uses a reshape plus ``F.linear``
    instead of PyTorch's slow 3D-convolution fallback.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=None,
        padding=0,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = _as_3tuple(kernel_size, "kernel_size")
        stride = kernel_size if stride is None else _as_3tuple(stride, "stride")
        padding = _as_3tuple(padding, "padding")
        if stride != kernel_size:
            raise ValueError(
                "NonOverlappingConv3dPatchifyLinear only supports "
                f"stride == kernel_size; got stride={stride}, kernel_size={kernel_size}."
            )
        if padding != (0, 0, 0):
            raise ValueError(
                "NonOverlappingConv3dPatchifyLinear only supports padding=0; "
                f"got padding={padding}."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B,C,T,H,W); got shape={tuple(x.shape)}.")
        b, c, t, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: got {c}, expected {self.in_channels}."
            )

        kt, kh, kw = self.kernel_size
        out_t = (t - kt) // kt + 1
        out_h = (h - kh) // kh + 1
        out_w = (w - kw) // kw + 1
        if out_t <= 0 or out_h <= 0 or out_w <= 0:
            raise ValueError(
                "Input is smaller than patchify kernel: "
                f"input={(t, h, w)}, kernel={self.kernel_size}."
            )

        # Match Conv3d floor behavior for non-overlapping windows by ignoring
        # incomplete trailing cells.
        x = x[:, :, : out_t * kt, : out_h * kh, : out_w * kw]
        patches = x.reshape(b, c, out_t, kt, out_h, kh, out_w, kw)
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7)
        patches = patches.reshape(b * out_t * out_h * out_w, c * kt * kh * kw)
        weight = self.weight.reshape(self.out_channels, c * kt * kh * kw)
        out = F.linear(patches, weight, self.bias)
        out = out.reshape(b, out_t, out_h, out_w, self.out_channels)
        return out.permute(0, 4, 1, 2, 3).contiguous()

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
        )


def make_conv3d_patchify(
    in_channels: int,
    out_channels: int,
    kernel_size,
    stride,
    padding=0,
    backend: Optional[str] = None,
) -> nn.Module:
    backend = (backend or os.environ.get("TRACK4D360_CONV3D_PATCHIFY_BACKEND", "conv3d")).lower()
    if backend in {"linear", "reshape_linear", "patchify_linear"}:
        module = NonOverlappingConv3dPatchifyLinear(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        module.patchify_backend = "linear"
        return module
    if backend in {"conv3d", "torch", "nn.conv3d"}:
        module = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        module.patchify_backend = "conv3d"
        return module
    raise ValueError(
        "Unknown TRACK4D360_CONV3D_PATCHIFY_BACKEND="
        f"{backend!r}; expected 'linear' or 'conv3d'."
    )


class SinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous values."""

    def __init__(self, num_freqs: int = 16, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.out_dim_per_channel = 2 * num_freqs + (1 if include_input else 0)
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_freq = x.unsqueeze(-1) * self.freqs * math.pi
        sin_enc = torch.sin(x_freq)
        cos_enc = torch.cos(x_freq)
        parts = []
        if self.include_input:
            parts.append(x.unsqueeze(-1))
        parts.append(sin_enc)
        parts.append(cos_enc)
        encoded = torch.cat(parts, dim=-1)
        shape = encoded.shape[:-2]
        return encoded.reshape(*shape, -1)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class Conv3dTrackAdapter(nn.Module):
    """Target-UV-splat Conv3d encoder for dense tracks.

    Takes raw track_seq + track_uv + track_vis, outputs block_tokens.
    Tracks are scattered to the TARGET view's pixel grid, then processed by
    Conv3d at the target latent resolution — correct for NVS settings.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_freqs: int = 16,
        hidden_dim: int = 156,
        out_dim: int = 156,
        num_mlp_layers: int = 3,
        proj_dim: int = 32,
        patch_size: int = 4,
        temporal_stride: int = 4,
        num_residual_blocks: int = 1,
        block_t: int = 1,
        block_h: int = 1,
        block_w: int = 1,
        patchify_backend: Optional[str] = None,
    ):
        """
        Args:
            in_channels: input channels per timestep (5: frame_idx, wx, wy, wz, vis).
            num_freqs: sinusoidal frequency bands per channel.
            hidden_dim: MLP hidden dimension.
            out_dim: output dimension per block token.
            num_mlp_layers: MLP depth.
            proj_dim: intermediate dim for Conv3d.
            patch_size: spatial kernel & stride (must equal latent_h = image_h // (8 * ps)).
            temporal_stride: temporal kernel & stride (4 for Wan VAE).
            num_residual_blocks: number of CausalResidualBlock3d after patchify.
            block_t/h/w: block sizes for interface compat; must be 1 with target-UV splat.
            patchify_backend: "conv3d" (default) uses the previous nn.Conv3d
                path; "linear" uses exact reshape+Linear for profiling.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.proj_dim = proj_dim
        self.patch_size = patch_size
        self.temporal_stride = temporal_stride
        self.block_t = block_t
        self.block_h = block_h
        self.block_w = block_w

        # Step 1: Sinusoidal encoding
        self.sin_enc = SinusoidalEncoding(num_freqs=num_freqs, include_input=True)
        enc_dim = in_channels * self.sin_enc.out_dim_per_channel  # 5 * 33 = 165

        # Step 2: Per-timestep MLP
        layers = []
        in_d = enc_dim
        for _ in range(num_mlp_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.GELU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        # Step 3: Project down before Conv3d
        self.proj_down = nn.Linear(hidden_dim, proj_dim)

        # Step 4: Non-overlapping patchify. The optional linear backend is an
        # exact reshape+Linear equivalent of Conv3d for profiling fallback
        # behavior without changing checkpoint parameter names or shapes.
        self.conv3d_patchify = make_conv3d_patchify(
            proj_dim, proj_dim,
            kernel_size=(temporal_stride, patch_size, patch_size),
            stride=(temporal_stride, patch_size, patch_size),
            padding=0,
            backend=patchify_backend,
        )
        nn.init.kaiming_normal_(self.conv3d_patchify.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.conv3d_patchify.bias)

        # Step 5: Causal residual blocks (stride-1, causal left-pad correct here)
        self.residual_blocks = nn.Sequential(
            *[CausalResidualBlock3d(proj_dim) for _ in range(num_residual_blocks)]
        )

        # Step 6: Project up
        self.proj_up = nn.Linear(proj_dim, out_dim)

        # Null token for unoccupied target blocks — zero-init so unoccupied cells inject nothing
        # (Koichi zeroes out unoccupied outputs after occupancy masking; this matches that)
        self.null_token = nn.Parameter(torch.zeros(out_dim))
        # Opt-in hard-zero null: freeze the empty-block token so it can never
        # drift toward occupied-token statistics (GWTT-style reserved value).
        # Default off — legacy checkpoints/configs keep the learnable null.
        if os.environ.get("TRACK4D360_TRACK_NULL_TOKEN_FROZEN", "0") == "1":
            self.null_token.requires_grad_(False)

    def forward(
        self,
        track_seq: torch.Tensor,
        track_uv: torch.Tensor,
        track_vis: torch.Tensor,
        image_h: int,
        image_w: int,
        num_frames: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            track_seq: (B, N, T, 5) — raw track features (frame_idx, wx, wy, wz, vis).
            track_uv:  (B, N, T, 2) — pixel coords in TARGET view (for splat).
            track_vis: (B, N, T)    — visibility mask.
            image_h, image_w: target image resolution.
            num_frames: T.
            latent_t, latent_h, latent_w: target latent grid dims.

        Returns:
            block_tokens: (B, num_blocks, out_dim)
            block_mask:   (B, num_blocks) bool — True where any track is visible.
        """
        B, N, T, C = track_seq.shape
        device = track_seq.device
        dtype = track_seq.dtype
        ps = self.patch_size
        ts = self.temporal_stride

        H_grid = latent_h * ps
        W_grid = latent_w * ps

        # --- Steps 1-4: sin-encode + MLP + proj_down + target-UV splat ---
        # Every intermediate in this block — `encoded`, the MLP `features`, the
        # `proj_down` output, `feats_flat`, the (B,N,T) index tensors — scales
        # LINEARLY with N (the track-point count). N is randomized per sample by
        # the sparse-track stride (2|4|8) and multi-query count (1|2|3), so it
        # swings by >10x between a sparse+single-query draw and a dense+
        # multi-query draw. Under FSDP these activations are NOT sharded, so a
        # rank that draws a heavy sample retains ~tens of GB more than its peers
        # → data-dependent activation imbalance and a random-rank OOM at the HBM
        # ceiling (observed 2026-06-11: 384p 8-node 14B, rank 14 at 92 GiB while
        # its node-mates sat at ~68 GiB).
        #
        # Gradient-checkpoint this region so the N-scaling intermediates are
        # dropped after the forward and recomputed in backward. Only the
        # FIXED-size scatter outputs (x_grid_flat / occ_flat, sized
        # total_cells×proj_dim and independent of N) survive — which decouples
        # per-rank memory from N and removes the imbalance. The recomputed work
        # (sin/MLP/scatter) is tiny next to the DiT denoiser, so the throughput
        # cost is negligible. Escape hatch:
        # TRACK4D360_TRACK_ADAPTER_CHECKPOINT=0 restores the old fully-
        # materialized path.
        # Optional identity-chunking of the per-point sin/MLP/proj_down. Even
        # with a fixed N, the transient (B, N, T, enc_dim≈165) + (B, N, T,
        # hidden_dim) tensors are the single largest forward allocation here.
        # Processing N in `nchunk` slices bounds that transient to ~1/nchunk
        # without changing the result (the ops are per-identity independent).
        # 0/1 = off (single pass, default). Stacks cleanly under the checkpoint.
        _nchunk = int(os.environ.get("TRACK4D360_TRACK_ADAPTER_NCHUNK", "0") or 0)

        def _encode_mlp_splat(track_seq, track_uv, track_vis):
            # --- Step 1-3: Sinusoidal encoding + MLP + project down ---
            # Output: features (B, N, T, proj_dim). Per-identity independent, so
            # it can be computed in N-slices to cap the transient peak.
            with track4d_record_function("track4d360/track_adapter/encode_mlp"):
                if _nchunk > 1 and N > _nchunk:
                    chunk = (N + _nchunk - 1) // _nchunk
                    feat_parts = []
                    for s in range(0, N, chunk):
                        e = min(s + chunk, N)
                        enc_c = self.sin_enc(track_seq[:, s:e])   # (B, n_c, T, enc_dim)
                        feat_c = self.mlp(enc_c)                  # (B, n_c, T, hidden_dim)
                        feat_c = self.proj_down(feat_c)           # (B, n_c, T, proj_dim)
                        feat_parts.append(feat_c)
                    features = torch.cat(feat_parts, dim=1)       # (B, N, T, proj_dim)
                else:
                    encoded = self.sin_enc(track_seq)      # (B, N, T, enc_dim)
                    features = self.mlp(encoded)           # (B, N, T, hidden_dim)
                    features = self.proj_down(features)    # (B, N, T, proj_dim)

            # --- Step 4: Target-UV splat into (B, proj_dim, T, H_grid, W_grid) ---
            # Map pixel coords → grid indices
            u_idx = (track_uv[..., 0] / image_w * W_grid).long().clamp(0, W_grid - 1)  # (B,N,T)
            v_idx = (track_uv[..., 1] / image_h * H_grid).long().clamp(0, H_grid - 1)  # (B,N,T)
            t_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, N, T)

            flat_idx = (t_idx * H_grid * W_grid + v_idx * W_grid + u_idx)  # (B, N, T)
            flat_idx = flat_idx.reshape(B, N * T)

            # Mask invisible entries before flattening. Do this in-place on the
            # proj_down output to avoid cloning B*N*T*proj_dim; that clone was a
            # 1.63GiB transient allocation in 288p q-heavy 14B runs.
            features.masked_fill_(~track_vis.bool().unsqueeze(-1), float('-inf'))

            # Flatten features for scatter.
            feats_flat = features.reshape(B, N * T, self.proj_dim)

            # Scatter targets MUST match the scatter SOURCE dtype (feats_flat),
            # which is the autocast compute dtype (bf16), NOT the raw input
            # dtype (track_seq.dtype). Under DDP/FSDP1 the input was already bf16
            # so `dtype` happened to match; FSDP2's mixed-precision path leaves
            # track_seq fp32 → scatter_reduce_ "Expected self.dtype == src.dtype".
            # Use feats_flat.dtype everywhere so all three sharding paths agree.
            compute_dtype = feats_flat.dtype

            with track4d_record_function("track4d360/track_adapter/scatter_reduce"):
                # Scatter features: amax into (B, T*H_grid*W_grid, proj_dim)
                total_cells = T * H_grid * W_grid
                x_grid_flat = torch.full(
                    (B, total_cells, self.proj_dim), float('-inf'), device=device, dtype=compute_dtype
                )
                idx_exp = flat_idx.unsqueeze(-1).expand(-1, -1, self.proj_dim)
                x_grid_flat.scatter_reduce_(1, idx_exp, feats_flat, reduce='amax', include_self=True)

                # Scatter occupancy: amax of vis indicator
                occ_flat = torch.zeros(B, total_cells, device=device, dtype=compute_dtype)
                vis_float = track_vis.to(device=device, dtype=compute_dtype).reshape(B, N * T)
                occ_flat.scatter_reduce_(1, flat_idx, vis_float, reduce='amax', include_self=True)

            # Replace -inf (unoccupied) with 0
            x_grid_flat = x_grid_flat.masked_fill(x_grid_flat == float('-inf'), 0.0)
            return x_grid_flat, occ_flat

        _use_ckpt = (
            self.training
            and torch.is_grad_enabled()
            and os.environ.get("TRACK4D360_TRACK_ADAPTER_CHECKPOINT", "1") == "1"
        )
        if _use_ckpt:
            # use_reentrant=False is required: the region has in-place ops
            # (masked_fill_, scatter_reduce_) and must preserve the autocast
            # context across recompute — both are handled by the non-reentrant
            # checkpoint implementation.
            x_grid_flat, occ_flat = torch.utils.checkpoint.checkpoint(
                _encode_mlp_splat, track_seq, track_uv, track_vis,
                use_reentrant=False,
            )
        else:
            x_grid_flat, occ_flat = _encode_mlp_splat(track_seq, track_uv, track_vis)

        # Reshape to volumes
        x_grid = x_grid_flat.reshape(B, T, H_grid, W_grid, self.proj_dim)
        x_grid = x_grid.permute(0, 4, 1, 2, 3)          # (B, proj_dim, T, H_grid, W_grid)
        occ_grid = occ_flat.reshape(B, 1, T, H_grid, W_grid)

        # --- Step 5: First-frame repetition (Koichi: ts-1 copies of frame 0) ---
        first_feat = x_grid[:, :, :1]
        x_grid = torch.cat([first_feat.expand(-1, -1, ts - 1, -1, -1), x_grid], dim=2)
        # x_grid: (B, proj_dim, T + ts - 1, H_grid, W_grid)

        first_occ = occ_grid[:, :, :1]
        occ_grid = torch.cat([first_occ.expand(-1, -1, ts - 1, -1, -1), occ_grid], dim=2)
        # occ_grid: (B, 1, T + ts - 1, H_grid, W_grid)

        # --- Step 6: Plain Conv3d patchify ---
        # Output: (B, proj_dim, (T+ts-1-ts)//ts+1, latent_h, latent_w)
        #       = (B, proj_dim, (T-1)//ts+1, latent_h, latent_w)
        with track4d_record_function("track4d360/track_adapter/conv3d_patchify"):
            x_latent = self.conv3d_patchify(x_grid)

        out_t = x_latent.shape[2]
        assert out_t == latent_t, (
            f"Conv3d output_t={out_t} != latent_t={latent_t}; "
            f"T={T}, ts={ts}, expected (T-1)//ts+1={(T-1)//ts+1}"
        )

        # --- Step 7: Residual blocks ---
        with track4d_record_function("track4d360/track_adapter/residual_blocks"):
            x_latent = self.residual_blocks(x_latent)

        # --- Step 8: block_mask from occupancy max_pool3d ---
        occ_latent = F.max_pool3d(
            occ_grid.float(),
            kernel_size=(ts, ps, ps),
            stride=(ts, ps, ps),
        )  # (B, 1, latent_t, latent_h, latent_w)
        block_mask = (occ_latent > 0).squeeze(1).reshape(B, -1)  # (B, num_blocks)

        # --- Step 9: Project up + assemble block tokens ---
        x_latent = x_latent.permute(0, 2, 3, 4, 1)          # (B, latent_t, lh, lw, proj_dim)
        with track4d_record_function("track4d360/track_adapter/proj_up"):
            block_tokens = self.proj_up(x_latent)             # (B, latent_t, lh, lw, out_dim)
        block_tokens = block_tokens.reshape(B, -1, self.out_dim)  # (B, num_blocks, out_dim)

        # Fill unoccupied blocks with null token
        null = self.null_token.view(1, 1, self.out_dim).expand_as(block_tokens)
        block_tokens = torch.where(block_mask.unsqueeze(-1), block_tokens, null)

        return block_tokens, block_mask
