"""
dense_target_control_encoder.py — Conv3d encoder for dense target-aligned maps.

Takes per-frame forward-warped maps already in target pixel space (produced by
``build_dense_target_reprojection``) and encodes them into block tokens for DiT
injection.  Architecturally mirrors the Conv3d patchify pipeline of
``Conv3dTrackAdapter`` (steps 5–9), but skips the scatter step because the
maps are already in target pixel space.

Input channel layout (``in_channels=5`` default):
  [0:3]  warped_rgb           — float32 [0, 1], 0 where invalid
  [3:4]  warped_depth_norm    — float32 [0, 1], caller-normalized, 0 where invalid
  [4:5]  warp_valid_float     — float32 {0, 1}, validity indicator

Vista4D-strict ablations may use ``in_channels=6``:
  [0:3]  warped_rgb
  [3:4]  warped_depth_norm or learned depth-null token
  [4:5]  target_motion_mask_float
  [5:6]  warp_valid_float

Vista4D nontrivial-mask ablations may use ``in_channels=7``:
  [0:3]  warped_rgb
  [3:4]  warped_depth_norm or learned depth-null token
  [4:5]  source_motion_mask reprojected into target space
  [5:6]  target_motion_mask_float
  [6:7]  warp_valid_float / point-cloud alpha

The legacy/full Vista4D mask-bundle layout may use ``in_channels=8`` by
inserting source alpha before source motion.

The valid-mask channel (index ``in_channels - 1``) doubles as model input
*and* occupancy signal for ``block_mask`` computation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use building blocks — no duplication
from holo4d.conditions.conv3d_track_adapter import (
    CausalResidualBlock3d,
    make_conv3d_patchify,
)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DenseTargetControlEncoder(nn.Module):
    """Conv3d-based encoder for dense target-aligned reprojection maps.

    Architecture (mirrors Conv3dTrackAdapter steps 5–9):
      1. First-frame repetition: prepend ``(temporal_stride - 1)`` copies of
         frame 0 so the first temporal patch sees real content rather than zeros.
      2. Plain ``nn.Conv3d`` patchify with ``kernel=(ts, ps, ps)``
         ``stride=(ts, ps, ps)``, ``padding=0``.
      3. ``CausalResidualBlock3d`` × ``num_residual_blocks`` (feature refine).
      4. ``proj_up``: ``nn.Linear(proj_dim → out_dim)``.
      5. ``block_mask``: ``F.max_pool3d`` on the warp-valid channel.
      6. Unoccupied blocks filled with ``null_token`` (learnable, zero-init).

    Args:
        in_channels: input channels. Default 5 = RGB(3) + depth(1) + valid(1);
            6 = RGB(3) + depth(1) + target motion mask(1) + valid(1);
            7 = 6 plus source motion reprojected into target space;
            8 = 7 plus source alpha reprojected into target space.
        proj_dim: channel dim inside the Conv3d stack.
        out_dim: output token dimension (must match DiT injection dim).
        patch_size: spatial patchify stride; ``image_h / (8 * patch_size)``
            must equal ``latent_h``.
        temporal_stride: temporal patchify stride; must match Wan VAE (4).
        num_residual_blocks: depth of ``CausalResidualBlock3d`` stack.
    """

    def __init__(
        self,
        in_channels: int = 5,
        proj_dim: int = 32,
        out_dim: int = 156,
        patch_size: int = 4,
        temporal_stride: int = 4,
        num_residual_blocks: int = 2,
        use_depth_null_token: bool = False,
        depth_channel_idx: int = 3,
        patchify_backend: Optional[str] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.proj_dim = proj_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.temporal_stride = temporal_stride
        # Index of the warp-valid channel (last channel by convention)
        self._valid_ch = in_channels - 1

        # Optional channel-level "no depth" mode for tier-B-style ablations.
        # Keeps the 5-channel input layout (so warm-start conv3d_patchify weight
        # loads cleanly), but replaces the depth slice in forward() with a
        # learnable scalar broadcast — letting the encoder learn the placeholder
        # value, matching the existing track_adapter.null_token pattern.
        self.use_depth_null_token = bool(use_depth_null_token)
        self.depth_channel_idx = int(depth_channel_idx)
        if self.use_depth_null_token:
            if in_channels < self.depth_channel_idx + 1:
                raise ValueError(
                    f"use_depth_null_token requires in_channels >= "
                    f"{self.depth_channel_idx + 1}; got {in_channels}."
                )
            # Zero-init: at step 0 the depth slice is all zeros, similar to
            # "very near plane"; the model learns to drift this during finetune.
            self.depth_null_token = nn.Parameter(torch.zeros(1))

        # Step 2: Non-overlapping patchify. The optional linear backend is an
        # exact reshape+Linear equivalent of Conv3d for profiling fallback
        # behavior without changing checkpoint parameter names or shapes.
        self.conv3d_patchify = make_conv3d_patchify(
            in_channels, proj_dim,
            kernel_size=(temporal_stride, patch_size, patch_size),
            stride=(temporal_stride, patch_size, patch_size),
            padding=0,
            backend=patchify_backend,
        )
        nn.init.kaiming_normal_(
            self.conv3d_patchify.weight, mode="fan_out", nonlinearity="linear"
        )
        nn.init.zeros_(self.conv3d_patchify.bias)

        # Step 3: Causal residual blocks (stride-1; causal left-pad correct)
        self.residual_blocks = nn.Sequential(
            *[CausalResidualBlock3d(proj_dim) for _ in range(num_residual_blocks)]
        )

        # Step 4: Project up to DiT dimension
        self.proj_up = nn.Linear(proj_dim, out_dim)

        # Null token for unoccupied blocks — zero-init so they inject nothing
        self.null_token = nn.Parameter(torch.zeros(out_dim))
        # Opt-in hard-zero null (TRACK4D360_TRACK_NULL_TOKEN_FROZEN=1): freeze the
        # empty-block token so it cannot drift toward occupied-token statistics.
        if os.environ.get("TRACK4D360_TRACK_NULL_TOKEN_FROZEN", "0") == "1":
            self.null_token.requires_grad_(False)

    # ------------------------------------------------------------------
    def forward(
        self,
        dense_maps: torch.Tensor,
        latent_t: int,
        latent_h: int,
        latent_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode dense target-aligned maps into block tokens.

        Args:
            dense_maps: ``(B, in_channels, T, H, W)`` float32.
                Channel layout is either ``[warped_rgb (3) |
                warped_depth_norm (1) | warp_valid_float (1)]`` or
                ``[warped_rgb (3) | warped_depth_norm (1) |
                target_motion_mask (1) | warp_valid_float (1)]``. The last
                channel must be the binary validity mask ({0, 1}).
            latent_t: expected temporal latent size.
            latent_h: expected spatial latent height.
            latent_w: expected spatial latent width.

        Returns:
            block_tokens: ``(B, latent_t * latent_h * latent_w, out_dim)``
                float32 — sequence aligned with ``Conv3dTrackAdapter`` output.
            block_mask: ``(B, latent_t * latent_h * latent_w)`` bool —
                True where at least one source pixel warped into that block.
        """
        B, C, T, H, W = dense_maps.shape
        ts = self.temporal_stride
        ps = self.patch_size

        # Step 0 (optional): depth-null mode — replace the warped-depth slice
        # with a learnable scalar broadcast. Used by tier-B-prime ablations
        # ("plucker + warped RGB, no depth") so the conv3d_patchify weight
        # stays at in_channels=5 and warm-starts cleanly from RGBD checkpoints.
        if self.use_depth_null_token:
            d = self.depth_channel_idx
            null_depth = self.depth_null_token.view(1, 1, 1, 1, 1).to(
                dtype=dense_maps.dtype, device=dense_maps.device
            ).expand(B, 1, T, H, W)
            dense_maps = torch.cat(
                [dense_maps[:, :d], null_depth, dense_maps[:, d + 1:]], dim=1
            )

        # ------------------------------------------------------------------
        # Step 1: First-frame repetition
        # Prepend (ts - 1) copies of frame 0 so the first Conv3d patch window
        # spans [frame0 * (ts-1), frame0, frame1, ..., frame(ts-1)] — matching
        # the causal convention used in Conv3dTrackAdapter.
        # After cat: (B, C, T + ts - 1, H, W)
        # ------------------------------------------------------------------
        first_frame = dense_maps[:, :, :1]                           # (B, C, 1, H, W)
        x = torch.cat(
            [first_frame.expand(-1, -1, ts - 1, -1, -1), dense_maps], dim=2
        )  # (B, C, T + ts - 1, H, W)

        # Also replicate the valid channel for occupancy tracking
        valid_vol = dense_maps[:, self._valid_ch : self._valid_ch + 1]  # (B, 1, T, H, W)
        first_valid = valid_vol[:, :, :1]
        valid_vol_padded = torch.cat(
            [first_valid.expand(-1, -1, ts - 1, -1, -1), valid_vol], dim=2
        )  # (B, 1, T + ts - 1, H, W)

        # ------------------------------------------------------------------
        # Step 2: Conv3d patchify
        # Input:  (B, C, T + ts - 1, H, W)
        # Output: (B, proj_dim, latent_t, latent_h, latent_w)
        #   latent_t = (T + ts - 1 - ts) // ts + 1 = (T - 1) // ts + 1  ✓
        # ------------------------------------------------------------------
        x_latent = self.conv3d_patchify(x)  # (B, proj_dim, lt, lh, lw)

        out_t = x_latent.shape[2]
        assert out_t == latent_t, (
            f"DenseTargetControlEncoder: patchify output_t={out_t} != "
            f"latent_t={latent_t}; T={T}, ts={ts}, "
            f"expected (T-1)//ts+1={(T - 1) // ts + 1}"
        )

        # ------------------------------------------------------------------
        # Step 3: Residual blocks
        # ------------------------------------------------------------------
        x_latent = self.residual_blocks(x_latent)

        # ------------------------------------------------------------------
        # Step 4: proj_up Linear
        # (B, proj_dim, lt, lh, lw) → permute → (B, lt, lh, lw, proj_dim)
        # → Linear → (B, lt, lh, lw, out_dim) → reshape → (B, num_blocks, out_dim)
        # ------------------------------------------------------------------
        x_latent = x_latent.permute(0, 2, 3, 4, 1)           # (B, lt, lh, lw, proj_dim)
        block_tokens = self.proj_up(x_latent)                 # (B, lt, lh, lw, out_dim)
        block_tokens = block_tokens.reshape(B, -1, self.out_dim)  # (B, num_blocks, out_dim)

        # ------------------------------------------------------------------
        # Step 5: block_mask via max_pool3d on valid channel
        # max_pool3d uses the same kernel/stride as patchify, so each block
        # gets True iff at least one valid source pixel fell inside it.
        # ------------------------------------------------------------------
        occ_latent = F.max_pool3d(
            valid_vol_padded.float(),
            kernel_size=(ts, ps, ps),
            stride=(ts, ps, ps),
        )  # (B, 1, latent_t, latent_h, latent_w)
        block_mask = (occ_latent > 0.0).squeeze(1).reshape(B, -1)  # (B, num_blocks)

        # ------------------------------------------------------------------
        # Step 6: fill unoccupied blocks with null_token
        # ------------------------------------------------------------------
        null = self.null_token.view(1, 1, self.out_dim).expand_as(block_tokens)
        block_tokens = torch.where(block_mask.unsqueeze(-1), block_tokens, null)

        return block_tokens, block_mask
