"""
DenseTrackConvAdapter: learnable Conv2d patchify adapter for dense regular-grid tracks.

This is an alternative adapter for the **persistent-query track branch** (frame-0-query
tracking with time-varying support).  It is NOT the dense same-timestep reprojection
branch described in `doc/new_features/2026-04-07-hybrid-spatial-control-design.md`.

Architecture role:
  - Input: (B, N, D) per-track embeddings from PointTrackEmbedder (temporal-aggregated)
  - Processing: Conv2d patchify on source-image grid layout, then scatter to target view
  - Output: (B, num_blocks, out_dim) block tokens, same interface as PointTrackAdapter
  - Drop-in replacement for PointTrackAdapter when tracks form a regular source grid

The dense same-timestep reprojection branch (TODO-8 / Phase 1) is a different module
to be created as `dense_target_reprojection_builder.py`, which operates on per-frame
source RGB/depth warped into target space (no PointTrackEmbedder involved).

Replaces PointTrackAdapter's scatter max-pool with a learned spatial aggregation.
Only works when tracks form a regular spatial grid on the source image (stride-S).

Key advantages over scatter max-pool:
  - Learnable aggregation (Conv2d) instead of fixed max-pool
  - Keeps all tracks at stride=1 (no input-side downsampling needed)
  - Spatial context: each Conv2d kernel sees a local patch of track embeddings,
    not just point-wise features scattered to a bucket

Architecture:
  1. Reshape time-invariant track embeddings (B, N, D) → (B, D, Hs, Ws) source grid
  2. proj_down: Linear(D → proj_dim) to reduce channel dim before conv
  3. Conv2d(proj_dim, proj_dim, kernel=patch_size, stride=patch_size) — patchify
     Output: (B, proj_dim, latent_h, latent_w) in SOURCE view
  4. proj_up: Linear(proj_dim → out_dim) to restore channel dim
  5. Source-to-target scatter: reproject source grid centers through target cameras
     to place each source-latent-cell into the correct target-latent-cell
  6. Temporal broadcast: replicate time-invariant features to all latent_t blocks

Patch size derivation:
  patch_size = spatial_div // track_spatial_stride
  For Wan 2.1 (spatial_div=16):
    stride=4 → patch_size=4,  stride=1 → patch_size=16

Why this works:
  - Koichi can't do this: his CoTracker tracks are irregular (sparse, non-grid).
  - We have dense mesh-tracked correspondences at regular stride on source image.
  - Conv patchify is the standard approach in ViTs (like DiT's own patchify layer).

Compatibility:
  - Same forward() signature as PointTrackAdapter
  - Same output shape: (B, num_blocks, out_dim), (B, num_blocks)
  - Drop-in replacement via config: adapter_type="conv_patchify"
"""

import os

import torch
import torch.nn as nn
from typing import Tuple


class DenseTrackConvAdapter(nn.Module):
    """Learnable Conv2d patchify adapter for dense regular-grid tracks.

    Converts per-track embeddings (on a regular source grid) into block-aligned
    latent tokens via learned spatial convolution, then scatters to target view.
    """

    def __init__(
        self,
        track_dim: int = 156,
        out_dim: int = 156,
        proj_dim: int = 32,
        patch_size: int = 4,
        block_t: int = 1,
        block_h: int = 1,
        block_w: int = 1,
    ):
        """
        Args:
            track_dim: dimension of per-track embeddings from PointTrackEmbedder.
            out_dim: output dimension per block token.
            proj_dim: intermediate dimension for conv (reduced from track_dim).
            patch_size: spatial patchify kernel & stride size.
                = spatial_div // track_spatial_stride.
                For Wan 2.1 (spatial_div=16): stride=4 → 4, stride=1 → 16.
            block_t: temporal block size in latent frames (kept for interface compat).
            block_h: spatial height block size in latent patches.
            block_w: spatial width block size in latent patches.
        """
        super().__init__()
        self.track_dim = track_dim
        self.out_dim = out_dim
        self.proj_dim = proj_dim
        self.patch_size = patch_size
        self.block_t = block_t
        self.block_h = block_h
        self.block_w = block_w

        # Project down before conv (156 → proj_dim channels is much cheaper to convolve)
        self.proj_down = nn.Linear(track_dim, proj_dim)

        # Conv2d patchify: kernel=patch_size, stride=patch_size
        self.conv = nn.Conv2d(
            proj_dim, proj_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.conv.bias)

        # Project up after conv (proj_dim → out_dim)
        self.proj_up = nn.Linear(proj_dim, out_dim)

        # Null token for empty blocks — zero-init so unoccupied cells inject nothing
        # (Koichi zeroes out unoccupied outputs after occupancy masking; this matches that)
        self.null_token = nn.Parameter(torch.zeros(out_dim))
        # Opt-in hard-zero null (TRACK4D360_TRACK_NULL_TOKEN_FROZEN=1): freeze the
        # empty-block token so it cannot drift toward occupied-token statistics.
        if os.environ.get("TRACK4D360_TRACK_NULL_TOKEN_FROZEN", "0") == "1":
            self.null_token.requires_grad_(False)

    def forward(
        self,
        track_embeddings: torch.Tensor,
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
            track_embeddings: (B, N, D) per-track embeddings from PointTrackEmbedder.
            track_uv: (B, N, T, 2) pixel coordinates (u=col, v=row) in target view.
            track_vis: (B, N, T) bool visibility mask.
            image_h, image_w: pixel resolution of target view.
            num_frames: number of input video frames.
            latent_t: temporal dimension of latent grid.
            latent_h: spatial height of latent grid (after patchify).
            latent_w: spatial width of latent grid (after patchify).

        Returns:
            block_tokens: (B, num_blocks, out_dim) — one token per latent block.
            block_mask: (B, num_blocks) — bool, True if block has at least one track.
        """
        B, N, D = track_embeddings.shape
        T = track_uv.shape[2]
        device = track_embeddings.device
        dtype = track_embeddings.dtype
        ps = self.patch_size

        num_bt = (latent_t + self.block_t - 1) // self.block_t
        num_bh = (latent_h + self.block_h - 1) // self.block_h
        num_bw = (latent_w + self.block_w - 1) // self.block_w
        num_blocks = num_bt * num_bh * num_bw

        # --- Step 1: Infer source grid dimensions from N and aspect ratio ---
        aspect = image_w / image_h
        Hs = int(round((N / aspect) ** 0.5))
        Ws = N // Hs
        if Hs * Ws != N:
            for delta in range(1, 10):
                if (Hs + delta) > 0 and N % (Hs + delta) == 0:
                    Hs = Hs + delta
                    Ws = N // Hs
                    break
                if (Hs - delta) > 0 and N % (Hs - delta) == 0:
                    Hs = Hs - delta
                    Ws = N // Hs
                    break
        assert Hs * Ws == N, (
            f"Cannot reshape N={N} tracks into grid with aspect {aspect:.3f}. "
            f"Check track_spatial_stride and image resolution."
        )

        # --- Step 2: Reshape to source spatial grid ---
        src_grid = track_embeddings.reshape(B, Hs, Ws, D)

        # --- Step 3: Project down ---
        src_proj = self.proj_down(src_grid)  # (B, Hs, Ws, proj_dim)
        src_proj = src_proj.permute(0, 3, 1, 2)  # (B, proj_dim, Hs, Ws)

        # --- Step 4: Conv2d patchify ---
        # Pad to exact multiple of patch_size if needed
        pad_h = (ps - Hs % ps) % ps
        pad_w = (ps - Ws % ps) % ps
        if pad_h > 0 or pad_w > 0:
            src_proj = nn.functional.pad(src_proj, (0, pad_w, 0, pad_h))

        src_latent = self.conv(src_proj)  # (B, proj_dim, conv_h, conv_w)
        conv_h, conv_w = src_latent.shape[2], src_latent.shape[3]

        # --- Step 5: Project up ---
        src_latent = src_latent.permute(0, 2, 3, 1)  # (B, conv_h, conv_w, proj_dim)
        src_tokens = self.proj_up(src_latent)  # (B, conv_h, conv_w, out_dim)

        # --- Step 6: Source-to-target scatter at latent resolution ---
        # For each source patch, compute its mean target pixel position from
        # the projected UVs of tracks within that patch.
        Hs_padded = (Hs + pad_h)
        Ws_padded = (Ws + pad_w)

        track_uv_grid = track_uv.reshape(B, Hs, Ws, T, 2)
        track_vis_grid = track_vis.reshape(B, Hs, Ws, T)

        # Pad track UV/vis to match padded conv input
        if pad_h > 0 or pad_w > 0:
            uv_pad = torch.zeros(B, Hs_padded, Ws_padded, T, 2, device=device, dtype=dtype)
            vis_pad = torch.zeros(B, Hs_padded, Ws_padded, T, device=device, dtype=torch.bool)
            uv_pad[:, :Hs, :Ws] = track_uv_grid
            vis_pad[:, :Hs, :Ws] = track_vis_grid
            track_uv_grid = uv_pad
            track_vis_grid = vis_pad

        # Reshape into patches: (B, conv_h, ps, conv_w, ps, T, 2)
        uv_patches = track_uv_grid.reshape(B, conv_h, ps, conv_w, ps, T, 2)
        vis_patches = track_vis_grid.reshape(B, conv_h, ps, conv_w, ps, T)

        # Mean UV per patch per frame (visible tracks only)
        vis_count = vis_patches.float().sum(dim=(2, 4))  # (B, conv_h, conv_w, T)
        vis_count_safe = vis_count.clamp(min=1.0)
        uv_masked = uv_patches * vis_patches.unsqueeze(-1).float()
        patch_uv = uv_masked.sum(dim=(2, 4)) / vis_count_safe.unsqueeze(-1)
        # patch_uv: (B, conv_h, conv_w, T, 2)
        patch_vis = vis_count > 0  # (B, conv_h, conv_w, T)

        # Map target pixel → target latent cell
        tgt_lw = (patch_uv[..., 0] / image_w * latent_w).clamp(0, latent_w - 1e-3).long()
        tgt_lh = (patch_uv[..., 1] / image_h * latent_h).clamp(0, latent_h - 1e-3).long()

        # Map frame → target temporal latent
        frame_idx = torch.arange(T, device=device).view(1, 1, 1, T).expand(
            B, conv_h, conv_w, T
        )
        tgt_lt = (frame_idx.float() / max(num_frames - 1, 1) * max(latent_t - 1, 1)).clamp(
            0, latent_t - 1e-3
        ).long()

        # Flat target block index: (B, conv_h, conv_w, T)
        tgt_block = (
            (tgt_lt // self.block_t).clamp(0, num_bt - 1) * (num_bh * num_bw) +
            (tgt_lh // self.block_h).clamp(0, num_bh - 1) * num_bw +
            (tgt_lw // self.block_w).clamp(0, num_bw - 1)
        )

        # Scatter source tokens to target blocks via max-pool
        # Expand src_tokens across T: (B, conv_h, conv_w, T, out_dim)
        src_expanded = src_tokens.unsqueeze(3).expand(B, conv_h, conv_w, T, self.out_dim)

        # Flatten for scatter
        num_obs = conv_h * conv_w * T
        src_flat = src_expanded.reshape(B, num_obs, self.out_dim)
        tgt_flat = tgt_block.reshape(B, num_obs)
        vis_flat = patch_vis.reshape(B, num_obs)

        # Mask invisible
        src_masked = src_flat.clone()
        src_masked[~vis_flat] = float("-inf")

        # Batch-offset for scatter
        batch_offset = torch.arange(B, device=device).view(B, 1) * num_blocks
        tgt_global = tgt_flat + batch_offset

        total_blocks_all = B * num_blocks

        src_all = src_masked.reshape(B * num_obs, self.out_dim)
        idx_all = tgt_global.reshape(B * num_obs)

        # Scatter max-pool
        block_vals = torch.full(
            (total_blocks_all, self.out_dim), float("-inf"),
            device=device, dtype=dtype,
        )
        idx_expanded = idx_all.unsqueeze(-1).expand(-1, self.out_dim)
        block_vals.scatter_reduce_(
            0, idx_expanded, src_all, reduce="amax", include_self=True,
        )

        # Determine occupied blocks
        occupied_flat = torch.zeros(total_blocks_all, dtype=torch.bool, device=device)
        valid_idx = idx_all[vis_flat.reshape(-1)]
        if valid_idx.numel() > 0:
            occupied_flat.scatter_(0, valid_idx, torch.ones_like(valid_idx, dtype=torch.bool))

        # Replace -inf with null_token for empty blocks
        null = self.null_token.unsqueeze(0).expand(total_blocks_all, -1)
        block_tokens = torch.where(occupied_flat.unsqueeze(-1), block_vals, null)

        block_tokens = block_tokens.reshape(B, num_blocks, self.out_dim)
        block_mask = occupied_flat.reshape(B, num_blocks)

        return block_tokens, block_mask
