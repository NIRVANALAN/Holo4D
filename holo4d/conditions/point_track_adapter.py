"""
PointTrackAdapter: convert per-track embeddings into block-aligned compressed
control tokens for injection into the Wan DiT.

Architecture:
  1. Divide the latent video grid into spatiotemporal blocks of (block_t, block_h, block_w).
  2. For each block, gather tracks whose projected 2D positions fall in that block.
  3. Concatenate each track embedding with its relative position inside the block.
  4. Apply MLP + max-pool over tracks in the block.
  5. Output one compressed token per block.

Latent grid dimensions (after VAE + patchify for 5B Wan):
  - VAE: temporal 4x, spatial 8x compression, z_dim=16
  - Patchify: spatial 2x (patch_size=[1,2,2])
  - So for input (T, H, W):
    latent_t = (T + 3) // 4
    latent_h = H // 8 // 2 = H // 16
    latent_w = W // 8 // 2 = W // 16

Default block size in latent space: (4, 2, 2)
  - This means each block covers:
    4 latent temporal steps × 2 latent spatial h × 2 latent spatial w
  - In pixel space: ~16 frames × 32 pixels × 32 pixels
"""

import os

import torch
import torch.nn as nn
from typing import Optional


class PointTrackAdapter(nn.Module):
    """Convert per-track embeddings into block-aligned latent control tokens.

    The output tokens are spatiotemporally aligned with the DiT's flattened
    latent sequence and can be channel-concatenated with video latent tokens.
    """

    def __init__(
        self,
        track_dim: int = 156,
        out_dim: int = 156,
        block_t: int = 1,
        block_h: int = 1,
        block_w: int = 1,
        rel_pos_dim: int = 3,
        hidden_dim: int = 156,
    ):
        """
        Args:
            track_dim: dimension of per-track embeddings from PointTrackEmbedder.
            out_dim: output dimension per block token.
            block_t: temporal block size in latent frames.
            block_h: spatial height block size in latent patches.
            block_w: spatial width block size in latent patches.
            rel_pos_dim: dimension of relative position encoding (t_rel, h_rel, w_rel).
            hidden_dim: MLP hidden dimension.
        """
        super().__init__()
        self.block_t = block_t
        self.block_h = block_h
        self.block_w = block_w
        self.out_dim = out_dim

        # MLP: (track_dim + rel_pos_dim) → hidden → out_dim
        self.mlp = nn.Sequential(
            nn.Linear(track_dim + rel_pos_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Null token for empty blocks — zero-init so unoccupied cells inject nothing
        # (Koichi zeroes out unoccupied outputs after occupancy masking; this matches that)
        self.null_token = nn.Parameter(torch.zeros(out_dim))
        # Opt-in hard-zero null (TRACK4D360_TRACK_NULL_TOKEN_FROZEN=1): freeze the
        # empty-block token so it cannot drift toward occupied-token statistics.
        if os.environ.get("TRACK4D360_TRACK_NULL_TOKEN_FROZEN", "0") == "1":
            self.null_token.requires_grad_(False)

    def _pixel_to_latent_coords(
        self,
        track_uv: torch.Tensor,
        track_frame_idx: torch.Tensor,
        image_h: int,
        image_w: int,
        num_frames: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
    ):
        """Convert pixel-space track positions to latent grid coordinates.

        Args:
            track_uv: (B, N, T, 2) pixel coordinates (u=col, v=row).
            track_frame_idx: (B, N, T) frame indices (0..num_frames-1).
            image_h, image_w: pixel image size.
            num_frames: total number of input video frames.
            latent_t, latent_h, latent_w: latent grid dimensions.

        Returns:
            latent_coords: (B, N, T, 3) — (lt, lh, lw) in latent grid, float.
        """
        # Map pixel (u, v) → latent (lw, lh)
        # Pixel → latent: divide by (image_size / latent_size)
        lw = track_uv[..., 0] / image_w * latent_w  # (B, N, T)
        lh = track_uv[..., 1] / image_h * latent_h  # (B, N, T)

        # Map frame_idx → latent temporal
        lt = track_frame_idx / max(num_frames - 1, 1) * max(latent_t - 1, 1)  # (B, N, T)

        return torch.stack([lt, lh, lw], dim=-1)  # (B, N, T, 3)

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
    ) -> torch.Tensor:
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
                num_blocks = ceil(latent_t/block_t) * ceil(latent_h/block_h) * ceil(latent_w/block_w)
            block_mask: (B, num_blocks) — bool, True if block has at least one track.
        """
        B, N, D = track_embeddings.shape
        _, _, T, _ = track_uv.shape
        device = track_embeddings.device

        # Compute block grid dimensions
        num_bt = (latent_t + self.block_t - 1) // self.block_t
        num_bh = (latent_h + self.block_h - 1) // self.block_h
        num_bw = (latent_w + self.block_w - 1) // self.block_w
        num_blocks = num_bt * num_bh * num_bw

        # Frame indices: (B, N, T)
        frame_idx = torch.arange(T, device=device, dtype=track_uv.dtype)
        frame_idx = frame_idx.view(1, 1, T).expand(B, N, T)

        # Convert to latent coords: (B, N, T, 3) — (lt, lh, lw)
        latent_coords = self._pixel_to_latent_coords(
            track_uv, frame_idx, image_h, image_w, num_frames,
            latent_t, latent_h, latent_w,
        )

        # Assign each (track, timestep) observation to a block
        lt = latent_coords[..., 0].clamp(0, latent_t - 1e-3)
        lh = latent_coords[..., 1].clamp(0, latent_h - 1e-3)
        lw = latent_coords[..., 2].clamp(0, latent_w - 1e-3)

        block_t_idx = (lt / self.block_t).long().clamp(0, num_bt - 1)  # (B, N, T)
        block_h_idx = (lh / self.block_h).long().clamp(0, num_bh - 1)
        block_w_idx = (lw / self.block_w).long().clamp(0, num_bw - 1)

        # Flat block index: (B, N, T)
        block_flat = block_t_idx * (num_bh * num_bw) + block_h_idx * num_bw + block_w_idx

        # Relative position inside block: [0, 1) per dimension
        rel_t = (lt / self.block_t - block_t_idx.float()).clamp(0, 1)  # (B, N, T)
        rel_h = (lh / self.block_h - block_h_idx.float()).clamp(0, 1)
        rel_w = (lw / self.block_w - block_w_idx.float()).clamp(0, 1)
        rel_pos = torch.stack([rel_t, rel_h, rel_w], dim=-1)  # (B, N, T, 3)

        # For each observation, concat track embedding + relative position
        # track_embeddings: (B, N, D) → expand to (B, N, T, D)
        track_emb_expanded = track_embeddings.unsqueeze(2).expand(B, N, T, D)

        # Concat: (B, N, T, D+3) — cast rel_pos to match embedding dtype
        rel_pos = rel_pos.to(dtype=track_emb_expanded.dtype)
        obs_features = torch.cat([track_emb_expanded, rel_pos], dim=-1)

        # Apply MLP per observation: (B, N, T, out_dim)
        obs_tokens = self.mlp(obs_features)

        # Aggregate into blocks via batched scatter max-pool (no Python for-loop)
        #
        # Strategy: offset block indices per batch element so we can scatter
        # all batch elements in a single call on a (B*num_blocks, out_dim) tensor.
        valid = track_vis.bool()  # (B, N, T)

        # Flatten observations: (B, N*T)
        obs_flat = obs_tokens.reshape(B, N * T, self.out_dim)     # (B, N*T, D)
        block_flat_2d = block_flat.reshape(B, N * T)               # (B, N*T)
        valid_flat = valid.reshape(B, N * T)                       # (B, N*T)

        # Mask invalid observations to -inf (won't win in max-pool)
        obs_masked = obs_flat.clone()
        obs_masked[~valid_flat] = float("-inf")

        # Add per-batch offset so block indices are unique across batch:
        # batch b gets block indices in [b*num_blocks, (b+1)*num_blocks)
        batch_offset = torch.arange(B, device=device).view(B, 1) * num_blocks  # (B, 1)
        block_global = block_flat_2d + batch_offset  # (B, N*T)

        # Flatten everything to 1D for scatter
        obs_all = obs_masked.reshape(B * N * T, self.out_dim)      # (B*N*T, D)
        idx_all = block_global.reshape(B * N * T)                   # (B*N*T,)

        # Scatter max-pool into (B*num_blocks, out_dim)
        total_blocks = B * num_blocks
        block_vals = torch.full(
            (total_blocks, self.out_dim), float("-inf"),
            device=device, dtype=obs_all.dtype,
        )
        idx_expanded = idx_all.unsqueeze(-1).expand(-1, self.out_dim)
        block_vals.scatter_reduce_(
            0, idx_expanded, obs_all, reduce="amax", include_self=True,
        )

        # Determine which blocks are occupied (received at least one valid obs)
        occupied_flat = torch.zeros(total_blocks, dtype=torch.bool, device=device)
        valid_idx = idx_all[valid_flat.reshape(-1)]
        occupied_flat.scatter_(0, valid_idx, torch.ones_like(valid_idx, dtype=torch.bool))

        # Replace -inf with null token for unoccupied blocks
        null = self.null_token.unsqueeze(0).expand(total_blocks, -1)
        block_tokens = torch.where(occupied_flat.unsqueeze(-1), block_vals, null)

        # Reshape back to (B, num_blocks, out_dim)
        block_tokens = block_tokens.reshape(B, num_blocks, self.out_dim)
        block_mask = occupied_flat.reshape(B, num_blocks)

        return block_tokens, block_mask


def expand_block_tokens_to_latent_seq(
    block_tokens: torch.Tensor,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    block_t: int,
    block_h: int,
    block_w: int,
) -> torch.Tensor:
    """Expand block tokens (B, num_blocks, D) to full latent sequence (B, L, D).

    Each latent position (t, h, w) maps to the block it belongs to.
    The resulting tensor is aligned with the DiT's flattened token sequence.

    Args:
        block_tokens: (B, num_blocks, D) — one token per spatiotemporal block.
        latent_t, latent_h, latent_w: latent grid dimensions.
        block_t, block_h, block_w: block sizes.

    Returns:
        expanded: (B, L, D) where L = latent_t * latent_h * latent_w.
    """
    B, num_blocks, D = block_tokens.shape
    num_bt = (latent_t + block_t - 1) // block_t
    num_bh = (latent_h + block_h - 1) // block_h
    num_bw = (latent_w + block_w - 1) // block_w

    t_idx = torch.arange(latent_t, device=block_tokens.device)
    h_idx = torch.arange(latent_h, device=block_tokens.device)
    w_idx = torch.arange(latent_w, device=block_tokens.device)
    tt, hh, ww = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")

    bt = (tt // block_t).clamp(max=num_bt - 1)
    bh = (hh // block_h).clamp(max=num_bh - 1)
    bw = (ww // block_w).clamp(max=num_bw - 1)
    block_idx = bt * (num_bh * num_bw) + bh * num_bw + bw
    block_idx_flat = block_idx.reshape(-1)  # (L,)

    block_idx_expanded = block_idx_flat.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
    expanded = block_tokens.gather(1, block_idx_expanded)
    return expanded
