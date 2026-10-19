"""
PointTrackEmbedder: encode a point track sequence into one embedding per track.

Architecture (Koichi-style, adapted for 3D dense tracks):
  1. Sinusoidal encoding of (frame_idx, world_x, world_y, world_z, vis) per timestep
  2. Shared MLP applied per timestep
  3. Temporal aggregation over timesteps (max-pool or cross-attention)
  4. Output: one embedding per track

Input:  track_seq (B, N, T, C)  with C=5 (frame_idx, world_x, world_y, world_z, vis)
Output: track_emb (B, N, D)

Temporal aggregation modes:
  - "max": Koichi-style temporal max-pool (default). Robust, parameter-free.
  - "attention": Learnable cross-attention with a single query token.
    Strictly more expressive (max-pool is a special case of hard-argmax attention).
    A learnable query attends to all T timesteps via multi-head cross-attention,
    with visibility used as key_padding_mask.

Key difference from Koichi (2D sparse tracks):
  - Our input is dense 3D world-space tracks, not projected 2D coordinates
  - Sinusoidal + MLP preserves high-frequency 3D spatial info
  - 2D projection is only used by the adapter for latent grid placement
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Literal


class SinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous values.

    Encodes each scalar value into a vector of (sin, cos) pairs at different frequencies.
    """

    def __init__(self, num_freqs: int = 16, include_input: bool = True):
        """
        Args:
            num_freqs: number of frequency bands.
            include_input: if True, prepend the raw input value.
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # Output dim per input scalar: 2*num_freqs (+ 1 if include_input)
        self.out_dim_per_channel = 2 * num_freqs + (1 if include_input else 0)

        # Precompute frequency bands: 2^0, 2^1, ..., 2^(num_freqs-1)
        freqs = 2.0 ** torch.arange(num_freqs).float()  # (num_freqs,)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., C) input values, each in roughly [0, 1].

        Returns:
            encoded: (..., C * out_dim_per_channel)
        """
        # x: (..., C)
        # Expand for frequency multiplication: (..., C, 1) * (num_freqs,) → (..., C, num_freqs)
        x_freq = x.unsqueeze(-1) * self.freqs * math.pi  # (..., C, num_freqs)

        sin_enc = torch.sin(x_freq)  # (..., C, num_freqs)
        cos_enc = torch.cos(x_freq)  # (..., C, num_freqs)

        parts = []
        if self.include_input:
            parts.append(x.unsqueeze(-1))  # (..., C, 1)
        parts.append(sin_enc)
        parts.append(cos_enc)

        # Concat along last dim: (..., C, out_dim_per_channel)
        encoded = torch.cat(parts, dim=-1)

        # Flatten C and encoding dims: (..., C * out_dim_per_channel)
        shape = encoded.shape[:-2]
        encoded = encoded.reshape(*shape, -1)

        return encoded


class TemporalCrossAttentionPool(nn.Module):
    """Pool T timesteps into one embedding via learnable cross-attention.

    A single learnable query token attends to all T key/value timesteps
    using multi-head cross-attention. Visibility is enforced via
    key_padding_mask (invisible timesteps are ignored).

    This is strictly more expressive than max-pool: max-pool is the special
    case where attention weight is a hard argmax. Cross-attention can learn
    soft weighted combinations and exploit temporal ordering (since the input
    already contains sinusoidal frame_idx encoding).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, T, D) per-timestep MLP outputs.
            mask: (B, N, T) bool — True = valid timestep.

        Returns:
            pooled: (B, N, D)
        """
        B, N, T, D = features.shape

        # Reshape to (B*N, T, D) for batched attention
        x = features.reshape(B * N, T, D)
        # MHA key_padding_mask: True = IGNORE
        key_padding_mask = ~mask.reshape(B * N, T)

        # Detect fully-masked tracks (all timesteps invisible)
        all_masked = key_padding_mask.all(dim=1)  # (B*N,)
        if all_masked.any():
            # Temporarily unmask to avoid NaN in softmax
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False

        query = self.query.expand(B * N, -1, -1)  # (B*N, 1, D)
        out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        out = out.squeeze(1)  # (B*N, D)

        # Zero out fully-masked tracks
        if all_masked.any():
            out = out.clone()
            out[all_masked] = 0.0

        return out.reshape(B, N, D)


class PointTrackEmbedder(nn.Module):
    """Encode per-track temporal sequences into fixed-size embeddings.

    Architecture:
        track_seq (B, N, T, C=5)
        → sinusoidal_encode → (B, N, T, C * enc_dim)
        → shared MLP per timestep → (B, N, T, out_dim)
        → temporal aggregation → (B, N, out_dim)

    Temporal aggregation modes:
        "max": masked temporal max-pool (Koichi default)
        "attention": learnable cross-attention with 1 query token
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_freqs: int = 16,
        hidden_dim: int = 156,
        out_dim: int = 156,
        num_mlp_layers: int = 3,
        temporal_pool: Literal["max", "attention"] = "max",
        temporal_attn_heads: int = 4,
    ):
        """
        Args:
            in_channels: number of input channels per timestep
                (default 5: frame_idx, world_x, world_y, world_z, vis).
            num_freqs: number of sinusoidal frequency bands per channel.
            hidden_dim: MLP hidden dimension.
            out_dim: output embedding dimension per track.
            num_mlp_layers: number of MLP layers (including output).
            temporal_pool: "max" (masked max-pool) or "attention" (cross-attention).
            temporal_attn_heads: number of attention heads (only for temporal_pool="attention").
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.temporal_pool = temporal_pool

        self.sin_enc = SinusoidalEncoding(num_freqs=num_freqs, include_input=True)
        enc_dim = in_channels * self.sin_enc.out_dim_per_channel

        # Shared MLP applied per timestep
        layers = []
        in_d = enc_dim
        for i in range(num_mlp_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.GELU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, out_dim))
        self.mlp = nn.Sequential(*layers)

        # Temporal aggregation
        if temporal_pool == "attention":
            self.temporal_attn_pool = TemporalCrossAttentionPool(
                dim=out_dim, num_heads=temporal_attn_heads,
            )
        else:
            self.temporal_attn_pool = None

    def forward(
        self,
        track_seq: torch.Tensor,
        track_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            track_seq: (B, N, T, C) per-timestep track features.
                C=5: (frame_idx, world_x, world_y, world_z, vis) — all in [0, 1] range.
            track_mask: (B, N, T) optional bool mask.
                True = valid timestep, False = masked.
                If None, uses last channel (vis) as mask (track_seq[..., -1] > 0.5).

        Returns:
            track_emb: (B, N, out_dim) one embedding per track.
        """
        B, N, T, C = track_seq.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Default mask from visibility channel (last channel)
        if track_mask is None:
            track_mask = track_seq[..., -1] > 0.5  # (B, N, T)

        # Sinusoidal encoding: (B, N, T, enc_dim)
        encoded = self.sin_enc(track_seq)

        # Shared MLP per timestep: (B, N, T, out_dim)
        features = self.mlp(encoded)  # (B, N, T, out_dim)

        if self.temporal_pool == "attention":
            # Cross-attention pooling
            track_emb = self.temporal_attn_pool(features, track_mask)
        else:
            # Masked temporal max-pool (Koichi default)
            mask_expanded = track_mask.unsqueeze(-1).expand_as(features)  # (B, N, T, out_dim)
            features = features.masked_fill(~mask_expanded, float("-inf"))
            track_emb, _ = features.max(dim=2)  # (B, N, out_dim)
            # Handle fully-masked tracks: replace -inf with 0
            all_masked = ~track_mask.any(dim=2)  # (B, N)
            track_emb = track_emb.masked_fill(all_masked.unsqueeze(-1), 0.0)

        return track_emb
