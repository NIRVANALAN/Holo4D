"""
TokenConcatProjector: fuse video latent tokens with track block tokens
via channel-wise concatenation + MLP projection back to DiT hidden dim.

In the Wan 5B DiT:
  - Video latent tokens: (B, L, dim) where dim=1536, L = latent_t * latent_h * latent_w
  - Track block tokens: (B, L, track_dim) — must be aligned to same L

This module:
  1. Concatenates video tokens and track tokens along channel dim
  2. Projects back to model hidden dim via MLP
"""

import torch
import torch.nn as nn


class TokenConcatProjector(nn.Module):
    """Channel-wise concatenation + projection for track injection.

    Concatenates video latent tokens with track condition tokens along
    the channel dimension, then projects back to the backbone hidden dim.
    """

    def __init__(
        self,
        video_dim: int,
        track_dim: int,
        hidden_dim: int,
        zero_init: bool = True,
    ):
        """
        Args:
            video_dim: dimension of video latent tokens (e.g. 1536 for Wan 5B).
            track_dim: dimension of track block tokens from PointTrackAdapter.
            hidden_dim: output dimension (should equal video_dim for DiT compatibility).
            zero_init: if True, zero-initialize the output layer so that
                the track injection has no effect at initialization.
                This preserves pretrained backbone behavior at start of training.
        """
        super().__init__()
        self.video_dim = video_dim
        self.track_dim = track_dim
        self.hidden_dim = hidden_dim

        in_dim = video_dim + track_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if zero_init:
            # ReCamMaster-style init for gradient flow:
            # - First layer (encoder side): zero-init → output starts at 0
            # - Last layer (projector side): identity-init → gradients flow freely
            # At init: proj(cat) = I @ GELU(0) = 0, so residual gives video + 0 = video.
            # But ∂output/∂hidden = I (not zero), so first layer gets gradient immediately.
            nn.init.zeros_(self.proj[0].weight)
            nn.init.zeros_(self.proj[0].bias)
            nn.init.eye_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        video_tokens: torch.Tensor,
        track_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_tokens: (B, L, video_dim) — latent tokens from DiT patchify.
            track_tokens: (B, L, track_dim) — track block tokens aligned to same L.
                L = latent_t * latent_h * latent_w (flattened spatiotemporal grid).

        Returns:
            fused: (B, L, hidden_dim) — tokens ready for DiT backbone.
        """
        B, L, Dv = video_tokens.shape
        _, Lt, Dt = track_tokens.shape

        assert L == Lt, (
            f"Token sequence length mismatch: video={L}, track={Lt}. "
            f"Track tokens must be aligned to the same spatiotemporal grid."
        )
        assert Dv == self.video_dim, f"Expected video_dim={self.video_dim}, got {Dv}"
        assert Dt == self.track_dim, f"Expected track_dim={self.track_dim}, got {Dt}"

        # Channel-wise concatenate: (B, L, video_dim + track_dim)
        concat = torch.cat([video_tokens, track_tokens], dim=-1)

        # Project back to hidden dim: (B, L, hidden_dim)
        fused = self.proj(concat)

        # Residual: add back video tokens so that zero-init starts as identity
        fused = video_tokens + fused

        return fused
