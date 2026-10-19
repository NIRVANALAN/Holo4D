"""Opt-in persistent trajectory identity conditioning.

The legacy Track4D adapter remains unchanged. This module adds a separate
identity path that computes one embedding per complete world-space trajectory,
then rasterizes the same embedding into target and, optionally, source tokens.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn


VALID_IDENTITY_MODES = {"legacy", "target", "shared"}


def normalize_persistent_identity_mode(mode: Optional[str]) -> str:
    value = str(mode or "legacy").strip().lower().replace("-", "_")
    aliases = {
        "off": "legacy",
        "none": "legacy",
        "id_target": "target",
        "target_only": "target",
        "id_shared": "shared",
        "source_target": "shared",
    }
    value = aliases.get(value, value)
    if value not in VALID_IDENTITY_MODES:
        raise ValueError(
            f"persistent_identity_mode={mode!r} is invalid; "
            f"expected one of {sorted(VALID_IDENTITY_MODES)}"
        )
    return value


def persistent_identity_only_enabled(mode: Optional[str]) -> bool:
    """Return the opt-in identity-only policy and reject a legacy mismatch.

    The flag changes token composition only; it does not change the module
    graph or checkpoint ABI.  This makes a trained hybrid identity checkpoint
    usable as the common initialization for field-on/field-off continuations.
    """
    enabled = os.environ.get("HOLO4D_TRACK_IDENTITY_ONLY", "0") == "1"
    normalized = normalize_persistent_identity_mode(mode)
    if enabled and normalized == "legacy":
        raise ValueError(
            "HOLO4D_TRACK_IDENTITY_ONLY=1 requires "
            "persistent_identity_mode=target or shared; legacy has no identity "
            "tokens to inject."
        )
    return enabled


def compose_target_identity_tokens(
    base_tokens: torch.Tensor,
    identity_tokens: torch.Tensor,
    *,
    identity_only: bool,
) -> torch.Tensor:
    """Compose the legacy geometric field and persistent target identity."""
    if tuple(base_tokens.shape) != tuple(identity_tokens.shape):
        raise ValueError(
            "base/identity target token shape mismatch: "
            f"{tuple(base_tokens.shape)} versus {tuple(identity_tokens.shape)}"
        )
    return identity_tokens if identity_only else base_tokens + identity_tokens


def compose_source_identity_tokens(
    legacy_source_null: torch.Tensor,
    identity_tokens: torch.Tensor,
    *,
    identity_only: bool,
) -> torch.Tensor:
    """Compose source tokens for shared identity mode."""
    if tuple(legacy_source_null.shape) != tuple(identity_tokens.shape):
        raise ValueError(
            "source-null/identity token shape mismatch: "
            f"{tuple(legacy_source_null.shape)} versus {tuple(identity_tokens.shape)}"
        )
    return identity_tokens if identity_only else legacy_source_null + identity_tokens


def build_persistent_trajectory_inputs(
    tracks_world: torch.Tensor,
    tracks_world_valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build target-camera-invariant ``(time, normalized XYZ)`` identity inputs."""
    if tracks_world.ndim == 3:
        tracks_world = tracks_world.unsqueeze(0)
    if tracks_world.ndim != 4 or tracks_world.shape[-1] != 3:
        raise ValueError(
            "tracks_world must be (N,T,3) or (B,N,T,3), got "
            f"{tuple(tracks_world.shape)}"
        )
    tracks = tracks_world.to(dtype=torch.float32)
    batch, identities, frames, _ = tracks.shape
    finite = torch.isfinite(tracks).all(dim=-1)
    if tracks_world_valid_mask is None:
        valid = finite
    else:
        valid = torch.as_tensor(
            tracks_world_valid_mask,
            device=tracks.device,
            dtype=torch.bool,
        )
        if valid.ndim == 2:
            valid = valid.unsqueeze(0)
        if valid.shape[0] == 1 and batch > 1:
            valid = valid.expand(batch, -1, -1)
        if tuple(valid.shape) != (batch, identities, frames):
            raise ValueError(
                "tracks_world_valid_mask must match tracks_world as (B,N,T): "
                f"got {tuple(valid.shape)}, expected {(batch, identities, frames)}"
            )
        valid = valid & finite

    safe = torch.where(valid.unsqueeze(-1), tracks, torch.zeros_like(tracks))
    positive_inf = torch.full_like(safe, float("inf"))
    negative_inf = torch.full_like(safe, float("-inf"))
    xyz_min = torch.where(valid.unsqueeze(-1), safe, positive_inf).reshape(
        batch, -1, 3
    ).amin(dim=1)
    xyz_max = torch.where(valid.unsqueeze(-1), safe, negative_inf).reshape(
        batch, -1, 3
    ).amax(dim=1)
    has_valid = valid.reshape(batch, -1).any(dim=1)
    xyz_min = torch.where(has_valid[:, None], xyz_min, torch.zeros_like(xyz_min))
    xyz_max = torch.where(has_valid[:, None], xyz_max, torch.ones_like(xyz_max))
    xyz_range = (xyz_max - xyz_min).clamp_min(1e-6)
    xyz = (safe - xyz_min[:, None, None]) / xyz_range[:, None, None]
    xyz = xyz.clamp(0.0, 1.0) * valid.unsqueeze(-1)
    time = torch.linspace(
        0.0,
        1.0,
        frames,
        device=tracks.device,
        dtype=tracks.dtype,
    ).view(1, 1, frames, 1).expand(batch, identities, -1, -1)
    sequence = torch.cat([time, xyz], dim=-1)
    return sequence, valid


class FourierEncoding(nn.Module):
    def __init__(self, channels: int, num_freqs: int):
        super().__init__()
        self.channels = int(channels)
        self.num_freqs = int(num_freqs)
        self.out_dim = self.channels * (1 + 2 * self.num_freqs)
        self.register_buffer(
            "freqs",
            2.0 ** torch.arange(self.num_freqs, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.channels:
            raise ValueError(
                f"Expected final dimension {self.channels}, got {value.shape[-1]}"
            )
        angles = value.unsqueeze(-1) * self.freqs.to(value) * math.pi
        encoded = torch.cat(
            [value.unsqueeze(-1), torch.sin(angles), torch.cos(angles)],
            dim=-1,
        )
        return encoded.flatten(start_dim=-2)


class PersistentTrajectoryIdentityAdapter(nn.Module):
    """Pool complete trajectories, then splat coherent point identities.

    Identity encoding is target-camera-independent. Source and target
    rasterization calls reuse the exact same ``identity_embedding`` tensor.
    Empty output tokens are hard-zeroed and are never represented by a learned
    null parameter.
    """

    def __init__(
        self,
        out_dim: int,
        identity_dim: int = 128,
        hidden_dim: int = 156,
        identity_num_freqs: int = 8,
        relative_num_freqs: int = 4,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.identity_dim = int(identity_dim)
        self.identity_encoding = FourierEncoding(4, identity_num_freqs)
        self.relative_encoding = FourierEncoding(3, relative_num_freqs)
        self.identity_mlp = nn.Sequential(
            nn.Linear(self.identity_encoding.out_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.identity_dim),
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(
                self.identity_dim + self.relative_encoding.out_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )
        # A newly enabled identity branch starts as a strict no-op on top of a
        # legacy checkpoint. Training must learn to open this branch.
        nn.init.zeros_(self.point_mlp[-1].weight)
        nn.init.zeros_(self.point_mlp[-1].bias)

    def encode_identities(
        self,
        identity_sequence: torch.Tensor,
        identity_valid: torch.Tensor,
    ) -> torch.Tensor:
        if identity_sequence.ndim != 4 or identity_sequence.shape[-1] != 4:
            raise ValueError(
                "identity_sequence must be (B,N,T,4), got "
                f"{tuple(identity_sequence.shape)}"
            )
        if tuple(identity_valid.shape) != tuple(identity_sequence.shape[:3]):
            raise ValueError(
                "identity_valid must match identity_sequence as (B,N,T), got "
                f"{tuple(identity_valid.shape)} versus {tuple(identity_sequence.shape[:3])}"
            )
        encoded = self.identity_encoding(identity_sequence)
        per_time = self.identity_mlp(encoded)
        valid = identity_valid.bool()
        per_time = per_time.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        identity = per_time.amax(dim=2)
        identity = torch.nan_to_num(identity, neginf=0.0, posinf=0.0)
        return identity

    @staticmethod
    def _latent_time_indices(
        frames: int,
        latent_t: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frame = torch.arange(frames, device=device)
        latent = torch.where(
            frame == 0,
            torch.zeros_like(frame),
            (frame - 1) // 4 + 1,
        ).clamp_max(latent_t - 1)
        relative = torch.where(
            frame == 0,
            torch.zeros_like(frame, dtype=torch.float32),
            ((frame - 1) % 4 + 1).float() / 4.0,
        )
        return latent, relative

    def rasterize(
        self,
        identity_embedding: torch.Tensor,
        identity_valid: torch.Tensor,
        track_uv: torch.Tensor,
        track_vis: torch.Tensor,
        *,
        image_h: int,
        image_w: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if track_uv.ndim != 4 or track_uv.shape[-1] != 2:
            raise ValueError(f"track_uv must be (B,N,T,2), got {tuple(track_uv.shape)}")
        if tuple(track_vis.shape) != tuple(track_uv.shape[:3]):
            raise ValueError("track_vis must match track_uv as (B,N,T)")
        batch, identities, frames = track_vis.shape
        if tuple(identity_embedding.shape[:2]) != (batch, identities):
            raise ValueError("identity_embedding must match track_uv as (B,N,D)")
        if tuple(identity_valid.shape) != (batch, identities, frames):
            raise ValueError("identity_valid must match track_uv as (B,N,T)")

        device = track_uv.device
        dtype = identity_embedding.dtype
        latent_frame, relative_t = self._latent_time_indices(
            frames, latent_t, device
        )
        u_scaled = track_uv[..., 0] / max(float(image_w), 1.0) * latent_w
        v_scaled = track_uv[..., 1] / max(float(image_h), 1.0) * latent_h
        u_index = torch.floor(u_scaled).long()
        v_index = torch.floor(v_scaled).long()
        inside = (
            (u_index >= 0)
            & (u_index < latent_w)
            & (v_index >= 0)
            & (v_index < latent_h)
        )
        visible = track_vis.bool() & identity_valid.bool() & inside
        u_index = u_index.clamp(0, latent_w - 1)
        v_index = v_index.clamp(0, latent_h - 1)
        t_index = latent_frame.view(1, 1, frames).expand(batch, identities, -1)

        token_count = latent_t * latent_h * latent_w
        tokens = None  # running max over chunks; built out-of-place for autograd
        occupancy = torch.zeros(
            (batch, token_count), device=device, dtype=torch.float32
        )

        # Chunk over the identity (track) dimension. point_feature is
        # (B, N, T, out_dim), and with DENSE tracking N == pixel count -- 147K at
        # 288x512 but 901K at 1280x704, and this runs twice for ID-shared (source
        # + target). Building it in one shot OOMs a 95 GB GPU (job 6021741 died
        # exactly here at rasterize()).
        #
        # Each chunk scatters into its OWN fresh -inf buffer (out-of-place
        # scatter_reduce), then we combine chunks with torch.maximum. This is
        # numerically identical to a single amax scatter (amax is associative)
        # AND autograd-safe: repeated IN-PLACE scatter_reduce_ into one shared
        # `tokens` bumps its version counter every chunk and breaks the backward
        # ("variable ... modified by an inplace operation ... ScatterReduceBackward0
        # is at version N; expected N-1" -- jobs 6035488/6035489). The token grid
        # is tiny (token_count x out_dim, a few MB), so one buffer per chunk is
        # cheap. occupancy carries no gradient (it only feeds a boolean mask), so
        # its in-place scatter_reduce_ is fine. 0 disables chunking.
        chunk = int(os.environ.get("HOLO4D_IDENTITY_RASTER_CHUNK", "32768"))
        if chunk <= 0 or chunk >= identities:
            chunk = identities
        identity_exp = identity_embedding.unsqueeze(2)  # (B, N, 1, D)
        for lo in range(0, identities, chunk):
            hi = min(lo + chunk, identities)
            rel_c = torch.stack(
                [
                    relative_t.view(1, 1, frames).expand(batch, hi - lo, -1),
                    (u_scaled[:, lo:hi] - torch.floor(u_scaled[:, lo:hi])).clamp(0.0, 1.0),
                    (v_scaled[:, lo:hi] - torch.floor(v_scaled[:, lo:hi])).clamp(0.0, 1.0),
                ],
                dim=-1,
            ).to(dtype=dtype)
            point_input = torch.cat(
                [
                    identity_exp[:, lo:hi].expand(-1, -1, frames, -1),
                    self.relative_encoding(rel_c),
                ],
                dim=-1,
            )
            point_feature = self.point_mlp(point_input)
            point_feature = point_feature.masked_fill(
                ~visible[:, lo:hi].unsqueeze(-1), float("-inf")
            )
            flat_index_c = (
                t_index[:, lo:hi] * latent_h * latent_w
                + v_index[:, lo:hi] * latent_w
                + u_index[:, lo:hi]
            ).reshape(batch, -1)
            flat_feature_c = point_feature.reshape(batch, -1, self.out_dim)
            chunk_tokens = torch.full(
                (batch, token_count, self.out_dim),
                float("-inf"),
                device=device,
                dtype=dtype,
            ).scatter_reduce(
                1,
                flat_index_c.unsqueeze(-1).expand(-1, -1, self.out_dim),
                flat_feature_c,
                reduce="amax",
                include_self=True,
            )
            tokens = chunk_tokens if tokens is None else torch.maximum(tokens, chunk_tokens)
            occupancy.scatter_reduce_(
                1,
                flat_index_c,
                visible[:, lo:hi].reshape(batch, -1).float(),
                reduce="amax",
                include_self=True,
            )
        mask = occupancy > 0
        tokens = torch.where(
            mask.unsqueeze(-1),
            torch.nan_to_num(tokens, neginf=0.0, posinf=0.0),
            torch.zeros_like(tokens),
        )
        return tokens, mask

    def forward(
        self,
        identity_sequence: torch.Tensor,
        identity_valid: torch.Tensor,
        target_uv: torch.Tensor,
        target_vis: torch.Tensor,
        *,
        image_h: int,
        image_w: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        source_uv: Optional[torch.Tensor] = None,
        source_vis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        identity = self.encode_identities(identity_sequence, identity_valid)
        target_tokens, target_mask = self.rasterize(
            identity,
            identity_valid,
            target_uv,
            target_vis,
            image_h=image_h,
            image_w=image_w,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        source_tokens = None
        source_mask = None
        if source_uv is not None or source_vis is not None:
            if source_uv is None or source_vis is None:
                raise ValueError("source_uv and source_vis must be provided together")
            source_tokens, source_mask = self.rasterize(
                identity,
                identity_valid,
                source_uv,
                source_vis,
                image_h=image_h,
                image_w=image_w,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
            )
        return target_tokens, target_mask, source_tokens, source_mask
