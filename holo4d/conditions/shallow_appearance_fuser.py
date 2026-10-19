"""
shallow_appearance_fuser.py — Shallow learned fusion for warped RGB appearance.

Pre-block injection: fuses warped RGB latent (appearance) with the patchified
DiT input. Runs ONCE before the DiT block stack, not per-block.

Two fusion modes:
  - "add": x = x + proj(warped_rgb_latent)
  - "concat_proj": x = proj(cat(x, warped_rgb_latent, dim=-1))

Both are zero-initialized so the model starts with no appearance injection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class ShallowAppearanceFuser(nn.Module):
    """Fuse warped RGB appearance latent with DiT token stream.

    The warped_rgb_latent is at VAE latent resolution (e.g. 8x spatial downsample).
    The DiT token stream x is at patchified resolution (e.g. 16x = 8x VAE + 2x patch).
    A Conv3d patch embedding bridges this resolution gap, following the same pattern
    as ``control_patch_embedding`` in the existing DiT.

    Args:
        dit_dim: DiT hidden dimension (e.g. 1536 for 1.3B).
        latent_dim: VAE latent channel dim (e.g. 16 for Wan2.1).
        patch_size: spatial patch size (e.g. (1, 2, 2) matching DiT patch_embedding).
    """

    def __init__(self, dit_dim: int, latent_dim: int, patch_size: tuple = (1, 2, 2)):
        super().__init__()
        self.dit_dim = dit_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        # Conv3d patch embedding: VAE latent res → patchified res → dit_dim
        # Same architecture as control_patch_embedding in wan_video_dit_v2v.py
        self.patch_proj = nn.Conv3d(
            latent_dim, dit_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        # Zero-init: no effect at training start
        with torch.no_grad():
            nn.init.zeros_(self.patch_proj.weight)
            nn.init.zeros_(self.patch_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        warped_rgb_latent: torch.Tensor,
        confidence: torch.Tensor | None = None,
        v2v_concat_mode: str | None = None,
        num_domains: int = 1,
        grid_f: int = 0,
        grid_h: int = 0,
        grid_w: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, dit_dim) — patchified DiT input tokens
            warped_rgb_latent: (B, latent_dim, T_vae, H_vae, W_vae) — VAE-encoded warped RGB
            confidence: (B, 1, T_vae, H_vae, W_vae) — optional confidence mask [0, 1]
            v2v_concat_mode: "width", "frame", or None (single-domain)
            num_domains: number of V2V domains (2 = source+target, 3 = source+target+depth)
            grid_f, grid_h, grid_w: patchified spatial dims of x

        Returns:
            x: (B, seq_len, dit_dim) — fused output
        """
        # Cast to module dtype (bf16 during training, float32 during some eval paths)
        module_dtype = self.patch_proj.weight.dtype
        warped_rgb_latent = warped_rgb_latent.to(dtype=module_dtype)

        # Apply confidence gating on 5D latent BEFORE patch projection
        if confidence is not None:
            warped_rgb_latent = warped_rgb_latent * confidence.to(dtype=module_dtype)

        # Patch-project: (B, latent_dim, T_vae, H_vae, W_vae) → (B, dit_dim, f, h, w)
        # This handles the VAE→patchified resolution gap (e.g. 2x spatial downsample)
        warp_5d = self.patch_proj(warped_rgb_latent)  # (B, dit_dim, f_tgt, h_tgt, w_tgt)
        B = warp_5d.shape[0]
        _, _, f_tgt, h_tgt, w_tgt = warp_5d.shape

        # Handle V2V multi-domain layout.
        # Domain order: [source | target_rgb | target_depth]
        # Warped appearance goes into target_rgb slot (domain index 1).
        TARGET_RGB_DOMAIN_IDX = 1

        if v2v_concat_mode == "width" and num_domains >= 2:
            # Width-concat: build full-width 5D tensor, insert warped into correct slice
            w_orig = grid_w // num_domains
            if w_tgt != w_orig:
                raise ValueError(
                    f"ShallowAppearanceFuser: width-concat patch dim mismatch. "
                    f"patch_proj output w={w_tgt}, expected w_orig={w_orig} "
                    f"(grid_w={grid_w}, num_domains={num_domains})."
                )
            full_5d = torch.zeros(
                B, self.dit_dim, f_tgt, h_tgt, grid_w,
                device=warp_5d.device, dtype=warp_5d.dtype,
            )
            w_start = TARGET_RGB_DOMAIN_IDX * w_orig
            full_5d[:, :, :, :, w_start:w_start + w_orig] = warp_5d
            warp_tokens = rearrange(full_5d, "b c f h w -> b (f h w) c")
        elif v2v_concat_mode == "frame" and num_domains >= 2:
            # Frame-concat: domains stacked along T
            pad = torch.zeros_like(warp_5d)
            domain_slots = [pad if d != TARGET_RGB_DOMAIN_IDX else warp_5d for d in range(num_domains)]
            full_5d = torch.cat(domain_slots, dim=2)  # cat along T
            warp_tokens = rearrange(full_5d, "b c f h w -> b (f h w) c")
        elif v2v_concat_mode is None or num_domains <= 1:
            # Single domain
            warp_tokens = rearrange(warp_5d, "b c f h w -> b (f h w) c")
        else:
            raise ValueError(
                f"ShallowAppearanceFuser: unsupported layout. "
                f"v2v_concat_mode={v2v_concat_mode!r}, num_domains={num_domains}."
            )

        if warp_tokens.shape[1] != x.shape[1]:
            raise ValueError(
                f"ShallowAppearanceFuser: seq length mismatch after domain routing. "
                f"warp_tokens.seq={warp_tokens.shape[1]}, x.seq={x.shape[1]}. "
                f"grid=({grid_f},{grid_h},{grid_w}), warp_5d=({f_tgt},{h_tgt},{w_tgt}), "
                f"num_domains={num_domains}, v2v_concat_mode={v2v_concat_mode!r}."
            )

        return x + warp_tokens


class GeometryTrackFuser(nn.Module):
    """Learned concat+project fusion for geometry tokens + track tokens.

    Replaces the old additive fusion (dense + alpha * track) with a learned
    projection: cond = proj(cat(geometry, track, dim=-1)).

    This preserves branch identity longer and lets the model learn how much
    geometry vs track information to keep.

    Args:
        geom_dim: geometry token dimension (out_dim of DenseTargetControlEncoder).
        track_dim: track token dimension (same as geom_dim by convention).
        out_dim: output dimension (same as input dim by default).
    """

    def __init__(self, geom_dim: int, track_dim: int, out_dim: int | None = None):
        super().__init__()
        if out_dim is None:
            out_dim = geom_dim  # same as input by default
        self.proj = nn.Linear(geom_dim + track_dim, out_dim, bias=True)
        # Identity-like init: output ≈ geometry at start (first half gets identity, second half gets zero)
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            # Geometry passes through, track starts at zero contribution
            eye_size = min(out_dim, geom_dim)
            self.proj.weight[:eye_size, :eye_size].copy_(torch.eye(eye_size))

    def forward(
        self,
        geom_tokens: torch.Tensor,
        track_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            geom_tokens: (B, seq_len, geom_dim)
            track_tokens: (B, seq_len, track_dim)

        Returns:
            fused: (B, seq_len, out_dim)
        """
        return self.proj(torch.cat([geom_tokens, track_tokens], dim=-1))


def initialize_geometry_track_fuser(pipe) -> GeometryTrackFuser:
    """Create and attach a GeometryTrackFuser to the pipeline."""
    # Get token dim from existing dense encoder or track adapter
    enc = getattr(pipe, "geometry_only_dense_encoder", None) or getattr(pipe, "dense_target_control_encoder", None)
    adapter = getattr(pipe, "track_adapter", None)
    geom_dim = int(getattr(enc, "out_dim", 156)) if enc else 156
    track_dim = int(getattr(adapter, "out_dim", geom_dim)) if adapter else geom_dim

    fuser = GeometryTrackFuser(geom_dim=geom_dim, track_dim=track_dim, out_dim=geom_dim)
    ref_param = next(pipe.dit.parameters(), None)
    if ref_param is not None:
        fuser = fuser.to(device=ref_param.device, dtype=ref_param.dtype)
    # Attach to pipe ONLY (not dit). This module is called in the dense unit's
    # process() which runs outside DDP's forward pass. Attaching to dit would
    # cause DDP unused-parameter errors.
    pipe.geometry_track_fuser = fuser
    return fuser


def initialize_shallow_appearance_fuser(pipe) -> ShallowAppearanceFuser:
    """Create and attach a ShallowAppearanceFuser to the pipeline.

    Reads dit_dim, VAE latent channels, and patch_size from the pipeline
    to match the existing architecture automatically.
    """
    dit_dim = int(getattr(pipe.dit, "dim", 1536))
    vae_z = int(getattr(getattr(getattr(pipe, "vae", None), "model", None), "z_dim", 16))
    patch_size = tuple(getattr(pipe.dit, "patch_size", (1, 2, 2)))

    fuser = ShallowAppearanceFuser(
        dit_dim=dit_dim,
        latent_dim=vae_z,
        patch_size=patch_size,
    )

    # Move to same device/dtype as DiT
    ref_param = next(pipe.dit.parameters(), None)
    if ref_param is not None:
        fuser = fuser.to(device=ref_param.device, dtype=ref_param.dtype)

    # Attach to both pipe and dit so model_fn (which only sees dit) can access it
    pipe.warped_appearance_fuser = fuser
    pipe.dit.warped_appearance_fuser = fuser
    return fuser
