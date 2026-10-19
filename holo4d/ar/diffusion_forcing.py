"""Diffusion Forcing helpers adapted to Wan rectified-flow timesteps.

This module intentionally ports the non-DMD part only: independent per-frame
noise levels for training and an autoregressive/pyramid schedule for sampling.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_timestep_matrix(timesteps: torch.Tensor, *, batch_size: int, num_frames: int) -> torch.Tensor:
    """Normalize scalar/vector timestep input to ``(B, F)``."""

    if timesteps.ndim == 0:
        return timesteps.reshape(1, 1).expand(batch_size, num_frames)
    if timesteps.ndim == 1:
        if timesteps.numel() == 1:
            return timesteps.reshape(1, 1).expand(batch_size, num_frames)
        if timesteps.numel() == num_frames:
            return timesteps.reshape(1, num_frames).expand(batch_size, num_frames)
        if timesteps.numel() == batch_size:
            return timesteps.reshape(batch_size, 1).expand(batch_size, num_frames)
    if timesteps.ndim == 2:
        if timesteps.shape == (batch_size, num_frames):
            return timesteps
        if timesteps.shape[0] == 1 and timesteps.shape[1] == num_frames:
            return timesteps.expand(batch_size, num_frames)
    raise ValueError(
        f"Cannot interpret timestep shape {tuple(timesteps.shape)} as (B,F)=({batch_size},{num_frames})."
    )


def sample_random_independent_timesteps(
    scheduler,
    *,
    batch_size: int,
    num_latent_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample independent Wan scheduler timesteps for each latent frame."""

    sched_steps = scheduler.timesteps.to(device=device, dtype=dtype)
    indices = torch.randint(
        low=0,
        high=int(sched_steps.numel()),
        size=(int(batch_size), int(num_latent_frames)),
        device=device,
        generator=generator,
    )
    return sched_steps[indices]


def parse_fixed_timestep_values(value: str, *, num_latent_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Parse a comma-separated timestep row for deterministic local tests."""

    vals = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise ValueError("Fixed diffusion-forcing timestep list is empty.")
    if len(vals) == 1:
        vals = vals * int(num_latent_frames)
    if len(vals) != int(num_latent_frames):
        raise ValueError(
            f"Expected {int(num_latent_frames)} fixed timestep values, got {len(vals)}."
        )
    return torch.tensor(vals, device=device, dtype=dtype).reshape(1, int(num_latent_frames))


def expand_latent_timesteps_to_token_timesteps(
    latent_timesteps: torch.Tensor,
    *,
    frame_seq_length: int,
) -> torch.Tensor:
    """Repeat each latent-frame timestep for all spatial patch tokens."""

    if latent_timesteps.ndim != 2:
        raise ValueError(f"latent_timesteps must be (B,F), got {tuple(latent_timesteps.shape)}")
    frame_seq_length = int(frame_seq_length)
    if frame_seq_length <= 0:
        raise ValueError(f"frame_seq_length must be positive, got {frame_seq_length}")
    return latent_timesteps.repeat_interleave(frame_seq_length, dim=1)


def flow_sigma_from_timesteps(timesteps: torch.Tensor, *, num_train_timesteps: int = 1000) -> torch.Tensor:
    """Wan rectified-flow sigma, using the local convention ``sigma=t/1000``."""

    return timesteps.float() / float(num_train_timesteps)


def flow_add_noise_per_frame(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Apply Wan flow noising with a separate timestep per latent frame."""

    if clean.shape != noise.shape:
        raise ValueError(f"clean and noise must have identical shapes, got {tuple(clean.shape)} vs {tuple(noise.shape)}")
    if clean.ndim != 5:
        raise ValueError(f"clean must be (B,C,F,H,W), got {tuple(clean.shape)}")
    bsz, _, frames, _, _ = clean.shape
    timestep_matrix = _as_timestep_matrix(timesteps.to(device=clean.device), batch_size=bsz, num_frames=frames)
    sigma = flow_sigma_from_timesteps(timestep_matrix, num_train_timesteps=num_train_timesteps)
    sigma = sigma.to(device=clean.device, dtype=clean.dtype).reshape(bsz, 1, frames, 1, 1)
    return clean + sigma * noise


def flow_training_target_per_frame(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Wan flow-matching target used by the repo scheduler: ``noise - clean``."""

    return noise - clean


def flow_step_per_frame(
    x: torch.Tensor,
    velocity: torch.Tensor,
    from_timesteps: torch.Tensor,
    to_timesteps: torch.Tensor,
    *,
    num_train_timesteps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One deterministic rectified-flow step with per-frame timesteps."""

    if x.shape != velocity.shape:
        raise ValueError(f"x and velocity must have identical shapes, got {tuple(x.shape)} vs {tuple(velocity.shape)}")
    bsz, _, frames, _, _ = x.shape
    from_matrix = _as_timestep_matrix(from_timesteps.to(device=x.device), batch_size=bsz, num_frames=frames)
    to_matrix = _as_timestep_matrix(to_timesteps.to(device=x.device), batch_size=bsz, num_frames=frames)
    sigma = flow_sigma_from_timesteps(from_matrix, num_train_timesteps=num_train_timesteps).to(device=x.device, dtype=x.dtype)
    next_sigma = flow_sigma_from_timesteps(to_matrix, num_train_timesteps=num_train_timesteps).to(device=x.device, dtype=x.dtype)
    sigma = sigma.reshape(bsz, 1, frames, 1, 1)
    next_sigma = next_sigma.reshape(bsz, 1, frames, 1, 1)
    x0 = x - sigma * velocity
    x_next = x + (next_sigma - sigma) * velocity
    return x_next, x0


def build_diffusion_forcing_scheduling_matrix(
    *,
    horizon: int,
    sampling_timesteps: int,
    num_train_timesteps: int = 1000,
    mode: str = "autoregressive",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a DF-style scheduling matrix in Wan timestep values.

    Rows are denoising states; columns are latent frames. ``0`` is clean and
    ``num_train_timesteps`` is pure noise under the Wan flow convention.
    """

    horizon = int(horizon)
    sampling_timesteps = int(sampling_timesteps)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if sampling_timesteps <= 0:
        raise ValueError(f"sampling_timesteps must be positive, got {sampling_timesteps}")
    if mode == "full_sequence":
        idx = np.arange(sampling_timesteps, -1, -1, dtype=np.float32)[:, None].repeat(horizon, axis=1)
    elif mode == "autoregressive":
        height = sampling_timesteps + horizon
        idx = np.zeros((height, horizon), dtype=np.float32)
        for m in range(height):
            for t in range(horizon):
                idx[m, t] = sampling_timesteps + t - m
        idx = np.clip(idx, 0, sampling_timesteps)
    else:
        raise ValueError(f"Unknown diffusion-forcing scheduling mode {mode!r}.")
    values = idx / float(sampling_timesteps) * float(num_train_timesteps)
    return torch.tensor(values, device=device, dtype=dtype)
