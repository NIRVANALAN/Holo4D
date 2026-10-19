"""Rolling-forcing helpers for Track4D-360 AR experiments.

The helpers here are intentionally small and tensor-shape focused. They do not
replace the Track4D condition architecture; they provide the cache and slicing
primitives used by the existing Wan V2V pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ARWindow:
    """A latent-frame window in a full sequence."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_denoising_step_list(value: str | Iterable[int]) -> list[int]:
    """Parse ``1000,800,...`` into a non-empty integer list."""

    if isinstance(value, str):
        steps = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        steps = [int(v) for v in value]
    if not steps:
        raise ValueError("AR denoising step list must contain at least one step.")
    return steps


def build_latent_windows(num_latent_frames: int, frames_per_block: int) -> list[ARWindow]:
    """Split a latent sequence into consecutive AR windows."""

    num_latent_frames = int(num_latent_frames)
    frames_per_block = int(frames_per_block)
    if num_latent_frames <= 0:
        raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}")
    if frames_per_block <= 0:
        raise ValueError(f"frames_per_block must be positive, got {frames_per_block}")
    return [
        ARWindow(start, min(start + frames_per_block, num_latent_frames))
        for start in range(0, num_latent_frames, frames_per_block)
    ]


def slice_latent_frames(tensor: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
    """Slice a ``(B,C,F,H,W)`` latent/video tensor on latent frame dim."""

    if tensor is None:
        return None
    if not torch.is_tensor(tensor) or tensor.ndim != 5:
        return tensor
    return tensor[:, :, int(start):int(end)]


def slice_tokens_by_latent_frames(
    tokens: torch.Tensor | None,
    *,
    start: int,
    end: int,
    frame_seq_length: int,
) -> torch.Tensor | None:
    """Slice flattened per-target-frame tokens by latent-frame offset."""

    if tokens is None:
        return None
    if not torch.is_tensor(tokens):
        return tokens
    if tokens.ndim >= 4:
        token_end = int(end)
        if token_end > int(tokens.shape[1]):
            raise ValueError(
                f"Frame-major token slice [{int(start)}:{token_end}] exceeds "
                f"frame length {int(tokens.shape[1])}"
            )
        return tokens[:, int(start):token_end, ...]
    if tokens.ndim != 3:
        return tokens
    frame_seq_length = int(frame_seq_length)
    if frame_seq_length <= 0:
        raise ValueError(f"frame_seq_length must be positive, got {frame_seq_length}")
    token_start = int(start) * frame_seq_length
    token_end = int(end) * frame_seq_length
    if token_end > int(tokens.shape[1]):
        raise ValueError(
            f"Token slice [{token_start}:{token_end}] exceeds token length "
            f"{int(tokens.shape[1])}; frame_seq_length={frame_seq_length}"
        )
    return tokens[:, token_start:token_end]


def allocate_kv_cache(num_layers: int) -> list[dict[str, torch.Tensor]]:
    """Create an empty per-layer KV cache list."""

    return [{} for _ in range(int(num_layers))]


def reset_kv_cache(kv_cache: list[dict[str, torch.Tensor]] | None) -> None:
    """Clear a cache in-place."""

    if kv_cache is None:
        return
    for entry in kv_cache:
        entry.clear()


def trim_kv_cache_tokens(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    frame_seq_length: int,
    max_latent_frames: int | None,
    keep_first_latent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim KV tensors, optionally preserving latent frame 0 as a sink."""

    if max_latent_frames is None:
        return k, v
    frame_seq_length = int(frame_seq_length)
    if frame_seq_length <= 0:
        raise ValueError(f"frame_seq_length must be positive, got {frame_seq_length}")
    keep_tokens = int(max_latent_frames) * frame_seq_length
    if keep_tokens <= 0 or int(k.shape[1]) <= keep_tokens:
        return k, v
    if not bool(keep_first_latent):
        return k[:, -keep_tokens:], v[:, -keep_tokens:]

    sink_tokens = frame_seq_length
    if int(k.shape[1]) <= sink_tokens + keep_tokens:
        return k, v
    return (
        torch.cat([k[:, :sink_tokens], k[:, -keep_tokens:]], dim=1),
        torch.cat([v[:, :sink_tokens], v[:, -keep_tokens:]], dim=1),
    )


def resolve_ar_timesteps(
    scheduler,
    denoising_steps: str | Iterable[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve nominal Wan step ids to the scheduler's nearest timesteps."""

    requested = parse_denoising_step_list(denoising_steps)
    sched_steps = scheduler.timesteps.detach().to(device="cpu", dtype=torch.float32)
    resolved = []
    for step in requested:
        idx = torch.argmin((sched_steps - float(step)).abs())
        resolved.append(scheduler.timesteps[idx])
    return torch.stack(resolved).to(device=device, dtype=dtype)


def select_window_timestep(
    timesteps: torch.Tensor,
    window_index: int,
    *,
    same_step_across_blocks: bool,
) -> torch.Tensor:
    """Pick the timestep tensor for one AR window."""

    if timesteps.ndim != 1:
        timesteps = timesteps.reshape(-1)
    idx = 0 if bool(same_step_across_blocks) else int(window_index) % int(timesteps.numel())
    return timesteps[idx].reshape(1)


def temporal_rope_slice(freqs: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """Return a temporal RoPE slice with an absolute latent-frame offset."""

    start = int(start)
    length = int(length)
    if start < 0:
        raise ValueError(f"RoPE start must be non-negative, got {start}")
    if length <= 0:
        raise ValueError(f"RoPE length must be positive, got {length}")
    end = start + length
    if end > int(freqs.shape[0]):
        raise ValueError(
            f"RoPE temporal slice [{start}:{end}] exceeds precomputed length {int(freqs.shape[0])}"
        )
    return freqs[start:end]
