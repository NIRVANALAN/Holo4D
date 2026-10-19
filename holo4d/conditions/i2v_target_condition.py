"""Small, testable helpers for Vista4D-style target-image conditioning.

The production pipeline unit lives in ``wan_video_track4d360.py`` because it
needs the VideoX-Fun pipeline/VAEs.  Tensor layout and augmentation logic lives
here so it can be tested without importing diffusers, transformers, or CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import torch


T = TypeVar("T")


def validate_i2v_condition_config(drop_prob: float, noise_level: float) -> None:
    """Validate the two Vista4D I2V augmentation strengths."""
    if not 0.0 <= float(drop_prob) <= 1.0:
        raise ValueError(f"i2v_condition_drop_prob must be in [0, 1], got {drop_prob}")
    if not 0.0 <= float(noise_level) <= 1.0:
        raise ValueError(f"i2v_condition_noise_level must be in [0, 1], got {noise_level}")


def augment_i2v_image_latents(
    image_latents: torch.Tensor,
    noise_level: float,
    *,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Vista4D's ``(1-alpha) * x + alpha * eps`` image-latent noise."""
    alpha = float(noise_level)
    if alpha == 0.0:
        return image_latents
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")
    if noise is None:
        noise = torch.randn_like(image_latents)
    if noise.shape != image_latents.shape:
        raise ValueError(
            f"noise must match image_latents, got {tuple(noise.shape)} vs "
            f"{tuple(image_latents.shape)}"
        )
    return (1.0 - alpha) * image_latents + alpha * noise.to(
        device=image_latents.device, dtype=image_latents.dtype
    )


def build_source_video_condition(
    source_latents: torch.Tensor,
    *,
    condition_channels: int = 20,
) -> torch.Tensor:
    """Build Wan-I2V's mask+latent condition for a fully observed source video.

    Official Wan-I2V appends four mask channels to the 16 VAE channels.  In
    Track4D V2V, the complete current source chunk is observed, so all four
    source mask channels are one and the source VAE latents stay nonzero for
    every latent frame.  This is distinct from the target-domain I2V condition,
    whose mask/latents only expose the carried boundary frame.
    """
    if source_latents.ndim != 5:
        raise ValueError("source_latents must be a BCTHW tensor")
    mask_channels = int(condition_channels) - int(source_latents.shape[1])
    if mask_channels <= 0:
        raise ValueError(
            "condition_channels must exceed source latent channels; "
            f"got {condition_channels} and {int(source_latents.shape[1])}"
        )
    source_mask = torch.ones(
        int(source_latents.shape[0]),
        mask_channels,
        int(source_latents.shape[2]),
        int(source_latents.shape[3]),
        int(source_latents.shape[4]),
        device=source_latents.device,
        dtype=source_latents.dtype,
    )
    return torch.cat([source_mask, source_latents], dim=1)


def expand_i2v_condition_to_v2v_domains(
    target_condition: torch.Tensor,
    full_latents: torch.Tensor,
    *,
    source_condition: torch.Tensor | None = None,
    num_domains: int = 2,
    source_domain: int = 0,
    target_domain: int = 1,
    concat_mode: str = "width",
) -> torch.Tensor:
    """Compose source-V2V and target-boundary I2V conditions by width domain.

    Track4D width-concatenates ``[source | target | ...]`` in space while Wan
    I2V's 20-channel tensor covers one video domain.  The source region carries
    the *current 49-frame source video* as a fully observed V2V condition; the
    target RGB region carries only the previous target chunk's last frame.
    Additional output domains remain zero.
    """
    if concat_mode != "width":
        raise NotImplementedError(
            "Vista4D target-image conditioning currently supports width-concat only"
        )
    if target_condition.ndim != 5 or full_latents.ndim != 5:
        raise ValueError("target_condition and full_latents must both be BCTHW tensors")
    if int(num_domains) < 2:
        raise ValueError(f"num_domains must be >= 2, got {num_domains}")
    if not 0 <= int(source_domain) < int(num_domains):
        raise ValueError(
            f"source_domain must be in [0, {int(num_domains) - 1}], got {source_domain}"
        )
    if not 0 <= int(target_domain) < int(num_domains):
        raise ValueError(
            f"target_domain must be in [0, {int(num_domains) - 1}], got {target_domain}"
        )
    if int(source_domain) == int(target_domain):
        raise ValueError("source_domain and target_domain must differ")
    if int(full_latents.shape[-1]) % int(num_domains) != 0:
        raise ValueError(
            f"full latent width {int(full_latents.shape[-1])} is not divisible by "
            f"num_domains={num_domains}"
        )
    domain_width = int(full_latents.shape[-1]) // int(num_domains)
    expected = (
        int(full_latents.shape[0]),
        int(target_condition.shape[1]),
        int(full_latents.shape[2]),
        int(full_latents.shape[3]),
        domain_width,
    )
    if tuple(target_condition.shape) != expected:
        raise ValueError(
            "target_condition must have single-domain shape "
            f"{expected}, got {tuple(target_condition.shape)}"
        )
    if source_condition is not None and tuple(source_condition.shape) != expected:
        raise ValueError(
            "source_condition must have single-domain shape "
            f"{expected}, got {tuple(source_condition.shape)}"
        )
    output = target_condition.new_zeros(
        int(full_latents.shape[0]),
        int(target_condition.shape[1]),
        int(full_latents.shape[2]),
        int(full_latents.shape[3]),
        int(full_latents.shape[4]),
    )
    if source_condition is not None:
        source_start = int(source_domain) * domain_width
        output[..., source_start : source_start + domain_width] = source_condition
    target_start = int(target_domain) * domain_width
    output[..., target_start : target_start + domain_width] = target_condition
    return output


def expand_i2v_condition_to_target_domain(
    condition: torch.Tensor,
    full_latents: torch.Tensor,
    *,
    num_domains: int = 2,
    target_domain: int = 1,
    concat_mode: str = "width",
) -> torch.Tensor:
    """Backward-compatible target-only wrapper for older callers/tests."""
    return expand_i2v_condition_to_v2v_domains(
        condition,
        full_latents,
        source_condition=None,
        num_domains=num_domains,
        target_domain=target_domain,
        concat_mode=concat_mode,
    )


@dataclass(frozen=True)
class LongHorizonChunk:
    """A fixed-size AR chunk and the number of newly emitted frames."""

    start: int
    indices: tuple[int, ...]
    valid_frames: int
    append_from: int

    def source_index(self, local_frame: int) -> int:
        """Map a local 49-frame index to its source-sequence frame index."""
        local_frame = int(local_frame)
        if not 0 <= local_frame < len(self.indices):
            raise IndexError(
                f"local frame must be in [0, {len(self.indices) - 1}], got {local_frame}"
            )
        return int(self.indices[local_frame])


@dataclass(frozen=True)
class ChunkTrackingQueryPlan:
    """Per-window tracking labels in local and original source coordinates."""

    local_indices: tuple[int, ...]
    source_indices: tuple[int, ...]


def plan_chunk_tracking_queries(
    chunk: LongHorizonChunk,
    *,
    num_queries: int = 2,
) -> ChunkTrackingQueryPlan:
    """Plan fresh tracking queries for one source chunk.

    Query labels are always local to the new 49-frame window.  For q2 this is
    local ``(0, 48)`` on every full chunk, mapping to source ``(0, 48)``, then
    ``(48, 96)``, and so on.  A padded tail maps its final local query to the
    repeated final real source frame.
    """
    total = len(chunk.indices)
    count = max(1, int(num_queries))
    if count >= total:
        local = tuple(range(total))
    elif count == 1:
        local = (0,)
    else:
        local = tuple(
            dict.fromkeys(int(round(i * (total - 1) / (count - 1))) for i in range(count))
        )
    source = tuple(chunk.source_index(i) for i in local)
    return ChunkTrackingQueryPlan(local_indices=local, source_indices=source)


def select_long_horizon_chunk(
    sequence: Sequence[T] | torch.Tensor,
    chunk: LongHorizonChunk,
    *,
    frame_dim: int = 0,
):
    """Select the current source/camera/track payload using chunk indices.

    Use ``frame_dim=0`` for frame lists, cameras, depth, and RGB tensors, and
    ``frame_dim=-2`` for ``tracks_world`` shaped ``(N,T,3)``.  Repeated indices
    pad the final short window using the same policy as generation/stitching.
    """
    if isinstance(sequence, torch.Tensor):
        dim = int(frame_dim) % sequence.ndim
        index = torch.tensor(chunk.indices, device=sequence.device, dtype=torch.long)
        return torch.index_select(sequence, dim, index)
    if int(frame_dim) != 0:
        raise ValueError("non-tensor sequences only support frame_dim=0")
    selected = [sequence[i] for i in chunk.indices]
    if isinstance(sequence, tuple):
        return tuple(selected)
    return selected


def plan_long_horizon_chunks(
    total_frames: int,
    *,
    chunk_frames: int = 49,
) -> list[LongHorizonChunk]:
    """Plan overlapping chunks where each continuation reuses one boundary frame.

    Wan requires ``4k+1`` frames.  Short final chunks are padded by repeating the
    final source index; callers keep only ``valid_frames`` outputs.  Chunk 0 emits
    all frames; later chunks discard output frame 0 because it is the previous
    chunk's carried condition.
    """
    total_frames = int(total_frames)
    chunk_frames = int(chunk_frames)
    if total_frames <= 0:
        return []
    if chunk_frames < 2 or chunk_frames % 4 != 1:
        raise ValueError(f"chunk_frames must be >= 2 and satisfy 4k+1, got {chunk_frames}")

    chunks: list[LongHorizonChunk] = []
    start = 0
    while start < total_frames:
        remaining = total_frames - start
        valid = min(chunk_frames, remaining)
        indices = list(range(start, start + valid))
        indices.extend([indices[-1]] * (chunk_frames - valid))
        append_from = 0 if not chunks else 1
        chunks.append(
            LongHorizonChunk(
                start=start,
                indices=tuple(indices),
                valid_frames=valid,
                append_from=append_from,
            )
        )
        if start + valid >= total_frames:
            break
        start += chunk_frames - 1
    return chunks


def stitch_long_horizon_chunks(
    generated_chunks: Sequence[Sequence[T]],
    plan: Sequence[LongHorizonChunk],
) -> list[T]:
    """Stitch generated chunks while removing overlap and padded tail frames."""
    if len(generated_chunks) != len(plan):
        raise ValueError(
            f"generated chunk count {len(generated_chunks)} does not match plan {len(plan)}"
        )
    output: list[T] = []
    for frames, window in zip(generated_chunks, plan):
        if len(frames) < window.valid_frames:
            raise ValueError(
                f"chunk at start={window.start} returned {len(frames)} frames, "
                f"needs at least {window.valid_frames}"
            )
        output.extend(frames[window.append_from : window.valid_frames])
    return output


def generate_long_horizon_chunks(
    plan: Sequence[LongHorizonChunk],
    *,
    generate_first: Callable[[LongHorizonChunk], Sequence[T]],
    generate_continuation: Callable[[LongHorizonChunk, T], Sequence[T]],
) -> tuple[list[T], list[list[T]]]:
    """Execute the Vista4D T2V/V2V-first, I2V-continuation protocol.

    The callbacks own dataset slicing and pipeline arguments.  This helper owns
    the easy-to-get-wrong autoregressive state transition: continuation chunk
    ``k`` receives the last *generated target frame* from chunk ``k-1``.
    """
    generated: list[list[T]] = []
    previous_target_frame: T | None = None
    for chunk_id, window in enumerate(plan):
        if chunk_id == 0:
            frames = list(generate_first(window))
        else:
            if previous_target_frame is None:
                raise RuntimeError("missing previous generated target frame")
            frames = list(generate_continuation(window, previous_target_frame))
        if len(frames) < window.valid_frames:
            raise ValueError(
                f"chunk {chunk_id} returned {len(frames)} frames, needs at least "
                f"{window.valid_frames}"
            )
        previous_target_frame = frames[window.valid_frames - 1]
        generated.append(frames)
    return stitch_long_horizon_chunks(generated, plan), generated
