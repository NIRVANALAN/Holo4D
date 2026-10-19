"""Causal-Forcing stage-1 helpers for Wan flow-matching experiments."""

from __future__ import annotations

import math

import torch


def build_teacher_forcing_causal_mask(
    *,
    num_latent_frames: int,
    frame_seq_length: int,
    num_frame_per_block: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the dense clean+noisy teacher-forcing mask used by Causal-Forcing.

    The token order is ``[clean tokens][noisy tokens]``. Clean tokens attend to
    clean tokens up to the end of their causal block. Noisy tokens attend to
    clean tokens from previous blocks plus noisy tokens in their own block.

    This dense helper is intended for low-resolution smoke tests. 480p/81f
    training should use a flex/block-mask implementation instead.
    """

    num_latent_frames = int(num_latent_frames)
    frame_seq_length = int(frame_seq_length)
    num_frame_per_block = int(num_frame_per_block)
    if num_latent_frames <= 0:
        raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}")
    if frame_seq_length <= 0:
        raise ValueError(f"frame_seq_length must be positive, got {frame_seq_length}")
    if num_frame_per_block <= 0:
        raise ValueError(f"num_frame_per_block must be positive, got {num_frame_per_block}")

    clean_tokens = num_latent_frames * frame_seq_length
    total_tokens = clean_tokens * 2
    attention_block_size = frame_seq_length * num_frame_per_block
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)

    num_blocks = int(math.ceil(clean_tokens / attention_block_size))
    for block_idx in range(num_blocks):
        clean_start = block_idx * attention_block_size
        clean_end = min((block_idx + 1) * attention_block_size, clean_tokens)
        noisy_start = clean_tokens + clean_start
        noisy_end = clean_tokens + clean_end

        # Clean context tokens are themselves causal by block.
        mask[clean_start:clean_end, :clean_end] = True
        # Noisy tokens see previous clean context blocks and their own noisy block.
        mask[noisy_start:noisy_end, :clean_start] = True
        mask[noisy_start:noisy_end, noisy_start:noisy_end] = True

    eye = torch.eye(total_tokens, dtype=torch.bool, device=device)
    return mask | eye

