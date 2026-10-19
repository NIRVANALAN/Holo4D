"""Shared artifact writers for Track4D-360 qualitative sampling outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from safetensors.torch import save_file


# Wan2.1/Wan2.2 release configurations define ``sample_fps=16``.  Keep the
# prediction artifact defaults here so individual evaluation drivers do not
# silently drift to a dataset container's playback rate.
WAN_SAMPLE_FPS = 16
PREDICTION_MP4_QUALITY = 9


def save_mp4(
    out_path: str | Path,
    frames: list[np.ndarray] | np.ndarray,
    *,
    fps: int | float = WAN_SAMPLE_FPS,
    quality: int = PREDICTION_MP4_QUALITY,
    codec: str = "libx264",
    pixelformat: str = "yuv420p",
    macro_block_size: int = 1,
    output_params: list[str] | None = None,
) -> None:
    """Save RGB uint8 frames with the Vista4D-style imageio settings."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video = np.asarray(frames, dtype=np.uint8)
    if video.ndim != 4 or video.shape[-1] not in (3, 4):
        raise ValueError(f"expected video shape (T,H,W,3/4), got {tuple(video.shape)}")

    kwargs: dict[str, Any] = {
        "fps": fps,
        "codec": codec,
        "macro_block_size": int(macro_block_size),
    }
    if output_params:
        kwargs["output_params"] = output_params
    else:
        kwargs["quality"] = int(quality)
    if pixelformat:
        kwargs["pixelformat"] = pixelformat
    writer = imageio.get_writer(str(out_path), **kwargs)
    try:
        for i in range(video.shape[0]):
            writer.append_data(video[i, ..., :3])
    finally:
        writer.close()


def to_bf16_tensor(value: Any) -> torch.Tensor:
    """Convert numeric arrays to contiguous bf16 tensors for compact inspection."""
    return torch.as_tensor(np.asarray(value), dtype=torch.float32).to(torch.bfloat16).contiguous()


def to_bool_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value).astype(bool)).contiguous()


def save_safetensors_with_json(
    out_path: str | Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Atomically save a safetensors payload plus a neighboring JSON manifest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    st_metadata = None
    if metadata is not None:
        st_metadata = {str(k): str(v) for k, v in metadata.items()}
    save_file({k: v.contiguous() for k, v in tensors.items()}, tmp, metadata=st_metadata)
    tmp.replace(out_path)
    if metadata is not None:
        payload = dict(metadata)
        payload.setdefault("stored_tensors", sorted(tensors))
        out_path.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def save_pred_latents(
    out_path: str | Path,
    latents: torch.Tensor,
    metadata: dict[str, Any] | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Save generated target-view latents in the shared Track4D eval format."""
    meta = dict(metadata or {})
    meta.setdefault("format_version", "1")
    meta.setdefault("kind", "track4d360_pred_target_latents")
    latents_save = latents.detach().to(dtype).contiguous()
    meta["latent_shape"] = "x".join(str(s) for s in latents_save.shape)
    meta["latent_dtype"] = str(latents_save.dtype).replace("torch.", "")
    save_safetensors_with_json(
        out_path,
        {"latents": latents_save},
        metadata={str(k): str(v) for k, v in meta.items()},
    )
