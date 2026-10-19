"""Layout helpers for Vista4D-style latent streams."""

from __future__ import annotations

import torch


def expand_vista4d_streams_for_v2v(
    source_stream: torch.Tensor,
    pointcloud_stream: torch.Tensor,
    *,
    x_shape: tuple[int, ...],
    v2v_concat_mode: str,
    num_domains: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place source and point-cloud streams into the V2V source/target grid."""
    if v2v_concat_mode not in ("frame", "width"):
        return source_stream, pointcloud_stream
    if int(num_domains) < 2:
        return source_stream, pointcloud_stream

    if len(x_shape) != 5:
        raise ValueError(f"Expected 5D token grid shape for x, got {x_shape}")
    _, _, f, h, w = x_shape

    def _expand_one(stream: torch.Tensor, *, name: str, domain_idx: int) -> torch.Tensor:
        if stream.shape[2:] == tuple(x_shape[2:]):
            return stream
        if v2v_concat_mode == "frame":
            if int(stream.shape[3]) != int(h) or int(stream.shape[4]) != int(w):
                raise ValueError(
                    f"Vista4D {name} stream grid mismatch for frame concat: "
                    f"stream={tuple(stream.shape)}, x={x_shape}"
                )
            if int(stream.shape[2]) * int(num_domains) != int(f):
                raise ValueError(
                    f"Vista4D {name} stream frame mismatch: stream={tuple(stream.shape)}, "
                    f"x={x_shape}, num_domains={int(num_domains)}"
                )
            parts = [torch.zeros_like(stream) for _ in range(int(num_domains))]
            parts[int(domain_idx)] = stream
            return torch.cat(parts, dim=2)

        if int(stream.shape[2]) != int(f) or int(stream.shape[3]) != int(h):
            raise ValueError(
                f"Vista4D {name} stream grid mismatch for width concat: "
                f"stream={tuple(stream.shape)}, x={x_shape}"
            )
        if int(stream.shape[4]) * int(num_domains) != int(w):
            raise ValueError(
                f"Vista4D {name} stream width mismatch: stream={tuple(stream.shape)}, "
                f"x={x_shape}, num_domains={int(num_domains)}"
            )
        parts = [torch.zeros_like(stream) for _ in range(int(num_domains))]
        parts[int(domain_idx)] = stream
        return torch.cat(parts, dim=4)

    # source_stream may be None in pointcloud-only mode (the clean source video is
    # already carried by x's source domain, so the Vista4D source stream is dropped).
    source_stream = (
        _expand_one(source_stream, name="source", domain_idx=0)
        if source_stream is not None
        else None
    )
    pointcloud_stream = _expand_one(pointcloud_stream, name="pointcloud", domain_idx=1)
    return source_stream, pointcloud_stream


__all__ = ["expand_vista4d_streams_for_v2v"]
