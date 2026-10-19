"""Deterministic exact-count wrapper around the optional TorchCluster FPS op."""

from __future__ import annotations

from collections.abc import Callable

import torch


def resolve_torch_cluster_fps(fps_fn: Callable | None = None) -> Callable:
    """Resolve the compiled op without providing a hidden fallback."""
    if fps_fn is not None:
        return fps_fn
    try:
        from torch_cluster import fps
    except Exception as exc:
        raise RuntimeError(
            "torch_cluster import failed; use a build matching the active Torch/CUDA ABI"
        ) from exc
    return fps


def torch_cluster_fps_indices(
    points: torch.Tensor,
    identity_ids: torch.Tensor,
    count: int,
    *,
    fps_fn: Callable | None = None,
) -> tuple[torch.Tensor, int]:
    """Return original identity rows with the repository's deterministic seed.

    TorchCluster starts from local row zero when ``random_start=False``. This
    wrapper moves the farthest-from-centroid row to zero, runs the compiled op,
    then maps through both permutations back to original identity indices.
    """
    ids = torch.as_tensor(identity_ids, device=points.device, dtype=torch.long).reshape(-1)
    target = min(max(int(count), 0), int(ids.numel()))
    if target == 0:
        return ids.new_empty((0,)), 0
    if target == ids.numel():
        return ids.clone(), int(ids.numel())
    fps_fn = resolve_torch_cluster_fps(fps_fn)

    pool = points[ids].float()
    order = torch.arange(pool.shape[0], device=pool.device)
    first = torch.sum((pool - pool.mean(dim=0, keepdim=True)) ** 2, dim=1).argmax()
    first_value = order[first].clone()
    order[first] = order[0]
    order[0] = first_value
    ratio = float(target) / float(pool.shape[0])
    reordered = fps_fn(pool[order], ratio=ratio, random_start=False)
    raw_count = int(reordered.numel())
    if raw_count < target:
        raise RuntimeError(
            f"torch_cluster.fps returned {raw_count} rows for requested {target}; "
            "an exact-count completion policy is required"
        )
    local = order[reordered[:target]]
    selected = ids[local]
    if int(torch.unique(selected).numel()) != target:
        raise RuntimeError("torch_cluster.fps returned duplicate identity rows")
    return selected, raw_count
