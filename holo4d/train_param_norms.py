"""Cheap distributed parameter norm helpers for Track4D-360 training logs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def _to_local_tensor(value: Any) -> torch.Tensor | None:
    tensor = value.detach() if hasattr(value, "detach") else value
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    tensor = tensor.detach() if hasattr(tensor, "detach") else tensor
    return tensor if torch.is_tensor(tensor) else None


def _is_sharded_dtensor(value: Any) -> bool:
    placements = getattr(value, "placements", None)
    if placements is None and hasattr(value, "detach"):
        placements = getattr(value.detach(), "placements", None)
    if placements is None:
        return False
    return any(type(placement).__name__.endswith("Shard") for placement in placements)


def _dist_is_ready(distributed: Any) -> bool:
    try:
        return bool(distributed.is_available() and distributed.is_initialized())
    except Exception:
        return False


def compute_l2_norms(
    entries: Mapping[str, Iterable[Any]],
    *,
    distributed: Any | None = None,
) -> dict[str, float]:
    """Compute L2 norms for named tensor groups.

    Normal tensors stay local for backward-compatible metrics. DTensor/FSDP2
    sharded tensors are handled by summing local shard squared norms and using
    one batched scalar all-reduce for all sharded entries in this call.
    """

    if distributed is None:
        import torch.distributed as distributed

    names: list[str] = []
    sums: list[torch.Tensor] = []
    reduce_flags: list[bool] = []

    for name, values in entries.items():
        local_sum = None
        needs_reduce = False
        for value in values:
            if value is None:
                continue
            needs_reduce = needs_reduce or _is_sharded_dtensor(value)
            local_tensor = _to_local_tensor(value)
            if local_tensor is None:
                continue
            part_sum = local_tensor.float().square().sum()
            local_sum = part_sum if local_sum is None else local_sum.to(part_sum.device) + part_sum

        if local_sum is None:
            local_sum = torch.tensor(0.0)
        names.append(name)
        sums.append(local_sum)
        reduce_flags.append(needs_reduce)

    if any(reduce_flags) and _dist_is_ready(distributed):
        reduce_indices = [i for i, flag in enumerate(reduce_flags) if flag]
        reduce_vec = torch.stack([sums[i].float() for i in reduce_indices])
        distributed.all_reduce(reduce_vec, op=distributed.ReduceOp.SUM)
        for pos, idx in enumerate(reduce_indices):
            sums[idx] = reduce_vec[pos]

    return {
        name: float(torch.clamp(sums[i], min=0.0).sqrt().item())
        for i, name in enumerate(names)
    }
