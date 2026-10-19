"""Shared temporal-query selection for Track4D NVS datasets."""

from __future__ import annotations

import numpy as np


def choose_track_query_indices(
    num_frames: int,
    num_queries: int,
    available_indices=None,
) -> list[int]:
    """Choose deterministic canonical anchors and optionally snap to availability."""
    num_frames = max(int(num_frames), 1)
    num_queries = max(int(num_queries), 1)
    if num_queries >= num_frames:
        desired = list(range(num_frames))
    elif num_queries == 1:
        desired = [0]
    elif num_queries == 2:
        desired = [0, num_frames - 1]
    elif num_queries == 3:
        desired = [0, (num_frames - 1) // 2, num_frames - 1]
    elif num_queries == 4:
        desired = [0, (num_frames - 1) // 4, (num_frames - 1) // 2, num_frames - 1]
    else:
        desired = [int(round(value)) for value in np.linspace(0, num_frames - 1, num_queries)]

    if available_indices is None:
        selected = []
        for query_idx in desired:
            query_idx = max(0, min(int(query_idx), num_frames - 1))
            if query_idx not in selected:
                selected.append(query_idx)
        return selected

    available = sorted(
        {int(query_idx) for query_idx in available_indices if 0 <= int(query_idx) < num_frames}
    )
    if not available:
        return [0]
    selected = []
    for query_idx in desired:
        nearest = min(available, key=lambda candidate: (abs(candidate - query_idx), candidate))
        if nearest not in selected:
            selected.append(nearest)
    return selected


def choose_canonical_random_query_indices(
    num_frames: int,
    num_queries: int,
    rng,
    available_indices=None,
) -> list[int]:
    """Sample an ordered subset of the canonical first/middle/last anchors."""
    pool = choose_track_query_indices(
        num_frames,
        min(3, max(int(num_frames), 1)),
        available_indices,
    )
    if not pool:
        return [0]
    count = min(max(int(num_queries), 1), len(pool))
    # choice(replace=False) already randomizes the returned order.
    chosen = rng.choice(np.asarray(pool, dtype=np.int64), size=count, replace=False)
    return [int(query_idx) for query_idx in np.asarray(chosen).reshape(-1).tolist()]
