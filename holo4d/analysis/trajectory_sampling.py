"""Target-independent sampling helpers for complete 3D trajectories."""

from __future__ import annotations

import numpy as np


def trajectory_descriptors(
    tracks_world: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    temporal_samples: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Build compact position-and-motion descriptors for full trajectories.

    Missing sampled positions are filled with that identity's robust temporal
    center. The returned eligibility mask still records which identities have
    enough real support to be sampled.
    """
    tracks = np.asarray(tracks_world, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"tracks_world must be (N,T,3), got {tracks.shape}")
    if valid_mask is None:
        valid = np.isfinite(tracks).all(axis=-1) & (np.linalg.norm(tracks, axis=-1) > 1e-8)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != tracks.shape[:2]:
            raise ValueError(f"valid_mask must be {tracks.shape[:2]}, got {valid.shape}")
        valid &= np.isfinite(tracks).all(axis=-1)
    eligible = valid.sum(axis=1) >= 2
    safe = np.where(valid[..., None], tracks, np.nan)
    with np.errstate(all="ignore"):
        center = np.nanmedian(safe, axis=1)
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    frame_ids = np.linspace(
        0, tracks.shape[1] - 1, max(2, int(temporal_samples))
    ).round().astype(np.int64)
    sampled = tracks[:, frame_ids].copy()
    sampled_valid = valid[:, frame_ids]
    sampled[~sampled_valid] = np.broadcast_to(center[:, None], sampled.shape)[~sampled_valid]
    velocity = np.diff(sampled, axis=1)
    descriptor = np.concatenate(
        [sampled.reshape(tracks.shape[0], -1), velocity.reshape(tracks.shape[0], -1)],
        axis=1,
    )
    return descriptor.astype(np.float32, copy=False), eligible


def trajectory_fps_indices(
    tracks_world: np.ndarray,
    valid_mask: np.ndarray | None,
    target_count: int,
    *,
    temporal_samples: int = 7,
) -> np.ndarray:
    """Deterministic farthest-point sampling over trajectory descriptors."""
    descriptor, eligible_mask = trajectory_descriptors(
        tracks_world, valid_mask, temporal_samples=temporal_samples
    )
    eligible = np.flatnonzero(eligible_mask)
    count = min(max(int(target_count), 0), int(eligible.size))
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    points = descriptor[eligible]
    centroid = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    chosen_local = np.empty((count,), dtype=np.int64)
    chosen_local[0] = first
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    min_distance[first] = -1.0
    for out_idx in range(1, count):
        next_idx = int(np.argmax(min_distance))
        chosen_local[out_idx] = next_idx
        distance = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[chosen_local[: out_idx + 1]] = -1.0
    return eligible[chosen_local]


def descriptor_coverage_error(
    tracks_world: np.ndarray,
    valid_mask: np.ndarray | None,
    selected_indices: np.ndarray,
    *,
    temporal_samples: int = 7,
    chunk_size: int = 4096,
) -> float:
    """Mean nearest selected descriptor distance over eligible candidates."""
    descriptor, eligible_mask = trajectory_descriptors(
        tracks_world, valid_mask, temporal_samples=temporal_samples
    )
    candidates = descriptor[eligible_mask]
    selected = descriptor[np.asarray(selected_indices, dtype=np.int64)]
    if candidates.size == 0 or selected.size == 0:
        return float("nan")
    nearest = []
    selected_norm = np.sum(selected * selected, axis=1)[None, :]
    for start in range(0, candidates.shape[0], max(int(chunk_size), 1)):
        chunk = candidates[start : start + chunk_size]
        squared = (
            np.sum(chunk * chunk, axis=1)[:, None]
            + selected_norm
            - 2.0 * (chunk @ selected.T)
        )
        nearest.append(np.sqrt(np.maximum(np.min(squared, axis=1), 0.0)))
    return float(np.concatenate(nearest).mean())


def cross_query_nearest_descriptor_distances(
    tracks_world: np.ndarray,
    valid_mask: np.ndarray | None,
    query_ids: np.ndarray,
    *,
    temporal_samples: int = 7,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Nearest full-trajectory descriptor distance to a different query pool."""
    descriptor, eligible_mask = trajectory_descriptors(
        tracks_world, valid_mask, temporal_samples=temporal_samples
    )
    query = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if query.shape != (descriptor.shape[0],):
        raise ValueError(f"query_ids must be {(descriptor.shape[0],)}, got {query.shape}")
    output = []
    eligible = np.flatnonzero(eligible_mask)
    for query_id in np.unique(query[eligible]):
        source_ids = eligible[query[eligible] == query_id]
        target_ids = eligible[query[eligible] != query_id]
        if source_ids.size == 0 or target_ids.size == 0:
            continue
        targets = descriptor[target_ids]
        target_norm = np.sum(targets * targets, axis=1)[None, :]
        for start in range(0, source_ids.size, max(int(chunk_size), 1)):
            sources = descriptor[source_ids[start : start + chunk_size]]
            squared = (
                np.sum(sources * sources, axis=1)[:, None]
                + target_norm
                - 2.0 * (sources @ targets.T)
            )
            output.append(np.sqrt(np.maximum(np.min(squared, axis=1), 0.0)))
    if not output:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(output).astype(np.float32, copy=False)
