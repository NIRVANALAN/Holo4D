from __future__ import annotations

from typing import Optional

import numpy as np


def _normalize_query_indices(
    query_idx_per_identity: Optional[np.ndarray],
    *,
    n_tracks: int,
    n_frames: int,
) -> np.ndarray:
    if query_idx_per_identity is None:
        qidx = np.zeros((n_tracks,), dtype=np.int64)
    else:
        qidx = np.asarray(query_idx_per_identity, dtype=np.int64).reshape(-1)
        if qidx.shape[0] != n_tracks:
            qidx = np.zeros((n_tracks,), dtype=np.int64)
    return np.clip(qidx, 0, max(int(n_frames) - 1, 0))


def _static_points_at_query(
    tracks_world: np.ndarray,
    ids: np.ndarray,
    qidx: np.ndarray,
) -> np.ndarray:
    if ids.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    row = ids.astype(np.int64, copy=False)
    return np.asarray(tracks_world[row, qidx[row], :], dtype=np.float32)


def _novel_static_mask_by_radius(
    base_points: np.ndarray,
    extra_points: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Return True for extra points without a nearby base static point."""
    base = np.asarray(base_points, dtype=np.float32)
    extra = np.asarray(extra_points, dtype=np.float32)
    if extra.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    finite_extra = np.isfinite(extra).all(axis=1)
    novel = finite_extra.copy()
    if base.shape[0] == 0:
        return novel

    finite_base = np.isfinite(base).all(axis=1)
    base = base[finite_base]
    if base.shape[0] == 0:
        return novel

    r = max(float(radius), 1e-8)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(base)
        dist, _ = tree.query(extra[finite_extra], k=1, distance_upper_bound=r)
        finite_ids = np.where(finite_extra)[0]
        novel[finite_ids] = ~np.isfinite(dist)
        return novel
    except Exception:
        # Fallback: conservative voxel occupancy. It may keep a few duplicate
        # boundary points, but never removes a clearly novel static point.
        base_vox = {tuple(v) for v in np.floor(base / r).astype(np.int64)}
        extra_vox = np.floor(extra / r).astype(np.int64)
        for i, vox in enumerate(extra_vox):
            if not finite_extra[i]:
                novel[i] = False
                continue
            overlap = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if (int(vox[0] + dx), int(vox[1] + dy), int(vox[2] + dz)) in base_vox:
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    break
            novel[i] = not overlap
        return novel


def fuse_static_background_valid_mask(
    tracks_valid: Optional[np.ndarray],
    dynamic_at_tracks: Optional[np.ndarray],
    tracks_shape: tuple[int, int],
    query_idx_per_identity: Optional[np.ndarray] = None,
    *,
    tracks_world: Optional[np.ndarray] = None,
    static_overlap_radius: float = 0.02,
) -> Optional[np.ndarray]:
    """Build q-aware CoW/VGGT validity with static-background dedup.

    Base-query static/background keeps the q1 behavior and stays valid after
    static freezing. Extra-query dynamic identities keep their original raw
    tracker valid mask. Extra-query static/background identities are kept only
    if their query-frame 3D point is not already covered by base-query static
    points; when kept, they still keep their original raw tracker valid mask
    so late-query background is not hallucinated as observed before it appears.
    """
    if dynamic_at_tracks is None:
        return tracks_valid

    dyn = np.asarray(dynamic_at_tracks, dtype=bool)
    if dyn.shape[:2] != tracks_shape:
        return tracks_valid

    n_tracks, n_frames = tracks_shape
    if tracks_valid is None:
        raw_valid = np.ones(tracks_shape, dtype=bool)
        out = raw_valid.copy()
    else:
        raw_valid = np.asarray(tracks_valid, dtype=bool)
        if raw_valid.shape[:2] != tracks_shape:
            raw_valid = np.ones(tracks_shape, dtype=bool)
        out = raw_valid.copy()

    qidx = _normalize_query_indices(
        query_idx_per_identity,
        n_tracks=n_tracks,
        n_frames=n_frames,
    )
    row = np.arange(n_tracks, dtype=np.int64)
    dynamic_ids = dyn[row, qidx]
    static_ids = ~dynamic_ids
    base_query = int(qidx.min()) if qidx.size else 0

    base_static_ids = static_ids & (qidx == base_query)
    extra_static_ids = static_ids & (qidx != base_query)
    out[base_static_ids, :] = True

    extra_rows = np.where(extra_static_ids)[0]
    if extra_rows.size == 0:
        return out
    out[extra_rows, :] = False

    if tracks_world is None:
        return out
    tracks = np.asarray(tracks_world, dtype=np.float32)
    if tracks.shape[:2] != tracks_shape or tracks.shape[-1] != 3:
        return out

    base_rows = np.where(base_static_ids)[0]
    base_points = _static_points_at_query(tracks, base_rows, qidx)
    extra_points = _static_points_at_query(tracks, extra_rows, qidx)
    novel = _novel_static_mask_by_radius(
        base_points,
        extra_points,
        radius=static_overlap_radius,
    )
    if np.any(novel):
        # Keep late-query novel static as additional evidence, but do not
        # promote it to persistent all-frame evidence. CoW/VGGT validity is an
        # observation/confidence mask, so a q48-only background point should not
        # become valid in earlier source frames where it may be occluded.
        out[extra_rows[novel], :] = raw_valid[extra_rows[novel]]
    return out
