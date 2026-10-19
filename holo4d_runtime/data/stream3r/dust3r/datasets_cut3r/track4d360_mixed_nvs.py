"""
Mixed multi-dataset NVS dataloaders for Track4D-360 training.

Extends Syn4D_Multi_NVS with additional dataset types:
  - Kubric_NVS:       Multi-camera Kubric (49/81 frames, 4 cameras) with bbox-tracked 3D
  - StaticScene_NVS:  Two-window NVS for static 3D scenes (TartanAir / DL3DV / MVS-Synth / PointOdyssey)
  - RE10K_NVS:        Camera-only NVS from RealEstate10K (no depth / no tracks)

All classes inherit from EasyDataset for YAML-compatible composition:
    "500 @ Kubric_NVS(ROOT='...', num_views=49)"

Each __getitem__ returns the same dict format as Syn4D_Multi_NVS.
"""

import json
import math
import os
import os.path as osp
import pickle
import random
import re
import subprocess
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import PIL.Image
import torch
from scipy.spatial.transform import Rotation

from .base.easy_dataset import EasyDataset
from .mixed_nvs_captions import (
    ChunkedCaptionStore,
    default_caption_dir_for_dataset,
    report_cache_miss,
    sanitize_mixed_nvs_caption,
)
from .track_query_sampling import (
    choose_canonical_random_query_indices as _choose_canonical_random_query_indices,
    choose_track_query_indices as _choose_track_query_indices,
)

# Updated 2026-04-30: replaces the previous "A multi-view video scene."
# placeholder with a task-meaningful prompt. Used as the fallback whenever
# a dataset can't produce a per-sample caption (cache miss, missing
# caption.txt, sanitize-empty). Must be in
# scripts/wan2.1_fun/train_track4d360.py:_DEFAULT_PROMPTS_TO_PRECOMPUTE so
# the strict-cache check finds a precomputed embedding and the trainer
# doesn't crash on legitimate fallback samples.
DEFAULT_PROMPT = "Generate the novel view trajectory video given the input video."


# ======================================================================
# Shared utilities
# ======================================================================

def _sanitize_indices(indices, total, target_len):
    """Defensive guard against negative / out-of-range frame indices.

    Python list indexing silently wraps negative indices to the end of the list
    (``lst[-3]`` returns the third-to-last element), so an off-by-one that
    produces ``indices = [-3, -2, -1, 0, 1, 2, ...]`` yields a video that
    appears to "start near the end" and then "wraps to frame 0". The static
    window samplers (``_getitem_overlap_window`` / ``_getitem_two_window``)
    can generate such indices when a scene's total frame count drops below
    ``num_views`` or when clamp math produces a negative ``shift``.

    This function filters out any index ``i`` with ``i < 0`` or ``i >= total``,
    preserves the remaining order, falls back to ``[0]`` if the filter wipes
    everything, and pads with the last valid index to reach ``target_len``.
    The result is always an ordered T-length list of non-negative indices
    strictly inside ``[0, total)``.

    The caller-side logic is NOT removed — the goal here is "fail predictably
    (freeze at last frame) instead of silently wrapping to frame 0".
    """
    if total <= 0:
        return [0] * int(target_len)
    cleaned = [int(i) for i in indices if 0 <= int(i) < int(total)]
    if not cleaned:
        cleaned = [0]
    while len(cleaned) < int(target_len):
        cleaned.append(cleaned[-1])
    return cleaned[: int(target_len)]


def _source_rgb_at_track_grid(src_frames, track_spatial_stride: int, track_query_indices=None) -> np.ndarray:
    """Sample source RGB on the same grid/order as depth-built 3D tracks.

    Returns ``(N * num_queries, T, 3)`` in [0, 1]. For each query chunk we sample
    the RGB frame at that query index and broadcast it over time, matching the
    track identity grid used by the Conv3D track adapter.
    """
    T = len(src_frames)
    if T <= 0:
        raise ValueError("src_frames must contain at least one frame")
    first = np.asarray(src_frames[0], dtype=np.float32)
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"source frame must be HxWx3 RGB, got {first.shape}")
    H, W = first.shape[:2]
    s = max(int(track_spatial_stride), 1)
    yy = np.arange(0, H, s)
    xx = np.arange(0, W, s)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    grid_y = grid_y.ravel()
    grid_x = grid_x.ravel()
    q_indices = [0] if track_query_indices is None else [int(q) for q in track_query_indices]
    chunks = []
    for q in q_indices:
        q = max(0, min(q, T - 1))
        img = np.asarray(src_frames[q], dtype=np.float32)
        if img.max(initial=0.0) > 2.0:
            img = img / 255.0
        rgb = img[grid_y, grid_x, :3].astype(np.float32, copy=False)
        chunks.append(np.broadcast_to(rgb[:, None, :], (rgb.shape[0], T, 3)).copy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _mask_at_track_grid(mask_thw, track_spatial_stride: int, track_query_indices=None) -> np.ndarray:
    """Sample a per-pixel mask on the SAME grid/order as `_source_rgb_at_track_grid`.

    Returns ``(N * num_queries, T)`` in {0,1}. Mirrors the RGB gather exactly:
    samples the query frame's mask on the stride grid and broadcasts over time
    (dynamic-ness is a per-identity property), so the returned rows are
    row-aligned with `source_rgb_at_tracks` / `tracks_world`.
    """
    m = np.asarray(mask_thw)
    if m.ndim == 2:  # (H, W)
        m = m[None]
    if m.ndim != 3:
        raise ValueError(f"mask must be (T,H,W) or (H,W), got {m.shape}")
    T, H, W = m.shape
    if T <= 0:
        raise ValueError("mask must contain at least one frame")
    s = max(int(track_spatial_stride), 1)
    yy = np.arange(0, H, s)
    xx = np.arange(0, W, s)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    grid_y = grid_y.ravel()
    grid_x = grid_x.ravel()
    q_indices = [0] if track_query_indices is None else [int(q) for q in track_query_indices]
    chunks = []
    for q in q_indices:
        q = max(0, min(q, T - 1))
        vals = m[q][grid_y, grid_x].astype(np.float32, copy=False)  # (N,)
        chunks.append(np.broadcast_to(vals[:, None], (vals.shape[0], T)).copy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _source_rgb_at_track_xy(src_frames, query_xy: np.ndarray, num_frames: int | None = None) -> np.ndarray:
    """Sample source RGB at arbitrary query points and broadcast over time."""
    T = len(src_frames) if num_frames is None else min(int(num_frames), len(src_frames))
    if T <= 0:
        raise ValueError("src_frames must contain at least one frame")
    img = np.asarray(src_frames[0], dtype=np.float32)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"source frame must be HxWx3 RGB, got {img.shape}")
    if img.max(initial=0.0) > 2.0:
        img = img / 255.0
    H, W = img.shape[:2]
    xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, W - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, H - 1)
    rgb = img[y, x, :3].astype(np.float32, copy=False)
    return np.broadcast_to(rgb[:, None, :], (rgb.shape[0], T, 3)).copy()


def _build_sparse_track_valid_mask(
    image_h: int,
    image_w: int,
    num_frames: int,
    stride: int,
    rng=None,
    block_jitter_prob: float = 0.0,
    block_size: int = 64,
    priority_mask=None,
) -> np.ndarray:
    """Return a random-subset validity mask for sparse-track training.

    The tensor shape remains `(H*W,T)` so the conv3d track adapter can be
    reused unchanged. Only selected sparse anchor identities are marked valid;
    all other grid cells flow to the adapter's null token through the visibility
    path. ``stride`` controls density as roughly one point per ``stride^2``
    dense-grid tracks, but unlike the older grid-offset sampler it draws a
    random subset to match DAVIS Cow/VGGT sparse-track visualizations.
    """
    H = int(image_h)
    W = int(image_w)
    T = int(num_frames)
    s = max(int(stride), 1)
    rng = rng if rng is not None else np.random.default_rng()
    total = H * W
    priority_flat = None
    if priority_mask is not None:
        priority_arr = np.asarray(priority_mask).astype(bool)
        if priority_arr.size == total:
            priority_flat = priority_arr.reshape(-1)
    candidate = np.ones((total,), dtype=bool)
    if priority_flat is not None:
        candidate &= ~priority_flat
    candidate_idx = np.flatnonzero(candidate)
    keep = max(1, int(round(len(candidate_idx) / float(s * s))))
    keep = min(keep, len(candidate_idx))
    weights = None
    if keep > 0 and s > 1 and rng.random() < max(0.0, min(1.0, float(block_jitter_prob))):
        b = max(int(block_size), 1)
        by = (H + b - 1) // b
        bx = (W + b - 1) // b
        block_weights = rng.choice(
            np.asarray([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float64),
            size=(by, bx),
            replace=True,
        )
        weights_2d = np.zeros((H, W), dtype=np.float64)
        for iy in range(by):
            for ix in range(bx):
                weights_2d[iy * b : min((iy + 1) * b, H), ix * b : min((ix + 1) * b, W)] = (
                    block_weights[iy, ix]
                )
        weights = weights_2d.reshape(-1)[candidate_idx]
        weights = weights / max(float(weights.sum()), 1e-12)
    chosen = rng.choice(candidate_idx, size=keep, replace=False, p=weights) if keep > 0 else np.asarray([], dtype=np.int64)
    flat = np.zeros((total,), dtype=bool)
    flat[chosen] = True
    if priority_flat is not None:
        flat |= priority_flat
    return np.repeat(flat[:, None], T, axis=1)


def _dynamic_track_priority_mask(sample: dict, image_h: int, image_w: int):
    mask = sample.get("source_dynamic_mask")
    if not isinstance(mask, np.ndarray):
        return None
    arr = np.asarray(mask).astype(bool)
    if arr.ndim == 3:
        arr = arr.any(axis=0)
    elif arr.ndim != 2:
        return None
    H = int(image_h)
    W = int(image_w)
    if arr.shape != (H, W):
        arr = cv2.resize(arr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not arr.any():
        return None
    return arr.reshape(-1)


def _resize_dynamic_priority_frame(mask_2d: np.ndarray, image_h: int, image_w: int):
    arr = np.asarray(mask_2d).astype(bool)
    H = int(image_h)
    W = int(image_w)
    if arr.shape != (H, W):
        arr = cv2.resize(arr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not arr.any():
        return None
    return arr.reshape(-1)


def _dynamic_track_priority_masks_for_query_grids(
    sample: dict,
    image_h: int,
    image_w: int,
    query_grid_repeats: int,
):
    mask = sample.get("source_dynamic_mask")
    repeats = max(int(query_grid_repeats), 1)
    if not isinstance(mask, np.ndarray):
        return [None] * repeats
    arr = np.asarray(mask).astype(bool)
    if arr.ndim == 2:
        one = _resize_dynamic_priority_frame(arr, image_h, image_w)
        return [one] * repeats
    if arr.ndim != 3:
        return [None] * repeats

    query_indices = sample.get("track_query_indices")
    if query_indices is None:
        query_indices = [int(sample.get("track_query_idx", 0))]
    query_indices = [int(q) for q in np.asarray(query_indices).reshape(-1).tolist()]
    if not query_indices:
        query_indices = [0]

    out = []
    T = int(arr.shape[0])
    for r in range(repeats):
        q = query_indices[min(r, len(query_indices) - 1)]
        q = max(0, min(int(q), T - 1))
        out.append(_resize_dynamic_priority_frame(arr[q], image_h, image_w))
    return out


def _parse_sparse_track_strides(value) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        raw = value
    elif isinstance(value, str):
        raw = [p for p in re.split(r"[,| ]+", value.strip()) if p]
    else:
        raw = [value]
    strides = []
    for item in raw:
        stride = int(item)
        if stride > 0:
            strides.append(stride)
    if not strides:
        strides = [8]
    return tuple(strides)


def _normalize_sparse_track_query_stride_mode(value) -> str:
    mode = str(value or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "0": "none",
        "q": "query_count_floor",
        "query": "query_count_floor",
        "query_count": "query_count_floor",
        "linear": "query_count_floor",
        "sqrt": "sqrt_budget",
        "budget": "sqrt_budget",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "query_count_floor", "sqrt_budget"}:
        raise ValueError(
            "sparse_track_query_stride_mode must be one of "
            "'none', 'query_count_floor', or 'sqrt_budget', "
            f"got {value!r}"
        )
    return mode


def _adjust_sparse_stride_for_query_budget(base_stride: int, query_grid_repeats: int, mode: str) -> int:
    base_stride = max(int(base_stride), 1)
    repeats = max(int(query_grid_repeats), 1)
    mode = _normalize_sparse_track_query_stride_mode(mode)
    if mode == "none" or repeats <= 1:
        return base_stride
    if mode == "sqrt_budget":
        # Point count scales as query_count / stride^2. sqrt(query_count) keeps
        # total valid tracks close to the single-query base-stride budget.
        return max(base_stride, int(math.ceil(math.sqrt(float(repeats)))))
    # Conservative cap requested for multi-query debugging/training: q=4 forces
    # at least s=4, q=3 at least s=3, etc.
    return max(base_stride, repeats)


def _parse_int_choices(value, default=(1,)) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        raw = value
    elif isinstance(value, str):
        raw = [p for p in re.split(r"[,| ]+", value.strip()) if p]
    else:
        raw = [value]
    choices = []
    for item in raw:
        v = int(item)
        if v > 0:
            choices.append(v)
    return tuple(choices or tuple(default))


def _choose_fourrc_training_query_indices(rng, max_queries: int = 1, allow_three: bool = True) -> list[int]:
    """Choose the training-time multi-query set for 49-frame 4RC tracks.

    Backward-compatible default is `[0]`. With `max_queries>=2`, q0 is always
    included and q48 may be added; with `max_queries>=3` and `allow_three`,
    `[0,24,48]` is also sampled. The actual dataloader can only load extra
    queries when a matching multi-query cache exists, but this helper defines
    the intended schedule consistently for training and preview tooling.
    """
    max_queries = int(max_queries)
    if max_queries <= 1:
        return [0]
    rng = rng if rng is not None else np.random.default_rng()
    choices = [[0], [0, 48]]
    if max_queries >= 3 and bool(allow_three):
        choices.append([0, 24, 48])
    return list(choices[int(rng.integers(len(choices)))])


def _copy_sample_for_geometry_mutation(sample: dict) -> dict:
    """Shallow-copy a dataset sample and detach mutable numpy arrays."""
    out = dict(sample)
    for key in (
        "tracks_world",
        "tracks_world_valid_mask",
        "source_depth",
        "source_depth_video",
        "source_depth_valid",
    ):
        value = out.get(key)
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
    return out


def _infer_track_grid_hw(sample: dict, num_tracks: int) -> tuple[int, int] | None:
    """Infer the dense source grid used by a `(H*W,T,3)` track tensor."""
    for key in ("source_depth_video", "source_depth", "target_depth"):
        arr = sample.get(key)
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim == 3:
            h, w = int(arr.shape[-2]), int(arr.shape[-1])
        elif arr.ndim == 2:
            h, w = int(arr.shape[0]), int(arr.shape[1])
        else:
            continue
        if h > 0 and w > 0 and h * w == int(num_tracks):
            return h, w
    return None


def _infer_source_grid_hw(sample: dict) -> tuple[int, int] | None:
    for key in ("source_depth_video", "source_depth", "target_depth"):
        arr = sample.get(key)
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim == 3:
            h, w = int(arr.shape[-2]), int(arr.shape[-1])
        elif arr.ndim == 2:
            h, w = int(arr.shape[0]), int(arr.shape[1])
        else:
            continue
        if h > 0 and w > 0:
            return h, w
    return None


def _valid_track_mask_from_tracks(tracks_world: np.ndarray) -> np.ndarray:
    tracks = np.asarray(tracks_world)
    valid = np.isfinite(tracks).all(axis=-1)
    valid &= np.linalg.norm(tracks, axis=-1) > 1e-8
    return valid.astype(bool, copy=False)


class GeometryRobustnessWrapper_NVS(EasyDataset):
    """Backward-compatible wrapper that simulates estimated geometry inputs.

    It can turn dense `(H*W,T,3)` tracks into sparse random-grid tracks through
    `tracks_world_valid_mask`, and can add small offset noise to conditioning
    tracks / source depth without touching target supervision.
    """

    def __init__(
        self,
        dataset,
        sparse_track_prob=0.0,
        sparse_track_stride=8,
        sparse_track_query_stride_mode="none",
        sparse_track_block_jitter_prob=0.5,
        sparse_track_block_size=64,
        multi_query_prob=0.0,
        multi_query_choices="1",
        multi_query_schedule="legacy_fixed",
        static_multi_query_visibility="query_valid_repeat",
        static_surface_depth_rel_tol=0.1,
        static_surface_depth_abs_tol=0.05,
        noise_prob=0.0,
        track_offset_std=0.0,
        track_temporal_jitter_std=None,
        track_xy_noise_ratio=0.25,
        track_frame_depth_jitter_ratio=0.5,
        depth_offset_std=0.0,
        depth_scale_jitter=0.0,
        source_rgb_noise_std=0.0,
        source_rgb_frame_jitter_std=0.0,
        dynamic_mask_aug_prob=0.0,
        dynamic_mask_morph_kernel=0,
        dynamic_mask_flip_prob=0.0,
        return_dynamic_masks=None,
        seed=None,
        apply_to_camera_only=False,
    ):
        if not isinstance(dataset, EasyDataset):
            raise TypeError(
                "GeometryRobustnessWrapper_NVS expects an EasyDataset instance, "
                f"got {type(dataset).__name__}"
            )
        self.dataset = dataset
        self.sparse_track_prob = float(sparse_track_prob)
        self.sparse_track_strides = _parse_sparse_track_strides(sparse_track_stride)
        self.sparse_track_stride = self.sparse_track_strides[0]
        self.sparse_track_query_stride_mode = _normalize_sparse_track_query_stride_mode(
            sparse_track_query_stride_mode
        )
        self.sparse_track_block_jitter_prob = float(sparse_track_block_jitter_prob)
        self.sparse_track_block_size = int(sparse_track_block_size)
        self.multi_query_prob = float(multi_query_prob)
        self.multi_query_choices = _parse_int_choices(multi_query_choices, default=(1,))
        self.multi_query_schedule = str(multi_query_schedule).strip().lower()
        if self.multi_query_schedule not in {"legacy_fixed", "canonical_random_subset"}:
            raise ValueError(f"unsupported multi_query_schedule={multi_query_schedule!r}")
        self.static_multi_query_visibility = str(static_multi_query_visibility).strip().lower()
        if self.static_multi_query_visibility not in {"query_valid_repeat", "surface"}:
            raise ValueError(
                f"unsupported static_multi_query_visibility={static_multi_query_visibility!r}"
            )
        self.static_surface_depth_rel_tol = float(static_surface_depth_rel_tol)
        self.static_surface_depth_abs_tol = float(static_surface_depth_abs_tol)
        self.noise_prob = float(noise_prob)
        self.track_offset_std = float(track_offset_std)
        self.track_temporal_jitter_std = (
            float(track_offset_std)
            if track_temporal_jitter_std is None
            else float(track_temporal_jitter_std)
        )
        self.track_xy_noise_ratio = float(track_xy_noise_ratio)
        self.track_frame_depth_jitter_ratio = float(track_frame_depth_jitter_ratio)
        self.depth_offset_std = float(depth_offset_std)
        self.depth_scale_jitter = float(depth_scale_jitter)
        self.source_rgb_noise_std = float(source_rgb_noise_std)
        self.source_rgb_frame_jitter_std = float(source_rgb_frame_jitter_std)
        self.dynamic_mask_aug_prob = float(dynamic_mask_aug_prob)
        self.dynamic_mask_morph_kernel = int(dynamic_mask_morph_kernel)
        self.dynamic_mask_flip_prob = float(dynamic_mask_flip_prob)
        self.seed = None if seed is None else int(seed)
        self.apply_to_camera_only = bool(apply_to_camera_only)
        if return_dynamic_masks is not None and hasattr(self.dataset, "return_dynamic_masks"):
            self.dataset.return_dynamic_masks = bool(return_dynamic_masks)

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return (
            "GeometryRobustnessWrapper_NVS("
            f"sparse_track_prob={self.sparse_track_prob}, "
            f"sparse_track_stride={self.sparse_track_strides}, "
            f"sparse_track_query_stride_mode={self.sparse_track_query_stride_mode}, "
            f"sparse_track_block_jitter_prob={self.sparse_track_block_jitter_prob}, "
            f"sparse_track_block_size={self.sparse_track_block_size}, "
            f"multi_query_prob={self.multi_query_prob}, "
            f"multi_query_choices={self.multi_query_choices}, "
            f"multi_query_schedule={self.multi_query_schedule}, "
            f"static_multi_query_visibility={self.static_multi_query_visibility}, "
            f"noise_prob={self.noise_prob}, "
            f"track_offset_std={self.track_offset_std}, "
            f"track_temporal_jitter_std={self.track_temporal_jitter_std}, "
            f"track_xy_noise_ratio={self.track_xy_noise_ratio}, "
            f"track_frame_depth_jitter_ratio={self.track_frame_depth_jitter_ratio}, "
            f"depth_offset_std={self.depth_offset_std}, "
            f"depth_scale_jitter={self.depth_scale_jitter}, "
            f"source_rgb_noise_std={self.source_rgb_noise_std}, "
            f"source_rgb_frame_jitter_std={self.source_rgb_frame_jitter_std}, "
            f"dynamic_mask_aug_prob={self.dynamic_mask_aug_prob}, "
            f"return_dynamic_masks={getattr(self.dataset, 'return_dynamic_masks', None)}, "
            f"dataset={repr(self.dataset)})"
        )

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    @property
    def num_views(self):
        return self.dataset.num_views

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def _rng_for_idx(self, idx):
        if self.seed is None:
            return np.random.default_rng()
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        return np.random.default_rng(self.seed + int(idx))

    def __getitem__(self, idx):
        sample = _copy_sample_for_geometry_mutation(self.dataset[idx])
        if sample.get("camera_only_tier", False) and not self.apply_to_camera_only:
            return sample
        rng = self._rng_for_idx(idx)
        self._maybe_apply_multi_query_tracks(sample, rng)
        self._maybe_sparsify_tracks(sample, rng)
        self._maybe_add_geometry_noise(sample, rng)
        self._maybe_augment_dynamic_masks(sample, rng)
        self._attach_source_dynamic_mask_at_tracks(sample)
        return sample

    def _attach_source_dynamic_mask_at_tracks(self, sample: dict) -> None:
        """Gather the per-pixel source dynamic mask onto track identities.

        Produces ``source_dynamic_mask_at_tracks`` (N,T) row-aligned with
        ``source_rgb_at_tracks`` / ``tracks_world`` using the SAME stride grid +
        query indices, so the Conv3D track adapter can carry one per-identity
        motion channel (reprojected to target by its target-UV splat). Runs after
        all mutations (sparse / multi-query rebuild / mask aug) so it matches the
        final tracks. No-op when the sample lacks a source dynamic mask or tracks.
        """
        import warnings
        mask = sample.get("source_dynamic_mask")
        tracks = sample.get("tracks_world")
        if not isinstance(mask, np.ndarray) or not isinstance(tracks, np.ndarray):
            return
        stride = max(int(getattr(self.dataset, "track_spatial_stride", 1) or 1), 1)
        q_indices = sample.get("track_query_indices")
        if not q_indices:
            q_indices = [int(sample.get("track_query_idx", 0))]
        try:
            at_tracks = _mask_at_track_grid(mask, stride, q_indices)
        except Exception as exc:  # gather is best-effort; builder zero-fills on absence
            warnings.warn(f"source_dynamic_mask_at_tracks gather failed: {exc}", stacklevel=2)
            return
        if at_tracks.shape[0] != tracks.shape[0]:
            warnings.warn(
                "source_dynamic_mask_at_tracks row count "
                f"{at_tracks.shape[0]} != tracks {tracks.shape[0]}; skipping mask channel.",
                stacklevel=2,
            )
            return
        sample["source_dynamic_mask_at_tracks"] = at_tracks

    def _maybe_apply_multi_query_tracks(self, sample: dict, rng) -> None:
        if "track_query_indices" in sample:
            return
        tracks = sample.get("tracks_world")
        if not isinstance(tracks, np.ndarray) or tracks.ndim != 3:
            return
        if rng.random() >= max(0.0, min(1.0, self.multi_query_prob)):
            return
        q_count = int(self.multi_query_choices[int(rng.integers(len(self.multi_query_choices)))])
        if self.multi_query_schedule == "canonical_random_subset":
            q_indices = _choose_canonical_random_query_indices(
                int(tracks.shape[1]), q_count, rng,
            )
        else:
            q_indices = _choose_track_query_indices(int(tracks.shape[1]), q_count)
            if len(q_indices) <= 1:
                return
        if bool(sample.get("has_dynamic_tracks", False)):
            # Dynamic datasets must synthesize their own multi-query tracks
            # (Syn4D_Multi_NVS does this internally and sets track_query_indices,
            # so such samples already returned at the top of this method). We do
            # NOT rebuild dynamic multi-query from depth here: the static path
            # below freezes each identity at its query-frame point, which is
            # wrong for moving objects. So a dynamic dataset without its own
            # multi-query stays single-query — a safe degradation, not silent
            # corruption. (Historically this branch tried to read a
            # `tracks_world_by_query` dict that no dataset ever populated.)
            return
        self._apply_static_multi_query_tracks(sample, q_indices)

    def _static_surface_validity(
        self,
        points_world: np.ndarray,
        depth_video: np.ndarray,
        source_cam2world: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Depth-test static query points against every source frame."""
        points = np.asarray(points_world, dtype=np.float32)
        depth = np.asarray(depth_video, dtype=np.float32)
        c2w = np.asarray(source_cam2world, dtype=np.float32)
        K_all = np.asarray(intrinsics, dtype=np.float32)
        T, H, W = depth.shape
        valid = np.zeros((points.shape[0], T), dtype=bool)
        finite_world = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-8)
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1,
        )
        for t in range(T):
            K_t = K_all[t] if K_all.ndim == 3 else K_all
            world2cam = np.linalg.inv(c2w[t]).astype(np.float32)
            cam = (points_h @ world2cam.T)[:, :3]
            z = cam[:, 2]
            z_safe = np.maximum(z, 1e-6)
            u = K_t[0, 0] * cam[:, 0] / z_safe + K_t[0, 2]
            v = K_t[1, 1] * cam[:, 1] / z_safe + K_t[1, 2]
            in_bounds = (
                finite_world
                & np.isfinite(u)
                & np.isfinite(v)
                & (z > 1e-6)
                & (u >= 0.0)
                & (u <= float(W - 1))
                & (v >= 0.0)
                & (v <= float(H - 1))
            )
            ids = np.where(in_bounds)[0]
            if ids.size == 0:
                continue
            ui = np.rint(u[ids]).astype(np.int64).clip(0, W - 1)
            vi = np.rint(v[ids]).astype(np.int64).clip(0, H - 1)
            observed = depth[t, vi, ui]
            tol = np.maximum(
                self.static_surface_depth_abs_tol,
                self.static_surface_depth_rel_tol * np.abs(observed),
            )
            visible = (
                np.isfinite(observed)
                & (observed > 1e-6)
                & (np.abs(z[ids] - observed) <= tol)
            )
            valid[ids[visible], t] = True
        return valid

    def _apply_static_multi_query_tracks(self, sample: dict, q_indices: list[int]) -> None:
        depth_video = sample.get("source_depth_video")
        source_cam2world = sample.get("source_cam2world")
        K = sample.get("source_intrinsics")
        if not (
            isinstance(depth_video, np.ndarray)
            and isinstance(source_cam2world, np.ndarray)
            and isinstance(K, np.ndarray)
        ):
            return
        depth_video = np.asarray(depth_video, dtype=np.float32)
        source_cam2world = np.asarray(source_cam2world, dtype=np.float32)
        if depth_video.ndim != 3 or source_cam2world.ndim != 3:
            return
        T, H, W = depth_video.shape
        if source_cam2world.shape[0] < T:
            return
        stride = max(int(getattr(self.dataset, "track_spatial_stride", 1) or 1), 1)
        grid_y = np.arange(0, H, stride, dtype=np.int64)
        grid_x = np.arange(0, W, stride, dtype=np.int64)
        chunks = []
        valid_chunks = []
        for q in q_indices:
            q = max(0, min(int(q), T - 1))
            K_q = K[q] if K.ndim == 3 else K
            pts_world, valid = depth_to_world_pts3d(depth_video[q], K_q, source_cam2world[q])
            pts_flat = pts_world[grid_y[:, None], grid_x[None, :]].reshape(-1, 3).astype(np.float32)
            valid_flat = valid[grid_y[:, None], grid_x[None, :]].reshape(-1)
            tracks_q = np.repeat(pts_flat[:, None, :], T, axis=1)
            tracks_q[~valid_flat] = 0.0
            chunks.append(tracks_q)
            if self.static_multi_query_visibility == "surface":
                valid_q = self._static_surface_validity(
                    pts_flat,
                    depth_video,
                    source_cam2world,
                    K,
                )
                valid_q &= valid_flat[:, None]
            else:
                valid_q = np.repeat(valid_flat[:, None], T, axis=1)
            valid_chunks.append(valid_q)
        sample["tracks_world"] = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        sample["tracks_world_valid_mask"] = np.concatenate(valid_chunks, axis=0).astype(bool, copy=False)
        sample["track_query_indices"] = [int(q) for q in q_indices]
        sample["track_query_idx"] = int(q_indices[0])
        sample["track_query_count"] = len(q_indices)
        sample["tracks_query_idx_per_identity"] = np.concatenate(
            [np.full((chunk.shape[0],), int(q), dtype=np.int32) for q, chunk in zip(q_indices, chunks)],
            axis=0,
        )
        if "source_rgb_video" in sample:
            sample["source_rgb_at_tracks"] = _source_rgb_at_track_grid(
                sample["source_rgb_video"],
                stride,
                q_indices,
            )
            if sample["source_rgb_at_tracks"].shape[0] != sample["tracks_world"].shape[0]:
                raise ValueError(
                    "source_rgb_at_tracks row count does not match rebuilt static "
                    f"multi-query tracks: rgb={sample['source_rgb_at_tracks'].shape[0]} "
                    f"tracks={sample['tracks_world'].shape[0]} stride={stride}"
                )
        elif "source_rgb_at_tracks" in sample:
            # Existing RGB-at-track tensors are query-anchor specific. Dropping
            # avoids shape mismatches when static tracks are rebuilt from depth.
            sample.pop("source_rgb_at_tracks", None)

    def _maybe_sparsify_tracks(self, sample: dict, rng) -> None:
        tracks = sample.get("tracks_world")
        if not isinstance(tracks, np.ndarray) or tracks.ndim != 3:
            return
        explicit_valid = sample.get("tracks_world_valid_mask")
        if isinstance(explicit_valid, np.ndarray):
            dense_valid = explicit_valid.astype(bool, copy=True)
        else:
            dense_valid = _valid_track_mask_from_tracks(tracks).astype(bool, copy=True)
        sample["tracks_world_dense_valid_mask"] = dense_valid
        if rng.random() >= max(0.0, min(1.0, self.sparse_track_prob)):
            sample["track_density_mode"] = sample.get("track_density_mode", "dense")
            return
        grid_hw = _infer_track_grid_hw(sample, int(tracks.shape[0]))
        query_grid_repeats = 1
        if grid_hw is None:
            source_grid_hw = _infer_source_grid_hw(sample)
            if source_grid_hw is None:
                sample["track_density_mode"] = sample.get("track_density_mode", "dense")
                return
            h0, w0 = source_grid_hw
            per_grid = h0 * w0
            if per_grid <= 0 or int(tracks.shape[0]) % per_grid != 0:
                sample["track_density_mode"] = sample.get("track_density_mode", "dense")
                return
            grid_hw = source_grid_hw
            query_grid_repeats = int(tracks.shape[0]) // per_grid
        h, w = grid_hw
        base_sparse_stride = int(self.sparse_track_strides[int(rng.integers(len(self.sparse_track_strides)))])
        sparse_stride = _adjust_sparse_stride_for_query_budget(
            base_sparse_stride,
            query_grid_repeats,
            self.sparse_track_query_stride_mode,
        )
        if sparse_stride <= 1:
            sample["track_density_mode"] = "dense"
            sample["track_sparse_stride"] = 1
            sample["track_sparse_base_stride"] = base_sparse_stride
            sample["track_sparse_stride_choices"] = np.asarray(self.sparse_track_strides, dtype=np.int32)
            sample["track_sparse_query_stride_mode"] = self.sparse_track_query_stride_mode
            return
        priority_masks = _dynamic_track_priority_masks_for_query_grids(
            sample, h, w, query_grid_repeats,
        )
        sparse_valid = np.concatenate(
            [
                _build_sparse_track_valid_mask(
                    h, w, int(tracks.shape[1]), stride=sparse_stride, rng=rng,
                    block_jitter_prob=self.sparse_track_block_jitter_prob,
                    block_size=self.sparse_track_block_size,
                    priority_mask=priority_masks[min(i, len(priority_masks) - 1)],
                )
                for i in range(query_grid_repeats)
            ],
            axis=0,
        )
        if isinstance(explicit_valid, np.ndarray):
            sparse_valid &= explicit_valid.astype(bool, copy=False)
        sparse_valid &= _valid_track_mask_from_tracks(tracks)
        sample["tracks_world_valid_mask"] = sparse_valid.astype(bool, copy=False)
        sample["track_density_mode"] = "sparse"
        sample["track_sparse_stride"] = sparse_stride
        sample["track_sparse_base_stride"] = base_sparse_stride
        sample["track_sparse_stride_choices"] = np.asarray(self.sparse_track_strides, dtype=np.int32)
        sample["track_sparse_query_stride_mode"] = self.sparse_track_query_stride_mode
        sample["track_sparse_block_jitter_prob"] = self.sparse_track_block_jitter_prob
        dynamic_priority = np.concatenate(
            [
                np.zeros((h * w,), dtype=bool) if p is None else np.asarray(p).astype(bool).reshape(-1)
                for p in priority_masks
            ],
            axis=0,
        )
        sample["track_dynamic_priority_fraction"] = (
            None if not dynamic_priority.any() else float(dynamic_priority.mean())
        )

    def _maybe_add_geometry_noise(self, sample: dict, rng) -> None:
        if rng.random() >= max(0.0, min(1.0, self.noise_prob)):
            sample["geometry_noise_applied"] = False
            return
        applied = False
        common_scale = float(sample.get("common_scale", 1.0) or 1.0)
        # Dataset coordinates/depth are already divided by common_scale. Treat
        # std knobs as raw-world-ish error magnitudes and map them into the
        # normalized training coordinates. Clamp at 1.0 so small-scale scenes do
        # not receive amplified noise.
        scale_denom = max(common_scale, 1.0)
        track_bias_scale = max(float(self.track_offset_std), 0.0) / scale_denom
        track_jitter_scale = max(float(self.track_temporal_jitter_std), 0.0) / scale_denom
        track_frame_depth_scale = (
            track_jitter_scale * max(float(self.track_frame_depth_jitter_ratio), 0.0)
        )
        depth_offset_scale = max(float(self.depth_offset_std), 0.0) / scale_denom
        tracks = sample.get("tracks_world")
        if (
            isinstance(tracks, np.ndarray)
            and tracks.ndim == 3
            and (track_bias_scale > 0 or track_jitter_scale > 0 or track_frame_depth_scale > 0)
        ):
            valid = sample.get("tracks_world_valid_mask")
            if isinstance(valid, np.ndarray):
                valid_mask = valid.astype(bool, copy=False)
            else:
                valid_mask = _valid_track_mask_from_tracks(tracks)
            noise = np.zeros(tracks.shape, dtype=np.float32)
            axis_weights = np.ones((tracks.shape[2],), dtype=np.float32)
            if tracks.shape[2] >= 3:
                xy_ratio = max(float(self.track_xy_noise_ratio), 0.0)
                axis_weights[:2] = xy_ratio
                axis_weights[2] = 1.0
            if track_bias_scale > 0:
                # Per-track bias simulates persistent depth/scale bias from
                # the 3D backend. CoW+VGGT errors are mostly depth-like, so xy
                # perturbation is intentionally smaller than z.
                bias = rng.normal(
                    loc=0.0,
                    scale=track_bias_scale,
                    size=(tracks.shape[0], 1, tracks.shape[2]),
                ).astype(np.float32)
                bias *= axis_weights.reshape(1, 1, -1)
                noise += bias
            if track_jitter_scale > 0:
                # Temporal jitter: every frame can move differently, with
                # moderate correlation. This matches video tracking/depth
                # backend flicker better than independent white 3D noise.
                eps = rng.normal(
                    loc=0.0,
                    scale=track_jitter_scale,
                    size=tracks.shape,
                ).astype(np.float32)
                eps *= axis_weights.reshape(1, 1, -1)
                jitter = np.zeros_like(eps)
                jitter[:, 0, :] = eps[:, 0, :]
                alpha = 0.35
                for t in range(1, tracks.shape[1]):
                    jitter[:, t, :] = alpha * jitter[:, t - 1, :] + (1.0 - alpha) * eps[:, t, :]
                noise += jitter
            if tracks.shape[2] >= 3 and track_frame_depth_scale > 0:
                # Frame-coherent z flicker approximates VGGT depth/camera
                # temporal inconsistency after CoW identities are lifted through
                # per-frame depth maps.
                eps = rng.normal(
                    loc=0.0,
                    scale=track_frame_depth_scale,
                    size=(tracks.shape[1],),
                ).astype(np.float32)
                frame_z = np.zeros_like(eps)
                frame_z[0] = eps[0]
                alpha = 0.5
                for t in range(1, tracks.shape[1]):
                    frame_z[t] = alpha * frame_z[t - 1] + (1.0 - alpha) * eps[t]
                noise[:, :, 2] += frame_z.reshape(1, -1)
            tracks = tracks.astype(np.float32, copy=True)
            tracks[valid_mask] += noise[valid_mask]
            sample["tracks_world"] = tracks
            applied = True

        depth_keys = ("source_depth_video", "source_depth")
        for key in depth_keys:
            depth = sample.get(key)
            if not isinstance(depth, np.ndarray):
                continue
            depth = depth.astype(np.float32, copy=True)
            valid = np.isfinite(depth) & (depth > 0)
            if key == "source_depth_video" and isinstance(sample.get("source_depth_valid"), np.ndarray):
                valid &= sample["source_depth_valid"].astype(bool, copy=False)
            if not valid.any():
                sample[key] = depth
                continue
            if self.depth_scale_jitter > 0:
                if depth.ndim >= 3:
                    num_frames = int(depth.shape[0])
                    eps = rng.uniform(
                        -self.depth_scale_jitter,
                        self.depth_scale_jitter,
                        size=(num_frames,),
                    ).astype(np.float32)
                    scale_delta = np.zeros_like(eps)
                    scale_delta[0] = eps[0]
                    alpha = 0.5
                    for t in range(1, num_frames):
                        scale_delta[t] = alpha * scale_delta[t - 1] + (1.0 - alpha) * eps[t]
                    reshape = (num_frames,) + (1,) * (depth.ndim - 1)
                    frame_scale = np.maximum(1.0 + scale_delta.reshape(reshape), 1e-3)
                    depth[valid] = (depth * frame_scale)[valid]
                else:
                    scale = 1.0 + float(rng.uniform(-self.depth_scale_jitter, self.depth_scale_jitter))
                    depth[valid] *= max(scale, 1e-3)
                applied = True
            if depth_offset_scale > 0:
                # VGGT-like depth error is smoother and frame-biased, not a
                # high-frequency salt-and-pepper depth map.
                depth_noise = rng.normal(0.0, depth_offset_scale * 0.25, size=depth.shape).astype(
                    np.float32
                )
                if depth.ndim >= 3:
                    frame_bias = rng.normal(
                        0.0,
                        depth_offset_scale * 0.5,
                        size=(int(depth.shape[0]),),
                    ).astype(np.float32)
                    frame_bias_smooth = np.zeros_like(frame_bias)
                    frame_bias_smooth[0] = frame_bias[0]
                    alpha = 0.5
                    for t in range(1, frame_bias.shape[0]):
                        frame_bias_smooth[t] = (
                            alpha * frame_bias_smooth[t - 1] + (1.0 - alpha) * frame_bias[t]
                        )
                    reshape = (int(depth.shape[0]),) + (1,) * (depth.ndim - 1)
                    depth_noise += frame_bias_smooth.reshape(reshape)
                depth[valid] = np.maximum(depth[valid] + depth_noise[valid], 0.0)
                applied = True
            sample[key] = depth
        rgb = sample.get("source_rgb_video")
        rgb_noise_std = max(float(self.source_rgb_noise_std), 0.0)
        rgb_frame_std = max(float(self.source_rgb_frame_jitter_std), 0.0)
        if isinstance(rgb, np.ndarray) and rgb.ndim == 4 and (rgb_noise_std > 0 or rgb_frame_std > 0):
            rgb_aug = rgb.astype(np.float32, copy=True)
            if (float(np.max(rgb_aug)) if rgb_aug.size else 0.0) > 2.0:
                rgb_aug /= 255.0
            if rgb_frame_std > 0:
                num_frames = int(rgb_aug.shape[0])
                gain_delta = rng.normal(0.0, rgb_frame_std, size=(num_frames, 1, 1, 3)).astype(np.float32)
                bias_delta = rng.normal(0.0, rgb_frame_std * 0.5, size=(num_frames, 1, 1, 3)).astype(np.float32)
                gain_smooth = np.zeros_like(gain_delta)
                bias_smooth = np.zeros_like(bias_delta)
                gain_smooth[0] = gain_delta[0]
                bias_smooth[0] = bias_delta[0]
                alpha = 0.5
                for t in range(1, num_frames):
                    gain_smooth[t] = alpha * gain_smooth[t - 1] + (1.0 - alpha) * gain_delta[t]
                    bias_smooth[t] = alpha * bias_smooth[t - 1] + (1.0 - alpha) * bias_delta[t]
                rgb_aug = rgb_aug * np.maximum(1.0 + gain_smooth, 0.0) + bias_smooth
                applied = True
            if rgb_noise_std > 0:
                rgb_aug += rng.normal(0.0, rgb_noise_std, size=rgb_aug.shape).astype(np.float32)
                applied = True
            sample["source_rgb_video"] = np.clip(rgb_aug, 0.0, 1.0).astype(np.float32, copy=False)
        sample["geometry_noise_applied"] = bool(applied)
        sample["geometry_noise_common_scale"] = common_scale
        sample["geometry_noise_track_bias_scale"] = track_bias_scale
        sample["geometry_noise_track_temporal_jitter_scale"] = track_jitter_scale
        sample["geometry_noise_track_xy_ratio"] = max(float(self.track_xy_noise_ratio), 0.0)
        sample["geometry_noise_track_frame_depth_scale"] = track_frame_depth_scale
        sample["geometry_noise_depth_offset_scale"] = depth_offset_scale
        sample["geometry_noise_source_rgb_noise_std"] = rgb_noise_std
        sample["geometry_noise_source_rgb_frame_jitter_std"] = rgb_frame_std

    def _maybe_augment_dynamic_masks(self, sample: dict, rng) -> None:
        if rng.random() >= max(0.0, min(1.0, self.dynamic_mask_aug_prob)):
            sample["dynamic_mask_aug_applied"] = False
            return
        applied = False
        for key in ("source_dynamic_mask", "target_dynamic_mask"):
            mask = sample.get(key)
            if not isinstance(mask, np.ndarray) or mask.ndim != 3:
                continue
            aug = mask.astype(bool, copy=True)
            k = max(int(self.dynamic_mask_morph_kernel), 0)
            if k >= 2:
                if k % 2 == 0:
                    k += 1
                kernel = np.ones((k, k), dtype=np.uint8)
                op = cv2.MORPH_DILATE if rng.random() < 0.5 else cv2.MORPH_ERODE
                aug = np.stack(
                    [
                        cv2.morphologyEx(
                            frame.astype(np.uint8),
                            op,
                            kernel,
                            iterations=1,
                        ).astype(bool, copy=False)
                        for frame in aug
                    ],
                    axis=0,
                )
                applied = True
            flip_prob = max(0.0, min(1.0, self.dynamic_mask_flip_prob))
            if flip_prob > 0:
                flips = rng.random(aug.shape) < flip_prob
                aug = np.logical_xor(aug, flips)
                applied = True
            sample[key] = aug.astype(bool, copy=False)
        sample["dynamic_mask_aug_applied"] = bool(applied)


def normalize_to_cam0(tracks_world, tgt_cam2world, src_cam0_pose, common_scale=None):
    """Transform tracks and cameras to frame-0 source camera space + common_scale.

    Identical to Syn4D_Multi_NVS._normalize_to_cam0.
    """
    world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)
    R = world_to_cam0[:3, :3]
    t = world_to_cam0[:3, 3]

    N, T, _ = tracks_world.shape
    tracks_cam0 = np.einsum("ij,ntj->nti", R, tracks_world) + t[None, None, :]
    tgt_cam2cam0 = world_to_cam0[None, :, :] @ tgt_cam2world

    if common_scale is None:
        valid_mask = np.any(tracks_cam0 != 0, axis=-1)
        valid_pts = tracks_cam0[valid_mask]
        if valid_pts.shape[0] > 0:
            common_scale = float(np.linalg.norm(valid_pts, axis=-1).mean())
        else:
            common_scale = 1.0
    else:
        common_scale = float(common_scale)
    common_scale = max(common_scale, 1e-6)

    tracks_cam0 = tracks_cam0 / common_scale
    tgt_cam2cam0[:, :3, 3] /= common_scale

    return tracks_cam0.astype(np.float32), tgt_cam2cam0.astype(np.float32), common_scale


def normalize_cameras_to_cam0(tgt_cam2world, src_cam0_pose):
    """Normalize cameras only (no tracks). Returns common_scale from camera baselines."""
    world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)
    tgt_cam2cam0 = world_to_cam0[None, :, :] @ tgt_cam2world

    dists = np.linalg.norm(tgt_cam2cam0[:, :3, 3], axis=-1)
    common_scale = float(np.mean(dists[dists > 1e-6])) if np.any(dists > 1e-6) else 1.0
    common_scale = max(common_scale, 1e-6)

    tgt_cam2cam0[:, :3, 3] /= common_scale
    return tgt_cam2cam0.astype(np.float32), common_scale


def _recam_cow_all10_rel_frames_tag(value: str) -> str:
    raw = str(value).strip()
    if ":" in raw:
        parts = [int(p) for p in raw.split(":")]
        if len(parts) == 2:
            frames = list(range(parts[0], parts[1], 1))
        elif len(parts) == 3:
            frames = list(range(parts[0], parts[1], parts[2]))
        else:
            raise ValueError(f"bad rel-frame range: {value}")
    else:
        frames = [int(p) for p in re.split(r"[,| ]+", raw) if p]
    if not frames:
        raise ValueError("empty ReCam all10 rel-frame list")
    if len(frames) >= 2:
        diffs = [b - a for a, b in zip(frames[:-1], frames[1:])]
        if len(set(diffs)) == 1:
            return f"rel{frames[0]:03d}-{frames[-1]:03d}s{diffs[0]}"
    return "rel" + "-".join(f"{f:03d}" for f in frames)


def _scale_intrinsics_to_resolution(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    K_out = np.asarray(K, dtype=np.float32).copy()
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    K_out[0, 0] *= sx
    K_out[0, 2] *= sx
    K_out[1, 1] *= sy
    K_out[1, 2] *= sy
    return K_out.astype(np.float32)


def depth_to_world_pts3d(depthmap, intrinsics, cam2world):
    """Unproject z-depth map to world-space (H,W,3) pointmap."""
    H, W = depthmap.shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depthmap.astype(np.float32)
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)

    valid = np.isfinite(z) & (z > 1e-6)

    R = cam2world[:3, :3]
    t = cam2world[:3, 3]
    pts_world = np.einsum("ij,hwj->hwi", R, pts_cam) + t[None, None, :]
    pts_world[~valid] = 0.0

    return pts_world.astype(np.float32), valid


def center_crop_and_resize(image_np, depthmap, intrinsics, target_w, target_h):
    """Center-crop to target aspect ratio, then resize.

    Returns: (image_pil, depthmap_or_None, intrinsics_adjusted)
    """
    H, W = image_np.shape[:2]
    target_aspect = target_w / target_h
    src_aspect = W / H
    K = intrinsics.copy()

    if src_aspect > target_aspect:
        new_W = int(round(H * target_aspect))
        left = (W - new_W) // 2
        image_np = image_np[:, left:left + new_W]
        if depthmap is not None:
            depthmap = depthmap[:, left:left + new_W]
        K[0, 2] -= left
    else:
        new_H = int(round(W / target_aspect))
        top = (H - new_H) // 2
        image_np = image_np[top:top + new_H]
        if depthmap is not None:
            depthmap = depthmap[top:top + new_H]
        K[1, 2] -= top

    crop_H, crop_W = image_np.shape[:2]
    scale_x = target_w / crop_W
    scale_y = target_h / crop_H

    image_pil = PIL.Image.fromarray(image_np)
    image_pil = image_pil.resize((target_w, target_h), PIL.Image.LANCZOS)

    if depthmap is not None:
        depthmap = cv2.resize(depthmap, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y

    return image_pil, depthmap, K.astype(np.float32)


def center_crop_and_resize_mask(mask_np, target_w, target_h):
    """Center-crop a mask to the target aspect ratio, then resize with NN."""
    resized = _center_crop_and_resize_np(
        mask_np,
        target_w=target_w,
        target_h=target_h,
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool, copy=False)


def _center_crop_and_resize_np(array_np, target_w, target_h, interpolation):
    """Center-crop an HxW array to target aspect ratio, then resize.

    Mirrors `center_crop_and_resize` without modifying camera intrinsics.
    Kubric segmentation and object-coordinate PNGs must use the same crop as
    RGB/depth before they are sampled on the output track grid.
    """
    array_np = np.asarray(array_np)
    H, W = array_np.shape[:2]
    target_aspect = target_w / target_h
    src_aspect = W / H

    if src_aspect > target_aspect:
        new_W = int(round(H * target_aspect))
        left = (W - new_W) // 2
        array_np = array_np[:, left:left + new_W]
    else:
        new_H = int(round(W / target_aspect))
        top = (H - new_H) // 2
        array_np = array_np[top:top + new_H]

    if array_np.dtype == np.bool_:
        array_np = array_np.astype(np.uint8)

    return cv2.resize(array_np, (target_w, target_h), interpolation=interpolation)


def quaternion_wxyz_to_rotation_matrix(quat_wxyz):
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat_wxyz)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quat_wxyz / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


BLENDER_TO_OPENCV = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float32)


def _split_contiguous(frame_ids, max_gap=1):
    """Split list of ints into runs where consecutive diff <= max_gap."""
    if not frame_ids:
        return []
    blocks, cur = [], [frame_ids[0]]
    for prev, nxt in zip(frame_ids, frame_ids[1:]):
        if nxt - prev <= max_gap:
            cur.append(nxt)
        else:
            blocks.append(cur)
            cur = [nxt]
    blocks.append(cur)
    return blocks


# ======================================================================
# Kubric_NVS — Multi-camera Kubric (49/81 frames, 4 cameras)
# ======================================================================

class Kubric_NVS(EasyDataset):
    """Multi-camera Kubric NVS dataset for Track4D-360.

    Uses preprocessed multi-camera Kubric data (movi_g_mv) with 4 cameras.
    Picks two different cameras as source/target for genuine multi-view NVS.
    3D tracks computed from depth + bbox-based object tracking.

    Data format:
        ROOT/
            train/  (or validation/)
                00001/
                    camera_0/ camera_1/ camera_2/ camera_3/
                        rgb_00.jpg ... rgb_48.jpg
                        depth_00.png ... depth_48.png
                        coords_00.png ... segm_00.png
            seq_data.train.npz  (cam_positions, cam_quaternions, focal_length, ...)
    """

    def __init__(
        self,
        ROOT,
        num_views=49,
        resolution=(512, 288),
        steps_per_epoch=2000,
        split="train",
        track_spatial_stride=1,
        num_cameras=4,
        reverse_prob=0.5,
        return_target_tracks=False,
        return_target_depth=False,
        return_source_depth_video=False,
        return_source_rgb_video=False,
        return_source_cameras=False,
        return_dynamic_masks=False,
        fixed_start_frame=None,
        fixed_scene_idx=None,
        camera_pairs=None,
        caption_cache_dir=None,  # if set: dir holding captions_all.json +
                                 # captions_t5_emb.safetensors built by
                                 # holo4d.precompute_t5_cache_kubric
        multi_query_prob=0.0,
        multi_query_choices="1",
        seed=None,
    ):
        self.ROOT = ROOT
        self.caption_cache_dir = (
            None if caption_cache_dir in (None, "") else str(caption_cache_dir)
        )
        self._num_views = int(num_views)
        self.steps_per_epoch = int(steps_per_epoch)
        self.split = split
        self.track_spatial_stride = int(track_spatial_stride)
        self.num_cameras = int(num_cameras)
        self.reverse_prob = float(reverse_prob)
        # Track4D training may auto-inject these kwargs for dynamic-GS runs.
        # Kubric_NVS already emits the available target-side tensors directly,
        # so keep the flags as compatibility no-ops instead of rejecting the
        # constructor call.
        self.return_target_tracks = bool(return_target_tracks)
        self.return_target_depth = bool(return_target_depth)
        self.return_source_depth_video = bool(return_source_depth_video)
        self.return_source_rgb_video = bool(return_source_rgb_video)
        self.return_source_cameras = bool(return_source_cameras)
        self.return_dynamic_masks = bool(return_dynamic_masks)
        self.multi_query_prob = float(multi_query_prob)
        self.multi_query_choices = _parse_int_choices(multi_query_choices, default=(1,))
        self.seed = None if seed is None else int(seed)
        self.fixed_start_frame = (
            None if fixed_start_frame is None else int(fixed_start_frame)
        )
        self.fixed_scene_idx = (
            None if fixed_scene_idx is None else int(fixed_scene_idx)
        )
        self.camera_pairs = self._parse_camera_pairs(camera_pairs)

        if isinstance(resolution, int):
            self.output_width = self.output_height = resolution
        elif isinstance(resolution, (tuple, list)):
            self.output_width, self.output_height = int(resolution[0]), int(resolution[1])
        else:
            self.output_width = self.output_height = int(resolution)

        # Load metadata
        seq_data_path = osp.join(ROOT, f"seq_data.{split}.npz")
        if not osp.exists(seq_data_path):
            raise FileNotFoundError(f"Kubric seq_data not found: {seq_data_path}")
        seq_data = np.load(seq_data_path)

        self.scene_idxs = seq_data["scene_idx"]
        self.camera_idxs = seq_data["camera_idx"]
        self.cam_positions = seq_data["cam_positions"]      # (N, T, 3)
        self.cam_quaternions = seq_data["cam_quaternions"]  # (N, T, 4)
        self.focal_length = seq_data["focal_length"]        # (N,)
        self.min_max_depth = seq_data["min_max_depth"]      # (N, 2)
        self.bboxes = seq_data["bboxes"]                    # (total_inst, T, 8, 3)
        self.bboxes_cumsum = seq_data["bboxes_cumsum"]      # (N+1,)
        self._n_frames = self.cam_positions.shape[1]

        # Build multi-camera scene groups: {scene_idx -> {cam_idx -> meta_idx}}
        self._scene_groups = {}
        for meta_idx in range(len(self.scene_idxs)):
            si = int(self.scene_idxs[meta_idx])
            ci = int(self.camera_idxs[meta_idx])
            if si not in self._scene_groups:
                self._scene_groups[si] = {}
            self._scene_groups[si][ci] = meta_idx

        # Keep only scenes with >= 2 cameras
        self._valid_scenes = [
            (si, cams) for si, cams in self._scene_groups.items()
            if len(cams) >= 2
        ]

        if self.fixed_scene_idx is not None:
            self._valid_scenes = [
                (si, cams) for si, cams in self._valid_scenes
                if int(si) == self.fixed_scene_idx
            ]
            if not self._valid_scenes:
                raise ValueError(
                    f"Kubric_NVS: fixed_scene_idx={self.fixed_scene_idx} not found in {ROOT}/{split}"
                )

        if not self._valid_scenes:
            raise ValueError(f"No multi-camera Kubric scenes in {ROOT}/{split}")

        # When caption_cache_dir is set, restrict _valid_scenes to (scene, cam)
        # pairs that have a precomputed T5 emb in the cache. Some on-disk
        # scenes lack caption.txt entirely (e.g. partial Kubric renders), and
        # the precompute correctly skips those — the dataset must skip them
        # too, or _load_camera_caption will hard-assert when the random
        # sampler picks one. Drops scenes that lose >=cams_required cameras
        # under the filter.
        if self.caption_cache_dir:
            cache = self._ensure_t5_emb_cache()
            if cache:
                covered = set(cache.keys())  # {"02000/camera_0", ...}
                filtered: list[tuple[int, dict[int, int]]] = []
                dropped_scenes = 0
                dropped_cams = 0
                for si, cams in self._valid_scenes:
                    scene_name = f"{int(si):05d}"
                    kept = {
                        ci: meta for ci, meta in cams.items()
                        if f"{scene_name}/camera_{int(ci)}" in covered
                    }
                    dropped_cams += len(cams) - len(kept)
                    if len(kept) >= 2:
                        filtered.append((si, kept))
                    else:
                        dropped_scenes += 1
                print(f"[Kubric_NVS] caption-cache filter: kept "
                      f"{len(filtered)}/{len(self._valid_scenes)} scenes "
                      f"(dropped {dropped_scenes} scenes with <2 captioned cams, "
                      f"{dropped_cams} cams missing captions). "
                      f"cache_dir={self.caption_cache_dir}")
                self._valid_scenes = filtered
                if not self._valid_scenes:
                    raise ValueError(
                        f"Kubric_NVS: every scene under {ROOT}/{split} got "
                        f"filtered out by the caption cache. cache_dir="
                        f"{self.caption_cache_dir}. Re-run "
                        f"holo4d.precompute_t5_cache_kubric or unset "
                        f"caption_cache_dir to disable the filter."
                    )

        total_cams = sum(len(cams) for _, cams in self._valid_scenes)
        print(f"[Kubric_NVS] {len(self._valid_scenes)} multi-cam scenes, "
              f"{total_cams} total cameras, {self._n_frames} frames/cam, "
              f"resolution={self.output_width}x{self.output_height}")

    @staticmethod
    def _parse_camera_pairs(camera_pairs):
        if camera_pairs in (None, ""):
            return None
        if isinstance(camera_pairs, str):
            parsed = []
            for token in camera_pairs.split(","):
                token = token.strip()
                if not token:
                    continue
                if "-" not in token:
                    raise ValueError(
                        f"Kubric_NVS camera_pairs token must look like '0-1', got {token!r}"
                    )
                src, tgt = token.split("-", 1)
                parsed.append((int(src), int(tgt)))
            return parsed or None

        parsed = []
        for pair in camera_pairs:
            if len(pair) != 2:
                raise ValueError(f"Kubric_NVS camera pair must have length 2, got {pair!r}")
            parsed.append((int(pair[0]), int(pair[1])))
        return parsed or None

    @property
    def _resolutions(self):
        return [(self.output_width, self.output_height)]

    @property
    def num_views(self):
        return self._num_views

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        pass

    @staticmethod
    def _extract_source_rgb_video(src_frames):
        return np.stack(
            [np.array(frame, dtype=np.float32) / 255.0 for frame in src_frames],
            axis=0,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _extract_source_cam2cam0(src_poses, src_cam0_pose, common_scale):
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)
        src_c2w_world = np.stack([pose.astype(np.float32) for pose in src_poses], axis=0)
        src_c2cam0 = world_to_cam0[None, :, :] @ src_c2w_world
        src_c2cam0 = src_c2cam0.copy()
        src_c2cam0[:, :3, 3] /= max(float(common_scale), 1e-6)
        return src_c2cam0.astype(np.float32, copy=False)

    @staticmethod
    def _extract_source_intrinsics(src_K):
        return src_K.astype(np.float32, copy=False)

    def _attach_source_dense_fields(
        self,
        result,
        src_frames,
        src_poses,
        src_K,
        src_depths,
        src_cam0_pose,
        common_scale,
    ):
        if self.return_source_rgb_video:
            result["source_rgb_video"] = self._extract_source_rgb_video(src_frames)
        if self.return_source_cameras:
            result["source_cam2world"] = self._extract_source_cam2cam0(
                src_poses, src_cam0_pose, common_scale,
            )
            result["source_intrinsics"] = self._extract_source_intrinsics(src_K)
        if self.return_source_depth_video and "source_depth_video" not in result and src_depths:
            result["source_depth_video"] = (
                np.stack(src_depths, axis=0).astype(np.float32)
                / max(float(common_scale), 1e-6)
            )

    @staticmethod
    def _normalize_tracks_to_cam0(tracks_world, src_cam0_pose, common_scale):
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)
        R = world_to_cam0[:3, :3]
        t = world_to_cam0[:3, 3]
        tracks_cam0 = np.einsum("ij,ntj->nti", R, tracks_world) + t[None, None, :]
        tracks_cam0 = tracks_cam0 / max(float(common_scale), 1e-6)
        return tracks_cam0.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_pointmap_to_cam0(pts_world, src_cam0_pose, common_scale):
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)
        R = world_to_cam0[:3, :3]
        t = world_to_cam0[:3, 3]
        pts_cam0 = np.einsum("ij,hwj->hwi", R, pts_world) + t[None, None, :]
        pts_cam0 = pts_cam0 / max(float(common_scale), 1e-6)
        return pts_cam0.astype(np.float32, copy=False)

    def _build_target_track_video_supervision(
        self,
        tgt_depths,
        tgt_poses,
        tgt_K,
        tgt_fg_masks,
        src_cam0_pose,
        common_scale,
    ):
        target_track_video = []
        target_track_valid_mask = []
        target_depth_valid_mask = []

        for depth_t, pose_t, fg_mask_t in zip(tgt_depths, tgt_poses, tgt_fg_masks):
            pts_world_t, valid_depth_t = depth_to_world_pts3d(depth_t, tgt_K, pose_t)
            valid_t = fg_mask_t.astype(bool) & valid_depth_t.astype(bool)
            pts_cam0_t = self._normalize_pointmap_to_cam0(
                pts_world_t, src_cam0_pose, common_scale,
            )
            pts_cam0_t = pts_cam0_t.copy()
            pts_cam0_t[~valid_t] = 0.0

            target_track_video.append(pts_cam0_t)
            target_track_valid_mask.append(valid_t.astype(bool, copy=False))
            target_depth_valid_mask.append(valid_t.astype(bool, copy=False))

        return (
            np.stack(target_track_video, axis=0).astype(np.float32, copy=False),
            np.stack(target_track_valid_mask, axis=0).astype(bool, copy=False),
            np.stack(target_depth_valid_mask, axis=0).astype(bool, copy=False),
        )

    def _ensure_t5_emb_cache(self):
        """Lazy-load the precomputed T5 emb cache (built by
        holo4d.precompute_t5_cache_kubric). Returns dict {key: bf16 emb}
        or empty dict if no cache. Cache dir layout:
            <caption_cache_dir>/captions_all.json   (key -> sanitized text)
            <caption_cache_dir>/captions_t5_emb.safetensors  (key -> [L, dim] bf16)
        Key format: f"{scene_idx:05d}/camera_{cam_id}" — must match what the
        precompute script wrote.
        """
        if not self.caption_cache_dir:
            return {}
        cache = getattr(self, "_caption_emb_cache", None)
        if cache is not None:
            return cache
        sft_path = osp.join(self.caption_cache_dir, "captions_t5_emb.safetensors")
        # Multi-shard fallback: when the precompute step skipped the all-in-memory
        # merge (KEEP_SHARDS=1, default for large datasets like ReCamMaster /
        # MultiCamVideo), the cache dir holds captions_t5_emb.rank*.part*.safetensors
        # instead of one merged file. Load and concatenate them.
        import glob as _glob
        shard_paths = sorted(_glob.glob(
            osp.join(self.caption_cache_dir, "captions_t5_emb.rank*.part*.safetensors")
        ))
        if not osp.isfile(sft_path) and not shard_paths:
            self._caption_emb_cache = {}
            return self._caption_emb_cache
        try:
            from safetensors.torch import load_file as _safetensors_load
            if osp.isfile(sft_path):
                self._caption_emb_cache = _safetensors_load(sft_path)
            else:
                merged: dict = {}
                for sp in shard_paths:
                    merged.update(_safetensors_load(sp))
                self._caption_emb_cache = merged
            should_log = os.environ.get("TRACK4D_VERBOSE_T5_CACHE", "0") == "1"
            try:
                from torch.utils.data import get_worker_info
                worker_info = get_worker_info()
                worker_id = None if worker_info is None else worker_info.id
            except Exception:
                worker_id = None
            if should_log or (os.environ.get("LOCAL_RANK", "0") == "0" and worker_id in (None, 0)):
                print(f"[Kubric_NVS] Loaded T5 emb cache: {sft_path} "
                      f"({len(self._caption_emb_cache)} keys)")
        except Exception as e:
            print(f"[Kubric_NVS] WARN: failed to load T5 emb cache {sft_path}: {e}")
            self._caption_emb_cache = {}
        return self._caption_emb_cache

    def _ensure_caption_text_table(self):
        """Lazy-load captions_all.json (key -> sanitized text) sibling to the
        T5 emb safetensors. Lets us serve captions directly from the cache
        dir without touching the on-disk caption.txt files at training time.
        Useful when (a) the data tree is on a read-only mount, (b) on-disk
        caption.txt files are missing or stale, (c) we want byte-exact
        consistency with what the T5 embeddings were computed from.
        """
        if not self.caption_cache_dir:
            return {}
        cached = getattr(self, "_caption_text_table", None)
        if cached is not None:
            return cached
        json_path = osp.join(self.caption_cache_dir, "captions_all.json")
        if not osp.isfile(json_path):
            self._caption_text_table = {}
            return self._caption_text_table
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self._caption_text_table = json.load(f)
        except Exception as e:
            print(f"[Kubric_NVS] WARN: failed to load caption text table {json_path}: {e}")
            self._caption_text_table = {}
        return self._caption_text_table

    def _load_camera_caption(self, scene_idx, cam_id):
        """Return (text, prompt_emb_or_None) for the given (scene, cam).

        - Reads per-camera ``caption.txt`` (Tarsier-format) under
          ``${ROOT}/${split}/${scene_idx:05d}/camera_${cam_id}/``.
        - If a precomputed T5 emb cache is available
          (``--caption_cache_dir`` set, built by
          ``holo4d.precompute_t5_cache_kubric``), the matching emb tensor
          is returned alongside the text — the pipeline's prompt-embedder unit
          then skips online T5 encode for these samples.
        - Falls back to DEFAULT_PROMPT (and emb=None) when caption.txt is
          missing or sanitizes empty.
        - Cached per (scene, cam) since train/eval iterate the same scene
          many times.

        Pre-2026-04-29 training and eval BOTH hardcoded DEFAULT_PROMPT and
        ignored these files; ismb288_3k was trained without them.
        """
        cache = getattr(self, "_caption_text_cache", None)
        if cache is None:
            cache = {}
            self._caption_text_cache = cache
        key = (int(scene_idx), int(cam_id))
        if key in cache:
            return cache[key]
        scene_name = f"{int(scene_idx):05d}"
        emb_key = f"{scene_name}/camera_{int(cam_id)}"

        # Caption text + emb both come from the precomputed cache_dir. We do
        # NOT touch the on-disk caption.txt at training time — that lets the
        # caption cache and the data tree live on different mounts (e.g. the
        # data on a read-only u6cl share, the cache on a writable u6ew share)
        # AND guarantees byte-exact agreement between the prompt text and the
        # T5 embedding (both produced by precompute_t5_cache_kubric from the
        # same caption.txt at cache-build time).
        emb_cache = self._ensure_t5_emb_cache()
        assert emb_cache, (
            "Kubric_NVS: caption_cache_dir is not set or the cache file is "
            "missing. With T5 deletion enabled (HOLO4D_OFFLOAD_T5_TO_CPU=1, the "
            "default), every kubric prompt must be served from a precomputed "
            "embedding. Build it:\n"
            "  torchrun --nproc_per_node=2 "
            "-m holo4d.precompute_t5_cache_kubric "
            f"--kubric_root <on-disk-data-tree> --split {self.split} "
            "--output_dir <where> \n"
            "then pass caption_cache_dir=<where> to Kubric_NVS."
        )
        emb = emb_cache.get(emb_key)
        assert emb is not None, (
            f"Kubric_NVS: T5 emb cache loaded but missing key {emb_key!r} "
            f"(cache has {len(emb_cache)} keys). Re-run "
            f"holo4d.precompute_t5_cache_kubric to refresh: this scene "
            f"was likely added after the cache was built. The init-time "
            f"caption-cache filter on _valid_scenes should have prevented "
            f"this from being sampled — file a bug if you hit this."
        )
        # Resolve the prompt text from captions_all.json (sibling to the
        # safetensors emb cache). The text is what was sanitized by
        # precompute_t5_cache_kubric and fed into T5 — using it here keeps
        # the in-memory `prompt` field consistent with what produced `emb`.
        # Falls back to DEFAULT_PROMPT (only used for logging / drop-text-prob;
        # the model-conditioning path goes through `emb` which is correct).
        text_table = self._ensure_caption_text_table()
        text = text_table.get(emb_key, DEFAULT_PROMPT)
        cache[key] = (text, emb)
        return cache[key]

    def _load_camera_fg_masks(self, scene_idx, cam_id, frame_ids):
        scene_name = f"{scene_idx:05d}"
        camera_dir = osp.join(self.ROOT, self.split, scene_name, f"camera_{cam_id}")

        masks = []
        for fi in frame_ids:
            segm_path = osp.join(camera_dir, f"segm_{fi:02d}.png")
            segm = cv2.imread(segm_path, cv2.IMREAD_UNCHANGED)
            if segm is None:
                raise FileNotFoundError(f"Segmentation not found: {segm_path}")
            if segm.ndim == 3:
                segm = segm[:, :, 0]
            fg_mask = center_crop_and_resize_mask(
                segm > 0,
                self.output_width,
                self.output_height,
            )
            masks.append(fg_mask)
        return masks

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Kubric_NVS: failed after {max_retries} retries: {e}"
                    ) from e

    def _getitem_impl(self, idx=0):
        rng = np.random.default_rng(None if self.seed is None else self.seed + int(idx))
        T = self._num_views
        n_frames = self._n_frames

        # 1. Pick random multi-camera scene
        scene_idx, cam_dict = self._valid_scenes[rng.integers(0, len(self._valid_scenes))]
        cam_ids = list(cam_dict.keys())

        # 2. Pick two different cameras (source, target)
        if self.camera_pairs:
            allowed_pairs = [
                (int(src), int(tgt))
                for src, tgt in self.camera_pairs
                if int(src) in cam_dict and int(tgt) in cam_dict and int(src) != int(tgt)
            ]
            if not allowed_pairs:
                raise ValueError(
                    f"Kubric_NVS: scene {scene_idx} has cameras {sorted(cam_ids)} but none of "
                    f"camera_pairs={self.camera_pairs} are available"
                )
            pair_idx = int(rng.integers(0, len(allowed_pairs)))
            src_cam_id, tgt_cam_id = allowed_pairs[pair_idx]
        else:
            pair = rng.choice(cam_ids, size=2, replace=False)
            src_cam_id, tgt_cam_id = int(pair[0]), int(pair[1])
        src_meta = cam_dict[src_cam_id]
        tgt_meta = cam_dict[tgt_cam_id]

        # 3. Frame selection: sequential, first T frames (or random start)
        max_start = max(0, n_frames - T)
        if self.fixed_start_frame is None:
            start = rng.integers(0, max_start + 1) if max_start > 0 else 0
        else:
            start = min(max(int(self.fixed_start_frame), 0), max_start)
        frame_ids = list(range(start, min(start + T, n_frames)))
        while len(frame_ids) < T:
            frame_ids.append(frame_ids[-1])

        # Optional reversal
        if random.random() < self.reverse_prob:
            frame_ids = list(reversed(frame_ids))

        if rng.random() < max(0.0, min(1.0, self.multi_query_prob)):
            q_count = int(self.multi_query_choices[int(rng.integers(len(self.multi_query_choices)))])
        else:
            q_count = 1
        track_query_indices = _choose_track_query_indices(len(frame_ids), q_count)
        query_frame_id = frame_ids[track_query_indices[0]]

        # 4. Load source camera data
        src_frames, src_depths, src_poses, src_K = self._load_camera_frames(
            scene_idx, src_cam_id, src_meta, frame_ids, load_depth=True)

        # 5. Load target camera data (RGB + depth + poses)
        tgt_frames, tgt_depths, tgt_poses, tgt_K = self._load_camera_frames(
            scene_idx, tgt_cam_id, tgt_meta, frame_ids, load_depth=True)

        # 6. Construct 3D tracks from source depth + bbox tracking
        H, W = self.output_height, self.output_width
        tracks_world_chunks = [
            self._compute_tracks(
                scene_idx, src_cam_id, src_meta, frame_ids, int(frame_ids[q_idx]),
                src_depths, src_poses, src_K, H, W,
            )
            for q_idx in track_query_indices
        ]
        tracks_world = np.concatenate(tracks_world_chunks, axis=0)

        # 7. Normalize
        tgt_cam2world = np.stack(tgt_poses, axis=0)
        src_cam0_pose = src_poses[0]
        if len(tracks_world_chunks) > 1:
            _base_tracks_cam0, _base_tgt_cam2cam0, common_scale = normalize_to_cam0(
                tracks_world_chunks[0], tgt_cam2world, src_cam0_pose,
            )
            tracks_cam0, tgt_cam2cam0, common_scale = normalize_to_cam0(
                tracks_world, tgt_cam2world, src_cam0_pose, common_scale=common_scale,
            )
        else:
            tracks_cam0, tgt_cam2cam0, common_scale = normalize_to_cam0(
                tracks_world, tgt_cam2world, src_cam0_pose,
            )

        # Build target depth array (T, H, W), normalized by common_scale
        if tgt_depths is not None and len(tgt_depths) == len(frame_ids):
            target_depth = np.stack(tgt_depths, axis=0).astype(np.float32) / common_scale
        else:
            target_depth = None
        has_depth = target_depth is not None

        target_track_video = None
        target_track_valid_mask = None
        target_depth_valid_mask = None
        target_tracks_world = None
        src_fg_masks = None
        tgt_fg_masks = None
        if getattr(self, "return_dynamic_masks", False):
            src_fg_masks = self._load_camera_fg_masks(scene_idx, src_cam_id, frame_ids)
            tgt_fg_masks = self._load_camera_fg_masks(scene_idx, tgt_cam_id, frame_ids)

        need_target_supervision = bool(
            getattr(self, "return_target_tracks", False)
            or getattr(self, "return_target_depth", False)
        )
        if need_target_supervision and has_depth:
            if tgt_fg_masks is None:
                tgt_fg_masks = self._load_camera_fg_masks(scene_idx, tgt_cam_id, frame_ids)
            (
                target_track_video,
                target_track_valid_mask,
                target_depth_valid_mask,
            ) = self._build_target_track_video_supervision(
                tgt_depths=tgt_depths,
                tgt_poses=tgt_poses,
                tgt_K=tgt_K,
                tgt_fg_masks=tgt_fg_masks,
                src_cam0_pose=src_cam0_pose,
                common_scale=common_scale,
            )
            if getattr(self, "return_target_tracks", False):
                target_tracks_world_raw = self._compute_tracks(
                    scene_idx, tgt_cam_id, tgt_meta, frame_ids, query_frame_id,
                    tgt_depths, tgt_poses, tgt_K, H, W,
                )
                target_tracks_world = self._normalize_tracks_to_cam0(
                    target_tracks_world_raw, src_cam0_pose, common_scale,
                )

        result = {
            "video": src_frames,
            "target_video": tgt_frames,
            "dataset_index": int(idx),
            "scene_idx": int(scene_idx),
            "scene_id": f"{int(scene_idx):05d}",
            "source_camera_id": int(src_cam_id),
            "target_camera_id": int(tgt_cam_id),
            "frame_ids": [int(frame_id) for frame_id in frame_ids],
            "tracks_world": tracks_cam0,
            "target_cam2world": tgt_cam2cam0,
            "target_intrinsics": tgt_K,
            "common_scale": common_scale,
            "track_query_idx": int(track_query_indices[0]),
            "track_query_indices": [int(q) for q in track_query_indices],
            "track_query_count": len(track_query_indices),
            # Per-camera caption shipped under <scene>/camera_<tgt>/caption.txt
            # (Tarsier-format). Loader sanitizes "Prediction:" extraction and
            # caches per (scene, cam). When --caption_cache_dir points at a
            # precomputed T5 emb safetensors (built by
            # holo4d.precompute_t5_cache_kubric), the matching emb tensor
            # is returned alongside the text and the pipeline's prompt-embedder
            # unit skips online T5 encode for that sample. Falls back to
            # DEFAULT_PROMPT (emb=None) when caption.txt is missing/empty.
            # Pre-2026-04-29 this was hardcoded DEFAULT_PROMPT for both
            # training and eval — the on-disk captions existed but were
            # never read.
            **dict(zip(("prompt", "prompt_emb"),
                       self._load_camera_caption(scene_idx, tgt_cam_id))),
            "has_tracks": True,
            "has_dynamic_tracks": True,
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "target_depth": target_depth,
            "has_depth": has_depth,
            "source_rgb_at_tracks": _source_rgb_at_track_grid(
                src_frames,
                self.track_spatial_stride,
                track_query_indices,
            ),
        }
        if target_track_video is not None:
            result["target_track_video"] = target_track_video
        if target_track_valid_mask is not None:
            result["target_track_valid_mask"] = target_track_valid_mask
        if target_depth_valid_mask is not None:
            result["target_depth_valid_mask"] = target_depth_valid_mask
            result["target_depth_valid"] = target_depth_valid_mask
        if target_tracks_world is not None:
            result["target_tracks_world"] = target_tracks_world
        if src_fg_masks is not None:
            result["source_dynamic_mask"] = np.stack(src_fg_masks, axis=0).astype(bool, copy=False)
        if tgt_fg_masks is not None and getattr(self, "return_dynamic_masks", False):
            result["target_dynamic_mask"] = np.stack(tgt_fg_masks, axis=0).astype(bool, copy=False)
        self._attach_source_dense_fields(
            result=result,
            src_frames=src_frames,
            src_poses=src_poses,
            src_K=src_K,
            src_depths=src_depths,
            src_cam0_pose=src_cam0_pose,
            common_scale=common_scale,
        )
        return result

    def _load_camera_frames(self, scene_idx, cam_id, meta_idx, frame_ids, load_depth=True):
        """Load RGB (+ optional depth) frames for a camera."""
        scene_name = f"{scene_idx:05d}"
        camera_dir = osp.join(self.ROOT, self.split, scene_name, f"camera_{cam_id}")

        focal = float(self.focal_length[meta_idx])
        min_d, max_d = self.min_max_depth[meta_idx]
        uint16_max = np.iinfo(np.uint16).max

        frames_pil = []
        depths = []
        poses = []
        K_out = None

        for fi in frame_ids:
            # RGB
            rgb_path = osp.join(camera_dir, f"rgb_{fi:02d}.jpg")
            img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"RGB not found: {rgb_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            orig_H, orig_W = img.shape[:2]
            fx = focal * orig_W
            fy = fx * (orig_H / orig_W)
            cx, cy = orig_W / 2.0, orig_H / 2.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            # Depth
            depth = None
            if load_depth:
                depth_path = osp.join(camera_dir, f"depth_{fi:02d}.png")
                raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise FileNotFoundError(f"Depth not found: {depth_path}")
                depth = raw.astype(np.float32) / uint16_max * (max_d - min_d) + min_d
                # Convert Euclidean depth to z-depth
                u, v = np.meshgrid(np.arange(orig_W, dtype=np.float32),
                                   np.arange(orig_H, dtype=np.float32))
                depth = depth / np.sqrt(1 + ((u - cx)/fx)**2 + ((v - cy)/fy)**2)

            # Camera pose (Blender -> OpenCV)
            pos = self.cam_positions[meta_idx, fi]
            quat = self.cam_quaternions[meta_idx, fi]
            R = quaternion_wxyz_to_rotation_matrix(quat)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R
            c2w[:3, 3] = pos.astype(np.float32)
            c2w = c2w @ BLENDER_TO_OPENCV

            # Crop + resize
            img_pil, depth_resized, K_resized = center_crop_and_resize(
                img, depth, K, self.output_width, self.output_height)

            frames_pil.append(img_pil)
            if depth_resized is not None:
                depths.append(depth_resized)
            poses.append(c2w)
            if K_out is None:
                K_out = K_resized

        return frames_pil, depths if depths else None, poses, K_out

    def _compute_tracks(self, scene_idx, cam_id, meta_idx, frame_ids,
                        query_frame_id, depths, poses, K, H, W):
        """Compute (N, T, 3) world-space 3D tracks using depth + bbox tracking."""
        T = len(frame_ids)
        s = self.track_spatial_stride
        yy = np.arange(0, H, s)
        xx = np.arange(0, W, s)
        grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
        grid_y_flat = grid_y.ravel()
        grid_x_flat = grid_x.ravel()
        N = len(grid_y_flat)

        # Query frame: unproject depth to world (static background)
        q_idx = frame_ids.index(query_frame_id) if query_frame_id in frame_ids else 0
        depth_q = depths[q_idx]
        pose_q = poses[q_idx]

        pts_world_q, valid_q = depth_to_world_pts3d(depth_q, K, pose_q)

        # Initialize all frames with static background (query frame world pts)
        tracks = np.zeros((N, T, 3), dtype=np.float32)
        valid_all = np.ones(N, dtype=bool)

        pts_q_grid = pts_world_q[grid_y_flat, grid_x_flat]  # (N, 3)
        valid_q_grid = valid_q[grid_y_flat, grid_x_flat]

        for t in range(T):
            tracks[:, t, :] = pts_q_grid
        valid_all &= valid_q_grid

        # Overlay dynamic objects using bboxes (same logic as KubricMultiCameraProcessed)
        bbox_start = self.bboxes_cumsum[meta_idx]
        bbox_end = self.bboxes_cumsum[meta_idx + 1]
        bboxes = self.bboxes[bbox_start:bbox_end]  # (n_inst, total_frames, 8, 3)

        if bboxes.shape[0] > 0:
            # Load segm + coords for query frame to identify object pixels
            scene_name = f"{scene_idx:05d}"
            camera_dir = osp.join(self.ROOT, self.split, scene_name, f"camera_{cam_id}")

            segm_path = osp.join(camera_dir, f"segm_{query_frame_id:02d}.png")
            segm = cv2.imread(segm_path, cv2.IMREAD_UNCHANGED)
            if segm is not None:
                if segm.ndim == 3:
                    segm = segm[:, :, 0]
                coords_path = osp.join(camera_dir, f"coords_{query_frame_id:02d}.png")
                obj_coords_raw = cv2.imread(coords_path, cv2.IMREAD_UNCHANGED)
                if obj_coords_raw is not None:
                    obj_coords = obj_coords_raw.astype(np.float32) / np.iinfo(obj_coords_raw.dtype).max

                    # Center-crop and resize segm + coords exactly like RGB/depth.
                    segm = _center_crop_and_resize_np(
                        segm.astype(np.float32),
                        self.output_width,
                        self.output_height,
                        cv2.INTER_NEAREST,
                    ).astype(np.int32)
                    obj_coords = _center_crop_and_resize_np(
                        obj_coords,
                        self.output_width,
                        self.output_height,
                        cv2.INTER_LINEAR,
                    ).astype(np.float32)

                    # For each dynamic object, override tracks
                    n_inst = bboxes.shape[0]
                    for seg_id in range(1, n_inst + 1):
                        mask_full = (segm == seg_id)
                        mask_grid = mask_full[grid_y_flat, grid_x_flat]
                        if not mask_grid.any():
                            continue
                        grid_idx = np.where(mask_grid)[0]
                        obj_local = obj_coords[grid_y_flat[grid_idx], grid_x_flat[grid_idx]]  # (M, 3)

                        for t, fi in enumerate(frame_ids):
                            bbox3d = bboxes[seg_id - 1, fi]  # (8, 3)
                            p0 = bbox3d[0].astype(np.float32)
                            ax = (bbox3d[4] - bbox3d[0]).astype(np.float32)
                            ay = (bbox3d[2] - bbox3d[0]).astype(np.float32)
                            az = (bbox3d[1] - bbox3d[0]).astype(np.float32)

                            pts_world = (
                                p0[None, :] +
                                obj_local[:, 0:1] * ax[None, :] +
                                obj_local[:, 1:2] * ay[None, :] +
                                obj_local[:, 2:3] * az[None, :]
                            ).astype(np.float32)

                            tracks[grid_idx, t, :] = pts_world

        tracks[~valid_all] = 0.0
        return tracks


# ======================================================================
# StaticScene_NVS — TartanAir + DL3DV + MVS-Synth + PointOdyssey
# ======================================================================

class StaticScene_NVS(EasyDataset):
    """Two-window NVS dataset for static/dynamic 3D scenes.

    Picks two separate temporal windows from the same camera trajectory.
    Source window provides RGB + depth -> 3D tracks (static for static scenes).
    Target window provides real RGB + camera poses.

    Supported dataset_type: 'tartanair', 'dl3dv', 'mvs_synth', 'pointodyssey', 'wildrgbd', 'co3d'
    """

    SUPPORTED_TYPES = ("tartanair", "dl3dv", "mvs_synth", "pointodyssey", "dynamic_replica", "wildrgbd", "co3d", "re10k")

    def __init__(
        self,
        ROOT,
        dataset_type,
        num_views=49,
        resolution=(512, 288),
        steps_per_epoch=1000,
        track_spatial_stride=1,
        min_window_gap=10,
        max_window_gap=100,
        min_overlap_ratio=0.0,
        split=None,
        allow_repeat=True,
        reverse_prob=0.5,
        freeze_time=False,
        freeze_time_drop_source_prob=1.0,
        return_target_depth=False,
        return_target_tracks=False,
        return_source_depth_video=False,
        return_source_rgb_video=False,
        return_source_cameras=False,
        caption_dir=None,
        max_scenes=None,
        exclude_sequence_ids=None,
        src_center_in_target_prob=0.0,
        use_da3_depth=False,
        min_shift_ratio=0.25,
        max_shift_ratio=0.75,
        shift_beta_ab=None,
        seed=None,
    ):
        """
        Args:
            min_window_gap: Min gap between sequential windows (overlap mode ignores this).
            max_window_gap: Max gap between sequential windows (overlap mode ignores this).
            min_overlap_ratio: If > 0, use OVERLAPPING windows instead of sequential gap.
                Source and target windows share at least this fraction of frames.
                E.g., 0.25 → ≥25% temporal overlap. Source [s:s+T], target [s+shift:s+shift+T]
                where shift ∈ [1, floor(T * (1 - min_overlap_ratio))].
                Randomly swaps source/target (reverse_prob). Recommended for static scenes.
            freeze_time: If True, use freeze-time frame duplication instead of two-window.
            freeze_time_drop_source_prob: Probability of dropping source video conditioning
                when freeze_time=True. Default 1.0 = always drop.
            src_center_in_target_prob: For forward-moving datasets (TartanAir), probability
                that source frame 0 lands exactly at the middle of the target window
                (shift = T//2). With probability 1 - src_center_in_target_prob the shift
                is drawn uniformly from [min_shift, max_shift] as usual. Default 0.0.
        """
        self.ROOT = ROOT
        self.dataset_type = dataset_type.lower()
        assert self.dataset_type in self.SUPPORTED_TYPES, \
            f"dataset_type must be one of {self.SUPPORTED_TYPES}, got {dataset_type}"
        self._num_views = int(num_views)
        self.steps_per_epoch = int(steps_per_epoch)
        self.track_spatial_stride = int(track_spatial_stride)
        self.min_window_gap = int(min_window_gap)
        self.max_window_gap = int(max_window_gap)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.reverse_prob = float(reverse_prob)
        self.split = split
        self.freeze_time = bool(freeze_time)
        self.freeze_time_drop_source_prob = float(freeze_time_drop_source_prob)
        # Track4D training may auto-inject these kwargs for depth-/track-
        # supervised GS runs. StaticScene_NVS already returns target_depth
        # when available, and does NOT have persistent 3D tracks (no
        # `target_tracks_world` emitted). Keep both as compatibility
        # no-ops instead of rejecting the constructor — this lets
        # mixed-dataset configs (e.g. v4 Kubric+MVS-Synth) inject
        # `return_target_tracks=True` globally without crashing on the
        # StaticScene_NVS entry. The v4 loss dispatch handles the absence
        # of target_tracks_world by zeroing track losses on those batches.
        self.return_target_depth = bool(return_target_depth)
        self.return_target_tracks = bool(return_target_tracks)  # no-op for static scenes
        self.return_source_depth_video = bool(return_source_depth_video)
        self.return_source_rgb_video = bool(return_source_rgb_video)
        self.return_source_cameras = bool(return_source_cameras)
        self.use_da3_depth = bool(use_da3_depth)
        self.seed = None if seed is None else int(seed)
        self.src_center_in_target_prob = float(src_center_in_target_prob)
        self.min_shift_ratio = float(min_shift_ratio)
        self.max_shift_ratio = float(max_shift_ratio)
        assert 0.0 <= self.min_shift_ratio <= self.max_shift_ratio <= 1.0, \
            f"need 0 ≤ min_shift_ratio ≤ max_shift_ratio ≤ 1, got " \
            f"{self.min_shift_ratio}, {self.max_shift_ratio}"
        if shift_beta_ab is not None:
            a, b = shift_beta_ab
            self.shift_beta_ab = (float(a), float(b))
        else:
            self.shift_beta_ab = None
        self.exclude_sequence_ids = {
            str(seq_id) for seq_id in (exclude_sequence_ids or [])
        }
        resolved_caption_dir = default_caption_dir_for_dataset(
            self.dataset_type, caption_root=caption_dir
        )
        self._caption_store = (
            ChunkedCaptionStore(resolved_caption_dir, DEFAULT_PROMPT)
            if resolved_caption_dir is not None and osp.isdir(resolved_caption_dir)
            else None
        )

        if isinstance(resolution, int):
            self.output_width = self.output_height = resolution
        elif isinstance(resolution, (tuple, list)):
            self.output_width, self.output_height = int(resolution[0]), int(resolution[1])
        else:
            self.output_width = self.output_height = int(resolution)

        self._scene_entries = self._scan_scenes()
        if not self._scene_entries:
            raise ValueError(f"No valid scenes found in {ROOT} for {dataset_type}")
        if self.exclude_sequence_ids:
            self._scene_entries = [
                scene for scene in self._scene_entries
                if str(scene.get("sequence_id")) not in self.exclude_sequence_ids
            ]
            if not self._scene_entries:
                raise ValueError(
                    f"All scenes were filtered out in {ROOT} for {dataset_type} by "
                    f"exclude_sequence_ids={sorted(self.exclude_sequence_ids)}"
                )

        if self.freeze_time:
            min_needed = 2
        elif self.min_overlap_ratio > 0:
            # Overlap mode: need T + 1 frames (shift ≥ 1)
            min_needed = self._num_views + 1
        else:
            min_needed = 2 * self._num_views + self.min_window_gap
        self._scene_entries = [s for s in self._scene_entries if len(s["frames"]) >= min_needed]
        if not self._scene_entries:
            raise ValueError(
                f"No scenes with >= {min_needed} frames in {ROOT} for {dataset_type}"
            )
        # Optional: truncate to first N scenes (by the sorted order produced by
        # the per-dataset _scan_*() method). Useful for "first 100 DL3DV" style
        # caption-dense subset training where the full dataset is too big.
        if max_scenes is not None:
            max_scenes_int = int(max_scenes)
            if max_scenes_int > 0 and len(self._scene_entries) > max_scenes_int:
                self._scene_entries = self._scene_entries[:max_scenes_int]
        total_scenes = len(self._scene_entries)
        total_frames = sum(len(s["frames"]) for s in self._scene_entries)
        if self.freeze_time:
            mode_str = "freeze-time"
        elif self.min_overlap_ratio > 0:
            mode_str = f"overlap, min_ratio={self.min_overlap_ratio:.0%}"
        else:
            mode_str = f"two-window, min_gap={self.min_window_gap}"
        print(f"[StaticScene_NVS/{dataset_type}] {total_scenes} scenes, "
              f"{total_frames} total frames, mode={mode_str}")

    @staticmethod
    def _extract_source_rgb_video(src_frames):
        return np.stack(
            [np.array(frame, dtype=np.float32) / 255.0 for frame in src_frames],
            axis=0,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _extract_source_cam2cam0(src_poses, src_cam0_pose, common_scale):
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)
        src_c2w_world = np.stack([pose.astype(np.float32) for pose in src_poses], axis=0)
        src_c2cam0 = world_to_cam0[None, :, :] @ src_c2w_world
        src_c2cam0 = src_c2cam0.copy()
        src_c2cam0[:, :3, 3] /= max(float(common_scale), 1e-6)
        return src_c2cam0.astype(np.float32, copy=False)

    @staticmethod
    def _extract_source_intrinsics(src_K):
        return src_K.astype(np.float32, copy=False)

    def _attach_source_dense_fields(
        self,
        result,
        src_frames,
        src_poses,
        src_K,
        src_depths,
        src_cam0_pose,
        common_scale,
    ):
        if self.return_source_rgb_video:
            result["source_rgb_video"] = self._extract_source_rgb_video(src_frames)
        if self.return_source_cameras:
            result["source_cam2world"] = self._extract_source_cam2cam0(
                src_poses, src_cam0_pose, common_scale,
            )
            result["source_intrinsics"] = self._extract_source_intrinsics(src_K)
        if self.return_source_depth_video and "source_depth_video" not in result and src_depths:
            result["source_depth_video"] = (
                np.stack(src_depths, axis=0).astype(np.float32) / max(float(common_scale), 1e-6)
            )

    def _scan_scenes(self):
        scanner = {
            "tartanair": self._scan_tartanair,
            "dl3dv": self._scan_dl3dv,
            "mvs_synth": self._scan_mvs_synth,
            "pointodyssey": self._scan_pointodyssey,
            "dynamic_replica": self._scan_dynamic_replica,
            "wildrgbd": self._scan_wildrgbd,
            "co3d": self._scan_co3d,
            "re10k": self._scan_re10k,
        }
        return scanner[self.dataset_type]()

    def _scan_tartanair(self):
        entries = []
        scene_dirs = sorted([d for d in os.listdir(self.ROOT)
                             if osp.isdir(osp.join(self.ROOT, d))])
        for scene in scene_dirs:
            for mode in ["Easy", "Hard"]:
                mode_dir = osp.join(self.ROOT, scene, mode)
                if not osp.isdir(mode_dir):
                    continue
                seq_dirs = sorted([d for d in os.listdir(mode_dir)
                                   if osp.isdir(osp.join(mode_dir, d))])
                for seq in seq_dirs:
                    seq_dir = osp.join(mode_dir, seq)
                    basenames = sorted([f[:-8] for f in os.listdir(seq_dir)
                                        if f.endswith("_rgb.png")])
                    if len(basenames) < 2:
                        continue
                    entries.append({
                        "scene_dir": seq_dir,
                        "frames": basenames,
                        "dataset": "tartanair",
                        "sequence_id": f"{scene}/{mode}/{seq}",
                    })
        return entries

    def _scan_dl3dv(self):
        entries = []
        for category in sorted(os.listdir(self.ROOT)):
            cat_dir = osp.join(self.ROOT, category)
            if not osp.isdir(cat_dir):
                continue
            for scene_hash in sorted(os.listdir(cat_dir)):
                dense_dir = osp.join(cat_dir, scene_hash, "dense")
                rgb_dir = osp.join(dense_dir, "rgb")
                if not osp.isdir(rgb_dir):
                    continue
                basenames = sorted([f[:-4] for f in os.listdir(rgb_dir)
                                    if f.endswith(".png")])
                if len(basenames) < 2:
                    continue
                if not osp.isdir(osp.join(dense_dir, "depth")):
                    continue
                if not osp.isdir(osp.join(dense_dir, "cam")):
                    continue
                # Robustness: do NOT gate scene inclusion on da3_depth/dense_depth
                # existence, even when use_da3_depth=True. Per-frame loader falls
                # back to the original depth/ when DA3 is missing or partially
                # transferred — this lets the run train on servers where DA3 is
                # still syncing or is missing entirely for some scenes.
                entries.append({
                    "scene_dir": dense_dir,
                    "frames": basenames,
                    "dataset": "dl3dv",
                    "sequence_id": f"{category}/{scene_hash}",
                })
        return entries

    def _scan_mvs_synth(self):
        entries = []
        for scene in sorted(os.listdir(self.ROOT)):
            scene_dir = osp.join(self.ROOT, scene)
            if not osp.isdir(scene_dir):
                continue
            rgb_dir = osp.join(scene_dir, "rgb")
            if not osp.isdir(rgb_dir):
                continue
            basenames = sorted([f[:-4] for f in os.listdir(rgb_dir)
                                if f.endswith(".jpg")])
            if len(basenames) < 2:
                continue
            if not osp.isdir(osp.join(scene_dir, "cam")):
                continue
            if not osp.isdir(osp.join(scene_dir, "depth")):
                continue
            entries.append({
                "scene_dir": scene_dir,
                "frames": basenames,
                "dataset": "mvs_synth",
                "sequence_id": scene,
            })
        return entries

    def _scan_pointodyssey(self):
        entries = []
        split = self.split or "train"
        root = osp.join(self.ROOT, split)
        if not osp.isdir(root):
            print(f"[StaticScene_NVS/pointodyssey] Split dir not found: {root}")
            return entries
        for scene in sorted(os.listdir(root)):
            scene_dir = osp.join(root, scene)
            if not osp.isdir(scene_dir):
                continue
            rgb_dir = osp.join(scene_dir, "rgb")
            if not osp.isdir(rgb_dir):
                continue
            basenames = sorted([f[:-4] for f in os.listdir(rgb_dir)
                                if f.endswith(".jpg")])
            if len(basenames) < 2:
                continue
            if not osp.isdir(osp.join(scene_dir, "cam")):
                continue
            entries.append({
                "scene_dir": scene_dir,
                "frames": basenames,
                "dataset": "pointodyssey",
                "sequence_id": scene,
            })
        return entries

    def _scan_dynamic_replica(self):
        entries = []
        split = self.split or "train"
        root = osp.join(self.ROOT, split)
        if not osp.isdir(root):
            print(f"[StaticScene_NVS/dynamic_replica] Split dir not found: {root}")
            return entries
        for scene in sorted(os.listdir(root)):
            # DynamicReplica uses {split}/{scene}/left/ structure
            scene_dir = osp.join(root, scene, "left")
            if not osp.isdir(scene_dir):
                continue
            rgb_dir = osp.join(scene_dir, "rgb")
            if not osp.isdir(rgb_dir):
                continue
            basenames = sorted([f[:-4] for f in os.listdir(rgb_dir)
                                if f.endswith(".png")],
                               key=lambda x: float(x))
            if len(basenames) < 2:
                continue
            if not osp.isdir(osp.join(scene_dir, "cam")):
                continue
            entries.append({
                "scene_dir": scene_dir,
                "frames": basenames,
                "dataset": "dynamic_replica",
                "sequence_id": scene,
            })
        return entries

    def _scan_wildrgbd(self):
        """Scan WildRGBD-MP processed dataset.

        Layout: ROOT/{category}/scenes/{scene_name}/{rgb,depth,metadata}/
                ROOT/{category}/selected_seqs_train.json → {scene_key: [frame_ids]}
        """
        entries = []
        if not osp.isdir(self.ROOT):
            return entries
        for category in sorted(os.listdir(self.ROOT)):
            cat_dir = osp.join(self.ROOT, category)
            train_json = osp.join(cat_dir, "selected_seqs_train.json")
            if not osp.isfile(train_json):
                continue
            with open(train_json) as f:
                scene_map = json.load(f)
            for scene_key, frame_ids in scene_map.items():
                scene_dir = osp.join(cat_dir, scene_key)
                if not osp.isdir(scene_dir):
                    continue
                if not osp.isdir(osp.join(scene_dir, "rgb")):
                    continue
                if len(frame_ids) < 2:
                    continue
                entries.append({
                    "scene_dir": scene_dir,
                    "frames": frame_ids,  # list of int frame IDs
                    "dataset": "wildrgbd",
                    "sequence_id": f"{category}/{scene_key.replace('scenes/', '', 1)}",
                })
        return entries

    def _load_wildrgbd_frame(self, scene_dir, frame_id):
        """Load one WildRGBD frame: RGB (jpg) + depth + safetensor cam.

        Depth sources (in priority order when use_da3_depth=True):
          1. da3_depth/{frame:05d}.npy  — DA3 dense metric depth (float32, metres)
             + da3_sky_mask/{frame:05d}.png zeroed (255 = sky)
          2. depth/{frame:05d}.png      — original uint16 PNG (mm → metres, fallback)
        """
        from safetensors.numpy import load_file
        img = cv2.imread(osp.join(scene_dir, "rgb", f"{frame_id:05d}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"RGB not found: {scene_dir}/rgb/{frame_id:05d}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        da3_path = osp.join(scene_dir, "da3_depth", f"{frame_id:05d}.npy")
        if self.use_da3_depth and osp.exists(da3_path):
            depth = np.load(da3_path).astype(np.float32)
            depth[~np.isfinite(depth)] = 0.0
            sky_path = osp.join(scene_dir, "da3_sky_mask", f"{frame_id:05d}.png")
            if osp.exists(sky_path):
                depth[cv2.imread(sky_path, cv2.IMREAD_UNCHANGED) >= 127] = 0.0
        else:
            depth_path = osp.join(scene_dir, "depth", f"{frame_id:05d}.png")
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise FileNotFoundError(f"Depth not found: {depth_path}")
            depth = depth_raw.astype(np.float32) / 1000.0  # mm → metres
            depth[depth <= 0] = 0.0

        cam = load_file(osp.join(scene_dir, "metadata", f"{frame_id:05d}.safetensor"))
        pose = cam["camera_pose"].astype(np.float32)
        K = cam["camera_intrinsics"].astype(np.float32)
        return img, depth, pose, K

    def _scan_co3d(self):
        """Scan Co3D processed dataset.

        Mirrors Co3d_Multi (co3d.py): reads root-level selected_seqs_train.json with
        structure {category: {scene_id: [frame_ids]}}.

        Layout: ROOT/selected_seqs_train.json → {category: {scene_id: [frame_ids]}}
                ROOT/{category}/{scene_id}/images/frame{N:06d}.jpg
                ROOT/{category}/{scene_id}/images/frame{N:06d}.safetensor (pose, K, maximum_depth)
                ROOT/{category}/{scene_id}/depths/frame{N:06d}.jpg.geometric.png (uint16)

        is_synthetic=False → has_depth=False (depth used for track construction only, not supervision).
        """
        entries = []
        train_json = osp.join(self.ROOT, "selected_seqs_train.json")
        if not osp.isfile(train_json):
            return entries
        with open(train_json) as f:
            all_scenes = json.load(f)
        for category, scene_map in all_scenes.items():
            for scene_id, frame_ids in scene_map.items():
                scene_dir = osp.join(self.ROOT, category, scene_id)
                if not osp.isdir(scene_dir):
                    continue
                if not osp.isdir(osp.join(scene_dir, "images")):
                    continue
                # Split into contiguous blocks (no gap > 1) to avoid camera jumps
                for block in _split_contiguous(frame_ids, max_gap=1):
                    if len(block) < 2:
                        continue
                    entries.append({
                        "scene_dir": scene_dir,
                        "frames": block,
                        "dataset": "co3d",
                    })
        return entries

    def _load_co3d_frame(self, scene_dir, frame_id):
        """Load one Co3D frame: RGB + depth (uint16, per-frame scale) + safetensor cam.

        Depth decode mirrors Co3d_Multi._read_depthmap (co3d.py):
            depth = raw_uint16 / 65535.0 * maximum_depth  (maximum_depth per-frame in safetensor).
        """
        from safetensors.numpy import load_file
        img_path = osp.join(scene_dir, "images", f"frame{frame_id:06d}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"RGB not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth_path = osp.join(scene_dir, "depths", f"frame{frame_id:06d}.jpg.geometric.png")
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Depth not found: {depth_path}")
        cam = load_file(osp.join(scene_dir, "images", f"frame{frame_id:06d}.safetensor"))
        maximum_depth = float(np.nan_to_num(cam["maximum_depth"]))
        depth = depth_raw.astype(np.float32) / 65535.0 * maximum_depth
        depth[depth <= 0] = 0.0
        pose = cam["camera_pose"].astype(np.float32)
        K = cam["camera_intrinsics"].astype(np.float32)
        return img, depth, pose, K

    def _scan_re10k(self):
        """Scan RE10K: {ROOT}/{scene_id}/{rgb,cam,depth}/."""
        entries = []
        for scene_id in sorted(os.listdir(self.ROOT)):
            scene_dir = osp.join(self.ROOT, scene_id)
            rgb_dir = osp.join(scene_dir, "rgb")
            cam_dir = osp.join(scene_dir, "cam")
            depth_dir = osp.join(scene_dir, "depth")
            if not osp.isdir(rgb_dir) or not osp.isdir(cam_dir):
                continue
            # depth/ is DA3-labeled (RE10K has no original depth)
            if not osp.isdir(depth_dir):
                continue
            basenames = sorted([
                osp.splitext(f)[0] for f in os.listdir(rgb_dir) if f.endswith(".png")
            ])
            if len(basenames) < 2:
                continue
            entries.append({
                "scene_dir": scene_dir,
                "frames": basenames,
                "dataset": "re10k",
                "sequence_id": scene_id,
            })
        return entries

    def _load_re10k_frame(self, scene_dir, basename):
        """Load RE10K frame: RGB + DA3 depth + cam (npz: intrinsics, pose c2w).

        Sky detection: DA3 sets sky pixels to conf=1.0 (non-sky >> 1).
        We zero out depth where conf <= 1.5.
        """
        img = cv2.imread(osp.join(scene_dir, "rgb", basename + ".png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth_path = osp.join(scene_dir, "depth", basename + ".npy")
        if osp.exists(depth_path):
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = np.zeros(img.shape[:2], dtype=np.float32)
        depth[~np.isfinite(depth)] = 0.0
        # DA3 sky mask from confidence: sky pixels have conf=1.0
        conf_path = osp.join(scene_dir, "conf", basename + ".npy")
        if osp.exists(conf_path):
            conf = np.load(conf_path).astype(np.float32)
            if conf.shape[:2] != depth.shape[:2]:
                conf = cv2.resize(conf, (depth.shape[1], depth.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            depth[conf <= 1.5] = 0.0  # sky
        depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
        valid_d = depth[depth > 0]
        if valid_d.size > 0:
            p98 = float(np.percentile(valid_d, 98))
            depth[depth > p98] = p98
        cam = np.load(osp.join(scene_dir, "cam", basename + ".npz"))
        return img, depth, cam["pose"].astype(np.float32), cam["intrinsics"].astype(np.float32)

    # --- EasyDataset interface ---

    @property
    def _resolutions(self):
        return [(self.output_width, self.output_height)]

    @property
    def num_views(self):
        return self._num_views

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        seed = None if self.seed is None else self.seed + int(idx)
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if seed is None:
                    return self._getitem_impl()
                attempt_seed = seed + attempt * 1000003
                py_state = random.getstate()
                np_state = np.random.get_state()
                try:
                    random.seed(attempt_seed)
                    np.random.seed(attempt_seed % (2**32 - 1))
                    return self._getitem_impl(np.random.default_rng(attempt_seed))
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"StaticScene_NVS/{self.dataset_type}: failed after {max_retries} retries: {e}"
                    ) from e

    def _getitem_impl(self, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        scene = self._scene_entries[rng.integers(0, len(self._scene_entries))]
        frames = scene["frames"]
        T = self._num_views
        total = len(frames)

        if self.freeze_time:
            return self._getitem_freeze_time(scene, rng)
        elif self.min_overlap_ratio > 0:
            return self._getitem_overlap_window(scene, rng)
        else:
            return self._getitem_two_window(scene, rng)

    def _getitem_overlap_window(self, scene, rng):
        """Overlap-window mode: two windows with 25%-75% temporal overlap.

        Constraints:
          1. Overlap ≥ 25% and ≤ 75% (shift ∈ [T*0.25, T*0.75])
          2. All indices in [0, total) — never restart
          3. Randomly swap src/tgt AND randomly reverse temporal order
             (camera poses stay aligned with frames)

        For forward-moving datasets (TartanAir): target is centered around
        source frame 0 so that the 3D tracks (computed from source frame 0
        depth) have maximum spatial coverage in the target view.
        For orbit datasets (DL3DV, MVS-Synth): regular shifted windows are fine
        since the camera orbits around the same scene center.
        """
        frames = scene["frames"]
        T = self._num_views
        total = len(frames)
        # is_forward routes the dataset through the "target-precedes-source +
        # optional source-0 centering" branch. TartanAir is forward-moving by
        # nature; any dataset that asks for src_center_in_target_prob > 0 (e.g.
        # DL3DV at high-overlap settings) opts in too.
        is_forward = (
            scene["dataset"] in ("tartanair",)
            or self.src_center_in_target_prob > 0.0
        )

        # Shift range: driven by min_shift_ratio / max_shift_ratio kwargs.
        # Lower max_shift_ratio → larger temporal overlap → stronger visual overlap.
        min_shift = max(1, int(np.ceil(T * self.min_shift_ratio)))
        max_shift = max(min_shift, int(T * self.max_shift_ratio))
        # Clamp to what the scene length actually allows. max(0, ...) guards
        # against scenes with total < T: an unclamped `total - T` there would go
        # negative, make `shift` negative, and produce a negative-index wrap
        # when building tgt_indices below (Python lst[-3] == 3rd-from-end).
        max_shift = max(0, min(max_shift, total - T))
        min_shift = max(0, min(min_shift, max_shift))
        if self.shift_beta_ab is not None and max_shift > min_shift:
            # Beta(a, b) biases shift toward the low end when a < b (mean = a/(a+b)).
            # E.g. Beta(2, 5) → E[shift_ratio] ≈ 0.29 of the [min_shift, max_shift] span
            # so most samples are high-overlap but a small tail keeps large-shift diversity.
            u = float(np.random.beta(*self.shift_beta_ab))
            shift = int(round(min_shift + u * (max_shift - min_shift)))
            shift = max(min_shift, min(shift, max_shift))
        else:
            shift = int(rng.integers(min_shift, max_shift + 1))

        if is_forward:
            # Target centered on source frame 0: target starts ~T/2 before source.
            # source_start = src_s, target_start = src_s - shift
            # So source frame 0 sits at target frame[shift] (near middle when shift≈T/2).
            #
            # src_center_in_target_prob: with this probability, snap shift to T//2 so
            # source frame 0 lands exactly at the center of the target window.
            if self.src_center_in_target_prob > 0.0 and random.random() < self.src_center_in_target_prob:
                shift = max(min_shift, min(T // 2, max_shift))
            max_src_start = max(0, total - T)
            min_src_start = shift  # target_start = src_s - shift ≥ 0
            if min_src_start > max_src_start:
                min_src_start = 0
                shift = max(0, min(shift, max_src_start))
            src_s = rng.integers(min_src_start, max(min_src_start + 1, max_src_start + 1))
            tgt_s = max(0, src_s - shift)
            src_indices = list(range(src_s, min(src_s + T, total)))
            tgt_indices = list(range(tgt_s, min(tgt_s + T, total)))
        else:
            # Regular overlap: two shifted windows. Guard start/shift so indices
            # never go negative (Python list[-1] wraps to end → would manifest
            # as "target video starts from frame 0" mid-clip).
            max_start = max(0, total - T - shift)
            start = rng.integers(0, max(1, max_start + 1))
            src_indices = list(range(start, start + T))
            tgt_indices = list(range(start + shift, start + shift + T))

        # Pad if needed (shouldn't happen with correct bounds)
        while len(src_indices) < T:
            src_indices.append(src_indices[-1])
        while len(tgt_indices) < T:
            tgt_indices.append(tgt_indices[-1])

        # Randomly swap source/target
        if random.random() < self.reverse_prob:
            src_indices, tgt_indices = tgt_indices, src_indices

        # Randomly reverse temporal order (both windows together, keeps alignment)
        if random.random() < 0.5:
            src_indices = src_indices[::-1]
            tgt_indices = tgt_indices[::-1]

        return self._load_and_normalize_windows(scene, src_indices, tgt_indices, rng)

    def _getitem_two_window(self, scene, rng):
        """Two-window mode: pick two NON-overlapping trajectory segments."""
        frames = scene["frames"]
        T = self._num_views
        total = len(frames)

        max_gap = min(self.max_window_gap, total - 2 * T)
        gap = rng.integers(self.min_window_gap, max(self.min_window_gap + 1, max_gap + 1))
        max_src_start = total - 2 * T - gap
        src_start = rng.integers(0, max(1, max_src_start + 1))
        tgt_start = src_start + T + gap

        src_indices = list(range(src_start, src_start + T))
        tgt_indices = list(range(tgt_start, min(tgt_start + T, total)))
        while len(tgt_indices) < T:
            tgt_indices.append(tgt_indices[-1])

        if random.random() < self.reverse_prob:
            src_indices, tgt_indices = tgt_indices, src_indices

        return self._load_and_normalize_windows(scene, src_indices, tgt_indices, rng)

    def _load_and_normalize_windows(self, scene, src_indices, tgt_indices, rng):
        """Shared load + normalize for both two-window and overlap-window modes."""
        src_frames, src_depths, src_poses, src_K = self._load_window(
            scene, src_indices, load_depth=True)
        is_synthetic = scene["dataset"] in ("tartanair", "mvs_synth")
        # Load target depth for any dataset that has depth on disk. DL3DV has
        # CUT3R-reconstructed depth; CO3D/synthetic have GT or pseudo-GT depth.
        # The GS render-side depth loss can use any of these — invalid pixels
        # are handled by the > 1e-6 fallback mask in gs_loss.py.
        load_tgt_depth = True
        tgt_frames, tgt_depths, tgt_poses, tgt_K = self._load_window(
            scene, tgt_indices, load_depth=load_tgt_depth)

        tracks_cam0, tgt_cam2cam0, common_scale = self._build_static_tracks_and_normalize(
            src_depths[0], src_K, src_poses[0], tgt_poses)

        is_dynamic = scene["dataset"] in ("pointodyssey", "dynamic_replica")
        source_depth = src_depths[0] / common_scale if src_depths else None

        if tgt_depths:
            target_depth = np.stack(tgt_depths, axis=0).astype(np.float32) / common_scale
            # Valid mask: finite, positive depth (CUT3R may produce 0/inf for
            # sky, reflective surfaces, or far background).
            target_depth_valid = (
                np.isfinite(target_depth) & (target_depth > 0)
            ).astype(np.float32)
        else:
            target_depth = None
            target_depth_valid = None
        has_depth = target_depth is not None
        prompt, prompt_emb = self._resolve_target_caption_with_emb(scene, tgt_indices)

        result = {
            "video": src_frames,
            "target_video": tgt_frames,
            "tracks_world": tracks_cam0,
            "target_cam2world": tgt_cam2cam0,
            "target_intrinsics": tgt_K,
            "common_scale": common_scale,
            "track_query_idx": 0,
            "prompt": prompt,
            "prompt_emb": prompt_emb,
            "has_tracks": True,
            "has_dynamic_tracks": is_dynamic,
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "source_depth": source_depth,
            "target_depth": target_depth,
            "target_depth_valid": target_depth_valid,
            "has_depth": has_depth,
            "source_rgb_at_tracks": _source_rgb_at_track_grid(
                src_frames,
                self.track_spatial_stride,
                [0],
            ),
        }
        self._attach_source_dense_fields(
            result=result,
            src_frames=src_frames,
            src_poses=src_poses,
            src_K=src_K,
            src_depths=src_depths,
            src_cam0_pose=src_poses[0],
            common_scale=common_scale,
        )
        return result

    def _getitem_freeze_time(self, scene, rng):
        """Freeze-time mode: duplicate single frames to construct 4D video structure.

        Paper: "For the static scenes, we duplicate and stack frames to construct a 4D
        video structure. To prevent the model from learning a trivial solution that simply
        replicates the first frame across all views, we find it necessary to remove frame
        conditioning when using freeze-time videos."

        Picks two different frame indices (= two different cameras/viewpoints).
        Source = frame_A duplicated T times. Target = frame_B duplicated T times.
        """
        frames = scene["frames"]
        T = self._num_views
        total = len(frames)

        # Pick two different frame indices (different viewpoints of static scene)
        idx_a, idx_b = rng.choice(total, size=2, replace=False)

        # Random viewpoint reversal
        if random.random() < self.reverse_prob:
            idx_a, idx_b = idx_b, idx_a

        # Load single source frame (with depth) and single target frame
        is_synthetic = scene["dataset"] in ("tartanair", "mvs_synth")
        src_frames_single, src_depths, src_poses, src_K = self._load_window(
            scene, [int(idx_a)], load_depth=True)
        tgt_frames_single, tgt_depths_single, tgt_poses, tgt_K = self._load_window(
            scene, [int(idx_b)], load_depth=is_synthetic)

        # Duplicate to T frames
        src_frames = src_frames_single * T
        tgt_frames = tgt_frames_single * T
        src_poses_T = src_poses * T
        src_depths_T = src_depths * T
        # Camera pose is static for all T frames
        tgt_poses_T = tgt_poses * T

        tracks_cam0, tgt_cam2cam0, common_scale = self._build_static_tracks_and_normalize(
            src_depths[0], src_K, src_poses[0], tgt_poses_T)

        # Build target depth: single frame tiled to (T, H, W), normalized by common_scale
        if is_synthetic and tgt_depths_single:
            target_depth_single = tgt_depths_single[0]
            target_depth = np.tile(target_depth_single[None, ...], (T, 1, 1)).astype(np.float32) / common_scale
            has_depth = True
        else:
            target_depth = None
            has_depth = False

        # source_depth: frame-0 dense depth for visualization
        source_depth = src_depths[0] / common_scale if src_depths else None

        # Drop source conditioning to prevent trivial copy
        drop_source = random.random() < self.freeze_time_drop_source_prob
        prompt, prompt_emb = self._resolve_target_caption_with_emb(scene, [int(idx_b)])

        result = {
            "video": src_frames,
            "target_video": tgt_frames,
            "tracks_world": tracks_cam0,
            "target_cam2world": tgt_cam2cam0,
            "target_intrinsics": tgt_K,
            "common_scale": common_scale,
            "track_query_idx": 0,
            "prompt": prompt,
            "prompt_emb": prompt_emb,
            "has_tracks": True,
            "has_dynamic_tracks": False,
            "track_video_loss_mask": 1.0,
            "drop_source_cond": drop_source,
            "source_depth": source_depth,
            "target_depth": target_depth,
            "has_depth": has_depth,
            "source_rgb_at_tracks": _source_rgb_at_track_grid(
                src_frames,
                self.track_spatial_stride,
                [0],
            ),
        }
        self._attach_source_dense_fields(
            result=result,
            src_frames=src_frames,
            src_poses=src_poses_T,
            src_K=src_K,
            src_depths=src_depths_T,
            src_cam0_pose=src_poses[0],
            common_scale=common_scale,
        )
        return result

    def _resolve_target_caption(self, scene, target_indices):
        # Legacy text-only path. Prefer _resolve_target_caption_with_emb in
        # __getitem__ so we forward both the string and the cached T5 embedding.
        text, _ = self._resolve_target_caption_with_emb(scene, target_indices)
        return text

    def _resolve_target_caption_with_emb(self, scene, target_indices):
        """Return (text, emb_or_none). The embedding is non-None only when:
          (a) ``captions_t5_emb.safetensors`` exists next to the caption JSON, and
          (b) the chunk-overlap selector matched a real chunk key (not the default
              fallback). DEFAULT_PROMPT samples always return (DEFAULT_PROMPT, None).
        """
        if self._caption_store is None:
            return DEFAULT_PROMPT, None
        sequence_id = scene.get("sequence_id")
        if not sequence_id:
            return DEFAULT_PROMPT, None
        return self._caption_store.get_with_emb(sequence_id, target_indices)

    def load_fixed_windows(self, scene_idx=0, src_start=0, tgt_start=None):
        """Deterministically load one static-scene source/target window pair.

        Used by eval scripts that want a stable sample instead of the default
        random two-window / overlap sampling.
        """
        scene = self._scene_entries[int(scene_idx)]
        T = self._num_views
        total = len(scene["frames"])
        src_start = int(src_start)
        if src_start < 0 or src_start + T > total:
            raise ValueError(f"src_start={src_start} invalid for scene length {total} and T={T}")

        if tgt_start is None:
            tgt_start = src_start + max(1, T // 2)
        tgt_start = int(tgt_start)
        if tgt_start < 0 or tgt_start + T > total:
            raise ValueError(f"tgt_start={tgt_start} invalid for scene length {total} and T={T}")

        src_indices = list(range(src_start, src_start + T))
        tgt_indices = list(range(tgt_start, tgt_start + T))
        return self._load_and_normalize_windows(scene, src_indices, tgt_indices, np.random.default_rng(0))

    def _build_static_tracks_and_normalize(self, depth_0, K, src_cam0_pose, tgt_poses_list):
        """Shared: build static tracks from frame-0 depth, normalize to cam0."""
        T = self._num_views
        H, W = self.output_height, self.output_width
        pts_world_0, valid_0 = depth_to_world_pts3d(depth_0, K, src_cam0_pose)

        s = self.track_spatial_stride
        yy = np.arange(0, H, s)
        xx = np.arange(0, W, s)
        grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
        grid_y = grid_y.ravel()
        grid_x = grid_x.ravel()

        pts_grid = pts_world_0[grid_y, grid_x]
        valid_grid = valid_0[grid_y, grid_x]
        tracks_world = np.tile(pts_grid[:, None, :], (1, T, 1))
        tracks_world[~valid_grid] = 0.0

        tgt_cam2world = np.stack(tgt_poses_list, axis=0)
        return normalize_to_cam0(tracks_world, tgt_cam2world, src_cam0_pose)

    def _load_window(self, scene, indices, load_depth=True):
        # Guard: drop out-of-range indices and pad to T so Python's negative-
        # index wrap (lst[-3] == last 3rd element) can never silently substitute
        # end-of-clip frames for frame 0.
        indices = _sanitize_indices(indices, len(scene["frames"]), self._num_views)
        frames_pil = []
        depths = []
        poses = []
        K_out = None

        for i in indices:
            basename = scene["frames"][i]
            ds = scene["dataset"]
            if ds == "tartanair":
                img, depth, pose, K = self._load_tartanair_frame(scene["scene_dir"], basename)
            elif ds == "dl3dv":
                img, depth, pose, K = self._load_dl3dv_frame(scene["scene_dir"], basename)
            elif ds == "mvs_synth":
                img, depth, pose, K = self._load_mvs_synth_frame(scene["scene_dir"], basename)
            elif ds == "pointodyssey":
                img, depth, pose, K = self._load_pointodyssey_frame(scene["scene_dir"], basename)
            elif ds == "dynamic_replica":
                img, depth, pose, K = self._load_dynamic_replica_frame(scene["scene_dir"], basename)
            elif ds == "wildrgbd":
                img, depth, pose, K = self._load_wildrgbd_frame(scene["scene_dir"], basename)
            elif ds == "co3d":
                img, depth, pose, K = self._load_co3d_frame(scene["scene_dir"], basename)
            elif ds == "re10k":
                img, depth, pose, K = self._load_re10k_frame(scene["scene_dir"], basename)
            else:
                raise ValueError(f"Unknown dataset: {ds}")

            img_pil, depth_resized, K_resized = center_crop_and_resize(
                img, depth if load_depth else None, K,
                self.output_width, self.output_height)

            frames_pil.append(img_pil)
            if load_depth and depth_resized is not None:
                depths.append(depth_resized)
            poses.append(pose)
            if K_out is None:
                K_out = K_resized

        return frames_pil, depths if depths else None, poses, K_out

    def _load_tartanair_frame(self, scene_dir, basename):
        from safetensors.numpy import load_file
        img = cv2.imread(osp.join(scene_dir, basename + "_rgb.png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = np.load(osp.join(scene_dir, basename + "_depth.npy"))
        depth[depth >= 1000] = 0.0
        depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
        cam = load_file(osp.join(scene_dir, basename + "_cam.safetensor"))
        return img, depth.astype(np.float32), cam["camera_pose"].astype(np.float32), cam["camera_intrinsics"].astype(np.float32)

    def _load_dl3dv_frame(self, scene_dir, basename):
        img = cv2.imread(osp.join(scene_dir, "rgb", basename + ".png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Prefer DA3 dense depth when available and requested.
        # DL3DV 7K uses da3_depth/, DL3DV 1-6K uses dense_depth/.
        # Robustness: if a DA3 file exists but fails to load (partial transfer
        # / corrupt / incomplete .npy header), silently fall through to the
        # next candidate. Final fallback is the original MVS `depth/` so a
        # partially-synced mirror can still train.
        def _try_load(path):
            if not osp.exists(path):
                return None
            try:
                return np.load(path).astype(np.float32)
            except Exception:
                return None

        depth = None
        using_da3 = False
        if self.use_da3_depth:
            depth = _try_load(osp.join(scene_dir, "da3_depth", basename + ".npy"))
            if depth is None:
                depth = _try_load(osp.join(scene_dir, "dense_depth", basename + ".npy"))
            using_da3 = depth is not None
        if depth is None:
            # Original MVS depth (always expected to be present).
            depth = np.load(osp.join(scene_dir, "depth", basename + ".npy")).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        # DA3 dense depth: only apply DA3 sky mask, skip MVS sky/outlier masks
        # MVS depth: apply all masks as before
        if using_da3:
            # DA3 sky detection: sky regions have conf set to 1.0 by
            # _handle_sky_regions (vs non-sky conf >> 1). Use conf <= 1.5
            # as sky proxy. Falls back to da3_sky_mask PNG if available.
            sky_path = osp.join(scene_dir, "da3_sky_mask", basename + ".png")
            sky_img = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED) if osp.exists(sky_path) else None
            if sky_img is not None:
                depth[sky_img >= 127] = 0.0
            else:
                conf = _try_load(osp.join(scene_dir, "da3_conf", basename + ".npy"))
                if conf is not None:
                    if conf.shape[:2] != depth.shape[:2]:
                        conf = cv2.resize(conf, (depth.shape[1], depth.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
                    depth[conf <= 1.5] = 0.0  # sky pixels have conf=1.0
        else:
            sky_path = osp.join(scene_dir, "sky_mask", basename + ".png")
            if osp.exists(sky_path):
                depth[cv2.imread(sky_path, cv2.IMREAD_UNCHANGED) >= 127] = 0.0
            outlier_path = osp.join(scene_dir, "outlier_mask", basename + ".png")
            if osp.exists(outlier_path):
                depth[cv2.imread(outlier_path, cv2.IMREAD_UNCHANGED) >= 127] = 0.0
        depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
        valid_d = depth[depth > 0]
        if valid_d.size > 0:
            p98 = float(np.percentile(valid_d, 98))
            depth[depth > p98] = p98
        cam = np.load(osp.join(scene_dir, "cam", basename + ".npz"))
        return img, depth, cam["pose"].astype(np.float32), cam["intrinsic"].astype(np.float32)

    def _load_mvs_synth_frame(self, scene_dir, basename):
        from safetensors.numpy import load_file
        img = cv2.imread(osp.join(scene_dir, "rgb", basename + ".jpg"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Prefer DA3 dense depth when available and requested
        da3_path = osp.join(scene_dir, "da3_depth", basename + ".npy")
        if self.use_da3_depth and osp.exists(da3_path):
            depth = np.load(da3_path).astype(np.float32)
        else:
            depth = np.load(osp.join(scene_dir, "depth", basename + ".npy")).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth > 1000] = 0.0
        valid_d = depth[depth > 0]
        if valid_d.size > 0:
            p98 = float(np.percentile(valid_d, 98))
            depth[depth > p98] = p98
        cam = load_file(osp.join(scene_dir, "cam", basename + ".safetensor"))
        return img, depth, cam["pose"].astype(np.float32), cam["intrinsics"].astype(np.float32)

    def _load_pointodyssey_frame(self, scene_dir, basename):
        from safetensors.numpy import load_file
        img = cv2.imread(osp.join(scene_dir, "rgb", basename + ".jpg"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = np.load(osp.join(scene_dir, "depth", basename + ".npy")).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth > 1000] = 0.0
        cam = load_file(osp.join(scene_dir, "cam", basename + ".safetensor"))
        return img, depth, cam["pose"].astype(np.float32), cam["intrinsics"].astype(np.float32)

    def _load_dynamic_replica_frame(self, scene_dir, basename):
        """Load DynamicReplica frame with pytorch3d->opencv camera fix."""
        from safetensors.numpy import load_file
        img = cv2.imread(osp.join(scene_dir, "rgb", basename + ".png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = np.load(osp.join(scene_dir, "depth", basename + ".npy")).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        cam = load_file(osp.join(scene_dir, "cam", basename + ".safetensor"))
        pose_raw = cam["pose"].astype(np.float32)
        intrinsics = cam["intrinsics"].astype(np.float32)

        # Fix: pytorch3d convention -> opencv convention (from dynamic_replica.py)
        pytorch3d_w2c = np.linalg.inv(pose_raw).astype(np.float32)
        pytorch3d_w2c[:2, 3] *= -1
        opencv_w2c = np.eye(4, dtype=np.float32)
        opencv_w2c[:3, :3] = pytorch3d_w2c[:3, :3].T
        opencv_w2c[:3, 3] = pytorch3d_w2c[:3, 3]
        opencv_w2c[:2] *= -1
        camera_pose = np.linalg.inv(opencv_w2c).astype(np.float32)

        return img, depth, camera_pose, intrinsics


class StaticScene_NVS_FixedScene(StaticScene_NVS):
    """StaticScene_NVS restricted to one deterministic scene entry by sorted index."""

    def __init__(self, *args, scene_idx=0, **kwargs):
        self.scene_idx = int(scene_idx)
        super().__init__(*args, **kwargs)
        if not self._scene_entries:
            raise ValueError("No static-scene entries available.")
        if self.scene_idx < 0 or self.scene_idx >= len(self._scene_entries):
            raise ValueError(
                f"scene_idx={self.scene_idx} out of range for {len(self._scene_entries)} scenes"
            )
        self._scene_entries = [self._scene_entries[self.scene_idx]]


class StaticScene_NVS_FixedWindows(StaticScene_NVS_FixedScene):
    """StaticScene_NVS restricted to one scene and one deterministic window pair.

    Useful for true overfit training where both the source and target
    trajectories must stay fixed across all training steps.
    """

    def __init__(self, *args, scene_idx=0, src_start=0, tgt_start=None, **kwargs):
        self.fixed_src_start = int(src_start)
        self.fixed_tgt_start = None if tgt_start is None else int(tgt_start)
        super().__init__(*args, scene_idx=scene_idx, **kwargs)

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        return self.load_fixed_windows(
            scene_idx=0,
            src_start=self.fixed_src_start,
            tgt_start=self.fixed_tgt_start,
        )


# ======================================================================
# RE10K_NVS — RealEstate10K, camera-only (no depth/tracks)
# ======================================================================

class RE10K_NVS(EasyDataset):
    """Two-window NVS for RealEstate10K with DA3 depth.

    DA3 depth is the only depth source (no MVS alternative):
      depth/{frame}.npy       — DA3 metric depth (float32, metres)
      sky_mask/{frame}.png    — sky mask (uint8, 255=sky)
      conf/{frame}.npy        — DA3 confidence (unused in dataloader)

    Builds static 3D tracks from source frame-0 depth exactly as
    StaticScene_NVS does.  has_tracks=True, has_depth=True.
    """

    def __init__(
        self,
        ROOT,
        num_views=49,
        resolution=(512, 288),
        steps_per_epoch=1000,
        min_window_gap=10,
        max_window_gap=100,
        reverse_prob=0.5,
        caption_dir=None,
        track_spatial_stride=1,
    ):
        self.ROOT = ROOT
        self.track_spatial_stride = int(track_spatial_stride)
        self._num_views = int(num_views)
        self.steps_per_epoch = int(steps_per_epoch)
        self.min_window_gap = int(min_window_gap)
        self.max_window_gap = int(max_window_gap)
        self.reverse_prob = float(reverse_prob)
        resolved_caption_dir = default_caption_dir_for_dataset(
            "re10k", caption_root=caption_dir
        )
        self._caption_store = (
            ChunkedCaptionStore(resolved_caption_dir, DEFAULT_PROMPT)
            if resolved_caption_dir is not None and osp.isdir(resolved_caption_dir)
            else None
        )

        if isinstance(resolution, int):
            self.output_width = self.output_height = resolution
        elif isinstance(resolution, (tuple, list)):
            self.output_width, self.output_height = int(resolution[0]), int(resolution[1])
        else:
            self.output_width = self.output_height = int(resolution)

        self._scene_entries = self._scan_scenes()
        min_needed = 2 * self._num_views + self.min_window_gap
        self._scene_entries = [s for s in self._scene_entries if len(s["frames"]) >= min_needed]
        if not self._scene_entries:
            raise ValueError(f"No RE10K scenes with >= {min_needed} frames in {ROOT}")
        print(f"[RE10K_NVS] {len(self._scene_entries)} scenes")

    def _scan_scenes(self):
        entries = []
        for scene in sorted(os.listdir(self.ROOT)):
            rgb_dir = osp.join(self.ROOT, scene, "rgb")
            cam_dir = osp.join(self.ROOT, scene, "cam")
            if not osp.isdir(rgb_dir) or not osp.isdir(cam_dir):
                continue
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")],
                key=lambda x: int(x),
            )
            if len(basenames) < 2:
                continue
            entries.append({
                "scene_dir": osp.join(self.ROOT, scene),
                "frames": basenames,
                "sequence_id": scene,
            })
        return entries

    @property
    def _resolutions(self):
        return [(self.output_width, self.output_height)]

    @property
    def num_views(self):
        return self._num_views

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return self._getitem_impl()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"RE10K_NVS: failed after {max_retries} retries: {e}") from e

    def _getitem_impl(self):
        rng = np.random.default_rng()
        scene = self._scene_entries[rng.integers(0, len(self._scene_entries))]
        frames = scene["frames"]
        T = self._num_views
        total = len(frames)

        max_gap = min(self.max_window_gap, total - 2 * T)
        gap = rng.integers(self.min_window_gap, max(self.min_window_gap + 1, max_gap + 1))
        max_src_start = total - 2 * T - gap
        src_start = rng.integers(0, max(1, max_src_start + 1))
        tgt_start = src_start + T + gap

        src_indices = list(range(src_start, src_start + T))
        tgt_indices = list(range(tgt_start, min(tgt_start + T, total)))
        while len(tgt_indices) < T:
            tgt_indices.append(tgt_indices[-1])

        if random.random() < self.reverse_prob:
            src_indices, tgt_indices = tgt_indices, src_indices

        src_frames, src_depths, src_poses, src_K = self._load_window(scene, src_indices)
        tgt_frames, tgt_depths, tgt_poses, tgt_K = self._load_window(scene, tgt_indices)

        tgt_cam2world = np.stack(tgt_poses, axis=0)
        src_cam0_pose = src_poses[0]

        # Build static tracks from source frame-0 DA3 depth (mirrors StaticScene_NVS)
        tracks_cam0, tgt_cam2cam0, common_scale = normalize_to_cam0(
            *self._build_static_tracks(src_depths[0], src_K, src_cam0_pose, tgt_cam2world)
        )

        target_depth = (
            np.stack(tgt_depths, axis=0).astype(np.float32) / common_scale
            if tgt_depths else None
        )
        target_depth_valid = (
            (np.isfinite(target_depth) & (target_depth > 0)).astype(np.float32)
            if target_depth is not None else None
        )

        prompt, prompt_emb = self._resolve_target_caption_with_emb(scene, tgt_indices)
        return {
            "video": src_frames,
            "target_video": tgt_frames,
            "tracks_world": tracks_cam0,
            "target_cam2world": tgt_cam2cam0,
            "target_intrinsics": tgt_K,
            "common_scale": common_scale,
            "track_query_idx": 0,
            "prompt": prompt,
            "prompt_emb": prompt_emb,
            "has_tracks": True,
            "has_dynamic_tracks": False,
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "source_depth": src_depths[0] / common_scale if src_depths else None,
            "target_depth": target_depth,
            "target_depth_valid": target_depth_valid,
            "has_depth": target_depth is not None,
        }

    def _build_static_tracks(self, depth_0, K, src_cam0_pose, tgt_cam2world):
        """Unproject frame-0 depth to world, tile across T, return (tracks_world, tgt_c2w)."""
        T = self._num_views
        H, W = self.output_height, self.output_width
        pts_world_0, valid_0 = depth_to_world_pts3d(depth_0, K, src_cam0_pose)
        s = self.track_spatial_stride
        yy, xx = np.arange(0, H, s), np.arange(0, W, s)
        gy, gx = np.meshgrid(yy, xx, indexing="ij")
        gy, gx = gy.ravel(), gx.ravel()
        pts_grid = pts_world_0[gy, gx]
        valid_grid = valid_0[gy, gx]
        tracks_world = np.tile(pts_grid[:, None, :], (1, T, 1))
        tracks_world[~valid_grid] = 0.0
        return tracks_world, tgt_cam2world, src_cam0_pose

    def _resolve_target_caption(self, scene, target_indices):
        # Legacy text-only path. Prefer _resolve_target_caption_with_emb in
        # __getitem__ to also forward the cached T5 embedding when available.
        text, _ = self._resolve_target_caption_with_emb(scene, target_indices)
        return text

    def _resolve_target_caption_with_emb(self, scene, target_indices):
        """Return (text, emb_or_none). See the matching helper in the static-scene
        sibling class for semantics."""
        if self._caption_store is None:
            return DEFAULT_PROMPT, None
        return self._caption_store.get_with_emb(scene["sequence_id"], target_indices)

    def _load_window(self, scene, indices):
        """Load RGB + DA3 depth + camera for each frame index.

        Returns (frames_pil, depths, poses, K):
          depths[i] — float32 (H, W) depth in metres, sky/invalid zeroed.
        """
        # Defensive guard (see _sanitize_indices docstring): any upstream off-by-
        # one producing negative indices would otherwise wrap-to-end via Python
        # list indexing and look like "video starts from frame 0 partway".
        indices = _sanitize_indices(indices, len(scene["frames"]), self._num_views)
        frames_pil = []
        depths = []
        poses = []
        K_out = None
        for i in indices:
            basename = scene["frames"][i]
            scene_dir = scene["scene_dir"]

            img = cv2.imread(osp.join(scene_dir, "rgb", basename + ".png"), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            depth_path = osp.join(scene_dir, "depth", basename + ".npy")
            depth = np.load(depth_path).astype(np.float32) if osp.exists(depth_path) else None
            if depth is not None:
                depth[~np.isfinite(depth)] = 0.0
                sky_path = osp.join(scene_dir, "sky_mask", basename + ".png")
                if osp.exists(sky_path):
                    depth[cv2.imread(sky_path, cv2.IMREAD_UNCHANGED) >= 127] = 0.0

            cam = np.load(osp.join(scene_dir, "cam", basename + ".npz"))
            K = cam["intrinsics"].astype(np.float32)
            pose = cam["pose"].astype(np.float32)

            img_pil, depth_resized, K_resized = center_crop_and_resize(
                img, depth, K, self.output_width, self.output_height)
            frames_pil.append(img_pil)
            if depth_resized is not None:
                depths.append(depth_resized)
            poses.append(pose)
            if K_out is None:
                K_out = K_resized
        return frames_pil, depths if depths else None, poses, K_out


def read_exr_depth(filepath):
    """Read single-channel (R) float32 depth from an OpenEXR file."""
    import OpenEXR
    import Imath

    exr_file = OpenEXR.InputFile(filepath)
    dw = exr_file.header()["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr_file.channel("R", pt)
    depth = np.frombuffer(raw, dtype=np.float32).reshape(H, W).copy()
    return depth


# ======================================================================
# Objaverse_NVS — Multi-trajectory Objaverse NVS
# ======================================================================

class Objaverse_NVS(EasyDataset):
    """Multi-trajectory NVS for Objaverse objects with depth→3D tracks.

    Each Objaverse object has multiple rendered trajectory folders (camera orbits,
    random trajectories, static views, etc.).  Training picks two random
    trajectories of the SAME object as source/target — no fixed source/target
    distinction.

    Source provides depth (frame-0) → 3D tracks (static, tiled across T frames).
    Target provides RGB + camera poses for supervision.

    Data format:
        ROOT/
            rendered_output_target_randtraj/
                {obj_id}_randtraj/
                    {traj_name}/
                        {traj_name}.mp4      (RGB, ~33 frames)
                        frame{i}.json        (camera: matrix_world, x_fov, y_fov)
                        frame{i}_depth{s:04d}.exr  (z-depth, R channel)
            rendered_output_change_radius_and_azim_dynamic/
                {obj_id}/{traj_name}/...
            ... (other trajectory folder types)
    """

    # Folder type → object-ID suffix
    DYNAMIC_FOLDERS = {
        "rendered_output_target_randtraj": "_randtraj",
        "rendered_output_change_radius_and_azim_dynamic": "",
        "rendered_output_change_radius_and_elevation_dynamic": "",
        "rendered_output_fixed_elevation": "",
        "rendered_output_random_fixed_randomradius": "_randfixed",
    }
    STATIC_FOLDERS = {
        "rendered_output_change_radius_and_elevation": "",
        "rendered_output_change_radius_and_elevation_traj": "",
    }

    def __init__(
        self,
        ROOT,
        train_list,
        num_views=49,
        resolution=(512, 288),
        steps_per_epoch=1000,
        track_spatial_stride=1,
        frames_per_traj=33,
        reverse_prob=0.5,
    ):
        """
        Args:
            ROOT: Parent directory containing rendered_output_* folders.
            train_list: Text file listing object IDs (one per line,
                e.g. "000-092/1b3c337ceec74b049f094ec467601d32").
            frames_per_traj: Number of frames per trajectory render (default 33).
                If num_views > frames_per_traj, last frame is repeated to pad.
        """
        self.ROOT = ROOT
        self._num_views = int(num_views)
        self.steps_per_epoch = int(steps_per_epoch)
        self.track_spatial_stride = int(track_spatial_stride)
        self.frames_per_traj = int(frames_per_traj)
        self.reverse_prob = float(reverse_prob)

        if isinstance(resolution, int):
            self.output_width = self.output_height = resolution
        elif isinstance(resolution, (tuple, list)):
            self.output_width, self.output_height = int(resolution[0]), int(resolution[1])
        else:
            self.output_width = self.output_height = int(resolution)

        # Load object list
        with open(train_list, "r") as f:
            object_ids = [line.strip() for line in f if line.strip()]

        self._object_entries = self._scan_objects(object_ids)
        if not self._object_entries:
            raise ValueError(f"No objects with >= 2 trajectories found in {ROOT}")

        total_trajs = sum(len(e["trajs"]) for e in self._object_entries)
        print(
            f"[Objaverse_NVS] {len(self._object_entries)} objects, "
            f"{total_trajs} total trajectories, "
            f"frames_per_traj={self.frames_per_traj}, "
            f"resolution={self.output_width}x{self.output_height}"
        )

    def _scan_objects(self, object_ids):
        """Discover objects with >= 2 available trajectory folders."""
        all_folders = {**self.DYNAMIC_FOLDERS, **self.STATIC_FOLDERS}
        entries = []
        for obj_id in object_ids:
            trajs = []
            for folder_name, suffix in all_folders.items():
                folder_root = osp.join(self.ROOT, folder_name)
                if not osp.isdir(folder_root):
                    continue
                obj_dir = osp.join(folder_root, f"{obj_id}{suffix}")
                if not osp.isdir(obj_dir):
                    continue
                for traj_name in sorted(os.listdir(obj_dir)):
                    traj_path = osp.join(obj_dir, traj_name)
                    if not osp.isdir(traj_path):
                        continue
                    # Must have at least frame0.json
                    if not osp.exists(osp.join(traj_path, "frame0.json")):
                        continue
                    trajs.append(
                        {
                            "path": traj_path,
                            "name": traj_name,
                            "folder": folder_name,
                            "is_static": folder_name in self.STATIC_FOLDERS,
                        }
                    )
            if len(trajs) >= 2:
                entries.append({"obj_id": obj_id, "trajs": trajs})
        return entries

    # --- EasyDataset interface ---

    @property
    def _resolutions(self):
        return [(self.output_width, self.output_height)]

    @property
    def num_views(self):
        return self._num_views

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return self._getitem_impl()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Objaverse_NVS: failed after {max_retries} retries: {e}"
                    ) from e

    def _getitem_impl(self):
        rng = np.random.default_rng()
        T = self._num_views

        # 1. Pick random object
        obj = self._object_entries[rng.integers(0, len(self._object_entries))]

        # 2. Pick two different trajectories (no source/target distinction)
        pair_idx = rng.choice(len(obj["trajs"]), size=2, replace=False)
        traj_a, traj_b = obj["trajs"][pair_idx[0]], obj["trajs"][pair_idx[1]]

        # Random assignment as source/target
        if random.random() < self.reverse_prob:
            src_traj, tgt_traj = traj_b, traj_a
        else:
            src_traj, tgt_traj = traj_a, traj_b

        # 3. Build frame index list (pad if frames_per_traj < T)
        n_avail = self.frames_per_traj
        frame_ids = list(range(min(n_avail, T)))
        while len(frame_ids) < T:
            frame_ids.append(frame_ids[-1])  # repeat last frame

        # 4. Load source and target (with depth for visualization; no training supervision)
        src_frames, src_depth_0, src_poses, src_K = self._load_trajectory(
            src_traj, frame_ids, load_depth_frame0=True
        )
        tgt_frames, tgt_depth_0, tgt_poses, tgt_K = self._load_trajectory(
            tgt_traj, frame_ids, load_depth_frame0=True
        )

        # 5. Normalize cameras only (no 3D tracks)
        tgt_cam2world = np.stack(tgt_poses, axis=0)  # (T, 4, 4)
        tgt_cam2cam0, common_scale = normalize_cameras_to_cam0(tgt_cam2world, src_poses[0])

        # Empty tracks placeholder (has_tracks=False → pipeline zeros track tokens)
        H, W = self.output_height, self.output_width
        s = self.track_spatial_stride
        N = (H // s) * (W // s)
        tracks_cam0 = np.zeros((N, T, 3), dtype=np.float32)

        # Depth for visualization only (has_depth=False → not used for training supervision)
        source_depth = src_depth_0 / common_scale if src_depth_0 is not None else None
        target_depth_tiled = (
            np.tile(tgt_depth_0[None, ...], (T, 1, 1)).astype(np.float32) / common_scale
            if tgt_depth_0 is not None else None
        )

        return {
            "video": src_frames,
            "target_video": tgt_frames,
            "tracks_world": tracks_cam0,
            "target_cam2world": tgt_cam2cam0,
            "target_intrinsics": tgt_K,
            "common_scale": common_scale,
            "track_query_idx": 0,
            "prompt": DEFAULT_PROMPT,
            "prompt_emb": None,
            "has_tracks": False,
            "has_dynamic_tracks": False,
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "source_depth": source_depth,
            "target_depth": target_depth_tiled,
            "has_depth": False,
        }

    def _load_trajectory(self, traj_info, frame_ids, load_depth_frame0=False):
        """Load RGB frames + camera poses (+ optional frame-0 depth) from a trajectory.

        Returns: (frames_pil, depth_0_or_None, poses, K)
        """
        traj_path = traj_info["path"]
        traj_name = traj_info["name"]
        is_static = traj_info["is_static"]

        # --- Load camera JSONs for requested frames ---
        # Camera data is the same for repeated frame indices, so cache per unique frame
        unique_frames = sorted(set(frame_ids))
        cam_cache = {}
        for fi in unique_frames:
            cam_path = osp.join(traj_path, f"frame{fi}.json")
            if not osp.exists(cam_path):
                raise FileNotFoundError(f"Camera JSON not found: {cam_path}")
            with open(cam_path, "r") as f:
                cam_cache[fi] = json.load(f)

        # --- Load RGB from MP4 ---
        # Try {traj_name}.mp4 first, then any *.mp4
        mp4_path = osp.join(traj_path, f"{traj_name}.mp4")
        if not osp.exists(mp4_path):
            mp4_candidates = [
                f for f in os.listdir(traj_path) if f.endswith(".mp4")
            ]
            if mp4_candidates:
                mp4_path = osp.join(traj_path, mp4_candidates[0])
            else:
                mp4_path = None

        if mp4_path and osp.exists(mp4_path):
            rgb_frames_raw = self._read_mp4_frames(mp4_path)
        else:
            # Fallback: individual PNGs
            rgb_frames_raw = self._read_png_frames(traj_path, unique_frames)

        # --- Process each requested frame ---
        frames_pil = []
        poses = []
        K_out = None
        depth_0 = None

        for fi in frame_ids:
            cam_data = cam_cache[fi]

            # Camera pose: Blender c2w → OpenCV c2w
            c2w = np.array(cam_data["matrix_world"], dtype=np.float32)
            c2w = c2w @ BLENDER_TO_OPENCV

            # Intrinsics from FOV
            x_fov = cam_data.get("x_fov", cam_data.get("fov", 0.8575560450553894))
            y_fov = cam_data.get("y_fov", x_fov)

            # RGB frame
            if fi < len(rgb_frames_raw):
                img_np = rgb_frames_raw[fi]
            else:
                img_np = rgb_frames_raw[-1]  # pad with last

            orig_H, orig_W = img_np.shape[:2]
            fx = orig_W / (2.0 * np.tan(float(x_fov) / 2.0))
            fy = orig_H / (2.0 * np.tan(float(y_fov) / 2.0))
            cx, cy = orig_W / 2.0, orig_H / 2.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            # Depth for frame 0 only (if requested)
            depth = None
            if load_depth_frame0 and fi == frame_ids[0] and depth_0 is None:
                # Static folders always use state 0000; dynamic uses state = frame index
                state_idx = 0 if is_static else fi
                depth_path = osp.join(traj_path, f"frame{fi}_depth{state_idx:04d}.exr")
                if osp.exists(depth_path):
                    depth = read_exr_depth(depth_path)
                    # Clip background / invalid
                    depth[~np.isfinite(depth)] = 0.0
                    depth[depth > 1000] = 0.0
                # else: EXR not found — depth remains None (viz only, not required)

            # Crop + resize (image + optional depth + intrinsics)
            img_pil, depth_resized, K_resized = center_crop_and_resize(
                img_np, depth, K, self.output_width, self.output_height
            )

            frames_pil.append(img_pil)
            poses.append(c2w)
            if K_out is None:
                K_out = K_resized
            if depth_resized is not None:
                depth_0 = depth_resized

        return frames_pil, depth_0, poses, K_out

    @staticmethod
    def _read_mp4_frames(mp4_path):
        """Read all frames from MP4 as list of (H, W, 3) uint8 numpy arrays."""
        reader = imageio.get_reader(mp4_path)
        frames = []
        try:
            for frame in reader:
                frames.append(np.asarray(frame))
        finally:
            reader.close()
        return frames

    @staticmethod
    def _read_png_frames(traj_path, unique_frame_ids):
        """Fallback: read individual PNG files frame{i}.png."""
        frames = {}
        for fi in unique_frame_ids:
            png_path = osp.join(traj_path, f"frame{fi}.png")
            if osp.exists(png_path):
                img = cv2.imread(png_path, cv2.IMREAD_COLOR)
                if img is not None:
                    frames[fi] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not frames:
            raise FileNotFoundError(f"No PNG frames found in {traj_path}")
        # Return as a list indexed by frame number (fill gaps with last)
        max_fi = max(frames.keys())
        result = []
        last_frame = None
        for i in range(max_fi + 1):
            if i in frames:
                last_frame = frames[i]
            if last_frame is not None:
                result.append(last_frame)
        return result


# ======================================================================
# HomographyNVS — Synthetic multi-view from monocular video via homography
# ======================================================================

class HomographyNVS(EasyDataset):
    """Applies random in-plane homography to monocular video frames to simulate multi-view.

    Wraps any depth-capable monocular dataset. For each sample:
    - Source: original frames + depth -> 3D tracks
    - Target: same frames warped by a SINGLE random rotation homography H
              (consistent across all T frames -> same "virtual camera angle" for all frames)
    - Target camera pose: original c2w rotated by R_delta_inv
    - Target depth: source depth warped by H

    The homography simulates the camera rotating in-place around its optical center
    (pan/tilt), equivalent to H = K @ R_delta @ K_inv.
    """

    def __init__(
        self,
        base_dataset,
        max_pan_deg=20.0,
        max_tilt_deg=20.0,
        max_roll_deg=5.0,
        border_mode="constant",
    ):
        """
        Args:
            base_dataset: Any EasyDataset that returns "video", "target_intrinsics", etc.
            max_pan_deg:  Max yaw perturbation in degrees.
            max_tilt_deg: Max pitch perturbation in degrees.
            max_roll_deg: Max roll perturbation in degrees.
            border_mode:  "constant" (zero-fill) or "replicate" for warpPerspective edges.
        """
        self.base_dataset = base_dataset
        self.max_pan_deg = float(max_pan_deg)
        self.max_tilt_deg = float(max_tilt_deg)
        self.max_roll_deg = float(max_roll_deg)
        self.border_mode = (
            cv2.BORDER_CONSTANT if border_mode == "constant" else cv2.BORDER_REPLICATE
        )

    @property
    def _resolutions(self):
        return self.base_dataset._resolutions

    @property
    def num_views(self):
        return self.base_dataset.num_views

    def __len__(self):
        return len(self.base_dataset)

    def set_epoch(self, epoch):
        self.base_dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        return self._apply_homography(sample)

    def _apply_homography(self, sample):
        """Apply a single random rotation homography to generate a synthetic target view."""
        rng = np.random.default_rng()
        T = len(sample["video"])
        K = sample["target_intrinsics"].astype(np.float64)  # (3, 3)
        H_out, W_out = np.array(sample["video"][0]).shape[:2]

        # Random pan/tilt/roll angles (degrees -> radians)
        pan  = np.deg2rad(rng.uniform(-self.max_pan_deg, self.max_pan_deg))
        tilt = np.deg2rad(rng.uniform(-self.max_tilt_deg, self.max_tilt_deg))
        roll = np.deg2rad(rng.uniform(-self.max_roll_deg, self.max_roll_deg))

        # Build rotation matrix R_delta = Ry(pan) @ Rx(tilt) @ Rz(roll)
        R_delta = (
            cv2.Rodrigues(np.array([0.0, pan, 0.0]))[0]
            @ cv2.Rodrigues(np.array([tilt, 0.0, 0.0]))[0]
            @ cv2.Rodrigues(np.array([0.0, 0.0, roll]))[0]
        )

        # H = K @ R_delta @ K_inv  (image-plane homography for in-place rotation)
        H_mat = K @ R_delta @ np.linalg.inv(K)  # (3, 3)
        H_mat_f32 = H_mat.astype(np.float32)

        # Warp each target frame
        target_video = []
        for img_pil in sample["video"]:
            img_np = np.array(img_pil)
            warped = cv2.warpPerspective(
                img_np, H_mat_f32, (W_out, H_out),
                flags=cv2.INTER_LINEAR,
                borderMode=self.border_mode,
            )
            target_video.append(PIL.Image.fromarray(warped))

        # Warp target depth if available
        src_target_depth = sample.get("target_depth")
        if src_target_depth is not None:
            new_depths = []
            for t in range(T):
                d = src_target_depth[t]
                warped_d = cv2.warpPerspective(
                    d, H_mat_f32, (W_out, H_out),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                new_depths.append(warped_d)
            new_target_depth = np.stack(new_depths, axis=0)
            new_has_depth = True
        else:
            new_target_depth = None
            new_has_depth = False

        # New target camera pose: original cam2world @ R_delta_inv embedded in 4x4.
        # R is orthogonal so R_delta_inv = R_delta.T.
        src_c2w = sample["target_cam2world"]  # (T, 4, 4) original target poses
        R_delta_inv = R_delta.T.astype(np.float32)
        R_delta_inv_4x4 = np.eye(4, dtype=np.float32)
        R_delta_inv_4x4[:3, :3] = R_delta_inv
        new_target_cam2world = src_c2w @ R_delta_inv_4x4[None, :, :]  # (T, 4, 4)

        return {
            **sample,
            "target_video": target_video,
            "target_cam2world": new_target_cam2world,
            "target_depth": new_target_depth,
            "has_depth": new_has_depth,
        }


# ======================================================================
# ReCamMaster_NVS — MultiCamVideo / SynCamVideo (UE5, 10 cameras × 81 frames)
# ======================================================================

class ReCamMaster_NVS(EasyDataset):
    """Multi-camera NVS from ReCamMaster/SynCamVideo datasets.

    Each scene has 10 static cameras × 81 frames. Source and target are
    different cameras viewing the same dynamic scene. DA3 depth (scale-aligned
    across cameras) provides the geometric conditioning.

    YAML usage:
        "500 @ ReCamMaster_NVS(ROOT='...', num_views=49)"
    """

    def __init__(
        self,
        ROOT,
        split="train",
        num_views=49,
        resolution=(256, 144),
        steps_per_epoch=500,
        track_spatial_stride=1,
        num_cameras=10,
        num_src_frames=81,
        frame_subsample=1,
        sensor_width_mm=23.76,
        return_source_depth_video=False,
        return_source_rgb_video=False,
        return_source_cameras=False,
        camera_only=False,
        caption_cache_dir=None,
        # 4RC-labeled training cache. See doc/track4d-360/4rc-labeled-recammaster.md.
        fourrc_cache_dir=None,
        fourrc_pair_prob=0.0,
        fourrc_pair_scales_path=None,  # default: <fourrc_cache_dir>/4rc_pair_scales.json
        fourrc_shard_prefix="4rc_labels",
        fourrc_source_dynamic_mask_dir=None,
        fourrc_track_density_mode="dense",
        fourrc_sparse_track_prob=0.0,
        fourrc_sparse_track_stride=8,
        fourrc_max_query_frames=1,
    ):
        self.ROOT = ROOT
        self.split = split
        self._num_views = int(num_views)
        self.output_width, self.output_height = int(resolution[0]), int(resolution[1])
        self.steps_per_epoch = int(steps_per_epoch)
        self.track_spatial_stride = int(track_spatial_stride)
        self.num_cameras = int(num_cameras)
        self.num_src_frames = int(num_src_frames)
        self.frame_subsample = int(frame_subsample)
        self.sensor_width_mm = float(sensor_width_mm)
        self.return_source_depth_video = bool(return_source_depth_video)
        self.return_source_rgb_video = bool(return_source_rgb_video)
        self.return_source_cameras = bool(return_source_cameras)
        self.camera_only = bool(camera_only)
        # Caption cache (precomputed by holo4d.precompute_t5_cache_recam).
        # Layout: <caption_cache_dir>/captions_all.json + either
        #   captions_t5_emb.safetensors (single, small datasets) or
        #   captions_t5_emb.rank*.part*.safetensors (sharded, big datasets).
        # Key format mirrors precompute: f"{split}/{focal}/{scene}/cam{NN:02d}".
        self.caption_cache_dir = (
            None if caption_cache_dir in (None, "") else str(caption_cache_dir)
        )
        # 4RC-labeled cache state. Lazy-loaded on first sample to keep worker
        # startup cheap. fourrc_pair_prob=0.0 disables this path entirely
        # (back-compat with the existing camera_only mode).
        self.fourrc_cache_dir = (
            None if fourrc_cache_dir in (None, "") else str(fourrc_cache_dir)
        )
        self.fourrc_pair_prob = float(fourrc_pair_prob)
        self.fourrc_shard_prefix = str(fourrc_shard_prefix)
        self.fourrc_pair_scales_path = (
            str(fourrc_pair_scales_path) if fourrc_pair_scales_path
            else (osp.join(self.fourrc_cache_dir, "4rc_pair_scales.json") if self.fourrc_cache_dir else None)
        )
        self.fourrc_source_dynamic_mask_dir = (
            None if fourrc_source_dynamic_mask_dir in (None, "") else str(fourrc_source_dynamic_mask_dir)
        )
        self.fourrc_track_density_mode = str(fourrc_track_density_mode or "dense").lower()
        self.fourrc_sparse_track_prob = float(fourrc_sparse_track_prob)
        self.fourrc_sparse_track_stride = int(fourrc_sparse_track_stride)
        self.fourrc_max_query_frames = int(fourrc_max_query_frames)

        self._scene_entries = self._scan_scenes()
        if not self._scene_entries:
            raise ValueError(f"No ReCamMaster scenes found in {ROOT}/{split}")
        mode_tag = "camera-only" if self.camera_only else "depth+tracks"
        cache_tag = f", caption_cache={self.caption_cache_dir}" if self.caption_cache_dir else ""
        fourrc_tag = (
            f", 4rc_cache={self.fourrc_cache_dir} (p={self.fourrc_pair_prob:.2f})"
            if self.fourrc_cache_dir and self.fourrc_pair_prob > 0 else ""
        )
        fg_mask_tag = (
            f", 4rc_fg_masks={self.fourrc_source_dynamic_mask_dir}"
            if self.fourrc_source_dynamic_mask_dir else ""
        )
        sparse_tag = (
            f", 4rc_track_density={self.fourrc_track_density_mode}"
            f"(p_sparse={self.fourrc_sparse_track_prob:.2f}, stride={self.fourrc_sparse_track_stride})"
            if self.fourrc_track_density_mode != "dense" or self.fourrc_sparse_track_prob > 0 else ""
        )
        print(f"[ReCamMaster_NVS] {len(self._scene_entries)} scenes, "
              f"{num_cameras} cams, {num_views} views, mode={mode_tag}{cache_tag}{fourrc_tag}{fg_mask_tag}{sparse_tag}")

    @property
    def _resolutions(self):
        return [(self.output_width, self.output_height)]

    @property
    def num_views(self):
        return self._num_views

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        pass

    def _scan_scenes(self):
        split_dir = osp.join(self.ROOT, self.split)
        entries = []
        for focal in sorted(os.listdir(split_dir)):
            focal_dir = osp.join(split_dir, focal)
            if not osp.isdir(focal_dir) or focal.startswith('.') or focal.endswith('.json'):
                continue
            for scene in sorted(os.listdir(focal_dir)):
                scene_dir = osp.join(focal_dir, scene)
                if not osp.isdir(osp.join(scene_dir, "videos")):
                    continue
                if not self.camera_only and not osp.isdir(osp.join(scene_dir, "da3_depth")):
                    continue
                cam_json = osp.join(scene_dir, "cameras", "camera_extrinsics.json")
                if not osp.isfile(cam_json):
                    continue
                entries.append({
                    "scene_dir": scene_dir,
                    "focal_folder": focal,
                    "scene_name": f"{focal}/{scene}",
                })
        return entries

    @staticmethod
    def _parse_camera_matrix(matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix, dtype=np.float32)

    @staticmethod
    def _ue5_to_opencv_c2w(c2w_ue5):
        c2w = c2w_ue5.T.copy()
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        c2w[:3, 3] /= 100.0
        return c2w

    def _compute_intrinsics(self, focal_folder, W, H):
        import re as _re
        match = _re.search(r"f(\d+(?:\.\d+)?)", focal_folder)
        focal_mm = float(match.group(1)) if match else 24.0
        w, h = float(W), float(H)
        if w >= h:
            eff_w = self.sensor_width_mm
            eff_h = self.sensor_width_mm * (h / w)
        else:
            eff_h = self.sensor_width_mm
            eff_w = self.sensor_width_mm * (w / h)
        fx = (focal_mm / eff_w) * w
        fy = (focal_mm / eff_h) * h
        return np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)

    def _ensure_caption_resources(self):
        """Lazy-init the (text-table, T5-emb-store) pair for caption lookup.

        Cached on self so the per-worker open cost (one captions_all.json
        load + one CachedT5EmbStore mmap) is paid once, not per __getitem__.
        """
        if not self.caption_cache_dir:
            return None, None
        emb_store = getattr(self, "_emb_store", None)
        text_table = getattr(self, "_caption_text_table", None)
        if emb_store is not None and text_table is not None:
            return text_table, emb_store
        from .mixed_nvs_captions import CachedT5EmbStore  # lazy
        emb_store = CachedT5EmbStore(self.caption_cache_dir)
        json_path = osp.join(self.caption_cache_dir, "captions_all.json")
        if osp.isfile(json_path):
            try:
                with open(json_path) as f:
                    text_table = json.load(f)
            except Exception:
                text_table = {}
        else:
            text_table = {}
        self._emb_store = emb_store
        self._caption_text_table = text_table
        return text_table, emb_store

    def _load_camera_caption(self, scene_name, cam_id):
        """Return (text, prompt_emb_or_None) for the given (scene, cam).

        ``scene_name`` is the entry's ``"{focal}/{scene}"`` (matches what
        precompute_t5_cache_recam wrote: keys ``f"{split}/{focal}/{scene}/cam{NN:02d}"``).

        Soft fallback (2026-04-30 evening): on cache miss, return
        (DEFAULT_PROMPT, None). The trainer's strict-cache check at
        wan_video.py:518 will find DEFAULT_PROMPT in
        pipe.precomputed_prompt_embs (set up by
        train_track4d360._setup_prompt_emb_cache) and serve the
        precomputed embedding for the unified NVS default — no crash, but
        also no silent loss of training signal because the prompt text
        and emb agree.
        """
        if not self.caption_cache_dir:
            return DEFAULT_PROMPT, None
        cache = getattr(self, "_caption_text_cache", None)
        if cache is None:
            cache = {}
            self._caption_text_cache = cache
        key = (scene_name, int(cam_id))
        if key in cache:
            return cache[key]
        text_table, emb_store = self._ensure_caption_resources()
        emb_key = f"{self.split}/{scene_name}/cam{int(cam_id):02d}"
        emb = emb_store.get_by_key(emb_key) if (emb_store and emb_store.available) else None
        _ds_label = f"ReCamMaster_NVS[{osp.basename(self.caption_cache_dir.rstrip('/'))}]"
        if emb is None:
            report_cache_miss(_ds_label, f"emb_key={emb_key!r}")
            cache[key] = (DEFAULT_PROMPT, None)
            return cache[key]
        text = text_table.get(emb_key) if text_table else None
        if not text:
            report_cache_miss(
                _ds_label,
                f"emb_key={emb_key!r} (emb present, text absent in captions_all.json)",
            )
            text = DEFAULT_PROMPT
        cache[key] = (text, emb)
        return cache[key]

    # -----------------------------------------------------------------
    # 4RC-labeled cache helpers
    # See doc/track4d-360/4rc-labeled-recammaster.md for the design.
    # -----------------------------------------------------------------
    def _ensure_fourrc_resources(self):
        """Lazy-init: scan all shards once, build (rel_dir → {cam → shard_name})
        and load pair_scales JSON. Cached on self for the worker's lifetime."""
        if not (self.fourrc_cache_dir and self.fourrc_pair_prob > 0):
            return None
        if getattr(self, "_fourrc_index", None) is not None:
            return self._fourrc_index
        from safetensors import safe_open as _safe_open
        cache_root = self.fourrc_cache_dir
        prefix = self.fourrc_shard_prefix
        idx_path = osp.join(cache_root, f"{prefix}_shard_index.json")
        scenes: dict = {}
        if osp.isfile(idx_path):
            with open(idx_path) as f:
                key_to_shard = json.load(f)
            for k, shard in key_to_shard.items():
                if not k.endswith("/extrinsic"):
                    continue
                parts = k.split("/")
                cam = parts[-2]
                rel_dir = "/".join(parts[:-2])
                scenes.setdefault(rel_dir, {})[cam] = shard
        else:
            # Fallback scan if index hasn't been merged.
            from glob import glob as _glob
            for sp in sorted(_glob(osp.join(cache_root, f"{prefix}.*.part*.safetensors"))):
                with _safe_open(sp, framework="pt") as f:
                    for k in f.keys():
                        if not k.endswith("/extrinsic"):
                            continue
                        parts = k.split("/")
                        cam = parts[-2]
                        rel_dir = "/".join(parts[:-2])
                        scenes.setdefault(rel_dir, {})[cam] = osp.basename(sp)
        # Load pair scales (small JSON; loaded once per worker).
        pair_scales: dict = {}
        if self.fourrc_pair_scales_path and osp.isfile(self.fourrc_pair_scales_path):
            with open(self.fourrc_pair_scales_path) as f:
                pair_scales = json.load(f)
        self._fourrc_index = scenes
        self._fourrc_pair_scales = pair_scales
        # Pair scales are source-anchored. The label cache only needs to tell
        # us which cameras can act as 4RC source videos; the target camera may
        # be any other GT camera in the scene. Older pair-scale files are keyed
        # as rel_dir|src_cam|some_labeled_tgt, but the value is estimated from
        # source motion only, so collapse them to rel_dir|src_cam here.
        scenes_with_scale = []
        source_scales: dict[tuple[str, str], float] = {}
        if pair_scales:
            for key, value in pair_scales.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    continue
                parts = key.split("|")
                if len(parts) != 3:
                    continue
                rel_dir, src_cam, _tgt_cam = parts
                source_scales[(rel_dir, src_cam)] = float(value)
            scenes_with_scale = [
                rel_dir for rel_dir, cams in scenes.items()
                if any((rel_dir, cam) in source_scales for cam in cams.keys())
            ]
        # Fallback: if there's no scales JSON (yet), expose all scenes so
        # the worker can still try; _getitem_4rc_pair will skip on miss.
        self._fourrc_source_scales = source_scales
        self._fourrc_scene_keys = scenes_with_scale or list(scenes.keys())
        if pair_scales:
            print(f"[ReCamMaster_NVS] 4rc filter: {len(self._fourrc_scene_keys)} scenes "
                  f"have source-scales out of {len(scenes)} cached")
        return scenes

    def _load_fourrc_task(self, rel_dir: str, cam: str) -> dict:
        """Read the 5 stored fields for one (rel_dir, cam) task from cache."""
        from safetensors import safe_open as _safe_open
        shard = self._fourrc_index[rel_dir][cam]
        sp = osp.join(self.fourrc_cache_dir, shard)
        out: dict = {}
        with _safe_open(sp, framework="pt") as f:
            for field in ("depth", "track", "extrinsic", "intrinsic",
                          "track_query_idx", "start_frame"):
                out[field] = f.get_tensor(f"{rel_dir}/{cam}/{field}")
        return out

    @staticmethod
    def _unpack_dynamic_mask_npz(npz_path: str) -> np.ndarray:
        """Load a SAM3 foreground sidecar saved as np.packbits."""
        with np.load(npz_path) as data:
            shape = tuple(int(v) for v in data["shape"])
            packed = data["dynamic_mask_packed"]
        mask = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape)
        return mask.astype(bool, copy=False)

    def _load_fourrc_source_dynamic_mask(self, rel_dir: str, cam: str, start_frame: int, num_frames: int):
        """Load source-view semantic foreground masks for the 4RC source clip.

        The production SAM3 cache is full-video (81 frames, start_frame=0),
        while the 4RC dense label picks a 49-frame source window. Slice using
        the cached 4RC start frame, then resize with nearest-neighbor to the
        current training canvas if the mask cache was produced at a lower
        inspection resolution.
        """
        if not self.fourrc_source_dynamic_mask_dir:
            return None
        npz_path = osp.join(
            self.fourrc_source_dynamic_mask_dir,
            rel_dir,
            cam,
            "dynamic_mask_packbits.npz",
        )
        if not osp.isfile(npz_path):
            return None
        try:
            mask = self._unpack_dynamic_mask_npz(npz_path)
        except Exception:
            return None
        start_frame = int(start_frame)
        end_frame = start_frame + int(num_frames)
        if mask.ndim != 3 or mask.shape[0] < end_frame:
            return None
        mask = mask[start_frame:end_frame]
        if mask.shape[1:] != (self.output_height, self.output_width):
            mask = np.stack([
                cv2.resize(
                    m.astype(np.uint8),
                    (self.output_width, self.output_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool, copy=False)
                for m in mask
            ], axis=0)
        return mask.astype(bool, copy=False)

    def _getitem_4rc_pair(self, rng):
        """Load a 4RC-labeled (src, tgt) v2v pair.

        Anchors on the source's `start_frame` (cached at label time). The
        cached cameras are SOURCE candidates only. The target camera is sampled
        from all GT cameras in the scene, then both src/tgt RGB windows are read
        at the source's exact cached 49-frame slice so the 4RC tracks/depth and
        RGB frames remain temporally aligned.

        Returns None on any cache miss / file-not-found / shape mismatch so
        the caller can fall back to the standard camera_only path.
        """
        try:
            scene_keys = self._fourrc_scene_keys
            if not scene_keys:
                return None
            rel_dir = scene_keys[rng.integers(len(scene_keys))]
            scene_cams = self._fourrc_index.get(rel_dir, {})
            source_scales = getattr(self, "_fourrc_source_scales", {})
            src_candidates = [
                cam for cam in sorted(scene_cams.keys())
                if not source_scales or (rel_dir, cam) in source_scales
            ]
            if not src_candidates:
                return None
            src_cam_name = src_candidates[int(rng.integers(len(src_candidates)))]

            # Target can be any other ReCam camera, not just one of the 4RC
            # labeled cameras. The 4RC labels are used only for source
            # reprojection; target RGB/camera are read from GT.
            all_cam_names = [f"cam{i:02d}" for i in range(1, self.num_cameras + 1)]
            target_candidates = [cam for cam in all_cam_names if cam != src_cam_name]
            if not target_candidates:
                return None
            tgt_cam_name = target_candidates[int(rng.integers(len(target_candidates)))]

            src_data = self._load_fourrc_task(rel_dir, src_cam_name)
            T = self._num_views
            src_start = int(src_data["start_frame"].item())

            depth = src_data["depth"].to(torch.float32).cpu().numpy()         # (T, Hs, Ws)
            track = src_data["track"].to(torch.float32).cpu().numpy()         # (T, Hs, Ws, 3)
            src_extr_4rc = src_data["extrinsic"].cpu().numpy()                # (T, 4, 4)
            K_save = src_data["intrinsic"].cpu().numpy()                      # (T, 3, 3)
            if depth.shape[0] < T or track.shape[0] < T:
                return None
            depth = depth[:T]; track = track[:T]
            Hs_cache, Ws_cache = depth.shape[1:]

            # Upsample cached 4RC fields (saved at 144×256 by default) to the
            # dataset's requested output resolution. depth + track use bilinear
            # to match the same channel-wise smooth-field convention as
            # `visualize_4rc_safetensors_pairs.py`. Intrinsic gets scaled.
            Hs, Ws = self.output_height, self.output_width
            if (Hs, Ws) != (Hs_cache, Ws_cache):
                depth = np.stack([
                    cv2.resize(depth[t], (Ws, Hs), interpolation=cv2.INTER_LINEAR)
                    for t in range(T)
                ], axis=0)
                track = np.stack([
                    cv2.resize(track[t], (Ws, Hs), interpolation=cv2.INTER_LINEAR)
                    for t in range(T)
                ], axis=0)
                sx = float(Ws) / float(Ws_cache)
                sy = float(Hs) / float(Hs_cache)
                K_save = K_save.copy()
                K_save[:, 0, 0] *= sx; K_save[:, 0, 2] *= sx
                K_save[:, 1, 1] *= sy; K_save[:, 1, 2] *= sy

            # ---- GT cameras at the source's window ----
            scene_dir = osp.join(self.ROOT, rel_dir)
            cam_json_path = osp.join(scene_dir, "cameras", "camera_extrinsics.json")
            if not osp.isfile(cam_json_path):
                return None
            with open(cam_json_path) as f:
                cam_data = json.load(f)
            src_cam_int = int(src_cam_name.replace("cam", ""))
            tgt_cam_int = int(tgt_cam_name.replace("cam", ""))
            gt_src_c2w = np.stack([
                self._ue5_to_opencv_c2w(self._parse_camera_matrix(
                    cam_data[f"frame{src_start + i}"][src_cam_name]))
                for i in range(T)
            ], axis=0).astype(np.float32)
            gt_tgt_c2w = np.stack([
                self._ue5_to_opencv_c2w(self._parse_camera_matrix(
                    cam_data[f"frame{src_start + i}"][tgt_cam_name]))
                for i in range(T)
            ], axis=0).astype(np.float32)

            # ---- source_scale to align GT translations with 4RC depth scale ----
            # Scale is anchored only on the selected source 4RC clip. It does
            # not depend on which GT target camera was sampled.
            pair_scale = float(source_scales.get((rel_dir, src_cam_name), 0.0) or 0.0)
            if pair_scale <= 0.0:
                # Backward-compatible fallback for legacy directed lookups.
                pair_key = f"{rel_dir}|{src_cam_name}|{tgt_cam_name}"
                pair_scale = float(self._fourrc_pair_scales.get(pair_key, 0.0) or 0.0)
            if pair_scale <= 0.0:
                return None  # missing source scale → fall back to camera_only

            # ---- Read MP4 frames for both cams at src_start window ----
            src_mp4 = osp.join(scene_dir, "videos", f"{src_cam_name}.mp4")
            tgt_mp4 = osp.join(scene_dir, "videos", f"{tgt_cam_name}.mp4")
            src_frames_raw = self._read_mp4_frames(src_mp4, self.num_src_frames)
            tgt_frames_raw = self._read_mp4_frames(tgt_mp4, self.num_src_frames)
            if len(src_frames_raw) < src_start + T or len(tgt_frames_raw) < src_start + T:
                return None
            H_orig, W_orig = src_frames_raw[0].shape[:2]
            # Center-crop + resize to (Ws, Hs). Use GT-derived intrinsics
            # (from focal-folder + raw resolution + crop), NOT 4RC's
            # predicted intrinsic — those differ by up to ~30% and would
            # break Plucker rays / dense reprojection downstream.
            dummy_d = np.zeros((H_orig, W_orig), np.float32)
            K_orig = self._compute_intrinsics(rel_dir.split("/")[1], W_orig, H_orig)
            src_frames, tgt_frames = [], []
            K_gt_cropped = None
            for i in range(T):
                fi = src_start + i
                src_pil, _, K_r = center_crop_and_resize(
                    src_frames_raw[fi], dummy_d, K_orig, Ws, Hs)
                tgt_pil, _, _ = center_crop_and_resize(
                    tgt_frames_raw[fi], dummy_d, K_orig, Ws, Hs)
                src_frames.append(src_pil)
                tgt_frames.append(tgt_pil)
                if K_gt_cropped is None:
                    K_gt_cropped = K_r  # all frames share the same crop, K_r is constant

            # ---- Express cams in src0 frame, scale GT translations ----
            world_to_src0 = np.linalg.inv(gt_src_c2w[0])
            src_cam2src0 = (world_to_src0[None] @ gt_src_c2w).astype(np.float32)
            tgt_cam2src0 = (world_to_src0[None] @ gt_tgt_c2w).astype(np.float32)
            src_cam2src0[:, :3, 3] *= pair_scale
            tgt_cam2src0[:, :3, 3] *= pair_scale

            # ---- Re-express 4RC tracks/depth in src cam_query frame ----
            # 4RC pred_track is in 4RC's own world frame (anchored at the
            # source clip's frame 0, but with a small non-identity offset).
            # Transform via inv(src_extr_4rc[query_idx]) to put tracks in
            # the source query camera frame, which we then identify with
            # GT src cam 0 (after pair_scale alignment). Same operation as
            # `transform_run_to_query_camera` in the visualizer.
            src_query_c2w_4rc = src_extr_4rc[int(src_data["track_query_idx"].item())]
            src_cam2src0_4rc = (
                np.linalg.inv(src_query_c2w_4rc)[None] @ src_extr_4rc[:T]
            ).astype(np.float32)
            R_q = src_query_c2w_4rc[:3, :3]
            t_q = src_query_c2w_4rc[:3, 3]
            # world_to_camera (row-vector form): pts_cam = (pts - t) @ R
            tracks_in_src0 = (track.reshape(-1, 3) - t_q[None, :]) @ R_q
            tracks_in_src0 = tracks_in_src0.reshape(track.shape)
            # (T, Hs, Ws, 3) -> (HW, T, 3) — the layout the dense conditioning
            # branch and visualize_multiframe_warp.py expect.
            tracks_in_src0 = tracks_in_src0.transpose(1, 2, 0, 3).reshape(-1, T, 3).astype(np.float32)

            common_scale = 1.0  # everything already in 4RC scale via pair_scale

            # ---- Assemble result, mirroring the depth+tracks branch ----
            prompt_text, prompt_emb = self._load_camera_caption(rel_dir.split("/", 1)[-1], tgt_cam_int)
            # NOTE: target-view depth supervision is unavailable on this path.
            # 4RC labels the source clip independently; we don't have target's
            # depth in the source's query-frame coords. Set has_depth=False
            # so the trainer's depth loss is skipped, but keep
            # source_depth_video populated for the dense-branch conditioning.
            result = {
                "video": src_frames,
                "target_video": tgt_frames,
                "tracks_world": tracks_in_src0,
                "target_cam2world": tgt_cam2src0,
                "target_intrinsics": K_gt_cropped.astype(np.float32),
                "common_scale": float(common_scale),
                "track_query_idx": int(src_data["track_query_idx"].item()),
                "camera_only_tier": False,
                "prompt": prompt_text,
                "prompt_emb": prompt_emb,
                "has_tracks": True,
                "has_dynamic_tracks": True,
                "has_depth": False,            # no target-view 4RC depth available
                "track_video_loss_mask": 1.0,
                "fourrc_rel_dir": rel_dir,
                "fourrc_source_cam_name": src_cam_name,
                "fourrc_target_cam_name": tgt_cam_name,
                "fourrc_start_frame": int(src_start),
                "fourrc_source_scale": float(pair_scale),
            }
            query_indices = _choose_fourrc_training_query_indices(
                rng,
                max_queries=self.fourrc_max_query_frames,
                allow_three=True,
            )
            result["fourrc_training_query_indices"] = np.asarray(query_indices, dtype=np.int32)
            density_mode = self.fourrc_track_density_mode
            use_sparse = density_mode == "sparse" or (
                density_mode == "mixed"
                and rng.random() < max(0.0, min(1.0, self.fourrc_sparse_track_prob))
            )
            if use_sparse:
                sparse_valid = _build_sparse_track_valid_mask(
                    Hs,
                    Ws,
                    T,
                    stride=self.fourrc_sparse_track_stride,
                    rng=rng,
                )
                depth_valid_at_tracks = (depth.reshape(T, Hs * Ws).T > 0)
                result["tracks_world_valid_mask"] = (
                    sparse_valid & depth_valid_at_tracks
                ).astype(bool, copy=False)
                result["track_density_mode"] = "sparse"
            else:
                result["track_density_mode"] = "dense"
            if self.return_source_depth_video:
                result["source_depth_video"] = depth.astype(np.float32)
                result["source_depth_valid"] = (depth > 0).astype(bool)
            if self.return_source_rgb_video:
                result["source_rgb_video"] = src_frames
            source_dynamic_mask = self._load_fourrc_source_dynamic_mask(
                rel_dir, src_cam_name, src_start, T,
            )
            if source_dynamic_mask is not None:
                result["source_dynamic_mask"] = source_dynamic_mask
            if self.return_source_cameras:
                # Source depth/tracks are 4RC predictions, so source poses
                # used for lifting source_depth_video must be 4RC-relative
                # poses. GT source poses are only used to anchor target GT
                # cameras into the source-query frame and estimate scale.
                result["source_cam2world"] = src_cam2src0_4rc
                result["source_intrinsics"] = K_save[0].astype(np.float32)
                # 4RC's predicted K, exposed for the dense-branch / viz to
                # audit label geometry when needed.
                result["source_intrinsics_4rc"] = K_save[0].astype(np.float32)
                # GT crop K — matches the actual target image, used by code
                # paths that project source RGB/depth into target view.
                result["source_intrinsics_gt"] = K_gt_cropped.astype(np.float32)
            return result
        except Exception:
            # Any cache miss / shape issue / IO error → fall through.
            return None

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return self._getitem_impl()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"ReCamMaster_NVS: failed after {max_retries} retries: {e}") from e

    def _getitem_impl(self):
        rng = np.random.default_rng()

        # 4RC-labeled branch — opt-in via `fourrc_pair_prob > 0`.
        # Falls through to the existing camera_only / depth+tracks code path
        # if (a) 4RC cache wasn't configured, (b) the random draw landed below
        # the prob, (c) the chosen scene has no 4RC labels (cache miss), or
        # (d) `_getitem_4rc_pair` returns None for any reason.
        if self.fourrc_cache_dir and self.fourrc_pair_prob > 0:
            self._ensure_fourrc_resources()
            if rng.random() < self.fourrc_pair_prob and self._fourrc_index:
                sample = self._getitem_4rc_pair(rng)
                if sample is not None:
                    return sample

        scene = self._scene_entries[rng.integers(len(self._scene_entries))]
        scene_dir = scene["scene_dir"]
        focal_folder = scene["focal_folder"]

        # Pick two random cameras
        cam_indices = list(range(1, self.num_cameras + 1))
        src_cam, tgt_cam = rng.choice(cam_indices, size=2, replace=False)

        # Load camera JSON
        with open(osp.join(scene_dir, "cameras", "camera_extrinsics.json")) as f:
            cam_data = json.load(f)

        # Determine frame indices (subsample if needed)
        total_frames = min(self.num_src_frames, len(cam_data))
        all_frame_ids = list(range(0, total_frames, self.frame_subsample))
        T = self._num_views
        if len(all_frame_ids) < T:
            # Repeat last frame
            all_frame_ids += [all_frame_ids[-1]] * (T - len(all_frame_ids))
        # Random start
        max_start = len(all_frame_ids) - T
        start = rng.integers(0, max(1, max_start + 1))
        frame_ids = all_frame_ids[start:start + T]

        # Read frames from MP4
        src_frames_raw = self._read_mp4_frames(
            osp.join(scene_dir, "videos", f"cam{src_cam:02d}.mp4"), total_frames)
        tgt_frames_raw = self._read_mp4_frames(
            osp.join(scene_dir, "videos", f"cam{tgt_cam:02d}.mp4"), total_frames)

        # Load poses for selected frames
        src_poses, tgt_poses = [], []
        for fi in frame_ids:
            fk = f"frame{fi}"
            src_mat = self._parse_camera_matrix(cam_data[fk][f"cam{src_cam:02d}"])
            tgt_mat = self._parse_camera_matrix(cam_data[fk][f"cam{tgt_cam:02d}"])
            src_poses.append(self._ue5_to_opencv_c2w(src_mat))
            tgt_poses.append(self._ue5_to_opencv_c2w(tgt_mat))

        # Intrinsics (from original video resolution, then adjust after crop+resize)
        H_orig, W_orig = src_frames_raw[0].shape[:2]
        K_orig = self._compute_intrinsics(focal_folder, W_orig, H_orig)

        tgt_poses_arr = np.stack(tgt_poses, axis=0)
        src_cam0_pose = src_poses[0]

        if self.camera_only:
            # Camera-only mode: dynamic scenes where depth/tracks are unreliable.
            # Skip all depth I/O; use camera baselines for scale; return zero tracks.
            src_frames, tgt_frames = [], []
            K_out = None
            dummy_d = np.zeros((H_orig, W_orig), np.float32)
            for fi in frame_ids:
                src_pil, _, K_r = center_crop_and_resize(
                    src_frames_raw[fi], dummy_d, K_orig, self.output_width, self.output_height)
                tgt_pil, _, _ = center_crop_and_resize(
                    tgt_frames_raw[fi], dummy_d, K_orig, self.output_width, self.output_height)
                src_frames.append(src_pil)
                tgt_frames.append(tgt_pil)
                if K_out is None:
                    K_out = K_r

            tgt_cam2cam0, common_scale = normalize_cameras_to_cam0(tgt_poses_arr, src_cam0_pose)
            s = self.track_spatial_stride
            N_pts = len(range(0, self.output_height, s)) * len(range(0, self.output_width, s))
            tracks_cam0 = np.zeros((N_pts, T, 3), dtype=np.float32)

            prompt_text, prompt_emb = self._load_camera_caption(scene["scene_name"], tgt_cam)
            result = {
                "video": src_frames,
                "target_video": tgt_frames,
                "tracks_world": tracks_cam0,
                "target_cam2world": tgt_cam2cam0,
                "target_intrinsics": K_out,
                "common_scale": common_scale,
                "track_query_idx": 0,
                "camera_only_tier": True,
                "prompt": prompt_text,
                "prompt_emb": prompt_emb,
                "has_tracks": False,
                "has_dynamic_tracks": False,
                "has_depth": False,
                "track_video_loss_mask": 0.0,
            }
            if self.return_source_cameras:
                src_poses_arr = np.stack(src_poses, axis=0)
                src_cam2cam0, _ = normalize_cameras_to_cam0(src_poses_arr, src_cam0_pose)
                src_cam2cam0[:, :3, 3] /= common_scale
                result["source_cam2world"] = src_cam2cam0
                result["source_intrinsics"] = K_out
            if self.return_source_rgb_video:
                result["source_rgb_video"] = src_frames
            if self.return_source_depth_video:
                # Camera-only ReCam/SynCam samples intentionally skip depth I/O.
                # Return an all-invalid placeholder so the dense branch keeps
                # the same trainable graph shape as depth-capable samples.
                source_depth = np.zeros((T, self.output_height, self.output_width), dtype=np.float32)
                result["source_depth_video"] = source_depth
                result["source_depth_valid"] = np.zeros_like(source_depth, dtype=bool)
            return result

        # Depth + tracks mode (default)
        src_depth_dir = osp.join(scene_dir, "da3_depth", f"cam{src_cam:02d}")
        tgt_depth_dir = osp.join(scene_dir, "da3_depth", f"cam{tgt_cam:02d}")

        # Build source and target frame lists with crop+resize
        src_frames, tgt_frames = [], []
        src_depths, tgt_depths = [], []
        K_out = None

        for i, fi in enumerate(frame_ids):
            # Source
            src_img = src_frames_raw[fi]
            src_d_path = osp.join(src_depth_dir, f"{fi:05d}.npy")
            src_d = np.load(src_d_path).astype(np.float32) if osp.exists(src_d_path) else np.zeros((H_orig, W_orig), np.float32)
            src_d[~np.isfinite(src_d)] = 0.0
            src_pil, src_d_r, K_r = center_crop_and_resize(src_img, src_d, K_orig, self.output_width, self.output_height)
            src_frames.append(src_pil)
            src_depths.append(src_d_r)
            if K_out is None:
                K_out = K_r

            # Target
            tgt_img = tgt_frames_raw[fi]
            tgt_d_path = osp.join(tgt_depth_dir, f"{fi:05d}.npy")
            tgt_d = np.load(tgt_d_path).astype(np.float32) if osp.exists(tgt_d_path) else np.zeros((H_orig, W_orig), np.float32)
            tgt_d[~np.isfinite(tgt_d)] = 0.0
            tgt_pil, tgt_d_r, _ = center_crop_and_resize(tgt_img, tgt_d, K_orig, self.output_width, self.output_height)
            tgt_frames.append(tgt_pil)
            tgt_depths.append(tgt_d_r)

        # Build static tracks from source frame-0 depth + normalize
        tracks_world = self._build_tracks(src_depths[0], K_out, src_cam0_pose, T)
        tracks_cam0, tgt_cam2cam0, common_scale = normalize_to_cam0(
            tracks_world, tgt_poses_arr, src_cam0_pose,
        )

        source_depth = src_depths[0] / max(common_scale, 1e-6)
        target_depth = np.stack(tgt_depths, axis=0).astype(np.float32) / max(common_scale, 1e-6)

        prompt_text, prompt_emb = self._load_camera_caption(scene["scene_name"], tgt_cam)
        result = {
            "video": src_frames,
            "target_video": tgt_frames,
            "tracks_world": tracks_cam0,
            "target_cam2world": tgt_cam2cam0,
            "target_intrinsics": K_out,
            "common_scale": common_scale,
            "track_query_idx": 0,
            "camera_only_tier": False,
            "prompt": prompt_text,
            "prompt_emb": prompt_emb,
            "has_tracks": True,
            "has_dynamic_tracks": False,
            "has_depth": True,
            "source_depth": source_depth,
            "target_depth": target_depth,
            "track_video_loss_mask": 1.0,
        }

        # Optional source dense fields
        if self.return_source_depth_video:
            src_d_arr = np.stack(src_depths, axis=0).astype(np.float32) / max(common_scale, 1e-6)
            result["source_depth_video"] = src_d_arr
        if self.return_source_rgb_video:
            result["source_rgb_video"] = src_frames
        if self.return_source_cameras:
            src_poses_arr = np.stack(src_poses, axis=0)
            src_cam2cam0, _ = normalize_cameras_to_cam0(src_poses_arr, src_cam0_pose)
            result["source_cam2world"] = src_cam2cam0
            result["source_intrinsics"] = K_out

        return result

    def _build_tracks(self, depth_0, K, src_cam0_pose, T):
        """Build static tracks from frame-0 depth."""
        H, W = self.output_height, self.output_width
        pts_world_0, valid_0 = depth_to_world_pts3d(depth_0, K, src_cam0_pose)
        s = self.track_spatial_stride
        yy, xx = np.arange(0, H, s), np.arange(0, W, s)
        gy, gx = np.meshgrid(yy, xx, indexing="ij")
        gy, gx = gy.ravel(), gx.ravel()
        pts_grid = pts_world_0[gy, gx]
        valid_grid = valid_0[gy, gx]
        tracks_world = np.tile(pts_grid[:, None, :], (1, T, 1))
        tracks_world[~valid_grid] = 0.0
        return tracks_world

    @staticmethod
    def _read_mp4_frames(mp4_path, max_frames):
        reader = imageio.get_reader(mp4_path)
        frames = []
        for f in reader:
            frames.append(f)
            if len(frames) >= max_frames:
                break
        reader.close()
        return frames


class ReCamMasterCowVGGT_NVS(EasyDataset):
    """ReCamMaster training pairs labeled by CoWTracker + VGGT-Omega.

    This is intentionally separate from ReCamMaster_NVS' 4RC branch. The cache
    is inference-like: CoWTracker dense 2D tracks queried only at frame 0,
    lifted through VGGT-Omega source49 depth/cameras and aligned with all10:s8.
    Therefore this dataset always emits dense q=[0] tracks and does not
    synthesize multi-query ReCam labels.
    """

    def __init__(
        self,
        ROOT,
        vggt_root,
        cow_root,
        source_vggt_root=None,
        pair1x49_label_root=None,
        geometry_source="all10_sim3",
        fg_mask_root=None,
        split="train",
        num_views=49,
        resolution=(256, 144),
        steps_per_epoch=5000,
        num_cameras=10,
        num_src_frames=81,
        sensor_width_mm=23.76,
        caption_cache_dir=None,
        all10_rel_frames="0:49:8",
        cow_conf_threshold=0.1,
        return_source_depth_video=True,
        return_source_rgb_video=True,
        return_source_cameras=True,
        max_index_scan=None,
        vggt_post_crop_mode="output_aspect",
        rgb_post_crop_mode="output_aspect",
        track_stride=1,
        track_samples=0,
        track_sample_mode="all",
        track_sample_seed=0,
        scene_offset_per_k=0,
        scene_limit_per_k=0,
        deterministic_index=False,
    ):
        self.ROOT = str(ROOT)
        self.vggt_root = Path(vggt_root)
        self.source_vggt_root = self.vggt_root if source_vggt_root in (None, "") else Path(source_vggt_root)
        self.pair1x49_label_root = None if pair1x49_label_root in (None, "") else Path(pair1x49_label_root)
        self.geometry_source = str(geometry_source)
        self.cow_root = Path(cow_root)
        self.fg_mask_root = None if fg_mask_root in (None, "") else Path(fg_mask_root)
        self.split = split
        self._num_views = int(num_views)
        self.output_width, self.output_height = int(resolution[0]), int(resolution[1])
        self.steps_per_epoch = int(steps_per_epoch)
        self.num_cameras = int(num_cameras)
        self.num_src_frames = int(num_src_frames)
        self.sensor_width_mm = float(sensor_width_mm)
        self.caption_cache_dir = None if caption_cache_dir in (None, "") else str(caption_cache_dir)
        self.all10_rel_frames = str(all10_rel_frames)
        self.all10_tag = _recam_cow_all10_rel_frames_tag(self.all10_rel_frames)
        self.cow_conf_threshold = float(cow_conf_threshold)
        self.return_source_depth_video = bool(return_source_depth_video)
        self.return_source_rgb_video = bool(return_source_rgb_video)
        self.return_source_cameras = bool(return_source_cameras)
        self.max_index_scan = None if max_index_scan in (None, "") else int(max_index_scan)
        self.vggt_post_crop_mode = str(vggt_post_crop_mode)
        self.rgb_post_crop_mode = str(rgb_post_crop_mode)
        self.track_stride = max(int(track_stride), 1)
        self.track_samples = int(track_samples)
        self.track_sample_mode = str(track_sample_mode)
        self.track_sample_seed = int(track_sample_seed)
        self.scene_offset_per_k = max(int(scene_offset_per_k), 0)
        self.scene_limit_per_k = max(int(scene_limit_per_k), 0)
        self.deterministic_index = bool(deterministic_index)
        if self.geometry_source not in {"all10_sim3", "pair1x49_label_scale_only"}:
            raise ValueError(
                "ReCamMasterCowVGGT_NVS geometry_source must be 'all10_sim3' "
                f"or 'pair1x49_label_scale_only', got {geometry_source!r}"
            )
        if self.geometry_source == "pair1x49_label_scale_only" and self.pair1x49_label_root is None:
            raise ValueError("pair1x49_label_root is required for geometry_source='pair1x49_label_scale_only'")

        self._recam = ReCamMaster_NVS(
            ROOT=self.ROOT,
            split=self.split,
            num_views=self._num_views,
            resolution=(self.output_width, self.output_height),
            steps_per_epoch=1,
            num_cameras=self.num_cameras,
            num_src_frames=self.num_src_frames,
            sensor_width_mm=self.sensor_width_mm,
            return_source_depth_video=False,
            return_source_rgb_video=False,
            return_source_cameras=False,
            camera_only=True,
            caption_cache_dir=self.caption_cache_dir,
        )
        self._samples = self._scan_labeled_samples()
        if not self._samples:
            raise ValueError(
                "No matched ReCamMaster CoW/VGGT labels found. "
                f"cow_root={self.cow_root}, vggt_root={self.vggt_root}, tag={self.all10_tag}"
            )
        print(
            f"[ReCamMasterCowVGGT_NVS] {len(self._samples)} matched q=[0] labels, "
            f"{self._num_views} views, resolution={self.output_width}x{self.output_height}, "
            f"geometry_source={self.geometry_source}, vggt={self.vggt_root}, "
            f"source_vggt={self.source_vggt_root}, pair1x49_labels={self.pair1x49_label_root}, "
            f"cow={self.cow_root}, track_sample_mode={self.track_sample_mode}, "
            f"track_stride={self.track_stride}, scene_offset_per_k={self.scene_offset_per_k}, "
            f"scene_limit_per_k={self.scene_limit_per_k}"
        )

    @property
    def _resolutions(self):
        return [(self.output_width, self.output_height)]

    @property
    def num_views(self):
        return self._num_views

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        pass

    @staticmethod
    def _import_cow_vggt_renderer():
        import importlib
        repo_root = Path(__file__).resolve().parents[5]
        repo_root_s = str(repo_root)
        if repo_root_s not in sys.path:
            sys.path.insert(0, repo_root_s)
        return importlib.import_module("bash_scripts.holo4d.tools.render_recammaster_cow_vggt_2x4_comparison")

    def _scan_labeled_samples(self):
        if self.geometry_source == "pair1x49_label_scale_only":
            return self._scan_pair1x49_labeled_samples()
        source_tracks_root = self.cow_root / "source_tracks"
        if not source_tracks_root.is_dir():
            raise ValueError(f"missing CoWTracker source_tracks root: {source_tracks_root}")
        proc = subprocess.run(
            ["find", str(source_tracks_root), "-type", "f", "-name", "*_pred_trackvideo.npz"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        paths = [Path(line) for line in proc.stdout.splitlines() if line]
        if self.max_index_scan is not None:
            paths = paths[: self.max_index_scan]
        out = []
        pat = re.compile(r"(cam\d+)_start(\d{4})_pred_trackvideo\.npz")
        for cow_npz in paths:
            rel = cow_npz.relative_to(source_tracks_root)
            m = pat.fullmatch(rel.name)
            if m is None:
                continue
            rel_dir = rel.parent.as_posix()
            cam = m.group(1)
            start_frame = int(m.group(2))
            source_json = (
                self.vggt_root / "source_windows" / rel_dir / f"{cam}_start{start_frame:04d}_q000.json"
            )
            all10_json = (
                self.vggt_root / "all10_windows" / rel_dir / f"start{start_frame:04d}_{self.all10_tag}.json"
            )
            if source_json.is_file() and all10_json.is_file():
                out.append({
                    "rel_dir": rel_dir,
                    "cam": cam,
                    "start_frame": start_frame,
                    "cow_npz": cow_npz,
                    "source_json": source_json,
                    "all10_json": all10_json,
                })
        return out

    def _scan_pair1x49_labeled_samples(self):
        label_root = self.pair1x49_label_root / "pair1x49_scale_labels"
        source_tracks_root = self.cow_root / "source_tracks"
        if not label_root.is_dir():
            raise ValueError(f"missing pair1x49 compact label root: {label_root}")
        if not source_tracks_root.is_dir():
            raise ValueError(f"missing CoWTracker source_tracks root: {source_tracks_root}")
        proc = subprocess.run(
            ["find", str(label_root), "-type", "f", "-name", "*_pair1x49_scale.json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        paths = [Path(line) for line in proc.stdout.splitlines() if line]
        if self.max_index_scan is not None:
            paths = paths[: self.max_index_scan]
        out = []
        pat = re.compile(r"(cam\d+)_to_(cam\d+)_start(\d{4})_pair1x49_scale\.json")
        for label_json in paths:
            rel = label_json.relative_to(label_root)
            m = pat.fullmatch(rel.name)
            if m is None:
                continue
            rel_dir = rel.parent.as_posix()
            if self.split and not rel_dir.startswith(f"{self.split}/"):
                continue
            if not self._rel_dir_in_scene_window(rel_dir):
                continue
            source_cam = m.group(1)
            target_cam = m.group(2)
            start_frame = int(m.group(3))
            cow_npz = (
                source_tracks_root
                / rel_dir
                / f"{source_cam}_start{start_frame:04d}_pred_trackvideo.npz"
            )
            if cow_npz.is_file():
                out.append({
                    "rel_dir": rel_dir,
                    "cam": source_cam,
                    "target_cam": target_cam,
                    "start_frame": start_frame,
                    "cow_npz": cow_npz,
                    "pair1x49_label_json": label_json,
                })
        return out

    def _rel_dir_in_scene_window(self, rel_dir):
        if self.scene_limit_per_k <= 0:
            return True
        parts = str(rel_dir).split("/")
        if len(parts) < 3:
            return True
        scene_name = parts[2]
        m = re.fullmatch(r"scene(\d+)", scene_name)
        if m is None:
            return True
        scene_idx = int(m.group(1))
        return self.scene_offset_per_k <= scene_idx < self.scene_offset_per_k + self.scene_limit_per_k

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        max_retries = 10
        last_error = None
        rng = np.random.default_rng(self.track_sample_seed + int(idx)) if self.deterministic_index else np.random.default_rng()
        for _attempt in range(max_retries):
            if self.deterministic_index:
                record = self._samples[(int(idx) + _attempt) % len(self._samples)]
            else:
                record = self._samples[int(rng.integers(len(self._samples)))]
            try:
                return self._build_sample(record, rng)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"ReCamMasterCowVGGT_NVS failed after {max_retries} retries: {last_error}") from last_error

    def _build_sample(self, record, rng):
        rv = self._import_cow_vggt_renderer()
        reproj_viz = rv.reproj_viz
        cache_probe = rv.cache_probe
        W, H = self.output_width, self.output_height
        rel_dir = record["rel_dir"]
        source_cam = record["cam"]
        start_frame = int(record["start_frame"])
        target_cam = record.get("target_cam")
        if target_cam is None:
            src_cam_int = int(source_cam.replace("cam", ""))
            target_candidates = [f"cam{i:02d}" for i in range(1, self.num_cameras + 1) if i != src_cam_int]
            target_cam = target_candidates[int(rng.integers(len(target_candidates)))]
        tgt_cam_int = int(target_cam.replace("cam", ""))

        if self.geometry_source == "pair1x49_label_scale_only":
            from safetensors.torch import load_file

            label_json = Path(record["pair1x49_label_json"])
            label_meta = json.loads(label_json.read_text())
            label_st_meta = Path(label_meta["label"])
            label_st_candidates = []
            if label_st_meta.is_absolute():
                label_st_candidates.extend([
                    label_st_meta,
                    label_json.with_suffix(".safetensors"),
                    label_json.parent / label_st_meta.name,
                ])
            else:
                label_st_candidates.append(label_json.parent / label_st_meta)
            label_st = next((path for path in label_st_candidates if path.is_file()), label_st_candidates[0])
            if not label_st.is_file():
                raise FileNotFoundError(
                    f"missing pair1x49 label safetensor for {label_json}; tried "
                    + ", ".join(str(path) for path in label_st_candidates)
                )
            label_tensors = load_file(str(label_st), device="cpu")
            source_json = Path(label_meta.get("source49_json", ""))
            if not source_json.is_file():
                source_json = rv._find_source49_json(
                    self.source_vggt_root, rel_dir, source_cam, start_frame
                )
            source_meta = json.loads(source_json.read_text())
            source_pred = reproj_viz.read_safetensors(
                reproj_viz.resolve_prediction_path(source_json, source_meta)
            )
            target_rel_frames = [
                int(x) for x in label_tensors["target_rel_frames"].cpu().numpy().reshape(-1).tolist()
            ]
            scale = float(label_tensors["source49_to_pair1x49_scale"].cpu().numpy().reshape(()))
            depth_all = reproj_viz.probe.squeeze_last_channel(source_pred["depth"]).astype(np.float32)
            depth_vggt = depth_all[target_rel_frames] * scale
            intr_vggt = np.asarray(source_pred["intrinsics"], dtype=np.float32)[target_rel_frames]
            source_c2w = np.asarray(source_pred["cam_c2w"], dtype=np.float32)[target_rel_frames].copy()
            source_c2w[:, :3, 3] *= scale
            target_c2w = label_tensors["target_cam_c2w"].cpu().numpy().astype(np.float32)
            if "target_intrinsics_cow_aspect_512x288" in label_tensors:
                target_intrinsics = label_tensors[
                    "target_intrinsics_cow_aspect_512x288"
                ].cpu().numpy().astype(np.float32)[0]
                target_intrinsics = _scale_intrinsics_to_resolution(
                    target_intrinsics,
                    src_w=512,
                    src_h=288,
                    dst_w=W,
                    dst_h=H,
                )
            else:
                target_intrinsics = label_tensors["target_intrinsics_raw"].cpu().numpy().astype(np.float32)[0]
            rel_frames = target_rel_frames
        else:
            source_meta = json.loads(Path(record["source_json"]).read_text())
            source_pred = reproj_viz.read_safetensors(reproj_viz.resolve_prediction_path(record["source_json"], source_meta))
            all10_meta = json.loads(Path(record["all10_json"]).read_text())
            all10_pred = reproj_viz.read_safetensors(reproj_viz.resolve_prediction_path(record["all10_json"], all10_meta))
            query_idx = int(source_meta.get("track_query_idx", 0))
            rel_frames = cache_probe.include_query_frame(reproj_viz.parse_rel_frames(self.all10_rel_frames), query_idx)
            task = cache_probe.FourRCTask(rel_dir=rel_dir, cam=source_cam, start_frame=start_frame, shard=None)
            gt = reproj_viz.gt_pair_geometry(self._recam, rel_dir, source_cam, target_cam, start_frame, H, W)
            gt_to_all10 = reproj_viz.fit_gt_to_all10(self._recam, task, all10_pred, list(all10_meta["frame_first_manifest"]))
            depth_scale = reproj_viz.estimate_source49_depth_to_all10_scale(
                source_pred,
                all10_pred,
                list(all10_meta["frame_first_manifest"]),
                source_cam,
            )
            source_c2w = reproj_viz.probe.align_probe.apply_sim3_to_c2w(gt["source_c2w"], gt_to_all10).astype(np.float32)
            target_c2w = reproj_viz.probe.align_probe.apply_sim3_to_c2w(gt["target_c2w"], gt_to_all10).astype(np.float32)
            depth_vggt = (
                reproj_viz.probe.squeeze_last_channel(source_pred["depth"]).astype(np.float32)
                * float(depth_scale["scale"])
            )
            intr_vggt = np.asarray(source_pred["intrinsics"], dtype=np.float32)
            target_intrinsics = gt["target_intrinsics"].astype(np.float32)
        T = min(self._num_views, depth_vggt.shape[0], source_c2w.shape[0], target_c2w.shape[0])
        depth_vggt = depth_vggt[:T]
        intr_vggt = intr_vggt[:T]
        source_c2w = source_c2w[:T]
        target_c2w = target_c2w[:T]
        if self.geometry_source == "pair1x49_label_scale_only" and self.rgb_post_crop_mode == "cow_aspect":
            source_rgb_np, _ = rv._read_rgb_stack_with_crop(
                Path(self.ROOT), rel_dir, source_cam, start_frame, W, H, crop_aspect_hw=(336, 560)
            )
            target_rgb_np, _ = rv._read_rgb_stack_with_crop(
                Path(self.ROOT), rel_dir, target_cam, start_frame, W, H, crop_aspect_hw=(336, 560)
            )
            source_rgb_np = source_rgb_np[:T]
            target_rgb_np = target_rgb_np[:T]
            # Crop-aware target intrinsics, consistent with BOTH the cow_aspect
            # (336x560) crop of the target RGB AND the source K (which is also
            # cow_hw=(336,560) crop-aware via _postprocess_vggt_depth_intrinsics
            # below). Replaces the `target_intrinsics_cow_aspect_512x288 ->
            # _scale_intrinsics_to_resolution` stretch path: that label is a 512x288
            # (aspect 1.778) canvas, so stretching it to a non-512x288 output (e.g.
            # 672x384, aspect 1.75; source cow crop is 1.667) left a ~6.7% source/
            # target framing mismatch -> target plucker rays inconsistent with the
            # target RGB. This is byte-for-byte the same K the 2x4 audit viz renders
            # with (rv._gt_intrinsics_for_rgb_crop). Revert to the stretch path with
            # TRACK4D360_RECAM_PAIR1X49_LEGACY_TARGET_K=1.
            if os.environ.get("TRACK4D360_RECAM_PAIR1X49_LEGACY_TARGET_K", "0") != "1":
                _crop_aware_target_k, _ = rv._gt_intrinsics_for_rgb_crop(
                    self._recam, rel_dir, target_cam, start_frame, (H, W), (336, 560)
                )
                if _crop_aware_target_k is not None:
                    target_intrinsics = np.asarray(_crop_aware_target_k, dtype=np.float32)
        else:
            source_rgb_np = reproj_viz.load_rgb_stack(Path(self.ROOT), rel_dir, source_cam, start_frame, W, H)[:T]
            target_rgb_np = reproj_viz.load_rgb_stack(Path(self.ROOT), rel_dir, target_cam, start_frame, W, H)[:T]

        with np.load(record["cow_npz"], allow_pickle=True) as cow:
            tracks_world, tracks_valid, query_xy, _, _ = rv.sample_cow_tracks(
                cow,
                depth_vggt,
                intr_vggt,
                source_c2w,
                track_stride=self.track_stride,
                track_samples=self.track_samples,
                track_sample_mode=self.track_sample_mode,
                track_sample_seed=self.track_sample_seed,
                cow_conf_threshold=self.cow_conf_threshold,
            )
            first = tracks_world[:, :1].copy()
            tracks_world = np.where(tracks_valid[..., None], tracks_world, first)

        tracks_cam0, target_cam2cam0, common_scale = normalize_to_cam0(
            tracks_world,
            target_c2w,
            source_c2w[0],
        )
        world_to_src0 = np.linalg.inv(source_c2w[0]).astype(np.float32)
        source_cam2cam0 = (world_to_src0[None] @ source_c2w).astype(np.float32)
        source_cam2cam0[:, :3, 3] /= max(float(common_scale), 1e-6)

        vggt_h, vggt_w = depth_vggt.shape[1:3]
        source_depth = np.stack(
            [cv2.resize(depth_vggt[t], (W, H), interpolation=cv2.INTER_LINEAR) for t in range(T)],
            axis=0,
        ).astype(np.float32)
        if self.geometry_source == "pair1x49_label_scale_only":
            source_depth, intr_vggt, _ = rv._postprocess_vggt_depth_intrinsics(
                depth_vggt,
                intr_vggt,
                (H, W),
                mode=self.vggt_post_crop_mode,
                cow_hw=(336, 560),
            )
        source_depth_valid = np.isfinite(source_depth) & (source_depth > 1e-6)
        source_depth = source_depth / max(float(common_scale), 1e-6)
        if self.geometry_source == "pair1x49_label_scale_only":
            source_intrinsics = np.asarray(intr_vggt[0], dtype=np.float32)
        else:
            source_intrinsics = _scale_intrinsics_to_resolution(intr_vggt[0], vggt_w, vggt_h, W, H)
        query_xy_out = query_xy.astype(np.float32).copy()
        query_xy_out[:, 0] *= float(W) / max(float(vggt_w), 1.0)
        query_xy_out[:, 1] *= float(H) / max(float(vggt_h), 1.0)

        prompt_text, prompt_emb = self._recam._load_camera_caption(rel_dir.split("/", 1)[-1], tgt_cam_int)
        source_frames = [PIL.Image.fromarray(frame) for frame in source_rgb_np]
        target_frames = [PIL.Image.fromarray(frame) for frame in target_rgb_np]

        sample = {
            "video": source_frames,
            "target_video": target_frames,
            "tracks_world": tracks_cam0.astype(np.float32),
            "tracks_world_valid_mask": tracks_valid[:, :T].astype(bool, copy=False),
            "tracks_valid": tracks_valid[:, :T].astype(bool, copy=False),
            "tracks_source_xy": query_xy_out.astype(np.float32),
            "source_rgb_at_tracks": _source_rgb_at_track_xy(source_rgb_np, query_xy_out, T),
            "target_cam2world": target_cam2cam0.astype(np.float32),
            "target_intrinsics": target_intrinsics.astype(np.float32),
            "common_scale": float(common_scale),
            "track_query_idx": 0,
            "track_query_indices": np.asarray([0], dtype=np.int32),
            "tracker_query_indices": np.asarray([0], dtype=np.int32),
            "track_query_count": 1,
            "track_density_mode": "dense" if self.track_sample_mode == "all" or self.track_stride <= 1 else "sparse",
            "track_sparse_stride": self.track_stride,
            "track_sample_mode": self.track_sample_mode,
            "camera_only_tier": False,
            "prompt": prompt_text,
            "prompt_emb": prompt_emb,
            "has_tracks": True,
            "has_dynamic_tracks": True,
            "has_depth": False,
            "track_video_loss_mask": 1.0,
            "recam_cow_vggt_rel_dir": rel_dir,
            "recam_cow_vggt_source_cam": source_cam,
            "recam_cow_vggt_target_cam": target_cam,
            "recam_cow_vggt_start_frame": int(start_frame),
            "recam_cow_vggt_all10_rel_frames": rel_frames,
            "recam_cow_vggt_geometry_source": self.geometry_source,
        }
        if self.return_source_depth_video:
            sample["source_depth_video"] = source_depth
            sample["source_depth_valid"] = source_depth_valid
        if self.return_source_rgb_video:
            sample["source_rgb_video"] = source_frames
        if self.return_source_cameras:
            sample["source_cam2world"] = source_cam2cam0
            sample["source_intrinsics"] = source_intrinsics
        if self.fg_mask_root is not None:
            source_dynamic_mask = rv.load_source_dynamic_mask(
                self.fg_mask_root, rel_dir, source_cam, start_frame, T, H, W,
            )
            if source_dynamic_mask is not None:
                sample["source_dynamic_mask"] = source_dynamic_mask
                sample["source_track_dynamic_mask"] = source_dynamic_mask
        return sample


SynCamMaster_NVS = ReCamMaster_NVS  # convenience alias; set camera_only=True in YAML

__all__ = [
    "Kubric_NVS",
    "StaticScene_NVS",
    "RE10K_NVS",
    "Objaverse_NVS",
    "HomographyNVS",
    "ReCamMaster_NVS",
    "ReCamMasterCowVGGT_NVS",
    "SynCamMaster_NVS",
    "GeometryRobustnessWrapper_NVS",
]
