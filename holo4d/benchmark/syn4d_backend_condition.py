"""Canonical source-condition I/O for the Syn4D backend matrix.

The canonical files contain source-only reconstruction outputs.  They are
already expressed in the backend's frame-0 camera coordinates and normalized
by a source-derived scale.  Loading one into an evaluation sample replaces
only source geometry/tracks; target RGB/depth are never read here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "tracks_xyz",
    "tracks_valid",
    "tracks_confidence",
    "track_query_uv",
    "track_query_frame",
    "source_cam2world",
    "source_intrinsics",
    "metadata_json",
)


def canonical_path(root: str | Path, backend: str, sequence: str) -> Path:
    return Path(root) / backend / sequence / "source_condition.npz"


def _decode_metadata(value: np.ndarray) -> dict:
    text = str(np.asarray(value).item())
    metadata = json.loads(text)
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object")
    return metadata


def load_canonical_condition(path: str | Path, *, expected_backend: str | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path}: missing canonical keys {missing}")
        result = {key: np.asarray(data[key]) for key in data.files}

    tracks = np.asarray(result["tracks_xyz"], dtype=np.float32)
    valid = np.asarray(result["tracks_valid"], dtype=bool)
    confidence = np.asarray(result["tracks_confidence"], dtype=np.float32)
    query_uv = np.asarray(result["track_query_uv"], dtype=np.float32)
    query_frame = np.asarray(result["track_query_frame"], dtype=np.int32)
    source_c2w = np.asarray(result["source_cam2world"], dtype=np.float32)
    source_k = np.asarray(result["source_intrinsics"], dtype=np.float32)
    metadata = _decode_metadata(result["metadata_json"])

    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"{path}: tracks_xyz must be (N,T,3), got {tracks.shape}")
    n, t = tracks.shape[:2]
    expected = {
        "tracks_valid": (n, t),
        "tracks_confidence": (n, t),
        "track_query_uv": (n, 2),
        "track_query_frame": (n,),
        "source_cam2world": (t, 4, 4),
        "source_intrinsics": (t, 3, 3),
    }
    actual = {
        "tracks_valid": valid.shape,
        "tracks_confidence": confidence.shape,
        "track_query_uv": query_uv.shape,
        "track_query_frame": query_frame.shape,
        "source_cam2world": source_c2w.shape,
        "source_intrinsics": source_k.shape,
    }
    bad = {key: (actual[key], shape) for key, shape in expected.items() if actual[key] != shape}
    if bad:
        raise ValueError(f"{path}: canonical shape mismatch {bad}")
    if t != 49:
        raise ValueError(f"{path}: expected 49 frames, got {t}")
    if not np.isfinite(tracks[valid]).all():
        raise ValueError(f"{path}: valid track entries contain non-finite XYZ")
    if not valid.any():
        raise ValueError(f"{path}: no valid track observations")
    if np.any(query_frame < 0) or np.any(query_frame >= t):
        raise ValueError(f"{path}: track_query_frame outside [0,{t})")
    backend = str(metadata.get("backend", ""))
    if expected_backend is not None and backend != expected_backend:
        raise ValueError(f"{path}: metadata backend={backend!r}, expected {expected_backend!r}")

    result.update(
        tracks_xyz=tracks,
        tracks_valid=valid,
        tracks_confidence=confidence,
        track_query_uv=query_uv,
        track_query_frame=query_frame,
        source_cam2world=source_c2w,
        source_intrinsics=source_k,
        metadata=metadata,
        path=path,
    )
    return result


def _video_frames_uint8(video) -> np.ndarray:
    frames = []
    for frame in video:
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if array.min() < 0:
                array = (array * 0.5 + 0.5) * 255.0
            elif array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        frames.append(array)
    return np.stack(frames, axis=0)


def _query_rgb(video, query_uv: np.ndarray, query_frame: np.ndarray) -> np.ndarray:
    frames = _video_frames_uint8(video)
    t, height, width = frames.shape[:3]
    x = np.rint(query_uv[:, 0]).astype(np.int64)
    y = np.rint(query_uv[:, 1]).astype(np.int64)
    q = np.asarray(query_frame, dtype=np.int64)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (q >= 0) & (q < t)
    rgb = np.zeros((len(query_uv), 3), dtype=np.float32)
    rgb[in_bounds] = frames[q[in_bounds], y[in_bounds], x[in_bounds]].astype(np.float32) / 255.0
    return np.broadcast_to(rgb[:, None], (len(rgb), t, 3)).copy()


def inject_canonical_condition(sample: dict, condition: dict) -> dict:
    """Return a sample with source tracks/cameras replaced by one backend."""
    result = dict(sample)
    tracks = condition["tracks_xyz"]
    valid = condition["tracks_valid"]
    qframe = condition["track_query_frame"]
    if len(sample["video"]) != tracks.shape[1]:
        raise ValueError(
            f"source video has {len(sample['video'])} frames but condition has {tracks.shape[1]}"
        )
    result["tracks_world"] = tracks
    result["tracks_world_valid_mask"] = valid
    result["tracks_query_idx_per_identity"] = qframe
    unique_queries = sorted(int(value) for value in np.unique(qframe))
    result["track_query_idx"] = unique_queries[0]
    result["track_query_indices"] = unique_queries
    result["track_query_count"] = len(unique_queries)
    rgb_query_frame = condition["metadata"].get("source_rgb_query_frame")
    if rgb_query_frame is None:
        # Legacy canonical files did not declare a separate appearance frame;
        # preserve their per-identity query-frame RGB behavior.
        rgb_query_indices = qframe
    else:
        rgb_query_frame = int(rgb_query_frame)
        if not 0 <= rgb_query_frame < len(sample["video"]):
            raise ValueError(
                f"source_rgb_query_frame={rgb_query_frame} outside source video "
                f"length {len(sample['video'])}"
            )
        rgb_query_indices = np.full(len(qframe), rgb_query_frame, dtype=np.int32)
    result["source_rgb_at_tracks"] = _query_rgb(
        sample["video"], condition["track_query_uv"], rgb_query_indices
    )
    # Force the track builder to derive target-camera depth from the backend
    # tracks and the prescribed target cameras.  GT track-aligned depths from
    # the original Syn4D sample must not survive this replacement.
    result.pop("reproj_src_depth_at_target", None)
    result.pop("reproj_src_depth_valid", None)
    result["source_cam2world"] = condition["source_cam2world"]
    result["source_intrinsics"] = condition["source_intrinsics"]
    target_alignment = condition["metadata"].get("target_camera_alignment")
    if target_alignment == "preserve_source_to_target_relative_pose":
        original_source_c2w = np.asarray(sample["source_cam2world"], dtype=np.float32)
        original_target_c2w = np.asarray(sample["target_cam2world"], dtype=np.float32)
        predicted_source_c2w = np.asarray(
            condition["source_cam2world"], dtype=np.float32
        )
        if original_source_c2w.shape != predicted_source_c2w.shape:
            raise ValueError(
                "cannot re-anchor target cameras: original/predicted source "
                f"camera shapes differ ({original_source_c2w.shape} vs "
                f"{predicted_source_c2w.shape})"
            )
        if original_target_c2w.shape != original_source_c2w.shape:
            raise ValueError(
                "cannot re-anchor target cameras: source/target camera shapes "
                f"differ ({original_source_c2w.shape} vs "
                f"{original_target_c2w.shape})"
            )
        # The requested NVS control is the per-frame source-to-target relative
        # pose. Canonical predicted tracks live in the VGGT source-camera
        # gauge, so compose that same relative control onto the predicted
        # source path. Keeping the original target poses would mix the GT
        # window gauge with the predicted reconstruction gauge and cause a
        # large discontinuity at each 49-frame boundary.
        source_to_target = (
            np.linalg.inv(original_source_c2w).astype(np.float32)
            @ original_target_c2w
        )
        translation_norms = np.linalg.norm(
            source_to_target[:, :3, 3], axis=-1
        )
        valid_norms = translation_norms[
            np.isfinite(translation_norms) & (translation_norms > 1e-8)
        ]
        if not len(valid_norms):
            raise ValueError("source-to-target relative translations are all zero/invalid")
        current_reference_norm = float(np.median(valid_norms))
        requested_reference_norm = condition["metadata"].get(
            "target_relative_translation_reference_norm"
        )
        if requested_reference_norm is None:
            requested_reference_norm = current_reference_norm
        requested_reference_norm = float(requested_reference_norm)
        if not np.isfinite(requested_reference_norm) or requested_reference_norm <= 1e-8:
            raise ValueError(
                "target_relative_translation_reference_norm must be positive; "
                f"got {requested_reference_norm}"
            )
        source_to_target[:, :3, 3] *= (
            requested_reference_norm / current_reference_norm
        )
        result["target_relative_translation_reference_norm"] = (
            requested_reference_norm
        )
        result["target_cam2world"] = (
            predicted_source_c2w @ source_to_target
        ).astype(np.float32)
        result["target_camera_alignment"] = target_alignment
    elif target_alignment not in (None, "keep_dataset_target_camera"):
        raise ValueError(
            f"unsupported target_camera_alignment={target_alignment!r}"
        )
    if "source_depth" in condition:
        depth = np.asarray(condition["source_depth"], dtype=np.float32)
        result["source_depth"] = depth[0]
        result["source_depth_video"] = depth
    result["source_condition_metadata"] = condition["metadata"]
    result["source_condition_path"] = str(condition["path"])
    return result


def normalize_cam2world_to_frame0(cam2world: np.ndarray) -> np.ndarray:
    """Express a camera path in its first-frame camera coordinate system."""
    cam2world = np.asarray(cam2world, dtype=np.float32)
    if cam2world.ndim != 3 or cam2world.shape[1:] != (4, 4):
        raise ValueError(f"cam2world must be (T,4,4), got {cam2world.shape}")
    return (np.linalg.inv(cam2world[0]).astype(np.float32)[None] @ cam2world).astype(
        np.float32
    )


def swap_source_camera_component(
    condition: dict,
    gt_source_cam2world: np.ndarray,
    *,
    component: str,
    gt_translation_scale: float,
) -> dict:
    """Replace only source-camera rotation or translation in one condition.

    The reconstructed 4RC observations are preserved in camera coordinates.
    After changing the camera path, every valid 3-D observation is therefore
    transformed from the old per-frame camera coordinate system into the new
    world coordinate system.  This avoids the invalid intervention in which a
    camera pose is changed while its point observations remain in the old
    world frame.
    """
    if component not in {"rotation", "translation"}:
        raise ValueError(f"component must be rotation or translation, got {component!r}")
    if not np.isfinite(gt_translation_scale) or gt_translation_scale <= 0:
        raise ValueError(
            f"gt_translation_scale must be finite and positive, got {gt_translation_scale}"
        )

    predicted = np.asarray(condition["source_cam2world"], dtype=np.float32)
    gt_local = normalize_cam2world_to_frame0(gt_source_cam2world)
    if predicted.shape != gt_local.shape:
        raise ValueError(
            f"predicted/GT source camera shapes differ: {predicted.shape} vs {gt_local.shape}"
        )

    hybrid = predicted.copy()
    if component == "rotation":
        hybrid[:, :3, :3] = gt_local[:, :3, :3]
    else:
        hybrid[:, :3, 3] = gt_local[:, :3, 3] * float(gt_translation_scale)

    tracks = np.asarray(condition["tracks_xyz"], dtype=np.float32)
    valid = np.asarray(condition["tracks_valid"], dtype=bool)
    old_world_to_camera = np.linalg.inv(predicted).astype(np.float32)
    camera_xyz = np.einsum(
        "tij,ntj->nti", old_world_to_camera[:, :3, :3], tracks
    ) + old_world_to_camera[None, :, :3, 3]
    hybrid_tracks = np.einsum(
        "tij,ntj->nti", hybrid[:, :3, :3], camera_xyz
    ) + hybrid[None, :, :3, 3]
    hybrid_tracks[~valid] = 0.0

    result = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in condition.items()
        if key not in {"metadata", "path"}
    }
    result["tracks_xyz"] = hybrid_tracks.astype(np.float32)
    result["source_cam2world"] = hybrid.astype(np.float32)
    return result


def apply_relative_camera_control(sample: dict, mode: str) -> tuple[dict, dict]:
    """Keep only relative source-to-target rotation or translation control."""
    if mode not in {"native_mixed", "rotation_only", "translation_only"}:
        raise ValueError(f"unsupported camera-control mode {mode!r}")
    result = dict(sample)
    source = np.asarray(sample["source_cam2world"], dtype=np.float32)
    target = np.asarray(sample["target_cam2world"], dtype=np.float32)
    if source.shape != target.shape or source.shape[1:] != (4, 4):
        raise ValueError(
            f"source/target camera shapes must match (T,4,4), got {source.shape}/{target.shape}"
        )
    relative = (np.linalg.inv(source).astype(np.float32) @ target).astype(np.float32)
    controlled = relative.copy()
    if mode == "rotation_only":
        controlled[:, :3, 3] = 0.0
    elif mode == "translation_only":
        controlled[:, :3, :3] = np.eye(3, dtype=np.float32)
    result["target_cam2world"] = (source @ controlled).astype(np.float32)
    provenance_record = {
        "mode": mode,
        "definition": "target_cam2world = source_cam2world @ controlled_source_to_target",
        "source_to_target_original": relative.tolist(),
        "source_to_target_controlled": controlled.tolist(),
        "target_cam2world_controlled": result["target_cam2world"].tolist(),
    }
    return result, provenance_record


def provenance(path: str | Path, metadata: dict) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "canonical_npz": str(path),
        "canonical_npz_sha256": digest.hexdigest(),
        "metadata": metadata,
    }
