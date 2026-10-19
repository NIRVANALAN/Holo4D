from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def anchor_trajectory_to_source_frame0(
    *,
    target_c2w: np.ndarray,
    target_intrinsics: np.ndarray,
    source_frame0_c2w: np.ndarray,
    source_intrinsics: np.ndarray,
    translation_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-anchor a target trajectory so frame 0 exactly matches the source view.

    Vista4D official target cameras can begin at a novel camera. For image/video
    conditioning runs where frame 0 should be shared with the source clip, keep
    Vista4D's relative motion shape but apply it on top of source frame 0.
    """
    target_c2w = np.asarray(target_c2w, dtype=np.float32)
    target_intrinsics = np.asarray(target_intrinsics, dtype=np.float32)
    source_frame0_c2w = np.asarray(source_frame0_c2w, dtype=np.float32)
    source_intrinsics = np.asarray(source_intrinsics, dtype=np.float32)

    if target_c2w.ndim != 3 or target_c2w.shape[1:] != (4, 4):
        raise ValueError(f"target_c2w must be (T,4,4), got {target_c2w.shape}")
    if target_intrinsics.ndim != 3 or target_intrinsics.shape[1:] != (3, 3):
        raise ValueError(
            f"target_intrinsics must be (T,3,3), got {target_intrinsics.shape}"
        )
    if source_frame0_c2w.shape != (4, 4):
        raise ValueError(
            f"source_frame0_c2w must be (4,4), got {source_frame0_c2w.shape}"
        )
    if source_intrinsics.shape != (3, 3):
        raise ValueError(f"source_intrinsics must be (3,3), got {source_intrinsics.shape}")
    if target_intrinsics.shape[0] != target_c2w.shape[0]:
        raise ValueError(
            "target_c2w and target_intrinsics must have matching frame counts, "
            f"got {target_c2w.shape[0]} and {target_intrinsics.shape[0]}"
        )

    if not np.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError(
            f"translation_scale must be finite and positive, got {translation_scale}"
        )

    target0_to_target_t = np.linalg.inv(target_c2w[0])[None] @ target_c2w
    target0_to_target_t[:, :3, 3] *= float(translation_scale)
    anchored_c2w = source_frame0_c2w[None] @ target0_to_target_t

    # Transfer focal evolution as an FOV ratio. Adding absolute focal deltas
    # from independent reconstructions can cross zero and flip dolly-zoom views.
    fx0 = float(target_intrinsics[0, 0, 0])
    fy0 = float(target_intrinsics[0, 1, 1])
    if fx0 <= 0.0 or fy0 <= 0.0:
        raise ValueError(f"target frame-0 focal length must be positive, got {fx0}, {fy0}")
    anchored_intrinsics = np.broadcast_to(
        source_intrinsics[None], target_intrinsics.shape
    ).copy()
    anchored_intrinsics[:, 0, 0] = (
        source_intrinsics[0, 0] * target_intrinsics[:, 0, 0] / fx0
    )
    anchored_intrinsics[:, 1, 1] = (
        source_intrinsics[1, 1] * target_intrinsics[:, 1, 1] / fy0
    )
    anchored_intrinsics[:, 0, 2] = source_intrinsics[0, 2] + (
        target_intrinsics[:, 0, 2] - target_intrinsics[0, 0, 2]
    )
    anchored_intrinsics[:, 1, 2] = source_intrinsics[1, 2] + (
        target_intrinsics[:, 1, 2] - target_intrinsics[0, 1, 2]
    )
    anchored_intrinsics[:, 2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if (
        not np.isfinite(anchored_intrinsics).all()
        or np.any(anchored_intrinsics[:, 0, 0] <= 0.0)
        or np.any(anchored_intrinsics[:, 1, 1] <= 0.0)
    ):
        raise ValueError("aligned Vista4D intrinsics contain non-positive focal lengths")
    anchored_c2w[0] = source_frame0_c2w
    anchored_intrinsics[0] = source_intrinsics
    return anchored_c2w.astype(np.float32), anchored_intrinsics.astype(np.float32)


def transform_native_target_to_source_gauge(
    *,
    target_c2w: np.ndarray,
    source_frame0_c2w: np.ndarray,
    translation_scale: float,
) -> np.ndarray:
    """Express Vista4D native target cameras in a query-camera source gauge.

    Released source and target cameras share Pi3's world coordinates, but a
    Holo4D/4RC condition is normalized so its query camera is identity.  First
    left-multiply every native target pose by the inverse Pi3 source frame-0
    pose, then scale translations into the 4RC reconstruction units.  This
    preserves the official target pose relative to the source without turning
    it into the separate same-first-frame protocol.
    """
    target_c2w = np.asarray(target_c2w, dtype=np.float32)
    source_frame0_c2w = np.asarray(source_frame0_c2w, dtype=np.float32)
    if target_c2w.ndim != 3 or target_c2w.shape[1:] != (4, 4):
        raise ValueError(f"target_c2w must be (T,4,4), got {target_c2w.shape}")
    if source_frame0_c2w.shape != (4, 4):
        raise ValueError(
            f"source_frame0_c2w must be (4,4), got {source_frame0_c2w.shape}"
        )
    if not np.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError(
            f"translation_scale must be finite and positive, got {translation_scale}"
        )
    target_in_source = np.linalg.inv(source_frame0_c2w)[None] @ target_c2w
    target_in_source[:, :3, 3] *= float(translation_scale)
    return target_in_source.astype(np.float32)


def discover_vista4d_pairs_from_metadata(
    metadata_csv: str | Path,
    video_filter: Iterable[str] | None = None,
    source_filter: str | None = None,
    max_cameras_per_video: int = 0,
) -> list[tuple[str, str]]:
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"missing {metadata_csv}; run hf download Eyeline-Labs/Vista4D-Eval-Data first"
        )

    video_set = set(video_filter) if video_filter else None
    source_norm = source_filter.lower() if source_filter else None
    per_video_count: dict[str, int] = defaultdict(int)
    pairs: list[tuple[str, str]] = []

    with metadata_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            video = row["video"]
            if source_norm and row.get("source", "").lower() != source_norm:
                continue
            if video_set is not None and video not in video_set:
                continue
            if max_cameras_per_video > 0 and per_video_count[video] >= max_cameras_per_video:
                continue
            pairs.append((video, row["camera"]))
            per_video_count[video] += 1
    return pairs


def lookup_vista4d_source_prompt(
    metadata_csv: str | Path,
    *,
    source_video: str,
    source_camera: str | None = None,
    default: str = "",
) -> str:
    """Return the Vista4D metadata caption for the source content video.

    In trajectory-transfer runs the output camera name may be a donor
    ``video__camera`` identifier. The text prompt must still describe the source
    video being generated, so this helper first tries the real source camera and
    then falls back to any caption row for the source video.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        return default
    with metadata_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if source_camera:
        for row in rows:
            if row.get("video") == source_video and row.get("camera") == source_camera:
                return row.get("prompt") or default
    for row in rows:
        if row.get("video") == source_video:
            return row.get("prompt") or default
    return default


def lookup_vista4d_pair_seed(
    metadata_csv: str | Path,
    *,
    video: str,
    camera: str,
) -> int:
    """Return Vista4D's published per-pair inference seed.

    The official evaluator does not use one global seed: every metadata row
    carries the seed used for that source/trajectory pair.  Keeping this lookup
    in the shared benchmark module lets Holo4D and Vista4D consume the same
    frozen protocol without duplicating CSV parsing in launchers.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"missing Vista4D metadata: {metadata_csv}")
    with metadata_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("video") == video and row.get("camera") == camera:
                value = row.get("seed", "")
                try:
                    return int(value)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid Vista4D seed {value!r} for {video}/{camera}"
                    ) from exc
    raise KeyError(f"Vista4D metadata has no pair {video}/{camera}")


def load_vista4d_source_fps(eval_root: str | Path, video: str) -> float:
    """Read the source FPS used by Vista4D's official output writer."""
    import cv2

    video_path = Path(eval_root) / "recon_and_seg" / video / "video.mp4"
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"could not read a valid FPS from {video_path}: {fps}")
    return fps


def _safe_transfer_camera_name(donor_video: str, donor_camera: str) -> str:
    safe_video = donor_video.replace("/", "_")
    safe_camera = donor_camera.replace("/", "_")
    return f"{safe_video}__{safe_camera}"


def _max_relative_forward_angle_deg(cam_c2w: np.ndarray) -> float:
    cam_c2w = np.asarray(cam_c2w, dtype=np.float32)
    if cam_c2w.ndim != 3 or cam_c2w.shape[1:] != (4, 4):
        raise ValueError(f"cam_c2w must be (T,4,4), got {cam_c2w.shape}")
    forward = cam_c2w[:, :3, 2]
    denom = np.linalg.norm(forward, axis=-1) * max(
        float(np.linalg.norm(forward[0])), 1e-8
    )
    dots = np.clip((forward @ forward[0]) / np.maximum(denom, 1e-8), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).max())


def _trajectory_is_front_facing(
    camera_root: Path,
    video: str,
    camera: str,
    max_forward_angle_deg: float,
) -> bool:
    if max_forward_angle_deg >= 180.0:
        return True
    npz_path = camera_root / video / f"{camera}.npz"
    if not npz_path.exists():
        return False
    data = np.load(npz_path, allow_pickle=True)
    if "cam_c2w" not in data.files:
        return False
    return _max_relative_forward_angle_deg(data["cam_c2w"]) <= max_forward_angle_deg


def discover_vista4d_transfer_pairs_from_metadata(
    metadata_csv: str | Path,
    *,
    camera_root: str | Path,
    video_filter: Iterable[str] | None = None,
    source_filter: str | None = None,
    donor_source_filter: str | None = None,
    trajectories_per_video: int = 10,
    seed: int = 0,
    max_forward_angle_deg: float = 90.0,
    exclude_same_video: bool = True,
) -> list[tuple[str, str, str, str]]:
    """Sample donor Vista4D trajectories for each requested source video.

    Returns tuples of ``(source_video, output_camera, donor_video, donor_camera)``.
    ``output_camera`` is stable and unique enough for output directories, while
    ``donor_video``/``donor_camera`` point at the actual camera NPZ to load.
    """
    metadata_csv = Path(metadata_csv)
    camera_root = Path(camera_root)
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"missing {metadata_csv}; run hf download Eyeline-Labs/Vista4D-Eval-Data first"
        )
    if trajectories_per_video <= 0:
        return []

    video_set = set(video_filter) if video_filter else None
    source_norm = source_filter.lower() if source_filter else None
    donor_source_norm = donor_source_filter.lower() if donor_source_filter else None
    rows: list[dict[str, str]] = []
    source_videos: list[str] = []

    with metadata_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            video = row["video"]
            source = row.get("source", "").lower()
            rows.append(row)
            if source_norm and source != source_norm:
                continue
            if video_set is not None and video not in video_set:
                continue
            if video not in source_videos:
                source_videos.append(video)

    donor_rows = []
    for row in rows:
        if donor_source_norm and row.get("source", "").lower() != donor_source_norm:
            continue
        if not _trajectory_is_front_facing(
            camera_root,
            row["video"],
            row["camera"],
            max_forward_angle_deg,
        ):
            continue
        donor_rows.append(row)

    rng = random.Random(seed)
    pairs: list[tuple[str, str, str, str]] = []
    for source_video in source_videos:
        candidates = [
            row for row in donor_rows
            if not exclude_same_video or row["video"] != source_video
        ]
        sample_count = min(int(trajectories_per_video), len(candidates))
        for row in rng.sample(candidates, sample_count):
            donor_video = row["video"]
            donor_camera = row["camera"]
            pairs.append((
                source_video,
                _safe_transfer_camera_name(donor_video, donor_camera),
                donor_video,
                donor_camera,
            ))
    return pairs
