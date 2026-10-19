"""Robustly align Vista4D/Pi3 geometry to a source reconstruction gauge.

Vista4D's released target cameras live in the world gauge recovered by Pi3.
Holo4D conditions, however, may come from a different reconstructor (4RC,
VGGT-Omega, ...).  A frame-0 depth ratio only resolves scale; it does not
resolve residual rotation and translation differences between the two
reconstructions.  This module estimates the full Sim(3) from corresponding
source-video pointmaps and applies it to the released target cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

# OpenCV disables OpenEXR unless this opt-in exists before ``cv2`` import.
# The sampler and the standalone effective-camera exporter import this module
# through different entrypoints, so keep the requirement at the data reader's
# single source of truth rather than relying on launcher order.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


@dataclass(frozen=True)
class PointmapSimilarity:
    """Similarity mapping ``target = scale * rotation @ source + translation``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    correspondences: int
    inliers: int
    median_residual: float
    p90_residual: float

    @classmethod
    def from_dict(cls, payload: dict) -> "PointmapSimilarity":
        """Restore the exact fitted transform saved beside a prediction."""
        return cls(
            scale=float(payload["scale"]),
            rotation=np.asarray(payload["rotation"], dtype=np.float64),
            translation=np.asarray(payload["translation"], dtype=np.float64),
            correspondences=int(payload["correspondences"]),
            inliers=int(payload["inliers"]),
            median_residual=float(payload["median_residual"]),
            p90_residual=float(payload["p90_residual"]),
        )

    def as_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "rotation": np.asarray(self.rotation, dtype=np.float64).tolist(),
            "translation": np.asarray(self.translation, dtype=np.float64).tolist(),
            "correspondences": int(self.correspondences),
            "inliers": int(self.inliers),
            "inlier_ratio": float(self.inliers / max(self.correspondences, 1)),
            "median_residual": float(self.median_residual),
            "p90_residual": float(self.p90_residual),
        }


def _intrinsics_to_matrix(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape == (3, 3):
        return intrinsics.copy()
    if intrinsics.shape == (4,):
        fx, fy, cx, cy = intrinsics.tolist()
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError(f"unsupported intrinsics shape {intrinsics.shape}")


def _read_depth(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(
            f"could not decode {path}; set OPENCV_IO_ENABLE_OPENEXR=1 before Python starts"
        )
    if depth.ndim == 3:
        depth = depth[..., 0]
    return np.asarray(depth, dtype=np.float64)


def _unproject_world(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    cam2world: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera = np.stack(
        [(xx - cx) * depth / fx, (yy - cy) * depth / fy, depth], axis=-1
    )
    rotation = np.asarray(cam2world[:3, :3], dtype=np.float64)
    translation = np.asarray(cam2world[:3, 3], dtype=np.float64)
    return camera @ rotation.T + translation


def _estimate_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the least-squares Sim(3) mapping row-vector points source->target."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"point arrays must both be (N,3), got {source.shape}, {target.shape}")
    if source.shape[0] < 3:
        raise ValueError("at least three correspondences are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= 1e-12:
        raise ValueError("degenerate source point cloud")
    covariance = target_centered.T @ source_centered / float(source.shape[0])
    left, singular_values, right_t = np.linalg.svd(covariance)
    parity = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0.0:
        parity[-1, -1] = -1.0
    rotation = left @ parity @ right_t
    scale = float(np.sum(singular_values * np.diag(parity)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"invalid similarity scale {scale}")
    return scale, rotation, translation


def _apply_to_points(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return float(scale) * (np.asarray(points) @ np.asarray(rotation).T) + np.asarray(translation)


def robust_similarity_from_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    *,
    trim_iterations: int = 6,
    mad_multiplier: float = 3.0,
    min_inliers: int = 512,
) -> PointmapSimilarity:
    """Estimate Sim(3) with iterative MAD trimming for dynamic/outlier pixels."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[finite], target[finite]
    if source.shape[0] < min_inliers:
        raise RuntimeError(
            f"insufficient pointmap correspondences: {source.shape[0]} < {min_inliers}"
        )
    keep = np.ones(source.shape[0], dtype=bool)
    for _ in range(max(int(trim_iterations), 1)):
        scale, rotation, translation = _estimate_umeyama(source[keep], target[keep])
        residual = np.linalg.norm(
            _apply_to_points(source, scale, rotation, translation) - target, axis=1
        )
        active = residual[keep]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        threshold = median + float(mad_multiplier) * 1.4826 * max(mad, 1e-9)
        next_keep = residual <= threshold
        if int(next_keep.sum()) < min_inliers:
            break
        if np.array_equal(next_keep, keep):
            keep = next_keep
            break
        keep = next_keep
    scale, rotation, translation = _estimate_umeyama(source[keep], target[keep])
    residual = np.linalg.norm(
        _apply_to_points(source, scale, rotation, translation) - target, axis=1
    )
    active = residual[keep]
    return PointmapSimilarity(
        scale=scale,
        rotation=rotation.astype(np.float64),
        translation=translation.astype(np.float64),
        correspondences=int(source.shape[0]),
        inliers=int(keep.sum()),
        median_residual=float(np.median(active)),
        p90_residual=float(np.quantile(active, 0.9)),
    )


def estimate_pi3_to_source_pointmap_similarity(
    *,
    source_query_pointmaps: list[np.ndarray],
    source_confidence_pointmaps: list[np.ndarray | None] | None = None,
    vista4d_recon_root: str | Path,
    num_frames: int,
    frame_stride: int = 4,
    pixel_stride: int = 8,
) -> PointmapSimilarity:
    """Estimate Pi3-world -> source-query Sim(3) from matched source frames.

    Correspondence is by source-frame pixel after resizing Pi3 depth to the
    source pointmap grid. Official Vista4D dynamic and sky masks are excluded;
    iterative trimming handles remaining reconstructor disagreement.
    """
    recon_root = Path(vista4d_recon_root)
    camera_path = recon_root / "cameras.npz"
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    cameras = np.load(camera_path, allow_pickle=True)
    pi3_c2w = np.asarray(cameras["cam_c2w"], dtype=np.float64)
    pi3_intrinsics = np.asarray(cameras["intrinsics"], dtype=np.float64)
    available = min(int(num_frames), len(source_query_pointmaps), pi3_c2w.shape[0])
    if available <= 0:
        raise ValueError("no source frames available for pointmap Sim(3)")

    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    for frame_index in range(0, available, max(int(frame_stride), 1)):
        source_query = np.asarray(source_query_pointmaps[frame_index], dtype=np.float64)
        if source_query.ndim != 3 or source_query.shape[-1] != 3:
            raise ValueError(
                f"source pointmap {frame_index} must be (H,W,3), got {source_query.shape}"
            )
        height, width = source_query.shape[:2]
        depth_native = _read_depth(recon_root / "depths" / f"{frame_index:05d}.exr")
        native_height, native_width = depth_native.shape
        depth = cv2.resize(
            depth_native, (width, height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float64)
        intrinsics = _intrinsics_to_matrix(pi3_intrinsics[frame_index])
        intrinsics[0, 0] *= width / float(native_width)
        intrinsics[0, 2] *= width / float(native_width)
        intrinsics[1, 1] *= height / float(native_height)
        intrinsics[1, 2] *= height / float(native_height)
        pi3_world = _unproject_world(depth, intrinsics, pi3_c2w[frame_index])

        valid = (
            np.isfinite(source_query).all(axis=-1)
            & np.isfinite(pi3_world).all(axis=-1)
            & np.isfinite(depth)
            & (depth > 1e-6)
        )
        if source_confidence_pointmaps is not None:
            confidence = source_confidence_pointmaps[frame_index]
            if confidence is not None:
                confidence = np.asarray(confidence, dtype=np.float64)
                if confidence.shape != (height, width):
                    confidence = cv2.resize(
                        confidence, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                finite_confidence = np.isfinite(confidence)
                if bool(finite_confidence.any()):
                    valid &= finite_confidence & (
                        confidence >= np.quantile(confidence[finite_confidence], 0.5)
                    )
        for folder in ("dynamic_mask", "sky_mask"):
            mask_path = recon_root / folder / f"{frame_index:05d}.png"
            if mask_path.is_file():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(
                        mask, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                    valid &= mask < 128
        yy, xx = np.mgrid[:height, :width]
        stride = max(int(pixel_stride), 1)
        valid &= (xx % stride == 0) & (yy % stride == 0)
        source_points.append(pi3_world[valid])
        target_points.append(source_query[valid])

    return robust_similarity_from_correspondences(
        np.concatenate(source_points, axis=0),
        np.concatenate(target_points, axis=0),
    )


def estimate_pi3_to_source_background_camera_similarity(
    *,
    source_query_pointmaps: list[np.ndarray],
    source_confidence_pointmaps: list[np.ndarray | None] | None = None,
    vista4d_recon_root: str | Path,
    pixel_stride: int = 4,
    mad_multiplier: float = 3.0,
    min_inliers: int = 512,
) -> PointmapSimilarity:
    """Estimate a source-camera-anchored Pi3-world -> query-camera Sim(3).

    A free point-cloud Sim(3) can rotate a target camera to compensate for
    non-rigid depth disagreement between two monocular reconstructors. That is
    not a valid gauge conversion. The source frame-0 camera fixes both origin
    and orientation; only scale is estimated from matched background
    depths. This preserves the official target motion relative to its source
    camera instead of moving that camera to reduce point-cloud residuals.

    Vista4D's released ``dynamic_mask`` and ``sky_mask`` are applied at the
    same pixels on both pointmaps. The resulting transform is valid for target
    cameras, pointmaps, and persistent tracks expressed in those gauges.
    """
    if not source_query_pointmaps:
        raise ValueError("source_query_pointmaps must contain frame 0")
    source_query = np.asarray(source_query_pointmaps[0], dtype=np.float64)
    if source_query.ndim != 3 or source_query.shape[-1] != 3:
        raise ValueError(
            f"source pointmap 0 must be (H,W,3), got {source_query.shape}"
        )

    recon_root = Path(vista4d_recon_root)
    camera_path = recon_root / "cameras.npz"
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    cameras = np.load(camera_path, allow_pickle=True)
    pi3_c2w0 = np.asarray(cameras["cam_c2w"][0], dtype=np.float64)
    pi3_intrinsics0 = _intrinsics_to_matrix(cameras["intrinsics"][0])

    height, width = source_query.shape[:2]
    depth_native = _read_depth(recon_root / "depths" / "00000.exr")
    native_height, native_width = depth_native.shape
    pi3_depth = cv2.resize(
        depth_native, (width, height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float64)
    pi3_intrinsics0[0, 0] *= width / float(native_width)
    pi3_intrinsics0[0, 2] *= width / float(native_width)
    pi3_intrinsics0[1, 1] *= height / float(native_height)
    pi3_intrinsics0[1, 2] *= height / float(native_height)
    pi3_world = _unproject_world(pi3_depth, pi3_intrinsics0, pi3_c2w0)

    valid = (
        np.isfinite(source_query).all(axis=-1)
        & np.isfinite(pi3_world).all(axis=-1)
        & np.isfinite(pi3_depth)
        & (source_query[..., 2] > 1e-6)
        & (pi3_depth > 1e-6)
    )
    if (
        source_confidence_pointmaps is not None
        and len(source_confidence_pointmaps) > 0
    ):
        confidence = source_confidence_pointmaps[0]
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float64)
            if confidence.shape != (height, width):
                confidence = cv2.resize(
                    confidence, (width, height), interpolation=cv2.INTER_LINEAR
                )
            # Confidence is an invalidity gate, not a percentile sampler.
            # Monocular confidence is spatially correlated with depth and
            # object layout; selecting only its upper half can change the
            # camera scale by almost 2x on Vista4D scenes. Robust background
            # masking and the MAD scale trim below handle noisy finite pixels.
            valid &= np.isfinite(confidence) & (confidence > 0.0)

    for folder in ("dynamic_mask", "sky_mask"):
        mask_path = recon_root / folder / "00000.png"
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
                valid &= mask < 128

    yy, xx = np.mgrid[:height, :width]
    stride = max(int(pixel_stride), 1)
    valid &= (xx % stride == 0) & (yy % stride == 0)
    ratios = source_query[..., 2][valid] / pi3_depth[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 1e-8)]
    if ratios.size < int(min_inliers):
        raise RuntimeError(
            f"insufficient static background depth ratios: {ratios.size} < {min_inliers}"
        )

    # Log-space trimming treats reciprocal scale errors symmetrically.
    log_ratios = np.log(ratios)
    median_log = float(np.median(log_ratios))
    mad_log = float(np.median(np.abs(log_ratios - median_log)))
    threshold = float(mad_multiplier) * 1.4826 * max(mad_log, 1e-9)
    keep = np.abs(log_ratios - median_log) <= threshold
    if int(keep.sum()) < int(min_inliers):
        raise RuntimeError(
            f"insufficient static background scale inliers: {int(keep.sum())} < {min_inliers}"
        )
    scale = float(np.exp(np.median(log_ratios[keep])))

    # x_world = R0 x_cam0 + C0, hence
    # x_query = scale * R0^T (x_world - C0). This maps Pi3's source camera
    # exactly to the 4RC query camera while retaining the native target offset.
    rotation = pi3_c2w0[:3, :3].T
    translation = -scale * (rotation @ pi3_c2w0[:3, 3])
    residual = np.linalg.norm(
        _apply_to_points(pi3_world[valid], scale, rotation, translation)
        - source_query[valid],
        axis=1,
    )
    active_residual = residual[np.isfinite(residual)]
    if active_residual.size == 0:
        raise RuntimeError("no finite background pointmap residuals")
    return PointmapSimilarity(
        scale=scale,
        rotation=rotation.astype(np.float64),
        translation=translation.astype(np.float64),
        correspondences=int(ratios.size),
        inliers=int(keep.sum()),
        median_residual=float(np.median(active_residual)),
        p90_residual=float(np.quantile(active_residual, 0.9)),
    )


def transform_cameras_by_similarity(
    cam2world: np.ndarray, similarity: PointmapSimilarity
) -> np.ndarray:
    """Map camera-to-world poses through a world-space Sim(3)."""
    cam2world = np.asarray(cam2world, dtype=np.float64)
    if cam2world.ndim != 3 or cam2world.shape[1:] != (4, 4):
        raise ValueError(f"cam2world must be (T,4,4), got {cam2world.shape}")
    result = np.broadcast_to(np.eye(4, dtype=np.float64), cam2world.shape).copy()
    result[:, :3, :3] = similarity.rotation[None] @ cam2world[:, :3, :3]
    result[:, :3, 3] = _apply_to_points(
        cam2world[:, :3, 3],
        similarity.scale,
        similarity.rotation,
        similarity.translation,
    )
    return result.astype(np.float32)
