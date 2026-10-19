#!/usr/bin/env python3
"""Evaluate Vista4D camera control with source-conditioned joint Pi3 poses.

This follows the camera protocol in Vista4D's supplementary material:

1. concatenate the source and generated clips and reconstruct them jointly;
2. fit the predicted source cameras to the released source cameras using only
   the source segment;
3. apply that source-only transform to the generated cameras; and
4. compare generated and native target extrinsics and vertical field of view.

The output reports both literal camera-center Umeyama and an orientation-
stabilized Sim(3).  Center-only Umeyama cannot identify a reliable global
rotation when the source camera is static or nearly static, whereas the
stabilized variant obtains that rotation from the source camera orientations.
Neither alignment observes target cameras.  That distinction is the main
difference from the repo's generated-video-only VGGT-Omega diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def generated_only_vggt_metrics(predicted_cameras: np.ndarray, target_cameras: np.ndarray) -> dict:
    """Return the legacy generated-only ATE/RPE protocol for one pair.

    The Vista4D/Pi3 headline metrics below use a source-only Sim(3) fitted on
    the first 49 source frames.  The older VGGT-Omega tables instead align the
    generated trajectory itself with Sim(3) and report ATE/RPE.  Keep that
    protocol available under an explicit namespace so the two alignments are
    never mistaken for one another.
    """
    eval_dir = Path(__file__).resolve().parents[2] / "eval" / "common"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from vipe_camera_eval import compute_pose_metrics

    metrics = compute_pose_metrics(
        np.asarray(predicted_cameras, dtype=np.float64),
        np.asarray(target_cameras, dtype=np.float64),
    )
    return {
        "ate": metrics.get("ate"),
        "rpe_translation": metrics.get("trans_err"),
        "rpe_rotation_deg": metrics.get("rot_err"),
        "alignment_mode": metrics.get("alignment_mode"),
        "ate_norm": metrics.get("ate_norm"),
        "rpe_translation_norm": metrics.get("trans_err_norm"),
    }


def read_pairs(path: Path, limit: int | None) -> list[tuple[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = [(row["video"], row.get("trajectory") or row["camera"]) for row in rows]
    return pairs if limit is None else pairs[:limit]


def resolve_prediction(root: Path, video: str, trajectory: str, names: Iterable[str]) -> Path:
    for name in names:
        candidate = root / video / trajectory / name.format(traj=trajectory)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"prediction missing for {video}/{trajectory} under {root}")


def load_standardized_video(path: Path, num_frames: int, height: int, width: int) -> np.ndarray:
    from utils.media import crop_and_resize_video, load_video

    video, _ = load_video(str(path))
    if video.shape[0] < num_frames:
        raise RuntimeError(f"{path} has {video.shape[0]} frames; expected at least {num_frames}")
    video = np.asarray(video[:num_frames], dtype=np.uint8)
    return crop_and_resize_video(video, height, width, resample="lanczos")


def umeyama_similarity(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``scale, rotation, translation, singular_values`` mapping source to target."""
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if source_xyz.shape != target_xyz.shape or source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError(f"invalid Umeyama inputs: {source_xyz.shape}, {target_xyz.shape}")

    source_mean = source_xyz.mean(axis=0)
    target_mean = target_xyz.mean(axis=0)
    source_centered = source_xyz - source_mean
    target_centered = target_xyz - target_mean
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= 1e-14:
        raise RuntimeError("predicted source-camera trajectory is degenerate for Umeyama alignment")

    covariance = target_centered.T @ source_centered / source_xyz.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    parity = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        parity[-1] = -1.0
    rotation = u @ np.diag(parity) @ vt
    scale = float(np.sum(singular_values * parity) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid Umeyama scale: {scale}")
    return scale, rotation, translation, singular_values


def pose_stabilized_similarity(
    source_cameras: np.ndarray,
    target_cameras: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a source-only Sim(3), stabilizing rotation with camera orientations.

    A camera-center path that barely translates is rank deficient: a literal
    point-only Umeyama fit can rotate almost arbitrarily while preserving its
    tiny positional residual.  Camera orientations still constrain the global
    rotation.  We compute their chordal SO(3) mean, then solve scale and
    translation by least squares with that rotation fixed.
    """
    source_cameras = np.asarray(source_cameras, dtype=np.float64)
    target_cameras = np.asarray(target_cameras, dtype=np.float64)
    if source_cameras.shape != target_cameras.shape or source_cameras.shape[1:] != (4, 4):
        raise ValueError(f"invalid pose-alignment inputs: {source_cameras.shape}, {target_cameras.shape}")

    relative_rotations = np.einsum(
        "tij,tkj->tik", target_cameras[:, :3, :3], source_cameras[:, :3, :3]
    )
    u, _, vt = np.linalg.svd(relative_rotations.sum(axis=0))
    parity = np.eye(3, dtype=np.float64)
    parity[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ parity @ vt

    source_xyz = np.einsum("ij,tj->ti", rotation, source_cameras[:, :3, 3])
    target_xyz = target_cameras[:, :3, 3]
    source_mean = source_xyz.mean(axis=0)
    target_mean = target_xyz.mean(axis=0)
    source_centered = source_xyz - source_mean
    target_centered = target_xyz - target_mean
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 1e-14:
        raise RuntimeError("predicted source-camera centers are constant; pose scale is not identifiable")
    target_energy = float(np.sum(target_centered * target_centered))
    if target_energy <= 1e-14:
        raise RuntimeError("reference source-camera centers are constant; pose scale is not identifiable")
    # Use the positive RMS motion ratio.  A signed dot-product fit is unstable
    # when the released source path has only millimetric motion: estimator
    # jitter can make the correlation negative even though Sim(3) scale must
    # be positive.
    scale = math.sqrt(target_energy / denominator)
    translation = target_mean - scale * source_mean
    return scale, rotation, translation


def apply_similarity(cameras: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    aligned = np.asarray(cameras, dtype=np.float64).copy()
    aligned[:, :3, :3] = np.einsum("ij,tjk->tik", rotation, aligned[:, :3, :3])
    aligned[:, :3, 3] = scale * np.einsum("ij,tj->ti", rotation, aligned[:, :3, 3]) + translation
    return aligned


def target_cameras_in_source_world(
    target_cameras: np.ndarray,
    source_reference: np.ndarray,
    target_camera_gauge: str,
) -> np.ndarray:
    """Express target cameras in the source-reference world frame.

    Vista4D's released target cameras and released source cameras already
    share one world frame.  Holo4D's exported *effective* cameras do not:
    the 4RC/VGGT backend normalizes its query camera to identity before the
    official trajectory is anchored to it.  Those targets are therefore
    source-frame-0-local cameras.  Comparing them directly with a joint-Pi3
    reconstruction aligned to Vista4D's released source cameras injects the
    backend-to-Pi3 gauge rotation into RotErr, even when generated frame zero
    exactly matches source frame zero.

    ``source_frame0_local`` performs the missing rigid gauge lift.  It changes
    only the coordinate frame used by the evaluator; it does not alter the
    requested relative camera trajectory or any generated video.
    """
    target_cameras = np.asarray(target_cameras, dtype=np.float64)
    source_reference = np.asarray(source_reference, dtype=np.float64)
    if target_camera_gauge == "shared_world":
        return target_cameras.copy()
    if target_camera_gauge == "source_frame0_local":
        return np.einsum("ij,tjk->tik", source_reference[0], target_cameras)
    raise ValueError(f"unsupported target camera gauge: {target_camera_gauge!r}")


def rotation_error_degrees(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    relative = np.einsum("tij,tkj->tik", reference[:, :3, :3], estimate[:, :3, :3])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def vertical_fov_degrees(intrinsics: np.ndarray, height: int) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    # Vista4D's released camera NPZ stores [fx, fy, cx, cy], while Syn4D
    # target cameras use the conventional 3x3 K matrix. Keep both formats in
    # the same evaluator instead of rewriting calibrated Syn4D camera files.
    if intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
        fy = intrinsics[:, 1, 1]
        principal_y = intrinsics[:, 1, 2]
    elif intrinsics.ndim == 2 and intrinsics.shape[1] >= 4:
        fy = intrinsics[:, 1]
        principal_y = intrinsics[:, 3]
    else:
        raise RuntimeError(f"unsupported intrinsic shape for FOV: {intrinsics.shape}")
    if np.any(fy <= 0):
        raise RuntimeError("non-positive focal length in Pi3/target intrinsics")
    # Camera files retain their native pixel calibration (the released target
    # cameras are 1280x720), while generated clips may be evaluated at 672x384.
    # FOV is resolution invariant, so derive the calibrated sensor height from
    # the centered principal point instead of mixing a resized height with a
    # native focal length.  Fall back to the supplied frame height only for a
    # malformed/non-centred entry.
    calibrated_height = 2.0 * principal_y
    calibrated_height = np.where(calibrated_height > 0, calibrated_height, float(height))
    return np.degrees(2.0 * np.arctan(calibrated_height / (2.0 * fy)))


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "n": int(array.size),
    }


def camera_errors(
    target_cameras: np.ndarray,
    aligned_cameras: np.ndarray,
) -> dict[str, float]:
    translation_delta = target_cameras[:, :3, 3] - aligned_cameras[:, :3, 3]
    translation_l2 = np.linalg.norm(translation_delta, axis=1)
    return {
        "translation_error_squared": float(np.sum(translation_delta * translation_delta, axis=1).mean()),
        "translation_error_l2": float(translation_l2.mean()),
        "rotation_error_deg": float(rotation_error_degrees(target_cameras, aligned_cameras).mean()),
    }


def valid_pose_cache(path: Path, expected_frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        data = np.load(path, allow_pickle=False)
        return (
            data["cam_c2w"].shape == (expected_frames, 4, 4)
            and data["intrinsics"].shape == (expected_frames, 4)
            and np.isfinite(data["cam_c2w"]).all()
            and np.isfinite(data["intrinsics"]).all()
        )
    except (OSError, ValueError, KeyError):
        return False


def save_pose_cache(path: Path, cameras: np.ndarray, intrinsics: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        cam_c2w=np.asarray(cameras, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
    )
    temporary.replace(path)


def evaluate_pair(
    video: str,
    trajectory: str,
    prediction: Path,
    source_root: Path,
    target_camera_root: Path,
    cache_path: Path,
    model,
    num_frames: int,
    height: int,
    width: int,
    pixel_limit: int,
    force_pose: bool,
    target_camera_gauge: str,
) -> dict:
    source_video_path = source_root / video / "video.mp4"
    source_camera_path = source_root / video / "cameras.npz"
    target_camera_path = target_camera_root / video / f"{trajectory}.npz"
    for required in (source_video_path, source_camera_path, target_camera_path, prediction):
        if not required.is_file():
            raise FileNotFoundError(required)

    expected_joint_frames = 2 * num_frames
    if force_pose or not valid_pose_cache(cache_path, expected_joint_frames):
        if model is None:
            raise RuntimeError(f"Pi3 model is required because pose cache is missing: {cache_path}")
        import torch
        from utils.recon_and_seg.recon_pi3 import run_pi3

        source_rgb = load_standardized_video(source_video_path, num_frames, height, width)
        generated_rgb = load_standardized_video(prediction, num_frames, height, width)
        joint_rgb = np.concatenate([source_rgb, generated_rgb], axis=0)
        # Vista4D's official preprocessing entrypoint decorates its caller with
        # ``torch.no_grad()``, while ``run_pi3`` itself does not.  Keep that
        # contract here; otherwise the 36-layer 98-frame decoder retains its
        # full autograd graph and can exhaust even a 141-GiB H200.
        with torch.inference_mode():
            _, joint_cameras, joint_intrinsics = run_pi3(
                joint_rgb,
                model=model,
                pixel_limit=pixel_limit,
                resize_output=True,
                dtype=torch.bfloat16,
                device="cuda",
                verbose=True,
            )
        save_pose_cache(cache_path, joint_cameras, joint_intrinsics)

    joint = np.load(cache_path, allow_pickle=False)
    predicted_source = np.asarray(joint["cam_c2w"][:num_frames], dtype=np.float64)
    predicted_generated = np.asarray(joint["cam_c2w"][num_frames:expected_joint_frames], dtype=np.float64)
    predicted_intrinsics = np.asarray(joint["intrinsics"][num_frames:expected_joint_frames], dtype=np.float64)

    source_reference = np.asarray(np.load(source_camera_path)["cam_c2w"][:num_frames], dtype=np.float64)
    target_data = np.load(target_camera_path)
    target_cameras = np.asarray(target_data["cam_c2w"][:num_frames], dtype=np.float64)
    target_cameras = target_cameras_in_source_world(
        target_cameras,
        source_reference,
        target_camera_gauge,
    )
    target_intrinsics = np.asarray(target_data["intrinsics"][:num_frames], dtype=np.float64)

    center_scale, center_rotation, center_translation, singular_values = umeyama_similarity(
        predicted_source[:, :3, 3], source_reference[:, :3, 3]
    )
    center_aligned_source = apply_similarity(predicted_source, center_scale, center_rotation, center_translation)
    center_aligned_generated = apply_similarity(
        predicted_generated, center_scale, center_rotation, center_translation
    )

    pose_scale, pose_rotation, pose_translation = pose_stabilized_similarity(
        predicted_source, source_reference
    )
    pose_aligned_source = apply_similarity(predicted_source, pose_scale, pose_rotation, pose_translation)
    pose_aligned_generated = apply_similarity(
        predicted_generated, pose_scale, pose_rotation, pose_translation
    )

    target_fov = vertical_fov_degrees(target_intrinsics, height)
    generated_fov = vertical_fov_degrees(predicted_intrinsics, height)
    fov_error_deg = np.abs(target_fov - generated_fov)
    try:
        generated_only_metrics = generated_only_vggt_metrics(predicted_generated, target_cameras)
    except Exception as exc:  # Keep the Pi3 score usable if evo is unavailable.
        generated_only_metrics = {
            "ate": None,
            "rpe_translation": None,
            "rpe_rotation_deg": None,
            "alignment_mode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "video": video,
        "trajectory": trajectory,
        "target_camera_gauge": target_camera_gauge,
        "prediction": str(prediction),
        "pose_cache": str(cache_path),
        "center_umeyama": {
            "scale": center_scale,
            "source_center_rmse": float(np.sqrt(np.mean(np.sum(
                (center_aligned_source[:, :3, 3] - source_reference[:, :3, 3]) ** 2, axis=1
            )))),
            "source_rotation_error_deg": float(rotation_error_degrees(
                source_reference, center_aligned_source
            ).mean()),
            "singular_values": singular_values.tolist(),
            **camera_errors(target_cameras, center_aligned_generated),
        },
        "pose_stabilized": {
            "scale": pose_scale,
            "source_center_rmse": float(np.sqrt(np.mean(np.sum(
                (pose_aligned_source[:, :3, 3] - source_reference[:, :3, 3]) ** 2, axis=1
            )))),
            "source_rotation_error_deg": float(rotation_error_degrees(
                source_reference, pose_aligned_source
            ).mean()),
            **camera_errors(target_cameras, pose_aligned_generated),
        },
        "intrinsics_fov_error_deg": float(fov_error_deg.mean()),
        "generated_only_vggt_metrics": generated_only_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--prediction-filenames", default="pred.mp4,pred_rgb.mp4,{traj}.mp4")
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-camera-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--vista4d-repo", type=Path, required=True)
    parser.add_argument("--model-id", default="yyfz233/Pi3")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--pixel-limit", type=int, default=255000)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--force-pose", action="store_true")
    parser.add_argument(
        "--target-camera-gauge",
        choices=("shared_world", "source_frame0_local"),
        default="shared_world",
        help=(
            "Coordinate frame of target-camera NPZs. Use source_frame0_local "
            "for Holo4D effective-camera exports whose frame zero is the "
            "backend-normalized source camera; released Vista4D cameras use "
            "shared_world."
        ),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.vista4d_repo))
    pairs = read_pairs(args.pairs_csv, args.limit_pairs)
    names = [name.strip() for name in args.prediction_filenames.split(",") if name.strip()]
    predictions = [resolve_prediction(args.prediction_root, video, trajectory, names) for video, trajectory in pairs]
    caches = [args.cache_root / f"{video}__{trajectory}" / "pi3_joint.npz" for video, trajectory in pairs]
    need_model = args.force_pose or not all(valid_pose_cache(path, 2 * args.num_frames) for path in caches)
    print(f"[pi3-camera] model={args.model_id} pairs={len(pairs)} need_model={need_model}")
    model = None
    if need_model:
        import torch
        from utils.recon_and_seg.recon_pi3 import init_pi3

        if not torch.cuda.is_available():
            raise RuntimeError("Pi3 camera evaluation requires CUDA when a pose cache is absent")
        model = init_pi3(model_id=args.model_id, device="cuda")

    per_pair = []
    for index, (video, trajectory) in enumerate(pairs, start=1):
        prediction = predictions[index - 1]
        cache_path = caches[index - 1]
        print(f"[pi3-camera] {index:03d}/{len(pairs):03d} {video}/{trajectory}")
        row = evaluate_pair(
            video=video,
            trajectory=trajectory,
            prediction=prediction,
            source_root=args.source_root,
            target_camera_root=args.target_camera_root,
            cache_path=cache_path,
            model=model,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            pixel_limit=args.pixel_limit,
            force_pose=args.force_pose,
            target_camera_gauge=args.target_camera_gauge,
        )
        per_pair.append(row)
        print(
            "[pi3-camera-result] "
            f"trans2={row['pose_stabilized']['translation_error_squared']:.6f} "
            f"trans={row['pose_stabilized']['translation_error_l2']:.6f} "
            f"rot={row['pose_stabilized']['rotation_error_deg']:.6f}deg "
            f"fov={row['intrinsics_fov_error_deg']:.6f}deg"
        )

    def aggregate_alignment(name: str) -> dict[str, dict[str, float | int]]:
        keys = (
            "translation_error_squared",
            "translation_error_l2",
            "rotation_error_deg",
            "source_center_rmse",
            "source_rotation_error_deg",
        )
        return {key: stats([row[name][key] for row in per_pair]) for key in keys}

    summary = {
        "protocol": "vista4d_joint_source_generated_pi3_source_only_pose_alignment",
        "pose_estimator": args.model_id,
        "prediction_root": str(args.prediction_root),
        "num_pairs": len(per_pair),
        "num_frames_per_segment": args.num_frames,
        "joint_frames": 2 * args.num_frames,
        "evaluation_resolution": [args.height, args.width],
        "primary_alignment": "pose_stabilized",
        "target_camera_gauge": args.target_camera_gauge,
        "pose_stabilized": aggregate_alignment("pose_stabilized"),
        "center_umeyama": aggregate_alignment("center_umeyama"),
        "intrinsics_fov_error_deg": stats([row["intrinsics_fov_error_deg"] for row in per_pair]),
        "per_pair": per_pair,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    headline = {
        key: value["mean"] for key, value in summary["pose_stabilized"].items()
        if key in ("translation_error_squared", "translation_error_l2", "rotation_error_deg")
    }
    headline["intrinsics_fov_error_deg"] = summary["intrinsics_fov_error_deg"]["mean"]
    print(f"[pi3-camera-summary] {json.dumps(headline, sort_keys=True)}")
    print(f"[pi3-camera] output={args.output_json}")


if __name__ == "__main__":
    main()
