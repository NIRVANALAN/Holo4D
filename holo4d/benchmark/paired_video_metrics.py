"""Compute strict paired RGB reconstruction metrics for saved eval videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


_LPIPS_MODELS: dict[tuple[str, str], object] = {}


def load_video(path: Path) -> np.ndarray:
    frames = [np.asarray(frame)[..., :3] for frame in iio.imiter(path, plugin="FFMPEG")]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.uint8, copy=False)


def resize_prediction_to_gt_canvas(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Area-resize every prediction frame to the paired GT canvas.

    This is a direct spatial resize (aspect ratio is not preserved), matching
    the established Syn4D-HF first-pair audit protocol. It is opt-in because
    other benchmarks may require strict native-shape matching instead.
    """
    import cv2

    height, width = gt.shape[1:3]
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in pred]
    )


def compute_psnr_ssim(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import structural_similarity

    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0
    psnr_values = []
    ssim_values = []
    for pred_frame, gt_frame in zip(pred_f, gt_f):
        mse = float(np.mean((pred_frame - gt_frame) ** 2))
        # Keep exact matches finite so the machine-readable JSON remains
        # standards-compliant. Generated videos will normally be below this.
        psnr_values.append(100.0 if mse == 0.0 else -10.0 * np.log10(mse))
        ssim_values.append(
            float(structural_similarity(pred_frame, gt_frame, data_range=1.0, channel_axis=-1))
        )
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def compute_lpips(pred: np.ndarray, gt: np.ndarray, batch_size: int = 8) -> float:
    import lpips
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (device, "alex")
    model = _LPIPS_MODELS.get(key)
    if model is None:
        model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        _LPIPS_MODELS[key] = model

    pred_t = torch.from_numpy(pred).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    gt_t = torch.from_numpy(gt).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    values = []
    with torch.inference_mode():
        for start in range(0, len(pred_t), batch_size):
            end = start + batch_size
            values.append(model(pred_t[start:end].to(device), gt_t[start:end].to(device)).cpu())
    return float(np.concatenate([value.numpy().reshape(-1) for value in values]).mean())


def metric_stats(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(len(values)),
    }


def evaluate_root(
    eval_root: Path,
    expected_pairs: int,
    expected_frames: int,
    compute_perceptual: bool,
    require_perceptual: bool = False,
    excluded_pair_substrings: tuple[str, ...] = (),
    resize_prediction_to_gt: bool = False,
    truncate_prediction_to_expected_frames: bool = False,
) -> tuple[list[dict], dict]:
    pred_paths = sorted(eval_root.rglob("pred_rgb.mp4"))
    if len(pred_paths) != expected_pairs:
        raise RuntimeError(
            f"expected {expected_pairs} predictions under {eval_root}, found {len(pred_paths)}"
        )
    pred_paths = [
        path
        for path in pred_paths
        if not any(token in str(path.relative_to(eval_root)) for token in excluded_pair_substrings)
    ]
    if not pred_paths:
        raise RuntimeError("pair exclusion removed every prediction")
    rows = []
    lpips_error = None
    for index, pred_path in enumerate(pred_paths, start=1):
        pair_dir = pred_path.parent
        gt_path = pair_dir / "gt_rgb.mp4"
        if not gt_path.is_file() or gt_path.stat().st_size == 0:
            raise FileNotFoundError(f"paired GT missing: {gt_path}")
        pred = load_video(pred_path)
        gt = load_video(gt_path)
        native_frames = len(pred)
        native_height, native_width = pred.shape[1:3]
        if truncate_prediction_to_expected_frames and len(pred) > expected_frames:
            pred = pred[:expected_frames]
        if resize_prediction_to_gt and pred.shape[1:3] != gt.shape[1:3]:
            pred = resize_prediction_to_gt_canvas(pred, gt)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch for {pair_dir}: pred={pred.shape}, gt={gt.shape}")
        if len(pred) != expected_frames:
            raise ValueError(
                f"frame-count mismatch for {pair_dir}: expected {expected_frames}, got {len(pred)}"
            )
        psnr, ssim = compute_psnr_ssim(pred, gt)
        row = {
            "pair": str(pair_dir.relative_to(eval_root)),
            "frames": int(len(pred)),
            "native_frames": int(native_frames),
            "height": int(pred.shape[1]),
            "width": int(pred.shape[2]),
            "native_height": int(native_height),
            "native_width": int(native_width),
            "spatial_resize": "cv2.INTER_AREA_to_gt" if resize_prediction_to_gt else "none",
            "temporal_selection": (
                "first_expected_frames" if truncate_prediction_to_expected_frames else "strict_native"
            ),
            "psnr": psnr,
            "ssim": ssim,
        }
        if compute_perceptual and lpips_error is None:
            try:
                row["lpips_alex"] = compute_lpips(pred, gt)
            except Exception as exc:
                lpips_error = f"{type(exc).__name__}: {exc}"
                if require_perceptual:
                    raise RuntimeError(f"LPIPS-Alex is required but unavailable: {lpips_error}") from exc
                print(
                    "[paired-metrics] WARNING: LPIPS-Alex unavailable; continuing with "
                    f"PSNR/SSIM only ({lpips_error})",
                    flush=True,
                )
        rows.append(row)
        lpips_text = f" LPIPS={row['lpips_alex']:.4f}" if "lpips_alex" in row else ""
        print(
            f"[paired-metrics] {index}/{len(pred_paths)} {row['pair']}: "
            f"PSNR={psnr:.3f} SSIM={ssim:.4f}{lpips_text}",
            flush=True,
        )
    lpips_rows = sum("lpips_alex" in row for row in rows)
    metric_status = {
        "psnr": "complete",
        "ssim": "complete",
        "lpips_alex": (
            "disabled" if not compute_perceptual
            else "complete" if lpips_rows == len(rows)
            else "unavailable"
        ),
        "lpips_alex_error": lpips_error,
    }
    return rows, metric_status


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=50)
    parser.add_argument("--expected-frames", type=int, default=49)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--eval-mode",
        choices=["same_view", "ext1_to_ext2", "nvs"],
        required=True,
    )
    parser.add_argument("--dataset-split", default="omni_droid_eval_heldout50")
    parser.add_argument(
        "--split-caveat",
        default=(
            "The deterministic heldout50 designation is shared across arms, but these qdata "
            "checkpoints used the combined OmniDROID training index and did not exclude it."
        ),
        help="Protocol note stored verbatim in the result JSON.",
    )
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument(
        "--resize-prediction-to-gt",
        action="store_true",
        help="Directly area-resize prediction frames to the paired GT HxW before computing metrics.",
    )
    parser.add_argument(
        "--truncate-prediction-to-expected-frames",
        action="store_true",
        help="If a prediction is longer than expected, evaluate its first expected frames.",
    )
    parser.add_argument(
        "--require-lpips",
        action="store_true",
        help="Fail instead of recording LPIPS as unavailable when pretrained weights are absent.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--exclude-pair-substring",
        action="append",
        default=[],
        help="Exclude matching relative pair paths from aggregation; may be repeated.",
    )
    args = parser.parse_args()

    if not args.eval_root.is_dir():
        raise FileNotFoundError(args.eval_root)
    rows, metric_status = evaluate_root(
        args.eval_root,
        expected_pairs=args.expected_pairs,
        expected_frames=args.expected_frames,
        compute_perceptual=not args.skip_lpips,
        require_perceptual=args.require_lpips,
        excluded_pair_substrings=tuple(args.exclude_pair_substring),
        resize_prediction_to_gt=args.resize_prediction_to_gt,
        truncate_prediction_to_expected_frames=args.truncate_prediction_to_expected_frames,
    )
    metric_keys = ["psnr", "ssim"]
    if metric_status["lpips_alex"] == "complete":
        metric_keys.append("lpips_alex")
    result = {
        "schema_version": 1,
        "method": args.method,
        "checkpoint_step": args.checkpoint_step,
        "eval_mode": args.eval_mode,
        "dataset_split": args.dataset_split,
        "eval_root": str(args.eval_root.resolve()),
        "pair_count": len(rows),
        "raw_pair_count": args.expected_pairs,
        "excluded_pair_substrings": args.exclude_pair_substring,
        "expected_pairs": args.expected_pairs,
        "expected_frames": args.expected_frames,
        "spatial_policy": (
            "Prediction frames are directly area-resized to each paired GT HxW with OpenCV INTER_AREA; aspect ratio is not preserved."
            if args.resize_prediction_to_gt else "Strict native prediction/GT shape match."
        ),
        "temporal_policy": (
            "If a prediction is longer than expected, evaluate its first expected frames."
            if args.truncate_prediction_to_expected_frames else "Strict native prediction frame-count match."
        ),
        "target_video_used_for_inference": False,
        "split_is_training_excluded": False,
        "split_caveat": args.split_caveat,
        "metric_status": metric_status,
        "aggregate": {key: metric_stats(rows, key) for key in metric_keys},
        "per_pair": rows,
    }
    output_json = args.output_json or args.eval_root / "paired_video_metrics.json"
    output_csv = args.output_csv or args.eval_root / "paired_video_metrics.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(rows, output_csv)
    print(f"[paired-metrics] JSON: {output_json}")
    print(f"[paired-metrics] CSV:  {output_csv}")


if __name__ == "__main__":
    main()
