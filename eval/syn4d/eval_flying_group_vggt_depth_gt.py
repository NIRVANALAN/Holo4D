#!/usr/bin/env python3
"""Evaluate VGGT-Omega depth from generated videos against Syn4D target GT depth.

The generated-video depth is monocular and has an arbitrary global scale. Each
49-frame prediction is therefore aligned once with a robust median scale over
all valid/confident pixels. No per-frame alignment is allowed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR


METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3")


def read_exr_depth(path: Path) -> np.ndarray:
    channels = OpenEXR.File(str(path)).channels()
    if "Depth" not in channels:
        raise KeyError(f"{path}: missing Depth channel; found {sorted(channels)}")
    depth = np.asarray(channels["Depth"].pixels, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"{path}: expected HxW depth, got {depth.shape}")
    return depth


def normalize_depth_array(array: np.ndarray) -> np.ndarray:
    depth = np.asarray(array, dtype=np.float32)
    while depth.ndim > 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    while depth.ndim > 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 3:
        raise ValueError(f"expected TxHxW depth, got {depth.shape}")
    return depth


def load_vggt_depth(path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    depth = normalize_depth_array(data["depth"])[:num_frames]
    conf = normalize_depth_array(data["depth_conf"])[:num_frames]
    if depth.shape != conf.shape or depth.shape[0] != num_frames:
        raise ValueError(f"{path}: invalid depth/conf shapes {depth.shape}/{conf.shape}")
    return depth, conf


def load_gt_depth(pair_dir: Path, num_frames: int, shape_hw: tuple[int, int]) -> np.ndarray:
    paths = sorted(pair_dir.glob("*.exr"))
    if len(paths) != num_frames:
        raise RuntimeError(f"{pair_dir}: expected {num_frames} EXRs, found {len(paths)}")
    target_h, target_w = shape_hw
    frames = []
    for path in paths:
        depth = read_exr_depth(path)
        if depth.shape != shape_hw:
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        frames.append(depth.astype(np.float32))
    return np.stack(frames) / 100.0  # Syn4D/Unreal centimetres -> metres.


def compute_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[dict, float]:
    if not np.any(valid):
        raise RuntimeError("no valid depth pixels")
    ratios = gt[valid] / np.maximum(pred[valid], 1e-8)
    scale = float(np.median(ratios[np.isfinite(ratios)]))
    pred_aligned = pred * scale
    p = np.maximum(pred_aligned[valid].astype(np.float64), 1e-8)
    g = np.maximum(gt[valid].astype(np.float64), 1e-8)
    diff = p - g
    ratio = np.maximum(p / g, g / p)
    metrics = {
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "sq_rel": float(np.mean(diff * diff / g)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "log_rmse": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "delta_2": float(np.mean(ratio < 1.25 ** 2)),
        "delta_3": float(np.mean(ratio < 1.25 ** 3)),
        "valid_pixels": int(valid.sum()),
    }
    return metrics, scale


def aggregate_rows(rows: list[dict]) -> dict:
    weights = np.asarray([row["valid_pixels"] for row in rows], dtype=np.float64)
    out = {"num_pairs": len(rows), "valid_pixels": int(weights.sum())}
    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = float(np.average(values, weights=weights))
        out[f"{key}_pair_mean"] = float(values.mean())
        out[f"{key}_pair_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    out["median_scale_pair_mean"] = float(np.mean([row["median_scale"] for row in rows]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--stage_manifest", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--conf_percentile", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    stage = json.loads(args.stage_manifest.read_text())
    rows = []
    failures = []
    for item in stage["rows"]:
        method, pair, sequence_name = item["method"], item["pair"], item["sequence_name"]
        cache = args.cache_root / sequence_name / "geometry" / "vggt_omega_depth.npz"
        gt_dir = args.bundle_root / "depth" / "target" / pair
        try:
            pred, conf = load_vggt_depth(cache, args.num_frames)
            gt = load_gt_depth(gt_dir, args.num_frames, pred.shape[1:])
            finite_conf = conf[np.isfinite(conf)]
            threshold = float(np.percentile(finite_conf, args.conf_percentile))
            valid = (
                np.isfinite(pred) & np.isfinite(gt) & np.isfinite(conf)
                & (pred > 1e-8) & (gt > 1e-8) & (conf >= threshold)
            )
            metrics, scale = compute_metrics(pred, gt, valid)
            row = {
                "method": method,
                "pair": pair,
                "reconstructor": "vggt_omega",
                "alignment": "one robust median scale per 49-frame video",
                "median_scale": scale,
                "confidence_percentile": args.conf_percentile,
                **metrics,
            }
            rows.append(row)
            print(f"[depth-gt] {method} {pair} AbsRel={row['abs_rel']:.6f} RMSE={row['rmse']:.6f}")
        except Exception as exc:
            failures.append({"method": method, "pair": pair, "error": repr(exc)})
            print(f"[depth-gt/fail] {method} {pair}: {exc}")
            if args.strict:
                raise

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    aggregate = {method: aggregate_rows(items) for method, items in sorted(grouped.items())}
    report = {
        "reference": "Syn4D flying_group target Depth EXR ground truth",
        "prediction": "VGGT-Omega depth reconstructed from generated target RGB video",
        "alignment": "one robust median scale per 49-frame video; no per-frame alignment",
        "gt_depth_unit": "metres (raw Syn4D Depth EXR divided by 100)",
        "rmse_unit": "metres",
        "mask": f"finite positive GT/pred and VGGT confidence >= pair percentile {args.conf_percentile}",
        "num_expected": stage["staged"],
        "num_evaluated": len(rows),
        "num_failures": len(failures),
        "aggregate": aggregate,
        "per_pair": rows,
        "failures": failures,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "depth_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["method", "pair", "median_scale", *METRIC_KEYS, "valid_pixels"]
    with (args.output_root / "depth_metrics_per_pair.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    with (args.output_root / "depth_metrics_aggregate.tsv").open("w", newline="") as handle:
        fields_agg = ["method", "num_pairs", *METRIC_KEYS, "valid_pixels"]
        writer = csv.DictWriter(handle, fieldnames=fields_agg, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for method, metrics in aggregate.items(): writer.writerow({"method": method, **metrics})
    if args.strict and len(rows) != stage["staged"]:
        raise RuntimeError(f"evaluated {len(rows)}/{stage['staged']}")
    print(f"[depth-gt] evaluated {len(rows)}/{stage['staged']}; output={args.output_root}")


if __name__ == "__main__":
    main()
