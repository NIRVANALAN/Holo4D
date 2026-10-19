#!/usr/bin/env python3
"""Evaluate 4RC depth from generated videos against Syn4D target GT depth."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
SHARED_EVALUATOR = THIS_DIR / "eval_flying_group_vggt_depth_gt.py"


def load_shared_evaluator():
    spec = importlib.util.spec_from_file_location("syn4d_depth_gt_shared", SHARED_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SHARED_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_4rc_depth_conf(path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    depth_frames = []
    confidence_frames = []
    for index in range(num_frames):
        points_key = f"pred_{index}_pts"
        confidence_key = f"pred_{index}_conf"
        if points_key not in data or confidence_key not in data:
            raise KeyError(f"{path}: missing {points_key} or {confidence_key}")
        points = np.asarray(data[points_key])
        confidence = np.asarray(data[confidence_key])
        if points.ndim == 4 and points.shape[0] == 1:
            points = points[0]
        if confidence.ndim == 3 and confidence.shape[0] == 1:
            confidence = confidence[0]
        depth_frames.append(points[..., 2].astype(np.float32))
        confidence_frames.append(confidence.astype(np.float32))
    return np.stack(depth_frames), np.stack(confidence_frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--recon_root", type=Path, required=True)
    parser.add_argument("--stage_manifest", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--conf_percentile", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    shared = load_shared_evaluator()
    stage = json.loads(args.stage_manifest.read_text())
    rows = []
    failures = []
    for item in stage["rows"]:
        method = item["method"]
        pair = item["pair"]
        sequence_name = item["sequence_name"]
        result_path = args.recon_root / sequence_name / "result.npz"
        gt_dir = args.bundle_root / "depth" / "target" / pair
        try:
            pred, confidence = load_4rc_depth_conf(result_path, args.num_frames)
            gt = shared.load_gt_depth(gt_dir, args.num_frames, pred.shape[1:])
            finite_confidence = confidence[np.isfinite(confidence)]
            threshold = float(np.percentile(finite_confidence, args.conf_percentile))
            valid = (
                np.isfinite(pred)
                & np.isfinite(gt)
                & np.isfinite(confidence)
                & (pred > 1e-8)
                & (gt > 1e-8)
                & (confidence >= threshold)
            )
            metrics, scale = shared.compute_metrics(pred, gt, valid)
            row = {
                "method": method,
                "pair": pair,
                "reconstructor": "4rc",
                "alignment": "one robust median scale per 49-frame video",
                "median_scale": scale,
                "confidence_percentile": args.conf_percentile,
                **metrics,
            }
            rows.append(row)
            print(
                f"[4rc-depth-gt] {method} {pair} "
                f"AbsRel={row['abs_rel']:.6f} RMSE={row['rmse']:.6f}"
            )
        except Exception as exc:
            failures.append({"method": method, "pair": pair, "error": repr(exc)})
            print(f"[4rc-depth-gt/fail] {method} {pair}: {exc}")
            if args.strict:
                raise

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    aggregate = {
        method: shared.aggregate_rows(method_rows)
        for method, method_rows in sorted(grouped.items())
    }
    report = {
        "reference": "Syn4D flying_group target Depth EXR ground truth",
        "prediction": "4RC depth reconstructed from generated target RGB video",
        "alignment": "one robust median scale per 49-frame video; no per-frame alignment",
        "gt_depth_unit": "metres (raw Syn4D Depth EXR divided by 100)",
        "rmse_unit": "metres",
        "mask": (
            "finite positive GT/pred and 4RC confidence >= pair percentile "
            f"{args.conf_percentile}"
        ),
        "num_expected": stage["staged"],
        "num_evaluated": len(rows),
        "num_failures": len(failures),
        "aggregate": aggregate,
        "per_pair": rows,
        "failures": failures,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "depth_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    per_pair_fields = [
        "method",
        "pair",
        "median_scale",
        *shared.METRIC_KEYS,
        "valid_pixels",
    ]
    with (args.output_root / "depth_metrics_per_pair.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=per_pair_fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    aggregate_fields = ["method", "num_pairs", *shared.METRIC_KEYS, "valid_pixels"]
    with (args.output_root / "depth_metrics_aggregate.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=aggregate_fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for method, metrics in aggregate.items():
            writer.writerow({"method": method, **metrics})
    if args.strict and len(rows) != stage["staged"]:
        raise RuntimeError(f"evaluated {len(rows)}/{stage['staged']}")
    print(f"[4rc-depth-gt] evaluated {len(rows)}/{stage['staged']}")


if __name__ == "__main__":
    main()
