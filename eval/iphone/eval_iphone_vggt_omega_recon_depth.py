#!/usr/bin/env python3
"""VGGT-Omega target-RGB pseudo-depth evaluation for DyCheck NVS."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def load_helpers(script_dir: Path):
    path = script_dir / "eval_iphone_4rc_recon_depth.py"
    spec = importlib.util.spec_from_file_location("iphone_4rc_depth", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_depth(path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    depth = np.asarray(data["depth"], dtype=np.float32)
    conf = np.asarray(data["depth_conf"], dtype=np.float32)
    # Current VGGT-Omega writer stores depth as (T,H,W) but confidence as
    # (1,T,H,W). Both describe the same T frames; remove singleton batch axes.
    while depth.ndim > 3 and depth.shape[0] == 1:
        depth = depth[0]
    while conf.ndim > 3 and conf.shape[0] == 1:
        conf = conf[0]
    while depth.ndim > 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    while conf.ndim > 3 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if depth.ndim != 3 or conf.ndim != 3 or depth.shape[0] < num_frames:
        raise ValueError(f"bad VGGT depth cache {path}: {depth.shape}/{conf.shape}")
    return depth[:num_frames], conf[:num_frames]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_root", type=Path, required=True)
    parser.add_argument("--geo4d_repo", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--mask_region", default="valid_covisible", choices=["fg_valid_covisible", "foreground", "valid_covisible", "all"])
    parser.add_argument("--conf_percentile", type=float, default=5.0)
    args = parser.parse_args()
    helpers = load_helpers(Path(__file__).parent)
    evaluator = helpers.load_depth_eval(args.geo4d_repo)
    manifest = json.loads((args.geometry_root / "manifest.json").read_text())
    rows = manifest["rows"]
    gt_rows = {row["iphone_sequence"]: row for row in rows if row["method"] == "gt"}
    out_rows = []
    for row in rows:
        if row["method"] == "gt":
            continue
        gt_row = gt_rows[row["iphone_sequence"]]
        gt_path = args.geometry_root / "vggt_omega" / gt_row["sequence_name"] / "geometry" / "vggt_omega_depth.npz"
        pred_path = args.geometry_root / "vggt_omega" / row["sequence_name"] / "geometry" / "vggt_omega_depth.npz"
        if not gt_path.exists() or not pred_path.exists():
            print(f"[skip] {row['sequence_name']}: missing VGGT cache")
            continue
        gt_depth, gt_conf = load_depth(gt_path, args.num_frames)
        pred_depth, pred_conf = load_depth(pred_path, args.num_frames)
        if gt_depth.shape != pred_depth.shape:
            raise ValueError(f"grid mismatch {row['sequence_name']}: {gt_depth.shape} vs {pred_depth.shape}")
        valid = np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > 0) & (pred_depth > 0)
        valid &= helpers.confidence_mask(gt_conf, pred_conf, args.conf_percentile)
        valid &= helpers.target_region_mask(Path(manifest["br_iphone"]), row["iphone_sequence"], args.num_frames, gt_depth.shape[2], gt_depth.shape[1], args.mask_region, int(gt_row.get("crop_width", gt_depth.shape[2])), int(gt_row.get("crop_height", gt_depth.shape[1])))
        metrics, _, _, _ = evaluator(pred_depth, gt_depth, custom_mask=valid, align_mask=torch.from_numpy(valid.reshape(-1, valid.shape[-1])), align_with_lad2=True, lr=1e-2, max_iters=5000, use_gpu=False)
        out_rows.append({"method": row["method"], "iphone_sequence": row["iphone_sequence"], **{key: float(value) for key, value in metrics.items()}})
    grouped: dict[str, list[dict]] = {}
    for row in out_rows:
        grouped.setdefault(row["method"], []).append(row)
    aggregate = {}
    for method, items in grouped.items():
        weights = np.asarray([max(float(item.get("valid_pixels", 0)), 0) for item in items])
        if weights.sum() <= 0: weights = np.ones(len(items))
        aggregate[method] = {"valid_pixels": float(weights.sum())}
        for key, value in items[0].items():
            if key not in {"method", "iphone_sequence", "valid_pixels"} and isinstance(value, float):
                aggregate[method][key] = float(np.average([item[key] for item in items], weights=weights))
    out_dir = args.geometry_root / "vggt_omega_depth_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"reference": "VGGT-Omega on GT target RGB; pseudo-reference, not target depth GT", "mask_region": args.mask_region, "per_sequence": out_rows, "aggregate": aggregate}
    (out_dir / "depth_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    keys = ["Abs Rel", "Sq Rel", "RMSE", "Log RMSE", "δ < 1.25", "δ < 1.25^2", "δ < 1.25^3", "valid_pixels"]
    with (out_dir / "depth_metrics_aggregate.tsv").open("w") as handle:
        handle.write("method\t" + "\t".join(keys) + "\n")
        for method, metrics in aggregate.items(): handle.write(method + "\t" + "\t".join(str(metrics.get(key, "")) for key in keys) + "\n")


if __name__ == "__main__":
    main()
