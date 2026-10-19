#!/usr/bin/env python3
"""Merge sharded camera-metric JSONs into one method JSON."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _stats(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None, "median": None, "n": 0}
    vals = [float(v) for v in vals]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    sorted_vals = sorted(vals)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2:
        median = sorted_vals[mid]
    else:
        median = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "median": median,
        "n": len(vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--shard_results_root", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--camera_json_name", default="eval_camera_metrics_vipe.json")
    args = parser.parse_args()

    shard_jsons = [
        args.shard_results_root / f"shard{i}" / args.camera_json_name
        for i in range(args.num_shards)
    ]
    missing = [str(path) for path in shard_jsons if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing shard JSON(s):\n" + "\n".join(missing))

    shard_data = [json.loads(path.read_text()) for path in shard_jsons]
    per_pair: list[dict] = []
    for shard_idx, data in enumerate(shard_data):
        for row in data.get("per_pair") or []:
            merged = dict(row)
            merged["shard_index"] = shard_idx
            per_pair.append(merged)

    per_pair.sort(key=lambda r: (str(r.get("seq", "")), str(r.get("traj", ""))))
    sim3 = [r for r in per_pair if r.get("alignment_mode") == "umeyama_align_scale"]
    origin = [r for r in per_pair if r.get("alignment_mode") == "origin_align_only"]
    has_gt_trans = [r for r in per_pair if r.get("ate_norm") is not None]

    first = shard_data[0]
    summary = {
        "method_pred_root": first.get("method_pred_root"),
        "num_pairs_evaluated": len(per_pair),
        "num_pairs_total": sum(int(d.get("num_pairs_total") or 0) for d in shard_data),
        "pose_estimator": first.get("pose_estimator"),
        "standardize_input": first.get("standardize_input"),
        "sharded_eval": {
            "method": args.method,
            "num_shards": args.num_shards,
            "shard_results_root": str(args.shard_results_root),
            "camera_json_name": args.camera_json_name,
        },
        "ate": _stats(per_pair, "ate"),
        "trans_err": _stats(per_pair, "trans_err"),
        "rot_err": _stats(per_pair, "rot_err"),
        "sim3_subset": {
            "ate": _stats(sim3, "ate"),
            "trans_err": _stats(sim3, "trans_err"),
            "rot_err": _stats(sim3, "rot_err"),
        },
        "origin_subset": {
            "ate": _stats(origin, "ate"),
            "trans_err": _stats(origin, "trans_err"),
            "rot_err": _stats(origin, "rot_err"),
        },
        "all_norm": {
            "ate": _stats(has_gt_trans, "ate_norm"),
            "trans_err": _stats(has_gt_trans, "trans_err_norm"),
            "rot_err": _stats(per_pair, "rot_err"),
        },
        "per_pair": per_pair,
    }
    for key in ("vggt_omega", "da3"):
        if key in first:
            summary[key] = first[key]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[merge-camera] wrote {args.output_json}")
    print(f"[merge-camera] pairs: {len(per_pair)} / {summary['num_pairs_total']}")


if __name__ == "__main__":
    main()
