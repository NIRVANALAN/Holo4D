"""Flatten per-method `eval_camera_metrics_vipe.json` files into per-pair TSVs.

Each method's JSON already stores its 100-pair breakdown under the
`per_pair` key (one record per (seq, traj) with ate / trans_err /
rot_err / ate_norm / trans_err_norm / alignment_mode / ...). This
helper walks the results directory and emits two TSVs:

  per_pair_camera_metrics.tsv         long format, one row per
                                       (method, seq, traj). All methods.
  per_pair_camera_metrics_wide.tsv    wide format, one row per (seq, traj),
                                       columns = {method}__{metric} for the
                                       dev subset (ours_* + vista4d_4rc by
                                       default; extend via --extra_methods).

Usage:
  python eval/export_per_pair_camera_metrics.py
  python ... --results_root /custom/path --out_dir /custom/path
  python ... --extra_methods trajcrafter_stage2_4rc,ex4d_calibrated

The two output files land alongside summary.tsv at
  ${BASELINES_ROOT}/eval/results/per_pair_camera_metrics{,_wide}.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Canonical column order for the long TSV.
LONG_COLS = [
    "method", "seq", "traj",
    "ate", "trans_err", "rot_err",
    "ate_norm", "trans_err_norm",
    "alignment_mode",
    "gt_centered_norm", "pred_centered_norm",
    "num_aligned_poses", "shard_index",
]

# Metrics to materialize as method-prefixed columns in the wide TSV.
WIDE_METRICS = ["ate_norm", "trans_err_norm", "rot_err", "alignment_mode"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_results = os.environ.get(
        "BASELINES_ROOT",
        "outputs/baselines",
    ) + "/eval/results"
    p.add_argument("--results_root", type=str, default=default_results,
                   help="Directory containing per-method <method>/eval_camera_metrics_vipe.json.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write the TSVs (default: --results_root).")
    p.add_argument("--extra_methods", type=str, default="",
                   help="Comma-separated extra method names to include in the wide TSV (beyond ours_* + vista4d_4rc).")
    p.add_argument("--long_only", action="store_true",
                   help="Skip the wide TSV (only emit the long-format one).")
    return p.parse_args()


def _dev_subset(method: str, extra: set[str]) -> bool:
    return method.startswith("ours_") or method == "vista4d_4rc" or method in extra


def _read_method_per_pair(json_path: Path) -> list[dict]:
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        print(f"[per-pair] WARN: failed to load {json_path}: {exc}", file=sys.stderr)
        return []
    pp = data.get("per_pair") or []
    if not isinstance(pp, list):
        return []
    return pp


def _emit_long(results_root: Path, out_path: Path) -> tuple[int, list[str]]:
    """Write per_pair_camera_metrics.tsv (all methods, long). Returns (n_rows, methods)."""
    methods = sorted(
        d.name for d in results_root.iterdir()
        if d.is_dir() and (d / "eval_camera_metrics_vipe.json").is_file()
    )
    n_rows = 0
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(LONG_COLS)
        for method in methods:
            for row in _read_method_per_pair(results_root / method / "eval_camera_metrics_vipe.json"):
                w.writerow([_fmt(row.get(c) if c != "method" else method) for c in LONG_COLS])
                n_rows += 1
    return n_rows, methods


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        # 6 sig figs is plenty for inspection; keeps the file diff-friendly.
        return f"{v:.6g}"
    return str(v)


def _emit_wide(results_root: Path, out_path: Path, extra: set[str]) -> tuple[int, list[str]]:
    """Write per_pair_camera_metrics_wide.tsv (dev subset only)."""
    candidate_methods = sorted(
        d.name for d in results_root.iterdir()
        if d.is_dir() and (d / "eval_camera_metrics_vipe.json").is_file()
    )
    methods = [m for m in candidate_methods if _dev_subset(m, extra)]

    # Aggregate (seq, traj) → {(method, metric) → value}.
    rows: dict[tuple[str, str], dict] = {}
    for method in methods:
        for row in _read_method_per_pair(results_root / method / "eval_camera_metrics_vipe.json"):
            key = (row.get("seq", ""), row.get("traj", ""))
            if not all(key):
                continue
            cell = rows.setdefault(key, {})
            for metric in WIDE_METRICS:
                cell[f"{method}__{metric}"] = row.get(metric)

    cols = ["seq", "traj"] + [f"{m}__{metric}" for m in methods for metric in WIDE_METRICS]
    # Stable per-(seq, traj) ordering: alphabetical on seq, then on traj.
    sorted_keys = sorted(rows.keys())

    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for k in sorted_keys:
            seq, traj = k
            cell = rows[k]
            w.writerow([seq, traj] + [_fmt(cell.get(c)) for c in cols[2:]])

    return len(sorted_keys), methods


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_root).resolve()
    if not results_root.is_dir():
        print(f"[per-pair] ERROR: results_root not found: {results_root}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(args.out_dir) if args.out_dir else results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = out_dir / "per_pair_camera_metrics.tsv"
    n_long, methods_long = _emit_long(results_root, long_path)
    print(f"[per-pair] wrote {long_path}  ({n_long} rows across {len(methods_long)} methods)")

    if not args.long_only:
        extra = {m.strip() for m in args.extra_methods.split(",") if m.strip()}
        wide_path = out_dir / "per_pair_camera_metrics_wide.tsv"
        n_pairs, methods_wide = _emit_wide(results_root, wide_path, extra)
        print(f"[per-pair] wrote {wide_path}  ({n_pairs} (seq,traj) pairs × {len(methods_wide)} methods)")
        print(f"[per-pair] wide subset methods: {', '.join(methods_wide)}")


if __name__ == "__main__":
    main()
