#!/usr/bin/env python3
"""Stage standardized flying_group predictions as flat MP4 links for reconstructors."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_METHODS = (
    "ex4d_gt_oracle,gen3c_gt_oracle,recammaster,trajcrafter_gt_oracle,"
    "vista4d_gt_oracle,ours_step19500_gt_tracks_ab12_v3,"
    "ours_step19500_4rc_q0_ab12_v1,"
    "ours_step19500_vggt_omega_cowtracker_q0_ab12_v1,"
    "ours_step19500_vdpm_q0_ab12_v1,"
    "ours_step19500_vggt_omega_opend4rt_q0_ab12_v1"
)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--limit_pairs", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    pair_rows = list(csv.DictReader((args.bundle_root / "test_pairs.csv").open()))
    if args.limit_pairs > 0:
        pair_rows = pair_rows[: args.limit_pairs]
    methods = split_csv(args.methods)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in methods:
        for pair_row in pair_rows:
            pair = pair_row["video"]
            source = args.bundle_root / "predictions" / method / pair / "target" / "pred.mp4"
            if not source.is_file() or source.stat().st_size == 0:
                if args.strict:
                    raise FileNotFoundError(source)
                print(f"[stage/skip] {source}")
                continue
            sequence_name = f"{method}__{pair}"
            target = args.output_root / f"{sequence_name}.mp4"
            if target.is_symlink() and target.resolve() != source.resolve():
                target.unlink()
            if not target.exists():
                target.symlink_to(source)
            rows.append({
                "method": method,
                "pair": pair,
                "sequence_name": sequence_name,
                "prediction": str(source),
                "staged_video": str(target),
            })

    expected = len(methods) * len(pair_rows)
    if args.strict and len(rows) != expected:
        raise RuntimeError(f"staged {len(rows)}/{expected} videos")
    manifest = {
        "bundle_root": str(args.bundle_root),
        "methods": methods,
        "num_pairs": len(pair_rows),
        "expected": expected,
        "staged": len(rows),
        "rows": rows,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[stage] {len(rows)}/{expected} videos -> {args.output_root}")


if __name__ == "__main__":
    main()
