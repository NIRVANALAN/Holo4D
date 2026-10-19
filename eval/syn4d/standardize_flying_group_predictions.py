#!/usr/bin/env python3
"""Resolve native baseline videos into the strict Syn4D prediction contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


PREDICTION_PATTERNS = {
    "recammaster": ("pred.mp4",),
    "gen3c": ("gen3c_target_first49.mp4", "gen3c_target.mp4", "pred.mp4"),
    "vista4d": ("video_seed=42.mp4", "pred.mp4"),
    "ex4d": ("output.mp4", "pred.mp4"),
    "trajcrafter": ("gen.mp4", "pred.mp4", "output.mp4"),
}


def find_prediction(root: Path, sequence: str, trajectory: str, method: str) -> Path:
    names = PREDICTION_PATTERNS[method]
    candidate_dirs = [
        root / sequence / trajectory,
        root / "predictions" / sequence / trajectory,
        root / "predictions" / sequence,
        root / f"{sequence}_{trajectory}",
        root / "results_384p" / f"{sequence}_{trajectory}",
    ]
    for directory in candidate_dirs:
        for name in names:
            path = directory / name
            if path.is_file() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(f"no {method} prediction for {sequence}/{trajectory} below {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(PREDICTION_PATTERNS))
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.pairs.open(newline="")))
    records = []
    for row in rows:
        sequence, trajectory = row["video"], row["trajectory"]
        try:
            native = find_prediction(args.native_root, sequence, trajectory, args.method)
        except FileNotFoundError:
            if args.allow_missing:
                continue
            raise
        output = args.output_root / sequence / trajectory / "pred.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or output.is_symlink():
            output.unlink()
        output.symlink_to(os.path.relpath(native.resolve(), output.parent))
        records.append(
            {"method": args.method, "video": sequence, "trajectory": trajectory, "native": str(native)}
        )
    (args.output_root / "standardization.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"[standardize] {args.method}: {len(records)}/{len(rows)} predictions")


if __name__ == "__main__":
    main()
