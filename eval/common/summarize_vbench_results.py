#!/usr/bin/env python3
"""Summarize VBench eval_results.json files into a TSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DIMENSIONS = (
    "aesthetic_quality",
    "imaging_quality",
    "subject_consistency",
    "background_consistency",
    "temporal_style",
)


def metric_value(payload: object) -> float | None:
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, list) and payload:
        head = payload[0]
        if isinstance(head, (int, float)):
            return float(head)
    if isinstance(payload, dict):
        for key in ("value", "score", "mean"):
            val = payload.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return None


def eval_jsons(vbench_dir: Path) -> list[Path]:
    return sorted(vbench_dir.glob("*_eval_results.json"))


def collect(eval_root: Path) -> list[dict[str, str]]:
    results_root = eval_root / "results"
    rows: list[dict[str, str]] = []
    for method_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        json_paths = eval_jsons(method_dir / "vbench")
        if not json_paths:
            continue
        row = {"method": method_dir.name}
        used_jsons: list[str] = []
        for dim in DIMENSIONS:
            val = None
            used_json = None
            for json_path in reversed(json_paths):
                data = json.loads(json_path.read_text())
                val = metric_value(data.get(dim))
                if val is not None:
                    used_json = json_path
                    break
            row[dim] = "" if val is None else f"{val:.6f}"
            if used_json is not None:
                used_jsons.append(str(used_json))
        row["json"] = ";".join(dict.fromkeys(used_jsons))
        rows.append(row)
    return rows


def write_tsv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["method", *DIMENSIONS, "json"]
    with out_path.open("w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(row.get(h, "") for h in headers) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min_rows", type=int, default=0)
    args = parser.parse_args()

    rows = collect(args.eval_root)
    if args.min_rows and len(rows) < args.min_rows:
        raise SystemExit(
            f"ERROR: expected at least {args.min_rows} VBench rows under "
            f"{args.eval_root}, found {len(rows)}"
        )
    out_path = args.out or args.eval_root / "results" / "summary_vbench.tsv"
    write_tsv(rows, out_path)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
