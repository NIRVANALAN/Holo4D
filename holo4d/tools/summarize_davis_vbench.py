"""Validate full-cohort DAVIS classic VBench scores and preserve exact JSON paths."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

DIMENSIONS = ("aesthetic_quality", "imaging_quality", "subject_consistency",
              "background_consistency", "temporal_style")


def scorer_provenance(destination: Path) -> None:
    import vbench
    version = importlib.metadata.version("vbench")
    if version != "0.1.5":
        raise ValueError(f"Required VBench v0.1.5, found {version}")
    package = Path(vbench.__file__).parent
    files = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(package.rglob("*.py"))}
    destination.write_text(json.dumps({"version": version, "source_sha256": files,
                                      "package": str(package)}, indent=2) + "\n")


def summarize(result_dir: Path, expected: int) -> dict:
    metrics = {}
    for path in sorted(result_dir.glob("*_eval_results.json")):
        for dimension, value in json.loads(path.read_text()).items():
            if dimension not in DIMENSIONS:
                continue
            mean, samples = value
            if len(samples) != expected:
                raise ValueError(f"{path}/{dimension}: {len(samples)} scores, expected {expected}")
            names = [sample["video_path"] for sample in samples]
            if len(set(names)) != expected:
                raise ValueError(f"Duplicate video IDs: {path}/{dimension}")
            metrics[dimension] = {"mean": mean, "n": len(samples), "json": str(path.resolve())}
    if set(metrics) != set(DIMENSIONS):
        raise ValueError(f"Incomplete classic metrics in {result_dir}: {sorted(metrics)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected", type=int, default=100)
    args = parser.parse_args()
    if args.provenance:
        scorer_provenance(args.provenance)
    if args.result_dir:
        result = summarize(args.result_dir, args.expected)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
