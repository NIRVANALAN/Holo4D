"""Compare metric summaries from an independent code replay, not video quality.

Paths and metadata may differ; explicitly selected numeric sections must agree.
This does not establish fresh reconstructor inference or diffusion equivalence.
"""
import argparse
import json
import math
from pathlib import Path


def compare(expected, actual, path="", tolerance=1e-10):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"{path}: different metric fields")
        for key, value in expected.items():
            compare(value, actual[key], f"{path}/{key}", tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isfinite(expected) or not math.isfinite(actual):
            raise ValueError(f"{path}: non-finite/non-numeric metric")
        if isinstance(expected, int) and isinstance(actual, int) and expected != actual:
            raise ValueError(f"{path}: different counts")
        if not math.isclose(expected, actual, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"{path}: {expected!r} != {actual!r}")
    elif expected != actual:
        raise ValueError(f"{path}: {expected!r} != {actual!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--section", action="append", required=True)
    args = parser.parse_args()
    expected, actual = json.loads(args.expected.read_text()), json.loads(args.actual.read_text())
    for section in args.section:
        compare(expected[section], actual[section], section)
    if "per_pair" in expected and "per_pair" in actual:
        old_ids = [(row.get("video", row.get("seq")), row.get("trajectory", row.get("traj"))) for row in expected["per_pair"]]
        new_ids = [(row.get("video", row.get("seq")), row.get("trajectory", row.get("traj"))) for row in actual["per_pair"]]
        if any(video is None or trajectory is None for video, trajectory in old_ids + new_ids):
            raise ValueError("Per-pair IDs absent")
        if old_ids != new_ids:
            raise ValueError("Per-pair IDs/order differ")
    print(json.dumps(dict(status="passed", sections=args.section,
                          tolerance=1e-10, expected=str(args.expected), actual=str(args.actual))))


if __name__ == "__main__":
    main()
