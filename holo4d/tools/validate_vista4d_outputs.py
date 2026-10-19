"""Validate that every requested Vista4D pair produced a decodable video."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio


def requested_pairs(metadata: Path, videos: set[str] | None, max_cameras: int):
    by_video: dict[str, list[str]] = {}
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if videos is not None and row["video"] not in videos:
                continue
            cameras = by_video.setdefault(row["video"], [])
            if row["camera"] not in cameras and len(cameras) < max_cameras:
                cameras.append(row["camera"])
    if videos is not None:
        missing = videos - set(by_video)
        if missing:
            raise ValueError(f"requested videos absent from metadata: {sorted(missing)}")
    return [(video, camera) for video, cameras in sorted(by_video.items()) for camera in cameras]


def decodable_frames(path: Path) -> int:
    reader = imageio.get_reader(path)
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-names", default="")
    parser.add_argument("--max-cameras", type=int, default=2)
    parser.add_argument("--expected-frames", type=int, default=49)
    args = parser.parse_args()
    videos = {v.strip() for v in args.video_names.split(",") if v.strip()} or None
    pairs = requested_pairs(args.metadata, videos, args.max_cameras)
    failures = []
    for video, camera in pairs:
        path = args.output_dir / video / camera / f"{camera}.mp4"
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"{video}/{camera}: missing {path}")
            continue
        try:
            frames = decodable_frames(path)
        except Exception as exc:
            failures.append(f"{video}/{camera}: decode failed: {exc}")
            continue
        if frames != args.expected_frames:
            failures.append(f"{video}/{camera}: {frames} frames, expected {args.expected_frames}")
    if failures:
        raise SystemExit("Vista4D output validation failed:\n  " + "\n  ".join(failures))
    print(f"Validated {len(pairs)} Vista4D outputs ({args.expected_frames} frames each)")


if __name__ == "__main__":
    main()
