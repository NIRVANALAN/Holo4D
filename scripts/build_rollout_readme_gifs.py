"""Export small, animated GitHub previews without changing the original MP4s.

Run from the repository root: python scripts/build_rollout_readme_gifs.py
Requires FFmpeg; preserves the full duration and aspect ratio of each video.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("tri_static_high", "figure0_orbit")


def export_gif(source: Path, destination: Path, width: int, fps: int) -> None:
    graph = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-threads", "1", "-filter_complex_threads", "1",
         "-i", str(source), "-filter_complex", graph, "-loop", "0", str(destination)],
        check=True,
    )
    print(f"{destination}: {destination.stat().st_size:,} bytes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "assets/demo/robot_long")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/demo/robot_long")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    if args.width <= 0 or args.fps <= 0:
        parser.error("width and fps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for example in EXAMPLES:
        export_gif(args.input_dir / f"{example}.mp4", args.output_dir / f"{example}.gif",
                   args.width, args.fps)


if __name__ == "__main__":
    main()
