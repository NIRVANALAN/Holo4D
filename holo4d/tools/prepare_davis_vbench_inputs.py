"""Make matched, 49-frame DAVIS VBench inputs from an existing prediction tree.

Both methods are encoded at the same FPS and quality. No diffusion sampling or
target frames are used; the output manifest preserves each source prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def read_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as handle:
        rows = [(row["video"].strip(), row["trajectory"].strip()) for row in csv.DictReader(handle)]
    if len(rows) != 100 or len(set(rows)) != 100:
        raise ValueError(f"Expected exactly 100 unique DAVIS pairs, got {len(rows)} rows")
    return rows


def read_captions(path: Path) -> dict[tuple[str, str], str]:
    with path.open(newline="") as handle:
        captions = {(row["video"].strip(), row["trajectory"].strip()):
                    row["caption"].strip() for row in csv.DictReader(handle)}
    if len(captions) != 100 or any(not value for value in captions.values()):
        raise ValueError(f"Expected 100 nonempty pair captions in {path}")
    return captions


def write_video(source: Path, destination: Path, frames: int, fps: int,
                width: int = 672, height: int = 384) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    count = 0
    reader = imageio.get_reader(source)
    try:
        with imageio.get_writer(temporary, fps=fps, codec="libx264", quality=9,
                                macro_block_size=None) as writer:
            for frame in reader:
                if count >= frames:
                    break
                # Explicit canvas stretch, not a crop: matches DAVIS camera scoring,
                # including square EX-4D and native-resolution 5B predictions.
                if frame.shape[:2] != (height, width):
                    frame = np.asarray(Image.fromarray(frame).resize(
                        (width, height), Image.Resampling.BILINEAR))
                writer.append_data(frame)
                count += 1
    finally:
        reader.close()
    if count != frames:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{source}: expected {frames} frames, decoded {count}")
    temporary.replace(destination)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--captions-csv", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--prediction-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; 0 means all 100")
    args = parser.parse_args()

    pairs = read_pairs(args.pairs_csv)
    captions = read_captions(args.captions_csv)
    if set(captions) != set(pairs):
        raise ValueError("Caption pair IDs do not match the frozen DAVIS pairs")
    selected = pairs[:args.limit] if args.limit else pairs
    # Check the entire selected cohort before writing any videos.
    inputs = [args.prediction_root / seq / traj /
              args.prediction_name.format(video=seq, trajectory=traj)
              for seq, traj in selected]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction(s): {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    prompt_map = {}
    for index, ((seq, traj), source) in enumerate(zip(selected, inputs)):
        destination = args.output_dir / f"video{index}.mp4"
        if not destination.is_file() or destination.stat().st_size == 0:
            write_video(source, destination, args.frames, args.fps, args.width, args.height)
        manifest.append({"index": index, "video": seq, "trajectory": traj,
                         "source": str(source.resolve()), "output": str(destination),
                         "frames": args.frames, "fps": args.fps,
                         "width": args.width, "height": args.height,
                         "codec": "libx264", "quality": 9, "resize": "bilinear_stretch",
                         "caption": captions[(seq, traj)]})
        prompt_map[destination.name] = captions[(seq, traj)]
        if (index + 1) % 10 == 0 or index + 1 == len(selected):
            print(f"[vbench-input] {index + 1}/{len(selected)}", flush=True)

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "prompts.json").write_text(json.dumps(prompt_map, indent=2) + "\n")
    print(f"[vbench-input] complete: {len(manifest)} videos -> {args.output_dir}")


if __name__ == "__main__":
    main()
