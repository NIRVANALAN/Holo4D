#!/usr/bin/env python3
"""Build a flat-mode prediction tree + metadata.csv from a baseline's
DAVIS-50 × 2-trajectory output, matching the contract `eval_video_quality.py
--mode flat` expects:

  <flat_root>/video<i>.mp4
  <flat_root>/dataset/videos/<seq>_<traj>.mp4   # source video for clip_v / fvd content-preservation proxy
  <flat_root>/dataset/metadata.csv               # columns: file_name,text  (caption from davis_trajectories_full.csv)

Usage:
    python build_flat_tree.py \\
        --method_pred_root <method>/predictions/ \\
        --flat_root <out>/_flat/<method> \\
        [--source_root ${BASELINES_ROOT}/davis_video] \\
        [--pairs_csv ${BASELINES_ROOT}/davis_trajectories_full.csv] \\
        [--num_frames 49]

Each row in `davis_trajectories_full.csv` becomes one (video<i>, source video, caption)
in the same order. Predictions are trimmed to leading `--num_frames` frames; the
canonical eval window is 49 (see baseline README's "Evaluation contract" section).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import string
import subprocess
from pathlib import Path


def mp4_from_frames_dir(frames_dir: Path, out_mp4: Path, num_frames: int, fps: int = 12) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp4 = out_mp4.with_name(f".{out_mp4.stem}.tmp{out_mp4.suffix}")
    if tmp_mp4.exists():
        tmp_mp4.unlink()
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(frames_dir / "*.jpg"),
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(tmp_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_mp4.replace(out_mp4)


def trim_mp4(in_mp4: Path, out_mp4: Path, num_frames: int) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp4 = out_mp4.with_name(f".{out_mp4.stem}.tmp{out_mp4.suffix}")
    if tmp_mp4.exists():
        tmp_mp4.unlink()
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_mp4),
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(tmp_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_mp4.replace(out_mp4)


def validate_pred_templates(raw_templates: list[str]) -> list[str]:
    valid: list[str] = []
    for tpl in raw_templates:
        try:
            fields = [field_name for _, field_name, _, _ in string.Formatter().parse(tpl) if field_name is not None]
        except ValueError as exc:
            print(f"[build-flat-tree] WARN: skip invalid pred filename template {tpl!r}: {exc}")
            continue
        bad_fields = [field_name for field_name in fields if field_name != "traj"]
        if bad_fields:
            print(f"[build-flat-tree] WARN: skip pred filename template {tpl!r}: unsupported fields {bad_fields}")
            continue
        valid.append(tpl)
    return valid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method_pred_root", required=True,
                   help="Directory containing <seq>/<traj>/{pred.mp4 | <traj>.mp4 | cam_type01/video0.mp4 | render.mp4 | ...}")
    p.add_argument("--flat_root", required=True,
                   help="Output flat tree dir. Creates <flat_root>/video<i>.mp4 + <flat_root>/dataset/{videos,metadata.csv}")
    p.add_argument("--source_root", default=os.environ.get("DAVIS_ROOT"),
                   help="DAVIS frames root (<source_root>/<seq>/00000.jpg ...).")
    p.add_argument("--pairs_csv", default=os.environ.get("PAIRS_FULL_CSV"),
                   help="Enriched 100-pair CSV with columns: video,trajectory,camera_npz,caption.")
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--source_fps", type=int, default=12)
    p.add_argument("--pred_filenames", default="pred.mp4,{traj}.mp4,cam_type01/video0.mp4,render.mp4,color_{traj}.mp4",
                   help="Comma-separated candidate paths under <seq>/<traj>/, the first existing one is used. "
                        "{traj} is substituted with the trajectory name.")
    p.add_argument("--limit_pairs", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=int(os.environ.get("FLAT_PROGRESS_EVERY", "10")),
                   help="Print progress every N successful predictions; 0 disables progress prints.")
    args = p.parse_args()

    for name in ("source_root", "pairs_csv"):
        if not getattr(args, name):
            p.error(f"--{name} or its matching environment variable is required")

    flat_root = Path(args.flat_root)
    flat_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = flat_root / "dataset"
    videos_dir = dataset_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    method_root = Path(args.method_pred_root)

    with open(args.pairs_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit_pairs > 0:
        rows = rows[: args.limit_pairs]

    metadata_rows: list[dict[str, str]] = []
    skipped: list[tuple[str, str, str]] = []
    raw_pred_candidate_templates = [s.strip() for s in args.pred_filenames.split(",") if s.strip()]
    pred_candidate_templates = validate_pred_templates(raw_pred_candidate_templates)
    if not pred_candidate_templates:
        raise SystemExit(f"no valid prediction filename templates in --pred_filenames={args.pred_filenames!r}")

    # Clean any stale flat-tree MP4s from a prior build of this method tag.
    # Otherwise an interrupted or no-skip run can leave corrupt source videos or
    # files at indices that collide with the new sequential numbering and trip
    # eval_video_quality.py.
    for stale in flat_root.glob("video*.mp4"):
        stale.unlink()
    for stale in videos_dir.glob("*.mp4"):
        stale.unlink()

    out_idx = 0  # sequential output index — increments only on success so
                 # video{out_idx}.mp4 stays aligned with metadata row out_idx
                 # even when some pairs are skipped (missing pred).
    for row in rows:
        seq, traj, caption = row["video"].strip(), row["trajectory"].strip(), row.get("caption", "").strip()
        # Resolve prediction path.
        candidates = [tpl.format(traj=traj) for tpl in pred_candidate_templates]
        pred_path = None
        for cand in candidates:
            cand_path = method_root / seq / traj / cand
            if cand_path.is_file():
                pred_path = cand_path
                break
            # also try one level up: method_pred_root WITHOUT predictions/ wrapper
            alt = method_root.parent / "predictions" / seq / traj / cand
            if alt.is_file():
                pred_path = alt
                break
        if pred_path is None:
            skipped.append((seq, traj, "missing pred"))
            continue

        # Build trimmed pred at <flat_root>/video<out_idx>.mp4
        out_pred = flat_root / f"video{out_idx}.mp4"
        trim_mp4(pred_path, out_pred, args.num_frames)

        # Build source MP4 at <flat_root>/dataset/videos/<seq>_<traj>.mp4 (re-encoded from frames)
        # The source is shared across both trajectories of the same video; cache it.
        src_name = f"{seq}_{traj}.mp4"
        src_out = videos_dir / src_name
        cache = videos_dir / f"_{seq}_first{args.num_frames}.mp4"
        if not cache.is_file():
            mp4_from_frames_dir(Path(args.source_root) / seq, cache, args.num_frames, args.source_fps)
        if not src_out.is_file():
            shutil.copy(cache, src_out)

        metadata_rows.append({"file_name": src_name, "text": caption})
        out_idx += 1
        if args.progress_every > 0 and (out_idx == 1 or out_idx % args.progress_every == 0):
            print(f"[build-flat-tree] {out_idx}/{len(rows)} {seq}/{traj} <- {pred_path.name}", flush=True)

    # Write metadata.csv
    meta_csv = dataset_dir / "metadata.csv"
    with open(meta_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_name", "text"])
        writer.writeheader()
        for r in metadata_rows:
            writer.writerow(r)

    print(f"flat_root: {flat_root}")
    print(f"  predictions: {len(metadata_rows)} (video0..video{len(metadata_rows)-1}.mp4)")
    print(f"  metadata.csv: {meta_csv}")
    if skipped:
        print(f"  skipped {len(skipped)} pairs:")
        for s in skipped[:10]:
            print("   ", s)


if __name__ == "__main__":
    main()
