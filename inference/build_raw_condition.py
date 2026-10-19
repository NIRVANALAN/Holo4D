#!/usr/bin/env python3
"""Build a portable Holo4D condition bundle from staged frames and reconstruction caches.

This program intentionally does not run a reconstructor or sample the diffusion
model. ``preprocess_raw_video.py`` orchestrates those external tools and calls
this cache-to-condition conversion after their outputs have been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trajectory_scale_contract(backend: str) -> dict:
    return dict(backend=backend, angle_units='degrees',
                translation_units='backend reconstruction units before common-scale normalization',
                cross_backend_distance_comparable=False,
                cow_vggt_scale_align='none' if backend == 'cow_vggt' else None,
                note='Equal distance/orbit-radius numbers need not imply equal physical camera motion across backends; inspect the condition before sampling.')


def load_video_frames(frame_dir: Path, expected_count: int) -> np.ndarray:
    paths = sorted(frame_dir.glob("frame_*.png"))
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} staged PNG frames, found {len(paths)} in {frame_dir}")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        frames.append(frame)
    video = np.stack(frames, axis=0)
    if video.shape[0] != expected_count or video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"invalid staged video array: {video.shape}")
    return video


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--backend", choices=("4rc", "cow_vggt"), required=True)
    parser.add_argument("--geometry-root", type=Path, required=True,
                        help="4RC run root, or VGGT-Omega geometry root for cow_vggt.")
    parser.add_argument("--cow-root", type=Path, help="CoWTracker output directory for cow_vggt.")
    parser.add_argument("--gmos-root", type=Path)
    parser.add_argument("--gmos-mask-subdir", default="mos_frame")
    parser.add_argument("--fuse-gmos-static-tracks", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--trajectory", default="orbit_right")
    parser.add_argument("--angle-deg", type=float, default=30.0)
    parser.add_argument("--distance", type=float, default=0.12)
    parser.add_argument("--orbit-radius", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    example_dir = repo_root / "examples" / "wan2.1_fun"
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))

    from eval_track4d360_davis_4rc_novel_traj import (
        apply_davis_track_condition_variant,
        load_davis_sample_4rc,
    )
    from holo4d.examples_io import FORMAT, file_sha256, validate_condition

    frame_dir = args.frame_dir.resolve()
    frame_video = load_video_frames(frame_dir, expected_count=49)
    height, width = frame_video.shape[1:3]
    if (height, width) != (args.height, args.width):
        raise ValueError(f"staged frame canvas {(height, width)} does not match requested {(args.height, args.width)}")

    if args.backend == "4rc":
        if args.fuse_gmos_static_tracks and args.gmos_root is None:
            raise ValueError("--fuse-gmos-static-tracks requires --gmos-root")
        if args.cow_root is not None:
            raise ValueError("--cow-root is only valid with --backend cow_vggt")
    else:
        if args.cow_root is None:
            raise ValueError("--backend cow_vggt requires --cow-root")
        if args.fuse_gmos_static_tracks:
            raise ValueError("4RC-specific --fuse-gmos-static-tracks is invalid for cow_vggt")

    sample = load_davis_sample_4rc(
        davis_root=str(frame_dir.parent),
        davis_4rc_root=str(args.geometry_root.resolve()) if args.backend == "4rc" else "",
        sequence_name=args.sequence_name,
        start_frame=0,
        num_frames=49,
        width=args.width,
        height=args.height,
        trajectory_name=args.trajectory,
        angle_deg=args.angle_deg,
        distance=args.distance,
        orbit_radius=args.orbit_radius,
        gmos_root=str(args.gmos_root.resolve()) if args.gmos_root else "",
        gmos_mask_subdir=args.gmos_mask_subdir,
        fuse_gmos_static_tracks=args.fuse_gmos_static_tracks,
        prompt=args.caption,
        source_backend=args.backend,
        cow_root=str(args.cow_root.resolve()) if args.cow_root else None,
        vggt_root=str(args.geometry_root.resolve()) if args.backend == "cow_vggt" else None,
        cow_vggt_scale_align="none",
        seed=args.seed,
    )
    apply_davis_track_condition_variant(
        sample, track_sample_mode="all", track_stride=1, seed=args.seed
    )

    source_video = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in sample["video"]])
    valid = sample.get("tracks_world_valid_mask")
    if valid is None:
        valid = np.ones(np.asarray(sample["tracks_world"]).shape[:2], dtype=bool)
    condition = {
        "source_video": source_video,
        "tracks_world": np.asarray(sample["tracks_world"], dtype=np.float32),
        "tracks_world_valid_mask": np.asarray(valid, dtype=bool),
        "source_cam2world": np.asarray(sample["source_cam2world"], dtype=np.float32),
        "source_intrinsics": np.asarray(sample["source_intrinsics"], dtype=np.float32),
        "target_cam2world": np.asarray(sample["target_cam2world"], dtype=np.float32),
        "target_intrinsics": np.asarray(sample["target_intrinsics"], dtype=np.float32),
        "source_depth_video": np.asarray(sample["source_depth_video"], dtype=np.float32),
        "source_rgb_at_tracks": np.asarray(sample["source_rgb_at_tracks"], dtype=np.float32),
        "track_query_idx": np.asarray(sample.get("track_query_idx", 0), dtype=np.int32),
        "common_scale": np.asarray(sample.get("common_scale", 1.0), dtype=np.float32),
        "trajectory_name": np.asarray(args.trajectory),
        "source_backend": np.asarray(args.backend),
    }
    optional_keys = (
        "tracks_world_confidence", "source_dynamic_mask", "source_dynamic_mask_at_tracks",
        "target_dynamic_mask", "tracks_source_xy", "reproj_src_depth_at_target",
        "reproj_src_depth_valid", "target_depth", "target_depth_valid",
    )
    for key in optional_keys:
        value = sample.get(key)
        if value is not None:
            condition[key] = np.asarray(value)
    report = validate_condition(condition)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_path = output_dir / "condition.npz"
    if condition_path.exists():
        raise FileExistsError(f"refusing to overwrite existing condition: {condition_path}")
    atomic_npz(condition_path, **condition)
    entry = {
        "id": args.sequence_name,
        "caption": args.caption,
        "condition": condition_path.name,
        "condition_sha256": file_sha256(condition_path),
        "source_backend": args.backend,
        "trajectory": args.trajectory,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "track_identities": report["identities"],
        "gmos": bool(args.gmos_root),
        "gmos_static_track_fusion": bool(args.fuse_gmos_static_tracks),
        "trajectory_scale": trajectory_scale_contract(args.backend),
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps({"format": FORMAT, "examples": [entry]}, indent=2) + "\n")
    provenance = {
        "source_frame_dir": str(frame_dir),
        "frame_hashes_sha256": {path.name: sha256(path) for path in sorted(frame_dir.glob("frame_*.png"))},
        "backend": args.backend,
        "geometry_root": str(args.geometry_root.resolve()),
        "cow_root": str(args.cow_root.resolve()) if args.cow_root else None,
        "gmos_root": str(args.gmos_root.resolve()) if args.gmos_root else None,
        "gmos_mask_subdir": args.gmos_mask_subdir if args.gmos_root else None,
        "gmos_static_track_fusion": bool(args.fuse_gmos_static_tracks),
        "caption": args.caption,
        "trajectory": args.trajectory,
        "resolution_hw": [args.height, args.width],
        "fps": args.fps,
        "seed": args.seed,
        "condition_report": report,
        "trajectory_scale": trajectory_scale_contract(args.backend),
        "source_condition_npz": str(condition_path),
        "source_condition_sha256": file_sha256(condition_path),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"condition": str(condition_path), "manifest": str(manifest), **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
