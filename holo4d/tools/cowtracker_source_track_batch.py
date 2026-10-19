#!/usr/bin/env python3
"""Batch CoWTracker source-track export with one model load per worker."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def assign_shard(sequences: list[str], num_shards: int, shard_index: int) -> list[str]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return [seq for idx, seq in enumerate(sequences) if idx % num_shards == shard_index]


def output_paths_for_sequence(output_dir: Path, sequence: str) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"{sequence}_pred_trackvideo.mp4",
        output_dir / f"{sequence}_pred_trackvideo.npz",
        output_dir / "logs" / f"{sequence}.log",
    )


def should_skip_sequence(mp4_path: Path, npz_path: Path, skip_existing: bool) -> bool:
    return bool(skip_existing and mp4_path.is_file() and npz_path.is_file())


def list_sequences(input_root: Path, seqs: str) -> list[str]:
    if seqs == "all":
        return sorted(p.name for p in input_root.iterdir() if p.is_dir())
    return [s.strip() for s in seqs.split(",") if s.strip()]


def read_frame_dir_video(
    frame_dir: Path,
    *,
    max_frames: int,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    import mediapy

    frame_paths = sorted(
        p for p in frame_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not frame_paths:
        raise FileNotFoundError(f"no image frames found under {frame_dir}")
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]

    frames = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        for path in frame_paths
    ]
    video = np.stack(frames, axis=0)
    source_shape = np.asarray(video.shape, dtype=np.int32)
    video = np.asarray(
        mediapy.resize_video(video, (int(target_height), int(target_width))),
        dtype=np.uint8,
    )
    return video, source_shape


def restore_reverse_time_outputs(
    tracks: np.ndarray,
    visibilities: np.ndarray,
    confidences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map reverse-input CoWTracker outputs back to original chronological time.

    CoWTracker's dense query grid is always placed on frame 0 of the tensor it
    sees. Feeding frames in reverse therefore makes the original last frame the
    query frame. Downstream DAVIS loaders still expect arrays indexed by the
    original source-video time, so we flip only the temporal axis of the outputs
    before saving.
    """
    if hasattr(tracks, "flip") and hasattr(visibilities, "flip") and hasattr(confidences, "flip"):
        return (
            tracks.flip(0).contiguous(),
            visibilities.flip(0).contiguous(),
            confidences.flip(0).contiguous(),
        )
    return (
        np.asarray(tracks)[::-1].copy(),
        np.asarray(visibilities)[::-1].copy(),
        np.asarray(confidences)[::-1].copy(),
    )


def append_npz_metadata(path: Path, metadata: dict[str, object]) -> None:
    payload = {}
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            payload[key] = data[key]
    payload.update(metadata)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **payload)
    tmp.replace(path)


def save_raw_predictions_npz(
    path: Path,
    tracks,
    visibilities,
    confidences,
    coord_convention: str,
    source_shape: np.ndarray,
    inference_shape: np.ndarray,
    fps: float,
) -> None:
    """Save the stable raw CoWTracker contract independent of demo.py version.

    The public CoWTracker demo has changed its visualization API several times
    and some releases have no ``save_predictions_npz`` helper.  The source
    cache needs only dense 2-D tracks plus provenance, so own that small file
    contract here rather than coupling cache generation to a demo utility.
    """
    def as_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    np.savez_compressed(
        path,
        tracks_xy=as_numpy(tracks).astype(np.float32),
        visibility=as_numpy(visibilities).astype(bool),
        confidence=as_numpy(confidences).astype(np.float32),
        coord_convention=np.asarray(coord_convention),
        source_shape=np.asarray(source_shape, dtype=np.int32),
        inference_shape=np.asarray(inference_shape, dtype=np.int32),
        fps=np.asarray(float(fps), dtype=np.float32),
        format=np.asarray("track4d360_cowtracker_raw_v1"),
    )


def import_cowtracker_demo(cowtracker_repo: Path):
    repo = str(cowtracker_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return importlib.import_module("demo")


def load_model(demo_module, checkpoint: str | None):
    import torch

    use_cuda = torch.cuda.is_available()
    return demo_module.CoWTracker.from_checkpoint(
        checkpoint,
        device="cuda" if use_cuda else "cpu",
        dtype=demo_module.inf_dtype if use_cuda else torch.float32,
    )


def process_sequence(args: argparse.Namespace, demo_module, model, sequence: str) -> None:
    import mediapy

    input_dir = Path(args.input_root) / sequence
    output_dir = Path(args.output_dir)
    mp4_path, npz_path, log_path = output_paths_for_sequence(output_dir, sequence)

    if should_skip_sequence(mp4_path, npz_path, args.skip_existing):
        print(f"[cow-batch][gpu{args.gpu}] skip existing {sequence}", flush=True)
        return

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cow-batch][gpu{args.gpu}] running {sequence} -> {log_path}", flush=True)
    with log_path.open("w", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print("=" * 60)
            print(f"CoWTracker batch sequence: {sequence}")
            print(f"Input frames: {input_dir}")
            print(f"Output MP4: {mp4_path}")
            print(f"Output NPZ: {npz_path}")
            print(f"FPS: {args.fps}")

            video, source_shape = read_frame_dir_video(
                input_dir,
                max_frames=args.max_frames,
                target_height=args.target_height,
                target_width=args.target_width,
            )
            print(f"Video shape: {video.shape}, source shape: {tuple(source_shape)}")

            model_video = video[::-1].copy() if args.reverse_time else video
            if args.reverse_time:
                print(
                    "Reverse-time mode: CoWTracker query is original last frame; "
                    "saved arrays are restored to original chronological order."
                )
            tracks, visibilities, confidences = demo_module.run_inference(model, model_video)
            if args.reverse_time:
                tracks, visibilities, confidences = restore_reverse_time_outputs(
                    tracks,
                    visibilities,
                    confidences,
                )
            print(f"Tracks shape: {tuple(tracks.shape)}")

            if args.coord_convention == "corner_aligned":
                tracks = tracks + 0.5

            if args.foreground_only:
                raise RuntimeError(
                    "foreground_only requires a CoWTracker demo fork with foreground-mask support; "
                    "the frozen Syn4D backend cache intentionally uses dense tracks."
                )

            painted_video = demo_module.create_visualization(
                video,
                tracks,
                visibilities,
                rate=args.rate,
                fps=args.fps,
                show_bkg=not args.no_bkg,
            )
            mediapy.write_video(str(mp4_path), painted_video, fps=float(args.fps))
            save_raw_predictions_npz(
                npz_path,
                tracks,
                visibilities,
                confidences,
                args.coord_convention,
                source_shape,
                np.asarray(video.shape, dtype=np.int32),
                float(args.fps),
            )
            if args.reverse_time:
                query_idx = int(video.shape[0] - 1)
                append_npz_metadata(
                    npz_path,
                    {
                        "query_frame_idx": np.asarray(query_idx, dtype=np.int32),
                        "query_frame_idx_original": np.asarray(query_idx, dtype=np.int32),
                        "time_order": np.asarray("original_chronological"),
                        "cowtracker_input_time_order": np.asarray("reverse"),
                    },
                )
            print(f"Saved output to: {mp4_path}")
            print(f"Saved dense predictions to: {npz_path}")
            print("=" * 60)

    print(f"[cow-batch][gpu{args.gpu}] done {sequence} -> {mp4_path}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cowtracker_repo", required=True)
    parser.add_argument("--seqs", default="all")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--rate", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=49)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--target_height", type=int, default=336)
    parser.add_argument("--target_width", type=int, default=560)
    parser.add_argument(
        "--coord_convention",
        choices=["pixel_center", "corner_aligned"],
        default="pixel_center",
    )
    parser.add_argument("--foreground_only", action="store_true")
    parser.add_argument("--foreground_black_threshold", type=float, default=8.0)
    parser.add_argument("--no_bkg", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--reverse_time",
        action="store_true",
        help=(
            "Run CoWTracker on frames in reverse order so the original last "
            "frame is the dense query frame, then save tracks in original time."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    input_root = Path(args.input_root)
    cowtracker_repo = Path(args.cowtracker_repo)
    from holo4d.cowtracker_compat import configure_cowtracker_attention
    configure_cowtracker_attention(cowtracker_repo)
    sequences = assign_shard(
        list_sequences(input_root, args.seqs),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    print(
        f"[cow-batch][gpu{args.gpu}] shard={args.shard_index}/{args.num_shards} "
        f"sequences={len(sequences)} input={input_root} output={args.output_dir}",
        flush=True,
    )
    if not sequences:
        return

    demo_module = import_cowtracker_demo(cowtracker_repo)
    checkpoint = args.checkpoint or None
    print(f"[cow-batch][gpu{args.gpu}] loading model once", flush=True)
    model = load_model(demo_module, checkpoint)
    print(f"[cow-batch][gpu{args.gpu}] model ready", flush=True)

    for sequence in sequences:
        process_sequence(args, demo_module, model, sequence)


if __name__ == "__main__":
    main()
