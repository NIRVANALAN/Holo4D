"""Build de-duplicated long videos from two-window I2V continuation cases.

The held-out Syn4D protocol stores two 49-frame windows per case.  The seed
window ends at the same absolute frame at which the continuation starts.  The
seed prediction's last target frame is the continuation I2V condition.  A
continuous visualization must therefore concatenate ``seed[0:49]`` with
``continuation[1:49]`` rather than retaining the shared boundary twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as imageio
import numpy as np

from holo4d.visualization.artifact_io import PREDICTION_MP4_QUALITY, WAN_SAMPLE_FPS


REQUIRED_STREAMS = ("source_rgb.mp4", "gt_rgb.mp4", "pred_rgb.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory containing flying_group/case_* or case_* directly.",
    )
    parser.add_argument(
        "--case-glob",
        default="case_*",
        help="Case-directory glob evaluated below flying_group (default: case_*).",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        action="append",
        default=[],
        help="Explicit case directory; repeat for a subset. Overrides --case-glob.",
    )
    parser.add_argument("--output-subdir", default="long_rollout")
    parser.add_argument("--fps", type=int, default=WAN_SAMPLE_FPS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_video(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = imageio.get_reader(path, output_params=["-threads", "1"])
    try:
        frames = [np.asarray(frame, dtype=np.uint8)[..., :3] for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError(f"Frame shapes change within {path}")
    return frames


def concatenate_at_shared_boundary(
    seed_frames: list[np.ndarray], continuation_frames: list[np.ndarray]
) -> list[np.ndarray]:
    if len(seed_frames) < 2 or len(continuation_frames) < 2:
        raise ValueError("Each I2V window must contain at least two frames")
    if seed_frames[0].shape != continuation_frames[0].shape:
        raise ValueError(
            "Seed/continuation resolution mismatch: "
            f"{seed_frames[0].shape} vs {continuation_frames[0].shape}"
        )
    return seed_frames + continuation_frames[1:]


def write_video(path: Path, frames: Iterable[np.ndarray], fps: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=PREDICTION_MP4_QUALITY,
        macro_block_size=1,
        pixelformat="yuv420p",
        ffmpeg_params=[
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-movflags",
            "+faststart",
        ],
    )
    count = 0
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
            count += 1
    finally:
        writer.close()
    return count


def video_info(path: Path) -> tuple[int, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = imageio.get_reader(path, output_params=["-threads", "1"])
    try:
        count = int(reader.count_frames())
        first = np.asarray(reader.get_data(0))
    finally:
        reader.close()
    if count <= 0:
        raise ValueError(f"No frames decoded from {path}")
    return count, (int(first.shape[0]), int(first.shape[1]))


def iter_chained_video(
    seed_path: Path,
    continuation_path: Path,
    *,
    counts: dict[str, object] | None = None,
):
    phase_counts: dict[str, int] = {}
    frame_shape: tuple[int, int] | None = None
    for path, skip_first in ((seed_path, False), (continuation_path, True)):
        phase = "continuation" if skip_first else "seed"
        decoded = 0
        reader = imageio.get_reader(path, output_params=["-threads", "1"])
        try:
            for index, frame in enumerate(reader):
                frame = np.asarray(frame, dtype=np.uint8)[..., :3]
                current_shape = (int(frame.shape[0]), int(frame.shape[1]))
                if frame_shape is None:
                    frame_shape = current_shape
                elif current_shape != frame_shape:
                    raise ValueError(
                        f"Frame resolution changed across chained videos: "
                        f"{current_shape} vs {frame_shape}"
                    )
                decoded += 1
                if skip_first and index == 0:
                    continue
                yield frame
        finally:
            reader.close()
        if decoded == 0:
            raise ValueError(f"No frames decoded from {path}")
        phase_counts[phase] = decoded
    if counts is not None:
        counts.update(phase_counts)
        counts["shape"] = frame_shape


def labelled_panel(frame: np.ndarray, label: str) -> np.ndarray:
    panel = frame.copy()
    height, width = panel.shape[:2]
    bar_height = max(30, int(round(height * 0.085)))
    cv2.rectangle(panel, (0, 0), (width, bar_height), (0, 0, 0), -1)
    scale = max(0.48, min(0.75, width / 1000.0))
    cv2.putText(
        panel,
        label,
        (10, int(round(bar_height * 0.70))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def crop_existing_comparison_frame(
    frame: np.ndarray, panel_height: int, panel_width: int
) -> dict[str, np.ndarray]:
    if frame.shape[0] < 2 * panel_height or frame.shape[1] < 5 * panel_width:
        raise ValueError(
            "Expected existing comparison.mp4 to contain a 2x5 grid at "
            f"panel size {panel_height}x{panel_width}; got {frame.shape}"
        )
    return {
        "source": frame[0:panel_height, 0:panel_width, :3].copy(),
        "target_gt": frame[
            0:panel_height, panel_width : 2 * panel_width, :3
        ].copy(),
        "generated": frame[
            0:panel_height, 2 * panel_width : 3 * panel_width, :3
        ].copy(),
        "dense_warped_rgb": frame[
            panel_height : 2 * panel_height, 0:panel_width, :3
        ].copy(),
        "dense_warped_depth": frame[
            panel_height : 2 * panel_height,
            panel_width : 2 * panel_width,
            :3,
        ].copy(),
        "tracks_on_generated": frame[
            panel_height : 2 * panel_height,
            4 * panel_width : 5 * panel_width,
            :3,
        ].copy(),
    }


def detailed_comparison_frame(
    source: np.ndarray,
    generated: np.ndarray,
    target_gt: np.ndarray,
    dense_warped_rgb: np.ndarray,
    dense_warped_depth: np.ndarray,
    tracks_on_prediction: np.ndarray,
    *,
    long_index: int,
    long_length: int,
    absolute_frame: int,
    boundary_frame: int,
    src_cam_idx: int,
    tgt_cam_idx: int,
    fps: int = WAN_SAMPLE_FPS,
    boundary_frames: Iterable[int] | None = None,
) -> np.ndarray:
    boundaries = tuple(sorted({int(boundary_frame), *(boundary_frames or ())}))
    first_boundary = boundaries[0]
    if absolute_frame < first_boundary:
        phase = "SEED (source-only T2V; GT target is reference only)"
        phase_color = (50, 160, 255)
    elif absolute_frame in boundaries:
        phase = "SHARED BOUNDARY: generated target is next I2V condition"
        phase_color = (0, 215, 255)
    else:
        phase = "AUTOREGRESSIVE CONTINUATION (generated boundary condition)"
        phase_color = (80, 210, 80)

    top_panels = [
        labelled_panel(source, f"SOURCE cam{src_cam_idx} (model input)"),
        labelled_panel(generated, f"GENERATED TARGET cam{tgt_cam_idx}"),
        labelled_panel(target_gt, f"TARGET cam{tgt_cam_idx} GT (reference only)"),
    ]
    bottom_panels = [
        labelled_panel(dense_warped_rgb, "SOURCE RGB -> TARGET (dense warp)"),
        labelled_panel(dense_warped_depth, "SOURCE DEPTH -> TARGET (z-buffer)"),
        labelled_panel(tracks_on_prediction, "TRACKS ON GENERATED TARGET"),
    ]
    canvas = np.concatenate(
        [np.concatenate(top_panels, axis=1), np.concatenate(bottom_panels, axis=1)],
        axis=0,
    )
    height, width = canvas.shape[:2]
    footer_height = max(32, int(round(source.shape[0] * 0.09)))
    cv2.rectangle(
        canvas,
        (0, height - footer_height),
        (width, height),
        (0, 0, 0),
        -1,
    )
    text = f"absolute frame {absolute_frame} | {phase}"
    cv2.putText(
        canvas,
        text,
        (12, height - max(9, footer_height // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        phase_color,
        2,
        cv2.LINE_AA,
    )
    frame_text = (
        f"frame {long_index + 1:03d}/{long_length:03d} | "
        f"abs {absolute_frame:04d} | {int(fps)} FPS"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.66
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(
        frame_text, font, scale, thickness
    )
    x0 = width - text_width - 18
    y0 = text_height + 16
    cv2.rectangle(
        canvas,
        (x0 - 8, 0),
        (width, y0 + 9),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        canvas,
        frame_text,
        (x0, y0),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def iter_detailed_comparison(
    seed_path: Path,
    continuation_path: Path,
    *,
    panel_height: int,
    panel_width: int,
    long_length: int,
    seed_start_frame: int,
    boundary_frame: int,
    src_cam_idx: int,
    tgt_cam_idx: int,
    fps: int = WAN_SAMPLE_FPS,
    counts: dict[str, object] | None = None,
):
    for long_index, frame in enumerate(
        iter_chained_video(seed_path, continuation_path, counts=counts)
    ):
        panels = crop_existing_comparison_frame(frame, panel_height, panel_width)
        absolute_frame = int(seed_start_frame) + long_index
        yield detailed_comparison_frame(
            panels["source"],
            panels["generated"],
            panels["target_gt"],
            panels["dense_warped_rgb"],
            panels["dense_warped_depth"],
            panels["tracks_on_generated"],
            long_index=long_index,
            long_length=long_length,
            absolute_frame=absolute_frame,
            boundary_frame=boundary_frame,
            src_cam_idx=src_cam_idx,
            tgt_cam_idx=tgt_cam_idx,
            fps=fps,
        )



def iter_multi_chained_video(
    chunk_paths: list[Path],
    *,
    valid_frames: list[int] | None = None,
    append_from: list[int] | None = None,
    counts: dict[str, object] | None = None,
):
    """Yield each planned chunk range, removing every shared boundary."""
    if not chunk_paths:
        raise ValueError("at least one chunk is required")
    if valid_frames is None:
        valid_frames = [video_info(path)[0] for path in chunk_paths]
    if append_from is None:
        append_from = [0] + [1] * (len(chunk_paths) - 1)
    if not (len(chunk_paths) == len(valid_frames) == len(append_from)):
        raise ValueError("chunk paths, valid-frame counts, and append offsets must align")
    frame_shape: tuple[int, int] | None = None
    phase_counts: dict[str, int] = {}
    for chunk_index, path in enumerate(chunk_paths):
        decoded = 0
        limit = int(valid_frames[chunk_index])
        first = int(append_from[chunk_index])
        if limit <= first:
            raise ValueError(
                f"chunk {chunk_index} has invalid planned range [{first}, {limit})"
            )
        reader = imageio.get_reader(path, output_params=["-threads", "1"])
        try:
            for frame_index, frame in enumerate(reader):
                if frame_index >= limit:
                    break
                frame = np.asarray(frame, dtype=np.uint8)[..., :3]
                shape = (int(frame.shape[0]), int(frame.shape[1]))
                if frame_shape is None:
                    frame_shape = shape
                elif shape != frame_shape:
                    raise ValueError(f"Frame resolution changed across chunks: {shape} vs {frame_shape}")
                decoded += 1
                if frame_index < first:
                    continue
                yield frame
        finally:
            reader.close()
        if decoded < limit:
            raise ValueError(f"{path} decoded {decoded} frames but the plan requires {limit}")
        phase_counts[f"chunk_{chunk_index}"] = decoded
    if counts is not None:
        counts.update(phase_counts)
        counts["shape"] = frame_shape


def iter_multi_detailed_comparison(
    chunk_paths: list[Path],
    *,
    panel_height: int,
    panel_width: int,
    long_length: int,
    seed_start_frame: int,
    boundary_frames: list[int],
    src_cam_idx: int,
    tgt_cam_idx: int,
    fps: int,
    valid_frames: list[int],
    append_from: list[int],
    counts: dict[str, object] | None = None,
):
    for long_index, frame in enumerate(
        iter_multi_chained_video(
            chunk_paths,
            valid_frames=valid_frames,
            append_from=append_from,
            counts=counts,
        )
    ):
        panels = crop_existing_comparison_frame(frame, panel_height, panel_width)
        absolute_frame = int(seed_start_frame) + long_index
        yield detailed_comparison_frame(
            panels["source"],
            panels["generated"],
            panels["target_gt"],
            panels["dense_warped_rgb"],
            panels["dense_warped_depth"],
            panels["tracks_on_generated"],
            long_index=long_index,
            long_length=long_length,
            absolute_frame=absolute_frame,
            boundary_frame=int(boundary_frames[0]),
            boundary_frames=boundary_frames,
            src_cam_idx=src_cam_idx,
            tgt_cam_idx=tgt_cam_idx,
            fps=fps,
        )


def resolve_chunk_dirs(case_dir: Path) -> list[Path]:
    """Resolve seed, continuation, continuation_2, ... in temporal order."""
    dirs = [case_dir / "seed"]
    index = 1
    while True:
        candidate = case_dir / ("continuation" if index == 1 else f"continuation_{index}")
        if not candidate.is_dir():
            break
        dirs.append(candidate)
        index += 1
    return dirs if all(path.is_dir() for path in dirs) else []


def build_multi_case(case_dir: Path, output_subdir: str, fps: int, overwrite: bool) -> dict:
    """Concatenate an arbitrary number of generated I2V chunks."""
    case = json.loads((case_dir / "case.json").read_text())
    chunk_dirs = resolve_chunk_dirs(case_dir)
    if len(chunk_dirs) < 3:
        raise ValueError("build_multi_case requires at least three chunks")
    for chunk_dir in chunk_dirs:
        for filename in REQUIRED_STREAMS + ("comparison.mp4",):
            if not (chunk_dir / filename).is_file():
                raise FileNotFoundError(chunk_dir / filename)
    prediction_paths = [path / "pred_rgb.mp4" for path in chunk_dirs]
    comparison_paths = [path / "comparison.mp4" for path in chunk_dirs]
    prediction_counts = [video_info(path)[0] for path in prediction_paths]
    comparison_counts = [video_info(path)[0] for path in comparison_paths]
    if prediction_counts != comparison_counts:
        raise ValueError(f"prediction/comparison lengths differ: {prediction_counts} vs {comparison_counts}")
    chunk_manifests = [json.loads((path / "chunk_manifest.json").read_text()) for path in chunk_dirs]
    valid_frames = [int(item["valid_frames"]) for item in chunk_manifests]
    append_from = [int(item["append_from"]) for item in chunk_manifests]
    for index, (decoded, valid) in enumerate(zip(prediction_counts, valid_frames, strict=True)):
        if decoded < valid:
            raise ValueError(f"chunk {index} decoded {decoded} frames but plan requires {valid}")
    long_length = sum(valid - first for valid, first in zip(valid_frames, append_from, strict=True))
    output_dir = case_dir / output_subdir
    generated_path = output_dir / f"target_generated_{long_length}f.mp4"
    comparison_path = output_dir / f"rgb_depth_tracks_2x3_{long_length}f.mp4"
    manifest_path = output_dir / "long_rollout.json"
    if not overwrite and all(path.is_file() for path in (generated_path, comparison_path, manifest_path)):
        try:
            manifest = json.loads(manifest_path.read_text())
            if int(manifest["long_frames"]) == long_length and int(manifest["fps"]) == int(fps):
                return {"case": case_dir.name, "status": "skipped", "output": str(output_dir)}
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_count = write_video(
        generated_path,
        iter_multi_chained_video(
            prediction_paths, valid_frames=valid_frames, append_from=append_from
        ),
        fps,
    )
    first_comparison = read_video(comparison_paths[0])[0]
    if first_comparison.shape[0] < 2 or first_comparison.shape[1] < 5:
        raise ValueError(f"invalid comparison frame shape {first_comparison.shape}")
    panel_height = first_comparison.shape[0] // 2
    panel_width = first_comparison.shape[1] // 5
    starts = []
    for index, chunk_dir in enumerate(chunk_dirs):
        marker = chunk_dir / "chunk_manifest.json"
        if marker.is_file():
            try:
                starts.append(int(json.loads(marker.read_text())["absolute_start_frame"]))
                continue
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        starts.append(int(case["seed_start_frame"]) + sum(prediction_counts[:index]) - index)
    boundaries = starts[1:]
    comparison_count = write_video(
        comparison_path,
        iter_multi_detailed_comparison(
            comparison_paths,
            panel_height=panel_height,
            panel_width=panel_width,
            long_length=long_length,
            seed_start_frame=int(case["seed_start_frame"]),
            boundary_frames=boundaries,
            src_cam_idx=int(case["src_cam_idx"]),
            tgt_cam_idx=int(case["tgt_cam_idx"]),
            fps=fps,
            valid_frames=valid_frames,
            append_from=append_from,
        ),
        fps,
    )
    if generated_count != long_length or comparison_count != long_length:
        raise RuntimeError(f"wrote {generated_count}/{comparison_count} frames, expected {long_length}")
    manifest = {
        "protocol": "chunk_0[all] + chunk_i[1:] (all shared boundaries de-duplicated)",
        "case": case_dir.name,
        "source_camera": int(case["src_cam_idx"]),
        "target_camera": int(case["tgt_cam_idx"]),
        "chunk_directories": [path.name for path in chunk_dirs],
        "chunk_window_frames": prediction_counts,
        "chunk_valid_frames": valid_frames,
        "chunk_append_from": append_from,
        "boundary_absolute_frames": boundaries,
        "long_absolute_frames": [int(case["seed_start_frame"]), int(case["seed_start_frame"]) + long_length - 1],
        "long_frames": long_length,
        "fps": int(fps),
        "generated_video": generated_path.name,
        "comparison_video": comparison_path.name,
        "comparison_overlay": "top-right label includes frame, absolute frame, and FPS; generated_video is unmodified",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"case": case_dir.name, "status": "written", "output": str(output_dir)}

def resolve_cases(args: argparse.Namespace) -> list[Path]:
    if args.case_dir:
        cases = [path.resolve() for path in args.case_dir]
    else:
        cases_root = args.run_root / "flying_group"
        if not cases_root.is_dir():
            cases_root = args.run_root
        cases = sorted(path.resolve() for path in cases_root.glob(args.case_glob))
    cases = [path for path in cases if path.is_dir()]
    if not cases:
        raise FileNotFoundError("No I2V continuation case directories selected")
    return cases


def validate_case_contract(case_dir: Path, case: dict) -> dict:
    seed_start = int(case["seed_start_frame"])
    continuation_start = int(case["continuation_start_frame"])
    condition_path = case_dir / "continuation" / "previous_generated_target_frame48.png"
    seed_condition_path = case_dir / "seed" / "last_target.png"
    previous_gt_path = case_dir / "continuation" / "previous_target_gt_frame48.png"
    current_gt_path = case_dir / "continuation" / "current_target_gt_frame0.png"
    for path in (
        condition_path,
        seed_condition_path,
        previous_gt_path,
        current_gt_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    condition_hash = sha256(condition_path)
    seed_condition_hash = sha256(seed_condition_path)
    previous_gt_hash = sha256(previous_gt_path)
    current_gt_hash = sha256(current_gt_path)
    if condition_hash != seed_condition_hash:
        raise ValueError(
            f"{case_dir}: continuation condition is not seed/last_target.png"
        )
    if previous_gt_hash != current_gt_hash:
        raise ValueError(f"{case_dir}: target GT is not identical at the boundary")
    return {
        "seed_start_frame": seed_start,
        "continuation_start_frame": continuation_start,
        "condition_sha256": condition_hash,
        "target_gt_boundary_sha256": previous_gt_hash,
    }


def build_case(
    case_dir: Path,
    output_subdir: str,
    fps: int,
    overwrite: bool,
) -> dict:
    case_path = case_dir / "case.json"
    if not case_path.is_file():
        raise FileNotFoundError(case_path)
    case = json.loads(case_path.read_text())
    chunk_dirs = resolve_chunk_dirs(case_dir)
    if len(chunk_dirs) > 2:
        return build_multi_case(case_dir, output_subdir, fps, overwrite)
    contract = validate_case_contract(case_dir, case)
    output_dir = case_dir / output_subdir
    generated_path = output_dir / "target_generated_97f.mp4"
    comparison_path = output_dir / "rgb_depth_tracks_2x3_97f.mp4"
    manifest_path = output_dir / "long_rollout.json"
    for filename in REQUIRED_STREAMS:
        for phase in ("seed", "continuation"):
            path = case_dir / phase / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    comparison_seed_path = case_dir / "seed" / "comparison.mp4"
    comparison_continuation_path = case_dir / "continuation" / "comparison.mp4"
    for path in (comparison_seed_path, comparison_continuation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not overwrite and all(
        path.is_file() for path in (generated_path, comparison_path, manifest_path)
    ):
        try:
            manifest_existing = json.loads(manifest_path.read_text())
            long_length_existing = int(manifest_existing["long_frames"])
            generated_count, _ = video_info(generated_path)
            comparison_count, _ = video_info(comparison_path)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            pass
        else:
            if (
                manifest_existing.get("comparison_video") == comparison_path.name
                and generated_count == long_length_existing
                and comparison_count == long_length_existing
            ):
                return {
                    "case": case_dir.name,
                    "status": "skipped",
                    "output": str(output_dir),
                }

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_phase_counts: dict[str, object] = {}
    generated_count = write_video(
        generated_path,
        iter_chained_video(
            case_dir / "seed" / "pred_rgb.mp4",
            case_dir / "continuation" / "pred_rgb.mp4",
            counts=generated_phase_counts,
        ),
        fps,
    )
    seed_length = int(generated_phase_counts["seed"])
    continuation_length = int(generated_phase_counts["continuation"])
    panel_shape = generated_phase_counts["shape"]
    if not isinstance(panel_shape, tuple) or len(panel_shape) != 2:
        raise RuntimeError(f"{case_dir}: missing generated-frame shape")
    panel_height, panel_width = int(panel_shape[0]), int(panel_shape[1])
    boundary_frame = int(contract["continuation_start_frame"])
    expected_boundary = int(contract["seed_start_frame"]) + seed_length - 1
    if boundary_frame != expected_boundary:
        raise ValueError(
            f"{case_dir}: continuation starts at {boundary_frame}, expected "
            f"shared seed boundary {expected_boundary}"
        )
    long_length = seed_length + continuation_length - 1
    src_cam_idx = int(case["src_cam_idx"])
    tgt_cam_idx = int(case["tgt_cam_idx"])
    comparison_phase_counts: dict[str, object] = {}
    comparison_count = write_video(
        comparison_path,
        iter_detailed_comparison(
            comparison_seed_path,
            comparison_continuation_path,
            panel_height=panel_height,
            panel_width=panel_width,
            long_length=long_length,
            seed_start_frame=int(contract["seed_start_frame"]),
            boundary_frame=boundary_frame,
            src_cam_idx=src_cam_idx,
            tgt_cam_idx=tgt_cam_idx,
            fps=fps,
            counts=comparison_phase_counts,
        ),
        fps,
    )
    comparison_lengths = (
        int(comparison_phase_counts["seed"]),
        int(comparison_phase_counts["continuation"]),
    )
    if comparison_lengths != (seed_length, continuation_length):
        raise ValueError(
            f"{case_dir}: comparison window lengths differ from prediction: "
            f"comparison={comparison_lengths}, "
            f"prediction={(seed_length, continuation_length)}"
        )
    comparison_shape = comparison_phase_counts["shape"]
    expected_comparison_shape = (2 * panel_height, 5 * panel_width)
    if (
        not isinstance(comparison_shape, tuple)
        or len(comparison_shape) != 2
        or any(
            int(actual) < expected
            for actual, expected in zip(
                comparison_shape, expected_comparison_shape, strict=True
            )
        )
    ):
        raise ValueError(
            f"{case_dir}: comparison grid too small: {comparison_shape}; "
            f"expected at least {expected_comparison_shape}"
        )
    if generated_count != long_length or comparison_count != long_length:
        raise RuntimeError(
            f"{case_dir}: wrote generated/comparison={generated_count}/"
            f"{comparison_count}, expected {long_length}"
        )
    manifest = {
        "protocol": "seed[all] + continuation[1:] (shared boundary de-duplicated)",
        "case": case_dir.name,
        "source_camera": src_cam_idx,
        "target_camera": tgt_cam_idx,
        "seed_absolute_frames": [
            int(contract["seed_start_frame"]),
            boundary_frame,
        ],
        "continuation_absolute_frames": [
            boundary_frame,
            boundary_frame + continuation_length - 1,
        ],
        "long_absolute_frames": [
            int(contract["seed_start_frame"]),
            boundary_frame + continuation_length - 1,
        ],
        "seed_frames": seed_length,
        "continuation_frames": continuation_length,
        "long_frames": long_length,
        "shared_boundary_absolute_frame": boundary_frame,
        "shared_boundary_long_index": seed_length - 1,
        "condition": "seed generated target last frame",
        "seed_bootstrap": "source-video-only T2V seed; GT target is reference only; seed window is not scored",
        "continuation": "generated-target-conditioned autoregressive window",
        "condition_sha256": contract["condition_sha256"],
        "target_gt_boundary_sha256": contract["target_gt_boundary_sha256"],
        "panel_source": "pure crops from seed/continuation comparison.mp4",
        "depth_panel": "source depth forward-warped into the target camera; not target GT depth",
        "track_panel": "bottom-right tracks-on-pred crop from each original comparison.mp4",
        "generated_video": generated_path.name,
        "comparison_video": comparison_path.name,
        "fps": int(fps),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"case": case_dir.name, "status": "written", "output": str(output_dir)}


def main() -> None:
    args = parse_args()
    cases = resolve_cases(args)
    written = 0
    skipped = 0
    for index, case_dir in enumerate(cases, start=1):
        result = build_case(
            case_dir,
            output_subdir=args.output_subdir,
            fps=args.fps,
            overwrite=args.overwrite,
        )
        written += result["status"] == "written"
        skipped += result["status"] == "skipped"
        print(
            f"[I2V long rollout] {index:03d}/{len(cases):03d} "
            f"{result['status']}: {result['case']} -> {result['output']}",
            flush=True,
        )
    print(
        f"[I2V long rollout] complete: selected={len(cases)} "
        f"written={written} skipped={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
