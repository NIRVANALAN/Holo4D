#!/usr/bin/env python3
"""Label ReCamMaster 4RC source clips with SAM3 foreground masks.

This script reads the existing 4RC safetensors cache to get the exact
scene/camera/start-frame tuples used for 4RC training, runs Vista4D's SAM3
video segmentation wrapper on the corresponding ReCamMaster RGB clip, and
writes a compact sidecar foreground cache keyed by the same prefix:

    <output_root>/train/f18_aperture10/scene1/cam03/dynamic_mask_packbits.npz

Masks are stored as packed bits to keep the full ReCamMaster source-cache
reasonable in size. Use ``np.unpackbits(..., count=np.prod(shape))`` when
loading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

import cv2
import imageio.v2 as imageio
import numpy as np
from safetensors import safe_open


DEFAULT_KEYWORDS = ("person", "human", "man", "woman", "character")
DEFAULT_384P_9X16_RESOLUTION = "672,384"


class LabelTask(NamedTuple):
    prefix: str
    video_path: Path
    start_frame: int
    shard_path: Path | None


def _parse_resolution(value: str) -> tuple[int, int]:
    try:
        w_str, h_str = value.split(",", 1)
        width, height = int(w_str), int(h_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"resolution must be 'W,H', got {value!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"resolution must be positive, got {value!r}")
    return width, height


def _center_crop_resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    target_aspect = float(width) / float(height)
    src_aspect = float(src_w) / float(src_h)
    if src_aspect > target_aspect:
        new_w = int(round(src_h * target_aspect))
        left = (src_w - new_w) // 2
        frame = frame[:, left:left + new_w]
    else:
        new_h = int(round(src_w / target_aspect))
        top = (src_h - new_h) // 2
        frame = frame[top:top + new_h]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def _resize_rgb_clip(clip: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.stack(
        [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for frame in clip
        ],
        axis=0,
    ).astype(np.uint8)


def _mask_video_frames(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8, copy=False) * 255
    return np.repeat(mask_u8[..., None], 3, axis=-1)


def _overlay_video_frames(
    clip: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    if clip.shape[:3] != mask.shape:
        raise ValueError(f"clip/mask shape mismatch: clip={clip.shape} mask={mask.shape}")
    out = clip.astype(np.float32, copy=True)
    red = np.asarray([255.0, 0.0, 0.0], dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + red * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _write_mp4(path: Path, frames: np.ndarray, *, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    writer = imageio.get_writer(
        str(tmp_path),
        fps=int(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        quality=8,
    )
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame.astype(np.uint8, copy=False)))
    finally:
        writer.close()
    os.replace(tmp_path, path)


def _expected_output_paths(
    out_dir: Path,
    *,
    save_mask_mp4: bool,
    save_overlay_mp4: bool,
) -> list[Path]:
    outputs = [out_dir / "dynamic_mask_packbits.npz"]
    if save_mask_mp4:
        outputs.append(out_dir / "dynamic_mask.mp4")
    if save_overlay_mp4:
        outputs.append(out_dir / "dynamic_mask_overlay.mp4")
    return outputs


def _mask_cache_matches(
    npz_path: Path,
    *,
    num_frames: int,
    output_resolution: tuple[int, int],
) -> bool:
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path) as data:
            shape = tuple(int(v) for v in data["shape"])
            cached_resolution = tuple(int(v) for v in data["output_resolution"])
    except Exception:
        return False
    out_w, out_h = output_resolution
    return shape == (int(num_frames), int(out_h), int(out_w)) and cached_resolution == (
        int(out_w),
        int(out_h),
    )


def _load_rgb_clip(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    width: int,
    height: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    frames: list[np.ndarray] = []
    last_frame = None
    for _ in range(int(num_frames)):
        ok, frame_bgr = cap.read()
        if not ok:
            if last_frame is None:
                cap.release()
                raise RuntimeError(f"failed to read first frame from {video_path}")
            frame_rgb = last_frame
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _center_crop_resize_rgb(frame_rgb, width=width, height=height)
            last_frame = frame_rgb
        frames.append(frame_rgb)
    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8)


def _iter_fourrc_prefixes(cache_root: Path) -> Iterable[tuple[Path, str]]:
    shard_paths = sorted(cache_root.glob("4rc_labels.shard*of*.part*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"no 4RC safetensors shards found in {cache_root}")
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            prefixes = sorted(
                key[: -len("/start_frame")]
                for key in handle.keys()
                if key.endswith("/start_frame")
            )
        for prefix in prefixes:
            yield shard_path, prefix


def _video_path_for_prefix(dataset_root: Path, prefix: str) -> Path:
    rel = _safe_rel_prefix(prefix)
    return dataset_root / Path(*rel.parts[:-1]) / "videos" / f"{rel.parts[-1]}.mp4"


def _iter_fourrc_tasks(dataset_root: Path, cache_root: Path) -> Iterable[LabelTask]:
    for shard_path, prefix in _iter_fourrc_prefixes(cache_root):
        start_frame = _load_start_frame(shard_path, prefix)
        yield LabelTask(
            prefix=prefix,
            video_path=_video_path_for_prefix(dataset_root, prefix),
            start_frame=start_frame,
            shard_path=shard_path,
        )


def _iter_fourrc_full_video_tasks(
    dataset_root: Path,
    cache_root: Path,
) -> Iterable[LabelTask]:
    seen: set[str] = set()
    for shard_path, prefix in _iter_fourrc_prefixes(cache_root):
        if prefix in seen:
            continue
        seen.add(prefix)
        yield LabelTask(
            prefix=prefix,
            video_path=_video_path_for_prefix(dataset_root, prefix),
            start_frame=0,
            shard_path=shard_path,
        )


def _iter_all_recammaster_tasks(
    dataset_root: Path,
    split: str,
    *,
    exclude_fourrc_prefixes: set[str] | None = None,
) -> Iterable[LabelTask]:
    scene_dirs = sorted((dataset_root / split).glob("*/*"))
    for scene_dir in scene_dirs:
        videos_dir = scene_dir / "videos"
        if not videos_dir.is_dir():
            continue
        for video_path in sorted(videos_dir.glob("cam*.mp4")):
            prefix = str((scene_dir.relative_to(dataset_root) / video_path.stem).as_posix())
            if exclude_fourrc_prefixes is not None and prefix in exclude_fourrc_prefixes:
                continue
            yield LabelTask(
                prefix=prefix,
                video_path=video_path,
                start_frame=0,
                shard_path=None,
            )


def _single_video_task(video_path: Path, prefix: str, start_frame: int) -> LabelTask:
    return LabelTask(
        prefix=prefix,
        video_path=video_path,
        start_frame=start_frame,
        shard_path=None,
    )


def _safe_rel_prefix(prefix: str) -> Path:
    parts = Path(prefix).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe 4RC prefix: {prefix!r}")
    return Path(*parts)


def _load_start_frame(shard_path: Path, prefix: str) -> int:
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        return int(handle.get_tensor(f"{prefix}/start_frame").item())


def _write_mask_cache(
    out_dir: Path,
    mask: np.ndarray,
    *,
    prefix: str,
    video_path: Path,
    start_frame: int,
    keywords: list[str],
    sam_resolution: tuple[int, int],
    output_resolution: tuple[int, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = mask.astype(bool, copy=False)
    shape = np.asarray(mask.shape, dtype=np.int32)
    packed = np.packbits(mask.reshape(-1).astype(np.uint8))

    npz_path = out_dir / "dynamic_mask_packbits.npz"
    tmp_npz = out_dir / "dynamic_mask_packbits.npz.tmp.npz"
    np.savez_compressed(
        tmp_npz,
        dynamic_mask_packed=packed,
        shape=shape,
        start_frame=np.asarray(start_frame, dtype=np.int32),
        sam_resolution=np.asarray(sam_resolution, dtype=np.int32),
        output_resolution=np.asarray(output_resolution, dtype=np.int32),
    )
    os.replace(tmp_npz, npz_path)

    meta = {
        "prefix": prefix,
        "video_path": str(video_path),
        "start_frame": int(start_frame),
        "num_frames": int(mask.shape[0]),
        "shape": [int(v) for v in mask.shape],
        "keywords": keywords,
        "sam_resolution": [int(v) for v in sam_resolution],
        "output_resolution": [int(v) for v in output_resolution],
        "dynamic_coverage": float(mask.mean()),
        "format": "np.packbits(flat bool mask); unpack with count=prod(shape)",
    }
    tmp_meta = out_dir / "meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp_meta, out_dir / "meta.json")


def _load_sam3(vista4d_root: Path):
    sys.path.insert(0, str(vista4d_root))
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video, run_sam3_video

    return init_sam3_video(), run_sam3_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="data/MultiCamVideo-Dataset",
        type=Path,
    )
    parser.add_argument(
        "--fourrc_cache_root",
        default="data/MultiCamVideo-Dataset_4RC",
        type=Path,
    )
    parser.add_argument(
        "--output_root",
        default="data/MultiCamVideo-Dataset_4RC_SAM3_FG",
        type=Path,
    )
    parser.add_argument(
        "--vista4d_root",
        default="third_party/Vista4D",
        type=Path,
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument(
        "--task_mode",
        choices=("fourrc", "fourrc_full_video", "all_recammaster", "single_video"),
        default="fourrc",
    )
    parser.add_argument("--single_video_path", default=None, type=Path)
    parser.add_argument("--single_prefix", default="single_video", type=str)
    parser.add_argument("--single_start_frame", default=0, type=int)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument(
        "--exclude_fourrc_prefixes",
        action="store_true",
        help="Only with --task_mode all_recammaster: skip source cameras already present in the 4RC cache.",
    )
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument(
        "--sam_resolution",
        type=_parse_resolution,
        default=_parse_resolution(DEFAULT_384P_9X16_RESOLUTION),
    )
    parser.add_argument(
        "--output_resolution",
        type=_parse_resolution,
        default=_parse_resolution(DEFAULT_384P_9X16_RESOLUTION),
    )
    parser.add_argument("--keywords", nargs="+", default=list(DEFAULT_KEYWORDS))
    parser.add_argument("--max_items", type=int, default=0, help="Per-shard cap; 0 means no cap.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reverse_order",
        action="store_true",
        help="Process the task list in reverse before sharding. Useful for a second resume job that starts from the tail.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_mask_mp4", action="store_true")
    parser.add_argument("--save_overlay_mp4", action="store_true")
    parser.add_argument("--mp4_fps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")

    keywords = [kw.strip() for kw in args.keywords if kw.strip()]
    if not keywords:
        raise ValueError("at least one --keywords value is required")

    print(
        "[sam3-fg] "
        f"dataset_root={args.dataset_root} fourrc_cache_root={args.fourrc_cache_root} "
        f"output_root={args.output_root} task_mode={args.task_mode} "
        f"shard={args.shard_id}/{args.num_shards} "
        f"keywords={keywords} sam_resolution={args.sam_resolution} "
        f"output_resolution={args.output_resolution}"
    )

    if args.task_mode == "single_video":
        if args.single_video_path is None:
            raise ValueError("--single_video_path is required with --task_mode single_video")
        tasks = [
            _single_video_task(
                args.single_video_path,
                args.single_prefix,
                args.single_start_frame,
            )
        ]
    elif args.task_mode == "fourrc":
        tasks = list(_iter_fourrc_tasks(args.dataset_root, args.fourrc_cache_root))
    elif args.task_mode == "fourrc_full_video":
        tasks = list(_iter_fourrc_full_video_tasks(args.dataset_root, args.fourrc_cache_root))
    else:
        exclude_prefixes = None
        if args.exclude_fourrc_prefixes:
            exclude_prefixes = {
                prefix for _shard_path, prefix in _iter_fourrc_prefixes(args.fourrc_cache_root)
            }
            print(f"[sam3-fg] excluding_fourrc_prefixes={len(exclude_prefixes)}")
        tasks = list(
            _iter_all_recammaster_tasks(
                args.dataset_root,
                args.split,
                exclude_fourrc_prefixes=exclude_prefixes,
            )
        )
    if args.reverse_order:
        tasks = list(reversed(tasks))
    shard_tasks = [
        task for idx, task in enumerate(tasks)
        if idx % args.num_shards == args.shard_id
    ]
    if args.max_items > 0:
        shard_tasks = shard_tasks[: args.max_items]
    print(f"[sam3-fg] total_tasks={len(tasks)} shard_tasks={len(shard_tasks)}")

    if args.dry_run:
        for task in shard_tasks[:10]:
            print(f"[dry-run] {task.prefix} start={task.start_frame} video={task.video_path}")
        return

    predictor, run_sam3_video = _load_sam3(args.vista4d_root)
    sam_w, sam_h = args.sam_resolution
    out_w, out_h = args.output_resolution

    done = 0
    skipped = 0
    failed = 0
    for local_idx, task in enumerate(shard_tasks):
        rel = _safe_rel_prefix(task.prefix)
        out_dir = args.output_root / rel
        expected_outputs = _expected_output_paths(
            out_dir,
            save_mask_mp4=args.save_mask_mp4,
            save_overlay_mp4=args.save_overlay_mp4,
        )
        out_npz = expected_outputs[0]
        cache_matches = _mask_cache_matches(
            out_npz,
            num_frames=args.num_frames,
            output_resolution=args.output_resolution,
        )
        if (
            cache_matches
            and all(path.exists() for path in expected_outputs)
            and not args.overwrite
        ):
            skipped += 1
            continue

        try:
            clip = _load_rgb_clip(
                task.video_path,
                start_frame=task.start_frame,
                num_frames=args.num_frames,
                width=sam_w,
                height=sam_h,
            )
            dynamic_mask, _seg_frames = run_sam3_video(clip, predictor, keywords)
            if (out_w, out_h) != (sam_w, sam_h):
                dynamic_mask = np.stack(
                    [_resize_mask(mask, width=out_w, height=out_h) for mask in dynamic_mask],
                    axis=0,
                )
            _write_mask_cache(
                out_dir,
                dynamic_mask,
                prefix=task.prefix,
                video_path=task.video_path,
                start_frame=task.start_frame,
                keywords=keywords,
                sam_resolution=args.sam_resolution,
                output_resolution=args.output_resolution,
            )
            if args.save_mask_mp4:
                _write_mp4(
                    out_dir / "dynamic_mask.mp4",
                    _mask_video_frames(dynamic_mask),
                    fps=args.mp4_fps,
                )
            if args.save_overlay_mp4:
                vis_clip = clip
                if (out_w, out_h) != (sam_w, sam_h):
                    vis_clip = _resize_rgb_clip(clip, width=out_w, height=out_h)
                _write_mp4(
                    out_dir / "dynamic_mask_overlay.mp4",
                    _overlay_video_frames(vis_clip, dynamic_mask),
                    fps=args.mp4_fps,
                )
            done += 1
            print(
                f"[sam3-fg] {args.shard_id}/{args.num_shards} "
                f"{local_idx + 1}/{len(shard_tasks)} wrote {out_npz} "
                f"coverage={float(dynamic_mask.mean()):.4f}"
            )
        except Exception as exc:
            failed += 1
            fail_dir = args.output_root / rel
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            print(f"[sam3-fg][ERROR] {task.prefix}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"[sam3-fg] complete shard={args.shard_id}: wrote={done} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
