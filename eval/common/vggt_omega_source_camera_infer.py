#!/usr/bin/env python3
"""VGGT-Omega camera/depth inference for source videos or frame directories.

This is source-video inference for roots such as DAVIS, Vista4D Pexels, and
WorldTrack original MP4s. It does not compute metrics because these roots are
not the DAVIS-50 x 2-traj prediction-vs-GT metric layout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}
MIN_SUPPORTED_ASPECT = 0.5
MAX_SUPPORTED_ASPECT = 2.0
VGGT_PATCH_SIZE = 16


def ensure_vggt_omega_import(vggt_omega_repo: Path) -> None:
    if str(vggt_omega_repo) not in sys.path:
        sys.path.insert(0, str(vggt_omega_repo))


def image_files(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def resolve_sequence_frame_dir(seq_dir: Path) -> Path | None:
    nested = seq_dir / "frames"
    if image_files(nested):
        return nested
    if image_files(seq_dir):
        return seq_dir
    return None


def discover_frame_sequences(input_root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for seq_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        frame_dir = resolve_sequence_frame_dir(seq_dir)
        if frame_dir is not None:
            items.append((seq_dir.name, frame_dir))
    return items


def discover_video_sequences(input_root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for video_path in sorted(p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES):
        seq_name = str(video_path.relative_to(input_root).with_suffix(""))
        items.append((seq_name, video_path))
    return items


def discover_sequences(input_root: Path, input_mode: str) -> list[tuple[str, Path]]:
    if input_mode == "frame_dirs":
        return discover_frame_sequences(input_root)
    if input_mode == "mp4":
        return discover_video_sequences(input_root)
    if input_mode == "auto":
        frame_items = discover_frame_sequences(input_root)
        return frame_items if frame_items else discover_video_sequences(input_root)
    raise ValueError(f"unknown input_mode={input_mode}")


def read_sequence_list(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split(",")[0].strip())
    return names


def shard_items(
    items: list[tuple[str, Path]],
    num_shards: int,
    shard_index: int,
) -> list[tuple[str, Path]]:
    if num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"--shard_index must be in [0, {num_shards}), got {shard_index}")
    return [item for i, item in enumerate(items) if i % num_shards == shard_index]


def load_model(
    checkpoint_path: Path,
    repo: Path,
    device: torch.device,
    enable_alignment: bool,
    enable_depth: bool,
):
    ensure_vggt_omega_import(repo)
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(enable_depth=enable_depth, enable_alignment=enable_alignment).to(device).eval()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[vggt-source] checkpoint load: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[vggt-source] missing head: {missing[:8]}")
        if unexpected:
            print(f"[vggt-source] unexpected head: {unexpected[:8]}")
    return model


def select_frames(frame_dir: Path, num_frames: int) -> list[Path]:
    frames = image_files(frame_dir)
    if not frames:
        raise RuntimeError(f"no images found under {frame_dir}")
    if len(frames) >= num_frames:
        return frames[:num_frames]
    return frames + [frames[-1]] * (num_frames - len(frames))


def read_video_frames(video_path: Path, num_frames: int) -> tuple[list[Image.Image], np.ndarray, list[str]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    images: list[Image.Image] = []
    source_hw: list[tuple[int, int]] = []
    labels: list[str] = []
    frame_idx = 0
    try:
        while len(images) < num_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            images.append(Image.fromarray(rgb))
            source_hw.append((height, width))
            labels.append(f"{video_path}#frame={frame_idx}")
            frame_idx += 1
    finally:
        cap.release()

    if not images:
        raise RuntimeError(f"no frames decoded from video: {video_path}")

    while len(images) < num_frames:
        images.append(images[-1].copy())
        source_hw.append(source_hw[-1])
        labels.append(f"{video_path}#frame={frame_idx - 1}:repeat")

    return images, np.asarray(source_hw, dtype=np.int32), labels


def image_sizes(frame_paths: list[Path]) -> np.ndarray:
    sizes = []
    for path in frame_paths:
        with Image.open(path) as image:
            width, height = image.size
        sizes.append((height, width))
    return np.asarray(sizes, dtype=np.int32)


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


def crop_to_supported_aspect_ratio(
    image: Image.Image,
    min_aspect_ratio: float = MIN_SUPPORTED_ASPECT,
    max_aspect_ratio: float = MAX_SUPPORTED_ASPECT,
) -> Image.Image:
    width, height = image.size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height))

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height))

    return image


def closest_vggt_shape(height: int, width: int, image_resolution: int, patch_size: int = VGGT_PATCH_SIZE) -> tuple[int, int]:
    """Choose the max-side grid whose aspect ratio is closest to the source."""
    if image_resolution % patch_size != 0:
        raise ValueError(f"image_resolution must be divisible by {patch_size}, got {image_resolution}")

    source_aspect = height / max(width, 1)
    candidates: list[tuple[int, int]] = []
    for candidate_h in range(patch_size, image_resolution + 1, patch_size):
        for candidate_w in range(patch_size, image_resolution + 1, patch_size):
            if max(candidate_h, candidate_w) != image_resolution:
                continue
            aspect = candidate_h / candidate_w
            if MIN_SUPPORTED_ASPECT <= aspect <= MAX_SUPPORTED_ASPECT:
                candidates.append((candidate_h, candidate_w))
    if not candidates:
        raise ValueError(f"no VGGT target shapes available for image_resolution={image_resolution}")

    return min(
        candidates,
        key=lambda hw: (
            abs(np.log((hw[0] / hw[1]) / source_aspect)),
            -hw[0] * hw[1],
        ),
    )


def preprocess_pil_images_closest_aspect(
    source_images: Iterable[Image.Image],
    image_resolution: int,
    patch_size: int = VGGT_PATCH_SIZE,
) -> torch.Tensor:
    images = []
    target_hw = None
    for source_image in source_images:
        image = crop_to_supported_aspect_ratio(source_image)
        width, height = image.size
        if target_hw is None:
            target_hw = closest_vggt_shape(height, width, image_resolution, patch_size)
        target_h, target_w = target_hw
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(images)


def load_and_preprocess_images_closest_aspect(
    frame_paths: list[Path],
    image_resolution: int,
    patch_size: int = VGGT_PATCH_SIZE,
) -> torch.Tensor:
    source_images = [load_rgb_image(path) for path in frame_paths]
    try:
        return preprocess_pil_images_closest_aspect(source_images, image_resolution, patch_size)
    finally:
        for image in source_images:
            image.close()


def infer_one(
    seq_name: str,
    source_path: Path,
    output_root: Path,
    model,
    repo: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    ensure_vggt_omega_import(repo)
    from vggt_omega.utils.geometry import closed_form_inverse_se3
    from vggt_omega.utils.pose_enc import encoding_to_camera
    if args.image_mode != "closest_aspect":
        from vggt_omega.utils.load_fn import load_and_preprocess_images

    input_type = "mp4" if source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES else "frame_dirs"
    seq_out = output_root / seq_name
    pose_dir = seq_out / "pose"
    geom_dir = seq_out / "geometry"
    pose_npz = pose_dir / "vggt_omega.npz"
    depth_npz = geom_dir / "vggt_omega_depth.npz"
    needs_depth = args.save_depth and not depth_npz.is_file()
    if pose_npz.is_file() and not needs_depth and not args.force:
        return {
            "sequence": seq_name,
            "source_path": str(source_path),
            "input_type": input_type,
            "pose_npz": str(pose_npz),
            "depth_npz": str(depth_npz) if depth_npz.is_file() else None,
            "source_hw_first": None,
            "vggt_hw": None,
            "status": "skipped",
        }

    if input_type == "mp4":
        video_images, src_hw, frame_labels = read_video_frames(source_path, args.num_frames)
        if args.image_mode != "closest_aspect":
            raise ValueError("MP4 input currently requires --image_mode closest_aspect")
        try:
            images = preprocess_pil_images_closest_aspect(
                video_images,
                image_resolution=args.image_resolution,
            ).to(device)
        finally:
            for image in video_images:
                image.close()
    else:
        frame_paths = select_frames(source_path, args.num_frames)
        src_hw = image_sizes(frame_paths)
        frame_labels = [str(p) for p in frame_paths]
        if args.image_mode == "closest_aspect":
            images = load_and_preprocess_images_closest_aspect(
                frame_paths,
                image_resolution=args.image_resolution,
            ).to(device)
        else:
            images = load_and_preprocess_images(
                [str(p) for p in frame_paths],
                mode=args.image_mode,
                image_resolution=args.image_resolution,
            ).to(device)

    if args.image_mode == "closest_aspect":
        source_note = "closest-aspect preprocessing"
    else:
        source_note = "VGGT-Omega load_fn preprocessing"

    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

    extr = extrinsics[0] if extrinsics.ndim == 4 else extrinsics
    intr_out = intrinsics[0] if intrinsics is not None and intrinsics.ndim == 4 else intrinsics
    extr = extr.detach().float().cpu()
    c2w = closed_form_inverse_se3(extr).detach().cpu().numpy().astype(np.float32)
    intr = intr_out.detach().float().cpu().numpy().astype(np.float32) if intr_out is not None else None
    model_hw = np.asarray(predictions["images"].shape[-2:], dtype=np.int32)

    pose_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pose_npz,
        cam_c2w=c2w,
        data=c2w,
        extrinsics_c2w=c2w,
        extrinsics_w2c=extr.numpy().astype(np.float32),
        intrinsics=intr if intr is not None else np.empty((0,), dtype=np.float32),
        frame_paths=np.asarray(frame_labels),
        source_path=np.asarray(str(source_path)),
        input_type=np.asarray(input_type),
        source_hw=src_hw,
        vggt_hw=model_hw,
        dataset=np.asarray(args.dataset_name),
        sequence=np.asarray(seq_name),
        image_resolution=np.asarray(args.image_resolution, dtype=np.int32),
        image_mode=np.asarray(args.image_mode),
        note=np.asarray(source_note),
    )
    depth_npz_out = None
    if args.save_depth:
        if "depth" not in predictions or "depth_conf" not in predictions:
            raise RuntimeError("VGGT-Omega depth output is missing; depth save requires depth enabled")
        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        depth_np = depth[0].detach().float().cpu().numpy() if depth.ndim == 5 else depth.detach().float().cpu().numpy()
        conf_np = depth_conf[0].detach().float().cpu().numpy() if depth_conf.ndim == 5 else depth_conf.detach().float().cpu().numpy()
        geom_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            depth_npz,
            depth=depth_np.astype(np.float32),
            depth_conf=conf_np.astype(np.float32),
            extrinsics_w2c=extr.numpy().astype(np.float32),
            intrinsics=intr,
            frame_paths=np.asarray(frame_labels),
            source_path=np.asarray(str(source_path)),
            input_type=np.asarray(input_type),
            source_hw=src_hw,
            vggt_hw=model_hw,
            dataset=np.asarray(args.dataset_name),
            sequence=np.asarray(seq_name),
            image_resolution=np.asarray(args.image_resolution, dtype=np.int32),
            image_mode=np.asarray(args.image_mode),
            note=np.asarray(source_note),
        )
        depth_npz_out = str(depth_npz)
    return {
        "sequence": seq_name,
        "source_path": str(source_path),
        "input_type": input_type,
        "pose_npz": str(pose_npz),
        "depth_npz": depth_npz_out,
        "source_hw_first": src_hw[0].tolist() if len(src_hw) else None,
        "vggt_hw": model_hw.tolist(),
        "status": "fresh",
    }


def write_manifest(output_root: Path, rows: Iterable[dict], shard_index: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    shard_json = output_root / f"summary_shard{shard_index}.json"
    shard_json.write_text(json.dumps(rows, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--input_root", type=Path, required=True)
    p.add_argument("--input_mode", default="frame_dirs", choices=["frame_dirs", "mp4", "auto"])
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--sequence_list", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vggt_omega_repo", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--image_resolution", type=int, default=512)
    p.add_argument("--image_mode", default="closest_aspect", choices=["closest_aspect", "balanced", "max_size"])
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--limit_sequences", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--enable_alignment", action="store_true")
    p.add_argument("--save_depth", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {args.input_root}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.vggt_omega_repo.is_dir():
        raise FileNotFoundError(f"VGGT-Omega repo not found: {args.vggt_omega_repo}")

    sequences = discover_sequences(args.input_root, args.input_mode)
    if args.sequence_list:
        keep = read_sequence_list(args.sequence_list)
        sequences = [item for item in sequences if item[0] in keep or Path(item[0]).name in keep]
    if args.limit_sequences:
        sequences = sequences[: args.limit_sequences]

    shard = shard_items(sequences, args.num_shards, args.shard_index)
    print(
        f"[vggt-source] dataset={args.dataset_name} total={len(sequences)} "
        f"shard={args.shard_index}/{args.num_shards} n={len(shard)}"
    )
    print(f"[vggt-source] input_root={args.input_root}")
    print(f"[vggt-source] input_mode={args.input_mode}")
    print(f"[vggt-source] output_root={args.output_root}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.checkpoint,
        args.vggt_omega_repo,
        device,
        args.enable_alignment,
        enable_depth=args.save_depth,
    )

    rows: list[dict] = []
    for idx, (seq_name, source_path) in enumerate(shard, start=1):
        try:
            row = infer_one(seq_name, source_path, args.output_root, model, args.vggt_omega_repo, device, args)
            rows.append(row)
            suffix = f", depth={row['depth_npz']}" if row.get("depth_npz") else ""
            hw = f", vggt_hw={row['vggt_hw']}" if row.get("vggt_hw") else ""
            print(f"  [{idx}/{len(shard)}] {seq_name}: {row['status']} -> {row['pose_npz']}{suffix}{hw}")
        except Exception as exc:
            row = {
                "sequence": seq_name,
                "source_path": str(source_path),
                "pose_npz": None,
                "status": "failed",
                "error": str(exc),
            }
            rows.append(row)
            print(f"  [{idx}/{len(shard)}] {seq_name}: FAIL ({exc})")

    write_manifest(args.output_root, rows, args.shard_index)
    n_failed = sum(1 for row in rows if row["status"] == "failed")
    if n_failed:
        raise SystemExit(f"{n_failed} sequence(s) failed")


if __name__ == "__main__":
    main()
