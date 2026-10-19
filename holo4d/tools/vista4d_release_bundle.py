"""Create or extract the checksummed, single-NPZ Vista4D inference bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from numpy.lib import format as npformat

FORMAT = "holo4d-vista4d-inference-bundle-v1"
MANIFEST_MEMBER = "manifest.json.npy"


def _write_bytes(zf: zipfile.ZipFile, member: str, source: Path) -> tuple[int, str]:
    size = source.stat().st_size
    digest = hashlib.sha256()
    with zf.open(member, "w", force_zip64=True) as dst, source.open("rb") as src:
        npformat.write_array_header_1_0(
            dst, {"descr": np.dtype("uint8").str, "fortran_order": False, "shape": (size,)}
        )
        while chunk := src.read(8 * 1024 * 1024):
            digest.update(chunk)
            dst.write(chunk)
    return size, digest.hexdigest()


def _write_inline(zf: zipfile.ZipFile, member: str, payload: bytes) -> None:
    with zf.open(member, "w", force_zip64=True) as dst:
        npformat.write_array_header_1_0(
            dst,
            {"descr": np.dtype("uint8").str, "fortran_order": False, "shape": (len(payload),)},
        )
        dst.write(payload)


def _read_header(src) -> int:
    version = npformat.read_magic(src)
    if version == (1, 0):
        shape, fortran, dtype = npformat.read_array_header_1_0(src)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran, dtype = npformat.read_array_header_2_0(src)
    else:
        raise ValueError(f"unsupported NPY version: {version}")
    if fortran or np.dtype(dtype) != np.dtype("uint8") or len(shape) != 1:
        raise ValueError("bundle members must be one-dimensional uint8 arrays")
    return int(shape[0])


def _add(files: dict[str, Path], rel: str, source: Path, required: bool = True) -> None:
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"required bundle input is missing: {source}")
        return
    if rel in files:
        raise ValueError(f"duplicate bundle path: {rel}")
    files[rel] = source


def collect(args: argparse.Namespace) -> tuple[dict[str, Path], list[str]]:
    eval_root, frame_root = Path(args.eval_root), Path(args.frame_root)
    cow_root, vggt_root = Path(args.cow_root), Path(args.vggt_root)
    metadata = eval_root / "metadata.csv"
    with metadata.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    camera_order: dict[str, list[str]] = {}
    for row in rows:
        cameras = camera_order.setdefault(row["video"], [])
        if row["camera"] not in cameras:
            cameras.append(row["camera"])
    available = sorted(
        video for video in camera_order
        if (frame_root / video).is_dir()
        and (cow_root / f"{video}_pred_trackvideo.npz").is_file()
        and (vggt_root / video).is_dir()
    )
    videos = [v.strip() for v in args.videos.split(",") if v.strip()] if args.videos else available
    unknown = sorted(set(videos) - set(available))
    if unknown:
        raise ValueError(f"videos missing required caches: {unknown}")
    videos = videos[: args.limit] if args.limit is not None else videos
    if not videos:
        raise ValueError("no Vista4D videos selected")

    files: dict[str, Path] = {}
    _add(files, "vista4d_eval/metadata.csv", metadata)
    _add(files, "vista4d_eval/README.md", eval_root / "README.md", False)
    for video in videos:
        frames = sorted(p for p in (frame_root / video).iterdir() if p.is_file())
        if len(frames) < 49:
            raise ValueError(f"{video}: expected 49 source frames, found {len(frames)}")
        for source in frames[:49]:
            _add(files, f"frames/{video}/{source.name}", source)
        _add(files, f"cow/{video}_pred_trackvideo.npz", cow_root / f"{video}_pred_trackvideo.npz")
        _add(files, f"cow/{video}_pred_trackvideo.mp4", cow_root / f"{video}_pred_trackvideo.mp4", False)
        for rel in ("geometry/vggt_omega_depth.npz", "pose/vggt_omega.npz"):
            _add(files, f"vggt/{video}/{rel}", vggt_root / video / rel)
        recon = eval_root / "recon_and_seg" / video
        _add(files, f"vista4d_eval/recon_and_seg/{video}/cameras.npz", recon / "cameras.npz")
        _add(files, f"vista4d_eval/recon_and_seg/{video}/depths/00000.exr", recon / "depths/00000.exr")
        for folder in ("dynamic_mask", "sky_mask"):
            _add(files, f"vista4d_eval/recon_and_seg/{video}/{folder}/00000.png", recon / folder / "00000.png", False)
        cameras = camera_order[video][: args.max_cameras]
        if len(cameras) != args.max_cameras:
            raise ValueError(f"{video}: expected {args.max_cameras} target cameras")
        for camera in cameras:
            _add(
                files,
                f"vista4d_eval/cameras/{video}/{camera}.npz",
                eval_root / "cameras" / video / f"{camera}.npz",
            )
    return files, videos


def pack(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.suffix != ".npz":
        raise ValueError("output must end in .npz")
    files, videos = collect(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    records = []
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for index, (rel, source) in enumerate(sorted(files.items())):
                member = f"files/{index:05d}.npy"
                size, digest = _write_bytes(zf, member, source)
                records.append({"path": rel, "member": member, "size": size, "sha256": digest})
                print(f"[{index + 1}/{len(files)}] {rel} ({size / 1024**2:.1f} MiB)", flush=True)
            manifest = {
                "format": FORMAT,
                "videos": videos,
                "max_cameras_per_video": args.max_cameras,
                "files": records,
                "sources": {
                    "vista4d_eval": "https://huggingface.co/datasets/Eyeline-Labs/Vista4D-Eval-Data",
                    "license": "Apache-2.0 for upstream Vista4D evaluation assets",
                },
            }
            _write_inline(zf, MANIFEST_MEMBER, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {output} ({output.stat().st_size / 1024**3:.2f} GiB; {len(videos)} videos)")


def _manifest(zf: zipfile.ZipFile) -> dict:
    with zf.open(MANIFEST_MEMBER) as src:
        manifest = json.loads(src.read(_read_header(src)).decode())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"unsupported bundle format: {manifest.get('format')!r}")
    return manifest


def extract(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "r", allowZip64=True) as zf:
        manifest = _manifest(zf)
        for index, entry in enumerate(manifest["files"]):
            rel = PurePosixPath(entry["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe path in bundle: {rel}")
            destination = output.joinpath(*rel.parts)
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite")
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with zf.open(entry["member"]) as src:
                remaining = _read_header(src)
                if remaining != entry["size"]:
                    raise ValueError(f"manifest size mismatch for {rel}")
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as dst:
                    temporary = Path(dst.name)
                    while remaining:
                        chunk = src.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise EOFError(f"truncated member for {rel}")
                        dst.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
            if digest.hexdigest() != entry["sha256"]:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"SHA-256 mismatch for {rel}")
            os.replace(temporary, destination)
            print(f"[{index + 1}/{len(manifest['files'])}] {rel}", flush=True)
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Extracted to {output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    create = sub.add_parser("pack")
    for name in ("eval-root", "frame-root", "cow-root", "vggt-root", "output"):
        create.add_argument(f"--{name}", required=True)
    create.add_argument("--videos")
    create.add_argument("--limit", type=int)
    create.add_argument("--max-cameras", type=int, default=2)
    create.set_defaults(func=pack)
    unpack = sub.add_parser("extract")
    unpack.add_argument("archive")
    unpack.add_argument("--output", required=True)
    unpack.add_argument("--overwrite", action="store_true")
    unpack.set_defaults(func=extract)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
