#!/usr/bin/env python3
"""Convert a raw MP4 into a portable, dense Holo4D source-condition bundle.

The 4RC path produces direct persistent 3D tracks. The CoW/VGGT-Omega path
tracks densely in 2D and lifts those tracks using VGGT-Omega depth and cameras.
External projects and checkpoints remain user-installed dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import warnings
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NUM_FRAMES = 49
TRAJECTORIES = (
    "pan_left", "pan_right", "pan_up", "pan_down", "translate_left", "translate_right",
    "zoom_in", "zoom_out", "orbit_left", "orbit_right", "static_left", "static_right",
    "static_up", "static_left_front", "static_right_front", "slow_pan_left", "slow_pan_right",
    "slow_truck_left", "slow_truck_right", "slow_dolly_in", "slow_crane_up", "slow_crane_down",
    "slow_left_front", "slow_right_front", "slow_crane_dolly_in",
)


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not name:
        raise ValueError("sequence name contains no safe filename characters")
    return name[:120]


def require_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"missing required path: {label}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def python_command(script: Path, *, env_name: str | None, conda_command: str) -> list[str]:
    if env_name:
        return [conda_command, "run", "--no-capture-output", "-n", env_name, "python", str(script)]
    return [sys.executable, str(script)]


def git_revision(repo: Path | None) -> str | None:
    if repo is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_fingerprint(value: str | Path | None) -> dict | None:
    if value is None or str(value) == "":
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        candidates = [path / name for name in ("model.safetensors", "pytorch_model.bin", "model.bin")]
        checkpoint_file = next((candidate for candidate in candidates if candidate.is_file()), None)
        if checkpoint_file is None:
            return {"path": str(path.resolve()), "reference": str(value)}
        path = checkpoint_file
    if not path.is_file():
        return {"reference": str(value)}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def run_step(
    label: str,
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    command_manifest: list[dict] | None = None,
) -> None:
    print(f"\n[raw-preprocess] {label}", flush=True)
    command_line = "[raw-preprocess] $ " + " ".join(repr(part) for part in command)
    print(command_line, flush=True)
    if command_manifest is not None:
        command_manifest.append({"label": label, "argv": command, "cwd": str(cwd) if cwd else None})
    with log_path.open("w") if log_path else open(os.devnull, "w") as log_handle:
        log_handle.write(command_line + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def warn_aspect_crop(video: Path, width: int, height: int) -> None:
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height:stream_tags=rotate:stream_side_data=rotation',
                             '-of', 'json', str(video)], capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)['streams'][0]
    source_w, source_h = stream['width'], stream['height']
    rotation = stream.get('tags', {}).get('rotate', 0)
    for side in stream.get('side_data_list', []):
        rotation = side.get('rotation', rotation)
    if abs(float(rotation)) % 180 == 90:
        source_w, source_h = source_h, source_w
    ratio = (source_w / source_h) / (width / height)
    if ratio > 1.25 or ratio < 0.8:
        warnings.warn(f'Input aspect {source_w}x{source_h} differs strongly from {width}x{height}; '
                      'center crop will discard substantial content. For portrait robot video use '
                      '--width 384 --height 672. This runner prepares one49-frame window only.',
                      RuntimeWarning, stacklevel=2)


def stage_video(
    video: Path,
    frame_dir: Path,
    *,
    width: int,
    height: int,
    fps: float,
    log_path: Path | None = None,
    command_manifest: list[dict] | None = None,
) -> int:
    warn_aspect_crop(video, width, height)
    frame_dir.mkdir(parents=True, exist_ok=False)
    vf = (
        f"fps={fps:.8g},scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", vf, "-frames:v", str(NUM_FRAMES), "-start_number", "0",
        str(frame_dir / "frame_%05d.png"),
    ]
    run_step(
        "decode/crop/resample exactly 49 source frames",
        command,
        log_path=log_path,
        command_manifest=command_manifest,
    )
    count = len(list(frame_dir.glob("frame_*.png")))
    if count != NUM_FRAMES:
        raise RuntimeError(
            f"Could not produce {NUM_FRAMES} frames at {fps} fps from {video}; got {count}. "
            "Provide a longer clip or change its frame rate/window before preprocessing."
        )
    return count


def make_caption_json(path: Path, sequence_name: str, caption: str) -> None:
    path.write_text(json.dumps({sequence_name: caption}, indent=2, ensure_ascii=False) + "\n")


def cowtracker_canvas(width: int, height: int, max_side: int = 560) -> tuple[int, int]:
    """Return H,W compatible with patch-14 tokens and the stride-8 feature pyramid."""
    if width <= 0 or height <= 0:
        raise ValueError("CoWTracker source dimensions must be positive")
    # lcm(14, 8) keeps DINO tokens and ResNet upsample/add feature maps aligned.
    patch = 56
    long_side = max(patch, int(round(max_side / patch)) * patch)
    scale = long_side / max(width, height)
    target_h = max(patch, int(round(height * scale / patch)) * patch)
    target_w = max(patch, int(round(width * scale / patch)) * patch)
    return target_h, target_w


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input source MP4/MOV (first 49 frames at --fps are used).")
    parser.add_argument("--caption", required=True, help="Correct source-video caption; stored in the manifest and used for sampling.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing directories are never overwritten.")
    parser.add_argument("--sequence-name", help="Stable safe ID; defaults to the input stem.")
    parser.add_argument("--backend", choices=("4rc", "cow_vggt"), default="4rc")
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--trajectory", choices=TRAJECTORIES, default="orbit_right")
    parser.add_argument("--angle-deg", type=float, default=30.0)
    parser.add_argument("--distance", type=float, default=0.12)
    parser.add_argument("--orbit-radius", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--conda-command", default="conda")
    parser.add_argument("--holo-env", default=os.environ.get("HOLO4D_ENV"))
    parser.add_argument("--fourrc-env", default=os.environ.get("FOURRC_ENV"))
    parser.add_argument("--cow-env", default=os.environ.get("COWTRACKER_ENV"))
    parser.add_argument("--vggt-env", default=os.environ.get("VGGT_OMEGA_ENV"))
    parser.add_argument("--gmos-env", default=os.environ.get("GMOS_ENV"))

    parser.add_argument("--fourrc-repo", type=Path, default=os.environ.get("FOURRC_REPO"))
    parser.add_argument("--fourrc-checkpoint", default=os.environ.get("FOURRC_CHECKPOINT", "Luo-Yihang/4RC"))
    parser.add_argument("--fourrc-size", type=int, default=512)
    parser.add_argument("--cowtracker-repo", type=Path, default=os.environ.get("COWTRACKER_REPO"))
    parser.add_argument("--cowtracker-checkpoint", default=os.environ.get("COWTRACKER_CHECKPOINT"))
    parser.add_argument("--vggt-repo", type=Path, default=os.environ.get("VGGT_OMEGA_REPO"))
    parser.add_argument("--vggt-checkpoint", type=Path, default=os.environ.get("VGGT_OMEGA_CHECKPOINT"))

    parser.add_argument("--gmos", choices=("none", "gmos", "gmos-s"), default="none")
    parser.add_argument("--gmos-repo", type=Path, default=os.environ.get("GMOS_REPO"))
    parser.add_argument("--pi3-checkpoint", default=os.environ.get("GMOS_PI3_CHECKPOINT"))
    parser.add_argument("--sam2-checkpoint", default=os.environ.get("GMOS_SAM2_CHECKPOINT"))
    parser.add_argument("--gmos-checkpoint", default=os.environ.get("GMOS_CHECKPOINT"))
    parser.add_argument("--fuse-gmos-static-tracks", action="store_true",
                        help="For 4RC only: fuse GMOS foreground labels into the persistent 4RC track condition.")
    parser.add_argument("--sample", action="store_true", help="After preprocessing, sample with inference/sample_examples.py.")
    parser.add_argument("--model", choices=("14b", "5b-commercial", "5b-full"), default="14b")
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--sampling-output", type=Path, help="Optional separate output directory for Holo4D generation.")
    parser.add_argument("--gpu", default="0", help="Single visible GPU index passed to backend tools (use CUDA_VISIBLE_DEVICES to select the physical device).")
    args = parser.parse_args()

    video = require_path(str(args.video), "--video")
    if video.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError("--video must be MP4, MOV, MKV or AVI")
    if not args.caption.strip():
        raise ValueError("--caption must not be empty")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be positive even pixel dimensions")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.fuse_gmos_static_tracks and args.backend != "4rc":
        raise ValueError("--fuse-gmos-static-tracks is valid only for --backend 4rc")
    if args.fuse_gmos_static_tracks and args.gmos == "none":
        raise ValueError("--fuse-gmos-static-tracks requires --gmos gmos or --gmos gmos-s")
    if args.sample and args.checkpoint is None:
        raise ValueError("--sample requires --checkpoint or CHECKPOINT")

    fourrc_repo = None
    cow_repo = None
    vggt_repo = None
    vggt_checkpoint = None
    gmos_repo = None
    pi3_ckpt = None
    sam2_ckpt = None
    gmos_ckpt = None
    if args.backend == "4rc":
        fourrc_repo = require_path(str(args.fourrc_repo) if args.fourrc_repo else None, "--fourrc-repo / FOURRC_REPO")
    else:
        cow_repo = require_path(str(args.cowtracker_repo) if args.cowtracker_repo else None, "--cowtracker-repo / COWTRACKER_REPO")
        vggt_repo = require_path(str(args.vggt_repo) if args.vggt_repo else None, "--vggt-repo / VGGT_OMEGA_REPO")
        vggt_checkpoint = require_path(str(args.vggt_checkpoint) if args.vggt_checkpoint else None, "--vggt-checkpoint / VGGT_OMEGA_CHECKPOINT")
        args.cowtracker_checkpoint = str(require_path(args.cowtracker_checkpoint, "--cowtracker-checkpoint / COWTRACKER_CHECKPOINT"))
    if args.gmos != "none":
        gmos_repo = require_path(str(args.gmos_repo) if args.gmos_repo else None, "--gmos-repo / GMOS_REPO")
        pi3_ckpt = require_path(args.pi3_checkpoint, "--pi3-checkpoint / GMOS_PI3_CHECKPOINT")
        sam2_ckpt = require_path(args.sam2_checkpoint, "--sam2-checkpoint / GMOS_SAM2_CHECKPOINT")
        gmos_ckpt = require_path(args.gmos_checkpoint, "--gmos-checkpoint / GMOS_CHECKPOINT")
    if args.sample:
        args.checkpoint = require_path(str(args.checkpoint), "--checkpoint / CHECKPOINT")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists; choose a new path: {output}")
    output.mkdir(parents=True, exist_ok=False)

    sequence_name = safe_name(args.sequence_name or video.stem)
    input_root = output / "input"
    frame_dir = input_root / sequence_name
    geometry_root = output / ("geometry_4rc" if args.backend == "4rc" else "geometry_vggt_omega")
    cow_root = output / "tracks_cow" if args.backend == "cow_vggt" else None
    gmos_root = output / "masks_gmos" if args.gmos != "none" else None
    condition_dir = output / "condition"
    logs = output / "logs"
    logs.mkdir()
    executed_commands: list[dict] = []

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to decode the raw video")
    stage_video(
        video,
        frame_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        log_path=logs / "stage_video.log",
        command_manifest=executed_commands,
    )
    caption_json = output / "caption.json"
    make_caption_json(caption_json, sequence_name, args.caption)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Some installed cluster environments expose both upstream and xFormers FA3
    # registrations. The validated release launchers disable FA3 before model
    # imports; carry that safe default through every isolated backend process.
    env.setdefault("DISABLE_FLASH_ATTN_3", "1")

    if args.backend == "4rc":
        run_step(
            "4RC direct dense 3D reconstruction/tracking",
            python_command(ROOT / "inference/run_4rc_reconstruction.py", env_name=args.fourrc_env, conda_command=args.conda_command)
            + ["--frames-dir", str(frame_dir), "--output-dir", str(geometry_root / sequence_name),
               "--fourrc-repo", str(fourrc_repo), "--checkpoint", args.fourrc_checkpoint,
               "--size", str(args.fourrc_size), "--query-index", "0", "--dtype", "bf16-mixed", "--device", f"cuda:{args.gpu}"],
            cwd=fourrc_repo,
            env=env,
            log_path=logs / "4rc.log",
            command_manifest=executed_commands,
        )
    else:
        cow_height, cow_width = cowtracker_canvas(args.width, args.height)
        run_step(
            "CoWTracker dense 2D tracking",
            python_command(ROOT / "holo4d/tools/cowtracker_source_track_batch.py", env_name=args.cow_env, conda_command=args.conda_command)
            + ["--input_root", str(input_root), "--output_dir", str(cow_root),
               "--cowtracker_repo", str(cow_repo), "--seqs", sequence_name, "--gpu", args.gpu,
               "--checkpoint", str(args.cowtracker_checkpoint), "--max_frames", str(NUM_FRAMES), "--fps", str(args.fps),
               "--target_height", str(cow_height), "--target_width", str(cow_width)],
            cwd=ROOT,
            env=env,
            log_path=logs / "cowtracker.log",
            command_manifest=executed_commands,
        )
        vggt_sequence_list = output / "vggt_sequences.txt"
        vggt_sequence_list.write_text(sequence_name + "\n")
        run_step(
            "VGGT-Omega source cameras and depth (aspect-preserving 512 canvas)",
            python_command(ROOT / "eval/common/vggt_omega_source_camera_infer.py", env_name=args.vggt_env, conda_command=args.conda_command)
            + ["--dataset_name", "holo4d_raw", "--input_root", str(input_root), "--input_mode", "frame_dirs",
               "--output_root", str(geometry_root), "--sequence_list", str(vggt_sequence_list),
               "--checkpoint", str(vggt_checkpoint), "--vggt_omega_repo", str(vggt_repo), "--gpu", args.gpu,
               "--num_frames", str(NUM_FRAMES), "--image_resolution", "512", "--image_mode", "closest_aspect", "--save_depth"],
            cwd=ROOT,
            env=env,
            log_path=logs / "vggt_omega.log",
            command_manifest=executed_commands,
        )

    gmos_mask_subdir = "mos_frame"
    if args.gmos != "none":
        if args.gmos == "gmos":
            gmos_script = gmos_repo / "inference_gmos.py"
            checkpoint_flag = "--gmos_ckpt"
        else:
            gmos_script = gmos_repo / "inference_gmos_s.py"
            checkpoint_flag = "--gmos_s_ckpt"
            gmos_mask_subdir = "frame"
        run_step(
            f"{args.gmos} foreground/motion masks",
            python_command(gmos_script, env_name=args.gmos_env, conda_command=args.conda_command)
            + ["--pi3_ckpt", str(pi3_ckpt), "--sam2_ckpt", str(sam2_ckpt), checkpoint_flag, str(gmos_ckpt),
               "--input_rgb", str(frame_dir), "--save_dir", str(gmos_root), "--save_mode", "mask",
               "--fps", str(args.fps), "--stride", "1"],
            cwd=gmos_repo,
            env=env,
            log_path=logs / f"{args.gmos}.log",
            command_manifest=executed_commands,
        )

    builder_command = python_command(ROOT / "inference/build_raw_condition.py", env_name=args.holo_env, conda_command=args.conda_command)
    builder_command += [
        "--repo-root", str(ROOT), "--frame-dir", str(frame_dir), "--sequence-name", sequence_name,
        "--backend", args.backend, "--geometry-root", str(geometry_root), "--output-dir", str(condition_dir),
        "--caption", args.caption, "--width", str(args.width), "--height", str(args.height), "--fps", str(args.fps),
        "--trajectory", args.trajectory, "--angle-deg", str(args.angle_deg), "--distance", str(args.distance),
        "--orbit-radius", str(args.orbit_radius), "--seed", str(args.seed),
    ]
    if cow_root is not None:
        builder_command += ["--cow-root", str(cow_root)]
    if gmos_root is not None:
        builder_command += ["--gmos-root", str(gmos_root), "--gmos-mask-subdir", gmos_mask_subdir]
    if args.fuse_gmos_static_tracks:
        builder_command.append("--fuse-gmos-static-tracks")
    run_step(
        "compose Holo4D dense condition bundle and validate its contract",
        builder_command,
        cwd=ROOT,
        env=env,
        log_path=logs / "build_condition.log",
        command_manifest=executed_commands,
    )

    run_manifest = {
        "input_video": str(video),
        "input_video_sha256": sha256(video),
        "caption": args.caption,
        "sequence_name": sequence_name,
        "backend": args.backend,
        "resolution_hw": [args.height, args.width],
        "fps": args.fps,
        "frames": NUM_FRAMES,
        "trajectory": {"name": args.trajectory, "angle_deg": args.angle_deg, "distance": args.distance, "orbit_radius": args.orbit_radius},
        "gmos": args.gmos,
        "gmos_static_track_fusion": bool(args.fuse_gmos_static_tracks),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "backend_repositories": {
            "fourrc": {"path": str(fourrc_repo) if fourrc_repo else None, "git_revision": git_revision(fourrc_repo)},
            "cowtracker": {"path": str(cow_repo) if cow_repo else None, "git_revision": git_revision(cow_repo)},
            "vggt_omega": {"path": str(vggt_repo) if vggt_repo else None, "git_revision": git_revision(vggt_repo)},
            "gmos": {"path": str(gmos_repo) if gmos_repo else None, "git_revision": git_revision(gmos_repo)},
        },
        "checkpoints": {
            "fourrc": asset_fingerprint(args.fourrc_checkpoint) if args.backend == "4rc" else None,
            "cowtracker": asset_fingerprint(args.cowtracker_checkpoint) if args.backend == "cow_vggt" else None,
            "vggt_omega": asset_fingerprint(vggt_checkpoint) if vggt_checkpoint else None,
            "pi3": asset_fingerprint(pi3_ckpt) if pi3_ckpt else None,
            "sam2": asset_fingerprint(sam2_ckpt) if sam2_ckpt else None,
            "gmos": asset_fingerprint(gmos_ckpt) if gmos_ckpt else None,
            "holo4d": asset_fingerprint(args.checkpoint) if args.sample else None,
        },
        "commands_logged_under": str(logs),
        "outputs": {"frames": str(frame_dir), "geometry": str(geometry_root), "cow_tracks": str(cow_root) if cow_root else None,
                    "gmos_masks": str(gmos_root) if gmos_root else None, "condition": str(condition_dir / "condition.npz"),
                    "condition_manifest": str(condition_dir / "manifest.json")},
        "commands": executed_commands,
        "environments": {
            "4rc_env": args.fourrc_env if args.backend == "4rc" else None,
            "cow_env": args.cow_env if args.backend == "cow_vggt" else None,
            "vggt_env": args.vggt_env if args.backend == "cow_vggt" else None,
            "gmos_env": args.gmos_env if args.gmos != "none" else None,
            "holo_env": args.holo_env,
        },
    }
    (output / "run.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    if args.sample:
        sampling_output = (args.sampling_output or output / "sampling").expanduser().resolve()
        if sampling_output.exists() and any(sampling_output.iterdir()):
            raise FileExistsError(f"sampling output is not empty; choose a new path: {sampling_output}")
        sampling_output.mkdir(parents=True, exist_ok=True)
        sample_env = env.copy()
        sample_env["CHECKPOINT"] = str(args.checkpoint)
        run_step(
            "Holo4D diffusion sampling from the generated condition",
            python_command(ROOT / "inference/sample_examples.py", env_name=args.holo_env, conda_command=args.conda_command)
            + ["--manifest", str(condition_dir / "manifest.json"), "--example", sequence_name,
               "--model", args.model, "--output-dir", str(sampling_output)],
            cwd=ROOT,
            env=sample_env,
            log_path=logs / "holo4d_sampling.log",
            command_manifest=executed_commands,
        )
        run_manifest["outputs"]["sampling"] = str(sampling_output)
        (output / "run.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    print(f"\n[raw-preprocess] complete. Portable condition: {condition_dir / 'condition.npz'}", flush=True)
    print(f"[raw-preprocess] condition manifest: {condition_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
