#!/usr/bin/env python3
"""VGGT-Omega camera-trajectory evaluation for DAVIS-50 x 2-traj.

This mirrors ``vipe_camera_eval.py`` but swaps the pose-estimation backend:

  pred mp4 -> standardized mp4 -> extracted PNG frames -> VGGT-Omega
  -> predicted c2w sequence -> same evo metrics as ViPE.

The output JSON intentionally matches the ViPE schema so downstream tooling can
compare estimator backends with the same aggregation and per-pair analysis code.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vipe_camera_eval import (
    compute_pose_metrics,
    load_gt_c2w,
    resolve_pred_mp4,
    standardize_pred_mp4,
)


def _ensure_vggt_omega_import(vggt_omega_repo: Path) -> None:
    if str(vggt_omega_repo) not in sys.path:
        sys.path.insert(0, str(vggt_omega_repo))


def _extract_frames(mp4_path: Path, frames_dir: Path, num_frames: int) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("frame_*.png"))
    if len(existing) >= num_frames and all(p.stat().st_mtime >= mp4_path.stat().st_mtime for p in existing[:num_frames]):
        return existing[:num_frames]

    for p in frames_dir.glob("frame_*.png"):
        p.unlink()

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open standardized mp4: {mp4_path}")

    paths: List[Path] = []
    idx = 0
    while idx < num_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out = frames_dir / f"frame_{idx:05d}.png"
        if not cv2.imwrite(str(out), frame_bgr):
            raise RuntimeError(f"failed to write extracted frame: {out}")
        paths.append(out)
        idx += 1
    cap.release()

    if not paths:
        raise RuntimeError(f"no frames extracted from {mp4_path}")

    # Keep the metric timeline length stable. VGGT can run shorter clips, but
    # padding here matches ViPE's c2w padding semantics.
    while len(paths) < num_frames:
        src = paths[-1]
        dst = frames_dir / f"frame_{len(paths):05d}.png"
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None or not cv2.imwrite(str(dst), img):
            raise RuntimeError(f"failed to pad frame {dst} from {src}")
        paths.append(dst)

    return paths


def _load_model(
    checkpoint_path: Path,
    repo: Path,
    device: torch.device,
    enable_alignment: bool,
    enable_depth: bool,
):
    import torch

    _ensure_vggt_omega_import(repo)
    from vggt_omega.models import VGGTOmega

    # Match the official VGGT-Omega inference contract by default:
    # `VGGTOmega()` => enable_depth=True, enable_alignment=False.
    # `--no_depth` exists only as an emergency low-memory fallback and uses a
    # non-strict load because the released checkpoint contains dense_head.*.
    print(
        "[vggt-omega] instantiate VGGTOmega("
        f"enable_depth={enable_depth}, enable_alignment={enable_alignment})"
    )
    model = VGGTOmega(enable_depth=enable_depth, enable_alignment=enable_alignment).to(device).eval()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=enable_depth)
    if missing or unexpected:
        print(f"[vggt-omega] checkpoint load: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[vggt-omega] missing head: {missing[:8]}")
        if unexpected:
            print(f"[vggt-omega] unexpected head: {unexpected[:8]}")
    return model


def run_vggt_omega_inference(
    pred_mp4: Path,
    out_dir: Path,
    model,
    vggt_omega_repo: Path,
    device: torch.device,
    num_frames: int,
    target_w: int,
    target_h: int,
    image_resolution: int,
    image_mode: str,
    force: bool,
) -> Tuple[Path, bool]:
    """Run VGGT-Omega and cache c2w under ``out_dir/pose/vggt_omega.npz``."""
    import torch

    _ensure_vggt_omega_import(vggt_omega_repo)
    from vggt_omega.utils.geometry import closed_form_inverse_se3
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    out_dir.mkdir(parents=True, exist_ok=True)
    std_mp4 = out_dir / f"_input_{target_h}x{target_w}x{num_frames}.mp4"
    standardize_pred_mp4(pred_mp4, std_mp4, target_w, target_h, num_frames)

    pose_dir = out_dir / "pose"
    pose_npz = pose_dir / "vggt_omega.npz"
    if (
        not force
        and pose_npz.is_file()
        and pose_npz.stat().st_mtime >= std_mp4.stat().st_mtime
    ):
        return pose_npz, True

    frame_paths = _extract_frames(std_mp4, out_dir / "frames", num_frames)
    images = load_and_preprocess_images(
        [str(p) for p in frame_paths],
        mode=image_mode,
        image_resolution=image_resolution,
    ).to(device)

    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

    # VGGT-Omega extrinsics are camera-from-world OpenCV 3x4 matrices.
    # Convert to world-from-camera / c2w 4x4 for the existing metric code.
    extr = extrinsics[0] if extrinsics.ndim == 4 else extrinsics
    intr_out = intrinsics[0] if intrinsics is not None and intrinsics.ndim == 4 else intrinsics
    extr = extr.detach().float().cpu()
    c2w = closed_form_inverse_se3(extr).detach().cpu().numpy().astype(np.float32)
    intr = intr_out.detach().float().cpu().numpy().astype(np.float32) if intr_out is not None else None

    pose_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pose_npz,
        data=c2w,
        cam_c2w=c2w,
        extrinsics_c2w=c2w,
        extrinsics_w2c=extr.numpy().astype(np.float32),
        intrinsics=intr if intr is not None else np.empty((0,), dtype=np.float32),
        image_resolution=np.asarray(image_resolution, dtype=np.int32),
        image_mode=np.asarray(image_mode),
    )
    return pose_npz, False


def _read_pairs(pairs_csv: Path, limit_pairs: Optional[int], num_shards: int, shard_index: int) -> List[Tuple[str, str]]:
    with open(pairs_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    pairs: List[Tuple[str, str]] = [(r["video"], r["trajectory"]) for r in rows]
    if limit_pairs:
        pairs = pairs[:limit_pairs]
    if num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"--shard_index must be in [0, {num_shards}), got {shard_index}")
    if num_shards > 1:
        pairs = [pair for i, pair in enumerate(pairs) if i % num_shards == shard_index]
    return pairs


def _stats(rows: list, key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None, "median": None, "n": 0}
    v = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "median": float(np.median(v)),
        "n": int(len(vals)),
    }


def _empty_summary(args, pairs_count: int) -> dict:
    return {
        "method_pred_root": str(args.method_pred_root),
        "num_pairs_evaluated": 0,
        "num_pairs_total": pairs_count,
        "pose_estimator": "vggt_omega",
        "standardize_input": {
            "target_w": args.target_w,
            "target_h": args.target_h,
            "num_frames": args.num_frames,
            "note": "ffmpeg scale=W:H stretch + truncate first N frames",
        },
        "vggt_omega": {
            "repo": str(args.vggt_omega_repo),
            "checkpoint": str(args.checkpoint),
            "image_resolution": args.image_resolution,
            "image_mode": args.image_mode,
            "enable_depth": not args.no_depth,
            "enable_alignment": args.enable_alignment,
        },
        "ate": {"mean": None, "std": None, "median": None, "n": 0},
        "trans_err": {"mean": None, "std": None, "median": None, "n": 0},
        "rot_err": {"mean": None, "std": None, "median": None, "n": 0},
        "sim3_subset": {
            "ate": {"mean": None, "std": None, "median": None, "n": 0},
            "trans_err": {"mean": None, "std": None, "median": None, "n": 0},
            "rot_err": {"mean": None, "std": None, "median": None, "n": 0},
        },
        "origin_subset": {
            "ate": {"mean": None, "std": None, "median": None, "n": 0},
            "trans_err": {"mean": None, "std": None, "median": None, "n": 0},
            "rot_err": {"mean": None, "std": None, "median": None, "n": 0},
        },
        "all_norm": {
            "ate": {"mean": None, "std": None, "median": None, "n": 0},
            "trans_err": {"mean": None, "std": None, "median": None, "n": 0},
            "rot_err": {"mean": None, "std": None, "median": None, "n": 0},
        },
        "per_pair": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method_pred_root", type=Path, required=True)
    ap.add_argument("--pred_filenames", default="pred.mp4,{traj}.mp4,gen.mp4,output.mp4,render.mp4,color_{traj}.mp4")
    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--cameras_root", type=Path, required=True)
    ap.add_argument("--vggt_cache_root", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vggt_omega_repo", type=Path, default=Path(os.environ.get("VGGT_OMEGA_REPO", "external/vggt-omega")))
    ap.add_argument("--image_resolution", type=int, default=512)
    ap.add_argument("--image_mode", choices=["balanced", "max_size"], default="balanced")
    ap.add_argument("--enable_alignment", action="store_true")
    ap.add_argument(
        "--no_depth",
        action="store_true",
        help="Low-memory fallback only; default matches official VGGTOmega() with enable_depth=True.",
    )
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--target_w", type=int, default=512)
    ap.add_argument("--target_h", type=int, default=288)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit_pairs", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--phase", choices=["all", "vggt", "metrics"], default="all")
    ap.add_argument("--force_pose", action="store_true")
    args = ap.parse_args()

    pred_filenames = [s.strip() for s in args.pred_filenames.split(",") if s.strip()]
    pairs = _read_pairs(args.pairs_csv, args.limit_pairs, args.num_shards, args.shard_index)
    print(f"[vggt-omega-cam] {len(pairs)} pairs from {args.pairs_csv}")

    if not pairs:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(_empty_summary(args, 0), indent=2))
        return

    model = None
    device = None
    if args.phase in ("all", "vggt"):
        # --gpu remains a logical integer; an interactive launcher may pin
        # physical visibility by UUID separately. Do not discard that pin.
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "EVAL_CUDA_VISIBLE_DEVICES_OVERRIDE", str(args.gpu)
        ) or str(args.gpu)
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("VGGT-Omega camera eval requires CUDA")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {args.checkpoint}")
        model = _load_model(
            args.checkpoint,
            args.vggt_omega_repo,
            device,
            args.enable_alignment,
            enable_depth=not args.no_depth,
        )

        num_pose_success = 0
        num_pose_failures = 0
        num_missing_predictions = 0
        for i, (seq, traj) in enumerate(pairs):
            pred_mp4 = resolve_pred_mp4(args.method_pred_root, seq, traj, pred_filenames)
            if pred_mp4 is None:
                num_missing_predictions += 1
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (pred mp4 missing)")
                continue
            out_dir = args.vggt_cache_root / f"{seq}__{traj}"
            try:
                pose_npz, was_cached = run_vggt_omega_inference(
                    pred_mp4=pred_mp4,
                    out_dir=out_dir,
                    model=model,
                    vggt_omega_repo=args.vggt_omega_repo,
                    device=device,
                    num_frames=args.num_frames,
                    target_w=args.target_w,
                    target_h=args.target_h,
                    image_resolution=args.image_resolution,
                    image_mode=args.image_mode,
                    force=args.force_pose,
                )
                num_pose_success += 1
                tag = "CACHED" if was_cached else "FRESH "
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: {tag} -> {pose_npz}")
            except Exception as exc:
                num_pose_failures += 1
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: FAIL ({exc})")
                if "Could not open libnvrtc" in str(exc):
                    raise RuntimeError(
                        "VGGT-Omega CUDA runtime cannot load NVRTC; expose the "
                        "matching libnvrtc directory through LD_LIBRARY_PATH"
                    ) from exc

        if num_pose_success == 0:
            raise RuntimeError(
                "VGGT-Omega inference produced zero poses "
                f"for {len(pairs)} requested pairs "
                f"({num_pose_failures} inference failures, "
                f"{num_missing_predictions} missing predictions)"
            )

    if args.phase == "vggt":
        return

    per_pair: List[dict] = []
    for i, (seq, traj) in enumerate(pairs):
        cam_npz = args.cameras_root / seq / f"{traj}.npz"
        if not cam_npz.is_file():
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (gt camera npz missing)")
            continue
        pose_npz = args.vggt_cache_root / f"{seq}__{traj}" / "pose" / "vggt_omega.npz"
        if not pose_npz.is_file():
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (VGGT-Omega pose missing)")
            continue
        try:
            pred_c2w = np.load(pose_npz)["data"][: args.num_frames]
            gt_c2w = load_gt_c2w(cam_npz, args.num_frames)
            metrics = compute_pose_metrics(pred_c2w, gt_c2w)
            per_pair.append({"seq": seq, "traj": traj, **metrics})
            print(
                f"  [{i+1}/{len(pairs)}] {seq}/{traj}: "
                f"ate={metrics['ate']:.4f}  trans={metrics['trans_err']:.4f}  "
                f"rot={metrics['rot_err']:.3f}"
            )
        except Exception as exc:
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: METRIC FAIL ({exc})")

    if not per_pair:
        print("[vggt-omega-cam] no successful pairs; writing empty summary")
        summary = _empty_summary(args, len(pairs))
    else:
        sim3 = [r for r in per_pair if r.get("alignment_mode") == "umeyama_align_scale"]
        origin = [r for r in per_pair if r.get("alignment_mode") == "origin_align_only"]
        has_gt_trans = [r for r in per_pair if r.get("ate_norm") is not None]
        summary = {
            "method_pred_root": str(args.method_pred_root),
            "num_pairs_evaluated": len(per_pair),
            "num_pairs_total": len(pairs),
            "pose_estimator": "vggt_omega",
            "standardize_input": {
                "target_w": args.target_w,
                "target_h": args.target_h,
                "num_frames": args.num_frames,
                "note": "ffmpeg scale=W:H stretch + truncate first N frames",
            },
            "vggt_omega": {
                "repo": str(args.vggt_omega_repo),
                "checkpoint": str(args.checkpoint),
                "image_resolution": args.image_resolution,
                "image_mode": args.image_mode,
                "enable_depth": not args.no_depth,
                "enable_alignment": args.enable_alignment,
            },
            "ate": _stats(per_pair, "ate"),
            "trans_err": _stats(per_pair, "trans_err"),
            "rot_err": _stats(per_pair, "rot_err"),
            "sim3_subset": {
                "ate": _stats(sim3, "ate"),
                "trans_err": _stats(sim3, "trans_err"),
                "rot_err": _stats(sim3, "rot_err"),
            },
            "origin_subset": {
                "ate": _stats(origin, "ate"),
                "trans_err": _stats(origin, "trans_err"),
                "rot_err": _stats(origin, "rot_err"),
            },
            "all_norm": {
                "ate": _stats(has_gt_trans, "ate_norm"),
                "trans_err": _stats(has_gt_trans, "trans_err_norm"),
                "rot_err": _stats(per_pair, "rot_err"),
            },
            "per_pair": per_pair,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(f"[vggt-omega-cam] {args.output_json}")


if __name__ == "__main__":
    main()
