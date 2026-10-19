#!/usr/bin/env python3
"""Depth Anything 3 camera-trajectory evaluation for DAVIS-50 x 2-traj.

This mirrors ``vggt_omega_camera_eval.py`` but uses DA3 as the pose estimator:

  pred mp4 -> standardized mp4 -> extracted PNG frames -> Depth Anything 3
  -> predicted c2w sequence -> same evo metrics as ViPE/VGGT-Omega.

DA3 returns OpenCV/COLMAP-style camera-from-world extrinsics. We invert them
to c2w before calling the shared metric code.
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


def _ensure_da3_import(da3_repo: Path) -> None:
    for p in (da3_repo / "src", da3_repo):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


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
    while len(paths) < num_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out = frames_dir / f"frame_{len(paths):05d}.png"
        if not cv2.imwrite(str(out), frame_bgr):
            raise RuntimeError(f"failed to write extracted frame: {out}")
        paths.append(out)
    cap.release()

    if not paths:
        raise RuntimeError(f"no frames extracted from {mp4_path}")

    while len(paths) < num_frames:
        src = paths[-1]
        dst = frames_dir / f"frame_{len(paths):05d}.png"
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None or not cv2.imwrite(str(dst), img):
            raise RuntimeError(f"failed to pad frame {dst} from {src}")
        paths.append(dst)
    return paths


def _load_model(model_name: str, da3_repo: Path, device: str):
    _ensure_da3_import(da3_repo)
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device).eval()
    return model


def _as_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsics, dtype=np.float32)
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] == (3, 4):
        bottom = np.zeros(ext.shape[:-2] + (1, 4), dtype=ext.dtype)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([ext, bottom], axis=-2)
    raise ValueError(f"unexpected DA3 extrinsics shape: {ext.shape}")


def run_da3_inference(
    pred_mp4: Path,
    out_dir: Path,
    model,
    num_frames: int,
    target_w: int,
    target_h: int,
    process_res: int,
    process_res_method: str,
    force: bool,
) -> Tuple[Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    std_mp4 = out_dir / f"_input_{target_h}x{target_w}x{num_frames}.mp4"
    standardize_pred_mp4(pred_mp4, std_mp4, target_w, target_h, num_frames)

    pose_dir = out_dir / "pose"
    pose_npz = pose_dir / "da3.npz"
    if not force and pose_npz.is_file() and pose_npz.stat().st_mtime >= std_mp4.stat().st_mtime:
        return pose_npz, True

    frame_paths = _extract_frames(std_mp4, out_dir / "frames", num_frames)
    prediction = model.inference(
        [str(p) for p in frame_paths],
        process_res=process_res,
        process_res_method=process_res_method,
    )
    if prediction.extrinsics is None:
        raise RuntimeError("DA3 prediction did not include extrinsics")

    w2c = _as_homogeneous(prediction.extrinsics)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    intrinsics = (
        np.asarray(prediction.intrinsics, dtype=np.float32)
        if prediction.intrinsics is not None
        else np.empty((0,), dtype=np.float32)
    )

    pose_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pose_npz,
        data=c2w,
        cam_c2w=c2w,
        extrinsics_c2w=c2w,
        extrinsics_w2c=w2c.astype(np.float32),
        intrinsics=intrinsics,
        depth=np.asarray(prediction.depth, dtype=np.float32),
        conf=np.asarray(prediction.conf, dtype=np.float32) if prediction.conf is not None else np.empty((0,), dtype=np.float32),
        process_res=np.asarray(process_res, dtype=np.int32),
        process_res_method=np.asarray(process_res_method),
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
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)), "n": int(len(vals))}


def _empty_summary(args, pairs_count: int) -> dict:
    return {
        "method_pred_root": str(args.method_pred_root),
        "num_pairs_evaluated": 0,
        "num_pairs_total": pairs_count,
        "pose_estimator": "da3",
        "standardize_input": {
            "target_w": args.target_w,
            "target_h": args.target_h,
            "num_frames": args.num_frames,
            "note": "ffmpeg scale=W:H stretch + truncate first N frames",
        },
        "da3": {
            "repo": str(args.da3_repo),
            "model_name": args.model_name,
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
        },
        "ate": {"mean": None, "std": None, "median": None, "n": 0},
        "trans_err": {"mean": None, "std": None, "median": None, "n": 0},
        "rot_err": {"mean": None, "std": None, "median": None, "n": 0},
        "sim3_subset": {"ate": {"mean": None, "std": None, "median": None, "n": 0}, "trans_err": {"mean": None, "std": None, "median": None, "n": 0}, "rot_err": {"mean": None, "std": None, "median": None, "n": 0}},
        "origin_subset": {"ate": {"mean": None, "std": None, "median": None, "n": 0}, "trans_err": {"mean": None, "std": None, "median": None, "n": 0}, "rot_err": {"mean": None, "std": None, "median": None, "n": 0}},
        "all_norm": {"ate": {"mean": None, "std": None, "median": None, "n": 0}, "trans_err": {"mean": None, "std": None, "median": None, "n": 0}, "rot_err": {"mean": None, "std": None, "median": None, "n": 0}},
        "per_pair": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method_pred_root", type=Path, required=True)
    ap.add_argument("--pred_filenames", default="pred.mp4,{traj}.mp4,gen.mp4,output.mp4,render.mp4,color_{traj}.mp4")
    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--cameras_root", type=Path, required=True)
    ap.add_argument("--da3_cache_root", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--da3_repo", type=Path, default=Path(os.environ.get("DA3_REPO", "external/Depth-Anything-3")))
    ap.add_argument("--model_name", default="depth-anything/DA3-GIANT-1.1")
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--process_res_method", default="upper_bound_resize")
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--target_w", type=int, default=512)
    ap.add_argument("--target_h", type=int, default=288)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit_pairs", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--phase", choices=["all", "da3", "metrics"], default="all")
    ap.add_argument("--force_pose", action="store_true")
    args = ap.parse_args()

    pred_filenames = [s.strip() for s in args.pred_filenames.split(",") if s.strip()]
    pairs = _read_pairs(args.pairs_csv, args.limit_pairs, args.num_shards, args.shard_index)
    print(f"[da3-cam] {len(pairs)} pairs from {args.pairs_csv}")

    if not pairs:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(_empty_summary(args, 0), indent=2))
        return

    if args.phase in ("all", "da3"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        model = _load_model(args.model_name, args.da3_repo, "cuda")
        for i, (seq, traj) in enumerate(pairs):
            pred_mp4 = resolve_pred_mp4(args.method_pred_root, seq, traj, pred_filenames)
            if pred_mp4 is None:
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (pred mp4 missing)")
                continue
            out_dir = args.da3_cache_root / f"{seq}__{traj}"
            try:
                pose_npz, was_cached = run_da3_inference(
                    pred_mp4=pred_mp4,
                    out_dir=out_dir,
                    model=model,
                    num_frames=args.num_frames,
                    target_w=args.target_w,
                    target_h=args.target_h,
                    process_res=args.process_res,
                    process_res_method=args.process_res_method,
                    force=args.force_pose,
                )
                tag = "CACHED" if was_cached else "FRESH "
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: {tag} -> {pose_npz}")
            except Exception as exc:
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: FAIL ({exc})")

    if args.phase == "da3":
        return

    per_pair: List[dict] = []
    for i, (seq, traj) in enumerate(pairs):
        cam_npz = args.cameras_root / seq / f"{traj}.npz"
        pose_npz = args.da3_cache_root / f"{seq}__{traj}" / "pose" / "da3.npz"
        if not cam_npz.is_file():
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (gt camera npz missing)")
            continue
        if not pose_npz.is_file():
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (DA3 pose missing)")
            continue
        try:
            pred_c2w = np.load(pose_npz)["data"][: args.num_frames]
            gt_c2w = load_gt_c2w(cam_npz, args.num_frames)
            metrics = compute_pose_metrics(pred_c2w, gt_c2w)
            per_pair.append({"seq": seq, "traj": traj, **metrics})
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: ate={metrics['ate']:.4f} trans={metrics['trans_err']:.4f} rot={metrics['rot_err']:.3f}")
        except Exception as exc:
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: METRIC FAIL ({exc})")

    if not per_pair:
        summary = _empty_summary(args, len(pairs))
    else:
        sim3 = [r for r in per_pair if r.get("alignment_mode") == "umeyama_align_scale"]
        origin = [r for r in per_pair if r.get("alignment_mode") == "origin_align_only"]
        has_gt_trans = [r for r in per_pair if r.get("ate_norm") is not None]
        summary = {
            "method_pred_root": str(args.method_pred_root),
            "num_pairs_evaluated": len(per_pair),
            "num_pairs_total": len(pairs),
            "pose_estimator": "da3",
            "standardize_input": {"target_w": args.target_w, "target_h": args.target_h, "num_frames": args.num_frames, "note": "ffmpeg scale=W:H stretch + truncate first N frames"},
            "da3": {"repo": str(args.da3_repo), "model_name": args.model_name, "process_res": args.process_res, "process_res_method": args.process_res_method},
            "ate": _stats(per_pair, "ate"),
            "trans_err": _stats(per_pair, "trans_err"),
            "rot_err": _stats(per_pair, "rot_err"),
            "sim3_subset": {"ate": _stats(sim3, "ate"), "trans_err": _stats(sim3, "trans_err"), "rot_err": _stats(sim3, "rot_err")},
            "origin_subset": {"ate": _stats(origin, "ate"), "trans_err": _stats(origin, "trans_err"), "rot_err": _stats(origin, "rot_err")},
            "all_norm": {"ate": _stats(has_gt_trans, "ate_norm"), "trans_err": _stats(has_gt_trans, "trans_err_norm"), "rot_err": _stats(per_pair, "rot_err")},
            "per_pair": per_pair,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(f"[da3-cam] {args.output_json}")


if __name__ == "__main__":
    main()
