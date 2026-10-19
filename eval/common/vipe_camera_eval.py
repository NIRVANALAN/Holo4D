"""
ViPE-based camera-trajectory evaluation for the DAVIS-50 × 2-trajectory benchmark.

Faster alternative to the BA-Track pipeline (`eval_batrack_camera.py`):
  ViPE  ≈ 30 s/video on H100 (single-pass feed-forward)
  BA-Track ≈ 5 min/video (Depth-Anything + UniDepth + sparse BA + dense BA)
  → 100 pairs: ~50 min (ViPE) vs ~8 hr (BA-Track), ~10× faster.

For each (seq, traj) pair the tool:
  1. Resolves the pred mp4 from a method tree (e.g. ${BR}/<method>/<seq>/<traj>/<pred>.mp4).
  2. Runs `vipe infer <pred_mp4> --output <vipe_cache>/<seq>__<traj>` if not cached.
  3. Loads the recovered c2w sequence from `<vipe_cache>/<seq>__<traj>/pose/<base>.npz`.
  4. Loads the GT target c2w from `cameras_2traj/<seq>/<traj>.npz` (key: `cam_c2w`).
  5. Computes ATE + RPE-translation + RPE-rotation via evo with Sim(3) (umeyama)
     alignment, mirroring `eval_batrack_camera.eval_metrics()` exactly.

Camera convention (verified):
  - ViPE pose npz `data: (T, 4, 4) float32` — c2w with frame 0 ≈ identity (camera-0
    world frame, OpenCV-style: +Z forward, -Y up). Confirmed by inspecting
    `assets/vipe/bear/pose/bear_first49.npz` (frame 0 norm-difference from I < 1e-3).
  - `cameras_2traj/<seq>/<traj>.npz` `cam_c2w[T,4,4]` — c2w with frame 0 == identity
    in 4RC world frame (same OpenCV convention as ViPE's source-recovered poses).
  - evo's `align=True correct_scale=True` does Sim(3) alignment which absorbs any
    residual frame-0 numerical drift + arbitrary world-frame rotation/scale, so the
    metrics measure relative-trajectory shape only.

Usage (per-method):
  python vipe_camera_eval.py \
      --method_pred_root ${BR}/TrajectoryCrafter_stage2 \
      --pred_filenames "gen.mp4,pred.mp4,{traj}.mp4" \
      --pairs_csv ${BR}/davis_trajectories.csv \
      --cameras_root ${BR}/cameras_2traj \
      --vipe_cache_root ${BR}/eval/_vipe/trajcrafter_stage2 \
      --output_json ${BR}/eval/results/trajcrafter_stage2/eval_camera_metrics_vipe.json \
      --vipe_conda_env vipe \
      --gpu 0

Or use the bash wrapper `eval_camera_metrics_vipe.sh` which handles env switches.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# -------------------------------------------------------------------------
# Step 1 — pair resolution + ViPE inference
# -------------------------------------------------------------------------

def resolve_pred_mp4(method_pred_root: Path, seq: str, traj: str,
                     pred_filename_templates: List[str]) -> Optional[Path]:
    """First existing candidate under <method_pred_root>/<seq>/<traj>/."""
    pair_dir = method_pred_root / seq / traj
    if not pair_dir.is_dir():
        # Some methods (e.g. recammaster) put pred under a `predictions/` subdir.
        pair_dir = method_pred_root / "predictions" / seq / traj
        if not pair_dir.is_dir():
            return None
    for tpl in pred_filename_templates:
        cand = pair_dir / tpl.format(seq=seq, traj=traj)
        if cand.is_file():
            return cand
    return None


def standardize_pred_mp4(src: Path, dst: Path,
                         target_w: int, target_h: int, num_frames: int) -> Path:
    """ffmpeg-resize and truncate <src> -> <dst> (W=target_w, H=target_h, T<=num_frames).

    Different baselines emit at different (W, H, T, fps): ours_1p3b/14b 512×288×49@12,
    ex4d 512×512×49@15, recammaster 832×480×81@12, trajcrafter 672×384×49@10.
    To keep ViPE's pose recovery comparable across methods we force-stretch every
    pred mp4 to the canonical (W=target_w, H=target_h) and TRUNCATE to first
    num_frames frames before feeding ViPE.

    NOTE on stretching: ffmpeg `scale=W:H` stretches without preserving aspect.
    ViPE re-estimates intrinsics per video, and the Sim(3)-aligned ATE absorbs the
    residual scale; the metric is robust to mild horizontal/vertical squash.
    NOTE on truncation: matches the user-spec `truncate later frames if there`.
    For methods like recammaster that emit 81 frames, taking the first 49 assumes
    those are the first 49 of the 49-frame GT trajectory. If recammaster instead
    re-samples the 49-frame target to 81 frames, the 49 frames here are NOT the
    full GT trajectory — flag with --frame_subsample uniform.

    Idempotent: if dst already exists and is newer than src, return dst.
    """
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"scale={target_w}:{target_h}:flags=bicubic",
        "-frames:v", str(num_frames),
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg standardize failed for {src} -> {dst}\n"
            f"stderr: {proc.stderr[:500]}"
        )
    return dst


def run_vipe_inference(pred_mp4: Path, vipe_out_dir: Path,
                       vipe_python: str, vipe_log: Path, num_frames: int,
                       target_w: int = 512, target_h: int = 288) -> Tuple[Path, bool]:
    """Standardize <pred_mp4> to <target_w>×<target_h>×<num_frames>, then run
    `vipe infer <std_mp4> --output <vipe_out_dir>` and return (pose_npz_path, was_cached).

    Cache key: if the cached pose npz exists AND its mtime >= the standardized mp4's
    mtime, skip ViPE and return (path, True). Otherwise wipe stale pose npz, run ViPE,
    return (path, False). This invalidates pre-standardization caches automatically.
    """
    vipe_out_dir.mkdir(parents=True, exist_ok=True)
    std_mp4 = vipe_out_dir / f"_input_{target_h}x{target_w}x{num_frames}.mp4"
    standardize_pred_mp4(pred_mp4, std_mp4, target_w, target_h, num_frames)

    pose_dir = vipe_out_dir / "pose"
    existing = sorted(pose_dir.glob("*.npz"))
    if existing and existing[0].stat().st_mtime >= std_mp4.stat().st_mtime:
        return existing[0], True   # cache hit — ViPE not run
    # Stale or missing — wipe pose dir before re-running ViPE on the std mp4
    if pose_dir.exists():
        for p in pose_dir.glob("*.npz"):
            p.unlink()

    vipe_log.parent.mkdir(parents=True, exist_ok=True)
    # vipe CLI lives at <env>/bin/vipe
    vipe_bin = Path(vipe_python).parent / "vipe"
    cmd = [str(vipe_bin), "infer", str(std_mp4), "--output", str(vipe_out_dir)]
    with open(vipe_log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"vipe infer failed for {std_mp4} (exit {proc.returncode}); see {vipe_log}")
    pose_files = sorted(pose_dir.glob("*.npz"))
    if not pose_files:
        raise RuntimeError(f"vipe ran but produced no pose npz under {pose_dir}; see {vipe_log}")
    return pose_files[0], False    # ViPE actually ran


# -------------------------------------------------------------------------
# Step 2 — c2w → trajectory + Sim(3)-aligned ATE/RPE (mirrors eval_batrack_camera)
# -------------------------------------------------------------------------

def load_vipe_c2w(pose_npz: Path, num_frames: int) -> np.ndarray:
    """Returns (T, 4, 4) c2w, T = min(num_frames, available)."""
    raw = np.load(pose_npz, allow_pickle=True)
    if "data" in raw.files:
        c2w = np.asarray(raw["data"], dtype=np.float64)
    else:
        # fallback: some ViPE builds use a different key
        c2w = np.asarray(raw[raw.files[0]], dtype=np.float64)
    c2w = c2w[:num_frames]
    if c2w.shape[0] < num_frames:
        # Pad with last pose so trajectory has the expected length
        pad = np.repeat(c2w[-1:], num_frames - c2w.shape[0], axis=0)
        c2w = np.concatenate([c2w, pad], axis=0)
    return c2w


def load_gt_c2w(cameras_npz: Path, num_frames: int) -> np.ndarray:
    """Returns (T, 4, 4) c2w from cameras_2traj/<seq>/<traj>.npz."""
    d = np.load(cameras_npz)
    if "cam_c2w" in d.files:
        c2w = np.asarray(d["cam_c2w"], dtype=np.float64)
    elif "c2w" in d.files:
        c2w = np.asarray(d["c2w"], dtype=np.float64)
    else:
        raise ValueError(f"no cam_c2w/c2w key in {cameras_npz} (keys: {d.files})")
    return c2w[:num_frames]


def relative_c2w(c2w_mats: np.ndarray) -> np.ndarray:
    inv0 = np.linalg.inv(c2w_mats[0])
    return np.stack([inv0 @ c for c in c2w_mats], axis=0)


def c2w_to_tum(c2w_mats: np.ndarray) -> np.ndarray:
    """Convert (T, 4, 4) c2w to TUM-style (T, 7) [x, y, z, qw, qx, qy, qz]."""
    from scipy.spatial.transform import Rotation
    out = []
    for c2w in c2w_mats:
        xyz = c2w[:3, 3]
        xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
        out.append(np.concatenate([xyz, wxyz], axis=0))
    return np.stack(out, axis=0)


def compute_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict:
    """Sim(3)-aligned ATE + RPE-trans + RPE-rot, mirroring eval_batrack_camera.eval_metrics()."""
    from evo.core import sync
    from evo.core.geometry import GeometryException
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D
    # evo.main_{ape,rpe} expose functions via `from <module> import <fn>`.
    from evo.main_ape import ape as main_ape_fn
    from evo.main_rpe import rpe as main_rpe_fn

    # Re-anchor both to frame 0 = identity, then convert to TUM.
    pred_rel = relative_c2w(pred_c2w)
    gt_rel = relative_c2w(gt_c2w)
    pred_tum = c2w_to_tum(pred_rel)
    gt_tum = c2w_to_tum(gt_rel)
    n = min(pred_tum.shape[0], gt_tum.shape[0])
    timestamps = np.arange(n, dtype=np.float64)

    pred_traj = PoseTrajectory3D(
        positions_xyz=pred_tum[:n, :3],
        orientations_quat_wxyz=pred_tum[:n, 3:],
        timestamps=timestamps,
    )
    gt_traj = PoseTrajectory3D(
        positions_xyz=gt_tum[:n, :3],
        orientations_quat_wxyz=gt_tum[:n, 3:],
        timestamps=timestamps,
    )
    gt_traj, pred_traj = sync.associate_trajectories(deepcopy(gt_traj), deepcopy(pred_traj))

    # Try Sim(3) alignment (umeyama with scale). For pan-only GT trajectories
    # (zero translation by design), Umeyama is rank-deficient → fall back to
    # origin-only alignment, mirroring eval_batrack_camera.py.
    align_mode = "umeyama_align_scale"
    try:
        ate = main_ape_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True, correct_scale=True, align_origin=False,
        )
        rpe_rot = main_rpe_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
            align=True, correct_scale=True, align_origin=False,
        )
        rpe_trans = main_rpe_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.translation_part,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
            align=True, correct_scale=True, align_origin=False,
        )
    except GeometryException:
        align_mode = "origin_align_only"
        ate = main_ape_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=False, correct_scale=False, align_origin=True,
        )
        rpe_rot = main_rpe_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
            align=False, correct_scale=False, align_origin=True,
        )
        rpe_trans = main_rpe_fn(
            gt_traj, pred_traj, est_name="traj",
            pose_relation=PoseRelation.translation_part,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
            align=False, correct_scale=False, align_origin=True,
        )
    # ----- Per-pair scale-normalized ATE/trans (works for ALL pairs with
    # rank-1+ GT translation, including 1D translates and zooms that
    # Umeyama-with-scale rejects as rank-deficient). The trick: scale the
    # PRED trajectory by ||gt_centered|| / ||pred_centered|| BEFORE evo, then
    # call evo with correct_scale=False align_origin=True. Result: ATE in GT
    # units, comparable across methods, on every pair where ||gt_centered|| >
    # eps (≈54/100 on DAVIS = orbits + 1D translates + zooms; pan-only n=46
    # has ||gt_centered||=0 by design and is the only class where ATE is
    # genuinely undefined — rot_err still applies there). -----------------
    gt_pos = gt_tum[:n, :3]
    pred_pos = pred_tum[:n, :3]
    gt_centered_norm  = float(np.linalg.norm(gt_pos  - gt_pos.mean(axis=0, keepdims=True)))
    pred_centered_norm = float(np.linalg.norm(pred_pos - pred_pos.mean(axis=0, keepdims=True)))
    eps = 1e-6
    ate_norm = trans_err_norm = None
    if gt_centered_norm > eps and pred_centered_norm > eps:
        scale_factor = gt_centered_norm / pred_centered_norm
        pred_pos_scaled = pred_pos * scale_factor
        pred_traj_scaled = PoseTrajectory3D(
            positions_xyz=pred_pos_scaled,
            orientations_quat_wxyz=pred_tum[:n, 3:],
            timestamps=timestamps,
        )
        gt_traj_2 = deepcopy(gt_traj)
        gt_traj_2, pred_traj_scaled = sync.associate_trajectories(gt_traj_2, pred_traj_scaled)
        try:
            ate_n = main_ape_fn(
                gt_traj_2, pred_traj_scaled, est_name="traj",
                pose_relation=PoseRelation.translation_part,
                align=False, correct_scale=False, align_origin=True,
            )
            rpe_t_n = main_rpe_fn(
                gt_traj_2, pred_traj_scaled, est_name="traj",
                pose_relation=PoseRelation.translation_part,
                delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
                align=False, correct_scale=False, align_origin=True,
            )
            ate_norm = float(ate_n.stats["rmse"])
            trans_err_norm = float(rpe_t_n.stats["rmse"])
        except Exception:
            pass

    return {
        "ate": float(ate.stats["rmse"]),
        "trans_err": float(rpe_trans.stats["rmse"]),
        "rot_err": float(rpe_rot.stats["rmse"]),
        # Scale-normalized variants (None for pan-only GT where ||gt||=0):
        "ate_norm": ate_norm,
        "trans_err_norm": trans_err_norm,
        "gt_centered_norm": gt_centered_norm,
        "pred_centered_norm": pred_centered_norm,
        "num_aligned_poses": int(gt_traj.num_poses),
        "alignment_mode": align_mode,
    }


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method_pred_root", type=Path, required=True,
                    help="<method root>/<seq>/<traj>/<pred>.mp4 (or .../predictions/<seq>/<traj>/...)")
    ap.add_argument("--pred_filenames", default="pred.mp4,{traj}.mp4,gen.mp4,output.mp4,render.mp4,color_{traj}.mp4",
                    help="Comma-separated candidates under <seq>/<traj>/.")
    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--cameras_root", type=Path, required=True,
                    help="cameras_2traj root: <root>/<seq>/<traj>.npz")
    ap.add_argument("--vipe_cache_root", type=Path, required=True,
                    help="Where to cache per-pair ViPE outputs.")
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--target_w", type=int, default=512,
                    help="Standardize pred mp4 width before ViPE (canonical 512).")
    ap.add_argument("--target_h", type=int, default=288,
                    help="Standardize pred mp4 height before ViPE (canonical 288).")
    ap.add_argument("--vipe_python", required=True,
                    help="Path to the vipe conda env's python (used to find sibling vipe CLI).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit_pairs", type=int, default=None,
                    help="Smoke run on first N pairs.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Split pairs by global row index modulo num_shards.")
    ap.add_argument("--shard_index", type=int, default=0,
                    help="Shard index to keep when num_shards > 1.")
    ap.add_argument("--phase", choices=["all", "vipe", "metrics"], default="all",
                    help="all = run ViPE then metrics; vipe = ViPE only; metrics = compute metrics from cached ViPE.")
    args = ap.parse_args()

    pred_filenames = [s.strip() for s in args.pred_filenames.split(",") if s.strip()]

    # Read pairs CSV (columns: video, trajectory)
    with open(args.pairs_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    pairs: List[Tuple[str, str]] = [(r["video"], r["trajectory"]) for r in rows]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(
            f"--shard_index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    if args.num_shards > 1:
        pairs = [pair for i, pair in enumerate(pairs) if i % args.num_shards == args.shard_index]
    print(f"[vipe-cam] {len(pairs)} pairs from {args.pairs_csv}")

    if not pairs:
        empty_summary = {
            "method_pred_root": str(args.method_pred_root),
            "num_pairs_evaluated": 0,
            "num_pairs_total": 0,
            "standardize_input": {
                "target_w": args.target_w,
                "target_h": args.target_h,
                "num_frames": args.num_frames,
                "note": "ffmpeg scale=W:H stretch + truncate first N frames",
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
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(empty_summary, fh, indent=2)
        print(f"[vipe-cam] empty shard -> {args.output_json}")
        return

    # Phase 1 — run ViPE on each pair (if not cached).
    if args.phase in ("all", "vipe"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for i, (seq, traj) in enumerate(pairs):
            pred_mp4 = resolve_pred_mp4(args.method_pred_root, seq, traj, pred_filenames)
            if pred_mp4 is None:
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (pred mp4 missing)")
                continue
            out_dir = args.vipe_cache_root / f"{seq}__{traj}"
            log_file = args.vipe_cache_root / "_logs" / f"{seq}__{traj}.log"
            try:
                pose_npz, was_cached = run_vipe_inference(
                    pred_mp4, out_dir, args.vipe_python, log_file,
                    args.num_frames, args.target_w, args.target_h)
                tag = "CACHED" if was_cached else "FRESH "
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: {tag} -> {pose_npz}")
            except Exception as e:
                print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: FAIL ({e})")

    if args.phase == "vipe":
        return

    # Phase 2 — compute metrics from cached ViPE outputs.
    per_pair: List[dict] = []
    for i, (seq, traj) in enumerate(pairs):
        cam_npz = args.cameras_root / seq / f"{traj}.npz"
        if not cam_npz.is_file():
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (gt camera npz missing)")
            continue
        out_dir = args.vipe_cache_root / f"{seq}__{traj}"
        pose_files = sorted((out_dir / "pose").glob("*.npz")) if (out_dir / "pose").exists() else []
        if not pose_files:
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: SKIP (vipe pose missing)")
            continue
        try:
            pred_c2w = load_vipe_c2w(pose_files[0], args.num_frames)
            gt_c2w = load_gt_c2w(cam_npz, args.num_frames)
            metrics = compute_pose_metrics(pred_c2w, gt_c2w)
            per_pair.append({"seq": seq, "traj": traj, **metrics})
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: ate={metrics['ate']:.4f}  "
                  f"trans={metrics['trans_err']:.4f}  rot={metrics['rot_err']:.3f}")
        except Exception as e:
            print(f"  [{i+1}/{len(pairs)}] {seq}/{traj}: METRIC FAIL ({e})")

    if not per_pair:
        print("[vipe-cam] no successful pairs; aborting summary")
        sys.exit(2)

    # Split by alignment mode. Sim(3) Umeyama only succeeds for trajectories with
    # rank-3 translation (orbit). Pan-only (||t||=0) and 1D-translate/zoom GT
    # trajectories fall back to origin_align_only — in that mode `ate` /
    # `trans_err` are NOT scale-normalized and conflate ViPE's per-method scene
    # scale with real trajectory error. ONLY rot_err is meaningful in fallback.
    sim3 = [r for r in per_pair if r.get("alignment_mode") == "umeyama_align_scale"]
    origin = [r for r in per_pair if r.get("alignment_mode") == "origin_align_only"]

    def _stats(rows: list, key: str) -> dict:
        if not rows:
            return {"mean": None, "std": None, "median": None, "n": 0}
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "median": None, "n": 0}
        v = np.array(vals)
        return {"mean": float(v.mean()), "std": float(v.std()),
                "median": float(np.median(v)), "n": int(len(vals))}

    # Pairs with rank-1+ GT translation (orbits + 1D translates + zooms +
    # any other non-pan-only) — the "all-pairs scale-normalized" metric set.
    has_gt_trans = [r for r in per_pair if r.get("ate_norm") is not None]

    summary = {
        "method_pred_root": str(args.method_pred_root),
        "num_pairs_evaluated": len(per_pair),
        "num_pairs_total": len(pairs),
        "standardize_input": {"target_w": args.target_w, "target_h": args.target_h,
                              "num_frames": args.num_frames,
                              "note": "ffmpeg scale=W:H stretch + truncate first N frames"},
        # ALL pairs (legacy, kept for backwards-compat — averages a meaningful
        # rot_err with a polluted ate/trans_err; do NOT cite ate/trans here).
        "ate":       _stats(per_pair, "ate"),
        "trans_err": _stats(per_pair, "trans_err"),
        "rot_err":   _stats(per_pair, "rot_err"),
        # Sim(3)-aligned subset (orbit-style trajectories with rank-3 translation).
        # ate/trans_err here ARE scale-normalized and directly comparable.
        "sim3_subset": {
            "ate":       _stats(sim3, "ate"),
            "trans_err": _stats(sim3, "trans_err"),
            "rot_err":   _stats(sim3, "rot_err"),
        },
        # Origin-only subset (pan / translate-1D / zoom).  ONLY rot_err is fair.
        "origin_subset": {
            "ate":       _stats(origin, "ate"),     # NOT comparable across methods
            "trans_err": _stats(origin, "trans_err"),  # NOT comparable across methods
            "rot_err":   _stats(origin, "rot_err"),
        },
        # All-pairs (with rank-1+ GT translation) per-pair scale-normalized
        # metric set. ate_norm / trans_err_norm here are computed by scaling
        # pred by ||gt_centered||/||pred_centered|| BEFORE evo (see
        # compute_pose_metrics) so the result is in GT trajectory units and
        # directly comparable across methods. n typically ≈ orbits + 1D
        # translates + zooms (≈54/100 on DAVIS); pan-only pairs (||gt||=0)
        # are excluded from ATE/trans by construction but contribute to
        # rot_err. THIS IS THE METRIC TO CITE FOR ATE ACROSS ALL VIDEOS.
        "all_norm": {
            "ate":       _stats(has_gt_trans, "ate_norm"),
            "trans_err": _stats(has_gt_trans, "trans_err_norm"),
            "rot_err":   _stats(per_pair,     "rot_err"),  # all 100 — rot is scale-invariant
        },
        "per_pair": per_pair,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print()
    print(f"[vipe-cam] {args.output_json}")
    print(f"  ALL n={len(per_pair)}: rot_err = {summary['rot_err']['mean']:.3f}° ± {summary['rot_err']['std']:.3f}°")
    s3 = summary["sim3_subset"]
    if s3["ate"]["n"]:
        print(f"  Sim(3) n={s3['ate']['n']:>3}: ate={s3['ate']['mean']:.4f}  trans={s3['trans_err']['mean']:.4f}  rot={s3['rot_err']['mean']:.3f}°")
    so = summary["origin_subset"]
    if so["ate"]["n"]:
        print(f"  origin n={so['ate']['n']:>3}: rot={so['rot_err']['mean']:.3f}°  (ate/trans NOT comparable in this mode)")
    an = summary["all_norm"]
    if an["ate"]["n"]:
        print(f"  ALL-NORM n={an['ate']['n']:>3}: ate={an['ate']['mean']:.4f}  trans={an['trans_err']['mean']:.4f}  (per-pair scale-norm; THE all-pair ATE)")


if __name__ == "__main__":
    main()
