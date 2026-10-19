"""Track4D-360 inference on the iPhone NVS benchmark.

Reads conditions/<seq>/ produced by `build_iphone_pairs.py`, assembles the
sample dict expected by Track4D-360's V2V pipeline (matches the DAVIS eval
protocol — see `examples/wan2.1_fun/eval_track4d360_davis_4rc_novel_traj.py`),
and uses GT cam-1/cam-2 poses as the target trajectory (instead of synthetic
trajectories). Reuses `build_eval_pipeline` and `_run_one_pair` from the main
hybrid_dense eval script so all the V2V plucker/dense/track plumbing stays in
one place.

Per-seq inputs (built once by build_iphone_pairs.py):
    src_frames_1x/, src_depth_1x/, src_video_1x.mp4, src_cameras.npz, tgt_cameras.npz

Per-seq output:
    predictions/<run_tag>/<seq>/
        pred.mp4                # 49 frames, model native (288x512), fps=fps
        comparison.mp4          # debug 2x4 viz (src/gt/pred/diff || dense panels)
        metadata.json
        args.json

Inference convention (frozen — see CLAUDE.md "Inference convention for V2V NVS"):
    num_inference_steps=50, cfg_scale=1.0, sigma_shift=5.0, seed=42

Resolution:
    --height/--width default 288/512 (ours canonical). Source video gets resized
    + center-cropped from 720x960 → 288x512 by `center_crop_and_resize` (same
    helper the DAVIS eval uses). Target intrinsics are scaled accordingly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import imageio.v3 as iio
import numpy as np
import PIL.Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLE_DIR = REPO_ROOT / "examples" / "wan2.1_fun"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

# Reuse pipeline + per-pair runner from the canonical eval script.
from eval_track4d360_hybrid_dense_syn4d import (  # type: ignore
    _run_one_pair,
    build_eval_pipeline,
)
from holo4d.data.track_reprojection import (  # type: ignore
    project_tracks_to_target_views,
)
from holo4d.novel_view.static_scene_novel_traj import (  # type: ignore
    _warp_source_to_target_frames,
)
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.track4d360_mixed_nvs import (  # type: ignore
    DEFAULT_PROMPT,
    center_crop_and_resize,
)


# -------------------------- Argument parsing --------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Track4D-360 iPhone NVS inference.")

    # Model / pipeline (mirrors eval_track4d360_davis_4rc_novel_traj.py)
    p.add_argument("--model_paths", type=str, default=None)
    p.add_argument("--model_id_with_origin_paths", type=str, default=None)
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--gs_checkpoint_path", type=str, default=None)
    p.add_argument("--lora_base_model", type=str, default=None)
    p.add_argument("--lora_target_modules", type=str, default="")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=float, default=None)
    p.add_argument("--lora_checkpoint", type=str, default=None)
    p.add_argument("--camera_control_checkpoint", type=str, default=None)
    p.add_argument("--fp8_models", type=str, default=None)
    p.add_argument("--offload_models", type=str, default=None)

    from holo4d.shared_args import (  # type: ignore
        add_cond_null_token_fallback_args,
        add_dense_branch_args,
        add_gs_dynamic_args,
        add_gs_output_args,
        add_perlayer_camera_args,
        add_plucker_args,
        add_query_impulse_args,
        add_track_config_args,
        add_warped_condition_args,
    )
    add_track_config_args(p)
    add_dense_branch_args(p)
    add_cond_null_token_fallback_args(p)
    add_query_impulse_args(p)
    add_plucker_args(p)
    add_perlayer_camera_args(p)
    add_warped_condition_args(p)
    add_gs_output_args(p)
    add_gs_dynamic_args(p)

    # iPhone-specific dataset paths
    p.add_argument("--iphone_conditions_root", type=str, required=True,
                   help="Root produced by build_iphone_pairs.py (contains conditions/<seq>/).")
    p.add_argument("--sequences", type=str, default="apple,block,paper-windmill,spin,teddy")

    # Source-geometry mode: "lidar" (default; matches paper) or "4rc" (use 4RC tracks/depth/poses,
    # scale-aligned to iPhone LIDAR via per-seq α from depth_scale_alpha.json).
    p.add_argument("--use_4rc_source", action="store_true", default=False,
                   help="Use 4RC tracks/depth/poses as source geometry (scale-aligned to iPhone "
                        "LIDAR via depth_scale_alpha.json). Requires --iphone_4rc_root.")
    p.add_argument("--iphone_4rc_root", type=str, default=None,
                   help="Root containing <seq>/result.npz from 4RC inference.")
    p.add_argument("--use_vggt_omega_source", action="store_true", default=False,
                   help="Use VGGT-Omega source depth/cameras and query-frame pseudo-tracks. "
                        "Mutually exclusive with --use_4rc_source.")
    p.add_argument("--iphone_vggt_omega_root", type=str, default=None,
                   help="Root containing <seq>/pose/vggt_omega.npz and "
                        "<seq>/geometry/vggt_omega_depth.npz.")
    p.add_argument("--use_vggt_omega_cowtracker_source", action="store_true", default=False,
                   help="Use persistent CoWTracker 2D identities lifted with VGGT-Omega "
                        "depth/cameras. Mutually exclusive with the other source overrides.")
    p.add_argument("--iphone_cowtracker_root", type=str, default=None,
                   help="Root containing <seq>_pred_trackvideo.npz CoWTracker caches.")
    p.add_argument("--use_vdpm_source", action="store_true", default=False,
                   help="Use the legacy V-DPM pointmap/camera source condition. Its pixel "
                        "identities are the fixed frame-0 grid, not explicit tracked identities.")
    p.add_argument("--iphone_vdpm_root", type=str, default=None,
                   help="Root containing <seq>/result.npz V-DPM caches.")
    p.add_argument("--use_vggt_omega_opend4rt_source", action="store_true", default=False,
                   help="Use Open-D4RT persistent tracks Sim(3)-aligned to VGGT-Omega "
                        "source depth/cameras. Mutually exclusive with other overrides.")
    p.add_argument("--iphone_opend4rt_root", type=str, default=None,
                   help="Root containing <seq>/result.npz Open-D4RT source-track caches.")
    p.add_argument("--track_conf_pct", type=float, default=2.5,
                   help="4RC viser conf_pct default — replaces low-conf 4RC tracks with the "
                        "static query-frame point. Matches DAVIS bench.")
    p.add_argument(
        "--null_tracks",
        action="store_true",
        default=False,
        help="Keep source geometry for dense reprojection but mark persistent tracks absent. "
             "This reproduces the camera-ablation null-token contract for RGB/RGBD tiers.",
    )
    p.add_argument("--source_preproc", type=str, default="center_crop",
                   choices=["center_crop", "resize"],
                   help="How to preprocess iPhone source 720x960 → model_w x model_h. "
                        "'center_crop' (default) does aspect-preserving center-crop to 16:9 + "
                        "resize, losing top/bottom of iPhone scene. 'resize' does aspect-distorting "
                        "direct LANCZOS resize, keeping the full scene at the cost of squashed "
                        "aspect (model sees iPhone content elongated horizontally). "
                        "4RC source mode (--use_4rc_source) auto-forces 'resize' since 4RC tracks "
                        "span the full iPhone scene.")

    # Sampling / shape
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--track_spatial_stride", type=int, default=1)
    p.add_argument("--no_vae_tiling", dest="vae_tiling", action="store_false")
    p.set_defaults(vae_tiling=True)
    p.add_argument("--capture_final_latents", action="store_true")

    # Output
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--force_rerun", action="store_true", default=False)
    p.add_argument("--gs_feedback_interval", type=int, default=0)

    # FSDP inference
    p.add_argument("--use_fsdp", action="store_true", default=False)
    return p.parse_args()


# -------------------------- iPhone sample assembly --------------------------


def _scale_intrinsics(K: np.ndarray, src_w: int, src_h: int, out_w: int, out_h: int) -> np.ndarray:
    """Aspect-distorting direct resize: independently scale fx/cx by W ratio and fy/cy by H ratio.

    Correct for the `--source_preproc resize` path where the image gets a non-uniform LANCZOS
    resize from (src_w, src_h) to (out_w, out_h). NOT correct for the center-crop path —
    use `_intrinsics_after_center_crop_resize` for that.
    """
    K = np.asarray(K, dtype=np.float32).copy()
    K[0, 0] *= float(out_w) / float(src_w)
    K[0, 2] *= float(out_w) / float(src_w)
    K[1, 1] *= float(out_h) / float(src_h)
    K[1, 2] *= float(out_h) / float(src_h)
    return K.astype(np.float32)


def _intrinsics_after_center_crop_resize(
    K: np.ndarray, src_w: int, src_h: int, out_w: int, out_h: int
) -> np.ndarray:
    """Correct intrinsics transform for `Holo4D runtime ... center_crop_and_resize`:
    center-crop to target aspect, then uniform-resize to (out_w, out_h).

    Steps must mirror `center_crop_and_resize` (track4d360_mixed_nvs.py:141):
        1. Crop to target aspect: shift cx/cy by -left/-top accordingly.
        2. Resize crop to (out_w, out_h): aspect now matches target so sx == sy.
    """
    K = np.asarray(K, dtype=np.float32).copy()
    target_aspect = float(out_w) / float(out_h)
    src_aspect = float(src_w) / float(src_h)
    if src_aspect > target_aspect:
        # Source wider than target — crop sides.
        new_w = int(round(src_h * target_aspect))
        left = (src_w - new_w) // 2
        K[0, 2] -= float(left)
        crop_w, crop_h = new_w, src_h
    else:
        # Source taller — crop top/bottom (this is the iPhone case).
        new_h = int(round(src_w / target_aspect))
        top = (src_h - new_h) // 2
        K[1, 2] -= float(top)
        crop_w, crop_h = src_w, new_h
    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)
    # sx and sy are equal (within rounding) since the crop matches target aspect.
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K.astype(np.float32)


def _resize_depth_video(depth_video: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    out = [
        cv2.resize(np.asarray(d, dtype=np.float32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        for d in depth_video
    ]
    return np.stack(out, axis=0).astype(np.float32)


def _load_4rc_source_geometry(
    rc_npz_path: Path,
    alpha: float,
    out_w: int,
    out_h: int,
    track_conf_pct: float,
) -> dict:
    """Mirror DAVIS' load_4rc_run + transform_run_to_query_camera + the
    track-conf filter, then multiply tracks/depth/cam-translations by α to
    bring them into iPhone metric scale. Returns a dict that supplies the
    source-side fields of the inference sample.
    """
    raw = np.load(rc_npz_path, allow_pickle=True)
    n_frames = int(raw["n_frames"])
    qidx = int(np.asarray(raw["pred_0_track_query_idx"]).reshape(-1)[0])

    pts_list, track_list, c2w_list, K_list = [], [], [], []
    for i in range(n_frames):
        def _sq(arr):
            arr = np.asarray(arr)
            while arr.ndim > 0 and arr.shape[0] == 1 and arr.ndim > 2:
                arr = arr[0]
            return arr
        pts_list.append(_sq(raw[f"pred_{i}_pts"]).astype(np.float32))
        track_list.append(_sq(raw[f"pred_{i}_track"]).astype(np.float32))
        c2w_list.append(_sq(raw[f"pred_{i}_extrinsic"]).astype(np.float32))
        K_list.append(_sq(raw[f"pred_{i}_intrinsic"]).astype(np.float32))

    H_rc, W_rc = pts_list[0].shape[:2]
    track_video = np.stack(track_list, axis=0)                               # (T, H, W, 3) world
    track_flat = track_video.reshape(n_frames, -1, 3).transpose(1, 0, 2)     # (N, T, 3)

    query_c2w = c2w_list[qidx]
    world_to_query = np.linalg.inv(query_c2w).astype(np.float32)
    query_pts = [
        ((world_to_query[:3, :3] @ p.reshape(-1, 3).T).T + world_to_query[:3, 3]).reshape(p.shape).astype(np.float32)
        for p in pts_list
    ]
    tracks_query = (
        (world_to_query[:3, :3] @ track_flat.reshape(-1, 3).T).T + world_to_query[:3, 3]
    ).reshape(track_flat.shape).astype(np.float32)

    # ---- Track-conf filter (matches DAVIS' load_davis_sample_4rc) ----
    static_pts_q = query_pts[qidx].reshape(-1, 3)
    n_pixels = tracks_query.shape[0]
    keep_mask = None

    conf_track_per_frame = []
    for fi in range(n_frames):
        ct_key = f"pred_{fi}_conf_track"
        if ct_key in raw.files:
            conf_track_per_frame.append(np.asarray(raw[ct_key]).reshape(-1))
    pts_conf_query = (
        np.asarray(raw[f"pred_{qidx}_conf"]).reshape(-1)
        if f"pred_{qidx}_conf" in raw.files else None
    )
    if conf_track_per_frame and pts_conf_query is not None and pts_conf_query.size == n_pixels:
        agg_track_conf = np.sum(np.stack(conf_track_per_frame, axis=0), axis=0)
        pts0 = tracks_query[:, 0, :]
        max_disp = np.zeros(n_pixels, dtype=np.float32)
        for t in range(1, tracks_query.shape[1]):
            disp = np.linalg.norm(tracks_query[:, t, :] - pts0, axis=-1)
            np.maximum(max_disp, disp, out=max_disp)
        dynamic_ratio = float((max_disp > 0.025).mean())
        if dynamic_ratio / 2.0 >= track_conf_pct / 100.0:
            effective_pct = track_conf_pct
        else:
            effective_pct = dynamic_ratio / 2.0
        n_keep_track = max(1, int(n_pixels * (100.0 - effective_pct) / 100.0))
        n_keep_pts = max(1, int(n_pixels * (100.0 - track_conf_pct) / 100.0))
        track_mask = np.zeros(n_pixels, dtype=bool)
        track_mask[np.argpartition(-agg_track_conf, n_keep_track - 1)[:n_keep_track]] = True
        pts_mask = np.zeros(n_pixels, dtype=bool)
        pts_mask[np.argpartition(-pts_conf_query, n_keep_pts - 1)[:n_keep_pts]] = True
        keep_mask = track_mask & pts_mask
        tracks_query[~keep_mask] = static_pts_q[~keep_mask, None, :]
        n_low = int((~keep_mask).sum())
        print(f"[4rc] conf_pct={track_conf_pct} dynamic_ratio={dynamic_ratio:.3f} "
              f"replaced {n_low}/{n_pixels} pixels with static query")
    else:
        print("[4rc] WARN: conf_track / conf_query missing in result.npz; "
              "skipping track-conf filter")

    # ---- Source depth = z-component of pts_world in the source-cam frame ----
    depth_list = []
    for pts, c2w in zip(pts_list, c2w_list):
        w2c = np.linalg.inv(c2w)
        pf = pts.reshape(-1, 3)
        d_cam = ((w2c[:3, :3] @ pf.T).T + w2c[:3, 3])[:, 2]
        d = d_cam.reshape(pts.shape[:2]).astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[d < 0.0] = 0.0
        depth_list.append(d)
    depth_video = np.stack(depth_list, axis=0).astype(np.float32)

    # ---- Resize depth + scale intrinsics to (out_w, out_h) ----
    # 4RC native is typically 280×504 landscape (when run with --crop_width 512
    # --crop_height 288, matching the model's center-crop convention). The model
    # operates at out_h × out_w (e.g. 288×512). We aspect-distort the depth and
    # intrinsics from 4RC's grid to the model's grid — small change since the
    # aspects already match. Tracks are NOT resampled — they stay at 4RC native
    # per-pixel grid (matches DAVIS bench convention; the persistent-track
    # branch's adapter is grid-agnostic).
    src_intrinsics = _scale_intrinsics(K_list[0], W_rc, H_rc, out_w, out_h).astype(np.float32)
    src_depth_resized = _resize_depth_video(depth_video, out_w, out_h)

    # ---- Source poses in query-cam frame ----
    src_c2ws_q = np.stack(
        [(world_to_query @ np.asarray(c2w, dtype=np.float32)).astype(np.float32) for c2w in c2w_list],
        axis=0,
    )

    # ---- Apply α (4RC arbitrary unit -> iPhone metric meters) ----
    a = float(alpha)
    src_c2ws_q[:, :3, 3] *= a
    src_depth_resized = src_depth_resized * a
    tracks_query = tracks_query * a

    return {
        "tracks_query": tracks_query.astype(np.float32),
        "source_intrinsics": src_intrinsics,
        "source_depth_video": src_depth_resized.astype(np.float32),
        "source_cam2world_q": src_c2ws_q.astype(np.float32),
        "rc_query_idx": qidx,
        "rc_hw": (int(H_rc), int(W_rc)),
        "n_pixels": int(n_pixels),                      # = H_rc * W_rc, 4RC native (e.g., 280*504)
        "alpha": a,
    }


def _load_vggt_omega_source_geometry(
    vggt_root: Path,
    seq: str,
    alpha: float,
    out_w: int,
    out_h: int,
) -> dict:
    """Build source depth/cameras and query-frame pseudo-tracks from VGGT-Omega.

    VGGT-Omega predicts per-frame depth/cameras, rather than persistent point
    identities. This control therefore keeps the same flattened pixel index
    across time after unprojecting every frame into the query-camera frame.
    """
    seq_root = vggt_root / seq
    pose_path = seq_root / "pose" / "vggt_omega.npz"
    depth_path = seq_root / "geometry" / "vggt_omega_depth.npz"
    if not pose_path.exists() or not depth_path.exists():
        raise FileNotFoundError(
            f"missing VGGT-Omega cache for {seq}: pose={pose_path} depth={depth_path}"
        )
    pose = np.load(pose_path)
    depth_data = np.load(depth_path)
    c2ws = np.asarray(pose["cam_c2w"], dtype=np.float32)
    intrinsics = np.asarray(pose["intrinsics"], dtype=np.float32)
    depth = np.asarray(depth_data["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3 or c2ws.shape[0] != depth.shape[0] or intrinsics.shape[0] != depth.shape[0]:
        raise ValueError(
            f"invalid VGGT-Omega shapes for {seq}: depth={depth.shape}, c2ws={c2ws.shape}, K={intrinsics.shape}"
        )

    src_h, src_w = depth.shape[1:]
    if (src_h, src_w) != (out_h, out_w):
        depth = np.stack(
            [cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR) for frame in depth],
            axis=0,
        ).astype(np.float32)
        intrinsics = intrinsics.copy()
        intrinsics[:, 0, [0, 2]] *= float(out_w) / float(src_w)
        intrinsics[:, 1, [1, 2]] *= float(out_h) / float(src_h)

    query_idx = 0
    world_to_query = np.linalg.inv(c2ws[query_idx]).astype(np.float32)
    c2ws_q = (world_to_query[None] @ c2ws).astype(np.float32)
    H, W = depth.shape[1:]
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    points_by_frame = []
    for frame_depth, K, c2w in zip(depth, intrinsics, c2ws):
        z = np.where(np.isfinite(frame_depth) & (frame_depth > 0.0), frame_depth, 0.0)
        x = (xx - float(K[0, 2])) * z / float(K[0, 0])
        y = (yy - float(K[1, 2])) * z / float(K[1, 1])
        points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        points_world = (c2w[:3, :3] @ points_cam.T).T + c2w[:3, 3]
        points_query = (world_to_query[:3, :3] @ points_world.T).T + world_to_query[:3, 3]
        points_by_frame.append(points_query.astype(np.float32))
    tracks_query = np.stack(points_by_frame, axis=1)
    valid = np.isfinite(tracks_query[:, query_idx]).all(axis=-1) & (tracks_query[:, query_idx, 2] > 1e-3)
    fallback = tracks_query[valid, query_idx].mean(axis=0) if valid.any() else np.zeros(3, dtype=np.float32)
    tracks_query[~np.isfinite(tracks_query).all(axis=-1)] = fallback

    scale = float(alpha)
    c2ws_q[:, :3, 3] *= scale
    return {
        "tracks_query": (tracks_query * scale).astype(np.float32),
        "source_intrinsics": intrinsics[query_idx].astype(np.float32),
        "source_depth_video": (depth * scale).astype(np.float32),
        "source_cam2world_q": c2ws_q.astype(np.float32),
        "source_kind": "vggt_omega",
    }


def _load_vggt_omega_cowtracker_source_geometry(
    cow_root: Path,
    vggt_root: Path,
    seq: str,
    alpha: float,
    out_w: int,
    out_h: int,
    num_frames: int,
) -> dict:
    """Load real CoW identities and lift them through VGGT-Omega geometry.

    This is deliberately distinct from ``_load_vggt_omega_source_geometry``:
    that control keeps a pixel index fixed through independently predicted
    pointmaps, whereas this path follows CoWTracker's 2D correspondences.
    """
    from holo4d.benchmark.cow_vggt_run import load_cow_vggt_run

    run = load_cow_vggt_run(
        seq,
        cow_root=str(cow_root),
        vggt_root=str(vggt_root),
        num_frames=num_frames,
        height=out_h,
        width=out_w,
        scale_align="none",
        track_sample_mode="all",
        cow_conf_threshold=0.5,
        depth_conf_threshold=0.0,
    )
    c2ws = np.stack(run["c2w"], axis=0).astype(np.float32)
    intrinsics = np.stack(run["K"], axis=0).astype(np.float32)
    query_idx = int(run["query_idx"])
    world_to_query = np.linalg.inv(c2ws[query_idx]).astype(np.float32)
    tracks_world = np.asarray(run["tracks"], dtype=np.float32)
    tracks_query = (
        (world_to_query[:3, :3] @ tracks_world.reshape(-1, 3).T).T
        + world_to_query[:3, 3]
    ).reshape(tracks_world.shape)
    c2ws_q = (world_to_query[None] @ c2ws).astype(np.float32)
    depth = np.stack(
        [np.asarray(p, dtype=np.float32)[..., 2] for p in run["pts"]], axis=0
    )
    scale = float(alpha)
    c2ws_q[:, :3, 3] *= scale
    return {
        "tracks_query": (tracks_query * scale).astype(np.float32),
        "source_intrinsics": intrinsics[query_idx].astype(np.float32),
        "source_depth_video": (depth * scale).astype(np.float32),
        "source_cam2world_q": c2ws_q.astype(np.float32),
        "source_kind": "vggt_omega+cowtracker",
        "track_valid": np.asarray(run["tracks_valid"], dtype=bool),
    }


def _load_vdpm_source_geometry(
    vdpm_root: Path,
    seq: str,
    lidar_depth_video: np.ndarray,
    out_w: int,
    out_h: int,
    num_frames: int,
) -> dict:
    """Adapt the frozen V-DPM cache into the inference source contract.

    V-DPM supplies a per-frame pointmap but no correspondence sidecar.  This
    is therefore a legacy fixed-grid condition: the pixel at (row, col) in
    frame zero defines the identity reused at the same grid coordinate in all
    frames.  Its scale is aligned to the source LiDAR depth at frame zero.
    """
    path = vdpm_root / seq / "result.npz"
    if not path.is_file():
        raise FileNotFoundError(f"V-DPM result.npz missing for {seq}: {path}")
    raw = np.load(path, allow_pickle=False)
    pts_world = np.asarray(raw["pts3d"], dtype=np.float32)[:num_frames]
    w2cs = np.asarray(raw["extrinsic_WC"], dtype=np.float32)[:num_frames]
    intrinsics = np.asarray(raw["intrinsic"], dtype=np.float32)[:num_frames].copy()
    if pts_world.ndim != 4 or pts_world.shape[-1] != 3:
        raise ValueError(f"{path}: expected pts3d=(T,H,W,3), got {pts_world.shape}")
    if w2cs.shape != (pts_world.shape[0], 4, 4):
        raise ValueError(f"{path}: extrinsic_WC shape {w2cs.shape} incompatible with pts3d")
    src_h, src_w = pts_world.shape[1:3]
    c2ws = np.linalg.inv(w2cs).astype(np.float32)
    query_idx = 0
    query_w2c = w2cs[query_idx]
    tracks_query = (
        (query_w2c[:3, :3] @ pts_world.reshape(-1, 3).T).T + query_w2c[:3, 3]
    ).reshape(pts_world.shape).reshape(pts_world.shape[0], -1, 3).transpose(1, 0, 2)
    depth_native = np.empty(pts_world.shape[:3], dtype=np.float32)
    for t in range(pts_world.shape[0]):
        depth_native[t] = (
            (w2cs[t, :3, :3] @ pts_world[t].reshape(-1, 3).T).T + w2cs[t, :3, 3]
        )[:, 2].reshape(src_h, src_w)
    depth_native[~np.isfinite(depth_native)] = 0.0
    depth_native[depth_native < 0] = 0.0
    depth = _resize_depth_video(depth_native, out_w, out_h)
    intrinsics[:, 0, [0, 2]] *= float(out_w) / float(src_w)
    intrinsics[:, 1, [1, 2]] *= float(out_h) / float(src_h)
    valid = (depth[0] > 1e-3) & np.isfinite(lidar_depth_video[0]) & (lidar_depth_video[0] > 1e-3)
    if int(valid.sum()) < 100:
        raise ValueError(f"{path}: fewer than 100 jointly valid source LiDAR/V-DPM pixels for scale alignment")
    alpha = float(np.median(lidar_depth_video[0][valid] / depth[0][valid]))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"{path}: invalid V-DPM -> LiDAR scale {alpha}")
    c2ws_q = (query_w2c[None] @ c2ws).astype(np.float32)
    c2ws_q[:, :3, 3] *= alpha
    return {
        "tracks_query": (tracks_query * alpha).astype(np.float32),
        "source_intrinsics": intrinsics[query_idx].astype(np.float32),
        "source_depth_video": (depth * alpha).astype(np.float32),
        "source_cam2world_q": c2ws_q.astype(np.float32),
        "source_kind": "vdpm_fixed_grid",
        "alpha": alpha,
    }


def _bilinear_sample_depth(depth: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample a depth grid at pixel-centre coordinates."""
    height, width = depth.shape
    x, y = uv[:, 0], uv[:, 1]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, width - 1); x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, height - 1); y1 = np.clip(y0 + 1, 0, height - 1)
    wx = x - x0; wy = y - y0
    values = (
        (1 - wx) * (1 - wy) * depth[y0, x0] + wx * (1 - wy) * depth[y0, x1]
        + (1 - wx) * wy * depth[y1, x0] + wx * wy * depth[y1, x1]
    ).astype(np.float32)
    return values, valid & np.isfinite(values) & (values > 1e-4)


def _umeyama_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping source XYZ into target XYZ."""
    src = np.asarray(source, dtype=np.float64); dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 3:
        raise ValueError(f"Sim(3) needs >=3 matched Nx3 points, got {src.shape}/{dst.shape}")
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src0, dst0 = src - mu_src, dst - mu_dst
    covariance = (dst0.T @ src0) / float(src.shape[0])
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vh) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vh
    variance = float(np.mean(np.sum(src0 * src0, axis=-1)))
    if variance <= 1e-12:
        raise ValueError("degenerate Open-D4RT anchor geometry for Sim(3)")
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _load_vggt_omega_opend4rt_source_geometry(
    vggt_root: Path,
    opend4rt_root: Path,
    seq: str,
    alpha: float,
    out_w: int,
    out_h: int,
    num_frames: int,
) -> dict:
    """Combine VGGT source camera/depth with Open-D4RT persistent trajectories.

    Open-D4RT coordinates are reference-0 but uncalibrated. We estimate one
    source-only Sim(3) from its q0 grid anchors to VGGT frame-0 unprojected
    points, then apply it to every trajectory position. No iPhone depth/camera
    label enters this alignment.
    """
    base = _load_vggt_omega_source_geometry(vggt_root, seq, alpha=1.0, out_w=out_w, out_h=out_h)
    path = opend4rt_root / seq / "result.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Open-D4RT cache missing for {seq}: {path}")
    raw = np.load(path, allow_pickle=False)
    tracks = np.asarray(raw["tracks_xyz_ref0"], dtype=np.float32)
    uv = np.asarray(raw["track_query_uv_px"], dtype=np.float32)
    query_frame = np.asarray(raw["track_query_frame"], dtype=np.int32)
    visibility = np.asarray(raw["tracks_visibility"], dtype=bool)
    confidence = np.asarray(raw["tracks_confidence"], dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[1:] != (num_frames, 3) or uv.shape != (tracks.shape[0], 2):
        raise ValueError(f"{path}: unexpected Open-D4RT shapes tracks={tracks.shape}, uv={uv.shape}")
    anchor = query_frame == 0
    depth0 = np.asarray(base["source_depth_video"][0], dtype=np.float32)
    sampled_depth, valid = _bilinear_sample_depth(depth0, uv)
    K = np.asarray(base["source_intrinsics"], dtype=np.float32)
    target_xyz = np.stack([
        (uv[:, 0] - K[0, 2]) * sampled_depth / K[0, 0],
        (uv[:, 1] - K[1, 2]) * sampled_depth / K[1, 1],
        sampled_depth,
    ], axis=-1)
    valid &= anchor & visibility[:, 0] & np.isfinite(confidence[:, 0]) & (confidence[:, 0] > 0.0)
    valid &= np.isfinite(tracks[:, 0]).all(axis=-1)
    if int(valid.sum()) < 16:
        raise ValueError(f"{path}: only {int(valid.sum())} valid q0 Open-D4RT/VGGT anchors; need >=16")
    scale, rotation, translation = _umeyama_sim3(tracks[valid, 0], target_xyz[valid])
    aligned = (scale * (rotation @ tracks.reshape(-1, 3).T)).T + translation
    aligned = aligned.reshape(tracks.shape).astype(np.float32)
    finite = np.isfinite(aligned).all(axis=-1)
    fallback = aligned[valid, 0].mean(axis=0)
    aligned[~finite] = fallback
    base.update({
        "tracks_query": (aligned * float(alpha)).astype(np.float32),
        "source_depth_video": (np.asarray(base["source_depth_video"], dtype=np.float32) * float(alpha)),
        "source_cam2world_q": np.asarray(base["source_cam2world_q"], dtype=np.float32),
        "source_kind": "vggt_omega+opend4rt",
        "opend4rt_sim3_scale": scale,
        "opend4rt_anchor_count": int(valid.sum()),
    })
    base["source_cam2world_q"][:, :3, 3] *= float(alpha)
    return base


def _load_iphone_sample(
    seq: str,
    conditions_root: Path,
    width: int,
    height: int,
    num_frames: int,
    use_4rc_source: bool = False,
    iphone_4rc_root: Path | None = None,
    use_vggt_omega_source: bool = False,
    iphone_vggt_omega_root: Path | None = None,
    use_vggt_omega_cowtracker_source: bool = False,
    iphone_cowtracker_root: Path | None = None,
    use_vdpm_source: bool = False,
    iphone_vdpm_root: Path | None = None,
    use_vggt_omega_opend4rt_source: bool = False,
    iphone_opend4rt_root: Path | None = None,
    track_conf_pct: float = 2.5,
    source_preproc: str = "center_crop",
) -> dict:
    """Build the inference `sample` dict for one iPhone sequence.

    Source = cam0 49 frames; target poses = GT cam-{1|2} poses from val.json.
    Tracks = static per-pixel point cloud unprojected from frame-0 lidar depth
    (matches the paper convention: "depth map labels for the point cloud input").
    All cameras are expressed in the QUERY-camera frame (= cam0 at t_start),
    matching DAVIS' transform_run_to_query_camera convention.
    """
    seq_dir = conditions_root / "conditions" / seq
    if not (seq_dir / "metadata.json").exists():
        raise FileNotFoundError(f"missing conditions for {seq}: {seq_dir}")
    with (seq_dir / "metadata.json").open() as f:
        meta = json.load(f)

    if int(meta["num_frames"]) != num_frames:
        raise ValueError(
            f"conditions built for num_frames={meta['num_frames']} "
            f"but inference asked for {num_frames}"
        )

    # ---------- load 1x source frames + depth ----------
    src_rgbs_1x = []
    src_depths_1x = []
    for i in range(num_frames):
        rgb = iio.imread(seq_dir / "src_frames_1x" / f"{i:03d}.png")
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        src_rgbs_1x.append(rgb.astype(np.uint8))
        src_depths_1x.append(np.load(seq_dir / "src_depth_1x" / f"{i:03d}.npy").astype(np.float32))
    src_rgbs_1x = np.stack(src_rgbs_1x, axis=0)        # (T, H1x, W1x, 3)
    src_depths_1x = np.stack(src_depths_1x, axis=0)    # (T, H1x, W1x)
    H1x, W1x = src_rgbs_1x.shape[1:3]

    # ---------- preprocess source frames: center_crop_and_resize OR direct resize ----------
    # `--source_preproc center_crop` (default, matches Vista4D + iPhone-NVS paper
    # protocol): center-crop iPhone to 16:9 strip + resize. 4RC must be run with
    # CROP_WIDTH=512 CROP_HEIGHT=288 so its tracks land in the same 16:9 view.
    # `--source_preproc resize` (debug/ablation): aspect-distorting direct resize
    # to model res. 4RC must be run with CROP_WIDTH=720 CROP_HEIGHT=960 (no crop)
    # so its tracks span the full iPhone scene matching the resized source.
    effective_preproc = source_preproc
    src_frames_pil: list[PIL.Image.Image] = []
    if effective_preproc == "center_crop":
        K_dummy = np.array(
            [[max(W1x, H1x), 0.0, W1x * 0.5],
             [0.0, max(W1x, H1x), H1x * 0.5],
             [0.0, 0.0, 1.0]], dtype=np.float32,
        )
        for t in range(num_frames):
            pil, _, _ = center_crop_and_resize(
                src_rgbs_1x[t], None, K_dummy, width, height
            )
            src_frames_pil.append(pil)
    else:  # "resize" — aspect-distorting LANCZOS direct resize
        for t in range(num_frames):
            pil = PIL.Image.fromarray(src_rgbs_1x[t]).resize((width, height), PIL.Image.LANCZOS)
            src_frames_pil.append(pil)
    src_rgb_video = np.stack(
        [np.asarray(f, dtype=np.float32) / 255.0 for f in src_frames_pil], axis=0
    )

    # ---------- cameras: load 1x intrinsics, then scale to (height, width) ----------
    src_cams = np.load(seq_dir / "src_cameras.npz")
    tgt_cams = np.load(seq_dir / "tgt_cameras.npz")
    src_Ks_1x = src_cams["Ks_1x"]                        # (T, 3, 3)
    src_c2ws_world = src_cams["c2ws"].astype(np.float32) # (T, 4, 4)
    tgt_Ks_1x = tgt_cams["Ks_1x"]                        # (T, 3, 3)
    tgt_c2ws_world = tgt_cams["c2ws"].astype(np.float32) # (T, 4, 4)

    # Scale intrinsics to (width, height) — must match the source-image transform.
    #   center_crop  → use `_intrinsics_after_center_crop_resize` (crop offset + uniform resize)
    #   resize       → use `_scale_intrinsics`                    (non-uniform direct resize)
    # Bug 2026-05-02 (recorded): originally used `_scale_intrinsics` for both paths, which
    # gave a wrong fy in the center_crop path (~0.42× truth on iPhone 720×960 → 288×512)
    # and silently destroyed NVS quality (~11 dB PSNR floor). Fixed by branching on preproc.
    out_w, out_h = width, height
    K_xform = (
        _intrinsics_after_center_crop_resize
        if effective_preproc == "center_crop"
        else _scale_intrinsics
    )
    src_Ks_out = np.stack([
        K_xform(K, W1x, H1x, out_w, out_h) for K in src_Ks_1x
    ], axis=0).astype(np.float32)
    tgt_Ks_out = np.stack([
        K_xform(K, W1x, H1x, out_w, out_h) for K in tgt_Ks_1x
    ], axis=0).astype(np.float32)

    # ---------- transform cameras into the query-cam frame (= cam0 at t=0) ----------
    query_idx = 0
    world_to_query = np.linalg.inv(src_c2ws_world[query_idx]).astype(np.float32)
    src_c2ws = (world_to_query[None] @ src_c2ws_world).astype(np.float32)
    tgt_c2ws = (world_to_query[None] @ tgt_c2ws_world).astype(np.float32)

    # Sanity: src[0] should be (very close to) identity.
    assert np.allclose(src_c2ws[0], np.eye(4, dtype=np.float32), atol=1e-3)

    # ---------- depth at output resolution ----------
    src_depth_video = _resize_depth_video(src_depths_1x, out_w, out_h)

    # ---------- (optional) 4RC source-geometry override ----------
    # When --use_4rc_source is set, we replace the lidar+COLMAP source path
    # (depth/tracks/poses/intrinsics) with the 4RC reconstruction, scaled by
    # per-seq α from depth_scale_alpha.json so 4RC's arbitrary unit lands in
    # iPhone's metric (LIDAR meters) frame. Target poses + target intrinsics
    # stay as iPhone GT (already in meters), so source and target now share
    # the same coordinate system and scale.
    override_count = sum((use_4rc_source, use_vggt_omega_source, use_vggt_omega_cowtracker_source, use_vdpm_source, use_vggt_omega_opend4rt_source))
    if override_count > 1:
        raise ValueError("Use only one source-geometry override")
    geometry_override: dict | None = None
    if use_4rc_source:
        if iphone_4rc_root is None:
            raise ValueError("--use_4rc_source requires --iphone_4rc_root")
        alpha_path = seq_dir / "depth_scale_alpha.json"
        if not alpha_path.exists():
            raise FileNotFoundError(
                f"missing α scale for {seq} — run align_4rc_to_iphone_lidar.py first: "
                f"{alpha_path}"
            )
        with alpha_path.open() as f:
            alpha = float(json.load(f)["alpha_4rc_to_lidar"])
        rc_npz = Path(iphone_4rc_root) / seq / "result.npz"
        if not rc_npz.exists():
            raise FileNotFoundError(f"4RC result.npz missing for {seq}: {rc_npz}")
        print(f"[4rc] {seq}: α={alpha:.4f}  loading {rc_npz}")
        geometry_override = _load_4rc_source_geometry(
            rc_npz_path=rc_npz,
            alpha=alpha,
            out_w=out_w,
            out_h=out_h,
            track_conf_pct=track_conf_pct,
        )
        # Override source-side fields. Target stays at iPhone GT.
        # Note: 4RC's query frame is at rc_query_idx (typically 0). The
        # source poses returned by _load_4rc_source_geometry are already
        # expressed in the 4RC query frame (× α for translation). Since we
        # already constructed the iPhone target poses in iPhone cam0_t=0
        # frame (= identical to 4RC's query frame at the anchor), and α
        # brought 4RC into iPhone meters, the two coordinate systems agree.
        src_depth_video = geometry_override["source_depth_video"]
        src_c2ws = geometry_override["source_cam2world_q"]
    elif use_vggt_omega_source:
        if iphone_vggt_omega_root is None:
            raise ValueError("--use_vggt_omega_source requires --iphone_vggt_omega_root")
        alpha_path = seq_dir / "vggt_omega_depth_scale_alpha.json"
        if not alpha_path.exists():
            raise FileNotFoundError(
                f"missing VGGT-Omega alpha for {seq}: {alpha_path}; run align_vggt_omega_to_iphone_lidar.py"
            )
        with alpha_path.open() as f:
            alpha = float(json.load(f)["alpha_vggt_omega_to_lidar"])
        geometry_override = _load_vggt_omega_source_geometry(
            vggt_root=iphone_vggt_omega_root,
            seq=seq,
            alpha=alpha,
            out_w=out_w,
            out_h=out_h,
        )
        print(f"[vggt-omega] {seq}: alpha={alpha:.4f}")
        src_depth_video = geometry_override["source_depth_video"]
        src_c2ws = geometry_override["source_cam2world_q"]
    elif use_vggt_omega_cowtracker_source:
        if iphone_vggt_omega_root is None or iphone_cowtracker_root is None:
            raise ValueError("CoW+VGGT source requires --iphone_vggt_omega_root and --iphone_cowtracker_root")
        alpha_path = seq_dir / "vggt_omega_depth_scale_alpha.json"
        if not alpha_path.exists():
            raise FileNotFoundError(f"missing VGGT-Omega alpha for {seq}: {alpha_path}")
        with alpha_path.open() as f:
            alpha = float(json.load(f)["alpha_vggt_omega_to_lidar"])
        geometry_override = _load_vggt_omega_cowtracker_source_geometry(
            cow_root=iphone_cowtracker_root, vggt_root=iphone_vggt_omega_root,
            seq=seq, alpha=alpha, out_w=out_w, out_h=out_h, num_frames=num_frames,
        )
        print(f"[vggt-omega+cowtracker] {seq}: alpha={alpha:.4f}")
        src_depth_video = geometry_override["source_depth_video"]
        src_c2ws = geometry_override["source_cam2world_q"]
    elif use_vdpm_source:
        if iphone_vdpm_root is None:
            raise ValueError("--use_vdpm_source requires --iphone_vdpm_root")
        geometry_override = _load_vdpm_source_geometry(
            vdpm_root=iphone_vdpm_root, seq=seq, lidar_depth_video=src_depth_video,
            out_w=out_w, out_h=out_h, num_frames=num_frames,
        )
        print(f"[vdpm] {seq}: alpha_vdpm_to_lidar={geometry_override['alpha']:.4f}")
        src_depth_video = geometry_override["source_depth_video"]
        src_c2ws = geometry_override["source_cam2world_q"]
    elif use_vggt_omega_opend4rt_source:
        if iphone_vggt_omega_root is None or iphone_opend4rt_root is None:
            raise ValueError("VGGT+Open-D4RT source requires --iphone_vggt_omega_root and --iphone_opend4rt_root")
        alpha_path = seq_dir / "vggt_omega_depth_scale_alpha.json"
        if not alpha_path.exists():
            raise FileNotFoundError(f"missing VGGT-Omega alpha for {seq}: {alpha_path}")
        with alpha_path.open() as f:
            alpha = float(json.load(f)["alpha_vggt_omega_to_lidar"])
        geometry_override = _load_vggt_omega_opend4rt_source_geometry(
            vggt_root=iphone_vggt_omega_root, opend4rt_root=iphone_opend4rt_root,
            seq=seq, alpha=alpha, out_w=out_w, out_h=out_h, num_frames=num_frames,
        )
        print(f"[vggt-omega+opend4rt] {seq}: alpha={alpha:.4f} "
              f"sim3={geometry_override['opend4rt_sim3_scale']:.4f} "
              f"anchors={geometry_override['opend4rt_anchor_count']}")
        src_depth_video = geometry_override["source_depth_video"]
        src_c2ws = geometry_override["source_cam2world_q"]

    # ---------- tracks_world (per-pixel static point cloud, in query-cam frame) ----------
    if geometry_override is not None:
        # 4RC mode: per-pixel temporal tracks come straight from 4RC, already
        # × α and already in the query-cam frame. The track tensor is at 4RC
        # native resolution (e.g. 280×504); we keep it at that native shape
        # since the model treats tracks as a pixel-grid attention input and
        # the per-pixel layout is what matters, not the W/H against the
        # output frame.
        tracks_query = geometry_override["tracks_query"]
        n_pixels = tracks_query.shape[0]
        # For source_intrinsics, swap in the rescaled 4RC K instead of COLMAP K.
        source_intrinsics_override = geometry_override["source_intrinsics"]
    else:
        # Default lidar+COLMAP path: unproject query-frame LIDAR depth → 3D
        # points in cam0 (=query) frame, then broadcast across T frames as a
        # static cloud. Matches the paper: "we adopt the camera pose and depth
        # map labels … for the point cloud input" — depth IS the point cloud.
        H, W = out_h, out_w
        K_q = src_Ks_out[query_idx]
        fx, fy = float(K_q[0, 0]), float(K_q[1, 1])
        cx, cy = float(K_q[0, 2]), float(K_q[1, 2])
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        z = src_depth_video[query_idx].astype(np.float32)
        x = (xx.astype(np.float32) - cx) * z / fx
        y = (yy.astype(np.float32) - cy) * z / fy
        pts_q = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        n_pixels = pts_q.shape[0]
        tracks_query = np.broadcast_to(
            pts_q[:, None, :], (n_pixels, num_frames, 3)
        ).astype(np.float32).copy()
        valid_pix = np.isfinite(z) & (z > 1e-3)
        if valid_pix.any():
            median_pt = pts_q[valid_pix.reshape(-1)].mean(axis=0)
        else:
            median_pt = np.zeros(3, dtype=np.float32)
        tracks_query[~valid_pix.reshape(-1)] = median_pt[None, :]
        source_intrinsics_override = None

    # ---------- common-scale normalisation (matches DAVIS) ----------
    finite = np.isfinite(tracks_query).all(axis=-1)
    valid_pts = tracks_query[finite]
    if valid_pts.shape[0] > 0:
        norms = np.linalg.norm(valid_pts, axis=-1)
        norms = norms[norms > 1e-3]
        common_scale = float(norms.mean()) if norms.size > 0 else 1.0
    else:
        common_scale = 1.0
    common_scale = max(common_scale, 1e-6)

    src_c2ws_n = src_c2ws.copy(); src_c2ws_n[:, :3, 3] /= common_scale
    tgt_c2ws_n = tgt_c2ws.copy(); tgt_c2ws_n[:, :3, 3] /= common_scale
    src_depth_video_n = src_depth_video / common_scale
    tracks_world_n = tracks_query / common_scale

    src_mode = geometry_override.get("source_kind", "4rc(α-aligned)") if geometry_override is not None else "lidar+COLMAP"
    print(f"[iphone] {seq}: H1x={H1x} W1x={W1x} -> {out_h}x{out_w}; "
          f"src_mode={src_mode}; common_scale={common_scale:.4f}")

    # ---------- assemble base sample (mirrors load_davis_sample_4rc) ----------
    target_intrinsics = tgt_Ks_out[query_idx]  # iPhone target intrinsics are constant per frame; pick query
    source_intrinsics = (
        source_intrinsics_override if source_intrinsics_override is not None
        else src_Ks_out[query_idx]
    )

    base_sample = {
        "video": src_frames_pil,
        "tracks_world": tracks_world_n,
        "target_intrinsics": target_intrinsics,
        "source_cam2world": src_c2ws_n,
        "source_intrinsics": source_intrinsics,
        "source_rgb_video": src_rgb_video,
        "source_depth_video": src_depth_video_n,
        "common_scale": np.float32(common_scale),
        "track_query_idx": int(query_idx),
        "prompt": DEFAULT_PROMPT,
        "has_tracks": True,
        "has_dynamic_tracks": False,   # iPhone source point cloud is static (depth-only)
        "has_depth": True,
        "track_video_loss_mask": 1.0,
        "sequence_name": seq,
        "trajectory_name": f"iphone_cam{int(tgt_cams['tgt_camera_id'])}",
    }

    # ---------- build target-view warped frames + projected tracks ----------
    # Mirrors `build_static_scene_novel_view_sample` but with externally
    # supplied target poses (from iPhone GT) instead of synthetic trajectories.
    target_frames, target_depth, target_depth_valid = _warp_source_to_target_frames(
        source_rgb_video=src_rgb_video,
        source_depth_video=src_depth_video_n,
        source_cam2world=src_c2ws_n,
        source_intrinsics=source_intrinsics,
        target_cam2world=tgt_c2ws_n,
    )
    _, track_vis, track_depth = project_tracks_to_target_views(
        torch.as_tensor(tracks_world_n, dtype=torch.float32),
        torch.as_tensor(tgt_c2ws_n, dtype=torch.float32),
        torch.as_tensor(target_intrinsics, dtype=torch.float32),
        image_h=int(height),
        image_w=int(width),
    )
    track_vis_np = track_vis[0].detach().cpu().numpy().astype(np.float32)
    track_depth_np = track_depth[0].detach().cpu().numpy().astype(np.float32) * track_vis_np

    sample = dict(base_sample)
    sample["target_video"] = target_frames
    sample["target_cam2world"] = tgt_c2ws_n
    sample["target_intrinsics"] = target_intrinsics
    sample["target_depth"] = target_depth
    sample["target_depth_valid"] = target_depth_valid
    sample["reproj_src_depth_at_target"] = track_depth_np
    sample["reproj_src_depth_valid"] = track_vis_np
    sample["trajectory_kind"] = "iphone_nvs"
    return sample


def main():
    args = parse_args()
    if getattr(args, "use_fsdp", False):
        from holo4d.eval_fsdp_helper import init_fsdp_early  # type: ignore
        init_fsdp_early(args)

    torch.manual_seed(args.seed)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    from holo4d.eval_fsdp_helper import is_main_process  # type: ignore
    if is_main_process():
        (output_root / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    print(f"[iPhone NVS] sequences={seqs}  output={output_root}")

    pipe = build_eval_pipeline(args)

    conditions_root = Path(args.iphone_conditions_root)
    for seq in seqs:
        pair_out_dir = output_root / seq
        pred_mp4 = pair_out_dir / "pred.mp4"
        pred_named_mp4 = pair_out_dir / f"iphone_cam{_target_cam_id(conditions_root, seq)}.mp4"
        if pred_mp4.exists() and not args.force_rerun:
            print(f"[iPhone NVS] skip {seq}: {pred_mp4} already exists")
            continue

        sample = _load_iphone_sample(
            seq=seq,
            conditions_root=conditions_root,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            use_4rc_source=bool(getattr(args, "use_4rc_source", False)),
            iphone_4rc_root=Path(args.iphone_4rc_root) if getattr(args, "iphone_4rc_root", None) else None,
            use_vggt_omega_source=bool(getattr(args, "use_vggt_omega_source", False)),
            iphone_vggt_omega_root=Path(args.iphone_vggt_omega_root) if getattr(args, "iphone_vggt_omega_root", None) else None,
            use_vggt_omega_cowtracker_source=bool(getattr(args, "use_vggt_omega_cowtracker_source", False)),
            iphone_cowtracker_root=Path(args.iphone_cowtracker_root) if getattr(args, "iphone_cowtracker_root", None) else None,
            use_vdpm_source=bool(getattr(args, "use_vdpm_source", False)),
            iphone_vdpm_root=Path(args.iphone_vdpm_root) if getattr(args, "iphone_vdpm_root", None) else None,
            use_vggt_omega_opend4rt_source=bool(getattr(args, "use_vggt_omega_opend4rt_source", False)),
            iphone_opend4rt_root=Path(args.iphone_opend4rt_root) if getattr(args, "iphone_opend4rt_root", None) else None,
            track_conf_pct=float(getattr(args, "track_conf_pct", 2.5)),
            source_preproc=str(getattr(args, "source_preproc", "center_crop")),
        )
        if args.null_tracks:
            # Mirror DAVIS --null_4rc_tracks: retain the source RGB/depth/camera
            # geometry used by the dense branch, while taking the learned
            # track-adapter null-token path instead of injecting real tracks.
            sample["has_tracks"] = False
            sample["has_dynamic_tracks"] = False
            sample["track_video_loss_mask"] = 0.0
            print(f"[iPhone NVS] {seq}: persistent tracks replaced by null token")
        print(f"[iPhone NVS] running {seq} -> {pair_out_dir}")
        _run_one_pair(pipe, sample, args, pair_out_dir)

        # _run_one_pair writes pred_rgb.mp4 + a trajectory-named mp4 from
        # `prediction_filename_for_sample`. Standardize to pred.mp4 for the
        # eval pipeline contract.
        if not pred_mp4.exists():
            for cand in (pred_named_mp4, pair_out_dir / "pred_rgb.mp4"):
                if cand.exists():
                    pred_mp4.write_bytes(cand.read_bytes())
                    break
        meta = {
            "sequence": seq,
            "checkpoint_path": args.checkpoint_path,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "sigma_shift": args.sigma_shift,
            "seed": args.seed,
        }
        (pair_out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[iPhone NVS] complete -> {output_root}")


def _target_cam_id(conditions_root: Path, seq: str) -> int:
    with (conditions_root / "conditions" / seq / "metadata.json").open() as f:
        return int(json.load(f)["tgt_camera_id"])


if __name__ == "__main__":
    main()
