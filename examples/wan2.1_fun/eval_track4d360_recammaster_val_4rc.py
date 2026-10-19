"""Eval Track4D-360 on ReCamMaster val with 4RC source depth + tracking.

This is the "full-conditioning" ReCamMaster benchmark path for the old
Track4D-360 checkpoints:

  - plucker camera control: ON
  - persistent Track4D branch: ON
  - dense source branch: ON
  - query-frame impulse: ON

Source geometry comes from precomputed 4RC outputs. Target camera trajectories
remain the GT ReCamMaster trajectories, but their translation scale is fit
per (source,target) pair into the 4RC source-geometry scale.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from eval_track4d360_hybrid_dense_syn4d import (  # noqa: E402
    _run_one_pair,
    build_eval_pipeline,
)
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.track4d360_mixed_nvs import (  # noqa: E402
    DEFAULT_PROMPT,
    center_crop_and_resize,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval Track4D-360 on ReCamMaster val with 4RC source depth + tracking."
    )

    parser.add_argument("--model_paths", type=str, default=None)
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--gs_checkpoint_path", type=str, default=None)
    parser.add_argument("--lora_base_model", type=str, default=None)
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--camera_control_checkpoint", type=str, default=None)

    from holo4d.shared_args import (
        add_cond_null_token_fallback_args,
        add_dense_branch_args,
        add_gs_output_args,
        add_perlayer_camera_args,
        add_plucker_args,
        add_query_impulse_args,
        add_track_config_args,
        add_warped_condition_args,
    )

    add_track_config_args(parser)
    add_dense_branch_args(parser)
    add_cond_null_token_fallback_args(parser)
    add_query_impulse_args(parser)
    add_plucker_args(parser)
    add_perlayer_camera_args(parser)
    add_warped_condition_args(parser)
    add_gs_output_args(parser)

    parser.add_argument(
        "--recammaster_root",
        type=str,
        default=os.environ.get("RECAMMASTER_ROOT", ""),
    )
    parser.add_argument(
        "--recammaster_4rc_root",
        type=str,
        default=os.environ.get("RECAMMASTER_4RC_ROOT", ""),
    )
    parser.add_argument("--val_subdirs", type=str, default="10basic_trajectories")
    parser.add_argument("--source_camera", type=str, default="cam01")
    parser.add_argument(
        "--target_cameras",
        type=str,
        default="cam02,cam03,cam04,cam05,cam06,cam07,cam08,cam09,cam10",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--sensor_width_mm", type=float, default=23.76)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--track_spatial_stride", type=int, default=1)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--capture_final_latents", action="store_true")
    parser.add_argument("--no_vae_tiling", dest="vae_tiling", action="store_false")
    parser.set_defaults(vae_tiling=True)
    parser.add_argument(
        "--drop_4rc_tracks",
        action="store_true",
        default=False,
        help="Load 4RC source RGB/depth but disable persistent 3D tracks by "
             "removing tracks_world. This skips the persistent placeholder path.",
    )
    parser.add_argument(
        "--null_4rc_tracks",
        action="store_true",
        default=False,
        help="Load 4RC source RGB/depth, keep tracks_world present, but set "
             "has_tracks=False so the persistent branch routes through its "
             "learned adapter null token. Use for track-drop ablations.",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _parse_camera_matrix(matrix_str: str) -> np.ndarray:
    rows = matrix_str.strip().split("] [")
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append(list(map(float, row.split())))
    return np.array(matrix, dtype=np.float32)


def _ue5_to_opencv_c2w(c2w_ue5: np.ndarray) -> np.ndarray:
    c2w = c2w_ue5.T.copy()
    c2w = c2w[:, [1, 2, 0, 3]]
    c2w[:3, 1] *= -1.0
    c2w[:3, 3] /= 100.0
    return c2w.astype(np.float32)


def _infer_focal_mm(val_subdir: str) -> float:
    match = re.search(r"f(\d+(?:\.\d+)?)", val_subdir)
    return float(match.group(1)) if match else 24.0


def _compute_intrinsics(focal_mm: float, width: int, height: int, sensor_width_mm: float) -> np.ndarray:
    w = float(width)
    h = float(height)
    if w >= h:
        eff_w = sensor_width_mm
        eff_h = sensor_width_mm * (h / w)
    else:
        eff_h = sensor_width_mm
        eff_w = sensor_width_mm * (w / h)
    fx = (focal_mm / eff_w) * w
    fy = (focal_mm / eff_h) * h
    return np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _read_mp4_frames(mp4_path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(mp4_path)
    frames = [frame for frame in reader]
    reader.close()
    if not frames:
        raise ValueError(f"No frames read from {mp4_path}")
    return frames


def _parse_caption(path: Path) -> str:
    if not path.exists():
        return DEFAULT_PROMPT
    text = path.read_text().strip()
    if "Prediction:" in text:
        text = text.split("Prediction:", 1)[1].strip()
    return text or DEFAULT_PROMPT


def _split_csv(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _squeeze_leading_singletons(arr):
    arr = np.asarray(arr)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _to_rgb_frame(view_img):
    arr = _squeeze_leading_singletons(view_img)
    if arr.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims after squeeze, got {arr.shape}")
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = ((arr * 0.5) + 0.5) * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def world_to_camera(points_world, c2w):
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return (points_world - t[None, :]) @ R


def project_points(points_world, c2w, K):
    pts_cam = world_to_camera(points_world, c2w)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    z_safe = np.where(valid, z, 1.0)
    u = K[0, 0] * pts_cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z_safe + K[1, 2]
    return u, v, z, valid


def render_points_to_target(points_world, colors_rgb, tgt_c2w, tgt_K, H, W):
    flat_points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    flat_colors = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
    u, v, z, valid = project_points(flat_points, tgt_c2w, tgt_K)
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    if not np.any(in_bounds):
        return rgb, mask

    z_ib = z[in_bounds]
    order = np.argsort(-z_ib)
    u_ib = u[in_bounds][order]
    v_ib = v[in_bounds][order]
    c_ib = flat_colors[in_bounds][order]
    rgb[v_ib, u_ib] = c_ib
    mask[v_ib, u_ib] = True
    return rgb, mask


def estimate_pair_translation_scale(source_run, target_frames, target_cam2src0, target_K):
    T = min(len(source_run["query_pts"]), len(target_frames), len(target_cam2src0))
    if T == 0:
        return 1.0

    trans_norms = np.linalg.norm(np.asarray(target_cam2src0)[:, :3, 3], axis=-1)
    candidate_frames = [i for i in range(T) if trans_norms[i] > 1e-6]
    if not candidate_frames:
        return 1.0

    if len(candidate_frames) > 4:
        idxs = np.linspace(0, len(candidate_frames) - 1, 4, dtype=int)
        candidate_frames = [candidate_frames[i] for i in idxs]

    H, W = source_run["frames"][0].shape[:2]
    scales = np.geomspace(0.02, 1.5, 16)
    best_score = None
    best_scale = 1.0

    for scale in scales:
        score_acc = 0.0
        valid_frames = 0
        for t in candidate_frames:
            pose = np.asarray(target_cam2src0[t], dtype=np.float32).copy()
            pose[:3, 3] *= float(scale)
            proj_rgb, proj_mask = render_points_to_target(
                source_run["query_pts"][t],
                source_run["frames"][t],
                pose,
                target_K[t],
                H,
                W,
            )
            n_valid = int(proj_mask.sum())
            if n_valid < 5000:
                continue
            tgt_rgb = np.asarray(target_frames[t], dtype=np.uint8)
            diff = np.abs(proj_rgb.astype(np.float32) - tgt_rgb.astype(np.float32)).mean(axis=2)
            photometric = float(diff[proj_mask].mean())
            coverage = float(n_valid) / float(H * W)
            score_acc += photometric - 5.0 * coverage + 0.75 * abs(np.log(float(scale)))
            valid_frames += 1

        if valid_frames == 0:
            continue
        score = score_acc / float(valid_frames)
        if best_score is None or score < best_score:
            best_score = score
            best_scale = float(scale)

    return float(best_scale)


def load_4rc_run(run_dir: Path):
    npz_path = run_dir / "result.npz"
    meta_path = run_dir / "metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    data = np.load(npz_path, allow_pickle=True)
    metadata = json.loads(meta_path.read_text())
    n_frames = int(data["n_frames"])

    frames = []
    pts_list = []
    confidence_list = []
    track_list = []
    c2w_list = []
    K_list = []
    for i in range(n_frames):
        frames.append(_to_rgb_frame(data[f"view_{i}_img"]))
        pts_list.append(_squeeze_leading_singletons(data[f"pred_{i}_pts"]).astype(np.float32))
        confidence_key = f"pred_{i}_conf"
        confidence_list.append(
            _squeeze_leading_singletons(data[confidence_key]).astype(np.float32)
            if confidence_key in data.files
            else None
        )
        track_list.append(_squeeze_leading_singletons(data[f"pred_{i}_track"]).astype(np.float32))
        c2w_list.append(_squeeze_leading_singletons(data[f"pred_{i}_extrinsic"]).astype(np.float32))
        K_list.append(_squeeze_leading_singletons(data[f"pred_{i}_intrinsic"]).astype(np.float32))

    query_idx = int(_squeeze_leading_singletons(data["pred_0_track_query_idx"]).reshape(-1)[0])
    track_video = np.stack(track_list, axis=0)
    track_flat = track_video.reshape(n_frames, -1, 3).transpose(1, 0, 2)

    return {
        "frames": frames,
        "pts": pts_list,
        "point_confidence": confidence_list,
        "tracks": track_flat,
        "c2w": c2w_list,
        "K": K_list,
        "query_idx": query_idx,
        "metadata": metadata,
    }


def transform_run_to_query_camera(run):
    query_idx = int(run["query_idx"])
    query_c2w = np.asarray(run["c2w"][query_idx], dtype=np.float32)
    query_pts = [
        world_to_camera(pts.reshape(-1, 3), query_c2w).reshape(pts.shape).astype(np.float32)
        for pts in run["pts"]
    ]
    query_tracks = world_to_camera(run["tracks"].reshape(-1, 3), query_c2w).reshape(run["tracks"].shape)
    return {
        **run,
        "query_pts": query_pts,
        "query_tracks": query_tracks.astype(np.float32),
    }


def extract_source_depth_video(run):
    depth_list = []
    for pts, c2w in zip(run["pts"], run["c2w"]):
        pts_cam = world_to_camera(pts.reshape(-1, 3), np.asarray(c2w, dtype=np.float32))
        depth = pts_cam[:, 2].reshape(pts.shape[:2]).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        depth_list.append(depth)
    return np.stack(depth_list, axis=0).astype(np.float32)


def resize_depth_video(depth_video, out_w, out_h):
    resized = [
        cv2.resize(np.asarray(depth, dtype=np.float32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        for depth in depth_video
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def load_recammaster_sample_4rc(
    recammaster_root: str,
    recammaster_4rc_root: str,
    val_subdir: str,
    source_camera: str,
    target_camera: str,
    start_frame: int,
    num_frames: int,
    width: int,
    height: int,
    sensor_width_mm: float,
) -> dict:
    val_dir = Path(recammaster_root) / "val" / val_subdir
    camera_json = val_dir / "cameras" / "camera_extrinsics.json"
    videos_dir = val_dir / "videos"
    if not camera_json.exists():
        raise FileNotFoundError(f"camera_extrinsics.json not found: {camera_json}")

    with camera_json.open() as f:
        cam_data = json.load(f)

    frame_keys_all = sorted(cam_data.keys(), key=lambda key: int(key.replace("frame", "")))
    total_frames = len(frame_keys_all)
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame={start_frame} outside [0, {total_frames - 1}]")
    frame_ids = list(range(start_frame, min(start_frame + num_frames, total_frames)))
    while len(frame_ids) < num_frames:
        frame_ids.append(frame_ids[-1])

    src_frames_raw = _read_mp4_frames(videos_dir / f"{source_camera}.mp4")
    tgt_frames_raw = _read_mp4_frames(videos_dir / f"{target_camera}.mp4")

    h_orig, w_orig = src_frames_raw[0].shape[:2]
    focal_mm = _infer_focal_mm(val_subdir)
    intrinsics_orig = _compute_intrinsics(focal_mm, w_orig, h_orig, sensor_width_mm)

    dummy_depth = np.zeros((h_orig, w_orig), dtype=np.float32)
    source_frames = []
    target_frames = []
    source_poses = []
    target_poses = []
    intrinsics_out = None

    for frame_id in frame_ids:
        frame_key = f"frame{frame_id}"
        src_pose = _ue5_to_opencv_c2w(_parse_camera_matrix(cam_data[frame_key][source_camera]))
        tgt_pose = _ue5_to_opencv_c2w(_parse_camera_matrix(cam_data[frame_key][target_camera]))
        source_poses.append(src_pose)
        target_poses.append(tgt_pose)

        src_img = src_frames_raw[frame_id]
        tgt_img = tgt_frames_raw[frame_id]
        src_pil, _, intrinsics_resized = center_crop_and_resize(
            src_img, dummy_depth, intrinsics_orig, width, height
        )
        tgt_pil, _, _ = center_crop_and_resize(
            tgt_img, dummy_depth, intrinsics_orig, width, height
        )
        source_frames.append(src_pil)
        target_frames.append(tgt_pil)
        if intrinsics_out is None:
            intrinsics_out = intrinsics_resized

    source_poses_arr = np.stack(source_poses, axis=0).astype(np.float32)
    target_poses_arr = np.stack(target_poses, axis=0).astype(np.float32)
    world_to_src0 = np.linalg.inv(source_poses_arr[0]).astype(np.float32)
    source_cam2src0 = world_to_src0[None, :, :] @ source_poses_arr
    target_cam2src0 = world_to_src0[None, :, :] @ target_poses_arr

    source_run_dir = Path(recammaster_4rc_root) / val_subdir / source_camera
    source_run = transform_run_to_query_camera(load_4rc_run(source_run_dir))
    fit_target_frames = [
        cv2.resize(np.asarray(frame, dtype=np.uint8), (source_run["frames"][0].shape[1], source_run["frames"][0].shape[0]))
        for frame in target_frames
    ]

    pair_scale = estimate_pair_translation_scale(
        source_run,
        fit_target_frames,
        target_cam2src0,
        np.asarray(source_run["K"], dtype=np.float32),
    )
    source_cam2src0[:, :3, 3] *= pair_scale
    target_cam2src0[:, :3, 3] *= pair_scale

    source_depth_video = resize_depth_video(extract_source_depth_video(source_run), width, height)
    caption = _parse_caption(videos_dir / f"{source_camera}_caption.txt")

    return {
        "video": source_frames,
        "target_video": target_frames,
        "tracks_world": source_run["query_tracks"],
        "target_cam2world": target_cam2src0,
        "target_intrinsics": intrinsics_out,
        "source_cam2world": source_cam2src0,
        "source_intrinsics": intrinsics_out,
        "source_rgb_video": source_frames,
        "source_depth_video": source_depth_video,
        "common_scale": np.float32(pair_scale),
        "track_query_idx": int(source_run["query_idx"]),
        "prompt": caption,
        "has_tracks": True,
        "has_dynamic_tracks": True,
        "has_depth": True,
        "track_video_loss_mask": 1.0,
        "pair_scale": np.float32(pair_scale),
    }


def main():
    import torch

    args = parse_args()
    torch.manual_seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    pipe = build_eval_pipeline(args)

    val_subdirs = _split_csv(args.val_subdirs)
    targets = _split_csv(args.target_cameras)

    for val_subdir in val_subdirs:
        for target_camera in targets:
            pair_out_dir = output_root / val_subdir / f"{args.source_camera}_to_{target_camera}"
            if (pair_out_dir / "pred_rgb.mp4").exists():
                print(f"[ReCamMaster4RC] skip {val_subdir} {args.source_camera}->{target_camera}: exists")
                continue

            sample = load_recammaster_sample_4rc(
                recammaster_root=args.recammaster_root,
                recammaster_4rc_root=args.recammaster_4rc_root,
                val_subdir=val_subdir,
                source_camera=args.source_camera,
                target_camera=target_camera,
                start_frame=args.start_frame,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                sensor_width_mm=args.sensor_width_mm,
            )
            if args.drop_4rc_tracks and args.null_4rc_tracks:
                raise ValueError("Use at most one of --drop_4rc_tracks and --null_4rc_tracks")
            if args.drop_4rc_tracks:
                sample["tracks_world"] = None
                sample["has_tracks"] = False
                sample["has_dynamic_tracks"] = False
                sample["track_video_loss_mask"] = 0.0
            elif args.null_4rc_tracks:
                sample["has_tracks"] = False
                sample["has_dynamic_tracks"] = False
                sample["track_video_loss_mask"] = 0.0
            print(
                f"[ReCamMaster4RC] running {val_subdir} {args.source_camera}->{target_camera} "
                f"pair_scale={float(sample['pair_scale']):.4f} "
                f"drop_4rc_tracks={bool(args.drop_4rc_tracks)} "
                f"null_4rc_tracks={bool(args.null_4rc_tracks)}"
            )
            _run_one_pair(pipe, sample, args, pair_out_dir)

    print(f"[ReCamMaster4RC] complete -> {output_root}")


if __name__ == "__main__":
    main()
