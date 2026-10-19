"""Eval Track4D-360 on DAVIS with 4RC source geometry and synthetic trajectories."""

import argparse
import csv as _csv
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import imageio.v2 as imageio
import numpy as np
import PIL.Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLE_DIR = REPO_ROOT / "examples" / "wan2.1_fun"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from eval_track4d360_hybrid_dense_syn4d import (  # noqa: E402
    NEGATIVE_PROMPT,
    _run_one_pair,
    _summarize_vista4d_plucker_modules,
    build_eval_pipeline,
    precompute_eval_prompt_embs,
)
from eval_track4d360_recammaster_val_4rc import (  # noqa: E402
    DEFAULT_PROMPT,
    extract_source_depth_video,
    load_4rc_run,
    resize_depth_video,
    transform_run_to_query_camera,
)
from holo4d.benchmark.cow_vggt_run import (  # noqa: E402
    load_cow_vggt_multiquery_run,
    load_cow_vggt_run,
    select_cow_vggt_query_anchor,
    split_multiquery_cow_roots,
)
from holo4d.benchmark.cow_vggt_static_fusion import (  # noqa: E402
    fuse_static_background_valid_mask,
)
from holo4d.benchmark.davis_utils import (  # noqa: E402
    DAVIS_TRAJECTORIES_13579_ZERO_BASED,
    resolve_davis_sequences,
)
from holo4d.novel_view.static_scene_novel_traj import (  # noqa: E402
    build_static_scene_novel_view_sample,
    parse_static_scene_trajectories,
    save_static_scene_novel_view_payload,
)
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.track4d360_mixed_nvs import (  # noqa: E402
    _source_rgb_at_track_xy,
    center_crop_and_resize,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval Track4D-360 on DAVIS first-49-frame clips with 4RC source geometry."
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
    parser.add_argument("--fp8_models", type=str, default=None)
    parser.add_argument("--offload_models", type=str, default=None)

    from holo4d.shared_args import (
        add_cond_null_token_fallback_args,
        add_dense_branch_args,
        add_gs_dynamic_args,
        add_gs_output_args,
        add_perlayer_camera_args,
        add_plucker_args,
        add_query_impulse_args,
        add_source_backend_args,
        add_track_config_args,
        add_warped_condition_args,
    )

    add_track_config_args(parser)
    add_dense_branch_args(parser)
    parser.add_argument(
        "--dense_only_condition_modules",
        action="store_true",
        default=False,
        help="Use dense-target control alignment without persistent Track4D track adapters.",
    )
    add_cond_null_token_fallback_args(parser)
    add_query_impulse_args(parser)
    add_plucker_args(parser)
    add_perlayer_camera_args(parser)
    add_warped_condition_args(parser)
    add_source_backend_args(parser)
    parser.add_argument(
        "--cow_multiquery_fusion_mode",
        choices=("legacy_delta", "raw_union", "single_anchor"),
        default="legacy_delta",
        help=(
            "CoW/VGGT q>1 policy. single_anchor consumes exactly one selected "
            "query, for q1-trained checkpoint compatibility."
        ),
    )
    parser.add_argument(
        "--cow_single_query_anchor",
        choices=("first", "last"),
        default="last",
        help=(
            "Anchor retained by --cow_multiquery_fusion_mode single_anchor. "
            "last selects q48 from a q0+q48 DAVIS cache."
        ),
    )
    add_gs_output_args(parser)
    add_gs_dynamic_args(parser)

    parser.add_argument(
        "--davis_root",
        type=str,
        default="data/geometry_cache/DAVIS_518x294",
    )
    parser.add_argument(
        "--davis_4rc_root",
        type=str,
        default="outputs/baselines/4RC/davis_49f",
    )
    parser.add_argument("--gmos_root", type=str, default="")
    parser.add_argument("--gmos_mask_subdir", type=str, default="mos_frame")
    parser.add_argument("--gmos_dilate_px", type=int, default=0)
    parser.add_argument("--no_gmos_mask", action="store_true")
    parser.add_argument("--sequence_names", type=str, default=None)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--track_spatial_stride", type=int, default=1)
    parser.add_argument(
        "--i2v_condition_mode",
        choices=("off", "source_first", "previous_target"),
        default="off",
        help=(
            "Target-boundary image policy. source_first is the canonical first-49 "
            "DAVIS evaluation because every official synthetic trajectory starts at "
            "the source camera. previous_target reads the prior chunk prediction's "
            "last frame from --i2v_previous_target_root."
        ),
    )
    parser.add_argument(
        "--i2v_previous_target_root",
        type=str,
        default=None,
        help=(
            "Previous-chunk prediction root for --i2v_condition_mode previous_target. "
            "Expected layout: <root>/<sequence>/<trajectory>/{last_target.png,"
            "pred_rgb.mp4,<trajectory>.mp4}."
        ),
    )

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera_condition_scale", type=float, default=1.0)
    parser.add_argument("--debug_camera_control", action="store_true", default=False)
    parser.add_argument(
        "--strict_vista4d_plucker_nonzero",
        action="store_true",
        default=False,
        help="Fail if vista4d_plucker per-block camera encoders remain zero after checkpoint load.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--capture_final_latents", action="store_true")
    parser.add_argument("--no_vae_tiling", dest="vae_tiling", action="store_false")
    parser.set_defaults(vae_tiling=True)

    parser.add_argument(
        "--trajectories",
        type=str,
        default=",".join(DAVIS_TRAJECTORIES_13579_ZERO_BASED),
        help="Comma-separated subset of the static-scene synthetic trajectory names.",
    )
    parser.add_argument("--angle_deg", type=float, default=8.0)
    parser.add_argument("--distance", type=float, default=0.12)
    parser.add_argument("--orbit_radius", type=float, default=2.5)
    parser.add_argument(
        "--orbit_start_deg",
        type=float,
        default=0.0,
        help="Start-angle offset for orbit_left/orbit_right: the sweep covers "
        "[start, start + angle]. Used by the orbit-chain loop to continue a "
        "previous round's orbit. 0.0 (default) keeps the historical behavior.",
    )
    parser.add_argument(
        "--drop_4rc_tracks",
        action="store_true",
        default=False,
        help="Load 4RC source RGB/depth but remove tracks_world, skipping the persistent placeholder path.",
    )
    parser.add_argument(
        "--null_4rc_tracks",
        action="store_true",
        default=False,
        help="Load 4RC source RGB/depth, keep tracks_world present, but set has_tracks=False so the persistent branch uses its learned adapter null token.",
    )
    parser.add_argument("--force_rerun", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gs_feedback_interval", type=int, default=0)
    parser.add_argument(
        "--profile_jsonl",
        type=str,
        default=None,
        help="Append synchronized per-pair runtime and CUDA-memory records.",
    )
    parser.add_argument(
        "--profile_stage_breakdown",
        action="store_true",
        default=False,
        help="Profile source-condition, track-encoding, denoising, and decode stages.",
    )

    # Batched mode: iterate the exact (seq, traj) rows of a CSV in a single
    # python process so the 14B model is loaded ONCE for all pairs instead
    # of once per pair. Skips the cartesian seq × traj loop entirely.
    # CSV format: header `video,trajectory` then one (seq,traj) per line.
    parser.add_argument(
        "--pairs_csv",
        type=str,
        default=None,
        help="Optional CSV of (video,trajectory) pairs. When set, the eval iterates only these pairs (not the cartesian product of --sequence_names × --trajectories) and reuses one loaded model across all of them.",
    )
    parser.add_argument(
        "--davis_captions_json",
        type=str,
        default=None,
        help="Optional JSON caption map for DAVIS, keyed by sequence name.",
    )
    parser.add_argument(
        "--require_davis_caption",
        action="store_true",
        default=False,
        help="Require every selected DAVIS sequence to have an entry in --davis_captions_json.",
    )
    parser.add_argument(
        "--track_sample_mode",
        type=str,
        default="all",
        choices=("all", "grid"),
        help="DAVIS q1 track condition density. all=stride1 dense; grid uses --track_stride validity masking.",
    )
    parser.add_argument("--track_stride", type=int, default=1)
    parser.add_argument(
        "--variant_sweep",
        type=str,
        default="",
        help="Optional comma/space-separated DAVIS q1 variants to run with one loaded model. "
             "Supported: q1_dense, q1_s2, q1_s4.",
    )
    parser.add_argument(
        "--dynamic_mask_dilate_kernel",
        type=int,
        default=0,
        help="Optional odd dilation kernel for 4RC-derived source/target dynamic masks. "
             "0 keeps the historical no-dilation behavior.",
    )

    # FSDP inference, no-op unless launched with accelerate.
    parser.add_argument("--use_fsdp", action="store_true", default=False)
    return parser.parse_args()


def _extract_caption_text(value) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("caption", "text", "prompt", "description"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def load_davis_captions(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    caption_path = Path(path)
    if not caption_path.is_file():
        raise FileNotFoundError(f"DAVIS captions JSON not found: {caption_path}")

    raw = json.loads(caption_path.read_text())
    captions: dict[str, str] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for record in raw:
            if not isinstance(record, dict):
                continue
            name = (
                record.get("sequence")
                or record.get("sequence_name")
                or record.get("video")
                or record.get("name")
            )
            if isinstance(name, str):
                items.append((name, record))
    else:
        raise ValueError(f"Unsupported DAVIS captions JSON format: {caption_path}")

    for name, value in items:
        if not isinstance(name, str):
            continue
        text = _extract_caption_text(value)
        if text is not None:
            captions[name.strip()] = text

    print(f"[DAVIS4RC] loaded {len(captions)} captions from {caption_path}", flush=True)
    return captions


def _read_frame_rgb(path: Path) -> np.ndarray:
    frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Failed to read DAVIS frame: {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _load_davis_frames(frame_paths: tuple[Path, ...], width: int, height: int) -> tuple[list[PIL.Image.Image], np.ndarray]:
    dummy_depth = None
    frames = []
    for path in frame_paths:
        frame_rgb = _read_frame_rgb(path)
        pil, _, _ = center_crop_and_resize(
            frame_rgb,
            dummy_depth,
            _default_intrinsics(frame_rgb.shape[1], frame_rgb.shape[0]),
            width,
            height,
        )
        frames.append(pil)
    return frames, np.stack([np.asarray(frame, dtype=np.float32) / 255.0 for frame in frames], axis=0)


def _default_intrinsics(width: int, height: int) -> np.ndarray:
    focal = float(max(width, height))
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _scale_intrinsics(K: np.ndarray, src_width: int, src_height: int, out_width: int, out_height: int) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32).copy()
    K[0, 0] *= float(out_width) / float(src_width)
    K[0, 2] *= float(out_width) / float(src_width)
    K[1, 1] *= float(out_height) / float(src_height)
    K[1, 2] *= float(out_height) / float(src_height)
    return K.astype(np.float32)


def _derive_projected_dynamic_mask_from_4rc_tracks(
    *,
    tracks_world: np.ndarray,
    cam2world: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
    motion_threshold: float,
) -> np.ndarray | None:
    """Project raw 4RC moving tracks into a camera sequence as a T,H,W mask."""
    try:
        import torch
        from holo4d.conditions.soft_splat_warp import (
            compute_projected_dynamic_mask_from_tracks_world,
        )

        tw_t = torch.as_tensor(np.asarray(tracks_world, dtype=np.float32)).unsqueeze(0)
        c2w_t = torch.as_tensor(np.asarray(cam2world, dtype=np.float32)).unsqueeze(0)
        k_t = torch.as_tensor(np.asarray(intrinsics, dtype=np.float32))
        dyn = compute_projected_dynamic_mask_from_tracks_world(
            tracks_world=tw_t,
            cam2world=c2w_t,
            intrinsics=k_t,
            image_h=int(height),
            image_w=int(width),
            motion_threshold=float(motion_threshold),
        )
        if dyn is None:
            return None
        return dyn[0].detach().cpu().numpy().astype(bool, copy=False)
    except Exception as exc:
        print(f"[load_davis_sample_4rc] WARN: failed to derive 4RC dynamic mask: {exc}")
        return None


def _dilate_dynamic_mask_video(mask: np.ndarray | None, kernel_size: int) -> np.ndarray | None:
    if mask is None:
        return None
    k = int(kernel_size)
    if k <= 1:
        return mask
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    frames = [
        cv2.dilate(np.asarray(frame).astype(np.uint8), kernel, iterations=1).astype(bool)
        for frame in np.asarray(mask).astype(bool)
    ]
    return np.stack(frames, axis=0).astype(bool, copy=False)


def _source_rgb_at_4rc_track_grid(
    *,
    source_rgb_video: np.ndarray,
    tracks_world: np.ndarray,
    image_h: int,
    image_w: int,
    track_query_idx: int,
) -> np.ndarray:
    """Sample query-frame source RGB in the same identity order as 4RC tracks."""
    src = np.asarray(source_rgb_video, dtype=np.float32)
    tracks = np.asarray(tracks_world)
    if src.ndim != 4 or src.shape[-1] != 3:
        raise ValueError(f"source_rgb_video must be (T,H,W,3), got {src.shape}")
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"tracks_world must be (N,T,3), got {tracks.shape}")

    num_tracks = int(tracks.shape[0])
    num_frames = int(tracks.shape[1])
    grid_h, grid_w = _infer_track_grid_hw(num_tracks, int(image_h), int(image_w))
    q = max(0, min(int(track_query_idx), int(src.shape[0]) - 1))
    rgb_q = src[q]
    if rgb_q.max(initial=0.0) > 2.0:
        rgb_q = rgb_q / 255.0
    if (int(rgb_q.shape[0]), int(rgb_q.shape[1])) != (grid_h, grid_w):
        rgb_q = cv2.resize(rgb_q, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
    rgb_q = np.clip(rgb_q, 0.0, 1.0).astype(np.float32, copy=False)
    rgb_flat = rgb_q.reshape(-1, 3)
    if int(rgb_flat.shape[0]) != num_tracks:
        raise ValueError(
            f"source RGB track grid produced {rgb_flat.shape[0]} identities, "
            f"but tracks_world has {num_tracks}"
        )
    return np.broadcast_to(rgb_flat[:, None, :], (num_tracks, num_frames, 3)).copy()


def _source_rgb_at_tracks(
    *,
    source_rgb_video: np.ndarray,
    tracks_world: np.ndarray,
    source_xy: np.ndarray | None,
    query_idx_per_identity: np.ndarray | None = None,
    image_h: int,
    image_w: int,
    track_query_idx: int,
) -> np.ndarray:
    if source_xy is not None:
        if query_idx_per_identity is not None:
            return _source_rgb_at_track_xy_per_identity(
                source_rgb_video,
                np.asarray(source_xy, dtype=np.float32),
                np.asarray(query_idx_per_identity, dtype=np.int32),
                int(np.asarray(tracks_world).shape[1]),
            )
        return _source_rgb_at_track_xy(
            source_rgb_video,
            np.asarray(source_xy, dtype=np.float32),
            int(np.asarray(tracks_world).shape[1]),
        )
    return _source_rgb_at_4rc_track_grid(
        source_rgb_video=source_rgb_video,
        tracks_world=tracks_world,
        image_h=image_h,
        image_w=image_w,
        track_query_idx=track_query_idx,
    )


def _source_rgb_at_track_xy_per_identity(
    src_frames,
    query_xy: np.ndarray,
    query_idx_per_identity: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    """Sample each identity's RGB from its own query frame and broadcast."""
    src = np.asarray(src_frames, dtype=np.float32)
    if src.ndim != 4 or src.shape[-1] != 3:
        raise ValueError(f"source_rgb_video must be (T,H,W,3), got {src.shape}")
    T_src, H, W, _ = src.shape
    T = min(int(num_frames), T_src)
    xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    qidx = np.asarray(query_idx_per_identity, dtype=np.int64).reshape(-1)
    if xy.shape[0] != qidx.shape[0]:
        raise ValueError(f"query_xy N={xy.shape[0]} != query_idx_per_identity N={qidx.shape[0]}")
    out = np.empty((xy.shape[0], 3), dtype=np.float32)
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, W - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, H - 1)
    for q in np.unique(np.clip(qidx, 0, T_src - 1)):
        ids = np.where(np.clip(qidx, 0, T_src - 1) == int(q))[0]
        img = src[int(q)]
        if img.max(initial=0.0) > 2.0:
            img = img / 255.0
        out[ids] = img[y[ids], x[ids], :3].astype(np.float32, copy=False)
    return np.broadcast_to(out[:, None, :], (out.shape[0], T, 3)).copy()


def _source_mask_at_track_xy(
    *,
    source_dynamic_mask: np.ndarray,
    source_xy: np.ndarray,
    num_frames: int,
    query_idx_per_identity: np.ndarray | None = None,
) -> np.ndarray:
    m = np.asarray(source_dynamic_mask)
    if m.ndim == 2:
        m = m[None]
    if m.ndim != 3:
        raise ValueError(f"source_dynamic_mask must be (T,H,W) or (H,W), got {m.shape}")
    xy = np.asarray(source_xy, dtype=np.float32).reshape(-1, 2)
    if query_idx_per_identity is None:
        qidx = np.zeros((xy.shape[0],), dtype=np.int64)
    else:
        qidx = np.asarray(query_idx_per_identity, dtype=np.int64).reshape(-1)
        if qidx.shape[0] != xy.shape[0]:
            raise ValueError(f"source_xy N={xy.shape[0]} != query_idx_per_identity N={qidx.shape[0]}")
    h, w = m.shape[1:3]
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
    vals = np.zeros((xy.shape[0],), dtype=np.float32)
    qidx_clip = np.clip(qidx, 0, m.shape[0] - 1)
    for q in np.unique(qidx_clip):
        ids = np.where(qidx_clip == int(q))[0]
        vals[ids] = (m[int(q), y[ids], x[ids]] >= 0.5).astype(np.float32)
    return np.broadcast_to(vals[:, None], (vals.shape[0], int(num_frames))).copy()


def _resolve_gmos_mask_dir(gmos_root: str, seq: str, mask_subdir: str) -> Path | None:
    if not gmos_root:
        return None
    root = Path(gmos_root)
    candidates = [
        root / mask_subdir / seq,
        root / seq / mask_subdir / seq,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def _load_gmos_source_dynamic_mask(
    *,
    gmos_root: str,
    seq: str,
    mask_subdir: str,
    height: int,
    width: int,
    num_frames: int,
) -> np.ndarray | None:
    mask_dir = _resolve_gmos_mask_dir(gmos_root, seq, mask_subdir)
    if mask_dir is None:
        return None
    paths = sorted(mask_dir.glob("*.png"))[: int(num_frames)]
    if len(paths) < int(num_frames):
        raise FileNotFoundError(
            f"GMOS mask dir {mask_dir} has {len(paths)} frames, expected {int(num_frames)}"
        )
    masks = []
    for path in paths:
        arr = np.asarray(PIL.Image.open(path))
        if arr.ndim == 3:
            mask = np.any(arr[..., :3] > 0, axis=-1)
        else:
            mask = arr > 0
        resized = PIL.Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (int(width), int(height)),
            PIL.Image.NEAREST,
        )
        masks.append(np.asarray(resized, dtype=np.uint8) > 0)
    return np.stack(masks, axis=0).astype(bool)


def _morph_mask_video(mask: np.ndarray, *, op: str, radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    if radius <= 0:
        return mask.astype(bool, copy=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    out = []
    for frame in np.asarray(mask).astype(np.uint8):
        if op == "dilate":
            out.append(cv2.dilate(frame, kernel) > 0)
        elif op == "erode":
            out.append(cv2.erode(frame, kernel) > 0)
        else:
            raise ValueError(f"Unsupported mask op: {op}")
    return np.stack(out, axis=0).astype(bool)


def _source_mask_at_4rc_track_grid(
    *,
    source_dynamic_mask: np.ndarray,   # (T, H, W) in {0, 1}
    tracks_world: np.ndarray,          # (N, T, 3)
    image_h: int,
    image_w: int,
    track_query_idx: int,
) -> np.ndarray:
    """Sample the per-pixel source dynamic mask on the SAME 4RC track grid/order
    as ``_source_rgb_at_4rc_track_grid`` -> (N, T) per-identity motion label.

    Closes the eval train/eval mismatch (codex #3): ab27/ab28 train with the
    per-track dynamic-mask channel, so eval must provide it too instead of
    zero-filling. Mirrors the RGB gather exactly (query frame, resized to the
    track grid, broadcast over time).
    """
    m = np.asarray(source_dynamic_mask).astype(np.float32)
    tracks = np.asarray(tracks_world)
    if m.ndim == 2:
        m = m[None]
    if m.ndim != 3:
        raise ValueError(f"source_dynamic_mask must be (T,H,W) or (H,W), got {m.shape}")
    num_tracks = int(tracks.shape[0])
    num_frames = int(tracks.shape[1])
    grid_h, grid_w = _infer_track_grid_hw(num_tracks, int(image_h), int(image_w))
    q = max(0, min(int(track_query_idx), int(m.shape[0]) - 1))
    mq = m[q]
    if (int(mq.shape[0]), int(mq.shape[1])) != (grid_h, grid_w):
        mq = cv2.resize(mq, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
    mq_flat = (mq.reshape(-1) >= 0.5).astype(np.float32)
    if int(mq_flat.shape[0]) != num_tracks:
        raise ValueError(
            f"source mask track grid produced {mq_flat.shape[0]} identities, "
            f"but tracks_world has {num_tracks}"
        )
    return np.broadcast_to(mq_flat[:, None], (num_tracks, num_frames)).copy()


def _freeze_static_track_identities(
    tracks_world: np.ndarray,
    dynamic_at_tracks: np.ndarray | None,
    query_idx_per_identity: np.ndarray | None = None,
) -> np.ndarray:
    """Freeze background/static identities at each identity's query-frame 3D point."""
    tracks = np.asarray(tracks_world, dtype=np.float32).copy()
    if dynamic_at_tracks is None or tracks.ndim != 3 or tracks.shape[1] == 0:
        return tracks
    dyn = np.asarray(dynamic_at_tracks, dtype=bool)
    if dyn.shape[:2] != tracks.shape[:2]:
        return tracks
    n_tracks, n_frames = tracks.shape[:2]
    if query_idx_per_identity is None:
        qidx = np.zeros((n_tracks,), dtype=np.int64)
    else:
        qidx = np.asarray(query_idx_per_identity, dtype=np.int64).reshape(-1)
        if qidx.shape[0] != n_tracks:
            qidx = np.zeros((n_tracks,), dtype=np.int64)
    qidx = np.clip(qidx, 0, n_frames - 1)
    row = np.arange(n_tracks, dtype=np.int64)
    static_ids = ~dyn[row, qidx]
    tracks[static_ids, :, :] = tracks[static_ids, qidx[static_ids], :][:, None, :]
    return tracks


def _dynamic_only_track_valid_mask(
    tracks_valid: np.ndarray | None,
    dynamic_at_tracks: np.ndarray | None,
    tracks_shape: tuple[int, int],
    query_idx_per_identity: np.ndarray | None = None,
    tracks_world: np.ndarray | None = None,
    static_overlap_radius: float = 0.02,
) -> np.ndarray | None:
    """Build q-aware CoW/VGGT validity from GMOS dynamic labels."""
    return fuse_static_background_valid_mask(
        tracks_valid,
        dynamic_at_tracks,
        tracks_shape,
        query_idx_per_identity,
        tracks_world=tracks_world,
        static_overlap_radius=static_overlap_radius,
    )


def _source_cam2query(run) -> np.ndarray:
    query_idx = int(run["query_idx"])
    query_c2w = np.asarray(run["c2w"][query_idx], dtype=np.float32)
    world_to_query = np.linalg.inv(query_c2w).astype(np.float32)
    c2w = np.stack([np.asarray(pose, dtype=np.float32) for pose in run["c2w"]], axis=0)
    return (world_to_query[None] @ c2w).astype(np.float32)


def load_davis_sample_4rc(
    *,
    davis_root: str,
    davis_4rc_root: str,
    sequence_name: str,
    start_frame: int,
    num_frames: int,
    width: int,
    height: int,
    trajectory_name: str,
    angle_deg: float,
    distance: float,
    orbit_radius: float,
    orbit_start_deg: float = 0.0,
    dynamic_motion_threshold: float = 0.005,
    dynamic_mask_dilate_kernel: int = 0,
    gmos_root: str = "",
    gmos_mask_subdir: str = "mos_frame",
    gmos_dilate_px: int = 0,
    no_gmos_mask: bool = False,
    prompt: str = DEFAULT_PROMPT,
    source_backend: str = "4rc",
    cow_root: str | None = None,
    vggt_root: str | None = None,
    cow_track_sample_mode: str = "all",
    cow_track_stride: int = 1,
    cow_conf_threshold: float = 0.5,
    cow_depth_conf_threshold: float = 0.0,
    cow_multiquery_fusion_mode: str = "legacy_delta",
    cow_single_query_anchor: str = "last",
    seed: int = 42,
) -> dict:
    source_condition_started = time.perf_counter()
    sequences = resolve_davis_sequences(
        davis_root,
        sequence_names=sequence_name,
        num_frames=num_frames,
        start_frame=start_frame,
    )
    if len(sequences) != 1:
        raise RuntimeError(f"Expected one DAVIS sequence for {sequence_name}, got {len(sequences)}")
    sequence = sequences[0]

    # ---- Source geometry/tracks: 4RC (default) or cotracker+vggt-omega ----
    # Both backends yield an identical ``source_run``; downstream code is shared.
    # Mirrors eval_track4d360_vista4d_official.py:258-281.
    if source_backend == "cow_vggt":
        if not cow_root or not vggt_root:
            raise ValueError("source_backend='cow_vggt' requires cow_root and vggt_root")
        cow_roots = split_multiquery_cow_roots(cow_root)
        if len(cow_roots) > 1:
            run = load_cow_vggt_multiquery_run(
                sequence.name,
                cow_roots=cow_roots,
                vggt_root=vggt_root,
                num_frames=num_frames,
                height=height,
                width=width,
                davis_4rc_root=davis_4rc_root,   # frame-0 depth reference for 4RC-scale align
                scale_align="4rc",
                track_sample_mode=cow_track_sample_mode,
                track_stride=cow_track_stride,
                cow_conf_threshold=cow_conf_threshold,
                depth_conf_threshold=cow_depth_conf_threshold,
                track_sample_seed=seed,
            )
            print(
                f"[load_davis_sample_4rc] {sequence.name}: cow_vggt multi-query "
                f"roots={len(cow_roots)} query_indices={run.get('track_query_indices')}"
            )
        else:
            run = load_cow_vggt_run(
                sequence.name,
                cow_root=cow_roots[0],
                vggt_root=vggt_root,
                num_frames=num_frames,
                height=height,
                width=width,
                davis_4rc_root=davis_4rc_root,   # frame-0 depth reference for 4RC-scale align
                scale_align="4rc",
                track_sample_mode=cow_track_sample_mode,
                track_stride=cow_track_stride,
                cow_conf_threshold=cow_conf_threshold,
                depth_conf_threshold=cow_depth_conf_threshold,
                track_sample_seed=seed,
            )
        if (
            str(cow_multiquery_fusion_mode).strip().lower() == "single_anchor"
            and len(run.get("track_query_indices", [])) > 1
        ):
            run = select_cow_vggt_query_anchor(
                run, anchor=cow_single_query_anchor,
            )
            print(
                f"[load_davis_sample_4rc] {sequence.name}: q>1 input reduced to "
                f"single {cow_single_query_anchor} anchor "
                f"q={run['track_query_indices'][0]} for q1 compatibility"
            )
        source_run = transform_run_to_query_camera(run)
    elif source_backend == "4rc":
        run_dir = Path(davis_4rc_root) / sequence.name
        source_run = transform_run_to_query_camera(load_4rc_run(run_dir))
    else:
        raise ValueError(f"source_backend must be '4rc' or 'cow_vggt', got {source_backend!r}")
    source_frames, source_rgb_video = _load_davis_frames(sequence.frame_paths, width, height)

    src_h, src_w = source_run["frames"][0].shape[:2]
    source_intrinsics = _scale_intrinsics(source_run["K"][0], src_w, src_h, width, height)
    source_cam2world = _source_cam2query(source_run)
    source_depth_video = resize_depth_video(extract_source_depth_video(source_run), width, height)

    # ---- (A) Replace low-confidence 4RC tracks with the static query-frame point ----
    # Mimic 4RC viser's default filtering exactly (arc/viz/viser_visualizer_track.py:
    # `min_conf_thr_percentile=2.5` initial slider value, with the dynamic_ratio
    # adaptive reduction). Pixels failing the filter get their temporal track
    # replaced with the static query-frame point — matching training's
    # _compute_tracks default (track4d360_mixed_nvs.py:880-888) which initializes
    # ALL tracks to pts_world_q then overlays bbox-tracked dynamic positions.
    #
    # 4RC viser combined filter:
    #   conf_pct = 2.5  (default initial slider)
    #   max_disp[n] = max_t ||tracks_track[n,t] - tracks_track[n,0]||  (per pixel)
    #   dynamic_ratio = (max_disp > 0.025).mean()
    #   effective_pct = conf_pct if (dynamic_ratio/2 >= conf_pct/100) else (dynamic_ratio/2)
    #   keep top (100 - effective_pct)% by aggregated conf_track
    #     AND top (100 - conf_pct)% by query-frame conf (point cloud conf)
    # The viser-style track-confidence filter below reads 4RC's per-frame
    # conf arrays from result.npz. That file (and `run_dir`) only exists for the
    # 4rc backend; cow_vggt already filters tracks inside load_cow_vggt_run
    # (invalid identities frozen to frame-0 via visibility), so we skip the
    # re-read and leave the conf arrays empty -> the filter below no-ops.
    qidx = int(source_run["query_idx"])
    conf_track_per_frame = []
    pts_conf_query = None
    if source_backend == "4rc":
        npz_path = run_dir / "result.npz"
        raw = np.load(npz_path, allow_pickle=True)
        n_frames_raw = int(raw["n_frames"])
        for fi in range(n_frames_raw):
            ct_key = f"pred_{fi}_conf_track"
            if ct_key in raw.files:
                conf_track_per_frame.append(np.asarray(raw[ct_key]).reshape(-1))
        pc_key = f"pred_{qidx}_conf"
        if pc_key in raw.files:
            pts_conf_query = np.asarray(raw[pc_key]).reshape(-1)

    tracks_query = np.asarray(source_run["query_tracks"], dtype=np.float32)  # (N, T, 3)
    raw_tracks_query = tracks_query.copy()
    source_tracks_valid = source_run.get("tracks_valid")
    if isinstance(source_tracks_valid, np.ndarray) and source_tracks_valid.shape == tracks_query.shape[:2]:
        source_tracks_valid = source_tracks_valid.astype(bool, copy=False)
    else:
        source_tracks_valid = None
    source_tracks_confidence = source_run.get("tracks_confidence")
    if (
        isinstance(source_tracks_confidence, np.ndarray)
        and source_tracks_confidence.shape == tracks_query.shape[:2]
    ):
        source_tracks_confidence = source_tracks_confidence.astype(np.float32, copy=False)
    else:
        source_tracks_confidence = None
    source_tracks_xy = source_run.get("tracks_source_xy")
    if not isinstance(source_tracks_xy, np.ndarray):
        source_tracks_xy = None
    source_query_idx_per_identity = source_run.get("tracks_query_idx_per_identity")
    if not isinstance(source_query_idx_per_identity, np.ndarray):
        source_query_idx_per_identity = None
    static_pts_q = np.asarray(source_run["query_pts"][qidx], dtype=np.float32).reshape(-1, 3)

    n_pixels = tracks_query.shape[0]
    keep_mask = None
    if conf_track_per_frame and pts_conf_query is not None and pts_conf_query.size == n_pixels:
        agg_track_conf = np.sum(np.stack(conf_track_per_frame, axis=0), axis=0)  # (N,)
        # Per-pixel max temporal displacement (matches 4RC viser line 337-340).
        # Use 4RC tracks (pre-replacement) in WORLD frame for displacement —
        # query_tracks is in query-cam frame and would over-count due to camera
        # motion, but at this point any frame is fine since we just need
        # max_disp scale. Use query frame as reference per viser code.
        pts0 = tracks_query[:, 0, :]
        max_disp = np.zeros(n_pixels, dtype=np.float32)
        for t in range(1, tracks_query.shape[1]):
            disp = np.linalg.norm(tracks_query[:, t, :] - pts0, axis=-1)
            np.maximum(max_disp, disp, out=max_disp)
        dynamic_ratio = float((max_disp > 0.025).mean())

        conf_pct_default = float(os.environ.get("TRACK_CONF_PCT", "2.5"))  # match viser default
        if dynamic_ratio / 2.0 >= conf_pct_default / 100.0:
            effective_pct = conf_pct_default
        else:
            effective_pct = dynamic_ratio / 2.0
        n_keep_track = max(1, int(n_pixels * (100.0 - effective_pct) / 100.0))
        n_keep_pts = max(1, int(n_pixels * (100.0 - conf_pct_default) / 100.0))

        top_track_idx = np.argpartition(-agg_track_conf, n_keep_track - 1)[:n_keep_track]
        track_mask = np.zeros(n_pixels, dtype=bool)
        track_mask[top_track_idx] = True

        top_pts_idx = np.argpartition(-pts_conf_query, n_keep_pts - 1)[:n_keep_pts]
        pts_mask = np.zeros(n_pixels, dtype=bool)
        pts_mask[top_pts_idx] = True

        keep_mask = track_mask & pts_mask  # combined filter
        tracks_query[~keep_mask] = static_pts_q[~keep_mask, None, :]
        n_low = int((~keep_mask).sum())
        print(f"[load_davis_sample_4rc] {sequence.name}: 4RC-viser filter — "
              f"conf_pct={conf_pct_default} dynamic_ratio={dynamic_ratio:.3f} "
              f"effective_pct={effective_pct:.3f} replaced {n_low}/{n_pixels} pixels with static")
    else:
        n_low = 0
        print(f"[load_davis_sample_4rc] {sequence.name}: WARN: missing conf_track / conf_query "
              "in result.npz; track-conf filter skipped (boundary noise will pass through to model)")

    # ---- (A2) OPTIONAL depth-discontinuity filter (off by default) -----------
    # Adapted from Vista4D's get_depths_outliers_pool (utils/point_cloud/filter.py)
    # which is itself "modified from GEN3C". At object boundaries, monocular
    # depth (from any DepthCrafter / pi3 / Depth-Anything style estimator)
    # carries a high local variance — when those pixels are reprojected they
    # become "floaters" that show as black halos around foreground objects.
    # This filter masks pixels whose local depth variance is high, then routes
    # those tracks to the static query-frame point (same handling as the
    # track-conf filter above).
    #
    # Off by default because: (1) 4RC's tracks already pass the conf filter,
    # which in practice covers ~80% of boundary noise, and (2) the depth-
    # outlier filter can be over-eager on thin geometry. Turn on via env:
    #   DEPTH_OUTLIER_FILTER=pool      # GEN3C-style local-range/mean ratio
    #   DEPTH_OUTLIER_FILTER=gaussian  # smooth-then-residual variant
    #   DEPTH_OUTLIER_FILTER=off       # default
    # Knobs (unused unless filter is on):
    #   DEPTH_OUTLIER_WINDOW=5         (pool window size, must be odd)
    #   DEPTH_OUTLIER_THRESHOLD=0.05   (pool ratio threshold)
    #   DEPTH_OUTLIER_SIGMA=4          (gaussian sigma)
    do_filter = os.environ.get("DEPTH_OUTLIER_FILTER", "off").lower()
    if do_filter in ("pool", "gaussian"):
        import torch as _torch
        import torch.nn.functional as _F
        depth_q = source_depth_video[qidx].astype(np.float32)  # (H, W)
        H, W = depth_q.shape
        d = _torch.from_numpy(depth_q)[None, None]  # (1, 1, H, W)
        if do_filter == "pool":
            ws = int(os.environ.get("DEPTH_OUTLIER_WINDOW", "5"))
            assert ws % 2 == 1
            thr = float(os.environ.get("DEPTH_OUTLIER_THRESHOLD", "0.05"))
            eps = 1e-6
            local_max = _F.max_pool2d(d, kernel_size=ws, stride=1, padding=ws // 2)
            local_min = -_F.max_pool2d(-d, kernel_size=ws, stride=1, padding=ws // 2)
            local_mean = _F.avg_pool2d(d, kernel_size=ws, stride=1, padding=ws // 2)
            outlier = ((local_max - local_min) / (local_mean + eps)) >= thr
            outlier = outlier[0, 0].numpy().astype(bool)
        else:  # gaussian
            from scipy.ndimage import gaussian_filter as _gauss
            sigma = float(os.environ.get("DEPTH_OUTLIER_SIGMA", "4"))
            thr = float(os.environ.get("DEPTH_OUTLIER_THRESHOLD", "0.1"))
            smoothed = _gauss(depth_q, sigma=sigma, mode="reflect")
            denom = np.abs(smoothed) + 1e-6
            outlier = (np.abs(depth_q - smoothed) / denom) >= thr
        depth_outlier_mask = outlier.reshape(-1)  # (H*W,) — pixel-aligned with tracks
        if depth_outlier_mask.size == n_pixels:
            n_before = int((~keep_mask).sum()) if keep_mask is not None else 0
            new_keep = (~depth_outlier_mask) if keep_mask is None else (keep_mask & ~depth_outlier_mask)
            tracks_query[~new_keep] = static_pts_q[~new_keep, None, :]
            keep_mask = new_keep
            n_after = int((~keep_mask).sum())
            print(f"[load_davis_sample_4rc] {sequence.name}: depth_outlier_filter={do_filter} "
                  f"replaced +{n_after - n_before}/{n_pixels} pixels with static")
        else:
            print(f"[load_davis_sample_4rc] WARN: depth_outlier mask size {depth_outlier_mask.size} "
                  f"!= n_pixels {n_pixels}; skipping depth filter")

    # ---- (B) Common-scale normalization (matches training:71-95) ----
    finite_mask = np.isfinite(tracks_query).all(axis=-1)
    valid_pts = tracks_query[finite_mask]
    if valid_pts.shape[0] > 0:
        norms = np.linalg.norm(valid_pts, axis=-1)
        norms = norms[norms > 1e-3]
        common_scale = float(norms.mean()) if norms.size > 0 else 1.0
    else:
        common_scale = 1.0
    common_scale = max(common_scale, 1e-6)

    source_cam2world_norm = source_cam2world.copy()
    source_cam2world_norm[:, :3, 3] /= common_scale
    source_depth_video_norm = source_depth_video / common_scale
    tracks_world_norm = tracks_query / common_scale
    raw_tracks_world_norm = raw_tracks_query / common_scale

    source_dynamic_mask = None
    source_dynamic_mask_kind = ""
    if not no_gmos_mask:
        source_dynamic_mask = _load_gmos_source_dynamic_mask(
            gmos_root=str(gmos_root or ""),
            seq=sequence.name,
            mask_subdir=str(gmos_mask_subdir or "mos_frame"),
            height=height,
            width=width,
            num_frames=num_frames,
        )
        if source_dynamic_mask is not None:
            source_dynamic_mask = _morph_mask_video(
                source_dynamic_mask,
                op="dilate",
                radius=int(gmos_dilate_px),
            )
            source_dynamic_mask_kind = "GMOS"
    if source_dynamic_mask is None:
        source_dynamic_mask = _derive_projected_dynamic_mask_from_4rc_tracks(
            tracks_world=raw_tracks_world_norm,
            cam2world=source_cam2world_norm,
            intrinsics=source_intrinsics,
            height=height,
            width=width,
            motion_threshold=float(dynamic_motion_threshold),
        )
        source_dynamic_mask = _dilate_dynamic_mask_video(source_dynamic_mask, int(dynamic_mask_dilate_kernel))
        source_dynamic_mask_kind = "4RC-derived"
    if source_dynamic_mask is not None:
        if source_dynamic_mask_kind == "GMOS":
            print(
                f"[load_davis_sample_4rc] {sequence.name}: GMOS source dynamic "
                f"coverage={float(source_dynamic_mask.mean()):.4f} "
                f"gmos_root={str(gmos_root or '')} "
                f"mask_subdir={str(gmos_mask_subdir or 'mos_frame')} "
                f"dilate_px={int(gmos_dilate_px)}"
            )
        else:
            print(
                f"[load_davis_sample_4rc] {sequence.name}: 4RC-derived source dynamic "
                f"coverage={float(source_dynamic_mask.mean()):.4f} "
                f"threshold={float(dynamic_motion_threshold):.4g} "
                f"dilate_kernel={int(dynamic_mask_dilate_kernel)}"
            )

    source_dynamic_mask_at_tracks = None
    if source_dynamic_mask is not None:
        if source_tracks_xy is not None:
            source_dynamic_mask_at_tracks = _source_mask_at_track_xy(
                source_dynamic_mask=source_dynamic_mask,
                source_xy=source_tracks_xy,
                num_frames=int(tracks_world_norm.shape[1]),
                query_idx_per_identity=source_query_idx_per_identity,
            )
        else:
            source_dynamic_mask_at_tracks = _source_mask_at_4rc_track_grid(
                source_dynamic_mask=source_dynamic_mask,
                tracks_world=tracks_world_norm,
                image_h=height,
                image_w=width,
                track_query_idx=int(source_run["query_idx"]),
            )
    source_tracks_valid_for_sample = source_tracks_valid
    if source_backend == "cow_vggt" and source_dynamic_mask_at_tracks is not None:
        tracks_world_norm = _freeze_static_track_identities(
            tracks_world_norm,
            source_dynamic_mask_at_tracks,
            source_query_idx_per_identity,
        )
        raw_tracks_world_norm = _freeze_static_track_identities(
            raw_tracks_world_norm,
            source_dynamic_mask_at_tracks,
            source_query_idx_per_identity,
        )
        fusion_mode = str(cow_multiquery_fusion_mode).strip().lower()
        if fusion_mode == "legacy_delta":
            source_tracks_valid_for_sample = _dynamic_only_track_valid_mask(
                source_tracks_valid,
                source_dynamic_mask_at_tracks,
                tracks_world_norm.shape[:2],
                source_query_idx_per_identity,
                tracks_world=tracks_world_norm,
            )
        elif fusion_mode == "raw_union":
            source_tracks_valid_for_sample = source_tracks_valid
        elif fusion_mode == "single_anchor":
            source_tracks_valid_for_sample = source_tracks_valid
        else:
            raise ValueError(
                f"unsupported cow_multiquery_fusion_mode={cow_multiquery_fusion_mode!r}"
            )

    print(f"[load_davis_sample_4rc] {sequence.name}: common_scale={common_scale:.4f}  "
          f"track_conf_filter: replaced {n_low}/{(keep_mask.size if keep_mask is not None else 0)} "
          f"low-conf pixels with static query point (matches training's static-bg distribution)")

    base_sample = {
        "video": source_frames,
        "tracks_world": tracks_world_norm,
        "target_intrinsics": source_intrinsics,
        "source_cam2world": source_cam2world_norm,
        "source_intrinsics": source_intrinsics,
        "source_rgb_video": source_rgb_video,
        "source_depth_video": source_depth_video_norm,
        "tracks_source_xy": source_tracks_xy,
        "source_rgb_at_tracks": _source_rgb_at_tracks(
            source_rgb_video=source_rgb_video,
            tracks_world=tracks_world_norm,
            source_xy=source_tracks_xy,
            query_idx_per_identity=source_query_idx_per_identity,
            image_h=height,
            image_w=width,
            track_query_idx=int(source_run["query_idx"]),
        ),
        "common_scale": np.float32(common_scale),
        "track_query_idx": int(source_run["query_idx"]),
        "track_query_indices": source_run.get("track_query_indices", [int(source_run["query_idx"])]),
        "track_query_count": int(source_run.get("track_query_count", 1)),
        "prompt": prompt,
        "has_tracks": True,
        "has_dynamic_tracks": True,
        "has_depth": True,
        "track_video_loss_mask": 1.0,
        "sequence_name": sequence.name,
    }
    if source_tracks_valid_for_sample is not None:
        base_sample["tracks_world_valid_mask"] = source_tracks_valid_for_sample.astype(bool, copy=False)
    if source_query_idx_per_identity is not None:
        base_sample["tracks_query_idx_per_identity"] = np.asarray(
            source_query_idx_per_identity, dtype=np.int32,
        )
    if source_tracks_confidence is not None:
        base_sample["tracks_world_confidence"] = source_tracks_confidence.astype(np.float32, copy=False)
    if source_dynamic_mask is not None:
        base_sample["source_dynamic_mask"] = source_dynamic_mask
        base_sample["source_track_dynamic_mask"] = source_dynamic_mask
        # per-track dynamic-mask channel for ab27/ab28 (codex #3): gather the
        # per-pixel mask on the same 4RC track grid as source_rgb_at_tracks so
        # eval uses the real motion label instead of zero-filling.
        if source_dynamic_mask_at_tracks is not None:
            base_sample["source_dynamic_mask_at_tracks"] = source_dynamic_mask_at_tracks

    source_condition_seconds = time.perf_counter() - source_condition_started
    dynamic_render_started = time.perf_counter()
    sample = build_static_scene_novel_view_sample(
        base_sample=base_sample,
        trajectory_name=trajectory_name,
        angle_deg=angle_deg,
        distance=distance,
        orbit_radius=orbit_radius,
        height=height,
        width=width,
        orbit_start_deg=orbit_start_deg,
    )
    target_dynamic_mask = _derive_projected_dynamic_mask_from_4rc_tracks(
        tracks_world=raw_tracks_world_norm,
        cam2world=np.asarray(sample["target_cam2world"], dtype=np.float32),
        intrinsics=np.asarray(sample["target_intrinsics"], dtype=np.float32),
        height=height,
        width=width,
        motion_threshold=float(dynamic_motion_threshold),
    )
    target_dynamic_mask = _dilate_dynamic_mask_video(target_dynamic_mask, int(dynamic_mask_dilate_kernel))
    if target_dynamic_mask is not None:
        sample["target_dynamic_mask"] = target_dynamic_mask
        print(
            f"[load_davis_sample_4rc] {sequence.name}: 4RC-derived target dynamic "
            f"coverage={float(target_dynamic_mask.mean()):.4f} "
            f"threshold={float(dynamic_motion_threshold):.4g} "
            f"dilate_kernel={int(dynamic_mask_dilate_kernel)}"
        )
    sample["_profile_stage_timings"] = {
        "static_source_condition_seconds": source_condition_seconds,
        "dynamic_render_zbuffer_fusion_seconds": (
            time.perf_counter() - dynamic_render_started
        ),
    }
    sample["sequence_name"] = sequence.name
    return sample


def _split_variants(value: str) -> list[str]:
    return [item for item in re.split(r"[,\s]+", str(value).strip()) if item]


def _variant_to_track_setup(variant: str) -> tuple[str, int]:
    if variant == "q1_dense":
        return "all", 1
    if variant == "q1_s2":
        return "grid", 2
    if variant == "q1_s4":
        return "grid", 4
    raise ValueError(f"Unsupported variant {variant!r}; expected q1_dense/q1_s2/q1_s4")


def _infer_track_grid_hw(num_tracks: int, image_h: int, image_w: int) -> tuple[int, int]:
    if int(num_tracks) == int(image_h) * int(image_w):
        return int(image_h), int(image_w)
    target_aspect = float(image_h) / max(float(image_w), 1.0)
    best = None
    for h_t in range(1, int(num_tracks) + 1):
        if int(num_tracks) % h_t != 0:
            continue
        w_t = int(num_tracks) // h_t
        err = abs((float(h_t) / max(float(w_t), 1.0)) - target_aspect)
        if best is None or err < best[0]:
            best = (err, h_t, w_t)
    if best is None:
        raise ValueError(f"Cannot infer 2D track grid for N={num_tracks}")
    return int(best[1]), int(best[2])


def apply_davis_track_condition_variant(
    sample: dict,
    *,
    track_sample_mode: str,
    track_stride: int,
    seed: int,
) -> None:
    """Apply q1 dense/sparse validity masks to a DAVIS+4RC sample in place."""
    tracks = sample.get("tracks_world")
    if not isinstance(tracks, np.ndarray) or tracks.ndim != 3:
        return
    if str(track_sample_mode) == "all" or int(track_stride) <= 1:
        dense_valid = sample.get("tracks_world_valid_mask")
        if isinstance(dense_valid, np.ndarray) and dense_valid.shape == tracks.shape[:2]:
            sample["tracks_world_dense_valid_mask"] = dense_valid.astype(bool, copy=False)
        sample["track_density_mode"] = "dense"
        sample["track_sparse_stride"] = 1
        return

    h_t, w_t = _infer_track_grid_hw(int(tracks.shape[0]), int(sample.get("height", 0) or 0), int(sample.get("width", 0) or 0))
    finite_valid = np.isfinite(tracks).all(axis=-1) & (np.abs(tracks).sum(axis=-1) > 1.0e-8)
    dense_valid = sample.get("tracks_world_valid_mask")
    if isinstance(dense_valid, np.ndarray) and dense_valid.shape == finite_valid.shape:
        dense_valid = dense_valid.astype(bool, copy=True) & finite_valid
    else:
        dense_valid = finite_valid.astype(bool, copy=True)

    from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.track4d360_mixed_nvs import (
        _build_sparse_track_valid_mask,
    )

    sparse_valid = _build_sparse_track_valid_mask(
        h_t,
        w_t,
        int(tracks.shape[1]),
        stride=int(track_stride),
        rng=np.random.default_rng(int(seed)),
    )
    sparse_valid &= dense_valid
    sample["tracks_world_dense_valid_mask"] = dense_valid.astype(bool, copy=False)
    sample["tracks_world_valid_mask"] = sparse_valid.astype(bool, copy=False)
    sample["track_density_mode"] = "sparse"
    sample["track_sparse_stride"] = int(track_stride)
    sample["track_sparse_base_stride"] = int(track_stride)


def _load_work_items(args):
    if args.pairs_csv:
        pairs = []
        with open(args.pairs_csv, newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                seq = (row.get("video") or row.get("sequence") or "").strip()
                traj = (row.get("trajectory") or row.get("traj") or "").strip()
                if not seq or not traj:
                    continue
                pairs.append((seq, traj))
        unique_seqs = sorted({s for s, _ in pairs})
        sequences = resolve_davis_sequences(
            args.davis_root,
            sequence_names=",".join(unique_seqs),
            num_frames=args.num_frames,
            start_frame=args.start_frame,
        )
        seq_by_name = {s.name: s for s in sequences}
        work = [(seq_by_name[seq], traj) for (seq, traj) in pairs if seq in seq_by_name]
        print(f"[DAVIS4RC] batched mode: {len(work)} pairs from {args.pairs_csv}")
        return work

    trajectory_names = parse_static_scene_trajectories(args.trajectories)
    sequences = resolve_davis_sequences(
        args.davis_root,
        sequence_names=args.sequence_names,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
    )
    print(f"[DAVIS4RC] cartesian mode: {len(sequences)} sequences × "
          f"{len(trajectory_names)} trajectories = {len(sequences) * len(trajectory_names)} pairs")
    return [(seq, traj) for seq in sequences for traj in trajectory_names]


def _maybe_print_source_rgb_track_debug(sample: dict, *, sequence_name: str, trajectory_name: str) -> None:
    if os.environ.get("TRACK4D360_DEBUG_SOURCE_RGB", "0") != "1":
        return
    rgb = sample.get("source_rgb_at_tracks")
    if rgb is None:
        print(
            f"[DAVIS4RC/debug] {sequence_name}/{trajectory_name}: "
            "source_rgb_at_tracks=None",
            flush=True,
        )
        return
    arr = np.asarray(rgb, dtype=np.float32)
    print(
        f"[DAVIS4RC/debug] {sequence_name}/{trajectory_name}: "
        f"source_rgb_at_tracks shape={arr.shape} "
        f"min={float(arr.min(initial=0.0)):.6f} "
        f"max={float(arr.max(initial=0.0)):.6f} "
        f"mean={float(arr.mean()):.6f}",
        flush=True,
    )


def _last_video_frame(path: Path) -> PIL.Image.Image:
    last = None
    reader = imageio.get_reader(path)
    try:
        for frame in reader:
            last = np.asarray(frame, dtype=np.uint8)
    finally:
        reader.close()
    if last is None:
        raise RuntimeError(f"No decodable frames in previous target video: {path}")
    return PIL.Image.fromarray(last, mode="RGB")


def _prepare_i2v_boundary_condition(
    sample: dict,
    args,
    *,
    sequence_name: str,
    trajectory_name: str,
) -> dict | None:
    mode = str(getattr(args, "i2v_condition_mode", "off"))
    if mode == "off":
        return None

    if mode == "source_first":
        source_c2w = np.asarray(sample["source_cam2world"], dtype=np.float32)
        target_c2w = np.asarray(sample["target_cam2world"], dtype=np.float32)
        source_k = np.asarray(sample["source_intrinsics"], dtype=np.float32)
        target_k = np.asarray(sample["target_intrinsics"], dtype=np.float32)
        if not np.allclose(source_c2w[0], target_c2w[0], atol=1e-5, rtol=1e-5):
            raise ValueError(
                "source_first I2V conditioning requires target camera frame 0 "
                "to equal source camera frame 0"
            )
        target_k0 = target_k[0] if target_k.ndim == 3 else target_k
        source_k0 = source_k[0] if source_k.ndim == 3 else source_k
        if not np.allclose(source_k0, target_k0, atol=1e-4, rtol=1e-5):
            raise ValueError(
                "source_first I2V conditioning requires matching frame-0 intrinsics"
            )
        image = sample["video"][0]
        metadata = {
            "mode": mode,
            "source": "sample.video[0]",
            "camera_alignment_checked": True,
        }
    elif mode == "previous_target":
        root_value = getattr(args, "i2v_previous_target_root", None)
        if not root_value:
            raise ValueError(
                "--i2v_condition_mode previous_target requires "
                "--i2v_previous_target_root"
            )
        pair_root = Path(root_value) / sequence_name / trajectory_name
        candidates = (
            pair_root / "last_target.png",
            pair_root / "pred_rgb.mp4",
            pair_root / f"{trajectory_name}.mp4",
        )
        condition_path = next((path for path in candidates if path.is_file()), None)
        if condition_path is None:
            raise FileNotFoundError(
                f"No previous target condition found under {pair_root}; tried "
                + ", ".join(path.name for path in candidates)
            )
        if condition_path.suffix.lower() == ".png":
            with PIL.Image.open(condition_path) as opened:
                image = opened.convert("RGB")
        else:
            image = _last_video_frame(condition_path)
        metadata = {
            "mode": mode,
            "source": str(condition_path),
            "camera_alignment_checked": False,
        }
    else:
        raise ValueError(f"Unsupported I2V condition mode: {mode}")

    sample["inference_input_image"] = image
    sample["use_target_image_condition"] = True
    return metadata


def _run_work(
    *,
    pipe,
    args,
    output_root: Path,
    work,
    captions_by_sequence: dict[str, str],
    variant: str,
    track_sample_mode: str,
    track_stride: int,
) -> tuple[int, int, int]:
    variant_args = vars(args).copy()
    variant_args.update(
        variant=variant,
        track_sample_mode=track_sample_mode,
        track_stride=track_stride,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "args.json").write_text(json.dumps(variant_args, indent=2, sort_keys=True))

    n_total = len(work)
    n_ok = n_skip = n_fail = 0
    for idx, (sequence, trajectory_name) in enumerate(work, 1):
        pair_out_dir = output_root / sequence.name / trajectory_name
        out_mp4 = pair_out_dir / f"{trajectory_name}.mp4"
        if out_mp4.exists() and not args.force_rerun:
            print(
                f"[DAVIS4RC] [{variant}] [{idx}/{n_total}] "
                f"skip {sequence.name}/{trajectory_name}: pred exists"
            )
            n_skip += 1
            continue

        try:
            sample = load_davis_sample_4rc(
                davis_root=args.davis_root,
                davis_4rc_root=args.davis_4rc_root,
                sequence_name=sequence.name,
                start_frame=args.start_frame,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                trajectory_name=trajectory_name,
                angle_deg=args.angle_deg,
                distance=args.distance,
                orbit_radius=args.orbit_radius,
                orbit_start_deg=float(getattr(args, "orbit_start_deg", 0.0)),
                dynamic_motion_threshold=float(getattr(args, "dense_warp_motion_threshold", 0.005)),
                dynamic_mask_dilate_kernel=int(getattr(args, "dynamic_mask_dilate_kernel", 0)),
                gmos_root=str(getattr(args, "gmos_root", "") or ""),
                gmos_mask_subdir=str(getattr(args, "gmos_mask_subdir", "mos_frame") or "mos_frame"),
                gmos_dilate_px=int(getattr(args, "gmos_dilate_px", 0)),
                no_gmos_mask=bool(getattr(args, "no_gmos_mask", False)),
                prompt=captions_by_sequence.get(sequence.name, DEFAULT_PROMPT),
                source_backend=str(getattr(args, "source_backend", "4rc")),
                cow_root=getattr(args, "cow_root", None),
                vggt_root=getattr(args, "vggt_root", None),
                cow_track_sample_mode=str(getattr(args, "cow_track_sample_mode", "all")),
                cow_track_stride=int(getattr(args, "cow_track_stride", 1)),
                cow_conf_threshold=float(getattr(args, "cow_conf_threshold", 0.5)),
                cow_depth_conf_threshold=float(getattr(args, "cow_depth_conf_threshold", 0.0)),
                cow_multiquery_fusion_mode=str(
                    getattr(args, "cow_multiquery_fusion_mode", "legacy_delta")
                ),
                cow_single_query_anchor=str(
                    getattr(args, "cow_single_query_anchor", "last")
                ),
                seed=int(args.seed),
            )
            sample["height"] = int(args.height)
            sample["width"] = int(args.width)
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
            else:
                apply_davis_track_condition_variant(
                    sample,
                    track_sample_mode=track_sample_mode,
                    track_stride=track_stride,
                    seed=int(args.seed) + idx - 1,
                )

            print(
                f"[DAVIS4RC] [{variant}] [{idx}/{n_total}] running {sequence.name}/{trajectory_name} "
                f"track_mode={sample.get('track_density_mode', 'dense')} "
                f"track_stride={sample.get('track_sparse_stride', 1)} "
                f"drop_4rc_tracks={bool(args.drop_4rc_tracks)} "
                f"null_4rc_tracks={bool(args.null_4rc_tracks)} "
                f"caption={'json' if sequence.name in captions_by_sequence else 'default'}"
            )
            _maybe_print_source_rgb_track_debug(
                sample,
                sequence_name=sequence.name,
                trajectory_name=trajectory_name,
            )
            i2v_metadata = _prepare_i2v_boundary_condition(
                sample,
                args,
                sequence_name=sequence.name,
                trajectory_name=trajectory_name,
            )
            if i2v_metadata is not None:
                pair_out_dir.mkdir(parents=True, exist_ok=True)
                (pair_out_dir / "i2v_condition.json").write_text(
                    json.dumps(i2v_metadata, indent=2, sort_keys=True) + "\n"
                )
            _run_one_pair(pipe, sample, args, pair_out_dir)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 — keep the batch going on per-pair errors
            n_fail += 1
            import traceback as _tb
            print(
                f"[DAVIS4RC] [{variant}] [{idx}/{n_total}] FAILED "
                f"{sequence.name}/{trajectory_name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _tb.print_exc()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"[DAVIS4RC] complete {variant} -> {output_root}  (ok={n_ok} skip={n_skip} fail={n_fail} total={n_total})")
    return n_ok, n_skip, n_fail


def main():
    import torch

    args = parse_args()
    if getattr(args, "use_fsdp", False):
        from holo4d.eval_fsdp_helper import init_fsdp_early
        init_fsdp_early(args)

    torch.manual_seed(args.seed)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    from holo4d.eval_fsdp_helper import is_main_process

    if is_main_process():
        (output_root / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    # Build the (sequence, trajectory) work list.  In --pairs_csv mode this is
    # the canonical DAVIS50 x 2-trajectory benchmark; otherwise it is the legacy
    # cartesian sequence x trajectory path.
    work = _load_work_items(args)

    captions_by_sequence = load_davis_captions(args.davis_captions_json)
    work_sequence_names = sorted({sequence.name for sequence, _ in work})
    if args.require_davis_caption:
        missing_captions = [
            name for name in work_sequence_names if name not in captions_by_sequence
        ]
        if missing_captions:
            preview = ", ".join(missing_captions[:10])
            raise RuntimeError(
                f"Missing DAVIS captions for {len(missing_captions)} selected sequences: {preview}"
            )

    # Load the model ONCE. Both modes share this.
    pipe = build_eval_pipeline(args)
    _summarize_vista4d_plucker_modules(pipe, args)
    prompts_to_cache = [DEFAULT_PROMPT, NEGATIVE_PROMPT]
    prompts_to_cache.extend(
        captions_by_sequence[name]
        for name in work_sequence_names
        if name in captions_by_sequence
    )
    precompute_eval_prompt_embs(pipe, prompts_to_cache)

    variants = _split_variants(args.variant_sweep)
    if variants:
        totals = []
        for variant in variants:
            mode, stride = _variant_to_track_setup(variant)
            totals.append(
                (variant,) + _run_work(
                    pipe=pipe,
                    args=args,
                    output_root=output_root / variant,
                    work=work,
                    captions_by_sequence=captions_by_sequence,
                    variant=variant,
                    track_sample_mode=mode,
                    track_stride=stride,
                )
            )
        print(f"[DAVIS4RC] all variants complete -> {output_root}  totals={totals}")
    else:
        _run_work(
            pipe=pipe,
            args=args,
            output_root=output_root,
            work=work,
            captions_by_sequence=captions_by_sequence,
            variant="single",
            track_sample_mode=args.track_sample_mode,
            track_stride=args.track_stride,
        )


if __name__ == "__main__":
    main()
