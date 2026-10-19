"""Build iPhone NVS conditions + GT trees from raw Shape-of-Motion data.

One-off, deterministic. Reads:
    data/iphone/<seq>/
        rgb/{1x,2x}/<f>.png
        depth/1x/<f>.npy                                 # lidar, meters
        flow3d_preprocessed/colmap/sparse/{cameras,images}.bin
        flow3d_preprocessed/track_anything/2x/<f>.png    # fg masks
        covisible/2x/{train,val}/<f>.png
        splits/{train,val}.json
        extra.json

Writes (all paths under --output_root):
    conditions/<seq>/
        src_video_1x.mp4                          # cam0 49 frames @ 720x960
        src_frames_1x/<idx>.png                   # cam0 RGB (no alpha) @ 720x960
        src_depth_1x/<idx>.npy                    # cam0 lidar depth (H,W) float32
        src_cameras.npz {Ks_1x:(49,3,3), w2cs:(49,4,4), c2ws:(49,4,4), src_hw:(2,)}
        tgt_cameras.npz {Ks_1x, w2cs, c2ws, tgt_hw, tgt_camera_id, time_ids}
        metadata.json
    gt/<seq>/
        target_frames_2x/<idx>.png               # cam{1|2} GT @ 360x480 (no alpha)
        covisible_2x/<idx>.png                   # binary covisible masks
        fg_mask_2x/<idx>.png                     # foreground masks (track_anything)
        valid_mask_2x/<idx>.png                  # alpha channel of source RGBA

Per-seq decisions match doc/evaluation/iphone_nvs_benchmark.md §2:
    apple          -> tgt_cam=2, t_start=0
    block          -> tgt_cam=1, t_start=0
    paper-windmill -> tgt_cam=1, t_start=0
    spin           -> tgt_cam=1, t_start=0
    teddy          -> tgt_cam=1, t_start=0

Run:
    python eval/iphone/build_iphone_pairs.py \
        --iphone_root data/iphone \
        --output_root outputs/benchmark/iphone_nvs

Idempotent — skips per-seq if metadata.json already exists, unless --force.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from colmap_io import get_colmap_camera_params  # noqa: E402

DEFAULT_SEQS = ("apple", "block", "paper-windmill", "spin", "teddy")
NUM_FRAMES = 49

# {seq: (target_camera_id, t_start)} — see doc/evaluation/iphone_nvs_benchmark.md §2
TARGET_PLAN: dict[str, tuple[int, int]] = {
    "apple":          (2, 0),
    "block":          (1, 0),
    "paper-windmill": (1, 0),
    "spin":           (1, 0),
    "teddy":          (1, 0),
}


def _load_split_frames(seq_dir: Path, split: str) -> dict:
    with (seq_dir / "splits" / f"{split}.json").open() as f:
        return json.load(f)


def _select_window(split_dict: dict, camera_id: int, t_start: int, n: int) -> list[str]:
    """Return n contiguous frame_names from camera `camera_id` starting at t_start.

    Asserts contiguity in time_ids.
    """
    by_t: dict[int, str] = {}
    for cid, tid, name in zip(
        split_dict["camera_ids"], split_dict["time_ids"], split_dict["frame_names"]
    ):
        if cid == camera_id:
            by_t[int(tid)] = name
    out = []
    for t in range(t_start, t_start + n):
        if t not in by_t:
            raise RuntimeError(
                f"camera {camera_id} missing frame at time_id={t} "
                f"(have time_ids: {sorted(by_t)[:10]}…)"
            )
        out.append(by_t[t])
    return out


def _load_lidar_depth(seq_dir: Path, frame_name: str, factor: int = 1) -> np.ndarray:
    """Load lidar depth at full resolution. Returns (H, W) float32, meters."""
    arr = np.load(seq_dir / "depth" / f"{factor}x" / f"{frame_name}.npy")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _load_rgba(path: Path) -> np.ndarray:
    """Load RGBA image; returns (H, W, 4) uint8."""
    arr = iio.imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3 + [np.full_like(arr, 255)], axis=-1)
    elif arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=-1)
    return arr.astype(np.uint8)


def _save_video_mp4(frames_uint8_rgb: np.ndarray, out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames_uint8_rgb, fps=float(fps), codec="libx264")


def _build_seq(
    seq: str,
    iphone_root: Path,
    output_root: Path,
    force: bool,
) -> dict:
    seq_dir = iphone_root / seq
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"missing seq dir: {seq_dir}")

    cond_dir = output_root / "conditions" / seq
    gt_dir = output_root / "gt" / seq
    meta_path = cond_dir / "metadata.json"
    if meta_path.exists() and not force:
        print(f"[{seq}] already built — skipping (use --force to rebuild). {meta_path}")
        return json.loads(meta_path.read_text())

    tgt_cam_id, t_start = TARGET_PLAN[seq]
    train = _load_split_frames(seq_dir, "train")
    val = _load_split_frames(seq_dir, "val")

    src_names = _select_window(train, camera_id=0, t_start=t_start, n=NUM_FRAMES)
    tgt_names = _select_window(val, camera_id=tgt_cam_id, t_start=t_start, n=NUM_FRAMES)

    print(f"[{seq}] tgt_cam={tgt_cam_id} t_start={t_start} "
          f"src[0,-1]={src_names[0]},{src_names[-1]} "
          f"tgt[0,-1]={tgt_names[0]},{tgt_names[-1]}")

    # ---------- COLMAP cameras (refined) ----------
    colmap_dir = seq_dir / "flow3d_preprocessed" / "colmap" / "sparse"
    if not (colmap_dir / "cameras.bin").exists():
        raise FileNotFoundError(f"COLMAP cameras.bin not found: {colmap_dir}")

    src_K_all, src_w2c_all = get_colmap_camera_params(
        str(colmap_dir), [f"{n}.png" for n in src_names]
    )
    tgt_K_all, tgt_w2c_all = get_colmap_camera_params(
        str(colmap_dir), [f"{n}.png" for n in tgt_names]
    )
    src_Ks_1x = src_K_all[:, :3, :3].astype(np.float32)
    tgt_Ks_1x = tgt_K_all[:, :3, :3].astype(np.float32)
    src_w2cs = src_w2c_all.astype(np.float32)
    tgt_w2cs = tgt_w2c_all.astype(np.float32)
    src_c2ws = np.linalg.inv(src_w2cs).astype(np.float32)
    tgt_c2ws = np.linalg.inv(tgt_w2cs).astype(np.float32)

    # ---------- Source RGB (1x, drop alpha for mp4) + lidar depth ----------
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / "src_frames_1x").mkdir(parents=True, exist_ok=True)
    (cond_dir / "src_depth_1x").mkdir(parents=True, exist_ok=True)

    src_rgbs = []
    src_depths = []
    for i, name in enumerate(src_names):
        rgba = _load_rgba(seq_dir / "rgb" / "1x" / f"{name}.png")
        rgb = rgba[..., :3]
        iio.imwrite(cond_dir / "src_frames_1x" / f"{i:03d}.png", rgb)
        depth = _load_lidar_depth(seq_dir, name, factor=1)
        if depth.shape[:2] != rgb.shape[:2]:
            raise RuntimeError(
                f"depth shape {depth.shape} != rgb shape {rgb.shape[:2]} for {name}"
            )
        np.save(cond_dir / "src_depth_1x" / f"{i:03d}.npy", depth)
        src_rgbs.append(rgb)
        src_depths.append(depth)
    src_rgbs_np = np.stack(src_rgbs, axis=0)
    src_h, src_w = src_rgbs_np.shape[1:3]

    # FPS from extra.json
    with (seq_dir / "extra.json").open() as f:
        fps = float(json.load(f).get("fps", 30.0))
    _save_video_mp4(src_rgbs_np, cond_dir / "src_video_1x.mp4", fps=fps)

    np.savez(
        cond_dir / "src_cameras.npz",
        Ks_1x=src_Ks_1x,
        w2cs=src_w2cs,
        c2ws=src_c2ws,
        src_hw=np.array([src_h, src_w], dtype=np.int32),
        time_ids=np.arange(t_start, t_start + NUM_FRAMES, dtype=np.int32),
    )

    np.savez(
        cond_dir / "tgt_cameras.npz",
        Ks_1x=tgt_Ks_1x,
        w2cs=tgt_w2cs,
        c2ws=tgt_c2ws,
        tgt_hw=np.array([src_h, src_w], dtype=np.int32),  # iPhone cams share intrinsics resolution
        tgt_camera_id=np.int32(tgt_cam_id),
        time_ids=np.arange(t_start, t_start + NUM_FRAMES, dtype=np.int32),
    )

    # ---------- GT @ 2x (target rgb, covisible, fg, valid alpha) ----------
    (gt_dir / "target_frames_2x").mkdir(parents=True, exist_ok=True)
    (gt_dir / "covisible_2x").mkdir(parents=True, exist_ok=True)
    (gt_dir / "fg_mask_2x").mkdir(parents=True, exist_ok=True)
    (gt_dir / "valid_mask_2x").mkdir(parents=True, exist_ok=True)

    cov_dir = seq_dir / "covisible" / "2x" / "val"
    fg_dir = seq_dir / "flow3d_preprocessed" / "track_anything" / "2x"
    rgb2x_dir = seq_dir / "rgb" / "2x"

    for i, name in enumerate(tgt_names):
        rgba = _load_rgba(rgb2x_dir / f"{name}.png")
        rgb = rgba[..., :3]
        valid = rgba[..., 3]  # 0 or 255
        iio.imwrite(gt_dir / "target_frames_2x" / f"{i:03d}.png", rgb)
        iio.imwrite(gt_dir / "valid_mask_2x" / f"{i:03d}.png", valid)
        cov_path = cov_dir / f"{name}.png"
        if not cov_path.exists():
            raise FileNotFoundError(f"missing covisible mask: {cov_path}")
        cov = iio.imread(cov_path)
        if cov.ndim == 3:
            cov = cov[..., 0]
        iio.imwrite(gt_dir / "covisible_2x" / f"{i:03d}.png", cov.astype(np.uint8))
        fg_path = fg_dir / f"{name}.png"
        if fg_path.exists():
            fg = iio.imread(fg_path)
            if fg.ndim == 3:
                fg = fg[..., 0]
            iio.imwrite(gt_dir / "fg_mask_2x" / f"{i:03d}.png", fg.astype(np.uint8))

    # ---------- metadata ----------
    meta = {
        "sequence": seq,
        "num_frames": NUM_FRAMES,
        "tgt_camera_id": int(tgt_cam_id),
        "t_start": int(t_start),
        "time_ids": list(range(t_start, t_start + NUM_FRAMES)),
        "src_frame_names": src_names,
        "tgt_frame_names": tgt_names,
        "src_hw_1x": [int(src_h), int(src_w)],
        "fps": fps,
        "iphone_root": str(iphone_root),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{seq}] -> {cond_dir}  +  {gt_dir}  ({NUM_FRAMES} frames, fps={fps})")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iphone_root", type=str,
                   default="data/iphone")
    p.add_argument("--output_root", type=str,
                   default="outputs/benchmark/iphone_nvs")
    p.add_argument("--sequences", type=str, default=",".join(DEFAULT_SEQS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    iphone_root = Path(args.iphone_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    all_meta = {}
    for seq in seqs:
        if seq not in TARGET_PLAN:
            raise ValueError(f"unknown seq {seq}; expected one of {DEFAULT_SEQS}")
        all_meta[seq] = _build_seq(seq, iphone_root, output_root, args.force)

    summary_path = output_root / "iphone_nvs_index.json"
    summary_path.write_text(json.dumps({
        "sequences": seqs,
        "num_frames": NUM_FRAMES,
        "per_seq": all_meta,
    }, indent=2))
    print(f"Index -> {summary_path}")


if __name__ == "__main__":
    main()
