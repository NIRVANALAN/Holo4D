"""Compute the per-seq scale α that brings 4RC depth/tracks/cam-translations
into iPhone's metric (LIDAR meters) frame.

Why: 4RC is an SfM-style reconstruction without absolute scale. Its outputs
(`pred_<i>_pts`, `pred_<i>_track`, `pred_<i>_extrinsic` translations) live in
an arbitrary unit. iPhone GT cameras + LIDAR depth are in metric meters. For
fair NVS evaluation we need 4RC source geometry expressed in the SAME unit as
the GT target cameras — otherwise the model sees a "1m camera shift" condition
mapped to a "0.07 4RC-unit" target trajectory and renders garbage. This script
estimates α := median(lidar / 4RC_depth) over valid pixels and saves it as
`conditions/<seq>/depth_scale_alpha.json` so the inference adapter can
multiply 4RC tracks / depth / cam-translations through.

Usage:
    python eval/iphone/align_4rc_to_iphone_lidar.py \
        --output_root outputs/iphone_nvs \
        --iphone_4rc_root outputs/iphone_nvs/4RC

Expected 4RC layout (mirrors the DAVIS bench's `<BASELINES_ROOT>/4RC/davis_49f/`):
    <iphone_4rc_root>/<seq>/result.npz                         # 4RC tracks/pts/cam outputs
    <iphone_4rc_root>/<seq>/metadata.json                      # 4RC's run metadata

Output:
    conditions/<seq>/depth_scale_alpha.json
        {
            "alpha_4rc_to_lidar": <float>,
            "n_valid_pixels": <int>,
            "lidar_median_m":   <float>,
            "rc_depth_median":  <float>,
            "rc_query_idx":     <int>,
        }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SEQS = ("apple", "block", "paper-windmill", "spin", "teddy")


def _load_4rc_query_depth(npz_path: Path) -> tuple[np.ndarray, int]:
    """Return (depth_4rc (H_rc, W_rc), query_idx)."""
    data = np.load(npz_path, allow_pickle=True)
    qidx = int(np.asarray(data["pred_0_track_query_idx"]).reshape(-1)[0])
    pts_q = np.asarray(data[f"pred_{qidx}_pts"], dtype=np.float32)
    if pts_q.ndim == 4 and pts_q.shape[0] == 1:
        pts_q = pts_q[0]
    extr = np.asarray(data[f"pred_{qidx}_extrinsic"], dtype=np.float32)
    if extr.ndim == 3 and extr.shape[0] == 1:
        extr = extr[0]
    pts_flat = pts_q.reshape(-1, 3)
    pts_world_h = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1), dtype=np.float32)], axis=-1)
    w2c = np.linalg.inv(extr)
    pts_cam = (w2c @ pts_world_h.T).T[:, :3]
    depth = pts_cam[:, 2].reshape(pts_q.shape[:2]).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth, qidx


def _load_lidar_at_query(conditions_root: Path, seq: str, qidx: int) -> np.ndarray:
    """Load LIDAR depth from conditions/<seq>/src_depth_1x/<qidx:03d>.npy."""
    path = conditions_root / "conditions" / seq / "src_depth_1x" / f"{qidx:03d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing lidar depth for {seq} qidx={qidx}: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _align_one_seq(
    seq: str,
    conditions_root: Path,
    iphone_4rc_root: Path,
    min_lidar_m: float,
    min_rc_depth: float,
) -> dict:
    rc_path = iphone_4rc_root / seq / "result.npz"
    if not rc_path.exists():
        raise FileNotFoundError(f"4RC result.npz missing for {seq}: {rc_path}")
    rc_depth, qidx = _load_4rc_query_depth(rc_path)
    H_rc, W_rc = rc_depth.shape
    lidar = _load_lidar_at_query(conditions_root, seq, qidx)
    if lidar.shape != (H_rc, W_rc):
        lidar_resized = cv2.resize(lidar, (W_rc, H_rc), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    else:
        lidar_resized = lidar

    valid = (lidar_resized > min_lidar_m) & (rc_depth > min_rc_depth) & np.isfinite(lidar_resized) & np.isfinite(rc_depth)
    if not valid.any():
        raise RuntimeError(f"no overlapping valid lidar/4RC depth pixels for {seq}")

    ratios = lidar_resized[valid] / rc_depth[valid]
    alpha = float(np.median(ratios))
    out = {
        "alpha_4rc_to_lidar": float(alpha),
        "n_valid_pixels": int(valid.sum()),
        "lidar_median_m": float(np.median(lidar_resized[valid])),
        "rc_depth_median": float(np.median(rc_depth[valid])),
        "rc_query_idx": int(qidx),
        "rc_hw": [int(H_rc), int(W_rc)],
        "min_lidar_m": float(min_lidar_m),
        "min_rc_depth": float(min_rc_depth),
        "ratio_p25_p50_p75": [float(np.quantile(ratios, q)) for q in (0.25, 0.5, 0.75)],
    }
    out_path = conditions_root / "conditions" / seq / "depth_scale_alpha.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{seq}] α={alpha:.4f}  n_valid={out['n_valid_pixels']}  "
          f"ratio p25/p50/p75={out['ratio_p25_p50_p75']}  -> {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=str,
                   default="outputs/benchmark/iphone_nvs")
    p.add_argument("--iphone_4rc_root", type=str, required=True,
                   help="Root containing <seq>/result.npz from 4RC inference.")
    p.add_argument("--sequences", type=str, default=",".join(DEFAULT_SEQS))
    p.add_argument("--min_lidar_m", type=float, default=0.05,
                   help="Drop LIDAR pixels below this many meters (sensor noise floor).")
    p.add_argument("--min_rc_depth", type=float, default=1e-3)
    args = p.parse_args()

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    output_root = Path(args.output_root)
    iphone_4rc_root = Path(args.iphone_4rc_root)

    summary = {}
    for seq in seqs:
        summary[seq] = _align_one_seq(
            seq=seq,
            conditions_root=output_root,
            iphone_4rc_root=iphone_4rc_root,
            min_lidar_m=args.min_lidar_m,
            min_rc_depth=args.min_rc_depth,
        )

    sum_path = output_root / "depth_scale_alpha_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {sum_path}")


if __name__ == "__main__":
    main()
