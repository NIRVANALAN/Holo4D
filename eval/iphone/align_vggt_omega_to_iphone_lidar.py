"""Scale-align cached VGGT-Omega iPhone source depth to source LIDAR.

This writes a separate VGGT-Omega alpha record and never overwrites the 4RC
alpha used by the headline source-geometry path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SEQS = ("apple", "block", "paper-windmill", "spin", "teddy")


def _load_lidar(conditions_root: Path, seq: str) -> np.ndarray:
    path = conditions_root / "conditions" / seq / "src_depth_1x" / "000.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing source LIDAR depth: {path}")
    depth = np.load(path).astype(np.float32)
    return depth[..., 0] if depth.ndim == 3 and depth.shape[-1] == 1 else depth


def _align_one(seq: str, output_root: Path, vggt_root: Path, min_lidar_m: float, min_depth: float) -> dict:
    depth_path = vggt_root / seq / "geometry" / "vggt_omega_depth.npz"
    if not depth_path.exists():
        raise FileNotFoundError(f"missing VGGT-Omega depth cache: {depth_path}")
    data = np.load(depth_path)
    depth = np.asarray(data["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    source_depth = depth[0]
    lidar = _load_lidar(output_root, seq)
    lidar = cv2.resize(lidar, (source_depth.shape[1], source_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(lidar) & np.isfinite(source_depth) & (lidar > min_lidar_m) & (source_depth > min_depth)
    if not valid.any():
        raise RuntimeError(f"no valid LIDAR/VGGT-Omega depth overlap for {seq}")
    ratios = lidar[valid] / source_depth[valid]
    alpha = float(np.median(ratios))
    record = {
        "alpha_vggt_omega_to_lidar": alpha,
        "n_valid_pixels": int(valid.sum()),
        "lidar_median_m": float(np.median(lidar[valid])),
        "vggt_omega_depth_median": float(np.median(source_depth[valid])),
        "vggt_omega_hw": [int(source_depth.shape[0]), int(source_depth.shape[1])],
        "ratio_p25_p50_p75": [float(np.quantile(ratios, q)) for q in (0.25, 0.5, 0.75)],
    }
    out_path = output_root / "conditions" / seq / "vggt_omega_depth_scale_alpha.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"[{seq}] alpha={alpha:.4f} n_valid={record['n_valid_pixels']} -> {out_path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("outputs/benchmark/iphone_nvs"))
    parser.add_argument("--iphone_vggt_omega_root", type=Path, required=True)
    parser.add_argument("--sequences", type=str, default=",".join(DEFAULT_SEQS))
    parser.add_argument("--min_lidar_m", type=float, default=0.05)
    parser.add_argument("--min_vggt_depth", type=float, default=1e-3)
    args = parser.parse_args()
    records = {}
    for seq in (item.strip() for item in args.sequences.split(",") if item.strip()):
        records[seq] = _align_one(seq, args.output_root, args.iphone_vggt_omega_root, args.min_lidar_m, args.min_vggt_depth)
    out_path = args.output_root / "vggt_omega_depth_scale_alpha_summary.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"Summary -> {out_path}")


if __name__ == "__main__":
    main()
