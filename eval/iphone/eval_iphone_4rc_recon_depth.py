#!/usr/bin/env python3
"""Evaluate 4RC reconstructed depth for iPhone NVS generated videos.

This compares each method's 4RC reconstruction against the 4RC reconstruction
of the GT target video. It intentionally uses Geo4D's depth_eval.py metric
function so the reported depth metrics match that codepath:

    outputs/baselines/Geo4D/dust3r/depth_eval.py::depth_evaluation

The metric is a proxy: GT depth here is pseudo-GT from 4RC-on-target-RGB, not
Shape-of-Motion target LIDAR depth (iPhone only ships cam0/source depth).
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.optimize import minimize


def load_depth_eval(geo4d_repo: Path):
    depth_eval_path = geo4d_repo / "dust3r" / "depth_eval.py"
    if not depth_eval_path.exists():
        raise FileNotFoundError(depth_eval_path)
    if str(geo4d_repo) not in sys.path:
        sys.path.insert(0, str(geo4d_repo))
    spec = importlib.util.spec_from_file_location("geo4d_depth_eval", depth_eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {depth_eval_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.depth_evaluation
    except ModuleNotFoundError as exc:
        print(
            f"[depth] normal Geo4D depth_eval import failed ({exc}); "
            "selectively loading metric functions only."
        )

    source = depth_eval_path.read_text()
    tree = ast.parse(source, filename=str(depth_eval_path))
    keep = {
        "depth2disparity",
        "absolute_error_loss",
        "absolute_value_scaling",
        "absolute_value_scaling2",
        "depth_evaluation",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in keep
    ]
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    namespace = {"torch": torch, "np": np, "minimize": minimize}
    exec(compile(mod, str(depth_eval_path), "exec"), namespace)
    return namespace["depth_evaluation"]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_4rc_depth_conf(result_npz: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(result_npz)
    depths = []
    confs = []
    for idx in range(num_frames):
        pts_key = f"pred_{idx}_pts"
        conf_key = f"pred_{idx}_conf"
        if pts_key not in data or conf_key not in data:
            raise KeyError(f"{result_npz}: missing {pts_key} or {conf_key}")
        pts = np.asarray(data[pts_key])
        conf = np.asarray(data[conf_key])
        if pts.ndim == 4 and pts.shape[0] == 1:
            pts = pts[0]
        if conf.ndim == 3 and conf.shape[0] == 1:
            conf = conf[0]
        depths.append(pts[..., 2].astype(np.float32))
        confs.append(conf.astype(np.float32))
    return np.stack(depths, axis=0), np.stack(confs, axis=0)


def confidence_mask(gt_conf: np.ndarray, pred_conf: np.ndarray, percentile: float) -> np.ndarray:
    if percentile <= 0:
        return np.ones_like(gt_conf, dtype=bool)
    gt_thr = np.percentile(gt_conf[np.isfinite(gt_conf)], percentile)
    pred_thr = np.percentile(pred_conf[np.isfinite(pred_conf)], percentile)
    return (gt_conf >= gt_thr) & (pred_conf >= pred_thr)


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape_hw
    out = [
        cv2.resize(m.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
        for m in mask
    ]
    return np.stack(out, axis=0)


def read_mask_stack(mask_dir: Path, num_frames: int) -> np.ndarray:
    frames = []
    for idx in range(num_frames):
        path = mask_dir / f"{idx:03d}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        img = np.asarray(imageio.imread(path))
        if img.ndim == 3:
            img = img.max(axis=-1)
        frames.append((img > 127).astype(np.uint8))
    return np.stack(frames, axis=0).astype(bool)


def target_region_mask(
    br: Path,
    seq: str,
    num_frames: int,
    width: int,
    height: int,
    mode: str,
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    if mode == "all":
        return np.ones((num_frames, height, width), dtype=bool)

    gt_dir = br / "gt" / seq
    masks = []
    if mode in {"foreground", "fg_valid_covisible"}:
        masks.append(read_mask_stack(gt_dir / "fg_mask_2x", num_frames))
    if mode in {"valid_covisible", "fg_valid_covisible"}:
        masks.append(read_mask_stack(gt_dir / "valid_mask_2x", num_frames))
        masks.append(read_mask_stack(gt_dir / "covisible_2x", num_frames))

    if not masks:
        raise ValueError(f"Unsupported mask mode: {mode}")
    mask = masks[0]
    for other in masks[1:]:
        mask = mask & other
    mask = center_crop_to_aspect_bool(mask, crop_width, crop_height)
    return resize_mask(mask, (height, width))


def center_crop_to_aspect_bool(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    height, width = mask.shape[1:3]
    target_aspect = float(target_w) / float(target_h)
    src_aspect = float(width) / float(height)
    if abs(src_aspect - target_aspect) < 1e-6:
        return mask
    if src_aspect > target_aspect:
        new_width = int(round(height * target_aspect))
        left = (width - new_width) // 2
        return mask[:, :, left:left + new_width]
    new_height = int(round(width / target_aspect))
    top = (height - new_height) // 2
    return mask[:, top:top + new_height]


def save_error_video(error_map: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for err in error_map:
        vis = np.clip(err, 0.0, 1.0)
        vis = (vis * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        frames.append(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def weighted_average(rows: list[dict]) -> dict:
    if not rows:
        return {}
    metric_keys = [
        key for key in rows[0]
        if key not in {"method", "iphone_sequence", "sequence_name"}
        and isinstance(rows[0].get(key), (int, float))
        and key != "valid_pixels"
    ]
    weights = np.asarray([max(float(row.get("valid_pixels", 0.0)), 0.0) for row in rows], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(rows), dtype=np.float64)
    out = {"valid_pixels": float(weights.sum())}
    for key in metric_keys:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = float(np.average(vals, weights=weights))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_root", default="outputs/benchmark/iphone_nvs/geometry_4rc")
    parser.add_argument("--recon_root", default=None,
                        help="Default: <geometry_root>/recon")
    parser.add_argument("--manifest", default=None,
                        help="Default: <geometry_root>/manifest.json")
    parser.add_argument("--geo4d_repo", default="third_party/Geo4D")
    parser.add_argument("--methods", default="",
                        help="Optional comma-separated subset; GT is always used only as reference.")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--conf_percentile", type=float, default=5.0,
                        help="Drop pixels below this confidence percentile in either GT or method reconstruction.")
    parser.add_argument("--mask_region", default="valid_covisible",
                        choices=["fg_valid_covisible", "foreground", "valid_covisible", "all"],
                        help="Target-view region for depth metrics. Default = valid covisible pixels; "
                             "the current iPhone export has empty fg_mask_2x directories.")
    parser.add_argument("--max_depth", type=float, default=0.0,
                        help="0 disables max-depth filtering; otherwise passed to Geo4D depth_evaluation.")
    parser.add_argument("--save_error_videos", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    geometry_root = Path(args.geometry_root)
    recon_root = Path(args.recon_root) if args.recon_root else geometry_root / "recon"
    manifest_path = Path(args.manifest) if args.manifest else geometry_root / "manifest.json"
    out_dir = geometry_root / "depth_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_evaluation = load_depth_eval(Path(args.geo4d_repo))
    manifest = load_manifest(manifest_path)
    br = Path(manifest["br_iphone"])
    rows_by_sequence = {
        row["sequence_name"]: row
        for row in manifest["rows"]
    }
    wanted_methods = set(split_csv(args.methods)) if args.methods else None

    gt_by_seq = {}
    gt_rows_by_seq = {}
    for sequence_name, row in rows_by_sequence.items():
        if row["method"] != "gt":
            continue
        result = recon_root / sequence_name / "result.npz"
        if result.exists():
            gt_by_seq[row["iphone_sequence"]] = result
            gt_rows_by_seq[row["iphone_sequence"]] = row

    eval_rows = []
    for sequence_name, row in sorted(rows_by_sequence.items()):
        method = row["method"]
        iphone_seq = row["iphone_sequence"]
        if method == "gt":
            continue
        if wanted_methods is not None and method not in wanted_methods:
            continue
        gt_result = gt_by_seq.get(iphone_seq)
        gt_row = gt_rows_by_seq.get(iphone_seq, {})
        pred_result = recon_root / sequence_name / "result.npz"
        if gt_result is None or not gt_result.exists():
            print(f"[skip] missing GT 4RC result for {iphone_seq}")
            continue
        if not pred_result.exists():
            print(f"[skip] missing method 4RC result for {sequence_name}: {pred_result}")
            continue

        gt_depth, gt_conf = load_4rc_depth_conf(gt_result, args.num_frames)
        pred_depth, pred_conf = load_4rc_depth_conf(pred_result, args.num_frames)
        if pred_depth.shape != gt_depth.shape:
            raise ValueError(f"shape mismatch {sequence_name}: pred={pred_depth.shape} gt={gt_depth.shape}")
        valid = (
            np.isfinite(gt_depth) & np.isfinite(pred_depth) &
            (gt_depth > 0) & (pred_depth > 0) &
            confidence_mask(gt_conf, pred_conf, args.conf_percentile)
        )
        region = target_region_mask(
            br,
            iphone_seq,
            args.num_frames,
            width=gt_depth.shape[2],
            height=gt_depth.shape[1],
            mode=args.mask_region,
            crop_width=int(gt_row.get("crop_width", gt_depth.shape[2])),
            crop_height=int(gt_row.get("crop_height", gt_depth.shape[1])),
        )
        valid = valid & region

        max_depth = None if args.max_depth <= 0 else float(args.max_depth)
        metrics, error_map, pred_aligned, gt_used = depth_evaluation(
            pred_depth,
            gt_depth,
            max_depth=max_depth,
            custom_mask=valid,
            align_mask=torch.from_numpy(valid.reshape(-1, valid.shape[-1])),
            align_with_lad2=True,
            lr=1e-2,
            max_iters=5000,
            use_gpu=False,
        )
        eval_row = {
            "method": method,
            "iphone_sequence": iphone_seq,
            "sequence_name": sequence_name,
            **{key: float(value) for key, value in metrics.items()},
        }
        eval_rows.append(eval_row)
        print(f"[depth] {sequence_name}: {metrics}")

        if args.save_error_videos:
            save_error_video(
                np.asarray(error_map.detach().cpu() if hasattr(error_map, "detach") else error_map),
                out_dir / "error_videos" / f"{sequence_name}.mp4",
                fps=args.fps,
            )

    by_method: dict[str, list[dict]] = {}
    for row in eval_rows:
        by_method.setdefault(row["method"], []).append(row)
    aggregate = {method: weighted_average(rows) for method, rows in sorted(by_method.items())}

    report = {
        "geometry_root": str(geometry_root),
        "recon_root": str(recon_root),
        "manifest": str(manifest_path),
        "geo4d_depth_eval": str(Path(args.geo4d_repo) / "dust3r" / "depth_eval.py"),
        "num_frames": args.num_frames,
        "conf_percentile": args.conf_percentile,
        "mask_region": args.mask_region,
        "per_sequence": eval_rows,
        "aggregate": aggregate,
    }
    (out_dir / "depth_metrics.json").write_text(json.dumps(report, indent=2))

    metric_keys = ["Abs Rel", "Sq Rel", "RMSE", "Log RMSE", "δ < 1.25", "δ < 1.25^2", "δ < 1.25^3", "valid_pixels"]
    with (out_dir / "depth_metrics.tsv").open("w") as f:
        f.write("method\tiphone_sequence\t" + "\t".join(metric_keys) + "\n")
        for row in eval_rows:
            f.write(
                f"{row['method']}\t{row['iphone_sequence']}\t" +
                "\t".join(str(row.get(key, "")) for key in metric_keys) +
                "\n"
            )
    with (out_dir / "depth_metrics_aggregate.tsv").open("w") as f:
        f.write("method\t" + "\t".join(metric_keys) + "\n")
        for method, metrics in aggregate.items():
            f.write(method + "\t" + "\t".join(str(metrics.get(key, "")) for key in metric_keys) + "\n")

    print(f"[depth] wrote {out_dir / 'depth_metrics.json'}")
    print(f"[depth] wrote {out_dir / 'depth_metrics_aggregate.tsv'}")


if __name__ == "__main__":
    main()
