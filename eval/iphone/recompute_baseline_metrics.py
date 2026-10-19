"""Strict five-scene iPhone/DyCheck RGB metric recomputation.

This wraps the public ``eval_iphone_nvs`` metric implementation without
copying videos or silently choosing an arbitrary MP4. Baseline inputs may be
the result-only archive's ``iphone_nvs/external_predictions_fixed`` directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import eval_iphone_nvs as metric


SCENES = ("apple", "block", "paper-windmill", "spin", "teddy")
TARGET_CAM = {scene: "iphone_cam2" if scene == "apple" else "iphone_cam1" for scene in SCENES}
FIXED_METHODS = ("gen3c_fixed", "vista4d_fixed", "trajcrafter_fixed", "recammaster_fixed")


def fixed_prediction(root: Path, method: str, scene: str) -> Path:
    camera = TARGET_CAM[scene]
    if method == "gen3c_fixed":
        return root / "gen3c" / scene / camera / f"gen3c_{camera}_first49.mp4"
    if method == "vista4d_fixed":
        return root / "vista4d_center_crop" / "results_384p" / f"{scene}_{camera}" / "video_seed=42.mp4"
    if method == "trajcrafter_fixed":
        return root / "trajcrafter" / scene / camera / "gen.mp4"
    if method == "recammaster_fixed":
        return root / "recammaster" / "predictions" / scene / camera / "pred.mp4"
    raise ValueError(f"Unknown fixed baseline: {method}")


def parse_standardized_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path or name in roots:
            raise ValueError(f"Expected unique NAME=PATH for --standardized-root, got {value!r}")
        roots[name] = Path(path)
    return roots


def prediction_paths(
    fixed_root: Path | None,
    standardized_roots: dict[str, Path],
) -> dict[str, dict[str, Path]]:
    methods: dict[str, dict[str, Path]] = {}
    if fixed_root is not None:
        for method in FIXED_METHODS:
            methods[method] = {scene: fixed_prediction(fixed_root, method, scene) for scene in SCENES}
    for method, root in standardized_roots.items():
        if method in methods:
            raise ValueError(f"Duplicate method name: {method}")
        methods[method] = {scene: root / scene / "pred.mp4" for scene in SCENES}
    return methods


def evaluate_method(
    name: str,
    paths: dict[str, Path],
    gt_root: Path,
    output_root: Path,
    device: torch.device,
) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    missing.extend(str(gt_root / scene) for scene in SCENES if not (gt_root / scene).is_dir())
    if missing:
        raise FileNotFoundError(f"{name}: incomplete five-scene evaluation:\n" + "\n".join(missing))

    per_seq = {
        scene: metric._eval_one_seq(
            paths[scene], gt_root / scene, 49, device, crop_gt_to_landscape=True,
        )
        for scene in SCENES
    }
    aggregate = metric._aggregate(per_seq)
    report = {
        "run_tag": name,
        "protocol": "iPhone/DyCheck five-scene RGB; first 49 frames; GT center-crop to 16:9",
        "canonical_resolution_hxw": [288, 512],
        "mask": "valid * covisible",
        "num_frames": 49,
        "prediction_paths": {scene: str(paths[scene].resolve()) for scene in SCENES},
        "per_seq": per_seq,
        "aggregate": aggregate,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_root / f"{name}.md").write_text(metric._format_md(per_seq, aggregate, name) + "\n")
    print(f"[iphone] {name}: 5/5 PSNR={aggregate['psnr']:.4f} SSIM={aggregate['ssim']:.4f} "
          f"LPIPS={aggregate['lpips']:.4f} mPSNR={aggregate['mpsnr']:.4f} "
          f"mSSIM={aggregate['mssim']:.4f} mLPIPS={aggregate['mlpips']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True, help="Prepared iPhone benchmark gt/<scene> root")
    parser.add_argument("--fixed-root", type=Path, help="Archive iphone_nvs/external_predictions_fixed root")
    parser.add_argument("--standardized-root", action="append", default=[], metavar="NAME=PATH",
                        help="Additional method with <PATH>/<scene>/pred.mp4; repeatable")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    methods = prediction_paths(args.fixed_root, parse_standardized_roots(args.standardized_root))
    if not methods:
        parser.error("Provide --fixed-root and/or --standardized-root")
    metric.CANON_H, metric.CANON_W = metric.CANON_PRESETS["landscape_model"]
    device = torch.device(args.device)
    for name, paths in methods.items():
        evaluate_method(name, paths, args.gt_root, args.output_root, device)


if __name__ == "__main__":
    main()
