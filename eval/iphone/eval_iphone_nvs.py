"""iPhone NVS metric evaluation (Dycheck protocol).

Reads predictions/<method>/<seq>/pred.mp4 + GT under <output_root>/gt/<seq>/
(produced by build_iphone_pairs.py). Trims/resizes everything to the canonical
360x480 (Shape-of-Motion 2x), computes both standard PSNR/SSIM/LPIPS and the
masked variants (mPSNR/mSSIM/mLPIPS) using `valid * covisible` per the
baseline-paper protocol, then writes:

    <output_root>/eval_results/<run_tag>.json   # per-seq + aggregate
    <output_root>/eval_results/<run_tag>.md     # markdown table

Run:
    python eval/iphone/eval_iphone_nvs.py \
        --output_root outputs/iphone_nvs \
        --predictions_dir outputs/iphone_nvs/predictions/ours_1p3b_step3000 \
        --run_tag ours_1p3b_step3000

Self-contained — only deps are torch + torchmetrics + torchvision + cv2 +
imageio (all in `holo4d` env).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from flow3d_metrics import mLPIPS, mPSNR, mSSIM  # noqa: E402

DEFAULT_SEQS = ("apple", "block", "paper-windmill", "spin", "teddy")
# Two canonical eval frames:
#   "portrait_2x"      — Shape-of-Motion 2x (480 H × 360 W, full iPhone view).
#                        Use with center_crop source preproc; pred at 288×512
#                        gets aspect-distort-resized to portrait 2x at eval.
#   "landscape_model"  — model's native 288 H × 512 W (16:9 landscape).
#                        Use with --source_preproc resize: pred is already at
#                        288×512, GT and masks (saved at 480×360) get
#                        aspect-distort-resized to 288×512 at eval. This keeps
#                        pred and GT in the same aspect-distorted iPhone view.
CANON_PRESETS = {
    "portrait_2x":     (480, 360),
    "landscape_model": (288, 512),
}
CANON_H, CANON_W = CANON_PRESETS["portrait_2x"]   # default — overridden in main()


# -------------------------- I/O helpers --------------------------


def _read_video_frames(mp4_path: Path) -> np.ndarray:
    """Return (T, H, W, 3) uint8 from an mp4."""
    frames = iio.imread(mp4_path, plugin="pyav")
    if frames.ndim == 3:
        frames = frames[None]
    return np.asarray(frames, dtype=np.uint8)


def _read_png_stack(dir_path: Path, n: int) -> np.ndarray:
    out = []
    for i in range(n):
        img = iio.imread(dir_path / f"{i:03d}.png")
        out.append(img)
    return np.stack(out, axis=0)


def _resize_video(frames: np.ndarray, h: int, w: int, mode: str = "rgb") -> np.ndarray:
    """frames: (T, H0, W0, ...) uint8 or float — bilinear for rgb, nearest for masks."""
    interp = cv2.INTER_AREA if mode == "rgb" else cv2.INTER_NEAREST
    out = []
    for f in frames:
        out.append(cv2.resize(f, (w, h), interpolation=interp))
    return np.stack(out, axis=0)


def _center_crop_to_aspect(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop the spatial dims (T, H, W[, C]) to match (target_w / target_h) aspect.

    Mirrors the model's `center_crop_and_resize` crop step, applied to GT/masks
    so they evaluate against pred on the same 16:9 strip the model rendered.
    """
    H, W = arr.shape[1], arr.shape[2]
    target_aspect = float(target_w) / float(target_h)
    src_aspect = float(W) / float(H)
    if abs(src_aspect - target_aspect) < 1e-6:
        return arr
    if src_aspect > target_aspect:
        new_W = int(round(H * target_aspect))
        left = (W - new_W) // 2
        return arr[:, :, left:left + new_W]
    else:
        new_H = int(round(W / target_aspect))
        top = (H - new_H) // 2
        return arr[:, top:top + new_H]


def _to_float01(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Any pixel >127 → 1.0; collapses a trailing channel axis if present.

    Accepts (H,W), (H,W,C), (T,H,W), or (T,H,W,C). The collapse rule keys on
    the last axis being a channel (size 3 or 4 for RGB/A; size 1 for explicit
    grayscale) so we never accidentally reduce the W axis of a stacked
    (T,H,W) grayscale mask.
    """
    last_is_channel = mask.ndim in (3, 4) and mask.shape[-1] in (1, 3, 4)
    if last_is_channel:
        mask = mask.max(axis=-1)
    return (mask.astype(np.float32) > 127.0).astype(np.float32)


# -------------------------- Per-seq metric loop --------------------------


def _eval_one_seq(
    pred_mp4: Path,
    gt_dir: Path,
    n_frames: int,
    device: torch.device,
    crop_gt_to_landscape: bool = False,
    crop_pred_to_landscape: bool = False,
) -> dict:
    pred = _read_video_frames(pred_mp4)
    if pred.shape[0] < n_frames:
        raise RuntimeError(f"{pred_mp4}: only {pred.shape[0]} frames (need ≥{n_frames})")
    pred = pred[:n_frames]
    if crop_pred_to_landscape:
        pred = _center_crop_to_aspect(pred, CANON_W, CANON_H)
    pred = _resize_video(pred, CANON_H, CANON_W, mode="rgb")

    gt = _read_png_stack(gt_dir / "target_frames_2x", n_frames)
    cov = _read_png_stack(gt_dir / "covisible_2x", n_frames)
    valid = _read_png_stack(gt_dir / "valid_mask_2x", n_frames)
    fg_dir = gt_dir / "fg_mask_2x"
    has_fg = fg_dir.exists() and any(fg_dir.iterdir())
    if has_fg:
        fg = _read_png_stack(fg_dir, n_frames)

    # Optional center-crop GT/masks to match the model's `center_crop_and_resize`
    # 16:9 strip — pred + GT then evaluate on the same view (matches Vista4D
    # convention). Apply BEFORE the resize-to-canon so we don't double-resize.
    if crop_gt_to_landscape:
        gt = _center_crop_to_aspect(gt, CANON_W, CANON_H)
        cov = _center_crop_to_aspect(cov, CANON_W, CANON_H)
        valid = _center_crop_to_aspect(valid, CANON_W, CANON_H)
        if has_fg:
            fg = _center_crop_to_aspect(fg, CANON_W, CANON_H)

    if gt.shape[1:3] != (CANON_H, CANON_W):
        gt = _resize_video(gt, CANON_H, CANON_W, mode="rgb")
    if cov.shape[1:3] != (CANON_H, CANON_W):
        cov = _resize_video(cov, CANON_H, CANON_W, mode="mask")
    if valid.shape[1:3] != (CANON_H, CANON_W):
        valid = _resize_video(valid, CANON_H, CANON_W, mode="mask")
    if has_fg and fg.shape[1:3] != (CANON_H, CANON_W):
        fg = _resize_video(fg, CANON_H, CANON_W, mode="mask")

    pred_t = torch.from_numpy(_to_float01(pred)).to(device)
    gt_t = torch.from_numpy(_to_float01(gt)).to(device)
    cov_b = torch.from_numpy(_binarize(cov)).to(device)
    valid_b = torch.from_numpy(_binarize(valid)).to(device)
    main_mask = (cov_b * valid_b).clamp(0, 1)

    metrics = {}
    # standard (no mask)
    for name, M in (("psnr", mPSNR()), ("ssim", mSSIM()), ("lpips", mLPIPS())):
        M = M.to(device)
        M.update(pred_t, gt_t)
        metrics[name] = float(M.compute().detach().cpu())

    # masked (Dycheck)
    for name, M in (("mpsnr", mPSNR()), ("mssim", mSSIM()), ("mlpips", mLPIPS())):
        M = M.to(device)
        M.update(pred_t, gt_t, main_mask)
        metrics[name] = float(M.compute().detach().cpu())

    if has_fg:
        fg_b = _binarize(fg)
        fg_t = torch.from_numpy(fg_b).to(device)
        bg_t = (1.0 - fg_t).clamp(0, 1)
        fg_mask = (fg_t * main_mask).clamp(0, 1)
        bg_mask = (bg_t * main_mask).clamp(0, 1)
        for tag, mask in (("fg", fg_mask), ("bg", bg_mask)):
            for name, M in (
                (f"{tag}_mpsnr", mPSNR()),
                (f"{tag}_mssim", mSSIM()),
                (f"{tag}_mlpips", mLPIPS()),
            ):
                M = M.to(device)
                M.update(pred_t, gt_t, mask)
                metrics[name] = float(M.compute().detach().cpu())

    return metrics


# -------------------------- Aggregate / report --------------------------


def _aggregate(per_seq: dict) -> dict:
    keys = sorted({k for v in per_seq.values() for k in v})
    return {k: float(np.mean([per_seq[s][k] for s in per_seq if k in per_seq[s]])) for k in keys}


def _format_md(per_seq: dict, aggregate: dict, run_tag: str) -> str:
    seqs = sorted(per_seq)
    headline_keys = ("psnr", "ssim", "lpips", "mpsnr", "mssim", "mlpips")
    lines = [f"# iPhone NVS results — `{run_tag}`", ""]

    lines.append("## Aggregate (mean over seqs)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k in headline_keys:
        if k in aggregate:
            lines.append(f"| {k.upper()} | {aggregate[k]:.4f} |")
    lines.append("")

    lines.append("## Per-seq\n")
    head = "| seq | " + " | ".join(k.upper() for k in headline_keys) + " |"
    sep = "|" + "|".join("---" for _ in range(len(headline_keys) + 1)) + "|"
    lines.append(head)
    lines.append(sep)
    for s in seqs:
        row = [s] + [
            f"{per_seq[s].get(k, float('nan')):.4f}" for k in headline_keys
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    fg_keys = [k for k in aggregate if k.startswith(("fg_", "bg_"))]
    if fg_keys:
        lines.append("## FG / BG breakdown (masked, optional)\n")
        for k in sorted(fg_keys):
            lines.append(f"- {k}: {aggregate[k]:.4f}")
        lines.append("")
    return "\n".join(lines)


# -------------------------- CLI --------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=str,
                   default="outputs/benchmark/iphone_nvs",
                   help="Directory containing gt/<seq>/. Eval results go to <output_root>/eval_results/.")
    p.add_argument("--predictions_dir", type=str, required=True,
                   help="Directory containing <seq>/pred.mp4 for each seq.")
    p.add_argument("--run_tag", type=str, required=True,
                   help="Identifier used for the output JSON / MD filenames (e.g. ours_1p3b_step3000).")
    p.add_argument("--sequences", type=str, default=",".join(DEFAULT_SEQS))
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--canon_mode", type=str, default="portrait_2x",
                   choices=sorted(CANON_PRESETS.keys()),
                   help="Canonical eval frame. 'portrait_2x' (default) = 480 H × 360 W (Shape-of-Motion 2x); "
                        "'landscape_model' = 288 H × 512 W (model native, paired with --source_preproc=resize "
                        "in predict_iphone_nvs.py).")
    p.add_argument("--crop_gt_to_landscape", action="store_true", default=False,
                   help="Center-crop GT/masks to the canon's aspect (16:9 when canon=landscape_model) "
                        "BEFORE the final resize-to-canon. Use with --source_preproc=center_crop in the "
                        "predictor so pred and GT evaluate on the same 16:9 strip (Vista4D protocol).")
    p.add_argument("--crop_pred_to_landscape", action="store_true", default=False,
                   help="Center-crop prediction to the canon's aspect before resizing. This is only for "
                        "diagnostic outputs that contain extra spatial region. Do not use it for EX-4D "
                        "square-boundary outputs; those should be directly resized back to the GT eval canvas.")
    args = p.parse_args()

    # Override module-level CANON_H/W from chosen preset.
    global CANON_H, CANON_W
    CANON_H, CANON_W = CANON_PRESETS[args.canon_mode]

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    output_root = Path(args.output_root)
    predictions_dir = Path(args.predictions_dir)
    eval_out_dir = output_root / "eval_results"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[eval] canon_mode={args.canon_mode}  CANON_H×W={CANON_H}×{CANON_W}")

    per_seq: dict[str, dict] = {}
    for seq in seqs:
        pred_mp4 = predictions_dir / seq / "pred.mp4"
        if not pred_mp4.exists():
            # Some launchers write trajectory-named mp4s instead.
            cands = list((predictions_dir / seq).glob("*.mp4")) if (predictions_dir / seq).exists() else []
            if not cands:
                print(f"[eval] skip {seq}: no mp4 under {predictions_dir / seq}")
                continue
            pred_mp4 = cands[0]
            print(f"[eval] {seq}: using {pred_mp4.name} as prediction")

        gt_dir = output_root / "gt" / seq
        if not gt_dir.exists():
            raise FileNotFoundError(f"GT not built for {seq}: {gt_dir}")

        print(f"[eval] {seq}: pred={pred_mp4}")
        per_seq[seq] = _eval_one_seq(
            pred_mp4, gt_dir, args.num_frames, device,
            crop_gt_to_landscape=bool(args.crop_gt_to_landscape),
            crop_pred_to_landscape=bool(args.crop_pred_to_landscape),
        )
        print(f"        -> {per_seq[seq]}")

    aggregate = _aggregate(per_seq) if per_seq else {}
    report = {
        "run_tag": args.run_tag,
        "predictions_dir": str(predictions_dir),
        "canonical_resolution_hxw": [CANON_H, CANON_W],
        "num_frames": args.num_frames,
        "per_seq": per_seq,
        "aggregate": aggregate,
    }
    (eval_out_dir / f"{args.run_tag}.json").write_text(json.dumps(report, indent=2))
    (eval_out_dir / f"{args.run_tag}.md").write_text(_format_md(per_seq, aggregate, args.run_tag))
    print(f"[eval] -> {eval_out_dir}/{args.run_tag}.{{json,md}}")
    if aggregate:
        print(f"[eval] aggregate: " +
              " ".join(f"{k}={aggregate[k]:.4f}" for k in
                       ("psnr", "ssim", "lpips", "mpsnr", "mssim", "mlpips") if k in aggregate))


if __name__ == "__main__":
    main()
