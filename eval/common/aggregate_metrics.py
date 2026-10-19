#!/usr/bin/env python3
"""Aggregate per-baseline metric JSONs into a comparison TSV.

Reads (in priority order — first match wins for camera columns):
  <results_root>/<method>/eval_video_metrics.json         (clip_t, clip_f, clip_v, fvd, fvd_v)
  <results_root>/<method>/eval_camera_metrics_vipe.json   (ate, trans_err, rot_err — ViPE-based, fast)
  <results_root>/<method>/eval_camera_metrics_vggt_omega.json (ate, trans_err, rot_err — VGGT-Omega)
  <results_root>/<method>/eval_camera_metrics_da3.json    (ate, trans_err, rot_err — Depth Anything 3)
  <results_root>/<method>/eval_camera_metrics.json        (ate, trans_err, rot_err — BA-Track, legacy)

Writes:
  <output_tsv>  with one row per method and columns:
    method  ATE/RPE camera columns  clip_t  clip_f  clip_v  fvd  n_video  n_ate  n_rot

By default `--camera_backend auto` preserves the historical priority:
ViPE, then VGGT-Omega or DA3 if ViPE is absent, then BA-Track. Use
`--camera_backend vggt_omega` or `--camera_backend da3` to force an
estimator table.
Skips dirs that contain neither metric file (e.g. `_trajc_audit_frames`).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VIDEO_METRICS = ["fvd", "clip_t", "clip_f", "clip_v"]   # fvd first (perceptual realism); fvd_v dropped (multiview-only, n/a here)
# Camera columns (only the recommended ones):
#   ate (n_max=54)       = per-pair scale-normalized ATE on translation-bearing pairs
#                          (orbits + 1D translates + zooms; 46 pan-only pairs excluded
#                          by construction since GT has zero translation)
#   trans_err (n_max=54) = per-pair scale-normalized RPE trans. on the same n=54
#   rot_err (n_max=100)  = RPE rot. in degrees, across all evaluated pairs
#                          (rotation is scale-invariant — every pair contributes)
#   n_ate / n_rot        = actual per-method counts (may be 53/99 if ViPE failed on a pair)
CAMERA_COL_ATE   = "ATE ↓"
CAMERA_COL_TRANS = "RPE trans. ↓"
CAMERA_COL_ROT   = "RPE rot. ↓"
CAMERA_COL_ROT_MED = "RPE rot. median ↓"   # better single-number rotation summary (heavy-tailed)
CAMERA_COL_ROT_OUT = "RPE rot. n_>1deg ↓"  # catastrophic-failure count
CAMERA_COLS  = [CAMERA_COL_ATE, CAMERA_COL_TRANS, CAMERA_COL_ROT, CAMERA_COL_ROT_MED, CAMERA_COL_ROT_OUT]
CAMERA_NCOLS = ["n_ate", "n_rot"]                                   # per-row counts (parked at end)

# Method grouping for the headline TSV layout. The TSV is sorted in two
# blocks (method ↓ first, then ablation ↓), then alphabetically inside each,
# matching the cross-comparison layout in
# doc/track4d-360/2026-05-03-camera-condition-ablation-status.md §3.
# Unlisted methods get category="other" and sink to the bottom.
METHOD_ORDER = [
    "ours_14b_softsplat",                                # 14B multiframe, ismb-trained
    "ours_14b",                                          # 14B singleframe, ismb-trained
    "vista4d_4rc",                                       # Vista4D
    "ex4d_calibrated",                                   # EX-4D
    "trajcrafter_stage2_4rc",                            # TrajCrafter
    "ours_1p3b",                                         # Our reference 1.3B (full conditions)
    "recammaster",                                       # RecamMaster
]
ABLATION_ORDER = [
    "ours_ablate_04_singleframe",                       # D: single-frame zbuffer RGBD + tracks (ours_1p3b alias)
    "ours_ablate_04_multiframe_rgbd_track3d",            # E: multiframe RGBD + tracks
    "ours_ablate_04_multiframe_rgbd_track3d_dynmask_fix",# E (post-fix re-run)
    "ours_ablate_02_zbuffer_rgbd_null_tracks",           # C: singleframe zbuffer RGBD, tracks null
    "ours_ablate_02b_zbuffer_rgb_dnull_null_tracks",     # B: singleframe zbuffer RGB only
    "ours_ablate_01_plucker_only",                       # A: plucker only
]
ABLATION_ALIASES = {
    "ours_ablate_04_singleframe": "ours_1p3b",
}
CATEGORY = (
    {m: ("method", i) for i, m in enumerate(METHOD_ORDER)}
    | {m: ("ablation", i) for i, m in enumerate(ABLATION_ORDER)}
)


def _is_number(v) -> bool:
    return isinstance(v, (int, float))


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _pair_key(row: dict) -> tuple[str, str] | None:
    seq = row.get("seq")
    traj = row.get("traj")
    if isinstance(seq, str) and isinstance(traj, str):
        return (seq, traj)
    return None


def _camera_metrics_from_pair_map(pair_map: dict, rot_pairs: set, ate_pairs: set) -> dict:
    """Recompute camera scalars on explicitly chosen common-success pair sets."""
    rot_vals = [
        pair_map[p]["rot_err"]
        for p in sorted(rot_pairs)
        if p in pair_map and _is_number(pair_map[p].get("rot_err"))
    ]
    ate_vals = [
        pair_map[p]["ate_norm"]
        for p in sorted(ate_pairs)
        if p in pair_map and _is_number(pair_map[p].get("ate_norm"))
    ]
    trans_vals = [
        pair_map[p]["trans_err_norm"]
        for p in sorted(ate_pairs)
        if p in pair_map and _is_number(pair_map[p].get("trans_err_norm"))
    ]
    return {
        CAMERA_COL_ATE:     _mean(ate_vals),
        CAMERA_COL_TRANS:   _mean(trans_vals),
        CAMERA_COL_ROT:     _mean(rot_vals),
        CAMERA_COL_ROT_MED: _median(rot_vals),
        CAMERA_COL_ROT_OUT: sum(1 for v in rot_vals if v >= 1.0) if rot_vals else None,
        "n_ate":            len(ate_vals),
        "n_rot":            len(rot_vals),
    }


def _common_success_rows(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Return rows with camera metrics restricted to pairs successful for every row.

    Rotation and ATE/trans have separate common-success sets: rot uses every
    pair where all rows have numeric `rot_err`, while ATE/trans use every pair
    where all rows have numeric `ate_norm` and `trans_err_norm`. This preserves
    the pan-only exclusion for ATE while still making each metric apples-to-apples.
    """
    eligible = [r for r in rows if r.get("_pair_map")]
    if len(eligible) < 2:
        return [], 0, 0

    rot_sets = []
    ate_sets = []
    for r in eligible:
        pair_map = r["_pair_map"]
        rot_sets.append({p for p, v in pair_map.items() if _is_number(v.get("rot_err"))})
        ate_sets.append({
            p for p, v in pair_map.items()
            if _is_number(v.get("ate_norm")) and _is_number(v.get("trans_err_norm"))
        })
    common_rot = set.intersection(*rot_sets) if rot_sets else set()
    common_ate = set.intersection(*ate_sets) if ate_sets else set()

    out = []
    eligible_methods = {r["method"] for r in eligible}
    for r in rows:
        if r.get("method") not in eligible_methods:
            continue
        nr = dict(r)
        nr.update(_camera_metrics_from_pair_map(nr["_pair_map"], common_rot, common_ate))
        out.append(nr)
    _sort_by_ate_best_bottom(out)
    return out, len(common_rot), len(common_ate)


def _sort_by_ate_best_bottom(rows: list[dict]) -> None:
    """Sort in-place by ATE descending so lower/better rows land at the bottom."""
    def _key(r: dict) -> tuple[float, str]:
        v = r.get(CAMERA_COL_ATE)
        score = v if isinstance(v, (int, float)) else float("inf")
        return (-score, r.get("method", ""))
    rows.sort(key=_key)


def _safe_load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _pull(data: dict, keys: list[str]) -> dict[str, float | None]:
    """Try a few common JSON nesting layouts to extract metric scalars."""
    out = {}
    for k in keys:
        v = None
        if isinstance(data, dict):
            if k in data and isinstance(data[k], (int, float)):
                v = float(data[k])
            elif k in data and isinstance(data[k], dict):
                cand = data[k]
                for nested in ("value", "mean"):
                    if nested in cand and isinstance(cand[nested], (int, float)):
                        v = float(cand[nested])
                        break
            elif "metrics" in data and isinstance(data["metrics"], dict) and k in data["metrics"]:
                cand = data["metrics"][k]
                if isinstance(cand, (int, float)):
                    v = float(cand)
                elif isinstance(cand, dict) and "mean" in cand:
                    v = float(cand["mean"])
            elif "summary" in data and isinstance(data["summary"], dict) and k in data["summary"]:
                cand = data["summary"][k]
                if isinstance(cand, (int, float)):
                    v = float(cand)
                elif isinstance(cand, dict) and "mean" in cand:
                    v = float(cand["mean"])
        out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True)
    p.add_argument("--output_tsv", required=True)
    p.add_argument(
        "--camera_backend",
        choices=["auto", "vipe", "vggt_omega", "da3", "batrack"],
        default="auto",
        help=(
            "Camera metric backend to aggregate. auto preserves the historical "
            "priority: vipe, then vggt_omega, then da3, then batrack."
        ),
    )
    args = p.parse_args()

    root = Path(args.results_root)
    rows = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        method = method_dir.name
        v_json = method_dir / "eval_video_metrics.json"
        c_vipe_json = method_dir / "eval_camera_metrics_vipe.json"
        c_vggt_json = method_dir / "eval_camera_metrics_vggt_omega.json"
        c_da3_json = method_dir / "eval_camera_metrics_da3.json"
        c_batrack_json = method_dir / "eval_camera_metrics.json"

        # Skip dirs with no metric files (audit dumps, scratch, etc.).
        if not (v_json.is_file() or c_vipe_json.is_file() or c_vggt_json.is_file() or c_da3_json.is_file() or c_batrack_json.is_file()):
            continue

        v = _safe_load(v_json)
        camera_candidates = {
            "vipe": c_vipe_json,
            "vggt_omega": c_vggt_json,
            "da3": c_da3_json,
            "batrack": c_batrack_json,
        }
        if args.camera_backend == "auto":
            camera_order = ["vipe", "vggt_omega", "da3", "batrack"]
        else:
            camera_order = [args.camera_backend]

        c = {}
        for camera_source in camera_order:
            c_path = camera_candidates[camera_source]
            if c_path.is_file():
                c = _safe_load(c_path)
                break

        vmetrics = _pull(v, VIDEO_METRICS)
        n_video = (
            v.get("num_samples")
            or v.get("n_samples")
            or (v.get("metadata", {}) or {}).get("num_samples")
            or len(v.get("per_sample") or [])
        )
        # Camera metrics — only the recommended ones. Pulled from the new
        # `all_norm` summary block when present; falls back to per-pair
        # recompute for older JSONs that don't have the block yet.
        all_norm = (c.get("all_norm") or {}) if isinstance(c, dict) else {}
        per_pair = c.get("per_pair") if isinstance(c, dict) else None
        if per_pair and not all_norm:
            import statistics
            n_pp = [r for r in per_pair if r.get("ate_norm") is not None]
            def _avg(rows, k):
                vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
                return statistics.fmean(vals) if vals else None
            all_norm = {
                "ate":       {"mean": _avg(n_pp, "ate_norm"),       "n": len(n_pp)},
                "trans_err": {"mean": _avg(n_pp, "trans_err_norm"), "n": len(n_pp)},
                "rot_err":   {"mean": _avg(per_pair, "rot_err"),    "n": len(per_pair)},
            }

        def _g(d, *path, default=None):
            cur = d
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    return default
                cur = cur[p]
            return cur

        n_rot = (
            _g(all_norm, "rot_err", "n", default=0)
            or (c.get("num_pairs_evaluated") if isinstance(c, dict) else 0)
            or (len(per_pair) if per_pair else 0)
            or 0
        )

        # Heavy-tailed rotation summary: median + count of catastrophic outliers.
        # Computed from per_pair when available — `all_norm` only stores mean.
        rot_median = None
        rot_n_outliers = None
        if per_pair:
            import statistics as _stat
            rot_vals = [r.get("rot_err") for r in per_pair if isinstance(r.get("rot_err"), (int, float))]
            if rot_vals:
                rot_median = _stat.median(rot_vals)
                rot_n_outliers = sum(1 for v in rot_vals if v >= 1.0)

        camera_metrics = {
            CAMERA_COL_ATE:     _g(all_norm, "ate", "mean"),
            CAMERA_COL_TRANS:   _g(all_norm, "trans_err", "mean"),
            CAMERA_COL_ROT:     _g(all_norm, "rot_err", "mean"),
            CAMERA_COL_ROT_MED: rot_median,
            CAMERA_COL_ROT_OUT: rot_n_outliers,
            "n_ate":            _g(all_norm, "ate", "n", default=0) or 0,
            "n_rot":            n_rot,
        }

        category, idx = CATEGORY.get(method, ("other", 999))
        row = {"_category": category, "_sort_idx": idx,
               "method": method, **vmetrics, **camera_metrics, "n_video": n_video}
        if per_pair:
            pair_map = {}
            for pp in per_pair:
                key = _pair_key(pp)
                if key is not None:
                    pair_map[key] = pp
            row["_pair_map"] = pair_map
        rows.append(row)

    out = Path(args.output_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", *CAMERA_COLS, *VIDEO_METRICS, "n_video", *CAMERA_NCOLS]

    # Split rows by category: methods + unlisted "other" go to summary.tsv,
    # ablation runs go to a sibling summary_ablation.tsv. The "other" tail
    # keeps ATE-ascending sort so newly-added methods sink in a sensible
    # order until they're explicitly placed in METHOD_ORDER.
    method_rows = [r for r in rows if r["_category"] in ("method", "other")]
    ablation_rows = [r for r in rows if r["_category"] == "ablation"]
    rows_by_method = {r["method"]: r for r in rows}
    for alias, source in ABLATION_ALIASES.items():
        source_row = rows_by_method.get(source)
        if source_row is None:
            continue
        category, idx = CATEGORY[alias]
        alias_row = dict(source_row)
        alias_row["method"] = alias
        alias_row["_category"] = category
        alias_row["_sort_idx"] = idx
        ablation_rows.append(alias_row)

    def _ate_key(r):
        v = r.get(CAMERA_COL_ATE)
        return v if isinstance(v, (int, float)) else float("inf")

    method_rows.sort(key=lambda r: (
        0 if r["_category"] == "method" else 1,
        r.get("_sort_idx", 999) if r["_category"] == "method" else _ate_key(r),
        r.get("method", ""),
    ))
    _sort_by_ate_best_bottom(ablation_rows)

    def _write(path: Path, rs: list[dict]) -> None:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for r in rs:
                writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
        print(f"wrote {len(rs)} rows -> {path}")

    _write(out, method_rows)
    ablation_out = out.with_name(out.stem + "_ablation" + out.suffix)
    _write(ablation_out, ablation_rows)

    # Sibling camera tables that recompute metrics on pair intersections.
    # These answer: "only compare samples where every method in this block
    # produced a valid camera result." Video metrics remain unchanged.
    common_method_rows, common_method_rot_n, common_method_ate_n = _common_success_rows(method_rows)
    if common_method_rows:
        common_out = out.with_name(out.stem + "_common_success" + out.suffix)
        _write(common_out, common_method_rows)
        print(f"  camera common-success: n_rot={common_method_rot_n}  n_ate={common_method_ate_n}")

    common_ablation_rows, common_ablation_rot_n, common_ablation_ate_n = _common_success_rows(ablation_rows)
    if common_ablation_rows:
        common_ablation_out = out.with_name(out.stem + "_ablation_common_success" + out.suffix)
        _write(common_ablation_out, common_ablation_rows)
        print(f"  ablation camera common-success: n_rot={common_ablation_rot_n}  n_ate={common_ablation_ate_n}")


if __name__ == "__main__":
    main()
