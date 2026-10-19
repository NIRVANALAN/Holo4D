#!/bin/bash
# ===========================================================================
# ViPE-based camera-trajectory eval for ONE baseline (DAVIS-50 × 2-traj).
#
# In-repo location: eval/common/eval_camera_metrics_vipe.sh
# Caches + result JSONs live at the data-side ${BASELINES_ROOT}/eval (not
# inside the repo).
#
# Drop-in replacement for eval_camera_metrics.sh (BA-Track), but ~10× faster:
#   ViPE      ≈ 30 s/video  → 100 pairs ≈ 50 min
#   BA-Track  ≈ 5 min/video → 100 pairs ≈ 8 hr
#
# Pipeline (two phases, two conda envs):
#   1. vipe env: ffmpeg-standardize each pred mp4 to TARGET_W×TARGET_H×NUM_FRAMES
#        (default 512×288×49) inside vipe_camera_eval.py, then run
#        `vipe infer <std.mp4>`. Pose npz cached at
#        ${EVAL_ROOT}/_vipe/<method>/<seq>__<traj>/pose/<base>.npz
#   2. batrack env (has evo + scipy): compare ViPE-recovered pred c2w vs
#        cameras_2traj/<seq>/<traj>.npz GT c2w using Sim(3)-aligned
#        ATE / RPE-trans / RPE-rot. Splits the per_pair list by
#        alignment_mode (umeyama_align_scale vs origin_align_only) so the
#        Sim(3) subset (orbit-style, n=19/100 on DAVIS) is reported separately
#        from the origin-only subset (pan/translate/zoom, n=81) where ate &
#        trans_err are NOT scale-comparable across methods.
#
# Output: ${BASELINES_ROOT}/eval/results/<method>/eval_camera_metrics_vipe.json
#
# Usage:
#   METHOD=trajcrafter_stage2 \
#   METHOD_PRED_ROOT=${BR}/TrajectoryCrafter_stage2 \
#   PRED_FILENAMES='gen.mp4' \
#     bash eval/common/eval_camera_metrics_vipe.sh
#
# Knobs:
#   FORCE=1                         re-run if output JSON exists
#   GPU=0                           CUDA device for ViPE
#   LIMIT_PAIRS=5                   smoke first N pairs
#   NUM_SHARDS=8 SHARD_INDEX=0      evaluate one row-index shard
#   PHASE=vipe|metrics|all          default 'all' (run both phases)
#   TARGET_W=512 TARGET_H=288 NUM_FRAMES=49   canonical ViPE input
# ===========================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/benchmark_env.sh"

EVAL_ROOT="${EVAL_ROOT:-${BASELINES_ROOT}/eval}"
METHOD="${METHOD:?METHOD env is required (e.g. ours_1p3b, recammaster, trajcrafter_stage2, ...)}"
METHOD_PRED_ROOT="${METHOD_PRED_ROOT:?METHOD_PRED_ROOT env is required}"
NUM_FRAMES="${NUM_FRAMES:-49}"
TARGET_W="${TARGET_W:-512}"
TARGET_H="${TARGET_H:-288}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
LIMIT_PAIRS="${LIMIT_PAIRS:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
PHASE="${PHASE:-all}"
VIPE_CONDA_ENV="${VIPE_CONDA_ENV:-vipe}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-batrack}"   # has evo + scipy
# Assets may live under ${BASELINES_ROOT} or one level up in the benchmark root
# (the layout the sampler reads/writes). Prefer baselines/, fall back to the
# benchmark root so metrics + sampling resolve the same asset.
_resolve_baseline_asset() {
  local rel="$1"
  if [ -e "${BASELINES_ROOT}/${rel}" ]; then
    printf '%s' "${BASELINES_ROOT}/${rel}"
  else
    printf '%s' "$(dirname "${BASELINES_ROOT}")/${rel}"
  fi
}
PAIRS_CSV="${PAIRS_CSV:-$(_resolve_baseline_asset davis_trajectories.csv)}"
CAMERAS_ROOT="${CAMERAS_ROOT:-$(_resolve_baseline_asset cameras_2traj)}"

# NB: avoid `${VAR:-...{traj}...}` (bash brace-counts inside the default).
if [ -z "${PRED_FILENAMES:-}" ]; then
  PRED_FILENAMES='pred.mp4,{traj}.mp4,gen.mp4,output.mp4,render.mp4,color_{traj}.mp4'
fi

vipe_cache="${EVAL_ROOT}/_vipe/${METHOD}"
out_json="${EVAL_ROOT}/results/${METHOD}/eval_camera_metrics_vipe.json"
mkdir -p "$(dirname "${out_json}")" "${vipe_cache}"

if [ "${FORCE}" != "1" ] && [ -f "${out_json}" ]; then
  echo "[eval-camera-vipe] skip ${METHOD}: ${out_json} exists (FORCE=1 to re-run)"
  cat "${out_json}" | head -25
  exit 0
fi

EXTRA_ARGS=()
[ -n "${LIMIT_PAIRS}" ] && EXTRA_ARGS+=(--limit_pairs "${LIMIT_PAIRS}")
if [ "${NUM_SHARDS}" != "1" ]; then
  EXTRA_ARGS+=(--num_shards "${NUM_SHARDS}" --shard_index "${SHARD_INDEX}")
fi

# --- Phase 1: ViPE inference --------------------------------------------------
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "vipe" ]; then
  echo "[eval-camera-vipe] phase=ViPE inference (${METHOD}; GPU=${GPU})"
  holo4d_activate_conda "${VIPE_CONDA_ENV}"
  VIPE_PYTHON="$(command -v python)"
  python "${SCRIPT_DIR}/vipe_camera_eval.py" \
    --method_pred_root "${METHOD_PRED_ROOT}" \
    --pred_filenames "${PRED_FILENAMES}" \
    --pairs_csv "${PAIRS_CSV}" \
    --cameras_root "${CAMERAS_ROOT}" \
    --vipe_cache_root "${vipe_cache}" \
    --output_json "${out_json}" \
    --num_frames "${NUM_FRAMES}" \
    --target_w "${TARGET_W}" \
    --target_h "${TARGET_H}" \
    --vipe_python "${VIPE_PYTHON}" \
    --gpu "${GPU}" \
    --phase vipe \
    "${EXTRA_ARGS[@]}"
fi

# --- Phase 2: trajectory comparison + metrics ---------------------------------
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "metrics" ]; then
  echo "[eval-camera-vipe] phase=metrics (${METHOD})"
  holo4d_activate_conda "${EVAL_CONDA_ENV}"
  python "${SCRIPT_DIR}/vipe_camera_eval.py" \
    --method_pred_root "${METHOD_PRED_ROOT}" \
    --pred_filenames "${PRED_FILENAMES}" \
    --pairs_csv "${PAIRS_CSV}" \
    --cameras_root "${CAMERAS_ROOT}" \
    --vipe_cache_root "${vipe_cache}" \
    --output_json "${out_json}" \
    --num_frames "${NUM_FRAMES}" \
    --target_w "${TARGET_W}" \
    --target_h "${TARGET_H}" \
    --vipe_python "/dev/null" \
    --gpu "${GPU}" \
    --phase metrics \
    "${EXTRA_ARGS[@]}"
fi

echo "[eval-camera-vipe] ${METHOD} -> ${out_json}"
