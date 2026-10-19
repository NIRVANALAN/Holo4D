#!/bin/bash
# ===========================================================================
# Depth Anything 3 camera-trajectory eval for ONE already-sampled baseline.
#
# Drop-in sibling of eval_camera_metrics_vggt_omega.sh. It keeps the same
# method/prediction-root contract, but estimates camera poses with DA3.
#
# Output:
#   ${BASELINES_ROOT}/eval/results/<method>/eval_camera_metrics_da3.json
#
# Usage:
#   METHOD=ours_14b \
#   METHOD_PRED_ROOT=${BR}/ours_davis50_2trajperseq_14b/288x512/step-5000 \
#   PRED_FILENAMES='pred_rgb.mp4,{traj}.mp4,pred.mp4' \
#   GPU=1 FORCE=1 \
#     bash eval/common/eval_camera_metrics_da3.sh
# ===========================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/benchmark_env.sh"

EVAL_ROOT="${EVAL_ROOT:-${BASELINES_ROOT}/eval}"
METHOD="${METHOD:?METHOD env is required}"
METHOD_PRED_ROOT="${METHOD_PRED_ROOT:?METHOD_PRED_ROOT env is required}"
DA3_REPO="${DA3_REPO:-${BASELINES_REPO_ROOT}/Depth-Anything-3}"
DA3_MODEL_NAME="${DA3_MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
DA3_PROCESS_RES="${DA3_PROCESS_RES:-504}"
DA3_PROCESS_RES_METHOD="${DA3_PROCESS_RES_METHOD:-upper_bound_resize}"
NUM_FRAMES="${NUM_FRAMES:-49}"
TARGET_W="${TARGET_W:-512}"
TARGET_H="${TARGET_H:-288}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
FORCE_POSE="${FORCE_POSE:-0}"
LIMIT_PAIRS="${LIMIT_PAIRS:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
PHASE="${PHASE:-all}"
DA3_CONDA_ENV="${DA3_CONDA_ENV:-holo4d}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-batrack}"
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

if [ -z "${PRED_FILENAMES:-}" ]; then
  PRED_FILENAMES='pred.mp4,{traj}.mp4,gen.mp4,output.mp4,render.mp4,color_{traj}.mp4'
fi

da3_cache="${EVAL_ROOT}/_da3/${METHOD}"
out_json="${EVAL_ROOT}/results/${METHOD}/eval_camera_metrics_da3.json"
mkdir -p "$(dirname "${out_json}")" "${da3_cache}"

if [ "${FORCE}" != "1" ] && [ -f "${out_json}" ]; then
  echo "[eval-camera-da3] skip ${METHOD}: ${out_json} exists (FORCE=1 to re-run)"
  head -25 "${out_json}"
  exit 0
fi

if command -v module >/dev/null 2>&1; then
  module load cudatoolkit PrgEnv-gnu || true
fi

EXTRA_ARGS=()
[ -n "${LIMIT_PAIRS}" ] && EXTRA_ARGS+=(--limit_pairs "${LIMIT_PAIRS}")
if [ "${NUM_SHARDS}" != "1" ]; then
  EXTRA_ARGS+=(--num_shards "${NUM_SHARDS}" --shard_index "${SHARD_INDEX}")
fi
[ "${FORCE_POSE}" = "1" ] && EXTRA_ARGS+=(--force_pose)

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "da3" ]; then
  echo "[eval-camera-da3] phase=DA3 inference (${METHOD}; GPU=${GPU})"
  holo4d_activate_conda "${DA3_CONDA_ENV}"
  python "${SCRIPT_DIR}/da3_camera_eval.py" \
    --method_pred_root "${METHOD_PRED_ROOT}" \
    --pred_filenames "${PRED_FILENAMES}" \
    --pairs_csv "${PAIRS_CSV}" \
    --cameras_root "${CAMERAS_ROOT}" \
    --da3_cache_root "${da3_cache}" \
    --output_json "${out_json}" \
    --da3_repo "${DA3_REPO}" \
    --model_name "${DA3_MODEL_NAME}" \
    --process_res "${DA3_PROCESS_RES}" \
    --process_res_method "${DA3_PROCESS_RES_METHOD}" \
    --num_frames "${NUM_FRAMES}" \
    --target_w "${TARGET_W}" \
    --target_h "${TARGET_H}" \
    --gpu "${GPU}" \
    --phase da3 \
    "${EXTRA_ARGS[@]}"
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "metrics" ]; then
  echo "[eval-camera-da3] phase=metrics (${METHOD})"
  holo4d_activate_conda "${EVAL_CONDA_ENV}"
  python "${SCRIPT_DIR}/da3_camera_eval.py" \
    --method_pred_root "${METHOD_PRED_ROOT}" \
    --pred_filenames "${PRED_FILENAMES}" \
    --pairs_csv "${PAIRS_CSV}" \
    --cameras_root "${CAMERAS_ROOT}" \
    --da3_cache_root "${da3_cache}" \
    --output_json "${out_json}" \
    --da3_repo "${DA3_REPO}" \
    --model_name "${DA3_MODEL_NAME}" \
    --process_res "${DA3_PROCESS_RES}" \
    --process_res_method "${DA3_PROCESS_RES_METHOD}" \
    --num_frames "${NUM_FRAMES}" \
    --target_w "${TARGET_W}" \
    --target_h "${TARGET_H}" \
    --gpu "${GPU}" \
    --phase metrics \
    "${EXTRA_ARGS[@]}"
fi

echo "[eval-camera-da3] ${METHOD} -> ${out_json}"
