#!/bin/bash
# ===========================================================================
# VGGT-Omega camera-trajectory eval for ONE baseline (DAVIS-50 x 2-traj).
#
# Drop-in sibling of eval_camera_metrics_vipe.sh. It keeps the same method
# roster inputs and output schema, but estimates camera poses with VGGT-Omega.
#
# Default checkpoint:
#   Set VGGT_OMEGA_CHECKPOINT to the downloaded vggt_omega_1b_512.pt.
#
# Output:
#   ${BASELINES_ROOT}/eval/results/<method>/eval_camera_metrics_vggt_omega.json
#
# Usage:
#   METHOD=ours_1p3b \
#   METHOD_PRED_ROOT=${BR}/ours_davis50_2trajperseq_1p3b/288x512/step-3000 \
#   PRED_FILENAMES='pred_rgb.mp4,{traj}.mp4,pred.mp4' \
#     bash eval/eval_camera_metrics_vggt_omega.sh
#
# Knobs mirror the ViPE wrapper:
#   FORCE=1                         re-run if output JSON exists
#   FORCE_POSE=1                    re-run VGGT-Omega even if pose cache exists
#   GPU=0                           CUDA device for VGGT-Omega
#   LIMIT_PAIRS=5                   smoke first N pairs
#   NUM_SHARDS=8 SHARD_INDEX=0      evaluate one row-index shard
#   PHASE=vggt|metrics|all          default 'all'
#   TARGET_W=512 TARGET_H=288 NUM_FRAMES=49
#   VGGT_OMEGA_REPO=/path/to/vggt-omega
#   VGGT_OMEGA_IMAGE_RESOLUTION=512
#   VGGT_OMEGA_IMAGE_MODE=balanced|max_size
# ===========================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/benchmark_env.sh"
source "${SCRIPT_DIR}/external_method_env.sh"
REPO_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"
track4d_configure_external_methods "${REPO_ROOT}"

EVAL_ROOT="${EVAL_ROOT:-${BASELINES_ROOT}/eval}"
METHOD="${METHOD:?METHOD env is required (e.g. ours_1p3b, recammaster, ...)}"
METHOD_PRED_ROOT="${METHOD_PRED_ROOT:?METHOD_PRED_ROOT env is required}"
VGGT_OMEGA_IMAGE_RESOLUTION="${VGGT_OMEGA_IMAGE_RESOLUTION:-512}"
VGGT_OMEGA_IMAGE_MODE="${VGGT_OMEGA_IMAGE_MODE:-balanced}"
NUM_FRAMES="${NUM_FRAMES:-49}"
TARGET_W="${TARGET_W:-672}"
TARGET_H="${TARGET_H:-384}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
FORCE="${FORCE:-0}"
FORCE_POSE="${FORCE_POSE:-0}"
LIMIT_PAIRS="${LIMIT_PAIRS:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
PHASE="${PHASE:-all}"
VGGT_OMEGA_CONDA_ENV="${VGGT_OMEGA_CONDA_ENV:-holo4d}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-${VGGT_OMEGA_CONDA_ENV}}"
PAIRS_CSV="${PAIRS_CSV:-$(_resolve_baseline_asset davis_trajectories.csv)}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ -z "${PRED_FILENAMES:-}" ]; then
  PRED_FILENAMES='pred.mp4,{traj}.mp4,gen.mp4,output.mp4,render.mp4,color_{traj}.mp4'
fi

vggt_cache="${EVAL_ROOT}/_vggt_omega/${METHOD}"
out_json="${EVAL_ROOT}/results/${METHOD}/eval_camera_metrics_vggt_omega.json"
mkdir -p "$(dirname "${out_json}")" "${vggt_cache}"

if [ "${FORCE}" != "1" ] && [ -f "${out_json}" ]; then
  echo "[eval-camera-vggt-omega] skip ${METHOD}: ${out_json} exists (FORCE=1 to re-run)"
  head -25 "${out_json}"
  exit 0
fi

if [ "${TRACK4D360_SKIP_CLUSTER_MODULES:-0}" != "1" ] && command -v module >/dev/null 2>&1; then
  module load cudatoolkit PrgEnv-gnu || true
fi

EXTRA_ARGS=()
[ -n "${LIMIT_PAIRS}" ] && EXTRA_ARGS+=(--limit_pairs "${LIMIT_PAIRS}")
if [ "${NUM_SHARDS}" != "1" ]; then
  EXTRA_ARGS+=(--num_shards "${NUM_SHARDS}" --shard_index "${SHARD_INDEX}")
fi
[ "${FORCE_POSE}" = "1" ] && EXTRA_ARGS+=(--force_pose)

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "vggt" ]; then
  echo "[eval-camera-vggt-omega] phase=VGGT-Omega inference (${METHOD}; GPU=${GPU})"
  if [ ! -f "${PAIRS_CSV}" ]; then
    echo "ERROR: missing PAIRS_CSV for VGGT-Omega metrics: ${PAIRS_CSV}" >&2
    echo "Set PAIRS_CSV explicitly or sync davis_trajectories.csv." >&2
    exit 1
  fi
  if [ ! -d "${CAMERAS_ROOT}" ]; then
    echo "ERROR: missing CAMERAS_ROOT for VGGT-Omega metrics: ${CAMERAS_ROOT}" >&2
    echo "Set CAMERAS_ROOT explicitly or sync cameras_2traj." >&2
    exit 1
  fi
  holo4d_activate_conda "${VGGT_OMEGA_CONDA_ENV}"
  ensure_ffmpeg_on_path
  python -u "${SCRIPT_DIR}/vggt_omega_camera_eval.py" \
    --method_pred_root "${METHOD_PRED_ROOT}" \
    --pred_filenames "${PRED_FILENAMES}" \
    --pairs_csv "${PAIRS_CSV}" \
    --cameras_root "${CAMERAS_ROOT}" \
    --vggt_cache_root "${vggt_cache}" \
    --output_json "${out_json}" \
    --checkpoint "${VGGT_OMEGA_CHECKPOINT}" \
    --vggt_omega_repo "${VGGT_OMEGA_REPO}" \
    --image_resolution "${VGGT_OMEGA_IMAGE_RESOLUTION}" \
    --image_mode "${VGGT_OMEGA_IMAGE_MODE}" \
    --num_frames "${NUM_FRAMES}" \
    --target_w "${TARGET_W}" \
    --target_h "${TARGET_H}" \
    --gpu "${GPU}" \
    --phase vggt \
    "${EXTRA_ARGS[@]}"
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "metrics" ]; then
  echo "[eval-camera-vggt-omega] phase=metrics (${METHOD})"
  if [ ! -f "${PAIRS_CSV}" ]; then
    echo "ERROR: missing PAIRS_CSV for VGGT-Omega metrics: ${PAIRS_CSV}" >&2
    echo "Set PAIRS_CSV explicitly or sync davis_trajectories.csv." >&2
    exit 1
  fi
  if [ ! -d "${CAMERAS_ROOT}" ]; then
    echo "ERROR: missing CAMERAS_ROOT for VGGT-Omega metrics: ${CAMERAS_ROOT}" >&2
    echo "Set CAMERAS_ROOT explicitly or sync cameras_2traj." >&2
    exit 1
  fi
  holo4d_activate_conda "${EVAL_CONDA_ENV}"
  ensure_ffmpeg_on_path
  python -u "${SCRIPT_DIR}/vggt_omega_camera_eval.py" \
    --method_pred_root "${METHOD_PRED_ROOT}" \
    --pred_filenames "${PRED_FILENAMES}" \
    --pairs_csv "${PAIRS_CSV}" \
    --cameras_root "${CAMERAS_ROOT}" \
    --vggt_cache_root "${vggt_cache}" \
    --output_json "${out_json}" \
    --checkpoint "${VGGT_OMEGA_CHECKPOINT}" \
    --vggt_omega_repo "${VGGT_OMEGA_REPO}" \
    --image_resolution "${VGGT_OMEGA_IMAGE_RESOLUTION}" \
    --image_mode "${VGGT_OMEGA_IMAGE_MODE}" \
    --num_frames "${NUM_FRAMES}" \
    --target_w "${TARGET_W}" \
    --target_h "${TARGET_H}" \
    --gpu "${GPU}" \
    --phase metrics \
    "${EXTRA_ARGS[@]}"
fi

echo "[eval-camera-vggt-omega] ${METHOD} -> ${out_json}"
