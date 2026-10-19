#!/bin/bash
# ===========================================================================
# Global Track4D-360 camera-metric entry point.
#
# Community-facing contract:
#   - this is the only top-level camera eval launcher;
#   - per-run metadata lives in eval/runs/<method>.sh;
#   - shared camera engines live in eval/common/eval_camera_metrics_<backend>.sh.
#
# Usage:
#   CAMERA_BACKEND=vipe GPUS=0,1 bash eval/run_eval_camera.sh
#   CAMERA_BACKEND=da3 METHODS=ours_14b,vista4d_4rc GPUS=0,1 FORCE=1 bash eval/run_eval_camera.sh
# ===========================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
BR="${BASELINES_ROOT:-${SCRIPT_DIR}/../outputs/benchmarks}"
EVAL_ROOT="${EVAL_ROOT:-${BR}/eval}"

join_csv() {
  local IFS=,
  echo "$*"
}

MAIN_BASELINE_METHODS=(
  ours_14b_softsplat
  ours_14b
  vista4d_4rc
  ex4d_calibrated
  trajcrafter_stage2_4rc
  ours_1p3b
  recammaster
  gen3c_calibrated
)

RAYMAP_METHODS=(
  ours_14b_raymap_resume3500_eb48_432x768_step2500
  ours_14b_raymap_resume3500_eb48_432x768_steps10
  ours_14b_raymap_t2v_init_steps5
  ours_14b_raymap_t2v_init_steps10
  ours_14b_raymap_t2v_init_steps25
  ours_14b_raymap_t2v_init_steps50
)

ABLATION_METHODS=(
  ours_ablate_01_plucker_only
  ours_ablate_02b_zbuffer_rgb_dnull_null_tracks
  ours_ablate_02_zbuffer_rgbd_null_tracks
  ours_ablate_04_multiframe_rgbd_track3d
  ours_ablate_04_multiframe_rgbd_track3d_dynmask_fix
)

DEFAULT_METHODS="$(join_csv "${MAIN_BASELINE_METHODS[@]}" "${RAYMAP_METHODS[@]}" "${ABLATION_METHODS[@]}")"
METHODS="${METHODS:-${DEFAULT_METHODS}}"
CAMERA_BACKEND="${CAMERA_BACKEND:-vggt_omega}"
GPUS="${GPUS:-0,1}"
FORCE="${FORCE:-0}"
FORCE_POSE="${FORCE_POSE:-0}"
PHASE="${PHASE:-all}"
LIMIT_PAIRS="${LIMIT_PAIRS:-}"
TARGET_W="${TARGET_W:-672}"
TARGET_H="${TARGET_H:-384}"
NUM_FRAMES="${NUM_FRAMES:-49}"

case "${CAMERA_BACKEND}" in
  vipe) CAMERA_JSON_NAME="eval_camera_metrics_vipe.json"; SUMMARY_NAME="summary.tsv" ;;
  vggt_omega) CAMERA_JSON_NAME="eval_camera_metrics_vggt_omega.json"; SUMMARY_NAME="summary_vggt_omega.tsv" ;;
  da3) CAMERA_JSON_NAME="eval_camera_metrics_da3.json"; SUMMARY_NAME="summary_da3.tsv" ;;
  *)
    echo "[eval-camera] ERROR: unknown CAMERA_BACKEND=${CAMERA_BACKEND}" >&2
    echo "[eval-camera] Known backends: vipe, vggt_omega, da3" >&2
    exit 1
    ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
IFS=',' read -ra METHOD_ARR <<< "${METHODS}"
N_SHARDS=${#GPU_ARR[@]}
if [ "${N_SHARDS}" -lt 1 ]; then
  echo "[eval-camera] ERROR: GPUS is empty" >&2
  exit 1
fi

run_one_method() {
  local method="$1"
  local run_sh="${RUNS_DIR}/${method}.sh"
  local shard_root="${EVAL_ROOT}/_${CAMERA_BACKEND}_shards/${method}"
  local final_json="${EVAL_ROOT}/results/${method}/${CAMERA_JSON_NAME}"
  local fail=0

  if [ ! -f "${run_sh}" ]; then
    echo "[eval-camera] ERROR: missing per-run wrapper: ${run_sh}" >&2
    return 1
  fi
  if [ "${FORCE}" != "1" ] && [ -f "${final_json}" ]; then
    echo "[eval-camera] skip ${method}: ${final_json} exists (FORCE=1 to re-run)"
    return 0
  fi

  mkdir -p "${shard_root}/_logs" "${shard_root}/results" "${EVAL_ROOT}/results/${method}"
  echo
  echo "================================================================"
  echo "[eval-camera] ${method} backend=${CAMERA_BACKEND} shards=${N_SHARDS} GPUS=${GPUS}"
  echo "================================================================"

  declare -a PIDS=()
  for shard_idx in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$shard_idx]}"
    log="${shard_root}/_logs/shard${shard_idx}_gpu${gpu}.log"
    echo "[eval-camera] launch ${method} shard ${shard_idx}/${N_SHARDS} on GPU ${gpu} -> ${log}"
    (
      EVAL_METRIC=camera \
      CAMERA_BACKEND="${CAMERA_BACKEND}" \
      EVAL_ROOT="${shard_root}" \
      EVAL_METHOD_OVERRIDE="shard${shard_idx}" \
      GPU="${gpu}" \
      FORCE="${FORCE}" \
      FORCE_POSE="${FORCE_POSE}" \
      TARGET_W="${TARGET_W}" \
      TARGET_H="${TARGET_H}" \
      NUM_FRAMES="${NUM_FRAMES}" \
      LIMIT_PAIRS="${LIMIT_PAIRS}" \
      NUM_SHARDS="${N_SHARDS}" \
      SHARD_INDEX="${shard_idx}" \
      PHASE="${PHASE}" \
        bash "${run_sh}" > "${log}" 2>&1
    ) &
    PIDS+=("$!")
  done

  for pid in "${PIDS[@]}"; do
    wait "${pid}" || fail=$((fail + 1))
  done
  if [ "${fail}" -ne 0 ]; then
    echo "[eval-camera] FAIL ${method}: ${fail} shard(s) failed" >&2
    return 1
  fi

  if [ "${CAMERA_BACKEND}" = "vggt_omega" ] && [ "${PHASE}" = "vggt" ]; then
    echo "[eval-camera] ${method}: VGGT-Omega pose inference complete; skip metric merge for PHASE=vggt"
    return 0
  fi

  python "${SCRIPT_DIR}/common/merge_sharded_camera_metrics_vipe.py" \
    --method "${method}" \
    --shard_results_root "${shard_root}/results" \
    --num_shards "${N_SHARDS}" \
    --camera_json_name "${CAMERA_JSON_NAME}" \
    --output_json "${final_json}"
}

echo "[eval-camera] METHODS=${METHODS}"
echo "[eval-camera] CAMERA_BACKEND=${CAMERA_BACKEND} GPUS=${GPUS} FORCE=${FORCE}"

fail=0
for method in "${METHOD_ARR[@]}"; do
  run_one_method "${method}" || fail=$((fail + 1))
done

if [ "${CAMERA_BACKEND}" = "vggt_omega" ] && [ "${PHASE}" = "vggt" ]; then
  echo
  echo "[eval-camera] PHASE=vggt: pose inference only; skip summary aggregation"
  exit "${fail}"
fi

echo
echo "[eval-camera] aggregating ${SUMMARY_NAME}..."
${PYTHON_BIN:-python} "${SCRIPT_DIR}/common/aggregate_metrics.py" \
  --results_root "${EVAL_ROOT}/results" \
  --output_tsv "${EVAL_ROOT}/results/${SUMMARY_NAME}" \
  --camera_backend "${CAMERA_BACKEND}"

echo "[eval-camera] summary -> ${EVAL_ROOT}/results/${SUMMARY_NAME}"
exit "${fail}"
