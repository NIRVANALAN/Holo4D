#!/bin/bash
# ===========================================================================
# Global Track4D-360 video-metric entry point.
#
# Community-facing contract:
#   - this is the only top-level video eval launcher;
#   - per-run metadata lives in eval/runs/<method>.sh;
#   - shared metric implementation lives in eval/common/eval_video_metrics.sh.
#
# Usage:
#   bash eval/run_eval_video.sh
#   METHODS=ours_14b,vista4d_4rc FORCE=1 GPU=0 bash eval/run_eval_video.sh
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
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"

IFS=',' read -ra METHOD_ARR <<< "${METHODS}"

echo "[eval-video] METHODS=${METHODS}"
echo "[eval-video] GPU=${GPU} FORCE=${FORCE}"

fail=0
for method in "${METHOD_ARR[@]}"; do
  run_sh="${RUNS_DIR}/${method}.sh"
  if [ ! -f "${run_sh}" ]; then
    echo "[eval-video] ERROR: missing per-run wrapper: ${run_sh}" >&2
    fail=$((fail + 1))
    continue
  fi
  echo
  echo "================================================================"
  echo "[eval-video] ${method}"
  echo "================================================================"
  EVAL_METRIC=video GPU="${GPU}" FORCE="${FORCE}" EVAL_ROOT="${EVAL_ROOT}" \
    bash "${run_sh}" || fail=$((fail + 1))
done

echo
echo "[eval-video] aggregating summary.tsv..."
${PYTHON_BIN:-python} "${SCRIPT_DIR}/common/aggregate_metrics.py" \
  --results_root "${EVAL_ROOT}/results" \
  --output_tsv "${EVAL_ROOT}/results/summary.tsv"

echo "[eval-video] summary -> ${EVAL_ROOT}/results/summary.tsv"
exit "${fail}"
