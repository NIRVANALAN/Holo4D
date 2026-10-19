#!/bin/bash
# ===========================================================================
# Compute video-quality metrics (clip_t, clip_f, clip_v, fvd) for ONE
# baseline's 100-pair output, using this repo's vendored eval_video_quality.py
# in `flat` mode.
#
# In-repo location: eval/common/eval_video_metrics.sh
# Caches + result JSONs live at the data-side ${BASELINES_ROOT}/eval (not
# inside the repo).
#
# FVD-V is intentionally excluded — it's a multiview-only metric (synchronized
# prediction/GT groups across views), and our flat-mode DAVIS NVS pipeline has
# exactly one rendered view per (seq, traj). To opt in anyway (e.g. for a
# syn4d-style multiview eval), set METRICS=...,fvd_v AND make sure the flat
# tree is grouped accordingly.
#
# Pipeline:
#   1. Build a flat tree at ${EVAL_ROOT}/_flat/<method>/{video<i>.mp4, dataset/}
#      from the baseline's predictions tree (build_flat_tree.py).
#   2. Run eval_video_quality.py --mode flat against that tree.
#   3. Outputs JSON + per-sample CSV at ${EVAL_ROOT}/results/<method>/eval_video_metrics.json
#
# Usage:
#   METHOD=ours_1p3b METHOD_PRED_ROOT=${BR}/ours_davis50_2trajperseq/288x512/step-3000 \
#     bash eval/common/eval_video_metrics.sh
#
# Required env:
#   METHOD             tag for the run (folder/file naming only)
#   METHOD_PRED_ROOT   the root containing <seq>/<traj>/<pred>.mp4
#                      (or .../predictions/<seq>/<traj>/pred.mp4 — auto-detected)
#
# Optional env:
#   PRED_FILENAMES     candidate filenames per pair (default tries pred.mp4, <traj>.mp4, cam_type01/video0.mp4, ...)
#   NUM_FRAMES=49      eval window
#   GPU=0              CUDA device for the metric models
#   METRICS=clip_t,clip_f,clip_v  (add fvd after installing the I3D weights)
#   FORCE=0            re-run if output already exists
# ===========================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/benchmark_env.sh"

EVAL_ROOT="${EVAL_ROOT:-${BASELINES_ROOT}/eval}"
VIDEO_QUALITY_ROOT="${VIDEO_QUALITY_ROOT:-${SCRIPT_DIR}/video_quality}"
VIDEO_QUALITY_SCRIPT="${VIDEO_QUALITY_SCRIPT:-${VIDEO_QUALITY_ROOT}/eval_video_quality.sh}"
COMMON_METRICS_ROOT="${COMMON_METRICS_ROOT:-${VIDEO_QUALITY_ROOT}/common_metrics_on_video_quality}"
RECAM_CONDA_ENV="${RECAM_CONDA_ENV:-${METRICS_CONDA_ENV:-holo4d}}"
METHOD="${METHOD:?METHOD env is required (e.g. ours_1p3b, recammaster, vista4d, ex4d, gen3c, trajcrafter)}"
METHOD_PRED_ROOT="${METHOD_PRED_ROOT:?METHOD_PRED_ROOT env is required}"
NUM_FRAMES="${NUM_FRAMES:-49}"
# Canonical eval resolution: every baseline (regardless of native render res:
# 288×512 ours, 384×672 vista4d, 480×832 recammaster, 704×1280 gen3c, ...)
# is resized to 288×512 inside eval_video_quality.py before computing
# fid/fvd/fvd_v/clip_*. This makes cross-method numbers directly comparable.
EVAL_HEIGHT="${EVAL_HEIGHT:-288}"
EVAL_WIDTH="${EVAL_WIDTH:-512}"
GPU="${GPU:-0}"
METRICS="${METRICS:-clip_t,clip_f,clip_v}"
# fvd_v is multiview-only — not applicable to flat single-view DAVIS NVS.
# Default off; set INCLUDE_FVD_V=1 explicitly only for syn4d/kubric multiview eval.
INCLUDE_FVD_V="${INCLUDE_FVD_V:-0}"
FORCE="${FORCE:-0}"
FLAT_PROGRESS_EVERY="${FLAT_PROGRESS_EVERY:-10}"
# NB: do NOT use `${VAR:-...{traj}...}` — bash brace-counting on the default
# value miscounts inner `{traj}` placeholders and leaves a stray `}` in the
# expansion, which python's str.format then chokes on.
if [ -z "${PRED_FILENAMES:-}" ]; then
  PRED_FILENAMES='pred.mp4,{traj}.mp4,cam_type01/video0.mp4,render.mp4,color_{traj}.mp4'
fi

flat_root="${EVAL_ROOT}/_flat/${METHOD}"
out_json="${EVAL_ROOT}/results/${METHOD}/eval_video_metrics.json"
out_csv="${EVAL_ROOT}/results/${METHOD}/eval_video_per_sample.csv"
mkdir -p "$(dirname "${out_json}")" "${flat_root}"

if [ "${FORCE}" != "1" ] && [ -f "${out_json}" ]; then
  echo "[eval-video] skip ${METHOD}: ${out_json} exists (set FORCE=1 to re-run)"
  cat "${out_json}"
  exit 0
fi

# 1. Build flat tree. Use the same metrics env by default; set METRICS_CONDA_ENV
# to use a separate metrics environment (default: the README holo4d environment).
holo4d_activate_conda "${HOLO4D_CONDA_ENV:-${METRICS_CONDA_ENV:-holo4d}}"
ensure_ffmpeg_on_path
if [ ! -d "${DAVIS_ROOT}" ]; then
  echo "ERROR: missing DAVIS_ROOT for video metrics: ${DAVIS_ROOT}" >&2
  echo "Set DAVIS_ROOT explicitly or sync the benchmark baselines bundle." >&2
  exit 1
fi
if [ ! -f "${PAIRS_FULL_CSV}" ]; then
  echo "ERROR: missing PAIRS_FULL_CSV for video metrics: ${PAIRS_FULL_CSV}" >&2
  echo "Set PAIRS_FULL_CSV explicitly or sync davis_trajectories_full.csv." >&2
  exit 1
fi
python "${EVAL_DIR}/build_flat_tree.py" \
  --method_pred_root "${METHOD_PRED_ROOT}" \
  --flat_root "${flat_root}" \
  --source_root "${DAVIS_ROOT}" \
  --pairs_csv "${PAIRS_FULL_CSV}" \
  --num_frames "${NUM_FRAMES}" \
  --pred_filenames "${PRED_FILENAMES}" \
  --progress_every "${FLAT_PROGRESS_EVERY}"

# 2. Run the in-repo eval_video_quality.py in flat mode.
holo4d_activate_conda "${RECAM_CONDA_ENV}"
cd "${VIDEO_QUALITY_ROOT}"
export PYTHONPATH="${VIDEO_QUALITY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export COMMON_METRICS_ROOT
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ ",${METRICS}," == *,fvd,* || ",${METRICS}," == *,fvd_v,* ]]; then
  if [ ! -f "${COMMON_METRICS_ROOT}/calculate_fvd.py" ]; then
    echo "ERROR: missing vendored FVD helper: ${COMMON_METRICS_ROOT}/calculate_fvd.py" >&2
    echo "Restore eval/common/video_quality/common_metrics_on_video_quality from the full source release." >&2
    exit 1
  fi
  for required_weight in \
    "${COMMON_METRICS_ROOT}/fvd/styleganv/i3d_torchscript.pt" \
    "${COMMON_METRICS_ROOT}/fvd/videogpt/i3d_pretrained_400.pt"; do
    if [ ! -s "${required_weight}" ]; then
      echo "ERROR: missing vendored FVD weight: ${required_weight}" >&2
      echo "Download the compatible I3D weight to this exact path before computing FVD." >&2
      exit 1
    fi
  done
fi

MODE=flat \
PRED_ROOT="${flat_root}" \
DATASET_PATH="${flat_root}/dataset" \
METADATA_CSV="${flat_root}/dataset/metadata.csv" \
PROMPT_FROM=source \
DEVICE=cuda \
METRICS="${METRICS}" \
OUTPUT_JSON="${out_json}" \
PER_SAMPLE_CSV="${out_csv}" \
MAX_FRAMES="${NUM_FRAMES}" \
EVAL_HEIGHT="${EVAL_HEIGHT}" \
EVAL_WIDTH="${EVAL_WIDTH}" \
INCLUDE_FVD_V="${INCLUDE_FVD_V}" \
  bash "${VIDEO_QUALITY_SCRIPT}"

echo "[eval-video] ${METHOD} -> ${out_json}"
