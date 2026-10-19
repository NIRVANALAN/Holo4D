#!/usr/bin/env bash
# Prediction vs paired target RGB, NOT the DAVIS source-proxy protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec python -u eval/common/video_quality/tools/eval_video_quality.py \
  --mode paired_csv --paired_csv "${PAIRED_CSV:?}" \
  --pred_root "${PRED_ROOT:?}" --metrics fvd --max_frames 49 \
  --eval_height 288 --eval_width 512 --fvd_method styleganv \
  --common_metrics_root "${COMMON_METRICS_ROOT:-${ROOT}/eval/common/video_quality/common_metrics_on_video_quality}" \
  --device "${DEVICE:-cuda}" --output_json "${OUTPUT_JSON:?}" \
  --per_sample_csv "${PER_SAMPLE_CSV:?}" "$@"
