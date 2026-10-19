#!/usr/bin/env bash
# Run in an environment with VBench 0.1.5 and its upstream dependencies/weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
RESULT_DIR="${RESULT_DIR:?}"
VIDEOS_PATH="${VIDEOS_PATH:?}"
mkdir -p "${RESULT_DIR}"
python -m holo4d.tools.summarize_davis_vbench --provenance "${RESULT_DIR}/scorer_provenance.json"
for dimension in aesthetic_quality imaging_quality subject_consistency background_consistency temporal_style; do
  vbench evaluate --dimension "${dimension}" --videos_path "${VIDEOS_PATH}" \
    --mode custom_input --prompt_file "${VIDEOS_PATH}/prompts.json" \
    --output_path "${RESULT_DIR}" --imaging_quality_preprocessing_mode longer --ngpus 1
done
exec python -m holo4d.tools.summarize_davis_vbench --result-dir "${RESULT_DIR}" \
  --output "${RESULT_DIR}/summary.json" --expected "${EXPECTED_VIDEOS:-100}"
