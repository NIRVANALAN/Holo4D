#!/usr/bin/env bash
# One-command 49-frame T2V/V2V demo. Missing public assets are downloaded once.
set -euo pipefail
ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HOLO4D_DEMO_BUNDLE_PATH="${HOLO4D_DEMO_BUNDLE_PATH:-$("${PYTHON_BIN}" -m holo4d.hub demo)}"
export SYN4D_ROOT="${SYN4D_ROOT:-${HOLO4D_DEMO_BUNDLE_PATH}}"
export SCENES=flying_group
export SEQUENCE_IDXS=0
export CAMERA_PAIRS=0-1
export START_FRAME=5
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/demo_t2v_49}"
exec bash "${ROOT}/inference/sample_19500_syn4d.sh"
