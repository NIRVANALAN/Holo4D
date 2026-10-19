#!/usr/bin/env bash
# One-command 49 + 48 frame T2V-to-I2V rollout demo.
set -euo pipefail
ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HOLO4D_DEMO_BUNDLE_PATH="${HOLO4D_DEMO_BUNDLE_PATH:-$("${PYTHON_BIN}" -m holo4d.hub demo)}"
export SYN4D_ROOT="${SYN4D_ROOT:-${HOLO4D_DEMO_BUNDLE_PATH}}"
export FIXED_SOURCE_SEQUENCES=seq_000000
export FIXED_SEED_START_FRAME=5
export NUM_PAIRS=1
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export I2V_RESUME_STAGE="${I2V_RESUME_STAGE:-both}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/demo_i2v_97}"
unset LONG_HORIZON_FRAMES
exec bash "${ROOT}/inference/sample_mixed_t2v_i2v_syn4d.sh"
