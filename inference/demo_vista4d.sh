#!/usr/bin/env bash
set -euo pipefail
ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export VIDEO_NAMES="${VIDEO_NAMES:-avocado-slice}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/vista4d_demo}"
exec bash "${ROOT}/inference/sample_vista4d.sh"
