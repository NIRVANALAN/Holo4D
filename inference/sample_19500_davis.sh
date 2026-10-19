#!/usr/bin/env bash
# DAVIS-50 / two-trajectory entrypoint. See eval/davis and README for the
# required 4RC/CoWTracker/VGGT-Omega source geometry cache contract.
set -euo pipefail
ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DISABLE_FLASH_ATTN_3="${DISABLE_FLASH_ATTN_3:-1}"
cd "${ROOT}"
exec python -u examples/wan2.1_fun/eval_track4d360_davis_4rc_novel_traj.py "$@"
