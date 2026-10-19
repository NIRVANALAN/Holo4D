#!/usr/bin/env bash
# Public prepared source geometry; no 4RC install or private dataset paths needed.
set -euo pipefail
ROOT="${HOLO4D_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PACK_ROOT="${ROBOT_ROLLOUT_PACK:-${HOLO4D_CACHE_DIR:-$ROOT/outputs/hub_cache}/robot_long2}"
if [[ ! -s "$PACK_ROOT/examples/robot_long2_20261011/manifest.json" ]]; then
  hf download yslan/Holo4D-2026 --include 'examples/robot_long2_20261011/*' \
    --local-dir "$PACK_ROOT" --revision "${ROBOT_ROLLOUT_REVISION:-8c203a6d6ce07908fb94e6c0e5cc8ae258a997be}"
fi
python "$ROOT/inference/sample_robot_rollouts.py" \
  --manifest "$PACK_ROOT/examples/robot_long2_20261011/manifest.json" \
  --example "${EXAMPLE:-tri_static_high}" \
  --output "${OUTPUT_DIR:-$ROOT/outputs/robot_rollouts}" "$@"
