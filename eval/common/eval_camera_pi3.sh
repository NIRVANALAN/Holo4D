#!/usr/bin/env bash
# Source+generated joint Pi3; source-only fit, never target-fitted alignment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec python -u -m holo4d.tools.eval_vista4d_camera_pi3 \
  --prediction-root "${METHOD_PRED_ROOT:?}" \
  --prediction-filenames "${PRED_FILENAMES:-pred.mp4,pred_rgb.mp4}" \
  --pairs-csv "${PAIRS_CSV:?}" \
  --source-root "${SOURCE_ROOT:?}" \
  --target-camera-root "${CAMERAS_ROOT:?}" \
  --vista4d-repo "${VISTA4D_REPO:?}" \
  --cache-root "${POSE_CACHE_ROOT:?}" \
  --output-json "${OUTPUT_JSON:?}" \
  --height 384 --width 672 --num-frames 49 \
  --target-camera-gauge "${TARGET_CAMERA_GAUGE:-shared_world}" "$@"
