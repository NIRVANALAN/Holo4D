#!/usr/bin/env bash
# Portable dense example; existing activated CUDA environment is respected.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${HOLO4D_FLASH_ATTN3:-0}" == "1" ]]; then
  export DISABLE_FLASH_ATTN_3=0
else
  export DISABLE_FLASH_ATTN_3="${DISABLE_FLASH_ATTN_3:-1}"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${EXAMPLE_MANIFEST:-}" ]]; then
  EXAMPLE_MANIFEST="$(python -m holo4d.hub examples)"
fi
exec python -u inference/sample_examples.py --manifest "$EXAMPLE_MANIFEST" \
  --example "${EXAMPLE:-davis_bear}" --model "${MODEL:-14b}" \
  --output-dir "${OUTPUT_DIR:-outputs/examples}" "$@"
