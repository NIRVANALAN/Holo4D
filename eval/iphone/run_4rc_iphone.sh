#!/bin/bash
# ===========================================================================
# Run 4RC on the iPhone NVS source videos (cam0, first 49 frames at 1x).
#
# Mirrors the DAVIS 4RC launcher (run_davis_49f_inference.sh) — same checkpoint,
# same crop-to-16:9 + resize-to-512 preproc — but pointed at the iPhone source
# frames produced by build_iphone_pairs.py.
#
# Output structure (matches predict_iphone_nvs.py + align_4rc_to_iphone_lidar.py):
#     <BR_IPHONE>/4RC/<seq>/result.npz
#     <BR_IPHONE>/4RC/<seq>/frames/frame_<idx>.jpg   (4RC's pre-crop staging)
#
# Usage:
#   bash eval/iphone/run_4rc_iphone.sh
#   DEVICE=cuda:0 SEQUENCES=apple bash ...                  # subset / pick GPU
#   FORCE_RECREATE_INPUTS=1 bash ...                         # re-stage input dir
#
# Pre-reqs:
#   - conditions/<seq>/src_frames_1x/<idx>.png exist (run build_iphone_pairs.py)
#   - 4RC repo at third_party/4RC
#   - conda env "4rc" installed (see that repo's README; NOT the holo4d environment)
# ===========================================================================
set -euo pipefail

HOLO4D_ROOT="${HOLO4D_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

LOGS_ROOT="outputs"
BR_IPHONE="${BR_IPHONE:-${LOGS_ROOT}/benchmark/iphone_nvs}"
FOURRC_REPO="${FOURRC_REPO:-third_party/4RC}"

[ -d "${BR_IPHONE}/conditions" ] || {
  echo "ERROR: ${BR_IPHONE}/conditions missing — run build_iphone_pairs.py first." >&2
  exit 1
}
[ -d "${FOURRC_REPO}" ] || {
  echo "ERROR: 4RC repo not found at ${FOURRC_REPO}" >&2
  exit 1
}

# Inference knobs (match the DAVIS bench)
SEQUENCES="${SEQUENCES:-apple,block,paper-windmill,spin,teddy}"
START_FRAME="${START_FRAME:-0}"
NUM_FRAMES="${NUM_FRAMES:-49}"
TRACK_QUERY_IDX="${TRACK_QUERY_IDX:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-Luo-Yihang/4RC}"
SIZE="${SIZE:-512}"
# DEFAULT — match Vista4D's 16:9 center-crop convention (the iPhone-NVS paper
# protocol's de-facto convention; TrajectoryCrafter aspect-distorts but
# Vista4D and the rest center-crop to 16:9). 4RC's center_crop_to_aspect with
# CROP_WIDTH=512 CROP_HEIGHT=288 (target_aspect = 1.7778) carves a 405H × 720W
# strip from iPhone 720×960 portrait, then `--size 512` resizes long-edge to
# 504 → 4RC outputs at 280×504 landscape (DAVIS-native shape). Tracks live in
# the same 16:9 view the DiT will see after `center_crop_and_resize`.
#
# Alternative — full-scene capture, no crop (4RC sees whole iPhone, outputs
# 378×504 portrait): set CROP_WIDTH=720 CROP_HEIGHT=960 (matching iPhone src
# aspect makes the center-crop a no-op). Use this only if you want to compare
# against direct-resize aspect-distorted variants.
CROP_WIDTH="${CROP_WIDTH:-512}"
CROP_HEIGHT="${CROP_HEIGHT:-288}"
DTYPE="${DTYPE:-bf16-mixed}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REFINE_TRACK_VISUALIZATION="${REFINE_TRACK_VISUALIZATION:-0}"

INPUT_ROOT="${INPUT_ROOT:-${BR_IPHONE}/4RC_input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BR_IPHONE}/4RC}"

# ------------------- Stage iPhone source frames into a 4RC-friendly tree -------------------
# 4RC's discover_sequences expects <root>/<seq>/<*.png> (flat). Our conditions
# layout is <root>/conditions/<seq>/src_frames_1x/<idx>.png. So symlink each
# src_frames_1x dir under INPUT_ROOT/<seq>/.
mkdir -p "${INPUT_ROOT}"
for seq in $(echo "${SEQUENCES}" | tr ',' ' '); do
  src="${BR_IPHONE}/conditions/${seq}/src_frames_1x"
  if [ ! -d "${src}" ]; then
    echo "ERROR: ${src} missing — re-run build_iphone_pairs.py for ${seq}" >&2
    exit 1
  fi
  link="${INPUT_ROOT}/${seq}"
  if [ "${FORCE_RECREATE_INPUTS:-0}" = "1" ] && { [ -L "${link}" ] || [ -d "${link}" ]; }; then
    rm -rf "${link}"
  fi
  if [ ! -e "${link}" ]; then
    ln -s "${src}" "${link}"
    echo "[stage] ${link} -> ${src}"
  fi
done

# ------------------- Run 4RC inference -------------------
if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
fi
if ! type module >/dev/null 2>&1; then
  for lmod_init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    [ -f "${lmod_init}" ] && source "${lmod_init}" && break
  done
fi
type module >/dev/null 2>&1 && module load all/CUDA/12.4.0 2>/dev/null || true
conda activate "${CONDA_ENV:-4rc}"

mkdir -p "${OUTPUT_ROOT}"

cat <<EOF
==========================================================================
4RC on iPhone source (cam0, ${START_FRAME}..$((START_FRAME + NUM_FRAMES - 1)))
  Repo:           ${FOURRC_REPO}
  Input root:     ${INPUT_ROOT}
  Output root:    ${OUTPUT_ROOT}
  Sequences:      ${SEQUENCES}
  Crop aspect:    ${CROP_WIDTH}x${CROP_HEIGHT} (no-op when CROP/SRC aspects match → resize-only)
  Long-edge:      ${SIZE}  (patch-rounds to 504; resulting 4RC native ≈ 378×504 portrait at default CROP)
  Track query:    frame ${TRACK_QUERY_IDX}
  Checkpoint:     ${CHECKPOINT_DIR}
  Device:         ${DEVICE}
==========================================================================
EOF

LOG="${OUTPUT_ROOT}/4rc.log"
: > "${LOG}"

cd "${FOURRC_REPO}"

extra_args=()
[ -n "${SEQUENCES}" ] && extra_args+=(--sequence_names "${SEQUENCES}")
[ "${REFINE_TRACK_VISUALIZATION}" = "1" ] && extra_args+=(--refine_track_visualization)
[ "${SKIP_EXISTING}" = "1" ] && extra_args+=(--skip_existing)

python inference_davis.py \
  --davis_root "${INPUT_ROOT}" \
  --start_frame "${START_FRAME}" \
  --num_frames "${NUM_FRAMES}" \
  --track_query_idx "${TRACK_QUERY_IDX}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --size "${SIZE}" \
  --crop_width "${CROP_WIDTH}" \
  --crop_height "${CROP_HEIGHT}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --output_root "${OUTPUT_ROOT}" \
  "${extra_args[@]}" 2>&1 | tee -a "${LOG}"

echo ""
echo "Done -> ${OUTPUT_ROOT}"
echo "  Per-seq results: ${OUTPUT_ROOT}/<seq>/result.npz"
echo ""
echo "Next: align 4RC depth to iPhone LIDAR meters per seq:"
echo "  python ${HOLO4D_ROOT}/eval/iphone/align_4rc_to_iphone_lidar.py \\"
echo "    --output_root ${BR_IPHONE} \\"
echo "    --iphone_4rc_root ${OUTPUT_ROOT}"
