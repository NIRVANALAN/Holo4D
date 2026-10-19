#!/usr/bin/env bash
# Reproduce the two-model long-video Syn4D rollout.
#
# Stage 1: source-video-only T2V with the public 384p step-19500 checkpoint.
# Stage 2: native I2V with the step-22500 checkpoint, conditioned on the
# generated target endpoint (last_target.png) from stage 1.  No ground-truth
# target image is supplied to either stage.
set -euo pipefail

ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SYN4D_ROOT="${SYN4D_ROOT:?Set SYN4D_ROOT to Syn4D_v1_stride_5 (or an equivalent local archive)}"
T2V_ROOT="${T2V_ROOT:-$("${PYTHON_BIN}" -m holo4d.hub wan t2v)}"
I2V_ROOT="${I2V_ROOT:-$("${PYTHON_BIN}" -m holo4d.hub wan i2v)}"
SEED_CHECKPOINT="${I2V_SEED_CHECKPOINT_PATH:-$("${PYTHON_BIN}" -m holo4d.hub checkpoint t2v)}"
CONTINUATION_CHECKPOINT="${I2V_CONTINUATION_CHECKPOINT_PATH:-$("${PYTHON_BIN}" -m holo4d.hub checkpoint i2v)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/mixed_t2v_i2v_syn4d}"
CAPTION_DIR="${SYN4D_CAPTION_DIR:-}"

# The two initialization bundles are explicit so the first stage cannot
# accidentally load I2V image-encoder weights and the second stage cannot
# accidentally omit them.
T2V_MODEL_PREFIX="${T2V_MODEL_PREFIX:-Wan-AI/Wan2.1-T2V-14B}"
I2V_MODEL_PREFIX="${I2V_MODEL_PREFIX:-Wan-AI/Wan2.1-I2V-14B-480P}"
T5_PATH="${T5_PATH:-${T2V_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"
VAE_PATH="${VAE_PATH:-${T2V_ROOT}/Wan2.1_VAE.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${T2V_ROOT}/google/umt5-xxl}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-${I2V_ROOT}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"
SEED_CAMERA_CONTROL_CHECKPOINT="${I2V_SEED_CAMERA_CONTROL_CHECKPOINT:-${T2V_ROOT}/diffusion_pytorch_model*.safetensors}"
CONTINUATION_CAMERA_CONTROL_CHECKPOINT="${I2V_CONTINUATION_CAMERA_CONTROL_CHECKPOINT:-${I2V_ROOT}/diffusion_pytorch_model*.safetensors}"

SEED_MODEL_ID="${I2V_SEED_MODEL_ID_WITH_ORIGIN_PATHS:-${T2V_MODEL_PREFIX}:${T2V_ROOT}/diffusion_pytorch_model*.safetensors,${T2V_MODEL_PREFIX}:${T5_PATH},${T2V_MODEL_PREFIX}:${VAE_PATH}}"
CONTINUATION_MODEL_ID="${I2V_CONTINUATION_MODEL_ID_WITH_ORIGIN_PATHS:-${I2V_MODEL_PREFIX}:${I2V_ROOT}/diffusion_pytorch_model*.safetensors,${I2V_MODEL_PREFIX}:${T5_PATH},${I2V_MODEL_PREFIX}:${VAE_PATH},${I2V_MODEL_PREFIX}:${IMAGE_ENCODER_PATH}}"

check_input() {
  local value="$1"
  local label="$2"
  if [[ -e "${value}" ]] || compgen -G "${value}" >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: ${label} is missing or matches no files: ${value}" >&2
  exit 1
}
check_input "${SEED_CHECKPOINT}" "seed checkpoint"
check_input "${CONTINUATION_CHECKPOINT}" "continuation checkpoint"
check_input "${T2V_ROOT}" "T2V model root"
check_input "${I2V_ROOT}" "I2V model root"
check_input "${SYN4D_ROOT}" "Syn4D root"
check_input "${T5_PATH}" "T5 encoder"
check_input "${VAE_PATH}" "VAE"
check_input "${TOKENIZER_PATH}" "tokenizer"
check_input "${IMAGE_ENCODER_PATH}" "I2V image encoder"
check_input "${SEED_CAMERA_CONTROL_CHECKPOINT}" "seed camera-control base"
check_input "${CONTINUATION_CAMERA_CONTROL_CHECKPOINT}" "continuation camera-control base"

export PYTHONUNBUFFERED=1
export HOLO4D_ALLOW_ONLINE_T5_ENCODE="${HOLO4D_ALLOW_ONLINE_T5_ENCODE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# th29py312 can expose both xformers FA3 and a second FA3 extension; loading
# both aborts at import time with a duplicate TORCH_LIBRARY registration. Use
# the tested FA2 path by default on every GPU. Set HOLO4D_FLASH_ATTN3=1 only
# in an environment with exactly one FA3 build.
if [[ "${HOLO4D_FLASH_ATTN3:-auto}" == "1" ]]; then
  unset DISABLE_FLASH_ATTN_3 || true
else
  export DISABLE_FLASH_ATTN_3=1
fi

ARGS=(
  --model_id_with_origin_paths "${CONTINUATION_MODEL_ID}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --checkpoint_path "${CONTINUATION_CHECKPOINT}"
  --i2v_seed_checkpoint_path "${SEED_CHECKPOINT}"
  --i2v_continuation_checkpoint_path "${CONTINUATION_CHECKPOINT}"
  --i2v_seed_model_id_with_origin_paths "${SEED_MODEL_ID}"
  --i2v_continuation_model_id_with_origin_paths "${CONTINUATION_MODEL_ID}"
  --i2v_seed_camera_control_checkpoint "${SEED_CAMERA_CONTROL_CHECKPOINT}"
  --i2v_continuation_camera_control_checkpoint "${CONTINUATION_CAMERA_CONTROL_CHECKPOINT}"
  --camera_control_checkpoint "${CONTINUATION_CAMERA_CONTROL_CHECKPOINT}"
  --track_config "${TRACK_CONFIG:-${ROOT}/config/holo4d/default_conv3d_patchify_srcdepth_srcrgb.yaml}"
  --track_injection_mode per_block
  --track_injection_block_mode concat_project
  --use_query_frame_impulse_condition
  --use_plucker_camera_control
  --enable_v2v_plucker_camera_control
  --use_perlayer_camera_control
  --camera_injection_mode projector
  --camera_plucker_encoder_mode patchify
  --use_plucker_raymap_tokens
  --no_dense_branch
  --syn4d_root "${SYN4D_ROOT}"
  --syn4d_valid_sequences_file "${SYN4D_VALID_SEQUENCES_FILE:-${ROOT}/config/syn4d_valid_sequences.txt}"
  --use_mp4_rgb
  --height "${HEIGHT:-384}" --width "${WIDTH:-672}" --num_frames 49
  --track_spatial_stride "${TRACK_SPATIAL_STRIDE:-1}"
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}"
  --cfg_scale "${CFG_SCALE:-5.0}" --sigma_shift "${SIGMA_SHIFT:-5.0}" --seed "${SEED:-42}"
  --i2v_continuation_pairs "${NUM_PAIRS:-1}"
  --i2v_continuation_seed "${SELECTION_SEED:-20260717}"
  --i2v_continuation_shard_index "${SHARD_INDEX:-0}"
  --i2v_continuation_num_shards "${NUM_SHARDS:-1}"
  --i2v_resume_stage "${I2V_RESUME_STAGE:-both}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ -n "${FIXED_SOURCE_SEQUENCES:-}" ]]; then
  ARGS+=(--i2v_fixed_source_sequences "${FIXED_SOURCE_SEQUENCES}" --i2v_fixed_seed_start_frame "${FIXED_SEED_START_FRAME:-5}")
fi
if (( ${LONG_HORIZON_FRAMES:-0} > 0 )); then
  if (( LONG_HORIZON_FRAMES < 50 )); then
    echo "ERROR: LONG_HORIZON_FRAMES must be at least 50" >&2
    exit 2
  fi
  ARGS+=(
    --i2v_long_horizon_frames "${LONG_HORIZON_FRAMES}"
    --i2v_long_horizon_seed_start_frame "${LONG_HORIZON_SEED_START_FRAME:-0}"
    --i2v_long_horizon_sequence_index "${LONG_HORIZON_SEQUENCE_INDEX:-0}"
    --i2v_long_horizon_case_index "${LONG_HORIZON_CASE_INDEX:-0}"
    --i2v_long_horizon_src_cam_idx "${LONG_HORIZON_SRC_CAM_IDX:-5}"
    --i2v_long_horizon_tgt_cam_idx "${LONG_HORIZON_TGT_CAM_IDX:-6}"
  )
  if [[ -n "${LONG_HORIZON_SOURCE_CONDITION_ROOT:-}" ]]; then
    ARGS+=(
      --i2v_long_horizon_source_condition_root "${LONG_HORIZON_SOURCE_CONDITION_ROOT}"
      --i2v_long_horizon_source_condition_backend "${LONG_HORIZON_SOURCE_CONDITION_BACKEND:?Set LONG_HORIZON_SOURCE_CONDITION_BACKEND}"
    )
  fi
fi
if [[ -n "${CAPTION_DIR}" ]]; then ARGS+=(--syn4d_caption_dir "${CAPTION_DIR}"); fi
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then ARGS+=(--force_rerun); fi
if [[ "${CAPTURE_FINAL_LATENTS:-0}" == "1" ]]; then ARGS+=(--capture_final_latents); fi
if [[ "${PROFILE_STAGE_BREAKDOWN:-0}" == "1" ]]; then ARGS+=(--profile_stage_breakdown); fi
if [[ -n "${PROFILE_JSONL:-}" ]]; then ARGS+=(--profile_jsonl "${PROFILE_JSONL}"); fi

cd "${ROOT}"
echo "[Holo4D] source-only T2V seed: ${SEED_CHECKPOINT}"
echo "[Holo4D] generated-boundary I2V continuation: ${CONTINUATION_CHECKPOINT}"
echo "[Holo4D] output: ${OUTPUT_DIR}"
python -u examples/wan2.1_fun/eval_track4d360_hybrid_dense_syn4d.py "${ARGS[@]}"

if [[ "${STITCH_OUTPUT:-1}" == "1" ]]; then
  python -m holo4d.tools.i2v_continuation_long_video \
    --run-root "${OUTPUT_DIR}" \
    --fps "${OUTPUT_FPS:-16}"
fi
