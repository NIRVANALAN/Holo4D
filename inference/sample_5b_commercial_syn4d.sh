#!/usr/bin/env bash
# Reproduce the released commercial-mixture Wan 2.2 TI2V-5B first-window run.
#
# This is a V2V/T2V-style 49-frame inference entrypoint.  The checkpoint was
# trained without the persistent-identity branch, at 384x672, so this script
# intentionally uses the plain RGB-D track config and does not silently reuse
# the 14B or 704x1280 recipes.  All data and model locations are overridable;
# the defaults download public model/checkpoint assets through holo4d.hub.
set -euo pipefail

ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SYN4D_ROOT="${SYN4D_ROOT:?Set SYN4D_ROOT to a Syn4D root or compact demo bundle}"
WAN_ROOT="${WAN_ROOT:-$("${PYTHON_BIN}" -m holo4d.hub wan 5b)}"
CHECKPOINT="${CHECKPOINT:-$("${PYTHON_BIN}" -m holo4d.hub checkpoint 5b-commercial)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/wan22_5b_commercial_syn4d_49f}"
CAPTION_DIR="${SYN4D_CAPTION_DIR:-}"
T5_PATH="${T5_PATH:-${WAN_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"
VAE_PATH="${VAE_PATH:-${WAN_ROOT}/Wan2.2_VAE.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WAN_ROOT}/google/umt5-xxl}"
DIT_GLOB="${DIT_GLOB:-${WAN_ROOT}/diffusion_pytorch_model*.safetensors}"

check_input() {
  local value="$1" label="$2"
  if [[ -e "${value}" ]] || compgen -G "${value}" >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: ${label} is missing or matches no files: ${value}" >&2
  exit 1
}

check_input "${SYN4D_ROOT}" "Syn4D root"
check_input "${WAN_ROOT}" "Wan 2.2 TI2V-5B model root"
check_input "${CHECKPOINT}" "Holo4D 5B checkpoint"
check_input "${T5_PATH}" "Wan T5 encoder"
check_input "${VAE_PATH}" "Wan 2.2 VAE"
check_input "${TOKENIZER_PATH}" "Wan tokenizer"
check_input "${DIT_GLOB}" "Wan 2.2 TI2V-5B DiT shards"
check_input "${ROOT}/config/holo4d/default_conv3d_patchify_srcdepth_srcrgb.yaml" "5B track config"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# The public release uses the tested FA2 route by default.  Opt into FA3 only
# when the environment contains exactly one compatible implementation.
if [[ "${HOLO4D_FLASH_ATTN3:-auto}" == "1" ]]; then
  unset DISABLE_FLASH_ATTN_3 || true
else
  export DISABLE_FLASH_ATTN_3=1
fi

MODEL_ID="Wan-AI/Wan2.2-TI2V-5B:${DIT_GLOB},Wan-AI/Wan2.2-TI2V-5B:${T5_PATH},Wan-AI/Wan2.2-TI2V-5B:${VAE_PATH}"
ARGS=(
  --model_id_with_origin_paths "${MODEL_ID}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --checkpoint_path "${CHECKPOINT}"
  # The base DiT is the same plain TI2V-5B family used during training.  The
  # loader copies its backbone before loading the trained Holo4D state dict.
  --camera_control_checkpoint "${DIT_GLOB}"
  --track_config "${ROOT}/config/holo4d/default_conv3d_patchify_srcdepth_srcrgb.yaml"
  --no_dense_branch
  --use_query_frame_impulse_condition
  --use_plucker_camera_control
  --enable_v2v_plucker_camera_control
  --use_perlayer_camera_control
  --camera_injection_mode projector
  --camera_plucker_encoder_mode patchify
  --use_plucker_raymap_tokens
  --track_injection_mode per_block
  --track_injection_block_mode concat_project
  --track_spatial_stride 1
  --syn4d_root "${SYN4D_ROOT}"
  --use_mp4_rgb
  --scenes "${SCENES:-flying_group}"
  --sequence_idxs "${SEQUENCE_IDXS:-0}"
  --camera_pairs "${CAMERA_PAIRS:-0-1}"
  --start_frame "${START_FRAME:-5}"
  --height 384
  --width 672
  --num_frames 49
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}"
  --cfg_scale "${CFG_SCALE:-1.0}"
  --sigma_shift "${SIGMA_SHIFT:-5.0}"
  --seed "${SEED:-42}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ -n "${CAPTION_DIR}" ]]; then
  ARGS+=(--syn4d_caption_dir "${CAPTION_DIR}")
fi
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  ARGS+=(--force_rerun)
fi

cd "${ROOT}"
echo "[Holo4D/5B] checkpoint=${CHECKPOINT}"
echo "[Holo4D/5B] Wan base=${WAN_ROOT}"
echo "[Holo4D/5B] resolution=384x672x49 steps=${NUM_INFERENCE_STEPS:-50} cfg=${CFG_SCALE:-1.0}"
echo "[Holo4D/5B] output=${OUTPUT_DIR}"
exec "${PYTHON_BIN}" -u examples/wan2.1_fun/eval_track4d360_hybrid_dense_syn4d.py "${ARGS[@]}"
