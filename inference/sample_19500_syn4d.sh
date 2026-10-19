#!/usr/bin/env bash
# Reproduce the released Holo4D 49-frame Syn4D protocol.
# All paths are environment overrides so this launcher contains no private
# dataset or checkpoint files.
set -euo pipefail

ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SYN4D_ROOT="${SYN4D_ROOT:?Set SYN4D_ROOT to Syn4D_v1_stride_5}"
WAN_ROOT="${WAN_ROOT:-$("${PYTHON_BIN}" -m holo4d.hub wan t2v)}"
CHECKPOINT="${CHECKPOINT:-$("${PYTHON_BIN}" -m holo4d.hub checkpoint t2v)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/syn4d_49_chunk}"
CAPTION_DIR="${SYN4D_CAPTION_DIR:-}"

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

MODEL_ID="Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/Wan2.1_VAE.pth"
ARGS=(
  --model_id_with_origin_paths "${MODEL_ID}"
  --tokenizer_path "${WAN_ROOT}/google/umt5-xxl"
  --checkpoint_path "${CHECKPOINT}"
  --camera_control_checkpoint "${WAN_ROOT}/diffusion_pytorch_model*.safetensors"
  --track_config "${ROOT}/config/holo4d/default_conv3d_patchify_srcdepth_srcrgb.yaml"
  --no_dense_branch
  --use_query_frame_impulse_condition
  --use_plucker_camera_control
  --enable_v2v_plucker_camera_control
  --use_plucker_raymap_tokens
  --use_perlayer_camera_control
  --camera_injection_mode projector
  --camera_plucker_encoder_mode patchify
  --track_injection_mode per_block
  --track_injection_block_mode concat_project
  --warped_condition_mode off
  --track_spatial_stride 1
  --syn4d_root "${SYN4D_ROOT}"
  --use_mp4_rgb
  --scenes "${SCENES:-flying_group}"
  --sequence_idxs "${SEQUENCE_IDXS:-0}"
  --camera_pairs "${CAMERA_PAIRS:-0-1}"
  --camera_pairs_by_sequence "${CAMERA_PAIRS_BY_SEQUENCE:-}"
  --start_frame "${START_FRAME:-5}"
  --height "${HEIGHT:-384}" --width "${WIDTH:-672}" --num_frames 49
  --num_inference_steps "${NUM_INFERENCE_STEPS:-25}"
  --cfg_scale "${CFG_SCALE:-5.0}" --sigma_shift 5.0 --seed "${SEED:-42}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ -n "${CAPTION_DIR}" ]]; then
  ARGS+=(--syn4d_caption_dir "${CAPTION_DIR}")
fi
if [[ -n "${SOURCE_CONDITION_ROOT:-}" ]]; then
  ARGS+=(--source_condition_root "${SOURCE_CONDITION_ROOT}" --source_condition_backend "${SOURCE_CONDITION_BACKEND:?Set SOURCE_CONDITION_BACKEND with SOURCE_CONDITION_ROOT}")
fi
if [[ "${CAPTURE_FINAL_LATENTS:-0}" == "1" ]]; then ARGS+=(--capture_final_latents); fi
if [[ "${PROFILE_STAGE_BREAKDOWN:-0}" == "1" ]]; then ARGS+=(--profile_stage_breakdown); fi
if [[ -n "${PROFILE_JSONL:-}" ]]; then ARGS+=(--profile_jsonl "${PROFILE_JSONL}"); fi

cd "${ROOT}"
exec python -u examples/wan2.1_fun/eval_track4d360_hybrid_dense_syn4d.py "${ARGS[@]}"
