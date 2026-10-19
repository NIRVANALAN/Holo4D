#!/usr/bin/env bash
# Holo4D 384p full-parameter training launcher.
set -euo pipefail

ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_PATH="${OUTPUT_PATH:?Set OUTPUT_PATH for checkpoints}"
WAN_ROOT="${WAN_ROOT:-$("${PYTHON_BIN}" -m holo4d.hub wan t2v)}"
DATA_CONFIG="${DATA_CONFIG:-${ROOT}/config/holo4d/train_384p.yaml}"
TRACK_CONFIG="${TRACK_CONFIG:-${ROOT}/config/holo4d/default_conv3d_patchify_srcdepth_srcrgb.yaml}"

for input_path in "${WAN_ROOT}" "${DATA_CONFIG}" "${TRACK_CONFIG}"; do
  if [[ ! -e "${input_path}" ]]; then
    echo "ERROR: required input does not exist: ${input_path}" >&2
    exit 1
  fi
done

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HOLO4D_OFFLOAD_T5_TO_CPU="${HOLO4D_OFFLOAD_T5_TO_CPU:-1}"

MODEL_ID="Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/Wan2.1_VAE.pth"
LAUNCH_ARGS=(--num_processes "${NUM_PROCESSES:-1}")
if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
  LAUNCH_ARGS+=(--config_file "${ACCELERATE_CONFIG}")
fi

ARGS=(
  --model_id_with_origin_paths "${MODEL_ID}"
  --tokenizer_path "${WAN_ROOT}/google/umt5-xxl"
  --camera_control_checkpoint "${WAN_ROOT}/diffusion_pytorch_model*.safetensors"
  --task sft:v2v
  --model_variant vt2v
  --v2v_concat_mode width
  --track_config "${TRACK_CONFIG}"
  --track4d360_data_config "${DATA_CONFIG}"
  --output_path "${OUTPUT_PATH}"
  --height "${HEIGHT:-384}"
  --width "${WIDTH:-672}"
  --num_frames 49
  --steps_per_epoch "${STEPS_PER_EPOCH:-25000}"
  --num_epochs "${NUM_EPOCHS:-200}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --save_steps "${SAVE_STEPS:-250}"
  --max_checkpoints_to_keep "${MAX_CHECKPOINTS_TO_KEEP:-5}"
  --remove_prefix_in_ckpt pipe.dit.
  --track_injection_mode per_block
  --track_injection_block_mode concat_project
  --use_query_frame_impulse_condition
  --use_plucker_camera_control
  --enable_v2v_plucker_camera_control
  --use_plucker_raymap_tokens
  --use_perlayer_camera_control
  --camera_injection_mode projector
  --camera_plucker_encoder_mode patchify
  --trainable_models "${TRAINABLE_MODELS:-dit.domain_embedding,dit.control_patch_embedding}"
  --dit_finetune_modules "${DIT_FINETUNE_MODULES:-self_attn}"
  --train_sigma_shift 5.0
  --cond_drop_geo_prob "${COND_DROP_GEO_PROB:-0.0}"
  --cond_drop_track_prob "${COND_DROP_TRACK_PROB:-0.0}"
  --cond_drop_dense_prob "${COND_DROP_DENSE_PROB:-0.0}"
  --dataset_num_workers "${DATASET_NUM_WORKERS:-6}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR:-2}"
  --drop_text_prob "${DROP_TEXT_PROB:-0.1}"
  --allow_existing_checkpoints
)
if [[ "${USE_GRADIENT_CHECKPOINTING:-1}" == 1 ]]; then ARGS+=(--use_gradient_checkpointing); fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}"); fi
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}"); fi

cd "${ROOT}"
exec accelerate launch "${LAUNCH_ARGS[@]}" train/train_track4d360.py "${ARGS[@]}"
