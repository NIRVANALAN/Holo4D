#!/bin/bash
# ===========================================================================
# Track4D-360 14B (H100 80GB) — iPhone (Shape-of-Motion) NVS benchmark.
#
# Same flag set as run_iphone_nvs_1p3b.sh; only the checkpoint + the 14B
# camera-control base camera-control base initialization weights differ. No FSDP — 14B fits on one H100.
# Set USE_FSDP=1 + GPUS=0,1 for 2-GPU FSDP if you want to test on smaller GPUs.
#
# Usage:
#   GPU=0 bash eval/iphone/run_iphone_nvs_14b.sh
#   USE_FSDP=1 GPUS=0,1 bash ...                        # 2-GPU FSDP
#   GPU=0 CKPT_STEP=8800 bash ...                        # custom ckpt
# ===========================================================================
set -euo pipefail

HOLO4D_ROOT="${HOLO4D_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${HOLO4D_ROOT}"

if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-holo4d}"
fi
if ! type module >/dev/null 2>&1; then
  for lmod_init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    [ -f "${lmod_init}" ] && source "${lmod_init}" && break
  done
fi
type module >/dev/null 2>&1 && module load all/CUDA/12.4.0 2>/dev/null || true

export PYTHONPATH="${HOLO4D_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOME}/.cache/triton}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HOLO4D_ALLOW_ONLINE_T5_ENCODE="${HOLO4D_ALLOW_ONLINE_T5_ENCODE:-1}"

LOGS_ROOT="outputs"
BR_IPHONE="${BR_IPHONE:-${LOGS_ROOT}/benchmark/iphone_nvs}"
CONDITIONS_ROOT="${CONDITIONS_ROOT:-${BR_IPHONE}}"

# ------------------- Checkpoint (14B FSDP-trained, see DAVIS launcher) -------------------
CKPT_STEP="${CKPT_STEP:-2000}"
CKPT_DIR="${LOGS_ROOT}/ismb_hybrid_dense_plucker_mixed_real_synth_syn4d_recam_syncam_savefix_cond_dropout_144x256x49_14b_16gpu_fsdp/train/Wan2.1-Fun-14B_track4d360_hybrid_dense_plucker_mixed_real_synth_syn4d_recam_syncam_savefix_cond_dropout"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/step-${CKPT_STEP}.safetensors}"
[ -f "${CKPT_PATH}" ] || { echo "ERROR: 14B ckpt not found: ${CKPT_PATH}" >&2; exit 1; }
CKPT_STEM="$(basename "${CKPT_PATH}" .safetensors)"

# ------------------- Pipeline weights (14B camera-control base variant) -------------------
TRACK_CONFIG="${HOLO4D_ROOT}/config/holo4d/default_conv3d_patchify_srcdepth.yaml"
CAMERA_CONTROL_CHECKPOINT_14B="${HOLO4D_ROOT}/weights/wan21-14b-controlCam/diffusion_pytorch_model.safetensors"
T5_PATH="${HOLO4D_ROOT}/weights/wan21-1p3b-run/models_t5_umt5-xxl-enc-bf16.safetensors"
VAE_PATH="${HOLO4D_ROOT}/weights/wan21-1p3b-run/Wan2.1_VAE.safetensors"
TOKENIZER_PATH="${HOLO4D_ROOT}/weights/wan21-1p3b-run/google/umt5-xxl"
MODEL_ID="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:${CAMERA_CONTROL_CHECKPOINT_14B},PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:${T5_PATH},PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:${VAE_PATH}"
[ -f "${CAMERA_CONTROL_CHECKPOINT_14B}" ] || { echo "ERROR: 14B init not found: ${CAMERA_CONTROL_CHECKPOINT_14B}" >&2; exit 1; }

# ------------------- Inference knobs (frozen — see CLAUDE.md) -------------------
HEIGHT="${HEIGHT:-288}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-49}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-1.0}"
SIGMA_SHIFT="${SIGMA_SHIFT:-5.0}"
SEED="${SEED:-42}"
DENSE_WARP_MODE="${DENSE_WARP_MODE:-hybrid_softsplat}"   # 14B was trained with softsplat (see ismb_sbatch_14b_..._softsplat_...sh)
MOTION_THRESHOLD="${MOTION_THRESHOLD:-0.005}"

GPU="${GPU:-0}"
SEQUENCES="${SEQUENCES:-apple,block,paper-windmill,spin,teddy}"
USE_4RC="${USE_4RC:-0}"
IPHONE_4RC_ROOT="${IPHONE_4RC_ROOT:-${BR_IPHONE}/4RC}"
# Default: center-crop everywhere (Vista4D protocol). Overrides:
#   RESIZE_MODE=1     → aspect-distort full scene, no GT crop.
#   PORTRAIT_CANON=1  → eval at SoM 2x (legacy convention).
RESIZE_MODE="${RESIZE_MODE:-0}"
PORTRAIT_CANON="${PORTRAIT_CANON:-0}"
if [ "${RESIZE_MODE}" = "1" ]; then
  SOURCE_PREPROC="resize"; CANON_MODE="landscape_model"; CROP_GT_FLAG=""
elif [ "${PORTRAIT_CANON}" = "1" ]; then
  SOURCE_PREPROC="center_crop"; CANON_MODE="portrait_2x"; CROP_GT_FLAG=""
else
  SOURCE_PREPROC="center_crop"; CANON_MODE="landscape_model"; CROP_GT_FLAG="--crop_gt_to_landscape"
fi
SUFFIX_4RC=""; [ "${USE_4RC}" = "1" ] && SUFFIX_4RC="_4rc"
SUFFIX_RESIZE=""; [ "${RESIZE_MODE}" = "1" ] && SUFFIX_RESIZE="_resize"
SUFFIX_CANON=""; [ "${PORTRAIT_CANON}" = "1" ] && SUFFIX_CANON="_portcanon"
RUN_TAG="${RUN_TAG:-ours_14b_${CKPT_STEM}_${HEIGHT}x${WIDTH}_${DENSE_WARP_MODE}${SUFFIX_4RC}${SUFFIX_RESIZE}${SUFFIX_CANON}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BR_IPHONE}/predictions/${RUN_TAG}}"
EXTRA_EVAL_ARGS=(--canon_mode "${CANON_MODE}")
[ -n "${CROP_GT_FLAG}" ] && EXTRA_EVAL_ARGS+=("${CROP_GT_FLAG}")
mkdir -p "${OUTPUT_ROOT}"
cp "$0" "${OUTPUT_ROOT}/" || true

# ------------------- FSDP plumbing (optional) -------------------
USE_FSDP="${USE_FSDP:-0}"
GPUS="${GPUS:-${GPU}}"
NUM_GPUS=$(echo "${GPUS}" | awk -F',' '{print NF}')
LAUNCHER=("python")
EXTRA_PY_ARGS=()
if [ "${USE_FSDP}" = "1" ]; then
  LAUNCHER=("accelerate" "launch"
            "--num_processes" "${NUM_GPUS}"
            "--num_machines" "1"
            "--mixed_precision" "bf16"
            "--use_fsdp")
  EXTRA_PY_ARGS+=(--use_fsdp)
fi
if [ "${USE_4RC}" = "1" ]; then
  # Note: USE_4RC=1 needs ≥24 GB VRAM. A4000 (16 GB) OOMs at track sin-encoding.
  for seq in $(echo "${SEQUENCES}" | tr ',' ' '); do
    [ -f "${CONDITIONS_ROOT}/conditions/${seq}/depth_scale_alpha.json" ] || {
      echo "ERROR: USE_4RC=1 but ${CONDITIONS_ROOT}/conditions/${seq}/depth_scale_alpha.json missing."
      echo "  Run align_4rc_to_iphone_lidar.py first."
      exit 1
    }
    [ -f "${IPHONE_4RC_ROOT}/${seq}/result.npz" ] || {
      echo "ERROR: USE_4RC=1 but ${IPHONE_4RC_ROOT}/${seq}/result.npz missing."
      exit 1
    }
  done
  EXTRA_PY_ARGS+=(--use_4rc_source --iphone_4rc_root "${IPHONE_4RC_ROOT}")
fi
EXTRA_PY_ARGS+=(--source_preproc "${SOURCE_PREPROC}")

cat <<EOF
==========================================================================
Track4D-360 14B — iPhone NVS benchmark @ ${HEIGHT}x${WIDTH}x${NUM_FRAMES}
  GPUs:               ${GPUS} (USE_FSDP=${USE_FSDP})
  Sequences:          ${SEQUENCES}
  Checkpoint:         ${CKPT_PATH}
  14B init:           ${CAMERA_CONTROL_CHECKPOINT_14B}
  Conditions root:    ${CONDITIONS_ROOT}
  Output root:        ${OUTPUT_ROOT}
  Dense warp mode:    ${DENSE_WARP_MODE}
  Sampling:           steps=${NUM_INFERENCE_STEPS} cfg=${CFG_SCALE} sigma_shift=${SIGMA_SHIFT} seed=${SEED}
==========================================================================
EOF

LOG="${OUTPUT_ROOT}/eval.${HOSTNAME:-host}.log"
: > "${LOG}"

for seq in $(echo "${SEQUENCES}" | tr ',' ' '); do
  meta="${CONDITIONS_ROOT}/conditions/${seq}/metadata.json"
  if [ ! -f "${meta}" ]; then
    echo "ERROR: missing conditions for ${seq}: ${meta}"
    echo "  Build them first with build_iphone_pairs.py — see README.md"
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES="${GPUS}" "${LAUNCHER[@]}" \
  eval/iphone/predict_iphone_nvs.py \
  --model_id_with_origin_paths "${MODEL_ID}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --checkpoint_path "${CKPT_PATH}" \
  --camera_control_checkpoint "${CAMERA_CONTROL_CHECKPOINT_14B}" \
  --track_config "${TRACK_CONFIG}" \
  --use_plucker_camera_control \
  --enable_v2v_plucker_camera_control \
  --use_query_frame_impulse_condition \
  --use_dense_branch \
  --dense_proj_dim 32 \
  --dense_num_residual_blocks 2 \
  --dense_alpha_track 1.0 \
  --track_spatial_stride 1 \
  --track_injection_mode per_block \
  --track_injection_block_mode concat_project \
  --warped_condition_mode off \
  --dense_in_channels 5 \
  --dense_warp_mode "${DENSE_WARP_MODE}" \
  --dense_warp_motion_threshold "${MOTION_THRESHOLD}" \
  --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" --sigma_shift "${SIGMA_SHIFT}" --seed "${SEED}" \
  --iphone_conditions_root "${CONDITIONS_ROOT}" \
  --sequences "${SEQUENCES}" \
  --output_dir "${OUTPUT_ROOT}" \
  "${EXTRA_PY_ARGS[@]}" \
  2>&1 | tee -a "${LOG}"

if [ "${SKIP_METRICS:-0}" = "1" ]; then
  echo "[metrics] SKIP_METRICS=1 — skipping eval_iphone_nvs.py"
else
  echo ""
  echo "=========================================================================="
  echo "Computing iPhone NVS metrics for ${RUN_TAG}..."
  echo "=========================================================================="
  CUDA_VISIBLE_DEVICES="${GPU}" python \
    eval/iphone/eval_iphone_nvs.py \
    --output_root "${BR_IPHONE}" \
    --predictions_dir "${OUTPUT_ROOT}" \
    --run_tag "${RUN_TAG}" \
    --sequences "${SEQUENCES}" \
    --num_frames "${NUM_FRAMES}" \
    "${EXTRA_EVAL_ARGS[@]}" \
    2>&1 | tee -a "${LOG}"
fi

echo ""
echo "Done -> ${OUTPUT_ROOT}"
echo "  Predictions:   ${OUTPUT_ROOT}/<seq>/pred.mp4"
echo "  Metrics JSON:  ${BR_IPHONE}/eval_results/${RUN_TAG}.json"
echo "  Metrics MD:    ${BR_IPHONE}/eval_results/${RUN_TAG}.md"
