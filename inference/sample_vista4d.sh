#!/usr/bin/env bash
# Canonical Holo4D 384p inference on the Vista4D evaluation set.
set -euo pipefail

ROOT="${HOLO4D_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAN_ROOT="${WAN_ROOT:-$("${PYTHON_BIN}" -m holo4d.hub wan t2v)}"
CHECKPOINT="${CHECKPOINT:-$("${PYTHON_BIN}" -m holo4d.hub checkpoint t2v)}"
if [[ -z "${VISTA4D_DATA_ROOT:-}" ]] && {
  [[ -z "${VISTA4D_EVAL_ROOT:-}" || -z "${FRAME_ROOT:-}" ||
     -z "${COW_ROOT:-}" || -z "${VGGT_ROOT:-}" ]];
}; then
  VISTA4D_DATA_ROOT="$("${PYTHON_BIN}" -m holo4d.hub vista4d)"
fi
VISTA4D_EVAL_ROOT="${VISTA4D_EVAL_ROOT:-${VISTA4D_DATA_ROOT}/vista4d_eval}"
FRAME_ROOT="${FRAME_ROOT:-${VISTA4D_DATA_ROOT}/frames}"
COW_ROOT="${COW_ROOT:-${VISTA4D_DATA_ROOT}/cow}"
VGGT_ROOT="${VGGT_ROOT:-${VISTA4D_DATA_ROOT}/vggt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/vista4d_384p}"
FOURRC_ROOT="${FOURRC_ROOT:-${ROOT}/outputs/unused_4rc_root}"

check_input() {
  local value="$1"
  local label="$2"
  if [[ -e "${value}" ]] || compgen -G "${value}" >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: ${label} is missing or matches no files: ${value}" >&2
  exit 1
}
for item in \
  "${WAN_ROOT}|Wan base" \
  "${CHECKPOINT}|Holo4D checkpoint" \
  "${VISTA4D_EVAL_ROOT}/metadata.csv|Vista4D metadata" \
  "${FRAME_ROOT}|source frames" \
  "${COW_ROOT}|CoWTracker cache" \
  "${VGGT_ROOT}|VGGT-Omega cache"; do
  check_input "${item%%|*}" "${item#*|}"
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export DISABLE_FLASH_ATTN_3="${DISABLE_FLASH_ATTN_3:-1}"
export HOLO4D_VISTA4D_TGT_T_DEPTH_FRAC="${VISTA4D_TARGET_TRANSLATION_DEPTH_FRAC:-0.6}"
# Native camera and FSFF half-viewpoint are different protocols, not aliases.
case "${TRAJECTORY_MODE:-native}" in
  native) ANCHOR=vista4d_frame0; DAMPING=1.0 ;;
  shared-first-frame-half) ANCHOR=source_frame0; DAMPING=0.5 ;;
  *) echo "Unknown TRAJECTORY_MODE; use native or shared-first-frame-half" >&2; exit 2 ;;
esac

MODEL_ID="Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/Wan2.1_VAE.pth"
if [[ -n "${EVAL_T5_CACHE:-}" ]]; then
  check_input "${EVAL_T5_CACHE}" "eval T5 prompt cache"
  export HOLO4D_EVAL_T5_CACHE="${EVAL_T5_CACHE}"
  export HOLO4D_REQUIRE_T5_CACHE=1
  export HOLO4D_ALLOW_ONLINE_T5_ENCODE=0
else
  check_input "${WAN_ROOT}/models_t5_umt5-xxl-enc-bf16.pth" "T5 encoder"
  MODEL_ID+=",Wan-AI/Wan2.1-T2V-14B:${WAN_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"
  export HOLO4D_ALLOW_ONLINE_T5_ENCODE=1
fi

ARGS=(
  --model_id_with_origin_paths "${MODEL_ID}"
  --tokenizer_path "${WAN_ROOT}/google/umt5-xxl"
  --checkpoint_path "${CHECKPOINT}"
  --camera_control_checkpoint "${WAN_ROOT}/diffusion_pytorch_model*.safetensors"
  --track_config "${TRACK_CONFIG:-${ROOT}/config/holo4d/default_conv3d_patchify_srcdepth_srcrgb.yaml}"
  --track_injection_mode per_block
  --track_injection_block_mode concat_project
  --track_spatial_stride 1
  --use_query_frame_impulse_condition
  --use_plucker_camera_control
  --enable_v2v_plucker_camera_control
  --use_perlayer_camera_control
  --camera_injection_mode projector
  --camera_plucker_encoder_mode patchify
  --use_plucker_raymap_tokens
  --no_dense_branch
  --height "${HEIGHT:-384}"
  --width "${WIDTH:-672}"
  --num_frames 49
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}"
  --cfg_scale "${CFG_SCALE:-5.0}"
  --sigma_shift "${SIGMA_SHIFT:-5.0}"
  --seed "${SEED:-42}"
  --davis_root "${FRAME_ROOT}"
  --davis_4rc_root "${FOURRC_ROOT}"
  --vista4d_eval_root "${VISTA4D_EVAL_ROOT}"
  --source_filter all
  --max_cameras_per_video "${MAX_CAMERAS_PER_VIDEO:-2}"
  --trajectory_anchor_mode "${ANCHOR}"
  --trajectory_damping "${TRAJECTORY_DAMPING:-${DAMPING}}"
  --vista4d_target_scale_align depth
  --source_backend cow_vggt
  --cow_root "${COW_ROOT}"
  --vggt_root "${VGGT_ROOT}"
  --cow_track_sample_mode all
  --cow_track_stride 1
  --cow_conf_threshold "${COW_CONF_THRESHOLD:-0.5}"
  --cow_depth_conf_threshold "${COW_DEPTH_CONF_THRESHOLD:-0.0}"
  --cow_vggt_scale_align none
  --output_dir "${OUTPUT_DIR}"
)
if [[ -n "${VIDEO_NAMES:-}" ]]; then ARGS+=(--video_names "${VIDEO_NAMES}"); fi
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then ARGS+=(--force_rerun); fi
if [[ "${CAPTURE_FINAL_LATENTS:-0}" == "1" ]]; then ARGS+=(--capture_final_latents); fi

cd "${ROOT}"
echo "[Holo4D/Vista4D] checkpoint=${CHECKPOINT}"
echo "[Holo4D/Vista4D] videos=${VIDEO_NAMES:-all} output=${OUTPUT_DIR}"
echo "[Holo4D/Vista4D] protocol=${TRAJECTORY_MODE:-native} anchor=${ANCHOR} damping=${TRAJECTORY_DAMPING:-${DAMPING}}"
python -u examples/wan2.1_fun/eval_track4d360_vista4d_official.py "${ARGS[@]}"
python -m holo4d.tools.validate_vista4d_outputs \
  --metadata "${VISTA4D_EVAL_ROOT}/metadata.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --video-names "${VIDEO_NAMES:-}" \
  --max-cameras "${MAX_CAMERAS_PER_VIDEO:-2}" \
  --expected-frames 49
