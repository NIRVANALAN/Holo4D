#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/video_quality"
cd "${REPO_ROOT}"

MODE="${MODE:-flat}"
PRED_ROOT="${PRED_ROOT:-./results/cam_type1}"
PROMPT_FROM="${PROMPT_FROM:-source}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_JSON="${OUTPUT_JSON:-${PRED_ROOT%/}/video_quality_metrics.json}"
PER_SAMPLE_CSV="${PER_SAMPLE_CSV:-${PRED_ROOT%/}/video_quality_per_sample.csv}"
CLIP_DETAIL_DIR="${CLIP_DETAIL_DIR:-${PRED_ROOT%/}/clip_details}"
PRED_TRAJ="${PRED_TRAJ:-}"
GT_TRAJ="${GT_TRAJ:-}"
GT_TRAJ_FORMAT="${GT_TRAJ_FORMAT:-tum}"
TRAJ_SKIP="${TRAJ_SKIP:-0}"
TRAJ_END="${TRAJ_END:--1}"
TRAJ_STRIDE="${TRAJ_STRIDE:-1}"
POSE_REPORT="${POSE_REPORT:-${PRED_ROOT%/}/pose_eval_report.txt}"
INCLUDE_FVD_V="${INCLUDE_FVD_V:-0}"

COMMON_ARGS=(
  --mode "${MODE}"
  --pred_root "${PRED_ROOT}"
  --prompt_from "${PROMPT_FROM}"
  --device "${DEVICE}"
  --output_json "${OUTPUT_JSON}"
  --per_sample_csv "${PER_SAMPLE_CSV}"
  --clip_detail_dir "${CLIP_DETAIL_DIR}"
)

if [[ -n "${METRICS:-}" ]]; then
  RESOLVED_METRICS="${METRICS}"
else
  if [[ "${MODE}" == "flat" ]]; then
    RESOLVED_METRICS="clip_t,clip_f,clip_v"
  else
    RESOLVED_METRICS="fvd,clip_t,clip_f,clip_v"
    if [[ "${INCLUDE_FVD_V}" == "1" ]]; then
      RESOLVED_METRICS="${RESOLVED_METRICS},fvd_v"
    fi
  fi
  if [[ -n "${PRED_TRAJ}" && -n "${GT_TRAJ}" ]]; then
    RESOLVED_METRICS="${RESOLVED_METRICS},rot_err,trans_err"
  fi
fi
COMMON_ARGS+=(--metrics "${RESOLVED_METRICS}")

if [[ -n "${MAX_FRAMES:-}" ]]; then
  COMMON_ARGS+=(--max_frames "${MAX_FRAMES}")
fi

if [[ -n "${EVAL_HEIGHT:-}" && -n "${EVAL_WIDTH:-}" ]]; then
  COMMON_ARGS+=(--eval_height "${EVAL_HEIGHT}" --eval_width "${EVAL_WIDTH}")
fi

if [[ -n "${BATCH_SIZE:-}" ]]; then
  COMMON_ARGS+=(--batch_size "${BATCH_SIZE}")
fi

if [[ -n "${FID_BATCH_SIZE:-}" ]]; then
  COMMON_ARGS+=(--fid_batch_size "${FID_BATCH_SIZE}")
fi

if [[ -n "${FVD_METHOD:-}" ]]; then
  COMMON_ARGS+=(--fvd_method "${FVD_METHOD}")
fi

if [[ -n "${CLIP_MODEL_NAME:-}" ]]; then
  COMMON_ARGS+=(--clip_model_name "${CLIP_MODEL_NAME}")
fi

if [[ -n "${CLIP_PRETRAINED:-}" ]]; then
  COMMON_ARGS+=(--clip_pretrained "${CLIP_PRETRAINED}")
fi

if [[ -n "${CLIP_BPE_PATH:-}" ]]; then
  COMMON_ARGS+=(--clip_bpe_path "${CLIP_BPE_PATH}")
fi

if [[ -n "${COMMON_METRICS_ROOT:-}" ]]; then
  COMMON_ARGS+=(--common_metrics_root "${COMMON_METRICS_ROOT}")
fi

if [[ -n "${SYN4D_MODALITY:-}" ]]; then
  COMMON_ARGS+=(--syn4d_modality "${SYN4D_MODALITY}")
fi

if [[ -n "${OFFICIAL_MODALITY:-}" ]]; then
  COMMON_ARGS+=(--official_modality "${OFFICIAL_MODALITY}")
fi

if [[ -n "${OFFICIAL_POINTMAP_SOURCE_MODE:-}" ]]; then
  COMMON_ARGS+=(--official_pointmap_source_mode "${OFFICIAL_POINTMAP_SOURCE_MODE}")
fi

if [[ -n "${PRED_TRAJ}" && -n "${GT_TRAJ}" ]]; then
  COMMON_ARGS+=(
    --pred_traj "${PRED_TRAJ}"
    --gt_traj "${GT_TRAJ}"
    --gt_traj_format "${GT_TRAJ_FORMAT}"
    --traj_skip "${TRAJ_SKIP}"
    --traj_end "${TRAJ_END}"
    --traj_stride "${TRAJ_STRIDE}"
    --pose_report "${POSE_REPORT}"
  )
fi

case "${MODE}" in
  flat)
    DATASET_PATH="${DATASET_PATH:-./example_test_data}"
    METADATA_CSV="${METADATA_CSV:-${DATASET_PATH%/}/metadata.csv}"
    python tools/eval_video_quality.py \
      "${COMMON_ARGS[@]}" \
      --dataset_path "${DATASET_PATH}" \
      --metadata_csv "${METADATA_CSV}" \
      "$@"
    ;;
  official_nvs)
    if [[ -n "${SHARED_EVAL_ROOT:-}" ]]; then
      RESOLVED_SHARED_EVAL_ROOT="${SHARED_EVAL_ROOT}"
    else
      latest_shared="$(ls -td "${REPO_ROOT}"/eval/official_eval_shared/*/ 2>/dev/null | head -n 1 || true)"
      if [[ -z "${latest_shared}" ]]; then
        echo "ERROR: set SHARED_EVAL_ROOT for official_nvs mode."
        exit 1
      fi
      RESOLVED_SHARED_EVAL_ROOT="${latest_shared%/}"
    fi
    SOURCE_VIDEO_NAME="${SOURCE_VIDEO_NAME:-source_cam01.mp4}"
    python tools/eval_video_quality.py \
      "${COMMON_ARGS[@]}" \
      --shared_eval_root "${RESOLVED_SHARED_EVAL_ROOT}" \
      --source_video_name "${SOURCE_VIDEO_NAME}" \
      "$@"
    ;;
  syn4d)
    SYN4D_DATASET_ROOT="${SYN4D_DATASET_ROOT:-}"
    if [[ -z "${SYN4D_DATASET_ROOT}" ]]; then
      echo "ERROR: set SYN4D_DATASET_ROOT for syn4d mode."
      exit 1
    fi
    python tools/eval_video_quality.py \
      "${COMMON_ARGS[@]}" \
      --syn4d_dataset_root "${SYN4D_DATASET_ROOT}" \
      "$@"
    ;;
  kubric)
    KUBRIC_DATASET_ROOT="${KUBRIC_DATASET_ROOT:-}"
    if [[ -z "${KUBRIC_DATASET_ROOT}" ]]; then
      echo "ERROR: set KUBRIC_DATASET_ROOT for kubric mode."
      exit 1
    fi
    KUBRIC_SPLIT="${KUBRIC_SPLIT:-train}"
    KUBRIC_SEQ_DATA_FILE="${KUBRIC_SEQ_DATA_FILE:-}"
    EXTRA_ARGS=(
      --kubric_dataset_root "${KUBRIC_DATASET_ROOT}"
      --kubric_split "${KUBRIC_SPLIT}"
    )
    if [[ -n "${KUBRIC_SEQ_DATA_FILE}" ]]; then
      EXTRA_ARGS+=(--kubric_seq_data_file "${KUBRIC_SEQ_DATA_FILE}")
    fi
    python tools/eval_video_quality.py \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      "$@"
    ;;
  *)
    echo "ERROR: unsupported MODE=${MODE}"
    exit 1
    ;;
esac
