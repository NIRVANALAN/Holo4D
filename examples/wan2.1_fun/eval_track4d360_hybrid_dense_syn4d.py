"""Eval script for the hybrid dense + persistent Track4D-360 branch.

This script is the inference-time counterpart of
``bash_scripts/holo4d/train_hybrid_dense_flying_group_seq0_2gpu.sh``.

It loads a checkpoint trained with ``--use_dense_branch`` and runs
inference with BOTH:

  1. the persistent frame-0-query conv3d track branch (existing), and
  2. the GEN3C-style per-frame dense target-space branch (new).

The two branches are fused additively in token space inside the pipeline:

    target_tokens = dense_ctrl_tokens + alpha_track * persistent_track_tokens

All CLI flags default to the values used by the 2-GPU hybrid dense training
launcher so you can run the eval with nothing more than ``--checkpoint_path``
and ``--output_dir``.

Usage example::

    python examples/wan2.1_fun/eval_track4d360_hybrid_dense_syn4d.py \
        --checkpoint_path /path/to/step-XXX.safetensors \
        --output_dir /path/to/eval_out

Or via the shell wrapper::

    bash bash_scripts/holo4d/eval_hybrid_dense_flying_group_seq0.sh \
        /path/to/step-XXX.safetensors
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Silence the "huggingface/tokenizers: The current process just got forked..."
# warning from DataLoader workers. Must be set BEFORE any tokenizers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Configure FA3 before importing torch/diffusers/xformers.  On H200 this keeps
# Holo4D's standalone FA3 kernel while hiding xFormers' duplicate extension.
_PRE_IMPORT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_PRE_IMPORT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRE_IMPORT_REPO_ROOT))
from holo4d.fa3_compat import configure_flash_attention_imports

configure_flash_attention_imports()

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm as _tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holo4d.config import load_track_config
from holo4d.conditions.i2v_target_condition import plan_long_horizon_chunks
from holo4d.eval_fsdp_helper import is_main_process
from holo4d.novel_view.object_centric_orbit import (
    build_fixed_height_look_at_orbit_sequence,
    build_object_centric_orbit_sequence,
    estimate_moving_track_pivots_world,
)
from holo4d.benchmark.syn4d_backend_condition import (
    apply_relative_camera_control,
    canonical_path,
    inject_canonical_condition,
    load_canonical_condition,
    provenance as source_condition_provenance,
)
from holo4d.visualization.artifact_io import (
    PREDICTION_MP4_QUALITY,
    WAN_SAMPLE_FPS,
    save_mp4,
    save_pred_latents,
)
from holo4d_runtime.core import ModelConfig, load_state_dict
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.syn4d_nvs import Syn4D_Multi_NVS
from holo4d_runtime.diffusion.training_module import DiffusionTrainingModule
from holo4d_runtime.pipeline.wan_video_track4d360 import WanVideoTrack4D360Pipeline
from holo4d_runtime.pipelines.wan_video_v2v import _ensure_domain_embedding


# Canonical VXF V2V negative prompt — must be passed to pipe(...) at inference
# time. Mirrors examples/wan2.1_fun/predict_v2v_control_camera.py:118 and
# examples/wan2.1_fun/predict_v2v_control_camera_dl3dv.py:34. Without this,
# CFG's negative branch runs with the null-text embedding and the guidance
# direction `(posi - nega)` fails to push the sample away from the generic
# Wan T2V "dreamy watercolor" prior, producing strong artifacts at cfg>=5.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval hybrid dense + persistent Track4D-360 NVS on a Syn4D sample."
    )

    # Model / checkpoint
    parser.add_argument("--model_paths", type=str, default=None,
                        help="JSON list of local model files/dirs, matching train_track4d360.py.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Comma-separated model_id:origin_path entries, matching train_track4d360.py.")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Trained Track4D-360 hybrid dense checkpoint (.safetensors).")
    parser.add_argument(
        "--i2v_seed_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Checkpoint for the source-video-only first T2V chunk. Used only "
            "with --i2v_continuation_pairs; defaults to --checkpoint_path."
        ),
    )
    parser.add_argument(
        "--i2v_continuation_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Checkpoint for the generated-target continuation chunk. Used only "
            "with --i2v_continuation_pairs; defaults to --checkpoint_path."
        ),
    )
    parser.add_argument(
        "--i2v_seed_model_id_with_origin_paths",
        type=str,
        default=None,
        help=(
            "T2V model bundle for the source-only first chunk. Used only with "
            "mixed I2V continuation; required when seed and continuation "
            "checkpoint paths differ."
        ),
    )
    parser.add_argument(
        "--i2v_continuation_model_id_with_origin_paths",
        type=str,
        default=None,
        help=(
            "I2V model bundle for the generated-boundary continuation chunk. "
            "Used only with mixed I2V continuation."
        ),
    )
    parser.add_argument(
        "--i2v_seed_camera_control_checkpoint",
        type=str,
        default=None,
        help="T2V VXF init weights for the source-only seed stage.",
    )
    parser.add_argument(
        "--i2v_continuation_camera_control_checkpoint",
        type=str,
        default=None,
        help="I2V VXF init weights for the generated-boundary continuation stage.",
    )
    parser.add_argument(
        "--i2v_resume_stage", choices=("both", "continuation"), default="both",
        help="Resume an existing seed without generating it again.",
    )
    parser.add_argument("--gs_checkpoint_path", type=str, default=None,
                        help="Optional GS-only overlay checkpoint. Load the base hybrid-dense+plucker "
                             "checkpoint via --checkpoint_path, then overlay gs_feature_head/gs_decoder "
                             "weights from this checkpoint.")
    parser.add_argument("--lora_base_model", type=str, default=None,
                        help="Comma-separated base model(s) to LoRA-wrap at eval, e.g. 'dit'.")
    parser.add_argument("--lora_target_modules", type=str, default="",
                        help="Comma-separated LoRA target module patterns. Must match training.")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank. Must match training when --lora_base_model is used.")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="Optional LoRA alpha. Defaults to rank when omitted.")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="LoRA checkpoint to load via PEFT mapping. "
                             "Use this instead of treating a LoRA checkpoint as a full DiT checkpoint.")
    parser.add_argument("--camera_control_checkpoint", type=str, default=None,
                        help="VideoX-Fun base 1.3B DiT (e.g. weights/wan21-1p3b/diffusion_pytorch_model.safetensors). "
                             "CRITICAL: the trained checkpoint only contains fine-tuned heads "
                             "(self_attn/ffn/domain_embedding/control_patch_embedding + track modules). "
                             "The rest of the DiT backbone MUST come from the VXF-native 1.3B weights, "
                             "NOT the DiffSynth-Studio 1.3B (which is a different model). Training always "
                             "passes --camera_control_checkpoint, and inference must do the same.")
    # VRAM knobs (forwarded to DiffusionTrainingModule.parse_model_configs).
    # Mainly useful for 14B inference on sub-40GB GPUs.
    parser.add_argument("--fp8_models", type=str, default=None,
                        help="Comma-separated model_id_with_origin_path entries to quantize "
                             "to fp8_e4m3fn storage with bf16 compute (~2x DiT memory savings).")
    parser.add_argument("--offload_models", type=str, default=None,
                        help="Comma-separated model_id_with_origin_path entries to keep on disk/CPU "
                             "and stream layer-by-layer (slow, but unblocks small-VRAM 14B inference).")

    from holo4d.shared_args import (
        add_track_config_args,
        add_dense_branch_args,
        add_cond_null_token_fallback_args,
        add_query_impulse_args,
        add_plucker_args,
        add_perlayer_camera_args,
        add_warped_condition_args,
        add_gs_output_args,
        add_gs_dynamic_args,
        add_ar_args,
        add_future_prediction_args,
        add_i2v_condition_args,
    )
    add_track_config_args(parser)
    add_dense_branch_args(parser)
    add_cond_null_token_fallback_args(parser)
    parser.add_argument(
        "--dense_only_condition_modules",
        action="store_true",
        default=False,
        help="Use dense-target control alignment without persistent Track4D track adapters.",
    )
    add_query_impulse_args(parser)
    add_plucker_args(parser)
    add_perlayer_camera_args(parser)
    add_warped_condition_args(parser)
    add_gs_output_args(parser)
    add_gs_dynamic_args(parser)
    add_ar_args(parser)
    add_future_prediction_args(parser)
    add_i2v_condition_args(parser)

    # Dataset — Syn4D direct mode (NOT YAML), matching training
    parser.add_argument("--syn4d_root", type=str,
                        default=os.environ.get("SYN4D_ROOT", ""))
    parser.add_argument(
        "--syn4d_caption_dir", type=str,
        default=os.environ.get("SYN4D_CAPTION_DIR"),
        help="Optional Syn4D caption/T5-cache directory for direct and continuation eval.",
    )
    parser.add_argument(
        "--source_condition_root", type=str, default=None,
        help="Optional canonical source-condition root.",
    )
    parser.add_argument(
        "--source_condition_backend", type=str, default=None,
        help="Canonical backend label used with --source_condition_root.",
    )
    parser.add_argument(
        "--camera_control_component",
        choices=("native_mixed", "rotation_only", "translation_only"),
        default="native_mixed",
        help=(
            "Optionally retain only source-to-target relative rotation or "
            "translation. This is an attribution control, not a headline NVS protocol."
        ),
    )
    parser.add_argument(
        "--track_xyz_noise_std_multiplier",
        type=float,
        default=0.0,
        help=(
            "Per-sample GT-track Gaussian-noise multiplier. Compute the "
            "population std of valid tracks independently for each XYZ axis."
        ),
    )
    parser.add_argument(
        "--track_xyz_noise_seed",
        type=int,
        default=20260725,
        help=(
            "Base seed for GT-track noise. The draw is keyed by scene and "
            "source sequence so target views and multipliers share it."
        ),
    )
    parser.add_argument(
        "--gt_track_orbit_deg",
        type=float,
        default=0.0,
        help=(
            "Replace the recorded target camera with a uniform object-centric "
            "orbit of this many degrees. The pivot is estimated only from "
            "moving Syn4D GT world-track identities. Zero disables the orbit."
        ),
    )
    parser.add_argument(
        "--gt_track_orbit_direction",
        choices=("left", "right"),
        default="left",
        help="Direction for --gt_track_orbit_deg.",
    )
    parser.add_argument(
        "--gt_track_orbit_mode",
        choices=("dynamic_rigid", "fixed_height_look_at"),
        default="dynamic_rigid",
        help=(
            "dynamic_rigid preserves the historical per-frame source/pivot "
            "composition. fixed_height_look_at uses one temporal-median pivot, "
            "the source frame-0 center, explicit Syn4D world-up, and a fresh "
            "look-at rotation at every frame."
        ),
    )
    parser.add_argument(
        "--gt_track_orbit_radius_scale",
        type=float,
        default=1.0,
        help=(
            "Scale the source frame-0 horizontal radius in fixed_height_look_at "
            "mode. Values above one move the arc away from near-field geometry."
        ),
    )
    parser.add_argument(
        "--gt_track_orbit_motion_threshold",
        type=float,
        default=0.01,
        help="Minimum maximum world-space displacement for a GT identity to define the orbit pivot.",
    )
    parser.add_argument(
        "--gt_track_orbit_min_identities",
        type=int,
        default=64,
        help="Fail rather than fall back unless at least this many moving GT identities define every pivot.",
    )
    parser.add_argument("--scenes", type=str, default="flying_group,train_group",
                        help="Comma-separated list of Syn4D scenes to evaluate.")
    parser.add_argument("--sequence_idxs", type=str, default="0",
                        help="Comma-separated list of sequence indices within each scene.")
    parser.add_argument("--camera_pairs", type=str, default="0-1,0-2,0-3,0-4,0-5,0-6,0-7",
                        help="Comma-separated (src_cam_idx)-(tgt_cam_idx) pairs. "
                             "Indices are positions in group['camera_names'], NOT "
                             "camera suffixes. For a group with 8 cameras the indices "
                             "0..7 map 1:1 to suffixes 0..7 in Syn4D.")
    parser.add_argument(
        "--camera_pairs_by_sequence", type=str, default="",
        help="Optional semicolon-separated sequence overrides, for example "
             "4:0-1,0-2,0-3,0-4. Sequences not listed use --camera_pairs.",
    )
    parser.add_argument("--start_frame", type=str, default="first",
                        help="'first' picks group['common_starts'][0]; integer picks "
                             "a specific absolute start frame.")
    parser.add_argument(
        "--use_mp4_rgb", action="store_true",
        help="Read Syn4D RGB from the scene mp4 directory. Required by compact Syn4D copies.",
    )
    parser.add_argument("--force_rerun", action="store_true",
                        help="Re-run pairs whose output mp4 already exists.")
    parser.add_argument(
        "--i2v_continuation_pairs", type=int, default=0,
        help="Run this many deterministic two-window I2V continuation cases instead "
             "of the ordinary direct Syn4D pair loop. The scored window starts at "
             "the prior generated target's final frame.",
    )
    parser.add_argument(
        "--i2v_continuation_seed", type=int, default=20260717,
        help="Deterministic candidate-selection seed for --i2v_continuation_pairs.",
    )
    parser.add_argument(
        "--i2v_continuation_shard_index", type=int, default=0,
        help="Zero-based shard index for the deterministic I2V continuation cases.",
    )
    parser.add_argument(
        "--i2v_continuation_num_shards", type=int, default=1,
        help="Number of deterministic I2V continuation shards.",
    )
    parser.add_argument(
        "--i2v_continuation_gt_boundary",
        action="store_true",
        default=False,
        help=(
            "DIAGNOSTIC: condition the continuation window on the GROUND-TRUTH "
            "previous target frame instead of the seed window's generated one. "
            "Training teacher-forces a GT boundary, so this isolates exposure "
            "bias (train/inference boundary distribution gap) from an actual "
            "mechanism bug: if the latent-frame-1 artifact disappears under "
            "this flag, the plumbing is correct and the model simply never "
            "learned to reconcile a drifted boundary. See "
            "doc/track4d-360/bugs/2026-08-23-ti2v-native-i2v-latent-frame1-artifact.md. "
            "Not a valid production protocol -- it leaks GT into the rollout."
        ),
    )
    parser.add_argument(
        "--i2v_fixed_source_sequences", type=str, default=None,
        help=(
            "Optional comma-separated flying_group base sequences for the exact "
            "source-camera-0 protocol (e.g. seq_000000,...,seq_000004). "
            "Uses seed start 5 and targets 1..7, except seq_000004 targets 1..4."
        ),
    )
    parser.add_argument(
        "--i2v_fixed_seed_start_frame", type=int, default=5,
        help="Absolute seed-window start used with --i2v_fixed_source_sequences.",
    )
    parser.add_argument(
        "--i2v_long_horizon_frames", type=int, default=0,
        help=(
            "Generate a resumable multi-chunk rollout. The first 49-frame "
            "chunk uses the source-only T2V seed checkpoint; every later "
            "chunk uses the generated previous target boundary with the I2V "
            "continuation checkpoint. Zero keeps the two-window protocol."
        ),
    )
    parser.add_argument(
        "--i2v_long_horizon_seed_start_frame", type=int, default=0,
        help="Absolute source frame for the first long-horizon chunk.",
    )
    parser.add_argument(
        "--i2v_long_horizon_sequence_index", type=int, default=0,
        help="Syn4D flying_group sequence index for the long-horizon case.",
    )
    parser.add_argument(
        "--i2v_long_horizon_case_index", type=int, default=0,
        help=(
            "Global output case index for collision-free one-case array jobs. "
            "It does not alter sequence, camera, or sampling semantics."
        ),
    )
    parser.add_argument(
        "--i2v_long_horizon_src_cam_idx", type=int, default=5,
        help="Source camera index for the long-horizon case.",
    )
    parser.add_argument(
        "--i2v_long_horizon_tgt_cam_idx", type=int, default=6,
        help="Target camera index for the long-horizon case.",
    )
    parser.add_argument(
        "--i2v_long_horizon_source_condition_root", type=str, default=None,
        help=(
            "Optional canonical source-condition root for long-horizon chunks. "
            "Files are selected as <base_seq>_chunk_<index:03d>."
        ),
    )
    parser.add_argument(
        "--i2v_long_horizon_source_condition_backend", type=str, default=None,
        help="Backend label stored in the long-horizon canonical condition metadata.",
    )
    parser.add_argument(
        "--syn4d_valid_sequences_file", type=str,
        default="config/syn4d_valid_sequences.txt",
        help="Valid-sequence list used for deterministic held-out Syn4D continuation eval.",
    )

    # Geometry — match training
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--track_spatial_stride", type=int, default=1)

    # Sampling defaults for backward-compatible direct Track4D-360 V2V NVS.
    #   num_inference_steps=50: matches canonical VXF V2V inference.
    #   cfg_scale=1.0: preserves the historical generic evaluator default. It
    #                   still uses the positive text embedding, but skips the
    #                   negative/unconditional guidance branch. Protocol
    #                   launchers that require classifier-free guidance must
    #                   override this explicitly (held-out I2V uses 2.5).
    #   sigma_shift=5.0: matches training (training_module.py:394 calls
    #                    set_timesteps(1000, training=True) which defaults to
    #                    shift=5 for the Wan template). shift=3 is null at
    #                    144×256 (confirmed same sweep) and is an inference
    #                    convention for 480p+; don't transfer it below 480p.
    #   seed=42: deterministic reference seed.
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera_condition_scale", type=float, default=1.0)
    parser.add_argument("--debug_camera_control", action="store_true", default=False)
    parser.add_argument(
        "--strict_vista4d_plucker_nonzero",
        action="store_true",
        default=False,
        help="Fail if vista4d_plucker per-block camera encoders remain zero after checkpoint load.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--capture_final_latents", action="store_true",
                        help="Save the final target latents from the rollout as final_target_latents.pt.")
    parser.add_argument("--no_vae_tiling", dest="vae_tiling", action="store_false",
                        help="Disable VAE tiled encode/decode during eval.")
    parser.set_defaults(vae_tiling=True)

    # FSDP inference (multi-GPU sharding for big models like 14B on
    # sub-80GB GPUs). When ON, the eval entry must be launched via
    # `accelerate launch --config_file <fsdp_yaml> ...`. Detected
    # automatically via WORLD_SIZE>1; this flag is a no-op when not
    # multi-process. See holo4d/eval_fsdp_helper.py for details.
    parser.add_argument("--use_fsdp", action="store_true", default=False,
                        help="Wrap pipe.dit with FSDP if launched via "
                             "accelerate launch (no-op single-process).")

    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--profile_jsonl", type=str, default=None,
        help=("Append one JSON record per generated pair with synchronized "
              "pipeline wall time, MP4-save time, and CUDA peak memory."),
    )
    parser.add_argument(
        "--profile_stage_breakdown", action="store_true", default=False,
        help=("Record synchronized condition/cache/track-token, Wan denoise, "
              "and VAE decode stage timings in --profile_jsonl."),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers (copied from eval_track4d360_channel_concat_syn4d.py)
# ---------------------------------------------------------------------------

def _video_to_uint8_frames(video) -> list[np.ndarray]:
    """Normalize a pipeline video output (list of PIL frames or BCTHW/CTHW
    tensor in [-1,1] or [0,1]) to a list of HWC uint8 arrays."""
    if isinstance(video, list):
        return [np.asarray(frame) for frame in video]
    if torch.is_tensor(video):
        tensor = video.detach().cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected video tensor with 4 dims after batch squeeze, got {tuple(tensor.shape)}"
            )
        tensor = tensor.permute(1, 2, 3, 0).float()
        if float(tensor.min()) < 0.0:
            tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0.0, 1.0)
        return [(frame.numpy() * 255.0).astype(np.uint8) for frame in tensor]
    raise TypeError(f"Unsupported video type: {type(video)}")


def _as_i2v_boundary_image(frame) -> Image.Image:
    """Convert a decoded/loaded boundary frame to the strict Wan-I2V type."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(
            f"I2V boundary must be an HWC RGB/RGBA frame, got {array.shape}"
        )
    if array.dtype != np.uint8:
        if not np.isfinite(array).all():
            raise ValueError("I2V boundary contains non-finite values")
        value_min = float(array.min())
        value_max = float(array.max())
        if np.issubdtype(array.dtype, np.floating):
            if 0.0 <= value_min and value_max <= 1.0:
                array = (array * 255.0).round()
            elif 0.0 <= value_min and value_max <= 255.0:
                array = array.round()
            else:
                raise ValueError(
                    "I2V floating boundary must be in [0, 1] or [0, 255], got "
                    f"range [{value_min}, {value_max}]"
                )
        elif not (0 <= value_min and value_max <= 255):
            raise ValueError(
                "I2V integer boundary must be in [0, 255], got "
                f"range [{value_min}, {value_max}]"
            )
        array = array.astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array, mode="RGB")


def _frame_to_uint8_rgb(frame) -> np.ndarray:
    """Normalize one visualization frame without silently clipping bad data."""
    return np.asarray(_as_i2v_boundary_image(frame), dtype=np.uint8)


def _read_first_video_frame(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        return _frame_to_uint8_rgb(reader.get_data(0))
    finally:
        reader.close()


def _captioned_frame(frame, caption: str, *, size: tuple[int, int]) -> np.ndarray:
    image = _as_i2v_boundary_image(frame).resize(size, Image.Resampling.LANCZOS)
    caption_height = 30
    canvas = Image.new("RGB", (size[0], size[1] + caption_height), "white")
    canvas.paste(image, (0, caption_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 6), caption, fill="black", font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _write_i2v_boundary_contact_sheet(
    path: Path,
    *,
    previous_source,
    previous_target_gt,
    generated_target_condition,
    current_source,
    current_target_gt,
    continuation_prediction,
) -> None:
    """Write the exact temporal/camera boundary contract as a 2x3 image."""
    frames = (
        (previous_source, "previous source f48 (src cam)"),
        (previous_target_gt, "previous target GT f48"),
        (generated_target_condition, "generated target f48 (I2V condition)"),
        (current_source, "current source f0 (src cam)"),
        (current_target_gt, "current target GT f0"),
        (continuation_prediction, "continuation prediction f0"),
    )
    panel_size = (384, 216)
    panels = [
        _captioned_frame(frame, caption, size=panel_size)
        for frame, caption in frames
    ]
    sheet = np.concatenate(
        [np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)],
        axis=0,
    )
    imageio.imwrite(path, sheet)


def save_video_png_frames(video, frames_dir: Path) -> list[Path]:
    """Dump per-frame PNGs (frame_00000.png, ...) from the in-RAM video.

    Used by the orbit-chain loop: the next round's 4RC preprocessing consumes
    a directory of frames, and dumping from the pre-encode tensor avoids the
    yuv420p quality loss of re-decoding the mp4.
    """
    if not is_main_process():
        return []
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, frame in enumerate(_video_to_uint8_frames(video)):
        frame_path = frames_dir / f"frame_{idx:05d}.png"
        imageio.imwrite(frame_path, frame)
        paths.append(frame_path)
    return paths


def save_video_frames(video, path: Path, fps: int = WAN_SAMPLE_FPS):
    # Under FSDP, only rank 0 writes outputs; other ranks return early
    # to avoid race conditions on the same output paths.
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _video_to_uint8_frames(video)
    # MP4 encoding quality. Use the shared Track4D artifact writer so eval and
    # sampling outputs stay aligned with the Vista4D-style imageio settings.
    import os as _os
    _crf = _os.environ.get("TRACK4D_EVAL_MP4_CRF")
    _preset = _os.environ.get("TRACK4D_EVAL_MP4_PRESET", "slow")
    _quality = int(
        _os.environ.get("TRACK4D_EVAL_MP4_QUALITY", str(PREDICTION_MP4_QUALITY))
    )
    _pix_fmt = _os.environ.get("TRACK4D_EVAL_MP4_PIX_FMT", "yuv420p")
    if str(path).endswith(".mp4"):
        output_params = []
        if _crf:
            output_params.extend(["-crf", str(_crf), "-preset", str(_preset)])
        save_mp4(
            path,
            np.stack(frames, axis=0),
            fps=fps,
            quality=_quality,
            pixelformat=_pix_fmt,
            output_params=output_params or None,
        )
    else:
        imageio.mimsave(path, frames, fps=fps)


# ---------------------------------------------------------------------------
# Pipeline setup — mirrors train_track4d360.py initialisation order exactly.
# ---------------------------------------------------------------------------

def build_eval_pipeline(
    args,
    checkpoint_path=None,
    model_id_with_origin_paths=None,
    camera_control_checkpoint=None,
):
    checkpoint_path = checkpoint_path or args.checkpoint_path
    model_id_with_origin_paths = (
        model_id_with_origin_paths
        if model_id_with_origin_paths is not None
        else args.model_id_with_origin_paths
    )
    camera_control_checkpoint = (
        camera_control_checkpoint
        if camera_control_checkpoint is not None
        else args.camera_control_checkpoint
    )
    if (
        getattr(args, "enable_v2v_plucker_camera_control", False)
        and not getattr(args, "use_plucker_camera_control", False)
    ):
        raise ValueError(
            "--enable_v2v_plucker_camera_control requires --use_plucker_camera_control."
        )
    if getattr(args, "use_plucker_raymap_tokens", False):
        if not getattr(args, "use_plucker_camera_control", False) or not getattr(args, "enable_v2v_plucker_camera_control", False):
            raise ValueError(
                "--use_plucker_raymap_tokens requires --use_plucker_camera_control "
                "and --enable_v2v_plucker_camera_control."
            )
        if not getattr(args, "use_perlayer_camera_control", False):
            print("[Eval] --use_plucker_raymap_tokens implies --use_perlayer_camera_control.")
        args.use_perlayer_camera_control = True
        if getattr(args, "camera_injection_mode", "projector") != "projector":
            raise ValueError("--use_plucker_raymap_tokens requires --camera_injection_mode projector.")
        if getattr(args, "camera_plucker_encoder_mode", "patchify") != "patchify":
            raise ValueError("--use_plucker_raymap_tokens requires --camera_plucker_encoder_mode patchify.")

    helper = DiffusionTrainingModule()
    model_configs = helper.parse_model_configs(
        args.model_paths,
        model_id_with_origin_paths,
        fp8_models=getattr(args, "fp8_models", None),
        offload_models=getattr(args, "offload_models", None),
        device=args.device,
    )
    tokenizer_config = (
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
        if args.tokenizer_path is None else ModelConfig(args.tokenizer_path)
    )

    pipe = WanVideoTrack4D360Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16 if "cuda" in str(args.device) else torch.float32,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    # NOTE: LoRA wrapping is DEFERRED until after VXF init (see below).
    # LoRA wrap renames self_attn.q.weight → self_attn.q.base_layer.weight,
    # which causes VXF init's key matching to silently miss all wrapped keys
    # (missing=1201). Same ordering invariant as train_track4d360.py.
    _lora_base_models = (
        [m.strip() for m in args.lora_base_model.split(",") if m.strip()]
        if args.lora_base_model is not None else []
    )
    if args.disable_track_modules:
        pipe.disable_track_conditioning_units()

    track_cfg = None
    if args.disable_track_modules:
        if args.use_dense_branch:
            if getattr(args, "dense_only_condition_modules", False):
                raise ValueError(
                    "--dense_only_condition_modules is only for dense-only inference mode; "
                    "do not combine with --disable_track_modules."
                )
            raise ValueError("--use_dense_branch is incompatible with --disable_track_modules.")
        if args.use_track_null_token_fallback or args.use_dense_null_token_fallback:
            raise ValueError("Null-token fallback requires Track4D modules; remove the fallback flags when --disable_track_modules is set.")
        print("[Eval] disable_track_modules=True: skipping persistent/dense Track4D init.")
    else:
        # 1. Persistent track adapter (matches training's track_config)
        track_cfg = load_track_config(args.track_config)
        dit_dim = int(getattr(pipe.dit, "dim", 1536))
        if getattr(args, "dense_only_condition_modules", False):
            if args.use_track_null_token_fallback:
                raise ValueError("--dense_only_condition_modules cannot use persistent track null-token fallback.")
            pipe.disable_persistent_track_conditioning_unit()
            pipe.initialize_dense_only_control_modules(
                track_embed_dim=track_cfg.embed_dim,
                dit_dim=dit_dim,
                track_injection_mode=args.track_injection_mode,
                track_injection_block_mode=getattr(args, "track_injection_block_mode", "additive"),
            )
        else:
            # Parity with train_track4d360: forward only the include_* flags the
            # track config declares (undeclared -> pipeline channel-count fallback),
            # so dynmask configs enable the per-track mask channels and in_channels
            # matches what the builder produces. Legacy configs declare none -> no-op.
            _include_key_map = {
                "include_source_depth": "source_depth",
                "include_source_rgb": "source_rgb",
                "include_track_anchor": "track_anchor",
                "include_source_dynamic_mask": "source_dynamic_mask",
                "include_target_dynamic_mask": "target_dynamic_mask",
                "include_reprojected_rgb": "reprojected_rgb",
            }
            track_include_flags = {
                short: bool(track_cfg.get(long))
                for long, short in _include_key_map.items()
                if long in track_cfg
            } or None
            pipe.initialize_track_modules(
                track_in_channels=track_cfg.get("in_channels", 5),
                track_embed_dim=track_cfg.embed_dim,
                track_num_freqs=track_cfg.num_freqs,
                track_hidden_dim=track_cfg.hidden_dim,
                track_num_mlp_layers=track_cfg.num_mlp_layers,
                dit_dim=dit_dim,
                track_injection_mode=args.track_injection_mode,
                track_injection_block_mode=getattr(args, "track_injection_block_mode", "additive"),
                adapter_type=track_cfg.get("adapter_type", "scatter"),
                conv_proj_dim=track_cfg.get("conv_proj_dim", 32),
                conv_track_stride=track_cfg.get("conv_track_stride", 4),
                conv3d_num_residual_blocks=track_cfg.get("conv3d_num_residual_blocks", 1),
                track_coordinate_mode=track_cfg.get("coordinate_mode", "world_xyz"),
                temporal_pool=track_cfg.get("temporal_pool", "max"),
                temporal_attn_heads=track_cfg.get("temporal_attn_heads", 4),
                track_include_flags=track_include_flags,
                persistent_identity_mode=track_cfg.get(
                    "persistent_identity_mode", "legacy"
                ),
                persistent_identity_dim=track_cfg.get(
                    "persistent_identity_dim", 128
                ),
            )
        if args.use_track_null_token_fallback:
            pipe.initialize_track_null_token()

        # 2. Dense target-space encoder (optional — matches training args)
        if args.use_dense_branch:
            persistent_out_dim = int(getattr(pipe.track_adapter, "out_dim", track_cfg.embed_dim))
            spatial_div = getattr(pipe, "height_division_factor", 16)
            dense_patch_size = spatial_div // int(track_cfg.get("conv_track_stride", 4))
            pipe.initialize_dense_target_control_encoder(
                in_channels=int(args.dense_in_channels),
                proj_dim=int(args.dense_proj_dim),
                out_dim=persistent_out_dim,
                patch_size=int(dense_patch_size),
                temporal_stride=4,
                num_residual_blocks=int(args.dense_num_residual_blocks),
                alpha_track=float(args.dense_alpha_track),
                use_depth_null_token=bool(getattr(args, "use_depth_null_token", False)),
            )
            if args.use_dense_null_token_fallback:
                pipe.initialize_dense_null_token()
        elif args.use_dense_null_token_fallback:
            raise ValueError("--use_dense_null_token_fallback requires --use_dense_branch.")

    # 3. Domain embedding for 2-domain width-concat V2V
    _ensure_domain_embedding(pipe.dit, num_domains=2)

    # 4. Query-frame impulse control patch embedding (matches training)
    if args.use_query_frame_impulse_condition:
        for name in ("dit", "dit2"):
            model = getattr(pipe, name, None)
            if model is None or not hasattr(model, "ensure_control_patch_embedding"):
                continue
            vae_z = int(getattr(getattr(getattr(pipe, "vae", None), "model", None), "z_dim", 16))
            dit_in = int(getattr(model, "in_dim", vae_z))
            control_c = max(int(dit_in), int(vae_z))
            model.ensure_control_patch_embedding(in_channels=control_c)

    # 4b. Plucker camera control adapter (matches training when enabled).
    #     MUST run BEFORE the VXF init load below so the control_adapter.*
    #     keys exist in pipe.dit.state_dict() and the vxf_init copy loop
    #     populates them from the Fun-Control-Camera weights.
    if args.use_plucker_camera_control:
        from holo4d_runtime.models.wan_video_camera_controller import (
            initialize_camera_control_adapter,
            initialize_perlayer_camera_encoders,
            initialize_camera_patchify_layers,
            initialize_camera_raymap_token_projection,
            initialize_vista4d_plucker_camera_encoders,
        )
        initialize_camera_control_adapter(pipe.dit, pipe.vae, in_dim_control_adapter=24)
        print("[Eval] Plucker camera control adapter initialized.")

        if getattr(args, "use_plucker_raymap_tokens", False):
            initialize_camera_raymap_token_projection(pipe.dit, pipe.vae, init_zero=True)
            print("[Eval] Plucker raymap-token projection initialized.")
        elif getattr(args, "use_perlayer_camera_control", False):
            cam_mode = getattr(args, "camera_injection_mode", "projector")
            enc_mode = getattr(args, "camera_plucker_encoder_mode", "patchify")
            add_projector = cam_mode in ("direct_raymap_add", "vista4d_plucker")
            if cam_mode == "vista4d_plucker":
                initialize_vista4d_plucker_camera_encoders(
                    pipe.dit, plucker_dim=6, add_projector=add_projector,
                )
            else:
                initialize_perlayer_camera_encoders(
                    pipe.dit, plucker_dim=6, add_projector=add_projector,
                )
            if enc_mode == "patchify" and cam_mode != "vista4d_plucker":
                initialize_camera_patchify_layers(pipe.dit, pipe.vae)
            print(f"[Eval] Per-layer camera control: mode={cam_mode}, "
                  f"encoder={enc_mode}, projector={add_projector}")

    # 4c-bis. Warped appearance conditioning (Lyra-2-inspired)
    warped_mode = getattr(args, "warped_condition_mode", "off")
    if warped_mode == "latent_fuse":
        from holo4d.conditions.shallow_appearance_fuser import (
            initialize_shallow_appearance_fuser,
            initialize_geometry_track_fuser,
        )
        initialize_shallow_appearance_fuser(pipe)
        initialize_geometry_track_fuser(pipe)
        print("[Eval] Warped appearance fuser + geometry+track fuser initialized")
        # NOTE: in latent_fuse + warped_geom_only_dense mode, the dense encoder
        # should already be initialized with in_channels=2 via --dense_in_channels 2.

    # 4c. GS output: initialize gs_feature_head on DiT + gs_decoder so the
    #     checkpoint's GS keys have somewhere to land. Without this, the
    #     gs_feature_head.* keys get resolved to the dit's main head.*
    #     (via _resolve_model_state_key's split-on-first-dot fallback)
    #     and corrupt the noise prediction, causing pure noise at eval.
    if getattr(args, "use_gs_output", False):
        gs_feat_dim = int(getattr(args, "gs_feat_dim", 1024))
        gs_sh_degree = int(getattr(args, "gs_sh_degree", 2))
        gs_npp = int(getattr(args, "gs_num_points_per_pixel", 2))
        pipe.dit.ensure_gs_feature_head(feat_dim=gs_feat_dim)
        from holo4d.models.gs_decoder import GaussianSplattingDecoder
        vae_decoder = pipe.vae.model.decoder
        gs_scale_cap = getattr(args, "gs_scale_cap", None)
        gs_render_patch_size = getattr(args, "gs_render_patch_size", None)
        pipe.gs_decoder = GaussianSplattingDecoder(
            vae_decoder=vae_decoder,
            feat_dim=gs_feat_dim,
            num_points_per_pixel=gs_npp,
            sh_degree=gs_sh_degree,
            use_network_checkpointing=False,
            use_render_checkpointing=False,
            scale_cap=gs_scale_cap,
            render_patch_size=(
                tuple(gs_render_patch_size) if gs_render_patch_size else None
            ),
            render_per_frame_gaussians=not bool(getattr(args, "gs_legacy_union_render", False)),
        )
        print(f"[Eval] GS output initialized: feat_dim={gs_feat_dim}, "
              f"sh_degree={gs_sh_degree}, npp={gs_npp}")
        if bool(getattr(args, "gs_legacy_union_render", False)):
            print("[Eval] GS render compatibility: legacy union render "
                  "(all per-frame Gaussian sets flattened for every view).")

        # Hybrid Lyra head swap (mirrors scripts/wan2.1_fun/train_track4d360.py
        # at the gs_head_style=hybrid block). Required to load v3 / v4 / any
        # exp-depth checkpoint, since the FlashWorld PixelAligned3DGS head
        # has different output channel layout from LyraPixelAligned3DGS.
        # Must run BEFORE ckpt load so gs_proj keys land on Lyra weights.
        _gs_head_style = str(getattr(args, "gs_head_style", "flashworld")).lower()
        _gs_dynamic_mode_eval = str(getattr(args, "gs_dynamic_mode", "static")).lower()
        if _gs_head_style == "hybrid":
            from holo4d.models.gs_decoder_lyra import LyraPixelAligned3DGS
            old_npp = int(pipe.gs_decoder.gs_head.num_points_per_pixel)
            old_embed_dim = int(pipe.gs_decoder.gs_head.gs_proj.in_channels)
            _v4_track_delta = (_gs_dynamic_mode_eval == "static_with_track")
            pipe.gs_decoder.gs_head = LyraPixelAligned3DGS(
                embed_dim=old_embed_dim,
                num_points_per_pixel=old_npp,
                dnear=float(getattr(args, "gs_lyra_dnear", 0.1)),
                dfar=float(getattr(args, "gs_lyra_dfar", 10.0)),
                scale_cap=0.3,
                predict_offset=True,
                depth_param=str(getattr(args, "gs_lyra_depth_param", "exp")),
                log_depth_shift=float(getattr(args, "gs_lyra_log_depth_shift", 0.0)),
                predict_track_delta=_v4_track_delta,
            ).to(
                device=pipe.gs_decoder.gs_head.gs_proj.weight.device,
                # Force float32 — the Lyra head's forward is autocast-disabled
                # and upcasts input to float32 internally, so its conv weights
                # must also be float32 (Conv2d requires matching dtypes when
                # autocast is off). Don't inherit bf16 from the FlashWorld head.
                dtype=torch.float32,
            )
            print(
                f"[Eval] gs_head_style=hybrid: Lyra head "
                f"(npp={old_npp}, embed_dim={old_embed_dim}, "
                f"depth_param={getattr(args, 'gs_lyra_depth_param', 'exp')}, "
                f"log_depth_shift={getattr(args, 'gs_lyra_log_depth_shift', 0.0)}, "
                f"predict_track_delta={_v4_track_delta})"
            )

        # Dynamic GS wrapper: must be attached BEFORE ckpt load so the
        # dynamic_head.proj.{weight,bias} keys in the checkpoint land on the
        # right module. Runs the canonical+deltas head from Lyra/FlashWorld
        # style training (gs_dynamic_mode=deltas).
        if _gs_dynamic_mode_eval == "deltas":
            from holo4d.models.gs_decoder_dynamic import (
                DynamicGaussianSplattingDecoder,
            )
            gs_dyn_max_t = int(getattr(args, "gs_dynamic_max_t_video", 64))
            predict_rot = bool(getattr(args, "gs_predict_rotation", False))
            pipe.gs_decoder.dynamic = DynamicGaussianSplattingDecoder(
                base_decoder=pipe.gs_decoder,
                max_t_video=gs_dyn_max_t,
                predict_rotation=predict_rot,
            )
            print(f"[Eval] GS dynamic mode: deltas "
                  f"(max_t_video={gs_dyn_max_t}, predict_rotation={predict_rot})")

    # 4d. Vista4D/GEN3C latent-stream patch embeddings. These are trained
    # DiT-side modules, so they must exist before checkpoint load; otherwise
    # their checkpoint keys are ignored as unexpected and inference silently
    # runs without the new-variant condition path.
    if getattr(args, "use_vista4d_latent_streams", False):
        from holo4d_runtime.models.wan_video_camera_controller import (
            initialize_vista4d_latent_stream_patch_embeddings,
        )
        initialize_vista4d_latent_stream_patch_embeddings(pipe.dit)
        print("[Eval] Vista4D latent stream patch embeddings initialized.")

    if getattr(args, "use_gen3c_warp_latent_channels", False):
        from holo4d_runtime.models.wan_video_camera_controller import (
            initialize_gen3c_warp_condition_patch_embedding,
        )
        initialize_gen3c_warp_condition_patch_embedding(pipe.dit, in_channels=32)
        print("[Eval] GEN3C warp condition patch embedding initialized.")

    # 5. VXF init: overwrite the DSTUDIO-loaded DiT backbone with VideoX-Fun's
    #    own 1.3B DiT weights.  CRITICAL: the trained checkpoint saves only
    #    the fine-tuned heads (self_attn/ffn/domain_embedding/
    #    control_patch_embedding) + track modules.  The rest of the DiT
    #    backbone (cross_attn, norms, patchify, input/output_proj, ...) MUST
    #    come from VXF's native 1.3B — NOT from DSTUDIO's 1.3B, which is a
    #    different model (different size: 3.23 GB vs 5.68 GB).  Training
    #    always passes ``--camera_control_checkpoint``; inference must replicate
    #    the EXACT same two-step init or the model runs with a DSTUDIO
    #    backbone + VXF-trained heads and silently produces garbage.
    #
    #    Mirrors scripts/wan2.1_fun/train_track4d360.py:404-431.
    if camera_control_checkpoint is not None:
        # Match train_track4d360.py:_load_safetensors_state_dict — accept a
        # single .safetensors path, a directory containing the HF-sharded
        # `diffusion_pytorch_model-*.safetensors`, or a glob pattern.
        # Plain Wan 2.1 T2V 14B ships sharded; passing the single-file path
        # is impossible without manual concat.
        import os as _os
        from glob import glob as _glob
        from safetensors.torch import load_file
        _raw = _os.fspath(camera_control_checkpoint)
        if _os.path.isdir(_raw):
            _vxf_paths = sorted(_glob(_os.path.join(_raw, "diffusion_pytorch_model*.safetensors")))
        else:
            _vxf_paths = sorted(_glob(_raw))
            if not _vxf_paths and _os.path.exists(_raw):
                _vxf_paths = [_raw]
        if not _vxf_paths:
            raise FileNotFoundError(f"--camera_control_checkpoint matched no safetensors files: {_raw}")
        vxf_sd = {}
        for _p in _vxf_paths:
            vxf_sd.update(load_file(_p))
        base_in_dim = pipe.dit.patch_embedding.weight.shape[1]
        target_state_keys = set(pipe.dit.state_dict().keys())
        remapped = {}
        for k, v in vxf_sd.items():
            if k == "patch_embedding.weight":
                if "patch_embedding_v2v.weight" in target_state_keys:
                    remapped["patch_embedding_v2v.weight"] = v
                source_c = int(v.shape[1])
                if source_c >= int(base_in_dim) * 2:
                    # VideoX-Fun stores [cond | noise]; warm-start from the
                    # noise half so the base patch embedding matches the
                    # target branch.
                    remapped["patch_embedding.weight"] = v[:, source_c - int(base_in_dim):]
                else:
                    remapped["patch_embedding.weight"] = v[:, :base_in_dim]
            elif k == "patch_embedding.bias":
                if "patch_embedding_v2v.bias" in target_state_keys:
                    remapped["patch_embedding_v2v.bias"] = v
                remapped["patch_embedding.bias"] = v
            elif k not in remapped and k in target_state_keys:
                remapped[k] = v
        m_vxf, u_vxf = pipe.dit.load_state_dict(remapped, strict=False)
        print(
            f"[Eval/VXF init] base_in_dim={base_in_dim}, "
            f"loaded from {camera_control_checkpoint}"
        )
        print(
            f"[Eval/VXF init] missing={len(m_vxf)} unexpected={len(u_vxf)} "
            f"(expected missing: domain_embedding + control_patch_embedding.*)"
        )
    else:
        print(
            "[Eval] WARNING: --camera_control_checkpoint NOT supplied. The DiT "
            "backbone will come from whatever was loaded by "
            "--model_id_with_origin_paths (typically a DiffSynth-Studio 1.3B "
            "that is NOT the model your checkpoint was trained on top of). "
            "This will likely produce garbage output. Pass "
            "--camera_control_checkpoint path/to/holo4d_runtime/weights/wan21-1p3b/"
            "diffusion_pytorch_model.safetensors to fix."
        )

    # 6. LoRA wrapping (AFTER VXF init, BEFORE checkpoint load).
    #    Must happen after VXF init so base weights are correct before PEFT
    #    renames them to base_layer.weight.
    if _lora_base_models:
        for base_model_name in _lora_base_models:
            if not hasattr(pipe, base_model_name) or getattr(pipe, base_model_name) is None:
                raise ValueError(f"--lora_base_model requested {base_model_name!r}, but the pipeline has no such model.")
            wrapped = helper.add_lora_to_model(
                getattr(pipe, base_model_name),
                target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
                lora_rank=int(args.lora_rank),
                lora_alpha=args.lora_alpha,
                upcast_dtype=pipe.torch_dtype,
            )
            setattr(pipe, base_model_name, wrapped)
        print(f"[Eval] LoRA applied to {', '.join(_lora_base_models)}")

    def _maybe_widen_feedback_patch_embedding(state_dict):
        if not hasattr(pipe, "dit") or not hasattr(pipe.dit, "widen_patch_embedding_for_feedback"):
            return
        patch_w = state_dict.get("patch_embedding.weight")
        if patch_w is None or patch_w.ndim < 2:
            return
        ckpt_in = int(patch_w.shape[1])
        model_in = int(pipe.dit.patch_embedding.in_channels)
        if ckpt_in > model_in:
            pipe.dit.widen_patch_embedding_for_feedback(ckpt_in - model_in)
            print(
                f"[Eval] widened DiT patch_embedding for feedback: "
                f"{model_in} -> {ckpt_in} input channels"
            )

    # 7. Load trained checkpoint.
    #    Non-LoRA checkpoints: load directly into DiT + Track4D modules.
    #    LoRA checkpoints: PEFT-load LoRA weights + restore non-LoRA params
    #    (domain_embedding/control_patch_embedding) separately.
    if args.lora_checkpoint is not None:
        lora_sd = load_state_dict(args.lora_checkpoint)
        lora_state_dict = helper.mapping_lora_state_dict(lora_sd)
        missing, unexpected = pipe.dit.load_state_dict(lora_state_dict, strict=False)
        print(f"[Eval] LoRA checkpoint load: missing={len(missing)} unexpected={len(unexpected)}")

        non_lora_params = {
            key: value
            for key, value in lora_sd.items()
            if "lora_A" not in key and "lora_B" not in key
        }
        if non_lora_params:
            non_lora_params = helper._adapt_non_lora_params_for_model(pipe.dit, non_lora_params)
            model_device = next(pipe.dit.parameters()).device
            non_lora_params = {
                k: v.to(device=model_device, dtype=pipe.torch_dtype)
                for k, v in non_lora_params.items()
            }
            missing_non_lora, unexpected_non_lora = pipe.dit.load_state_dict(non_lora_params, strict=False)
            loaded_non_lora = len(non_lora_params) - len(missing_non_lora)
            print(
                f"[Eval] Non-LoRA param restore: loaded={loaded_non_lora} "
                f"missing={len(missing_non_lora)} unexpected={len(unexpected_non_lora)}"
            )
        if not args.disable_track_modules:
            pipe.load_track_checkpoint(args.lora_checkpoint, verbose=True)
    else:
        # Mixed T2V/I2V stages resolve checkpoints as Path objects; the shared loader
        # expects a string for suffix checks.
        checkpoint_path = str(checkpoint_path)
        ckpt_sd = load_state_dict(checkpoint_path)

        # Split out gs_decoder keys BEFORE _adapt_non_lora_params_for_model,
        # which has a split-on-first-dot fallback that would mis-resolve
        # gs_feature_head.head.weight → head.weight (the dit's main output
        # head), corrupting noise predictions.
        gs_decoder_sd = {}
        dit_sd = {}
        for k, v in ckpt_sd.items():
            if k.startswith("pipe.gs_decoder."):
                gs_decoder_sd[k[len("pipe.gs_decoder."):]] = v
            elif k.startswith("gs_decoder."):
                gs_decoder_sd[k[len("gs_decoder."):]] = v
            else:
                dit_sd[k] = v

        _maybe_widen_feedback_patch_embedding(dit_sd)
        dit_sd = helper._adapt_non_lora_params_for_model(pipe.dit, dit_sd)
        missing, unexpected = pipe.dit.load_state_dict(dit_sd, strict=False)
        print(f"[Eval] DiT checkpoint load: missing={len(missing)} unexpected={len(unexpected)}")

        if gs_decoder_sd and hasattr(pipe, "gs_decoder") and pipe.gs_decoder is not None:
            m_gs, u_gs = pipe.gs_decoder.load_state_dict(gs_decoder_sd, strict=False)
            loaded_gs = len(gs_decoder_sd) - len([k for k in m_gs if k in gs_decoder_sd])
            print(f"[Eval] GS decoder checkpoint load: {len(gs_decoder_sd)} keys, "
                  f"missing={len(m_gs)} unexpected={len(u_gs)}")
        elif gs_decoder_sd:
            print(f"[Eval] WARNING: checkpoint has {len(gs_decoder_sd)} gs_decoder keys "
                  f"but pipe.gs_decoder is None (--use_gs_output not set?). "
                  f"GS decoder weights NOT loaded.")

        if not args.disable_track_modules:
            pipe.load_track_checkpoint(checkpoint_path, verbose=True)
            # Strict active-parameter coverage for the released dense-track
            # recipes. Frozen base parameters are supplied by earlier loads.
            # LoRA/dense-render research recipes retain their own contracts.
            if (not args.use_dense_branch and getattr(args, 'use_plucker_raymap_tokens', False)
                    and getattr(args, 'track_injection_mode', None) == 'per_block'):
                from holo4d.checkpoint_validation import validate_dense_checkpoint
                pipe._checkpoint_coverage = validate_dense_checkpoint(
                    pipe, ckpt_sd, dit_sd, remapped if camera_control_checkpoint else {})
                print('[Eval] strict dense checkpoint coverage PASS: '
                      + str(pipe._checkpoint_coverage['required_parameter_counts']), flush=True)

    gs_checkpoint_path = getattr(args, "gs_checkpoint_path", None)
    if gs_checkpoint_path is not None:
        gs_sd_full = load_state_dict(gs_checkpoint_path)
        gs_decoder_sd = {}
        dit_overlay_sd = {}
        for k, v in gs_sd_full.items():
            if k.startswith("pipe.gs_decoder."):
                gs_decoder_sd[k[len("pipe.gs_decoder."):]] = v
            elif k.startswith("gs_decoder."):
                gs_decoder_sd[k[len("gs_decoder."):]] = v
            else:
                dit_overlay_sd[k] = v

        if dit_overlay_sd:
            _maybe_widen_feedback_patch_embedding(dit_overlay_sd)
            dit_overlay_sd = helper._adapt_non_lora_params_for_model(pipe.dit, dit_overlay_sd)
            missing_gsdit, unexpected_gsdit = pipe.dit.load_state_dict(dit_overlay_sd, strict=False)
            loaded_gsdit = len(dit_overlay_sd) - len(unexpected_gsdit)
            print(f"[Eval] GS overlay DiT load: loaded={loaded_gsdit} "
                  f"missing={len(missing_gsdit)} unexpected={len(unexpected_gsdit)}")

        if gs_decoder_sd:
            if hasattr(pipe, "gs_decoder") and pipe.gs_decoder is not None:
                m_gs, u_gs = pipe.gs_decoder.load_state_dict(gs_decoder_sd, strict=False)
                print(f"[Eval] GS overlay decoder load: {len(gs_decoder_sd)} keys, "
                      f"missing={len(m_gs)} unexpected={len(u_gs)}")
            else:
                print(f"[Eval] WARNING: gs_checkpoint_path has {len(gs_decoder_sd)} gs_decoder keys "
                      f"but pipe.gs_decoder is None (--use_gs_output not set?). "
                      f"GS decoder weights NOT loaded.")

    # Move gs_decoder to the same device/dtype as the pipeline so it's ready
    # for both in-loop GS feedback and post-loop _render_gs_eval.
    if hasattr(pipe, "gs_decoder") and pipe.gs_decoder is not None:
        pipe.gs_decoder = pipe.gs_decoder.to(device=pipe.device, dtype=pipe.torch_dtype)
        # LyraPixelAligned3DGS.forward is decorated `@autocast(enabled=False)`
        # and explicitly upcasts its input to float32 before the gs_proj /
        # track_delta_proj convs. With autocast off, Conv2d requires input
        # and weight to share dtype — so the head's parameters MUST stay
        # in float32 even though the rest of gs_decoder (extra_conv_in +
        # decoder body) lives in pipe.torch_dtype (bf16). Re-cast the
        # head back to float32 to override the recursive bf16 cast above.
        from holo4d.models.gs_decoder_lyra import LyraPixelAligned3DGS
        if (
            getattr(pipe.gs_decoder, "gs_head", None) is not None
            and isinstance(pipe.gs_decoder.gs_head, LyraPixelAligned3DGS)
        ):
            pipe.gs_decoder.gs_head = pipe.gs_decoder.gs_head.to(dtype=torch.float32)
            print("[Eval] LyraPixelAligned3DGS gs_head pinned to float32 "
                  "(forward runs autocast-disabled, requires fp32 conv weights)")

    # Sanity dtype/device print — catches the "bf16 flag set but model still in
    # fp32" class of bug before it manifests as OOM at forward time.
    patch_w = pipe.dit.patch_embedding.weight
    print(
        f"[Eval] DiT patch_embedding.weight dtype={patch_w.dtype} device={patch_w.device}, "
        f"pipe.torch_dtype={getattr(pipe, 'torch_dtype', None)}"
    )

    # ---- Optional FSDP wrap (multi-GPU sharded inference) ----
    # No-op when not launched via `accelerate launch` (WORLD_SIZE<=1).
    # Mirrors the training-side FSDP setup at
    # train_track4d360.py:1052-1090 (hide VAE + frozen DiT submodules,
    # then accelerator.prepare(pipe.dit) so DiTBlocks get FSDP-wrapped).
    # MUST run AFTER all weight loading (VXF init / DiT ckpt /
    # LoRA wrap / GS overlay) — wrapping renames keys to
    # _fsdp_wrapped_module.* which would silently miss subsequent
    # state-dict loads (same Invariant B class as the training-side
    # LoRA-vs-VXF-init ordering).
    if getattr(args, "use_fsdp", False):
        from holo4d.eval_fsdp_helper import maybe_init_fsdp
        maybe_init_fsdp(pipe)

    return pipe


def _summarize_vista4d_plucker_modules(pipe, args):
    if getattr(args, "camera_injection_mode", "projector") != "vista4d_plucker":
        return
    if not getattr(args, "use_perlayer_camera_control", False):
        return

    blocks = list(getattr(pipe.dit, "blocks", []) or [])
    if not blocks:
        raise RuntimeError("vista4d_plucker eval found no DiT blocks")

    cam_weight_norms = []
    cam_bias_norms = []
    projector_delta_norms = []
    for block in blocks:
        enc = getattr(block, "cam_encoder", None)
        if enc is None:
            raise RuntimeError("vista4d_plucker eval missing block.cam_encoder after checkpoint load")
        cam_weight_norms.append(float(enc.weight.detach().float().norm().item()))
        if getattr(enc, "bias", None) is not None:
            cam_bias_norms.append(float(enc.bias.detach().float().norm().item()))
        proj = getattr(block, "camera_projector", None)
        if proj is None:
            raise RuntimeError("vista4d_plucker eval missing block.camera_projector after checkpoint load")
        eye = torch.eye(
            int(proj.weight.shape[0]),
            int(proj.weight.shape[1]),
            device=proj.weight.device,
            dtype=torch.float32,
        )
        projector_delta_norms.append(float((proj.weight.detach().float() - eye).norm().item()))

    def _fmt(values):
        if not values:
            return "none"
        arr = np.asarray(values, dtype=np.float64)
        return f"min={arr.min():.6g} mean={arr.mean():.6g} max={arr.max():.6g}"

    print(
        "[Eval] Vista4D-Plucker module audit: "
        f"cam_encoder.weight_norm({_fmt(cam_weight_norms)}), "
        f"cam_encoder.bias_norm({_fmt(cam_bias_norms)}), "
        f"camera_projector.delta_identity_norm({_fmt(projector_delta_norms)})",
        flush=True,
    )
    if getattr(args, "strict_vista4d_plucker_nonzero", False) and max(cam_weight_norms) <= 1.0e-8:
        raise RuntimeError(
            "vista4d_plucker cam_encoder weights are still zero after checkpoint load. "
            "This means the trained camera modules were not loaded, or the checkpoint did not train them."
        )


def precompute_eval_prompt_embs(pipe, prompts):
    """Populate ``pipe.precomputed_prompt_embs`` for eval prompt strings.

    When ``HOLO4D_EVAL_T5_CACHE`` is set, load the exact prompt table from
    disk before touching the tokenizer or text encoder. This permits an eval
    model bundle that omits T5 entirely. Strict coverage uses the existing
    ``HOLO4D_REQUIRE_T5_CACHE`` / ``HOLO4D_ALLOW_ONLINE_T5_ENCODE`` knobs.
    """
    unique_prompts = []
    seen = set()
    for prompt in prompts:
        prompt = "" if prompt is None else str(prompt)
        if prompt in seen:
            continue
        seen.add(prompt)
        unique_prompts.append(prompt)

    disk_cache = os.environ.get("HOLO4D_EVAL_T5_CACHE", "").strip()
    if disk_cache:
        from holo4d.eval_prompt_t5_cache import load_eval_prompt_cache

        allow_online = os.environ.get("HOLO4D_ALLOW_ONLINE_T5_ENCODE", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        require_cache = os.environ.get("HOLO4D_REQUIRE_T5_CACHE", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        loaded = load_eval_prompt_cache(
            disk_cache,
            expected_prompts=unique_prompts,
            strict=require_cache and not allow_online,
        )
        precomputed = dict(getattr(pipe, "precomputed_prompt_embs", None) or {})
        precomputed.update(loaded)
        pipe.precomputed_prompt_embs = precomputed
        missing = [prompt for prompt in unique_prompts if prompt not in precomputed]
        print(
            f"[Eval/T5 disk cache] loaded={len(loaded)} required={len(unique_prompts)} "
            f"missing={len(missing)} path={disk_cache}",
            flush=True,
        )
        if not missing:
            return

    text_encoder = getattr(pipe, "text_encoder", None)
    tokenizer = getattr(pipe, "tokenizer", None)
    if text_encoder is None or tokenizer is None:
        missing_count = len(missing) if disk_cache else len(unique_prompts)
        raise RuntimeError(
            "Eval prompt cache is incomplete, but text_encoder/tokenizer is absent; "
            f"cannot encode {missing_count} missing prompt(s)."
        )

    precomputed = dict(getattr(pipe, "precomputed_prompt_embs", None) or {})
    missing_prompts = [prompt for prompt in unique_prompts if prompt not in precomputed]
    if not missing_prompts:
        print(f"[Eval/T5 cache] all {len(unique_prompts)} prompt embeddings already cached", flush=True)
        pipe.precomputed_prompt_embs = precomputed
        return

    device = next(text_encoder.parameters()).device
    was_training = bool(getattr(text_encoder, "training", False))
    text_encoder.eval()
    with torch.no_grad():
        for prompt_str in missing_prompts:
            ids, mask = tokenizer(prompt_str, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            emb = text_encoder(ids, mask)
            for _, v in enumerate(seq_lens):
                emb[:, v:] = 0
            L = max(int(seq_lens.item()), 1)
            precomputed[prompt_str] = (
                emb[0, :L].to(dtype=torch.bfloat16, device="cpu").contiguous().clone()
            )
    if was_training:
        text_encoder.train()

    pipe.precomputed_prompt_embs = precomputed
    preview = ", ".join(repr(p[:32]) for p in missing_prompts[:3])
    print(
        f"[Eval/T5 cache] precomputed {len(missing_prompts)} prompt embeddings "
        f"(table={len(precomputed)}): {preview}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Dataset — direct Syn4D_Multi_NVS with dense-branch source fields on.
# ---------------------------------------------------------------------------

def build_dataset(args, scene: str, sequence_idx: int) -> Syn4D_Multi_NVS:
    """Instantiate the per-scene dataset.

    ``HOLO4D_COMPACT_BUNDLE_ROOT`` is an explicit smoke-test-only escape
    hatch: the frozen bundle contains the already-materialized 49-frame source
    tracks and camera/video artifacts, so no shared Syn4D mesh metadata is
    needed. The normal loader remains the default for all training/evaluation.
    """
    demo_path = os.environ.get("HOLO4D_DEMO_BUNDLE_PATH")
    if demo_path:
        from holo4d.data.demo_rollout_bundle import DemoRolloutBundle
        return DemoRolloutBundle(demo_path, sequence_idx=sequence_idx, num_frames=args.num_frames)
    compact_root = os.environ.get("HOLO4D_COMPACT_BUNDLE_ROOT")
    if compact_root:
        from holo4d.data.compact_syn4d_bundle import CompactSyn4DBundle
        return CompactSyn4DBundle(compact_root, sequence_idx=sequence_idx, num_frames=args.num_frames)
    need_source_rgb = args.use_dense_branch and not args.disable_track_modules
    need_source_cameras = (
        need_source_rgb or args.enable_v2v_plucker_camera_control
    )
    return Syn4D_Multi_NVS(
        ROOT=args.syn4d_root,
        num_views=args.num_frames,
        resolution=(args.width, args.height),
        steps_per_epoch=1,
        track_spatial_stride=args.track_spatial_stride,
        scene=scene,
        sequence_idx=sequence_idx,
        fixed_start_frame=args.start_frame,
        return_source_depth_video=True,
        return_source_rgb_video=need_source_rgb,
        return_source_cameras=need_source_cameras,
        return_target_depth=False,
        return_target_tracks=False,
        use_mp4_rgb=bool(args.use_mp4_rgb),
        caption_dir=args.syn4d_caption_dir,
    )


def resolve_start_frame(dataset: Syn4D_Multi_NVS, args, group_idx: int = 0) -> int:
    """Resolve the ``--start_frame`` arg against the picked NVS group."""
    group = dataset._nvs_groups[group_idx]
    common_starts = group["common_starts"]
    if str(args.start_frame).lower() == "first":
        return int(common_starts[0])
    start = int(args.start_frame)
    if start not in common_starts:
        raise ValueError(
            f"start_frame={start} not in common_starts for "
            f"{group['scene_name']} seq={group_idx} "
            f"(first five: {common_starts[:5]})"
        )
    return start


def _parse_camera_pairs(spec: str):
    """Parse '0-1,0-2,...' into [(0, 1), (0, 2), ...]."""
    pairs = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" not in raw:
            raise ValueError(f"bad camera pair '{raw}' (expected 'src-tgt')")
        a, b = raw.split("-", 1)
        pairs.append((int(a), int(b)))
    return pairs


def _parse_camera_pairs_by_sequence(spec: str):
    """Parse seq:src-tgt,...;seq:src-tgt,... overrides."""
    overrides = {}
    for raw_group in str(spec).split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        if ":" not in raw_group:
            raise ValueError(f"bad sequence camera-pair override {raw_group!r}")
        raw_seq, raw_pairs = raw_group.split(":", 1)
        seq_idx = int(raw_seq.strip())
        if seq_idx in overrides:
            raise ValueError(f"duplicate sequence camera-pair override for {seq_idx}")
        overrides[seq_idx] = _parse_camera_pairs(raw_pairs)
    return overrides


def _parse_int_list(spec: str):
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def _apply_online_track_xyz_noise(
    sample: dict,
    *,
    multiplier: float,
    base_seed: int,
    scene: str,
    sequence_idx: int,
):
    """Add deterministic per-sample Gaussian noise to valid GT 3-D tracks."""
    if multiplier < 0:
        raise ValueError(f"track XYZ noise multiplier must be non-negative, got {multiplier}")
    tracks = np.asarray(sample["tracks_world"], dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"tracks_world must have shape (N,T,3), got {tracks.shape}")

    valid_raw = sample.get("tracks_world_valid_mask")
    if valid_raw is None:
        valid = np.isfinite(tracks).all(axis=-1)
        valid &= np.linalg.norm(tracks, axis=-1) > 1e-8
    else:
        valid = np.asarray(valid_raw, dtype=bool)
        if valid.shape != tracks.shape[:2]:
            raise ValueError(
                "tracks_world_valid_mask shape mismatch: "
                f"expected {tracks.shape[:2]}, got {valid.shape}"
            )
        valid &= np.isfinite(tracks).all(axis=-1)
    if not np.any(valid):
        raise ValueError("cannot compute online track noise statistics: no valid tracks")

    valid_xyz = tracks[valid].astype(np.float64, copy=False)
    mean_xyz = valid_xyz.mean(axis=0)
    std_xyz = valid_xyz.std(axis=0, ddof=0)
    safe_std_xyz = np.maximum(std_xyz, 1e-8)
    source_key = f"{base_seed}:{scene}:{int(sequence_idx)}"
    digest = hashlib.sha256(source_key.encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(derived_seed)
    base_noise = rng.standard_normal(tracks.shape).astype(np.float32)

    delta = np.zeros_like(tracks, dtype=np.float32)
    delta[valid] = (
        float(multiplier)
        * base_noise[valid]
        * safe_std_xyz.astype(np.float32)[None, :]
    )
    noisy = tracks.copy()
    noisy[valid] += delta[valid]
    result = dict(sample)
    result["tracks_world"] = noisy
    # Clean-track reprojection fields are inconsistent after XYZ perturbation.
    result.pop("reproj_src_depth_at_target", None)
    result.pop("reproj_src_depth_valid", None)

    delta_valid = delta[valid].astype(np.float64, copy=False)
    provenance = {
        "protocol": "per_sample_valid_xyz_axis_population_std_gaussian",
        "multiplier": float(multiplier),
        "base_seed": int(base_seed),
        "derived_seed": int(derived_seed),
        "source_key": source_key,
        "valid_observation_count": int(valid.sum()),
        "mean_xyz": mean_xyz.tolist(),
        "std_xyz": std_xyz.tolist(),
        "safe_std_xyz": safe_std_xyz.tolist(),
        "actual_noise_std_xyz": delta_valid.std(axis=0, ddof=0).tolist(),
        "actual_noise_rms_xyz": float(np.sqrt(np.mean(np.sum(delta_valid**2, axis=-1)))),
    }
    return result, provenance


def _apply_gt_track_object_orbit(sample: dict, args) -> tuple[dict, dict]:
    """Build a uniform subject orbit from the sample's GT 3-D trajectories."""
    sweep = float(getattr(args, "gt_track_orbit_deg", 0.0))
    if abs(sweep) <= 1e-8:
        return sample, {}
    tracks = np.asarray(sample["tracks_world"], dtype=np.float32)
    valid = sample.get("tracks_world_valid_mask")
    pivots, moving = estimate_moving_track_pivots_world(
        tracks_world=tracks,
        valid_mask=None if valid is None else np.asarray(valid, dtype=bool),
        motion_threshold=float(args.gt_track_orbit_motion_threshold),
        min_identities=int(args.gt_track_orbit_min_identities),
        temporal_smooth_radius=2,
    )
    source_c2w = np.asarray(sample["source_cam2world"], dtype=np.float32)
    direction = str(args.gt_track_orbit_direction)
    mode = str(args.gt_track_orbit_mode)
    fixed_pivot = None
    world_up = None
    if mode == "fixed_height_look_at":
        if sample.get("source_world_up") is None:
            raise ValueError(
                "fixed_height_look_at requires source_world_up from the direct "
                "Syn4D loader; do not use a legacy compact bundle"
            )
        world_up = np.asarray(sample["source_world_up"], dtype=np.float32)
        target_c2w, angles, radii, fixed_pivot = (
            build_fixed_height_look_at_orbit_sequence(
                source_cam2world=source_c2w,
                pivots_world=pivots,
                trajectory_name=f"orbit_{direction}",
                orbit_angle_deg=sweep,
                world_up=world_up,
                radius_scale=float(args.gt_track_orbit_radius_scale),
            )
        )
        protocol = "syn4d_fixed_height_look_at_gt_track_arc_v2"
    else:
        target_c2w, angles, radii = build_object_centric_orbit_sequence(
            source_cam2world=source_c2w,
            pivots_world=pivots,
            trajectory_name=f"orbit_{direction}",
            orbit_angle_deg=sweep,
        )
        protocol = "syn4d_moving_gt_track_object_orbit_v1"
    result = dict(sample)
    result["target_cam2world"] = target_c2w
    result["target_intrinsics"] = np.asarray(sample["source_intrinsics"], dtype=np.float32)
    result.pop("target_intrinsics_per_frame", None)
    # The recorded Syn4D target view no longer matches this synthetic camera.
    # Explicitly prevent it from entering the target latent condition.
    result["inference_target_video"] = None
    result.pop("reproj_src_depth_at_target", None)
    result.pop("reproj_src_depth_valid", None)
    result["gt_track_orbit"] = {
        "protocol": protocol,
        "mode": mode,
        "direction": direction,
        "sweep_deg": sweep,
        "radius_scale": float(args.gt_track_orbit_radius_scale),
        "motion_threshold": float(args.gt_track_orbit_motion_threshold),
        "moving_identity_count": int(moving.sum()),
        "source_identity_count": int(tracks.shape[0]),
        "angles_deg": angles.tolist(),
        "pivots_world": pivots.tolist(),
        "fixed_pivot_world": None if fixed_pivot is None else fixed_pivot.tolist(),
        "world_up": None if world_up is None else world_up.tolist(),
        "source_cam2world": source_c2w.tolist(),
        "target_cam2world": target_c2w.tolist(),
        "radii": radii.tolist(),
        "radius_median": float(np.median(radii)),
        "camera_heights": (
            None
            if world_up is None
            else (
                target_c2w[:, :3, 3] @ (world_up / np.linalg.norm(world_up))
            ).tolist()
        ),
        "camera_center_path_length": float(
            np.linalg.norm(np.diff(target_c2w[:, :3, 3], axis=0), axis=-1).sum()
        ),
    }
    return result, result["gt_track_orbit"]


def _t(x, dtype=torch.float32):
    """Convert numpy / list / tensor to a float tensor, preserving None."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


# ---------------------------------------------------------------------------
# GS eval rendering
# ---------------------------------------------------------------------------

def _render_gs_eval(pipe, sample, args, pair_out_dir: Path, pred_video=None,
                    gs_features_override=None, latents_override=None, output_suffix=""):
    """Render the trained 3DGS head at eval time and save gs_eval.mp4.

    By default uses ``pipe._last_gs_features`` (captured from the FINAL clean
    step of the diffusion loop) and ``pipe._debug_last_target_latents``.
    Pass ``gs_features_override`` / ``latents_override`` to render from a
    mid-trajectory capture (e.g. the σ≈0.242 step matching training). The
    ``output_suffix`` is appended to gs_eval/gs_rgb/gs_depth filenames so
    multiple captures don't collide.
    """
    from holo4d.models.gs_camera_utils import c2w_K_to_flashworld_cameras

    device = pipe.device
    gs_decoder = pipe.gs_decoder

    # Retrieve gs_features (override if provided, else last-step capture).
    gs_features = gs_features_override if gs_features_override is not None else getattr(pipe, "_last_gs_features", None)
    if gs_features is None:
        print("[Eval/GS] no gs_features captured from inference loop — "
              "gs_feature_head may not be loaded or inference didn't run.")
        return

    # Build target cameras in FlashWorld format
    c2w = torch.as_tensor(np.asarray(sample["target_cam2world"]), dtype=torch.float32)
    K = torch.as_tensor(np.asarray(sample["target_intrinsics"]), dtype=torch.float32)
    H_img, W_img = args.height, args.width
    target_cameras = c2w_K_to_flashworld_cameras(c2w, K, H_img, W_img).to(device)
    if target_cameras.dim() == 2:
        target_cameras = target_cameras.unsqueeze(0)  # (1, T_video, 11)
    T_video = int(target_cameras.shape[1])

    # Extract target-domain features (right half of width-concat)
    split = gs_features.shape[4] // 2
    gs_features_target = gs_features[:, :, :, :, split:]
    if bool(getattr(args, "gs_zero_features", False)):
        gs_features_target = torch.zeros_like(gs_features_target)
        print("[Eval/GS] gs_zero_features=True: zeroed target GS feature stream")

    # z input to the GS decoder: use the model's predicted target latents
    # when available (fair eval — matches what the model actually produced),
    # fall back to VAE-encoding the GT target video otherwise.
    if latents_override is not None:
        # Override is full-width (source|target). Split to target half.
        _lw_split = latents_override.shape[4] // 2
        predicted_target_latents = latents_override[:, :, :, :, _lw_split:]
    else:
        predicted_target_latents = getattr(pipe, "_debug_last_target_latents", None)
    if predicted_target_latents is not None:
        tgt_latents = predicted_target_latents.to(dtype=pipe.torch_dtype, device=device)
        if tgt_latents.dim() == 4:
            tgt_latents = tgt_latents.unsqueeze(0)
        print("[Eval/GS] using predicted target latents for GS decode (fair eval)")
    else:
        target_video_raw = sample.get("target_video")
        if target_video_raw is None:
            print("[Eval/GS] no target_video in sample — skipping GS render.")
            return

        tgt_tensor = pipe.preprocess_video(
            target_video_raw, torch_dtype=torch.float32, device=device,
        ).to(dtype=pipe.torch_dtype, device=device)

        with torch.no_grad():
            tgt_latents = pipe.vae.encode(
                tgt_tensor, device=device, tiled=True,
                tile_size=(30, 52), tile_stride=(15, 26),
            ).to(dtype=pipe.torch_dtype, device=device)
        if tgt_latents.dim() == 4:
            tgt_latents = tgt_latents.unsqueeze(0)
        print("[Eval/GS] using GT-encoded target latents for GS decode (use --capture_final_latents for fair eval)")

    # Decode + render the full T_video timeline.
    # If the dynamic wrapper is attached (canonical + per-frame deltas), route
    # through it so the dynamic_head weights are actually exercised. This is
    # what makes a dynamic-trained checkpoint actually look dynamic in eval —
    # the static path (gs_decoder.forward) would ignore the dynamic head.
    dyn_wrap = getattr(gs_decoder, "dynamic", None)
    with torch.no_grad():
        if dyn_wrap is not None:
            from holo4d.models.gs_loss_dynamic import _compose_frame_gaussians
            from holo4d.models.gs_renderer import gaussian_render
            T_video = int(target_cameras.shape[1])
            canonical_params, delta_params = dyn_wrap(
                gs_features_target, tgt_latents, target_cameras[:, 0], T_video,
            )
            # canonical: (B, npp, H, W, D); delta: (B, T_video-1, npp, H, W, ch)
            B = canonical_params.shape[0]
            npp, H_full, W_full, D = canonical_params.shape[1:]
            sh_deg = dyn_wrap.sh_degree
            predict_rotation = bool(dyn_wrap.predict_rotation)

            # Build target c2w + intrinsics once (reuse the camera adapter logic
            # from gs_loss_dynamic — do the same 11-param → c2w/intr conversion
            # that PixelAligned3DGS uses under the hood).
            from holo4d.models.gs_camera_utils import quaternion_to_matrix
            test_c2ws = torch.eye(4, device=target_cameras.device)[None, None].repeat(
                B, T_video, 1, 1
            ).float()
            test_c2ws[:, :, :3, :3] = quaternion_to_matrix(target_cameras[:, :, :4])
            test_c2ws[:, :, :3, 3] = target_cameras[:, :, 4:7]
            fx, fy, cx, cy = target_cameras[:, :, 7:11].split([1, 1, 1, 1], dim=-1)
            test_intr = torch.cat([
                fx * W_img, fy * H_img, cx * W_img, cy * H_img,
            ], dim=-1)

            rendered_rgb_frames = []
            rendered_depth_frames = []
            for t_idx in range(T_video):
                full_t, _, _ = _compose_frame_gaussians(
                    canonical_params,
                    delta_params,
                    predict_rotation,
                    bool(getattr(args, "gs_disable_delta_opacity", False)),
                    t_idx,
                )
                flat = full_t.reshape(B, npp * H_full * W_full, D).unsqueeze(1)
                rgb_t, depth_t = gaussian_render(
                    flat, test_c2ws[:, t_idx:t_idx + 1], test_intr[:, t_idx:t_idx + 1],
                    W_img, H_img,
                    sh_degree=sh_deg, bg_mode="white", use_checkpoint=False,
                    patch_size=getattr(dyn_wrap, "render_patch_size", None),
                )
                # rgb_t: (B, 1, 3, H, W), depth_t: (B, 1, 1, H, W)
                rendered_rgb_frames.append(rgb_t)
                rendered_depth_frames.append(depth_t)
            rendered_rgb = torch.cat(rendered_rgb_frames, dim=1)
            rendered_depth = torch.cat(rendered_depth_frames, dim=1)
            print(f"[Eval/GS] dynamic render: T_video={T_video}, "
                  f"canonical|delta-composed per-frame")
        else:
            gaussian_params = gs_decoder(gs_features_target, tgt_latents, target_cameras)
            # gaussian_params: (B, T_video, npp, H, W, D)
            keep_frac = float(getattr(args, "gs_opacity_prune_keep_frac", 1.0))
            if keep_frac < 1.0:
                from holo4d.models.gs_loss import apply_opacity_pruning
                gaussian_params = apply_opacity_pruning(
                    gaussian_params,
                    keep_frac=keep_frac,
                )
                print(f"[Eval/GS] opacity pruning before render: keep_frac={keep_frac}")

            rendered_rgb, rendered_depth = gs_decoder.render(
                gaussian_params, target_cameras,
                height=H_img, width=W_img, bg_mode="white",
            )
    # rendered_rgb: (B, T_video, 3, H, W) in [-1, 1]
    # rendered_depth: (B, T_video, 1, H, W)

    # --- Build per-frame panels: [GT | VAE pred | GS RGB | GS depth] with captions ---
    # VAE pred (passed from _run_one_pair)
    if pred_video is not None:
        if isinstance(pred_video, list):
            pred_np = [np.asarray(f, dtype=np.uint8) for f in pred_video]
        elif torch.is_tensor(pred_video):
            t = pred_video.detach().cpu()
            if t.ndim == 5:
                t = t[0]
            t = t.permute(1, 2, 3, 0).float()
            if float(t.min()) < 0.0:
                t = (t + 1.0) / 2.0
            pred_np = [(f.clamp(0, 1).numpy() * 255).astype(np.uint8) for f in t]
        else:
            pred_np = None
    else:
        pred_np = None

    # GT target
    gt_target = sample.get("target_video")
    if gt_target is not None:
        if isinstance(gt_target, list):
            gt_np = [np.asarray(f, dtype=np.uint8) for f in gt_target]
        elif torch.is_tensor(gt_target):
            t = gt_target.detach().cpu()
            if t.ndim == 5:
                t = t[0]
            t = t.permute(1, 2, 3, 0).float()
            if float(t.min()) < 0.0:
                t = (t + 1.0) / 2.0
            gt_np = [(f.clamp(0, 1).numpy() * 255).astype(np.uint8) for f in t]
        else:
            gt_np = None
    else:
        gt_np = None

    # GS RGB: (B, T, 3, H, W) → list of (H, W, 3) uint8
    gs_rgb_tensor = rendered_rgb[0].permute(0, 2, 3, 1).float()  # (T, H, W, 3)
    gs_rgb_np = [
        ((f.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
        for f in gs_rgb_tensor
    ]

    # GS depth: use the same Spectral_r palette as debug_latents for visual parity.
    from holo4d_runtime.diffusion.debug_latents import _colorize_depth_tensor
    depth_tensor = rendered_depth[0, :, 0].float()  # (T, H, W)
    d_min = depth_tensor.min()
    d_max = depth_tensor.max()
    gs_depth_np = _colorize_depth_tensor(depth_tensor)

    # Caption bar height
    try:
        from PIL import Image, ImageDraw, ImageFont
        _has_pil = True
    except ImportError:
        _has_pil = False
    CAPTION_H = 20

    def _add_caption(frame: np.ndarray, text: str) -> np.ndarray:
        """Append a white caption bar below frame."""
        H, W = frame.shape[:2]
        if not _has_pil:
            bar = np.full((CAPTION_H, W, 3), 255, dtype=np.uint8)
            return np.concatenate([frame, bar], axis=0)
        bar = Image.new("RGB", (W, CAPTION_H), (255, 255, 255))
        draw = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((4, 2), text, fill=(0, 0, 0), font=font)
        bar_np = np.asarray(bar, dtype=np.uint8)
        return np.concatenate([frame, bar_np], axis=0)

    # Assemble width-concat frames
    n_frames = len(gs_rgb_np)
    concat_frames = []
    for i in range(n_frames):
        panels = []
        captions = []
        if gt_np is not None and i < len(gt_np):
            panels.append(gt_np[i])
            captions.append("GT target")
        if pred_np is not None and i < len(pred_np):
            panels.append(pred_np[i])
            captions.append("VAE pred")
        panels.append(gs_rgb_np[i])
        captions.append("GS RGB")
        panels.append(gs_depth_np[i])
        captions.append(f"GS depth [{d_min.item():.1f}, {d_max.item():.1f}]")

        # Resize all panels to same H×W (they should match, but guard)
        target_h = panels[0].shape[0]
        target_w = panels[0].shape[1]
        resized = []
        for p, cap in zip(panels, captions):
            if p.shape[0] != target_h or p.shape[1] != target_w:
                from PIL import Image as _PILImage
                p = np.asarray(
                    _PILImage.fromarray(p).resize((target_w, target_h)),
                    dtype=np.uint8,
                )
            resized.append(_add_caption(p, cap))
        concat_frames.append(np.concatenate(resized, axis=1))

    out_path = str(pair_out_dir / f"gs_eval{output_suffix}.mp4")
    save_mp4(out_path, np.stack(concat_frames, axis=0), fps=12)

    # Also save individual GS rgb + depth for programmatic use
    save_mp4(pair_out_dir / f"gs_rgb{output_suffix}.mp4", gs_rgb_np, fps=12)
    save_mp4(pair_out_dir / f"gs_depth{output_suffix}.mp4", gs_depth_np, fps=12)

    print(f"[Eval/GS] saved gs_eval.mp4 (width-concat, {n_frames} frames, "
          f"depth range [{d_min.item():.2f}, {d_max.item():.2f}])")


# ---------------------------------------------------------------------------
# Per-pair inference
# ---------------------------------------------------------------------------

def _sample_with_captured_track_selection(sample, captured_selection):
    """Gather visualization payloads with the exact pipeline-selected rows."""
    if captured_selection is None:
        return sample
    indices = torch.as_tensor(captured_selection["indices"])
    valid_indices = torch.as_tensor(captured_selection["valid"]).bool()
    if indices.ndim != 2 or indices.shape[0] != 1 or valid_indices.shape != indices.shape:
        raise ValueError(
            "Captured visualization selection must have shape (1,K), got "
            f"indices={tuple(indices.shape)} valid={tuple(valid_indices.shape)}"
        )
    selected_rows = indices[0, valid_indices[0]].numpy()
    identity_count = int(np.asarray(sample["tracks_world"]).shape[0])
    if np.any(selected_rows < 0) or np.any(selected_rows >= identity_count):
        raise IndexError("Captured visualization selection contains an out-of-range row")
    comparison_sample = dict(sample)
    row_aligned_keys = (
        "tracks_world",
        "tracks_world_valid_mask",
        "tracks_world_dense_valid_mask",
        "tracks_query_idx_per_identity",
        "reproj_src_depth_at_target",
        "reproj_src_depth_valid",
        "source_rgb_at_tracks",
        "source_dynamic_mask_at_tracks",
        "target_dynamic_mask_at_tracks",
    )
    for key in row_aligned_keys:
        value = sample.get(key)
        if value is None:
            continue
        if int(value.shape[0]) != identity_count:
            raise ValueError(
                f"Visualization payload {key} has N={value.shape[0]}, expected {identity_count}"
            )
        if torch.is_tensor(value):
            row_ids = torch.as_tensor(selected_rows, device=value.device, dtype=torch.long)
            comparison_sample[key] = value.index_select(0, row_ids)
        else:
            comparison_sample[key] = np.asarray(value)[selected_rows]
    comparison_sample["track_visualization_selection"] = {
        "mode": str(captured_selection["mode"]),
        "selected_count": int(selected_rows.size),
        "source_identity_count": identity_count,
    }
    return comparison_sample


def _run_one_pair(pipe, sample, args, pair_out_dir: Path):
    """Run inference on a single (src, tgt, start_frame) sample and save outputs."""
    pair_out_dir.mkdir(parents=True, exist_ok=True)

    # Persistent-branch depth comes from the dataset's reprojected source
    # depth under ``reproj_src_depth_at_target`` / ``reproj_src_depth_valid``.
    # The trainer renames those to ``source_depth_at_tracks`` /
    # ``source_depth_valid_at_tracks`` when forwarding to the pipe; we mirror
    # that here.
    source_depth_at_tracks = sample.get("reproj_src_depth_at_target")
    source_depth_valid_at_tracks = sample.get("reproj_src_depth_valid")

    source_camera = None
    if sample.get("source_cam2world") is not None and sample.get("source_intrinsics") is not None:
        source_camera = {
            "c2w_opencv": _t(sample.get("source_cam2world")),
            "intrinsics": _t(sample.get("source_intrinsics")),
        }
    target_camera = None
    target_k_for_camera = sample.get("target_intrinsics_per_frame")
    if target_k_for_camera is None:
        target_k_for_camera = sample.get("target_intrinsics")
    if sample.get("target_cam2world") is not None and target_k_for_camera is not None:
        target_camera = {
            "c2w_opencv": _t(sample.get("target_cam2world")),
            "intrinsics": _t(target_k_for_camera),
        }

    boundary_condition_requested = bool(
        sample.get(
            "use_target_image_condition",
            getattr(args, "use_target_image_condition", False),
        )
    )
    ti2v_native_i2v = bool(getattr(args, "ti2v_native_i2v", False))

    call_kwargs = dict(
        prompt=sample["prompt"],
        # Cached T5 embedding from the dataset's safetensors lookup (same
        # cache the trainer reads). When present, WanVideoUnit_PromptEmbedder
        # skips T5 online encode entirely. None for novel-trajectory samples
        # where the dataset didn't ship one — the unit's cache table or the
        # online-encode fallback handles it.
        prompt_emb=sample.get("prompt_emb"),
        negative_prompt=NEGATIVE_PROMPT,
        # Eval launchers precompute this string into pipe.precomputed_prompt_embs
        # once per process; leave the direct arg unset so the unit uses that
        # shared table.
        negative_prompt_emb=None,
        input_video=sample["video"],
        # Wan-I2V target-boundary conditioning is opt-in. The mixed Syn4D
        # protocol keeps the T2V seed source-only and passes the previous
        # generated target chunk's final frame for continuation.
        input_image=sample.get("inference_input_image"),
        use_target_image_condition=(
            boundary_condition_requested and not ti2v_native_i2v
        ),
        use_ti2v_native_i2v=(boundary_condition_requested and ti2v_native_i2v),
        # Dropout/noise are training augmentations. Evaluation always uses the
        # supplied boundary image exactly.
        i2v_condition_drop_prob=0.0,
        i2v_condition_noise_level=0.0,
        # Novel-view datasets may retain a real-camera target video only as a
        # visualization reference. If inference_target_video is explicitly
        # present (including None), honor it instead of leaking that reference
        # latent into the generated target domain.
        target_video=(
            sample["inference_target_video"]
            if "inference_target_video" in sample
            else sample.get("target_video")
        ),
        # Persistent track branch
        tracks_world=_t(sample["tracks_world"]),
        tracks_world_valid_mask=_t(sample.get("tracks_world_valid_mask")),
        tracks_query_idx_per_identity=_t(
            sample.get("tracks_query_idx_per_identity"), dtype=torch.long,
        ),
        # Preserve the historic direct-Syn4D eval behavior (global-RNG identity
        # selection). The new autoregressive protocol is deliberately seeded
        # so every shard can resume the same selected continuation cases.
        track_identity_sampling_seed=(
            int(getattr(args, "seed", 0))
            if (
                int(getattr(args, "i2v_continuation_pairs", 0)) > 0
                or int(getattr(args, "i2v_long_horizon_frames", 0)) > 0
            )
            else None
        ),
        tracks_world_dense_valid_mask=_t(sample.get("tracks_world_dense_valid_mask")),
        target_cam2world=_t(sample["target_cam2world"]),
        target_intrinsics=_t(sample["target_intrinsics"]),
        # Per-frame target K (Vista4D zoom trajectories). When set, the pipeline
        # uses it for plucker-ray construction so camera-control rays match the
        # per-frame FOV evolution of the dense-conditioning warp. None for
        # static-K samples (training, DAVIS-2trajperseq, iPhone NVS, etc.).
        target_intrinsics_per_frame=_t(sample.get("target_intrinsics_per_frame")),
        # Persistent 6-channel extras (match training config_srcdepth.yaml)
        source_depth_video=_t(sample.get("source_depth_video")),
        source_depth_at_tracks=_t(source_depth_at_tracks),
        source_depth_valid_at_tracks=(
            torch.as_tensor(np.asarray(source_depth_valid_at_tracks))
            if source_depth_valid_at_tracks is not None else None
        ),
        source_rgb_at_tracks=_t(sample.get("source_rgb_at_tracks")),
        # dynmask track config: per-identity motion label (None on eval samples that
        # don't provide it -> builder zero-fills; strict-check is training-only).
        source_dynamic_mask_at_tracks=_t(sample.get("source_dynamic_mask_at_tracks")),
        has_tracks=sample.get("has_tracks", True),
        source_camera=source_camera,
        target_camera=target_camera,
        # Query-frame impulse conditioning (matches training)
        use_query_frame_impulse_condition=args.use_query_frame_impulse_condition,
        track_query_idx=sample.get("track_query_idx", 0),
        # Plucker camera control: pipeline's WanVideoUnit_CameraControl_NVS
        # auto-builds the rays from target_cam2world + target_intrinsics
        # already in this kwargs dict. Dataset's c2w is in OpenCV convention
        # (verified — same euler_to_rotation_matrix + zxy_xyz Blender→OpenCV
        # flip used in predict_v2v_control_camera_syn4d.py:115).
        use_plucker_camera_control=args.use_plucker_camera_control,
        enable_v2v_plucker_camera_control=args.enable_v2v_plucker_camera_control,
        # Per-layer camera control (Lyra-2-style KQ-only injection)
        use_perlayer_camera_control=getattr(args, "use_perlayer_camera_control", False),
        camera_injection_mode=getattr(args, "camera_injection_mode", "projector"),
        camera_plucker_encoder_mode=getattr(args, "camera_plucker_encoder_mode", "patchify"),
        use_plucker_raymap_tokens=getattr(args, "use_plucker_raymap_tokens", False),
        use_ray_origin=getattr(args, "plucker_use_ray_origin", False),
        # Warped appearance conditioning (Lyra-2-inspired)
        warped_condition_mode=getattr(args, "warped_condition_mode", "off"),
        warped_appearance_fusion=getattr(args, "warped_appearance_fusion", "concat_proj"),
        warped_geom_only_dense=getattr(args, "warped_geom_only_dense", True),
        use_vista4d_latent_streams=getattr(args, "use_vista4d_latent_streams", False),
        vista4d_streams_pointcloud_only=getattr(args, "vista4d_streams_pointcloud_only", False),
        use_gen3c_warp_latent_channels=getattr(args, "use_gen3c_warp_latent_channels", False),
        # Bug 2 fix (2026-06-17): forward the dynamic-mask disable contract so
        # eval matches the trained config. Without this the dense / vista4d units
        # silently fall back to their default (disable=False) regardless of the
        # CLI flag, so a mask-OFF-trained config (e.g. ab23) was evaluated
        # mask-ON. Reaches the units via the pipeline's extra_kwargs ->
        # inputs_shared catch-all (base_pipeline.py:465).
        dense_hybrid_disable_dynamic_masks=bool(getattr(args, "dense_hybrid_disable_dynamic_masks", False)),
        # Dense reprojection mode: "zbuffer" (default) or "hybrid_softsplat"
        # (Lyra-2/GEN3C bilinear soft-splat with multi-frame fusion at static
        # pixels). Plumbs into inputs_shared via the pipeline's extra_kwargs
        # catch-all (wan_video_track4d360.py:1780 inputs_shared.update),
        # which DenseTargetControlUnit.process reads. Without this, eval
        # silently runs zbuffer regardless of the CLI flag.
        dense_warp_mode=getattr(args, "dense_warp_mode", "zbuffer"),
        dense_warp_motion_threshold=float(getattr(args, "dense_warp_motion_threshold", 0.005)),
        # Shape / sampling
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        v2v_concat_mode="width",
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        camera_condition_scale=float(getattr(args, "camera_condition_scale", 1.0)),
        _debug_camera_control=bool(getattr(args, "debug_camera_control", False)),
        tiled=args.vae_tiling,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        use_domain_embedding=True,
        # Always capture the final post-denoising latents. Cost: one
        # detach+clone+CPU copy per pair (≈ 3-6 MB at fp32 for our
        # 384×672×49 setup), well under the per-pair side-mp4 budget.
        # Saved as pred_latents.safetensors below so the user can
        # re-decode at any quality (CRF / preset / resolution scale)
        # without re-running the 50-step denoising loop.
        debug_capture_final_latents=True,
        profile_stage_timings=bool(getattr(args, "profile_stage_breakdown", False)),
        gs_capture_progress_idx=getattr(args, "gs_capture_progress_idx", None),
        # Show the denoising loop as a tqdm bar (e.g. "32/50 [02:30<01:30, 4.5s/it]").
        # Wrapped with `leave=False` so the bar disappears after each pair completes,
        # keeping the log scrollable rather than littered with stale bars.
        progress_bar_cmd=lambda xs: _tqdm(xs, desc="denoise", leave=False),
    )
    call_kwargs.update(
        ar_training=bool(getattr(args, "ar_training", False)),
        ar_num_frame_per_block=int(getattr(args, "ar_num_frame_per_block", 3)),
        ar_denoising_step_list=str(getattr(args, "ar_denoising_step_list", "1000,800,600,400,200")),
        ar_same_step_across_blocks=bool(getattr(args, "ar_same_step_across_blocks", False)),
        ar_context_noise=float(getattr(args, "ar_context_noise", 0.0)),
        ar_cache_max_latent_frames=int(getattr(args, "ar_cache_max_latent_frames", 21)),
        ar_loss_last_latent_frames=int(getattr(args, "ar_loss_last_latent_frames", 21)),
    )

    if args.use_dense_branch or args.enable_v2v_plucker_camera_control:
        call_kwargs.update(
            source_cam2world=_t(sample.get("source_cam2world")),
            source_intrinsics=_t(sample.get("source_intrinsics")),
        )

        if args.use_dense_branch:
            call_kwargs.update(
                use_dense_branch=True,
                use_dense_null_token_fallback=bool(getattr(args, "use_dense_null_token_fallback", False)),
                source_rgb_video=_t(sample.get("source_rgb_video")),
                source_depth_valid=_t(sample.get("source_depth_valid", sample.get("source_depth_valid_mask"))),
                source_dynamic_mask=_t(sample.get("source_dynamic_mask")),
                target_dynamic_mask=_t(sample.get("target_dynamic_mask")),
            )
            missing_fields = [
                k for k in (
                    "source_rgb_video",
                    "source_depth_video",
                    "source_cam2world",
                    "source_intrinsics",
                ) if call_kwargs.get(k) is None
            ]
            if missing_fields:
                print(
                    f"[Eval] WARNING: --use_dense_branch is on but the dataset did not "
                    f"emit {missing_fields}. Dense branch will silently no-op."
                )

    # ab27/ab28: the track config may need dense source fields even when the dense
    # branch is OFF — channel-injected reprojected RGB (include_reprojected_rgb) and
    # the per-track dynamic-mask gather both read source video/depth/cameras + the
    # per-pixel dynamic masks. Forward them when pipe.track_include_flags asks, so
    # ab28 eval actually uses reprojected RGB (the pipeline strict-checks otherwise).
    _tflags = getattr(pipe, "track_include_flags", None) or {}
    if _tflags.get("reprojected_rgb") or _tflags.get("source_dynamic_mask"):
        call_kwargs.setdefault("source_cam2world", _t(sample.get("source_cam2world")))
        call_kwargs.setdefault("source_intrinsics", _t(sample.get("source_intrinsics")))
        call_kwargs.setdefault("source_dynamic_mask", _t(sample.get("source_dynamic_mask")))
        call_kwargs.setdefault("target_dynamic_mask", _t(sample.get("target_dynamic_mask")))
        if _tflags.get("reprojected_rgb"):
            call_kwargs.setdefault("source_rgb_video", _t(sample.get("source_rgb_video")))
            call_kwargs.setdefault(
                "source_depth_valid",
                _t(sample.get("source_depth_valid", sample.get("source_depth_valid_mask"))),
            )

    if getattr(args, "use_track_null_token_fallback", False):
        call_kwargs["use_track_null_token_fallback"] = True

    if args.enable_v2v_plucker_camera_control:
        missing_camera_fields = [
            k for k in ("source_cam2world", "source_intrinsics")
            if call_kwargs.get(k) is None
        ]
        if missing_camera_fields:
            raise ValueError(
                "--enable_v2v_plucker_camera_control requires source camera fields "
                f"from the dataset, but got missing {missing_camera_fields}."
            )

    # torch.no_grad() is CRITICAL — without it autograd keeps the positive-cfg
    # branch's activation graph alive while the negative branch runs, effectively
    # doubling peak memory and OOM'ing on 48 GB GPUs for width-concat V2V.
    # Matches the official VideoX-Fun inference pattern at
    # examples/wan2.1_fun/predict_v2v_control_camera.py:261.
    gs_feedback_interval = int(getattr(args, "gs_feedback_interval", 0))
    if gs_feedback_interval > 0:
        # Pass GS feedback interval into the pipeline call so the denoising
        # loop can inject 3D feedback at selected steps.
        call_kwargs["gs_feedback_interval"] = gs_feedback_interval
    # One-time activation log so logs make it obvious which warp path fired
    # (silent multiframe failures are pre-2026-04-29 the #1 fingerprint of a
    # missing-plumbing bug).
    if not getattr(_run_one_pair, "_dense_warp_mode_logged", False):
        print(f"[Eval] dense_warp_mode={call_kwargs['dense_warp_mode']} "
              f"motion_threshold={call_kwargs['dense_warp_motion_threshold']}", flush=True)
        _run_one_pair._dense_warp_mode_logged = True  # type: ignore[attr-defined]
    profile_enabled = bool(getattr(args, "profile_jsonl", None)) and torch.cuda.is_available()
    if profile_enabled:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    pipeline_started = time.perf_counter()
    with torch.no_grad():
        output = pipe(**call_kwargs)
    if profile_enabled:
        torch.cuda.synchronize()
    pipeline_seconds = time.perf_counter() - pipeline_started
    peak_allocated_bytes = (
        int(torch.cuda.max_memory_allocated()) if profile_enabled else None
    )
    peak_reserved_bytes = (
        int(torch.cuda.max_memory_reserved()) if profile_enabled else None
    )
    video = output["video"] if isinstance(output, dict) else output

    comparison_sample = _sample_with_captured_track_selection(
        sample,
        getattr(pipe, "_debug_last_track_identity_selection", None),
    )

    # Latent dump.
    # Writes pred_latents.safetensors (the post-denoising target-view latent
    # tensor, compact bf16) so the user can re-decode
    # to mp4 at any quality / resolution scale via the companion utility
    # `examples/wan2.1_fun/decode_pred_latents.py` without re-running the
    # 50-step denoising loop.
    #
    # Must be rank-0 only — both ranks compute identical latents under FSDP
    # and would race-write the same file.
    # The legacy --capture_final_latents .pt dumps (debug) are still gated
    # by the flag for backwards compat with older debug-flow scripts.
    if is_main_process():
        captured_adapter_inputs = getattr(
            pipe, "_debug_last_track_adapter_inputs", None,
        )
        if captured_adapter_inputs is not None:
            capture_path = pair_out_dir / "track_adapter_inputs.pt"
            torch.save(captured_adapter_inputs, capture_path)
            print(f"[Eval] saved exact track-adapter inputs -> {capture_path}", flush=True)

        final_target_latents = getattr(pipe, "_debug_last_target_latents", None)
        if final_target_latents is not None:
            try:
                save_pred_latents(
                    pair_out_dir / "pred_latents.safetensors",
                    final_target_latents,
                    metadata={
                        # Tags that decode_pred_latents.py reads back. Strings only —
                        # safetensors metadata is str→str.
                        "v2v_concat_mode": str(call_kwargs.get("v2v_concat_mode", "width")),
                        "height": str(call_kwargs.get("height", "")),
                        "width": str(call_kwargs.get("width", "")),
                        "num_frames": str(call_kwargs.get("num_frames", "")),
                        "num_inference_steps": str(call_kwargs.get("num_inference_steps", "")),
                        "cfg_scale": str(call_kwargs.get("cfg_scale", "")),
                        "sigma_shift": str(call_kwargs.get("sigma_shift", "")),
                        "seed": str(call_kwargs.get("seed", "")),
                    },
                )
            except Exception as _exc:
                print(f"[Eval] warn: pred_latents.safetensors save failed: {_exc}")

    # Legacy .pt dumps (gated, debug-only — superseded by pred_latents.safetensors).
    if args.capture_final_latents and is_main_process():
        final_target_latents = getattr(pipe, "_debug_last_target_latents", None)
        if final_target_latents is not None:
            torch.save(final_target_latents, pair_out_dir / "final_target_latents.pt")
        final_decode_latents = getattr(pipe, "_debug_last_decode_latents", None)
        if final_decode_latents is not None:
            torch.save(final_decode_latents, pair_out_dir / "final_decode_latents.pt")

    output_fps = float(sample.get("output_fps", WAN_SAMPLE_FPS))
    if not np.isfinite(output_fps) or output_fps <= 0.0:
        raise ValueError(f"sample output_fps must be finite and positive, got {output_fps}")
    video_save_started = time.perf_counter()
    save_video_frames(video, pair_out_dir / "pred_rgb.mp4", fps=output_fps)
    video_save_seconds = time.perf_counter() - video_save_started
    if getattr(args, "profile_jsonl", None) and is_main_process():
        profile_path = Path(args.profile_jsonl)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_record = {
            "pair_output_dir": str(pair_out_dir),
            "pipeline_seconds": pipeline_seconds,
            "pred_mp4_save_seconds": video_save_seconds,
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "num_inference_steps": int(args.num_inference_steps),
            "cfg_scale": float(args.cfg_scale),
            "sigma_shift": float(args.sigma_shift),
            "seed": int(args.seed),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
        stage_timings = getattr(pipe, "_debug_last_stage_timings", None)
        if stage_timings:
            profile_record.update(stage_timings)
        sample_stage_timings = sample.get("_profile_stage_timings")
        if isinstance(sample_stage_timings, dict):
            profile_record.update(
                {
                    str(key): float(value)
                    for key, value in sample_stage_timings.items()
                }
            )
        with profile_path.open("a") as profile_file:
            profile_file.write(json.dumps(profile_record, sort_keys=True) + "\n")
        print("[Eval/profile] " + json.dumps(profile_record, sort_keys=True), flush=True)
    # A continuation window consumes the exact decoded target endpoint, not a
    # frame re-read from the lossy MP4. Keep this lightweight handoff artifact
    # beside every I2V-conditioned pair so the next window can use it directly.
    if bool(
        sample.get("use_target_image_condition", False)
        or sample.get("save_i2v_boundary", False)
    ) and is_main_process():
        try:
            target_frames = _video_to_uint8_frames(video)
            if not target_frames:
                raise RuntimeError("prediction contains no frames")
            imageio.imwrite(pair_out_dir / "first_target.png", target_frames[0])
            imageio.imwrite(pair_out_dir / "last_target.png", target_frames[-1])
        except Exception as e:
            print(f"[Eval] warn: I2V boundary PNG save failed: {e}")
    try:
        from holo4d.novel_view.static_scene_novel_traj import (
            prediction_filename_for_sample,
        )

        prediction_name = prediction_filename_for_sample(sample)
        if prediction_name != "pred_rgb.mp4":
            save_video_frames(video, pair_out_dir / prediction_name, fps=output_fps)
    except Exception as e:
        print(f"[Eval] warn: trajectory-named mp4 save failed: {e}")

    # Opt-in per-frame PNG dump (off by default). Two triggers:
    #   - args.save_pred_frames_dir: explicit target dir (orbit-chain driver)
    #   - HOLO4D_DUMP_PRED_FRAMES=1: dump under <pair_out_dir>/pred_frames
    pred_frames_dir = getattr(args, "save_pred_frames_dir", None)
    if not pred_frames_dir and os.environ.get("HOLO4D_DUMP_PRED_FRAMES", "0") == "1":
        pred_frames_dir = pair_out_dir / "pred_frames"
    if pred_frames_dir:
        try:
            written = save_video_png_frames(video, Path(pred_frames_dir))
            if written:
                print(f"[Eval] dumped {len(written)} pred frames to {pred_frames_dir}")
        except Exception as e:
            print(f"[Eval] warn: pred frame dump failed: {e}")

    # --- 3DGS rendering (if gs_decoder is available) ---
    # Under FSDP, gs_decoder is replicated (not sharded — see
    # eval_fsdp_helper.py). Running its forward on every rank produces
    # identical outputs that would race-write the same mp4s, so we gate
    # the entire call to rank 0.
    if hasattr(pipe, "gs_decoder") and pipe.gs_decoder is not None and is_main_process():
        try:
            _render_gs_eval(pipe, sample, args, pair_out_dir, pred_video=video)
        except Exception as e:
            import traceback
            print(f"[Eval] warn: GS rendering failed: {e}")
            traceback.print_exc()

        # Mid-trajectory capture render (training-σ snapshot, e.g. t≈242).
        if getattr(pipe, "_gs_features_at_capture", None) is not None:
            try:
                _t_cap = int(round(float(pipe._capture_timestep)))
                _render_gs_eval(
                    pipe, sample, args, pair_out_dir, pred_video=video,
                    gs_features_override=pipe._gs_features_at_capture,
                    latents_override=pipe._x0_hat_at_capture,
                    output_suffix=f"_t{_t_cap}",
                )
                print(f"[Eval/GS] mid-trajectory render saved with suffix _t{_t_cap}")
            except Exception as e:
                import traceback
                print(f"[Eval] warn: GS mid-capture rendering failed: {e}")
                traceback.print_exc()

    # Save a paired target GT when one exists. Novel-view datasets can retain a
    # real-camera image only as a reference; do not mislabel that as gt_rgb.mp4.
    target_video = sample.get("target_video")
    if target_video is not None:
        try:
            target_name = (
                "gt_rgb.mp4"
                if sample.get("target_video_is_ground_truth", True)
                else "target_reference_rgb.mp4"
            )
            save_video_frames(target_video, pair_out_dir / target_name, fps=output_fps)
        except Exception as e:
            print(f"[Eval] warn: could not save target/reference RGB: {e}")

    # Save source video for CLIP-V (view consistency vs source)
    source_video = sample.get("video")
    if source_video is not None:
        try:
            save_video_frames(source_video, pair_out_dir / "source_rgb.mp4", fps=output_fps)
        except Exception as e:
            print(f"[Eval] warn: could not save source_rgb.mp4: {e}")

    # Diagnostic 2x4 comparison.mp4 (src/gt/pred/diff || dense_rgb/depth/valid/tracks).
    # Reuses the in-RAM ``video`` tensor we just saved as pred_rgb.mp4 — no
    # extra inference, no disk re-read.
    # Under FSDP, only rank 0 writes; non-main processes skip this block.
    if getattr(args, "skip_comparison", False) or not is_main_process():
        return
    try:
        if sample.get("omni_droid_novel_view_mode"):
            from holo4d.visualization.omni_droid_panels import (
                write_omni_droid_comparison_mp4 as write_comparison_mp4,
            )
        else:
            from holo4d.visualization.hybrid_panels import (
                write_eval_comparison_mp4 as write_comparison_mp4,
            )

        if isinstance(video, list):
            pred_frames = [np.asarray(f, dtype=np.uint8) for f in video]
        elif torch.is_tensor(video):
            tensor = video.detach().cpu()
            if tensor.ndim == 5:
                tensor = tensor[0]
            tensor = tensor.permute(1, 2, 3, 0).float()
            if float(tensor.min()) < 0.0:
                tensor = (tensor + 1.0) / 2.0
            tensor = tensor.clamp(0.0, 1.0)
            pred_frames = [(f.numpy() * 255.0).astype(np.uint8) for f in tensor]
        else:
            pred_frames = None

        if pred_frames is not None:
            tag = (
                f"{pair_out_dir.parent.parent.name}/"
                f"{pair_out_dir.parent.name}/{pair_out_dir.name}"
            )
            comparison_path = pair_out_dir / "comparison.mp4"
            selection_meta = comparison_sample.get("track_visualization_selection")
            if selection_meta is not None:
                tag += (
                    f" | tracks={selection_meta['mode']} "
                    f"K={selection_meta['selected_count']}/{selection_meta['source_identity_count']}"
                )
            write_comparison_mp4(
                comparison_path,
                comparison_sample,
                pred_frames,
                fps=output_fps,
                caption_extra=tag,
                dense_warp_mode=getattr(args, "dense_warp_mode", "zbuffer"),
                dense_warp_motion_threshold=float(getattr(args, "dense_warp_motion_threshold", 0.005)),
            )
            try:
                from holo4d.novel_view.static_scene_novel_traj import (
                    comparison_filename_for_sample,
                )

                comparison_name = comparison_filename_for_sample(sample)
                if comparison_name != "comparison.mp4":
                    write_comparison_mp4(
                        pair_out_dir / comparison_name,
                        comparison_sample,
                        pred_frames,
                        fps=output_fps,
                        caption_extra=tag,
                        dense_warp_mode=getattr(args, "dense_warp_mode", "zbuffer"),
                        dense_warp_motion_threshold=float(getattr(args, "dense_warp_motion_threshold", 0.005)),
                    )
            except Exception as e:
                print(f"[Eval] warn: trajectory-named comparison mp4 save failed: {e}")
    except Exception as e:
        print(f"[Eval] warn: comparison.mp4 viz failed: {e}")

    # sample_meta.npz removed — strict subset of novel_view_payload.npz, and
    # everything in it is deterministic from ${DAVIS_4RC_ROOT}/<seq>/result.npz
    # plus the trajectory CLI args (angle_deg / distance / orbit_radius). Cache
    # was ~57 MB/pair (≈ 5.7 GB / 100-pair benchmark) for zero functional gain.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_i2v_continuation_dataset(args) -> Syn4D_Multi_NVS:
    """Build the held-out Syn4D dataset used by the autoregressive I2V check.

    ``flying_group`` is excluded by both balanced-q2 training configs.  Keep
    the dataset construction explicit here instead of relying on the normal
    one-sequence direct-eval helper, because this protocol samples cases from
    all held-out sequences and needs the source-camera fields for V2V rays.
    """
    demo_path = os.environ.get("HOLO4D_DEMO_BUNDLE_PATH")
    if demo_path:
        from holo4d.data.demo_rollout_bundle import DemoRolloutBundle
        return DemoRolloutBundle(demo_path, sequence_idx=0, num_frames=args.num_frames)
    return Syn4D_Multi_NVS(
        ROOT=args.syn4d_root,
        num_views=args.num_frames,
        resolution=(args.width, args.height),
        steps_per_epoch=1,
        track_spatial_stride=args.track_spatial_stride,
        scene="flying_group",
        valid_sequences_file=args.syn4d_valid_sequences_file,
        caption_dir=args.syn4d_caption_dir,
        use_mp4_rgb=True,
        return_source_depth_video=True,
        return_source_rgb_video=False,
        return_source_cameras=True,
        return_dynamic_masks=False,
    )


def _enumerate_i2v_continuation_cases(dataset: Syn4D_Multi_NVS, num_frames: int):
    """Enumerate exact-boundary two-window Syn4D continuation cases.

    The seed window is ``[s, s + 48]``. The scored window starts at exactly
    ``s + 48``, so the target pixel at its frame 0 is the preceding model
    output's final frame. Syn4D stores tracking anchors every five frames; the
    scored window therefore uses the earliest in-window GT anchor (usually
    local frame 2) rather than shifting the window away from the boundary.
    """
    boundary_offset = int(num_frames) - 1
    cases = []
    for group_idx, group in enumerate(dataset._nvs_groups):
        common_starts = set(int(v) for v in group["common_starts"])
        for seed_start in sorted(common_starts):
            continuation_start = seed_start + boundary_offset
            query_idx = next(
                (
                    local_idx
                    for local_idx in range(int(num_frames))
                    if continuation_start + local_idx in common_starts
                ),
                None,
            )
            if query_idx is None:
                continue
            # The continuation anchor itself is a complete 49-frame Syn4D
            # sample, so this also proves the continuation window fits.
            anchor = continuation_start + query_idx
            for src_cam_idx in range(len(group["camera_names"])):
                for tgt_cam_idx in range(len(group["camera_names"])):
                    if src_cam_idx == tgt_cam_idx:
                        continue
                    cases.append(
                        {
                            "group_idx": group_idx,
                            "scene_name": group["scene_name"],
                            "base_seq": group["base_seq"],
                            "src_cam_idx": src_cam_idx,
                            "tgt_cam_idx": tgt_cam_idx,
                            "seed_start_frame": seed_start,
                            "continuation_start_frame": continuation_start,
                            "continuation_track_anchor_frame": anchor,
                            "continuation_track_query_idx": query_idx,
                        }
                    )
    return cases


def _select_i2v_continuation_cases(dataset: Syn4D_Multi_NVS, args):
    cases = _enumerate_i2v_continuation_cases(dataset, args.num_frames)
    fixed_sequences = getattr(args, "i2v_fixed_source_sequences", None)
    if fixed_sequences:
        allowed = {item.strip() for item in str(fixed_sequences).split(",") if item.strip()}
        cases = [
            case for case in cases
            if case["base_seq"] in allowed
            and int(case["src_cam_idx"]) == 0
            and int(case["seed_start_frame"]) == int(getattr(args, "i2v_fixed_seed_start_frame", 5))
            and int(case["tgt_cam_idx"]) > 0
            and (case["base_seq"] != "seq_000004" or int(case["tgt_cam_idx"]) <= 4)
        ]
        cases.sort(key=lambda case: (case["base_seq"], int(case["tgt_cam_idx"])))
        requested = int(args.i2v_continuation_pairs)
        if requested <= 0 or requested > len(cases):
            requested = len(cases)
        return cases[:requested]
    requested = int(args.i2v_continuation_pairs)
    if len(cases) < requested:
        raise RuntimeError(
            f"Syn4D held-out continuation has only {len(cases)} valid ordered cases; "
            f"requested {requested}"
        )
    rng = np.random.default_rng(int(args.i2v_continuation_seed))
    selected_indices = sorted(rng.choice(len(cases), size=requested, replace=False).tolist())
    return [cases[index] for index in selected_indices]


def _i2v_checkpoint_path(args, role: str) -> Path:
    """Resolve the checkpoint used by one stage of the I2V protocol."""
    if role == "seed":
        configured = getattr(args, "i2v_seed_checkpoint_path", None)
    elif role == "continuation":
        configured = getattr(args, "i2v_continuation_checkpoint_path", None)
    else:
        raise ValueError(f"Unknown I2V checkpoint role: {role}")
    return Path(configured or args.checkpoint_path).expanduser()


def _i2v_checkpoint_provenance(checkpoint_path: Path, role: str) -> dict:
    """Return lightweight identity data without hashing a multi-GB checkpoint."""
    path = checkpoint_path.expanduser()
    size_bytes = path.stat().st_size if path.is_file() else None
    return {
        "role": role,
        "checkpoint_path": str(path),
        "checkpoint_resolved_path": str(path.resolve(strict=False)),
        "checkpoint_name": path.name,
        "checkpoint_size_bytes": size_bytes,
    }


def _write_i2v_checkpoint_provenance(stage_dir: Path, checkpoint_path: Path, role: str) -> dict:
    """Record the exact checkpoint that produced a seed/continuation artifact."""
    payload = _i2v_checkpoint_provenance(checkpoint_path, role)
    if is_main_process():
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "checkpoint_provenance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return payload


def _i2v_checkpoint_provenance_matches(
    stage_dir: Path, checkpoint_path: Path, role: str
) -> bool:
    """Check that a resumable artifact belongs to the requested checkpoint."""
    marker_path = stage_dir / "checkpoint_provenance.json"
    if not marker_path.is_file():
        return False
    try:
        actual = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = _i2v_checkpoint_provenance(checkpoint_path, role)
    return (
        actual.get("role") == expected["role"]
        and actual.get("checkpoint_resolved_path") == expected["checkpoint_resolved_path"]
        and actual.get("checkpoint_size_bytes") == expected["checkpoint_size_bytes"]
    )


def _sync_i2v_processes():
    """Synchronize ranks when a rank-0 marker is written between stages."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _write_i2v_continuation_manifest(output_root: Path, selected_cases, args):
    shard_index = int(args.i2v_continuation_shard_index)
    shard_count = int(args.i2v_continuation_num_shards)
    payload = {
        "protocol": "syn4d_gt_tracks_i2v_exact_boundary_v1",
        "scene": "flying_group",
        "held_out_from_training": True,
        "seed_bootstrap": "source-video-only T2V seed; not scored",
        "scored_continuation": "previous generated target frame 48 -> next target frame 0",
        "tracks": "Syn4D GT 3D tracks; continuation query uses first available in-window GT anchor",
        "seed_checkpoint_path": str(_i2v_checkpoint_path(args, "seed")),
        "continuation_checkpoint_path": str(_i2v_checkpoint_path(args, "continuation")),
        "checkpoint_protocol": (
            "source-only T2V seed uses the seed checkpoint; the generated target "
            "boundary seeds continuation with the I2V continuation checkpoint"
        ),
        "selection_seed": int(args.i2v_continuation_seed),
        "num_requested": int(args.i2v_continuation_pairs),
        "shard_index": shard_index,
        "num_shards": shard_count,
        "cases": [
            {"global_index": index, **case}
            for index, case in enumerate(selected_cases)
            if index % shard_count == shard_index
        ],
    }
    path = output_root / f"continuation_manifest_shard{shard_index}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_i2v_continuation_context_artifacts(
    *,
    continuation_dir: Path,
    seed_sample: dict,
    continuation_sample: dict,
    previous_target_path: Path,
    case: dict,
) -> dict:
    """Save explicit source/boundary assets for one generated-target handoff.

    ``previous_source_frame48.png`` is a useful temporal reference but is not
    the I2V condition.  The actual I2V condition is the generated target frame
    saved as ``previous_generated_target_frame48.png``.  Keeping both avoids
    the common source-versus-target boundary ambiguity during visual review.
    """
    continuation_dir.mkdir(parents=True, exist_ok=True)
    previous_source = _video_to_uint8_frames(seed_sample["video"])
    current_source = _video_to_uint8_frames(continuation_sample["video"])
    previous_target_gt = _video_to_uint8_frames(seed_sample["target_video"])
    current_target_gt = _video_to_uint8_frames(continuation_sample["target_video"])
    if not all((previous_source, current_source, previous_target_gt, current_target_gt)):
        raise RuntimeError("I2V continuation source/target chunks contain no RGB frames")
    save_video_frames(
        previous_source,
        continuation_dir / "previous_source_rgb.mp4",
        fps=WAN_SAMPLE_FPS,
    )
    save_video_frames(
        current_source,
        continuation_dir / "current_source_rgb.mp4",
        fps=WAN_SAMPLE_FPS,
    )
    imageio.imwrite(
        continuation_dir / "previous_source_frame48.png", previous_source[-1]
    )
    imageio.imwrite(
        continuation_dir / "current_source_frame0.png", current_source[0]
    )
    previous_target_gt_boundary = _frame_to_uint8_rgb(previous_target_gt[-1])
    current_target_gt_boundary = _frame_to_uint8_rgb(current_target_gt[0])
    imageio.imwrite(
        continuation_dir / "previous_target_gt_frame48.png",
        previous_target_gt_boundary,
    )
    imageio.imwrite(
        continuation_dir / "current_target_gt_frame0.png",
        current_target_gt_boundary,
    )
    boundary = _as_i2v_boundary_image(imageio.imread(previous_target_path))
    generated_target_condition = np.asarray(boundary, dtype=np.uint8)
    imageio.imwrite(
        continuation_dir / "previous_generated_target_frame48.png",
        generated_target_condition,
    )
    prediction_first_path = continuation_dir / "first_target.png"
    prediction_first = None
    if prediction_first_path.is_file():
        prediction_first = _frame_to_uint8_rgb(imageio.imread(prediction_first_path))
    elif (continuation_dir / "pred_rgb.mp4").is_file():
        prediction_first = _read_first_video_frame(continuation_dir / "pred_rgb.mp4")
        imageio.imwrite(prediction_first_path, prediction_first)
    if prediction_first is not None:
        imageio.imwrite(
            continuation_dir / "continuation_pred_frame0.png", prediction_first
        )
        _write_i2v_boundary_contact_sheet(
            continuation_dir / "boundary_context.png",
            previous_source=previous_source[-1],
            previous_target_gt=previous_target_gt_boundary,
            generated_target_condition=generated_target_condition,
            current_source=current_source[0],
            current_target_gt=current_target_gt_boundary,
            continuation_prediction=prediction_first,
        )
    if previous_target_gt_boundary.shape != current_target_gt_boundary.shape:
        raise ValueError(
            "Previous/current target GT boundary frames have different shapes: "
            f"{previous_target_gt_boundary.shape} vs {current_target_gt_boundary.shape}"
        )
    target_boundary_diff = np.abs(
        previous_target_gt_boundary.astype(np.int16)
        - current_target_gt_boundary.astype(np.int16)
    )
    target_boundary_max_abs_diff = int(target_boundary_diff.max())
    payload = {
        "protocol": "generated target frame 48 conditions next target window",
        "previous_source_video": "previous_source_rgb.mp4",
        "current_source_video": "current_source_rgb.mp4",
        "previous_source_last_frame": "previous_source_frame48.png",
        "current_source_first_frame": "current_source_frame0.png",
        "previous_target_gt_last_frame": "previous_target_gt_frame48.png",
        "current_target_gt_first_frame": "current_target_gt_frame0.png",
        "actual_i2v_condition": "previous_generated_target_frame48.png",
        "continuation_prediction_first_frame": (
            "continuation_pred_frame0.png" if prediction_first is not None else None
        ),
        "boundary_contact_sheet": (
            "boundary_context.png" if prediction_first is not None else None
        ),
        "condition_is_previous_source_last_frame": False,
        "target_gt_boundary_identical": target_boundary_max_abs_diff == 0,
        "target_gt_boundary_max_abs_diff": target_boundary_max_abs_diff,
        "seed_source_absolute_frames": [
            int(case["seed_start_frame"]),
            int(case["seed_start_frame"]) + 48,
        ],
        "continuation_source_absolute_frames": [
            int(case["continuation_start_frame"]),
            int(case["continuation_start_frame"]) + 48,
        ],
        "boundary_absolute_frame": int(case["continuation_start_frame"]),
        "continuation_track_query_idx": int(case["continuation_track_query_idx"]),
        "continuation_track_anchor_frame": int(case["continuation_track_anchor_frame"]),
    }
    (continuation_dir / "i2v_context.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload



def _long_horizon_case_plan(dataset: Syn4D_Multi_NVS, args) -> tuple[dict, list[dict]]:
    """Build one exact source-window plan for a resumable AR rollout.

    ``plan_long_horizon_chunks`` uses the same 49-frame/one-frame-overlap
    contract as the model.  We deliberately reject a chunk whose source
    window is not present instead of padding or repeating source frames: a
    long video must remain a valid source-conditioned Syn4D evaluation.
    """
    sequence_index = int(args.i2v_long_horizon_sequence_index)
    expected_seq = f"seq_{sequence_index:06d}"
    matches = [
        (group_idx, group)
        for group_idx, group in enumerate(dataset._nvs_groups)
        if str(group.get("base_seq")) == expected_seq
    ]
    if not matches:
        raise RuntimeError(
            f"No flying_group sequence {expected_seq} in Syn4D dataset; "
            f"available={[group.get('base_seq') for group in dataset._nvs_groups]}"
        )
    group_idx, group = matches[0]
    src_cam_idx = int(args.i2v_long_horizon_src_cam_idx)
    tgt_cam_idx = int(args.i2v_long_horizon_tgt_cam_idx)
    camera_count = len(group["camera_names"])
    if not (0 <= src_cam_idx < camera_count and 0 <= tgt_cam_idx < camera_count):
        raise ValueError(
            f"camera indices {src_cam_idx}->{tgt_cam_idx} outside [0,{camera_count})"
        )
    if src_cam_idx == tgt_cam_idx:
        raise ValueError("long-horizon source and target cameras must differ")

    seed_start = int(args.i2v_long_horizon_seed_start_frame)
    total_frames = int(args.i2v_long_horizon_frames)
    if total_frames <= 0:
        raise ValueError("--i2v_long_horizon_frames must be positive")
    common_starts = {int(value) for value in group["common_starts"]}
    if seed_start not in common_starts:
        raise ValueError(
            f"seed start frame {seed_start} is not a common Syn4D window start"
        )
    planned = plan_long_horizon_chunks(total_frames, chunk_frames=int(args.num_frames))
    chunks = []
    for chunk_index, window in enumerate(planned):
        absolute_start = seed_start + int(window.start)
        if chunk_index == 0:
            query_idx = 0
        else:
            query_idx = next(
                (
                    local_idx
                    for local_idx in range(int(args.num_frames))
                    if absolute_start + local_idx in common_starts
                ),
                None,
            )
            if query_idx is None:
                raise RuntimeError(
                    f"No in-window Syn4D track anchor for chunk {chunk_index} "
                    f"starting at absolute frame {absolute_start}"
                )
        anchor = absolute_start + int(query_idx)
        # load_pair performs the authoritative bounds check for all camera
        # arrays, including source RGB/depth and target trajectory metadata.
        dataset.load_pair(
            src_cam_idx=src_cam_idx,
            tgt_cam_idx=tgt_cam_idx,
            start_frame=anchor,
            window_start_frame=absolute_start if chunk_index else None,
            group_idx=group_idx,
            track_query_idx=int(query_idx),
        )
        chunks.append(
            {
                "chunk_index": chunk_index,
                "absolute_start_frame": absolute_start,
                "valid_frames": int(window.valid_frames),
                "window_frames": int(args.num_frames),
                "append_from": int(window.append_from),
                "track_query_idx": int(query_idx),
                "track_anchor_frame": anchor,
                "stage": "seed" if chunk_index == 0 else "continuation",
            }
        )
    case = {
        "group_idx": int(group_idx),
        "scene_name": group["scene_name"],
        "base_seq": group["base_seq"],
        "src_cam_idx": src_cam_idx,
        "tgt_cam_idx": tgt_cam_idx,
        "seed_start_frame": seed_start,
        "continuation_start_frame": chunks[1]["absolute_start_frame"] if len(chunks) > 1 else None,
        "protocol": "source-only T2V seed plus generated-target I2V autoregressive rollout",
    }
    return case, chunks


def _inject_i2v_long_horizon_source_condition(sample: dict, args, case: dict, chunk_index: int) -> dict:
    root = getattr(args, "i2v_long_horizon_source_condition_root", None)
    backend = getattr(args, "i2v_long_horizon_source_condition_backend", None)
    if bool(root) != bool(backend):
        raise ValueError(
            "--i2v_long_horizon_source_condition_root and backend must be set together"
        )
    if not root:
        return sample
    sequence = f"{case['base_seq']}_chunk_{int(chunk_index):03d}"
    path = canonical_path(root, backend, sequence)
    condition = load_canonical_condition(path, expected_backend=backend)
    condition = dict(condition)
    condition["metadata"] = dict(condition["metadata"])
    reference_key = "target_relative_translation_reference_norm"
    if int(chunk_index) > 0 and reference_key in case:
        condition["metadata"][reference_key] = float(case[reference_key])
    result = inject_canonical_condition(sample, condition)
    if reference_key in result:
        if reference_key in case and not np.isclose(
            float(case[reference_key]), float(result[reference_key]), rtol=1e-6
        ):
            raise ValueError(
                "long-horizon target translation reference changed across chunks: "
                f"{case[reference_key]} vs {result[reference_key]}"
            )
        case[reference_key] = float(result[reference_key])
    result["source_condition_metadata"] = condition["metadata"]
    print(
        f"[I2V/long] chunk {chunk_index}: predicted source condition {path} "
        f"tracks={condition['tracks_xyz'].shape}",
        flush=True,
    )
    return result


def _long_horizon_chunk_dir(case_dir: Path, chunk_index: int) -> Path:
    if int(chunk_index) == 0:
        return case_dir / "seed"
    if int(chunk_index) == 1:
        return case_dir / "continuation"
    return case_dir / f"continuation_{int(chunk_index)}"


def _write_i2v_long_chunk_manifest(
    chunk_dir: Path,
    *,
    chunk: dict,
    checkpoint_path: Path,
    case: dict,
) -> dict:
    payload = {
        "protocol": "syn4d_generated_target_i2v_long_horizon_v1",
        "case": case["base_seq"],
        "chunk_index": int(chunk["chunk_index"]),
        "stage": chunk["stage"],
        "absolute_start_frame": int(chunk["absolute_start_frame"]),
        "valid_frames": int(chunk["valid_frames"]),
        "window_frames": int(chunk["window_frames"]),
        "append_from": int(chunk["append_from"]),
        "track_query_idx": int(chunk["track_query_idx"]),
        "track_anchor_frame": int(chunk["track_anchor_frame"]),
        "checkpoint": _i2v_checkpoint_provenance(checkpoint_path, chunk["stage"]),
        "source_camera": int(case["src_cam_idx"]),
        "target_camera": int(case["tgt_cam_idx"]),
    }
    if is_main_process():
        chunk_dir.mkdir(parents=True, exist_ok=True)
        (chunk_dir / "chunk_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return payload


def _run_i2v_long_horizon_eval(pipe, args, output_root: Path, stage: str = "both"):
    """Run/resume a multi-chunk T2V-seed -> I2V autoregressive rollout."""
    if stage not in {"seed", "continuation", "both"}:
        raise ValueError(f"Unknown long-horizon stage: {stage}")
    dataset = _build_i2v_continuation_dataset(args)
    case, chunks = _long_horizon_case_plan(dataset, args)
    case_index = int(getattr(args, "i2v_long_horizon_case_index", 0))
    if case_index < 0:
        raise ValueError(f"i2v_long_horizon_case_index must be non-negative, got {case_index}")
    case_tag = (
        f"case_{case_index:03d}_{case['base_seq']}_src{case['src_cam_idx']}_"
        f"tgt{case['tgt_cam_idx']}_seed{case['seed_start_frame']:04d}_"
        f"next{chunks[1]['absolute_start_frame']:04d}"
    )
    case_dir = output_root / "flying_group" / case_tag
    case_dir.mkdir(parents=True, exist_ok=True)
    seed_checkpoint_path = _i2v_checkpoint_path(args, "seed")
    continuation_checkpoint_path = _i2v_checkpoint_path(args, "continuation")
    long_manifest = {
        "protocol": "source-only T2V seed; generated-target I2V continuation chunks",
        "base_seq": case["base_seq"],
        "source_camera": int(case["src_cam_idx"]),
        "target_camera": int(case["tgt_cam_idx"]),
        "seed_start_frame": int(case["seed_start_frame"]),
        "requested_frames": int(args.i2v_long_horizon_frames),
        "planned_chunks": chunks,
        "seed_checkpoint_path": str(seed_checkpoint_path),
        "continuation_checkpoint_path": str(continuation_checkpoint_path),
    }
    if is_main_process():
        (case_dir / "long_horizon_manifest.json").write_text(
            json.dumps(long_manifest, indent=2, sort_keys=True) + "\n"
        )
        case_json = case_dir / "case.json"
        existing = json.loads(case_json.read_text()) if case_json.is_file() else {}
        case_json.write_text(
            json.dumps({**existing, **case, "long_horizon": long_manifest}, indent=2, sort_keys=True)
            + "\n"
        )

    def chunk_ready(chunk_dir: Path, checkpoint: Path, role: str) -> bool:
        return (
            (chunk_dir / "pred_rgb.mp4").is_file()
            and (chunk_dir / "last_target.png").is_file()
            and _i2v_checkpoint_provenance_matches(chunk_dir, checkpoint, role)
        )

    if stage in {"seed", "both"}:
        chunk = chunks[0]
        seed_dir = _long_horizon_chunk_dir(case_dir, 0)
        if chunk_ready(seed_dir, seed_checkpoint_path, "seed") and not args.force_rerun:
            print(f"[I2V/long] skip seed: {seed_dir}", flush=True)
        else:
            sample = dataset.load_pair(
                src_cam_idx=case["src_cam_idx"],
                tgt_cam_idx=case["tgt_cam_idx"],
                start_frame=chunk["absolute_start_frame"],
                group_idx=case["group_idx"],
                track_query_idx=0,
            )
            sample = _inject_i2v_long_horizon_source_condition(sample, args, case, 0)
            sample["use_target_image_condition"] = False
            sample["save_i2v_boundary"] = True
            sample["inference_target_video"] = None
            _run_one_pair(pipe, sample, args, seed_dir)
            _write_i2v_checkpoint_provenance(seed_dir, seed_checkpoint_path, "seed")
        _write_i2v_long_chunk_manifest(seed_dir, chunk=chunk, checkpoint_path=seed_checkpoint_path, case=case)
        if stage == "seed":
            _sync_i2v_processes()
            return

    if not (case_dir / "seed" / "last_target.png").is_file():
        raise RuntimeError("long-horizon seed is missing its last_target.png")

    completed = skipped = 0
    for chunk in chunks[1:]:
        index = int(chunk["chunk_index"])
        previous_dir = _long_horizon_chunk_dir(case_dir, index - 1)
        current_dir = _long_horizon_chunk_dir(case_dir, index)
        prediction = current_dir / "pred_rgb.mp4"
        if chunk_ready(current_dir, continuation_checkpoint_path, "continuation") and not args.force_rerun:
            print(f"[I2V/long] skip chunk {index}: {current_dir}", flush=True)
            _write_i2v_long_chunk_manifest(
                current_dir, chunk=chunk, checkpoint_path=continuation_checkpoint_path, case=case
            )
            skipped += 1
            continue
        previous_target = previous_dir / "last_target.png"
        if not previous_target.is_file():
            raise RuntimeError(f"chunk {index} cannot find generated boundary {previous_target}")
        previous_chunk = chunks[index - 1]
        previous_sample = dataset.load_pair(
            src_cam_idx=case["src_cam_idx"],
            tgt_cam_idx=case["tgt_cam_idx"],
            start_frame=previous_chunk["track_anchor_frame"],
            window_start_frame=previous_chunk["absolute_start_frame"] if index - 1 else None,
            group_idx=case["group_idx"],
            track_query_idx=previous_chunk["track_query_idx"],
        )
        previous_sample = _inject_i2v_long_horizon_source_condition(
            previous_sample, args, case, index - 1
        )
        current_sample = dataset.load_pair(
            src_cam_idx=case["src_cam_idx"],
            tgt_cam_idx=case["tgt_cam_idx"],
            start_frame=chunk["track_anchor_frame"],
            window_start_frame=chunk["absolute_start_frame"],
            group_idx=case["group_idx"],
            track_query_idx=chunk["track_query_idx"],
        )
        current_sample = _inject_i2v_long_horizon_source_condition(
            current_sample, args, case, index
        )
        current_sample["inference_input_image"] = _as_i2v_boundary_image(
            imageio.imread(previous_target)
        )
        current_sample["use_target_image_condition"] = True
        current_sample["inference_target_video"] = None
        handoff_case = {
            "seed_start_frame": previous_chunk["absolute_start_frame"],
            "continuation_start_frame": chunk["absolute_start_frame"],
            "continuation_track_query_idx": chunk["track_query_idx"],
            "continuation_track_anchor_frame": chunk["track_anchor_frame"],
        }
        _write_i2v_continuation_context_artifacts(
            continuation_dir=current_dir,
            seed_sample=previous_sample,
            continuation_sample=current_sample,
            previous_target_path=previous_target,
            case=handoff_case,
        )
        print(
            f"[I2V/long] chunk {index}/{len(chunks)-1}: "
            f"source [{chunk['absolute_start_frame']}, "
            f"{chunk['absolute_start_frame'] + chunk['window_frames'] - 1}] "
            f"condition={previous_target}",
            flush=True,
        )
        _run_one_pair(pipe, current_sample, args, current_dir)
        _write_i2v_continuation_context_artifacts(
            continuation_dir=current_dir,
            seed_sample=previous_sample,
            continuation_sample=current_sample,
            previous_target_path=previous_target,
            case=handoff_case,
        )
        _write_i2v_checkpoint_provenance(current_dir, continuation_checkpoint_path, "continuation")
        _write_i2v_long_chunk_manifest(
            current_dir, chunk=chunk, checkpoint_path=continuation_checkpoint_path, case=case
        )
        completed += 1
    _sync_i2v_processes()
    print(
        f"[I2V/long] done: chunks={len(chunks)} completed={completed} skipped={skipped} "
        f"output={case_dir}",
        flush=True,
    )

def _run_i2v_continuation_eval(pipe, args, output_root: Path, stage: str = "both"):
    """Evaluate one or both stages of the held-out Syn4D I2V protocol.

    ``stage=seed`` creates the source-video-only T2V first chunk, while
    ``stage=continuation`` consumes that chunk's generated target boundary.
    Keeping the stages separate lets the caller load a different checkpoint
    for each stage without keeping two 14B pipelines resident at once.
    """
    if stage not in {"seed", "continuation", "both"}:
        raise ValueError(f"Unknown I2V evaluation stage: {stage}")
    shard_index = int(args.i2v_continuation_shard_index)
    shard_count = int(args.i2v_continuation_num_shards)
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"invalid continuation shard {shard_index}/{shard_count}"
        )

    dataset = _build_i2v_continuation_dataset(args)
    selected_cases = _select_i2v_continuation_cases(dataset, args)
    manifest_path = _write_i2v_continuation_manifest(output_root, selected_cases, args)
    seed_checkpoint_path = _i2v_checkpoint_path(args, "seed")
    continuation_checkpoint_path = _i2v_checkpoint_path(args, "continuation")
    shard_cases = [
        (index, case)
        for index, case in enumerate(selected_cases)
        if index % shard_count == shard_index
    ]
    print(
        f"[I2V/Syn4D] selected={len(selected_cases)} shard={shard_index}/{shard_count} "
        f"cases={len(shard_cases)} stage={stage} manifest={manifest_path}",
        flush=True,
    )
    print(
        f"[I2V/Syn4D] seed_checkpoint={seed_checkpoint_path} "
        f"continuation_checkpoint={continuation_checkpoint_path}",
        flush=True,
    )

    completed = 0
    skipped = 0
    for case_index, case in shard_cases:
        case_tag = (
            f"case_{case_index:03d}_{case['base_seq']}_"
            f"src{case['src_cam_idx']}_tgt{case['tgt_cam_idx']}_"
            f"seed{case['seed_start_frame']:04d}_next{case['continuation_start_frame']:04d}"
        )
        case_dir = output_root / "flying_group" / case_tag
        seed_dir = case_dir / "seed"
        continuation_dir = case_dir / "continuation"
        continuation_pred = continuation_dir / "pred_rgb.mp4"
        print(
            f"[I2V/Syn4D] case {case_index}: {case['base_seq']} "
            f"src={case['src_cam_idx']} tgt={case['tgt_cam_idx']} "
            f"seed={case['seed_start_frame']} boundary={case['continuation_start_frame']} "
            f"q={case['continuation_track_query_idx']}",
            flush=True,
        )

        seed_sample = None
        seed_ready = (
            (seed_dir / "last_target.png").is_file()
            and _i2v_checkpoint_provenance_matches(
                seed_dir, seed_checkpoint_path, "seed"
            )
        )
        if stage in {"seed", "both"}:
            if seed_ready and not args.force_rerun:
                print(
                    f"[I2V/Syn4D] skip {case_tag}: seed matches {seed_checkpoint_path}",
                    flush=True,
                )
            else:
                seed_sample = dataset.load_pair(
                    src_cam_idx=case["src_cam_idx"],
                    tgt_cam_idx=case["tgt_cam_idx"],
                    start_frame=case["seed_start_frame"],
                    group_idx=case["group_idx"],
                    track_query_idx=0,
                )
                # The original step-19500 model is T2V/V2V: Syn4D NVS only
                # supplies the source video for this seed window. Save its
                # generated target endpoint for the later I2V handoff, but do
                # not pass GT target RGB as a condition (it does not exist in
                # the NVS input contract).
                seed_sample["use_target_image_condition"] = False
                seed_sample["save_i2v_boundary"] = True
                seed_sample["inference_target_video"] = None
                print(
                    f"[I2V/Syn4D] source-only T2V seed with {seed_checkpoint_path}",
                    flush=True,
                )
                _run_one_pair(pipe, seed_sample, args, seed_dir)
                _write_i2v_checkpoint_provenance(
                    seed_dir, seed_checkpoint_path, "seed"
                )
                seed_ready = True

            if stage == "seed":
                completed += 1
                continue

        if not seed_ready:
            raise RuntimeError(
                f"{case_tag} has no seed artifact produced by {seed_checkpoint_path}; "
                "run the seed stage first"
            )

        previous_target = seed_dir / "last_target.png"
        if not previous_target.is_file():
            raise RuntimeError(f"seed rollout did not produce {previous_target}")
        continuation_ready = (
            continuation_pred.is_file()
            and _i2v_checkpoint_provenance_matches(
                continuation_dir,
                continuation_checkpoint_path,
                "continuation",
            )
        )
        if continuation_ready and not args.force_rerun:
            print(
                f"[I2V/Syn4D] skip {case_tag}: continuation matches "
                f"{continuation_checkpoint_path}",
                flush=True,
            )
            skipped += 1
            continue

        if seed_sample is None:
            seed_sample = dataset.load_pair(
                src_cam_idx=case["src_cam_idx"],
                tgt_cam_idx=case["tgt_cam_idx"],
                start_frame=case["seed_start_frame"],
                group_idx=case["group_idx"],
                track_query_idx=0,
            )
        if bool(getattr(args, "i2v_continuation_gt_boundary", False)):
            # DIAGNOSTIC path: teacher-force the GT boundary the way training
            # does, instead of the seed window's own (drifted) generation.
            _gt_prev_frames = _video_to_uint8_frames(seed_sample["target_video"])
            if not _gt_prev_frames:
                raise RuntimeError(
                    "--i2v_continuation_gt_boundary requires the seed sample's "
                    "target_video, but it decoded to zero frames"
                )
            boundary_image = _as_i2v_boundary_image(
                _frame_to_uint8_rgb(_gt_prev_frames[-1])
            )
        else:
            boundary_image = _as_i2v_boundary_image(imageio.imread(previous_target))

        continuation_sample = dataset.load_pair(
            src_cam_idx=case["src_cam_idx"],
            tgt_cam_idx=case["tgt_cam_idx"],
            start_frame=case["continuation_track_anchor_frame"],
            window_start_frame=case["continuation_start_frame"],
            group_idx=case["group_idx"],
            track_query_idx=case["continuation_track_query_idx"],
        )
        continuation_sample["inference_input_image"] = boundary_image
        continuation_sample["use_target_image_condition"] = True
        # Keep target_video only for gt_rgb.mp4/comparison.mp4. The pipeline
        # must start from noise in target latent space for this NVS rollout.
        continuation_sample["inference_target_video"] = None
        context = _write_i2v_continuation_context_artifacts(
            continuation_dir=continuation_dir,
            seed_sample=seed_sample,
            continuation_sample=continuation_sample,
            previous_target_path=previous_target,
            case=case,
        )
        print(
            f"[I2V/Syn4D] continuation with {continuation_checkpoint_path}",
            flush=True,
        )
        _run_one_pair(pipe, continuation_sample, args, continuation_dir)
        context = _write_i2v_continuation_context_artifacts(
            continuation_dir=continuation_dir,
            seed_sample=seed_sample,
            continuation_sample=continuation_sample,
            previous_target_path=previous_target,
            case=case,
        )
        continuation_provenance = _write_i2v_checkpoint_provenance(
            continuation_dir, continuation_checkpoint_path, "continuation"
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process():
            (case_dir / "case.json").write_text(
                json.dumps(
                    {
                        **case,
                        "protocol": "source-only T2V seed; scored generated-target I2V continuation",
                        "seed_checkpoint_path": str(seed_checkpoint_path),
                        "continuation_checkpoint_path": str(continuation_checkpoint_path),
                        "seed_checkpoint_provenance": _i2v_checkpoint_provenance(
                            seed_checkpoint_path, "seed"
                        ),
                        "continuation_checkpoint_provenance": continuation_provenance,
                        "boundary_png": str(previous_target),
                        "continuation_pred": str(continuation_pred),
                        "continuation_gt": str(continuation_dir / "gt_rgb.mp4"),
                        "continuation_context": context,
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n"
            )
        completed += 1

    # Synchronize once after the complete shard, rather than after each case:
    # modulo sharding gives different ranks different case counts.
    _sync_i2v_processes()
    print(
        f"[I2V/Syn4D] done: completed={completed} skipped={skipped} "
        f"output={output_root}",
        flush=True,
    )

def _release_eval_pipeline(pipe):
    """Release a large inference pipeline before loading the next checkpoint."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def main():
    args = parse_args()
    if bool(args.source_condition_root) != bool(args.source_condition_backend):
        raise ValueError("--source_condition_root and --source_condition_backend must be set together")
    if args.track_xyz_noise_std_multiplier < 0:
        raise ValueError("--track_xyz_noise_std_multiplier must be non-negative")
    if args.track_xyz_noise_std_multiplier > 0 and args.source_condition_root:
        raise ValueError("GT-track noise cannot be combined with an estimated source condition")
    if args.camera_control_component != "native_mixed" and abs(float(args.gt_track_orbit_deg)) > 1e-8:
        raise ValueError("camera-control component splits cannot be combined with GT-track orbit")
    if getattr(args, "future_prediction", False):
        # The future-prediction flags are exposed on the shared parser but the
        # eval inputs path does not consume them yet (Regime-2 / eval rollout is
        # not wired). Fail loud rather than silently ignore the flag.
        raise NotImplementedError(
            "future_prediction is not yet supported in eval (eval-side future "
            "rollout / Regime-2 is not implemented). Use it on the train path only."
        )
    torch.manual_seed(args.seed)

    # FSDP early init: must run BEFORE build_eval_pipeline so each rank's
    # local CUDA device is set before from_pretrained allocates 14B/T5/VAE.
    # No-op when not under accelerate launch.
    if getattr(args, "use_fsdp", False):
        from holo4d.eval_fsdp_helper import init_fsdp_early
        init_fsdp_early(args)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        (output_root / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    seq_idxs = _parse_int_list(args.sequence_idxs)
    pairs = _parse_camera_pairs(args.camera_pairs)
    pairs_by_sequence = _parse_camera_pairs_by_sequence(args.camera_pairs_by_sequence)

    if args.i2v_continuation_pairs or args.i2v_long_horizon_frames > 0:
        seed_checkpoint_path = _i2v_checkpoint_path(args, "seed")
        continuation_checkpoint_path = _i2v_checkpoint_path(args, "continuation")
        same_i2v_checkpoint = (
            seed_checkpoint_path.resolve(strict=False)
            == continuation_checkpoint_path.resolve(strict=False)
        )
        print(f"[Eval] I2V seed checkpoint: {seed_checkpoint_path}")
        print(f"[Eval] I2V continuation checkpoint: {continuation_checkpoint_path}")
        if args.lora_checkpoint and not same_i2v_checkpoint:
            raise ValueError(
                "Mixed I2V checkpoints currently require full checkpoints; "
                "--lora_checkpoint cannot be combined with different seed and "
                "continuation checkpoint paths."
            )
    else:
        print(f"[Eval] checkpoint: {args.checkpoint_path}")
    print(f"[Eval] scenes={scenes} seqs={seq_idxs} pairs={pairs} overrides={pairs_by_sequence}")
    print(f"[Eval] resolution: {args.height}x{args.width}x{args.num_frames}")
    print(f"[Eval] use_dense_branch={args.use_dense_branch} "
          f"(in_ch={args.dense_in_channels} proj_dim={args.dense_proj_dim} "
          f"residual={args.dense_num_residual_blocks} alpha={args.dense_alpha_track})")
    print(f"[Eval] track_injection_mode={args.track_injection_mode}")
    if args.source_condition_root:
        print(f"[Eval] canonical source condition: backend={args.source_condition_backend} root={args.source_condition_root}")
    if args.track_xyz_noise_std_multiplier > 0:
        print(
            "[Eval] online GT-track noise: "
            f"multiplier={args.track_xyz_noise_std_multiplier} "
            f"base_seed={args.track_xyz_noise_seed}"
        )

    if args.i2v_continuation_pairs or args.i2v_long_horizon_frames > 0:
        if same_i2v_checkpoint:
            # Preserve the historical single-checkpoint behavior when the two
            # explicit paths are identical, including resume semantics.
            pipe = build_eval_pipeline(args, checkpoint_path=continuation_checkpoint_path)
            _summarize_vista4d_plucker_modules(pipe, args)
            if args.i2v_long_horizon_frames > 0:
                _run_i2v_long_horizon_eval(pipe, args, output_root, stage=args.i2v_resume_stage)
            else:
                _run_i2v_continuation_eval(pipe, args, output_root, stage=args.i2v_resume_stage)
            _release_eval_pipeline(pipe)
            return

        # The production protocol intentionally uses two checkpoints. Build
        # them sequentially so the 14B DiT/T5/VAE from the seed stage is
        # released before the continuation checkpoint is loaded.
        seed_model_id = getattr(args, "i2v_seed_model_id_with_origin_paths", None)
        continuation_model_id = getattr(
            args, "i2v_continuation_model_id_with_origin_paths", None
        )
        seed_vxf_init = getattr(args, "i2v_seed_camera_control_checkpoint", None)
        continuation_vxf_init = getattr(
            args, "i2v_continuation_camera_control_checkpoint", None
        )
        missing_stage_config = [
            name
            for name, value in (
                ("--i2v_seed_model_id_with_origin_paths", seed_model_id),
                (
                    "--i2v_continuation_model_id_with_origin_paths",
                    continuation_model_id,
                ),
                ("--i2v_seed_camera_control_checkpoint", seed_vxf_init),
                (
                    "--i2v_continuation_camera_control_checkpoint",
                    continuation_vxf_init,
                ),
            )
            if not value
        ]
        if missing_stage_config:
            raise ValueError(
                "Mixed T2V-seed/I2V-continuation evaluation requires explicit "
                "stage model bundles: " + ", ".join(missing_stage_config)
            )
        if args.i2v_resume_stage == "continuation":
            print("[I2V/Syn4D] resuming from completed seed with I2V continuation", flush=True)
            continuation_pipe = build_eval_pipeline(
                args, checkpoint_path=continuation_checkpoint_path,
                model_id_with_origin_paths=continuation_model_id,
                camera_control_checkpoint=continuation_vxf_init,
            )
            _summarize_vista4d_plucker_modules(continuation_pipe, args)
            if args.i2v_long_horizon_frames > 0:
                _run_i2v_long_horizon_eval(continuation_pipe, args, output_root, stage="continuation")
            else:
                _run_i2v_continuation_eval(continuation_pipe, args, output_root, stage="continuation")
            _release_eval_pipeline(continuation_pipe)
            return
        print(
            "[I2V/Syn4D] loading source-only T2V seed pipeline",
            flush=True,
        )
        seed_pipe = build_eval_pipeline(
            args,
            checkpoint_path=seed_checkpoint_path,
            model_id_with_origin_paths=seed_model_id,
            camera_control_checkpoint=seed_vxf_init,
        )
        _summarize_vista4d_plucker_modules(seed_pipe, args)
        if args.i2v_long_horizon_frames > 0:
            _run_i2v_long_horizon_eval(seed_pipe, args, output_root, stage="seed")
        else:
            _run_i2v_continuation_eval(seed_pipe, args, output_root, stage="seed")
        _release_eval_pipeline(seed_pipe)
        # ``_release_eval_pipeline`` cannot delete the caller's reference.
        # Drop it here before constructing the second 14B pipeline; otherwise
        # the seed DiT/T5/VAE remains live and an 80-GB A100 OOMs during I2V
        # model loading even though the seed stage has completed.
        del seed_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        _sync_i2v_processes()

        print(
            "[I2V/Syn4D] loading I2V continuation pipeline; later chunk uses generated boundary",
            flush=True,
        )
        continuation_pipe = build_eval_pipeline(
            args,
            checkpoint_path=continuation_checkpoint_path,
            model_id_with_origin_paths=continuation_model_id,
            camera_control_checkpoint=continuation_vxf_init,
        )
        _summarize_vista4d_plucker_modules(continuation_pipe, args)
        if args.i2v_long_horizon_frames > 0:
            _run_i2v_long_horizon_eval(
                continuation_pipe, args, output_root, stage="continuation"
            )
        else:
            _run_i2v_continuation_eval(
                continuation_pipe, args, output_root, stage="continuation"
            )
        _release_eval_pipeline(continuation_pipe)
        return

    # Build pipe once, reuse across all (scene, seq, pair) combinations.
    pipe = build_eval_pipeline(args)
    _summarize_vista4d_plucker_modules(pipe, args)

    total_pairs = 0
    skipped = 0
    ran = 0
    source_condition_cache = {}
    source_provenance_cache = {}

    for scene in scenes:
        for seq_idx in seq_idxs:
            print(f"\n[Eval] === scene={scene} seq={seq_idx} ===")
            try:
                dataset = build_dataset(args, scene=scene, sequence_idx=seq_idx)
            except Exception as e:
                print(f"[Eval] skip {scene}/seq{seq_idx}: dataset build failed: {e}")
                continue
            group = dataset._nvs_groups[0]
            cam_names = group["camera_names"]
            num_cams = len(cam_names)
            start_frame = resolve_start_frame(dataset, args, group_idx=0)
            print(f"[Eval]   group '{group['scene_name']}' cameras={cam_names} "
                  f"start_frame={start_frame}")

            condition = None
            condition_provenance = None
            if args.source_condition_root:
                sequence_name = f"seq_{seq_idx:06d}"
                condition_path = canonical_path(args.source_condition_root, args.source_condition_backend, sequence_name)
                cache_key = str(condition_path)
                if cache_key not in source_condition_cache:
                    source_condition_cache[cache_key] = load_canonical_condition(condition_path, expected_backend=args.source_condition_backend)
                    source_provenance_cache[cache_key] = source_condition_provenance(condition_path, source_condition_cache[cache_key]["metadata"])
                condition = source_condition_cache[cache_key]
                condition_provenance = source_provenance_cache[cache_key]

            sequence_pairs = pairs_by_sequence.get(seq_idx, pairs)
            for src_idx, tgt_idx in sequence_pairs:
                total_pairs += 1
                if src_idx >= num_cams or tgt_idx >= num_cams:
                    print(f"[Eval]   skip pair {src_idx}-{tgt_idx}: group only has "
                          f"{num_cams} cameras ({cam_names})")
                    skipped += 1
                    continue

                pair_tag = f"src{src_idx}_tgt{tgt_idx}"
                pair_out_dir = output_root / scene / f"seq{seq_idx}" / pair_tag
                out_mp4 = pair_out_dir / "pred_rgb.mp4"
                if out_mp4.exists() and not args.force_rerun:
                    print(f"[Eval]   skip pair {pair_tag}: {out_mp4} already exists "
                          f"(use --force_rerun to overwrite)")
                    skipped += 1
                    continue

                print(f"[Eval]   running pair {pair_tag} → {pair_out_dir}")
                try:
                    sample = dataset.load_pair(
                        src_cam_idx=src_idx,
                        tgt_cam_idx=tgt_idx,
                        start_frame=start_frame,
                        group_idx=0,
                        track_query_idx=0,
                    )
                except Exception as e:
                    print(f"[Eval]   skip pair {pair_tag}: load_pair failed: {e}")
                    skipped += 1
                    continue

                if args.track_xyz_noise_std_multiplier > 0:
                    sample, noise_provenance = _apply_online_track_xyz_noise(
                        sample,
                        multiplier=args.track_xyz_noise_std_multiplier,
                        base_seed=args.track_xyz_noise_seed,
                        scene=scene,
                        sequence_idx=seq_idx,
                    )
                    pair_out_dir.mkdir(parents=True, exist_ok=True)
                    (pair_out_dir / "track_noise_provenance.json").write_text(
                        json.dumps(noise_provenance, indent=2, sort_keys=True) + "\n"
                    )
                if abs(float(args.gt_track_orbit_deg)) > 1e-8:
                    if condition is not None:
                        raise ValueError(
                            "--gt_track_orbit_deg requires direct Syn4D GT tracks; "
                            "do not combine it with --source_condition_root"
                        )
                    sample, orbit_provenance = _apply_gt_track_object_orbit(sample, args)
                    pair_out_dir.mkdir(parents=True, exist_ok=True)
                    (pair_out_dir / "gt_track_orbit.json").write_text(
                        json.dumps(orbit_provenance, indent=2, sort_keys=True) + "\n"
                    )
                if condition is not None:
                    sample = inject_canonical_condition(sample, condition)
                    pair_out_dir.mkdir(parents=True, exist_ok=True)
                    (pair_out_dir / "source_condition_provenance.json").write_text(
                        json.dumps(condition_provenance, indent=2, sort_keys=True) + "\n"
                    )
                if args.camera_control_component != "native_mixed":
                    sample, control_provenance = apply_relative_camera_control(
                        sample, args.camera_control_component
                    )
                    pair_out_dir.mkdir(parents=True, exist_ok=True)
                    (pair_out_dir / "camera_control_component.json").write_text(
                        json.dumps(control_provenance, indent=2, sort_keys=True) + "\n"
                    )
                _run_one_pair(pipe, sample, args, pair_out_dir)
                ran += 1

    print(
        f"\n[Eval] done: {ran}/{total_pairs} pairs run, {skipped} skipped. "
        f"Outputs under: {output_root}"
    )


if __name__ == "__main__":
    main()
