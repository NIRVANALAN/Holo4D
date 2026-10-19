"""
Track4D-360 Training Script.

Trains a Wan 5B V2V model with point-track injection for camera-controlled
video generation. Uses the same accelerate training pipeline as train_wan_v2v.py.

Track module config is loaded from a YAML file (configs/holo4d/default.yaml).
Block sizes (block_t/h/w) are auto-derived from the VAE + DiT patchify architecture.
"""

import os
import time as _time
from datetime import timedelta

_LAUNCH_T0 = _time.perf_counter()
_LAST_T = _LAUNCH_T0
_IS_RANK0 = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _tick(label: str) -> None:
    """Print cumulative + delta import timing on rank 0 only."""
    global _LAST_T
    if not _IS_RANK0:
        return
    now = _time.perf_counter()
    total = now - _LAUNCH_T0
    delta = now - _LAST_T
    _LAST_T = now
    print(f"[launch-timer] +{delta:6.2f}s  (total {total:7.2f}s)  {label}", flush=True)


_tick("entered train_track4d360.py")

import torch
_tick("import torch")

# Enable TF32 for fp32 ops (LayerNorm grads, fp32 master params under FSDP
# MP-bf16, optimizer state, RoPE rotation, etc.). bf16 matmuls are unaffected
# — those run on tensor cores at full bf16 throughput either way. The TF32
# matmul path on H100 is +2–5% on the fp32-ops fraction at near-zero
# numerical cost (10-bit mantissa vs 23-bit, far above bf16's 7-bit).
# cudnn flag is for any cudnn fp32 conv (rare in this codebase but free to
# enable). Override with NVIDIA_TF32_OVERRIDE=0 env if you want strict fp32
# for a debug run.
def _enable_tf32_backend_precision():
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        torch.backends.cudnn.allow_tf32 = True


_enable_tf32_backend_precision()
_tick("enabled TF32 (matmul + cudnn)")
import argparse
import accelerate
import numpy as np
from accelerate.utils import InitProcessGroupKwargs
_tick("import accelerate")
import re
import warnings
from PIL import Image

from holo4d_runtime.diffusion import *
_tick("from holo4d_runtime.diffusion import *")
from holo4d_runtime.diffusion.training_module import DiffusionTrainingModule
_tick("holo4d_runtime.diffusion.training_module")
from holo4d_runtime.diffusion.runner import launch_training_task
_tick("holo4d_runtime.diffusion.runner")
from holo4d_runtime.diffusion.logger import ModelLogger
from holo4d_runtime.diffusion.logging import create_logger
_tick("holo4d_runtime.diffusion.{logger,logging}")
from holo4d_runtime.pipelines.wan_video_track4d360 import (
    WanVideoTrack4D360Pipeline,
    ModelConfig,
)
_tick("holo4d_runtime.pipelines.wan_video_track4d360")
from holo4d.config import load_track_config, add_track4d360_args
_tick("holo4d.config")
from holo4d.train_param_norms import compute_l2_norms
_tick("holo4d.train_param_norms")
from holo4d.utils.torch_profile import enable_track4d_conv3d_module_profile_spans
_tick("holo4d.utils.torch_profile")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _is_rank0_process():
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    except Exception:
        return True


def _rank0_print(*args, **kwargs):
    if _is_rank0_process():
        print(*args, **kwargs)


def _summarize_state_dict_keys(label, keys, preview=6):
    keys = list(keys)
    if not keys:
        return f"{label}=0"
    head = ", ".join(str(k) for k in keys[:preview])
    suffix = " ..." if len(keys) > preview else ""
    return f"{label}={len(keys)} [{head}{suffix}]"


def _load_safetensors_state_dict(path_or_pattern):
    """Load safetensors from a single file, directory, or glob pattern."""
    from glob import glob
    from safetensors.torch import load_file

    raw = os.fspath(path_or_pattern)
    if os.path.isdir(raw):
        paths = sorted(glob(os.path.join(raw, "diffusion_pytorch_model*.safetensors")))
    else:
        paths = sorted(glob(raw))
        if not paths and os.path.exists(raw):
            paths = [raw]
    if not paths:
        raise FileNotFoundError(f"No safetensors files matched: {raw}")

    state_dict = {}
    for path in paths:
        state_dict.update(load_file(path))
    return state_dict, paths


def resolve_track4d360_mode(*, joint_track_prediction=False, joint_depth_prediction=False, joint_track_depth_prediction=False):
    """Resolve mutually-exclusive Track4D-360 output modes into a config dict."""
    flags = {
        "joint_track_prediction": bool(joint_track_prediction),
        "joint_depth_prediction": bool(joint_depth_prediction),
        "joint_track_depth_prediction": bool(joint_track_depth_prediction),
    }
    if sum(flags.values()) > 1:
        raise ValueError("Track4D-360 modes are mutually exclusive: use only one of "
                         "--joint_track_prediction / --joint_depth_prediction / --joint_track_depth_prediction.")
    if flags["joint_track_depth_prediction"]:
        return dict(joint_track_prediction=False, joint_depth_prediction=False,
                    joint_track_depth_prediction=True, multimodality_nvs=False,
                    multimodality_tracking=False, multimodality_track_depth=True,
                    num_domains=4, return_target_tracks=True, return_target_depth=True,
                    target_track_use_source_stats=True)
    if flags["joint_depth_prediction"]:
        return dict(joint_track_prediction=False, joint_depth_prediction=True,
                    joint_track_depth_prediction=False, multimodality_nvs=True,
                    multimodality_tracking=False, multimodality_track_depth=False,
                    num_domains=3, return_target_tracks=False, return_target_depth=True,
                    target_track_use_source_stats=False)
    if flags["joint_track_prediction"]:
        return dict(joint_track_prediction=True, joint_depth_prediction=False,
                    joint_track_depth_prediction=False, multimodality_nvs=False,
                    multimodality_tracking=True, multimodality_track_depth=False,
                    num_domains=3, return_target_tracks=True, return_target_depth=False,
                    target_track_use_source_stats=False)
    return dict(joint_track_prediction=False, joint_depth_prediction=False,
                joint_track_depth_prediction=False, multimodality_nvs=False,
                multimodality_tracking=False, multimodality_track_depth=False,
                num_domains=2, return_target_tracks=False, return_target_depth=False,
                target_track_use_source_stats=False)


def resolve_channel_concat_mode(*, stage="A", use_residual_tracks=True):
    """Resolve channel-concat multi-modal V2V config (Stage A: rgb+depth, Stage B: +residual_track)."""
    if stage == "A":
        return dict(channel_concat=True, channel_concat_stage="A",
                    channel_modalities=["rgb", "depth"], num_channel_modalities=2,
                    num_domains=2, return_target_tracks=False, return_target_depth=True,
                    return_source_depth_video=True, use_residual_tracks=False,
                    multimodality_nvs=False, multimodality_tracking=False,
                    multimodality_track_depth=False, joint_track_prediction=False,
                    joint_depth_prediction=False, joint_track_depth_prediction=False,
                    target_track_use_source_stats=False)
    elif stage == "B":
        return dict(channel_concat=True, channel_concat_stage="B",
                    channel_modalities=["rgb", "depth", "residual_track"],
                    num_channel_modalities=3, num_domains=2,
                    return_target_tracks=True, return_target_depth=True,
                    return_source_depth_video=True, use_residual_tracks=use_residual_tracks,
                    multimodality_nvs=False, multimodality_tracking=False,
                    multimodality_track_depth=False, joint_track_prediction=False,
                    joint_depth_prediction=False, joint_track_depth_prediction=False,
                    target_track_use_source_stats=True)
    else:
        raise ValueError(f"Unknown channel_concat_stage: {stage!r}. Expected 'A' or 'B'.")


def get_plucker_camera_control_trainables(
    *,
    use_plucker_camera_control=False,
    train_plucker_camera_control=False,
    use_perlayer_camera_control=False,
    use_plucker_raymap_tokens=False,
    camera_injection_mode="projector",
):
    """Return extra trainable modules for Plucker camera control."""
    modules = []
    if bool(use_plucker_camera_control) and bool(train_plucker_camera_control):
        modules.append("dit.control_adapter")
    if bool(use_plucker_raymap_tokens):
        # Ecam: patchified 6D Plucker raymap -> DiT token dim, zero-init.
        modules.append("dit._camera_patchify_unshuffle")
        modules.append("dit._camera_patchify_conv")
        modules.append("dit._camera_plucker_patchify_encoder")
        # ReCamMaster-style identity projector around self-attention output.
        modules.append("dit.blocks.*.camera_projector")
        modules.append("dit.blocks.*.self_attn")
    elif bool(use_perlayer_camera_control):
        # Per-block cam_encoder: Conv3d(6 -> dim) per block
        modules.append("dit.blocks.*.cam_encoder")
        # Per-block self_attn: Lyra-2 trains self-attention alongside cam_encoder
        modules.append("dit.blocks.*.self_attn")
        if camera_injection_mode in ("direct_raymap_add", "vista4d_plucker"):
            # ReCamMaster-style projector (not used in kq_only)
            modules.append("dit.blocks.*.camera_projector")
        if camera_injection_mode != "vista4d_plucker":
            # Camera patchify layers (24D -> 6D extraction). Vista4D-style
            # mode builds raw 6D Plucker directly at DiT token resolution.
            modules.append("dit._camera_patchify_unshuffle")
            modules.append("dit._camera_patchify_conv")
    return modules


def _ensure_dataset_constructor_kwarg(dataset_expr: str, key: str, value_literal: str) -> str:
    """Ensure a YAML dataset constructor string has ``key=value_literal``.

    Supports strings of the form ``"25000 @ DatasetClass(...)"``. If the key is
    already present, it is replaced in-place; otherwise it is appended before the
    final closing ``)``.
    """
    pattern = rf"(?P<prefix>\b{re.escape(key)}\s*=\s*)(True|False|None|[^,)]+)"
    if re.search(pattern, dataset_expr):
        return re.sub(pattern, rf"\g<prefix>{value_literal}", dataset_expr, count=1)

    insert_at = dataset_expr.rfind(")")
    if insert_at == -1:
        raise ValueError(f"Malformed dataset expression (missing closing ')'): {dataset_expr}")

    before = dataset_expr[:insert_at].rstrip()
    suffix = "" if before.endswith("(") else ", "
    return f"{before}{suffix}{key}={value_literal}{dataset_expr[insert_at:]}"


def _apply_mode_cfg_to_yaml_train_datasets(train_datasets, mode_cfg):
    """Mirror auto-enabled model-side mode flags into YAML dataset constructors."""
    patched = list(train_datasets)
    if bool(mode_cfg.get("return_target_tracks", False)):
        patched = [
            _ensure_dataset_constructor_kwarg(expr, "return_target_tracks", "True")
            for expr in patched
        ]
    if bool(mode_cfg.get("return_target_depth", False)):
        patched = [
            _ensure_dataset_constructor_kwarg(expr, "return_target_depth", "True")
            for expr in patched
        ]
    return patched


def _first_int(value, default=0):
    """Read one scalar metadata value without relying on randomized hash()."""
    if value is None:
        return int(default)
    if torch.is_tensor(value):
        return int(value.detach().reshape(-1)[0].item()) if value.numel() else int(default)
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0]) if value.size else int(default)
    if isinstance(value, (list, tuple)):
        return _first_int(value[0], default) if value else int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def derive_track_identity_sampling_seed(
    global_step,
    micro_step,
    process_index,
    dataset_index,
    base_seed=0,
):
    """Derive a stable dedicated selector seed from training-step metadata."""
    mask = (1 << 63) - 1
    value = int(base_seed) & mask
    for component in (global_step, micro_step, process_index, dataset_index):
        item = int(component) & mask
        value ^= (item + 0x9E3779B97F4A7C15 + ((value << 6) & mask) + (value >> 2)) & mask
        value &= mask
    return value


class WanTrack4D360TrainingModule(DiffusionTrainingModule):
    """
    Training module for Track4D-360.

    Extends DiffusionTrainingModule to:
      - Use WanVideoTrack4D360Pipeline (with track injection)
      - Initialize track modules (embedder, adapter, projector)
      - Handle paired camera data from Syn4D
    """

    def filter_exported_trainable_state_dict(self, state_dict):
        state_dict = super().filter_exported_trainable_state_dict(state_dict)
        if bool(getattr(self, "disable_domain_embedding", False)):
            state_dict = {
                name: param
                for name, param in state_dict.items()
                if "domain_embedding" not in name
            }
        return state_dict

    def set_training_step_context(self, global_step, micro_step=0, process_index=0):
        """Receive deterministic step metadata from the generic training runner."""
        self._training_global_step = int(global_step)
        self._training_micro_step = int(micro_step)
        self._training_process_index = int(process_index)

    def _track_identity_sampling_seed(self, data):
        if os.environ.get("TRACK4D360_DEDICATED_IDENTITY_SAMPLING_SEED", "0") != "1":
            return None
        dataset_index = _first_int(data.get("dataset_index") if isinstance(data, dict) else None)
        return derive_track_identity_sampling_seed(
            getattr(self, "_training_global_step", 0),
            getattr(self, "_training_micro_step", 0),
            getattr(self, "_training_process_index", 0),
            dataset_index,
            base_seed=int(os.environ.get("TRACK4D360_IDENTITY_SAMPLING_BASE_SEED", "0") or 0),
        )

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_alpha=None,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft:v2v",
        v2v_concat_mode="width",
        model_variant="vt2v",
        # Track config (OmegaConf)
        track_cfg=None,
        drop_text_prob=0.0,
        # Vista4D-style continuation training: condition on target frame 0
        # using the official Wan-I2V VAE+CLIP image path.
        i2v_target_condition=False,
        i2v_condition_drop_prob=0.3,
        i2v_condition_noise_level=0.05,
        # Koichi-style: directly finetune DiT self-attn/FFN (full params, not LoRA)
        dit_finetune_modules="",
        # Track injection mode: "single" (one projector before blocks) or "per_block"
        track_injection_mode="single",
        # Per-block injection style: "additive" or "concat_project" (Koichi-style)
        track_injection_block_mode="additive",
        # Resume from a non-LoRA checkpoint (safetensors with trainable params)
        resume_checkpoint=None,
        # Thin gs_head/decoder delta to overlay AFTER teacher warm-start.
        # Use this (instead of resume_checkpoint) when starting a new training
        # run from a teacher's frozen DiT + a previous run's gs_decoder/gs_head
        # weights. resume_checkpoint=teacher takes priority 4 in the launcher
        # and warms the DiT/track/dense/plucker; gs_head_init_checkpoint is then
        # applied on top to overlay any matching keys (gs_decoder.*, gs_head.*,
        # dit.gs_feature_head.*, dit.control_patch_embedding.*, etc.).
        # Keys absent from the file are silently skipped (zero-init by builder).
        gs_head_init_checkpoint=None,
        # camera-control base checkpoint to init from (32ch→dual-16ch remap)
        camera_control_checkpoint=None,
        # Stage 2: joint NVS + 3D tracking (3-domain V2V)
        joint_track_prediction=False,
        # Stage 2: joint NVS + depth only (3-domain V2V)
        joint_depth_prediction=False,
        # Stage 2 multimodal: joint NVS + 3D tracking + depth (4-domain V2V)
        joint_track_depth_prediction=False,
        # Channel-concat multi-modal V2V (Stage A: rgb+depth, Stage B: rgb+depth+track)
        channel_concat_stage=None,
        use_residual_tracks=True,
        # Plucker ray camera control
        use_plucker_camera_control=False,
        enable_v2v_plucker_camera_control=False,
        train_plucker_camera_control=False,
        use_plucker_raymap_tokens=False,
        freeze_lora_after_load=False,
        skip_v2v_required_trainables=False,
        disable_domain_embedding=False,
        gs_only_training=False,
        gs_head_style="flashworld",
        gs_lyra_depth_param="exp",
        gs_lyra_log_depth_shift=0.0,
        gs_lyra_dnear=0.1,
        gs_lyra_dfar=10.0,
        fixed_step_frac=None,
        fixed_num_inference_steps=50,
        fixed_sigma=None,
        skip_diffusion_loss=False,
        freeze_track_block_injector=False,
        ar_training=False,
        ar_training_algorithm="windowed_teacher_forcing",
        ar_num_frame_per_block=3,
        ar_denoising_step_list="1000,800,600,400,200",
        ar_same_step_across_blocks=False,
        ar_grad_through_context_cache=False,
        ar_context_noise=0.0,
        ar_context_noise_step=0,
        ar_rolling_window_blocks=0,
        ar_cache_max_latent_frames=21,
        ar_cache_keep_first_latent=False,
        ar_loss_last_latent_frames=21,
        # Arbitrary query-frame conditioning for Stage 2
        use_query_frame_impulse_condition=False,
        # Future prediction (temporal extrapolation) — all default OFF.
        future_prediction=False,
        future_num_latent_frames=0,
        future_loss_weight=1.0,
        future_source_null=False,
        target_depth_log_cap=14.0,
        target_depth_norm_type="log_depth",
        target_depth_max_disparity=5.0,
        # GS output: 3D Gaussian Splatting from diffusion features
        use_gs_output=False,
        gs_zero_features=False,
        use_gs_feedback_training=False,
        gs_feat_dim=1024,
        gs_loss_weight=1.0,
        gs_mse_loss_weight=1.0,
        gs_lpips_loss_weight=0.5,
        gs_depth_loss_weight=0.05,
        gs_abs_depth_loss_weight=0.0,
        gs_opacity_loss_weight=0.1,
        gs_opacity_prune_keep_frac=1.0,
        gs_sh_degree=2,
        gs_num_points_per_pixel=2,
        gs_scale_cap=None,
        gs_render_patch_size=None,
        gs_legacy_union_render=False,
        gs_replay_cache_size=0,
        no_gs_latent_detach=False,
        # Dynamic GS: canonical first-frame Gaussians + per-frame deltas
        gs_dynamic_mode="static",
        gs_pos_loss_weight=1.0,
        gs_dyn_depth_loss_weight=0.5,
        gs_opa_tv_weight=0.01,
        gs_xyz_tv_weight=0.01,
        gs_predict_rotation=False,
        gs_dynamic_max_t_video=64,
        # Track-only loss weights (consumed by gs_loss_dynamic deltas mode
        # AND v4 static_with_track mode). Default 0 → loss term inactive.
        # Cache-v3 recipe: 1.0 / 1.0 / 1.0 / 5.0 / 0.5 with quantile=0.98.
        gs_track_canonical_loss_weight=0.0,
        gs_track_delta_loss_weight=0.0,
        gs_track_final_pos_loss_weight=0.0,
        gs_assoc_track_2d_loss_weight=0.0,
        gs_track_accel_loss_weight=0.0,
        gs_track_loss_quantile=0.98,
        # v4: detach canonical from gs_proj graph before track losses. Default
        # False preserves cache-v3 behavior — track GT supervises render-head's
        # frame-0 xyz directly, anchoring monocular depth that pure photometric
        # can't disambiguate. Set True only as an ablation that isolates the
        # track-head from render-head supervision.
        gs_track_canonical_detach=False,
        # v4 joint-render: at every t, also render canonical warped by Δxyz[t]
        # alongside the per-frame Gaussians (alpha-blended via z-buffer in one
        # gaussian_render call). track_delta_proj then receives photometric
        # gradient — not only track-GT gradient — and slot-permutation between
        # render-head's npp and track-head's npp self-resolves through joint
        # photometric optimization. Default False preserves current behavior;
        # True ≈ 2× render compute at t>0 (≈ doubled Gaussian count).
        gs_track_joint_render=False,
        gs_track_supervision_mode="pointmap",
        disable_track_modules=False,
        force_t2v_drop_source_cond=False,
        raw_t2v_target_only=False,
        dense_only_condition_modules=False,
        freeze_track4d_condition_modules=False,
        # Dense GEN3C-style target-space control branch
        use_dense_branch=False,
        dense_in_channels=5,
        dense_proj_dim=32,
        dense_num_residual_blocks=2,
        dense_alpha_track=1.0,
        dense_conditioning_noise_max_level=0.0,
        source_mask_boundary_depth_dropout_prob=0.0,
        source_mask_boundary_kernel_size=5,
        dense_warp_mode="zbuffer",
        dense_warp_motion_threshold=0.005,
        dense_hybrid_disable_dynamic_masks=False,
        use_vista4d_latent_streams=False,
        vista4d_streams_pointcloud_only=False,
        use_gen3c_warp_latent_channels=False,
        use_depth_null_token=False,
        # Conditioning dropout
        cond_drop_track_prob=0.0,
        cond_drop_dense_prob=0.0,
        cond_drop_geo_prob=0.0,
        # Learned null-token fallback for missing/dropped geometry conditions
        use_track_null_token_fallback=False,
        use_dense_null_token_fallback=False,
        # Per-layer camera control
        use_perlayer_camera_control=False,
        camera_injection_mode="projector",
        camera_plucker_encoder_mode="patchify",
        plucker_use_ray_origin=False,
        # Warped appearance conditioning (Lyra-2-inspired)
        warped_condition_mode="off",
        warped_appearance_fusion="concat_proj",
        warped_geom_only_dense=True,
        multi_frame_warped_appearance=False,
        multi_frame_motion_threshold=0.01,
    ):
        super().__init__()
        self.channel_concat_stage = channel_concat_stage
        self.use_plucker_camera_control = bool(use_plucker_camera_control)
        self.enable_v2v_plucker_camera_control = bool(enable_v2v_plucker_camera_control)
        self.train_plucker_camera_control = bool(train_plucker_camera_control)
        self.use_plucker_raymap_tokens = bool(use_plucker_raymap_tokens)
        self.use_perlayer_camera_control = bool(use_perlayer_camera_control)
        self.freeze_track4d_condition_modules = bool(freeze_track4d_condition_modules)
        self.use_vista4d_latent_streams = bool(use_vista4d_latent_streams)
        self.vista4d_streams_pointcloud_only = bool(vista4d_streams_pointcloud_only)
        self.use_gen3c_warp_latent_channels = bool(use_gen3c_warp_latent_channels)
        self.camera_injection_mode = str(camera_injection_mode)
        self.camera_plucker_encoder_mode = str(camera_plucker_encoder_mode)
        self.plucker_use_ray_origin = bool(plucker_use_ray_origin)
        self.warped_condition_mode = str(warped_condition_mode)
        self.warped_appearance_fusion = str(warped_appearance_fusion)
        self.warped_geom_only_dense = bool(warped_geom_only_dense)
        self.multi_frame_warped_appearance = bool(multi_frame_warped_appearance)
        self.multi_frame_motion_threshold = float(multi_frame_motion_threshold)
        self.freeze_track_block_injector = bool(freeze_track_block_injector)
        if self.enable_v2v_plucker_camera_control and not self.use_plucker_camera_control:
            raise ValueError(
                "--enable_v2v_plucker_camera_control requires --use_plucker_camera_control."
            )
        if self.use_plucker_raymap_tokens:
            if not self.use_plucker_camera_control or not self.enable_v2v_plucker_camera_control:
                raise ValueError(
                    "--use_plucker_raymap_tokens requires --use_plucker_camera_control "
                    "and --enable_v2v_plucker_camera_control."
                )
            if not self.use_perlayer_camera_control:
                _rank0_print(
                    "[Track4D-360] --use_plucker_raymap_tokens implies "
                    "--use_perlayer_camera_control."
                )
            self.use_perlayer_camera_control = True
            if self.camera_injection_mode != "projector":
                raise ValueError("--use_plucker_raymap_tokens requires --camera_injection_mode projector.")
            if self.camera_plucker_encoder_mode != "patchify":
                raise ValueError("--use_plucker_raymap_tokens requires --camera_plucker_encoder_mode patchify.")
        if channel_concat_stage is not None:
            self.mode_cfg = resolve_channel_concat_mode(stage=channel_concat_stage, use_residual_tracks=use_residual_tracks)
        else:
            self.mode_cfg = resolve_track4d360_mode(
                joint_track_prediction=joint_track_prediction,
                joint_depth_prediction=joint_depth_prediction,
                joint_track_depth_prediction=joint_track_depth_prediction,
            )
        # Mirror the main()-side auto-enables here so self.return_target_*
        # flags drive the inputs_shared injection (line 1755+) consistently
        # with the dataset YAML patch in main(). Without this, the dataset
        # emits target_depth but __getitem__'s `if self.return_target_depth:`
        # gate is False → gs_depth_loss / gs_abs_depth_loss stay at 0.
        if bool(use_gs_output) and not self.mode_cfg.get("return_target_depth", False):
            self.mode_cfg["return_target_depth"] = True
            _rank0_print("[Track4D-360] use_gs_output=True (init): auto-enabled "
                         "self.mode_cfg['return_target_depth']=True so target depth "
                         "lands in inputs_shared.")
        if str(gs_dynamic_mode).lower() in ("deltas", "static_with_track") and \
                not self.mode_cfg.get("return_target_tracks", False):
            self.mode_cfg["return_target_tracks"] = True
            _rank0_print(f"[Track4D-360] gs_dynamic_mode={gs_dynamic_mode} (init): "
                         "auto-enabled self.mode_cfg['return_target_tracks']=True.")
        self.joint_track_depth_prediction = bool(self.mode_cfg.get("joint_track_depth_prediction", False))
        self.joint_depth_prediction = bool(self.mode_cfg.get("joint_depth_prediction", False))
        self.joint_track_prediction = bool(self.mode_cfg.get("joint_track_prediction", False))
        self.return_target_tracks = bool(self.mode_cfg["return_target_tracks"])
        self.return_target_depth = bool(self.mode_cfg["return_target_depth"])
        self.i2v_target_condition = bool(i2v_target_condition)
        self.i2v_condition_drop_prob = float(i2v_condition_drop_prob)
        self.i2v_condition_noise_level = float(i2v_condition_noise_level)
        from holo4d.conditions.i2v_target_condition import (
            validate_i2v_condition_config,
        )
        validate_i2v_condition_config(
            self.i2v_condition_drop_prob, self.i2v_condition_noise_level
        )
        self.use_query_frame_impulse_condition = bool(use_query_frame_impulse_condition)
        self.future_prediction = bool(future_prediction)
        self.future_num_latent_frames = int(future_num_latent_frames)
        self.future_loss_weight = float(future_loss_weight)
        self.future_source_null = bool(future_source_null)
        self.target_depth_log_cap = float(target_depth_log_cap)
        self.target_depth_norm_type = str(target_depth_norm_type)
        self.target_depth_max_disparity = float(target_depth_max_disparity)

        if task == "sft:v2v":
            assert model_variant == "vt2v", (
                "V2V training requires --model_variant vt2v"
            )

        # Load pipeline
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ) if tokenizer_path is None else ModelConfig(tokenizer_path)

        _tick("before WanVideoTrack4D360Pipeline.from_pretrained")
        self.pipe = WanVideoTrack4D360Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        _tick("after WanVideoTrack4D360Pipeline.from_pretrained (T5/VAE/DiT loaded)")
        if self.i2v_target_condition:
            vae_z = int(
                getattr(getattr(getattr(self.pipe, "vae", None), "model", None), "z_dim", 16)
            )
            dit_in = int(getattr(self.pipe.dit, "in_dim", vae_z))
            if dit_in <= vae_z or not bool(getattr(self.pipe.dit, "has_image_input", False)):
                raise ValueError(
                    "--i2v_target_condition requires a Wan I2V backbone (expected "
                    f"in_dim > VAE z_dim and has_image_input=True; got in_dim={dit_in}, "
                    f"vae_z_dim={vae_z}, has_image_input="
                    f"{getattr(self.pipe.dit, 'has_image_input', None)})."
                )
            if getattr(self.pipe, "image_encoder", None) is None:
                raise ValueError(
                    "--i2v_target_condition requires the Wan CLIP image encoder. Add "
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth to "
                    "--model_id_with_origin_paths."
                )
            _rank0_print(
                "[Track4D-360 I2V] target-frame-0 conditioning enabled: "
                f"drop_prob={self.i2v_condition_drop_prob}, "
                f"latent_noise={self.i2v_condition_noise_level}, DiT in_dim={dit_in}"
            )
        if self.mode_cfg.get("channel_concat"):
            self.pipe.enable_channel_concat(self.mode_cfg)
        elif self.joint_track_depth_prediction:
            self.pipe.enable_joint_track_depth_prediction()
        elif self.joint_depth_prediction:
            self.pipe.enable_joint_depth_prediction()
        elif self.joint_track_prediction:
            self.pipe.enable_joint_track_prediction()
        if disable_track_modules:
            self.pipe.disable_track_conditioning_units()

        # Ensure impulse-conditioning control patch embedding exists before
        # checkpoint loading / DDP wrapping, matching the ECCV V2V path.
        if self.use_query_frame_impulse_condition:
            for name in ("dit", "dit2"):
                model = getattr(self.pipe, name, None)
                if model is None or not hasattr(model, "ensure_control_patch_embedding"):
                    continue
                vae_z = int(
                    getattr(getattr(getattr(self.pipe, "vae", None), "model", None), "z_dim", 16)
                )
                dit_in = int(getattr(model, "in_dim", vae_z))
                control_c = max(int(dit_in), int(vae_z))
                model.ensure_control_patch_embedding(in_channels=control_c)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Precompute T5 embeddings for the small set of training-time prompt
        # strings that DON'T come from the per-chunk caption cache:
        #   - ""                              (drop-text-prob CFG-dropout path)
        #   - "A multi-view video scene."     (DEFAULT_PROMPT in track4d360_mixed_nvs)
        #   - "A dynamic multi-view video scene."  (DEFAULT_PROMPT in syn4d_nvs)
        # These are stored as `pipe.precomputed_prompt_embs`. The prompt
        # embedder unit looks them up by exact string match before falling
        # through to an online T5 encode. Combined with `data["prompt_emb"]`
        # forwarded by `get_pipeline_inputs`, this means online T5 fires only
        # on cache misses (stale / partial precompute) — never in steady state.
        #
        # Cheap (~1 sec, three forwards) and idempotent across ranks.
        self._setup_prompt_emb_cache()

        # Store config
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.task = task
        self.v2v_concat_mode = v2v_concat_mode
        self.disable_domain_embedding = bool(disable_domain_embedding)
        self.drop_text_prob = float(drop_text_prob)
        self.cond_drop_track_prob = float(cond_drop_track_prob)
        self.cond_drop_dense_prob = float(cond_drop_dense_prob)
        self.cond_drop_geo_prob = float(cond_drop_geo_prob)
        self.use_track_null_token_fallback = bool(use_track_null_token_fallback)
        self.use_dense_null_token_fallback = bool(use_dense_null_token_fallback)
        self.ar_training = bool(ar_training)
        if self.ar_training and self.future_prediction:
            # future prediction only flows through FlowMatchV2VLoss so far; the AR
            # loss (FlowMatchTrack4D360ARLoss) does NOT apply source-null / future
            # loss-weight, yet the track unit would still null future visibility →
            # a silently-partial combo. Block it until AR supports future.
            raise NotImplementedError(
                "future_prediction is not yet supported with ar_training: the AR loss "
                "does not apply future source-null / loss-weight. Run them separately."
            )
        self.ar_training_algorithm = str(ar_training_algorithm)
        _ar_algos = {
            "windowed_teacher_forcing",
            "rolling_forcing",
            "diffusion_forcing",
            "causal_teacher_forcing",
        }
        if self.ar_training_algorithm not in _ar_algos:
            # argparse enforces this for CLI runs; guard direct Python construction
            # so a typo can't silently fall back to teacher forcing.
            raise ValueError(
                f"ar_training_algorithm must be one of {sorted(_ar_algos)}, "
                f"got {self.ar_training_algorithm!r}."
            )
        self.ar_config = {
            "ar_num_frame_per_block": int(ar_num_frame_per_block),
            "ar_denoising_step_list": str(ar_denoising_step_list),
            "ar_same_step_across_blocks": bool(ar_same_step_across_blocks),
            "ar_detach_context_cache": not bool(ar_grad_through_context_cache),
            "ar_context_noise": float(ar_context_noise),
            "ar_context_noise_step": int(ar_context_noise_step),
            "ar_rolling_window_blocks": int(ar_rolling_window_blocks),
            "ar_cache_max_latent_frames": int(ar_cache_max_latent_frames),
            "ar_cache_keep_first_latent": bool(ar_cache_keep_first_latent),
            "ar_loss_last_latent_frames": int(ar_loss_last_latent_frames),
        }
        self._latest_log_metrics = {}
        self._latest_graph_signature = {}
        self._pending_input_metrics = {}
        self._printed_runtime_grad_ckpt = False

        def _ar_loss(pipe, inputs_shared, inputs_posi):
            if self.ar_training_algorithm == "rolling_forcing":
                # Faithful self-forcing rollout (clean detached KV cache, exit-block
                # gradient). Imported lazily to keep the default path's import graph
                # unchanged.
                from holo4d_runtime.diffusion.loss import FlowMatchTrack4D360RollingForcingLoss
                return FlowMatchTrack4D360RollingForcingLoss(pipe, **inputs_shared, **inputs_posi)
            if self.ar_training_algorithm == "diffusion_forcing":
                from holo4d_runtime.diffusion.loss import FlowMatchTrack4D360DiffusionForcingLoss
                return FlowMatchTrack4D360DiffusionForcingLoss(pipe, **inputs_shared, **inputs_posi)
            if self.ar_training_algorithm == "causal_teacher_forcing":
                from holo4d_runtime.diffusion.loss import FlowMatchTrack4D360CausalTeacherForcingLoss
                return FlowMatchTrack4D360CausalTeacherForcingLoss(pipe, **inputs_shared, **inputs_posi)
            # Default: windowed teacher forcing (noises GT windows; NOT self-forcing).
            return FlowMatchTrack4D360ARLoss(pipe, **inputs_shared, **inputs_posi)

        self.task_to_loss = {
            "sft:v2v": lambda pipe, inputs_shared, inputs_posi, inputs_nega: (
                _ar_loss(pipe, inputs_shared, inputs_posi)
                if self.ar_training
                else FlowMatchV2VLoss(pipe, **inputs_shared, **inputs_posi)
            ),
        }

        # Ensure domain embedding unless this is an explicit raw-T2V sanity run.
        # Channel-concat: always 2 domains (source, target). Legacy width-concat:
        # 2-4 domains depending on mode.
        if not self.disable_domain_embedding:
            from holo4d_runtime.pipelines.wan_video_v2v import _ensure_domain_embedding
            num_domains = int(self.mode_cfg["num_domains"])
            _ensure_domain_embedding(self.pipe.dit, num_domains=num_domains)
        else:
            _rank0_print(
                "[Track4D-360] disable_domain_embedding=True: skipping V2V "
                "domain_embedding initialization/use for raw T2V-style runs."
            )

        # Channel-concat: initialize multimodal patch embedding
        if self.mode_cfg.get("channel_concat"):
            num_mod = int(self.mode_cfg["num_channel_modalities"])
            self.pipe.dit._ensure_patch_embedding_multimodal(num_mod)
            print(f"[Track4D-360] Channel-concat stage {channel_concat_stage}: "
                  f"{num_mod} modalities, {num_mod}*{self.pipe.dit.in_dim}="
                  f"{num_mod * self.pipe.dit.in_dim} input channels")

        if self.use_plucker_camera_control:
            from holo4d_runtime.models.wan_video_camera_controller import (
                initialize_camera_control_adapter,
                initialize_perlayer_camera_encoders,
                initialize_camera_patchify_layers,
                initialize_camera_raymap_token_projection,
                initialize_vista4d_plucker_camera_encoders,
            )
            initialize_camera_control_adapter(self.pipe.dit, self.pipe.vae, in_dim_control_adapter=24)
            train_state = "trainable" if self.train_plucker_camera_control else "frozen"
            _rank0_print(f"[Track4D-360] Plucker camera control: SimpleAdapter initialized (24→dim, {train_state})")

            if self.use_plucker_raymap_tokens:
                initialize_camera_raymap_token_projection(self.pipe.dit, self.pipe.vae, init_zero=True)
                _rank0_print(
                    "[Track4D-360] Plucker raymap tokens: initialized zero Ecam "
                    "(6D raymap→dim) + identity attention projectors"
                )
            elif self.use_perlayer_camera_control:
                # Initialize per-layer camera encoders.
                # kq_only: strict Lyra-2 style — no projector, standard gate on attn output.
                # direct_raymap_add: ReCamMaster style — projector wraps attn output.
                add_projector = self.camera_injection_mode in ("direct_raymap_add", "vista4d_plucker")
                if self.camera_injection_mode == "vista4d_plucker":
                    initialize_vista4d_plucker_camera_encoders(
                        self.pipe.dit, plucker_dim=6, add_projector=add_projector,
                    )
                else:
                    initialize_perlayer_camera_encoders(
                        self.pipe.dit, plucker_dim=6, add_projector=add_projector,
                    )
                if self.camera_plucker_encoder_mode == "patchify" and self.camera_injection_mode != "vista4d_plucker":
                    initialize_camera_patchify_layers(self.pipe.dit, self.pipe.vae)
                _rank0_print(
                    f"[Track4D-360] Per-layer camera control: mode={self.camera_injection_mode}, "
                    f"encoder={self.camera_plucker_encoder_mode}, projector={add_projector}"
                )

        if self.use_vista4d_latent_streams:
            from holo4d_runtime.models.wan_video_camera_controller import (
                initialize_vista4d_latent_stream_patch_embeddings,
            )
            initialize_vista4d_latent_stream_patch_embeddings(self.pipe.dit)
            _rank0_print(
                "[Track4D-360] Vista4D latent streams: initialized source/pointcloud "
                "RGB patch embeddings + zero-init mask patch embeddings"
            )

        if self.use_gen3c_warp_latent_channels:
            from holo4d_runtime.models.wan_video_camera_controller import (
                initialize_gen3c_warp_condition_patch_embedding,
            )
            initialize_gen3c_warp_condition_patch_embedding(self.pipe.dit, in_channels=32)
            _rank0_print(
                "[Track4D-360] GEN3C warp latent channels: initialized zero "
                "patch embedding for VAE(warped RGB) + VAE(valid mask)"
            )

        # Warped appearance conditioning (Lyra-2-inspired)
        if self.warped_condition_mode == "latent_fuse":
            from holo4d.conditions.shallow_appearance_fuser import (
                initialize_shallow_appearance_fuser,
                initialize_geometry_track_fuser,
            )
            initialize_shallow_appearance_fuser(self.pipe)
            _rank0_print("[Track4D-360] Warped appearance fuser initialized")
            # Initialize geometry+track concat fuser
            initialize_geometry_track_fuser(self.pipe)
            _rank0_print("[Track4D-360] Geometry+track concat fuser initialized")
            # NOTE: in latent_fuse + warped_geom_only_dense mode, the dense encoder
            # should be initialized with in_channels=2 (depth+valid only) via
            # --dense_in_channels 2 in the bash script. No separate geometry encoder
            # is needed — the standard dense_target_control_encoder handles it.

        # GS output: initialize GS feature head + GS decoder
        self.use_gs_output = bool(use_gs_output)
        self.gs_detach_predicted_latents = not bool(no_gs_latent_detach)
        self.use_gs_feedback_training = bool(use_gs_feedback_training) and bool(use_gs_output)
        self.gs_only_training = bool(gs_only_training)
        # Sigma resolution is deferred until AFTER switch_pipe_to_training_mode
        # has called scheduler.set_timesteps() — only then does the scheduler
        # have .sigmas. For now, just stash the requested knobs.
        self.fixed_step_frac = None if fixed_step_frac is None else float(fixed_step_frac)
        self.fixed_num_inference_steps = int(fixed_num_inference_steps)
        # --fixed_sigma takes precedence: when set, it overrides step_frac
        # resolution below (direct σ target, no shift math). Resolved as-is
        # post-switch_pipe_to_training_mode via argmin on scheduler.sigmas.
        self._requested_fixed_sigma = None if fixed_sigma is None else float(fixed_sigma)
        self.fixed_sigma = None  # final value, resolved post-switch_pipe_to_training_mode
        self.skip_diffusion_loss = bool(skip_diffusion_loss)
        if self.fixed_step_frac is not None:
            if not (0.0 <= self.fixed_step_frac <= 1.0):
                raise ValueError(
                    f"fixed_step_frac must be in [0, 1]; got {self.fixed_step_frac}"
                )
            if self.fixed_num_inference_steps <= 0:
                raise ValueError(
                    f"fixed_num_inference_steps must be > 0 when fixed_step_frac "
                    f"is set; got {self.fixed_num_inference_steps}"
                )
        if self.skip_diffusion_loss:
            _rank0_print(
                "[Track4D-360] skip_diffusion_loss=True: FM MSE term will be "
                "zeroed (GS-only mode — DiT forward still runs for gs_features)."
            )
        self.gs_dynamic_mode = str(gs_dynamic_mode).lower()
        self.gs_predict_rotation = bool(gs_predict_rotation)
        self.disable_track_modules = bool(disable_track_modules)
        self.force_t2v_drop_source_cond = bool(force_t2v_drop_source_cond)
        self.raw_t2v_target_only = bool(raw_t2v_target_only)
        self.dense_only_condition_modules = bool(dense_only_condition_modules)
        self.gs_config = {}
        if self.use_gs_output:
            from holo4d.models.gs_decoder import GaussianSplattingDecoder
            # Init GS feature head on DiT (zero-init, lazy)
            self.pipe.dit.ensure_gs_feature_head(feat_dim=int(gs_feat_dim))
            # Init GS decoder from VAE decoder
            vae_decoder = self.pipe.vae.model.decoder
            gs_decoder = GaussianSplattingDecoder(
                vae_decoder=vae_decoder,
                feat_dim=int(gs_feat_dim),
                num_points_per_pixel=int(gs_num_points_per_pixel),
                sh_degree=int(gs_sh_degree),
                use_network_checkpointing=True,
                use_render_checkpointing=True,
                scale_cap=gs_scale_cap,
                render_patch_size=(
                    tuple(gs_render_patch_size) if gs_render_patch_size else None
                ),
                render_per_frame_gaussians=not bool(gs_legacy_union_render),
            )
            self.pipe.gs_decoder = gs_decoder
            if bool(gs_legacy_union_render):
                _rank0_print(
                    "[Track4D-360] GS render compatibility: legacy union render "
                    "(all per-frame Gaussian sets flattened for every view)."
                )

            # Optional: swap the activation head to Lyra-style ('hybrid' head).
            # Warm VAE decoder body (from gs_decoder above) + fresh Lyra head —
            # matches the pattern in train_gs_head_online.py:527 and
            # train_gs_head_from_cache.py. Default 'flashworld' is a no-op
            # so existing launchers / checkpoints are unaffected.
            if str(gs_head_style).lower() == "hybrid":
                from holo4d.models.gs_decoder_lyra import LyraPixelAligned3DGS
                old_npp = int(gs_decoder.gs_head.num_points_per_pixel)
                old_embed_dim = int(gs_decoder.gs_head.gs_proj.in_channels)
                # v4: when gs_dynamic_mode=static_with_track, the Lyra head
                # also gets a parallel track_delta_proj head for per-pixel
                # per-frame 3D-delta supervision (track-only, never enters
                # rendering). See doc/track4d-360/2026-04-26-v4-per-frame-track-head.md.
                _v4_track_delta = str(self.gs_dynamic_mode).lower() == "static_with_track"
                if _v4_track_delta and int(old_npp) > 1:
                    _rank0_print(
                        f"[Track4D-360] WARNING: gs_dynamic_mode=static_with_track with npp={old_npp}. "
                        f"track_delta_proj's npp slots are only soft-aligned to gs_proj's slots via the "
                        f"shared per-pixel feature backbone — frame-0 association uses render-head xyz "
                        f"to pick slot k, then expects track-head's slot k to predict the matching delta. "
                        f"With npp=1 this risk vanishes. If track losses fail to converge, try npp=1."
                    )
                gs_decoder.gs_head = LyraPixelAligned3DGS(
                    embed_dim=old_embed_dim,
                    num_points_per_pixel=old_npp,
                    dnear=float(gs_lyra_dnear),
                    dfar=float(gs_lyra_dfar),
                    scale_cap=0.3,
                    predict_offset=True,
                    depth_param=str(gs_lyra_depth_param),
                    log_depth_shift=float(gs_lyra_log_depth_shift),
                    predict_track_delta=_v4_track_delta,
                )
                _rank0_print(
                    f"[Track4D-360] gs_head_style=hybrid: warm VAE decoder body + "
                    f"fresh Lyra head (npp={old_npp}, embed_dim={old_embed_dim}, "
                    f"depth_param={gs_lyra_depth_param}, "
                    f"log_depth_shift={gs_lyra_log_depth_shift}, "
                    f"dnear={gs_lyra_dnear}, dfar={gs_lyra_dfar}, "
                    f"predict_track_delta={_v4_track_delta})"
                )
            elif str(gs_head_style).lower() != "flashworld":
                raise ValueError(
                    f"Unknown gs_head_style={gs_head_style!r}; expected "
                    f"'flashworld' or 'hybrid'."
                )

            # FlashWorld-style feedback: widen patch embedding to accept
            # prev_feats (gs_feat_dim) + prev_latents_pred (in_dim=16).
            if self.use_gs_feedback_training:
                feedback_channels = int(gs_feat_dim) + int(self.pipe.dit.in_dim)
                self.pipe.dit.widen_patch_embedding_for_feedback(feedback_channels)
                print(
                    f"[Track4D-360] GS feedback training: widened patch embedding "
                    f"by {feedback_channels} channels "
                    f"(prev_feats={gs_feat_dim} + prev_latents_pred={self.pipe.dit.in_dim})"
                )

            # Dynamic GS: attach the canonical+deltas wrapper as a child of
            # gs_decoder so the existing trainable_models entry "gs_decoder"
            # picks it up recursively. The base decoder is shared (composition,
            # not deep-copied), so the static path is unaffected.
            if self.gs_dynamic_mode == "deltas":
                from holo4d.models.gs_decoder_dynamic import (
                    DynamicGaussianSplattingDecoder,
                )
                gs_decoder.dynamic = DynamicGaussianSplattingDecoder(
                    base_decoder=gs_decoder,
                    max_t_video=int(gs_dynamic_max_t_video),
                    predict_rotation=self.gs_predict_rotation,
                )
                print(
                    f"[Track4D-360] GS dynamic mode: deltas "
                    f"(max_t_video={gs_dynamic_max_t_video}, "
                    f"predict_rotation={self.gs_predict_rotation})"
                )
            elif self.gs_dynamic_mode == "static_with_track":
                # v4: rendering uses per-frame static head (no DynamicDecoder
                # wrap). Track supervision comes from the gs_head's
                # `track_delta_proj` head, plumbed in the hybrid-swap block
                # above when gs_head_style=hybrid.
                if str(gs_head_style).lower() != "hybrid":
                    raise ValueError(
                        f"gs_dynamic_mode='static_with_track' (v4) requires "
                        f"gs_head_style='hybrid' (the FlashWorld PixelAligned3DGS "
                        f"head doesn't have a track_delta_proj). "
                        f"Got gs_head_style={gs_head_style!r}."
                    )
                print(
                    "[Track4D-360] GS dynamic mode: static_with_track (v4) — "
                    "per-frame static rendering + parallel track-only delta head "
                    "(supervised by target_tracks_world when present)."
                )
            elif self.gs_dynamic_mode != "static":
                raise ValueError(
                    f"Unknown gs_dynamic_mode={self.gs_dynamic_mode}; "
                    f"must be 'static', 'deltas', or 'static_with_track'"
                )

            self.gs_config = {
                "gs_loss_weight": float(gs_loss_weight),
                "mse_loss_weight": float(gs_mse_loss_weight),
                "lpips_loss_weight": float(gs_lpips_loss_weight),
                "depth_loss_weight": float(gs_depth_loss_weight),
                "abs_depth_loss_weight": float(gs_abs_depth_loss_weight),
                "opacity_loss_weight": float(gs_opacity_loss_weight),
                "opacity_prune_keep_frac": float(gs_opacity_prune_keep_frac),
                # When True, predicted clean latents stay connected for GS loss (joint VDM+3DGS).
                "detach_predicted_latents": bool(self.gs_detach_predicted_latents),
                # Dynamic GS knobs (consumed by the loss-site dispatch)
                "dynamic_mode": self.gs_dynamic_mode,
                "pos_loss_weight": float(gs_pos_loss_weight),
                "dyn_depth_loss_weight": float(gs_dyn_depth_loss_weight),
                "opa_tv_weight": float(gs_opa_tv_weight),
                "xyz_tv_weight": float(gs_xyz_tv_weight),
                "predict_rotation": bool(gs_predict_rotation),
                # Zero the DiT-side feature stream at the GS decoder.
                # Matches cache overfit's --feature_mode x0_mid semantics
                # (train_gs_head_from_cache.py:697-700 substitutes zero
                # features). With gs_zero_features=True the decoder output
                # depends purely on the x0-hat latent; the projector still
                # runs but its output is discarded.
                "zero_features": bool(gs_zero_features),
                # Overfit accelerator: cap the in-memory GS-loss-input cache
                # in holo4d_runtime.diffusion.loss._GS_REPLAY_CACHE. Once the
                # cache hits this size, FlowMatchV2VLoss skips the DiT
                # forward and replays a random cached entry. 0 disables.
                "replay_cache_size": int(gs_replay_cache_size),
                # Track-only loss weights — consumed by deltas-mode and
                # v4 static_with_track-mode dispatches in
                # holo4d_runtime/diffusion/loss.py. Default 0 keeps each term
                # silent; cache-v3 recipe sets canon/delta/final = 1.0,
                # assoc_2d = 5.0, accel = 0.5, quantile = 0.98.
                "track_canonical_loss_weight": float(gs_track_canonical_loss_weight),
                "track_delta_loss_weight": float(gs_track_delta_loss_weight),
                "track_final_pos_loss_weight": float(gs_track_final_pos_loss_weight),
                "assoc_track_2d_loss_weight": float(gs_assoc_track_2d_loss_weight),
                "track_accel_loss_weight": float(gs_track_accel_loss_weight),
                "track_loss_quantile": float(gs_track_loss_quantile),
                "track_canonical_detach": bool(gs_track_canonical_detach),
                "track_joint_render": bool(gs_track_joint_render),
                "gs_track_supervision_mode": str(gs_track_supervision_mode),
            }
            _latent_mode = (
                "detach_latents_for_gs_decoder"
                if self.gs_detach_predicted_latents
                else "joint_backprop_through_predicted_latents"
            )
            _rank0_print(f"[Track4D-360] GS output: feat_dim={gs_feat_dim}, "
                         f"sh_degree={gs_sh_degree}, npp={gs_num_points_per_pixel}, "
                         f"loss_weight={gs_loss_weight}, "
                         f"mse={gs_mse_loss_weight}, lpips={gs_lpips_loss_weight}, "
                         f"depth={gs_depth_loss_weight}, opacity={gs_opacity_loss_weight}, "
                         f"opacity_prune_keep_frac={gs_opacity_prune_keep_frac}, "
                         f"dynamic_mode={self.gs_dynamic_mode}, latent_grad={_latent_mode}")

        self.use_dense_branch = bool(use_dense_branch)
        self.dense_alpha_track = float(dense_alpha_track)
        self.dense_warp_mode = str(dense_warp_mode)
        self.dense_warp_motion_threshold = float(dense_warp_motion_threshold)
        self.dense_hybrid_disable_dynamic_masks = bool(dense_hybrid_disable_dynamic_masks)
        # Vista4D-style opt-in noise augmentation (0.0 = off, backward-compatible).
        # Stored on the pipe so WanVideoUnit_DenseTargetControl can read it at forward time.
        self.dense_conditioning_noise_max_level = float(dense_conditioning_noise_max_level)
        self.source_mask_boundary_depth_dropout_prob = float(source_mask_boundary_depth_dropout_prob)
        self.source_mask_boundary_kernel_size = int(source_mask_boundary_kernel_size)
        if hasattr(self, "pipe") and self.pipe is not None:
            self.pipe.dense_conditioning_noise_max_level = self.dense_conditioning_noise_max_level
            self.pipe.source_mask_boundary_depth_dropout_prob = self.source_mask_boundary_depth_dropout_prob
            self.pipe.source_mask_boundary_kernel_size = self.source_mask_boundary_kernel_size
            self.pipe.use_vista4d_latent_streams = self.use_vista4d_latent_streams
            self.pipe.use_gen3c_warp_latent_channels = self.use_gen3c_warp_latent_channels
        # Initialize track modules (block_t/h/w auto-derived from architecture)
        # unless this run is explicitly plucker-only/raw V2V.
        if self.disable_track_modules:
            if self.use_dense_branch:
                raise ValueError("--use_dense_branch requires persistent Track4D modules; remove it when --disable_track_modules is set.")
            if self.use_track_null_token_fallback or self.use_dense_null_token_fallback:
                raise ValueError("Null-token fallback requires persistent Track4D modules; remove the fallback flags when --disable_track_modules is set.")
            _rank0_print("[Track4D-360] disable_track_modules=True: skipping persistent/dense Track4D modules.")
        elif self.dense_only_condition_modules:
            if not self.use_dense_branch:
                raise ValueError("--dense_only_condition_modules requires --use_dense_branch.")
            if self.use_track_null_token_fallback:
                raise ValueError("--dense_only_condition_modules cannot use persistent track null-token fallback.")
            if track_cfg is None:
                track_cfg = load_track_config()
            dit_dim = int(getattr(self.pipe.dit, "dim", 1536))
            self.pipe.disable_persistent_track_conditioning_unit()
            self.pipe.initialize_dense_only_control_modules(
                track_embed_dim=track_cfg.embed_dim,
                dit_dim=dit_dim,
                track_injection_mode=track_injection_mode,
                track_injection_block_mode=track_injection_block_mode,
            )
            persistent_out_dim = int(track_cfg.embed_dim)
            spatial_div = getattr(self.pipe, "height_division_factor", 16)
            dense_patch_size = spatial_div // int(track_cfg.get("conv_track_stride", 4))
            self.pipe.initialize_dense_target_control_encoder(
                in_channels=int(dense_in_channels),
                proj_dim=int(dense_proj_dim),
                out_dim=persistent_out_dim,
                patch_size=int(dense_patch_size),
                temporal_stride=4,
                num_residual_blocks=int(dense_num_residual_blocks),
                alpha_track=float(dense_alpha_track),
                use_depth_null_token=bool(use_depth_null_token),
            )
            if self.use_dense_null_token_fallback:
                self.pipe.initialize_dense_null_token()

            track_trainable = ["track_adapter"]
            if track_injection_mode == "per_block" and not self.freeze_track_block_injector:
                track_trainable.append("track_block_injector")
            else:
                track_trainable.append("track_projector")
            track_trainable.append("dense_target_control_encoder")
            if self.use_dense_null_token_fallback:
                track_trainable.append("dense_null_token")
            if bool(self.freeze_track4d_condition_modules):
                _rank0_print(
                    "[Track4D-360] freeze_track4d_condition_modules=True: "
                    "dense-only condition modules stay in the forward pass but "
                    "are not added to trainable_models."
                )
            elif not bool(gs_only_training):
                if trainable_models:
                    existing = [m.strip() for m in trainable_models.split(",")]
                    all_trainable = existing + track_trainable
                    trainable_models = ",".join(all_trainable)
                else:
                    trainable_models = ",".join(track_trainable)
            else:
                _rank0_print("[Track4D-360] gs_only_training=True: dense-only modules kept in pipeline but not trainable.")
        else:
            if track_cfg is None:
                track_cfg = load_track_config()
            dit_dim = int(getattr(self.pipe.dit, "dim", 1536))
            # Explicit per-channel include flags (new track configs). Only keys the
            # config actually declares are forwarded; the pipeline falls back to the
            # legacy channel-count map for any undeclared key, so existing configs
            # (which declare none) behave exactly as before. Short names match the
            # pipeline's track_include_flags lookup.
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
            self.pipe.initialize_track_modules(
                track_in_channels=track_cfg.get("in_channels", 5),
                track_embed_dim=track_cfg.embed_dim,
                track_num_freqs=track_cfg.num_freqs,
                track_hidden_dim=track_cfg.hidden_dim,
                track_num_mlp_layers=track_cfg.num_mlp_layers,
                dit_dim=dit_dim,
                track_injection_mode=track_injection_mode,
                track_injection_block_mode=track_injection_block_mode,
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
            if self.use_track_null_token_fallback:
                self.pipe.initialize_track_null_token()

            if self.use_dense_branch:
                # ``out_dim`` MUST equal the persistent track adapter's out_dim
                # for additive fusion to work.
                persistent_out_dim = int(getattr(self.pipe.track_adapter, "out_dim", track_cfg.embed_dim))
                spatial_div = getattr(self.pipe, "height_division_factor", 16)
                dense_patch_size = spatial_div // int(track_cfg.get("conv_track_stride", 4))
                self.pipe.initialize_dense_target_control_encoder(
                    in_channels=int(dense_in_channels),
                    proj_dim=int(dense_proj_dim),
                    out_dim=persistent_out_dim,
                    patch_size=int(dense_patch_size),
                    temporal_stride=4,
                    num_residual_blocks=int(dense_num_residual_blocks),
                    alpha_track=float(dense_alpha_track),
                    use_depth_null_token=bool(use_depth_null_token),
                )
                if self.use_dense_null_token_fallback:
                    self.pipe.initialize_dense_null_token()
            elif self.use_dense_null_token_fallback:
                raise ValueError("--use_dense_null_token_fallback requires --use_dense_branch.")

            track_trainable = ["track_adapter"]
            if self.pipe.track_embedder is not None:
                track_trainable.insert(0, "track_embedder")
            if self.pipe.track_identity_adapter is not None:
                track_trainable.append("track_identity_adapter")
            if track_injection_mode == "per_block" and not self.freeze_track_block_injector:
                track_trainable.append("track_block_injector")
            else:
                track_trainable.append("track_projector")
            if self.use_dense_branch:
                track_trainable.append("dense_target_control_encoder")
            if self.use_track_null_token_fallback:
                track_trainable.append("track_null_token")
            if self.use_dense_null_token_fallback:
                track_trainable.append("dense_null_token")
            if bool(self.freeze_track4d_condition_modules):
                _rank0_print(
                    "[Track4D-360] freeze_track4d_condition_modules=True: "
                    "track/dense condition modules stay in the forward pass but "
                    "are not added to trainable_models."
                )
            elif not bool(gs_only_training):
                if trainable_models:
                    existing = [m.strip() for m in trainable_models.split(",")]
                    all_trainable = existing + track_trainable
                    trainable_models = ",".join(all_trainable)
                else:
                    trainable_models = ",".join(track_trainable)
            else:
                _rank0_print("[Track4D-360] gs_only_training=True: track modules kept in pipeline "
                             "but FROZEN (not added to trainable_models).")

        # Add V2V-specific trainable params.
        # ``domain_embedding`` is required for normal multi-domain V2V. The
        # zero-init ``control_patch_embedding`` is only materialized for
        # control-latent paths such as the query-frame impulse control.
        v2v_required = [] if self.disable_domain_embedding else ["dit.domain_embedding"]
        if self.future_prediction and self.future_source_null:
            # Learned per-channel source-domain null for future frames. Wan2.1 VAE
            # latent z_dim = 16 (= input_latents channels in the loss). Created here
            # so it lands in trainable_models and the optimizer updates it; without
            # this it would stay zero-init. No-op for non-future runs (module
            # state_dict identical to today when future_prediction is off).
            init_future_null = getattr(self.pipe.dit, "initialize_future_source_null", None)
            if init_future_null is not None:
                init_future_null(16)
                v2v_required.append("dit.future_source_null")
        if self.use_query_frame_impulse_condition:
            v2v_required.append("dit.control_patch_embedding")
        # Channel-concat: multimodal patch embedding is trainable
        if self.mode_cfg.get("channel_concat"):
            num_mod = int(self.mode_cfg["num_channel_modalities"])
            v2v_required.append(f"dit._multimodal_patch_embeddings.{num_mod}")
        v2v_required.extend(
            get_plucker_camera_control_trainables(
                use_plucker_camera_control=self.use_plucker_camera_control,
                train_plucker_camera_control=self.train_plucker_camera_control,
                use_perlayer_camera_control=self.use_perlayer_camera_control,
                use_plucker_raymap_tokens=self.use_plucker_raymap_tokens,
                camera_injection_mode=self.camera_injection_mode,
            )
        )
        # Warped appearance conditioning trainables
        if self.warped_condition_mode == "latent_fuse":
            v2v_required.append("dit.warped_appearance_fuser")
            v2v_required.append("geometry_track_fuser")
        if self.use_vista4d_latent_streams:
            # pointcloud_only drops the redundant source-video stream (already
            # carried by x's source domain in width-concat V2V), so its two patch
            # embeddings must NOT be trainable — they never enter the forward graph
            # and would otherwise trip FSDP/DDP unused-parameter handling.
            v2v_required.extend([
                "dit.vista4d_pointcloud_patch_embedding",
                "dit.vista4d_pointcloud_mask_patch_embedding",
            ])
            if not self.vista4d_streams_pointcloud_only:
                v2v_required.extend([
                    "dit.vista4d_source_patch_embedding",
                    "dit.vista4d_source_mask_patch_embedding",
                ])
        if self.use_gen3c_warp_latent_channels:
            v2v_required.append("dit.gen3c_warp_condition_patch_embedding")

        if not bool(skip_v2v_required_trainables) and not bool(gs_only_training):
            for required in v2v_required:
                if required not in trainable_models:
                    trainable_models += f",{required}"

        # GS output: add GS trainable modules
        if self.use_gs_output:
            # When gs_zero_features=True, the DiT feature-head output is
            # zeroed at the loss site before entering the GS decoder. No
            # gradient ever reaches gs_feature_head → DDP would raise
            # "parameter did not receive grad". Skip making it trainable
            # in that case (matches the cache overfit, where the projector
            # is only instantiated for the `--feature_mode hidden` path and
            # is structurally absent in x0_mid / x0_final).
            if bool(gs_zero_features):
                gs_trainable = ["gs_decoder"]
                _rank0_print(
                    "[Track4D-360] gs_zero_features=True: skipping "
                    "dit.gs_feature_head from trainable_models (zero output → "
                    "zero gradient → would trigger DDP unused-parameter error)."
                )
            else:
                gs_trainable = ["dit.gs_feature_head", "gs_decoder"]
            if self.use_gs_feedback_training:
                # The widened feedback channels live in the main patch embedding,
                # so they must be trainable when FlashWorld-style feedback training is on.
                gs_trainable.append("dit.patch_embedding")
            # gs_only_training=True with no --trainable_models passed leaves this
            # None; the earlier v2v_required / track_trainable blocks were skipped
            # so nothing has filled it in yet. Normalize to "" before appending.
            if trainable_models is None:
                trainable_models = ""
            for gt in gs_trainable:
                if gt not in trainable_models:
                    trainable_models += (f",{gt}" if trainable_models else gt)

        # Koichi-style: directly finetune DiT self-attn/FFN submodules (full params).
        if dit_finetune_modules:
            for mod_name in dit_finetune_modules.split(","):
                mod_name = mod_name.strip()
                if mod_name:
                    pattern = f"dit.blocks.*.{mod_name}"
                    if pattern not in trainable_models:
                        trainable_models += f",{pattern}"
                        _rank0_print(f"[Track4D-360] Koichi-style finetune: {pattern} (full params, not LoRA)")

        # Defensive pre-freeze initialization. The camera modules must exist
        # before switch_pipe_to_training_mode() expands wildcard trainables and
        # calls freeze_except(); otherwise freeze_except fails while resolving
        # dit.blocks.*.cam_encoder. This call is idempotent when the earlier
        # camera-control init block already created the modules.
        #
        # Trigger on the presence of the camera modules in trainable_models, but
        # ONLY in vista4d_plucker mode. The trainable_models list is exactly what
        # freeze_except will resolve, so keying off it guarantees the modules
        # exist no matter how they were requested (explicit --trainable_models,
        # the ablate auto-add, or a mode flag), and can never desync from freeze.
        #
        # The mode guard is load-bearing: initialize_vista4d_plucker_camera_encoders
        # builds Linear encoders and asserts block.cam_encoder is Linear. But
        # `blocks.*.camera_projector` is ALSO added by projector + raymap-token
        # mode (ReCamMaster-style projector, ab12 and its descendants ab23/24/25),
        # where cam_encoder is Conv3d. Without the mode guard, a projector-mode
        # config that carries camera_projector trainables would wrongly invoke the
        # vista4d init and crash ("Found Conv3d; use a fresh model ..."). In
        # projector mode the camera modules are already created by the earlier
        # use_perlayer_camera_control init block, so no pre-freeze init is needed.
        trainable_models_for_check = trainable_models or ""
        _trainables_need_vista4d_cam = (
            self.camera_injection_mode == "vista4d_plucker"
            and (
                "blocks.*.cam_encoder" in trainable_models_for_check
                or "blocks.*.camera_projector" in trainable_models_for_check
            )
        )
        if (
            (self.use_perlayer_camera_control and self.camera_injection_mode == "vista4d_plucker")
            or _trainables_need_vista4d_cam
        ):
            from holo4d_runtime.models.wan_video_camera_controller import (
                initialize_vista4d_plucker_camera_encoders,
            )

            initialize_vista4d_plucker_camera_encoders(
                self.pipe.dit,
                plucker_dim=6,
                add_projector=True,
            )
            blocks = list(getattr(self.pipe.dit, "blocks", []) or [])
            missing_pre_freeze = []
            if not blocks:
                missing_pre_freeze.append("dit.blocks=missing")
            else:
                for module_name in ("cam_encoder", "camera_projector"):
                    if not hasattr(blocks[0], module_name):
                        missing_pre_freeze.append(f"blocks.0.{module_name}=missing")
            if missing_pre_freeze:
                raise RuntimeError(
                    "vista4d_plucker pre-freeze initialization failed: "
                    + ", ".join(missing_pre_freeze)
                )

        # camera-control base initialization MUST happen BEFORE switch_pipe_to_training_mode, because
        # switch_pipe_to_training_mode applies LoRA wrapping which renames
        # target-module weights from e.g. `blocks.0.self_attn.q.weight` to
        # `blocks.0.self_attn.q.base_layer.weight` (PEFT convention). If camera-control base
        # init runs after wrapping, the `k in target_state_keys` filter below
        # silently drops every LoRA-wrapped key and the self_attn/ffn base
        # weights end up coming from whatever generic loader preloaded, NOT from
        # camera-control base. LoRA deltas trained on top of camera-control base
        # would then be applied on top of a subtly-different generic loader base,
        # which is silently wrong.
        if camera_control_checkpoint is not None:
            control_sd, control_paths = _load_safetensors_state_dict(camera_control_checkpoint)
            # infer base in_dim from the initialised model (16 for 1.3B, 48 for 5B)
            base_in_dim = self.pipe.dit.patch_embedding.weight.shape[1]
            target_state_keys = set(self.pipe.dit.state_dict().keys())
            remapped = {}
            for k, v in control_sd.items():
                if k == "patch_embedding.weight":
                    if "patch_embedding_v2v.weight" in target_state_keys:
                        remapped["patch_embedding_v2v.weight"] = v
                    source_c = int(v.shape[1])
                    if source_c >= int(base_in_dim) * 2:
                        # camera-control runtime stores [cond | noise]. Track4D warm-starts from the
                        # noise half so the base patch embedding matches the target branch.
                        remapped["patch_embedding.weight"] = v[:, source_c - int(base_in_dim):]
                    else:
                        remapped["patch_embedding.weight"] = v[:, :base_in_dim]
                elif k == "patch_embedding.bias":
                    if "patch_embedding_v2v.bias" in target_state_keys:
                        remapped["patch_embedding_v2v.bias"] = v
                    remapped["patch_embedding.bias"] = v
                elif k not in remapped and k in target_state_keys:
                    remapped[k] = v
            remapped = self._adapt_non_lora_params_for_model(self.pipe.dit, remapped)
            m, u = self.pipe.dit.load_state_dict(remapped, strict=False)
            _rank0_print(
                f"[camera-control base initialization] base_in_dim={base_in_dim}, loaded "
                f"{len(control_paths)} safetensors file(s) from {camera_control_checkpoint}"
            )
            _rank0_print(
                "[camera-control base initialization] "
                + _summarize_state_dict_keys("missing", m)
                + " "
                + _summarize_state_dict_keys("unexpected", u)
            )

        # Switch to training mode (apply LoRA wrappers, load LoRA checkpoint if any).
        # Runs AFTER camera-control base initialization so LoRA deltas sit on top of camera-control base.
        self.switch_pipe_to_training_mode(
            pipe=self.pipe,
            trainable_models=trainable_models,
            lora_base_model=lora_base_model,
            lora_target_modules=lora_target_modules,
            lora_rank=lora_rank,
            lora_checkpoint=lora_checkpoint,
            lora_alpha=lora_alpha,
            freeze_lora_after_load=freeze_lora_after_load,
            task=task,
            train_sigma_shift=getattr(args, "train_sigma_shift", None),
        )

        # ``switch_pipe_to_training_mode`` restores LoRA/DiT tensors, but its
        # generic model loader does not own the pipeline-level ``track_*``
        # modules.  Restore those through the same migration-aware API used by
        # full-resume and eval.  This is intentionally allowed to find zero
        # track keys (for a pure LoRA checkpoint), but any incompatible track
        # tensor that is present must fail loudly rather than being skipped.
        if lora_checkpoint is not None and getattr(self.pipe, "track_adapter", None) is not None:
            self.pipe.load_track_checkpoint(lora_checkpoint, verbose=True)
        if self.use_perlayer_camera_control and self.camera_injection_mode == "vista4d_plucker":
            blocks = list(getattr(self.pipe.dit, "blocks", []) or [])
            if not blocks:
                raise RuntimeError("vista4d_plucker training found no DiT blocks after training-mode setup")
            missing_trainables = []
            for module_name in ("cam_encoder", "camera_projector"):
                module = getattr(blocks[0], module_name, None)
                if module is None:
                    missing_trainables.append(f"blocks.0.{module_name}=missing")
                    continue
                if not any(bool(getattr(param, "requires_grad", False)) for param in module.parameters()):
                    missing_trainables.append(f"blocks.0.{module_name}=frozen")
            if missing_trainables:
                raise RuntimeError(
                    "vista4d_plucker training requires trainable per-block camera modules, "
                    "but found: "
                    + ", ".join(missing_trainables)
                    + ". Add dit.blocks.*.cam_encoder and dit.blocks.*.camera_projector "
                    "to --trainable_models."
                )

        # Codex review F1: freeze_except() re-enables grads module-wide and
        # resume checkpoints restore a drifted nonzero null. Re-enforce the
        # hard-zero frozen-null policy AFTER training-mode switch + checkpoint
        # load so it wins. No-op unless TRACK4D360_TRACK_NULL_TOKEN_FROZEN=1.
        from holo4d.conditions.null_token_policy import (
            enforce_frozen_null_tokens_on_pipe,
        )
        enforce_frozen_null_tokens_on_pipe(self.pipe)

        # Persistent-identity arms MUST have trainable identity weights. The
        # identity branch is zero-initialized, so an untrained/frozen adapter
        # silently degenerates the arm into an exact legacy reproduction —
        # the experiment would "succeed" while testing nothing. Fail loudly.
        # Codex review F7: checking "any parameter" is insufficient — the
        # zero-init FINAL layer specifically must be trainable, otherwise the
        # output delta stays zero forever even with trainable earlier layers.
        if getattr(self.pipe, "persistent_identity_mode", "legacy") != "legacy":
            from holo4d.conditions.persistent_trajectory_identity import (
                persistent_identity_only_enabled,
            )

            identity_adapter = getattr(self.pipe, "track_identity_adapter", None)
            if identity_adapter is None:
                raise RuntimeError(
                    "persistent_identity_mode != legacy but track_identity_adapter "
                    "was not initialized before training-mode setup."
                )
            final_layer = identity_adapter.point_mlp[-1]
            if not (
                final_layer.weight.requires_grad
                and (final_layer.bias is None or final_layer.bias.requires_grad)
            ):
                raise RuntimeError(
                    "persistent_identity_mode != legacy but the ZERO-INITIALIZED "
                    "final identity layer (point_mlp[-1]) is not trainable — the "
                    "identity delta would stay exactly zero and this arm would "
                    "silently train as legacy. Add track_identity_adapter to "
                    "trainable modules."
                )
            if not any(p.requires_grad for p in identity_adapter.parameters()):
                raise RuntimeError(
                    "persistent_identity_mode != legacy but no track_identity_adapter "
                    "parameter is trainable."
                )
            # NOTE (2026-08-06 fix): the identity-only VALUE guard (identity
            # output + injector track columns must be nonzero) MOVED to run
            # AFTER checkpoint load — see `_verify_identity_only_after_load`
            # below. Running it here (the original bug) always saw the
            # zero-init adapter, because `resume_checkpoint` is applied later,
            # so every identity-only continuation (5866060/5866064) failed at
            # startup before step 0. The requires_grad checks above stay here
            # (they do not depend on loaded values).
            _rank0_print(
                "[Track4D-360] persistent-identity trainability verified: "
                f"{sum(p.numel() for p in identity_adapter.parameters() if p.requires_grad):,} "
                "trainable identity params (final layer trainable: True)"
            )

        # Resolve --fixed_step_frac → sigma. Must run AFTER
        # switch_pipe_to_training_mode because that call is what populates
        # pipe.scheduler.sigmas / pipe.scheduler.timesteps (via set_timesteps).
        # We build a temporary reference sigmas array using the scheduler's
        # own set_timesteps_fn at the requested --fixed_num_inference_steps
        # (default 50 — matches online GS-trainer spec), passing the same
        # shift that training is configured with, then pluck sigma at
        # step_idx = round(num_inference_steps * step_frac). This replicates
        # the online spec's "step 12 of 50 @ shift=5 → σ≈0.94" convention
        # exactly rather than approximating via index interpolation.
        # --fixed_sigma takes precedence if provided; otherwise resolve via
        # step_frac on a reference inference schedule. Both paths end up
        # populating self.fixed_sigma; inputs_shared["fixed_sigma"] threads
        # that value into FlowMatchV2VLoss + debug_latents.
        if self._requested_fixed_sigma is not None:
            self.fixed_sigma = float(self._requested_fixed_sigma)
            _shift = getattr(args, "train_sigma_shift", None)
            _step_idx = None  # not from a reference schedule
        elif self.fixed_step_frac is not None:
            _shift = getattr(args, "train_sigma_shift", None)
            _step_idx = int(round(self.fixed_num_inference_steps * float(self.fixed_step_frac)))
            _step_idx = max(0, min(_step_idx, self.fixed_num_inference_steps - 1))
            _kwargs = {"num_inference_steps": int(self.fixed_num_inference_steps)}
            if _shift is not None:
                _kwargs["shift"] = float(_shift)
            _tmp_sigmas, _tmp_ts = self.pipe.scheduler.set_timesteps_fn(**_kwargs)
            self.fixed_sigma = float(_tmp_sigmas[_step_idx].item())
        else:
            _shift = None
            _step_idx = None
        if self.fixed_sigma is not None:
            # For transparency, also resolve the nearest TRAINING-schedule
            # index (1000 steps under the same shift) so the log matches
            # what FlowMatchV2VLoss will actually pick.
            import torch as _torch
            _train_sigmas = self.pipe.scheduler.sigmas.detach().cpu().float()
            _train_idx = int(
                _torch.argmin((_train_sigmas - float(self.fixed_sigma)).abs()).item()
            )
            _actual_sigma = float(_train_sigmas[_train_idx].item())
            _rank0_print(
                f"[Track4D-360] Timestep pinned: "
                f"step_frac={self.fixed_step_frac} of "
                f"{self.fixed_num_inference_steps}-step trajectory "
                f"(shift={_shift}) → step_idx={_step_idx} → "
                f"target sigma={self.fixed_sigma:.4f}; in the "
                f"{len(_train_sigmas)}-step training schedule this picks "
                f"index {_train_idx} (actual sigma={_actual_sigma:.4f}). "
                f"Overrides random sampling in FlowMatchV2VLoss."
            )

        # Resume from a non-LoRA checkpoint (full-param finetuning).
        # Uses the same API as eval (create_eval_pipeline): load DiT params with
        # strict=False, then load_track_checkpoint for track modules.
        if resume_checkpoint is not None:
            from holo4d_runtime.core import load_state_dict
            ckpt_sd = load_state_dict(resume_checkpoint)
            ckpt_sd = self._adapt_non_lora_params_for_model(self.pipe.dit, ckpt_sd)

            # Channel-concat Stage B resume from Stage A: auto-expand 2-modal→3-modal
            # patch embedding. Detect by checking if checkpoint has a 2-modal embedding
            # but the current model expects 3-modal.
            if self.mode_cfg.get("channel_concat"):
                cur_num = int(self.mode_cfg["num_channel_modalities"])
                self._channel_concat_resume_expand_patch_embedding(
                    ckpt_sd, self.pipe.dit, cur_num,
                )

            # Load DiT params (blocks.*.self_attn/ffn, domain_embedding, control_patch_embedding)
            m, u = self.pipe.dit.load_state_dict(ckpt_sd, strict=False)
            loaded_dit = len(ckpt_sd) - len(m)
            _rank0_print(f"[Resume] Loaded {loaded_dit} DiT params from {resume_checkpoint}")
            if u:
                _rank0_print(f"[Resume] {len(u)} unexpected keys (ignored)")
            # Load track modules (embedder, adapter, injector/projector)
            self.pipe.load_track_checkpoint(resume_checkpoint, verbose=True)

            # Resurrect dead GS-feature-head weights from old checkpoints
            # trained under the cascade zero-init bug (diagnosed
            # 2026-04-16; see
            # doc/new_features/2026-04-16-kubric-dynamic-gs-head-dead-path-analysis.md).
            # Old checkpoints carry ``gs_feature_head.head.weight`` ==
            # 0 exactly, which stayed locked forever because the
            # decoder-side ``extra_conv_in.weight`` was also zero (no
            # gradient could flow). On resume, overwrite that dead tensor
            # with a fresh Kaiming init so gradient starts flowing. The
            # decoder side stays zero-init by design so the GS decoder
            # continues to behave like the VAE decoder at the first
            # resume step.
            gs_head_mod = getattr(self.pipe.dit, "gs_feature_head", None)
            if gs_head_mod is not None:
                try:
                    W = gs_head_mod.head.weight
                    if W.abs().max().item() == 0.0:
                        import torch.nn as _nn
                        with torch.no_grad():
                            _nn.init.kaiming_uniform_(W, a=5 ** 0.5)
                            if gs_head_mod.head.bias is not None:
                                _nn.init.zeros_(gs_head_mod.head.bias)
                        _rank0_print(
                            "[Resume/GS-fix] Detected dead gs_feature_head.head "
                            "(all-zero weight from pre-2026-04-16 cascade bug); "
                            "re-initialised with Kaiming so gradient starts flowing."
                        )
                except Exception as _gs_exc:
                    _rank0_print(f"[Resume/GS-fix] skipped ({_gs_exc})")

        # Stage-2 overlay: thin gs_head/decoder delta on top of teacher warm-start.
        # Routes ``dit.*`` keys onto pipe.dit, ``gs_decoder.*`` onto pipe.gs_decoder,
        # everything else onto pipe.dit (so legacy track-module keys still resolve).
        if gs_head_init_checkpoint is not None:
            from holo4d_runtime.core import load_state_dict
            init_sd = load_state_dict(gs_head_init_checkpoint)
            dit_sd, gsd_sd, other_sd = {}, {}, {}
            for k, v in init_sd.items():
                if k.startswith("gs_decoder."):
                    gsd_sd[k[len("gs_decoder."):]] = v
                elif k.startswith("dit."):
                    dit_sd[k[len("dit."):]] = v
                else:
                    other_sd[k] = v
            n_dit = n_gsd = n_other = 0
            if dit_sd:
                m, u = self.pipe.dit.load_state_dict(dit_sd, strict=False)
                n_dit = len(dit_sd) - len(u)
                if u:
                    _rank0_print(f"[GS-init] dit overlay: {len(u)} unexpected keys ignored")
            gs_dec_mod = getattr(self.pipe, "gs_decoder", None)
            if gsd_sd and gs_dec_mod is not None:
                m, u = gs_dec_mod.load_state_dict(gsd_sd, strict=False)
                n_gsd = len(gsd_sd) - len(u)
                if u:
                    _rank0_print(f"[GS-init] gs_decoder overlay: {len(u)} unexpected keys ignored")
            if other_sd:
                # Legacy track-module keys without prefix → use the same
                # migration-aware loader as resume/eval.  Do not swallow shape
                # or occupancy-mode mismatches: continuing with random track
                # weights would invalidate the training arm.
                self.pipe.load_track_checkpoint(gs_head_init_checkpoint, verbose=False)
                n_other = len(other_sd)
            _rank0_print(
                f"[GS-init] Loaded {n_dit} dit + {n_gsd} gs_decoder + {n_other} other "
                f"keys from {gs_head_init_checkpoint}"
            )

        # Identity-only VALUE guard — MUST run AFTER all checkpoint loading
        # (resume_checkpoint / gs_head_init), because the trained identity
        # output and injector track columns only become nonzero once the
        # ID-shared continuation checkpoint is applied. The original code ran
        # this before the resume load and failed every identity-only
        # continuation on the zero-init adapter (diagnosed 2026-08-06 from
        # jobs 5866060/5866064).
        if getattr(self.pipe, "persistent_identity_mode", "legacy") != "legacy":
            from holo4d.conditions.persistent_trajectory_identity import (
                persistent_identity_only_enabled,
            )
            if persistent_identity_only_enabled(self.pipe.persistent_identity_mode):
                _id_adapter = getattr(self.pipe, "track_identity_adapter", None)
                _final = _id_adapter.point_mlp[-1] if _id_adapter is not None else None
                if _final is None:
                    raise RuntimeError(
                        "TRACK4D360_TRACK_IDENTITY_ONLY=1 but track_identity_adapter "
                        "is missing after setup."
                    )
                final_magnitude = float(_final.weight.detach().abs().sum())
                if _final.bias is not None:
                    final_magnitude += float(_final.bias.detach().abs().sum())
                if final_magnitude == 0.0:
                    raise RuntimeError(
                        "TRACK4D360_TRACK_IDENTITY_ONLY=1 found an all-zero identity "
                        "output layer after checkpoint load. Identity-only cannot "
                        "start from the double-zero identity/injector initialization; "
                        "resume from a trained persistent-ID shared checkpoint."
                    )
                _injector = getattr(self.pipe, "track_block_injector", None)
                _projector = getattr(self.pipe, "track_projector", None)
                injector_track_magnitude = 0.0
                if _injector is not None:
                    for encoder in _injector.encoders:
                        if _injector.mode == "concat_project":
                            track_weight = encoder.weight[:, -int(_injector.track_dim):]
                        else:
                            track_weight = encoder.weight
                        injector_track_magnitude += float(track_weight.detach().abs().sum())
                elif _projector is not None:
                    first_layer = _projector.proj[0]
                    track_weight = first_layer.weight[:, -int(_projector.track_dim):]
                    injector_track_magnitude = float(track_weight.detach().abs().sum())
                if injector_track_magnitude == 0.0:
                    raise RuntimeError(
                        "TRACK4D360_TRACK_IDENTITY_ONLY=1 found all-zero injector "
                        "track columns after checkpoint load. Resume from a trained "
                        "hybrid persistent-ID checkpoint."
                    )
                _rank0_print(
                    "[Track4D-360] identity-only continuation verified: "
                    f"resumed identity output magnitude={final_magnitude:.6g}, "
                    f"injector track-column magnitude={injector_track_magnitude:.6g}"
                )

        # Debug: verify trainable parameters exist after setup
        n_trainable = sum(p.requires_grad for p in self.pipe.parameters())
        n_total = sum(1 for _ in self.pipe.parameters())
        print(f"[Track4D-360] Trainable params: {n_trainable}/{n_total}", flush=True)
        if n_trainable == 0:
            print("[Track4D-360] WARNING: No trainable parameters! Check freeze_except logic.", flush=True)
            # List what was supposed to be trainable
            print(f"[Track4D-360] trainable_models arg was: {trainable_models}", flush=True)

        n_profiled_conv3d_modules = enable_track4d_conv3d_module_profile_spans(self.pipe)
        if n_profiled_conv3d_modules:
            _rank0_print(
                "[Track4D-360] TRACK4D360_PROFILE_CONV3D_MODULES=1: "
                f"wrapped {n_profiled_conv3d_modules} Conv3d modules with profiler spans."
            )

        # Hide frozen VAE from FSDP's module tree.
        # Why: pipe.vae access path is `.encode()`/`.decode()` (plain Python methods),
        # never `vae(...)` (__call__). With SIZE_BASED_WRAP at 10M threshold the VAE
        # gets sharded under FULL_SHARD, but no `__call__` ever triggers the FSDP
        # pre-forward hook to gather shards — params stay at shape (0,) on some
        # ranks, producing "weight should have at least 3 dimensions" / "size of
        # tensor a (96) must match tensor b (0)" inside Encoder3d.forward.
        # How to apply: detach `vae` from `self.pipe._modules` and re-attach as a
        # plain instance attribute via `object.__setattr__`, bypassing
        # torch.nn.Module.__setattr__'s auto-registration. FSDP traverses _modules,
        # so the VAE becomes invisible to it; ordinary attribute lookup still works.
        _vae = self.pipe.vae
        del self.pipe.vae
        object.__setattr__(self.pipe, "vae", _vae)

        # Like VAE, Wan's frozen image encoder is called through encode_image()
        # instead of Module.__call__. Keep it outside FSDP's module tree so no
        # sharded parameter is accessed without a pre-forward all-gather.
        if self.i2v_target_condition:
            _image_encoder = self.pipe.image_encoder
            del self.pipe.image_encoder
            object.__setattr__(self.pipe, "image_encoder", _image_encoder)

        # Hide small FROZEN DiT root submodules that are accessed directly via
        # `dit.X(...)` in model_fn_wan_video_v2v (bypassing dit.__call__).
        # Why: `dit.time_embedding(...)`, `dit.text_embedding(...)`,
        # `dit.time_projection(...)`, `dit.head(x, t)`, `dit.img_emb(...)`,
        # `dit.patch_embedding(...)` all skip the DiT's forward hook. SIZE_BASED
        # wraps any Linear >=10M inside these as its own FSDP unit, but the
        # OTHER Linears (e.g. time_embedding's 256→5120 Linear at ~1.3M) end up
        # in some outer flat_param whose pre-forward hook isn't reliably gathering
        # the view through Sequential iteration on torch 2.10 — causing
        # "mat2 must be a matrix, got 1-D tensor" inside F.linear.
        # Fix: same object.__setattr__ trick as VAE — pull these submodules out
        # of `dit._modules` so FSDP doesn't see them at all. They stay as plain
        # unsharded full-precision tensors on each rank. Total cost ~233 MB at
        # bf16 (mostly time_projection's 157 M Linear), trivial vs. the 14 B DiT.
        # All listed modules are FROZEN, so no DDP/FSDP gradient sync is needed.
        # Note: `dit.blocks` and `dit.control_adapter` are NOT hidden — blocks
        # carry the bulk of the weight (sharding mandatory) and control_adapter
        # is invoked via __call__ (`dit.control_adapter(x)`) so its FSDP unit
        # hook fires correctly. Same for trainable `dit.domain_embedding` and
        # `dit.control_patch_embedding` which are tiny and live in the root
        # FSDP unit's flat_param (gathered when `model(data)` fires).
        _dit = self.pipe.dit
        for _name in (
            "time_embedding",
            "time_projection",
            "text_embedding",
            "head",
            "img_emb",
            "patch_embedding",
            "ref_conv",
            "pre_dit_cam_encoder",
        ):
            _mod = getattr(_dit, _name, None)
            if _mod is None or not isinstance(_mod, torch.nn.Module):
                continue
            delattr(_dit, _name)
            object.__setattr__(_dit, _name, _mod)

    @staticmethod
    def _channel_concat_resume_expand_patch_embedding(ckpt_sd, dit, cur_num_modalities):
        """Handle Stage A→B patch embedding expansion during checkpoint resume.

        If the checkpoint contains a 2-modal patch embedding but the current model
        expects 3-modal, auto-expand by copying the 2-modal weights and zero-initializing
        the new track channels. Also handles loading a Stage 1 (no multimodal) checkpoint
        into a Stage A model.
        """
        cur_key = f"_multimodal_patch_embeddings.{cur_num_modalities}.weight"
        cur_bias_key = f"_multimodal_patch_embeddings.{cur_num_modalities}.bias"
        in_dim = int(dit.in_dim)

        if cur_key in ckpt_sd:
            # Exact match: checkpoint already has the right num_modalities
            ckpt_c = ckpt_sd[cur_key].shape[1]
            expected_c = cur_num_modalities * in_dim
            if ckpt_c == expected_c:
                print(f"[Resume] Channel-concat: checkpoint has matching {cur_num_modalities}-modal "
                      f"patch embedding ({ckpt_c}ch)")
                return
            # Channel mismatch within same key — shouldn't happen, warn and fall through
            print(f"[Resume] WARNING: checkpoint {cur_key} has {ckpt_c}ch, "
                  f"expected {expected_c}ch for {cur_num_modalities} modalities")

        # Search for a smaller multimodal embedding in the checkpoint
        ckpt_num = None
        ckpt_weight_key = None
        for n in range(cur_num_modalities - 1, 0, -1):
            k = f"_multimodal_patch_embeddings.{n}.weight"
            if k in ckpt_sd:
                ckpt_num = n
                ckpt_weight_key = k
                break

        if ckpt_num is not None:
            # Found a smaller multimodal embedding → expand
            print(f"[Resume] Channel-concat: expanding {ckpt_num}-modal → "
                  f"{cur_num_modalities}-modal patch embedding (zero-init new channels)")
            old_weight = ckpt_sd.pop(ckpt_weight_key)
            old_bias_key = f"_multimodal_patch_embeddings.{ckpt_num}.bias"
            old_bias = ckpt_sd.pop(old_bias_key, None)

            # Build expanded weight
            new_weight = torch.zeros(
                old_weight.shape[0],  # out_channels (dim)
                cur_num_modalities * in_dim,
                *old_weight.shape[2:],  # kernel spatial dims
                device=old_weight.device,
                dtype=old_weight.dtype,
            )
            new_weight[:, : ckpt_num * in_dim] = old_weight
            ckpt_sd[cur_key] = new_weight

            if old_bias is not None:
                ckpt_sd[cur_bias_key] = old_bias
            # Also update the model's live parameter to match
            dit.expand_multimodal_patch_embedding(ckpt_num, cur_num_modalities)
        else:
            # No multimodal embedding in checkpoint at all (e.g., Stage 1 → Stage A).
            # The model's _ensure_patch_embedding_multimodal already initialized from
            # patch_embedding.weight tiling, which is the right starting point.
            print(f"[Resume] Channel-concat: no multimodal patch embedding in checkpoint. "
                  f"Using tiled init from base patch_embedding for {cur_num_modalities} modalities.")

    # ---- Track4D-360 cached-T5 prompt embedding setup -----------------------
    # The dataloader puts a per-chunk Wan-T5 embedding into ``data["prompt_emb"]``
    # when a precomputed safetensors cache is present next to the caption JSONs
    # (see holo4d/precompute_t5_cache.py). For the small set of prompts
    # that the cache doesn't cover (the empty CFG-drop string and the
    # per-dataset DEFAULT_PROMPTs), we encode once at training init and stash
    # them on ``pipe.precomputed_prompt_embs``. WanVideoUnit_PromptEmbedder
    # looks up that table before it falls through to an online T5 forward.
    #
    # Setting the env var ``HOLO4D_OFFLOAD_T5_TO_CPU=1`` ALSO moves T5 to CPU
    # immediately after the precompute. With a complete cache the online
    # fallback never fires; if it does, the unit moves T5 back to the device
    # for that single encode (see wan_video.py:WanVideoUnit_PromptEmbedder).
    # That's safe but slower per cache miss — only flip the env var when you
    # trust the cache covers all data.
    # Soft strict-cache mode (2026-04-30, evening): include the empty string
    # for drop-text CFG dropout, plus the unified Track4D-360 NVS default
    # prompt. The default is task-meaningful — "Generate the novel view
    # trajectory video given the input video." — replacing the previous
    # generic "A multi-view video scene." / "A dynamic multi-view video
    # scene." placeholders that didn't describe the actual training task.
    #
    # Every dataset (Syn4D, ReCam/SynCam, Kubric, StaticScene/cut3r) now
    # uses this same string as its fallback when a per-sample caption is
    # missing (cache miss, sanitize-empty, etc.) — see
    # holo4d_runtime/data/stream3r/dust3r/datasets_cut3r/{track4d360_mixed_nvs,
    # syn4d_nvs}.py:DEFAULT_PROMPT.
    #
    # Why include the default here even though we want every sample to have
    # a real cached caption: precomputing the default's embedding once means
    # the strict-cache check at wan_video.py:518 has a table hit for
    # legitimate edge-case fallbacks (rare scene with missing caption.txt,
    # a precompute that crashed mid-encode, etc.) instead of crashing on
    # every fallback. The hard-error still fires for any UNKNOWN prompt
    # text, which is the actual coverage-bug surface we want to catch.
    _DEFAULT_PROMPTS_TO_PRECOMPUTE = (
        "",                                                                  # drop-text CFG dropout — REQUIRED
        "Generate the novel view trajectory video given the input video.",   # unified NVS task default
    )

    def _setup_prompt_emb_cache(self):
        import os as _os
        pipe = self.pipe
        text_encoder = getattr(pipe, "text_encoder", None)
        tokenizer = getattr(pipe, "tokenizer", None)
        if text_encoder is None or tokenizer is None:
            print("[T5 cache] text_encoder or tokenizer missing on pipe; skipping precompute", flush=True)
            return

        device = next(text_encoder.parameters()).device
        precomputed = {}
        with torch.no_grad():
            for prompt_str in self._DEFAULT_PROMPTS_TO_PRECOMPUTE:
                ids, mask = tokenizer(prompt_str, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device)
                seq_lens = mask.gt(0).sum(dim=1).long()
                emb = text_encoder(ids, mask)  # [1, seq_len, dim]
                # Match the online encode's mask zero-out so cached/online
                # outputs are identical for these strings.
                for i, v in enumerate(seq_lens):
                    emb[:, v:] = 0
                # Strip batch dim and trim to non-pad length, matching the
                # safetensors-cache layout used by the per-chunk path. The
                # unit's `_format_cached_emb` re-pads to CONTEXT_SEQ_LEN.
                L = max(int(seq_lens.item()), 1)
                precomputed[prompt_str] = emb[0, :L].to(dtype=torch.bfloat16, device="cpu").contiguous().clone()

        pipe.precomputed_prompt_embs = precomputed
        keys_preview = ", ".join(repr(k[:20]) for k in precomputed.keys())
        print(f"[T5 cache] precomputed {len(precomputed)} default prompt embeddings: {keys_preview}", flush=True)

        # ---- T5 disposal policy (default = DELETE) -----------------------
        # Why DELETE is the default (changed 2026-04-28):
        #   The legacy offload approach was `pipe.text_encoder.to("cpu")`,
        #   but text_encoder is registered as a submodule of `self.pipe`,
        #   which is itself a submodule of the trainer (`self`). When the
        #   runner calls `accelerator.prepare(self)` AFTER this method runs,
        #   accelerate cascades `.to(device)` through `self.modules()` and
        #   pulls T5 right back to GPU. The HBM probe printed `~3 GB` here
        #   because it ran *between* offload and accelerator.prepare; by step
        #   1 of training, T5 was back on GPU and the offload was undone.
        #
        #   Deleting (`pipe.text_encoder = None`) survives accelerator.prepare
        #   because there's nothing left to cascade `.to()` through. With the
        #   strict-cache guard in WanVideoUnit_PromptEmbedder, no online
        #   encode can fire to need T5 back — the 3 default-prompt embeddings
        #   on pipe.precomputed_prompt_embs + the per-sample safetensors
        #   cache cover every prompt the trainer sees.
        #
        # Env var resolution (precedence top→bottom):
        #   HOLO4D_KEEP_T5=1            → keep T5 on GPU (debug / legacy)
        #   HOLO4D_OFFLOAD_T5_TO_CPU=0  → keep T5 on GPU (legacy alias for the
        #                              old "no offload" mode; kept so existing
        #                              `..._t5_on_gpu.sh` ablation wrapper
        #                              still works without edits)
        #   anything else (including unset / =1)
        #                            → DELETE T5 (default). The legacy
        #                              HOLO4D_OFFLOAD_T5_TO_CPU=1 maps here too,
        #                              so existing offload launchers now
        #                              actually save HBM end-to-end.
        keep_env = _os.environ.get("HOLO4D_KEEP_T5", "").strip().lower()
        offload_env = _os.environ.get("HOLO4D_OFFLOAD_T5_TO_CPU", "").strip().lower()
        delete_env = _os.environ.get("HOLO4D_DELETE_T5_AFTER_PRECOMPUTE", "").strip().lower()

        keep_t5 = (
            keep_env in ("1", "true", "yes", "on")
            or offload_env in ("0", "false", "no", "off")
        )

        if keep_t5:
            print(
                "[T5 cache] keeping pipe.text_encoder on GPU "
                f"(HOLO4D_KEEP_T5={keep_env or 'unset'}, HOLO4D_OFFLOAD_T5_TO_CPU={offload_env or 'unset'}).",
                flush=True,
            )
        else:
            # Default path: drop T5 entirely. The 3 default prompt embeddings
            # already live on `pipe.precomputed_prompt_embs`; the per-sample
            # cache covers everything else; the strict-cache hard error in
            # WanVideoUnit_PromptEmbedder ensures no online encode fires.
            # Saves ~10 GB GPU + ~10 GB CPU.
            try:
                import gc as _gc
                pipe.text_encoder.to("meta")  # release tensor storage on every device
            except Exception:
                pass
            pipe.text_encoder = None
            try:
                _gc.collect()
            except Exception:
                pass
            torch.cuda.empty_cache()
            why = []
            if delete_env in ("1", "true", "yes", "on"):
                why.append("HOLO4D_DELETE_T5_AFTER_PRECOMPUTE=1 (explicit)")
            if offload_env in ("1", "true", "yes", "on"):
                why.append("HOLO4D_OFFLOAD_T5_TO_CPU=1 (legacy → maps to delete; .to('cpu') was reverted by accelerator.prepare)")
            if not why:
                why.append("default policy (set HOLO4D_KEEP_T5=1 to opt out)")
            print(
                "[T5 cache] dropped pipe.text_encoder entirely "
                "(cache table is now the only prompt path). Reason: "
                + "; ".join(why),
                flush=True,
            )

        # HBM probe — the only reliable proof that the offload took effect.
        # `memory_allocated` = live CUDA tensor bytes (drops by ~10 GB when T5
        # moves to CPU); `memory_reserved` = caching-allocator high-water
        # (matches nvidia-smi / wandb system.gpu.*.memory) and converges
        # between offload and on-gpu runs after activations warm up. Stash on
        # self so the training loop can wandb.log() it once at step 0.
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            self._hbm_after_t5_setup = {
                "hbm/post_t5_allocated_gb": allocated_gb,
                "hbm/post_t5_reserved_gb": reserved_gb,
            }
            print(
                f"[HBM after T5 setup] allocated_gb={allocated_gb:.2f} "
                f"reserved_gb={reserved_gb:.2f} (offload_env={offload_env or 'unset'})",
                flush=True,
            )

    def get_pipeline_inputs(self, data):
        """Convert batch data to pipeline inputs."""
        import random

        prompt = data["prompt"]
        # Cached T5 embedding from the dataloader (None when no chunk caption
        # matched, or when the dataset uses DEFAULT_PROMPT directly). Becomes
        # `prompt_emb` in inputs_posi so WanVideoUnit_PromptEmbedder can skip
        # the online T5 forward; falls back to pipe.precomputed_prompt_embs
        # (set up in __init__) when None and prompt is a known default string.
        prompt_emb_cached = data.get("prompt_emb", None)
        if self.training and random.random() < self.drop_text_prob:
            prompt = ""
            # Empty-prompt embedding lives in pipe.precomputed_prompt_embs[""];
            # leaving prompt_emb_cached at whatever it was means the unit will
            # match the prompt-string fallback table on "". Clear it explicitly
            # so we don't accidentally feed a real-caption embedding for what's
            # supposed to be the unconditional CFG-drop branch.
            prompt_emb_cached = None

        # -- Conditioning dropout: drop track/dense so model learns plucker-only --
        # Like text CFG dropout (drop_text_prob → prompt=""), we keep the
        # computation graph identical and only change the DATA:
        #   track drop  → has_tracks=False (track unit emits learned null token, still runs)
        #   dense drop  → has_depth=False + zero source RGB/all-False valid mask
        # This avoids DDP desync — all units always run, same params get grads.
        # Each rank rolls independently (like drop_text_prob), which is safe
        # because the graph structure never changes.
        def _coerce_batch_bool(x, default=False):
            if x is None:
                return bool(default)
            if torch.is_tensor(x):
                return bool(x.bool().all().item())
            if isinstance(x, np.ndarray):
                return bool(np.asarray(x).astype(bool).all())
            if isinstance(x, (list, tuple)):
                return bool(all(bool(v) for v in x))
            return bool(x)

        _drop_tracks = False
        _drop_dense = False
        _camera_only_tier = _coerce_batch_bool(data.get("camera_only_tier", False), default=False)
        if _camera_only_tier:
            # ReCam/SynCam camera-only samples are intended to train plucker
            # control without Track4D or dense-view cues. Keep the graph
            # shape intact, but force the same data-level no-op path every
            # time instead of relying on random conditioning dropout.
            _drop_tracks = True
            _drop_dense = True
        if self.training and (self.cond_drop_geo_prob > 0 or self.cond_drop_track_prob > 0 or self.cond_drop_dense_prob > 0):
            if _camera_only_tier:
                pass
            elif random.random() < self.cond_drop_geo_prob:
                _drop_tracks = True
                _drop_dense = True
            else:
                if random.random() < self.cond_drop_track_prob:
                    _drop_tracks = True
                if random.random() < self.cond_drop_dense_prob:
                    _drop_dense = True
        # Always write dropout metrics (even when 0) so every rank has the
        # same metric keys — accelerator.gather() runs per-key and mismatched
        # keys across ranks cause NCCL desync.
        self._pending_input_metrics = {}
        if self.cond_drop_geo_prob > 0 or self.cond_drop_track_prob > 0 or self.cond_drop_dense_prob > 0:
            self._pending_input_metrics["cond_dropout/drop_tracks"] = float(_drop_tracks)
            self._pending_input_metrics["cond_dropout/drop_dense"] = float(_drop_dense)

        # Convert numpy arrays to tensors (dataset returns numpy)
        def _to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return x

        def _zero_like_dense_rgb(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return torch.zeros_like(x)
            if isinstance(x, np.ndarray):
                return np.zeros_like(x)
            if isinstance(x, (list, tuple)):
                zero_frames = []
                for frame in x:
                    if isinstance(frame, Image.Image):
                        zero_frames.append(Image.new("RGB", frame.size))
                    else:
                        arr = np.asarray(frame)
                        zero_frames.append(np.zeros_like(arr))
                return zero_frames
            return x

        def _zero_like_dense_valid(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return torch.zeros_like(x, dtype=torch.bool)
            if isinstance(x, np.ndarray):
                return np.zeros_like(x, dtype=bool)
            if isinstance(x, (list, tuple)):
                return [False for _ in x]
            return x

        # Freeze-time: drop source video conditioning to prevent trivial copy.
        # When drop_source_cond=True, replace source video with None → T2V mode.
        # Note: per-sample mixed batches (tensor([True, False])) are not supported —
        # this is a whole-batch flag. With batch_size=1 (our default), this is fine.
        drop_source_cond = data.get("drop_source_cond", False)
        if torch.is_tensor(drop_source_cond):
            # For batch_size=1: scalar or 1-element tensor
            # For batch_size>1: all() — drop only if ALL samples request it
            drop_source_cond = bool(drop_source_cond.all())
        if bool(getattr(self, "force_t2v_drop_source_cond", False)):
            drop_source_cond = True
        source_video = None if drop_source_cond else data["video"]

        # `prompt_emb` is consumed by WanVideoUnit_PromptEmbedder.process which
        # skips the T5 forward when this is a non-None bf16 tensor. None falls
        # through to the per-prompt-string fallback table (pipe.precomputed_
        # prompt_embs) and finally to online T5 encode if neither is cached.
        inputs_posi = {"prompt": prompt, "prompt_emb": prompt_emb_cached}
        inputs_nega = {}
        inputs_shared = {
            # Source video (condition) — None when freeze-time drop is active
            "input_video": source_video,
            # The current 49-frame input_video remains the full source-domain
            # V2V condition.  WanVideoUnit_TargetImageCondition additionally
            # selects target_video[0] for the target-domain boundary condition;
            # input_image is reserved for the previous generated target frame
            # during chunk-autoregressive inference.
            "input_image": None,
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Target video (prediction target)
            "target_video": data["target_video"],
            # Track conditioning (always passed; has_tracks=False zeros the output)
            "tracks_world": _to_tensor(data["tracks_world"]),
            "target_cam2world": _to_tensor(data["target_cam2world"]),
            "target_intrinsics": _to_tensor(data["target_intrinsics"]),
            # Pipeline config
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "v2v_concat_mode": None if self.raw_t2v_target_only else self.v2v_concat_mode,
            "raw_t2v_target_only": self.raw_t2v_target_only,
            "use_domain_embedding": not self.disable_domain_embedding,
            "use_query_frame_impulse_condition": self.use_query_frame_impulse_condition,
            # Future prediction (temporal extrapolation). No-ops in loss/pipeline
            # when future_prediction is False → default V2V path unchanged.
            "future_prediction": self.future_prediction,
            "future_num_latent_frames": int(self.future_num_latent_frames),
            "future_loss_weight": float(self.future_loss_weight),
            "track_query_idx": data.get("track_query_idx", 0),
            "target_track_use_source_stats": self.mode_cfg["target_track_use_source_stats"],
            "target_depth_log_cap": self.target_depth_log_cap,
            "target_depth_norm_type": self.target_depth_norm_type,
            "target_depth_max_disparity": self.target_depth_max_disparity,
            # Disable features not used in Track4D-360
            "use_perlayer_camera_control": self.use_perlayer_camera_control,
            "use_pre_dit_camera_injection": False,
            "use_noise_camera_injection": False,
            "multimodality_nvs": self.mode_cfg["multimodality_nvs"],
            "multimodality_tracking": self.mode_cfg["multimodality_tracking"],
            "multimodality_track_depth": self.mode_cfg["multimodality_track_depth"],
            "drop_video_prob": 0.0,
            "drop_t2v_prob": None,
            "drop_i2v_prob": None,
            "input_video_noise_strength": 0.0,
            "input_video_noise_random_max": 500,
            "use_target_image_condition": self.i2v_target_condition,
            "i2v_condition_drop_prob": self.i2v_condition_drop_prob,
            "i2v_condition_noise_level": self.i2v_condition_noise_level,
            "camera_condition_scale": 1.0,
            "camera_injection_mode": self.camera_injection_mode,
            "camera_plucker_encoder_mode": self.camera_plucker_encoder_mode,
            "use_plucker_raymap_tokens": self.use_plucker_raymap_tokens,
            "use_ray_origin": self.plucker_use_ray_origin,
            # Warped appearance conditioning
            "warped_condition_mode": self.warped_condition_mode,
            "warped_appearance_fusion": self.warped_appearance_fusion,
            "warped_geom_only_dense": self.warped_geom_only_dense,
            # Multi-frame hybrid mode (Lyra-2 forward_warp_multiframes style):
            # when on, the warped-appearance unit fuses all T source frames
            # into each target view at static pixels, and falls back to per-
            # frame warp at dynamic pixels. Static / dynamic comes from the
            # per-pixel motion range of `tracks_world` (Kubric_NVS / Syn4D).
            "multi_frame_warped_appearance": self.multi_frame_warped_appearance,
            "multi_frame_motion_threshold": self.multi_frame_motion_threshold,
            # Dense-branch warp backend. "zbuffer" (default) = classic per-
            # frame hard z-buffer scatter via build_dense_target_reprojection.
            # "softsplat" = GEN3C-style per-frame bilinear soft-splat.
            # "hybrid_softsplat" = Lyra-2/GEN3C bilinear soft-splat with
            # multi-frame fusion at static pixels + per-frame fallback at
            # dynamic pixels. Same channel layout (RGB|depth|valid), same
            # encoder, drop-in. See dense_target_reprojection_builder.py.
            "dense_warp_mode": self.dense_warp_mode,
            "dense_warp_motion_threshold": self.dense_warp_motion_threshold,
            "dense_hybrid_disable_dynamic_masks": self.dense_hybrid_disable_dynamic_masks,
            "use_vista4d_latent_streams": self.use_vista4d_latent_streams,
            "vista4d_streams_pointcloud_only": self.vista4d_streams_pointcloud_only,
            "use_gen3c_warp_latent_channels": self.use_gen3c_warp_latent_channels,
        }
        if os.environ.get("TRACK4D360_ADAPTER_MEM_PROFILE", "0") == "1":
            profile_keys = (
                "dataset_name",
                "dataset_type",
                "scene_id",
                "sequence_id",
                "source_cam",
                "target_cam",
                "track_density_mode",
                "track_sparse_stride",
                "track_sparse_base_stride",
                "track_query_idx",
                "track_query_indices",
                "track_query_count",
            )
            inputs_shared["track_profile_meta"] = {
                key: data[key]
                for key in profile_keys
                if key in data
            }
        if data.get("tracks_world_valid_mask") is not None:
            inputs_shared["tracks_world_valid_mask"] = _to_tensor(data["tracks_world_valid_mask"])
        if data.get("tracks_query_idx_per_identity") is not None:
            inputs_shared["tracks_query_idx_per_identity"] = _to_tensor(
                data["tracks_query_idx_per_identity"]
            ).long()
        identity_sampling_seed = self._track_identity_sampling_seed(data)
        if identity_sampling_seed is not None:
            inputs_shared["track_identity_sampling_seed"] = identity_sampling_seed

        # Channel-concat multi-modal config
        inputs_shared["channel_concat"] = bool(self.mode_cfg.get("channel_concat", False))
        inputs_shared["channel_modalities"] = self.mode_cfg.get("channel_modalities", None)
        inputs_shared["num_channel_modalities"] = self.mode_cfg.get("num_channel_modalities", 0)
        inputs_shared["use_plucker_camera_control"] = getattr(self, "use_plucker_camera_control", False)
        inputs_shared["enable_v2v_plucker_camera_control"] = getattr(
            self, "enable_v2v_plucker_camera_control", False
        )
        inputs_shared["use_track_null_token_fallback"] = getattr(
            self, "use_track_null_token_fallback", False
        )
        inputs_shared["use_dense_null_token_fallback"] = getattr(
            self, "use_dense_null_token_fallback", False
        )

        if _camera_only_tier and self.use_query_frame_impulse_condition:
            # Negative query index means "encode an all-zero impulse video"
            # so the control-patch path stays alive without leaking a query
            # frame prior into camera-only plucker batches.
            q = data.get("track_query_idx", 0)
            if torch.is_tensor(q):
                inputs_shared["track_query_idx"] = torch.full_like(q, -1)
            elif isinstance(q, np.ndarray):
                inputs_shared["track_query_idx"] = np.full_like(q, -1)
            elif isinstance(q, (list, tuple)):
                inputs_shared["track_query_idx"] = [-1 for _ in q]
            else:
                inputs_shared["track_query_idx"] = -1

        # Timestep pinning: FlowMatchV2VLoss resolves `fixed_sigma` to the
        # nearest scheduler index (deterministic, single t per step). Sigma
        # semantics are load-bearing here — see the docstring of
        # add_gs_training_timestep_args for why we don't use index fractions.
        if getattr(self, "fixed_sigma", None) is not None:
            inputs_shared["fixed_sigma"] = float(self.fixed_sigma)
        # Zero out the FM MSE in FlowMatchV2VLoss when DiT is frozen.
        inputs_shared["skip_diffusion_loss"] = bool(getattr(self, "skip_diffusion_loss", False))

        # Source depth video (per-frame, for channel-concat source-side encoding)
        source_depth_video = data.get("source_depth_video")
        if source_depth_video is not None:
            inputs_shared["source_depth_video"] = _to_tensor(source_depth_video)
        if getattr(self, "enable_v2v_plucker_camera_control", False):
            source_cam2world = data.get("source_cam2world")
            source_intrinsics = data.get("source_intrinsics")
            if source_cam2world is not None and source_intrinsics is not None:
                inputs_shared["source_camera"] = {
                    "c2w_opencv": _to_tensor(source_cam2world),
                    "intrinsics": _to_tensor(source_intrinsics),
                }
            if data.get("target_cam2world") is not None and data.get("target_intrinsics") is not None:
                inputs_shared["target_camera"] = {
                    "c2w_opencv": _to_tensor(data["target_cam2world"]),
                    "intrinsics": _to_tensor(data["target_intrinsics"]),
                }

        # ------------------------------------------------------------------
        # Dense GEN3C-style target-space control branch inputs.
        # Forwarded only when present in the data dict; the pipeline unit
        # gracefully no-ops if any required tensor is missing.
        #
        # When --use_dense_branch is on we LOUDLY warn at the first batch if
        # the dataset isn't producing the required fields — otherwise the
        # dense branch silently contributes nothing and we waste a run.
        # ------------------------------------------------------------------
        _use_dense = bool(getattr(self, "use_dense_branch", False))
        inputs_shared["use_dense_branch"] = _use_dense
        inputs_shared["use_vista4d_latent_streams"] = bool(getattr(self, "use_vista4d_latent_streams", False))
        inputs_shared["vista4d_streams_pointcloud_only"] = bool(getattr(self, "vista4d_streams_pointcloud_only", False))
        inputs_shared["use_gen3c_warp_latent_channels"] = bool(getattr(self, "use_gen3c_warp_latent_channels", False))
        for src_key, dst_key in (
            ("source_rgb_video", "source_rgb_video"),
            ("source_depth_valid", "source_depth_valid"),
            ("source_cam2world", "source_cam2world"),
            ("source_intrinsics", "source_intrinsics"),
            ("source_intrinsics_4rc", "source_intrinsics_4rc"),
            ("source_intrinsics_gt", "source_intrinsics_gt"),
            ("source_dynamic_mask", "source_dynamic_mask"),
            ("target_dynamic_mask", "target_dynamic_mask"),
        ):
            v = data.get(src_key)
            if _drop_dense and src_key == "source_rgb_video":
                # Keep the dense encoder in-graph on every rank. Dropping the
                # key entirely makes the dense unit early-return `{}`, which
                # skips `dense_target_control_encoder` on only a subset of
                # ranks/samples and reintroduces DDP/FSDP unused-param hangs.
                v = _zero_like_dense_rgb(v)
            elif _drop_dense and src_key == "source_depth_valid":
                v = _zero_like_dense_valid(v)
            if v is not None:
                inputs_shared[dst_key] = _to_tensor(v)

        if _use_dense and not getattr(self, "_printed_dense_branch_warning", False):
            required = ("source_rgb_video", "source_depth_video",
                        "source_cam2world", "source_intrinsics")
            missing = [k for k in required if data.get(k) is None]
            if missing and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                import warnings
                warnings.warn(
                    "[Track4D-360] use_dense_branch=True but the dataset is "
                    f"not producing: {missing}. The dense target-space control "
                    "branch will silently no-op for every batch missing these "
                    "fields. Direct-dataset mode turns the right flags on "
                    "automatically; for YAML mode add "
                    "return_source_rgb_video=True, return_source_cameras=True, "
                    "return_source_depth_video=True to the dataset kwargs in "
                    "your data config.",
                    stacklevel=2,
                )
            self._printed_dense_branch_warning = True

        # Modality indicators from mixed-dataset training
        inputs_shared["has_tracks"] = False if _drop_tracks else data.get("has_tracks", True)
        inputs_shared["has_dynamic_tracks"] = False if _drop_tracks else data.get("has_dynamic_tracks", True)

        # Per-domain loss masking: track_video_loss_mask controls whether
        # non-RGB target domains (track, depth) should have diffusion loss.
        # 1.0 = full loss, 0.0 = masked. Used by FlowMatchV2VLoss when
        # multimodality_tracking/multimodality_track_depth is enabled.
        # Shape: scalar, (1,), or (B,) — _coerce_sample_weight_vector in loss.py handles all.
        track_video_loss_mask = 0.0 if _drop_tracks else data.get("track_video_loss_mask", None)
        # Override: force_track_video_loss_mask from YAML config (e.g., 0.0 to disable track supervision)
        if getattr(self, "_force_track_video_loss_mask", None) is not None:
            track_video_loss_mask = self._force_track_video_loss_mask
        if track_video_loss_mask is not None:
            inputs_shared["track_video_loss_mask"] = _to_tensor(track_video_loss_mask)

        # Stage 2: pass target-view dense 3D tracks
        if self.return_target_tracks:
            target_track_video = data.get("target_track_video")
            target_track_valid_mask = data.get("target_track_valid_mask")
            if target_track_video is not None:
                inputs_shared["target_track_video"] = _to_tensor(target_track_video)
            if target_track_valid_mask is not None:
                inputs_shared["target_track_valid_mask"] = _to_tensor(target_track_valid_mask)
            # v4 / cache-v3 persistent-tracks: also forward the row-major
            # `(N=H_full*W_full, T, 3)` `target_tracks_world` tensor when
            # the dataset emits it (Kubric_NVS does; StaticScene_NVS does
            # not, in which case we just don't add the key and v4's loss
            # dispatch falls back to track-less / pointmap mode). Without
            # this forwarding the v4 dispatch saw `inputs.get
            # ("target_tracks_world") == None` for every batch and never
            # exercised the persistent-track path.
            target_tracks_world = data.get("target_tracks_world")
            target_tracks_world_valid_mask = data.get(
                "target_tracks_world_valid_mask",
                data.get("target_tracks_world_valid"),
            )
            if target_tracks_world is not None:
                inputs_shared["target_tracks_world"] = _to_tensor(target_tracks_world)
            if target_tracks_world_valid_mask is not None:
                inputs_shared["target_tracks_world_valid_mask"] = _to_tensor(
                    target_tracks_world_valid_mask
                )
        if self.return_target_depth:
            target_depth_video = data.get("target_depth_video")
            if target_depth_video is None:
                target_depth_video = data.get("target_depth")
            target_depth_valid_mask = data.get("target_depth_valid_mask")
            if target_depth_valid_mask is None:
                target_depth_valid_mask = data.get("target_depth_valid")
            if target_depth_video is not None:
                inputs_shared["target_depth_video"] = _to_tensor(target_depth_video)
            if target_depth_valid_mask is not None:
                inputs_shared["target_depth_valid_mask"] = _to_tensor(target_depth_valid_mask)

        # Depth loss mask: zero out depth domain loss for datasets without depth GT
        has_depth = False if _drop_dense else data.get("has_depth", True)
        if torch.is_tensor(has_depth):
            depth_loss_mask = has_depth.float()
        else:
            depth_loss_mask = 1.0 if has_depth else 0.0
        inputs_shared["depth_loss_mask"] = depth_loss_mask

        if self.ar_training:
            inputs_shared["ar_training"] = True
            inputs_shared.update(self.ar_config)

        # Reprojected source depth → target camera (optional, for 6-channel track injection)
        reproj_src_depth = data.get("reproj_src_depth_at_target")
        reproj_src_valid = data.get("reproj_src_depth_valid")
        if reproj_src_depth is not None:
            inputs_shared["source_depth_at_tracks"] = _to_tensor(reproj_src_depth)
        if reproj_src_valid is not None:
            inputs_shared["source_depth_valid_at_tracks"] = _to_tensor(reproj_src_valid)

        # Source RGB at track positions (optional, for RGB-augmented track injection)
        source_rgb_at_tracks = data.get("source_rgb_at_tracks")
        if source_rgb_at_tracks is not None:
            inputs_shared["source_rgb_at_tracks"] = _to_tensor(source_rgb_at_tracks)

        # Per-identity dynamic mask at track positions (optional, for the dynmask
        # track config: source SAM3/GT motion label, reprojected to target by the
        # adapter splat). Strict-check (fail on absence) lives in the track unit.
        source_dynamic_mask_at_tracks = data.get("source_dynamic_mask_at_tracks")
        if source_dynamic_mask_at_tracks is not None:
            inputs_shared["source_dynamic_mask_at_tracks"] = _to_tensor(source_dynamic_mask_at_tracks)

        # GS output: pass flags and GT data for splatting loss
        if self.use_gs_output:
            inputs_shared["use_gs_output"] = True
            inputs_shared["use_gs_feedback_training"] = self.use_gs_feedback_training
            inputs_shared["gs_config"] = {
                **self.gs_config,
                "image_height": data["video"][0].size[1],
                "image_width": data["video"][0].size[0],
            }
            # GT target RGB in [-1,1] at full resolution for GS rendering loss
            _tgt_vid = data.get("target_video")
            if isinstance(_tgt_vid, list) and len(_tgt_vid) > 0 and hasattr(_tgt_vid[0], "convert"):
                import torchvision.transforms.functional as _TF
                target_video_tensor = torch.stack(
                    [_TF.to_tensor(img) * 2.0 - 1.0 for img in _tgt_vid], dim=1
                )  # (3, T, H, W)
            else:
                target_video_tensor = _to_tensor(_tgt_vid)
            if target_video_tensor is not None:
                inputs_shared["gt_target_rgb"] = target_video_tensor  # (3, T, H, W) per-sample
            if "target_depth_video" in inputs_shared:
                inputs_shared["gt_target_depth"] = inputs_shared["target_depth_video"]
            if "target_depth_valid_mask" in inputs_shared:
                inputs_shared["depth_valid_mask"] = inputs_shared["target_depth_valid_mask"]
            # Target cameras in FlashWorld 11-param format
            target_cam2world = data.get("target_cam2world")
            target_intrinsics = data.get("target_intrinsics")
            if target_cam2world is not None and target_intrinsics is not None:
                from holo4d.models.gs_camera_utils import c2w_K_to_flashworld_cameras
                c2w = _to_tensor(target_cam2world)  # (T, 4, 4)
                K = _to_tensor(target_intrinsics)    # (3, 3)
                H_img = data["video"][0].size[1]
                W_img = data["video"][0].size[0]
                inputs_shared["target_cameras_flashworld"] = c2w_K_to_flashworld_cameras(c2w, K, H_img, W_img)

        if self.training and not self._printed_runtime_grad_ckpt:
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(
                    "[Track4D-360] First train batch runtime flags: "
                    f"use_gradient_checkpointing={inputs_shared['use_gradient_checkpointing']}, "
                    f"use_gradient_checkpointing_offload={inputs_shared['use_gradient_checkpointing_offload']}"
                )
            self._printed_runtime_grad_ckpt = True
        dense_path = 0
        if _use_dense:
            has_target_cam = inputs_shared.get("target_cam2world") is not None
            has_source_rgb = inputs_shared.get("source_rgb_video") is not None
            has_source_depth = inputs_shared.get("source_depth_video") is not None
            has_source_cam = inputs_shared.get("source_cam2world") is not None
            if not has_target_cam:
                dense_path = 0
            elif has_source_rgb and has_source_depth and has_source_cam:
                dense_path = 1
            elif bool(inputs_shared.get("use_dense_null_token_fallback", False)):
                dense_path = 2
            else:
                dense_path = 3

        track_path = 0
        if not bool(getattr(self, "disable_track_modules", False)):
            has_track_world = inputs_shared.get("tracks_world") is not None
            has_target_cam = inputs_shared.get("target_cam2world") is not None
            has_target_k = inputs_shared.get("target_intrinsics") is not None
            if has_track_world and has_target_cam and has_target_k:
                track_path = 1
            elif bool(inputs_shared.get("use_track_null_token_fallback", False)):
                track_path = 2
            else:
                track_path = 3

        self._latest_graph_signature = {
            "camera_only_tier": int(_camera_only_tier),
            "drop_tracks": int(_drop_tracks),
            "drop_dense": int(_drop_dense),
            "track_path": int(track_path),
            "dense_path": int(dense_path),
        }
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        """Forward pass: run pipeline units → compute loss."""
        if isinstance(data, list):
            data = data[0]
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        else:
            # Cached pipeline inputs predate the runner's step context. Inject
            # the dedicated identity seed at consumption time as well.
            identity_sampling_seed = self._track_identity_sampling_seed(data)
            if identity_sampling_seed is not None:
                inputs[0]["track_identity_sampling_seed"] = identity_sampling_seed
        input_metrics = dict(getattr(self, "_pending_input_metrics", {}) or {})
        self._pending_input_metrics = {}
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        # Opt-in per-unit + DiT forward timing to localize forward_s variance to a
        # specific control component (track_adapter / dense warp / plucker) vs the
        # DiT denoising. cuda.synchronize is required for true GPU timing, so this
        # adds sync overhead — keep OFF (default) for production; set
        # TRACK4D360_PROFILE_UNITS=1 when diagnosing. Results land in wandb as
        # timing/unit/<UnitClassName>_s and timing/dit_loss_fn_s.
        _profile_units = os.environ.get("TRACK4D360_PROFILE_UNITS", "0") == "1"
        _unit_timings = {}
        for unit in self.pipe.units:
            if _profile_units and torch.cuda.is_available():
                torch.cuda.synchronize()
                _u0 = _time.perf_counter()
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            if _profile_units and torch.cuda.is_available():
                torch.cuda.synchronize()
                _key = f"timing/unit/{type(unit).__name__}_s"
                _unit_timings[_key] = _unit_timings.get(_key, 0.0) + (_time.perf_counter() - _u0)
        if _profile_units and torch.cuda.is_available():
            torch.cuda.synchronize()
            _l0 = _time.perf_counter()
        loss_or_tuple = self.task_to_loss[self.task](self.pipe, *inputs)
        if _profile_units and torch.cuda.is_available():
            torch.cuda.synchronize()
            _unit_timings["timing/dit_loss_fn_s"] = _time.perf_counter() - _l0
            input_metrics.update(_unit_timings)
        if isinstance(loss_or_tuple, tuple):
            loss, metrics = loss_or_tuple
            merged_metrics = dict(input_metrics)
            if isinstance(metrics, dict):
                merged_metrics.update(metrics)
            self._latest_log_metrics = merged_metrics
        else:
            loss = loss_or_tuple
            self._latest_log_metrics = input_metrics
        return loss

    def consume_latest_log_metrics(self):
        metrics = self._latest_log_metrics
        self._latest_log_metrics = {}
        # Append track module health metrics (weight/gradient norms)
        metrics.update(self._compute_track_health_metrics())
        # One-shot HBM probe from _setup_prompt_emb_cache(): only emitted on
        # the first step that hits this path, then cleared. allocated_gb is
        # the offload proof (≈10 GB lower when text_encoder.cpu() ran);
        # reserved_gb tracks the allocator high-water mark.
        hbm_probe = getattr(self, "_hbm_after_t5_setup", None)
        if hbm_probe:
            metrics.update(hbm_probe)
            self._hbm_after_t5_setup = {}
        # Per-step HBM snapshot — gives a continuous wandb chart that holds
        # the offload delta (~11 GB for HOLO4D_OFFLOAD_T5_TO_CPU=1 runs) for the
        # entire training run, instead of relying on the system-gpu panel
        # whose nvml-sourced reading sometimes hides the gap behind the
        # caching-allocator high-water mark. Cheap (two cudaMemGetInfo
        # equivalents per step).
        if torch.cuda.is_available():
            metrics["hbm/allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics["hbm/reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        # Per-step T5 device probe. 1.0 = text_encoder lives on a CUDA device
        # (offload was busted, online encode brought it back, etc.); 0.0 =
        # CPU; -1.0 = pipe.text_encoder is None (HOLO4D_DELETE_T5_AFTER_PRECOMPUTE
        # path). Watching this on wandb is the only way to definitively tell
        # whether the offload is holding throughout training, since
        # `hbm/allocated_gb` can be inflated by allocator-greedy caching of
        # activation blocks that aren't T5 weights.
        try:
            te = getattr(self.pipe, "text_encoder", None)
            if te is None:
                metrics["t5/on_cuda"] = -1.0
            else:
                dev = next(te.parameters()).device
                metrics["t5/on_cuda"] = 1.0 if dev.type == "cuda" else 0.0
        except StopIteration:
            metrics["t5/on_cuda"] = -2.0  # text_encoder has no params (shouldn't happen)
        except Exception:
            pass
        return metrics

    def consume_latest_graph_signature(self):
        signature = self._latest_graph_signature
        self._latest_graph_signature = {}
        return signature

    def _compute_track_health_metrics(self):
        """Compute weight and gradient norms for track injection modules.

        Returns dict of scalar metrics keyed like:
          track/injector/block_00/weight_norm
          track/injector/block_00/grad_norm
          track/injector/weight_norm_mean   (across blocks)
          track/injector/grad_norm_mean
          track/embedder/weight_norm
          track/adapter/weight_norm
          ...
        """
        metrics = {}
        # In gs_only_training the track/dense/plucker/injector modules are
        # all frozen (requires_grad=False) so weight_norm is constant and
        # grad_norm is always 0 — nothing actionable to log. Short-circuit.
        if getattr(self, "gs_only_training", False):
            return metrics
        pipe = self.pipe

        norm_entries = {}

        # --- Per-block injector encoders ---
        injector = getattr(pipe, "track_block_injector", None)
        if injector is not None:
            for i, enc in enumerate(injector.encoders):
                prefix = f"track/injector/block_{i:02d}"
                norm_entries[f"{prefix}/weight_norm"] = [enc.weight]
                norm_entries[f"{prefix}/grad_norm"] = [enc.weight.grad] if enc.weight.grad is not None else []

        # --- Track embedder ---
        embedder = getattr(pipe, "track_embedder", None)
        if embedder is not None:
            embedder_params = list(embedder.parameters())
            norm_entries["track/embedder/weight_norm"] = embedder_params
            norm_entries["track/embedder/grad_norm"] = [
                p.grad for p in embedder_params if p.grad is not None
            ]

        # --- Track adapter ---
        adapter = getattr(pipe, "track_adapter", None)
        if adapter is not None:
            adapter_params = list(adapter.parameters())
            norm_entries["track/adapter/weight_norm"] = adapter_params
            norm_entries["track/adapter/grad_norm"] = [
                p.grad for p in adapter_params if p.grad is not None
            ]

        if norm_entries:
            metrics.update(compute_l2_norms(norm_entries))
            block_w_norms = [
                value
                for key, value in metrics.items()
                if key.startswith("track/injector/block_") and key.endswith("/weight_norm")
            ]
            block_g_norms = [
                value
                for key, value in metrics.items()
                if key.startswith("track/injector/block_") and key.endswith("/grad_norm")
            ]
            if block_w_norms:
                metrics["track/injector/weight_norm_mean"] = sum(block_w_norms) / len(block_w_norms)
                metrics["track/injector/weight_norm_max"] = max(block_w_norms)
            if block_g_norms:
                metrics["track/injector/grad_norm_mean"] = sum(block_g_norms) / len(block_g_norms)
                metrics["track/injector/grad_norm_max"] = max(block_g_norms)

        return metrics

    # ------------------------------------------------------------------
    # Debug visualization (called by runner.py hook)
    # ------------------------------------------------------------------

    def log_debug_visualization(self, batch_data, output_root, global_step, fps=16, quality=5):
        """Save debug video as 2×3 grid matching the reference visualization.

        Layout:
          [source        ] [target GT     ] [track residual (3D motion)]
          [src vis map   ] [block density ] [proj UV+density on target ]
        """
        import numpy as np
        from PIL import Image

        from holo4d.visualization.track_panels import (
            render_track_residual,
            render_source_vis_map,
            render_block_density,
            render_proj_uv_density_merged,
            compose_2x3_grid,
            save_video,
        )

        step_dir = os.path.join(output_root, f"step-{int(global_step):06d}")
        os.makedirs(step_dir, exist_ok=True)

        # --- Extract data ---
        src_frames = batch_data.get("video", [])
        tgt_frames = batch_data.get("target_video", [])
        tracks_world = batch_data.get("tracks_world")
        cam2world = batch_data.get("target_cam2world")
        intrinsics = batch_data.get("target_intrinsics")

        has_rgb = (
            isinstance(src_frames, list) and len(src_frames) > 0
            and isinstance(src_frames[0], Image.Image)
        )
        has_tracks = tracks_world is not None
        has_proj = has_tracks and cam2world is not None and intrinsics is not None

        if not has_rgb and not has_tracks:
            return

        if has_rgb:
            W, H = src_frames[0].size
            T = len(src_frames)
        else:
            H, W = 144, 256
            T = cam2world.shape[0] if hasattr(cam2world, 'shape') else 49

        # Convert to numpy
        src_np = [np.asarray(f.resize((W, H))) for f in src_frames] if has_rgb else [np.zeros((H, W, 3), dtype=np.uint8)] * T
        tgt_np = [np.asarray(f.resize((W, H))) for f in tgt_frames] if has_rgb and isinstance(tgt_frames, list) else [np.zeros((H, W, 3), dtype=np.uint8)] * T
        black = [np.zeros((H, W, 3), dtype=np.uint8)] * T

        # Row 1: source, target GT, track residual
        residual_frames = render_track_residual(tracks_world, H, W) if has_tracks else black

        # Row 2: src vis map, block density, proj UV+density merged
        if has_proj:
            src_vis_frames = render_source_vis_map(
                tracks_world, cam2world, intrinsics, H, W, src_frames_np=src_np)
            density_frames = render_block_density(
                tracks_world, cam2world, intrinsics, H, W, T, target_frames_np=tgt_np)
            merged_frames = render_proj_uv_density_merged(
                tracks_world, cam2world, intrinsics, H, W, T, target_frames_np=tgt_np,
                marker_size=2)
        else:
            src_vis_frames = density_frames = merged_frames = black

        prompt = batch_data.get("prompt", "")

        combined = compose_2x3_grid(
            top_panels=[src_np, tgt_np, residual_frames],
            bot_panels=[src_vis_frames, density_frames, merged_frames],
            top_labels=["source", "target GT", "track residual"],
            bot_labels=["src vis (grn/red/org)", "block density", "proj UV + density"],
            caption=prompt,
        )

        save_video(combined, os.path.join(step_dir, "debug_combined.mp4"), fps=fps)

        # Also produce ``debug_full.mp4`` that vstacks debug_combined (which carries
        # the proj-UV tracking branch) with the latest debug_t*.mp4 (latent panels).
        # Mirrors the comparison.mp4 style from benchmark_davis_4rc_1p3b.sh.
        try:
            import glob as _glob
            import imageio.v2 as _imageio
            import cv2 as _cv2
            t_paths = sorted(
                _glob.glob(os.path.join(step_dir, "debug_t*.mp4")),
                key=lambda p: os.path.getmtime(p),
            )
            combined_path = os.path.join(step_dir, "debug_combined.mp4")
            if t_paths and os.path.exists(combined_path):
                t_mp4 = t_paths[-1]

                def _read(p):
                    rdr = _imageio.get_reader(p)
                    fs = [np.asarray(f, dtype=np.uint8)[..., :3] for f in rdr]
                    rdr.close()
                    return fs

                top_fs = _read(combined_path)
                bot_fs = _read(t_mp4)
                if top_fs and bot_fs:
                    target_w = max(top_fs[0].shape[1], bot_fs[0].shape[1])

                    def _resize_w(fs, w):
                        out = []
                        for f in fs:
                            h, w0 = f.shape[:2]
                            if w0 != w:
                                new_h = int(round(h * (w / w0)))
                                f = _cv2.resize(f, (w, new_h), interpolation=_cv2.INTER_AREA)
                            out.append(f)
                        return out

                    top_fs = _resize_w(top_fs, target_w)
                    bot_fs = _resize_w(bot_fs, target_w)
                    n = min(len(top_fs), len(bot_fs))
                    stacked = [np.concatenate([top_fs[i], bot_fs[i]], axis=0) for i in range(n)]
                    # libx264 (yuv420p) requires both H and W to be even.
                    # imageio's writer auto-pads W (macro_block_size=16) but
                    # leaves H untouched → odd H crashes the encoder.
                    # Trim a row / column if needed; cheap and lossless.
                    if stacked:
                        sh, sw = stacked[0].shape[:2]
                        if sh % 2 == 1 or sw % 2 == 1:
                            new_h = sh - (sh % 2)
                            new_w = sw - (sw % 2)
                            stacked = [f[:new_h, :new_w] for f in stacked]
                    save_video(stacked, os.path.join(step_dir, "debug_full.mp4"), fps=fps)
        except Exception as _e:
            print(f"[debug_full] Failed to vstack debug videos: {_e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def track4d360_parser():
    parser = argparse.ArgumentParser(description="Track4D-360 Training")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)

    parser.add_argument("--tokenizer_path", type=str, default=None)

    # Koichi-style direct finetuning (full params, not LoRA)
    parser.add_argument("--dit_finetune_modules", type=str, default="",
                        help="Comma-separated DiT submodules to finetune with full params. "
                             "E.g. 'self_attn,ffn' for Koichi-style finetuning.")
    parser.add_argument(
        "--i2v_target_condition",
        action="store_true",
        default=False,
        help=(
            "Opt in to Vista4D-style continuation training with Wan I2V: use "
            "the full source video as source-domain V2V condition and "
            "target_video[0] as the additional target boundary condition. "
            "Requires an I2V backbone and CLIP image encoder; default off "
            "preserves existing runs."
        ),
    )
    parser.add_argument(
        "--i2v_condition_drop_prob",
        type=float,
        default=0.3,
        help="Probability of dropping target-boundary VAE and CLIP conditions during training (Vista4D: 0.3); source V2V remains.",
    )
    parser.add_argument(
        "--i2v_condition_noise_level",
        type=float,
        default=0.05,
        help=(
            "Image-latent augmentation alpha in (1-alpha)*x + alpha*noise "
            "during training (Vista4D: 0.05)."
        ),
    )
    parser.add_argument("--track_injection_mode", type=str, default="single",
                        choices=["single", "per_block"],
                        help="Track injection mode: 'single' (one projector before all blocks) "
                             "or 'per_block' (encoder+projector at each DiT block).")
    parser.add_argument("--track_injection_block_mode", type=str, default="additive",
                        choices=["additive", "concat_project"],
                        help="Per-block injection style: 'additive' (x + Linear(track)) "
                             "or 'concat_project' (Linear(cat(x, track)), Koichi-style).")

    parser.add_argument("--train_sigma_shift", type=float, default=None,
                        help="Override the noise-schedule shift for training timesteps. "
                             "Default (None) uses the Wan 2.1 default of 5.0. "
                             "Lower values (e.g. 3.0) give more training at medium noise levels.")

    # Data
    parser.add_argument("--dummy", default=False, action="store_true",
                        help="Use dummy random data for CPU testing.")
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true",
                        help="Load model on CPU first (required for ZeRO offload on small GPUs).")

    # Resume from non-LoRA checkpoint
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a .safetensors checkpoint from a previous run. "
                             "Loads trainable params (DiT self_attn/ffn + track modules) "
                             "on top of the base model. Use with --initial_step and "
                             "--allow_existing_checkpoints.")
    parser.add_argument("--gs_head_init_checkpoint", type=str, default=None,
                        help="Optional thin gs_decoder/gs_head delta checkpoint to "
                             "overlay AFTER --resume_checkpoint. Use this when warm-"
                             "starting the new run from a frozen teacher (resume_"
                             "checkpoint=teacher) and seeding gs_decoder/gs_head from "
                             "a previous run's safetensors. Missing keys are skipped, "
                             "so checkpoints saved as 'trainable subset only' that "
                             "lack DiT body weights won't clobber the teacher.")

    # Stage 2: joint NVS + 3D tracking
    parser.add_argument("--joint_track_prediction", default=False, action="store_true",
                        help="Stage 2: jointly predict target RGB + target-view dense 3D tracks. "
                             "Uses 3-domain V2V (source_rgb | target_rgb | target_track). "
                             "Requires dataset with return_target_tracks=True.")
    parser.add_argument("--joint_track_depth_prediction", default=False, action="store_true",
                        help="Stage 2 multimodal: jointly predict target RGB + target-view dense 3D tracks + depth. "
                             "Uses 4-domain V2V (source_rgb | target_rgb | target_track | target_depth). "
                             "Requires dataset with return_target_tracks=True and return_target_depth=True.")
    parser.add_argument("--joint_depth_prediction", default=False, action="store_true",
                        help="Stage 2: jointly predict target RGB + depth (3-domain V2V, no tracks).")
    parser.add_argument("--target_depth_log_cap", type=float, default=14.0,
                        help="Fixed cap for log-depth normalization.")
    parser.add_argument("--target_depth_norm_type", type=str, default="log_depth",
                        choices=["log_depth", "disparity"],
                        help="Depth normalization: 'log_depth' (default) or 'disparity' (1/depth).")
    parser.add_argument("--target_depth_max_disparity", type=float, default=5.0,
                        help="Max disparity cap when --target_depth_norm_type=disparity.")
    parser.add_argument("--camera_control_checkpoint", type=str, default=None,
                        help="Path to camera-control base .safetensors. Applies 32ch→dual-16ch "
                             "patch_embedding remap then loads into WanModelV2V with strict=False.")

    # Channel-concat multi-modal V2V (memory-efficient architecture)
    parser.add_argument("--channel_concat_stage", type=str, default=None,
                        choices=["A", "B"],
                        help="Channel-concat multi-modal mode. "
                             "'A' = RGB+depth (2 modalities, 2*C_vae channels). "
                             "'B' = RGB+depth+residual_track (3 modalities, 3*C_vae). "
                             "None = legacy width-concat multi-domain.")
    parser.add_argument("--use_residual_tracks", default=True,
                        help="Predict target-view tracks in residual form (track[t] - track[query]). "
                             "Only used when --channel_concat_stage=B.")
    parser.add_argument("--no_residual_tracks", dest="use_residual_tracks", action="store_false",
                        help="Predict absolute target-view tracks (not residual).")
    parser.add_argument("--freeze_lora_after_load", action="store_true",
                        help="Load LoRA weights from --lora_checkpoint, then freeze all LoRA adapter params. "
                             "Useful when training only auxiliary heads on top of a fixed LoRA backbone.")
    parser.add_argument("--skip_v2v_required_trainables", action="store_true",
                        help="Do not auto-add V2V trainables such as domain_embedding/control_patch_embedding. "
                             "Useful for GS-head-only training on top of a fixed V2V checkpoint.")
    parser.add_argument("--disable_domain_embedding", action="store_true",
                        help="Do not use, auto-train, or export V2V domain_embedding. "
                             "Default off; intended only for raw Wan T2V-style RF sanity tests "
                             "that force input_video=None and disable Track4D/control modules.")
    parser.add_argument("--gs_only_training", action="store_true",
                        help="Train ONLY the GS head (dit.gs_feature_head + gs_decoder). "
                             "Skips auto-adding track modules, V2V modules, domain_embedding, "
                             "and control_patch_embedding to the trainable set. These stay in "
                             "the pipeline for correct forward pass but are frozen from the "
                             "loaded checkpoint. Combines with --resume_checkpoint to study "
                             "GS decoding on a pre-trained frozen base model.")
    parser.add_argument("--freeze_track_block_injector", action="store_true",
                        help="Keep per-block Track4D injector modules in the pipeline but "
                             "do not add them to the trainable set. Useful when continuing "
                             "a Track4D run with new auxiliary heads and the injector path "
                             "is not guaranteed to participate in every DDP step.")
    parser.add_argument("--freeze_track4d_condition_modules", action="store_true",
                        help="Keep persistent track and dense condition modules in the "
                             "forward pass, but do not auto-add their parameters to "
                             "trainable_models. Use this when continuing a checkpoint "
                             "and only adapting DiT/camera-control parameters.")

    from holo4d.shared_args import (
        add_gs_output_args,
        add_gs_dynamic_args,
        add_cond_dropout_args,
        add_cond_null_token_fallback_args,
        add_plucker_args,
        add_perlayer_camera_args,
        add_warped_condition_args,
        add_gs_training_timestep_args,
        add_ar_args,
        add_future_prediction_args,
    )
    add_gs_output_args(parser)
    add_gs_dynamic_args(parser)
    add_cond_dropout_args(parser)
    add_cond_null_token_fallback_args(parser)
    # Defines --use_plucker_camera_control, --enable_v2v_plucker_camera_control,
    # --train_plucker_camera_control, --use_plucker_raymap_tokens, and
    # --plucker_use_ray_origin from the shared train/eval source of truth.
    add_plucker_args(parser)
    add_perlayer_camera_args(parser)
    add_warped_condition_args(parser)
    add_gs_training_timestep_args(parser)
    add_ar_args(parser)
    add_future_prediction_args(parser)

    # Dense GEN3C-style target-space control branch
    parser.add_argument("--use_dense_branch", action="store_true",
                        help="Enable the dense GEN3C-style target-space control branch. "
                             "Requires the dataset to produce source_rgb_video, source_depth_video, "
                             "source_cam2world, and source_intrinsics per sample. In direct-dataset "
                             "mode these are turned on automatically; in YAML mode the dataset must "
                             "set return_source_rgb_video=True, return_source_cameras=True, "
                             "return_source_depth_video=True explicitly.")
    parser.add_argument("--dense_in_channels", type=int, default=5,
                        help="Dense encoder input channels. 5 = [rgb(3)|depth_norm(1)|pointcloud_alpha(1)]; "
                             "6 = [rgb(3)|depth_norm/null(1)|target_motion_mask(1)|pointcloud_alpha(1)]; "
                             "7 = [rgb(3)|depth_norm/null(1)|source_motion_mask(1)|"
                             "target_motion_mask(1)|pointcloud_alpha(1)]; "
                             "8 = [rgb(3)|depth_norm/null(1)|source_alpha_mask(1)|"
                             "source_motion_mask(1)|target_motion_mask(1)|pointcloud_alpha(1)].")
    parser.add_argument("--dense_proj_dim", type=int, default=32,
                        help="Dense encoder internal Conv3d channel dim.")
    parser.add_argument("--dense_num_residual_blocks", type=int, default=2,
                        help="Number of CausalResidualBlock3d stages in dense encoder.")
    parser.add_argument("--dense_alpha_track", type=float, default=1.0,
                        help="Scale on the persistent branch in late additive fusion: "
                             "target_tokens = dense_ctrl_tokens + alpha_track * persistent_track_tokens.")
    parser.add_argument("--use_depth_null_token", action="store_true", default=False,
                        help="Tier-B-prime ablation: keep dense_in_channels=5 and replace "
                             "the warped-depth slice with a learnable scalar broadcast in the "
                             "encoder. Avoids re-initializing conv3d_patchify when the "
                             "warm-start ckpt was trained with RGBD. See EXPERIMENTS.md §9.")
    parser.add_argument("--dense_conditioning_noise_max_level", type=float, default=0.0,
                        help="Vista4D-style continuous noise augmentation on dense and track "
                             "conditioning tokens (opt-in). 0.0 (default) = off, backward-compatible. When >0, "
                             "samples level ~ U(0, max) per batch at training and adds Gaussian "
                             "noise scaled to token std. Inference unchanged. Typical: 0.1–0.4.")
    parser.add_argument("--source_mask_boundary_depth_dropout_prob", type=float, default=0.0,
                        help="Train-time probability of invalidating source-depth pixels on the "
                             "foreground/background boundary of source_dynamic_mask before hybrid "
                             "soft-splat fusion. 0.0 = off/backward-compatible; 0.5 reduces "
                             "SAM/GMOS boundary train-inference gap.")
    parser.add_argument("--source_mask_boundary_kernel_size", type=int, default=5,
                        help="Odd morphology kernel for source_mask_boundary_depth_dropout_prob.")
    parser.add_argument("--dense_warp_mode", type=str, default="zbuffer",
                        choices=["zbuffer", "softsplat", "hybrid_softsplat"],
                        help="Dense-branch warped-RGB mode for the source-to-target reprojection. "
                             "'zbuffer' (default, parent recipe): hard z-buffered per-pixel splat. "
                             "'softsplat': GEN3C-style per-frame bilinear soft-splat with "
                             "depth-weighted blending and the same [RGB|depth|valid] contract. "
                             "'hybrid_softsplat': Lyra-2 / GEN3C bilinear soft-splat — multi-frame "
                             "fused at static target pixels (n_views=T, depth-weighted), per-frame "
                             "s=t→t splat at dynamic target pixels. Static / dynamic comes from "
                             "per-frame Kubric segmentation masks (or `tracks_world` projection as "
                             "a fallback for track-only dynamic datasets). Routed through the same "
                             "5-channel [RGB|depth|valid] dense encoder — only the contents of the "
                             "warped tensor change.")
    parser.add_argument("--dense_warp_motion_threshold", type=float, default=0.005,
                        help="Per-pixel motion-range threshold (in cam0-normalized coords) above "
                             "which a target pixel is considered DYNAMIC and falls back to per-"
                             "frame splat. Used only when --dense_warp_mode=hybrid_softsplat AND "
                             "the dataset doesn't provide an explicit per-frame segmentation mask. "
                             "Default 0.005 = 0.5%% of scene scale (tightened from 0.01 on "
                             "2026-04-28 after Syn4D 288×512 leakage verification — Syn4D actors "
                             "have smaller-amplitude motion than Kubric balls and need the "
                             "stricter cutoff). If still leaking, drop to 0.003.")
    parser.add_argument("--dense_hybrid_disable_dynamic_masks", action="store_true", default=False,
                        help="For --dense_warp_mode=hybrid_softsplat, ignore explicit dynamic "
                             "masks and track-derived dynamic masks. The branch still performs "
                             "multi-frame softsplat, but treats the target as all-static. Used by "
                             "the multiframe-RGB no-dynamic-mask ablation.")
    parser.add_argument("--use_vista4d_latent_streams", action="store_true", default=False,
                        help="Opt in to Vista4D-style source/point-cloud latent streams. "
                             "Source RGB and rendered point-cloud RGB are VAE-encoded as extra token streams; "
                             "source/point-cloud alpha+motion masks are packed into zero-init mask patch embeddings. "
                             "Default off preserves the old dense-control path.")
    parser.add_argument("--vista4d_streams_pointcloud_only", action="store_true", default=False,
                        help="With --use_vista4d_latent_streams: drop the redundant Vista4D source-video "
                             "stream and append only the reprojected point-cloud render stream (token "
                             "sequence [output | point-cloud], stream_count 2 instead of 3). In width/frame "
                             "V2V the clean source video is already carried by x's source domain, so the "
                             "separate source stream double-encodes it. Use this for the 'ab12 tracks + "
                             "token-injected 05-style point-cloud render' experiment.")
    parser.add_argument("--use_gen3c_warp_latent_channels", action="store_true", default=False,
                        help="Opt in to GEN3C-style warped RGB injection. The rendered point-cloud RGB "
                             "and its valid mask are VAE-encoded and added through a zero-init input "
                             "patch embedding, instead of the old dense-control module.")
    parser.add_argument("--disable_track_modules", action="store_true",
                        help="Disable Track4D persistent/dense module initialization entirely. "
                             "Useful for raw plucker-only V2V runs that still reuse the fixed-window DL3DV loaders.")
    parser.add_argument("--force_t2v_drop_source_cond", action="store_true",
                        help="Force input_video=None for every batch so the V2V concat unit enters "
                             "T2V mode. Default off; intended for raw Wan RF sanity tests without "
                             "source-video/control conditioning.")
    parser.add_argument("--raw_t2v_target_only", action="store_true",
                        help="With --force_t2v_drop_source_cond, bypass V2V fake-source concatenation "
                             "and train on target latents only (v2v_concat_mode=None in the model call). "
                             "Intended for raw causal-teacher experiments.")
    parser.add_argument("--dense_only_condition_modules", action="store_true",
                        help="Enable dense target-space conditioning without constructing the "
                             "persistent 3D-track input encoder. Keeps only the dense encoder "
                             "plus token-injection support needed to align dense tokens into "
                             "the V2V sequence.")
    parser.add_argument("--torch_profile_active_steps", type=int, default=0,
                        help="Enable torch.profiler on rank 0 for this many optimizer steps. 0 disables profiling.")
    parser.add_argument("--torch_profile_wait_steps", type=int, default=1,
                        help="torch.profiler schedule wait steps before warmup.")
    parser.add_argument("--torch_profile_warmup_steps", type=int, default=1,
                        help="torch.profiler schedule warmup steps before active capture.")
    parser.add_argument("--torch_profile_repeat", type=int, default=1,
                        help="Number of profiler schedule repeats.")
    parser.add_argument("--torch_profile_dir", type=str, default=None,
                        help="Directory for TensorBoard-compatible torch.profiler traces. "
                             "Default: <output_path>/torch_profiler")
    parser.add_argument("--torch_profile_record_shapes", default=True,
                        help="Record tensor shapes in profiler traces.")
    parser.add_argument("--no_torch_profile_record_shapes", dest="torch_profile_record_shapes", action="store_false",
                        help="Disable tensor-shape recording in profiler traces.")
    parser.add_argument("--torch_profile_memory", default=True,
                        help="Record memory events in profiler traces.")
    parser.add_argument("--no_torch_profile_memory", dest="torch_profile_memory", action="store_false",
                        help="Disable memory event recording in profiler traces.")
    parser.add_argument("--torch_profile_with_stack", action="store_true",
                        help="Capture Python stacks in profiler traces. Higher overhead, but useful for attribution.")

    # Shared Track4D-360 args (track_config, syn4d_root, scene, sequence_idx, track_spatial_stride)
    add_track4d360_args(parser)
    return parser


if __name__ == "__main__":
    _tick("entered __main__ (all imports done)")
    parser = track4d360_parser()
    args = parser.parse_args()
    _tick("parsed argparse")

    if getattr(args, "debug_missing_grads", False):
        os.environ["TRACK4D_DEBUG_MISSING_GRADS"] = "1"
    _torch_dd = getattr(args, "torch_distributed_debug", None)
    if _torch_dd:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = str(_torch_dd)

    # Force V2V task
    args.task = "sft:v2v"

    # Load track config unless this is an explicit plucker-only / raw V2V run.
    track_cfg = None
    if not getattr(args, "disable_track_modules", False):
        track_cfg = load_track_config(args.track_config)
        _rank0_print(f"[Track4D-360] Track config: {dict(track_cfg)}")
    else:
        _rank0_print("[Track4D-360] disable_track_modules=True: track config ignored.")

    # Resolve mode: channel-concat takes precedence over legacy width-concat flags
    channel_concat_stage = getattr(args, "channel_concat_stage", None)
    if channel_concat_stage is not None:
        mode_cfg = resolve_channel_concat_mode(
            stage=channel_concat_stage,
            use_residual_tracks=getattr(args, "use_residual_tracks", True),
        )
        _rank0_print(f"[Track4D-360] Channel-concat mode: stage={channel_concat_stage}, "
                     f"modalities={mode_cfg['channel_modalities']}")
    else:
        mode_cfg = resolve_track4d360_mode(
            joint_track_prediction=getattr(args, "joint_track_prediction", False),
            joint_depth_prediction=getattr(args, "joint_depth_prediction", False),
            joint_track_depth_prediction=getattr(args, "joint_track_depth_prediction", False),
        )

    # Dynamic GS mode requires the dense per-pixel target track GT for the
    # position-map loss term AND the target-view depth GT for the depth loss
    # term — auto-flip both dataset emitters on so the launcher stays a
    # single-flag change. Mutates the dict in place so both the dataset
    # constructor and the model side see the same view.
    if getattr(args, "gs_dynamic_mode", "static") == "deltas":
        if not mode_cfg.get("return_target_tracks", False):
            mode_cfg["return_target_tracks"] = True
            _rank0_print("[Track4D-360] gs_dynamic_mode=deltas: auto-enabled "
                         "mode_cfg['return_target_tracks']=True")
        if not mode_cfg.get("return_target_depth", False):
            mode_cfg["return_target_depth"] = True
            _rank0_print("[Track4D-360] gs_dynamic_mode=deltas: auto-enabled "
                         "mode_cfg['return_target_depth']=True")

    # use_gs_output requires target depth GT for gs_depth_loss /
    # gs_abs_depth_loss (static-mode render-side losses). Auto-enable
    # regardless of gs_dynamic_mode so static-mode runs (v3b) actually
    # see depth supervision instead of silently zeroing the depth loss.
    if getattr(args, "use_gs_output", False):
        if not mode_cfg.get("return_target_depth", False):
            mode_cfg["return_target_depth"] = True
            _rank0_print("[Track4D-360] use_gs_output=True: auto-enabled "
                         "mode_cfg['return_target_depth']=True (needed for "
                         "gs_depth_loss / gs_abs_depth_loss).")
    if getattr(args, "ar_training", False) and getattr(args, "use_gs_output", False):
        raise NotImplementedError(
            "--ar_training with --use_gs_output is intentionally disabled for the first AR implementation."
        )

    # v4 (static_with_track): also auto-enable target_tracks emission so
    # mixed datasets that supply 3D tracks (Kubric_NVS) feed the track
    # head. Datasets that don't (StaticScene_NVS) ignore the flag — the
    # track loss branch handles None gracefully.
    if getattr(args, "gs_dynamic_mode", "static") == "static_with_track":
        if not mode_cfg.get("return_target_tracks", False):
            mode_cfg["return_target_tracks"] = True
            _rank0_print("[Track4D-360] gs_dynamic_mode=static_with_track: "
                         "auto-enabled mode_cfg['return_target_tracks']=True")

    # Any GS-output run: force return_target_depth=True. The static-path
    # depth loss needs it (gated on gs_depth_loss_weight>0 internally),
    # the dynamic path always needs it for dyn_depth_loss, and the
    # debug_latents tgt_depth panel needs it regardless of loss weights.
    # Previously gated on gs_depth_loss_weight > 0; that left the panel
    # empty when the weight defaulted to 0 (e.g. dynamic launchers that
    # set gs_depth_loss_weight=0 and rely on gs_dyn_depth_loss_weight
    # instead). 2026-04-26.
    if (getattr(args, "use_gs_output", False)
            and not mode_cfg.get("return_target_depth", False)):
        mode_cfg["return_target_depth"] = True
        _rank0_print("[Track4D-360] use_gs_output: auto-enabled "
                     "mode_cfg['return_target_depth']=True")

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        visible_cuda_devices = int(torch.cuda.device_count())
        if visible_cuda_devices <= 0:
            raise RuntimeError("torch.cuda.is_available() is True but torch.cuda.device_count() == 0")
        if local_rank >= visible_cuda_devices:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {visible_cuda_devices} CUDA device(s) are visible "
                f"in this process (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}). "
                "This would place multiple ranks on the same GPU. Fix the launcher/device visibility."
            )
        torch.cuda.set_device(local_rank)

    ddp_kwargs = accelerate.DistributedDataParallelKwargs(
        find_unused_parameters=bool(getattr(args, "find_unused_parameters", False)),
    )
    dist_timeout_s = int(os.environ.get("TORCH_DISTRIBUTED_DEFAULT_TIMEOUT", "1800"))
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=dist_timeout_s))
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
    )
    if _IS_RANK0:
        print(f"[distributed] process group timeout: {dist_timeout_s}s", flush=True)
    print("gradient accumulation:", args.gradient_accumulation_steps)

    # Create dataset
    if args.dummy:
        from holo4d.data.syn4d_paired_adapter import DummySyn4DPairedDataset

        dataset = DummySyn4DPairedDataset(
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            track_spatial_stride=args.track_spatial_stride,
            steps_per_epoch=args.steps_per_epoch,
        )
        print(f"[Track4D-360] Using DUMMY dataset: {len(dataset)} samples")
    elif args.track4d360_data_config:
        # YAML-based compositional dataset loading (preferred).
        # Builds a CatDataset from "N @ DatasetClass(...)" strings, then lets
        # launch_training_task create the DataLoader with its standard collate.
        from pathlib import Path
        from omegaconf import OmegaConf

        config_path = Path(args.track4d360_data_config).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Track4D-360 data config not found: {config_path}")

        cfg = OmegaConf.load(str(config_path))
        # Override resolution/num_views with runtime launcher args so that
        # ${num_views} and ${resolution} in train_datasets strings resolve to
        # the actual CLI values, not the YAML defaults.
        cfg.num_views = args.num_frames
        cfg.resolution = [args.width, args.height]
        cfg = OmegaConf.to_container(cfg, resolve=True)

        train_datasets = cfg.get("train_datasets")
        if not train_datasets:
            raise ValueError("Track4D-360 data config must provide `train_datasets` list.")
        train_datasets = _apply_mode_cfg_to_yaml_train_datasets(train_datasets, mode_cfg)
        if (str(getattr(args, "dense_warp_mode", "zbuffer")).lower() == "hybrid_softsplat"
                and not bool(getattr(args, "dense_hybrid_disable_dynamic_masks", False))):
            # Auto-enable per-frame foreground masks on datasets that support
            # them. Both Kubric_NVS and Syn4D_Multi_NVS emit
            # `source_dynamic_mask` / `target_dynamic_mask` from per-frame
            # segmentation (Kubric: segm PNGs; Syn4D: NOT(union of `_env*`
            # masks)). The dense unit uses these directly and bypasses the
            # threshold-based `tracks_world` motion fallback, which
            # under-classifies Syn4D actor motion.
            train_datasets = [
                _ensure_dataset_constructor_kwarg(expr, "return_dynamic_masks", "True")
                if ("Kubric_NVS" in expr or "Syn4D_Multi_NVS" in expr) else expr
                for expr in train_datasets
            ]

        # eval() each dataset string via datasets_cut3r namespace (imports all classes).
        # "N @ Syn4D_Multi_NVS(...)" → ResizedDataset, joined by " + " → CatDataset.
        import holo4d_runtime.data.stream3r.dust3r.datasets_cut3r as _ds_ns  # noqa: for eval scope
        train_datasets_concat = " + ".join(train_datasets)

        if accelerator.is_main_process:
            print(f"[Track4D-360] YAML config: {config_path}")
            print(f"[Track4D-360] Dataset string: {train_datasets_concat}")

        dataset = eval(train_datasets_concat, vars(_ds_ns))
        print(f"[Track4D-360] Dataset ready: {type(dataset).__name__}, {len(dataset)} samples")

        # Optional: force track_video_loss_mask override from YAML
        _force_track_mask = cfg.get("force_track_video_loss_mask", None)
        if _force_track_mask is not None:
            print(f"[Track4D-360] force_track_video_loss_mask={_force_track_mask} (from YAML)")
        else:
            _force_track_mask = None
    else:
        # Direct mode (fallback): single dataset from CLI args
        from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.syn4d_nvs import Syn4D_Multi_NVS

        # Dense GEN3C-style target-space control branch requires per-frame
        # source RGB, source depth, and source cameras.  When --use_dense_branch
        # is set we flip on the corresponding dataset emitters automatically.
        _use_dense = bool(getattr(args, "use_dense_branch", False))
        _need_source_depth = _use_dense or bool(mode_cfg.get("return_source_depth_video", False))
        dataset = Syn4D_Multi_NVS(
            ROOT=args.syn4d_root,
            num_views=args.num_frames,
            resolution=(args.width, args.height),
            steps_per_epoch=args.steps_per_epoch,
            track_spatial_stride=args.track_spatial_stride,
            scene=args.scene,
            sequence_idx=args.sequence_idx,
            return_target_tracks=bool(mode_cfg["return_target_tracks"]),
            return_target_depth=bool(mode_cfg["return_target_depth"]),
            return_source_depth_video=_need_source_depth,
            return_source_rgb_video=_use_dense,
            return_source_cameras=_use_dense,
        )
        print(f"[Track4D-360] Loaded Syn4D_Multi_NVS: {dataset}")

    # Create model
    model = WanTrack4D360TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if (args.dummy or args.initialize_model_on_cpu) else accelerator.device,
        v2v_concat_mode=args.v2v_concat_mode,
        model_variant=args.model_variant,
        track_cfg=track_cfg,
        drop_text_prob=args.drop_text_prob,
        i2v_target_condition=getattr(args, "i2v_target_condition", False),
        i2v_condition_drop_prob=getattr(args, "i2v_condition_drop_prob", 0.3),
        i2v_condition_noise_level=getattr(args, "i2v_condition_noise_level", 0.05),
        dit_finetune_modules=args.dit_finetune_modules,
        track_injection_mode=args.track_injection_mode,
        track_injection_block_mode=getattr(args, "track_injection_block_mode", "additive"),
        resume_checkpoint=getattr(args, "resume_checkpoint", None),
        gs_head_init_checkpoint=getattr(args, "gs_head_init_checkpoint", None),
        camera_control_checkpoint=getattr(args, "camera_control_checkpoint", None),
        joint_track_prediction=getattr(args, "joint_track_prediction", False),
        joint_depth_prediction=getattr(args, "joint_depth_prediction", False),
        joint_track_depth_prediction=getattr(args, "joint_track_depth_prediction", False),
        use_query_frame_impulse_condition=getattr(args, "use_query_frame_impulse_condition", False),
        future_prediction=getattr(args, "future_prediction", False),
        future_num_latent_frames=getattr(args, "future_num_latent_frames", 0),
        future_loss_weight=getattr(args, "future_loss_weight", 1.0),
        future_source_null=getattr(args, "future_source_null", False),
        target_depth_log_cap=getattr(args, "target_depth_log_cap", 14.0),
        target_depth_norm_type=getattr(args, "target_depth_norm_type", "log_depth"),
        target_depth_max_disparity=getattr(args, "target_depth_max_disparity", 5.0),
        channel_concat_stage=channel_concat_stage,
        use_residual_tracks=getattr(args, "use_residual_tracks", True),
        use_plucker_camera_control=getattr(args, "use_plucker_camera_control", False),
        enable_v2v_plucker_camera_control=getattr(args, "enable_v2v_plucker_camera_control", False),
        train_plucker_camera_control=getattr(args, "train_plucker_camera_control", False),
        use_plucker_raymap_tokens=getattr(args, "use_plucker_raymap_tokens", False),
        use_perlayer_camera_control=getattr(args, "use_perlayer_camera_control", False),
        camera_injection_mode=getattr(args, "camera_injection_mode", "projector"),
        camera_plucker_encoder_mode=getattr(args, "camera_plucker_encoder_mode", "patchify"),
        freeze_lora_after_load=getattr(args, "freeze_lora_after_load", False),
        skip_v2v_required_trainables=getattr(args, "skip_v2v_required_trainables", False),
        disable_domain_embedding=getattr(args, "disable_domain_embedding", False),
        gs_only_training=getattr(args, "gs_only_training", False),
        gs_head_style=getattr(args, "gs_head_style", "flashworld"),
        gs_lyra_depth_param=args.gs_lyra_depth_param,
        gs_lyra_log_depth_shift=args.gs_lyra_log_depth_shift,
        gs_lyra_dnear=args.gs_lyra_dnear,
        gs_lyra_dfar=args.gs_lyra_dfar,
        fixed_step_frac=getattr(args, "fixed_step_frac", None),
        fixed_num_inference_steps=getattr(args, "fixed_num_inference_steps", 50),
        fixed_sigma=getattr(args, "fixed_sigma", None),
        skip_diffusion_loss=getattr(args, "skip_diffusion_loss", False),
        freeze_track_block_injector=getattr(args, "freeze_track_block_injector", False),
        ar_training=getattr(args, "ar_training", False),
        ar_training_algorithm=getattr(args, "ar_training_algorithm", "windowed_teacher_forcing"),
        ar_num_frame_per_block=getattr(args, "ar_num_frame_per_block", 3),
        ar_denoising_step_list=getattr(args, "ar_denoising_step_list", "1000,800,600,400,200"),
        ar_same_step_across_blocks=getattr(args, "ar_same_step_across_blocks", False),
        ar_grad_through_context_cache=getattr(args, "ar_grad_through_context_cache", False),
        ar_context_noise=getattr(args, "ar_context_noise", 0.0),
        ar_context_noise_step=getattr(args, "ar_context_noise_step", 0),
        ar_rolling_window_blocks=getattr(args, "ar_rolling_window_blocks", 0),
        ar_cache_max_latent_frames=getattr(args, "ar_cache_max_latent_frames", 21),
        ar_cache_keep_first_latent=getattr(args, "ar_cache_keep_first_latent", False),
        ar_loss_last_latent_frames=getattr(args, "ar_loss_last_latent_frames", 21),
        use_gs_output=getattr(args, "use_gs_output", False),
        gs_zero_features=getattr(args, "gs_zero_features", False),
        use_gs_feedback_training=getattr(args, "use_gs_feedback_training", False),
        gs_feat_dim=getattr(args, "gs_feat_dim", 1024),
        gs_loss_weight=getattr(args, "gs_loss_weight", 1.0),
        gs_mse_loss_weight=(
            getattr(args, "gs_mse_loss_weight", 1.0)
            if getattr(args, "gs_rgb_loss_weight", None) is None
            else getattr(args, "gs_rgb_loss_weight")
        ),
        gs_lpips_loss_weight=getattr(args, "gs_lpips_loss_weight", 0.5),
        gs_abs_depth_loss_weight=args.gs_abs_depth_loss_weight,
        gs_depth_loss_weight=getattr(args, "gs_depth_loss_weight", 0.05),
        gs_opacity_loss_weight=getattr(args, "gs_opacity_loss_weight", 0.1),
        gs_opacity_prune_keep_frac=getattr(args, "gs_opacity_prune_keep_frac", 1.0),
        gs_sh_degree=getattr(args, "gs_sh_degree", 2),
        gs_num_points_per_pixel=getattr(args, "gs_num_points_per_pixel", 2),
        gs_scale_cap=getattr(args, "gs_scale_cap", None),
        gs_render_patch_size=getattr(args, "gs_render_patch_size", None),
        gs_legacy_union_render=getattr(args, "gs_legacy_union_render", False),
        gs_replay_cache_size=getattr(args, "gs_replay_cache_size", 0),
        no_gs_latent_detach=getattr(args, "no_gs_latent_detach", False),
        gs_dynamic_mode=getattr(args, "gs_dynamic_mode", "static"),
        gs_pos_loss_weight=getattr(args, "gs_pos_loss_weight", 1.0),
        gs_dyn_depth_loss_weight=getattr(args, "gs_dyn_depth_loss_weight", 0.5),
        gs_opa_tv_weight=getattr(args, "gs_opa_tv_weight", 0.01),
        gs_xyz_tv_weight=getattr(args, "gs_xyz_tv_weight", 0.01),
        gs_predict_rotation=getattr(args, "gs_predict_rotation", False),
        gs_dynamic_max_t_video=getattr(args, "gs_dynamic_max_t_video", 64),
        gs_track_canonical_loss_weight=args.gs_track_canonical_loss_weight,
        gs_track_delta_loss_weight=args.gs_track_delta_loss_weight,
        gs_track_final_pos_loss_weight=args.gs_track_final_pos_loss_weight,
        gs_assoc_track_2d_loss_weight=args.gs_assoc_track_2d_loss_weight,
        gs_track_accel_loss_weight=args.gs_track_accel_loss_weight,
        gs_track_loss_quantile=args.gs_track_loss_quantile,
        gs_track_canonical_detach=getattr(args, "gs_track_canonical_detach", False),
        gs_track_joint_render=getattr(args, "gs_track_joint_render", False),
        gs_track_supervision_mode=args.gs_track_supervision_mode,
        disable_track_modules=getattr(args, "disable_track_modules", False),
        force_t2v_drop_source_cond=getattr(args, "force_t2v_drop_source_cond", False),
        raw_t2v_target_only=getattr(args, "raw_t2v_target_only", False),
        dense_only_condition_modules=getattr(args, "dense_only_condition_modules", False),
        freeze_track4d_condition_modules=getattr(args, "freeze_track4d_condition_modules", False),
        use_dense_branch=getattr(args, "use_dense_branch", False),
        dense_in_channels=getattr(args, "dense_in_channels", 5),
        dense_proj_dim=getattr(args, "dense_proj_dim", 32),
        dense_num_residual_blocks=getattr(args, "dense_num_residual_blocks", 2),
        dense_alpha_track=getattr(args, "dense_alpha_track", 1.0),
        dense_conditioning_noise_max_level=getattr(args, "dense_conditioning_noise_max_level", 0.0),
        source_mask_boundary_depth_dropout_prob=getattr(args, "source_mask_boundary_depth_dropout_prob", 0.0),
        source_mask_boundary_kernel_size=getattr(args, "source_mask_boundary_kernel_size", 5),
        dense_hybrid_disable_dynamic_masks=getattr(args, "dense_hybrid_disable_dynamic_masks", False),
        use_depth_null_token=getattr(args, "use_depth_null_token", False),
        cond_drop_track_prob=getattr(args, "cond_drop_track_prob", 0.0),
        cond_drop_dense_prob=getattr(args, "cond_drop_dense_prob", 0.0),
        cond_drop_geo_prob=getattr(args, "cond_drop_geo_prob", 0.0),
        use_track_null_token_fallback=getattr(args, "use_track_null_token_fallback", False),
        use_dense_null_token_fallback=getattr(args, "use_dense_null_token_fallback", False),
        # Warped appearance conditioning
        warped_condition_mode=getattr(args, "warped_condition_mode", "off"),
        warped_appearance_fusion=getattr(args, "warped_appearance_fusion", "concat_proj"),
        warped_geom_only_dense=getattr(args, "warped_geom_only_dense", True),
        multi_frame_warped_appearance=getattr(args, "multi_frame_warped_appearance", False),
        multi_frame_motion_threshold=getattr(args, "multi_frame_motion_threshold", 0.01),
        # Dense-branch warp backend (zbuffer / hybrid_softsplat)
        dense_warp_mode=getattr(args, "dense_warp_mode", "zbuffer"),
        dense_warp_motion_threshold=getattr(args, "dense_warp_motion_threshold", 0.005),
        use_vista4d_latent_streams=getattr(args, "use_vista4d_latent_streams", False),
        vista4d_streams_pointcloud_only=getattr(args, "vista4d_streams_pointcloud_only", False),
        use_gen3c_warp_latent_channels=getattr(args, "use_gen3c_warp_latent_channels", False),
    )

    # Apply force_track_video_loss_mask from YAML config (if set)
    if "_force_track_mask" in dir() and _force_track_mask is not None:
        model._force_track_video_loss_mask = float(_force_track_mask)
    else:
        model._force_track_video_loss_mask = None

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        log_freq=args.wandb_log_freq,
        wandb_log_gradients=args.wandb_log_gradients,
        wandb_log_parameters=args.wandb_log_parameters,
        max_checkpoints_to_keep=args.max_checkpoints_to_keep,
        allow_existing_checkpoints=args.allow_existing_checkpoints,
        initial_step=getattr(args, "initial_step", 0),
    )

    _tick("about to call launch_training_task (pipeline+model already built)")
    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
    )
