"""
Wan video-to-video (V2V) pipeline.

This pipeline subclasses `diffsynth.pipelines.wan_video.WanVideoPipeline` to keep feature
parity and API compatibility (e.g. `from_pretrained`, `__call__`, `enable_usp`), and adds:
- `disp_video` or `depth_normalized` -> `depth_latents` encoding via the shared VAE
- concatenation of RGB + depth latents for training (`frame`, `width`, or `channel`)
- a V2V-aware `model_fn` that injects domain embeddings and adjusts RoPE positions
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import Optional, Literal, Union, Dict

import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from ..core import ModelConfig
from ..diffusion.base_pipeline import PipelineUnit
try:
    from ..models.longcat_video_dit import LongCatVideoTransformer3DModel
except ImportError:
    LongCatVideoTransformer3DModel = None  # Not needed for Wan V2V
from ..models.wan_video_dit import DiTBlock, modulate, sinusoidal_embedding_1d
from ..models.wan_video_dit_v2v import WanModelV2V
from ..utils.camera_control_nvs import (
    process_camera_pair_to_plucker,
    process_camera_pair_to_plucker_dit_resolution,
    prepare_camera_control_latents,
    prepare_camera_control_pose_latents,
    prepare_camera_control_pose_latents_raw,
)
from ..utils.geometry import compute_signed_log_quantiles, normalize_track_signed_log
from holo4d.utils.torch_profile import track4d_record_function

from .vista4d_stream_layout import (
    expand_vista4d_streams_for_v2v,
)
from .wan_video import (
    WanVideoPipeline as WanVideoPipelineBase,
    WanVideoUnit_InputVideoEmbedder,
    WanVideoUnit_ShapeChecker,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
    WanVideoUnit_ImageEmbedderFused,
    WanVideoUnit_UnifiedSequenceParallel,
    WanVideoUnit_TeaCache,
    WanVideoUnit_CfgMerger,
    TeaCache,
    TemporalTiler_BCTHW,
    model_fn_wan_video,
)

from pdb import set_trace as st


def _ensure_domain_embedding(dit: Optional[torch.nn.Module], num_domains: int = 2) -> None:
    """
    Ensure domain_embedding exists with the correct number of domains.

    Args:
        dit: DiT model
        num_domains: Number of domains (2 for RGB-only, 3 for RGB+Depth multi-modality)
    """
    if dit is None:
        return
    if not hasattr(dit, "dim"):
        raise AttributeError("V2V requires `dit.dim` to create `domain_embedding`.")

    # Check if domain_embedding exists and has correct size
    if hasattr(dit, "domain_embedding"):
        existing_embedding = dit.domain_embedding
        if existing_embedding.shape[0] == num_domains:
            return  # Already correct size

        # Resize domain_embedding if size mismatch (expected when fine-tuning for multi-modality)
        print(f"ℹ️  Resizing domain_embedding: {existing_embedding.shape[0]} → {num_domains} domains")

        # Get device and dtype from existing embedding
        device, dtype = existing_embedding.device, existing_embedding.dtype
        dim = int(getattr(dit, "dim"))

        if existing_embedding.shape[0] < num_domains:
            # Expanding: keep existing tokens and add new ones
            # Important: New tokens (e.g., depth modality) should be freshly initialized,
            # NOT copied from RGB tokens, since they represent different modalities
            new_tokens_count = num_domains - existing_embedding.shape[0]
            new_tokens = torch.zeros(new_tokens_count, dim, device=device, dtype=dtype)
            torch.nn.init.normal_(new_tokens, std=0.02)

            # Concatenate existing tokens with new tokens
            # For 2→3 expansion: [source_rgb, target_rgb] → [source_rgb, target_rgb, target_depth]
            # The new target_depth token is randomly initialized, not copied from target_rgb
            expanded_embedding = torch.cat([existing_embedding.data, new_tokens], dim=0)
            dit.domain_embedding.data = expanded_embedding
            print(f"  ✓ Expanded domain_embedding by adding {new_tokens_count} new token(s) (randomly initialized)")
        else:
            # Shrinking: truncate to keep only first num_domains tokens
            dit.domain_embedding.data = existing_embedding.data[:num_domains]
            print(f"  ✓ Truncated domain_embedding to {num_domains} tokens")

        return

    # Create new domain_embedding if it doesn't exist
    ref_param = next(dit.parameters(), None)
    if ref_param is None or getattr(ref_param.device, "type", None) == "meta":
        device, dtype = "cpu", torch.float32
    else:
        device, dtype = ref_param.device, ref_param.dtype

    # Create domain_embedding: learnable tokens to distinguish different modalities/domains
    # For RGB-only (num_domains=2):
    #   - domain_embedding[0] = source RGB
    #   - domain_embedding[1] = target RGB
    # For RGB+Depth multi-modality (num_domains=3):
    #   - domain_embedding[0] = source RGB
    #   - domain_embedding[1] = target RGB
    #   - domain_embedding[2] = target Depth
    # For full multi-modality (num_domains=4):
    #   - domain_embedding[0] = source RGB
    #   - domain_embedding[1] = target RGB
    #   - domain_embedding[2] = source Depth
    #   - domain_embedding[3] = target Depth
    domain_embedding = torch.nn.Parameter(torch.zeros(num_domains, int(getattr(dit, "dim")), device=device, dtype=dtype))
    torch.nn.init.normal_(domain_embedding, std=0.02)
    setattr(dit, "domain_embedding", domain_embedding)
    print(f"  ✓ Created domain_embedding with {num_domains} tokens")


def _module_or_param_requires_grad(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, torch.nn.Parameter):
        return bool(obj.requires_grad)
    if torch.is_tensor(obj):
        return bool(getattr(obj, "requires_grad", False))
    if isinstance(obj, torch.nn.Module):
        return any(param.requires_grad for param in obj.parameters())
    return False


def _dit_hidden_state_has_trainable_sources(
    dit: Optional[torch.nn.Module],
    extra_trainable_sources=None,
) -> bool:
    """Return True when any trainable source can still influence the DiT hidden state.

    This is stricter than checking the first parameter of the first block:
    partial fine-tuning commonly freezes some early tensors (for example the
    modulation weights) while enabling gradients for deeper submodules such as
    ``self_attn``/``ffn`` or optional control adapters. Detaching hidden states
    in that case severs valid gradients and causes DDP unused-parameter errors.
    """
    if dit is None:
        return False

    trainable_sources = [
        getattr(dit, "blocks", None),
        getattr(dit, "control_adapter", None),
        getattr(dit, "control_patch_embedding", None),
        getattr(dit, "patch_embedding", None),
        getattr(dit, "patch_embedding_v2v", None),
        getattr(dit, "_multimodal_patch_embeddings", None),
        getattr(dit, "domain_embedding", None),
        getattr(dit, "_camera_patchify_conv", None),
        getattr(dit, "_camera_plucker_patchify_encoder", None),
        getattr(dit, "vista4d_source_patch_embedding", None),
        getattr(dit, "vista4d_pointcloud_patch_embedding", None),
        getattr(dit, "vista4d_source_mask_patch_embedding", None),
        getattr(dit, "vista4d_pointcloud_mask_patch_embedding", None),
        getattr(dit, "gen3c_warp_condition_patch_embedding", None),
    ]
    if extra_trainable_sources:
        trainable_sources.extend(extra_trainable_sources)

    return any(_module_or_param_requires_grad(source) for source in trainable_sources)


_TRACK4D_DIT_ORIGINAL_FORWARD = DiTBlock.forward
_TRACK4D_RUNTIME_ATTRS = (
    "_track4d_runtime_mode",
    "_track4d_runtime_camera_features",
    "_track4d_runtime_track_tokens",
    "_track4d_runtime_track_injector",
    "_track4d_runtime_block_idx",
    "_track4d_runtime_kv_cache",
    "_track4d_runtime_updating_cache",
    "_track4d_runtime_frame_seq_length",
    "_track4d_runtime_cache_max_latent_frames",
    "_track4d_runtime_cache_keep_first_latent",
    "_track4d_runtime_causal_attn_mask",
    "_camera_condition_scale",
    "_camera_injection_mode",
)
_TRACK4D_RUNTIME_LOGGED_MODES = set()


def _track4d_project_vista4d_camera(
    block: DiTBlock,
    camera_features: Optional[torch.Tensor],
    camera_scale: float,
) -> Optional[torch.Tensor]:
    if camera_features is None:
        return None
    if not hasattr(block, "cam_encoder"):
        raise RuntimeError("vista4d_plucker mode requires per-block Linear cam_encoder modules.")
    if camera_features.ndim != 3 or int(camera_features.shape[-1]) != 6:
        raise ValueError(
            "vista4d_plucker mode expects raw Plucker tokens with shape "
            f"(B, L, 6), got {tuple(camera_features.shape)}"
        )
    return block.cam_encoder(camera_features) * camera_scale


def _track4d_extract_modulation(block: DiTBlock, t_mod: torch.Tensor):
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=chunk_dim)
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
        )
    return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _track4d_self_attn(block: DiTBlock, x: torch.Tensor, freqs: torch.Tensor, **kwargs) -> torch.Tensor:
    """Call self-attention with optional AR cache state attached to the block."""

    ar_kwargs = {
        "kv_cache": getattr(block, "_track4d_runtime_kv_cache", None),
        "updating_cache": bool(getattr(block, "_track4d_runtime_updating_cache", True)),
        "frame_seq_length": getattr(block, "_track4d_runtime_frame_seq_length", None),
        "ar_cache_max_latent_frames": getattr(block, "_track4d_runtime_cache_max_latent_frames", None),
        "ar_cache_keep_first_latent": getattr(block, "_track4d_runtime_cache_keep_first_latent", False),
    }
    ar_kwargs.update(kwargs)
    return block.self_attn(x, freqs, **ar_kwargs)


def _track4d_forward_block_with_camera(
    block: DiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    camera_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if camera_features is None or not hasattr(block, "camera_projector"):
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    camera_scale = float(getattr(block, "_camera_condition_scale", 1.0))
    if camera_scale == 0.0:
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        _track4d_extract_modulation(block, t_mod)
    )
    input_x = modulate(block.norm1(x), shift_msa, scale_msa)

    cam_injection_mode = getattr(block, "_camera_injection_mode", None)
    if cam_injection_mode == "rt_projector":
        if not hasattr(block, "camera_pose_encoder"):
            raise RuntimeError("rt_projector mode requires per-layer `camera_pose_encoder` modules on DiT blocks.")
        camera_features = block.camera_pose_encoder(camera_features)

    input_x = input_x + camera_features * camera_scale
    attn_out = _track4d_self_attn(block, input_x, freqs)
    x = x + gate_msa * block.camera_projector(attn_out)
    x = x + block.cross_attn(block.norm3(x), context)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = x + gate_mlp * block.ffn(input_x)
    return x


def _track4d_forward_block_with_camera_add(
    block: DiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    camera_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if camera_features is None or not hasattr(block, "cam_add_projector"):
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    camera_scale = float(getattr(block, "_camera_condition_scale", 1.0))
    if camera_scale != 0.0:
        x = x + camera_scale * block.cam_add_projector(camera_features)
    return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)


def _track4d_forward_block_with_direct_raymap(
    block: DiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    camera_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if camera_features is None or not hasattr(block, "cam_encoder"):
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    camera_scale = float(getattr(block, "_camera_condition_scale", 1.0))
    if camera_scale == 0.0:
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    cam_emb_5d = block.cam_encoder(camera_features) * camera_scale
    _, _, f, h, w = cam_emb_5d.shape
    cam_emb = rearrange(cam_emb_5d, "b c f h w -> b (f h w) c").contiguous()
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        _track4d_extract_modulation(block, t_mod)
    )
    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    input_x = input_x + cam_emb
    attn_out = _track4d_self_attn(block, input_x, freqs)
    if not hasattr(block, "camera_projector"):
        raise RuntimeError(
            "direct_raymap_add mode requires per-layer camera_projector. "
            "Initialize camera encoders with add_projector=True."
        )
    x = x + gate_msa * block.camera_projector(attn_out)
    x = x + block.cross_attn(block.norm3(x), context)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = x + gate_mlp * block.ffn(input_x)
    return x


def _track4d_forward_block_with_vista4d_plucker(
    block: DiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    camera_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if camera_features is None or not hasattr(block, "cam_encoder"):
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    camera_scale = float(getattr(block, "_camera_condition_scale", 1.0))
    if camera_scale == 0.0:
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    cam_emb = _track4d_project_vista4d_camera(block, camera_features, camera_scale)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        _track4d_extract_modulation(block, t_mod)
    )
    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    input_x = input_x + cam_emb
    attn_out = _track4d_self_attn(block, input_x, freqs)
    if not hasattr(block, "camera_projector"):
        raise RuntimeError(
            "vista4d_plucker mode requires per-layer camera_projector modules. "
            "Initialize camera encoders with add_projector=True."
        )
    x = x + gate_msa * block.camera_projector(attn_out)
    x = x + block.cross_attn(block.norm3(x), context)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = x + gate_mlp * block.ffn(input_x)
    return x


def _track4d_forward_block_with_kq_only(
    block: DiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    camera_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if camera_features is None or not hasattr(block, "cam_encoder"):
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    camera_scale = float(getattr(block, "_camera_condition_scale", 1.0))
    if camera_scale == 0.0:
        return _TRACK4D_DIT_ORIGINAL_FORWARD(block, x, context, t_mod, freqs)

    cam_emb_5d = block.cam_encoder(camera_features) * camera_scale
    cam_emb = rearrange(cam_emb_5d, "b c f h w -> b (f h w) c").contiguous()
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        _track4d_extract_modulation(block, t_mod)
    )
    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    attn_out = _track4d_self_attn(block, input_x, freqs, kq_bias=cam_emb)
    if hasattr(block, "camera_projector"):
        x = x + gate_msa * block.camera_projector(attn_out)
    else:
        x = x + gate_msa * attn_out
    x = x + block.cross_attn(block.norm3(x), context)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = x + gate_mlp * block.ffn(input_x)
    return x


def _track4d_forward_block_with_track(
    block: DiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    camera_features: Optional[torch.Tensor],
    track_tokens: torch.Tensor,
    track_injector: torch.nn.Module,
    block_idx: int,
) -> torch.Tensor:
    with track4d_record_function(f"track4d360/dit/block{block_idx:02d}/modulate"):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            _track4d_extract_modulation(block, t_mod)
        )
        input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    input_x = track_injector.inject(input_x, track_tokens, block_idx)

    if camera_features is not None and hasattr(block, "camera_projector"):
        with track4d_record_function(f"track4d360/dit/block{block_idx:02d}/camera_condition"):
            camera_scale = float(getattr(block, "_camera_condition_scale", 1.0))
            cam_injection_mode = getattr(block, "_camera_injection_mode", None)
            if cam_injection_mode == "rt_projector" and hasattr(block, "camera_pose_encoder"):
                camera_features = block.camera_pose_encoder(camera_features)
            elif cam_injection_mode == "vista4d_plucker":
                camera_features = _track4d_project_vista4d_camera(block, camera_features, camera_scale)
                camera_scale = 1.0
            input_x = input_x + camera_features * camera_scale
        with track4d_record_function(f"track4d360/dit/block{block_idx:02d}/self_attn"):
            attn_out = _track4d_self_attn(block, input_x, freqs)
        with track4d_record_function(f"track4d360/dit/block{block_idx:02d}/camera_projector"):
            x = block.gate(x, gate_msa, block.camera_projector(attn_out))
    else:
        with track4d_record_function(f"track4d360/dit/block{block_idx:02d}/self_attn"):
            x = block.gate(x, gate_msa, _track4d_self_attn(block, input_x, freqs))

    with track4d_record_function(f"track4d360/dit/block{block_idx:02d}/cross_attn"):
        x = x + block.cross_attn(block.norm3(x), context)
    with track4d_record_function(f"track4d360/dit/block{block_idx:02d}/ffn"):
        input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(input_x))
    return x


def _track4d_maybe_log_runtime_mode_once(self: DiTBlock, mode: str) -> None:
    if os.environ.get("RANK", "0") != "0":
        return
    if mode in _TRACK4D_RUNTIME_LOGGED_MODES:
        return
    block_idx = getattr(self, "_track4d_runtime_block_idx", None)
    print(
        "[track4d-runtime] rank0 patched DiTBlock.forward saw "
        f"mode={mode} block_idx={block_idx}",
        flush=True,
    )
    _TRACK4D_RUNTIME_LOGGED_MODES.add(mode)


def _track4d_patched_dit_block_forward(
    self: DiTBlock,
    x,
    context,
    t_mod,
    freqs,
    kv_cache=None,
    updating_cache: bool = True,
    frame_seq_length: int | None = None,
    ar_cache_max_latent_frames: int | None = None,
    ar_cache_keep_first_latent: bool = False,
    causal_attn_mask: Optional[torch.Tensor] = None,
):
    mode = getattr(self, "_track4d_runtime_mode", None)
    if mode is None:
        if kv_cache is None:
            kv_cache = getattr(self, "_track4d_runtime_kv_cache", None)
            updating_cache = bool(getattr(self, "_track4d_runtime_updating_cache", updating_cache))
            frame_seq_length = getattr(self, "_track4d_runtime_frame_seq_length", frame_seq_length)
            ar_cache_max_latent_frames = getattr(
                self,
                "_track4d_runtime_cache_max_latent_frames",
                ar_cache_max_latent_frames,
            )
            ar_cache_keep_first_latent = bool(getattr(
                self,
                "_track4d_runtime_cache_keep_first_latent",
                ar_cache_keep_first_latent,
            ))
            causal_attn_mask = getattr(self, "_track4d_runtime_causal_attn_mask", causal_attn_mask)
        return _TRACK4D_DIT_ORIGINAL_FORWARD(
            self,
            x,
            context,
            t_mod,
            freqs,
            kv_cache=kv_cache,
            updating_cache=updating_cache,
            frame_seq_length=frame_seq_length,
            ar_cache_max_latent_frames=ar_cache_max_latent_frames,
            ar_cache_keep_first_latent=ar_cache_keep_first_latent,
            causal_attn_mask=causal_attn_mask,
        )
    _track4d_maybe_log_runtime_mode_once(self, mode)
    if mode == "camera":
        return _track4d_forward_block_with_camera(
            self, x, context, t_mod, freqs,
            getattr(self, "_track4d_runtime_camera_features", None),
        )
    if mode == "camera_add":
        return _track4d_forward_block_with_camera_add(
            self, x, context, t_mod, freqs,
            getattr(self, "_track4d_runtime_camera_features", None),
        )
    if mode == "direct_raymap_add":
        return _track4d_forward_block_with_direct_raymap(
            self, x, context, t_mod, freqs,
            getattr(self, "_track4d_runtime_camera_features", None),
        )
    if mode == "vista4d_plucker":
        return _track4d_forward_block_with_vista4d_plucker(
            self, x, context, t_mod, freqs,
            getattr(self, "_track4d_runtime_camera_features", None),
        )
    if mode == "kq_only":
        return _track4d_forward_block_with_kq_only(
            self, x, context, t_mod, freqs,
            getattr(self, "_track4d_runtime_camera_features", None),
        )
    if mode == "track":
        return _track4d_forward_block_with_track(
            self, x, context, t_mod, freqs,
            getattr(self, "_track4d_runtime_camera_features", None),
            getattr(self, "_track4d_runtime_track_tokens"),
            getattr(self, "_track4d_runtime_track_injector"),
            int(getattr(self, "_track4d_runtime_block_idx")),
        )
    raise RuntimeError(f"Unknown Track4D DiTBlock runtime mode: {mode}")


def _track4d_patch_dit_block_forward() -> None:
    if getattr(DiTBlock, "_track4d_forward_is_patched", False):
        return
    DiTBlock.forward = _track4d_patched_dit_block_forward
    DiTBlock._track4d_forward_is_patched = True


def _track4d_call_block_with_runtime_state(
    block: DiTBlock,
    runtime_state: Optional[Dict[str, object]],
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    kv_cache = None if not runtime_state else runtime_state.get("_track4d_runtime_kv_cache")
    updating_cache = True if not runtime_state else bool(runtime_state.get("_track4d_runtime_updating_cache", True))
    frame_seq_length = None if not runtime_state else runtime_state.get("_track4d_runtime_frame_seq_length")
    cache_max_latent_frames = None if not runtime_state else runtime_state.get("_track4d_runtime_cache_max_latent_frames")
    cache_keep_first_latent = False if not runtime_state else bool(runtime_state.get("_track4d_runtime_cache_keep_first_latent", False))
    if not runtime_state:
        with track4d_record_function("track4d360/dit/block/base"):
            return block(
                x,
                context,
                t_mod,
                freqs,
                kv_cache=kv_cache,
                updating_cache=updating_cache,
                frame_seq_length=frame_seq_length,
                ar_cache_max_latent_frames=cache_max_latent_frames,
                ar_cache_keep_first_latent=cache_keep_first_latent,
            )

    # Under FSDP TRANSFORMER_BASED_WRAP at DiTBlock, each `dit.blocks[i]` is a
    # `FullyShardedDataParallel(DiTBlock)` instance. setattr on the wrapper goes
    # to wrapper.__dict__, but the patched forward runs with `self = inner` (the
    # unsharded DiTBlock), and getattr on inner doesn't see attrs set on the
    # wrapper. Set on the inner module directly when the block is FSDP-wrapped;
    # otherwise (DDP or unwrapped), `block` IS the DiTBlock and this is a no-op
    # rename.
    target = getattr(block, "_fsdp_wrapped_module", block)
    for name, value in runtime_state.items():
        setattr(target, name, value)
    try:
        block_idx = runtime_state.get("_track4d_runtime_block_idx", "camera")
        with track4d_record_function(f"track4d360/dit/block/{block_idx}"):
            return block(x, context, t_mod, freqs)
    finally:
        for name in _TRACK4D_RUNTIME_ATTRS:
            if hasattr(target, name):
                delattr(target, name)


_track4d_patch_dit_block_forward()


def _convert_dit_to_v2v(dit: Optional[torch.nn.Module]) -> Optional[torch.nn.Module]:
    if dit is None:
        return None
    if LongCatVideoTransformer3DModel is not None and isinstance(dit, (LongCatVideoTransformer3DModel,)):
        return dit
    if isinstance(dit, WanModelV2V):
        _ensure_domain_embedding(dit)
        return dit
    if not hasattr(dit, "blocks") or not hasattr(dit, "head") or not hasattr(dit, "patch_size"):
        _ensure_domain_embedding(dit)
        return dit

    ffn_dim = int(getattr(dit.blocks[0], "ffn_dim"))
    num_heads = int(getattr(dit.blocks[0].self_attn, "num_heads"))
    num_layers = int(len(dit.blocks))
    eps = float(getattr(dit.head.norm, "eps"))
    out_features = int(getattr(dit.head.head, "out_features"))
    patch_prod = int(math.prod(getattr(dit, "patch_size")))
    out_dim = out_features // patch_prod
    text_dim = int(getattr(dit.text_embedding[0], "in_features"))

    add_control_adapter = getattr(dit, "control_adapter", None) is not None
    in_dim_control_adapter = 24
    if add_control_adapter and hasattr(dit.control_adapter, "conv"):
        in_dim_control_adapter = int(getattr(dit.control_adapter.conv, "in_channels")) // 64

    cfg = dict(
        dim=int(getattr(dit, "dim")),
        in_dim=int(getattr(dit, "in_dim")),
        ffn_dim=ffn_dim,
        out_dim=out_dim,
        text_dim=text_dim,
        freq_dim=int(getattr(dit, "freq_dim")),
        eps=eps,
        patch_size=tuple(getattr(dit, "patch_size")),
        num_heads=num_heads,
        num_layers=num_layers,
        has_image_input=bool(getattr(dit, "has_image_input", False)),  # Default False for S2V models
        has_image_pos_emb=bool(getattr(dit, "has_image_pos_emb", False)),
        has_ref_conv=bool(getattr(dit, "has_ref_conv", False)),
        add_control_adapter=bool(add_control_adapter),
        in_dim_control_adapter=int(in_dim_control_adapter),
        seperated_timestep=bool(getattr(dit, "seperated_timestep", False)),
        require_vae_embedding=bool(getattr(dit, "require_vae_embedding", True)),
        require_clip_embedding=bool(getattr(dit, "require_clip_embedding", True)),
        fuse_vae_embedding_in_latents=bool(getattr(dit, "fuse_vae_embedding_in_latents", False)),
    )

    device = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype
    training = dit.training

    # Always create V2V model on CPU first to avoid OOM with large models (e.g., 14B)
    # accelerator.prepare() will handle device placement with proper partitioning (ZeRO-3)
    dit_v2v = WanModelV2V(**cfg)
    dit_v2v.load_state_dict(dit.state_dict(), strict=False)
    dit_v2v.train(training)

    # Only move to device if original model was already on a device (not CPU)
    # This allows ZeRO-3 initialization to work properly
    if device.type != 'cpu':
        # Destroy the old DiT's tensor storage in-place before materializing the V2V
        # variant on GPU. Without this we transiently hold 2x DiT (~56 GB for 14B) and
        # OOM on 95 GB GH200; naively moving the old DiT to CPU instead trades GPU OOM
        # for CPU OOM (4 ranks * 28 GB = 112 GB on top of ZeRO-1 optimizer offload).
        # Zeroing p.data releases both GPU AND CPU storage in one shot — the Module
        # object stays valid until the caller reassigns `pipe.dit = dit_v2v`.
        for p in dit.parameters():
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        for b in dit.buffers():
            b.data = torch.empty(0, dtype=b.dtype, device=b.device)
        torch.cuda.empty_cache()
        dit_v2v = dit_v2v.to(device=device, dtype=dtype)
    else:
        dit_v2v = dit_v2v.to(dtype=dtype)

    return dit_v2v


class WanVideoUnit_ModeDropoutAugmentation(PipelineUnit):
    """
    Applies T2V/I2V/V2V mode switching by controlling which video input is available.
    Following ReCamMaster official training (paper Section 4.2):

    Training modes (mutually exclusive):
    - T2V mode (drop_video_prob * 0.5): Text + camera control only
      → Remove video input entirely (input_video=None)
      → Keep text prompt (with CFG dropout applied separately)
      → Model generates from noise using text + camera guidance

    - I2V mode (drop_video_prob * 0.5): Image + text + camera control
      → Keep only first frame of video (input_video=[first_frame])
      → Keep text prompt (with CFG dropout applied separately)
      → Model generates from first frame using text + camera guidance

    - V2V mode (remaining probability): Video + text + camera control
      → Keep full video input
      → Keep text prompt (with CFG dropout applied separately)
      → Optional: Add noise to conditional video latent (0-500 timesteps) for synthetic data robustness

    Note: All modes keep the text prompt. CFG text dropout is handled separately in training script.
    """
    def __init__(self):
        super().__init__(
            # `drop_video_prob` is legacy "total prob split equally between T2V and I2V".
            # `drop_t2v_prob` / `drop_i2v_prob` allow explicit control (e.g. I2V-only).
            input_params=("input_video", "drop_video_prob", "drop_t2v_prob", "drop_i2v_prob"),
            output_params=("input_video", "mode_type"),
        )

    def process(self, pipe: "WanVideoPipeline", input_video, drop_video_prob, drop_t2v_prob=None, drop_i2v_prob=None):
        import random

        # Default parameters if not provided (for backward compatibility)
        drop_video_prob = drop_video_prob if drop_video_prob is not None else 0.0
        drop_t2v_prob = drop_t2v_prob if drop_t2v_prob is not None else None
        drop_i2v_prob = drop_i2v_prob if drop_i2v_prob is not None else None

        # If no input video is provided, treat it as explicit T2V mode.
        if input_video is None:
            return {"input_video": None, "mode_type": "t2v"}

        # Resolve explicit vs legacy probability configuration.
        if drop_t2v_prob is None and drop_i2v_prob is None:
            # Legacy: split total prob equally between T2V and I2V (ReCamMaster 0.4 -> 0.2/0.2/0.6).
            if drop_video_prob == 0:
                return {"mode_type": "v2v"}  # Default mode (keep full video)
            t2v_prob = float(drop_video_prob) * 0.5
            i2v_prob = float(drop_video_prob) * 0.5
        else:
            # Explicit: allow disabling T2V while keeping I2V (or vice versa).
            t2v_prob = float(drop_t2v_prob or 0.0)
            i2v_prob = float(drop_i2v_prob or 0.0)
            if t2v_prob < 0 or i2v_prob < 0:
                raise ValueError(f"drop_t2v_prob and drop_i2v_prob must be >= 0. Got {t2v_prob}, {i2v_prob}.")
            if t2v_prob + i2v_prob > 1.0 + 1e-8:
                raise ValueError(f"drop_t2v_prob + drop_i2v_prob must be <= 1. Got {t2v_prob + i2v_prob}.")
            if t2v_prob == 0.0 and i2v_prob == 0.0:
                return {"mode_type": "v2v"}

        # Mode selection (T2V / I2V / V2V)
        mode_rand = random.random()

        if mode_rand < t2v_prob:
            # T2V mode: Remove video input entirely (text + camera control only)
            return {"input_video": None, "mode_type": "t2v"}
        if mode_rand < t2v_prob + i2v_prob:
            # I2V mode: Keep only first frame (image + text + camera control)
            return {"input_video": [input_video[0]], "mode_type": "i2v"}
        else:
            # V2V mode: Keep full video + text + camera control
            # Note: Latent noise augmentation will be applied in a separate unit after encoding
            return {"mode_type": "v2v"}


class WanVideoUnit_LatentNoiseAugmentation(PipelineUnit):
    """
    Adds noise augmentation to conditional video latent for V2V mode.
    Following ReCamMaster paper style: applies moderate noise (default 0-500 timesteps) to reduce
    domain gap between synthetic and real data.

    Only applies in V2V mode when input_video_noise_strength != 0.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_latents", "mode_type", "input_video_noise_strength", "input_video_noise_random_max"),
            output_params=("input_latents",),
        )

    def process(self, pipe: "WanVideoPipeline", input_latents, mode_type, input_video_noise_strength, input_video_noise_random_max):
        import random
        import torch

        # Default parameters
        input_video_noise_strength = input_video_noise_strength if input_video_noise_strength is not None else 0.0
        input_video_noise_random_max = 500 if input_video_noise_random_max is None else int(input_video_noise_random_max)
        mode_type = mode_type if mode_type is not None else "v2v"

        # Only apply noise in V2V mode when input_latents exist and noise_strength > 0
        if input_latents is None or mode_type != "v2v" or input_video_noise_strength == 0:
            return {}

        # Determine noise strength (can be fixed or random in range [0, input_video_noise_random_max])
        if input_video_noise_strength < 0:
            # Negative value means random in [0, input_video_noise_random_max]
            noise_timestep = random.randint(0, max(0, input_video_noise_random_max))
        elif input_video_noise_strength >= 1.0:
            # Interpret as fixed timestep value (e.g., 250 out of 1000)
            noise_timestep = int(input_video_noise_strength)
        else:
            # Interpret as fraction [0, 1] and scale to [0, 1000]
            noise_timestep = int(input_video_noise_strength * int(getattr(pipe.scheduler, "num_train_timesteps", 1000)))

        num_train_timesteps = int(getattr(pipe.scheduler, "num_train_timesteps", 1000))
        noise_timestep = max(0, min(int(noise_timestep), num_train_timesteps))
        timestep_tensor = torch.tensor([float(noise_timestep)], dtype=torch.float32, device=pipe.device)

        # Add noise to input latents
        latent_noise = torch.randn_like(input_latents)
        input_latents = pipe.scheduler.add_noise(input_latents, latent_noise, timestep_tensor)

        return {"input_latents": input_latents}


class WanVideoUnit_3DModalityVideo(PipelineUnit):
    """
    Unified unit for encoding depth/disparity modalities.

    Handles two input types:
    1. disp_video: List of PIL images (disparity/depth maps for V2V, legacy) → depth_latents
       Note: disp_video is deprecated in favor of depth_normalized for better precision
    2. depth_normalized: Single-channel tensor (1, T, H, W) in [-1, 1] → depth_latents
       Preferred input type. Preserves float32 precision (avoids PIL uint8 quantization).
       Supports all normalization modes (log_depth, disparity, fixed_min_max).
    """
    def __init__(self):
        super().__init__(
            input_params=("depth_video", "tiled", "tile_size", "tile_stride"),
            output_params=("depth_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipelineBase, depth_video, tiled, tile_size, tile_stride):
        result = {}

        # Handle depth_normalized (single-channel tensor, needs replication to 3 channels)
        if depth_video is not None:
            pipe.load_models_to_device(self.onload_model_names)
            # depth_normalized is (1, T, H, W) - single channel depth map in [-1, 1]
            # VAE expects 3-channel input (B, C, T, H, W), so replicate to 3 channels
            if isinstance(depth_video, torch.Tensor):
                if depth_video.ndim == 4:  # (1, T, H, W)
                    # Replicate single-channel depth to 3 channels
                    depth_video_tensor = depth_video.unsqueeze(0)  # (1, 1, T, H, W)
                    depth_video_tensor = depth_video_tensor.repeat(1, 3, 1, 1, 1)  # (1, 3, T, H, W)
                elif depth_video.ndim == 5:  # (B, 1, T, H, W)
                    # Replicate to 3 channels
                    depth_video_tensor = depth_video.repeat(1, 3, 1, 1, 1)  # (B, 3, T, H, W)
                else:
                    raise ValueError(f"depth_normalized must be (1, T, H, W) or (B, 1, T, H, W), got {tuple(depth_video.shape)}")
            else:
                raise TypeError(f"depth_normalized must be a tensor, got {type(depth_video)}")

            depth_video_tensor = depth_video_tensor.to(dtype=pipe.torch_dtype, device=pipe.device)
            depth_latents = pipe.vae.encode(
                depth_video_tensor,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            result["depth_latents"] = depth_latents

        # Handle disp_video (PIL images, legacy - output as depth_latents for consistency)
        # if disp_video is not None:
        #     pipe.load_models_to_device(self.onload_model_names)
        #     disp_video_tensor = pipe.preprocess_video(disp_video)
        #     depth_latents = pipe.vae.encode(
        #         disp_video_tensor,
        #         device=pipe.device,
        #         tiled=tiled,
        #         tile_size=tile_size,
        #         tile_stride=tile_stride,
        #     ).to(dtype=pipe.torch_dtype, device=pipe.device)
        #     result["depth_latents"] = depth_latents

        return result


class WanVideoUnit_PointmapVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("pointmap", "tiled", "tile_size", "tile_stride"),
            output_params=("pointmap_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipelineBase, pointmap, tiled, tile_size, tile_stride):
        if pointmap is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        # Pointmap is already a tensor (3, T, H, W) with (x, y, z) coordinates
        # Convert to (B, C, T, H, W) format expected by VAE
        if isinstance(pointmap, torch.Tensor):
            if pointmap.ndim == 4:  # (C, T, H, W)
                pointmap_video = pointmap.unsqueeze(0)  # (1, C, T, H, W)
            elif pointmap.ndim == 5:  # (B, C, T, H, W)
                pointmap_video = pointmap
            else:
                raise ValueError(f"pointmap must be (C, T, H, W) or (B, C, T, H, W), got {tuple(pointmap.shape)}")
        else:
            raise TypeError(f"pointmap must be a tensor, got {type(pointmap)}")

        pointmap_video = pointmap_video.to(dtype=pipe.torch_dtype, device=pipe.device)
        pointmap_latents = pipe.vae.encode(
            pointmap_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pointmap_latents": pointmap_latents}


class WanVideoUnit_TrackVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "track",
                "track_valid_mask",
                "track_xyz_normalization",
                "tiled",
                "tile_size",
                "tile_stride",
                "track_xyz_quantile_low",
                "track_xyz_quantile_high",
                "predict_track_residual",
                "residual_zero_center_normalization",
                "residual_linear_normalization",
                "residual_log_scale_factor",
                "xyz_normalization_stats",
            ),
            output_params=("track_latents",),
            onload_model_names=("vae",),
        )

    @staticmethod
    def _coerce_track_valid_mask(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, _c, t, h, w = track_video.shape
        if track_valid_mask is None:
            return torch.ones((b, t, h, w), device=track_video.device, dtype=torch.bool)

        valid = track_valid_mask.to(device=track_video.device, dtype=torch.bool)
        if valid.ndim == 3:  # (T,H,W)
            valid = valid.unsqueeze(0)
        if valid.ndim == 4:  # (B,T,H,W) or (1,T,H,W)
            if valid.shape[0] == 1 and b > 1:
                valid = valid.expand(b, -1, -1, -1)
        elif valid.ndim == 5 and valid.shape[1] == 1:  # (B,1,T,H,W)
            valid = valid[:, 0]
        else:
            raise ValueError(
                f"track_valid_mask must be (T,H,W), (1,T,H,W), (B,T,H,W), or (B,1,T,H,W), got {tuple(valid.shape)}"
            )
        if valid.shape != (b, t, h, w):
            raise ValueError(f"track_valid_mask shape {tuple(valid.shape)} must match track spatial/temporal dims {(b, t, h, w)}")
        return valid

    @staticmethod
    def _normalize_track_sequencewise_minmax(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
        *,
        torch_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Sequence-wise normalization (match visualize_kubric_track_vae_roundtrip.py):
        - Compute per-channel min/max across (T,H,W) using valid pixels only
        - Map to [-1, 1], set invalid pixels to 0
        """
        b, c, t, h, w = track_video.shape
        valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)

        track_f32 = track_video.float()
        track_flat = track_f32.reshape(b, c, -1)  # (B,C,N)
        valid_flat = valid.reshape(b, -1)  # (B,N)

        pos_inf = torch.tensor(float("inf"), device=track_f32.device, dtype=track_f32.dtype)
        neg_inf = torch.tensor(float("-inf"), device=track_f32.device, dtype=track_f32.dtype)
        min_vals = torch.where(valid_flat.unsqueeze(1), track_flat, pos_inf).amin(dim=-1)  # (B,C)
        max_vals = torch.where(valid_flat.unsqueeze(1), track_flat, neg_inf).amax(dim=-1)  # (B,C)

        has_any_valid = valid_flat.any(dim=-1)  # (B,)
        if not bool(has_any_valid.all()):
            min_vals = torch.where(has_any_valid[:, None], min_vals, torch.zeros_like(min_vals))
            max_vals = torch.where(has_any_valid[:, None], max_vals, torch.ones_like(max_vals))

        eps = torch.tensor(1e-6, device=track_f32.device, dtype=track_f32.dtype)
        ranges = (max_vals - min_vals).clamp_min(eps)  # (B,C)
        norm_flat = (track_flat - min_vals.unsqueeze(-1)) / ranges.unsqueeze(-1) * 2.0 - 1.0
        norm_flat = torch.where(valid_flat.unsqueeze(1), norm_flat, torch.zeros_like(norm_flat))
        return norm_flat.reshape(b, c, t, h, w).clamp(-1.0, 1.0).to(dtype=torch_dtype)

    @staticmethod
    def _compute_track_sequencewise_minmax_stats(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-(B,C) min/max across (T,H,W) on valid pixels.

        Returns:
            (min_vals, max_vals): each (B,C) float tensors on track_video.device
        """
        b, c, _t, _h, _w = track_video.shape
        valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)

        track_f32 = track_video.float()
        track_flat = track_f32.reshape(b, c, -1)  # (B,C,N)
        valid_flat = valid.reshape(b, -1)  # (B,N)

        pos_inf = torch.tensor(float("inf"), device=track_f32.device, dtype=track_f32.dtype)
        neg_inf = torch.tensor(float("-inf"), device=track_f32.device, dtype=track_f32.dtype)
        min_vals = torch.where(valid_flat.unsqueeze(1), track_flat, pos_inf).amin(dim=-1)  # (B,C)
        max_vals = torch.where(valid_flat.unsqueeze(1), track_flat, neg_inf).amax(dim=-1)  # (B,C)

        has_any_valid = valid_flat.any(dim=-1)  # (B,)
        if not bool(has_any_valid.all()):
            min_vals = torch.where(has_any_valid[:, None], min_vals, torch.zeros_like(min_vals))
            max_vals = torch.where(has_any_valid[:, None], max_vals, torch.ones_like(max_vals))

        return min_vals, max_vals

    @staticmethod
    def _apply_track_sequencewise_minmax_stats(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
        *,
        min_vals: torch.Tensor,
        max_vals: torch.Tensor,
        torch_dtype: torch.dtype,
        zero_center: bool = False,
    ) -> torch.Tensor:
        """
        Apply precomputed min/max stats to normalize a track-like tensor to [-1,1].

        Default mode (zero_center=False):
            norm = (x - min) / (max - min) * 2 - 1

        Zero-center mode (zero_center=True):
            norm = x / (max(|max|, |min|))
            This ensures x=0 → norm=0.

        Invalid pixels (per `track_valid_mask`) are set to 0.
        """
        b, c, t, h, w = track_video.shape
        valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)

        track_f32 = track_video.float()
        track_flat = track_f32.reshape(b, c, -1)  # (B,C,N)
        valid_flat = valid.reshape(b, -1)  # (B,N)

        min_vals = min_vals.to(device=track_video.device, dtype=track_f32.dtype)
        max_vals = max_vals.to(device=track_video.device, dtype=track_f32.dtype)

        eps = torch.tensor(1e-6, device=track_f32.device, dtype=track_f32.dtype)

        if zero_center:
            # Zero-centered: scale by max absolute value
            scale = torch.maximum(torch.abs(max_vals), torch.abs(min_vals)).clamp_min(eps)  # (B,C)
            norm_flat = track_flat / scale.unsqueeze(-1)
            norm_flat = norm_flat.clamp(-1.0, 1.0)
        else:
            # Standard: translate and scale
            ranges = (max_vals - min_vals).clamp_min(eps)  # (B,C)
            norm_flat = (track_flat - min_vals.unsqueeze(-1)) / ranges.unsqueeze(-1) * 2.0 - 1.0
            norm_flat = norm_flat.clamp(-1.0, 1.0)
        norm_flat = torch.where(valid_flat.unsqueeze(1), norm_flat, torch.zeros_like(norm_flat))
        return norm_flat.reshape(b, c, t, h, w).to(dtype=torch_dtype)

    @staticmethod
    def _normalize_track_sequencewise_signed_log_p2_p98(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
        *,
        torch_dtype: torch.dtype,
        q_low: float = 0.02,
        q_high: float = 0.98,
        half_range: float = 0.75,  # For PPD-style [-1.0, 0.5] range
    ) -> torch.Tensor:
        """
        PPD-style signed-log1p + percentile normalization for XYZ:
        1. y = sign(x) * log1p(|x|)
        2. Normalize using [q_low, q_high] quantile range (values can exceed [0,1])
        3. Clip to [-0.5, 1.0]
        4. Shift by -0.5 to get [-1.0, 0.5]

        Note: Quantiles define normalization range, NOT clipping range.
              Values outside quantiles → normalized to <0 or >1 → then clipped.

        Reference: pixel-perfect-depth/ppd/models/ppd_train.py:102
        """
        b, c, t, h, w = track_video.shape
        valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)

        track_f32 = track_video.float()
        y = torch.sign(track_f32) * torch.log1p(torch.abs(track_f32))  # (B,C,T,H,W)
        y_flat = y.reshape(b, c, -1)  # (B,C,N)
        valid_flat = valid.reshape(b, -1)  # (B,N)

        ql = torch.zeros((b, c), device=y.device, dtype=y.dtype)
        qh = torch.ones((b, c), device=y.device, dtype=y.dtype)
        for bi in range(b):
            for ci in range(c):
                vals = y_flat[bi, ci][valid_flat[bi]]
                if vals.numel() == 0:
                    ql[bi, ci] = 0.0
                    qh[bi, ci] = 1.0
                else:
                    ql[bi, ci] = torch.quantile(vals, q_low)
                    qh[bi, ci] = torch.quantile(vals, q_high)

        eps = torch.tensor(1e-6, device=y.device, dtype=y.dtype)
        denom = (qh - ql).clamp_min(eps)  # (B,C)

        # Normalize using quantile range (can exceed [0,1])
        y_normalized = (y_flat - ql.unsqueeze(-1)) / denom.unsqueeze(-1)

        # PPD-style: clip to [-0.5, 1.0], then shift to [-1.0, 0.5]
        y_clipped = y_normalized.clamp(-0.5, 1.0)
        y_final = y_clipped - 0.5  # [-1.0, 0.5]

        y_final = torch.where(valid_flat.unsqueeze(1), y_final, torch.zeros_like(y_final))
        return y_final.reshape(b, c, t, h, w).to(dtype=torch_dtype)

    @staticmethod
    def _normalize_track_linear_fixed(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
        *,
        torch_dtype: torch.dtype,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Fixed symmetric linear normalization for residual targets:
            x_norm = clamp(x * K, [-1, 1])
        This preserves x=0 -> 0 exactly (no log transform, no per-sequence stats).
        """
        valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)
        k = float(scale_factor)
        if not (k > 0.0):
            k = 1.0
        x_norm = (track_video.float() * k).clamp(-1.0, 1.0)
        x_norm = torch.where(valid.unsqueeze(1), x_norm, torch.zeros_like(x_norm))
        return x_norm.to(dtype=torch_dtype)

    @staticmethod
    def _compute_track_sequencewise_signed_log_stats(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
        *,
        q_low: float = 0.02,
        q_high: float = 0.98,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-(B,C) signed-log percentile stats on valid pixels:
            y = sign(x) * log1p(|x|), then q_low/q_high percentiles.
        """
        b, c, _t, _h, _w = track_video.shape
        valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)

        track_f32 = track_video.float()
        y = torch.sign(track_f32) * torch.log1p(torch.abs(track_f32))  # (B,C,T,H,W)
        y_flat = y.reshape(b, c, -1)  # (B,C,N)
        valid_flat = valid.reshape(b, -1)  # (B,N)

        ql = torch.zeros((b, c), device=y.device, dtype=y.dtype)
        qh = torch.ones((b, c), device=y.device, dtype=y.dtype)
        for bi in range(b):
            for ci in range(c):
                vals = y_flat[bi, ci][valid_flat[bi]]
                if vals.numel() == 0:
                    ql[bi, ci] = 0.0
                    qh[bi, ci] = 1.0
                else:
                    ql[bi, ci] = torch.quantile(vals, q_low)
                    qh[bi, ci] = torch.quantile(vals, q_high)
        return ql, qh

    @staticmethod
    def _apply_track_sequencewise_signed_log_stats(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
        *,
        ql: torch.Tensor,
        qh: torch.Tensor,
        torch_dtype: torch.dtype,
        half_range: float = 0.75,  # For PPD-style [-1.0, 0.5] range
        zero_center: bool = False,
        post_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply precomputed signed-log percentile stats to normalize a track-like tensor.

        Default mode (zero_center=False):
            PPD-style: normalize → clip to [-0.5, 1.0] → shift to [-1.0, 0.5]

        Zero-center mode (zero_center=True):
            Scale only (no translation): y_norm = y / scale where scale = (qh - ql) / 2
            This ensures x=0 → y=0 → y_norm=0, making static backgrounds always map to the same value.
            Clip to symmetric range [-1, 1].

        Invalid pixels (per `track_valid_mask`) are set to 0.
        """
        b, c, t, h, w = track_video.shape
        valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)

        track_f32 = track_video.float()
        y = torch.sign(track_f32) * torch.log1p(torch.abs(track_f32))  # (B,C,T,H,W)
        y_flat = y.reshape(b, c, -1)  # (B,C,N)
        valid_flat = valid.reshape(b, -1)  # (B,N)

        ql = ql.to(device=track_video.device, dtype=y.dtype)
        qh = qh.to(device=track_video.device, dtype=y.dtype)

        eps = torch.tensor(1e-6, device=y.device, dtype=y.dtype)
        denom = (qh - ql).clamp_min(eps)  # (B,C)

        if zero_center:
            # Zero-centered normalization: scale only, no translation
            # scale = (qh - ql) / 2, so y_norm = y / scale = y * 2 / (qh - ql)
            scale = denom / 2.0  # (B,C)
            y_normalized = y_flat / scale.unsqueeze(-1)  # (B,C,N)
            if float(post_scale) != 1.0:
                y_normalized = y_normalized * float(post_scale)
            # Clip to symmetric range [-1, 1]
            y_final = y_normalized.clamp(-1.0, 1.0)
        else:
            # Standard normalization: translate and scale
            y_normalized = (y_flat - ql.unsqueeze(-1)) / denom.unsqueeze(-1)
            # PPD-style: clip to [-0.5, 1.0], then shift to [-1.0, 0.5]
            y_clipped = y_normalized.clamp(-0.5, 1.0)
            y_final = y_clipped - 0.5  # [-1.0, 0.5]

        y_final = torch.where(valid_flat.unsqueeze(1), y_final, torch.zeros_like(y_final))
        return y_final.reshape(b, c, t, h, w).to(dtype=torch_dtype)

    @staticmethod
    def _normalize_track_sequencewise_with_reference(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
        *,
        reference_track_video: Optional[torch.Tensor],
        reference_track_valid_mask: Optional[torch.Tensor],
        track_xyz_normalization: str,
        torch_dtype: torch.dtype,
        q_low: float = 0.02,
        q_high: float = 0.98,
        zero_center: bool = False,
        post_scale: float = 1.0,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Normalize `track_video` using stats computed from `reference_track_video` (same scene).

        If reference is None, falls back to normalizing using `track_video` itself.

        Args:
            zero_center: If True, normalize such that x=0 maps to normalized=0 (no translation).
                        This ensures residual=0 (static background) always appears as the same color.
            return_stats: If True, also return the computed normalization stats.

        Returns:
            If return_stats=False: normalized tensor
            If return_stats=True: (normalized tensor, stats dict)
        """
        ref_video = reference_track_video if reference_track_video is not None else track_video
        ref_mask = reference_track_valid_mask if reference_track_video is not None else track_valid_mask

        # st()
        norm = str(track_xyz_normalization or "minmax")
        if norm == "minmax":
            min_vals, max_vals = WanVideoUnit_TrackVideo._compute_track_sequencewise_minmax_stats(ref_video, ref_mask)
            normalized = WanVideoUnit_TrackVideo._apply_track_sequencewise_minmax_stats(
                track_video,
                track_valid_mask,
                min_vals=min_vals,
                max_vals=max_vals,
                torch_dtype=torch_dtype,
                zero_center=zero_center,
            )
            if return_stats:
                return normalized, {"norm_type": "minmax", "min_vals": min_vals, "max_vals": max_vals}
            return normalized
        if norm in ("signed_log_p2_p98", "signed_log_p0_p100"):
            ql_arg = 0.0 if norm == "signed_log_p0_p100" else float(q_low)
            qh_arg = 1.0 if norm == "signed_log_p0_p100" else float(q_high)

            valid_track_bthw = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)
            valid_ref_bthw = WanVideoUnit_TrackVideo._coerce_track_valid_mask(ref_video, ref_mask)

            b, c, t, h, w = track_video.shape
            if c != 3:
                raise ValueError(f"Expected C=3 for signed_log track normalization, got shape {tuple(track_video.shape)}")

            ql_bc = torch.zeros((b, c), device=track_video.device, dtype=torch.float32)
            qh_bc = torch.ones((b, c), device=track_video.device, dtype=torch.float32)
            norm_out = torch.zeros((b, c, t, h, w), device=track_video.device, dtype=torch.float32)

            for bi in range(b):
                if bool(zero_center):
                    ql_bc[bi] = torch.tensor([-1.0, -1.0, -1.0], device=track_video.device, dtype=torch.float32)
                    qh_bc[bi] = torch.tensor([1.0, 1.0, 1.0], device=track_video.device, dtype=torch.float32)
                else:
                    ref_thwc = ref_video[bi].float().permute(1, 2, 3, 0).contiguous()  # (T,H,W,3)
                    ref_valid_thw = valid_ref_bthw[bi]
                    ql_b, qh_b = compute_signed_log_quantiles(ref_thwc, ref_valid_thw, q_low=ql_arg, q_high=qh_arg)
                    ql_bc[bi] = ql_b.to(device=track_video.device, dtype=torch.float32)
                    qh_bc[bi] = qh_b.to(device=track_video.device, dtype=torch.float32)

                track_thwc = track_video[bi].float().permute(1, 2, 3, 0).contiguous()  # (T,H,W,3)
                track_valid_thw = valid_track_bthw[bi]
                norm_thwc, _params = normalize_track_signed_log(
                    track_thwc,
                    valid_mask=track_valid_thw,
                    ql=ql_bc[bi],
                    qh=qh_bc[bi],
                    q_low=ql_arg,
                    q_high=qh_arg,
                    zero_center=bool(zero_center),
                    post_scale=float(post_scale),
                )
                norm_out[bi] = norm_thwc.permute(3, 0, 1, 2).contiguous()

            normalized = norm_out.to(dtype=torch_dtype)
            if return_stats:
                return normalized, {"norm_type": norm, "ql": ql_bc.to(dtype=torch_dtype), "qh": qh_bc.to(dtype=torch_dtype)}
            return normalized
        raise ValueError(f"Unknown track_xyz_normalization={norm!r}")

    @staticmethod
    def _normalize_track_sequencewise(
        track_video: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor],
        *,
        track_xyz_normalization: str,
        torch_dtype: torch.dtype,
        q_low: float = 0.02,
        q_high: float = 0.98,
    ) -> torch.Tensor:
        norm = str(track_xyz_normalization or "minmax")
        if norm == "minmax":
            return WanVideoUnit_TrackVideo._normalize_track_sequencewise_minmax(track_video, track_valid_mask, torch_dtype=torch_dtype)
        if norm in ("signed_log_p2_p98", "signed_log_p0_p100"):
            return WanVideoUnit_TrackVideo._normalize_track_sequencewise_with_reference(
                track_video,
                track_valid_mask,
                reference_track_video=None,
                reference_track_valid_mask=None,
                track_xyz_normalization=norm,
                torch_dtype=torch_dtype,
                q_low=q_low,
                q_high=q_high,
            )
        raise ValueError(f"Unknown track_xyz_normalization={norm!r}")

    def process(
        self,
        pipe: WanVideoPipelineBase,
        track,
        track_valid_mask,
        track_xyz_normalization,
        tiled,
        tile_size,
        tile_stride,
        track_xyz_quantile_low=0.02,
        track_xyz_quantile_high=0.98,
        predict_track_residual=False,
        residual_zero_center_normalization=False,
        residual_linear_normalization=False,
        residual_log_scale_factor: float = 1.0,
        xyz_normalization_stats=None,
    ):
        if track is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        # Track is expected to be a tensor (3, T, H, W) with (x, y, z) coordinates (or batched).
        # Convert to (B, C, T, H, W) format expected by VAE.
        if isinstance(track, torch.Tensor):
            if track.ndim == 4:  # (C, T, H, W)
                track_video = track.unsqueeze(0)  # (1, C, T, H, W)
            elif track.ndim == 5:  # (B, C, T, H, W)
                track_video = track
            else:
                raise ValueError(f"track must be (C, T, H, W) or (B, C, T, H, W), got {tuple(track.shape)}")
        else:
            raise TypeError(f"track must be a tensor, got {type(track)}")

        track_video = track_video.to(device=pipe.device)

        # Residual target ablations:
        # - signed-log zero-center (legacy/default when enabled)
        # - fixed linear zero-preserving scaling (new ablation, stats-free)
        use_linear_residual = bool(predict_track_residual) and bool(residual_linear_normalization)
        zero_center = bool(predict_track_residual) and bool(residual_zero_center_normalization) and (not use_linear_residual)
        post_scale = float(residual_log_scale_factor) if zero_center else 1.0

        if use_linear_residual:
            # Fixed linear residual normalization ignores per-sequence xyz stats by design.
            track_video = self._normalize_track_linear_fixed(
                track_video,
                track_valid_mask,
                torch_dtype=pipe.torch_dtype,
                scale_factor=float(residual_log_scale_factor),
            )
        # If xyz_normalization_stats provided (from xyz_video_cond), use those stats
        # This ensures xyz_video_cond and track are normalized with the same stats
        elif xyz_normalization_stats is not None:
            norm_type = xyz_normalization_stats.get("norm_type")
            if norm_type == "minmax":
                track_video = self._apply_track_sequencewise_minmax_stats(
                    track_video,
                    track_valid_mask,
                    min_vals=xyz_normalization_stats["min_vals"],
                    max_vals=xyz_normalization_stats["max_vals"],
                    torch_dtype=pipe.torch_dtype,
                    zero_center=zero_center,
                )
            elif norm_type in ("signed_log_p2_p98", "signed_log_p0_p100"):
                ql_stats = xyz_normalization_stats["ql"]
                qh_stats = xyz_normalization_stats["qh"]
                if not isinstance(ql_stats, torch.Tensor) or not isinstance(qh_stats, torch.Tensor):
                    raise TypeError("xyz_normalization_stats['ql'/'qh'] must be torch.Tensors for signed_log modes")

                ql_arg = 0.0 if str(norm_type) == "signed_log_p0_p100" else float(track_xyz_quantile_low)
                qh_arg = 1.0 if str(norm_type) == "signed_log_p0_p100" else float(track_xyz_quantile_high)

                valid_track_bthw = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_valid_mask)
                b, c, t, h, w = track_video.shape
                if c != 3:
                    raise ValueError(f"Expected C=3 for signed_log track normalization, got shape {tuple(track_video.shape)}")

                out = torch.zeros((b, c, t, h, w), device=track_video.device, dtype=torch.float32)
                for bi in range(b):
                    track_thwc = track_video[bi].float().permute(1, 2, 3, 0).contiguous()  # (T,H,W,3)
                    track_valid_thw = valid_track_bthw[bi]
                    ql_b = ql_stats[bi].to(device=track_video.device, dtype=torch.float32)
                    qh_b = qh_stats[bi].to(device=track_video.device, dtype=torch.float32)
                    norm_thwc, _params = normalize_track_signed_log(
                        track_thwc,
                        valid_mask=track_valid_thw,
                        ql=ql_b,
                        qh=qh_b,
                        q_low=ql_arg,
                        q_high=qh_arg,
                        zero_center=bool(zero_center),
                        post_scale=float(post_scale),
                    )
                    out[bi] = norm_thwc.permute(3, 0, 1, 2).contiguous()
                track_video = out.to(dtype=pipe.torch_dtype)
            else:
                raise ValueError(f"Unknown xyz_normalization_stats norm_type={norm_type!r}")
        else:
            # Legacy: self-normalize using track's own stats
            track_video = self._normalize_track_sequencewise_with_reference(
                track_video,
                track_valid_mask,
                reference_track_video=track_video,  # Self-normalize
                reference_track_valid_mask=track_valid_mask,
                track_xyz_normalization=track_xyz_normalization,
                torch_dtype=pipe.torch_dtype,
                q_low=track_xyz_quantile_low,
                q_high=track_xyz_quantile_high,
                zero_center=zero_center,
                post_scale=post_scale,
            )

        track_latents = pipe.vae.encode(
            track_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"track_latents": track_latents}


class WanVideoUnit_TrackVideoSecondary(WanVideoUnit_TrackVideo):
    """
    Encode an optional secondary tracking target (e.g., track-video) for joint V2V diffusion.

    This shares the same normalization pipeline as `WanVideoUnit_TrackVideo`, but writes
    to `track_video_latents`.
    """
    def __init__(self):
        PipelineUnit.__init__(
            self,
            input_params=(
                "track_video",
                "track_video_valid_mask",
                "track_xyz_normalization",
                "tiled",
                "tile_size",
                "tile_stride",
                "track_xyz_quantile_low",
                "track_xyz_quantile_high",
                "track_video_prenormalized",
                "track_video_prenormalized_signed_log",
                "depth_video_raw",
            ),
            output_params=("track_video_latents",),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipelineBase,
        track_video,
        track_video_valid_mask,
        track_xyz_normalization,
        tiled,
        tile_size,
        tile_stride,
        track_xyz_quantile_low=0.02,
        track_xyz_quantile_high=0.98,
        track_video_prenormalized=False,
        track_video_prenormalized_signed_log=False,
        depth_video_raw=False,
    ):
        if track_video is None:
            return {}

        # Depth video normalization: log-space + per-sequence percentile → [-1, 1].
        # Input: raw depth / common_scale replicated to 3ch (all channels identical).
        # This runs on-the-fly in the pipeline, NOT in the shared data adapter.
        if depth_video_raw:
            pipe.load_models_to_device(self.onload_model_names)
            if isinstance(track_video, torch.Tensor):
                if track_video.ndim == 4:
                    track_video = track_video.unsqueeze(0)  # (1,C,T,H,W)
            depth = track_video[:, 0:1].float()  # (1,1,T,H,W) — single channel
            valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_video_valid_mask)
            # Log-space: log1p(depth), depth is already / common_scale, positive
            depth_log = torch.log1p(depth.clamp(min=0))
            # Per-sequence percentile normalization (2%/98% over full video)
            valid_vals = depth_log[valid.unsqueeze(1)]  # flatten valid
            if valid_vals.numel() > 0:
                d_p02 = torch.quantile(valid_vals, 0.02)
                d_p98 = torch.quantile(valid_vals, 0.98)
            else:
                d_p02, d_p98 = depth_log.min(), depth_log.max()
            d_range = max(float(d_p98 - d_p02), 1e-6)
            # PPD-style: (x - p02)/(p98 - p02) → clamp[-0.5, 1.0] → -0.5 → [-1.0, 0.5]
            depth_norm = (depth_log - d_p02) / d_range
            depth_norm = depth_norm.clamp(-0.5, 1.0) - 0.5
            depth_norm = torch.where(valid.unsqueeze(1), depth_norm, torch.zeros_like(depth_norm))
            # Replicate to 3 channels for VAE
            track_video = depth_norm.expand(-1, 3, -1, -1, -1)  # (1,3,T,H,W)
            track_video = track_video.to(dtype=pipe.torch_dtype, device=pipe.device)
            track_latents = pipe.vae.encode(
                track_video, device=pipe.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"track_video_latents": track_latents}

        if track_video_prenormalized:
            # Data is already quantile-normalized to [-1,1] by preprocessing.
            # Optionally apply zero-preserving signed-log compression, then encode.
            pipe.load_models_to_device(self.onload_model_names)
            if isinstance(track_video, torch.Tensor):
                if track_video.ndim == 4:  # (C, T, H, W)
                    track_video = track_video.unsqueeze(0)  # (1, C, T, H, W)
            track_video = track_video.to(device=pipe.device)
            if track_video_prenormalized_signed_log:
                valid = WanVideoUnit_TrackVideo._coerce_track_valid_mask(track_video, track_video_valid_mask)
                track_video = torch.sign(track_video.float()) * torch.log1p(torch.abs(track_video.float()))
                track_video = torch.where(valid.unsqueeze(1), track_video, -torch.ones_like(track_video))
            track_video = track_video.to(dtype=pipe.torch_dtype, device=pipe.device)
            track_latents = pipe.vae.encode(
                track_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"track_video_latents": track_latents}

        out = super().process(
            pipe=pipe,
            track=track_video,
            track_valid_mask=track_video_valid_mask,
            track_xyz_normalization=track_xyz_normalization,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            track_xyz_quantile_low=track_xyz_quantile_low,
            track_xyz_quantile_high=track_xyz_quantile_high,
            predict_track_residual=False,
            residual_zero_center_normalization=False,
            residual_linear_normalization=False,
            residual_log_scale_factor=1.0,
            xyz_normalization_stats=None,
        )
        return {"track_video_latents": out["track_latents"]}


class WanVideoUnit_InputVideoEmbedder_XYZCondition(WanVideoUnit_TrackVideo):
    """
    Encode RGB conditioning video, with an optional per-frame XYZ "video" added pixel-wise in RGB space.

    This is used for VDPM-style tasks where we want: (RGB + XYZ_from_depth) -> track,
    without introducing extra tokens/channels later (just an additive conditioning).
    """

    def __init__(self):
        PipelineUnit.__init__(
            self,
            input_params=(
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
                # Optional RGB+XYZ conditioning
                "use_xyz_video_condition",
                "xyz_video_cond",
                "xyz_video_valid_mask",
                "xyz_video_condition_scale",
                # Optional query-frame impulse conditioning (VDM tracking)
                "use_query_frame_impulse_condition",
                "track_query_idx",
                # Reuse the same normalization choice as track
                "track_xyz_normalization",
                "track_xyz_quantile_low",
                "track_xyz_quantile_high",
                # Optional reference stats (use GT track when available)
                "track",
                "track_valid_mask",
                # Training augmentation for xyz_video_cond
                "xyz_video_condition_aug_prob",
                "xyz_video_condition_noise_std",
                "xyz_video_condition_z_noise_std",
                "xyz_video_condition_frame_scale_std",
                "xyz_video_condition_dropout_prob",
                # If true, xyz augmentation (noise/dropout) has already been applied by the dataset adapter.
                # This avoids double-augmenting (which would break residual-target consistency).
                "xyz_video_cond_augmented_in_dataset",
                # Residual prediction mode
                "predict_track_residual",
                "residual_zero_center_normalization",
                # If True, normalize track using xyz_video_cond stats (instead of self-normalizing).
                # This ensures xyz_video_cond and track are in the same normalized space,
                # and allows inference without GT track (since stats come from xyz_video_cond).
                "normalize_track_with_xyz_cond_stats",
                # Conditioning mode: "xyz" (full 3D), "depth" (Z-channel + log norm), "none" (skip)
                "xyz_video_condition_mode",
                # CFG-style full-signal dropout probability (zero entire conditioning per sample)
                "xyz_video_condition_cfg_dropout_prob",
            ),
            output_params=("latents", "input_latents", "control_xyz_latents_input", "xyz_normalization_stats"),
            onload_model_names=("vae",),
        )

    @staticmethod
    def _as_bool(v) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                raise ValueError(f"Expected a scalar tensor for bool flag, got shape {tuple(v.shape)}")
            return bool(v.detach().cpu().item())
        return bool(v)

    @staticmethod
    def _as_float(v, default: float) -> float:
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                raise ValueError(f"Expected a scalar tensor for float value, got shape {tuple(v.shape)}")
            return float(v.detach().cpu().item())
        return float(v)

    @staticmethod
    def _coerce_xyz_video(xyz_video: torch.Tensor, *, batch: int) -> torch.Tensor:
        if xyz_video.ndim == 4:  # (C,T,H,W)
            xyz_bcthw = xyz_video.unsqueeze(0)
        elif xyz_video.ndim == 5:  # (B,C,T,H,W)
            xyz_bcthw = xyz_video
        else:
            raise ValueError(f"xyz_video_cond must be (C,T,H,W) or (B,C,T,H,W), got {tuple(xyz_video.shape)}")
        if int(xyz_bcthw.shape[1]) != 3:
            raise ValueError(f"xyz_video_cond must have C=3, got shape {tuple(xyz_bcthw.shape)}")
        if int(xyz_bcthw.shape[0]) == 1 and int(batch) > 1:
            xyz_bcthw = xyz_bcthw.expand(int(batch), -1, -1, -1, -1)
        if int(xyz_bcthw.shape[0]) != int(batch):
            raise ValueError(f"xyz_video_cond batch must match input_video batch={batch}, got {int(xyz_bcthw.shape[0])}")
        return xyz_bcthw

    @staticmethod
    def _apply_xyz_augmentation(
        xyz_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        noise_std: float,
        z_noise_std: float,
        frame_scale_std: float,
        dropout_prob: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Apply training augmentation to xyz_video_cond to bridge train/inference domain gap.

        Args:
            xyz_norm: (B,3,T,H,W) normalized XYZ pointmap
            valid_mask: (B,1,T,H,W) or (1,T,H,W) valid mask
            noise_std: Gaussian noise std for all XYZ channels (in normalized units)
            z_noise_std: Extra Gaussian noise std for Z channel only
            frame_scale_std: Per-frame log-normal scale jitter std (simulates depth flicker)
            dropout_prob: Randomly zero out valid pixels with this probability
            device: torch device

        Returns:
            xyz_aug: (B,3,T,H,W) augmented XYZ pointmap
        """
        xyz_aug = xyz_norm.clone()
        b, c, t, h, w = xyz_aug.shape

        # 1. Additive Gaussian noise (all channels)
        if noise_std > 0:
            noise = torch.randn_like(xyz_aug, device=device) * noise_std
            xyz_aug = xyz_aug + noise

        # 2. Extra Z-channel (depth) noise
        if z_noise_std > 0:
            z_noise = torch.randn(b, 1, t, h, w, device=device, dtype=xyz_aug.dtype) * z_noise_std
            xyz_aug[:, 2:3, :, :, :] = xyz_aug[:, 2:3, :, :, :] + z_noise

        # 3. Per-frame scale jitter (simulate depth scale inconsistency across frames)
        if frame_scale_std > 0:
            # Log-normal: scale = exp(N(0, std))
            log_scale = torch.randn(b, 1, t, 1, 1, device=device, dtype=xyz_aug.dtype) * frame_scale_std
            frame_scale = torch.exp(log_scale)  # (B,1,T,1,1)
            xyz_aug = xyz_aug * frame_scale

        # 4. Dropout: randomly zero out valid pixels
        if dropout_prob > 0:
            # Handle different valid_mask shapes
            if valid_mask.ndim == 4:  # (1,T,H,W)
                valid_mask_b = valid_mask.expand(b, -1, -1, -1).unsqueeze(1)  # (B,1,T,H,W)
            elif valid_mask.ndim == 5 and valid_mask.shape[1] == 1:  # (B,1,T,H,W)
                valid_mask_b = valid_mask
            else:
                raise ValueError(f"valid_mask must be (1,T,H,W) or (B,1,T,H,W), got {tuple(valid_mask.shape)}")

            dropout_mask = torch.rand(b, 1, t, h, w, device=device, dtype=xyz_aug.dtype) > dropout_prob
            # Only drop pixels that were originally valid
            keep_mask = (~valid_mask_b.bool()) | dropout_mask
            xyz_aug = torch.where(keep_mask, xyz_aug, torch.zeros_like(xyz_aug))

        return xyz_aug

    @staticmethod
    def _apply_depth_augmentation(
        depth_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        noise_std: float,
        z_noise_std: float,
        frame_scale_std: float,
        dropout_prob: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Apply train-time augmentation to a depth-only conditioning signal.

        Unlike XYZ augmentation, depth-only conditioning is semantically 1-channel and is
        later replicated to 3 channels only to satisfy the VAE input shape. Keep the three
        replicated channels identical by augmenting the single depth channel before repeat.
        """
        depth_aug = depth_norm.clone()
        b, c, t, h, w = depth_aug.shape
        if c != 1:
            raise ValueError(f"depth_norm must have C=1, got shape {tuple(depth_aug.shape)}")

        total_noise_std = float(noise_std) + float(z_noise_std)
        if total_noise_std > 0:
            noise = torch.randn_like(depth_aug, device=device) * total_noise_std
            depth_aug = depth_aug + noise

        if frame_scale_std > 0:
            log_scale = torch.randn(b, 1, t, 1, 1, device=device, dtype=depth_aug.dtype) * frame_scale_std
            depth_aug = depth_aug * torch.exp(log_scale)

        if valid_mask.ndim == 4:
            valid_mask_b = valid_mask.expand(b, -1, -1, -1).unsqueeze(1)
        elif valid_mask.ndim == 5 and valid_mask.shape[1] == 1:
            valid_mask_b = valid_mask
        else:
            raise ValueError(f"valid_mask must be (1,T,H,W) or (B,1,T,H,W), got {tuple(valid_mask.shape)}")

        if dropout_prob > 0:
            dropout_mask = torch.rand(b, 1, t, h, w, device=device, dtype=depth_aug.dtype) > dropout_prob
            keep_mask = (~valid_mask_b.bool()) | dropout_mask
            depth_aug = torch.where(keep_mask, depth_aug, torch.zeros_like(depth_aug))

        return depth_aug * valid_mask_b.to(dtype=depth_aug.dtype, device=device)

    def process(
        self,
        pipe: WanVideoPipelineBase,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
        use_xyz_video_condition,
        xyz_video_cond,
        xyz_video_valid_mask,
        xyz_video_condition_scale,
        use_query_frame_impulse_condition,
        track_query_idx,
        track_xyz_normalization,
        track_xyz_quantile_low,
        track_xyz_quantile_high,
        track,
        track_valid_mask,
        xyz_video_condition_aug_prob,
        xyz_video_condition_noise_std,
        xyz_video_condition_z_noise_std,
        xyz_video_condition_frame_scale_std,
        xyz_video_condition_dropout_prob,
        xyz_video_cond_augmented_in_dataset,
        predict_track_residual,
        residual_zero_center_normalization,
        normalize_track_with_xyz_cond_stats=False,
        xyz_video_condition_mode=None,
        xyz_video_condition_cfg_dropout_prob=None,
    ):
        if input_video is None:
            return {"latents": noise}

        pipe.load_models_to_device(self.onload_model_names)

        # Preprocess RGB conditioning video in [-1,1]
        cond_video = pipe.preprocess_video(input_video, torch_dtype=torch.float32, device=pipe.device)
        b, c, t, h, w = cond_video.shape
        if c != 3:
            raise ValueError(f"input_video must have 3 channels after preprocessing, got shape {tuple(cond_video.shape)}")

        cond_video = cond_video.to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = pipe.vae.encode(
            cond_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        control_xyz_latents_input = None
        xyz_normalization_stats = None  # Will store (ql, qh) if normalize_track_with_xyz_cond_stats is True
        cond_mode = str(xyz_video_condition_mode or "xyz")
        if self._as_bool(use_xyz_video_condition):
            if xyz_video_cond is None:
                raise ValueError("use_xyz_video_condition=True but xyz_video_cond is None")
            if not isinstance(xyz_video_cond, torch.Tensor):
                raise TypeError(f"xyz_video_cond must be a torch.Tensor, got {type(xyz_video_cond)}")

            # CFG-style full-signal dropout: zero entire conditioning with some probability
            cfg_drop_prob = self._as_float(xyz_video_condition_cfg_dropout_prob, 0.0)
            cfg_dropped = pipe.scheduler.training and cfg_drop_prob > 0.0 and torch.rand(1).item() < cfg_drop_prob

            # mode="none": skip geometric conditioning entirely (adapter still provides xyz for residual target)
            if cond_mode == "none" or cfg_dropped:
                pass  # control_xyz_latents_input stays None
            else:
                xyz_bcthw = self._coerce_xyz_video(xyz_video_cond.to(device=pipe.device), batch=b).float()
                if xyz_bcthw.shape[2:] != (t, h, w):
                    raise ValueError(
                        f"xyz_video_cond must match input_video (T,H,W)=({t},{h},{w}), got {tuple(xyz_bcthw.shape[2:])}"
                    )

                if cond_mode == "depth":
                    # --- Depth-only conditioning: Z-channel with log normalization ---
                    # Better dynamic range than full XYZ pointmap for conditioning.
                    depth_z = xyz_bcthw[:, 2:3]  # (B, 1, T, H, W) — depth from cam0-aligned pointmap

                    # Build valid mask (reuse xyz mask + exclude behind-camera)
                    if xyz_video_valid_mask is not None:
                        vm = xyz_video_valid_mask
                        if vm.ndim == 4:  # (1,T,H,W)
                            vm = vm.unsqueeze(0).expand(b, -1, -1, -1, -1)
                        elif vm.ndim == 5 and vm.shape[1] != 1:
                            vm = vm[:, :1]
                        valid_5d = vm.float().to(device=pipe.device)
                    else:
                        valid_5d = torch.ones(b, 1, t, h, w, device=pipe.device)
                    valid_5d = valid_5d * (depth_z > 0).float()

                    # Log normalization: log1p(z) for z >= 0
                    depth_log = torch.log1p(depth_z.clamp(min=0))

                    # PPD-style p2/p98 normalization over full video sequence → [-1.0, 0.5]
                    # (depth - p02) / (p98 - p02) → clamp[-0.5, 1.0] → subtract 0.5
                    # Shallow outliers (below p02) reach -1.0; deep outliers cap at 0.5.
                    for i in range(b):
                        valid_pixels = depth_log[i][valid_5d[i].bool()]
                        if valid_pixels.numel() > 0:
                            d_p02 = torch.quantile(valid_pixels, 0.02)
                            d_p98 = torch.quantile(valid_pixels, 0.98)
                            d_range = d_p98 - d_p02
                            if d_range > 1e-6:
                                depth_log[i] = (depth_log[i] - d_p02) / d_range
                                depth_log[i] = depth_log[i].clamp(-0.5, 1.0) - 0.5
                            else:
                                depth_log[i] = torch.zeros_like(depth_log[i])
                        else:
                            depth_log[i] = torch.zeros_like(depth_log[i])

                    depth_log = depth_log * valid_5d  # zero-out invalid pixels; already in [-1.0, 0.5]

                    # Apply augmentation (noise/dropout still useful for train/inference gap)
                    if pipe.scheduler.training:
                        if not self._as_bool(xyz_video_cond_augmented_in_dataset):
                            aug_prob = self._as_float(xyz_video_condition_aug_prob, 0.0)
                            if aug_prob > 0.0 and torch.rand(1).item() < aug_prob:
                                depth_log = self._apply_depth_augmentation(
                                    depth_log,
                                    valid_5d,
                                    noise_std=self._as_float(xyz_video_condition_noise_std, 0.0),
                                    z_noise_std=self._as_float(xyz_video_condition_z_noise_std, 0.0),
                                    frame_scale_std=self._as_float(xyz_video_condition_frame_scale_std, 0.0),
                                    dropout_prob=self._as_float(xyz_video_condition_dropout_prob, 0.0),
                                    device=pipe.device,
                                )

                    # Replicate to 3 channels for VAE after single-channel depth augmentation.
                    xyz_norm = depth_log.expand(-1, 3, -1, -1, -1).contiguous()

                else:
                    # --- Default: full XYZ pointmap conditioning ---
                    # Determine reference for normalization:
                    # - If normalize_track_with_xyz_cond_stats=True: use xyz_video_cond for stats (self-normalize)
                    # - Otherwise (legacy): use GT track for stats (if available)
                    ref_track_video = None
                    ref_track_mask = None
                    if self._as_bool(normalize_track_with_xyz_cond_stats):
                        # NEW: Compute stats from xyz_video_cond itself, store for track normalization
                        ref_track_video = None  # Will self-normalize
                        ref_track_mask = None
                    elif isinstance(track, torch.Tensor):
                        if track.ndim == 4:  # (C,T,H,W)
                            ref_track_video = track.unsqueeze(0)
                        elif track.ndim == 5:  # (B,C,T,H,W)
                            ref_track_video = track
                        else:
                            raise ValueError(f"track must be (C,T,H,W) or (B,C,T,H,W), got {tuple(track.shape)}")
                        if int(ref_track_video.shape[1]) != 3:
                            raise ValueError(f"track must have C=3, got shape {tuple(ref_track_video.shape)}")
                        if int(ref_track_video.shape[0]) == 1 and int(b) > 1:
                            ref_track_video = ref_track_video.expand(int(b), -1, -1, -1, -1)
                        if int(ref_track_video.shape[0]) != int(b):
                            raise ValueError(f"track batch must match input_video batch={b}, got {int(ref_track_video.shape[0])}")
                        if tuple(ref_track_video.shape[2:]) != (t, h, w):
                            raise ValueError(
                                f"track must match input_video (T,H,W)=({t},{h},{w}), got {tuple(ref_track_video.shape[2:])}"
                            )
                        ref_track_video = ref_track_video.to(device=pipe.device)
                        ref_track_mask = track_valid_mask

                    # Zero-center normalization for residual prediction: ensures residual=0 maps to norm=0
                    zero_center = self._as_bool(predict_track_residual) and self._as_bool(residual_zero_center_normalization)

                    # If normalize_track_with_xyz_cond_stats=True, return stats so WanVideoUnit_TrackVideo
                    # can normalize the track target/prediction with the same signed-log stats.
                    should_return_stats = self._as_bool(normalize_track_with_xyz_cond_stats)

                    # NOTE: xyz_video_cond does NOT need residual post-scale: it is conditioning, not the VAE target.
                    result = WanVideoUnit_TrackVideo._normalize_track_sequencewise_with_reference(
                        xyz_bcthw,
                        xyz_video_valid_mask,
                        reference_track_video=ref_track_video,
                        reference_track_valid_mask=ref_track_mask,
                        track_xyz_normalization=str(track_xyz_normalization or "minmax"),
                        torch_dtype=torch.float32,
                        q_low=track_xyz_quantile_low,
                        q_high=track_xyz_quantile_high,
                        zero_center=zero_center,
                        return_stats=should_return_stats,
                    )

                    if should_return_stats:
                        xyz_norm, xyz_normalization_stats = result
                    else:
                        xyz_norm = result

                    # Apply training augmentation to bridge train/inference domain gap
                    # (GT xyz at train time vs Stream3R/DA3 xyz at inference time)
                    if pipe.scheduler.training:
                        # If the dataset adapter already handled xyz augmentation (and residual target construction),
                        # do not apply another, independent augmentation here.
                        if self._as_bool(xyz_video_cond_augmented_in_dataset):
                            aug_prob = 0.0
                        else:
                            aug_prob = self._as_float(xyz_video_condition_aug_prob, 0.0)
                        if aug_prob > 0.0 and torch.rand(1).item() < aug_prob:
                            xyz_norm = self._apply_xyz_augmentation(
                                xyz_norm,
                                xyz_video_valid_mask,
                                noise_std=self._as_float(xyz_video_condition_noise_std, 0.0),
                                z_noise_std=self._as_float(xyz_video_condition_z_noise_std, 0.0),
                                frame_scale_std=self._as_float(xyz_video_condition_frame_scale_std, 0.0),
                                dropout_prob=self._as_float(xyz_video_condition_dropout_prob, 0.0),
                                device=pipe.device,
                            )

                # VAE-encode and scale (shared by xyz and depth modes)
                scale = self._as_float(xyz_video_condition_scale, 1.0)
                xyz_norm = xyz_norm.to(dtype=pipe.torch_dtype, device=pipe.device)
                control_xyz_latents_input = pipe.vae.encode(
                    xyz_norm,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                control_xyz_latents_input = control_xyz_latents_input * float(scale)

        # Optional query-frame impulse conditioning (binary 0/1 video with an impulse at the query frame).
        # We encode it with the same VAE to produce control latents at the same spatiotemporal resolution.
        if self._as_bool(use_query_frame_impulse_condition):
            # Determine query index per sample (default: 0).
            if track_query_idx is None:
                q_idxs = [0] * int(b)
            elif isinstance(track_query_idx, torch.Tensor):
                if track_query_idx.numel() == 1:
                    q = int(track_query_idx.detach().cpu().item())
                    q_idxs = [q] * int(b)
                else:
                    flat = track_query_idx.detach().cpu().view(-1).tolist()
                    if len(flat) != int(b):
                        raise ValueError(
                            f"track_query_idx must be scalar or have batch length {int(b)}, got {tuple(track_query_idx.shape)}"
                        )
                    q_idxs = [int(v) for v in flat]
            elif isinstance(track_query_idx, (list, tuple)):
                if len(track_query_idx) != int(b):
                    raise ValueError(f"track_query_idx must have batch length {int(b)}, got len={len(track_query_idx)}")
                q_idxs = [int(v) for v in track_query_idx]
            else:
                q = int(track_query_idx)
                q_idxs = [q] * int(b)

            impulse_b1thw = torch.zeros((int(b), 1, int(t), int(h), int(w)), device=pipe.device, dtype=torch.float32)
            for bi, q in enumerate(q_idxs):
                q = int(q)
                if q < 0:
                    continue
                q = max(0, min(q, int(t) - 1))
                impulse_b1thw[bi, 0, q, :, :] = 1.0

            # Wan VAE expects 3-channel inputs; repeat impulse to 3 channels.
            impulse_b3thw = impulse_b1thw.repeat(1, 3, 1, 1, 1).to(dtype=pipe.torch_dtype, device=pipe.device)
            impulse_latents = pipe.vae.encode(
                impulse_b3thw,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

            if control_xyz_latents_input is None:
                control_xyz_latents_input = impulse_latents
            else:
                # Concatenate XYZ control latents and impulse control latents along channels.
                control_xyz_latents_input = torch.cat([control_xyz_latents_input, impulse_latents], dim=1)

        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image, torch_dtype=torch.float32, device=pipe.device)
            vace_reference_image = vace_reference_image.to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)

        if pipe.scheduler.training:
            return {
                "latents": noise,
                "input_latents": input_latents,
                "control_xyz_latents_input": control_xyz_latents_input,
                "xyz_normalization_stats": xyz_normalization_stats,
            }

        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {
            "latents": latents,
            "input_latents": input_latents,
            "control_xyz_latents_input": control_xyz_latents_input,
            "xyz_normalization_stats": xyz_normalization_stats,
        }


class WanVideoUnit_V2VConcatenate(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_latents",
                "target_rgb_latents",
                "track_latents",
                "track_video_latents",
                "pointmap_latents",
                "depth_latents",
                "v2v_concat_mode",
                "multimodality_nvs",
                "multimodality_tracking",
                "multimodality_track_depth",
                "mode_type",
                "channel_concat",
                "raw_t2v_target_only",
            ),
            output_params=("input_latents", "cond_latents", "v2v_concat_mode"),
        )

    @staticmethod
    def _make_i2v_latents(cond_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
        if target_latents.shape[2] <= 1:
            return cond_latents[:, :, :1]
        noise_rest = torch.randn(
            (target_latents.shape[0], target_latents.shape[1], target_latents.shape[2] - 1, target_latents.shape[3], target_latents.shape[4]),
            device=target_latents.device,
            dtype=target_latents.dtype,
        )
        return torch.cat([cond_latents[:, :, :1], noise_rest], dim=2)

    def process(
        self,
        pipe: WanVideoPipelineBase,
        input_latents,
        target_rgb_latents,
        track_latents,
        track_video_latents,
        pointmap_latents,
        depth_latents,
        v2v_concat_mode=None,
        multimodality_nvs=False,
        multimodality_tracking=False,
        multimodality_track_depth=False,
        mode_type=None,
        channel_concat=False,
        raw_t2v_target_only=False,
    ):
        # Channel-concat mode assembles latents in WanVideoUnit_ChannelConcatV2V; skip this unit.
        if channel_concat:
            return {}
        mode_type = "v2v" if mode_type is None else str(mode_type)
        active_multi_modes = int(bool(multimodality_nvs)) + int(bool(multimodality_tracking)) + int(bool(multimodality_track_depth))
        if active_multi_modes > 1:
            raise ValueError("Only one of multimodality_nvs, multimodality_tracking, or multimodality_track_depth can be enabled at a time.")

        # Multi-modality NVS: concatenate [source_rgb | target_rgb | target_depth]
        if multimodality_nvs:
            if target_rgb_latents is None:
                raise ValueError("multimodality_nvs=True requires target_rgb_latents (encoded target RGB).")
            if depth_latents is None:
                raise ValueError("multimodality_nvs=True requires depth_latents (encoded target depth).")

            if mode_type == "t2v":
                input_latents = torch.randn_like(target_rgb_latents)
            elif mode_type == "i2v":
                if input_latents is None:
                    input_latents = torch.randn_like(target_rgb_latents)
                else:
                    input_latents = self._make_i2v_latents(input_latents, target_rgb_latents)
            elif input_latents is None:
                # In multi-modality mode we always model 3 domains; if source conditioning is absent,
                # the caller should use mode_type='t2v' (handled above) to synthesize an unconditioned sample.
                raise ValueError("multimodality_nvs=True requires input_latents unless mode_type is 't2v' or 'i2v'.")

            mode = "width" if v2v_concat_mode is None else v2v_concat_mode
            assert target_rgb_latents is not None
            if mode not in ("frame", "width"):
                raise ValueError(f"Multi-modality NVS only supports frame/width concat mode, got: {mode}")

            # Validate shapes match (except concat dimension).
            if (
                input_latents.shape[:2] != target_rgb_latents.shape[:2]
                or input_latents.shape[:2] != depth_latents.shape[:2]
                or input_latents.shape[3:] != target_rgb_latents.shape[3:]
                or input_latents.shape[3:] != depth_latents.shape[3:]
            ):
                raise ValueError(
                    "Multi-modality: latent shapes must match (B,C,T,H,W) across source/target_rgb/target_depth. "
                    f"Got input_latents={tuple(input_latents.shape)}, target_rgb_latents={tuple(target_rgb_latents.shape)}, depth_latents={tuple(depth_latents.shape)}."
                )

            if mode == "frame":
                if (
                    input_latents.shape[2] != target_rgb_latents.shape[2]
                    or input_latents.shape[2] != depth_latents.shape[2]
                ):
                    raise ValueError(
                        "Frame concat requires equal T across domains. "
                        f"Got input_latents.T={input_latents.shape[2]}, target_rgb_latents.T={target_rgb_latents.shape[2]}, depth_latents.T={depth_latents.shape[2]}."
                    )
                out = torch.cat([input_latents, target_rgb_latents, depth_latents], dim=2)
            else:  # width
                if (
                    input_latents.shape[2] != target_rgb_latents.shape[2]
                    or input_latents.shape[2] != depth_latents.shape[2]
                ):
                    raise ValueError(
                        "Width concat requires equal T across domains. "
                        f"Got input_latents.T={input_latents.shape[2]}, target_rgb_latents.T={target_rgb_latents.shape[2]}, depth_latents.T={depth_latents.shape[2]}."
                    )
                out = torch.cat([input_latents, target_rgb_latents, depth_latents], dim=4)

            return {"input_latents": out, "v2v_concat_mode": mode}

        # Multi-modality tracking: concatenate [source_rgb | target_track | target_track_video]
        if multimodality_tracking:
            if track_latents is None:
                raise ValueError("multimodality_tracking=True requires track_latents (encoded residual/track target).")
            if track_video_latents is None:
                raise ValueError("multimodality_tracking=True requires track_video_latents (encoded track-video target).")

            if mode_type == "t2v":
                input_latents = torch.randn_like(track_latents)
            elif mode_type == "i2v":
                if input_latents is None:
                    input_latents = torch.randn_like(track_latents)
                else:
                    input_latents = self._make_i2v_latents(input_latents, track_latents)
            elif input_latents is None:
                raise ValueError("multimodality_tracking=True requires input_latents unless mode_type is 't2v' or 'i2v'.")

            mode = "width" if v2v_concat_mode is None else v2v_concat_mode
            # NOTE: joint tracking multimodality is intentionally width-only for now.
            if mode == "frame":
                raise NotImplementedError(
                    "multimodality_tracking with frame-wise concat is not implemented. "
                    "Use v2v_concat_mode='width'."
                )
            if mode != "width":
                raise ValueError(f"Multi-modality tracking only supports width concat mode, got: {mode}")

            if (
                input_latents.shape[:2] != track_latents.shape[:2]
                or input_latents.shape[:2] != track_video_latents.shape[:2]
                or input_latents.shape[3:] != track_latents.shape[3:]
                or input_latents.shape[3:] != track_video_latents.shape[3:]
            ):
                raise ValueError(
                    "Multi-modality tracking: latent shapes must match (B,C,T,H,W) across source/track/track_video. "
                    f"Got input_latents={tuple(input_latents.shape)}, track_latents={tuple(track_latents.shape)}, track_video_latents={tuple(track_video_latents.shape)}."
                )

            if (
                input_latents.shape[2] != track_latents.shape[2]
                or input_latents.shape[2] != track_video_latents.shape[2]
            ):
                raise ValueError(
                    "Width concat requires equal T across domains. "
                    f"Got input_latents.T={input_latents.shape[2]}, track_latents.T={track_latents.shape[2]}, track_video_latents.T={track_video_latents.shape[2]}."
                )
            out = torch.cat([input_latents, track_latents, track_video_latents], dim=4)

            return {"input_latents": out, "v2v_concat_mode": mode}

        # Multi-modality tracking+depth: concatenate
        # [source_rgb | target_rgb | target_track | target_depth]
        if multimodality_track_depth:
            if track_latents is None:
                raise ValueError("multimodality_track_depth=True requires track_latents (encoded target RGB remap).")
            if track_video_latents is None:
                raise ValueError("multimodality_track_depth=True requires track_video_latents (encoded target pointmap).")
            if depth_latents is None:
                raise ValueError("multimodality_track_depth=True requires depth_latents (encoded target depth).")

            if mode_type == "t2v":
                input_latents = torch.randn_like(track_latents)
            elif mode_type == "i2v":
                if input_latents is None:
                    input_latents = torch.randn_like(track_latents)
                else:
                    input_latents = self._make_i2v_latents(input_latents, track_latents)
            elif input_latents is None:
                raise ValueError("multimodality_track_depth=True requires input_latents unless mode_type is 't2v' or 'i2v'.")

            mode = "width" if v2v_concat_mode is None else v2v_concat_mode
            if mode != "width":
                raise ValueError(f"multimodality_track_depth only supports width concat, got: {mode}")

            if (
                input_latents.shape[:2] != track_latents.shape[:2]
                or input_latents.shape[:2] != track_video_latents.shape[:2]
                or input_latents.shape[:2] != depth_latents.shape[:2]
                or input_latents.shape[3:] != track_latents.shape[3:]
                or input_latents.shape[3:] != track_video_latents.shape[3:]
                or input_latents.shape[3:] != depth_latents.shape[3:]
            ):
                raise ValueError(
                    "Multi-modality tracking+depth: latent shapes must match (B,C,T,H,W) across source/target_rgb/target_track/target_depth. "
                    f"Got input_latents={tuple(input_latents.shape)}, track_latents={tuple(track_latents.shape)}, "
                    f"track_video_latents={tuple(track_video_latents.shape)}, depth_latents={tuple(depth_latents.shape)}."
                )

            if (
                input_latents.shape[2] != track_latents.shape[2]
                or input_latents.shape[2] != track_video_latents.shape[2]
                or input_latents.shape[2] != depth_latents.shape[2]
            ):
                raise ValueError(
                    "Width concat requires equal T across domains. "
                    f"Got input_latents.T={input_latents.shape[2]}, track_latents.T={track_latents.shape[2]}, "
                    f"track_video_latents.T={track_video_latents.shape[2]}, depth_latents.T={depth_latents.shape[2]}."
                )

            out = torch.cat([input_latents, track_latents, track_video_latents, depth_latents], dim=4)
            return {"input_latents": out, "v2v_concat_mode": mode}

        # Standard V2V: single target modality (priority: depth > pointmap > target_rgb for NVS)
        target_latents = None
        modality_name = None

        if depth_latents is not None:
            target_latents = depth_latents
            modality_name = "depth"
        elif track_latents is not None:
            target_latents = track_latents
            modality_name = "track"
        elif pointmap_latents is not None:
            target_latents = pointmap_latents
            modality_name = "pointmap"
        elif target_rgb_latents is not None:
            # NVS training case: concatenate [source_rgb | target_rgb]
            target_latents = target_rgb_latents
            modality_name = "rgb"

        if target_latents is None:
            if v2v_concat_mode is not None:
                raise ValueError("`v2v_concat_mode` is set but no target modality (pointmap_latents, depth_latents, or target_rgb_latents) is provided.")
            return {}

        if bool(raw_t2v_target_only) and mode_type == "t2v":
            return {"input_latents": target_latents, "v2v_concat_mode": None}

        if mode_type == "t2v":
            input_latents = torch.randn_like(target_latents)
        elif mode_type == "i2v":
            if input_latents is None:
                input_latents = torch.randn_like(target_latents)
            else:
                input_latents = self._make_i2v_latents(input_latents, target_latents)
        elif input_latents is None:
            # Backward-compatible behavior: no source conditioning present.
            return {"input_latents": target_latents, "v2v_concat_mode": v2v_concat_mode}

        mode = "frame" if v2v_concat_mode is None else v2v_concat_mode
        if mode not in ("frame", "width", "channel"):
            raise ValueError(f"Invalid v2v_concat_mode: {mode}. Expected 'frame', 'width', or 'channel'.")

        if mode == "channel":
            if input_latents.shape != target_latents.shape:
                raise ValueError(
                    f"Channel concat requires identical latent shapes. "
                    f"Got input_latents={tuple(input_latents.shape)}, {modality_name}_latents={tuple(target_latents.shape)}."
                )
            return {"cond_latents": input_latents, "input_latents": target_latents, "v2v_concat_mode": mode}

        if input_latents.shape[:2] != target_latents.shape[:2] or input_latents.shape[3:] != target_latents.shape[3:]:
            raise ValueError(
                f"RGB/{modality_name} latent shapes must match (except concat dim). "
                f"Got input_latents={tuple(input_latents.shape)}, {modality_name}_latents={tuple(target_latents.shape)}."
            )

        if mode == "frame":
            if input_latents.shape[2] != target_latents.shape[2]:
                raise ValueError(
                    f"Frame concat requires equal T. Got input_latents.T={input_latents.shape[2]} vs {modality_name}_latents.T={target_latents.shape[2]}."
                )
            out = torch.cat([input_latents, target_latents], dim=2)
        else:  # width mode
            if input_latents.shape[2] != target_latents.shape[2]:
                raise ValueError(
                    f"Width concat requires equal T. Got input_latents.T={input_latents.shape[2]} vs {modality_name}_latents.T={target_latents.shape[2]}."
                )
            out = torch.cat([input_latents, target_latents], dim=4)

        return {"input_latents": out, "v2v_concat_mode": mode}


class WanVideoUnit_CameraControl_NVS(PipelineUnit):
    """
    Camera control for novel view synthesis (NVS) training.

    Processes source and target camera poses to generate relative Plücker ray embeddings
    that condition the video generation model on camera transformations.
    """
    def __init__(self):
        super().__init__(
            input_params=("nvs_video", "source_camera", "target_camera", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride", "camera_injection_mode", "camera_pose_mode", "camera_pose_frame_stride", "use_ray_origin"),
            output_params=("control_camera_latents_input", "control_camera_pose_input", "control_camera_plucker_input"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipelineBase,
        tiled, tile_size, tile_stride,
        source_camera: Optional[Dict] = None,
        target_camera: Optional[Dict] = None,
        nvs_video: Optional[torch.Tensor] = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 49,
        camera_injection_mode: str = "projector",
        camera_pose_mode: str = "packed",
        camera_pose_frame_stride: int = 4,
        use_ray_origin: bool = False,
    ):
        """
        Process camera pair to generate control latents.

        Note: The dataset now always returns c2w_opencv as (T, 4, 4) for both static and
        per-frame cameras, ensuring unified format throughout the pipeline.

        Args:
            pipe: Pipeline instance
            source_camera: Source camera dict with 'c2w_opencv' (T, 4, 4) and 'intrinsics' (3, 3)
            target_camera: Target camera dict with 'c2w_opencv' (T, 4, 4) and 'intrinsics' (3, 3)
            height: Video frame height
            width: Video frame width
            num_frames: Number of frames

        Returns:
            Dict with 'control_camera_latents_input' key
        """
        # Skip if no cameras provided
        # st()
        if source_camera is None or target_camera is None:
            return {}

        # Check if model has camera control capability
        # For "projector" and "add" modes: requires control_adapter (SimpleAdapter)
        # For "direct_raymap_add" mode: uses per-layer cam_encoder (no global control_adapter)
        # Either way, we still compute the Plücker rays and return them
        # Note: control_adapter is NOT used in this function - only in cfg_guided_model_fn
        # So we don't need to assert its existence here

        out: Dict = {
            "control_camera_latents_input": None,
            "control_camera_pose_input": None,
            "control_camera_plucker_input": None,
        }

        if camera_injection_mode == "rt_projector":
            # ReCamMaster-style RT conditioning (no Plücker rays):
            # relative pose convention (match ReCamMaster train_recammaster.py):
            #   rel = inv(source_c2w_0) @ target_c2w_t
            # i.e. treat source camera (t=0) as the reference "world" frame.
            src = torch.as_tensor(source_camera["c2w_opencv"], dtype=torch.float32, device=pipe.device)  # (T,4,4)
            tgt = torch.as_tensor(target_camera["c2w_opencv"], dtype=torch.float32, device=pipe.device)  # (T,4,4)
            if int(src.shape[0]) != int(num_frames) or int(tgt.shape[0]) != int(num_frames):
                raise ValueError(
                    f"Camera frames must match num_frames={int(num_frames)}; got src={int(src.shape[0])}, tgt={int(tgt.shape[0])}"
                )
            src0_inv = torch.linalg.inv(src[0])  # (4,4)
            rel = src0_inv.unsqueeze(0) @ tgt  # (T,4,4)
            rt = rel[:, :3, :4].reshape(int(num_frames), 12)  # (T,12)
            if str(camera_pose_mode).lower() == "raw":
                stride = max(1, int(camera_pose_frame_stride))
                expected_f = int((int(num_frames) - 1) // 4 + 1)
                indices = list(range(0, int(num_frames), stride))
                if len(indices) != expected_f:
                    indices = torch.linspace(0, int(num_frames) - 1, expected_f).round().long().tolist()
                rt_latent = rt[indices]  # (F,12)
                out["control_camera_pose_input"] = prepare_camera_control_pose_latents_raw(
                    rt_latent,
                    dtype=pipe.torch_dtype,
                ).to(device=pipe.device, dtype=pipe.torch_dtype)
            else:
                out["control_camera_pose_input"] = prepare_camera_control_pose_latents(
                    rt,
                    num_frames=num_frames,
                    dtype=pipe.torch_dtype,
                ).to(device=pipe.device, dtype=pipe.torch_dtype)
        elif camera_injection_mode == "vista4d_plucker":
            height_dit = int(height) // int(pipe.height_division_factor)
            width_dit = int(width) // int(pipe.width_division_factor)
            time_stride = int(getattr(pipe, "time_division_factor", 4))
            out["control_camera_plucker_input"] = process_camera_pair_to_plucker_dit_resolution(
                source_camera,
                target_camera,
                height,
                width,
                num_frames,
                height_dit,
                width_dit,
                time_stride=time_stride,
                device=str(pipe.device),
                use_ray_origin=use_ray_origin,
            ).to(device=pipe.device, dtype=pipe.torch_dtype)
        else:
            # Plücker raymap conditioning (projector/add/direct_raymap_add)
            plucker_embedding = process_camera_pair_to_plucker(
                source_camera,
                target_camera,
                height,
                width,
                num_frames,
                device=str(pipe.device),
                use_ray_origin=use_ray_origin,
            )  # (T, H, W, 6)

            out["control_camera_latents_input"] = prepare_camera_control_latents(
                plucker_embedding,
                num_frames,
                dtype=pipe.torch_dtype,
            ).to(device=pipe.device, dtype=pipe.torch_dtype)

        # During training: encode nvs_video (target RGB) to get target_rgb_latents
        # During inference: nvs_video is None (we're generating it), so skip this
        if nvs_video is not None:
            pipe.load_models_to_device(self.onload_model_names)

            # Handle different input formats for nvs_video
            if isinstance(nvs_video, torch.Tensor):
                # Accept either already-encoded VAE latents (C == z_dim) or RGB video (C == 3).
                z_dim = None
                try:
                    z_dim = int(getattr(getattr(pipe, "vae", None), "model", None).z_dim)  # type: ignore[attr-defined]
                except Exception:
                    z_dim = None

                c_in = int(nvs_video.shape[-4])
                is_latents = (z_dim is not None and c_in == z_dim)
                if is_latents:  # Already latents: (B, z_dim, T, H, W) or (z_dim, T, H, W)
                    if nvs_video.ndim == 4:  # (z_dim, T, H, W) -> (1, z_dim, T, H, W)
                        target_rgb_latents = nvs_video.unsqueeze(0)
                    else:  # Already (B, z_dim, T, H, W)
                        target_rgb_latents = nvs_video
                    # Ensure correct dtype for latents
                    target_rgb_latents = target_rgb_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
                else:  # RGB video: (C, T, H, W) with C=3
                    if c_in != 3:
                        raise ValueError(
                            f"nvs_video tensor must have C=3 (RGB) or C={z_dim} (VAE latents); got shape {tuple(nvs_video.shape)}."
                        )
                    if nvs_video.ndim == 4:
                        input_video = nvs_video.unsqueeze(0)  # Add batch dimension
                    else:
                        input_video = nvs_video
                    # Ensure correct dtype for VAE input
                    input_video = input_video.to(dtype=pipe.torch_dtype, device=pipe.device)
                    target_rgb_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            else:
                # List of PIL images: use standard preprocessing
                input_video = pipe.preprocess_video(nvs_video)
                # preprocess_video should return correct dtype, but ensure it
                input_video = input_video.to(dtype=pipe.torch_dtype, device=pipe.device)
                target_rgb_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            out.update({'target_rgb_latents': target_rgb_latents})

        return out


def model_fn_wan_video_v2v(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input: Optional[torch.Tensor] = None,
    control_camera_pose_input: Optional[torch.Tensor] = None,
    control_camera_plucker_input: Optional[torch.Tensor] = None,
    vista4d_source_video_latents: Optional[torch.Tensor] = None,
    vista4d_pointcloud_video_latents: Optional[torch.Tensor] = None,
    vista4d_source_mask_latents: Optional[torch.Tensor] = None,
    vista4d_pointcloud_mask_latents: Optional[torch.Tensor] = None,
    gen3c_warp_condition_latents: Optional[torch.Tensor] = None,
    fuse_vae_embedding_in_latents: bool = False,
    # Width-concat TI2V boundary marking. 0 keeps the historical behavior
    # (frame 0 marked t=0 across the FULL concatenated width). Set to the domain
    # count to restrict the t=0 marking to the boundary domain's frame-0 tokens.
    # See the block guarded by `ti2v_boundary_domains` below.
    ti2v_boundary_domains: int = 0,
    ti2v_boundary_target_domain: int = 1,
    v2v_concat_mode: Optional[Literal["frame", "width", "channel"]] = None,
    cond_latents: Optional[torch.Tensor] = None,
    reference_latents: Optional[torch.Tensor] = None,
    audio_embeds: Optional[torch.Tensor] = None,
    multimodality_nvs: bool = False,
    multimodality_tracking: bool = False,
    multimodality_track_depth: bool = False,
    debug_onehot_camera: bool = False,
    trajectory_pair_index: Optional[torch.Tensor] = None,
    feedback_input: Optional[torch.Tensor] = None,
    kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    current_start: int = 0,
    cache_start: int = 0,
    updating_cache: bool = True,
    ar_num_frame_per_block: int = 3,
    ar_cache_max_latent_frames: Optional[int] = None,
    ar_cache_keep_first_latent: bool = False,
    causal_teacher_clean_latents: Optional[torch.Tensor] = None,
    causal_teacher_num_frame_per_block: int = 1,
    # High-level training parameters (passed by loss function but not directly used by DiT)
    **kwargs,
    # input_video=None,
    # disp_video=None,
    # source_camera: Optional[Dict] = None,
    # target_camera: Optional[Dict] = None,
    # height: Optional[int] = None,
    # width: Optional[int] = None,
    # num_frames: Optional[int] = None,
    # cfg_scale: Optional[float] = None,
    # tiled: Optional[bool] = None,
    # rand_device: Optional[str] = None,
    # vace_scale: Optional[float] = None,
    # max_timestep_boundary: Optional[float] = None,
    # min_timestep_boundary: Optional[float] = None,
):
    # NOTE: `v2v_concat_mode=None` is allowed for standard T2V/I2V usage (no V2V concat).
    if v2v_concat_mode is not None and v2v_concat_mode not in ("frame", "width", "channel"):
        raise ValueError(f"Invalid v2v_concat_mode: {v2v_concat_mode}. Expected 'frame', 'width', 'channel', or None.")
    ar_mode = kv_cache is not None
    causal_teacher_mode = causal_teacher_clean_latents is not None
    if ar_mode:
        if use_unified_sequence_parallel:
            raise NotImplementedError("Track4D-360 AR mode does not support unified sequence parallel.")
        if tea_cache is not None:
            raise NotImplementedError("Track4D-360 AR mode does not support TeaCache.")
        if sliding_window_size is not None or sliding_window_stride is not None:
            raise NotImplementedError("Track4D-360 AR mode does not support temporal sliding-window tiling.")
    if causal_teacher_mode:
        if use_unified_sequence_parallel:
            raise NotImplementedError("causal_teacher_forcing does not support sequence parallel.")
        if tea_cache is not None:
            raise NotImplementedError("causal_teacher_forcing does not support TeaCache.")
        if sliding_window_size is not None or sliding_window_stride is not None:
            raise NotImplementedError("causal_teacher_forcing does not support temporal tiling.")
        if kwargs.get("use_domain_embedding", True):
            raise NotImplementedError("causal_teacher_forcing first cut requires --disable_domain_embedding.")
        unsupported = [
            name for name, value in {
                "control_camera_latents_input": control_camera_latents_input,
                "control_camera_pose_input": control_camera_pose_input,
                "control_camera_plucker_input": control_camera_plucker_input,
                "control_latents_input": cond_latents,
                "reference_latents": reference_latents,
                "audio_embeds": audio_embeds,
            }.items()
            if value is not None
        ]
        if unsupported:
            raise NotImplementedError(
                "causal_teacher_forcing first cut is raw/no-control only; "
                f"got unsupported inputs {unsupported}."
            )
    active_multi_modes = int(bool(multimodality_nvs)) + int(bool(multimodality_tracking)) + int(bool(multimodality_track_depth))
    if active_multi_modes > 1:
        raise ValueError("Only one of multimodality_nvs, multimodality_tracking, or multimodality_track_depth can be enabled.")
    multi_num_domains = 4 if bool(multimodality_track_depth) else (3 if (bool(multimodality_nvs) or bool(multimodality_tracking)) else 2)
    multi_any = active_multi_modes > 0
    if v2v_concat_mode is None and multi_any:
        raise ValueError("multimodality mode requires v2v_concat_mode in {'frame','width'} (V2V modes only).")
    if (bool(multimodality_tracking) or bool(multimodality_track_depth)) and v2v_concat_mode == "frame":
        raise NotImplementedError(
            "multimodality tracking width-concat mode is required. Use v2v_concat_mode='width'."
        )

    if v2v_concat_mode == "channel":
        if cond_latents is None:
            raise ValueError("Channel concat requires `cond_latents` (clean conditioning RGB latents).")
        if latents is None:
            raise ValueError("Channel concat requires `latents` (target latents to denoise).")
        if hasattr(dit, "in_dim") and int(latents.shape[1]) != int(getattr(dit, "in_dim")):
            raise ValueError(
                "Channel concat assumes `latents.shape[1] == dit.in_dim` so the model patchify can accept "
                "`torch.cat([cond_latents, latents], dim=1)` as 2×in_dim input."
            )
        if cond_latents.shape != latents.shape:
            raise ValueError(
                f"Channel concat requires cond_latents and latents to have identical shapes. "
                f"Got cond_latents={tuple(cond_latents.shape)}, latents={tuple(latents.shape)}."
            )
        if y is not None:
            raise NotImplementedError("Channel concat currently does not support `y` / VAE-embedding conditioning.")
        latents = torch.cat([cond_latents, latents], dim=1)
        return model_fn_wan_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            sliding_window_size=sliding_window_size,
            sliding_window_stride=sliding_window_stride,
            cfg_merge=cfg_merge,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

    # Optional "control" latents that are channel-concatenated with x before patchify (VideoX-Fun style).
    # Expected to be VAE latents in the *source* domain shape; we expand to V2V concat shape with zeros.
    control_latents_input = kwargs.get("control_latents_input", None)
    if control_latents_input is None:
        control_latents_input = kwargs.get("control_xyz_latents_input", None)

    if control_latents_input is not None:
        if v2v_concat_mode not in ("frame", "width"):
            raise NotImplementedError("control_latents_input is only supported for v2v_concat_mode in {'frame','width'}.")
        if latents is None:
            raise ValueError("control_latents_input requires `latents` to be set.")

        num_domains = multi_num_domains
        control_latents_input = control_latents_input.to(device=latents.device, dtype=latents.dtype)

        # Accept either full V2V-latent shape or source-only shape (will be expanded with zeros for target domains).
        # NOTE: control_latents_input channel count is allowed to differ from latents.C (e.g., concat XYZ + query impulse).
        if int(control_latents_input.shape[0]) != int(latents.shape[0]):
            raise ValueError(
                "control_latents_input batch must match latents batch; "
                f"got control_latents_input={tuple(control_latents_input.shape)} vs latents={tuple(latents.shape)}."
            )
        ctrl_c = int(control_latents_input.shape[1])

        if v2v_concat_mode == "width":
            if int(control_latents_input.shape[2]) != int(latents.shape[2]) or int(control_latents_input.shape[3]) != int(latents.shape[3]):
                raise ValueError(
                    "control_latents_input (T,H) must match latents for width-concat; "
                    f"got control_latents_input={tuple(control_latents_input.shape)} vs latents={tuple(latents.shape)}."
                )
            full_w = int(latents.shape[4])
            split_w = int(full_w // num_domains)
            if int(control_latents_input.shape[4]) == full_w:
                pass  # already full shape
            elif int(control_latents_input.shape[4]) == split_w:
                zeros = torch.zeros_like(control_latents_input)
                control_latents_input = torch.cat([control_latents_input] + [zeros] * (num_domains - 1), dim=4)
            else:
                expected_full = (int(latents.shape[0]), ctrl_c, int(latents.shape[2]), int(latents.shape[3]), full_w)
                expected_src = (int(latents.shape[0]), ctrl_c, int(latents.shape[2]), int(latents.shape[3]), split_w)
                raise ValueError(
                    f"control_latents_input must be either full shape {expected_full} or source-only shape {expected_src} for width-concat. "
                    f"Got {tuple(control_latents_input.shape)}."
                )
        else:  # frame
            if int(control_latents_input.shape[3]) != int(latents.shape[3]) or int(control_latents_input.shape[4]) != int(latents.shape[4]):
                raise ValueError(
                    "control_latents_input (H,W) must match latents for frame-concat; "
                    f"got control_latents_input={tuple(control_latents_input.shape)} vs latents={tuple(latents.shape)}."
                )
            full_t = int(latents.shape[2])
            split_t = int(full_t // num_domains)
            if int(control_latents_input.shape[2]) == full_t:
                pass  # already full shape
            elif int(control_latents_input.shape[2]) == split_t:
                zeros = torch.zeros_like(control_latents_input)
                control_latents_input = torch.cat([control_latents_input] + [zeros] * (num_domains - 1), dim=2)
            else:
                expected_full = (int(latents.shape[0]), ctrl_c, full_t, int(latents.shape[3]), int(latents.shape[4]))
                expected_src = (int(latents.shape[0]), ctrl_c, split_t, int(latents.shape[3]), int(latents.shape[4]))
                raise ValueError(
                    f"control_latents_input must be either full shape {expected_full} or source-only shape {expected_src} for frame-concat. "
                    f"Got {tuple(control_latents_input.shape)}."
                )

        # Enforce: control applies to SOURCE tokens only.
        # For full-shape inputs, this zeros out any accidental target-domain control.
        if v2v_concat_mode == "width":
            split_w = int(latents.shape[4] // num_domains)
            control_latents_input = control_latents_input.clone()
            control_latents_input[..., split_w:] = 0
        else:  # frame
            split_t = int(latents.shape[2] // num_domains)
            control_latents_input = control_latents_input.clone()
            control_latents_input[:, :, split_t:] = 0

    if v2v_concat_mode is not None and v2v_concat_mode not in ("frame", "width"):
        raise ValueError(f"Invalid v2v_concat_mode: {v2v_concat_mode}. Expected 'frame', 'width', 'channel', or None.")
    if reference_latents is not None:
        raise NotImplementedError("V2V mode does not currently support `reference_latents`.")
    if LongCatVideoTransformer3DModel is not None and isinstance(dit, LongCatVideoTransformer3DModel):
        raise NotImplementedError("V2V mode is not supported for LongCat-Video DiT.")
    if audio_embeds is not None:
        raise NotImplementedError("V2V mode is not supported for S2V/audio-conditioned forward.")

    # if use_unified_sequence_parallel:
    #     import torch.distributed as dist
    #     from xfuser.core.distributed import (
    #         get_sequence_parallel_rank,
    #         get_sequence_parallel_world_size,
    #         get_sp_group,
    #     )
    # else:
    dist = None

    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            # motion_controller=motion_controller,
            # vace=vace,
            # vap=vap,
            # animate_adapter=animate_adapter,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            # vace_context=vace_context,
            # vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            # motion_bucket_id=motion_bucket_id,
            v2v_concat_mode=v2v_concat_mode,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video_v2v,
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )

    if getattr(dit, "seperated_timestep", False) and fuse_vae_embedding_in_latents:
        _tokens_per_frame = latents.shape[3] * latents.shape[4] // 4
        _nd = int(ti2v_boundary_domains or 0)
        if _nd >= 2:
            # WIDTH-CONCAT FIX (2026-08-23). ``latents.shape[4]`` is the FULL
            # concatenated width, so the default row below marks EVERY token of
            # latent frame 0 as timestep 0 -- the source half and any non-RGB
            # target domains included. That is right for official single-domain
            # TI2V, where frame 0 genuinely IS the conditioning image, but wrong
            # here: only the target-RGB domain's frame 0 is clamped to the
            # boundary. Declaring the SOURCE half of frame 0 to be t=0
            # conditioning content makes the pretrained TI2V prior continue that
            # source view forward, which leaks source appearance into the
            # temporally adjacent latent frame 1. Measured signature and
            # controls: doc/track4d-360/bugs/2026-08-23-ti2v-native-i2v-latent-frame1-artifact.md
            #
            # Mark t=0 for the boundary domain's frame-0 tokens only; every
            # other domain keeps the ordinary diffusion timestep, which for the
            # source domain matches how it is labelled on all its other frames.
            _td = int(ti2v_boundary_target_domain)
            if not 0 <= _td < _nd:
                raise ValueError(
                    f"ti2v_boundary_target_domain must be in [0, {_nd - 1}], got {_td}"
                )
            _rows = int(latents.shape[3]) // 2
            _cols = int(latents.shape[4]) // 2
            if _cols % _nd != 0:
                raise ValueError(
                    f"token columns {_cols} not divisible by ti2v_boundary_domains={_nd}"
                )
            _dom_cols = _cols // _nd
            _first = torch.ones(
                (_rows, _cols), dtype=latents.dtype, device=latents.device
            ) * timestep
            _first[:, _td * _dom_cols : (_td + 1) * _dom_cols] = 0
            _first = _first.reshape(1, _tokens_per_frame)
        else:
            _first = torch.zeros(
                (1, _tokens_per_frame),
                dtype=latents.dtype,
                device=latents.device,
            )
        timestep = torch.concat(
            [
                _first,
                torch.ones(
                    (latents.shape[2] - 1, _tokens_per_frame),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [
                torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0)
                for chunk in t_chunks
            ]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    elif timestep.dim() != 1:
        patch_size = tuple(getattr(dit, "patch_size", (1, 2, 2)))
        patch_h = int(patch_size[1]) if len(patch_size) >= 2 else 1
        patch_w = int(patch_size[2]) if len(patch_size) >= 3 else 1
        seq_len = int(latents.shape[2] * latents.shape[3] * latents.shape[4] // max(1, patch_h * patch_w))
        if timestep.size(1) < seq_len:
            pad_size = seq_len - int(timestep.size(1))
            padding = timestep[:, -1:].repeat(1, pad_size)
            timestep = torch.cat([timestep, padding], dim=1)
        elif timestep.size(1) > seq_len:
            raise ValueError(
                f"Token-level timestep length {int(timestep.size(1))} exceeds patch token length {seq_len}."
            )
        bt = int(timestep.size(0))
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep.flatten()).unflatten(0, (bt, seq_len))
        )
        if use_unified_sequence_parallel and dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [
                torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0)
                for chunk in t_chunks
            ]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    clean_teacher_x = causal_teacher_clean_latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
        if clean_teacher_x is not None:
            clean_teacher_x = torch.concat([clean_teacher_x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Some Wan checkpoints have `dit.in_dim > vae.z_dim` and expect extra channels appended
    # to the VAE latents before patchify. For V2V training we often don't have that signal;
    # pad with zeros to satisfy the DiT input shape.
    dit_in_dim = getattr(dit, "in_dim", None)
    if dit_in_dim is not None:
        expected_c = int(dit_in_dim)
        current_c = int(x.shape[1])
        if current_c < expected_c:
            if y is not None:
                if int(y.shape[0]) != int(x.shape[0]):
                    raise ValueError(f"`y` batch must match latents batch; got y={tuple(y.shape)} vs latents={tuple(x.shape)}.")
                if tuple(y.shape[2:]) != tuple(x.shape[2:]):
                    raise ValueError(f"`y` spatial/temporal shape must match latents; got y={tuple(y.shape)} vs latents={tuple(x.shape)}.")
                if int(y.shape[1]) != expected_c - current_c:
                    raise ValueError(
                        "Invalid `y` channel count for this DiT. "
                        f"Expected y.C == dit.in_dim - latents.C ({int(y.shape[1])} != {expected_c - current_c})."
                    )
                x = torch.cat([x, y], dim=1)
            else:
                pad_c = expected_c - current_c
                pad = torch.zeros((int(x.shape[0]), int(pad_c), int(x.shape[2]), int(x.shape[3]), int(x.shape[4])), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
                if clean_teacher_x is not None:
                    if tuple(clean_teacher_x.shape[2:]) != tuple(x.shape[2:]):
                        raise ValueError(
                            "causal_teacher_clean_latents spatial/temporal shape must match latents; "
                            f"got clean={tuple(clean_teacher_x.shape)} vs latents={tuple(x.shape)}."
                        )
                    clean_teacher_x = torch.cat([clean_teacher_x, torch.zeros_like(pad)], dim=1)

            if control_latents_input is not None:
                if int(control_latents_input.shape[0]) != int(x.shape[0]):
                    raise ValueError(
                        "`control_latents_input` batch must match latents batch; "
                        f"got control_latents_input={tuple(control_latents_input.shape)} vs latents={tuple(x.shape)}."
                    )
                # Legacy compatibility: if control has fewer channels than dit.in_dim (e.g., vae.z_dim),
                # pad with zeros so older checkpoints that expect in_dim-sized control still work.
                # If control has MORE channels (e.g., concat XYZ + query-impulse), keep as-is and let
                # `dit.control_patch_embedding` adapt its input channel count.
                if int(control_latents_input.shape[1]) < expected_c:
                    pad_c = expected_c - int(control_latents_input.shape[1])
                    pad = torch.zeros(
                        (int(control_latents_input.shape[0]), int(pad_c), int(control_latents_input.shape[2]), int(control_latents_input.shape[3]), int(control_latents_input.shape[4])),
                        device=control_latents_input.device,
                        dtype=control_latents_input.dtype,
                    )
                    control_latents_input = torch.cat([control_latents_input, pad], dim=1)
        elif current_c > expected_c:
            raise ValueError(f"Unexpected input channel count: got latents.C={current_c}, expected dit.in_dim={expected_c}.")
    if clip_feature is not None and getattr(dit, "require_clip_embedding", False):
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # ============================================================================
    # Per-layer camera control mode (alternative to standard mode)
    # ============================================================================
    # Check if per-layer camera control is enabled
    use_perlayer_camera_control = kwargs.get("use_perlayer_camera_control", False)
    camera_features_perlayer = None
    camera_features_spatial = None  # Will be set by per-layer processing if enabled

    # Camera injection mode: "projector" (ReCamMaster-style) or "add" (video4spatial_ttt-style)
    # - "projector": input_x = norm(x) + cam_feat, x = x + gate * projector(attn(input_x))
    # - "add": x = x + cam_add_projector(cam_feat), then standard attention
    camera_injection_mode = kwargs.get("camera_injection_mode", "projector")
    camera_condition_scale = float(kwargs.get("camera_condition_scale", 1.0))
    camera_targets_only = bool(kwargs.get("camera_targets_only", False))
    camera_plucker_encoder_mode = str(kwargs.get("camera_plucker_encoder_mode", "patchify"))
    use_plucker_raymap_tokens = bool(kwargs.get("use_plucker_raymap_tokens", False))
    use_pre_dit_camera_injection = bool(kwargs.get("use_pre_dit_camera_injection", False))
    use_vista4d_latent_streams = bool(kwargs.get("use_vista4d_latent_streams", False))
    vista4d_streams_pointcloud_only = bool(kwargs.get("vista4d_streams_pointcloud_only", False))
    use_gen3c_warp_latent_channels = bool(kwargs.get("use_gen3c_warp_latent_channels", False))
    
    # Debug flag - set to True to enable verbose debugging of camera control
    _debug_camera_control = kwargs.get("_debug_camera_control", False)
    if _debug_camera_control:
        print(f"\n[DEBUG] Camera control mode check:")
        print(f"  use_perlayer_camera_control: {use_perlayer_camera_control}")
        print(f"  use_pre_dit_camera_injection: {use_pre_dit_camera_injection}")
        print(f"  use_plucker_raymap_tokens: {use_plucker_raymap_tokens}")
        print(f"  camera_injection_mode: {camera_injection_mode}")
        print(f"  camera_condition_scale: {camera_condition_scale}")
        print(f"  control_camera_latents_input is not None: {control_camera_latents_input is not None}")
        print(f"  control_camera_pose_input is not None: {control_camera_pose_input is not None}")
        print(f"  control_camera_plucker_input is not None: {control_camera_plucker_input is not None}")
        if control_camera_latents_input is not None:
            print(f"  control_camera_latents_input shape: {control_camera_latents_input.shape}")
        if control_camera_pose_input is not None:
            print(f"  control_camera_pose_input shape: {control_camera_pose_input.shape}")
        if control_camera_plucker_input is not None:
            print(f"  control_camera_plucker_input shape: {control_camera_plucker_input.shape}")
        print(f"  dit has control_adapter: {hasattr(dit, 'control_adapter') and dit.control_adapter is not None}")
        if hasattr(dit, 'blocks') and len(dit.blocks) > 0:
            print(f"  blocks[0] has camera_projector: {hasattr(dit.blocks[0], 'camera_projector')}")
            print(f"  blocks[0] has cam_add_projector: {hasattr(dit.blocks[0], 'cam_add_projector')}")
            print(f"  blocks[0] has cam_encoder: {hasattr(dit.blocks[0], 'cam_encoder')}")
            if hasattr(dit.blocks[0], "cam_encoder"):
                try:
                    enc = dit.blocks[0].cam_encoder
                    w_norm = enc.weight.detach().float().norm().item()
                    b_norm = enc.bias.detach().float().norm().item() if getattr(enc, "bias", None) is not None else 0.0
                    print(f"  blocks[0].cam_encoder weight_norm: {w_norm:.6f}, bias_norm: {b_norm:.6f}")
                except Exception as e:
                    print(f"  blocks[0].cam_encoder norm check failed: {e}")

    if camera_injection_mode == "rt_projector":
        has_camera_cond = control_camera_pose_input is not None
    elif camera_injection_mode == "vista4d_plucker":
        has_camera_cond = control_camera_plucker_input is not None
    else:
        has_camera_cond = control_camera_latents_input is not None

    # Process camera features for per-layer injection (if enabled)
    # This processes the camera features that will be used in each DiT block
    if use_perlayer_camera_control and has_camera_cond:
        # Process camera features separately (not in patchify)
        if camera_injection_mode == "vista4d_plucker":
            if control_camera_plucker_input is None:
                raise RuntimeError("vista4d_plucker mode requires control_camera_plucker_input")
            if not hasattr(dit, "blocks") or not dit.blocks or not hasattr(dit.blocks[0], "cam_encoder"):
                raise RuntimeError(
                    "vista4d_plucker mode requires per-block Linear cam_encoder modules. "
                    "Initialize them via initialize_vista4d_plucker_camera_encoders()."
                )
            if not hasattr(dit.blocks[0], "camera_projector"):
                raise RuntimeError(
                    "vista4d_plucker mode requires identity-initialized per-block camera_projector modules."
                )
            if control_camera_plucker_input.ndim != 5 or int(control_camera_plucker_input.shape[-1]) != 6:
                raise ValueError(
                    "vista4d_plucker mode expects control_camera_plucker_input shape "
                    f"(B, F, H, W, 6), got {tuple(control_camera_plucker_input.shape)}"
                )
            camera_features_spatial = rearrange(
                control_camera_plucker_input,
                "b f h w d -> b d f h w",
            ).contiguous()
            if _debug_camera_control and not getattr(dit, "_logged_vista4d_plucker_runtime", False):
                cam_stats = camera_features_spatial.detach().float()
                print(
                    "[CameraControl] vista4d_plucker runtime "
                    f"shape={tuple(control_camera_plucker_input.shape)} "
                    f"spatial_shape={tuple(camera_features_spatial.shape)} "
                    f"device={control_camera_plucker_input.device} "
                    f"dtype={control_camera_plucker_input.dtype} "
                    f"mean={cam_stats.mean().item():.6g} "
                    f"std={cam_stats.std(unbiased=False).item():.6g} "
                    f"absmax={cam_stats.abs().max().item():.6g}",
                    flush=True,
                )
                dit._logged_vista4d_plucker_runtime = True
            x = dit.patchify(x, None, v2v_concat_mode, multi_num_domains if multi_any else False, control_latents_input=control_latents_input, feedback_input=feedback_input)
        elif use_plucker_raymap_tokens:
            if camera_injection_mode != "projector":
                raise ValueError(
                    "use_plucker_raymap_tokens expects camera_injection_mode='projector' "
                    "so Ecam(raymaps) is added to video tokens before self-attention."
                )
            if camera_plucker_encoder_mode != "patchify":
                raise ValueError(
                    "use_plucker_raymap_tokens requires camera_plucker_encoder_mode='patchify'."
                )
            if control_camera_latents_input is None:
                raise RuntimeError("use_plucker_raymap_tokens requires control_camera_latents_input")
            if not hasattr(dit, "_camera_patchify_unshuffle") or not hasattr(dit, "_camera_patchify_conv"):
                raise RuntimeError(
                    "use_plucker_raymap_tokens requires camera patchify layers. "
                    "Initialize them via initialize_camera_raymap_token_projection()."
                )
            if not hasattr(dit, "_camera_plucker_patchify_encoder"):
                raise RuntimeError(
                    "use_plucker_raymap_tokens requires `_camera_plucker_patchify_encoder`. "
                    "Initialize it via initialize_camera_raymap_token_projection()."
                )
            if not hasattr(dit, "blocks") or not dit.blocks or not hasattr(dit.blocks[0], "camera_projector"):
                raise RuntimeError(
                    "use_plucker_raymap_tokens requires identity-initialized per-block camera_projector modules. "
                    "Initialize them via initialize_camera_raymap_token_projection()."
                )

            B, C, T_cam, H, W = control_camera_latents_input.shape
            camera_rays = control_camera_latents_input.permute(0, 2, 1, 3, 4).contiguous().view(B * T_cam, C, H, W)
            camera_rays = dit._camera_patchify_unshuffle(camera_rays)
            camera_features_6 = dit._camera_patchify_conv(camera_rays)
            _, C6, Hc, Wc = camera_features_6.shape
            camera_features_6 = camera_features_6.view(B, T_cam, C6, Hc, Wc).permute(0, 2, 1, 3, 4).contiguous()
            camera_features_spatial = dit._camera_plucker_patchify_encoder(camera_features_6)
            if _debug_camera_control:
                print(f"  ✓ Plucker raymap token mode: camera_features_spatial shape: {camera_features_spatial.shape}")
                print(f"    camera_features_spatial mean: {camera_features_spatial.mean():.6f}, std: {camera_features_spatial.std():.6f}")
            x = dit.patchify(x, None, v2v_concat_mode, multi_num_domains if multi_any else False, control_latents_input=control_latents_input, feedback_input=feedback_input)
        elif camera_injection_mode in ("direct_raymap_add", "kq_only"):
            # Direct raymap / KQ-only mode with per-layer cam_encoder
            # Both use per-layer Conv3d encoding of raw 6D Plücker rays.
            # Difference is injection: direct_raymap_add adds to input_x, kq_only passes as kq_bias to Q/K only.
            # Supports two Plücker encoder modes:
            # - "patchify": patchify layers extract 6D rays, then cam_encoder(6→dim) per-layer
            # - "control_adapter": SimpleAdapter encodes 24D→dim, then cam_encoder(dim→dim) per-layer

            B, C, T, H, W = control_camera_latents_input.shape
            # C=24 for 4 patches with 6D rays each (24 = 6D × 2×2 patch_size)

            # Check per-layer cam_encoder initialization
            if not hasattr(dit, "blocks") or dit.blocks is None or len(dit.blocks) == 0 or not hasattr(dit.blocks[0], "cam_encoder"):
                raise RuntimeError(
                    f"{camera_injection_mode} mode requires per-layer `cam_encoder` modules, but none were found on `dit.blocks`.\n"
                    "Initialize them via `initialize_perlayer_camera_encoders()` before inference/training and ensure the checkpoint loads them."
                )

            if camera_plucker_encoder_mode == "patchify":
                # PATCHIFY MODE: Extract 6D Plücker rays, per-layer cam_encoder does Conv3d(6→dim)
                # Use PixelUnshuffle + Conv to patchify and extract 6D Plücker rays spatially
                # This matches SimpleAdapter's structure but extracts 6D instead of encoding to dim

                # Extract the patch_size and downscale_factor from dit/vae
                patch_size = tuple(getattr(dit, "patch_size", (1, 2, 2)))  # (T, H, W)
                ph, pw = patch_size[1], patch_size[2]

                # Prepare for spatial processing: (B, 24, T, H, W) → (B*T, 24, H, W)
                camera_rays = control_camera_latents_input.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 24, H, W)
                camera_rays = camera_rays.view(B * T, C, H, W)  # (B*T, 24, H, W)

                # Patchify layers MUST be initialized before forward pass (via initialize_camera_patchify_layers)
                # to avoid DDP issues and ensure checkpoint compatibility
                if not hasattr(dit, "_camera_patchify_unshuffle") or not hasattr(dit, "_camera_patchify_conv"):
                    raise RuntimeError(
                        "camera_plucker_encoder_mode='patchify' with direct_raymap_add requires patchify layers! "
                        "Call initialize_camera_patchify_layers() before using this mode. "
                        "This is required for DDP compatibility and checkpoint loading."
                    )

                # DEBUG MODE: One-hot embeddings based on trajectory pair index
                if debug_onehot_camera:
                    if trajectory_pair_index is None:
                        raise ValueError(
                            "debug_onehot_camera mode requires trajectory_pair_index to be provided! "
                            "Make sure the dataset returns 'trajectory_pair_index' and it's passed through the pipeline."
                        )

                    # Validate trajectory_pair_index is in valid range
                    if isinstance(trajectory_pair_index, torch.Tensor):
                        trajectory_indices = trajectory_pair_index.cpu().tolist()
                        if len(set(trajectory_indices)) == 1 and B > 1:
                            print(f"⚠ Warning: All samples have same trajectory_pair_index={trajectory_indices[0]} (batch size={B})")
                    else:
                        trajectory_indices = [trajectory_pair_index]

                    # Create one-hot embeddings instead of processing Plücker rays
                    # This helps diagnose if the issue is with camera encoding or injection mechanism

                    # Apply PixelUnshuffle to get correct spatial dimensions
                    camera_rays = dit._camera_patchify_unshuffle(camera_rays)
                    # Apply Conv2d to get correct spatial dimensions (but we'll replace the values)
                    camera_features_spatial = dit._camera_patchify_conv(camera_rays)
                    _, C_out, H_out, W_out = camera_features_spatial.shape

                    # Create one-hot embeddings: (B, 6, T, H', W')
                    # Each trajectory pair gets a unique one-hot pattern across the 6 channels
                    camera_features_spatial = torch.zeros(B, T, 6, H_out, W_out,
                                                         device=camera_features_spatial.device,
                                                         dtype=camera_features_spatial.dtype)

                    # Set one-hot pattern for each sample based on trajectory_pair_index
                    for b in range(B):
                        idx = trajectory_pair_index[b].item() if isinstance(trajectory_pair_index, torch.Tensor) else trajectory_pair_index
                        if idx < 0 or idx > 5:
                            raise ValueError(f"trajectory_pair_index[{b}]={idx} out of valid range [0, 5]")
                        camera_features_spatial[b, :, idx, :, :] = 1.0  # Set channel idx to 1.0 for all spatial/temporal locations

                    # Reshape to (B, 6, T, H', W')
                    camera_features_spatial = camera_features_spatial.permute(0, 2, 1, 3, 4).contiguous()

                    if _debug_camera_control:
                        print(f"  🐛 DEBUG ONE-HOT MODE: camera_features_spatial shape: {camera_features_spatial.shape}")
                        print(f"    trajectory_pair_indices: {trajectory_pair_index if isinstance(trajectory_pair_index, int) else trajectory_pair_index.tolist()}")
                        print(f"    one-hot pattern (first sample): channel sums = {[camera_features_spatial[0, i].sum().item() for i in range(6)]}")
                else:
                    # NORMAL MODE: Process Plücker rays through patchify layers
                    # Apply PixelUnshuffle: (B*T, 24, H, W) → (B*T, 24*downscale^2, H//downscale, W//downscale)
                    camera_rays = dit._camera_patchify_unshuffle(camera_rays)

                    # Apply Conv2d: (B*T, 24*downscale^2, H', W') → (B*T, 6, H'//ph, W'//pw)
                    camera_features_spatial = dit._camera_patchify_conv(camera_rays)

                    # Reshape back to (B, 6, T, H', W')
                    _, C_out, H_out, W_out = camera_features_spatial.shape
                    camera_features_spatial = camera_features_spatial.view(B, T, C_out, H_out, W_out)
                    camera_features_spatial = camera_features_spatial.permute(0, 2, 1, 3, 4).contiguous()  # (B, 6, T, H', W')

                    if _debug_camera_control:
                        print(f"  ✓ Direct raymap mode (patchify): camera_features_spatial shape: {camera_features_spatial.shape}")
                        print(f"    camera_features_spatial mean: {camera_features_spatial.mean():.6f}, std: {camera_features_spatial.std():.6f}")

            elif camera_plucker_encoder_mode == "control_adapter":
                # CONTROL_ADAPTER MODE: SimpleAdapter encodes 24D→dim, per-layer cam_encoder does Conv3d(dim→dim)
                # This replaces patchify layers with SimpleAdapter while keeping per-layer processing

                if not hasattr(dit, "control_adapter") or dit.control_adapter is None:
                    raise RuntimeError(
                        "camera_plucker_encoder_mode='control_adapter' with direct_raymap_add requires control_adapter! "
                        "Initialize it via initialize_camera_controller() before using this mode."
                    )

                # Process with SimpleAdapter: (B, 24, T, H, W) → (B, dim, T, H', W')
                camera_features_spatial = dit.control_adapter(control_camera_latents_input)

                if _debug_camera_control:
                    print(f"  ✓ Direct raymap mode (control_adapter): camera_features_spatial shape: {camera_features_spatial.shape}")
                    print(f"    camera_features_spatial mean: {camera_features_spatial.mean():.6f}, std: {camera_features_spatial.std():.6f}")
            else:
                raise ValueError(
                    f"Unsupported camera_plucker_encoder_mode='{camera_plucker_encoder_mode}' for direct_raymap_add. "
                    f"Supported modes: 'patchify', 'control_adapter'"
                )

            # Don't add to patchified features yet - will encode per-layer with Conv3d
            x = dit.patchify(x, None, v2v_concat_mode, multi_num_domains if multi_any else False, control_latents_input=control_latents_input, feedback_input=feedback_input)  # Pass None to skip camera control in patchify
        elif camera_injection_mode == "rt_projector":
            # ReCamMaster-style RT conditioning uses packed pose features (no SimpleAdapter).
            # We'll expand to token grid after patchify.
            x = dit.patchify(x, None, v2v_concat_mode, multi_num_domains if multi_any else False, control_latents_input=control_latents_input, feedback_input=feedback_input)  # Skip camera control in patchify
            camera_features_spatial = None
        elif camera_plucker_encoder_mode == "patchify":
            # Patchify-based Plücker encoding:
            # 24D packed rays -> (PixelUnshuffle+Conv2d) -> 6D raymap -> Conv3d(6->dim)
            if control_camera_latents_input is None:
                raise RuntimeError("camera_plucker_encoder_mode='patchify' requires control_camera_latents_input")
            if not hasattr(dit, "_camera_patchify_unshuffle") or not hasattr(dit, "_camera_patchify_conv"):
                raise RuntimeError(
                    "camera_plucker_encoder_mode='patchify' requires camera patchify layers.\n"
                    "Initialize them via `initialize_camera_patchify_layers()` before loading checkpoints."
                )
            if not hasattr(dit, "_camera_plucker_patchify_encoder"):
                raise RuntimeError(
                    "camera_plucker_encoder_mode='patchify' requires `_camera_plucker_patchify_encoder`.\n"
                    "Initialize it via `initialize_camera_patchify_plucker_encoder()`."
                )

            B, C, T_cam, H, W = control_camera_latents_input.shape
            camera_rays = control_camera_latents_input.permute(0, 2, 1, 3, 4).contiguous().view(B * T_cam, C, H, W)
            camera_rays = dit._camera_patchify_unshuffle(camera_rays)
            camera_features_6 = dit._camera_patchify_conv(camera_rays)  # (B*T, 6, H', W')
            _, C6, Hc, Wc = camera_features_6.shape
            camera_features_6 = camera_features_6.view(B, T_cam, C6, Hc, Wc).permute(0, 2, 1, 3, 4).contiguous()  # (B, 6, T, H', W')
            camera_features_spatial = dit._camera_plucker_patchify_encoder(camera_features_6)  # (B, dim, T, H', W')
            if _debug_camera_control:
                print(f"  ✓ Per-layer mode (patchify encoder): camera_features_spatial shape: {camera_features_spatial.shape}")
                print(f"    camera_features_spatial mean: {camera_features_spatial.mean():.6f}, std: {camera_features_spatial.std():.6f}")
            # Don't add to patchified features yet - will add per-layer
            x = dit.patchify(x, None, v2v_concat_mode, multi_num_domains if multi_any else False, control_latents_input=control_latents_input, feedback_input=feedback_input)  # Pass None to skip camera control in patchify
        elif hasattr(dit, "control_adapter") and dit.control_adapter is not None:
            # Standard per-layer modes: Use SimpleAdapter (control_adapter) to encode globally
            camera_features_spatial = dit.control_adapter(control_camera_latents_input)  # (B, dim, T, H, W)
            if _debug_camera_control:
                print(f"  ✓ Per-layer mode: camera_features_spatial shape: {camera_features_spatial.shape}")
                print(f"    camera_features_spatial mean: {camera_features_spatial.mean():.6f}, std: {camera_features_spatial.std():.6f}")
            x = dit.patchify(x, None, v2v_concat_mode, multi_num_domains if multi_any else False, control_latents_input=control_latents_input, feedback_input=feedback_input)
        else:
            raise RuntimeError(f"Per-layer camera control (mode={camera_injection_mode}) requires appropriate initialization")
    else:
        # Standard mode: add camera control in patchify (if present)
        if _debug_camera_control:
            print(f"  → Using standard mode (not per-layer)")
        x = dit.patchify(x, control_camera_latents_input, v2v_concat_mode, multi_num_domains if multi_any else False, control_latents_input=control_latents_input, feedback_input=feedback_input)
        camera_features_spatial = None

    f, h, w = x.shape[2:]
    clean_teacher_tokens = None
    if causal_teacher_mode:
        if clean_teacher_x is None:
            raise RuntimeError("causal_teacher_mode requires clean_teacher_x")
        if tuple(clean_teacher_x.shape) != tuple(latents.shape):
            # Compare to the original latent tensor here; channel padding is handled above.
            if tuple(clean_teacher_x.shape[2:]) != tuple(x.shape[2:]):
                raise ValueError(
                    "causal_teacher_clean_latents must match latents before patchify; "
                    f"got clean={tuple(clean_teacher_x.shape)}, patchified x={tuple(x.shape)}."
                )
        clean_patch = dit.patchify(clean_teacher_x, None, v2v_concat_mode, multi_num_domains if multi_any else False)
        if tuple(clean_patch.shape[2:]) != (int(f), int(h), int(w)):
            raise ValueError(
                "causal_teacher_clean_latents patch grid mismatch: "
                f"clean_patch={tuple(clean_patch.shape)}, x={tuple(x.shape)}."
            )
        clean_teacher_tokens = rearrange(clean_patch, "b c f h w -> b (f h w) c").contiguous()

    is_v2v_layout = v2v_concat_mode in ("frame", "width")
    # Determine number of domains:
    # - Standard (v2v_concat_mode=None): 1 domain
    # - V2V RGB-only: 2 domains (source_rgb, target_rgb)
    # - V2V RGB+Depth: 3 domains (source_rgb, target_rgb, target_depth)
    num_domains = multi_num_domains if is_v2v_layout else 1

    # Validate spatial dimensions based on number of domains
    if is_v2v_layout:
        if v2v_concat_mode == "frame":
            if f % num_domains != 0:
                raise ValueError(
                    f"V2V frame mode with {num_domains} domains requires frame length divisible by {num_domains}, got f={f}."
                )
        else:  # width
            if w % num_domains != 0:
                raise ValueError(
                    f"V2V width mode with {num_domains} domains requires width divisible by {num_domains}, got w={w}."
                )

    # ============================================================================
    # Optional camera injection into "noisy" features (VideoX-Fun style, patch-level)
    # ============================================================================
    # Inject camera-conditioned features right after patch embedding and BEFORE domain tokens / DiT blocks.
    # This is independent from pre-DiT injection and per-layer injection.
    use_noise_camera_injection = bool(kwargs.get("use_noise_camera_injection", False))
    if use_noise_camera_injection and has_camera_cond and camera_injection_mode != "rt_projector":
        if not hasattr(dit, "noise_cam_encoder") or dit.noise_cam_encoder is None:
            if getattr(dit, "training", False):
                raise RuntimeError(
                    "use_noise_camera_injection requires dit.noise_cam_encoder to be initialized before training. "
                    "Initialize it via initialize_noise_camera_encoder() before DDP/optimizer setup."
                )
            from holo4d_runtime.models.wan_video_camera_controller import initialize_noise_camera_encoder

            initialize_noise_camera_encoder(dit)

        if not hasattr(dit, "control_adapter") or dit.control_adapter is None:
            raise RuntimeError(
                "use_noise_camera_injection requires dit.control_adapter (SimpleAdapter) to be initialized."
            )

        # Prefer reuse of already-encoded camera features from per-layer section to avoid redundant control_adapter call.
        # Only reuse if it is in dim-space (Conv3d expects dim channels).
        camera_features_encoded = None
        if camera_features_spatial is not None:
            try:
                if int(camera_features_spatial.shape[1]) == int(getattr(dit, "dim")):
                    camera_features_encoded = camera_features_spatial
            except Exception:
                camera_features_encoded = None
        if camera_features_encoded is None:
            camera_features_encoded = dit.control_adapter(control_camera_latents_input)

        # Ensure camera features match x's layout (standard or width/frame concat).
        _, _, cam_f, cam_h, cam_w = camera_features_encoded.shape
        if cam_f != f or cam_h != h or cam_w != w:
            if is_v2v_layout:
                if v2v_concat_mode == "width" and cam_f == f and cam_h == h and cam_w * num_domains == w:
                    camera_features_encoded = camera_features_encoded.repeat(1, 1, 1, 1, num_domains)
                elif v2v_concat_mode == "frame" and cam_h == h and cam_w == w and cam_f * num_domains == f:
                    camera_features_encoded = camera_features_encoded.repeat(1, 1, num_domains, 1, 1)
                else:
                    raise ValueError(
                        f"use_noise_camera_injection: encoded camera features shape {tuple(camera_features_encoded.shape)} "
                        f"is incompatible with patchified x shape {tuple(x.shape)} (v2v_concat_mode={v2v_concat_mode}, num_domains={num_domains})."
                    )
            else:
                raise ValueError(
                    f"use_noise_camera_injection: encoded camera features shape {tuple(camera_features_encoded.shape)} "
                    f"must match patchified x shape {tuple(x.shape)} in standard mode (v2v_concat_mode=None)."
                )

        camera_features_noise = dit.noise_cam_encoder(camera_features_encoded)

        # Optional: apply camera_targets_only (zero out source domain). No-op in standard mode (single domain).
        if camera_targets_only and is_v2v_layout:
            _, _, nf, nh, nw = camera_features_noise.shape
            if v2v_concat_mode == "width":
                split = nw // num_domains
                camera_features_noise[:, :, :, :, :split] = 0
            else:  # frame
                split = nf // num_domains
                camera_features_noise[:, :, :split, :, :] = 0

        camera_features_noise = camera_features_noise * camera_condition_scale
        x = x + camera_features_noise

        if _debug_camera_control:
            print(f"\n[DEBUG] Noise-camera injection enabled (patch-level)")
            print(f"  camera_features_noise shape: {camera_features_noise.shape}")
            print(f"  x shape after noise-camera injection: {x.shape}")

    use_domain_embedding = kwargs.get("use_domain_embedding", None)
    if use_domain_embedding is None:
        use_domain_embedding = True

    if is_v2v_layout and use_domain_embedding:
        # Create domain embedding with correct number of domains
        _ensure_domain_embedding(dit, num_domains=num_domains)

        # Assign domain IDs based on concatenation mode and multi-modality
        domain_ids = torch.zeros((f, h, w), dtype=torch.long, device=x.device)
        if v2v_concat_mode == "frame":
            split = f // num_domains
            for i in range(1, num_domains):
                domain_ids[i * split : (i + 1) * split if i < num_domains - 1 else f, :, :] = i
        else:  # width mode
            split = w // num_domains
            if num_domains == 3:
                # 3-way split: [source | target_1 | target_2]
                domain_ids[:, :, split : 2 * split] = 1
                domain_ids[:, :, 2 * split :] = 2
            else:
                # 2-way split: [source_rgb | target_rgb]
                domain_ids[:, :, split :] = 1

        domain_tokens = dit.domain_embedding[domain_ids].permute(3, 0, 1, 2).unsqueeze(0).to(dtype=x.dtype)
        x = x + domain_tokens

    if use_gen3c_warp_latent_channels:
        if gen3c_warp_condition_latents is None:
            raise RuntimeError("use_gen3c_warp_latent_channels=True but missing gen3c_warp_condition_latents")
        if not hasattr(dit, "gen3c_warp_condition_patch_embedding"):
            raise RuntimeError(
                "use_gen3c_warp_latent_channels=True requires dit.gen3c_warp_condition_patch_embedding; "
                "initialize via initialize_gen3c_warp_condition_patch_embedding()."
            )
        gen3c_cond = dit.gen3c_warp_condition_patch_embedding(
            gen3c_warp_condition_latents.to(device=x.device, dtype=x.dtype)
        )
        if is_v2v_layout:
            if v2v_concat_mode == "width":
                if int(gen3c_cond.shape[2]) != int(f) or int(gen3c_cond.shape[3]) != int(h):
                    raise ValueError(
                        f"GEN3C condition grid mismatch: condition={tuple(gen3c_cond.shape)}, x={tuple(x.shape)}"
                    )
                if int(gen3c_cond.shape[4]) * int(num_domains) == int(w):
                    gen3c_cond = torch.cat(
                        [torch.zeros_like(gen3c_cond), gen3c_cond.repeat(1, 1, 1, 1, int(num_domains) - 1)],
                        dim=4,
                    )
                elif int(gen3c_cond.shape[4]) != int(w):
                    raise ValueError(
                        f"GEN3C condition width mismatch: condition={tuple(gen3c_cond.shape)}, x={tuple(x.shape)}, "
                        f"num_domains={int(num_domains)}"
                    )
            elif v2v_concat_mode == "frame":
                if int(gen3c_cond.shape[3]) != int(h) or int(gen3c_cond.shape[4]) != int(w):
                    raise ValueError(
                        f"GEN3C condition grid mismatch: condition={tuple(gen3c_cond.shape)}, x={tuple(x.shape)}"
                    )
                if int(gen3c_cond.shape[2]) * int(num_domains) == int(f):
                    gen3c_cond = torch.cat(
                        [torch.zeros_like(gen3c_cond), gen3c_cond.repeat(1, 1, int(num_domains) - 1, 1, 1)],
                        dim=2,
                    )
                elif int(gen3c_cond.shape[2]) != int(f):
                    raise ValueError(
                        f"GEN3C condition frame mismatch: condition={tuple(gen3c_cond.shape)}, x={tuple(x.shape)}, "
                        f"num_domains={int(num_domains)}"
                    )
        if gen3c_cond.shape[2:] != x.shape[2:]:
            raise ValueError(
                "GEN3C condition token grid mismatch after expansion: "
                f"x={tuple(x.shape)}, condition={tuple(gen3c_cond.shape)}"
            )
        x = x + gen3c_cond

    vista4d_stream_count = 1
    if use_vista4d_latent_streams:
        # pointcloud_only: in width/frame-concat V2V the clean source video is
        # already carried by the source domain of x, so the Vista4D source-video
        # stream is redundant. Drop it and append only the reprojected
        # point-cloud render stream (sequence becomes [output | point-cloud],
        # stream_count 3 -> 2). The camera-feature repeat (`:3507`) and RoPE
        # freqs repeat (`:3569`) key off `vista4d_stream_count`, so they adapt
        # automatically; the track-token aligner null-pads to x.shape[1].
        required_streams = {
            "vista4d_pointcloud_video_latents": vista4d_pointcloud_video_latents,
            "vista4d_pointcloud_mask_latents": vista4d_pointcloud_mask_latents,
        }
        required_embeddings = [
            "vista4d_pointcloud_patch_embedding",
            "vista4d_pointcloud_mask_patch_embedding",
        ]
        if not vista4d_streams_pointcloud_only:
            required_streams["vista4d_source_video_latents"] = vista4d_source_video_latents
            required_streams["vista4d_source_mask_latents"] = vista4d_source_mask_latents
            required_embeddings = [
                "vista4d_source_patch_embedding",
                "vista4d_source_mask_patch_embedding",
            ] + required_embeddings
        missing = [name for name, value in required_streams.items() if value is None]
        if missing:
            raise RuntimeError(f"use_vista4d_latent_streams=True but missing {missing}")
        for name in required_embeddings:
            if not hasattr(dit, name):
                raise RuntimeError(
                    f"use_vista4d_latent_streams=True requires dit.{name}; "
                    "initialize via initialize_vista4d_latent_stream_patch_embeddings()."
                )

        pointcloud_stream = dit.vista4d_pointcloud_patch_embedding(
            vista4d_pointcloud_video_latents.to(device=x.device, dtype=x.dtype)
        )
        pointcloud_stream = pointcloud_stream + dit.vista4d_pointcloud_mask_patch_embedding(
            vista4d_pointcloud_mask_latents.to(device=x.device, dtype=x.dtype)
        )
        source_stream = None
        if not vista4d_streams_pointcloud_only:
            source_stream = dit.vista4d_source_patch_embedding(
                vista4d_source_video_latents.to(device=x.device, dtype=x.dtype)
            )
            source_stream = source_stream + dit.vista4d_source_mask_patch_embedding(
                vista4d_source_mask_latents.to(device=x.device, dtype=x.dtype)
            )
        if is_v2v_layout:
            source_stream, pointcloud_stream = expand_vista4d_streams_for_v2v(
                source_stream,
                pointcloud_stream,
                x_shape=tuple(x.shape),
                v2v_concat_mode=str(v2v_concat_mode),
                num_domains=int(num_domains),
            )
        if pointcloud_stream.shape[2:] != x.shape[2:] or (
            source_stream is not None and source_stream.shape[2:] != x.shape[2:]
        ):
            raise ValueError(
                "Vista4D stream token grid mismatch: "
                f"x={tuple(x.shape)}, "
                f"source={None if source_stream is None else tuple(source_stream.shape)}, "
                f"pointcloud={tuple(pointcloud_stream.shape)}"
            )
        if source_stream is None:
            x = torch.cat([x, pointcloud_stream], dim=2)
            vista4d_stream_count = 2
        else:
            x = torch.cat([x, pointcloud_stream, source_stream], dim=2)
            vista4d_stream_count = 3

    # ============================================================================
    # Pre-DiT Camera Injection (VideoX-Fun style)
    # ============================================================================
    # Add camera features to x AFTER patch embedding and domain tokens but BEFORE DiT blocks
    # This is ADDITIVE with per-layer camera injection - both can be enabled simultaneously
    if use_pre_dit_camera_injection and use_perlayer_camera_control and has_camera_cond:
        if _debug_camera_control:
            print(f"\n[DEBUG] Pre-DiT camera injection mode enabled (in addition to per-layer)")
            print(f"  x shape before pre-DiT camera injection: {x.shape}")

        # Check if pre-DiT cam_encoder exists
        if not hasattr(dit, 'pre_dit_cam_encoder') or dit.pre_dit_cam_encoder is None:
            if getattr(dit, "training", False):
                raise RuntimeError(
                    "use_pre_dit_camera_injection requires dit.pre_dit_cam_encoder to be initialized before training. "
                    "Initialize it via initialize_pre_dit_camera_encoder() before DDP/optimizer setup."
                )
            from holo4d_runtime.models.wan_video_camera_controller import initialize_pre_dit_camera_encoder

            initialize_pre_dit_camera_encoder(dit)

        # Process camera features using control_adapter (SimpleAdapter) first
        if not hasattr(dit, 'control_adapter') or dit.control_adapter is None:
            raise RuntimeError(
                "use_pre_dit_camera_injection requires dit.control_adapter to be initialized! "
                "Ensure control_adapter (SimpleAdapter) is created before inference/training."
            )

        # Reuse camera_features_spatial from per-layer section if available (avoid redundant control_adapter call)
        # camera_features_spatial is already computed in per-layer section above (line 1449 or 1497)
        if camera_features_spatial is not None:
            # Reuse existing encoded features from per-layer processing
            camera_features_encoded = camera_features_spatial
            if _debug_camera_control:
                print(f"  ✓ Reusing camera_features_spatial from per-layer section (avoiding redundant control_adapter call)")
        else:
            # Encode camera features: (B, 24, T, H, W) → (B, dim, T, H', W')
            camera_features_encoded = dit.control_adapter(control_camera_latents_input)

        if _debug_camera_control:
            print(f"  control_camera_latents_input shape: {control_camera_latents_input.shape}")
            print(f"  camera_features_encoded shape (after control_adapter): {camera_features_encoded.shape}")

        # Apply pre-DiT cam_encoder: (B, dim, T, H', W') → (B, dim, T, H', W')
        # This is a Conv3d(dim, dim, kernel_size=1) module, zero-initialized
        camera_features_predit = dit.pre_dit_cam_encoder(camera_features_encoded)

        if _debug_camera_control:
            print(f"  camera_features_predit shape (after pre_dit_cam_encoder): {camera_features_predit.shape}")

        # Optional: apply camera_targets_only (zero out source domain)
        if camera_targets_only:
            _, _, cam_f, cam_h, cam_w = camera_features_predit.shape
            if v2v_concat_mode == "width":
                # Zero out source domain (first 1/num_domains of width)
                split = cam_w // num_domains
                camera_features_predit[:, :, :, :, :split] = 0
            elif v2v_concat_mode == "frame":
                split = cam_f // num_domains
                camera_features_predit[:, :, :split, :, :] = 0

        # Apply camera_condition_scale
        camera_features_predit = camera_features_predit * camera_condition_scale

        # Add encoded camera features to x (broadcast to match x's shape)
        # x shape: (B, C, F, H, W), camera_features_predit shape: (B, dim, T, H', W')
        x = x + camera_features_predit

        if _debug_camera_control:
            print(f"  x shape after pre-DiT camera injection: {x.shape}")
            print(f"  camera_features_predit mean: {camera_features_predit.mean():.6f}, std: {camera_features_predit.std():.6f}")

    # ============================================================================
    # Optional noise injection before DiT blocks (token-space augmentation)
    # ============================================================================
    pre_dit_noise_injection_std = float(kwargs.get("pre_dit_noise_injection_std", 0.0) or 0.0)
    if pre_dit_noise_injection_std > 0:
        noise = torch.randn_like(x, dtype=torch.float32).to(dtype=x.dtype)
        x = x + noise * pre_dit_noise_injection_std
        if _debug_camera_control:
            print(
                f"[DEBUG] pre_dit_noise_injection_std={pre_dit_noise_injection_std} "
                f"(mean={noise.mean().item():.6f}, std={noise.std().item():.6f})"
            )

    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    base_token_count = int(f * h * w)
    causal_teacher_mask = None
    if causal_teacher_mode:
        if clean_teacher_tokens is None:
            raise RuntimeError("causal_teacher_mode expected clean_teacher_tokens")
        if t_mod.ndim != 4:
            seq_len = int(base_token_count)
            scalar_timestep = timestep.reshape(-1, 1).to(device=x.device, dtype=x.dtype).expand(int(x.shape[0]), seq_len)
            t_clean = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, torch.zeros_like(scalar_timestep).flatten()).unflatten(0, (int(x.shape[0]), seq_len))
            )
            t_noisy = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, scalar_timestep.flatten()).unflatten(0, (int(x.shape[0]), seq_len))
            )
            t_mod = torch.cat([dit.time_projection(t_clean), dit.time_projection(t_noisy)], dim=1).unflatten(2, (6, dit.dim))
        else:
            clean_timestep = torch.zeros((int(x.shape[0]), int(base_token_count)), device=x.device, dtype=timestep.dtype)
            t_clean = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, clean_timestep.flatten()).unflatten(0, (int(x.shape[0]), int(base_token_count)))
            )
            t_mod_clean = dit.time_projection(t_clean).unflatten(2, (6, dit.dim))
            t_mod = torch.cat([t_mod_clean, t_mod], dim=1)
        x = torch.cat([clean_teacher_tokens, x], dim=1)
        from holo4d.ar.causal_teacher_forcing import build_teacher_forcing_causal_mask

        causal_teacher_mask = build_teacher_forcing_causal_mask(
            num_latent_frames=int(f),
            frame_seq_length=int(h) * int(w),
            num_frame_per_block=int(causal_teacher_num_frame_per_block),
            device=x.device,
        ).view(1, 1, int(base_token_count) * 2, int(base_token_count) * 2)

    # Prepare per-layer camera features if using per-layer mode
    if use_perlayer_camera_control and (camera_features_spatial is not None or camera_injection_mode == "rt_projector"):
        if camera_injection_mode == "rt_projector":
            if control_camera_pose_input is None:
                raise RuntimeError("rt_projector mode requires control_camera_pose_input")
            if control_camera_pose_input.ndim != 5:
                raise ValueError(
                    f"Expected control_camera_pose_input shape (B, 48, F, 1, 1), got {tuple(control_camera_pose_input.shape)}"
                )
            pose_b, pose_c, pose_f, pose_h, pose_w = control_camera_pose_input.shape
            if int(pose_h) != 1 or int(pose_w) != 1:
                raise ValueError(f"Expected control_camera_pose_input spatial dims (1,1), got ({int(pose_h)},{int(pose_w)})")
            if int(pose_f) != int(f) and int(pose_f) * int(num_domains) != int(f):
                raise ValueError(f"Pose latent time F={int(pose_f)} is incompatible with token F={int(f)} (num_domains={int(num_domains)})")
            if int(pose_c) not in (12, 48):
                raise ValueError(f"Expected pose channel dim 12 (raw) or 48 (12*4 packed), got {int(pose_c)}")

            # Expand packed RT features spatially to match token grid.
            # Start from base grid (before width/frame concat), then reuse the existing repeat logic below.
            if v2v_concat_mode == "width":
                w_base = int(w // num_domains)
                camera_features_spatial = control_camera_pose_input.expand(int(pose_b), int(pose_c), int(pose_f), int(h), int(w_base)).contiguous()
            else:
                camera_features_spatial = control_camera_pose_input.expand(int(pose_b), int(pose_c), int(pose_f), int(h), int(w)).contiguous()

        # Reshape camera features to match token dimension: (B, dim, T, H, W) → (B, T*H*W, dim)
        # Note: camera_features_spatial may have different spatial dims due to width/frame concat
        _, _, cam_f, cam_h, cam_w = camera_features_spatial.shape
        if _debug_camera_control:
            print(f"\n[DEBUG] Preparing per-layer camera features:")
            print(f"  camera_features_spatial shape: {camera_features_spatial.shape}")
            print(f"  patchified latents shape (f, h, w): ({f}, {h}, {w})")

        # For width/frame concat modes, camera features need to match concatenated dimensions
        if v2v_concat_mode == "width":
            # Expected width after concatenation: base_width * num_domains
            # If camera features have different width, repeat to match
            expected_w = cam_w * num_domains
            if w == expected_w:
                # Repeat along width dimension to match concatenated width
                # num_domains may be 3 (RGB+depth / RGB+track) or 4 (RGB+track+depth).
                camera_features_spatial = camera_features_spatial.repeat(1, 1, 1, 1, num_domains)
            elif w != cam_w:
                raise ValueError(
                    f"Per-layer camera control: Width mismatch in width-concat mode. "
                    f"Expected w={expected_w} (cam_w={cam_w} * num_domains={num_domains}) but got w={w}. "
                    f"Camera features shape: {camera_features_spatial.shape}, latent shape after patchify: (*, *, {f}, {h}, {w})"
                )
        elif v2v_concat_mode == "frame":
            # Expected frames after concatenation: base_frames * num_domains
            # If camera features have different frames, repeat to match
            expected_f = cam_f * num_domains
            if f == expected_f:
                # Repeat along frame dimension to match concatenated frames
                camera_features_spatial = camera_features_spatial.repeat(1, 1, num_domains, 1, 1)
            elif f != cam_f:
                raise ValueError(
                    f"Per-layer camera control: Frame mismatch in frame-concat mode. "
                    f"Expected f={expected_f} (cam_f={cam_f} * num_domains={num_domains}) but got f={f}. "
                    f"Camera features shape: {camera_features_spatial.shape}, latent shape after patchify: (*, *, {f}, {h}, {w})"
                )

        # Optional: apply camera conditioning only to target domains (keep source domain camera-free).
        # This can help preserve appearance/color consistency from the clean source video.
        # assert camera_targets_only, 'should be more reasonable to have camera_targets_only=True' 
        if camera_targets_only and is_v2v_layout:
            if v2v_concat_mode == "width":
                # domain 0 (source RGB) occupies [:split] along width
                split = int(w // num_domains)
                camera_features_spatial[:, :, :, :, :split] = 0
            else:  # frame
                split = int(f // num_domains)
                camera_features_spatial[:, :, :split, :, :] = 0

        # Store camera scale on blocks so per-block forward can scale *after* projection/encoding
        # (important for direct_raymap_add where cam_encoder has bias; scaling inputs alone won't disable it).
        try:
            for block in getattr(dit, "blocks", []) or []:
                setattr(block, "_camera_condition_scale", camera_condition_scale)
                setattr(block, "_camera_injection_mode", camera_injection_mode)
        except Exception:
            pass

        # For direct_raymap_add/kq_only modes with Conv3d encoder, keep 5D shape (B, C, F, H, W).
        # For projector/rt_projector/vista4d_plucker modes with Linear projection, flatten to 3D.
        if camera_injection_mode in ("direct_raymap_add", "kq_only"):
            # Keep 5D for Conv3d encoder
            camera_features_perlayer = camera_features_spatial
            if _debug_camera_control:
                print(f"  camera_features_perlayer (5D for Conv3d) shape: {camera_features_perlayer.shape}")
                print(f"  camera_features_perlayer mean: {camera_features_perlayer.mean():.6f}, std: {camera_features_perlayer.std():.6f}")
        else:
            # Flatten to 3D for Linear projection
            camera_features_perlayer = rearrange(camera_features_spatial, "b c f h w -> b (f h w) c").contiguous()
        if use_vista4d_latent_streams and camera_features_perlayer is not None:
            camera_features_perlayer = camera_features_perlayer.repeat(1, vista4d_stream_count, 1)
            if _debug_camera_control:
                print(f"  camera_features_perlayer (3D for Linear) shape: {camera_features_perlayer.shape}")
                print(f"  camera_features_perlayer mean: {camera_features_perlayer.mean():.6f}, std: {camera_features_perlayer.std():.6f}")
    # ============================================================================

    # ============================================================================
    # Warped RGB Appearance Conditioning (Lyra-2-inspired, opt-in mode)
    # ============================================================================
    # Inject warped source appearance ONCE before DiT blocks via shallow fusion.
    # Only active when warped_condition_mode == "latent_fuse".
    warped_condition_mode = kwargs.get("warped_condition_mode", "off")
    warped_rgb_latent = kwargs.get("warped_rgb_latent")
    if warped_condition_mode == "latent_fuse" and warped_rgb_latent is not None:
        warped_appearance_confidence = kwargs.get("warped_appearance_confidence")
        fuser = getattr(dit, "warped_appearance_fuser", None)
        if fuser is None:
            raise RuntimeError(
                "warped_condition_mode='latent_fuse' requires dit.warped_appearance_fuser! "
                "Initialize via initialize_shallow_appearance_fuser() before training/inference."
            )
        # x is (B, seq_len, dim) at this point — fuser expects this shape
        x = fuser(
            x,
            warped_rgb_latent.to(device=x.device, dtype=x.dtype),
            confidence=warped_appearance_confidence,
            v2v_concat_mode=v2v_concat_mode,
            num_domains=num_domains,
            grid_f=f, grid_h=h, grid_w=w,
        )
    # ============================================================================

    # AR rolling-forcing temporal RoPE offset: each window's tokens must be rotated
    # at their ABSOLUTE latent-frame position so cached (past-window) K/V and the
    # current window share one continuous time axis. `current_start` is the window's
    # first latent-frame index; it defaults to 0 → `temporal_rope_slice(freqs,0,f)`
    # is byte-identical to the previous `dit.freqs[0][:f]` (non-AR path unchanged).
    from holo4d.ar.rolling_forcing import temporal_rope_slice
    if not is_v2v_layout:
        freqs = torch.cat(
            [
                temporal_rope_slice(dit.freqs[0], current_start, f).view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)
    elif v2v_concat_mode == "frame":
        f_orig = f // num_domains
        freqs = torch.cat(
            [
                temporal_rope_slice(dit.freqs[0], current_start, f_orig).view(f_orig, 1, 1, -1).expand(f_orig, h, w, -1).repeat(num_domains, 1, 1, 1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)
    else:  # width
        w_orig = w // num_domains
        freqs = torch.cat(
            [
                temporal_rope_slice(dit.freqs[0], current_start, f).view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w_orig].view(1, 1, w_orig, -1).expand(f, h, w_orig, -1).repeat(1, 1, num_domains, 1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)
    if use_vista4d_latent_streams:
        freqs = freqs.repeat(vista4d_stream_count, 1, 1)
    if causal_teacher_mode:
        freqs = torch.cat([freqs, freqs], dim=0)

    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if use_unified_sequence_parallel and dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
        chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
        pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
        chunks = [
            torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0)
            for chunk in chunks
        ]
        x = chunks[get_sequence_parallel_rank()]

    if tea_cache_update:
        x = tea_cache.update(x)
    else:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        def create_custom_forward_with_state(block, runtime_state):
            def custom_forward(*inputs):
                return _track4d_call_block_with_runtime_state(
                    block, runtime_state, *inputs
                )

            return custom_forward

        if camera_injection_mode == "add":
            camera_runtime_mode = "camera_add"
        elif camera_injection_mode == "direct_raymap_add":
            camera_runtime_mode = "direct_raymap_add"
        elif camera_injection_mode == "vista4d_plucker":
            camera_runtime_mode = "vista4d_plucker"
        elif camera_injection_mode == "kq_only":
            camera_runtime_mode = "kq_only"
        else:
            camera_runtime_mode = "camera"

        # Optional pre-block token-level injection hook (e.g., Track4D-360 single-projector mode).
        # Backward-compatible: callers that don't pass this kwarg see no change.
        _pre_block_hook = kwargs.get("_pre_block_hook")
        if callable(_pre_block_hook):
            x = _pre_block_hook(x, f=f, h=h, w=w, num_domains=num_domains, v2v_concat_mode=v2v_concat_mode)

        # Track4D-360 per-block injection (ReCamMaster-style: inject after layernorm+modulate).
        _track_align_fn = kwargs.get("_track_align_fn")
        _track_block_injector = kwargs.get("_track_block_injector")
        _track_tokens_aligned = None
        if _track_block_injector is not None:
            if callable(_track_align_fn):
                _track_tokens_aligned = _track_align_fn(
                    x, f=f, h=h, w=w, num_domains=num_domains, v2v_concat_mode=v2v_concat_mode
                )
            else:
                # Conditional-dropout / camera-only-tier / target_cam2world=None paths
                # have no track signal to inject. The track unit upstream returns
                # {} in these cases, so _track_align_fn is None. But we MUST still
                # run the per-block injector encoders on a zero-valued track-tokens
                # tensor — otherwise their weight grads stay None, and DDP hangs at
                # the next gradient sync (with grad_accum, this fires on the last
                # micro-batch of every opt step, so any rank that drew a no-track
                # sample on that micro-batch deadlocks the allreduce). Multiply-by-
                # zero matches the existing fallback at wan_video_track4d360.py:916
                # for the has_tracks=False path: encoder fires on zeros, weights
                # see a zero gradient, parameters are marked "ready" for allreduce,
                # DDP is happy. No find_unused_parameters required.
                _track_dim = int(getattr(_track_block_injector, "track_dim", 0))
                if _track_dim > 0:
                    _track_tokens_aligned = torch.zeros(
                        x.shape[0], x.shape[1], _track_dim,
                        device=x.device, dtype=x.dtype,
                    )
            if (
                _track_tokens_aligned is not None
                and os.environ.get("HOLO4D_DEBUG_TRACK_TOKENS", "0") == "1"
                and not bool(getattr(dit, "_track4d_debug_logged_aligned_tokens", False))
            ):
                setattr(dit, "_track4d_debug_logged_aligned_tokens", True)
                with torch.no_grad():
                    track_abs_mean = float(_track_tokens_aligned.detach().abs().mean().cpu())
                    track_abs_max = float(_track_tokens_aligned.detach().abs().amax().cpu())
                print(
                    "[Track4D-360/debug] aligned_track_tokens "
                    f"shape={tuple(_track_tokens_aligned.shape)} "
                    f"abs_mean={track_abs_mean:.6f} abs_max={track_abs_max:.6f} "
                    f"use_track_injector={_track_block_injector is not None}",
                    flush=True,
                )

        # Determine which forward function to use for each block
        _use_track_injection = (_track_tokens_aligned is not None and _track_block_injector is not None)

        # When the DiT backbone is frozen (gs_only_training / frozen-base),
        # run the entire block loop under torch.no_grad() to skip autograd
        # graph construction for all 30 blocks. The output x will be detached
        # before gs_feature_head and dit.head anyway, so the graph is never
        # used — building it wastes ~10-20% of forward time and stores
        # intermediate activations that are immediately discarded.
        _dit_backbone_frozen = not _dit_hidden_state_has_trainable_sources(
            dit, extra_trainable_sources=(_track_block_injector,)
        )
        _no_grad_ctx = torch.no_grad() if _dit_backbone_frozen else nullcontext()

        with _no_grad_ctx:
            for block_idx, block in enumerate(dit.blocks):
                runtime_state = None
                if _use_track_injection:
                    runtime_state = {
                        "_track4d_runtime_mode": "track",
                        "_track4d_runtime_camera_features": camera_features_perlayer,
                        "_track4d_runtime_track_tokens": _track_tokens_aligned,
                        "_track4d_runtime_track_injector": _track_block_injector,
                        "_track4d_runtime_block_idx": block_idx,
                        "_camera_condition_scale": camera_condition_scale,
                        "_camera_injection_mode": camera_injection_mode,
                    }
                elif use_perlayer_camera_control and camera_features_perlayer is not None:
                    runtime_state = {
                        "_track4d_runtime_mode": camera_runtime_mode,
                        "_track4d_runtime_camera_features": camera_features_perlayer,
                        "_camera_condition_scale": camera_condition_scale,
                        "_camera_injection_mode": camera_injection_mode,
                    }
                elif ar_mode:
                    runtime_state = {
                        "_track4d_runtime_block_idx": block_idx,
                    }
                elif causal_teacher_mode:
                    runtime_state = {
                        "_track4d_runtime_block_idx": block_idx,
                        "_track4d_runtime_causal_attn_mask": causal_teacher_mask,
                    }
                if ar_mode:
                    if runtime_state is None:
                        runtime_state = {"_track4d_runtime_block_idx": block_idx}
                    runtime_state.update(
                        {
                            "_track4d_runtime_kv_cache": kv_cache[block_idx],
                            "_track4d_runtime_updating_cache": bool(updating_cache),
                            "_track4d_runtime_frame_seq_length": int(h) * int(w),
                            "_track4d_runtime_cache_max_latent_frames": ar_cache_max_latent_frames,
                            "_track4d_runtime_cache_keep_first_latent": bool(ar_cache_keep_first_latent),
                        }
                    )

                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        if runtime_state is None:
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x, context, t_mod, freqs, use_reentrant=False,
                            )
                        else:
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward_with_state(block, runtime_state),
                                x, context, t_mod, freqs, use_reentrant=False,
                            )
                elif use_gradient_checkpointing:
                    if runtime_state is None:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, use_reentrant=False,
                        )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_with_state(block, runtime_state),
                            x, context, t_mod, freqs, use_reentrant=False,
                        )
                else:
                    x = _track4d_call_block_with_runtime_state(
                        block, runtime_state, x, context, t_mod, freqs
                    )

        if tea_cache is not None:
            tea_cache.store(x)

    if causal_teacher_mode:
        x = x[:, int(base_token_count):]
        freqs = freqs[int(base_token_count):]

    if use_vista4d_latent_streams:
        x = x[:, :base_token_count]

    # GS feature head: run parallel head on pre-head hidden states if available
    gs_features = None
    if getattr(dit, "gs_feature_head", None) is not None:
        # When the DiT backbone is frozen (no LoRA, no dit_finetune_modules),
        # detach x before feeding to gs_feature_head so gs_loss.backward()
        # does NOT traverse the entire DiT computation graph. Without this,
        # the backward through gsplat + LPIPS (the expensive GS loss path)
        # would backprop through all 1.3B DiT params just to compute gradients
        # for the ~1.5M gs_feature_head weights — a massive waste.
        # When the DiT IS trainable (LoRA or full fine-tune), keep x live so
        # the GS loss can provide 3D-geometry gradients to the DiT backbone.
        gs_input = x.detach() if _dit_backbone_frozen else x
        gs_features = dit.gs_feature_head(gs_input, t)
        if use_unified_sequence_parallel and dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
            gs_features = get_sp_group().all_gather(gs_features, dim=1)
            gs_features = gs_features[:, :-pad_shape] if pad_shape > 0 else gs_features
        gs_features = dit.unpatchify(gs_features, (f, h, w))

    # When the DiT backbone is frozen and GS is the only training target,
    # also detach x before dit.head so the diffusion loss backward does NOT
    # traverse the full DiT graph either. dit.head itself is always frozen
    # (not in dit_finetune_modules or LoRA targets), so no useful gradients
    # are lost. Combined with the gs_feature_head detach above, this cuts
    # BOTH backward paths through the DiT, reducing iteration time from
    # ~2s to <1s for frozen-base GS experiments.
    if gs_features is not None and _dit_backbone_frozen:
        x = x.detach()
    x = dit.head(x, t)
    if use_unified_sequence_parallel and dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)
        x = x[:, :-pad_shape] if pad_shape > 0 else x
    x = dit.unpatchify(x, (f, h, w))

    if gs_features is not None:
        # Also surface the inputs to gs_feature_head so the GS replay
        # cache can store them and re-run the head on every replay step
        # (instead of caching the post-head features and skipping the
        # head entirely, which would leave gs_feature_head's params
        # without gradient and trip DDP). Shapes:
        #   gs_pre_head_input: (B, N_tokens, dim)  — already detached if
        #                      DiT backbone is frozen
        #   gs_pre_head_t:     timestep tensor passed into the head
        #   gs_unpatchify_shape: (f, h, w) latent grid for unpatchify
        return {
            "noise_pred": x,
            "gs_features": gs_features,
            "gs_pre_head_input": gs_input,
            "gs_pre_head_t": t,
            "gs_unpatchify_shape": (int(f), int(h), int(w)),
        }
    return x


class WanVideoPipeline(WanVideoPipelineBase):
    default_training_task = "sft:v2v"

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.model_fn = model_fn_wan_video_v2v
        self.motion_controller = None
        self.vace = None
        self.vace2 = None
        self.vap = None
        self.animate_adapter = None
        self.in_iteration_models = ("dit",)
        self.in_iteration_models_2 = ("dit2",)

        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),  # Encodes text prompt to context
            WanVideoUnit_ModeDropoutAugmentation(),  # ReCamMaster T2V/I2V/V2V mode switching (after prompt encoding)
            WanVideoUnit_InputVideoEmbedder_XYZCondition(),  # Encodes video to latents
            WanVideoUnit_LatentNoiseAugmentation(),  # ReCamMaster-style V2V latent noise augmentation (default 0-500 timesteps)
            WanVideoUnit_CameraControl_NVS(),  # Add camera control for NVS training
            WanVideoUnit_3DModalityVideo(),  # Handles both disp_video and depth_normalized
            WanVideoUnit_PointmapVideo(),
            WanVideoUnit_TrackVideo(),
            WanVideoUnit_TrackVideoSecondary(),
            WanVideoUnit_V2VConcatenate(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        pipe = super().from_pretrained(*args, **kwargs)
        pipe.dit = _convert_dit_to_v2v(getattr(pipe, "dit", None))
        pipe.dit2 = _convert_dit_to_v2v(getattr(pipe, "dit2", None))
        return pipe

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if self.tokenizer is None or self.text_encoder is None:
            raise ValueError("`tokenizer` and `text_encoder` must be initialized (use `from_pretrained`).")
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb.to(dtype=self.torch_dtype, device=self.device)

    @staticmethod
    def _crop_and_resize(frame: Image.Image, target_height: int, target_width: int) -> Image.Image:
        """
        Match training preprocessing (`ImageCropAndResize`):
        resize-to-cover (preserve aspect ratio) + center crop.
        """
        width, height = frame.size
        scale = max(target_width / width, target_height / height)
        new_width, new_height = round(width * scale), round(height * scale)
        resample = getattr(Image, "Resampling", Image).BILINEAR
        frame = frame.resize((new_width, new_height), resample=resample)
        left = max(0, (new_width - target_width) // 2)
        top = max(0, (new_height - target_height) // 2)
        return frame.crop((left, top, left + target_width, top + target_height))

    @staticmethod
    def _v2v_concat(cond_latents: torch.Tensor, target_latents: torch.Tensor, mode: Literal["frame", "width"]) -> torch.Tensor:
        if mode == "frame":
            return torch.cat([cond_latents, target_latents], dim=2)
        return torch.cat([cond_latents, target_latents], dim=4)

    @staticmethod
    def _v2v_target_slice(cond_latents: torch.Tensor, mode: Literal["frame", "width"]):
        if mode == "frame":
            split = int(cond_latents.shape[2])
            return (slice(None), slice(None), slice(split, None), Ellipsis)
        split = int(cond_latents.shape[4])
        return (slice(None), slice(None), slice(None), slice(None), slice(split, None))

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        input_video: Optional[Union[list[Image.Image], torch.Tensor]] = None,
        # depth_normalized: Optional[Union[list[Image.Image], torch.Tensor]] = None,
        *,
        input_image: Optional[Image.Image] = None,
        # Optional RGB+XYZ conditioning (for video->track style tasks)
        use_xyz_video_condition: bool = False,
        xyz_video_cond: Optional[torch.Tensor] = None,
        xyz_video_valid_mask: Optional[torch.Tensor] = None,
        xyz_video_condition_scale: float = 1.0,
        xyz_video_condition_mode: str = "xyz",
        # Optional query-frame impulse conditioning (VDM tracking)
        use_query_frame_impulse_condition: bool = False,
        track_query_idx: Optional[Union[int, torch.Tensor]] = None,
        track_xyz_normalization: str = "minmax",
        track_xyz_quantile_low: float = 0.02,
        track_xyz_quantile_high: float = 0.98,
        # Residual mode flags (used for xyz normalization parity with training)
        predict_track_residual: bool = False,
        residual_zero_center_normalization: bool = False,
        residual_linear_normalization: bool = False,
        # Extra scaling applied to residuals AFTER signed-log normalization (zero-center mode only).
        # This is a linear knob to make residuals occupy more of [-1,1] without changing xyz stats.
        residual_log_scale_factor: float = 1.0,
        # Optional reference for xyz normalization (use GT track when available)
        xyz_video_ref_track: Optional[torch.Tensor] = None,
        xyz_video_ref_valid_mask: Optional[torch.Tensor] = None,
        v2v_concat_mode: Literal["frame", "width", "channel"] = "channel",
        v2v_channel_patch_embed_mode: Literal["add", "cat"] = "add",
        mode_type: Literal["v2v", "i2v", "t2v"] = "v2v",
        denoising_strength: float = 1.0,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: str = "",
        progress_bar_cmd=tqdm,
        return_decoded_tensor: bool = False,
        # Camera control for NVS
        source_camera: Optional[Dict] = None,
        target_camera: Optional[Dict] = None,
        use_perlayer_camera_control: bool = False,
        use_pre_dit_camera_injection: bool = False,
        use_noise_camera_injection: bool = False,
        pre_dit_noise_injection_std: float = 0.0,
        camera_injection_mode: Literal["projector", "add", "direct_raymap_add", "rt_projector", "kq_only", "vista4d_plucker"] = "projector",
        use_domain_embedding: bool = True,
        camera_pose_mode: Literal["packed", "raw"] = "packed",
        camera_pose_frame_stride: int = 4,
        camera_condition_scale: float = 1.0,
        camera_targets_only: bool = False,
        camera_plucker_encoder_mode: Literal["control_adapter", "patchify"] = "patchify",
        use_plucker_raymap_tokens: bool = False,
        # Multi-modality NVS (RGB + Depth)
        multimodality_nvs: bool = False,
        multimodality_tracking: bool = False,
        multimodality_track_depth: bool = False,
        depth_normalized: Optional[torch.Tensor] = None,
        nvs_video: Optional[Union[list[Image.Image], torch.Tensor]] = None,
        track_video: Optional[Union[list[Image.Image], torch.Tensor]] = None,
        # Debug flags
        _debug_camera_control: bool = False,
        debug_onehot_camera: bool = False,
        trajectory_pair_index: Optional[Union[int, torch.Tensor]] = None,
    ) -> Union[list[Image.Image], Dict, tuple[list[Image.Image], torch.Tensor]]:
        """
        V2V inference (e.g., RGB -> depth/disparity or multi-target generation).

        `mode_type` controls which visual conditioning is available (text prompt is always used):
        - "v2v": conditioning comes from full `input_video` (V2V concat path)
        - "i2v": conditioning comes from `input_image` (or first frame of `input_video`) using the official TI2V-style image injection
        - "t2v": no visual conditioning (standard T2V denoising path)

        For single-modality (RGB only):
            - Returns: list[Image.Image] (for backward compatibility)

        For multi-modality (RGB + Depth):
            - Returns: Dict with keys {"rgb": list[Image.Image], "depth": list[Image.Image]}

        Process:
        - Encode conditioning video to latents
        - Denoise target latents (optionally with multi-modality concatenation)
        - Decode and return generated outputs
        """

        mode_type = str(mode_type or "v2v")
        if mode_type not in ("v2v", "i2v", "t2v"):
            raise ValueError(f"Invalid mode_type: {mode_type}. Expected 'v2v', 'i2v', or 't2v'.")
        # Auto-infer mode when callers use the official T2V/I2V calling convention.
        if mode_type == "v2v" and input_video is None:
            mode_type = "i2v" if input_image is not None else "t2v"
        if mode_type == "v2v" and input_video is None:
            raise ValueError("mode_type='v2v' requires `input_video` (conditioning video).")
        if mode_type == "i2v" and input_video is None and input_image is None:
            raise ValueError("I2V inference requires either `input_video` (to take first frame) or `input_image`.")
        if mode_type == "v2v":
            if v2v_concat_mode not in ("frame", "width", "channel"):
                raise ValueError(f"Invalid v2v_concat_mode: {v2v_concat_mode}. Expected 'frame', 'width', or 'channel'.")
        if denoising_strength <= 0 or denoising_strength > 1:
            raise ValueError("`denoising_strength` must be in (0, 1].")
        if mode_type != "v2v" and (use_xyz_video_condition or xyz_video_cond is not None):
            raise ValueError("xyz_video_cond conditioning is only supported in mode_type='v2v'.")

        active_multi_modes = int(bool(multimodality_nvs)) + int(bool(multimodality_tracking)) + int(bool(multimodality_track_depth))
        if active_multi_modes > 1:
            raise ValueError("Only one of multimodality_nvs, multimodality_tracking, or multimodality_track_depth can be enabled.")
        multimodality_any = active_multi_modes > 0
        multimodality_num_domains = 4 if bool(multimodality_track_depth) else (3 if multimodality_any else 2)

        # Validate multi-modality inputs
        if multimodality_nvs:
            if depth_normalized is None:
                raise ValueError("Multi-modality NVS requires `depth_normalized` input (target depth map).")
            if nvs_video is None and depth_normalized is None:
                raise ValueError("Multi-modality NVS requires `nvs_video` or `disp_video` (target RGB video).")
            if v2v_concat_mode == "channel":
                raise ValueError("Multi-modality NVS does not support channel concat mode. Use 'width' or 'frame'.")
        if (multimodality_tracking or multimodality_track_depth) and v2v_concat_mode == "channel":
            raise ValueError("multimodality tracking modes do not support channel concat mode. Use 'width'.")
        if (multimodality_tracking or multimodality_track_depth) and v2v_concat_mode == "frame":
            raise NotImplementedError(
                "multimodality tracking modes with frame-wise concat are not implemented. Use v2v_concat_mode='width'."
            )

        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # ============================================================================
        # UNIT PROCESSING: Process raw inputs through Units (for training/inference consistency)
        # ============================================================================
        # Initialize camera control latents (internal variable, not a user input)
        control_camera_latents_input = None
        control_camera_pose_input = None
        control_camera_plucker_input = None

        # If raw camera dicts are provided, process them through WanVideoUnit_CameraControl_NVS
        # This matches the training flow and eliminates code duplication
        if source_camera is not None and target_camera is not None:
            # Use Unit to process cameras (same as training!)
            camera_unit = WanVideoUnit_CameraControl_NVS()
            camera_outputs = camera_unit.process(
                pipe=self,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                source_camera=source_camera,
                target_camera=target_camera,
                nvs_video=nvs_video,  # None during inference (we're generating it)
                height=height,
                width=width,
                num_frames=num_frames,
                camera_injection_mode=camera_injection_mode,
                camera_pose_mode=camera_pose_mode,
                camera_pose_frame_stride=camera_pose_frame_stride,
            )
            # Extract processed camera control latents
            control_camera_latents_input = camera_outputs.get("control_camera_latents_input")
            control_camera_pose_input = camera_outputs.get("control_camera_pose_input")
            control_camera_plucker_input = camera_outputs.get("control_camera_plucker_input")
            # During training, also get target_rgb_latents if nvs_video was provided
            target_rgb_latents_from_unit = camera_outputs.get("target_rgb_latents")

        if (
            enable_v2v_plucker_camera_control
            and camera_injection_mode == "vista4d_plucker"
            and use_perlayer_camera_control
        ):
            if control_camera_plucker_input is None:
                raise RuntimeError(
                    "vista4d_plucker per-layer camera control requires source_camera and target_camera "
                    "so WanVideoUnit_CameraControl_NVS can build control_camera_plucker_input."
                )
            if not getattr(self, "_logged_vista4d_plucker_input", False):
                print(
                    "[CameraControl] vista4d_plucker "
                    f"shape={tuple(control_camera_plucker_input.shape)} "
                    f"device={control_camera_plucker_input.device} "
                    f"dtype={control_camera_plucker_input.dtype}",
                    flush=True,
                )
                self._logged_vista4d_plucker_input = True

        if camera_injection_mode == "rt_projector" and not use_perlayer_camera_control:
            raise ValueError("camera_injection_mode='rt_projector' requires use_perlayer_camera_control=True (RT conditioning is per-layer only).")
        # ============================================================================

        self.load_models_to_device(["text_encoder"])
        context_posi = self.encode_prompt(prompt)
        context_nega = self.encode_prompt(negative_prompt) if cfg_scale != 1.0 else None

        self.load_models_to_device(["vae"])

        # ---------------------------------------------------------------------
        # T2V / I2V (official TI2V-style image injection)
        # ---------------------------------------------------------------------
        # In these modes we run the base Wan2.2 denoising path (no V2V concat),
        # but still allow NVS camera conditioning via `control_camera_*`.
        if mode_type in ("t2v", "i2v"):
            if multimodality_any:
                raise ValueError("multimodality inference is only supported in mode_type='v2v'.")

            conditioning_image: Optional[Image.Image] = None
            if mode_type == "i2v":
                if isinstance(input_image, Image.Image):
                    conditioning_image = input_image
                elif isinstance(input_video, list) and len(input_video) > 0 and isinstance(input_video[0], Image.Image):
                    conditioning_image = input_video[0]
                else:
                    raise ValueError(
                        "mode_type='i2v' requires `input_image` (PIL) or `input_video` as a non-empty list of PIL frames."
                    )

            # Initialize latent noise exactly like WanVideoUnit_NoiseInitializer.
            latent_length = (int(num_frames) - 1) // 4 + 1
            latent_h = int(height) // int(self.vae.upsampling_factor)
            latent_w = int(width) // int(self.vae.upsampling_factor)
            z_dim = int(self.vae.model.z_dim)
            noise = self.generate_noise(
                (1, z_dim, int(latent_length), int(latent_h), int(latent_w)),
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=torch.float32,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            latents = noise

            clip_feature = None
            y = None
            first_frame_latents = None
            fuse_in_latents = False

            if conditioning_image is not None:
                # Match official TI2V injection: CLIP feature + (optional) VAE embedding `y`
                # and (optional) fused-first-frame injection into `latents`.
                clip_out = WanVideoUnit_ImageEmbedderCLIP().process(
                    self,
                    input_image=conditioning_image,
                    end_image=None,
                    height=height,
                    width=width,
                )
                clip_feature = clip_out.get("clip_feature")

                y_out = WanVideoUnit_ImageEmbedderVAE().process(
                    self,
                    input_image=conditioning_image,
                    end_image=None,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                y = y_out.get("y")

                fused_out = WanVideoUnit_ImageEmbedderFused().process(
                    self,
                    input_image=conditioning_image,
                    latents=latents,
                    noise=noise,
                    height=height,
                    width=width,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                latents = fused_out.get("latents", latents)
                fuse_in_latents = bool(fused_out.get("fuse_vae_embedding_in_latents", False))
                first_frame_latents = fused_out.get("first_frame_latents")

            tea_cache_posi = TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None
            tea_cache_nega = TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None

            self.load_models_to_device(["dit"])
            use_usp = bool(getattr(self, "use_unified_sequence_parallel", False))

            trajectory_pair_index_tensor = None
            if debug_onehot_camera and trajectory_pair_index is not None:
                if isinstance(trajectory_pair_index, torch.Tensor):
                    trajectory_pair_index_tensor = trajectory_pair_index.to(device=self.device, dtype=torch.long)
                else:
                    trajectory_pair_index_tensor = torch.tensor([trajectory_pair_index], dtype=torch.long, device=self.device)

            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

                noise_pred_posi = self.model_fn(
                    dit=self.dit,
                    latents=latents,
                    timestep=timestep,
                    context=context_posi,
                    clip_feature=clip_feature,
                    y=y,
                    tea_cache=tea_cache_posi,
                    use_unified_sequence_parallel=use_usp,
                    sliding_window_size=sliding_window_size,
                    sliding_window_stride=sliding_window_stride,
                    use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False,
                    control_camera_latents_input=control_camera_latents_input,
                    control_camera_pose_input=control_camera_pose_input,
                    control_camera_plucker_input=control_camera_plucker_input,
                    fuse_vae_embedding_in_latents=fuse_in_latents,
                    v2v_concat_mode=None,
                    multimodality_nvs=False,
                    multimodality_tracking=False,
                    debug_onehot_camera=debug_onehot_camera,
                    trajectory_pair_index=trajectory_pair_index_tensor,
                    use_perlayer_camera_control=use_perlayer_camera_control,
                    use_pre_dit_camera_injection=use_pre_dit_camera_injection,
                    use_noise_camera_injection=use_noise_camera_injection,
                    pre_dit_noise_injection_std=pre_dit_noise_injection_std,
                    camera_injection_mode=camera_injection_mode,
                    camera_condition_scale=camera_condition_scale,
                    camera_targets_only=camera_targets_only,
                    camera_plucker_encoder_mode=camera_plucker_encoder_mode,
                    use_plucker_raymap_tokens=use_plucker_raymap_tokens,
                    _debug_camera_control=_debug_camera_control,
                )

                if cfg_scale != 1.0:
                    noise_pred_nega = self.model_fn(
                        dit=self.dit,
                        latents=latents,
                        timestep=timestep,
                        context=context_nega,
                        clip_feature=clip_feature,
                        y=y,
                        tea_cache=tea_cache_nega,
                        use_unified_sequence_parallel=use_usp,
                        sliding_window_size=sliding_window_size,
                        sliding_window_stride=sliding_window_stride,
                        use_gradient_checkpointing=False,
                        use_gradient_checkpointing_offload=False,
                        control_camera_latents_input=control_camera_latents_input,
                        control_camera_pose_input=control_camera_pose_input,
                        control_camera_plucker_input=control_camera_plucker_input,
                        fuse_vae_embedding_in_latents=fuse_in_latents,
                        v2v_concat_mode=None,
                        multimodality_nvs=False,
                        multimodality_tracking=False,
                        debug_onehot_camera=debug_onehot_camera,
                        trajectory_pair_index=trajectory_pair_index_tensor,
                        use_perlayer_camera_control=use_perlayer_camera_control,
                        use_pre_dit_camera_injection=use_pre_dit_camera_injection,
                        use_noise_camera_injection=use_noise_camera_injection,
                        pre_dit_noise_injection_std=pre_dit_noise_injection_std,
                        camera_injection_mode=camera_injection_mode,
                        camera_condition_scale=camera_condition_scale,
                        camera_targets_only=camera_targets_only,
                        camera_plucker_encoder_mode=camera_plucker_encoder_mode,
                        use_plucker_raymap_tokens=use_plucker_raymap_tokens,
                        _debug_camera_control=_debug_camera_control,
                    )
                    noise_pred = noise_pred_nega + float(cfg_scale) * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                if first_frame_latents is not None and fuse_in_latents:
                    latents[:, :, 0:1] = first_frame_latents

            self.load_models_to_device(["vae"])
            decoded_tensor = self.vae.decode(latents, device=self.device, **tiler_kwargs)
            decoded_frames = self.vae_output_to_video(decoded_tensor)
            self.load_models_to_device([])
            if return_decoded_tensor:
                return decoded_frames, decoded_tensor
            return decoded_frames

        # Encode conditioning (source) video / image into latents.
        def _as_bool(v) -> bool:
            if v is None:
                return False
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    raise ValueError(f"Expected scalar tensor for bool flag, got shape {tuple(v.shape)}")
                return bool(v.detach().cpu().item())
            return bool(v)

        def _as_float(v, default: float) -> float:
            if v is None:
                return float(default)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    raise ValueError(f"Expected scalar tensor for float value, got shape {tuple(v.shape)}")
                return float(v.detach().cpu().item())
            return float(v)

        def _coerce_xyz_video(xyz: torch.Tensor, *, batch: int) -> torch.Tensor:
            if xyz.ndim == 4:  # (C,T,H,W)
                xyz_bcthw = xyz.unsqueeze(0)
            elif xyz.ndim == 5:  # (B,C,T,H,W)
                xyz_bcthw = xyz
            else:
                raise ValueError(f"`xyz_video_cond` must be (C,T,H,W) or (B,C,T,H,W), got {tuple(xyz.shape)}.")
            if int(xyz_bcthw.shape[1]) != 3:
                raise ValueError(f"`xyz_video_cond` must have C=3, got shape {tuple(xyz_bcthw.shape)}.")
            if int(xyz_bcthw.shape[0]) == 1 and int(batch) > 1:
                xyz_bcthw = xyz_bcthw.expand(int(batch), -1, -1, -1, -1)
            if int(xyz_bcthw.shape[0]) != int(batch):
                raise ValueError(f"`xyz_video_cond` batch must match input batch={batch}, got {int(xyz_bcthw.shape[0])}.")
            return xyz_bcthw

        if isinstance(input_video, torch.Tensor):
            cond_video = input_video
            if cond_video.ndim == 4:
                cond_video = cond_video.unsqueeze(0)
            if cond_video.ndim != 5:
                raise ValueError(
                    f"`input_video` must be a frame list or a tensor [B,C,T,H,W] / [C,T,H,W], got {tuple(cond_video.shape)}."
                )
            if int(cond_video.shape[2]) != int(num_frames):
                raise ValueError(f"`input_video` tensor T={int(cond_video.shape[2])} does not match num_frames={int(num_frames)}.")
            cond_video = cond_video.to(device=self.device)
        else:
            frames = list(input_video)
            if len(frames) == 0:
                raise ValueError("Got an empty `input_video` (0 frames).")
            if len(frames) < num_frames:
                frames = frames + [frames[-1]] * (num_frames - len(frames))
            elif len(frames) > num_frames:
                frames = frames[:num_frames]
            frames = [self._crop_and_resize(frame, height, width) for frame in frames]
            preprocess_dtype = torch.float32 if _as_bool(use_xyz_video_condition) else self.torch_dtype
            cond_video = self.preprocess_video(frames, torch_dtype=preprocess_dtype, device=self.device)

        cond_video = cond_video.to(dtype=self.torch_dtype, device=self.device)
        input_latents = self.vae.encode(cond_video, device=self.device, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

        control_xyz_latents_input = None
        cond_mode = str(xyz_video_condition_mode or "xyz")
        if _as_bool(use_xyz_video_condition) and cond_mode != "none":
            if xyz_video_cond is None:
                raise ValueError("use_xyz_video_condition=True but `xyz_video_cond` is None")
            if not isinstance(xyz_video_cond, torch.Tensor):
                raise TypeError(f"`xyz_video_cond` must be a torch.Tensor, got {type(xyz_video_cond)}")

            b, c, t, h, w = cond_video.shape
            if c != 3:
                raise ValueError(f"`input_video` must have 3 channels, got shape {tuple(cond_video.shape)}")

            xyz_bcthw = _coerce_xyz_video(xyz_video_cond.to(device=self.device), batch=b).float()
            if xyz_bcthw.shape[2:] != (t, h, w):
                raise ValueError(
                    f"`xyz_video_cond` must match input_video (T,H,W)=({t},{h},{w}), got {tuple(xyz_bcthw.shape[2:])}"
                )

            if cond_mode == "depth":
                # Depth-only conditioning: Z-channel with log normalization (matches training unit)
                depth_z = xyz_bcthw[:, 2:3]  # (B, 1, T, H, W)
                valid_5d = (depth_z > 0).float()
                if xyz_video_valid_mask is not None:
                    vm = xyz_video_valid_mask
                    if vm.ndim == 4:
                        vm = vm.unsqueeze(0).expand(b, -1, -1, -1, -1)
                    elif vm.ndim == 5 and vm.shape[1] != 1:
                        vm = vm[:, :1]
                    valid_5d = valid_5d * vm.float().to(device=self.device)
                depth_log = torch.log1p(depth_z.clamp(min=0))
                for i in range(b):
                    valid_pixels = depth_log[i][valid_5d[i].bool()]
                    if valid_pixels.numel() > 0:
                        d_p02 = torch.quantile(valid_pixels, 0.02)
                        d_p98 = torch.quantile(valid_pixels, 0.98)
                        d_range = d_p98 - d_p02
                        if d_range > 1e-6:
                            depth_log[i] = (depth_log[i] - d_p02) / d_range
                            depth_log[i] = depth_log[i].clamp(-0.5, 1.0) - 0.5
                        else:
                            depth_log[i] = torch.zeros_like(depth_log[i])
                    else:
                        depth_log[i] = torch.zeros_like(depth_log[i])
                depth_log = depth_log * valid_5d
                xyz_norm = depth_log.expand(-1, 3, -1, -1, -1).contiguous()
            else:
                # Default: full XYZ pointmap conditioning
                ref_track_video = None
                ref_track_mask = None
                if isinstance(xyz_video_ref_track, torch.Tensor):
                    if xyz_video_ref_track.ndim == 4:  # (C,T,H,W)
                        ref_track_video = xyz_video_ref_track.unsqueeze(0)
                    elif xyz_video_ref_track.ndim == 5:  # (B,C,T,H,W)
                        ref_track_video = xyz_video_ref_track
                    else:
                        raise ValueError(
                            f"xyz_video_ref_track must be (C,T,H,W) or (B,C,T,H,W), got {tuple(xyz_video_ref_track.shape)}"
                        )
                    if int(ref_track_video.shape[1]) != 3:
                        raise ValueError(f"xyz_video_ref_track must have C=3, got shape {tuple(ref_track_video.shape)}")
                    if int(ref_track_video.shape[0]) == 1 and int(b) > 1:
                        ref_track_video = ref_track_video.expand(int(b), -1, -1, -1, -1)
                    if int(ref_track_video.shape[0]) != int(b):
                        raise ValueError(f"xyz_video_ref_track batch must match input_video batch={b}, got {int(ref_track_video.shape[0])}")
                    if tuple(ref_track_video.shape[2:]) != (t, h, w):
                        raise ValueError(
                            f"xyz_video_ref_track must match input_video (T,H,W)=({t},{h},{w}), got {tuple(ref_track_video.shape[2:])}"
                        )
                    ref_track_video = ref_track_video.to(device=self.device)
                    ref_track_mask = xyz_video_ref_valid_mask

                xyz_norm = WanVideoUnit_TrackVideo._normalize_track_sequencewise_with_reference(
                    xyz_bcthw,
                    xyz_video_valid_mask,
                    reference_track_video=ref_track_video,
                    reference_track_valid_mask=ref_track_mask,
                    track_xyz_normalization=str(track_xyz_normalization or "minmax"),
                    torch_dtype=torch.float32,
                    q_low=track_xyz_quantile_low,
                    q_high=track_xyz_quantile_high,
                    zero_center=_as_bool(predict_track_residual) and _as_bool(residual_zero_center_normalization),
                )

            xyz_norm = xyz_norm.to(dtype=self.torch_dtype, device=self.device)
            scale = _as_float(xyz_video_condition_scale, 1.0)
            control_xyz_latents_input = self.vae.encode(xyz_norm, device=self.device, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            control_xyz_latents_input = control_xyz_latents_input * float(scale)

        # Optional query-frame impulse conditioning (Version A: frame-level binary impulse).
        # Encoded with the same VAE and concatenated to the control latents along channels.
        if _as_bool(use_query_frame_impulse_condition):
            b, c, t, h, w = cond_video.shape

            # Determine query index per sample (default: 0).
            if track_query_idx is None:
                q_idxs = [0] * int(b)
            elif isinstance(track_query_idx, torch.Tensor):
                if track_query_idx.numel() == 1:
                    q = int(track_query_idx.detach().cpu().item())
                    q_idxs = [q] * int(b)
                else:
                    flat = track_query_idx.detach().cpu().view(-1).tolist()
                    if len(flat) != int(b):
                        raise ValueError(
                            f"track_query_idx must be scalar or have batch length {int(b)}, got {tuple(track_query_idx.shape)}"
                        )
                    q_idxs = [int(v) for v in flat]
            elif isinstance(track_query_idx, (list, tuple)):
                if len(track_query_idx) != int(b):
                    raise ValueError(f"track_query_idx must have batch length {int(b)}, got len={len(track_query_idx)}")
                q_idxs = [int(v) for v in track_query_idx]
            else:
                q = int(track_query_idx)
                q_idxs = [q] * int(b)

            impulse_b1thw = torch.zeros((int(b), 1, int(t), int(h), int(w)), device=self.device, dtype=torch.float32)
            for bi, q in enumerate(q_idxs):
                q = int(q)
                if q < 0:
                    continue
                q = max(0, min(q, int(t) - 1))
                impulse_b1thw[bi, 0, q, :, :] = 1.0

            impulse_b3thw = impulse_b1thw.repeat(1, 3, 1, 1, 1).to(dtype=self.torch_dtype, device=self.device)
            impulse_latents = self.vae.encode(impulse_b3thw, device=self.device, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

            if control_xyz_latents_input is None:
                control_xyz_latents_input = impulse_latents
            else:
                control_xyz_latents_input = torch.cat([control_xyz_latents_input, impulse_latents], dim=1)

        # Multi-modality targets (3-domain V2V)
        if multimodality_nvs:
            # Use nvs_video if provided, otherwise fall back to disp_video
            target_rgb_video = nvs_video if nvs_video is not None else depth_normalized

            # Encode target RGB video (use same logic as input_video)
            if isinstance(target_rgb_video, torch.Tensor):
                tgt_video = target_rgb_video
                if tgt_video.ndim == 4:
                    tgt_video = tgt_video.unsqueeze(0)
                if tgt_video.ndim != 5:
                    raise ValueError(f"Target RGB video must be [B,C,T,H,W] / [C,T,H,W], got {tuple(tgt_video.shape)}.")
                if int(tgt_video.shape[2]) != int(num_frames):
                    raise ValueError(f"Target RGB tensor T={int(tgt_video.shape[2])} does not match num_frames={int(num_frames)}.")
                tgt_video = tgt_video.to(dtype=self.torch_dtype, device=self.device)
            else:
                tgt_frames = list(target_rgb_video)
                if len(tgt_frames) == 0:
                    raise ValueError("Got an empty target RGB video (0 frames).")
                if len(tgt_frames) < num_frames:
                    tgt_frames = tgt_frames + [tgt_frames[-1]] * (num_frames - len(tgt_frames))
                elif len(tgt_frames) > num_frames:
                    tgt_frames = tgt_frames[:num_frames]
                tgt_frames = [self._crop_and_resize(frame, height, width) for frame in tgt_frames]
                tgt_video = self.preprocess_video(tgt_frames).to(dtype=self.torch_dtype, device=self.device)

            target_rgb_latents = self.vae.encode(tgt_video, device=self.device, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

            # Encode depth using WanVideoUnit_3DModalityVideo
            depth_unit = WanVideoUnit_3DModalityVideo()
            depth_outputs = depth_unit.process(
                pipe=self,
                # disp_video=None,
                depth_video=depth_normalized,
                **tiler_kwargs
            )
            depth_latents = depth_outputs["depth_latents"]

            # For multi-modality NVS:
            # - cond_latents should be source_RGB (clean condition)
            # - target_latents should be [target_RGB | target_depth] (to be denoised)
            cond_latents = input_latents  # Source RGB (clean)

            # Concatenate target modalities: [target_RGB | target_depth]
            if v2v_concat_mode == "width":
                targets_concat = torch.cat([target_rgb_latents, depth_latents], dim=4)
            elif v2v_concat_mode == "frame":
                targets_concat = torch.cat([target_rgb_latents, depth_latents], dim=2)
            else:
                raise ValueError(f"Multi-modality NVS only supports width/frame concat, got: {v2v_concat_mode}")

            # Generate noise matching the target shape (both RGB and depth)
            noise = self.generate_noise(
                targets_concat.shape,
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=torch.float32,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            target_latents = noise
            target_slice = None  # We'll handle splitting later
        elif multimodality_tracking:
            cond_latents = input_latents
            if v2v_concat_mode == "width":
                targets_shape = (
                    int(cond_latents.shape[0]),
                    int(cond_latents.shape[1]),
                    int(cond_latents.shape[2]),
                    int(cond_latents.shape[3]),
                    int(cond_latents.shape[4]) * 2,
                )
            elif v2v_concat_mode == "frame":
                raise NotImplementedError(
                    "multimodality_tracking with frame-wise concat is not implemented. Use v2v_concat_mode='width'."
                )
            else:
                raise ValueError(f"multimodality_tracking only supports width concat, got: {v2v_concat_mode}")

            # Start both target domains from noise.
            # Layout: [target_track | target_track_video] (second stage concatenation with source is done below).
            target_latents = self.generate_noise(
                targets_shape,
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=torch.float32,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            target_slice = None  # We'll handle splitting later
            _ = track_video  # Reserved for future target initialization / ablations.
        elif multimodality_track_depth:
            cond_latents = input_latents
            if v2v_concat_mode == "width":
                targets_shape = (
                    int(cond_latents.shape[0]),
                    int(cond_latents.shape[1]),
                    int(cond_latents.shape[2]),
                    int(cond_latents.shape[3]),
                    int(cond_latents.shape[4]) * 3,
                )
            else:
                raise ValueError(f"multimodality_track_depth only supports width concat, got: {v2v_concat_mode}")

            target_latents = self.generate_noise(
                targets_shape,
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=torch.float32,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            target_slice = None

        else:
            # Single-modality: use original logic
            cond_latents = input_latents

            noise = self.generate_noise(
                cond_latents.shape,
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=torch.float32,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            if depth_normalized is None:
                target_latents = noise
            else:
                if isinstance(depth_normalized, torch.Tensor):
                    tgt_video = depth_normalized
                    if tgt_video.ndim == 4:
                        tgt_video = tgt_video.unsqueeze(0)
                    if tgt_video.ndim != 5:
                        raise ValueError(f"`disp_video` must be a frame list or a tensor [B,C,T,H,W] / [C,T,H,W], got {tuple(tgt_video.shape)}.")
                    if int(tgt_video.shape[2]) != int(num_frames):
                        raise ValueError(f"`disp_video` tensor T={int(tgt_video.shape[2])} does not match num_frames={int(num_frames)}.")
                    tgt_video = tgt_video.to(dtype=self.torch_dtype, device=self.device)
                else:
                    tgt_frames = list(depth_normalized)
                    if len(tgt_frames) == 0:
                        raise ValueError("Got an empty `disp_video` (0 frames).")
                    if len(tgt_frames) < num_frames:
                        tgt_frames = tgt_frames + [tgt_frames[-1]] * (num_frames - len(tgt_frames))
                    elif len(tgt_frames) > num_frames:
                        tgt_frames = tgt_frames[:num_frames]
                    tgt_frames = [self._crop_and_resize(frame, height, width) for frame in tgt_frames]
                    tgt_video = self.preprocess_video(tgt_frames).to(dtype=self.torch_dtype, device=self.device)

                tgt_latents_init = self.vae.encode(tgt_video, device=self.device, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
                if tgt_latents_init.shape != cond_latents.shape:
                    raise ValueError(
                        f"cond/target latent shapes must match. Got cond_latents={tuple(cond_latents.shape)} vs tgt_latents={tuple(tgt_latents_init.shape)}."
                    )
                target_latents = self.scheduler.add_noise(tgt_latents_init, noise, timestep=self.scheduler.timesteps[0])

            target_slice = None if v2v_concat_mode == "channel" else self._v2v_target_slice(cond_latents, v2v_concat_mode)

        tea_cache_posi = TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None
        tea_cache_nega = TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None

        self.load_models_to_device(["dit"])
        # Ensure domain embedding with correct number of domains (unless disabled for ReCamMaster compatibility)
        if use_domain_embedding:
            num_domains = multimodality_num_domains
            _ensure_domain_embedding(self.dit, num_domains=num_domains)

        if hasattr(self.dit, "set_v2v_channel_patch_embed_mode"):
            self.dit.set_v2v_channel_patch_embed_mode(v2v_channel_patch_embed_mode)
        elif hasattr(self.dit, "v2v_channel_patch_embed_mode"):
            setattr(self.dit, "v2v_channel_patch_embed_mode", v2v_channel_patch_embed_mode)
        use_usp = bool(getattr(self, "use_unified_sequence_parallel", False))

        # Convert trajectory_pair_index to tensor if needed (for debug_onehot_camera mode)
        if debug_onehot_camera and trajectory_pair_index is not None:
            if not isinstance(trajectory_pair_index, torch.Tensor):
                # Single integer for batch_size=1
                trajectory_pair_index = torch.tensor([trajectory_pair_index], dtype=torch.long, device=self.device)
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            if multimodality_any:
                # Multi-modality: cond_latents = source_RGB (clean), target_latents = [target_RGB | target_depth] (noisy)
                # Concatenate to form: [source_RGB | noisy_target_RGB | noisy_target_depth] = 3 domains
                latents_input = self._v2v_concat(cond_latents, target_latents, v2v_concat_mode)
                inputs_shared = {"latents": latents_input}
            elif v2v_concat_mode == "channel":
                inputs_shared = {"latents": target_latents, "cond_latents": cond_latents}
            else:
                latents_input = self._v2v_concat(cond_latents, target_latents, v2v_concat_mode)
                inputs_shared = {"latents": latents_input}

            inputs_posi = {"context": context_posi, "tea_cache": tea_cache_posi}
            inputs_nega = {"context": context_nega, "tea_cache": tea_cache_nega} if context_nega is not None else {}

            noise_pred = self.cfg_guided_model_fn(
                self.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                dit=self.dit,
                timestep=timestep,
                v2v_concat_mode=v2v_concat_mode,
                multimodality_nvs=multimodality_nvs,  # Pass multimodality flag
                multimodality_tracking=multimodality_tracking,
                multimodality_track_depth=multimodality_track_depth,
                cfg_merge=False,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
                use_unified_sequence_parallel=use_usp,
                sliding_window_size=sliding_window_size,
                sliding_window_stride=sliding_window_stride,
                fuse_vae_embedding_in_latents=False,
                control_camera_latents_input=control_camera_latents_input,
                control_camera_pose_input=control_camera_pose_input,
                control_camera_plucker_input=control_camera_plucker_input,
                control_xyz_latents_input=control_xyz_latents_input,
                use_perlayer_camera_control=use_perlayer_camera_control,
                use_pre_dit_camera_injection=use_pre_dit_camera_injection,
                use_noise_camera_injection=use_noise_camera_injection,
                pre_dit_noise_injection_std=pre_dit_noise_injection_std,
                camera_injection_mode=camera_injection_mode,
                use_domain_embedding=use_domain_embedding,
                camera_condition_scale=camera_condition_scale,
                camera_targets_only=camera_targets_only,
                camera_plucker_encoder_mode=camera_plucker_encoder_mode,
                use_plucker_raymap_tokens=use_plucker_raymap_tokens,
                _debug_camera_control=_debug_camera_control and progress_id == 0,  # Only debug first timestep
                debug_onehot_camera=debug_onehot_camera,
                trajectory_pair_index=trajectory_pair_index,
            )

            if multimodality_any:
                # Multi-modality: denoise only the target part [target_RGB | target_depth]
                # noise_pred and latents_input have structure: [source | target_RGB | target_depth]
                # Extract target slice (everything except source domain)
                if v2v_concat_mode == "width":
                    source_width = cond_latents.shape[4]
                    target_slice = (slice(None), slice(None), slice(None), slice(None), slice(source_width, None))
                elif v2v_concat_mode == "frame":
                    if multimodality_tracking or multimodality_track_depth:
                        raise NotImplementedError(
                            "multimodality tracking modes with frame-wise concat are not implemented. Use v2v_concat_mode='width'."
                        )
                    source_frames = cond_latents.shape[2]
                    target_slice = (slice(None), slice(None), slice(source_frames, None), Ellipsis)
                else:
                    raise ValueError(f"Multi-modality only supports width/frame concat, got: {v2v_concat_mode}")

                # Denoise only the target part
                target_latents = self.scheduler.step(
                    noise_pred[target_slice],
                    self.scheduler.timesteps[progress_id],
                    latents_input[target_slice],
                )
            elif v2v_concat_mode == "channel":
                target_latents = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    target_latents,
                )
            else:
                target_latents = self.scheduler.step(
                    noise_pred[target_slice],
                    self.scheduler.timesteps[progress_id],
                    latents_input[target_slice],
                )

        self.load_models_to_device(["vae"])

        # Decode outputs based on modality
        if multimodality_nvs:
            # Split the denoised target latents: [target_rgb | target_depth]
            if v2v_concat_mode == "width":
                split = target_latents.shape[4] // 2
                target_rgb_latents = target_latents[:, :, :, :, :split]
                target_depth_latents = target_latents[:, :, :, :, split:]
            elif v2v_concat_mode == "frame":
                split = target_latents.shape[2] // 2
                target_rgb_latents = target_latents[:, :, :split]
                target_depth_latents = target_latents[:, :, split:]
            else:
                raise ValueError(f"Multi-modality NVS does not support v2v_concat_mode={v2v_concat_mode}")

            # Decode target RGB
            target_rgb_tensor = self.vae.decode(target_rgb_latents, device=self.device, **tiler_kwargs)
            target_rgb_frames = self.vae_output_to_video(target_rgb_tensor)

            # Decode target depth
            target_depth_tensor = self.vae.decode(target_depth_latents, device=self.device, **tiler_kwargs)
            # Average the 3 output channels to get final depth prediction
            # Input to VAE was replicated 3 times, output channels should be similar but may have slight differences
            target_depth_single = target_depth_tensor.mean(dim=1, keepdim=True)  # (B, 1, T, H, W)

            self.load_models_to_device([])

            # Return as dict for multi-modality
            # Note: Return tensors in [-1, 1] range (normalized) for proper visualization in inference script
            output_dict = {
                "rgb": target_rgb_frames,  # List of PIL Images
                "depth_tensor": target_depth_single,  # (B, 1, T, H, W) in [-1, 1] range
            }

            if return_decoded_tensor:
                output_dict["rgb_tensor"] = target_rgb_tensor

            return output_dict
        elif multimodality_tracking:
            # Split denoised target latents: [target_track | target_track_video]
            if v2v_concat_mode != "width":
                raise NotImplementedError(
                    "multimodality_tracking decode supports width-wise concat only."
                )
            split = target_latents.shape[4] // 2
            target_track_latents = target_latents[:, :, :, :, :split]
            target_track_video_latents = target_latents[:, :, :, :, split:]

            target_track_tensor = self.vae.decode(target_track_latents, device=self.device, **tiler_kwargs)
            target_track_frames = self.vae_output_to_video(target_track_tensor)

            target_track_video_tensor = self.vae.decode(target_track_video_latents, device=self.device, **tiler_kwargs)
            target_track_video_frames = self.vae_output_to_video(target_track_video_tensor)

            self.load_models_to_device([])
            output_dict = {
                "track": target_track_frames,
                "track_video": target_track_video_frames,
            }
            if return_decoded_tensor:
                output_dict["track_tensor"] = target_track_tensor
                output_dict["track_video_tensor"] = target_track_video_tensor
            return output_dict

        else:
            # Single-modality: return frames directly (backward compatible)
            decoded_tensor = self.vae.decode(target_latents, device=self.device, **tiler_kwargs)
            decoded_frames = self.vae_output_to_video(decoded_tensor)
            self.load_models_to_device([])

            if return_decoded_tensor:
                return decoded_frames, decoded_tensor
            return decoded_frames


__all__ = [
    "ModelConfig",
    "WanVideoPipeline",
    "WanVideoUnit_3DModalityVideo",
    "WanVideoUnit_PointmapVideo",
    "WanVideoUnit_TrackVideo",
    "WanVideoUnit_InputVideoEmbedder_XYZCondition",
    "WanVideoUnit_V2VConcatenate",
    "WanVideoUnit_CameraControl_NVS",
    "model_fn_wan_video_v2v",
]
