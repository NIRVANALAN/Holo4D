"""
Track4D-360 Pipeline: Wan Video V2V with Point Track Injection.

Extends the standard V2V pipeline to add Koichi-style point-track
conditioning for camera-controlled video generation.

Architecture:
  Source RGB (cam A) + 3D tracks → predict Target RGB (cam B)

  V2V layout: [source_rgb | target_rgb] along width (2 domains)
  Track injection: token-level concat+project BEFORE DiT blocks

Training:
  - FlowMatchV2VLoss on target domain (right half of width)
  - Track modules (embedder, adapter, projector) are trainable
  - DiT LoRA adapts the backbone
"""

from typing import Optional, Literal
import os
import time

import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from .wan_video_v2v import (
    WanVideoPipeline as WanVideoPipelineV2V,
    WanVideoUnit_ShapeChecker,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_ModeDropoutAugmentation,
    WanVideoUnit_InputVideoEmbedder_XYZCondition,
    WanVideoUnit_LatentNoiseAugmentation,
    WanVideoUnit_V2VConcatenate,
    WanVideoUnit_CameraControl_NVS,
    WanVideoUnit_UnifiedSequenceParallel,
    WanVideoUnit_TeaCache,
    WanVideoUnit_CfgMerger,
    _ensure_domain_embedding,
    ModelConfig,
    model_fn_wan_video_v2v,
)
from ..diffusion.base_pipeline import PipelineUnit
from ..utils.geometry import (
    compute_signed_log_quantiles,
    normalize_depth_log_fixed_cap,
    normalize_track_signed_log,
)

from holo4d.conditions.point_track_embedder import PointTrackEmbedder
from holo4d.conditions.point_track_adapter import (
    PointTrackAdapter,
    expand_block_tokens_to_latent_seq,
)
from holo4d.conditions.dense_track_conv_adapter import DenseTrackConvAdapter
from holo4d.conditions.conv3d_track_adapter import Conv3dTrackAdapter
from holo4d.conditions.persistent_trajectory_identity import (
    PersistentTrajectoryIdentityAdapter,
    build_persistent_trajectory_inputs,
    compose_source_identity_tokens,
    compose_target_identity_tokens,
    normalize_persistent_identity_mode,
    persistent_identity_only_enabled,
)
from holo4d.conditions.variable_identity_count import (
    resolve_variable_identity_count,
)
from holo4d.conditions.target_view_track_builder import (
    apply_random_track_point_dropout,
    build_target_view_tracks,
    compact_track_tensors_by_visibility,
    fix_track_identity_count,
    sample_track_identities_query_balanced,
)
from holo4d.conditions.canonical_object_sampler import (
    foreground_at_query_anchors,
    foreground_from_row_aligned_mask,
    gather_identity_payload,
    pad_canonical_object_selection,
    select_canonical_object_identities,
    select_canonical_object_identities_query_aware,
)
from holo4d.data.track_reprojection import project_tracks_to_target_views
from holo4d.conditions.dense_target_reprojection_builder import (
    build_dense_target_reprojection,
)
from holo4d.conditions.dense_target_control_encoder import (
    DenseTargetControlEncoder,
)
from holo4d.conditions.i2v_target_condition import (
    augment_i2v_image_latents,
    build_source_video_condition,
    expand_i2v_condition_to_v2v_domains,
    validate_i2v_condition_config,
)
from holo4d.conditions.ti2v_native_i2v import (
    extract_target_first_frame_latents,
    inject_target_first_frame_latents,
)
# Dense target-space control unit + helpers live in
# holo4d/conditions/dense_target_control_unit.py so they can be imported
# in environments without diffusers/deepspeed/CUDA (the test suite imports
# them directly to avoid triggering this package's heavy __init__.py).
# Re-exported here so production callers keep working unchanged.
from holo4d.conditions.dense_target_control_unit import (
    WanVideoUnit_DenseTargetControl,
    _apply_condition_token_noise,
    _align_track_to_sequence,
    _normalize_warped_depth_per_frame,
    track_occupancy_channel_enabled,
)
from holo4d.conditions.null_token_policy import (
    enforce_frozen_null_tokens_on_pipe,
    migrate_state_for_occupancy_channel,
)
from holo4d.conditions.vista4d_stream_unit import WanVideoUnit_Vista4DStreams
from holo4d.models.token_concat_projector import TokenConcatProjector
from holo4d.utils.torch_profile import track4d_record_function


_TRACK4D_ADAPTER_PROFILE_CALL = 0


def _track4d_env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _track4d_adapter_profile_enabled() -> bool:
    if not _track4d_env_flag("HOLO4D_ADAPTER_MEM_PROFILE", "0"):
        return False
    rank_filter = os.environ.get("HOLO4D_ADAPTER_MEM_PROFILE_RANK", "").strip()
    if rank_filter:
        return os.environ.get("RANK", "?") in {x.strip() for x in rank_filter.split(",") if x.strip()}
    local_filter = os.environ.get("HOLO4D_ADAPTER_MEM_PROFILE_LOCAL_RANK", "").strip()
    if local_filter:
        return os.environ.get("LOCAL_RANK", "?") in {x.strip() for x in local_filter.split(",") if x.strip()}
    return True


def _track4d_tensor_gib(numel: int, element_size: int) -> float:
    return float(numel) * float(element_size) / (1024.0 ** 3)


def _track4d_cuda_mem_fields(device) -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    if _track4d_env_flag("HOLO4D_ADAPTER_MEM_PROFILE_SYNC", "0"):
        torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    return (
        f"cuda_alloc={allocated / (1024 ** 3):.2f}GiB "
        f"cuda_reserved={reserved / (1024 ** 3):.2f}GiB "
        f"cuda_max_reserved={max_reserved / (1024 ** 3):.2f}GiB "
        f"gpu_used={(total - free) / (1024 ** 3):.2f}GiB "
        f"gpu_free={free / (1024 ** 3):.2f}GiB"
    )


def _track4d_profile_meta_str(meta) -> str:
    if not isinstance(meta, dict):
        return ""
    parts = []
    for key in (
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
        "raw_compact_n",
        "fixed_track_n",
        "selected_foreground_counts",
        "selected_background_counts",
        "selected_query_counts",
        "selected_padding_counts",
        "torch_cluster_foreground_raw_returned_counts",
        "torch_cluster_background_raw_returned_counts",
        "selector_wall_time_ms",
        "selector_peak_allocated_bytes",
        "canonical_missing_mask_fallback",
        "canonical_reference_frame",
        "canonical_foreground_fraction",
    ):
        if key in meta:
            parts.append(f"{key}={meta[key]}")
    return " ".join(parts)


def _track4d_log_adapter_profile(
    where: str,
    device,
    *,
    call_id: int,
    adapter_type: str,
    image_h: int,
    image_w: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    track_seq=None,
    track_vis=None,
    track_profile_meta=None,
    adapter=None,
) -> None:
    if not _track4d_adapter_profile_enabled():
        return
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    host = os.uname().nodename
    shape = tuple(track_seq.shape) if torch.is_tensor(track_seq) else None
    B = N = T = C = 0
    track_gib = 0.0
    feature_gib = 0.0
    grid_gib = 0.0
    visible = None
    valid_frac = None
    if torch.is_tensor(track_seq):
        B, N, T, C = [int(x) for x in track_seq.shape]
        elem_size = int(track_seq.element_size())
        track_gib = _track4d_tensor_gib(track_seq.numel(), elem_size)
        proj_dim = int(getattr(adapter, "proj_dim", 0) or 0)
        ps = int(getattr(adapter, "patch_size", 4) or 4)
        if proj_dim > 0:
            feature_gib = _track4d_tensor_gib(B * N * T * proj_dim, elem_size)
            grid_gib = _track4d_tensor_gib(B * T * (latent_h * ps) * (latent_w * ps) * proj_dim, elem_size)
    if torch.is_tensor(track_vis):
        visible = int(track_vis.bool().sum().item())
        valid_frac = float(visible) / max(float(track_vis.numel()), 1.0)
    meta_s = _track4d_profile_meta_str(track_profile_meta)
    print(
        "[track-adapter-profile] "
        f"rank={rank} local_rank={local_rank} host={host} call={call_id} where={where} "
        f"adapter={adapter_type} image={image_h}x{image_w} latent={latent_t}x{latent_h}x{latent_w} "
        f"shape={shape} B={B} N={N} T={T} C={C} visible={visible} valid_frac={valid_frac} "
        f"est_track={track_gib:.3f}GiB est_proj_features={feature_gib:.3f}GiB "
        f"est_splat_grid={grid_gib:.3f}GiB {meta_s} {_track4d_cuda_mem_fields(device)}",
        flush=True,
    )
from holo4d.models.per_block_track_injector import PerBlockTrackInjector


def _is_rank0_process():
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    except Exception:
        return True


def _rank0_print(*args, **kwargs):
    if _is_rank0_process():
        print(*args, **kwargs)


class _DenseOnlyTokenAlignAdapter(torch.nn.Module):
    """Alignment shim for dense-only conditioning.

    It provides the null-token/block-size surface used by
    ``_align_track_to_sequence`` without constructing a persistent 3D-track
    encoder. Dense-only ablations still need this source-domain padding token
    so target-space dense tokens align with the full V2V token sequence.
    """

    def __init__(self, out_dim: int, block_t: int, block_h: int, block_w: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.block_t = int(block_t)
        self.block_h = int(block_h)
        self.block_w = int(block_w)
        self.null_token = torch.nn.Parameter(torch.zeros(1, 1, self.out_dim))


def _prepare_gs_feedback_rgb_for_vae(
    rendered_rgb: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert GS-rendered RGB from (B,T,3,H,W) to VAE input layout/dtype."""
    return rendered_rgb.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)


def _coerce_tracks_world_bnt3(tracks_world: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert tracks to (B, N, T, 3) float32 plus valid mask."""
    if not isinstance(tracks_world, torch.Tensor):
        import numpy as np
        if isinstance(tracks_world, np.ndarray):
            tracks_world = torch.from_numpy(tracks_world)
        else:
            raise TypeError(f"tracks_world must be a torch.Tensor, got {type(tracks_world)}")
    tw = tracks_world.to(device=device, dtype=torch.float32)
    if tw.ndim == 3:
        tw = tw.unsqueeze(0)
    if tw.ndim != 4 or tw.shape[-1] != 3:
        raise ValueError(f"tracks_world must be (N,T,3) or (B,N,T,3), got {tuple(tw.shape)}")
    valid = torch.any(torch.abs(tw) > 0, dim=-1)
    return tw, valid


def _compute_source_track_signed_log_stats(
    tracks_world: torch.Tensor,
    device: torch.device,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> dict:
    """Compute source-track signed-log stats for target pointmap normalization."""
    tw_bnt3, valid_bnt = _coerce_tracks_world_bnt3(tracks_world, device=device)
    b = tw_bnt3.shape[0]
    ql = torch.zeros((b, 3), device=device, dtype=torch.float32)
    qh = torch.ones((b, 3), device=device, dtype=torch.float32)
    for bi in range(b):
        ql_b, qh_b = compute_signed_log_quantiles(
            tw_bnt3[bi],
            valid_mask=valid_bnt[bi],
            q_low=q_low,
            q_high=q_high,
        )
        ql[bi] = ql_b.to(device=device, dtype=torch.float32)
        qh[bi] = qh_b.to(device=device, dtype=torch.float32)
    return {
        "norm_type": "signed_log_p2_p98",
        "ql": ql,
        "qh": qh,
        "q_low": float(q_low),
        "q_high": float(q_high),
    }


def _compute_projected_depth_log_stats(
    tracks_world: torch.Tensor,
    target_cam2world: torch.Tensor,
    device: torch.device,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> dict:
    """Project source tracks into the target cameras and use their positive depths as log-depth stats."""
    tw_bnt3, valid_bnt = _coerce_tracks_world_bnt3(tracks_world, device=device)
    if not isinstance(target_cam2world, torch.Tensor):
        import numpy as np
        if isinstance(target_cam2world, np.ndarray):
            target_cam2world = torch.from_numpy(target_cam2world)
        else:
            raise TypeError(f"target_cam2world must be a torch.Tensor, got {type(target_cam2world)}")
    c2w = target_cam2world.to(device=device, dtype=torch.float32)
    if c2w.ndim == 3:
        c2w = c2w.unsqueeze(0)
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"target_cam2world must be (T,4,4) or (B,T,4,4), got {tuple(c2w.shape)}")
    if c2w.shape[0] == 1 and tw_bnt3.shape[0] > 1:
        c2w = c2w.expand(tw_bnt3.shape[0], -1, -1, -1)
    if c2w.shape[0] != tw_bnt3.shape[0]:
        raise ValueError(f"Batch mismatch between tracks_world {tuple(tw_bnt3.shape)} and target_cam2world {tuple(c2w.shape)}")

    b, _n, t, _ = tw_bnt3.shape
    ql = torch.zeros((b,), device=device, dtype=torch.float32)
    qh = torch.ones((b,), device=device, dtype=torch.float32)
    for bi in range(b):
        w2c = torch.linalg.inv(c2w[bi])  # (T,4,4)
        pts_world = tw_bnt3[bi]  # (N,T,3)
        rot = w2c[:, :3, :3]  # (T,3,3)
        trans = w2c[:, :3, 3]  # (T,3)
        pts_cam = torch.einsum("tij,ntj->nti", rot, pts_world) + trans.unsqueeze(0)
        depth_vals = pts_cam[..., 2]
        valid = valid_bnt[bi] & torch.isfinite(depth_vals) & (depth_vals > 1e-8)
        if bool(valid.any()):
            log_vals = torch.log(depth_vals[valid].clamp_min(1e-8))
            ql_b = torch.quantile(log_vals, q_low)
            qh_b = torch.quantile(log_vals, q_high)
        else:
            ql_b = torch.zeros((), device=device, dtype=torch.float32)
            qh_b = torch.ones((), device=device, dtype=torch.float32)
        ql[bi] = ql_b.to(device=device, dtype=torch.float32)
        qh[bi] = qh_b.to(device=device, dtype=torch.float32)
    return {
        "norm_type": "log_depth_q",
        "ql": ql,
        "qh": qh,
        "q_low": float(q_low),
        "q_high": float(q_high),
    }


# ---------------------------------------------------------------------------
# Pipeline Units
# ---------------------------------------------------------------------------

class WanVideoUnit_TargetVideoEncoder(PipelineUnit):
    """Encode target-view RGB video via VAE → target_rgb_latents.

    During inference (target_video=None), creates random noise latents matching
    the source input_latents shape so V2V concat can proceed.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "target_video",
                "input_latents",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("target_rgb_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe, target_video=None, input_latents=None, tiled=False,
                tile_size=None, tile_stride=None):
        if target_video is None:
            # Inference mode: target is noise with same shape as source latents
            if input_latents is not None:
                return {"target_rgb_latents": torch.randn_like(input_latents)}
            return {}

        tile_size = tile_size or (30, 52)
        tile_stride = tile_stride or (15, 26)

        # target_video is a list of PIL Images (same format as input_video)
        # Explicit device/dtype to match WanVideoUnit_InputVideoEmbedder_XYZCondition
        # (wan_video_v2v.py:1352-1364). Without this, preprocess_video returns a
        # CPU tensor, the VAE encode output stays on CPU, and width-concat later
        # fails with "Expected all tensors to be on the same device".
        video_tensor = pipe.preprocess_video(
            target_video, torch_dtype=torch.float32, device=pipe.device,
        )  # (1, C, T, H, W)
        video_tensor = video_tensor.to(dtype=pipe.torch_dtype, device=pipe.device)
        target_rgb_latents = pipe.vae.encode(
            video_tensor,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"target_rgb_latents": target_rgb_latents}


class WanVideoUnit_TargetImageCondition(PipelineUnit):
    """Track4D V2V source condition plus target-boundary Wan-I2V condition.

    During training the image comes from ``target_video[0]``.  During
    autoregressive inference the caller passes the preceding generated chunk's
    final frame as ``input_image``.  The resulting 20-channel tensor keeps the
    current source chunk fully observed in source domain 0 and exposes only the
    carried boundary frame in target RGB domain 1.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "use_target_image_condition",
                "input_image",
                "target_video",
                "input_latents",
                "mode_type",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "v2v_concat_mode",
                "multimodality_nvs",
                "multimodality_tracking",
                "multimodality_track_depth",
                "i2v_condition_drop_prob",
                "i2v_condition_noise_level",
            ),
            output_params=("y", "clip_feature"),
            onload_model_names=("vae", "image_encoder"),
        )

    @staticmethod
    def _condition_image(input_image, target_video):
        if input_image is not None:
            image = input_image
        elif target_video:
            image = target_video[0]
        else:
            return None
        if not isinstance(image, Image.Image):
            raise TypeError(
                "Target I2V condition must be a PIL.Image. During training it is "
                "taken from target_video[0]; during AR inference pass input_image. "
                f"Got {type(image).__name__}."
            )
        return image

    def process(
        self,
        pipe,
        use_target_image_condition=False,
        input_image=None,
        target_video=None,
        input_latents=None,
        mode_type="v2v",
        num_frames=49,
        height=384,
        width=672,
        tiled=False,
        tile_size=None,
        tile_stride=None,
        v2v_concat_mode="width",
        multimodality_nvs=False,
        multimodality_tracking=False,
        multimodality_track_depth=False,
        i2v_condition_drop_prob=0.3,
        i2v_condition_noise_level=0.05,
    ):
        if not bool(use_target_image_condition):
            return {}
        validate_i2v_condition_config(
            float(i2v_condition_drop_prob), float(i2v_condition_noise_level)
        )
        if input_latents is None:
            raise ValueError(
                "Target I2V conditioning must run after V2V concatenation; input_latents is missing."
            )
        if v2v_concat_mode != "width":
            raise NotImplementedError(
                "Target I2V conditioning currently supports Track4D width-concat only."
            )
        if not bool(getattr(pipe.dit, "require_vae_embedding", False)):
            raise ValueError(
                "Target I2V conditioning requires a Wan I2V DiT with VAE image embedding."
            )
        if getattr(pipe, "image_encoder", None) is None:
            raise ValueError(
                "Target I2V conditioning requires the Wan CLIP image encoder weights."
            )

        is_training = bool(getattr(pipe.scheduler, "training", False))
        drop_target_image = False
        if is_training and float(i2v_condition_drop_prob) > 0.0:
            if torch.rand((), device="cpu").item() < float(i2v_condition_drop_prob):
                # Drop only the target boundary-image condition.  The current
                # source chunk remains a full V2V condition in source domain 0.
                drop_target_image = True

        image = self._condition_image(input_image, target_video)
        if image is None and not drop_target_image:
            raise ValueError(
                "Target I2V conditioning is enabled but neither input_image nor target_video was provided."
            )

        height = int(height)
        width = int(width)
        num_frames = int(num_frames)
        tile_size = tile_size or (30, 52)
        tile_stride = tile_stride or (15, 26)

        num_domains = 4 if bool(multimodality_track_depth) else (
            3 if (bool(multimodality_nvs) or bool(multimodality_tracking)) else 2
        )
        if int(input_latents.shape[-1]) % int(num_domains) != 0:
            raise ValueError(
                f"full latent width {int(input_latents.shape[-1])} is not divisible by "
                f"num_domains={num_domains}"
            )
        domain_width = int(input_latents.shape[-1]) // int(num_domains)
        source_latents = input_latents[..., :domain_width]
        source_condition = None
        if str(mode_type or "v2v") == "v2v":
            source_condition = build_source_video_condition(
                source_latents,
                condition_channels=20,
            )

        # Vista4D's 30% image dropout applies to the carried target image and
        # its CLIP feature, not to the source V2V video condition.
        if drop_target_image:
            target_condition = source_latents.new_zeros(
                int(source_latents.shape[0]),
                20,
                int(source_latents.shape[2]),
                int(source_latents.shape[3]),
                int(source_latents.shape[4]),
            )
            y = expand_i2v_condition_to_v2v_domains(
                target_condition,
                input_latents,
                source_condition=source_condition,
                num_domains=num_domains,
                source_domain=0,
                target_domain=1,
                concat_mode=v2v_concat_mode,
            )
            return {"y": y}

        pipe.load_models_to_device(self.onload_model_names)

        resized = image.resize((width, height))
        image_tensor = pipe.preprocess_image(resized).to(pipe.device)

        clip_feature = pipe.image_encoder.encode_image([image_tensor])
        clip_feature = clip_feature.to(dtype=pipe.torch_dtype, device=pipe.device)

        vae_input = torch.cat(
            [
                image_tensor.transpose(0, 1),
                torch.zeros(
                    3,
                    num_frames - 1,
                    height,
                    width,
                    device=image_tensor.device,
                    dtype=image_tensor.dtype,
                ),
            ],
            dim=1,
        )
        image_latents = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        if is_training:
            image_latents = augment_i2v_image_latents(
                image_latents, float(i2v_condition_noise_level)
            )

        mask = torch.ones(
            1, num_frames, height // 8, width // 8,
            device=pipe.device, dtype=pipe.torch_dtype,
        )
        mask[:, 1:] = 0
        mask = torch.cat(
            [torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]],
            dim=1,
        )
        mask = mask.view(1, mask.shape[1] // 4, 4, height // 8, width // 8)
        mask = mask.transpose(1, 2)[0]
        single_domain_y = torch.cat([mask, image_latents], dim=0).unsqueeze(0)

        y = expand_i2v_condition_to_v2v_domains(
            single_domain_y,
            input_latents,
            source_condition=source_condition,
            num_domains=num_domains,
            source_domain=0,
            target_domain=1,
            concat_mode=v2v_concat_mode,
        )
        return {"y": y, "clip_feature": clip_feature}


class WanVideoUnit_TI2VNativeFirstFrameCondition(PipelineUnit):
    """Build Wan 2.2's clean target-frame-0 latent condition.

    Training reuses frame 0 from the already encoded target video, preserving
    the exact causal-VAE encoding used by the diffusion target. Inference
    encodes the explicitly supplied boundary image as a one-frame video.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "use_ti2v_native_i2v",
                "input_image",
                "target_video",
                "input_latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "v2v_concat_mode",
                "multimodality_nvs",
                "multimodality_tracking",
                "multimodality_track_depth",
                "channel_concat",
                "raw_t2v_target_only",
                "i2v_condition_drop_prob",
                "i2v_condition_noise_level",
            ),
            output_params=(
                "first_frame_latents",
                "ti2v_native_condition_active",
                "fuse_vae_embedding_in_latents",
                "ti2v_boundary_domains",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        use_ti2v_native_i2v=False,
        input_image=None,
        target_video=None,
        input_latents=None,
        height=384,
        width=672,
        tiled=False,
        tile_size=None,
        tile_stride=None,
        v2v_concat_mode="width",
        multimodality_nvs=False,
        multimodality_tracking=False,
        multimodality_track_depth=False,
        channel_concat=False,
        raw_t2v_target_only=False,
        i2v_condition_drop_prob=0.3,
        i2v_condition_noise_level=0.05,
    ):
        if not bool(use_ti2v_native_i2v):
            return {}
        if v2v_concat_mode != "width":
            raise ValueError("TI2V-native I2V requires width-concat V2V")
        if bool(channel_concat) or bool(raw_t2v_target_only):
            raise ValueError(
                "TI2V-native I2V requires an explicit width-concatenated source "
                "and target RGB layout"
            )
        if bool(multimodality_tracking):
            raise ValueError(
                "TI2V-native I2V does not support multimodality_tracking because "
                "that layout has no target RGB domain"
            )
        if input_latents is None:
            raise ValueError(
                "TI2V-native I2V must run after V2V concatenation; input_latents is missing"
            )
        if not bool(getattr(pipe.dit, "fuse_vae_embedding_in_latents", False)):
            raise ValueError(
                "TI2V-native I2V requires fuse_vae_embedding_in_latents=True"
            )
        if not bool(getattr(pipe.dit, "seperated_timestep", False)):
            raise ValueError("TI2V-native I2V requires seperated_timestep=True")
        validate_i2v_condition_config(
            float(i2v_condition_drop_prob), float(i2v_condition_noise_level)
        )

        is_training = bool(getattr(pipe.scheduler, "training", False))
        if is_training and float(i2v_condition_drop_prob) > 0.0:
            if torch.rand((), device="cpu").item() < float(i2v_condition_drop_prob):
                return {
                    "ti2v_native_condition_active": False,
                    "fuse_vae_embedding_in_latents": False,
                    "ti2v_boundary_domains": 0,
                }

        num_domains = 4 if bool(multimodality_track_depth) else (
            3 if (bool(multimodality_nvs) or bool(multimodality_tracking)) else 2
        )
        if is_training and target_video is not None:
            first_frame_latents = extract_target_first_frame_latents(
                input_latents, num_domains=num_domains, target_domain=1
            )
        else:
            image = WanVideoUnit_TargetImageCondition._condition_image(
                input_image, target_video
            )
            if image is None:
                raise ValueError(
                    "TI2V-native I2V inference requires input_image (or target_video[0])"
                )
            pipe.load_models_to_device(self.onload_model_names)
            tile_size = tile_size or (30, 52)
            tile_stride = tile_stride or (15, 26)
            image = image.resize((int(width), int(height)))
            # Match WanVideoUnit_ImageEmbedderFused exactly rather than routing
            # a one-frame image through the general video preprocessor.
            image_tensor = pipe.preprocess_image(image).transpose(0, 1)
            first_frame_latents = pipe.vae.encode(
                [image_tensor.to(dtype=pipe.torch_dtype, device=pipe.device)],
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=input_latents.dtype, device=input_latents.device)
            if int(first_frame_latents.shape[2]) != 1:
                first_frame_latents = first_frame_latents[:, :, 0:1]

        if is_training:
            first_frame_latents = augment_i2v_image_latents(
                first_frame_latents, float(i2v_condition_noise_level)
            )
        # Width-concat boundary marking. Historical behavior (env unset / "0")
        # marks ALL of latent frame 0 as t=0, including the source half -- which
        # leaks source appearance into latent frame 1. Setting
        # HOLO4D_TI2V_TARGET_ONLY_T0=1 restricts the marking to the boundary
        # (target-RGB) domain, the only domain actually clamped clean. Kept
        # opt-in so already-evaluated checkpoints keep reproducing bit-for-bit
        # until the A/B measurement lands. See
        # doc/track4d-360/bugs/2026-08-23-ti2v-native-i2v-latent-frame1-artifact.md
        _target_only_t0 = os.environ.get("HOLO4D_TI2V_TARGET_ONLY_T0", "0") == "1"
        return {
            "first_frame_latents": first_frame_latents,
            "ti2v_native_condition_active": True,
            # Activates Wan 2.2's separated-timestep branch: latent frame 0
            # receives timestep zero while generated frames use diffusion t.
            "fuse_vae_embedding_in_latents": True,
            "ti2v_boundary_domains": int(num_domains) if _target_only_t0 else 0,
        }


class WanVideoUnit_TargetTrackEncoder(PipelineUnit):
    """Encode target-view dense 3D position maps via VAE → target_track_latents.

    Stage 2: joint NVS + 3D tracking. The target-view 3D positions (cam0-normalized,
    common_scale applied) are converted to a video-like tensor via signed_log
    normalization, then VAE-encoded to latent space alongside the target RGB.

    During inference (target_track_video=None), creates random noise latents
    matching the source input_latents shape.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "target_track_video",
                "target_track_valid_mask",
                "tracks_world",
                "input_latents",
                "tiled",
                "tile_size",
                "tile_stride",
                "target_track_use_source_stats",
            ),
            output_params=("track_video_latents", "target_track_normalization_stats"),
            onload_model_names=("vae",),
        )

    @staticmethod
    def _normalize_with_source_stats(track_video, valid_mask, stats):
        out = torch.zeros_like(track_video, dtype=torch.float32)
        for bi in range(track_video.shape[0]):
            track_thwc = track_video[bi].float().permute(1, 2, 3, 0).contiguous()
            valid_thw = None if valid_mask is None else valid_mask[bi]
            norm_thwc, _ = normalize_track_signed_log(
                track_thwc,
                valid_mask=valid_thw,
                ql=stats["ql"][bi],
                qh=stats["qh"][bi],
                q_low=float(stats.get("q_low", 0.02)),
                q_high=float(stats.get("q_high", 0.98)),
            )
            out[bi] = norm_thwc.permute(3, 0, 1, 2).contiguous()
        return out

    def process(self, pipe, target_track_video=None, target_track_valid_mask=None,
                tracks_world=None, input_latents=None, tiled=False, tile_size=None,
                tile_stride=None, target_track_use_source_stats=False):
        source_stats = None
        if bool(target_track_use_source_stats) and tracks_world is not None:
            source_stats = _compute_source_track_signed_log_stats(
                tracks_world,
                device=pipe.device,
            )
        if target_track_video is None:
            # Inference: target track domain is noise
            if input_latents is not None:
                return {
                    "track_video_latents": torch.randn_like(input_latents),
                    "target_track_normalization_stats": source_stats,
                }
            return {}

        tile_size = tile_size or (30, 52)
        tile_stride = tile_stride or (15, 26)

        # target_track_video: (T, H, W, 3) numpy or tensor → (1, 3, T, H, W)
        if isinstance(target_track_video, torch.Tensor):
            ttv = target_track_video.float()
        else:
            import numpy as np
            ttv = torch.from_numpy(np.asarray(target_track_video)).float()

        if ttv.ndim == 4:  # (T, H, W, 3)
            ttv = ttv.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
        elif ttv.ndim == 5 and ttv.shape[-1] == 3:  # (B, T, H, W, 3)
            ttv = ttv.permute(0, 4, 1, 2, 3)  # (B, 3, T, H, W)

        ttv = ttv.to(device=pipe.device)

        # Handle valid mask
        tvm = target_track_valid_mask
        if tvm is not None:
            if isinstance(tvm, torch.Tensor):
                tvm = tvm.to(device=pipe.device, dtype=torch.bool)
            else:
                tvm = torch.from_numpy(np.asarray(tvm)).to(device=pipe.device, dtype=torch.bool)
            if tvm.ndim == 3:
                tvm = tvm.unsqueeze(0)

        if source_stats is not None:
            ttv_norm = self._normalize_with_source_stats(ttv, tvm, source_stats)
        else:
            # Legacy Stage-2 path: target self-normalization.
            from .wan_video_v2v import WanVideoUnit_TrackVideo
            ttv_norm = WanVideoUnit_TrackVideo._normalize_track_sequencewise_signed_log_p2_p98(
                ttv, tvm,
                torch_dtype=torch.float32,
            )
        ttv_norm = ttv_norm.to(dtype=pipe.torch_dtype)

        # VAE encode
        target_track_latents = pipe.vae.encode(
            ttv_norm,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return {
            "track_video_latents": target_track_latents,
            "target_track_normalization_stats": source_stats,
        }


class WanVideoUnit_TargetDepthEncoder(PipelineUnit):
    """Encode target-view depth maps using source-projected log-depth stats."""

    def __init__(self):
        super().__init__(
            input_params=(
                "target_depth_video",
                "target_depth_valid_mask",
                "tracks_world",
                "target_cam2world",
                "input_latents",
                "tiled",
                "tile_size",
                "tile_stride",
                "target_depth_log_cap",
                "target_depth_norm_type",
                "target_depth_max_disparity",
            ),
            output_params=("depth_latents", "target_depth_normalization_stats"),
            onload_model_names=("vae",),
        )

    @staticmethod
    def _normalize_depth_with_fixed_cap(depth_video, valid_mask, max_depth):
        b = depth_video.shape[0]
        out = torch.zeros_like(depth_video, dtype=torch.float32)
        for bi in range(b):
            norm_b, _ = normalize_depth_log_fixed_cap(
                depth_video[bi],
                valid_mask=None if valid_mask is None else valid_mask[bi],
                max_depth=float(max_depth),
            )
            out[bi] = norm_b
        return out

    @staticmethod
    def _normalize_depth_as_disparity(depth_video, valid_mask, max_disparity):
        from holo4d_runtime.utils.geometry import normalize_disparity_fixed_cap
        b = depth_video.shape[0]
        out = torch.zeros_like(depth_video, dtype=torch.float32)
        for bi in range(b):
            norm_b, _ = normalize_disparity_fixed_cap(
                depth_video[bi],
                valid_mask=None if valid_mask is None else valid_mask[bi],
                max_disparity=float(max_disparity),
                invalid_value=-1.0,
            )
            out[bi] = norm_b
        return out

    def process(
        self,
        pipe,
        target_depth_video=None,
        target_depth_valid_mask=None,
        tracks_world=None,
        target_cam2world=None,
        input_latents=None,
        tiled=False,
        tile_size=None,
        tile_stride=None,
        target_depth_log_cap=14.0,
        target_depth_norm_type="log_depth",
        target_depth_max_disparity=5.0,
    ):
        norm_type = str(target_depth_norm_type)
        if norm_type == "disparity":
            depth_stats = {
                "norm_type": "disparity_fixed_cap",
                "max_disparity": float(target_depth_max_disparity),
            }
        else:
            depth_stats = {
                "norm_type": "log_depth_fixed_cap",
                "max_depth": float(target_depth_log_cap),
            }

        if target_depth_video is None:
            if input_latents is not None:
                # No depth GT → zeros (loss masked by depth_loss_mask=0.0)
                return {
                    "depth_latents": torch.zeros_like(input_latents),
                    "target_depth_normalization_stats": depth_stats,
                }
            return {}

        tile_size = tile_size or (30, 52)
        tile_stride = tile_stride or (15, 26)

        if isinstance(target_depth_video, torch.Tensor):
            tdv = target_depth_video.float()
        else:
            import numpy as np
            tdv = torch.from_numpy(np.asarray(target_depth_video)).float()

        if tdv.ndim == 3:  # (T,H,W)
            tdv = tdv.unsqueeze(0)
        elif tdv.ndim == 5 and tdv.shape[1] == 1:  # (B,1,T,H,W)
            tdv = tdv[:, 0]
        elif tdv.ndim != 4:
            raise ValueError(f"target_depth_video must be (T,H,W), (B,T,H,W), or (B,1,T,H,W); got {tuple(tdv.shape)}")
        tdv = tdv.to(device=pipe.device)

        dvm = target_depth_valid_mask
        if dvm is not None:
            if isinstance(dvm, torch.Tensor):
                dvm = dvm.to(device=pipe.device, dtype=torch.bool)
            else:
                import numpy as np
                dvm = torch.from_numpy(np.asarray(dvm)).to(device=pipe.device, dtype=torch.bool)
            if dvm.ndim == 3:
                dvm = dvm.unsqueeze(0)

        if norm_type == "disparity":
            tdv_norm = self._normalize_depth_as_disparity(tdv, dvm, target_depth_max_disparity)
        else:
            tdv_norm = self._normalize_depth_with_fixed_cap(tdv, dvm, target_depth_log_cap)
        tdv_norm = tdv_norm.unsqueeze(1).repeat(1, 3, 1, 1, 1).to(dtype=pipe.torch_dtype)

        depth_latents = pipe.vae.encode(
            tdv_norm,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return {
            "depth_latents": depth_latents,
            "target_depth_normalization_stats": depth_stats,
        }


class WanVideoUnit_JointTrackRemap(PipelineUnit):
    """Remap target_rgb_latents → track_latents for 3-domain V2V concat.

    The V2V concat's multimodality_tracking path expects:
      - track_latents (domain 2)
      - track_video_latents (domain 3)
    But Track4D-360's TargetVideoEncoder outputs target_rgb_latents.
    This unit bridges the naming when joint_track_prediction is enabled.
    """

    def __init__(self):
        super().__init__(
            input_params=("target_rgb_latents",),
            output_params=("track_latents",),
        )

    def process(self, pipe, target_rgb_latents=None):
        if target_rgb_latents is None:
            return {}
        return {"track_latents": target_rgb_latents}


# =========================================================================
# Channel-concat multi-modal V2V units (ported from DiffSynth-Studio)
# =========================================================================


class WanVideoUnit_SourceMultiModalEncoder(PipelineUnit):
    """Encode source depth (and optionally residual tracks) for channel-concat V2V."""

    def __init__(self):
        super().__init__(
            input_params=("source_depth_video", "tracks_world", "input_latents",
                          "tiled", "tile_size", "tile_stride",
                          "channel_concat", "channel_modalities",
                          "target_depth_log_cap", "target_depth_norm_type",
                          "target_depth_max_disparity", "track_query_idx",
                          "target_track_use_source_stats"),
            output_params=("source_depth_latents", "source_track_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, source_depth_video=None, tracks_world=None,
                input_latents=None, tiled=False, tile_size=None, tile_stride=None,
                channel_concat=False, channel_modalities=None,
                target_depth_log_cap=14.0, target_depth_norm_type="log_depth",
                target_depth_max_disparity=5.0, track_query_idx=0,
                target_track_use_source_stats=False):
        if not channel_concat:
            return {}
        tile_size = tile_size or (30, 52)
        tile_stride = tile_stride or (15, 26)
        result = {}
        if channel_modalities and "depth" in channel_modalities:
            if source_depth_video is not None:
                import numpy as np
                sdv = torch.from_numpy(source_depth_video).float() if isinstance(source_depth_video, np.ndarray) else source_depth_video.float()
                if sdv.ndim == 3:  # (T,H,W)
                    sdv = sdv.unsqueeze(0)
                elif sdv.ndim == 5 and sdv.shape[1] == 1:  # (B,1,T,H,W)
                    sdv = sdv[:, 0]
                elif sdv.ndim != 4:
                    raise ValueError(f"source_depth_video must be (T,H,W), (B,T,H,W), or (B,1,T,H,W); got {tuple(sdv.shape)}")
                sdv = sdv.to(device=pipe.device)
                if str(target_depth_norm_type) == "disparity":
                    sdv_norm = WanVideoUnit_TargetDepthEncoder._normalize_depth_as_disparity(sdv, None, target_depth_max_disparity)
                else:
                    sdv_norm = WanVideoUnit_TargetDepthEncoder._normalize_depth_with_fixed_cap(sdv, None, target_depth_log_cap)
                sdv_norm = sdv_norm.unsqueeze(1).repeat(1, 3, 1, 1, 1).to(dtype=pipe.torch_dtype)
                result["source_depth_latents"] = pipe.vae.encode(sdv_norm, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            elif input_latents is not None:
                result["source_depth_latents"] = torch.zeros_like(input_latents)
        if channel_modalities and "residual_track" in channel_modalities:
            if input_latents is not None:
                result["source_track_latents"] = torch.zeros_like(input_latents)
        return result


class WanVideoUnit_TargetResidualTrackEncoder(PipelineUnit):
    """Encode target residual 3D tracks via VAE for channel-concat Stage B."""

    def __init__(self):
        super().__init__(
            input_params=("target_track_video", "target_track_valid_mask", "tracks_world",
                          "input_latents", "tiled", "tile_size", "tile_stride",
                          "channel_concat", "channel_modalities", "track_query_idx",
                          "target_track_use_source_stats"),
            output_params=("target_track_latents", "target_track_normalization_stats"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, target_track_video=None, target_track_valid_mask=None,
                tracks_world=None, input_latents=None, tiled=False, tile_size=None,
                tile_stride=None, channel_concat=False, channel_modalities=None,
                track_query_idx=0, target_track_use_source_stats=False):
        if not channel_concat or not channel_modalities or "residual_track" not in channel_modalities:
            return {}
        if target_track_video is None:
            if input_latents is not None:
                return {"target_track_latents": torch.zeros_like(input_latents),
                        "target_track_normalization_stats": None}
            return {}
        tile_size = tile_size or (30, 52)
        tile_stride = tile_stride or (15, 26)
        import numpy as np
        ttv = torch.from_numpy(target_track_video).float() if isinstance(target_track_video, np.ndarray) else target_track_video.float()
        if ttv.ndim == 4:
            ttv = ttv.unsqueeze(0)
        ttv = ttv.to(device=pipe.device)
        query_idx = int(track_query_idx)
        residual = ttv - ttv[:, query_idx:query_idx + 1]
        residual_bctHW = residual.permute(0, 4, 1, 2, 3).contiguous()
        tvm = target_track_valid_mask
        if tvm is not None:
            tvm = (torch.from_numpy(tvm) if isinstance(tvm, np.ndarray) else tvm).to(device=pipe.device, dtype=torch.bool)
            if tvm.ndim == 3:
                tvm = tvm.unsqueeze(0)
        from holo4d_runtime.pipelines.wan_video_v2v import WanVideoUnit_TrackVideo
        ttv_norm = WanVideoUnit_TrackVideo._normalize_track_sequencewise_signed_log_p2_p98(residual_bctHW, tvm, torch_dtype=torch.float32)
        ttv_norm = ttv_norm.to(dtype=pipe.torch_dtype)
        target_track_latents = pipe.vae.encode(ttv_norm, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"target_track_latents": target_track_latents, "target_track_normalization_stats": None}


class WanVideoUnit_ChannelConcatV2V(PipelineUnit):
    """Channel-concat modalities within each view, then width-concat source and target."""

    def __init__(self):
        super().__init__(
            input_params=("input_latents", "target_rgb_latents", "source_depth_latents",
                          "source_track_latents", "depth_latents", "target_track_latents",
                          "channel_concat", "channel_modalities"),
            output_params=("input_latents", "cond_latents"),
        )

    def process(self, pipe, input_latents=None, target_rgb_latents=None,
                source_depth_latents=None, source_track_latents=None,
                depth_latents=None, target_track_latents=None,
                channel_concat=False, channel_modalities=None):
        if not channel_concat:
            return {}
        if input_latents is None or target_rgb_latents is None:
            return {}
        channel_modalities = channel_modalities or []
        source_parts = [input_latents]
        if "depth" in channel_modalities and source_depth_latents is not None:
            source_parts.append(source_depth_latents)
        if "residual_track" in channel_modalities and source_track_latents is not None:
            source_parts.append(source_track_latents)
        target_parts = [target_rgb_latents]
        if "depth" in channel_modalities and depth_latents is not None:
            target_parts.append(depth_latents)
        if "residual_track" in channel_modalities and target_track_latents is not None:
            target_parts.append(target_track_latents)
        # Validate spatial shapes match before concatenating.
        ref_shape = input_latents.shape
        for p in source_parts[1:] + target_parts[1:]:
            if p.shape[0] != ref_shape[0] or p.shape[2:] != ref_shape[2:]:
                raise ValueError(
                    f"channel-concat: all latent parts must share (B, *, T, H, W) shape; "
                    f"reference={tuple(ref_shape)}, offending={tuple(p.shape)}"
                )
        source_mm = torch.cat(source_parts, dim=1)
        target_mm = torch.cat(target_parts, dim=1)
        return {"input_latents": torch.cat([source_mm, target_mm], dim=4), "cond_latents": source_mm}


# =========================================================================


class WanVideoUnit_TrackPointCondition(PipelineUnit):
    """
    Process world-space 3D tracks into latent-space control tokens.

    Steps:
      1. Project world tracks to target camera view → 2D tracks
      2. Embed per-track sequences via PointTrackEmbedder
      3. Aggregate into block tokens via PointTrackAdapter
      4. Expand blocks to full latent sequence
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "tracks_world",
                "target_cam2world",
                "target_intrinsics",
                "target_intrinsics_per_frame",
                "height",
                "width",
                "num_frames",
                "tracks_world_valid_mask",
                "tracks_query_idx_per_identity",
                "track_identity_sampling_seed",
                "source_depth_at_tracks",
                "source_depth_valid_at_tracks",
                "source_rgb_at_tracks",
                "source_dynamic_mask_at_tracks",
                "target_dynamic_mask_at_tracks",
                # ab28: dense source video/depth/cameras to compute the fg/bg-fused
                # reprojected RGB in the track path (only used when the dynmask+
                # reproj track config sets include_reprojected_rgb).
                "source_rgb_video",
                "source_depth_video",
                "source_depth_valid",
                "source_cam2world",
                "source_intrinsics",
                "tracks_world_dense_valid_mask",
                "source_dynamic_mask",
                "target_dynamic_mask",
                "has_tracks",
                "use_track_null_token_fallback",
                "track_profile_meta",
                # Future prediction (temporal extrapolation): when on, future
                # frames' track visibility is zeroed so the per-block null_token
                # fills future latent-frame blocks. Default off → no-op.
                "future_prediction",
                "future_num_latent_frames",
            ),
            output_params=("track_point_tokens", "_pre_block_hook", "_track_align_fn", "_track_block_injector"),
        )

    def process(
        self,
        pipe,
        tracks_world=None,
        target_cam2world=None,
        target_intrinsics=None,
        target_intrinsics_per_frame=None,
        height=None,
        width=None,
        num_frames=None,
        tracks_world_valid_mask=None,
        tracks_query_idx_per_identity=None,
        track_identity_sampling_seed=None,
        source_depth_at_tracks=None,
        source_depth_valid_at_tracks=None,
        source_rgb_at_tracks=None,
        source_dynamic_mask_at_tracks=None,
        target_dynamic_mask_at_tracks=None,
        source_rgb_video=None,
        source_depth_video=None,
        source_depth_valid=None,
        source_cam2world=None,
        source_intrinsics=None,
        tracks_world_dense_valid_mask=None,
        source_dynamic_mask=None,
        target_dynamic_mask=None,
        # Explicit conditioning flags (must also be listed in input_params above
        # so the unit_runner forwards them). No **kwargs catch-all: every
        # input_params entry is a named arg here, so any drift between
        # input_params and this signature surfaces as a TypeError instead of a
        # silently-dropped input.
        has_tracks=True,                      # False on track-dropout samples → null-mask the track tokens
        use_track_null_token_fallback=False,  # emit a learned null token when tracks_world is absent
        track_profile_meta=None,
        future_prediction=False,
        future_num_latent_frames=0,
    ):
        if target_cam2world is None:
            return {}
        if getattr(pipe, "track_adapter", None) is None:
            # Raw plucker-only / disabled-Track4D runs remove the track modules
            # entirely. If this unit is still present for any reason, fail safe.
            return {}

        device = pipe.device
        dtype = pipe.torch_dtype

        # Move inputs to device (keep float32 for geometric precision)
        if torch.is_tensor(tracks_world):
            tracks_world = tracks_world.to(device=device, dtype=torch.float32)
        if torch.is_tensor(target_cam2world):
            target_cam2world = target_cam2world.to(device=device, dtype=torch.float32)
        if torch.is_tensor(target_intrinsics):
            target_intrinsics = target_intrinsics.to(device=device, dtype=torch.float32)
        if torch.is_tensor(target_intrinsics_per_frame):
            target_intrinsics_per_frame = target_intrinsics_per_frame.to(
                device=device,
                dtype=torch.float32,
            )
        # A single K remains the legacy/default contract.  Zoom and dolly-zoom
        # trajectories provide K_t; those intrinsics must govern persistent
        # track projection as well as Plucker rays, otherwise the two strongest
        # camera conditions describe different target views.
        target_projection_intrinsics = (
            target_intrinsics_per_frame
            if target_intrinsics_per_frame is not None
            else target_intrinsics
        )
        if (
            target_intrinsics_per_frame is not None
            and _track4d_env_flag("HOLO4D_DEBUG_INTRINSICS", "0")
        ):
            debug_k = torch.as_tensor(target_intrinsics_per_frame).detach().cpu()
            if debug_k.ndim == 4:
                debug_k = debug_k[0]
            print(
                "[Track4D-360/intrinsics] persistent-track projection uses K_t "
                f"shape={tuple(debug_k.shape)} "
                f"fx0={float(debug_k[0, 0, 0]):.6f} "
                f"fx_last={float(debug_k[-1, 0, 0]):.6f} "
                f"fy0={float(debug_k[0, 1, 1]):.6f} "
                f"fy_last={float(debug_k[-1, 1, 1]):.6f}",
                flush=True,
            )
        if torch.is_tensor(tracks_world_valid_mask):
            tracks_world_valid_mask = tracks_world_valid_mask.to(device=device)
        if tracks_query_idx_per_identity is not None:
            if not torch.is_tensor(tracks_query_idx_per_identity):
                tracks_query_idx_per_identity = torch.as_tensor(tracks_query_idx_per_identity)
            tracks_query_idx_per_identity = tracks_query_idx_per_identity.to(
                device=device, dtype=torch.long,
            )

        image_h = int(height)
        image_w = int(width)
        T = int(num_frames)
        B = (
            int(target_cam2world.shape[0])
            if torch.is_tensor(target_cam2world) and target_cam2world.ndim == 4
            else 1
        )
        spatial_div = getattr(pipe, 'height_division_factor', 16)
        latent_t = (T - 1) // 4 + 1
        latent_h = image_h // spatial_div
        latent_w = image_w // spatial_div
        profile_call_id = None
        if _track4d_adapter_profile_enabled():
            global _TRACK4D_ADAPTER_PROFILE_CALL
            _TRACK4D_ADAPTER_PROFILE_CALL += 1
            profile_call_id = _TRACK4D_ADAPTER_PROFILE_CALL
            every = max(int(os.environ.get("HOLO4D_ADAPTER_MEM_PROFILE_EVERY", "1") or 1), 1)
            if profile_call_id % every != 0:
                profile_call_id = None

        # Structural occupancy channel state for this forward (opt-in via
        # HOLO4D_TRACK_OCCUPANCY_CHANNEL=1). Populated after the adapter
        # runs; None on fallback paths → appended as all-unobserved zeros.
        # source_occupancy_seq: shared-identity source tokens carry REAL
        # identity observations, so their occupancy comes from the identity
        # rasterizer's source mask (Codex review F6) — constant 0 would
        # contradict the nonzero identity features.
        track_occupancy_seq = None
        source_occupancy_seq = None

        def _make_track_null_tokens():
            if not use_track_null_token_fallback:
                return None
            null_token = getattr(pipe, "track_null_token", None)
            if null_token is None:
                return None
            return null_token.to(device=device, dtype=dtype).expand(
                B, latent_t * latent_h * latent_w, -1
            )

        def _build_injection_payload(
            track_point_tokens,
            source_track_point_tokens=None,
        ):
            if track_occupancy_channel_enabled():
                # Append the structural occupancy channel LAST, after identity
                # deltas / noise / null-swap, so those stages keep operating on
                # the legacy channel layout. Source-domain occupancy uses the
                # identity rasterizer's source mask in shared mode (F6) and 0
                # otherwise (no observations live on source tokens then).
                if track_occupancy_seq is not None:
                    occ = track_occupancy_seq.to(
                        device=track_point_tokens.device,
                        dtype=track_point_tokens.dtype,
                    )
                else:
                    occ = torch.zeros_like(track_point_tokens[..., :1])
                track_point_tokens = torch.cat([track_point_tokens, occ], dim=-1)
                if source_track_point_tokens is not None:
                    if source_occupancy_seq is not None:
                        src_occ = source_occupancy_seq.to(
                            device=source_track_point_tokens.device,
                            dtype=source_track_point_tokens.dtype,
                        )
                    else:
                        src_occ = torch.zeros_like(source_track_point_tokens[..., :1])
                    source_track_point_tokens = torch.cat(
                        [source_track_point_tokens, src_occ], dim=-1
                    )
            injection_mode = getattr(pipe, "track_injection_mode", "single")

            if injection_mode == "per_block":
                def _make_per_block_align(tpt, source_tpt, adapter, injector):
                    _cached = {"grad": None, "nograd": None}

                    def align_fn(x, f, h, w, num_domains=2, v2v_concat_mode="width"):
                        cache_key = "grad" if torch.is_grad_enabled() else "nograd"
                        if _cached[cache_key] is None:
                            with track4d_record_function("track4d360/track_condition/align_to_sequence"):
                                _cached[cache_key] = _align_track_to_sequence(
                                    tpt,
                                    adapter,
                                    x,
                                    f,
                                    h,
                                    w,
                                    num_domains,
                                    v2v_concat_mode,
                                    source_tpt=source_tpt,
                                )
                        return _cached[cache_key]

                    return align_fn

                align_fn = _make_per_block_align(
                    track_point_tokens,
                    source_track_point_tokens,
                    pipe.track_adapter,
                    pipe.track_block_injector,
                )
                return {
                    "track_point_tokens": track_point_tokens,
                    "_track_align_fn": align_fn,
                    "_track_block_injector": pipe.track_block_injector,
                }

            def _make_pre_block_hook(tpt, source_tpt, projector, adapter):
                def hook(x, f, h, w, num_domains=2, v2v_concat_mode="width"):
                    if tpt is None:
                        return x
                    with track4d_record_function("track4d360/track_condition/align_to_sequence"):
                        track_full = _align_track_to_sequence(
                            tpt,
                            adapter,
                            x,
                            f,
                            h,
                            w,
                            num_domains,
                            v2v_concat_mode,
                            source_tpt=source_tpt,
                        )
                    with track4d_record_function("track4d360/track_condition/pre_block_projector"):
                        return projector(x, track_full)
                return hook

            hook_fn = _make_pre_block_hook(
                track_point_tokens,
                source_track_point_tokens,
                pipe.track_projector,
                pipe.track_adapter,
            )
            return {
                "track_point_tokens": track_point_tokens,
                "_pre_block_hook": hook_fn,
            }

        if tracks_world is None:
            track_point_tokens = _make_track_null_tokens()
            if track_point_tokens is None:
                return {}
            return _build_injection_payload(track_point_tokens)

        # 1. Build track sequences: 3D world coords for embedder, 2D pixel for adapter
        if source_depth_at_tracks is not None:
            if torch.is_tensor(source_depth_at_tracks):
                source_depth_at_tracks = source_depth_at_tracks.to(device=device, dtype=torch.float32)
        if source_depth_valid_at_tracks is not None:
            if torch.is_tensor(source_depth_valid_at_tracks):
                source_depth_valid_at_tracks = source_depth_valid_at_tracks.to(device=device)
        # Auto-detect track channel config from adapter.
        # Channel layout (must match Conv3dTrackAdapter in_channels):
        #   5  = (frame_idx, wx, wy, wz, vis)
        #   6  = 5 + depth
        #   8  = 5 + R, G, B
        #   9  = 5 + depth + R, G, B
        #   11 = 5 + anchor(3) + R, G, B
        #   12 = 5 + anchor(3) + depth + R, G, B
        track_in_channels = getattr(pipe.track_adapter, 'in_channels', 5) if hasattr(pipe, 'track_adapter') else 5
        # Explicit per-channel flags (new track configs declare them) take priority;
        # when absent (legacy configs) fall back to the channel-count map so existing
        # 9/11/12-channel runs + checkpoints are byte-identical. Dynamic-mask channels
        # only ever turn on via explicit flags (never inferred from channel count).
        _flags = getattr(pipe, "track_include_flags", None)
        if _flags:
            include_reproj_depth = bool(_flags.get("source_depth", track_in_channels in {6, 9, 12}))
            include_source_rgb = bool(_flags.get("source_rgb", track_in_channels in {8, 9, 11, 12}))
            include_track_anchor = bool(_flags.get("track_anchor", track_in_channels in {11, 12}))
            include_source_dynamic_mask = bool(_flags.get("source_dynamic_mask", False))
            include_target_dynamic_mask = bool(_flags.get("target_dynamic_mask", False))
            include_reprojected_rgb = bool(_flags.get("reprojected_rgb", False))
        else:
            include_reproj_depth = track_in_channels in {6, 9, 12}
            include_source_rgb = track_in_channels in {8, 9, 11, 12}
            include_track_anchor = track_in_channels in {11, 12}
            include_source_dynamic_mask = False
            include_target_dynamic_mask = False
            include_reprojected_rgb = False

        # Retrieve source_rgb_at_tracks and move to device (only when RGB config)
        source_rgb_at_tracks = source_rgb_at_tracks if include_source_rgb else None
        if source_rgb_at_tracks is not None and torch.is_tensor(source_rgb_at_tracks):
            source_rgb_at_tracks = source_rgb_at_tracks.to(device=device, dtype=torch.float32)
        elif include_source_rgb and source_rgb_at_tracks is None:
            if os.environ.get("HOLO4D_STRICT_SOURCE_RGB", "0") == "1":
                raise ValueError(
                    f"track_in_channels={track_in_channels} requires source_rgb_at_tracks, "
                    "but it was not provided. Disable HOLO4D_STRICT_SOURCE_RGB=1 "
                    "only for intentional zero-RGB ablations."
                )
            import warnings
            warnings.warn(
                f"track_in_channels={track_in_channels} includes RGB but source_rgb_at_tracks "
                "not provided; RGB channels will be zeros.",
                stacklevel=2,
            )
            tw = tracks_world
            if not torch.is_tensor(tw):
                tw = torch.as_tensor(tw, device=device, dtype=torch.float32)
            if tw.ndim == 3:
                rgb_shape = (1, int(tw.shape[0]), int(tw.shape[1]), 3)
            elif tw.ndim == 4:
                rgb_shape = (int(tw.shape[0]), int(tw.shape[1]), int(tw.shape[2]), 3)
            else:
                raise ValueError(
                    f"tracks_world must be (N,T,3) or (B,N,T,3) before RGB fallback, got {tuple(tw.shape)}"
                )
            source_rgb_at_tracks = torch.zeros(rgb_shape, device=device, dtype=torch.float32)

        # Preserve the row-aligned mask for canonical FG/BG selection even when
        # it is not configured as an adapter input channel.
        canonical_source_dynamic_mask_at_tracks = source_dynamic_mask_at_tracks
        if (
            canonical_source_dynamic_mask_at_tracks is not None
            and torch.is_tensor(canonical_source_dynamic_mask_at_tracks)
        ):
            canonical_source_dynamic_mask_at_tracks = canonical_source_dynamic_mask_at_tracks.to(
                device=device
            )

        # Per-track Vista4D-style dynamic masks (only when explicitly enabled).
        source_dynamic_mask_at_tracks = source_dynamic_mask_at_tracks if include_source_dynamic_mask else None
        if source_dynamic_mask_at_tracks is not None and torch.is_tensor(source_dynamic_mask_at_tracks):
            source_dynamic_mask_at_tracks = source_dynamic_mask_at_tracks.to(device=device, dtype=torch.float32)
        elif include_source_dynamic_mask and source_dynamic_mask_at_tracks is None:
            # Mixed training intentionally combines dynamic datasets (Syn4D/Kubric,
            # which emit masks) with STATIC scenes (dl3dv/re10k/mvs_synth/tartanair,
            # which have no motion -> a zero dynamic-mask channel is CORRECT, not a
            # bug). So zero-fill is the default. HOLO4D_STRICT_DYNMASK=1 opts in
            # to a hard fail on absence (useful only for dynamic-ONLY debug runs, to
            # catch a misconfigured mask path); never enable it for mixed data.
            if getattr(pipe.dit, "training", False) and os.environ.get("HOLO4D_STRICT_DYNMASK", "0") == "1":
                raise RuntimeError(
                    "HOLO4D_STRICT_DYNMASK=1 and include_source_dynamic_mask is on "
                    "but source_dynamic_mask_at_tracks is absent. For dynamic-only runs "
                    "ensure the dataset emits it; for mixed static+dynamic data, unset "
                    "HOLO4D_STRICT_DYNMASK (static scenes correctly have no mask)."
                )
        target_dynamic_mask_at_tracks = target_dynamic_mask_at_tracks if include_target_dynamic_mask else None
        if target_dynamic_mask_at_tracks is not None and torch.is_tensor(target_dynamic_mask_at_tracks):
            target_dynamic_mask_at_tracks = target_dynamic_mask_at_tracks.to(device=device, dtype=torch.float32)

        # ab28: compute the fg/bg-hybrid-fused + boundary-cleaned reprojected RGB in
        # the track path (same warp ab23/24 inject via token stream) so the builder
        # can sample it at each track's target-UV and replace the query-anchored
        # src_rgb. Gated on include_reprojected_rgb; any missing input / failure ->
        # None -> the builder keeps query-anchored src_rgb (backward-compatible).
        # NOTE: this adds a dense soft-splat warp to the track path (cost), so ab28
        # is faster than the ab23/24 token stream but not pure ab12 speed.
        reprojected_rgb_dense = None
        reprojected_rgb_valid = None
        _reproj_inputs_present = (
            source_rgb_video is not None
            and source_depth_video is not None
            and source_cam2world is not None
        )
        _allow_reproj_fallback = (
            os.environ.get("HOLO4D_ALLOW_REPROJECTED_RGB_FALLBACK", "0") == "1"
        )
        if include_reprojected_rgb and not _reproj_inputs_present and not _allow_reproj_fallback:
            # GPT review: do NOT silently fall back to query-anchored RGB — that would
            # make ab28 secretly evaluate/train as ab27. Fail loudly so the missing
            # forwarding (e.g. eval not requesting source_rgb_video) is caught.
            raise RuntimeError(
                "include_reprojected_rgb is on but source_rgb_video/source_depth_video/"
                "source_cam2world were not forwarded to the track unit -> ab28 would "
                "silently fall back to query-anchored src_rgb (== ab27). Forward the dense "
                "source fields (train does; eval must forward them when the track config "
                "sets reprojected_rgb), or set HOLO4D_ALLOW_REPROJECTED_RGB_FALLBACK=1 "
                "to intentionally fall back."
            )
        if include_reprojected_rgb and _reproj_inputs_present:
            try:
                from holo4d.conditions.dense_target_reprojection_builder import (
                    build_dense_target_reprojection_hybrid,
                )
                from holo4d.conditions.dense_target_control_unit import (
                    WanVideoUnit_DenseTargetControl as _DTC,
                    _maybe_remove_source_mask_boundary_depth,
                )
                from holo4d.conditions.soft_splat_warp import (
                    compute_target_dynamic_mask_from_tracks_world,
                )
                _T, _H, _W = int(num_frames), int(image_h), int(image_w)
                _tgt = target_cam2world if torch.is_tensor(target_cam2world) else torch.as_tensor(target_cam2world)
                _B = _tgt.shape[0] if _tgt.ndim == 4 else 1
                _src_rgb = _DTC._coerce_source_rgb_video(source_rgb_video, _B, _T, _H, _W, device)
                _src_depth = _DTC._coerce_source_depth_video(source_depth_video, _B, _T, _H, _W, device)
                _src_valid = _DTC._coerce_source_depth_valid(source_depth_valid, _src_depth)
                _src_c2w = _DTC._coerce_cam2world(source_cam2world, _B, _T, device)
                _tgt_c2w = _DTC._coerce_cam2world(target_cam2world, _B, _T, device)
                _K_in = source_intrinsics if source_intrinsics is not None else target_intrinsics
                _K = _DTC._coerce_intrinsics(_K_in, _B, _T, device)
                _K_tgt = (
                    _DTC._coerce_intrinsics(
                        target_projection_intrinsics,
                        _B,
                        _T,
                        device,
                    )
                    if target_projection_intrinsics is not None
                    else _K
                )
                _src_dyn = None
                if source_dynamic_mask is not None and torch.is_tensor(source_dynamic_mask):
                    _src_dyn = source_dynamic_mask.to(device=device).bool()
                    if _src_dyn.ndim == 3:
                        _src_dyn = _src_dyn.unsqueeze(0).expand(_B, -1, -1, -1)
                # boundary-depth cleaning (drops edge points before the warp)
                _src_valid = _maybe_remove_source_mask_boundary_depth(
                    _src_valid, _src_dyn,
                    prob=float(getattr(pipe, "source_mask_boundary_depth_dropout_prob", 0.0)),
                    kernel_size=int(getattr(pipe, "source_mask_boundary_kernel_size", 5)),
                    training=bool(getattr(pipe.dit, "training", False)),
                )
                _tgt_dyn = None
                if target_dynamic_mask is not None and torch.is_tensor(target_dynamic_mask):
                    _tgt_dyn = target_dynamic_mask.to(device=device).bool()
                    if _tgt_dyn.ndim == 3:
                        _tgt_dyn = _tgt_dyn.unsqueeze(0).expand(_B, -1, -1, -1)
                elif _src_dyn is not None:
                    _tgt_dyn = compute_target_dynamic_mask_from_tracks_world(
                        tracks_world=tracks_world, tgt_cam2world=_tgt_c2w, tgt_intrinsics=_K_tgt,
                        image_h=_H, image_w=_W, motion_threshold=0.005,
                    )
                _warped_rgb, _, _warp_valid = build_dense_target_reprojection_hybrid(
                    source_rgb=_src_rgb, source_depth=_src_depth, source_depth_valid=_src_valid,
                    src_cam2world=_src_c2w, tgt_cam2world=_tgt_c2w, intrinsics=_K,
                    image_h=_H, image_w=_W,
                    source_static_mask=None if _src_dyn is None else ~_src_dyn,
                    target_dynamic_mask=_tgt_dyn,
                    dynamic_tracks_world=tracks_world,
                    dynamic_tracks_valid_mask=(
                        tracks_world_dense_valid_mask
                        if tracks_world_dense_valid_mask is not None
                        else tracks_world_valid_mask
                    ),
                    target_intrinsics=_K_tgt,
                )
                reprojected_rgb_dense = _warped_rgb
                reprojected_rgb_valid = _warp_valid
            except Exception as _e:  # GPT review: fail loudly, don't silently become ab27
                if not _allow_reproj_fallback:
                    raise
                import warnings as _w
                _w.warn(
                    f"ab28 reprojected-RGB warp failed ({_e}); "
                    "HOLO4D_ALLOW_REPROJECTED_RGB_FALLBACK=1 -> falling back to "
                    "query-anchored src_rgb.",
                    stacklevel=2,
                )
                reprojected_rgb_dense = None
                reprojected_rgb_valid = None

        fixed_identity_count = int(os.environ.get("HOLO4D_TRACK_IDENTITY_FIXED_N", "0") or 0)
        identity_sampling_mode = str(
            os.environ.get("HOLO4D_TRACK_IDENTITY_SAMPLING_MODE", "legacy_uniform_order")
        ).strip().lower()
        canonical_methods = {
            "canonical_object_fps": "exact_fps",
            "canonical_object_morton": "morton_stratified",
            "canonical_object_torch_cluster_fps": "torch_cluster_fps",
        }
        count_sampling_seed = (
            None
            if track_identity_sampling_seed is None
            else int(torch.as_tensor(track_identity_sampling_seed).reshape(-1)[0].item())
        )
        variable_identity_count = resolve_variable_identity_count(
            fixed_identity_count,
            sampling_seed=count_sampling_seed,
            # The diffusion scheduler is the pipeline's authoritative
            # train/eval signal. DiT modules may remain in eval mode while
            # selected parameters are optimized by the training wrapper.
            training=bool(getattr(pipe.scheduler, "training", False)),
        )
        active_identity_count = variable_identity_count.active_count
        if variable_identity_count.enabled and identity_sampling_mode not in canonical_methods:
            raise ValueError(
                "Train-time variable identity count currently requires a canonical "
                "selector (canonical_object_morton/fps/torch_cluster_fps), got "
                f"{identity_sampling_mode!r}."
            )
        if variable_identity_count.enabled:
            printed = int(getattr(pipe, "_variable_identity_count_prints", 0))
            if printed < 5:
                print(
                    "[Track4D-360/variable-N] "
                    f"rank={os.environ.get('RANK', '?')} "
                    f"active_N={active_identity_count} "
                    f"range=[{variable_identity_count.minimum},"
                    f"{variable_identity_count.maximum}] "
                    f"padded_N={variable_identity_count.padded_count} "
                    f"seed={count_sampling_seed}",
                    flush=True,
                )
                pipe._variable_identity_count_prints = printed + 1
        canonical_selection = None
        if identity_sampling_mode in canonical_methods:
            if fixed_identity_count <= 0:
                raise ValueError(
                    f"HOLO4D_TRACK_IDENTITY_SAMPLING_MODE={identity_sampling_mode} "
                    "requires HOLO4D_TRACK_IDENTITY_FIXED_N > 0"
                )
            exact_fps_limit = int(
                os.environ.get("HOLO4D_TRACK_CANONICAL_EXACT_FPS_MAX_K", "8192") or 8192
            )
            if (
                identity_sampling_mode == "canonical_object_fps"
                and active_identity_count > exact_fps_limit
            ):
                raise ValueError(
                    "Exact canonical FPS is sequential O(KN) and is disabled for "
                    f"K={active_identity_count} > {exact_fps_limit}. Use "
                    "canonical_object_morton for the H*W capacity arm or explicitly "
                    "raise HOLO4D_TRACK_CANONICAL_EXACT_FPS_MAX_K after benchmarking."
                )
            tracks_for_shape = tracks_world if torch.is_tensor(tracks_world) else torch.as_tensor(tracks_world)
            if tracks_for_shape.ndim == 3:
                canonical_batch_size, canonical_identity_count, canonical_frame_count = (
                    1,
                    int(tracks_for_shape.shape[0]),
                    int(tracks_for_shape.shape[1]),
                )
            elif tracks_for_shape.ndim == 4:
                canonical_batch_size, canonical_identity_count, canonical_frame_count = map(
                    int, tracks_for_shape.shape[:3]
                )
            else:
                raise ValueError(
                    "tracks_world must be (N,T,3) or (B,N,T,3), got "
                    f"{tuple(tracks_for_shape.shape)}"
                )
            if tracks_query_idx_per_identity is None:
                canonical_query_indices = torch.zeros(
                    (canonical_batch_size, canonical_identity_count),
                    device=device,
                    dtype=torch.long,
                )
            else:
                canonical_query_indices = tracks_query_idx_per_identity

            foreground_identity_mask = None
            if canonical_source_dynamic_mask_at_tracks is not None:
                foreground_identity_mask = foreground_from_row_aligned_mask(
                    canonical_source_dynamic_mask_at_tracks,
                    canonical_query_indices,
                    batch_size=canonical_batch_size,
                    identity_count=canonical_identity_count,
                    frame_count=canonical_frame_count,
                ).to(device=device)
            elif (
                source_dynamic_mask is not None
                and source_cam2world is not None
                and source_intrinsics is not None
            ):
                source_cameras = (
                    source_cam2world.to(device=device, dtype=torch.float32)
                    if torch.is_tensor(source_cam2world)
                    else torch.as_tensor(source_cam2world, device=device, dtype=torch.float32)
                )
                source_camera_intrinsics = (
                    source_intrinsics.to(device=device, dtype=torch.float32)
                    if torch.is_tensor(source_intrinsics)
                    else torch.as_tensor(source_intrinsics, device=device, dtype=torch.float32)
                )
                source_uv, _, _ = project_tracks_to_target_views(
                    tracks_world,
                    source_cameras,
                    source_camera_intrinsics,
                    image_h,
                    image_w,
                )
                foreground_identity_mask = foreground_at_query_anchors(
                    source_uv,
                    canonical_query_indices,
                    source_dynamic_mask,
                    image_hw=(image_h, image_w),
                )
            elif os.environ.get("HOLO4D_TRACK_CANONICAL_STRICT_FG_MASK", "0") == "1":
                raise RuntimeError(
                    "Canonical object sampling requires a source dynamic mask when "
                    "HOLO4D_TRACK_CANONICAL_STRICT_FG_MASK=1"
                )

            canonical_missing_mask_fallback = foreground_identity_mask is None
            canonical_reference_frame = int(
                os.environ.get("HOLO4D_TRACK_CANONICAL_REFERENCE_FRAME", "0") or 0
            )
            canonical_foreground_fraction = float(
                os.environ.get("HOLO4D_TRACK_FOREGROUND_FRACTION", "0.75") or 0.75
            )
            canonical_profile = _track4d_env_flag(
                "HOLO4D_TRACK_CANONICAL_PROFILE", "0"
            )
            canonical_query_aware = (
                os.environ.get("HOLO4D_TRACK_CANONICAL_QUERY_AWARE", "0") == "1"
            )
            if canonical_query_aware and tracks_query_idx_per_identity is None:
                # Codex review F5: silently falling back to the t0 policy would
                # make the flag a no-op on exactly the samples it targets.
                raise RuntimeError(
                    "HOLO4D_TRACK_CANONICAL_QUERY_AWARE=1 requires "
                    "tracks_query_idx_per_identity (row-level query provenance); "
                    "the dataset did not provide it. Fix the producer or unset "
                    "the flag."
                )
            if canonical_query_aware:
                # Per-query eligibility: each identity is selected against its
                # OWN query anchor frame, so q_last-only disocclusion rows
                # (invalid at t0) stay eligible. The global
                # HOLO4D_TRACK_CANONICAL_REFERENCE_FRAME is ignored here.
                canonical_selection = select_canonical_object_identities_query_aware(
                    tracks_world,
                    tracks_world_valid_mask,
                    foreground_identity_mask,
                    active_identity_count,
                    query_idx_per_identity=canonical_query_indices,
                    foreground_fraction=canonical_foreground_fraction,
                    method=canonical_methods[identity_sampling_mode],
                    morton_bits=int(
                        os.environ.get("HOLO4D_TRACK_CANONICAL_MORTON_BITS", "10") or 10
                    ),
                    profile=canonical_profile,
                )
            else:
                canonical_selection = select_canonical_object_identities(
                    tracks_world,
                    tracks_world_valid_mask,
                    foreground_identity_mask,
                    active_identity_count,
                    reference_frame=canonical_reference_frame,
                    foreground_fraction=canonical_foreground_fraction,
                    method=canonical_methods[identity_sampling_mode],
                    morton_bits=int(
                        os.environ.get("HOLO4D_TRACK_CANONICAL_MORTON_BITS", "10") or 10
                    ),
                    profile=canonical_profile,
                )
            if variable_identity_count.enabled:
                canonical_selection = pad_canonical_object_selection(
                    canonical_selection,
                    variable_identity_count.padded_count,
                )
            if canonical_profile:
                rank = os.environ.get("RANK", "?")
                local_rank = os.environ.get("LOCAL_RANK", "?")
                profiled_queries = gather_identity_payload(
                    canonical_query_indices,
                    canonical_selection.indices,
                    expected_identity_count=canonical_identity_count,
                )
                for batch_index in range(canonical_selection.indices.shape[0]):
                    fg_candidates = int(
                        canonical_selection.foreground_candidate_counts[batch_index].item()
                    )
                    bg_candidates = int(
                        canonical_selection.background_candidate_counts[batch_index].item()
                    )
                    valid_row = canonical_selection.valid[batch_index]
                    query_counts = {
                        int(query): int(
                            (profiled_queries[batch_index, valid_row] == query).sum().item()
                        )
                        for query in torch.unique(
                            profiled_queries[batch_index, valid_row], sorted=True
                        )
                    }
                    print(
                        "[canonical-selector-profile] "
                        f"rank={rank} local_rank={local_rank} batch={batch_index} "
                        f"mode={identity_sampling_mode} raw_eligible_n={fg_candidates + bg_candidates} "
                        f"active_target_n={active_identity_count} "
                        f"padded_n={variable_identity_count.padded_count} "
                        f"fg_candidates={fg_candidates} bg_candidates={bg_candidates} "
                        f"fg_selected={int(canonical_selection.selected_foreground_counts[batch_index].item())} "
                        f"bg_selected={int(canonical_selection.selected_background_counts[batch_index].item())} "
                        f"selected_query_counts={query_counts} "
                        f"padding={int((~canonical_selection.valid[batch_index]).sum().item())} "
                        f"fg_raw_returned={int(canonical_selection.foreground_raw_returned_counts[batch_index].item())} "
                        f"bg_raw_returned={int(canonical_selection.background_raw_returned_counts[batch_index].item())} "
                        f"selector_ms={float(canonical_selection.selector_wall_time_ms[batch_index].item()):.3f} "
                        f"selector_peak_allocated_bytes={int(canonical_selection.selector_peak_allocated_bytes[batch_index].item())} "
                        f"missing_mask_fallback={int(canonical_missing_mask_fallback)} "
                        f"reference_frame={canonical_reference_frame} "
                        f"foreground_fraction={canonical_foreground_fraction:.6f}",
                        flush=True,
                    )
            selected_indices = canonical_selection.indices
            if _track4d_env_flag("HOLO4D_CAPTURE_SELECTED_TRACKS_FOR_VIZ", "0"):
                pipe._debug_last_track_identity_selection = {
                    "mode": identity_sampling_mode,
                    "indices": selected_indices.detach().cpu(),
                    "valid": canonical_selection.valid.detach().cpu(),
                }
            gather_kwargs = {"expected_identity_count": canonical_identity_count}
            tracks_world = gather_identity_payload(
                tracks_world, selected_indices, **gather_kwargs
            )
            tracks_world_valid_mask = gather_identity_payload(
                tracks_world_valid_mask, selected_indices, **gather_kwargs
            )
            tracks_query_idx_per_identity = gather_identity_payload(
                canonical_query_indices, selected_indices, **gather_kwargs
            )
            source_depth_at_tracks = gather_identity_payload(
                source_depth_at_tracks, selected_indices, **gather_kwargs
            )
            source_depth_valid_at_tracks = gather_identity_payload(
                source_depth_valid_at_tracks, selected_indices, **gather_kwargs
            )
            source_rgb_at_tracks = gather_identity_payload(
                source_rgb_at_tracks, selected_indices, **gather_kwargs
            )
            source_dynamic_mask_at_tracks = gather_identity_payload(
                source_dynamic_mask_at_tracks, selected_indices, **gather_kwargs
            )
            target_dynamic_mask_at_tracks = gather_identity_payload(
                target_dynamic_mask_at_tracks, selected_indices, **gather_kwargs
            )

        persistent_identity_mode = normalize_persistent_identity_mode(
            getattr(pipe, "persistent_identity_mode", "legacy")
        )
        identity_only = persistent_identity_only_enabled(persistent_identity_mode)
        identity_sequence = None
        identity_valid = None
        source_identity_uv = None
        source_identity_vis = None
        if persistent_identity_mode != "legacy":
            identity_sequence, identity_valid = build_persistent_trajectory_inputs(
                tracks_world,
                tracks_world_valid_mask,
            )
            if persistent_identity_mode == "shared":
                if source_cam2world is None or source_intrinsics is None:
                    raise ValueError(
                        "persistent_identity_mode=shared requires source_cam2world "
                        "and source_intrinsics so the same identities can be placed "
                        "on source and target tokens."
                    )
                source_cameras = torch.as_tensor(
                    source_cam2world,
                    device=device,
                    dtype=torch.float32,
                )
                source_k = torch.as_tensor(
                    source_intrinsics,
                    device=device,
                    dtype=torch.float32,
                )
                if (
                    source_k.ndim == 3
                    and source_k.shape[0] == identity_sequence.shape[0]
                    and source_k.shape[0] != identity_sequence.shape[2]
                ):
                    source_k = source_k[:, None].expand(
                        -1, identity_sequence.shape[2], -1, -1
                    )
                source_identity_uv, source_identity_vis, _ = (
                    project_tracks_to_target_views(
                        tracks_world,
                        source_cameras,
                        source_k,
                        image_h,
                        image_w,
                    )
                )
                source_identity_vis = source_identity_vis & identity_valid

        with track4d_record_function("track4d360/track_condition/build_target_view_tracks"):
            track_seq, track_uv_pixel, track_vis = build_target_view_tracks(
                tracks_world,
                target_cam2world,
                target_projection_intrinsics,
                image_h,
                image_w,
                source_depth_at_tracks=source_depth_at_tracks,
                source_depth_valid_at_tracks=source_depth_valid_at_tracks,
                include_reproj_depth=include_reproj_depth,
                source_rgb_at_tracks=source_rgb_at_tracks,
                include_track_anchor=include_track_anchor,
                tracks_world_valid_mask=tracks_world_valid_mask,
                track_coordinate_mode=getattr(pipe, "track_coordinate_mode", "world_xyz"),
                source_dynamic_mask_at_tracks=source_dynamic_mask_at_tracks,
                include_source_dynamic_mask=include_source_dynamic_mask,
                target_dynamic_mask_at_tracks=target_dynamic_mask_at_tracks,
                include_target_dynamic_mask=include_target_dynamic_mask,
                reprojected_rgb_dense=reprojected_rgb_dense,
                reprojected_rgb_valid=reprojected_rgb_valid,
                include_reprojected_rgb=include_reprojected_rgb,
            )
        # track_seq: (B, N, T, C)
        #   C=5 base, C=6 +depth, C=8 +RGB, C=9 +depth+RGB
        #   C=11 +anchor+RGB, C=12 +anchor+depth+RGB
        # track_uv_pixel: (B, N, T, 2)  pixel coords for adapter spatial assignment
        # track_vis: (B, N, T) bool
        if int(track_seq.shape[-1]) != int(track_in_channels):
            raise ValueError(
                "Track condition channel mismatch before adapter: "
                f"builder produced C={track_seq.shape[-1]} but track_adapter.in_channels={track_in_channels}. "
                f"include_reproj_depth={include_reproj_depth} include_source_rgb={include_source_rgb} "
                f"include_track_anchor={include_track_anchor} "
                f"source_depth_at_tracks={'yes' if source_depth_at_tracks is not None else 'no'} "
                f"source_rgb_at_tracks={'yes' if source_rgb_at_tracks is not None else 'no'} "
                f"track_coordinate_mode={getattr(pipe, 'track_coordinate_mode', 'world_xyz')}"
            )
        debug_track_tokens = os.environ.get("HOLO4D_DEBUG_TRACK_TOKENS", "0") == "1"
        if debug_track_tokens:
            with torch.no_grad():
                vis_mean = float(track_vis.float().mean().detach().cpu())
                seq_abs_mean = float(track_seq.abs().mean().detach().cpu())
                seq_abs_max = float(track_seq.abs().amax().detach().cpu())
            debug_track_channels = int(track_seq.shape[-1])
            debug_track_count = int(track_seq.shape[1])
            debug_track_frames = int(track_seq.shape[2])
            print(
                "[Track4D-360/debug] track_seq "
                f"C={debug_track_channels} N={debug_track_count} T={debug_track_frames} "
                f"vis_mean={vis_mean:.6f} abs_mean={seq_abs_mean:.6f} abs_max={seq_abs_max:.6f} "
                f"coord_mode={getattr(pipe, 'track_coordinate_mode', 'world_xyz')} "
                f"source_rgb={'yes' if source_rgb_at_tracks is not None else 'no'}",
                flush=True,
            )

        B = track_seq.shape[0]

        # Sparse/q-budgeted backendmix keeps the dense grid tensor shape in the
        # dataloader for backward compatibility, but the Conv3D adapter's MLP
        # scales with N before the scatter. Compact to identities that are
        # visible in at least one target frame so invalid sparse identities do
        # not allocate q-count-sized activations. Use a batch-union mask so the
        # tensor remains rectangular for B>1.
        if identity_sampling_mode in canonical_methods:
            if isinstance(track_profile_meta, dict):
                track_profile_meta = dict(track_profile_meta)
                track_profile_meta["identity_sampling_mode"] = identity_sampling_mode
                track_profile_meta["raw_compact_n"] = int(
                    (
                        canonical_selection.foreground_candidate_counts
                        + canonical_selection.background_candidate_counts
                    ).max().item()
                )
                track_profile_meta["fixed_track_n"] = int(track_seq.shape[1])
                track_profile_meta["active_track_n"] = int(active_identity_count)
                track_profile_meta["variable_track_n_enabled"] = bool(
                    variable_identity_count.enabled
                )
                track_profile_meta["selected_foreground_counts"] = (
                    canonical_selection.selected_foreground_counts.tolist()
                )
                track_profile_meta["selected_background_counts"] = (
                    canonical_selection.selected_background_counts.tolist()
                )
                track_profile_meta["torch_cluster_foreground_raw_returned_counts"] = (
                    canonical_selection.foreground_raw_returned_counts.tolist()
                )
                track_profile_meta["torch_cluster_background_raw_returned_counts"] = (
                    canonical_selection.background_raw_returned_counts.tolist()
                )
                track_profile_meta["selector_wall_time_ms"] = (
                    canonical_selection.selector_wall_time_ms.tolist()
                )
                track_profile_meta["selector_peak_allocated_bytes"] = (
                    canonical_selection.selector_peak_allocated_bytes.tolist()
                )
                track_profile_meta["canonical_missing_mask_fallback"] = (
                    canonical_missing_mask_fallback
                )
                track_profile_meta["canonical_reference_frame"] = canonical_reference_frame
                track_profile_meta["canonical_foreground_fraction"] = (
                    canonical_foreground_fraction
                )
                track_profile_meta["foreground_unique_cell_counts"] = (
                    canonical_selection.foreground_unique_cell_counts.tolist()
                )
                track_profile_meta["background_unique_cell_counts"] = (
                    canonical_selection.background_unique_cell_counts.tolist()
                )
                track_profile_meta["foreground_cell_collapse_ratios"] = [
                    1.0 - float(unique) / max(float(candidate), 1.0)
                    for unique, candidate in zip(
                        canonical_selection.foreground_unique_cell_counts.tolist(),
                        canonical_selection.foreground_candidate_counts.tolist(),
                    )
                ]
                track_profile_meta["background_cell_collapse_ratios"] = [
                    1.0 - float(unique) / max(float(candidate), 1.0)
                    for unique, candidate in zip(
                        canonical_selection.background_unique_cell_counts.tolist(),
                        canonical_selection.background_candidate_counts.tolist(),
                    )
                ]
                track_profile_meta["selected_query_counts"] = [
                    {
                        int(query): int((row[valid_row] == query).sum().item())
                        for query in torch.unique(row[valid_row], sorted=True)
                    }
                    for row, valid_row in zip(
                        tracks_query_idx_per_identity, canonical_selection.valid
                    )
                ]
                track_profile_meta["selected_padding_counts"] = [
                    int(value) for value in (~canonical_selection.valid).sum(dim=1).tolist()
                ]
        elif identity_sampling_mode == "query_balanced_random":
            if fixed_identity_count <= 0:
                raise ValueError(
                    "HOLO4D_TRACK_IDENTITY_SAMPLING_MODE=query_balanced_random "
                    "requires HOLO4D_TRACK_IDENTITY_FIXED_N > 0"
                )
            raw_compact_n = None
            if isinstance(track_profile_meta, dict):
                # Profiling explicitly opts into this synchronization; the
                # normal training path performs no GPU-to-CPU validity read.
                raw_compact_n = int(
                    track_vis.bool().any(dim=2).sum(dim=1).max().item()
                )
            sample_seed = (
                None if track_identity_sampling_seed is None
                else int(torch.as_tensor(track_identity_sampling_seed).reshape(-1)[0].item())
            )
            track_seq, track_uv_pixel, track_vis, selected_query_ids, selected_source_idx = (
                sample_track_identities_query_balanced(
                    track_seq,
                    track_uv_pixel,
                    track_vis,
                    fixed_identity_count,
                    tracks_query_idx_per_identity,
                    seed=sample_seed,
                )
            )
            if identity_sequence is not None:
                gather_kwargs = {
                    "expected_identity_count": int(identity_sequence.shape[1])
                }
                identity_sequence = gather_identity_payload(
                    identity_sequence, selected_source_idx, **gather_kwargs
                )
                identity_valid = gather_identity_payload(
                    identity_valid, selected_source_idx, **gather_kwargs
                )
                source_identity_uv = gather_identity_payload(
                    source_identity_uv, selected_source_idx, **gather_kwargs
                )
                source_identity_vis = gather_identity_payload(
                    source_identity_vis, selected_source_idx, **gather_kwargs
                )
            if isinstance(track_profile_meta, dict):
                track_profile_meta = dict(track_profile_meta)
                track_profile_meta["identity_sampling_mode"] = identity_sampling_mode
                track_profile_meta["raw_compact_n"] = raw_compact_n
                track_profile_meta["fixed_track_n"] = int(track_seq.shape[1])
                track_profile_meta["selected_query_counts"] = [
                    {
                        int(q): int((row == q).sum().item())
                        for q in torch.unique(row[row >= 0], sorted=True)
                    }
                    for row in selected_query_ids
                ]
                track_profile_meta["selected_padding_counts"] = [
                    int((row < 0).sum().item()) for row in selected_source_idx
                ]
        else:
            if identity_sampling_mode not in {"legacy_uniform_order", "legacy", "uniform_order"}:
                raise ValueError(
                    "Unsupported HOLO4D_TRACK_IDENTITY_SAMPLING_MODE="
                    f"{identity_sampling_mode!r}"
                )
            track_seq, track_uv_pixel, track_vis, identity_keep = compact_track_tensors_by_visibility(
                track_seq, track_uv_pixel, track_vis,
            )
            if identity_sequence is not None:
                identity_sequence = identity_sequence[:, identity_keep]
                identity_valid = identity_valid[:, identity_keep]
                if source_identity_uv is not None:
                    source_identity_uv = source_identity_uv[:, identity_keep]
                    source_identity_vis = source_identity_vis[:, identity_keep]
            # Optional dense-tracking robustness augmentation: train-time-only,
            # drops a per-call random fraction (sampled from
            # [HOLO4D_TRACK_RANDOM_DROPOUT_MIN, _MAX]) of whole track
            # identities' visibility, mirroring "drop the bottom-confidence
            # fraction of a dense candidate pool" at inference. Defaults to
            # off (0.0/0.0) for backward compatibility.
            dropout_min = float(os.environ.get("HOLO4D_TRACK_RANDOM_DROPOUT_MIN", "0.0") or 0.0)
            dropout_max = float(os.environ.get("HOLO4D_TRACK_RANDOM_DROPOUT_MAX", "0.0") or 0.0)
            if dropout_max > 0.0 and bool(getattr(pipe.scheduler, "training", False)):
                track_vis = apply_random_track_point_dropout(
                    track_vis, dropout_min, dropout_max, seed=count_sampling_seed,
                )
            raw_compact_n = int(track_seq.shape[1])
            all_invisible = identity_keep.numel() > 0 and not bool(identity_keep.any())

        if (
            fixed_identity_count > 0
            and identity_sampling_mode != "query_balanced_random"
            and identity_sampling_mode not in canonical_methods
        ):
            auxiliary_fixed_indices = None
            if identity_sequence is not None:
                current_n = int(identity_sequence.shape[1])
                if current_n > fixed_identity_count:
                    row = torch.linspace(
                        0,
                        current_n - 1,
                        fixed_identity_count,
                        device=identity_sequence.device,
                    ).round().long().clamp_(0, current_n - 1)
                else:
                    row = torch.arange(
                        current_n,
                        device=identity_sequence.device,
                        dtype=torch.long,
                    )
                    if current_n < fixed_identity_count:
                        row = torch.cat(
                            [
                                row,
                                torch.full(
                                    (fixed_identity_count - current_n,),
                                    -1,
                                    device=row.device,
                                    dtype=torch.long,
                                ),
                            ]
                        )
                auxiliary_fixed_indices = row.unsqueeze(0).expand(
                    identity_sequence.shape[0], -1
                )
            # Fixed-N path: ALWAYS route through the adapter with the same
            # (B, fixed_N, T, C) shape — including the all-invisible case. When
            # nothing is visible, compact_track_tensors_by_visibility already
            # produced an N=0 tensor, and fix_track_identity_count pads it up to
            # fixed_N with vis=False rows. The adapter renders those as all-null
            # occupancy (every block falls back to its null token), which is
            # numerically identical to the _make_track_null_tokens() early
            # return — but keeps the forward graph shape-consistent across FSDP
            # ranks. Without this, a rank that drew an all-invisible sample would
            # take the null-token early return while its peers ran the adapter,
            # diverging the collective sequence → hang / shape mismatch.
            track_seq, track_uv_pixel, track_vis = fix_track_identity_count(
                track_seq,
                track_uv_pixel,
                track_vis,
                fixed_identity_count,
            )
            if auxiliary_fixed_indices is not None:
                gather_kwargs = {
                    "expected_identity_count": int(identity_sequence.shape[1])
                }
                identity_sequence = gather_identity_payload(
                    identity_sequence, auxiliary_fixed_indices, **gather_kwargs
                )
                identity_valid = gather_identity_payload(
                    identity_valid, auxiliary_fixed_indices, **gather_kwargs
                )
                source_identity_uv = gather_identity_payload(
                    source_identity_uv, auxiliary_fixed_indices, **gather_kwargs
                )
                source_identity_vis = gather_identity_payload(
                    source_identity_vis, auxiliary_fixed_indices, **gather_kwargs
                )
            if isinstance(track_profile_meta, dict):
                track_profile_meta = dict(track_profile_meta)
                track_profile_meta["raw_compact_n"] = raw_compact_n
                track_profile_meta["fixed_track_n"] = int(track_seq.shape[1])
        elif (
            identity_sampling_mode != "query_balanced_random"
            and identity_sampling_mode not in canonical_methods
            and all_invisible
        ):
            # Legacy path (fixed-N disabled): no visible identities → inject null
            # tokens directly. NOT FSDP-safe if ranks diverge here; enable
            # HOLO4D_TRACK_IDENTITY_FIXED_N for shape-stable distributed
            # training.
            track_point_tokens = _make_track_null_tokens()
            if track_point_tokens is None:
                return {}
            return _build_injection_payload(track_point_tokens)
        adapter_type = getattr(pipe, 'track_adapter_type', 'scatter')
        if profile_call_id is not None:
            _track4d_log_adapter_profile(
                "after_compact_cpu",
                device,
                call_id=profile_call_id,
                adapter_type=adapter_type,
                image_h=image_h,
                image_w=image_w,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                track_seq=track_seq,
                track_vis=track_vis,
                track_profile_meta=track_profile_meta,
                adapter=getattr(pipe, "track_adapter", None),
            )

        # 2. Move to device
        track_seq = track_seq.to(device=device, dtype=dtype)
        track_uv_pixel = track_uv_pixel.to(device=device, dtype=dtype)
        track_vis = track_vis.to(device=device)
        if identity_sequence is not None:
            identity_sequence = identity_sequence.to(device=device, dtype=dtype)
            identity_valid = identity_valid.to(device=device, dtype=torch.bool)
            if source_identity_uv is not None:
                source_identity_uv = source_identity_uv.to(device=device, dtype=dtype)
                source_identity_vis = source_identity_vis.to(
                    device=device, dtype=torch.bool
                )
        # Future prediction: zero track visibility on FUTURE video frames so the
        # adapter's per-block null_token fills future latent-frame blocks (genuine
        # per-latent-frame null via visibility, not the per-sample has_tracks
        # gate). No-op when future prediction is off → default path unchanged.
        if bool(future_prediction) and int(future_num_latent_frames) > 0:
            from holo4d.future_prediction import future_pixel_mask
            _latent_t = (int(T) - 1) // 4 + 1
            _n_fut = int(future_num_latent_frames)
            if 0 < _n_fut < _latent_t:
                _obs = torch.ones(int(track_vis.shape[0]), _latent_t, dtype=torch.bool, device=track_vis.device)
                _obs[:, _latent_t - _n_fut:] = False
                _keep = (~future_pixel_mask(_obs, int(T))).unsqueeze(1)  # (B,1,T) True=observed
                track_vis = torch.where(_keep, track_vis, torch.zeros_like(track_vis))
        if profile_call_id is not None:
            _track4d_log_adapter_profile(
                "after_to_device",
                device,
                call_id=profile_call_id,
                adapter_type=adapter_type,
                image_h=image_h,
                image_w=image_w,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                track_seq=track_seq,
                track_vis=track_vis,
                track_profile_meta=track_profile_meta,
                adapter=getattr(pipe, "track_adapter", None),
            )

        # Opt-in exact adapter-input capture for the null-token collision
        # probe. Capture after fixed-N selection, device conversion, and
        # future-visibility masking, immediately before the adapter call.
        # Keeping this behind an environment flag avoids the CPU transfer and
        # memory cost in normal train/eval paths.
        if _track4d_env_flag("HOLO4D_CAPTURE_TRACK_ADAPTER_INPUTS", "0"):
            pipe._debug_last_track_adapter_inputs = {
                "track_seq": track_seq.detach().cpu(),
                "track_uv": track_uv_pixel.detach().cpu(),
                "track_vis": track_vis.detach().cpu(),
                "image_h": int(image_h),
                "image_w": int(image_w),
                "num_frames": int(T),
                "adapter_type": str(adapter_type),
                "source": "post-selection-pre-adapter",
            }

        # Compute latent dimensions after VAE + patchify.
        # Wan 2.1 VAE (WanVideoVAE): spatial 8x.  Wan 2.2 (WanVideoVAE38): spatial 16x.
        # DiT patchify (Conv3d stride [1,2,2]): additional spatial 2x.
        # Total spatial division = vae.upsampling_factor * 2  (16 for Wan 2.1, 32 for Wan 2.2).
        # Track tokens must align with DiT's post-patchify grid.
        # 3. Embed + Adapt to blocks
        if adapter_type == "conv3d_patchify":
            # Conv3dTrackAdapter: unified encoder — takes track_seq directly,
            # replaces both embedder and adapter in a single forward pass.
            if profile_call_id is not None:
                _track4d_log_adapter_profile(
                    "before_adapter",
                    device,
                    call_id=profile_call_id,
                    adapter_type=adapter_type,
                    image_h=image_h,
                    image_w=image_w,
                    latent_t=latent_t,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    track_seq=track_seq,
                    track_vis=track_vis,
                    track_profile_meta=track_profile_meta,
                    adapter=pipe.track_adapter,
                )
            with track4d_record_function("track4d360/track_condition/adapter_forward"):
                block_tokens, block_mask = pipe.track_adapter(
                    track_seq,
                    track_uv_pixel,
                    track_vis,
                    image_h,
                    image_w,
                    T,
                    latent_t,
                    latent_h,
                    latent_w,
                )
            if debug_track_tokens:
                with torch.no_grad():
                    token_abs_mean = float(block_tokens.abs().mean().detach().cpu())
                    token_abs_max = float(block_tokens.abs().amax().detach().cpu())
                    mask_mean = float(block_mask.float().mean().detach().cpu())
                print(
                    "[Track4D-360/debug] block_tokens "
                    f"shape={tuple(block_tokens.shape)} mask_mean={mask_mean:.6f} "
                    f"abs_mean={token_abs_mean:.6f} abs_max={token_abs_max:.6f}",
                    flush=True,
                )
            if profile_call_id is not None:
                _track4d_log_adapter_profile(
                    "after_adapter",
                    device,
                    call_id=profile_call_id,
                    adapter_type=adapter_type,
                    image_h=image_h,
                    image_w=image_w,
                    latent_t=latent_t,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    track_seq=track_seq,
                    track_vis=track_vis,
                    track_profile_meta=track_profile_meta,
                    adapter=pipe.track_adapter,
                )
        else:
            # Standard two-stage: PointTrackEmbedder → PointTrackAdapter/DenseTrackConvAdapter
            with track4d_record_function("track4d360/track_condition/embedder_forward"):
                track_embeddings = pipe.track_embedder(track_seq, track_vis)
            # (B, N, embed_dim)
            with track4d_record_function("track4d360/track_condition/adapter_forward"):
                block_tokens, block_mask = pipe.track_adapter(
                    track_embeddings,
                    track_uv_pixel,
                    track_vis,
                    image_h,
                    image_w,
                    T,
                    latent_t,
                    latent_h,
                    latent_w,
                )
        # block_tokens: (B, num_blocks, adapter_out_dim)

        # 5. Expand to target-view latent sequence
        with track4d_record_function("track4d360/track_condition/expand_tokens"):
            track_point_tokens = expand_block_tokens_to_latent_seq(
                block_tokens,
                latent_t,
                latent_h,
                latent_w,
                pipe.track_adapter.block_t,
                pipe.track_adapter.block_h,
                pipe.track_adapter.block_w,
            )
        # (B, L_target, track_dim) where L_target = latent_t * latent_h * latent_w

        if track_occupancy_channel_enabled():
            track_occupancy_seq = expand_block_tokens_to_latent_seq(
                block_mask.unsqueeze(-1).to(dtype=block_tokens.dtype),
                latent_t,
                latent_h,
                latent_w,
                pipe.track_adapter.block_t,
                pipe.track_adapter.block_h,
                pipe.track_adapter.block_w,
            )
            # (B, L_target, 1) — 1.0 where any selected track occupies the block

        source_track_point_tokens = None
        if persistent_identity_mode != "legacy":
            identity_adapter = getattr(pipe, "track_identity_adapter", None)
            if identity_adapter is None:
                raise RuntimeError(
                    "Persistent identity mode is enabled but "
                    "pipe.track_identity_adapter is not initialized."
                )
            (
                target_identity_tokens,
                target_identity_mask,
                source_identity_tokens,
                source_identity_mask,
            ) = identity_adapter(
                identity_sequence,
                identity_valid,
                track_uv_pixel,
                track_vis,
                image_h=image_h,
                image_w=image_w,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                source_uv=source_identity_uv,
                source_vis=source_identity_vis,
            )
            target_identity_tokens = target_identity_tokens.to(
                device=track_point_tokens.device,
                dtype=track_point_tokens.dtype,
            )
            track_point_tokens = compose_target_identity_tokens(
                track_point_tokens,
                target_identity_tokens,
                identity_only=identity_only,
            )
            if identity_only and track_occupancy_channel_enabled():
                track_occupancy_seq = target_identity_mask.unsqueeze(-1).to(
                    dtype=track_point_tokens.dtype,
                    device=track_point_tokens.device,
                )
            if persistent_identity_mode == "shared":
                if source_identity_tokens is None:
                    raise RuntimeError(
                        "Shared persistent identity did not produce source tokens."
                    )
                source_identity_tokens = source_identity_tokens.to(
                    device=track_point_tokens.device,
                    dtype=track_point_tokens.dtype,
                )
                legacy_source_null = pipe.track_adapter.null_token.to(
                    device=track_point_tokens.device,
                    dtype=track_point_tokens.dtype,
                ).view(1, 1, -1).expand_as(source_identity_tokens)
                # The learned identity map is a delta over the exact legacy
                # source-null baseline. Because point_mlp is zero-initialized,
                # enabling shared mode on an old checkpoint is initially a
                # strict no-op rather than replacing its trained null token.
                source_track_point_tokens = compose_source_identity_tokens(
                    legacy_source_null,
                    source_identity_tokens,
                    identity_only=identity_only,
                )
                if track_occupancy_channel_enabled() and source_identity_mask is not None:
                    # (B, L) bool from the identity rasterizer, already at
                    # latent-sequence resolution — 1.0 where a projected
                    # identity actually lands on a source block.
                    source_occupancy_seq = source_identity_mask.unsqueeze(-1).to(
                        dtype=source_track_point_tokens.dtype,
                        device=source_track_point_tokens.device,
                    )
            if _track4d_env_flag(
                "HOLO4D_CAPTURE_PERSISTENT_IDENTITY_FOR_VIZ", "0"
            ):
                pipe._debug_last_persistent_identity = {
                    "mode": persistent_identity_mode,
                    "identity_only": identity_only,
                    "target_mask": target_identity_mask.detach().cpu(),
                    "source_mask": (
                        None
                        if source_identity_mask is None
                        else source_identity_mask.detach().cpu()
                    ),
                    "target_uv": track_uv_pixel.detach().cpu(),
                    "target_vis": track_vis.detach().cpu(),
                    "source_uv": (
                        None
                        if source_identity_uv is None
                        else source_identity_uv.detach().cpu()
                    ),
                    "source_vis": (
                        None
                        if source_identity_vis is None
                        else source_identity_vis.detach().cpu()
                    ),
                }

        # 5a. Use the same relative token-noise knob as the dense branch so
        # track conditioning learns robustness to imperfect inference tracks.
        # The following has_tracks gate replaces dropped samples with the null
        # token afterward, so noise is only retained for real track conditions.
        _noise_max = float(getattr(pipe, "dense_conditioning_noise_max_level", 0.0))
        _training = bool(getattr(getattr(pipe, "scheduler", None), "training", False))
        track_point_tokens = _apply_condition_token_noise(
            track_point_tokens,
            _noise_max,
            training=_training,
        )

        # 5b. Swap dropped/missing tracks to the adapter's learned null token
        #     without changing the graph path. Keep a zero-gradient connection
        #     to the normal encoder path so DDP still sees those parameters as
        #     used when different ranks make independent dropout decisions.
        #     `has_tracks` is an explicit process() arg (see signature above).
        adapter_null = getattr(pipe.track_adapter, "null_token", None)
        if adapter_null is not None:
            adapter_null = adapter_null.to(
                device=track_point_tokens.device,
                dtype=track_point_tokens.dtype,
            ).view(1, 1, -1).expand_as(track_point_tokens)
        if torch.is_tensor(has_tracks):
            if has_tracks.ndim == 0:
                if not has_tracks.item():
                    if adapter_null is not None:
                        track_point_tokens = track_point_tokens * 0.0 + adapter_null
                    else:
                        track_point_tokens = track_point_tokens * 0.0
            else:
                mask = has_tracks.bool().to(track_point_tokens.device)
                mask_view = mask.view(-1, 1, 1).to(dtype=track_point_tokens.dtype)
                dropped_view = 1.0 - mask_view
                if adapter_null is not None:
                    track_point_tokens = track_point_tokens * mask_view + adapter_null * dropped_view
                else:
                    track_point_tokens = track_point_tokens * mask_view
        elif not has_tracks:
            if adapter_null is not None:
                track_point_tokens = track_point_tokens * 0.0 + adapter_null
            else:
                track_point_tokens = track_point_tokens * 0.0

        if source_track_point_tokens is not None:
            if adapter_null is None:
                source_null = torch.zeros_like(source_track_point_tokens)
            else:
                source_null = adapter_null[:, :1].expand_as(
                    source_track_point_tokens
                )
            if torch.is_tensor(has_tracks):
                if has_tracks.ndim == 0:
                    if not has_tracks.item():
                        source_track_point_tokens = (
                            source_track_point_tokens * 0.0 + source_null
                        )
                else:
                    source_mask = has_tracks.bool().to(
                        source_track_point_tokens.device
                    ).view(-1, 1, 1)
                    source_track_point_tokens = torch.where(
                        source_mask,
                        source_track_point_tokens,
                        source_null,
                    )
            elif not has_tracks:
                source_track_point_tokens = (
                    source_track_point_tokens * 0.0 + source_null
                )

        # 5c. Occupancy must mirror the has_tracks dropout: a dropped sample's
        # condition is the null token everywhere, so its occupancy is 0.
        # Applies to BOTH target and shared-identity source occupancy (F6).
        def _apply_has_tracks_to_occupancy(occ_seq):
            if occ_seq is None:
                return None
            if torch.is_tensor(has_tracks):
                if has_tracks.ndim == 0:
                    if not has_tracks.item():
                        return torch.zeros_like(occ_seq)
                    return occ_seq
                _occ_mask = (
                    has_tracks.bool()
                    .to(occ_seq.device)
                    .view(-1, 1, 1)
                    .to(dtype=occ_seq.dtype)
                )
                return occ_seq * _occ_mask
            if not has_tracks:
                return torch.zeros_like(occ_seq)
            return occ_seq

        track_occupancy_seq = _apply_has_tracks_to_occupancy(track_occupancy_seq)
        source_occupancy_seq = _apply_has_tracks_to_occupancy(source_occupancy_seq)

        # 6. Align track tokens to full V2V sequence (pad source domain with null)
        # NOTE: helper lifted to module scope as ``_align_track_to_sequence`` so
        # the dense-control unit can reuse it without copy/paste.
        return _build_injection_payload(
            track_point_tokens,
            source_track_point_tokens=source_track_point_tokens,
        )


# =========================================================================
#
# WanVideoUnit_DenseTargetControl + helpers (_align_track_to_sequence,
# _normalize_warped_depth_per_frame) live in
# holo4d/conditions/dense_target_control_unit.py and are re-exported at
# the top of this file.  This indirection lets the test suite import them
# without triggering ``holo4d_runtime/pipeline/__init__.py`` (which pulls in
# diffusers / deepspeed and dies in CUDA-less envs).
#
# The block that previously defined them inline is intentionally removed; the
# next line below is the original cut point so the rest of this file stays
# unchanged.
#
# =========================================================================



# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class WanVideoTrack4D360Pipeline(WanVideoPipelineV2V):
    """
    Track4D-360 pipeline: V2V with point-track injection.

    Inherits the full V2V pipeline (WanVideoPipeline from wan_video_v2v)
    and adds:
      - track_embedder (PointTrackEmbedder)
      - track_adapter (PointTrackAdapter)
      - track_projector (TokenConcatProjector)
      - WanVideoUnit_TargetVideoEncoder
      - WanVideoUnit_TrackPointCondition
    """

    default_training_task = "sft:v2v"

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16,
                 joint_track_prediction=False,
                 joint_depth_prediction=False,
                 joint_track_depth_prediction=False):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.channel_concat_mode_cfg = None
        self.joint_track_prediction = bool(joint_track_prediction or joint_track_depth_prediction)
        self.joint_depth_prediction = bool(joint_depth_prediction)
        self.joint_track_depth_prediction = bool(joint_track_depth_prediction)
        self.track_conditioning_enabled = True
        self.track_embedder = None
        self.track_adapter = None
        self.track_identity_adapter = None
        self.persistent_identity_mode = "legacy"
        self.persistent_identity_only = False
        self.track_projector = None
        self.track_block_injector = None
        self.track_null_token = None
        # Dense GEN3C-style target-space control branch (off until
        # ``initialize_dense_target_control_encoder`` is called).
        self.dense_target_control_encoder = None
        self.dense_branch_alpha_track = 1.0
        # Opt-in Vista4D-style continuous noise augmentation on dense conditioning
        # tokens. 0.0 = off (backward-compatible). Set from args at train init.
        self.dense_conditioning_noise_max_level = 0.0
        self.dense_null_token = None

        # Override unit list for Track4D-360
        units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ModeDropoutAugmentation(),
            WanVideoUnit_InputVideoEmbedder_XYZCondition(),
            WanVideoUnit_LatentNoiseAugmentation(),
            WanVideoUnit_CameraControl_NVS(),
            WanVideoUnit_TargetVideoEncoder(),
        ]

        if self.joint_track_prediction:
            units.append(WanVideoUnit_TargetTrackEncoder())
            units.append(WanVideoUnit_JointTrackRemap())
        if self.joint_depth_prediction or self.joint_track_depth_prediction:
            units.append(WanVideoUnit_TargetDepthEncoder())

        # Warped appearance unit: soft-splat warp + VAE encode (Lyra-2-inspired).
        # No-op when warped_condition_mode == "off" (default).
        from holo4d.conditions.warped_appearance_unit import WanVideoUnit_WarpedAppearance
        units.append(WanVideoUnit_WarpedAppearance())

        units.extend([
            WanVideoUnit_TrackPointCondition(),
            # Dense target-space control unit MUST run after the persistent
            # track unit so it can fuse with ``track_point_tokens``.  It is a
            # graceful no-op when ``use_dense_branch=False`` or the dense
            # encoder hasn't been initialised.
            WanVideoUnit_Vista4DStreams(),
            WanVideoUnit_DenseTargetControl(),
            WanVideoUnit_V2VConcatenate(),
            # No-op unless use_target_image_condition=True. It must run after
            # V2VConcatenate so it can preserve the current source video in
            # source domain 0 and add the boundary-image condition to target 1.
            WanVideoUnit_TargetImageCondition(),
            # No-op unless use_ti2v_native_i2v=True. Unlike the Wan 2.1 unit
            # above, this emits a clean target-domain frame-0 latent and no
            # CLIP/extra-channel condition.
            WanVideoUnit_TI2VNativeFirstFrameCondition(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ])
        self.units = units

    def disable_track_conditioning_units(self):
        """Remove persistent/dense Track4D units for raw plucker-only V2V runs."""
        kept_units = []
        removed = 0
        for unit in self.units:
            if isinstance(unit, (WanVideoUnit_TrackPointCondition, WanVideoUnit_DenseTargetControl)):
                removed += 1
                continue
            kept_units.append(unit)
        self.units = kept_units
        self.track_conditioning_enabled = False
        if removed:
            _rank0_print(f"[Track4D-360] Disabled track conditioning units ({removed} removed)")

    def disable_persistent_track_conditioning_unit(self):
        """Remove only the persistent 3D-track input unit.

        Dense-only ablations keep ``WanVideoUnit_DenseTargetControl`` but must
        not run ``WanVideoUnit_TrackPointCondition``, otherwise the batch's
        ``tracks_world`` would still be encoded as a conditioning modality.
        """
        kept_units = []
        removed = 0
        for unit in self.units:
            if isinstance(unit, WanVideoUnit_TrackPointCondition):
                removed += 1
                continue
            kept_units.append(unit)
        self.units = kept_units
        if removed:
            _rank0_print(f"[Track4D-360] Disabled persistent track conditioning unit ({removed} removed)")

    def _num_v2v_domains(self):
        if self.joint_track_depth_prediction:
            return 4
        if self.joint_depth_prediction:
            return 3
        if self.joint_track_prediction:
            return 3
        return 2

    def enable_joint_track_prediction(self):
        """Enable Stage 2: joint NVS + 3D tracking (3-domain V2V).

        Inserts TargetTrackEncoder and JointTrackRemap units into the pipeline.
        Can be called after from_pretrained() to upgrade a 2-domain pipeline to 3-domain.
        """
        if self.joint_track_prediction:
            return  # already enabled
        self.channel_concat_mode_cfg = None
        self.joint_track_prediction = True

        # Find the position of TrackPointCondition and insert before it
        insert_idx = None
        for i, unit in enumerate(self.units):
            if isinstance(unit, WanVideoUnit_TrackPointCondition):
                insert_idx = i
                break
        if insert_idx is None:
            raise RuntimeError("Cannot find WanVideoUnit_TrackPointCondition in pipeline units")

        self.units.insert(insert_idx, WanVideoUnit_JointTrackRemap())
        self.units.insert(insert_idx, WanVideoUnit_TargetTrackEncoder())
        _rank0_print("[Track4D-360] Enabled joint_track_prediction (3-domain V2V)")

    def enable_joint_depth_prediction(self):
        """Enable Stage 2: joint NVS + depth only (3-domain V2V)."""
        if self.joint_depth_prediction:
            return
        self.channel_concat_mode_cfg = None
        self.joint_depth_prediction = True
        self.joint_track_prediction = False
        self.joint_track_depth_prediction = False

        insert_idx = None
        for i, unit in enumerate(self.units):
            if isinstance(unit, WanVideoUnit_TrackPointCondition):
                insert_idx = i
                break
        if insert_idx is None:
            raise RuntimeError("Cannot find WanVideoUnit_TrackPointCondition in pipeline units")
        if not any(isinstance(u, WanVideoUnit_TargetDepthEncoder) for u in self.units):
            self.units.insert(insert_idx, WanVideoUnit_TargetDepthEncoder())
        _rank0_print("[Track4D-360] Enabled joint_depth_prediction (3-domain RGB+depth V2V)")

    def enable_joint_track_depth_prediction(self):
        """Enable Stage 2 multimodal target prediction: RGB + target pointmap + target depth."""
        if self.joint_track_depth_prediction:
            return
        self.channel_concat_mode_cfg = None
        if not self.joint_track_prediction:
            self.enable_joint_track_prediction()
        self.joint_depth_prediction = False
        self.joint_track_prediction = True
        self.joint_track_depth_prediction = True

        insert_idx = None
        for i, unit in enumerate(self.units):
            if isinstance(unit, WanVideoUnit_TrackPointCondition):
                insert_idx = i
                break
        if insert_idx is None:
            raise RuntimeError("Cannot find WanVideoUnit_TrackPointCondition in pipeline units")
        self.units.insert(insert_idx, WanVideoUnit_TargetDepthEncoder())
        _rank0_print("[Track4D-360] Enabled joint_track_depth_prediction (4-domain V2V)")

    def enable_channel_concat(self, mode_cfg: dict):
        """Enable channel-concat multi-modal V2V (Stage A: rgb+depth, Stage B: rgb+depth+track)."""
        stage = mode_cfg.get("channel_concat_stage", "A")
        modalities = mode_cfg.get("channel_modalities", ["rgb", "depth"])
        self.channel_concat_mode_cfg = dict(mode_cfg)
        self.joint_track_prediction = False
        self.joint_depth_prediction = False
        self.joint_track_depth_prediction = False
        units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ModeDropoutAugmentation(),
            WanVideoUnit_InputVideoEmbedder_XYZCondition(),
            WanVideoUnit_LatentNoiseAugmentation(),
            WanVideoUnit_TargetVideoEncoder(),
            WanVideoUnit_SourceMultiModalEncoder(),
            WanVideoUnit_TargetDepthEncoder(),
        ]
        if "residual_track" in modalities:
            units.append(WanVideoUnit_TargetResidualTrackEncoder())
        units.extend([
            WanVideoUnit_TrackPointCondition(),
            WanVideoUnit_DenseTargetControl(),
            WanVideoUnit_ChannelConcatV2V(),
            WanVideoUnit_V2VConcatenate(),   # fallback: runs only when channel_concat=False
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ])
        self.units = units
        _rank0_print(f"[Track4D-360] Enabled channel-concat: stage={stage}, modalities={modalities}")

    def _decode_channel_concat_target_latents(
        self,
        target_latents,
        *,
        tiled=False,
        tile_size=None,
        tile_stride=None,
        track_normalization_stats=None,
        depth_normalization_stats=None,
    ):
        """Decode channel-concat target latents into modality-specific outputs."""
        mode_cfg = getattr(self, "channel_concat_mode_cfg", None) or {}
        modalities = list(mode_cfg.get("channel_modalities", ["rgb", "depth"]))
        if not modalities:
            raise ValueError("channel_concat_mode_cfg must define at least one modality")

        base_latent_c = int(getattr(getattr(self, "dit", None), "in_dim", 0))
        if base_latent_c <= 0:
            if target_latents.shape[1] % len(modalities) != 0:
                raise ValueError(
                    f"Cannot infer base latent channels from target latents shape {tuple(target_latents.shape)} "
                    f"for modalities={modalities}"
                )
            base_latent_c = target_latents.shape[1] // len(modalities)

        expected_c = base_latent_c * len(modalities)
        if target_latents.shape[1] != expected_c:
            raise ValueError(
                f"Channel-concat target latents have {target_latents.shape[1]} channels, "
                f"expected {expected_c} for modalities={modalities} and base_latent_c={base_latent_c}"
            )

        outputs = {
            "channel_modalities": modalities,
        }
        offset = 0
        for modality in modalities:
            modality_latents = target_latents[:, offset: offset + base_latent_c]
            offset += base_latent_c
            decoded = self.vae.decode(
                modality_latents,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if modality == "rgb":
                outputs["video"] = self.vae_output_to_video(decoded)
            elif modality == "depth":
                outputs["depth_tensor"] = decoded.mean(dim=1, keepdim=True)
                outputs["depth_normalization_stats"] = depth_normalization_stats
            elif modality == "residual_track":
                outputs["track_tensor"] = decoded
                outputs["track_normalization_stats"] = track_normalization_stats
            else:
                outputs[f"{modality}_tensor"] = decoded
        return outputs

    def _derive_track_block_sizes(self):
        """Track adapter block sizes in latent space.

        Following Koichi's design: 1:1 block-to-latent-token mapping.
        Each latent position (after VAE 4x temporal, 8x spatial + DiT 2x patchify)
        is one block = one adapter output token. No further grouping.
        """
        block_t = 1
        block_h = 1
        block_w = 1
        return block_t, block_h, block_w

    def initialize_track_modules(
        self,
        track_in_channels=5,
        track_embed_dim=156,
        track_num_freqs=16,
        track_hidden_dim=156,
        track_num_mlp_layers=3,
        dit_dim=1536,
        track_injection_mode="single",
        track_injection_block_mode="additive",
        adapter_type="scatter",
        conv_proj_dim=32,
        conv_track_stride=4,
        conv3d_num_residual_blocks=1,
        track_coordinate_mode="world_xyz",
        temporal_pool="max",
        temporal_attn_heads=4,
        track_include_flags=None,
        persistent_identity_mode="legacy",
        persistent_identity_dim=128,
    ):
        """
        Create and register track conditioning modules.

        Called by the training module BEFORE switch_pipe_to_training_mode()
        so that the modules exist for freeze/unfreeze and checkpoint loading.
        block_t/h/w are auto-derived from the model architecture.

        Args:
            track_injection_mode: "single" (one projector before all blocks) or
                "per_block" (encoder+projector at each DiT block, ReCamMaster-style).
            adapter_type: "scatter" (PointTrackAdapter, Koichi-style max-pool),
                "conv_patchify" (DenseTrackConvAdapter, learnable Conv2d patchify), or
                "conv3d_patchify" (Conv3dTrackAdapter, causal Conv3d, replaces both
                embedder and adapter).
            conv_proj_dim: intermediate channel dim for conv_patchify / conv3d_patchify.
            conv_track_stride: track spatial stride (for computing conv patch_size
                = spatial_div // track_stride). Used for conv_patchify/conv3d_patchify.
            conv3d_num_residual_blocks: number of CausalResidualBlock3d after patchify
                (only for conv3d_patchify).
        """
        self.track_injection_mode = track_injection_mode
        self.track_adapter_type = adapter_type
        self.track_coordinate_mode = str(track_coordinate_mode or "world_xyz")
        self.persistent_identity_mode = normalize_persistent_identity_mode(
            persistent_identity_mode
        )
        self.persistent_identity_only = persistent_identity_only_enabled(
            self.persistent_identity_mode
        )
        if (
            self.persistent_identity_mode != "legacy"
            and self.track_coordinate_mode.lower()
            not in {"world", "world_xyz", "cam0", "cam0_xyz"}
        ):
            raise ValueError(
                "Persistent trajectory identity requires target-camera-invariant "
                "coordinate_mode=world_xyz."
            )
        # Explicit per-channel include flags from the track config (new path). When
        # None, build_track_condition falls back to the legacy channel-count map so
        # existing 9/11/12-channel configs + checkpoints are byte-identical.
        self.track_include_flags = dict(track_include_flags) if track_include_flags else None
        block_t, block_h, block_w = self._derive_track_block_sizes()

        spatial_div = getattr(self, 'height_division_factor', 16)
        conv_patch_size = spatial_div // conv_track_stride

        if adapter_type == "conv3d_patchify":
            # Conv3dTrackAdapter: unified encoder — replaces both embedder and adapter.
            # No separate track_embedder needed.
            self.track_embedder = None
            self.track_adapter = Conv3dTrackAdapter(
                in_channels=track_in_channels,
                num_freqs=track_num_freqs,
                hidden_dim=track_hidden_dim,
                out_dim=track_embed_dim,
                num_mlp_layers=track_num_mlp_layers,
                proj_dim=conv_proj_dim,
                patch_size=conv_patch_size,
                temporal_stride=4,
                num_residual_blocks=conv3d_num_residual_blocks,
                block_t=block_t,
                block_h=block_h,
                block_w=block_w,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            self.track_embedder = PointTrackEmbedder(
                in_channels=track_in_channels,
                num_freqs=track_num_freqs,
                hidden_dim=track_hidden_dim,
                out_dim=track_embed_dim,
                num_mlp_layers=track_num_mlp_layers,
                temporal_pool=temporal_pool,
                temporal_attn_heads=temporal_attn_heads,
            ).to(device=self.device, dtype=self.torch_dtype)

            if adapter_type == "conv_patchify":
                self.track_adapter = DenseTrackConvAdapter(
                    track_dim=track_embed_dim,
                    out_dim=track_embed_dim,
                    proj_dim=conv_proj_dim,
                    patch_size=conv_patch_size,
                    block_t=block_t,
                    block_h=block_h,
                    block_w=block_w,
                ).to(device=self.device, dtype=self.torch_dtype)
            else:
                self.track_adapter = PointTrackAdapter(
                    track_dim=track_embed_dim,
                    out_dim=track_embed_dim,
                    block_t=block_t,
                    block_h=block_h,
                    block_w=block_w,
                ).to(device=self.device, dtype=self.torch_dtype)

        self.track_identity_adapter = None
        if self.persistent_identity_mode != "legacy":
            self.track_identity_adapter = PersistentTrajectoryIdentityAdapter(
                out_dim=track_embed_dim,
                identity_dim=int(persistent_identity_dim),
                hidden_dim=track_hidden_dim,
            ).to(device=self.device, dtype=self.torch_dtype)

        # --- Injection modules ---
        # Opt-in structural occupancy channel: the injected track sequence
        # carries one extra binary channel (observed / unobserved block), so
        # the injector input dim grows by 1. Default off = legacy shapes, so
        # legacy checkpoints load unchanged; occupancy-mode checkpoints must be
        # loaded with the same env flag (shape mismatch fails loudly otherwise).
        occupancy_extra = 1 if track_occupancy_channel_enabled() else 0
        injector_track_dim = track_embed_dim + occupancy_extra
        if occupancy_extra:
            _rank0_print(
                "[Track4D-360] HOLO4D_TRACK_OCCUPANCY_CHANNEL=1 -> "
                f"injector track_dim={injector_track_dim} (+1 structural occupancy)"
            )
        if track_injection_mode == "per_block":
            num_blocks = len(self.dit.blocks)
            self.track_block_injector = PerBlockTrackInjector(
                num_blocks=num_blocks,
                track_dim=injector_track_dim,
                dit_dim=dit_dim,
                mode=track_injection_block_mode,
            ).to(device=self.device, dtype=self.torch_dtype)
            self.track_projector = None
        else:
            self.track_projector = TokenConcatProjector(
                video_dim=dit_dim,
                track_dim=injector_track_dim,
                hidden_dim=dit_dim,
                zero_init=True,
            ).to(device=self.device, dtype=self.torch_dtype)
            self.track_block_injector = None

        # --- Print summary ---
        track_modules = [m for m in (self.track_embedder, self.track_adapter,
                                     self.track_identity_adapter,
                                     self.track_block_injector, self.track_projector) if m is not None]
        total_params = sum(p.numel() for m in track_modules for p in m.parameters())
        _rank0_print(f"[Track4D-360] Initialized track modules ({track_injection_mode}, adapter={adapter_type}): {total_params:,} params")
        if adapter_type == "conv3d_patchify":
            patchify_backend = getattr(self.track_adapter.conv3d_patchify, "patchify_backend", "conv3d")
            _rank0_print(f"  track_adapter (unified): conv3d_patchify(proj_dim={conv_proj_dim}, "
                         f"patch_size={conv_patch_size}, stride={conv_track_stride}, "
                         f"residual_blocks={conv3d_num_residual_blocks}, "
                         f"patchify_backend={patchify_backend})")
            _rank0_print(f"  track_coordinate_mode: {self.track_coordinate_mode}")
        else:
            _rank0_print(f"  track_embedder: in={track_in_channels}, embed_dim={track_embed_dim}, freqs={track_num_freqs}, layers={track_num_mlp_layers}, temporal_pool={temporal_pool}")
            if adapter_type == "conv_patchify":
                _rank0_print(f"  track_adapter: conv_patchify(proj_dim={conv_proj_dim}, patch_size={conv_patch_size}, stride={conv_track_stride})")
            else:
                _rank0_print(f"  track_adapter: scatter(block=({block_t},{block_h},{block_w}))")
        if track_injection_mode == "per_block":
            injector_params = sum(p.numel() for p in self.track_block_injector.parameters())
            _rank0_print(f"  track_block_injector: {len(self.dit.blocks)} blocks x Linear({track_embed_dim},{dit_dim}) = {injector_params:,} params")
        else:
            _rank0_print(f"  track_projector: {dit_dim}+{track_embed_dim} -> {dit_dim}")
        _rank0_print(
            "  persistent_identity: "
            f"mode={self.persistent_identity_mode}, dim={int(persistent_identity_dim)}, "
            f"identity_only={self.persistent_identity_only}"
        )

    def initialize_dense_only_control_modules(
        self,
        track_embed_dim=156,
        dit_dim=1536,
        track_injection_mode="per_block",
        track_injection_block_mode="concat_project",
    ):
        """Create dense-control injection support without 3D-track encoders."""
        self.track_injection_mode = track_injection_mode
        self.track_adapter_type = "dense_only_align"
        block_t, block_h, block_w = self._derive_track_block_sizes()
        self.track_embedder = None
        self.track_identity_adapter = None
        self.persistent_identity_mode = "legacy"
        self.persistent_identity_only = False
        self.track_adapter = _DenseOnlyTokenAlignAdapter(
            out_dim=track_embed_dim,
            block_t=block_t,
            block_h=block_h,
            block_w=block_w,
        ).to(device=self.device, dtype=self.torch_dtype)

        occupancy_extra = 1 if track_occupancy_channel_enabled() else 0
        injector_track_dim = track_embed_dim + occupancy_extra
        if track_injection_mode == "per_block":
            num_blocks = len(self.dit.blocks)
            self.track_block_injector = PerBlockTrackInjector(
                num_blocks=num_blocks,
                track_dim=injector_track_dim,
                dit_dim=dit_dim,
                mode=track_injection_block_mode,
            ).to(device=self.device, dtype=self.torch_dtype)
            self.track_projector = None
        else:
            self.track_projector = TokenConcatProjector(
                video_dim=dit_dim,
                track_dim=injector_track_dim,
                hidden_dim=dit_dim,
                zero_init=True,
            ).to(device=self.device, dtype=self.torch_dtype)
            self.track_block_injector = None

        modules = [m for m in (self.track_adapter, self.track_block_injector, self.track_projector) if m is not None]
        total_params = sum(p.numel() for m in modules for p in m.parameters())
        _rank0_print(
            f"[Track4D-360] Initialized dense-only control injection "
            f"({track_injection_mode}, {total_params:,} params; no persistent track encoder)"
        )

    def initialize_track_null_token(self):
        """Create a learned placeholder token for missing/dropped track conditioning."""
        if getattr(self, "track_adapter", None) is None:
            raise ValueError("initialize_track_null_token requires track_adapter to exist first.")
        out_dim = int(getattr(self.track_adapter, "out_dim"))
        self.track_null_token = torch.nn.Parameter(
            torch.randn(1, 1, out_dim, device=self.device, dtype=self.torch_dtype)
        )
        _rank0_print(
            f"[Track4D-360] Initialized track_null_token: shape={tuple(self.track_null_token.shape)}"
        )

    def initialize_dense_target_control_encoder(
        self,
        in_channels: int = 5,
        proj_dim: int = 32,
        out_dim: int = 156,
        patch_size: int = 4,
        temporal_stride: int = 4,
        num_residual_blocks: int = 2,
        alpha_track: float = 1.0,
        use_depth_null_token: bool = False,
    ):
        """Create the dense GEN3C-style target-space control encoder.

        Must be called by the training/inference setup *after*
        ``initialize_track_modules`` so that ``out_dim`` can be matched against
        the persistent track adapter's output dimension.

        Args:
            in_channels: 5 = ``[warped_rgb (3) | warped_depth_norm (1) | warp_valid (1)]``.
            proj_dim: internal Conv3d channel dim.
            out_dim: token dim — must equal the persistent track adapter's
                ``out_dim`` so the additive fusion is dimension-compatible.
            patch_size: spatial patchify stride; must equal
                ``height_division_factor // conv_track_stride`` for the
                persistent path.
            temporal_stride: temporal patchify stride; must equal Wan VAE
                temporal stride (4).
            num_residual_blocks: depth of CausalResidualBlock3d stack.
            alpha_track: scaling factor on the persistent branch in the late
                additive fusion.  Stored on ``self.dense_branch_alpha_track``.
        """
        # Sanity-check out_dim against the persistent adapter when present.
        if hasattr(self, "track_adapter") and self.track_adapter is not None:
            persistent_out = getattr(self.track_adapter, "out_dim", None)
            if persistent_out is not None and persistent_out != out_dim:
                raise ValueError(
                    f"DenseTargetControlEncoder out_dim={out_dim} != "
                    f"track_adapter.out_dim={persistent_out}; "
                    "the two streams must share the same token dimension for "
                    "late additive fusion."
                )

        self.dense_target_control_encoder = DenseTargetControlEncoder(
            in_channels=in_channels,
            proj_dim=proj_dim,
            out_dim=out_dim,
            patch_size=patch_size,
            temporal_stride=temporal_stride,
            num_residual_blocks=num_residual_blocks,
            use_depth_null_token=bool(use_depth_null_token),
        ).to(device=self.device, dtype=self.torch_dtype)
        self.dense_branch_alpha_track = float(alpha_track)

        n_params = sum(p.numel() for p in self.dense_target_control_encoder.parameters())
        patchify_backend = getattr(
            self.dense_target_control_encoder.conv3d_patchify,
            "patchify_backend",
            "conv3d",
        )
        _rank0_print(
            f"[Track4D-360] Initialized dense_target_control_encoder: "
            f"in_ch={in_channels}, proj_dim={proj_dim}, out_dim={out_dim}, "
            f"patch_size={patch_size}, ts={temporal_stride}, "
            f"residual_blocks={num_residual_blocks}, alpha_track={alpha_track}, "
            f"depth_null_token={bool(use_depth_null_token)}, "
            f"patchify_backend={patchify_backend} "
            f"({n_params:,} params)"
        )

    def initialize_dense_null_token(self):
        """Create a learned placeholder token for missing/dropped dense conditioning."""
        encoder = getattr(self, "dense_target_control_encoder", None)
        if encoder is None:
            raise ValueError("initialize_dense_null_token requires dense_target_control_encoder first.")
        out_dim = int(getattr(encoder, "out_dim"))
        self.dense_null_token = torch.nn.Parameter(
            torch.randn(1, 1, out_dim, device=self.device, dtype=self.torch_dtype)
        )
        _rank0_print(
            f"[Track4D-360] Initialized dense_null_token: shape={tuple(self.dense_null_token.shape)}"
        )

    def load_track_checkpoint(self, checkpoint_path, verbose=True):
        """
        Load track module weights from a training checkpoint.

        Training checkpoints saved with --remove_prefix_in_ckpt "pipe.dit." contain:
          - LoRA weights: blocks.0.self_attn.q.lora_A.weight (for DiT)
          - DiT params: domain_embedding, control_patch_embedding.*
          - Track modules: pipe.track_embedder.*, pipe.track_adapter.*, pipe.track_projector.*

        This method loads only the track module weights (pipe.track_*).
        LoRA weights should be loaded separately via load_lora or PEFT.
        """
        from safetensors.torch import load_file as load_safetensors

        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_safetensors(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        checkpoint_has_persistent_identity = any(
            key.startswith("pipe.track_identity_adapter.")
            or key.startswith("track_identity_adapter.")
            for key in state_dict
        )
        if (
            checkpoint_has_persistent_identity
            and getattr(self, "track_identity_adapter", None) is None
        ):
            raise ValueError(
                "Checkpoint contains persistent trajectory identity weights, but "
                "the current track config uses persistent_identity_mode=legacy. "
                "Load it with the matching target/shared track config."
            )

        track_modules = {}
        track_embedder = getattr(self, "track_embedder", None)
        track_adapter = getattr(self, "track_adapter", None)
        track_identity_adapter = getattr(self, "track_identity_adapter", None)
        track_projector = getattr(self, "track_projector", None)
        track_block_injector = getattr(self, "track_block_injector", None)
        if track_embedder is not None:
            track_modules["track_embedder"] = track_embedder
        if track_adapter is not None:
            track_modules["track_adapter"] = track_adapter
        if track_identity_adapter is not None:
            track_modules["track_identity_adapter"] = track_identity_adapter
        if track_projector is not None:
            track_modules["track_projector"] = track_projector
        if track_block_injector is not None:
            track_modules["track_block_injector"] = track_block_injector
        if getattr(self, "dense_target_control_encoder", None) is not None:
            track_modules["dense_target_control_encoder"] = self.dense_target_control_encoder
        # Warped-appearance latent-fuse path (--warped_condition_mode latent_fuse):
        # GeometryTrackFuser is attached to `pipe` only (not `pipe.dit`), so its
        # saved keys carry the `pipe.geometry_track_fuser.*` prefix and are
        # rejected as "unexpected" by the earlier `pipe.dit.load_state_dict`
        # call — they must land here. Without this entry the fuser runs with
        # random init and the warped branch emits pure noise at inference.
        if getattr(self, "geometry_track_fuser", None) is not None:
            track_modules["geometry_track_fuser"] = self.geometry_track_fuser
        if getattr(self, "gs_decoder", None) is not None:
            track_modules["gs_decoder"] = self.gs_decoder

        loaded_count = 0
        for module_name, module in track_modules.items():
            # Only "pipe.dit." is stripped on save, so auxiliary modules like
            # gs_decoder still appear as "pipe.gs_decoder.*" in checkpoints.
            prefix_options = [f"pipe.{module_name}.", f"{module_name}."]
            module_state = {}
            for key, value in state_dict.items():
                for prefix in prefix_options:
                    if key.startswith(prefix):
                        param_name = key[len(prefix):]
                        module_state[param_name] = value
                        break

            if module_state:
                current_state = module.state_dict()
                skipped_shape = []
                # Injector/projector input dims change with the occupancy
                # channel flag (D vs D+1). Silent shape-skip here previously
                # left those layers at random init (Codex review F3): migrate
                # legacy → occupancy with an explicit zero column, and fail
                # loudly on any other mismatch for these modules.
                if module_name in ("track_projector", "track_block_injector"):
                    module_state, occ_migrated = migrate_state_for_occupancy_channel(
                        module_state,
                        current_state,
                        occupancy_enabled=track_occupancy_channel_enabled(),
                        module_name=module_name,
                    )
                    if occ_migrated and verbose:
                        print(
                            f"  [Track4D-360] {module_name}: zero-padded occupancy "
                            f"input column on {len(occ_migrated)} key(s) "
                            f"(legacy checkpoint -> occupancy model)"
                        )
                filtered_state = {}
                for param_name, value in module_state.items():
                    current = current_state.get(param_name)
                    if current is not None and tuple(current.shape) != tuple(value.shape):
                        skipped_shape.append((param_name, tuple(value.shape), tuple(current.shape)))
                        continue
                    filtered_state[param_name] = value
                missing, unexpected = module.load_state_dict(filtered_state, strict=False)
                loaded_count += len(filtered_state) - len(missing)
                if verbose:
                    print(f"  [Track4D-360] Loaded {module_name}: "
                          f"{len(filtered_state)}/{len(module_state)} keys, "
                          f"{len(missing)} missing, {len(unexpected)} unexpected")
                    if skipped_shape:
                        preview = ", ".join(
                            f"{name}: ckpt{src}->model{dst}"
                            for name, src, dst in skipped_shape[:5]
                        )
                        suffix = " ..." if len(skipped_shape) > 5 else ""
                        print(f"    Skipped shape-mismatched keys: {preview}{suffix}")
                    if missing:
                        print(f"    Missing: {missing}")
            elif verbose:
                print(f"  [Track4D-360] WARNING: No weights found for {module_name}")

        for param_name in ("track_null_token", "dense_null_token"):
            param = getattr(self, param_name, None)
            if param is None:
                continue
            loaded = False
            for prefix in (f"pipe.{param_name}", param_name):
                if prefix in state_dict:
                    with torch.no_grad():
                        param.copy_(state_dict[prefix].to(device=param.device, dtype=param.dtype))
                    loaded = True
                    if verbose:
                        print(f"  [Track4D-360] Loaded {param_name}: shape={tuple(param.shape)}")
                    break
            if not loaded and verbose:
                print(f"  [Track4D-360] WARNING: No weights found for {param_name}")

        # Also load DiT non-LoRA params (domain_embedding, control_patch_embedding)
        dit_params = {}
        for key, value in state_dict.items():
            if key in ("domain_embedding",) or key.startswith("control_patch_embedding"):
                dit_params[key] = value
        if dit_params and hasattr(self, "dit") and self.dit is not None:
            model_state = self.dit.state_dict()
            if "domain_embedding" in dit_params and "domain_embedding" in model_state:
                ckpt_domain = dit_params["domain_embedding"]
                model_domain = model_state["domain_embedding"]
                if (
                    torch.is_tensor(ckpt_domain)
                    and torch.is_tensor(model_domain)
                    and tuple(ckpt_domain.shape) != tuple(model_domain.shape)
                ):
                    if (
                        ckpt_domain.ndim == 2
                        and model_domain.ndim == 2
                        and int(ckpt_domain.shape[1]) == int(model_domain.shape[1])
                    ):
                        merged = model_domain.detach().clone().to(dtype=ckpt_domain.dtype)
                        rows_to_copy = min(int(ckpt_domain.shape[0]), int(model_domain.shape[0]))
                        merged[:rows_to_copy] = ckpt_domain[:rows_to_copy].to(dtype=merged.dtype)
                        dit_params["domain_embedding"] = merged
                        if verbose:
                            print(
                                "  [Track4D-360] Partial domain_embedding load: "
                                f"ckpt={tuple(ckpt_domain.shape)} -> model={tuple(model_domain.shape)}, "
                                f"copied first {rows_to_copy} row(s), kept new rows initialized"
                            )
                    else:
                        dit_params.pop("domain_embedding", None)
                        if verbose:
                            print(
                                "  [Track4D-360] WARNING: Skipping incompatible domain_embedding "
                                f"ckpt={tuple(ckpt_domain.shape)} vs model={tuple(model_domain.shape)}"
                            )

            for key in list(dit_params.keys()):
                if not key.startswith("control_patch_embedding") or key not in model_state:
                    continue
                ckpt_tensor = dit_params[key]
                model_tensor = model_state[key]
                if not torch.is_tensor(ckpt_tensor) or not torch.is_tensor(model_tensor):
                    continue
                if tuple(ckpt_tensor.shape) == tuple(model_tensor.shape):
                    continue
                if ckpt_tensor.ndim != model_tensor.ndim:
                    dit_params.pop(key, None)
                    if verbose:
                        print(
                            f"  [Track4D-360] WARNING: Skipping incompatible {key}: "
                            f"ckpt={tuple(ckpt_tensor.shape)} vs model={tuple(model_tensor.shape)}"
                        )
                    continue
                merged = model_tensor.detach().clone().to(dtype=ckpt_tensor.dtype)
                copy_shape = [min(int(a), int(b)) for a, b in zip(ckpt_tensor.shape, model_tensor.shape)]
                copy_slices = tuple(slice(0, n) for n in copy_shape)
                merged[copy_slices] = ckpt_tensor[copy_slices].to(dtype=merged.dtype)
                dit_params[key] = merged
                if verbose:
                    print(
                        f"  [Track4D-360] Partial {key} load: "
                        f"ckpt={tuple(ckpt_tensor.shape)} -> model={tuple(model_tensor.shape)}, "
                        f"copied overlap={tuple(copy_shape)}"
                    )

            missing, unexpected = self.dit.load_state_dict(dit_params, strict=False)
            loaded_dit = len(dit_params) - len(missing)
            if verbose and loaded_dit > 0:
                print(f"  [Track4D-360] Loaded {loaded_dit} DiT params (domain_embedding, control_patch_embedding)")

        if verbose:
            print(f"  [Track4D-360] Total: {loaded_count} track module parameters loaded from {checkpoint_path}")

        # Codex review F1: checkpoint loads restore a drifted nonzero null and
        # can re-enable its grad. Re-enforce the hard-zero frozen policy LAST,
        # so it wins over both freeze_except() and any loaded value. No-op
        # unless HOLO4D_TRACK_NULL_TOKEN_FROZEN=1.
        enforce_frozen_null_tokens_on_pipe(self)

    def _record_stage_profile(
        self,
        *,
        enabled: bool,
        total_started: float,
        decode_started: float,
        condition_seconds: float,
        denoise_seconds: float,
    ) -> None:
        """Finalize synchronized per-stage inference timings for the last call."""
        if not enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        decode_seconds = time.perf_counter() - decode_started
        unit_timings = getattr(self, "_debug_last_condition_unit_timings", {}) or {}
        track_seconds = float(unit_timings.get("track_encoding_seconds", 0.0))
        dense_seconds = float(unit_timings.get("dense_condition_seconds", 0.0))
        self._debug_last_stage_timings = {
            "condition_cache_track_token_seconds": condition_seconds,
            "track_encoding_seconds": track_seconds,
            "dense_condition_seconds": dense_seconds,
            "other_condition_seconds": max(
                condition_seconds - track_seconds - dense_seconds,
                0.0,
            ),
            "wan_denoise_seconds": denoise_seconds,
            "vae_decode_and_transition_seconds": decode_seconds,
            "stage_total_seconds": time.perf_counter() - total_started,
        }

    def __call__(
        self,
        # Prompt
        prompt: str = "",
        negative_prompt: Optional[str] = "",
        # Optional precomputed T5 embeddings — when supplied, the prompt-
        # embedder unit's cached fast-path uses these directly and skips T5
        # online encode. Eval scripts can pull these from the dataset's
        # `_resolve_target_caption_with_emb` / `_find_caption_with_emb`
        # output, matching the safetensors cache the trainer reads.
        prompt_emb: Optional[torch.Tensor] = None,
        negative_prompt_emb: Optional[torch.Tensor] = None,
        # Source video (condition)
        input_video: Optional[list] = None,
        # Vista4D continuation condition: previous target chunk's last frame.
        # Ignored unless use_target_image_condition=True.
        input_image: Optional[Image.Image] = None,
        denoising_strength: Optional[float] = 1.0,
        # V2V mode
        v2v_concat_mode: Optional[Literal["frame", "width"]] = "width",
        # Track4D-360 specific: world-space 3D tracks + target camera
        tracks_world: Optional[torch.Tensor] = None,
        tracks_query_idx_per_identity: Optional[torch.Tensor] = None,
        track_identity_sampling_seed: Optional[int] = None,
        target_cam2world: Optional[torch.Tensor] = None,
        target_intrinsics: Optional[torch.Tensor] = None,
        # Optional per-frame K (T, 3, 3). When set, used by both Plucker camera
        # rays and persistent 3D-track projection so all camera conditions share
        # the same per-frame FOV evolution. A single `target_intrinsics` remains
        # the backward-compatible fixed-K fallback.
        target_intrinsics_per_frame: Optional[torch.Tensor] = None,
        # Target video (only for training / evaluation; not needed for inference)
        target_video: Optional[list] = None,
        source_depth_video: Optional[torch.Tensor] = None,
        # Dense GEN3C-style target-space control branch (optional)
        source_rgb_video: Optional[torch.Tensor] = None,
        source_depth_valid: Optional[torch.Tensor] = None,
        source_cam2world: Optional[torch.Tensor] = None,
        source_intrinsics: Optional[torch.Tensor] = None,
        use_dense_branch: bool = False,
        # Stage 2: target-view dense 3D position maps (joint tracking)
        target_track_video: Optional[torch.Tensor] = None,
        target_track_valid_mask: Optional[torch.Tensor] = None,
        target_depth_video: Optional[torch.Tensor] = None,
        target_depth_valid_mask: Optional[torch.Tensor] = None,
        target_depth_log_cap: float = 14.0,
        # Arbitrary query-frame conditioning
        use_query_frame_impulse_condition: bool = False,
        track_query_idx: Optional[torch.Tensor] = None,
        enable_v2v_plucker_camera_control: bool = False,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        num_frames: int = 49,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple] = (30, 52),
        tile_stride: Optional[tuple] = (15, 26),
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # V2V settings
        use_domain_embedding: bool = True,
        # Wan-I2V continuation settings (all opt-in / backward-compatible).
        use_target_image_condition: bool = False,
        use_ti2v_native_i2v: bool = False,
        i2v_condition_drop_prob: float = 0.0,
        i2v_condition_noise_level: float = 0.0,
        # Progress
        progress_bar_cmd=tqdm,
        # Extra kwargs passed through to model_fn
        **extra_kwargs,
    ):
        """
        Track4D-360 inference: generate target-view video given source video + 3D tracks.

        Args:
            prompt: Text prompt for generation.
            input_video: Source-view video frames (list of PIL Images).
            tracks_world: Dense 3D tracks in world space (N, T, 3) or (B, N, T, 3).
            target_cam2world: Target camera extrinsics (T, 4, 4).
            target_intrinsics: Target camera intrinsics (3, 3).
            v2v_concat_mode: V2V concatenation mode ("width" or "frame").
            Other args: Standard generation parameters.

        Returns:
            List of PIL Images (generated target-view video).
        """
        debug_capture_final_latents = bool(extra_kwargs.pop("debug_capture_final_latents", False))
        profile_stage_timings = bool(extra_kwargs.pop("profile_stage_timings", False))
        gs_feedback_interval = int(extra_kwargs.pop("gs_feedback_interval", 0))
        # gs_capture_progress_idx: capture gs_features + predicted x0 at this
        # mid-trajectory step so eval can render the GS head from features
        # extracted at the same σ training operated at (e.g. step 47/50 with
        # shift=5 → σ≈0.242 = training's fixed_step_frac=0.94 point).
        gs_capture_progress_idx = extra_kwargs.pop("gs_capture_progress_idx", None)
        gs_capture_progress_idx = (
            None if gs_capture_progress_idx is None else int(gs_capture_progress_idx)
        )
        self._debug_last_target_latents = None
        self._debug_last_decode_latents = None
        self._debug_last_track_identity_selection = None
        self._debug_last_track_adapter_inputs = None
        self._debug_last_stage_timings = None
        self._debug_last_condition_unit_timings = {}
        if profile_stage_timings and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        stage_total_started = time.perf_counter()
        self._gs_features_at_capture = None
        self._x0_hat_at_capture = None
        self._capture_progress_idx = None
        self._capture_timestep = None
        if bool(use_target_image_condition) and bool(use_ti2v_native_i2v):
            raise ValueError(
                "use_target_image_condition and use_ti2v_native_i2v are mutually exclusive"
            )

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        channel_concat_mode_cfg = getattr(self, "channel_concat_mode_cfg", None)
        channel_concat = bool(channel_concat_mode_cfg)
        channel_modalities = list(channel_concat_mode_cfg.get("channel_modalities", [])) if channel_concat else None
        num_channel_modalities = len(channel_modalities) if channel_modalities is not None else 0

        # Build inputs dicts
        inputs_posi = {
            "prompt": prompt,
            "prompt_emb": prompt_emb,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_prompt_emb": negative_prompt_emb,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_video": input_video,
            "input_image": input_image,
            "denoising_strength": denoising_strength,
            "v2v_concat_mode": v2v_concat_mode,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            # Track4D-360 specific
            "tracks_world": tracks_world,
            "tracks_query_idx_per_identity": tracks_query_idx_per_identity,
            "track_identity_sampling_seed": track_identity_sampling_seed,
            "target_cam2world": target_cam2world,
            "target_intrinsics": target_intrinsics,
            "target_intrinsics_per_frame": target_intrinsics_per_frame,
            "target_video": target_video,
            "use_ti2v_native_i2v": bool(use_ti2v_native_i2v),
            "source_depth_video": source_depth_video,
            # Dense GEN3C-style target-space control branch (optional)
            "source_rgb_video": source_rgb_video,
            "source_depth_valid": source_depth_valid,
            "source_cam2world": source_cam2world,
            "source_intrinsics": source_intrinsics,
            "use_dense_branch": bool(use_dense_branch),
            # Stage 2: target-view 3D tracks (joint tracking)
            "target_track_video": target_track_video,
            "target_track_valid_mask": target_track_valid_mask,
            "target_depth_video": target_depth_video,
            "target_depth_valid_mask": target_depth_valid_mask,
            "target_depth_log_cap": float(target_depth_log_cap),
            "target_track_use_source_stats": bool(channel_concat_mode_cfg.get("target_track_use_source_stats", False))
            if channel_concat else self.joint_track_depth_prediction,
            "channel_concat": channel_concat,
            "channel_modalities": channel_modalities,
            "num_channel_modalities": num_channel_modalities,
            "use_query_frame_impulse_condition": use_query_frame_impulse_condition,
            "track_query_idx": track_query_idx,
            "enable_v2v_plucker_camera_control": bool(enable_v2v_plucker_camera_control),
            # V2V settings
            "use_domain_embedding": use_domain_embedding,
            # Per-layer camera control defaults (overridable via extra_kwargs)
            "use_perlayer_camera_control": False,
            "use_pre_dit_camera_injection": False,
            "use_noise_camera_injection": False,
            "multimodality_nvs": False,
            "multimodality_tracking": self.joint_track_prediction and (not self.joint_track_depth_prediction),
            "multimodality_track_depth": self.joint_track_depth_prediction,
            "drop_video_prob": 0.0,
            "drop_t2v_prob": None,
            "drop_i2v_prob": None,
            "input_video_noise_strength": 0.0,
            "use_target_image_condition": bool(use_target_image_condition),
            # Drop/noise are train-only inside the condition unit. Inference
            # accepts these values but deterministically uses the supplied image.
            "i2v_condition_drop_prob": float(i2v_condition_drop_prob),
            "i2v_condition_noise_level": float(i2v_condition_noise_level),
        }
        if enable_v2v_plucker_camera_control:
            if not extra_kwargs.get("use_plucker_camera_control", False):
                raise ValueError(
                    "enable_v2v_plucker_camera_control=True requires use_plucker_camera_control=True."
                )
            if source_cam2world is not None and source_intrinsics is not None:
                inputs_shared["source_camera"] = {
                    "c2w_opencv": source_cam2world,
                    "intrinsics": source_intrinsics,
                }
            if target_cam2world is not None and target_intrinsics is not None:
                # Prefer per-frame K for plucker construction so zoom-trajectory
                # rays evolve with FOV. Falls back to single (3,3) K otherwise.
                _tgt_K_for_plucker = (
                    target_intrinsics_per_frame
                    if target_intrinsics_per_frame is not None
                    else target_intrinsics
                )
                inputs_shared["target_camera"] = {
                    "c2w_opencv": target_cam2world,
                    "intrinsics": _tgt_K_for_plucker,
                }
        # Merge any extra kwargs
        inputs_shared.update(extra_kwargs)

        # Run pipeline units (shape check, noise init, VAE encode, track conditioning, V2V concat, etc.)
        for unit in self.units:
            timed_unit_key = None
            if profile_stage_timings:
                if isinstance(unit, WanVideoUnit_TrackPointCondition):
                    timed_unit_key = "track_encoding_seconds"
                elif isinstance(unit, WanVideoUnit_DenseTargetControl):
                    timed_unit_key = "dense_condition_seconds"
            if timed_unit_key is not None and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            unit_started = time.perf_counter() if timed_unit_key is not None else None
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
            if timed_unit_key is not None:
                assert unit_started is not None
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                self._debug_last_condition_unit_timings[timed_unit_key] = (
                    self._debug_last_condition_unit_timings.get(timed_unit_key, 0.0)
                    + time.perf_counter() - unit_started
                )

        # V2V inference fix: after V2V concat widens `input_latents`,
        # `latents` (noise) must match. Re-initialize noise with the concat shape.
        il = inputs_shared.get("input_latents")
        lt = inputs_shared.get("latents")
        num_domains = self._num_v2v_domains()
        if il is not None and lt is not None and il.shape != lt.shape:
            noise = torch.randn(il.shape, device=lt.device, dtype=lt.dtype, generator=torch.Generator(lt.device).manual_seed(seed))
            # Add noise at the first scheduler timestep (full denoising)
            inputs_shared["latents"] = self.scheduler.add_noise(il, noise, timestep=self.scheduler.timesteps[0])
            # Keep source portion clean (condition)
            if v2v_concat_mode == "width":
                w_src = il.shape[4] // num_domains
                inputs_shared["latents"][:, :, :, :, :w_src] = il[:, :, :, :, :w_src]
            elif v2v_concat_mode == "frame":
                t_src = il.shape[2] // num_domains
                inputs_shared["latents"][:, :, :t_src] = il[:, :, :t_src]

        if profile_stage_timings and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        condition_seconds = time.perf_counter() - stage_total_started
        denoise_started = time.perf_counter()

        # ------------------------------------------------------------------
        # Source clean-force helper (mirrors training FlowMatchV2VLoss which
        # does ``noisy_latents[source_slice] = input_latents[source_slice]``
        # before every forward pass). Without this, ``scheduler.step()``
        # updates the entire latent tensor — including the source domain
        # that should stay clean as conditioning — and the DiT sees an
        # off-distribution source on every iteration after step 0. That
        # compounds over the denoising schedule and produces dreamy
        # VDM-prior-looking output instead of a precise reconstruction of
        # the target. See doc/new_features/2026-04-09-hybrid-dense-plucker-
        # training-failure-analysis.md for the full forensics.
        # ------------------------------------------------------------------
        def _force_source_clean():
            latents = inputs_shared.get("latents")
            if il is None or latents is None or il.shape != latents.shape:
                return
            if v2v_concat_mode == "width":
                w_src = il.shape[4] // num_domains
                latents[:, :, :, :, :w_src] = il[:, :, :, :, :w_src]
            elif v2v_concat_mode == "frame":
                t_src = il.shape[2] // num_domains
                latents[:, :, :t_src] = il[:, :, :t_src]

        def _force_ti2v_native_boundary():
            if not bool(inputs_shared.get("ti2v_native_condition_active", False)):
                return
            first_frame = inputs_shared.get("first_frame_latents")
            if first_frame is None:
                raise RuntimeError(
                    "TI2V-native condition is active but first_frame_latents is missing"
                )
            inputs_shared["latents"] = inject_target_first_frame_latents(
                inputs_shared["latents"],
                first_frame,
                num_domains=num_domains,
                target_domain=1,
            )

        # Native I2V must present the same clean source and target boundary to
        # the first DiT call as it does after every scheduler step. Keep this
        # extra pre-loop write opt-in so legacy inference remains unchanged.
        if bool(inputs_shared.get("ti2v_native_condition_active", False)):
            _force_source_clean()
            _force_ti2v_native_boundary()

        # Denoise
        _iteration_models = list(self.in_iteration_models)
        _has_gs_decoder = getattr(self, "gs_decoder", None) is not None
        _feedback_channels = getattr(self.dit, "_feedback_channels", 0)
        # Trained feedback mode: model was trained with FlashWorld-style
        # prev_feats + prev_latents_pred conditioning channels.
        _gs_fb_trained = _feedback_channels > 0 and _has_gs_decoder and v2v_concat_mode == "width" and il is not None
        # Legacy diagnostic mode: untrained latent-replacement hack.
        _gs_fb_legacy = (
            gs_feedback_interval > 0
            and not _gs_fb_trained
            and _has_gs_decoder
            and v2v_concat_mode == "width"
            and il is not None
        )
        _needs_vae_in_loop = _gs_fb_trained or _gs_fb_legacy
        if _needs_vae_in_loop and "vae" not in _iteration_models:
            _iteration_models.append("vae")
        self.load_models_to_device(_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        # --- GS feedback setup (cameras, split point) ---
        _gs_fb_cameras = None
        _gs_fb_split = None
        if _gs_fb_trained or _gs_fb_legacy:
            from holo4d.models.gs_camera_utils import c2w_K_to_flashworld_cameras
            _gs_fb_split = il.shape[4] // num_domains
            _tc2w = inputs_shared.get("target_cam2world")
            _tK = inputs_shared.get("target_intrinsics")
            if _tc2w is not None and _tK is not None:
                _tc2w_f = _tc2w.float() if torch.is_tensor(_tc2w) else torch.as_tensor(_tc2w, dtype=torch.float32)
                _tK_f = _tK.float() if torch.is_tensor(_tK) else torch.as_tensor(_tK, dtype=torch.float32)
                _gs_fb_cameras = c2w_K_to_flashworld_cameras(
                    _tc2w_f, _tK_f, height, width,
                ).to(device=self.device, dtype=self.torch_dtype)
                if _gs_fb_cameras.dim() == 2:
                    _gs_fb_cameras = _gs_fb_cameras.unsqueeze(0)
            else:
                _gs_fb_trained = False
                _gs_fb_legacy = False

        if _gs_fb_trained:
            print(f"[GS feedback] trained mode: feedback channels every step "
                  f"({len(self.scheduler.timesteps)}-step rollout)")
        elif _gs_fb_legacy:
            print(f"[GS feedback] legacy mode: latent replacement every {gs_feedback_interval} steps")

        # Feedback state for trained mode (chained across steps)
        _prev_feedback_input = None  # None → zeros on step 0
        _ar_eval = bool(inputs_shared.get("ar_training", False))
        _ar_frames_per_block = int(inputs_shared.get("ar_num_frame_per_block", 3))
        _ar_cache_max_frames = int(inputs_shared.get("ar_cache_max_latent_frames", 21))
        _ar_cache_keep_first_latent = bool(inputs_shared.get("ar_cache_keep_first_latent", False))

        def _run_model_maybe_ar(cfg_inputs, timestep, feedback_input=None):
            if not _ar_eval:
                return self.model_fn(
                    **models, **inputs_shared, **cfg_inputs, timestep=timestep,
                    feedback_input=feedback_input,
                )
            from holo4d.ar.rolling_forcing import allocate_kv_cache, build_latent_windows, slice_latent_frames
            from holo4d_runtime.diffusion.loss import _slice_ar_condition_payload

            full_latents = inputs_shared["latents"]
            windows = build_latent_windows(int(full_latents.shape[2]), _ar_frames_per_block)
            kv_cache = allocate_kv_cache(len(getattr(self.dit, "blocks", [])))
            pred_chunks = []
            for window in windows:
                window_shared = dict(inputs_shared)
                window_shared["latents"] = slice_latent_frames(full_latents, window.start, window.end)
                if window_shared.get("input_latents") is not None:
                    window_shared["input_latents"] = slice_latent_frames(
                        window_shared["input_latents"], window.start, window.end
                    )
                window_shared = _slice_ar_condition_payload(self, window_shared, window.start, window.end)
                # These AR cache controls are passed explicitly below.  Keep them
                # out of the expanded payload so Python does not receive duplicate
                # values for model_fn's named parameters.
                window_shared.pop("ar_num_frame_per_block", None)
                window_shared.pop("ar_cache_max_latent_frames", None)
                window_shared.pop("ar_cache_keep_first_latent", None)
                chunk = self.model_fn(
                    **models,
                    **window_shared,
                    **cfg_inputs,
                    timestep=timestep,
                    feedback_input=feedback_input,
                    kv_cache=kv_cache,
                    current_start=window.start,
                    cache_start=max(0, window.end - _ar_cache_max_frames),
                    updating_cache=True,
                    ar_num_frame_per_block=_ar_frames_per_block,
                    ar_cache_max_latent_frames=_ar_cache_max_frames,
                    ar_cache_keep_first_latent=_ar_cache_keep_first_latent,
                )
                pred_chunks.append(chunk)
            if pred_chunks and isinstance(pred_chunks[0], dict):
                out = dict(pred_chunks[0])
                out["noise_pred"] = torch.cat([chunk["noise_pred"] for chunk in pred_chunks], dim=2)
                if pred_chunks[0].get("gs_features") is not None:
                    out["gs_features"] = torch.cat([chunk["gs_features"] for chunk in pred_chunks], dim=2)
                return out
            return torch.cat(pred_chunks, dim=2)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            current_latents = inputs_shared["latents"]

            noise_pred_posi = _run_model_maybe_ar(inputs_posi, timestep, feedback_input=_prev_feedback_input)
            # model_fn returns a dict when gs_feature_head is active.
            gs_features = None
            if isinstance(noise_pred_posi, dict):
                gs_features = noise_pred_posi.get("gs_features")
                self._last_gs_features = gs_features
                noise_pred_posi = noise_pred_posi["noise_pred"]
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = _run_model_maybe_ar(inputs_nega, timestep, feedback_input=_prev_feedback_input)
                    if isinstance(noise_pred_nega, dict):
                        noise_pred_nega = noise_pred_nega["noise_pred"]
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # --- Mid-trajectory capture (training-σ snapshot for fair GS eval) ---
            if (
                gs_capture_progress_idx is not None
                and progress_id == gs_capture_progress_idx
                and gs_features is not None
            ):
                _sigma_cap = self.scheduler.sigmas[progress_id]
                self._gs_features_at_capture = gs_features.detach().clone()
                self._x0_hat_at_capture = (current_latents + noise_pred * (0 - _sigma_cap)).detach().clone()
                self._capture_progress_idx = int(progress_id)
                self._capture_timestep = float(timestep.item())
                print(f"[GS capture] progress_id={progress_id} timestep={float(timestep.item()):.1f} "
                      f"sigma={float(_sigma_cap.item()):.4f} — saved gs_features + x0_hat for eval render.")

            # --- Trained feedback: GS decode + render + VAE encode for next step ---
            if _gs_fb_trained and gs_features is not None and progress_id < len(self.scheduler.timesteps) - 1:
                sigma = self.scheduler.sigmas[progress_id]
                x0_hat = current_latents + noise_pred * (0 - sigma)
                gs_features_target = gs_features[:, :, :, :, _gs_fb_split:]
                x0_hat_target = x0_hat[:, :, :, :, _gs_fb_split:]

                gs_params = self.gs_decoder(gs_features_target, x0_hat_target, _gs_fb_cameras)
                rendered_rgb, _ = self.gs_decoder.render(
                    gs_params, _gs_fb_cameras,
                    height=height, width=width, bg_mode="white",
                )
                rgb_seq = _prepare_gs_feedback_rgb_for_vae(
                    rendered_rgb,
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                z_3d = self.vae.encode(
                    rgb_seq, device=self.device, tiled=tiled,
                    tile_size=tile_size, tile_stride=tile_stride,
                )
                if z_3d.dim() == 4:
                    z_3d = z_3d.unsqueeze(0)
                z_3d = z_3d.to(dtype=self.torch_dtype, device=self.device)

                # Build feedback_input for next step: cat([prev_feats, prev_latents_pred])
                # Source half = zeros, target half = signal
                prev_feats = torch.zeros_like(gs_features)
                prev_feats[:, :, :, :, _gs_fb_split:] = gs_features_target
                prev_latents_pred = torch.zeros_like(current_latents)
                prev_latents_pred[:, :, :, :, _gs_fb_split:] = z_3d
                _prev_feedback_input = torch.cat([prev_feats, prev_latents_pred], dim=1)

                # Re-noise from 3D latent for next step (FlashWorld-style)
                t_next = self.scheduler.timesteps[progress_id + 1]
                fresh_noise = torch.randn_like(z_3d)
                z_3d_noisy = self.scheduler.add_noise(z_3d, fresh_noise, t_next)

                # Standard scheduler step, then replace target with 3D re-noised
                inputs_shared["latents"] = self.scheduler.step(
                    noise_pred, self.scheduler.timesteps[progress_id], current_latents
                )
                inputs_shared["latents"][:, :, :, :, _gs_fb_split:] = z_3d_noisy
            else:
                # Standard scheduler step (no feedback, or last step, or legacy mode)
                inputs_shared["latents"] = self.scheduler.step(
                    noise_pred, self.scheduler.timesteps[progress_id], current_latents
                )

                # --- Legacy diagnostic feedback (untrained, latent replacement only) ---
                if (
                    _gs_fb_legacy
                    and gs_features is not None
                    and progress_id > 0
                    and progress_id < len(self.scheduler.timesteps) - 1
                    and progress_id % gs_feedback_interval == 0
                ):
                    sigma = self.scheduler.sigmas[progress_id]
                    x0_hat = current_latents + noise_pred * (0 - sigma)
                    gs_features_target = gs_features[:, :, :, :, _gs_fb_split:]
                    x0_hat_target = x0_hat[:, :, :, :, _gs_fb_split:]

                    gs_params = self.gs_decoder(gs_features_target, x0_hat_target, _gs_fb_cameras)
                    rendered_rgb, _ = self.gs_decoder.render(
                        gs_params, _gs_fb_cameras,
                        height=height, width=width, bg_mode="white",
                    )
                    rgb_seq = _prepare_gs_feedback_rgb_for_vae(
                        rendered_rgb,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                    z_3d = self.vae.encode(
                        rgb_seq, device=self.device, tiled=tiled,
                        tile_size=tile_size, tile_stride=tile_stride,
                    )
                    if z_3d.dim() == 4:
                        z_3d = z_3d.unsqueeze(0)
                    z_3d = z_3d.to(dtype=self.torch_dtype, device=self.device)

                    t_next = self.scheduler.timesteps[progress_id + 1]
                    fresh_noise = torch.randn_like(z_3d)
                    z_3d_noisy = self.scheduler.add_noise(z_3d, fresh_noise, t_next)
                    inputs_shared["latents"][:, :, :, :, _gs_fb_split:] = z_3d_noisy

            # CRITICAL: re-force source domain clean to match training distribution.
            _force_source_clean()
            if bool(inputs_shared.get("ti2v_native_condition_active", False)):
                _force_ti2v_native_boundary()
            elif "first_frame_latents" in inputs_shared:
                # Legacy full-frame hook retained for existing external callers.
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # Final forward to capture gs_features matching the fully-denoised state.
        # Inside the loop, _last_gs_features is from the model_fn call at step
        # N-1 (before the final scheduler.step). That forward ran on latents at
        # sigma_{N-1} (small but non-zero). For a fully matched (features, latents)
        # pair we run one extra forward on the post-rollout clean latents at t=0.
        # The scheduler excludes exact zero from timesteps (linspace[:-1]), so we
        # construct it directly.
        if getattr(self, "_last_gs_features", None) is not None:
            t_zero = torch.zeros(1, dtype=self.torch_dtype, device=self.device)
            final_out = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=t_zero, feedback_input=_prev_feedback_input)
            if isinstance(final_out, dict) and "gs_features" in final_out:
                self._last_gs_features = final_out["gs_features"]

        if profile_stage_timings and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        denoise_seconds = time.perf_counter() - denoise_started
        decode_started = time.perf_counter()

        # Decode — extract target-domain latents (V2V width concat: skip source domain)
        decode_latents = inputs_shared["latents"]
        target_latents_all = None
        if v2v_concat_mode == "width" and il is not None:
            w_src = decode_latents.shape[4] // num_domains
            target_latents_all = decode_latents[:, :, :, :, w_src:]
        if debug_capture_final_latents:
            latents_to_capture = target_latents_all if target_latents_all is not None else decode_latents
            self._debug_last_target_latents = latents_to_capture.detach().to(device="cpu", dtype=torch.float32).clone()
            self._debug_last_decode_latents = decode_latents.detach().to(device="cpu", dtype=torch.float32).clone()

        self.load_models_to_device(["vae"])

        if channel_concat and v2v_concat_mode == "width":
            if target_latents_all is None:
                raise RuntimeError(
                    "channel_concat=True requires input_latents to be set by WanVideoUnit_ChannelConcatV2V. "
                    "Check that source/target latents were encoded successfully."
                )
            output = self._decode_channel_concat_target_latents(
                target_latents_all,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                track_normalization_stats=inputs_shared.get("target_track_normalization_stats"),
                depth_normalization_stats=inputs_shared.get("target_depth_normalization_stats"),
            )
            self.load_models_to_device([])
            self._record_stage_profile(
                enabled=profile_stage_timings,
                total_started=stage_total_started,
                decode_started=decode_started,
                condition_seconds=condition_seconds,
                denoise_seconds=denoise_seconds,
            )
            return output
        if self.joint_track_depth_prediction and v2v_concat_mode == "width":
            w_per_domain = decode_latents.shape[4] // num_domains
            target_rgb_lat = target_latents_all[:, :, :, :, :w_per_domain]
            target_track_lat = target_latents_all[:, :, :, :, w_per_domain : 2 * w_per_domain]
            target_depth_lat = target_latents_all[:, :, :, :, 2 * w_per_domain :]

            video = self.vae.decode(
                target_rgb_lat, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
            video = self.vae_output_to_video(video)

            target_track_decoded = self.vae.decode(
                target_track_lat, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
            target_depth_decoded = self.vae.decode(
                target_depth_lat, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            ).mean(dim=1, keepdim=True)
            output = {
                "video": video,
                "track_tensor": target_track_decoded,
                "depth_tensor": target_depth_decoded,
                "track_normalization_stats": inputs_shared.get("target_track_normalization_stats"),
                "depth_normalization_stats": inputs_shared.get("target_depth_normalization_stats"),
            }
            self.load_models_to_device([])
            self._record_stage_profile(
                enabled=profile_stage_timings,
                total_started=stage_total_started,
                decode_started=decode_started,
                condition_seconds=condition_seconds,
                denoise_seconds=denoise_seconds,
            )
            return output
        if self.joint_track_prediction and v2v_concat_mode == "width":
            # 3-domain: split target_latents_all into [target_rgb | target_track]
            w_per_domain = decode_latents.shape[4] // num_domains
            target_rgb_lat = target_latents_all[:, :, :, :, :w_per_domain]
            target_track_lat = target_latents_all[:, :, :, :, w_per_domain:]

            # Decode target RGB
            video = self.vae.decode(
                target_rgb_lat, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
            video = self.vae_output_to_video(video)

            # Decode target track (returns raw tensor, not video frames)
            target_track_decoded = self.vae.decode(
                target_track_lat, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
            self.load_models_to_device([])
            self._record_stage_profile(
                enabled=profile_stage_timings,
                total_started=stage_total_started,
                decode_started=decode_started,
                condition_seconds=condition_seconds,
                denoise_seconds=denoise_seconds,
            )
            return video, target_track_decoded
        else:
            # 2-domain: decode target RGB only
            decode_latents = target_latents_all if (v2v_concat_mode == "width" and il is not None) else decode_latents
            video = self.vae.decode(
                decode_latents, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
            video = self.vae_output_to_video(video)
            self.load_models_to_device([])
            self._record_stage_profile(
                enabled=profile_stage_timings,
                total_started=stage_total_started,
                decode_started=decode_started,
                condition_seconds=condition_seconds,
                denoise_seconds=denoise_seconds,
            )
            return video
