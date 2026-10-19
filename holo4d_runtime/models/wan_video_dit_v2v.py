"""
V2V-aware Wan DiT model.

`WanModelV2V` is a strict superset of `WanModel` that adds a trainable domain embedding
used by the V2V pipeline to distinguish conditioning (RGB) and target (disp/depth)
tokens after concatenation.

The model keeps the same forward/patchify behavior as the base `WanModel`; V2V-specific
logic lives in `diffsynth.pipelines.wan_video_v2v.model_fn_wan_video_v2v`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .wan_video_dit import WanModel as WanModelBase, Head, sinusoidal_embedding_1d
from pdb import set_trace as st


class WanModelV2V(WanModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_embedding = nn.Parameter(torch.zeros(2, self.dim))
        nn.init.normal_(self.domain_embedding, std=0.02)
        self.v2v_channel_patch_embed_mode = "add"
        self.patch_embedding_v2v: Optional[nn.Conv3d] = None
        self.control_patch_embedding: Optional[nn.Conv3d] = None
        self._multimodal_patch_embeddings: nn.ModuleDict = nn.ModuleDict()
        # GS feature head (lazy-init via ensure_gs_feature_head)
        self.gs_feature_head: Optional[Head] = None
        # FlashWorld-style feedback channels (0 = disabled)
        self._feedback_channels: int = 0
        # Future-prediction source-domain null (lazy-init, see
        # initialize_future_source_null). None for every non-future run → the
        # module state_dict is byte-identical to today unless future prediction
        # is explicitly enabled.
        self.future_source_null: Optional[nn.Parameter] = None

    def initialize_future_source_null(self, num_channels: int) -> nn.Parameter:
        """Create the learned per-channel source-domain null for future frames.

        Zero-initialized so that, even before the optimizer touches it, the
        future-frame source region is just zeros (a clean OOD sentinel). Only
        created when --future_prediction is set; must be added to
        ``trainable_models`` (``dit.future_source_null``) to actually learn.
        """
        if self.future_source_null is None:
            ref = self.patch_embedding.weight
            param = nn.Parameter(torch.zeros(int(num_channels)))
            if getattr(ref.device, "type", None) != "meta":
                param = nn.Parameter(param.data.to(device=ref.device, dtype=ref.dtype))
            self.future_source_null = param
        return self.future_source_null

    def _ensure_patch_embedding_multimodal(self, num_modalities: int) -> nn.Conv3d:
        """Conv3d for N-modal channel-concat patchify. Initialized by tiling base weight."""
        key = str(num_modalities)
        if key in self._multimodal_patch_embeddings:
            return self._multimodal_patch_embeddings[key]
        conv = nn.Conv3d(
            self.in_dim * num_modalities, self.dim,
            kernel_size=self.patch_size, stride=self.patch_size,
            bias=self.patch_embedding.bias is not None,
        )
        ref_weight = self.patch_embedding.weight
        if getattr(ref_weight.device, "type", None) != "meta":
            conv = conv.to(device=ref_weight.device, dtype=ref_weight.dtype)
            with torch.no_grad():
                for m in range(num_modalities):
                    conv.weight[:, m * self.in_dim:(m + 1) * self.in_dim].copy_(ref_weight)
                if conv.bias is not None and self.patch_embedding.bias is not None:
                    conv.bias.copy_(self.patch_embedding.bias * num_modalities)
        self._multimodal_patch_embeddings[key] = conv
        return conv

    def expand_multimodal_patch_embedding(self, old_num: int, new_num: int) -> nn.Conv3d:
        """Expand N-modal → M-modal patch embedding; new channels zero-initialized."""
        if new_num <= old_num:
            raise ValueError(f"new_num ({new_num}) must be > old_num ({old_num})")
        old_conv = self._ensure_patch_embedding_multimodal(old_num)
        conv = nn.Conv3d(
            self.in_dim * new_num, self.dim,
            kernel_size=self.patch_size, stride=self.patch_size,
            bias=old_conv.bias is not None,
        )
        if getattr(old_conv.weight.device, "type", None) != "meta":
            conv = conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        with torch.no_grad():
            nn.init.zeros_(conv.weight)
            conv.weight[:, :old_num * self.in_dim].copy_(old_conv.weight)
            if conv.bias is not None and old_conv.bias is not None:
                conv.bias.copy_(old_conv.bias)
        self._multimodal_patch_embeddings[str(new_num)] = conv
        return conv

    def _ensure_patch_embedding_v2v(self) -> nn.Conv3d:
        if self.patch_embedding_v2v is not None:
            return self.patch_embedding_v2v

        conv = nn.Conv3d(
            self.in_dim * 2,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.patch_embedding.bias is not None,
        )

        ref_weight = self.patch_embedding.weight
        if getattr(ref_weight.device, "type", None) != "meta":
            conv = conv.to(device=ref_weight.device, dtype=ref_weight.dtype)
            with torch.no_grad():
                conv.weight[:, : self.in_dim].copy_(self.patch_embedding.weight)
                conv.weight[:, self.in_dim :].copy_(self.patch_embedding.weight)
                if conv.bias is not None and self.patch_embedding.bias is not None:
                    conv.bias.copy_(self.patch_embedding.bias * 2)

        self.patch_embedding_v2v = conv
        return conv

    def ensure_control_patch_embedding(self, in_channels: Optional[int] = None) -> nn.Conv3d:
        """
        Ensure the zero-init control patch embedding exists.

        This layer is only needed when `control_latents_input` is provided (xyz-video control).
        Keeping it lazy avoids introducing redundant parameters for other tasks (e.g., camera control).
        """
        if in_channels is None:
            in_channels = int(self.in_dim)
        in_channels = int(in_channels)

        if self.control_patch_embedding is not None:
            if int(getattr(self.control_patch_embedding, "in_channels", -1)) == in_channels:
                return self.control_patch_embedding

            old = self.control_patch_embedding
            conv = nn.Conv3d(
                in_channels,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=old.bias is not None,
            )

            ref_weight = old.weight
            if getattr(ref_weight.device, "type", None) != "meta":
                conv = conv.to(device=ref_weight.device, dtype=ref_weight.dtype)

            with torch.no_grad():
                nn.init.zeros_(conv.weight)
                if conv.bias is not None:
                    if old.bias is not None and getattr(old.bias.device, "type", None) != "meta":
                        conv.bias.copy_(old.bias)
                    else:
                        nn.init.zeros_(conv.bias)

                copy_c = min(int(old.in_channels), int(in_channels))
                if getattr(ref_weight.device, "type", None) != "meta":
                    conv.weight[:, :copy_c].copy_(old.weight[:, :copy_c])

            self.control_patch_embedding = conv
            return conv

        conv = nn.Conv3d(
            in_channels,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.patch_embedding.bias is not None,
        )

        ref_weight = self.patch_embedding.weight
        if getattr(ref_weight.device, "type", None) != "meta":
            conv = conv.to(device=ref_weight.device, dtype=ref_weight.dtype)

        with torch.no_grad():
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        self.control_patch_embedding = conv
        return conv

    def widen_patch_embedding_for_feedback(self, feedback_channels: int) -> None:
        """Widen patch_embedding to accept feedback channels (FlashWorld-style).

        After this call, patchify() expects input of shape
        ``(B, in_dim + feedback_channels, T, H, W)`` where the last
        ``feedback_channels`` are ``cat([prev_feats, prev_latents_pred])``.

        The new input-channel slices are zero-initialized so that loading
        old checkpoints (which only have ``in_dim`` channels) and passing
        zeros for feedback reproduces the original behavior exactly.
        """
        if feedback_channels <= 0:
            return
        if self._feedback_channels == feedback_channels:
            return  # already widened

        old_conv = self.patch_embedding
        old_in = int(old_conv.in_channels)
        new_in = old_in + feedback_channels

        new_conv = nn.Conv3d(
            new_in, self.dim,
            kernel_size=self.patch_size, stride=self.patch_size,
            bias=old_conv.bias is not None,
        )

        ref_weight = old_conv.weight
        if getattr(ref_weight.device, "type", None) != "meta":
            new_conv = new_conv.to(device=ref_weight.device, dtype=ref_weight.dtype)

        with torch.no_grad():
            new_conv.weight[:, :old_in].copy_(ref_weight)
            new_conv.weight[:, old_in:].zero_()
            if new_conv.bias is not None and old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.patch_embedding = new_conv
        self._feedback_channels = feedback_channels

    def set_v2v_channel_patch_embed_mode(self, mode: str) -> None:
        mode = "cat" if mode == "concat" else mode
        if mode not in ("add", "cat"):
            raise ValueError(f"Invalid v2v_channel_patch_embed_mode: {mode}. Expected 'add' or 'cat'.")
        if mode == "cat":
            self._ensure_patch_embedding_v2v()
        self.v2v_channel_patch_embed_mode = mode

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        v2v_concat_mode: Optional[str] = None,
        multimodality_nvs: bool = False,
        control_latents_input: Optional[torch.Tensor] = None,
        feedback_input: Optional[torch.Tensor] = None,
    ):
        # Handle camera control in V2V mode: camera rays describe the target /
        # novel-view trajectory, so they should condition the target domains,
        # not the clean source-conditioning domain.
        has_camera_control = self.control_adapter is not None and control_camera_latents_input is not None

        if int(x.shape[1]) == int(self.in_dim):
            # Standard mode: single input OR width/frame concat (not channel concat)
            if control_latents_input is not None:
                if int(control_latents_input.shape[0]) != int(x.shape[0]):
                    raise ValueError(
                        f"control_latents_input batch must match x. "
                        f"Got control_latents_input={tuple(control_latents_input.shape)} vs x={tuple(x.shape)}."
                    )
                if tuple(control_latents_input.shape[2:]) != tuple(x.shape[2:]):
                    raise ValueError(
                        f"control_latents_input (T,H,W) must match x. "
                        f"Got control_latents_input={tuple(control_latents_input.shape)} vs x={tuple(x.shape)}."
                    )

            # FlashWorld-style feedback: concatenate prev_feats + prev_latents_pred
            # to the latent input before the main patch embedding.
            if self._feedback_channels > 0:
                if feedback_input is not None:
                    x = torch.cat([x, feedback_input], dim=1)
                else:
                    x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, self._feedback_channels))

            x = self.patch_embedding(x)
            # assert control_latents_input is not None # just to debug
            if control_latents_input is not None:
                control_patch_embedding = self.ensure_control_patch_embedding(int(control_latents_input.shape[1]))
                x = x + control_patch_embedding(control_latents_input)
            if has_camera_control:
                y_camera = self.control_adapter(control_camera_latents_input) # B dim T H W

                # For frame/width-concat modes, camera control needs to be repeated
                # ``multimodality_nvs`` is historically a bool, but V2V callers may
                # now pass an explicit integer domain count for >3-domain layouts.
                if isinstance(multimodality_nvs, bool):
                    num_domains = 3 if multimodality_nvs else 2
                else:
                    num_domains = int(multimodality_nvs)
                    if num_domains < 2:
                        raise ValueError(f"Expected multimodality domain count >= 2, got {num_domains}")
                if v2v_concat_mode == "width":
                    # Width-concat layout is [source | target_1 | target_2 | ...].
                    # Keep the clean source domain unconditioned and apply the
                    # camera signal to the target-side domains only.
                    if num_domains >= 2:
                        y_camera = torch.cat(
                            [
                                torch.zeros_like(y_camera),
                                y_camera.repeat(1, 1, 1, 1, num_domains - 1),
                            ],
                            dim=4,
                        )
                elif v2v_concat_mode == "frame":
                    # Frame-concat layout is [source ; target_1 ; target_2 ; ...].
                    # Again, only the target-side domains receive the camera signal.
                    if num_domains >= 2:
                        y_camera = torch.cat(
                            [
                                torch.zeros_like(y_camera),
                                y_camera.repeat(1, 1, num_domains - 1, 1, 1),
                            ],
                            dim=2,
                        )

                x = x + y_camera

        elif int(x.shape[1]) == int(self.in_dim) * 2:
            # V2V mode: channel wise concatenated source + target
            if control_latents_input is not None:
                raise NotImplementedError("control_latents_input is not supported when x is already channel-concatenated (2*in_dim).")
            mode = getattr(self, "v2v_channel_patch_embed_mode", "add")
            mode = "cat" if mode == "concat" else mode

            if mode == "add":
                x_cond, x_target = x.chunk(2, dim=1)
                x_cond_patched = self.patch_embedding(x_cond)
                x_target_patched = self.patch_embedding(x_target)

                # Channel-concat add-mode still keeps source and target separate
                # before fusion, so only the target half should receive the
                # target-camera signal.
                if has_camera_control:
                    y_camera = self.control_adapter(control_camera_latents_input)
                    x_target_patched = x_target_patched + y_camera

                # Combine source and target
                x = x_cond_patched + x_target_patched

            elif mode == "cat":
                if self.patch_embedding_v2v is None:
                    raise RuntimeError(
                        "v2v_channel_patch_embed_mode='cat' requires patch_embedding_v2v to be initialized. "
                        "Call set_v2v_channel_patch_embed_mode('cat') before running forward (training/inference)."
                    )
                x = self.patch_embedding_v2v(x)
                # In cat mode, camera control is added to entire concatenated features
                if has_camera_control:
                    y_camera = self.control_adapter(control_camera_latents_input)
                    x = x + y_camera
            else:
                raise ValueError(f"Invalid v2v_channel_patch_embed_mode: {mode}. Expected 'add' or 'cat'.")
        elif int(x.shape[1]) > int(self.in_dim) and int(x.shape[1]) % int(self.in_dim) == 0:
            # N-modal channel-concat (N > 2): Conv3d(N*in_dim, dim, ...)
            num_modalities = int(x.shape[1]) // int(self.in_dim)
            conv = self._ensure_patch_embedding_multimodal(num_modalities)
            x = conv(x)
            if control_latents_input is not None:
                raise NotImplementedError(
                    f"control_latents_input not supported for {num_modalities}-modal channel-concat."
                )
            if has_camera_control:
                y_camera = self.control_adapter(control_camera_latents_input)
                if isinstance(multimodality_nvs, bool):
                    num_domains = 3 if multimodality_nvs else 2
                else:
                    num_domains = int(multimodality_nvs)
                if v2v_concat_mode == "width":
                    y_camera = y_camera.repeat(1, 1, 1, 1, num_domains)
                elif v2v_concat_mode == "frame":
                    y_camera = y_camera.repeat(1, 1, num_domains, 1, 1)
                x = x + y_camera
        else:
            raise ValueError(
                f"Unexpected input channel count for V2V patchify: got C={int(x.shape[1])}, "
                f"expected C={self.in_dim} (standard) or C={self.in_dim * 2} (channel concat)."
            )

        return x

    def ensure_gs_feature_head(self, feat_dim: int = 1024) -> "Head":
        """Lazy-init a parallel GS feature head.

        Reads the same pre-head hidden states as the existing Head and
        produces feat_dim-channel features for the GS decoder.

        Initialization policy (FIXED 2026-04-16 after the cascade
        zero-init diagnosis — see
        doc/new_features/2026-04-16-kubric-dynamic-gs-head-dead-path-analysis.md):
          - Do NOT zero-init ``gs_head.head.weight``. Use the default
            Kaiming init ``nn.Linear`` provides. Zero-init here combined
            with zero-init on the GS decoder's ``extra_conv_in`` (which
            consumes these features) creates a dead-gradient cascade: both
            layers get zero gradient until the other is non-zero, so they
            stay at zero forever. Observed in step-15000 of the Kubric49
            pair01 overfit run.
          - The "no disruption at init" guarantee is preserved on the
            decoder side by ``extra_conv_in`` remaining zero-init
            (``holo4d/models/gs_decoder.py:248-250``), so the GS
            decoder still behaves identically to the VAE decoder at step
            0. Only the DiT-side adapter carries random init, which is
            fine because its output only flows into ``extra_conv_in`` and
            is gated to zero by that layer's own zero weight at init.
        """
        if self.gs_feature_head is not None:
            return self.gs_feature_head
        eps = self.head.norm.eps if hasattr(self.head.norm, "eps") else 1e-6
        gs_head = Head(self.dim, feat_dim, self.patch_size, eps)
        # Leave ``gs_head.head.weight/bias`` at nn.Linear's default Kaiming
        # init — intentionally non-zero. Zero-init here is what breaks the
        # gradient cascade into extra_conv_in on the decoder side.
        # Move to same device/dtype as the existing head
        ref = self.head.head.weight
        if getattr(ref.device, "type", None) != "meta":
            gs_head = gs_head.to(device=ref.device, dtype=ref.dtype)
        self.gs_feature_head = gs_head
        return gs_head


__all__ = ["WanModelV2V", "sinusoidal_embedding_1d"]
