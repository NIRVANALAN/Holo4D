"""Shared CLI argument groups for Track4D-360 train + eval scripts.

Both ``scripts/wan2.1_fun/train_track4d360.py`` and the eval scripts under
``examples/wan2.1_fun/eval_track4d360_*.py`` need identical arg definitions
for GS output, plucker camera control, track config, and dense branch.
Defining them once here avoids the "eval parser missing a flag that training
added" class of bug (e.g., the --use_gs_output incident).

Usage::

    from holo4d.shared_args import (
        add_gs_output_args,
        add_plucker_args,
        add_track_config_args,
        add_dense_branch_args,
    )
    parser = argparse.ArgumentParser(...)
    add_gs_output_args(parser)
    add_plucker_args(parser)
    ...
"""

from __future__ import annotations

import argparse
import os


def add_i2v_condition_args(parser: argparse.ArgumentParser) -> None:
    """I2V conditioning modes shared by training and evaluation."""
    group = parser.add_argument_group("I2V conditioning")
    group.add_argument(
        "--i2v_target_condition",
        action="store_true",
        default=False,
        help=(
            "Use the legacy Wan 2.1 I2V VAE+CLIP extra-channel path. Requires "
            "an I2V backbone; mutually exclusive with --ti2v_native_i2v."
        ),
    )
    group.add_argument(
        "--ti2v_native_i2v",
        action="store_true",
        default=False,
        help=(
            "Use Wan 2.2 TI2V-native target-frame-0 latent conditioning. "
            "Keeps the V2V width-concat source and masks conditioned frame 0 "
            "from the target RGB diffusion loss."
        ),
    )
    group.add_argument(
        "--i2v_condition_drop_prob",
        type=float,
        default=0.3,
        help=(
            "Training probability of dropping the target boundary condition "
            "and falling back to ordinary V2V for that step."
        ),
    )
    group.add_argument(
        "--i2v_condition_noise_level",
        type=float,
        default=0.05,
        help="Boundary-latent augmentation alpha in (1-alpha)*x + alpha*noise.",
    )


def add_gs_output_args(parser: argparse.ArgumentParser) -> None:
    """3D Gaussian Splatting output head — shared between train + eval."""
    group = parser.add_argument_group("GS output (3D Gaussian Splatting)")
    group.add_argument("--gs_head_style", type=str, default="flashworld",
                       choices=["flashworld", "hybrid"],
                       help="GS head activation style. 'flashworld' (default — "
                            "existing main-trainer behavior): warm-start from the VAE "
                            "decoder output head. 'hybrid': warm VAE decoder body + "
                            "fresh Lyra-style activation head (LyraPixelAligned3DGS, "
                            "direct RGB + sigmoid-lerp depth + exp+cap scale). Swap "
                            "is applied at GS decoder build time so warm-start / "
                            "resume paths both see the final head structure. "
                            "See holo4d/models/gs_decoder_lyra.py.")
    group.add_argument("--use_gs_output", action="store_true", default=False,
                       help="Enable 3DGS output branch: parallel GS feature head + GS decoder + splatting loss.")
    group.add_argument("--gs_zero_features", action="store_true", default=False,
                       help="Zero out `gs_features_target` before the GS decoder "
                            "call. Matches the cache-based overfit's "
                            "`--feature_mode x0_mid` semantics (train_gs_head_from_cache.py "
                            "substitutes zero features at that flag). Use when "
                            "you want the decoder output to depend purely on "
                            "the x0-hat latent (`z = latents_pred.detach()`), "
                            "with no contribution from the DiT feature-head / "
                            "`extra_conv_in` side channel. Projector still runs "
                            "but its output is discarded.")
    group.add_argument("--gs_feat_dim", type=int, default=1024,
                       help="Feature dimension for GS feature head output.")
    group.add_argument("--gs_loss_weight", type=float, default=1.0,
                       help="Outer weight of the GS loss relative to diffusion loss.")
    group.add_argument("--gs_mse_loss_weight", type=float, default=1.0,
                       help="Weight of RGB MSE within the GS loss.")
    group.add_argument("--gs_lpips_loss_weight", type=float, default=0.5,
                       help="Weight of LPIPS(VGG) within the GS loss.")
    group.add_argument("--gs_depth_loss_weight", type=float, default=0.05,
                       help="Weight of scale-invariant depth loss within the GS loss.")
    group.add_argument("--gs_abs_depth_loss_weight", type=float, default=0.0,
                       help="Weight of absolute foreground log-depth L1. "
                            "Complements --gs_depth_loss_weight (scale-invariant) "
                            "by tying down the absolute scale of the rendered "
                            "depth on track-less batches. Foreground masking via "
                            "the dataset's target_depth_valid_mask (sky / invalid "
                            "pixels excluded). Recommended for v4 when batches "
                            "include MVS-Synth / DL3DV (no persistent tracks → "
                            "absolute scale would otherwise drift). 0 disables.")
    group.add_argument("--gs_opacity_loss_weight", type=float, default=0.1,
                       help="Weight of opacity L1 regularization within the GS loss.")
    group.add_argument("--gs_opacity_prune_keep_frac", type=float, default=1.0,
                       help="Legacy static-GS render pruning knob. 1.0 (default) "
                            "renders all predicted Gaussians. Values in (0,1), "
                            "for example 0.2, zero opacity outside the top "
                            "fraction per frame before render. Kept only for "
                            "old v3b compatibility; new per-frame runs should "
                            "leave this at 1.0 so train/debug/eval render the "
                            "same cloud.")
    group.add_argument("--gs_rgb_loss_weight", type=float, default=None,
                       help="Deprecated alias for --gs_mse_loss_weight.")
    group.add_argument("--gs_sh_degree", type=int, default=2,
                       help="Spherical harmonics degree for Gaussian color representation.")
    group.add_argument("--gs_num_points_per_pixel", type=int, default=2,
                       help="Number of Gaussians predicted per pixel.")
    group.add_argument("--no_gs_latent_detach", action="store_true", default=False,
                       help="Backprop GS/render losses through predicted clean target latents into "
                            "dit.head (joint VDM+3DGS). Default keeps detaching latents for the GS decoder.")
    group.add_argument("--use_gs_feedback_training", action="store_true", default=False,
                       help="FlashWorld-style 2-forward GS feedback training: forward 1 (bootstrap, "
                            "no feedback) → GS decode + render + VAE encode → forward 2 (with "
                            "prev_feats + prev_latents_pred feedback channels). Requires widened "
                            "patch embedding (auto-initialized).")
    group.add_argument("--gs_scale_cap", type=float, default=None,
                       help="Hard max on GS scales (world units) after activation. "
                            "None = disabled (default). Lyra-1 convention is 0.3.")
    group.add_argument("--gs_render_patch_size", type=int, nargs=2, default=None,
                       metavar=("PH", "PW"),
                       help="Tile size (height width) for patch-based GS rendering. "
                            "None = full-frame (default). Must divide H and W exactly. "
                            "Lyra-1 DeferredBPPatch style: reduces peak GS render memory.")
    group.add_argument("--gs_legacy_union_render", action="store_true", default=False,
                       help="Render every target view from the union of all per-frame "
                            "Gaussian sets, matching pre-2026-04-26 v3b GS-head "
                            "training. Default renders view t from Gaussian set t "
                            "only, which is the intended per-frame video behavior "
                            "for new runs but is not compatible with old v3b "
                            "checkpoints.")
    group.add_argument("--gs_lyra_depth_param", type=str, default="exp",
                       choices=["sigmoid_lerp", "exp"],
                       help="Lyra/hybrid head depth activation. 'exp' (default, "
                            "DAV3 / FlashWorld convention): trunc_exp(distance + "
                            "log_depth_shift) — unbounded, log-parameterised, no "
                            "dnear/dfar tuning needed. 'sigmoid_lerp' (legacy): "
                            "dnear+(dfar-dnear)*sigmoid(distance) — bounded, "
                            "needs per-scene dfar tuning, hits a ceiling on "
                            "wide-baseline outdoor data.")
    group.add_argument("--gs_lyra_log_depth_shift", type=float, default=0.0,
                       help="Bias added to the predicted distance before exp() "
                            "in 'exp' mode. 0.0 → init depth = exp(0) = 1.0 "
                            "(matches post-common_scale unit-mean). Ignored "
                            "under 'sigmoid_lerp'.")
    group.add_argument("--gs_lyra_dnear", type=float, default=0.1,
                       help="Lyra head near-plane (only used when "
                            "--gs_lyra_depth_param sigmoid_lerp).")
    group.add_argument("--gs_lyra_dfar", type=float, default=10.0,
                       help="Lyra head far-plane (only used when "
                            "--gs_lyra_depth_param sigmoid_lerp).")
    group.add_argument("--gs_replay_cache_size", type=int, default=0,
                       help="Overfit accelerator: cache the GS-loss inputs "
                            "(gs_features_target, latents_pred_target, GT) from the "
                            "first N steps, then on every subsequent step pick a "
                            "random cached entry and skip the DiT forward entirely. "
                            "≈2x speed-up for GS-only training. 0 disables. Only "
                            "honoured when --skip_diffusion_loss is also set "
                            "(replay path produces no DiT gradient). Recommended: "
                            "100 for 1-sample overfits, larger for multi-sample. "
                            "Cache lives on CPU; ~50 MB/entry at 144x256x49.")


def add_gs_dynamic_args(parser: argparse.ArgumentParser) -> None:
    """Dynamic 3DGS head: canonical first-frame Gaussians + per-frame deltas."""
    group = parser.add_argument_group("Dynamic GS (canonical + deltas)")
    group.add_argument("--gs_dynamic_mode", type=str, default="static",
                       choices=["static", "deltas", "static_with_track"],
                       help="GS head mode. 'static' = per-latent-frame "
                            "independent Gaussians (rendering only). 'deltas' = "
                            "canonical first-frame Gaussians + per-frame deltas "
                            "(both rendering AND tracking; assumes shared geometry "
                            "across time). 'static_with_track' (v4) = per-frame "
                            "independent Gaussians for rendering + parallel "
                            "per-pixel-per-frame 3D delta head supervised by "
                            "target_tracks_world only (track losses ZERO-out for "
                            "datasets without 3D tracks like MVS-Synth/DL3DV; "
                            "fully active for Kubric). Best of both: per-frame "
                            "freedom for static scenes + dynamic-mode track supervision.")
    group.add_argument("--gs_track_supervision_mode", type=str, default="pointmap",
                       choices=["pointmap", "persistent_tracks"],
                       help="Geometry supervision target for the dynamic path. "
                            "'pointmap' (default): rendered XYZ pointmap vs target_track_video. "
                            "'persistent_tracks': frame-0 query tracks from target_tracks_world "
                            "with direct canonical/delta/final-position supervision.")
    group.add_argument("--gs_pos_loss_weight", type=float, default=1.0,
                       help="Weight of the rendered-position-vs-target_track_video L1 term.")
    group.add_argument("--gs_dyn_depth_loss_weight", type=float, default=0.5,
                       help="Weight of the rendered-depth-vs-target_depth_video masked L1 term "
                            "(dynamic path only, independent of --gs_depth_loss_weight).")
    group.add_argument("--gs_opa_tv_weight", type=float, default=0.01,
                       help="Weight of per-frame opacity total-variation smoothness.")
    group.add_argument("--gs_xyz_tv_weight", type=float, default=0.01,
                       help="Weight of per-frame delta-xyz total-variation smoothness.")
    group.add_argument("--gs_predict_rotation", action="store_true",
                       help="Predict a 4-channel quaternion delta per Gaussian per frame.")
    group.add_argument("--gs_disable_delta_opacity", action="store_true",
                       help="Keep opacity canonical across time by ignoring the per-frame "
                            "Δopacity_logit channels. Useful for isolating whether dynamic "
                            "opacity is hurting tracking quality.")
    group.add_argument("--gs_direct_pos_loss_weight", type=float, default=0.0,
                       help="SoM-style direct-center supervision: masked L1 between the "
                            "frame-0-associated Gaussian center xyz and GT target_track_video. "
                            "0 = disabled (default). Recommended start: 0.05–0.1.")
    group.add_argument("--gs_track_canonical_loss_weight", type=float, default=0.0,
                       help="Persistent-track mode only: direct frame-0 anchor supervision on "
                            "canonical_xyz against target_tracks_world[:, 0].")
    group.add_argument("--gs_track_delta_loss_weight", type=float, default=0.0,
                       help="Persistent-track mode only: direct supervision on delta_xyz against "
                            "target_tracks_world[:, t] - target_tracks_world[:, 0].")
    group.add_argument("--gs_track_final_pos_loss_weight", type=float, default=0.0,
                       help="Persistent-track mode only: optional direct supervision on the final "
                            "position canonical_xyz + delta_xyz against target_tracks_world[:, t].")
    group.add_argument("--gs_dynamic_max_t_video", type=int, default=64,
                       help="Maximum number of target-video frames the dynamic GS head "
                            "reserves delta output channels for. Must be >= T_video at runtime.")

    # -- SoM-inspired track losses (Objective A per 2026-04-22 spec §1-bis) ----
    group.add_argument("--gs_assoc_track_2d_loss_weight", type=float, default=0.0,
                       help="Objective A (from 2026-04-22-som-track-supervision-spec.md §1-bis). "
                            "L1 loss in pixel units between the associated Gaussian's projected "
                            "trajectory and the persistent target_tracks_world reshaped to "
                            "(T, H, W, 3) row-major, projected through the same per-frame "
                            "target cameras. Quantile-clipped and normalized by max(H, W). "
                            "Addresses the GT-source semantic mismatch: target_tracks_world is "
                            "persistent, target_track_video is per-frame re-raster. The overlay "
                            "panel draws the former. Default 0 = disabled.")
    group.add_argument("--gs_track_accel_loss_weight", type=float, default=0.0,
                       help="Associated-trajectory acceleration smoothness: L1 of "
                            "|2*xyz[t] − xyz[t-1] − xyz[t+1]| on the per-pixel associated Gaussian "
                            "trajectory (world units, masked on valid pixels). Directly suppresses "
                            "per-frame velocity jitter (`pxv` metric). Start lower on _noopa runs.")
    group.add_argument("--gs_track_loss_quantile", type=float, default=0.98,
                       help="Quantile cutoff for masked_l1_loss used by the assoc 2D track loss: "
                            "values above this quantile are clipped to zero so outlier pixels do "
                            "not dominate. Ported from SoM flow3d/loss_utils.py:50.")
    group.add_argument("--gs_track_canonical_detach", type=lambda s: s.lower() in ("1", "true", "yes"),
                       default=False,
                       help="v4 only: detach gaussian_params[:, 0] before track losses. Default "
                            "False — preserves cache-v3 behavior where track GT directly supervises "
                            "render-head's gs_proj at frame 0, anchoring monocular depth (photometric "
                            "alone is depth-ambiguous: many xyz solutions render the same RGB). Set "
                            "True for an ablation isolating track-head learning from render-head "
                            "supervision (matches the doc claim 'track head never enters rendering' "
                            "literally). If detach=False causes early-step PSNR regression because "
                            "the world-unit canonical loss (~1m) dwarfs RGB MSE (~0.05), prefer "
                            "halving --gs_track_canonical_loss_weight over flipping this to True.")
    group.add_argument("--gs_track_joint_render", type=lambda s: s.lower() in ("1", "true", "yes"),
                       default=False,
                       help="v4 only: at every t, also render canonical warped by Δxyz[t] alongside "
                            "the per-frame Gaussians (alpha-blended via z-buffer in one render call). "
                            "track_delta_proj then receives photometric gradient — not only track-GT "
                            "gradient — and the npp slot-permutation risk between render and track "
                            "heads self-resolves through joint optimization. Cost: ≈2× render compute "
                            "at t>0 (Gaussian count doubles). Default False (current v4 behavior).")


def add_plucker_args(parser: argparse.ArgumentParser) -> None:
    """Plucker ray camera control adapter."""
    group = parser.add_argument_group("Plucker camera control")
    group.add_argument("--use_plucker_camera_control", action="store_true", default=False,
                       help="Enable Plucker-ray camera control via SimpleAdapter.")
    group.add_argument("--enable_v2v_plucker_camera_control", action="store_true", default=False,
                       help="Actually route Track4D-360 V2V camera rays through the Plucker adapter. "
                            "Default keeps legacy behavior where --use_plucker_camera_control only loads it.")
    group.add_argument("--train_plucker_camera_control", action="store_true",
                       help="Finetune the Plucker SimpleAdapter. Default keeps it frozen.")
    group.add_argument("--use_plucker_raymap_tokens", action="store_true", default=False,
                       help="Opt in to Plucker raymap token conditioning: patchified 6D raymaps "
                            "are projected with zero-init Ecam and added to video tokens before "
                            "DiT self-attention. Default preserves the legacy adapter/noisy-latent path.")
    group.add_argument("--plucker_use_ray_origin", action="store_true", default=False,
                       help="Use [rays_o, rays_d] representation (Lyra-2 convention) instead of "
                            "cross-product [rays_o x rays_d, rays_d]. Must match between train and eval.")


def add_warped_condition_args(parser: argparse.ArgumentParser) -> None:
    """Lyra-2-inspired warped RGB appearance conditioning mode."""
    group = parser.add_argument_group("Warped appearance conditioning")
    group.add_argument("--warped_condition_mode", type=str, default="off",
                       choices=["off", "latent_fuse"],
                       help="Warped appearance conditioning mode. "
                            "'off' (default): existing dense+track pipeline unchanged. "
                            "'latent_fuse': soft-splat warped source RGB -> VAE encode -> "
                            "shallow fusion before DiT blocks (Lyra-2-inspired).")
    group.add_argument("--warped_appearance_fusion", type=str, default="concat_proj",
                       choices=["add", "concat_proj"],
                       help="How to fuse warped RGB latent with DiT input. "
                            "'add': simple additive (x = x + proj(warped)). "
                            "'concat_proj': concat then project (x = proj(cat(x, warped))).")
    group.add_argument("--warped_geom_only_dense", action="store_true", default=True,
                       help="In latent_fuse mode, make dense branch geometry-only "
                            "(depth+valid, no RGB). Default True to avoid double-counting appearance.")
    group.add_argument("--no_warped_geom_only_dense", dest="warped_geom_only_dense",
                       action="store_false",
                       help="Keep RGB in dense branch even in latent_fuse mode.")
    group.add_argument("--multi_frame_warped_appearance", action="store_true",
                       default=False,
                       help="When set together with --warped_condition_mode latent_fuse, "
                            "produce a HYBRID warped RGB: at static target pixels, fuse "
                            "all T source frames (Lyra-2 forward_warp_multiframes); at "
                            "dynamic target pixels, fall back to the per-frame s=t->t "
                            "splat. Static / dynamic separation comes from the per-pixel "
                            "motion range of `tracks_world` (Kubric_NVS / Syn4D); when "
                            "tracks are absent the dataset is treated as pure static and "
                            "the multi-frame fill covers everywhere. Default False (= "
                            "classic per-frame Lyra-2 Path A, fully back-compat).")
    group.add_argument("--multi_frame_motion_threshold", type=float, default=0.01,
                       help="Per-pixel max-frame motion threshold (cam0-normalised "
                            "scene units; the dataset divides by `common_scale` so 1.0 "
                            "is roughly scene depth). 0.01 = 1%% of scene scale, tight "
                            "enough to filter Kubric falling balls out of the multi-"
                            "frame fill. Only used when "
                            "--multi_frame_warped_appearance is set.")


def add_perlayer_camera_args(parser: argparse.ArgumentParser) -> None:
    """Per-layer camera control — Lyra-2-style per-block encoding + KQ-only injection."""
    group = parser.add_argument_group("Per-layer camera control")
    group.add_argument("--use_perlayer_camera_control", action="store_true", default=False,
                       help="Enable per-layer camera control (per-block cam_encoder).")
    group.add_argument("--camera_injection_mode", type=str, default="projector",
                       choices=["projector", "add", "direct_raymap_add", "rt_projector", "kq_only", "vista4d_plucker"],
                       help="Camera injection mode. "
                            "'kq_only' (Lyra-2-style): per-layer cam_encoder, camera bias affects only Q/K, not V. "
                            "'vista4d_plucker': raw 6D Plucker tokens at DiT resolution, per-block "
                            "zero-init Linear(6->dim), identity attention projector.")
    group.add_argument("--camera_plucker_encoder_mode", type=str, default="patchify",
                       choices=["patchify", "control_adapter"],
                       help="Plucker encoder for per-layer modes: "
                            "'patchify' (PixelUnshuffle+Conv2d 24->6, then per-block Conv3d 6->dim), "
                            "'control_adapter' (SimpleAdapter 24->dim, then per-block Conv3d dim->dim).")


def add_track_config_args(parser: argparse.ArgumentParser) -> None:
    """Track module configuration — must match between train + eval."""
    group = parser.add_argument_group("Track modules")
    group.add_argument("--track_config", type=str,
                       default="config/holo4d/default_conv3d_patchify_srcdepth.yaml",
                       help="Track module config YAML. Must match training.")
    group.add_argument("--track_injection_mode", type=str, default="per_block",
                       choices=["single", "per_block"])
    group.add_argument("--track_injection_block_mode", type=str, default="additive",
                       choices=["additive", "concat_project"],
                       help="Per-block injection style. Must match training.")
    group.add_argument("--disable_track_modules", action="store_true", default=False,
                       help="Skip Track4D persistent/dense module initialization entirely.")


def add_dense_branch_args(parser: argparse.ArgumentParser) -> None:
    """Dense GEN3C-style target-space control branch — must match between train + eval."""
    group = parser.add_argument_group("Dense branch")
    group.add_argument("--use_dense_branch", action="store_true", default=True,
                       help="Enable the dense GEN3C-style target-space branch.")
    group.add_argument("--no_dense_branch", dest="use_dense_branch", action="store_false",
                       help="Disable the dense branch.")
    group.add_argument("--dense_in_channels", type=int, default=5,
                       help="Dense encoder input channels. 5 = [rgb|depth|pointcloud_alpha]; "
                            "6 = [rgb|depth/null|target_motion_mask|pointcloud_alpha]; "
                            "7 = [rgb|depth/null|source_motion_mask|target_motion_mask|pointcloud_alpha]; "
                            "8 = [rgb|depth/null|source_alpha_mask|source_motion_mask|"
                            "target_motion_mask|pointcloud_alpha].")
    group.add_argument("--dense_proj_dim", type=int, default=32)
    group.add_argument("--dense_num_residual_blocks", type=int, default=2)
    group.add_argument("--dense_alpha_track", type=float, default=1.0)
    # Tier-B "no depth channel" ablation. With dense_in_channels=5 (so the
    # conv3d_patchify weight loads cleanly from RGBD warm-starts), replace the
    # warped-depth slice with a learnable scalar broadcast in the dense encoder.
    # Lets the encoder learn the placeholder value instead of confounding
    # "no depth signal" with a re-initialized input projection.
    group.add_argument("--use_depth_null_token", action="store_true", default=False,
                       help="Replace dense-encoder depth channel with a learnable "
                            "scalar placeholder (matches track_adapter.null_token "
                            "pattern). Requires --dense_in_channels >= 4. Used by "
                            "the camcond ablation tier B (plucker + warped RGB, "
                            "no depth).")
    # Vista4D-style continuous noise augmentation on dense and track conditioning tokens.
    # 0.0 (default) = no-op, backward-compatible. When >0, samples
    # level ~ U(0, max) per batch at training time and adds Gaussian noise
    # scaled to token std before fusion/injection. Inference path unchanged.
    group.add_argument("--dense_conditioning_noise_max_level", type=float, default=0.0,
                       help="Max noise level for Vista4D-style opt-in noise augmentation "
                            "on dense and track conditioning tokens. 0.0 = off. Typical: 0.1–0.4.")
    group.add_argument("--dense_warp_mode", type=str, default="zbuffer",
                       choices=["zbuffer", "softsplat", "hybrid_softsplat"],
                       help="Backend that builds the dense branch's warped "
                            "[RGB|depth|valid] inputs. "
                            "'zbuffer' (default, fully back-compat): per-frame "
                            "hard nearest-z scatter via "
                            "build_dense_target_reprojection. "
                            "'softsplat': GEN3C-style per-frame bilinear "
                            "soft-splat with depth-weighted blending. "
                            "'hybrid_softsplat': bilinear soft-splat (Lyra-2 / "
                            "GEN3C, depth-weighted) with multi-frame fusion at "
                            "static target pixels and per-frame fallback at "
                            "dynamic pixels. Static / dynamic comes from the "
                            "per-pixel motion range of `tracks_world` "
                            "(Kubric_NVS / Syn4D); datasets without tracks are "
                            "treated as pure static.")
    group.add_argument("--dense_warp_motion_threshold", type=float, default=0.005,
                       help="Per-pixel max-frame motion threshold (cam0-normalised "
                            "scene units; 1.0 ~= scene depth). Pixels whose 3D "
                            "position moves less than this across the video are "
                            "marked static and contribute to the multi-frame "
                            "fill. Default 0.005 = 0.5%% of scene scale (tightened "
                            "from 0.01 on 2026-04-28 after Syn4D 288×512 leakage "
                            "verification — Kubric uses per-frame segm masks so "
                            "is unaffected, but Syn4D's track-projection "
                            "fallback needs the stricter cutoff to catch "
                            "smaller-amplitude actor motion). Only used when "
                            "--dense_warp_mode hybrid_softsplat.")
    group.add_argument("--dense_hybrid_disable_dynamic_masks", action="store_true", default=False,
                       help="For --dense_warp_mode hybrid_softsplat, ignore explicit "
                            "source/target dynamic masks and track-derived dynamic masks. "
                            "The dense branch still uses multi-frame softsplat RGB/depth, "
                            "but treats the whole target as static. Intended for the "
                            "multiframe-RGB no-dynamic-mask ablation.")
    group.add_argument("--use_vista4d_latent_streams", action="store_true", default=False,
                       help="Opt in to Vista4D-style source/point-cloud latent streams. "
                            "Source RGB and rendered point-cloud RGB are VAE-encoded as "
                            "separate token streams; alpha/motion masks are packed into "
                            "zero-init mask patch embeddings. Default off preserves the "
                            "existing dense-control path.")
    group.add_argument("--vista4d_streams_pointcloud_only", action="store_true", default=False,
                       help="With --use_vista4d_latent_streams: drop the redundant "
                            "source-video stream and append ONLY the reprojected "
                            "point-cloud render stream (ab20/ab21 lean variant). Must "
                            "match training; default off keeps the full source+pointcloud "
                            "streams (ab22/ab09v faithful Vista4D).")
    group.add_argument("--use_gen3c_warp_latent_channels", action="store_true", default=False,
                       help="Opt in to GEN3C-style warped RGB injection. The rendered "
                            "point-cloud RGB and valid mask are VAE-encoded and added "
                            "through a zero-init input patch embedding.")


def add_query_impulse_args(parser: argparse.ArgumentParser) -> None:
    """Query-frame impulse conditioning — must match between train + eval."""
    group = parser.add_argument_group("Query-frame impulse")
    group.add_argument("--use_query_frame_impulse_condition", action="store_true", default=True)
    group.add_argument("--no_query_impulse", dest="use_query_frame_impulse_condition",
                       action="store_false")


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Sampling / inference defaults for Track4D-360 V2V NVS."""
    group = parser.add_argument_group("Sampling")
    group.add_argument("--num_inference_steps", type=int, default=50)
    group.add_argument("--cfg_scale", type=float, default=1.0,
                       help="Text-CFG scale. cfg=1 skips the negative DiT pass (cheaper). "
                            "For V2V NVS, text-CFG is a near-null knob.")
    group.add_argument("--sigma_shift", type=float, default=5.0,
                       help="Noise schedule shift. 5.0 matches training default.")
    group.add_argument("--seed", type=int, default=42)


def add_gs_training_timestep_args(parser: argparse.ArgumentParser) -> None:
    """Training-only GS-head knobs that override FlowMatchV2VLoss defaults.

    Pin the FM timestep to a specific fraction of a reference denoising
    trajectory (matches online GS-trainer spec: step_idx of N_inference)
    and/or skip the FM loss (frozen DiT → diffusion gradient is a no-op;
    forward still runs for gs_features). Imported by
    scripts/wan2.1_fun/train_track4d360.py.

    Why step-fraction, not σ-directly, and not scheduler-index-fraction:
    - Scheduler index fraction is the WRONG knob — under --train_sigma_shift
      it is non-linear: index-frac 0.25 @ shift=5 resolves to σ≈0.94.
    - σ-directly is confusing because users think in "step k of N_inference
      denoising steps" (matches --gs_teacher_step_idx in the online spec).
    - Step-fraction is the natural knob: fraction 0.25 on a 50-step ref
      trajectory means step_idx=12, which is where the online trainer pins.
    The trainer resolves step_frac → σ via the scheduler's own
    set_timesteps_fn(N_inference, shift), then picks the nearest index in
    the 1000-step training schedule via argmin. Resolution is deferred to
    after switch_pipe_to_training_mode() so the scheduler is initialised.
    """
    group = parser.add_argument_group("GS training timestep (train-only)")
    group.add_argument("--fixed_step_frac", type=float, default=None,
                       help="If set (e.g., 0.25), pin the FM timestep to this "
                            "fraction of a reference denoising trajectory. "
                            "Matches the online GS-trainer spec: step_frac=0.25 "
                            "on a 50-step trajectory → step_idx=12. The sigma "
                            "at that step (under --train_sigma_shift) is "
                            "computed via the scheduler's set_timesteps_fn, "
                            "then used deterministically for every training "
                            "step. Example (shift=5, 50-step ref): "
                            "step_frac=0.25 → sigma≈0.9406.")
    group.add_argument("--fixed_num_inference_steps", type=int, default=50,
                       help="Reference denoising-trajectory length for "
                            "--fixed_step_frac. Default 50 matches the online "
                            "GS-trainer spec's --gs_teacher_num_inference_steps.")
    group.add_argument("--fixed_sigma", type=float, default=None,
                       help="Direct sigma target (0 = clean, 1 = pure noise). "
                            "Training-timestep = sigma * 1000. If set, takes "
                            "precedence over --fixed_step_frac. Example: "
                            "--fixed_sigma 0.25 → t=250, 25%% noise. Contrast "
                            "with --fixed_step_frac 0.25 on a 50-step ref @ "
                            "shift=5 which → σ=0.94, t=940.")
    group.add_argument("--skip_diffusion_loss", action="store_true", default=False,
                       help="Zero out the flow-matching MSE term. Use with "
                            "--gs_only_training (frozen DiT → FM gradient is a "
                            "no-op; still computes forward for gs_features).")


def add_ar_args(parser: argparse.ArgumentParser) -> None:
    """Rolling-forcing AR train/eval knobs shared by Track4D entrypoints."""

    group = parser.add_argument_group("Track4D-360 rolling-forcing AR")
    group.add_argument("--ar_training", action="store_true", default=False,
                       help="Use the opt-in AR V2V loss (see --ar_training_algorithm).")
    group.add_argument("--ar_training_algorithm", type=str, default="windowed_teacher_forcing",
                       choices=[
                           "windowed_teacher_forcing",
                           "rolling_forcing",
                           "diffusion_forcing",
                           "causal_teacher_forcing",
                       ],
                       help="AR training objective. 'windowed_teacher_forcing' (default, "
                            "backward-compatible): noise GT latent windows with a causal KV "
                            "cache — NOT the self-forcing RollingForcing objective. "
                            "'rolling_forcing': faithful self-rollout (model consumes its own "
                            "denoised context via a clean detached KV cache; gradient only "
                            "through a random exit block). 'diffusion_forcing': non-DMD "
                            "per-latent-frame independent noise levels in one full-sequence "
                            "Wan flow-matching forward. 'causal_teacher_forcing': raw/no-control "
                            "Causal-Forcing stage-1 style clean-context teacher forcing. See "
                            "doc/track4d-360/2026-06-15-rolling-forcing-training-rewrite.md.")
    group.add_argument("--ar_context_noise_step", type=int, default=0,
                       help="rolling_forcing: scheduler step id used to (near-)clean re-run the "
                            "denoised block into the KV cache (reference context_noise; 0 = clean).")
    group.add_argument("--ar_rolling_window_blocks", type=int, default=0,
                       help="rolling_forcing: blocks per sliding window (0 = len(denoising_step_list)).")
    group.add_argument("--ar_num_frame_per_block", type=int, default=3,
                       help="Number of latent frames per AR block.")
    group.add_argument("--ar_denoising_step_list", type=str, default="1000,800,600,400,200",
                       help="Comma-separated scheduler timesteps used by AR windows.")
    group.add_argument("--ar_same_step_across_blocks", action="store_true", default=False,
                       help="Use the first AR timestep for every latent block.")
    group.add_argument("--ar_grad_through_context_cache", action="store_true", default=False,
                       help="Opt out of detaching the cached past windows during AR training. "
                            "Default (off) = teacher forcing: past context is fixed/detached so "
                            "each window only trains 'predict this window | given past'. Set this "
                            "to keep the (memory-heavy, window-coupled) grad-through-cache behavior.")
    group.add_argument("--ar_context_noise", type=float, default=0.0,
                       help="Reserved context-noise knob for future AR ablations; 0 disables it.")
    group.add_argument("--ar_cache_max_latent_frames", type=int, default=21,
                       help="Maximum latent frames retained in the rolling KV cache.")
    group.add_argument("--ar_cache_keep_first_latent", action="store_true", default=False,
                       help="Keep latent frame 0 in the rolling KV cache as an attention sink "
                            "in addition to the local cache window. Default off.")
    group.add_argument("--ar_loss_last_latent_frames", type=int, default=21,
                       help="Compute AR diffusion loss only on the last N latent frames.")


def add_future_prediction_args(parser: argparse.ArgumentParser) -> None:
    """Future-prediction (temporal extrapolation) knobs shared by train/eval.

    All default OFF: with --future_prediction unset, the train/eval forward,
    dataset fields, and loss behaviour are byte-identical to the V2V NVS path.
    Future prediction marks the trailing N latent frames as UNOBSERVED — their
    source domain is replaced by a learned null and their track visibility is
    zeroed (camera/plucker stays), while the target is supervised on all frames.
    """

    group = parser.add_argument_group("Future prediction (temporal extrapolation)")
    group.add_argument("--future_prediction", action="store_true", default=False,
                       help="Enable future prediction: mark the last "
                            "--future_num_latent_frames latent frames as unobserved "
                            "and predict them from observed context + camera.")
    group.add_argument("--future_num_latent_frames", type=int, default=0,
                       help="Number of trailing LATENT frames treated as future/"
                            "unobserved. Must be >0 and < latent_t when "
                            "--future_prediction is set. The observed/future split "
                            "lands on a latent-frame boundary by construction.")
    group.add_argument("--future_loss_weight", type=float, default=1.0,
                       help="Multiplier applied to the diffusion loss on future "
                            "(predicted) latent frames; observed frames keep weight 1.")
    group.add_argument("--future_source_null", action="store_true", default=False,
                       help="Use the learned per-channel `future_source_null` latent "
                            "for the source domain of future frames (vs plain zeros).")


def add_cond_dropout_args(parser: argparse.ArgumentParser) -> None:
    """Conditioning dropout — train the model to work without track/dense signals."""
    group = parser.add_argument_group("Conditioning dropout")
    group.add_argument("--cond_drop_track_prob", type=float, default=0.0,
                       help="Per-batch prob of dropping track conditioning "
                            "(sets tracks_world=None, masks track loss). "
                            "Simulates depth-only dynamic tier.")
    group.add_argument("--cond_drop_dense_prob", type=float, default=0.0,
                       help="Per-batch prob of dropping dense target-space branch conditioning "
                            "(removes dense-branch RGB/camera inputs, masks depth loss).")
    group.add_argument("--cond_drop_geo_prob", type=float, default=0.0,
                       help="Per-batch prob of dropping BOTH track AND dense jointly. "
                            "Simulates camera-only tier. Recommended: 0.1 for mixed training.")


def add_cond_null_token_fallback_args(parser: argparse.ArgumentParser) -> None:
    """Learned fallback tokens for missing/dropped geometric conditions."""
    group = parser.add_argument_group("Conditioning null-token fallback")
    group.add_argument("--use_track_null_token_fallback", action="store_true", default=False,
                       help="When track conditioning is missing or dropped, inject a learned "
                            "track placeholder token sequence instead of a pure no-op path.")
    group.add_argument("--use_dense_null_token_fallback", action="store_true", default=False,
                       help="When dense target-space inputs are missing or dropped, inject a "
                            "learned dense placeholder token sequence instead of a pure no-op path.")


def add_gs_teacher_args(parser: argparse.ArgumentParser) -> None:
    """Online GS-head trainer: frozen teacher extraction configuration."""
    group = parser.add_argument_group("GS teacher extraction (online trainer)")
    group.add_argument("--gs_teacher_num_inference_steps", type=int, default=50,
                       help="Number of inference steps for the teacher's noise schedule. "
                            "Used to derive sigma_S from the scheduler. Default 50.")
    group.add_argument("--gs_teacher_step_idx", type=int, default=None,
                       help="Fixed denoising step index (0-indexed) at which to extract "
                            "the teacher feature. Default: auto from --gs_teacher_sigma_frac.")
    group.add_argument("--gs_teacher_sigma_frac", type=float, default=0.25,
                       help="Fraction of the denoising trajectory for the teacher step. "
                            "step_idx = round(num_inference_steps * sigma_frac). "
                            "Ignored when --gs_teacher_step_idx is set explicitly. Default 0.25.")
    group.add_argument("--gs_teacher_block_idx", type=int, default=19,
                       help="DiT block index to tap for h_mid (0-indexed). "
                            "Wan 1.3B has 30 blocks; 19 ≈ 62.5%% depth. Default 19.")


def add_source_backend_args(parser: argparse.ArgumentParser) -> None:
    """Source geometry/tracks backend for DAVIS-source NVS eval drivers.

    Shared by eval_track4d360_vista4d_official.py and
    eval_track4d360_davis_4rc_novel_traj.py so the cow_vggt flag vector never
    drifts between the two (the --use_gs_output parser-drift lesson). Both
    drivers build an identical ``source_run`` from either cached 4RC predictions
    or cotracker + VGGT-Omega caches.
    """
    group = parser.add_argument_group("Source geometry/tracks backend")
    group.add_argument(
        "--source_backend", type=str, default="4rc", choices=["4rc", "cow_vggt"],
        help="Source geometry/tracks: 4rc (default) or cotracker+vggt-omega.",
    )
    group.add_argument(
        "--cow_root", type=str,
        default=os.environ.get("COW_ROOT"),
        help="CoWTracker source tracks root (<root>/<seq>_pred_trackvideo.npz); set COW_ROOT or pass --cow_root for cow_vggt.",
    )
    group.add_argument(
        "--vggt_root", type=str,
        default=os.environ.get("VGGT_ROOT"),
        help="VGGT-Omega cameras/depth root (<root>/<seq>/{pose,geometry}/*.npz); set VGGT_ROOT or pass --vggt_root.",
    )
    group.add_argument("--cow_track_sample_mode", type=str, default="all",
                       choices=["all", "grid", "jittered", "random"])
    group.add_argument("--cow_track_stride", type=int, default=1)
    group.add_argument("--cow_conf_threshold", type=float, default=0.5)
    group.add_argument("--cow_depth_conf_threshold", type=float, default=0.0)
