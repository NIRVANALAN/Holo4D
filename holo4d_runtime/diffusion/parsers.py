import argparse


def add_xyz_video_condition_config(parser: argparse.ArgumentParser):
    """
    Optional RGB+XYZ conditioning (adds a normalized XYZ "video" to RGB pixel-wise before VAE encoding).
    Disabled by default.
    """
    parser.add_argument(
        "--use_xyz_video_condition",
        default=False,
        action="store_true",
        help="Enable RGB+XYZ conditioning: pixel-wise add normalized XYZ (from depth/camera) to the RGB input video before VAE encoding.",
    )
    parser.add_argument(
        "--xyz_video_condition_scale",
        type=float,
        default=1.0,
        help="Scale for the added XYZ conditioning term (RGB := clamp(RGB + scale * XYZ_norm, -1, 1)).",
    )
    parser.add_argument(
        "--xyz_video_condition_aug_prob",
        type=float,
        default=0.0,
        help=(
            "(Training) Probability to apply xyz_video_cond noise augmentation to a sample (0 disables). "
            "Recommended: 0.5 to augment half of training samples. "
            "This bridges train/inference domain gap (GT xyz vs Stream3R/DA3 predicted xyz)."
        ),
    )
    parser.add_argument(
        "--xyz_video_condition_noise_std",
        type=float,
        default=0.0,
        help=(
            "(Training) Additive Gaussian noise std applied to xyz_video_cond after /common_scale (in normalized XYZ units). "
            "Recommended: 0.15 (based on Stream3R median error ~0.10, mean ~0.15). "
            "Simulates monocular 3D prediction errors from Stream3R/DA3 at inference time."
        ),
    )
    parser.add_argument(
        "--xyz_video_condition_z_noise_std",
        type=float,
        default=0.0,
        help=(
            "(Training) Extra Gaussian noise std applied only to the Z channel (depth axis) of xyz_video_cond. "
            "Recommended: 0.0-0.1 (depth is often less accurate than XY in monocular models)."
        ),
    )
    parser.add_argument(
        "--xyz_video_condition_frame_scale_std",
        type=float,
        default=0.0,
        help=(
            "(Training) Per-frame multiplicative scale jitter (log-normal std) to simulate depth scale flicker. "
            "Recommended: 0.05-0.1 (simulates per-frame scale drift in relative depth models like DA3Mono)."
        ),
    )
    parser.add_argument(
        "--xyz_video_condition_dropout_prob",
        type=float,
        default=0.0,
        help=(
            "(Training) Randomly drop valid xyz pixels with this probability (mask them invalid and zero values). "
            "Recommended: 0.05-0.1 (simulates confidence-based masking and missing data at inference time)."
        ),
    )
    parser.add_argument(
        "--xyz_video_condition_mode",
        type=str,
        default="xyz",
        choices=["xyz", "depth", "none"],
        help=(
            "Conditioning mode for xyz_video_cond: "
            "'xyz' (default) uses full 3D pointmap (X,Y,Z); "
            "'depth' uses only Z-channel with log normalization (better dynamic range); "
            "'none' disables geometric conditioning (adapter still computes xyz for residual target)."
        ),
    )
    parser.add_argument(
        "--xyz_video_condition_cfg_dropout_prob",
        type=float,
        default=0.0,
        help=(
            "(Training) CFG-style full-signal dropout: probability to zero out the ENTIRE xyz/depth "
            "conditioning signal per sample. Enables classifier-free guidance at inference time. "
            "Recommended: 0.10-0.15."
        ),
    )
    parser.add_argument(
        "--camera_condition_cfg_dropout_prob",
        type=float,
        default=0.0,
        help=(
            "(Training) CFG-style full-signal dropout: probability to zero out the ENTIRE camera pose "
            "conditioning signal per sample. Enables classifier-free guidance at inference time. "
            "Recommended: 0.10-0.15."
        ),
    )
    return parser


def add_query_frame_impulse_condition_config(parser: argparse.ArgumentParser):
    """
    Optional query-frame impulse conditioning for VDM 3D tracking.

    When enabled, the pipeline encodes a binary impulse video (1 at the query frame, else 0) with the Wan VAE
    and concatenates it to the existing xyz control latents along channels. Disabled by default.
    """
    parser.add_argument(
        "--use_query_frame_impulse_condition",
        default=False,
        action="store_true",
        help="Enable query-frame impulse conditioning (binary impulse video encoded by VAE and concatenated to control latents).",
    )
    return parser


def add_inference_config(parser: argparse.ArgumentParser):
    """
    Add common inference CLI args shared by WanVideo inference scripts.

    DEPRECATED: Use add_essential_inference_config for new scripts.
    This function is kept for backward compatibility only.
    """
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion denoising steps.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu).",
    )
    return parser


def add_essential_inference_config(parser: argparse.ArgumentParser):
    """
    Add only essential user-facing inference arguments.

    Technical diffusion parameters (cfg_scale, num_inference_steps, sigma_shift,
    tiled, tile_size, tile_stride) should be hardcoded in the script as reasonable
    defaults, following the official inference example pattern.
    """
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu).",
    )
    return parser


def add_video_output_config(parser: argparse.ArgumentParser):
    """
    Add video output quality args shared by inference scripts.
    """
    parser.add_argument(
        "--video_quality",
        type=int,
        default=6,
        help="Video quality (1-10): 1=smallest file, 10=largest.",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=16,
        help="Video FPS (frames per second).",
    )
    return parser


def add_camera_control_config(
    parser: argparse.ArgumentParser,
    *,
    default_camera_injection_mode: str = "projector",
    default_plucker_encoder_mode: str = "patchify",
    allow_auto_plucker_encoder_mode: bool = False,
    include_camera_condition_scale: bool = False,
    include_debug_camera_control: bool = False,
    include_use_perlayer_camera_projector: bool = False,
):
    """
    Add common camera-control CLI args shared by training/inference.

    Args:
        parser: ArgumentParser instance to add arguments to
        default_camera_injection_mode: Default camera injection mode
        default_plucker_encoder_mode: Default Plücker encoder mode ("patchify" or "control_adapter")
        allow_auto_plucker_encoder_mode: Allow "auto" mode for inference (infer from checkpoint)
        include_camera_condition_scale: Include --camera_condition_scale arg (for inference)
        include_debug_camera_control: Include --debug_camera_control arg (for verbose debugging)
        include_use_perlayer_camera_projector: Include --use_perlayer_camera_projector arg

    Notes:
    - `camera_plucker_encoder_mode` determines how Plücker rays are encoded:
      * "patchify": PixelUnshuffle+Conv2d (24D → 6D) + per-layer Conv3d (6D → dim)
      * "control_adapter": SimpleAdapter (24D → dim) + per-layer Conv3d (dim → dim)
    - For `direct_raymap_add` mode, plucker_encoder_mode chooses between patchify or control_adapter
    - `camera_condition_scale` is an inference-facing knob; keep it optional here.
    """
    parser.add_argument(
        "--use_perlayer_camera_control",
        default=False,
        action="store_true",
        help="Use per-layer camera control: adds camera features to each DiT block.",
    )
    parser.add_argument(
        "--camera_injection_mode",
        type=str,
        default=default_camera_injection_mode,
        choices=["projector", "add", "direct_raymap_add", "rt_projector", "kq_only", "vista4d_plucker"],
        help="Camera injection mode for per-layer control: "
        "'projector' (ReCamMaster-style: camera features added to normalized input, attention output projected), "
        "'add' (video4spatial_ttt-style: camera features directly added to x before attention), "
        "'direct_raymap_add' (paper-style: per-layer encoding of raw 6D Plücker raymaps, no global SimpleAdapter), "
        "'rt_projector' (ReCamMaster-style: per-layer projector + per-layer Linear RT encoder; no Plücker rays), "
        "'kq_only' (Lyra-2-style: per-layer cam_encoder, camera bias affects only Q/K projections, not V), or "
        "'vista4d_plucker' (Vista4D-style raw 6D Plücker tokens, per-block zero-init Linear, identity projector).",
    )
    parser.add_argument(
        "--recammaster_compat",
        default=False,
        action="store_true",
        help="Enable ReCamMaster compatibility mode (e.g., disable domain tokens and suppress domain_embedding warnings).",
    )
    parser.add_argument(
        "--camera_pose_mode",
        type=str,
        default="packed",
        choices=["packed", "raw"],
        help="RT pose feature mode for rt_projector: "
        "'packed' (default, 12*4=48 features per latent frame), "
        "or 'raw' (12D RT per latent frame, ReCamMaster official style).",
    )
    parser.add_argument(
        "--camera_pose_dim",
        type=int,
        default=None,
        help="Override RT pose feature dimension for rt_projector. "
        "Defaults to 48 for packed mode or 12 for raw mode when unset.",
    )
    parser.add_argument(
        "--camera_pose_frame_stride",
        type=int,
        default=4,
        help="Frame stride when camera_pose_mode=raw (default: 4, matches Wan VAE temporal compression).",
    )
    if include_camera_condition_scale:
        parser.add_argument(
            "--camera_condition_scale",
            type=float,
            default=1.0,
            help="Scale factor for camera conditioning features (1.0 = default).",
        )
    parser.add_argument(
        "--camera_targets_only",
        default=False,
        action="store_true",
        help="Apply camera conditioning only on target domains (keeps source domain camera-free; can help color consistency).",
    )

    plucker_choices = ["control_adapter", "patchify"]
    if allow_auto_plucker_encoder_mode:
        plucker_choices = ["auto"] + plucker_choices
    if default_plucker_encoder_mode not in plucker_choices:
        raise ValueError(
            f"Invalid default_plucker_encoder_mode={default_plucker_encoder_mode!r}; "
            f"expected one of {plucker_choices}."
        )
    parser.add_argument(
        "--camera_plucker_encoder_mode",
        type=str,
        default=default_plucker_encoder_mode,
        choices=plucker_choices,
        help="Plücker encoder for projector/add modes: "
        + ("'auto' (infer from checkpoint), " if allow_auto_plucker_encoder_mode else "")
        + "'control_adapter' (SimpleAdapter), or "
        "'patchify' (PixelUnshuffle+Conv2d 24->6, then Conv3d 6->dim).",
    )
    if include_debug_camera_control:
        parser.add_argument(
            "--debug_camera_control",
            default=False,
            action="store_true",
            help="Enable verbose debugging for camera control (prints shapes and values).",
        )
    parser.add_argument(
        "--debug_onehot_camera",
        default=False,
        action="store_true",
        help="DEBUG MODE: Use one-hot embeddings based on trajectory pair index instead of processing Plücker rays. "
        "Helps diagnose if camera control issue is with encoding or injection mechanism.",
    )
    if include_use_perlayer_camera_projector:
        parser.add_argument(
            "--use_perlayer_camera_projector",
            default=True,
            action="store_true",
            help="For direct_raymap_add mode: add ReCamMaster-style identity-initialized projector on attention output (DEFAULT=True, REQUIRED). "
            "Enables: x = x + gate_msa * projector(self_attn(input_x + cam_emb))",
        )
    parser.add_argument(
        "--use_pre_dit_camera_injection",
        default=False,
        action="store_true",
        help="Use camera-control runtime style camera injection: add camera features to x after patch embedding "
        "but BEFORE DiT blocks. This is ADDITIVE with per-layer injection (both can be enabled). "
        "Requires --use_perlayer_camera_control. Uses control_adapter + pre_dit_cam_encoder. "
        "When enabled, camera features are injected TWICE: once before DiT blocks (pre-DiT) "
        "and again in each DiT block (per-layer). "
        "NOTE: pre_dit_cam_encoder is always initialized for checkpoint compatibility, but only "
        "trained if included in --trainable_models.",
    )
    parser.add_argument(
        "--use_noise_camera_injection",
        default=False,
        action="store_true",
        help="Inject camera features into the token feature map immediately after patch embedding "
        "(i.e., into the 'noisy' features) before domain tokens / DiT blocks. "
        "Requires `dit.control_adapter` and initializes `dit.noise_cam_encoder` (Conv3d(dim→dim), zero-init). "
        "Can be combined with per-layer and/or pre-DiT camera injections.",
    )
    parser.add_argument(
        "--pre_dit_noise_injection_std",
        type=float,
        default=0.0,
        help="Additive Gaussian noise (std) injected into the token feature map right before DiT blocks "
        "(after patchify/conditioning, before flattening). Intended as extra robustness regularization; "
        "0 disables.",
    )
    return parser


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path",
                        type=str,
                        default=None,
                        required=False,
                        help="Base path of the dataset (required unless --use_stream3r_data is enabled).")
    parser.add_argument("--dataset_metadata_path",
                        type=str,
                        default=None,
                        help="Path to the metadata file of the dataset.")
    parser.add_argument(
        "--dataset_repeat",
        type=int,
        default=1,
        help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers",
                        type=int,
                        default=4,
                        help="Number of workers for data loading.")
    parser.add_argument("--dataloader_prefetch_factor",
                        type=int,
                        default=2,
                        help="Number of batches to prefetch per worker. Higher values reduce GPU idle time but use more memory.")
    parser.add_argument(
        "--data_file_keys",
        type=str,
        default="image,video",
        help="Data file keys in the metadata. Comma-separated.")
    return parser


def add_pointmap_stats_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--pointmap_stats_path",
        type=str,
        default=None,
        help="Path to JSON file with global pointmap statistics (used when predicting pointmaps or tracks).",
    )
    return parser


def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=
        "Height of images. Leave `height` and `width` empty to enable dynamic resolution."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=
        "Width of images. Leave `height` and `width` empty to enable dynamic resolution."
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help="Maximum number of pixels per frame, used for dynamic resolution."
    )
    return parser


def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=
        "Height of images. Leave `height` and `width` empty to enable dynamic resolution."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=
        "Width of images. Leave `height` and `width` empty to enable dynamic resolution."
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help="Maximum number of pixels per frame, used for dynamic resolution."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help=
        "Number of frames per video. Frames are sampled from the video prefix."
    )
    return parser


def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths",
                        type=str,
                        default=None,
                        help="Paths to load models. In JSON format.")
    parser.add_argument(
        "--model_id_with_origin_paths",
        type=str,
        default=None,
        help=
        "Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated."
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="it2v",
        choices=["it2v", "vt2v"],
        help=
        "Model variant: 'it2v' (default Wan, for T2V/I2V) or 'vt2v' (Video-to-Video with domain tokens and V2V-specific RoPE)."
    )
    parser.add_argument("--extra_inputs",
                        default=None,
                        help="Additional model inputs, comma-separated. "
                             "Examples: 'disp_video' (for depth prediction), 'pointmap' (for 3D point prediction). "
                             "This determines the prediction target and affects data preprocessing.")
    parser.add_argument(
        "--track_xyz_normalization",
        type=str,
        default="minmax",
        choices=["minmax", "signed_log_p2_p98", "signed_log_p0_p100"],
        help=(
            "Normalization applied to `track` XYZ (after cam1 alignment / common_scale) before VAE encoding. "
            "'minmax': per-sequence per-channel min/max -> [-1, 1] (valid pixels only). "
            "'signed_log_p2_p98': PPD-style signed-log1p -> quantile-range normalize (2%%-98%%) -> clamp to [-1.0, 0.5]. "
            "'signed_log_p0_p100': same as signed_log_p2_p98 but uses 0%%-100%% (min/max) instead of 2%%-98%% (no truncation)."
        ),
    )
    parser.add_argument(
        "--track_xyz_quantile_low",
        type=float,
        default=0.02,
        help="Lower quantile for signed_log_p2_p98 normalization (default: 0.02 = 2%%). Set to 0.0 to use actual min (no clipping)."
    )
    parser.add_argument(
        "--track_xyz_quantile_high",
        type=float,
        default=0.98,
        help="Upper quantile for signed_log_p2_p98 normalization (default: 0.98 = 98%%). Set to 1.0 to use actual max (no clipping)."
    )
    parser.add_argument(
        "--predict_track_residual",
        default=False,
        action="store_true",
        help="Predict track residual (track - xyz_video_cond) instead of absolute track. "
        "Requires --use_xyz_video_condition. Background pixels (no motion) will have near-zero residuals, "
        "making the task easier and more focused on moving objects. At inference, predicted residual is added back to xyz_video_cond."
    )
    parser.add_argument(
        "--residual_target_xyz_source",
        type=str,
        default="clean",
        choices=["augmented", "clean"],
        help=(
            "(Training, Stream3R adapter) When --predict_track_residual is enabled, "
            "choose whether the residual target is computed against the augmented xyz_video_cond "
            "(after xyz noise/dropout) or the clean xyz_video_cond (before augmentation). "
            "Default: clean. Inference behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--residual_zero_center_normalization",
        default=False,
        action="store_true",
        help=(
            "When --predict_track_residual is enabled, normalize residuals such that residual=0 "
            "(static background) always maps to 0 in normalized space. "
            "This ensures consistent background color across all sequences, making it easier for the model "
            "to learn that static regions should be zero. If disabled (default), residual=0 maps to different "
            "values per sequence depending on the min/max/quantile range. "
            "IMPORTANT: if you compare GT vs pred in normalized space, normalize GT using the SAME stats source as inference."
        ),
    )
    parser.add_argument(
        "--residual_linear_normalization",
        default=False,
        action="store_true",
        help=(
            "When --predict_track_residual is enabled, use fixed zero-preserving linear normalization "
            "(clamp(residual * K, [-1,1])) instead of signed-log residual normalization. "
            "Reuses --residual_log_scale_factor as the linear K for backward compatibility."
        ),
    )
    parser.add_argument(
        "--residual_log_scale_factor",
        type=float,
        default=3.0,
        help=(
            "Extra linear scale applied to residuals AFTER signed-log normalization in zero-center mode. "
            "Effective only when --predict_track_residual and --residual_zero_center_normalization are enabled. "
            "Larger values make residuals occupy more of [-1,1] without changing xyz quantile stats."
        ),
    )
    parser.add_argument(
        "--stream3r_common_scale_stat",
        type=str,
        default="mean",
        choices=["mean", "median", "q02_q98_mean", "q02_q98_median", "query_p60", "dynamic_mean", "dynamic_q60"],
        help=(
            "Statistic used to compute Stream3R common_scale from valid XYZ norms before any downstream "
            "normalization. 'mean' matches legacy behavior; 'median' is a more robust alternative. "
            "'q02_q98_*' trims the outer 2%% tails of valid point norms before reducing. "
            "'query_p60' uses p60 of query-frame-only norms (best cross-dataset alignment). "
            "'dynamic_mean' uses mean query-frame norm over GT dynamic points only (oracle analysis mode). "
            "'dynamic_q60' uses p60 query-frame norm over GT dynamic points only (oracle analysis mode)."
        ),
    )
    parser.add_argument(
        "--explicit_motion_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for explicit decoded 3D motion L2 supervision on x0_hat (target motion prediction). "
            "0 disables this term. When enabled, prediction is denormalized (including residual K-scale) "
            "and compared to GT in metric 3D space using common_scale."
        ),
    )
    explicit_snr_group = parser.add_mutually_exclusive_group()
    explicit_snr_group.add_argument(
        "--explicit_motion_loss_snr_weight",
        dest="explicit_motion_loss_snr_weight",
        action="store_true",
        help=(
            "Apply SNR weighting to explicit motion loss (default). "
            "Weight function is controlled by --explicit_motion_loss_snr_sigmoid."
        ),
    )
    explicit_snr_group.add_argument(
        "--no_explicit_motion_loss_snr_weight",
        dest="explicit_motion_loss_snr_weight",
        action="store_false",
        help="Disable SNR weighting for explicit motion loss.",
    )
    parser.set_defaults(explicit_motion_loss_snr_weight=True)
    explicit_snr_mode_group = parser.add_mutually_exclusive_group()
    explicit_snr_mode_group.add_argument(
        "--explicit_motion_loss_snr_sigmoid",
        dest="explicit_motion_loss_snr_sigmoid",
        action="store_true",
        help=(
            "Use sigmoid-compressed log-SNR weighting for explicit motion loss (default). "
            "weight = sigmoid(a * (log_snr - b))."
        ),
    )
    explicit_snr_mode_group.add_argument(
        "--no_explicit_motion_loss_snr_sigmoid",
        dest="explicit_motion_loss_snr_sigmoid",
        action="store_false",
        help="Use legacy explicit-motion SNR weighting snr/(1+snr).",
    )
    parser.set_defaults(explicit_motion_loss_snr_sigmoid=True)
    parser.add_argument(
        "--explicit_motion_loss_snr_sigmoid_a",
        type=float,
        default=1.0,
        help="Steepness 'a' for explicit motion sigmoid-logSNR weighting.",
    )
    parser.add_argument(
        "--explicit_motion_loss_snr_sigmoid_b",
        type=float,
        default=-5.0,
        help="Center 'b' for explicit motion sigmoid-logSNR weighting (default -5.0).",
    )
    explicit_decode_ckpt_group = parser.add_mutually_exclusive_group()
    explicit_decode_ckpt_group.add_argument(
        "--explicit_motion_loss_decode_checkpoint",
        dest="explicit_motion_loss_decode_checkpoint",
        action="store_true",
        help=(
            "Use activation checkpointing in VAE decode for explicit motion loss. "
            "Reduces VRAM at the cost of extra compute."
        ),
    )
    explicit_decode_ckpt_group.add_argument(
        "--no_explicit_motion_loss_decode_checkpoint",
        dest="explicit_motion_loss_decode_checkpoint",
        action="store_false",
        help="Disable activation checkpointing in explicit motion loss VAE decode.",
    )
    parser.set_defaults(explicit_motion_loss_decode_checkpoint=False)
    parser.add_argument(
        "--explicit_motion_loss_decode_chunk_size",
        type=int,
        default=0,
        help=(
            "Micro-batch size for explicit motion loss VAE decode. "
            "Set >0 to enable chunked decode; 0 keeps default full decode path."
        ),
    )
    parser.add_argument(
        "--normalize_track_with_xyz_cond_stats",
        default=False,
        action="store_true",
        help=(
            "Compute normalization stats from xyz_video_cond and apply the same stats to both xyz_video_cond "
            "and track target. This ensures both tensors are in the same normalized space and enables "
            "inference without GT track (since stats come from xyz_video_cond which is always available). "
            "IMPORTANT: Use this flag for models that need to work with Stream3R/DA3 xyz at inference time. "
            "Without this flag, xyz_video_cond is normalized using GT track stats (legacy behavior), which "
            "causes scale mismatch when GT track is not available at inference."
        ),
    )
    parser.add_argument(
        "--prenormalized_track_video_signed_log",
        default=False,
        action="store_true",
        help=(
            "When a dataset provides `track_video_prenormalized` in [-1,1], apply zero-preserving "
            "signed-log1p compression to track_video before VAE encoding (no extra K scaling)."
        ),
    )
    parser = add_xyz_video_condition_config(parser)
    parser.add_argument("--fp8_models",
                        default=None,
                        help="Models with FP8 precision, comma-separated.")
    parser.add_argument(
        "--offload_models",
        default=None,
        help=
        "Models with offload, comma-separated. Only used in splited training.")
    parser.add_argument(
        "--v2v_concat_mode",
        type=str,
        default="channel",
        choices=["frame", "width", "channel"],
        help=
        "V2V concatenation mode: 'channel' (concatenate RGB condition with target along channels; cheapest), "
        "'frame' (along temporal dim), or 'width' (along spatial dim for shorter interaction distance)."
    )
    parser.add_argument(
        "--v2v_channel_patch_embed_mode",
        type=str,
        default="add",
        choices=["add", "cat"],
        help=(
            "Only used when `--v2v_concat_mode channel`. "
            "'add' uses the original patch embedding on each half and sums the embeddings (no new weights). "
            "'cat' uses a dedicated V2V patch embedding with 2× input channels (new weights)."
        ),
    )
    return parser


def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=1,
                        help="Number of epochs.")
    parser.add_argument("--trainable_models",
                        type=str,
                        default=None,
                        help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--find_unused_parameters",
                        default=False,
                        action="store_true",
                        help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01,
                        help="Weight decay.")
    parser.add_argument("--task",
                        type=str,
                        default="sft",
                        required=False,
                        help="Task type.")
    parser.add_argument("--drop_text_prob",
                        type=float,
                        default=0.0,
                        help="Probability of dropping text prompts during training for classifier-free guidance (0.0=no dropout, 0.5=50%% dropout). "
                             "Following ControlNet training, empty prompts enable CFG at inference time.")
    parser.add_argument("--drop_video_prob",
                        type=float,
                        default=0.0,
                        help="Probability of T2V/I2V mode switching during training. "
                             "Following ReCamMaster paper Section 4.2, this enables camera-controlled T2V and I2V generation. "
                             "Split equally: drop_video_prob*0.5 → T2V (text+camera, no video), drop_video_prob*0.5 → I2V (image+text+camera). "
                             "Default: 0.0 (no dropout, V2V only). ReCamMaster official: 0.4 (20%% T2V + 20%% I2V + 60%% V2V).")
    parser.add_argument("--drop_t2v_prob",
                        type=float,
                        default=None,
                        help="Explicit probability for T2V mode (text+camera only). "
                             "If set (or if --drop_i2v_prob is set), overrides the legacy --drop_video_prob equal-split behavior. "
                             "Use 0.0 to disable T2V while keeping I2V/V2V.")
    parser.add_argument("--drop_i2v_prob",
                        type=float,
                        default=None,
                        help="Explicit probability for I2V mode (first frame + text + camera). "
                             "If set (or if --drop_t2v_prob is set), overrides the legacy --drop_video_prob equal-split behavior. "
                             "Example: I2V-only 20%% -> --drop_t2v_prob 0 --drop_i2v_prob 0.2.")
    parser.add_argument("--input_video_noise_strength",
                        type=float,
                        default=0.0,
                        help="Noise strength for V2V conditional video latent augmentation (timestep in noise schedule). "
                             "Following ReCamMaster paper, adds robustness on synthetic data by injecting noise to conditional video latent. "
                             "Default: 0.0 (no noise). "
                             "Use negative value (e.g., -1) for random timestep in [0, --input_video_noise_random_max].")
    parser.add_argument("--input_video_noise_random_max",
                        type=int,
                        default=500,
                        help="Upper bound for random latent-noise timestep when --input_video_noise_strength < 0. "
                             "Default: 500 (legacy behavior).")
    return parser


def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path",
                        type=str,
                        default="./models",
                        help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt",
                        type=str,
                        default="pipe.dit.",
                        help="Remove prefix in ckpt.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2500,
        help=
        "Number of checkpoint saving invervals (optimizer steps). If None and save_every_n_epochs > 0, will auto-calculate. Otherwise saves every epoch."
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=2,
        help=
        "Auto-calculate save_steps to save every N epochs (default: 2). Only used if --save_steps is None. Set to 0 to disable auto-calculation."
    )
    parser.add_argument(
        "--max_checkpoints_to_keep",
        type=int,
        default=5,
        help=
        "Maximum number of rolling checkpoints to keep. For step checkpoints, "
        "multiples of 5000 are preserved as milestones and do not count toward "
        "this limit. Set to -1 to keep all checkpoints."
    )
    parser.add_argument(
        "--allow_existing_checkpoints",
        action="store_true",
        help=
        "Allow training in a directory that already contains checkpoint files. By default, training will abort if checkpoints are found to prevent mixing old and new runs."
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=0,
        help="Initial optimizer step count. Use when resuming from a LoRA checkpoint so step numbering continues (e.g., --initial_step 1000)."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Optional hard cap on optimizer steps. Training stops once this step count is reached, independent of num_epochs."
    )
    return parser


def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model",
                        type=str,
                        default=None,
                        help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules",
                        type=str,
                        default="q,k,v,o,ffn.0,ffn.2",
                        help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank",
                        type=int,
                        default=32,
                        help="Rank of LoRA.")
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha (scaling). Defaults to `lora_rank` when unset.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help=
        "Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint."
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help=
        "Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model."
    )
    parser.add_argument("--preset_lora_model",
                        type=str,
                        default=None,
                        help="Which model the preset LoRA is fused to.")
    return parser


def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing",
                        default=False,
                        action="store_true",
                        help="Whether to use gradient checkpointing.")
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Gradient accumulation steps.")
    return parser


def add_wandb_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_wandb",
                        default=False,
                        action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project",
                        type=str,
                        default="wanvideo-training",
                        help="WandB project name")
    parser.add_argument("--wandb_name",
                        type=str,
                        default=None,
                        help="WandB run name (auto-generated if None)")
    # Backwards/alternative flag name used by some scripts.
    parser.add_argument("--wandb_run_name",
                        dest="wandb_name",
                        type=str,
                        default=None,
                        help="Alias for --wandb_name")
    parser.add_argument("--wandb_entity",
                        type=str,
                        default=None,
                        help="WandB entity (username or team)")
    parser.add_argument('--wandb_tags',
                        type=str,
                        nargs='*',
                        default=[],
                        help='Wandb tags')
    parser.add_argument("--wandb_notes",
                        type=str,
                        default=None,
                        help="WandB run notes")
    parser.add_argument("--wandb_log_gradients",
                        default=False,
                        action="store_true",
                        help="Log model gradients to WandB")
    parser.add_argument("--wandb_log_parameters",
                        default=False,
                        action="store_true",
                        help="Log model parameters to WandB")
    parser.add_argument(
        "--wandb_log_freq",
        type=int,
        default=100,
        help="Frequency of detailed logging (gradients, parameters)")
    parser.add_argument("--wandb_watch_model",
                        default=False,
                        action="store_true",
                        help="Use wandb.watch() to track model")
    return parser


def add_debug_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--debug_latents",
        default=False,
        action="store_true",
        help="Enable latent debugging: decode and save cond/xt/x0_hat videos every N optimizer steps.",
    )
    parser.add_argument(
        "--debug_latents_every",
        type=int,
        default=100,
        help="Log decoded latents every N optimizer steps (only when --debug_latents is enabled).",
    )
    parser.add_argument(
        "--debug_latents_fps",
        type=int,
        default=16,
        help="FPS for saved latent debug mp4s.",
    )
    parser.add_argument(
        "--debug_latents_quality",
        type=int,
        default=5,
        help="imageio/ffmpeg quality for saved latent debug mp4s (lower is better quality; typical 0-10).",
    )
    parser.add_argument(
        "--debug_latents_max_frames",
        type=int,
        default=None,
        help="Optional cap on number of frames written per debug mp4 (helps keep files small).",
    )
    parser.add_argument(
        "--debug_latents_outdir",
        type=str,
        default="debug_latents",
        help="Subdirectory (under --output_path) to store latent debug outputs.",
    )
    tiled_group = parser.add_mutually_exclusive_group()
    tiled_group.add_argument(
        "--debug_latents_vae_tiled",
        dest="debug_latents_vae_tiled",
        action="store_true",
        help="Use tiled VAE decode for latent debug outputs (default).",
    )
    tiled_group.add_argument(
        "--no_debug_latents_vae_tiled",
        dest="debug_latents_vae_tiled",
        action="store_false",
        help="Disable tiled VAE decode for latent debug outputs.",
    )
    parser.set_defaults(debug_latents_vae_tiled=True)
    parser.add_argument(
        "--debug_latents_vae_tile_size",
        type=int,
        nargs=2,
        default=(30, 52),
        help="Tile size for tiled VAE decode (H W tiles in latent-debug decoding).",
    )
    parser.add_argument(
        "--debug_latents_vae_tile_stride",
        type=int,
        nargs=2,
        default=(15, 26),
        help="Tile stride for tiled VAE decode (H W stride in latent-debug decoding).",
    )
    parser.add_argument(
        "--debug_missing_grads",
        default=False,
        action="store_true",
        help="After each backward, print trainable parameter names that still have grad=None (per rank). "
             "Uses TRACK4D_DEBUG_MISSING_GRADS in holo4d_runtime/diffusion/runner.py.",
    )
    parser.add_argument(
        "--torch_distributed_debug",
        type=str,
        default=None,
        choices=["INFO", "DETAIL"],
        help="Set TORCH_DISTRIBUTED_DEBUG before distributed init (e.g. DETAIL to map unused DDP param indices). "
             "Verbose; for diagnosis only.",
    )
    return parser


def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_pointmap_stats_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_wandb_config(parser)
    parser = add_debug_config(parser)
    parser = add_query_frame_impulse_condition_config(parser)
    return parser


def add_vdpm_kubric_dataset_config(parser: argparse.ArgumentParser):
    """
    VDPM (video -> track) dataset args (KubricWithTracks via Stream3R adapter).
    """
    parser.add_argument("--data_root", type=str, required=True, help="KubricWithTracks ROOT directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--num_views", type=int, default=17, help="Number of frames (should be 4k+1 for VAE)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512], help="width height")
    parser.add_argument("--select_idx", type=int, default=None, help="Fixed dataset index (default: random sampling)")
    parser.add_argument("--overfitting", action="store_true", help="Use dataset overfitting mode")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to run")
    return parser


def add_vdpm_track_inference_config(parser: argparse.ArgumentParser):
    """
    VDPM (video -> track) inference args shared by scripts.

    Note: V2V architecture parameters (v2v_concat_mode, v2v_channel_patch_embed_mode),
    diffusion parameters (denoising_strength, sigma_shift), and tiling parameters
    should be hardcoded in scripts as constants from inference_defaults.py.
    """
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to finetuned checkpoint (optional if using LoRA)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")

    parser.add_argument("--prompt", type=str, default="Generate dense 3D point tracking for this video.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")

    parser.add_argument(
        "--track_xyz_normalization",
        type=str,
        default="minmax",
        choices=["minmax", "signed_log_p2_p98", "signed_log_p0_p100"],
        help="Must match training `--track_xyz_normalization`.",
    )
    parser.add_argument(
        "--track_xyz_quantile_low",
        type=float,
        default=0.02,
        help="Lower quantile for signed_log_p2_p98 normalization. Must match training value."
    )
    parser.add_argument(
        "--track_xyz_quantile_high",
        type=float,
        default=0.98,
        help="Upper quantile for signed_log_p2_p98 normalization. Must match training value."
    )
    # Residual prediction (track - xyz_query) training/inference mode.
    # Used by Stream3RDatasetAdapter and visualization/denormalization utilities.
    parser.add_argument(
        "--predict_track_residual",
        default=False,
        action="store_true",
        help="Predict residual tracks (track - xyz_query). Requires xyz_video_cond in the dataset.",
    )
    parser.add_argument(
        "--residual_zero_center_normalization",
        default=False,
        action="store_true",
        help="When predicting residual tracks, normalize with zero-center mode so residual=0 maps to norm=0.",
    )
    parser.add_argument(
        "--residual_linear_normalization",
        default=False,
        action="store_true",
        help=(
            "Use fixed zero-preserving linear residual normalization (clamp(residual * K, [-1,1])) "
            "instead of signed-log residual normalization. Reuses --residual_log_scale_factor as linear K."
        ),
    )
    parser.add_argument(
        "--residual_log_scale_factor",
        type=float,
        default=3.0,
        help=(
            "Extra linear scale applied to residuals AFTER signed-log normalization in zero-center mode. "
            "Effective only when --predict_track_residual and --residual_zero_center_normalization are enabled."
        ),
    )
    parser.add_argument(
        "--stream3r_common_scale_stat",
        type=str,
        default="mean",
        choices=["mean", "median", "q02_q98_mean", "q02_q98_median", "query_p60", "dynamic_mean", "dynamic_q60"],
        help="Must match training common_scale statistic for Stream3R preprocessing.",
    )
    parser.add_argument(
        "--normalize_track_with_xyz_cond_stats",
        default=False,
        action="store_true",
        help="Use xyz_video_cond stats for normalization (matches training when enabled).",
    )
    parser.add_argument(
        "--prenormalized_track_video_signed_log",
        default=False,
        action="store_true",
        help=(
            "Training/inference compatibility flag for preloaded multimodality track_video targets: "
            "apply zero-preserving signed-log1p before VAE encoding when track_video is already in [-1,1]."
        ),
    )

    parser = add_xyz_video_condition_config(parser)
    parser = add_query_frame_impulse_condition_config(parser)

    # Tile parameters - kept for VDPM since they need to match data resolution
    parser.add_argument("--tile_size", type=int, nargs=2, default=[30, 52])
    parser.add_argument("--tile_stride", type=int, nargs=2, default=[15, 26])
    return parser
