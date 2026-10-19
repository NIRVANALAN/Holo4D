"""
Track4D-360 shared configuration.

Single source of truth for track module config, model defaults, and
CLI argument setup. Used by both training and inference scripts.
"""

import os
from omegaconf import OmegaConf

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TRACK_CONFIG = os.path.join(_REPO_ROOT, "configs", "holo4d", "default.yaml")

# ---------------------------------------------------------------------------
# Track config loading
# ---------------------------------------------------------------------------

def load_track_config(path=None):
    """Load track module config from YAML. Returns OmegaConf DictConfig."""
    path = path or DEFAULT_TRACK_CONFIG
    cfg = OmegaConf.load(path)
    return cfg.track


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

# Model presets: (model_id_with_origin_paths, dit_dim)
# VAE spatial compression is auto-detected at runtime:
#   Wan2.1 VAE: upsampling_factor=8  → total spatial = 8*2=16x (patchify 2x)
#   Wan2.2 VAE: upsampling_factor=16 → total spatial = 16*2=32x (patchify 2x)
MODEL_PRESETS = {
    "5b": {
        "model_id_with_origin_paths": (
            "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,"
            "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,"
            "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
        ),
    },
    "1.3b": {
        "model_id_with_origin_paths": (
            "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,"
            "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,"
            "Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth"
        ),
    },
}

# Default: 5B
MODEL_ID_WITH_ORIGIN_PATHS = MODEL_PRESETS["5b"]["model_id_with_origin_paths"]

LORA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"
LORA_RANK = 64

# Data defaults
SYN4D_ROOT = os.environ.get("SYN4D_ROOT", "data/Syn4D")
DEFAULT_RESOLUTION = (512, 288)  # (W, H)
DEFAULT_NUM_FRAMES = 49


# ---------------------------------------------------------------------------
# Shared argparse setup
# ---------------------------------------------------------------------------

def get_model_preset(name):
    """Get model preset by name ('5b' or '1.3b')."""
    name = name.lower().strip()
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset '{name}'. Available: {list(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[name]


def add_track4d360_args(parser):
    """Add Track4D-360 specific args to an existing argparse parser."""
    parser.add_argument("--track_config", type=str, default=DEFAULT_TRACK_CONFIG,
                        help="YAML config for track module hyperparameters.")
    # YAML-based compositional dataset loading (preferred)
    parser.add_argument("--track4d360_data_config", type=str, default=None,
                        help="Path to YAML dataset config (e.g. configs/holo4d/syn4d_overfit_arena.yaml). "
                             "When provided, overrides --syn4d_root/--scene/--sequence_idx.")
    # Direct dataset args (fallback when no YAML config)
    parser.add_argument("--syn4d_root", type=str, default=SYN4D_ROOT)
    parser.add_argument("--scene", type=str, default=None,
                        help="Syn4D scene name filter (partial match). None = all scenes.")
    parser.add_argument("--sequence_idx", type=int, default=None,
                        help="Syn4D sequence index (None = all sequences)")
    parser.add_argument("--track_spatial_stride", type=int, default=1)
    return parser
