import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import os
from typing_extensions import Literal

class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1, downscale_factor=8):
        super(SimpleAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by downscale_factor
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)

        # After pixel_unshuffle, channels multiply by downscale_factor²
        # e.g., downscale_factor=8 → 8² = 64, downscale_factor=16 → 16² = 256
        channel_multiplier = downscale_factor ** 2

        # Convolution: reduce spatial dimensions by a factor of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * channel_multiplier, out_dim, kernel_size=kernel_size, stride=stride, padding=0)

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    # def zero_init_weights(self):
    #     """Zero-initialize all weights to ensure camera control doesn't affect outputs at first step."""
    #     with torch.no_grad():
    #         # Zero-initialize convolution
    #         self.conv.weight.zero_()
    #         if self.conv.bias is not None:
    #             self.conv.bias.zero_()

    #         # Zero-initialize residual blocks
    #         for module in self.residual_blocks.modules():
    #             if isinstance(module, nn.Conv2d):
    #                 module.weight.zero_()
    #                 if module.bias is not None:
    #                     module.bias.zero_()

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out
    
    def process_camera_coordinates(
        self,
        direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"],
        length: int,
        height: int,
        width: int,
        speed: float = 1/54,
        origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    ):
        if origin is None:
            origin = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        coordinates = generate_camera_coordinates(direction, length, speed, origin)
        plucker_embedding = process_pose_file(coordinates, width, height)
        return plucker_embedding
        
    

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out
    
class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def custom_meshgrid(*args):
    # torch>=2.0.0 only
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def process_pose_file(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding



def generate_camera_coordinates(
    direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"],
    length: int,
    speed: float = 1/54,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
):
    coordinates = [list(origin)]
    while len(coordinates) < length:
        coor = coordinates[-1].copy()
        if "Left" in direction:
            coor[9] += speed
        if "Right" in direction:
            coor[9] -= speed
        if "Up" in direction:
            coor[13] += speed
        if "Down" in direction:
            coor[13] -= speed
        if "In" in direction:
            coor[18] -= speed
        if "Out" in direction:
            coor[18] += speed
        coordinates.append(coor)
    return coordinates


def initialize_camera_control_adapter(model, vae=None, in_dim_control_adapter=24):
    """
    Initialize camera control adapter for NVS (Novel View Synthesis).

    This function creates a SimpleAdapter for camera control and attaches it to the model.
    It auto-detects VAE spatial downsampling factor and model dimensions.

    Args:
        model: DiT model (dit or dit2) to attach adapter to
        vae: VAE model for auto-detecting spatial downsampling (optional)
        in_dim_control_adapter: Input dimension for camera control (default: 24 for Plücker rays)

    Returns:
        The model with camera control adapter initialized (for chaining)
    """
    if model is None:
        return None

    # Skip if adapter already exists
    if hasattr(model, "control_adapter") and model.control_adapter is not None:
        return model

    # Extract model dimensions
    patch_size = tuple(getattr(model, "patch_size", (1, 2, 2)))
    dim = int(getattr(model, "dim"))

    # Auto-detect VAE spatial downsampling factor.
    # camera-control's WanVideoVAE / WanVideoVAE38 exposes this as `upsampling_factor`
    # (8 for Wan2.1 1.3B, 16 for Wan2.2 5B). Diffusers-style VAEs use
    # `spatial_compression_ratio`. Check both so the adapter matches the
    # shape that was saved in the pretrained checkpoint (see
    # holo4d_runtime/pipeline/wan_video_v2v.py:151 which assumes downscale²=64
    # for Wan2.1).
    if vae is not None:
        if hasattr(vae, 'upsampling_factor'):
            downscale_factor = int(vae.upsampling_factor)
        elif hasattr(vae, 'spatial_compression_ratio'):
            downscale_factor = int(vae.spatial_compression_ratio)
        else:
            downscale_factor = 16
    else:
        downscale_factor = 16

    # Create adapter
    model.control_adapter = SimpleAdapter(
        in_dim_control_adapter,
        dim,
        kernel_size=patch_size[1:],
        stride=patch_size[1:],
        downscale_factor=downscale_factor,
    )

    # NOTE: Do NOT zero-initialize adapter weights!
    # Zero-init makes the model learn to ignore camera control (camera signal = 0 at start).
    # Instead, use default PyTorch initialization so camera signal is present from the start.
    # The cam_add_projector (initialized with small std=0.02) will modulate the signal strength.
    # Reference: video4spatial_ttt does not zero-init the camera encoder.
    # model.control_adapter.zero_init_weights()  # REMOVED!

    # Move to same device/dtype as model
    ref_param = next(model.parameters(), None)
    if ref_param is not None:
        model.control_adapter = model.control_adapter.to(
            device=ref_param.device,
            dtype=ref_param.dtype
        )

    return model


def initialize_perlayer_camera_projectors(model):
    """
    Initialize per-layer camera projectors for NVS with per-layer camera control.

    This is an alternative to the standard camera control that adds camera features
    to each DiT block instead of only in patchify.

    Architecture (ReCamMaster-style):
    1. control_adapter processes Plücker rays → camera features (B, dim, T, H, W)
    2. Each DiT block gets a trainable projection layer: Linear(dim, dim)
    3. Camera features are added before self-attention in each block:
       - Reshape camera features to (B, T*H*W, dim)
       - Add to normalized input: input_x = norm1(x) + camera_features
       - Apply self-attention and project: x = x + gate_msa * projector(self_attn(input_x, ...))

    ReCamMaster initialization (train_recammaster.py line 296-300):
        block.projector.weight = nn.Parameter(torch.eye(dim))  # IDENTITY MATRIX
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    With identity initialization, projector(x) = x at start, so the forward pass:
        x = x + gate_msa * projector(self_attn(input_x, freqs))
    is equivalent to the standard DiTBlock behavior initially.

    Args:
        model: DiT model (dit or dit2) with blocks attribute

    Returns:
        The model with per-layer camera projectors initialized (for chaining)
    """
    if model is None:
        return None

    # Check if model has blocks
    if not hasattr(model, "blocks") or model.blocks is None:
        return model

    dim = int(getattr(model, "dim"))

    # Add camera_projector to each block
    for block in model.blocks:
        if not hasattr(block, "camera_projector"):
            # Create projection layer: Linear(dim, dim)
            camera_projector = nn.Linear(dim, dim, bias=True)

            # ReCamMaster-style IDENTITY initialization:
            # - weight = identity matrix: projector(x) = x initially
            # - bias = zeros
            # This ensures the forward pass matches pretrained behavior at initialization:
            #   x = x + gate_msa * projector(attn_out) = x + gate_msa * attn_out (standard DiTBlock)
            with torch.no_grad():
                camera_projector.weight.copy_(torch.eye(dim))
                camera_projector.bias.zero_()

            # Move to same device/dtype as block
            ref_param = next(block.parameters(), None)
            if ref_param is not None:
                camera_projector = camera_projector.to(
                    device=ref_param.device,
                    dtype=ref_param.dtype
                )

            block.camera_projector = camera_projector

    return model


def initialize_perlayer_camera_pose_encoders(model, pose_dim: int = 48, init_scale: float = 0.0):
    """
    Initialize per-layer camera pose encoders for ReCamMaster-style camera control using RT (no Plücker rays).

    In ReCamMaster, each block applies a linear camera encoder before adding camera features to the
    normalized token stream:
        cam_emb = cam_encoder(pose_feat)
        input_x = norm1(x) + cam_emb
        x = x + gate * projector(self_attn(input_x))

    This helper attaches a per-block `camera_pose_encoder: nn.Linear(pose_dim, dim)` so the pipeline can
    pass packed RT features (e.g., 12*4=48 per latent frame after temporal packing).

    Initialization:
    - By default, ZERO initialization (init_scale=0.0) so RT conditioning is a no-op until trained,
      preserving base model behavior (ReCamMaster-style safe init).
    - If init_scale > 0, uses weight ~ N(0, init_scale), bias = 0.

    Args:
        model: DiT model (dit or dit2) with blocks attribute
        pose_dim: Input pose feature dimension (default: 48 = 12 RT * 4 temporal pack)
        init_scale: Std for normal init (default: 0.0 => zero-init)

    Returns:
        The model with per-layer camera pose encoders initialized
    """
    if model is None:
        return None

    if not hasattr(model, "blocks") or model.blocks is None:
        return model

    dim = int(getattr(model, "dim"))

    for block in model.blocks:
        if not hasattr(block, "camera_pose_encoder"):
            enc = nn.Linear(int(pose_dim), dim, bias=True)
            with torch.no_grad():
                if float(init_scale) == 0.0:
                    enc.weight.zero_()
                else:
                    nn.init.normal_(enc.weight, std=float(init_scale))
                if enc.bias is not None:
                    enc.bias.zero_()

            ref_param = next(block.parameters(), None)
            if ref_param is not None:
                enc = enc.to(device=ref_param.device, dtype=ref_param.dtype)

            block.camera_pose_encoder = enc

    return model


def initialize_perlayer_camera_encoders(model, plucker_dim=6, init_scale=0.02, add_projector=True):
    """
    Initialize per-layer camera encoders for NVS with direct raymap encoding.

    This follows the paper approach (CogVideoX-ControlNet style) where:
    1. Each block has a cam_encoder: Conv3d(plucker_dim, dim, kernel_size=1)
       - Pointwise 3D convolution to preserve spatial-temporal structure
    2. Camera features (B, 6, T, H', W') are encoded per-layer to (B, dim, T, H', W')
    3. Encoded features are directly ADDED to x before self-attention: x = x + cam_emb

    If add_projector=True (ReCamMaster style):
    4. Each block also has camera_projector: Linear(dim, dim) - identity-initialized
    5. Self-attention output is projected: x = x + gate_msa * projector(self_attn(input_x))

    Reference: https://github.com/TheDenk/cogvideox-controlnet/blob/e21a4a3407eee8f372373fa1593297d7526caeeb/cogvideo_controlnet.py#L66
    - Uses Conv2d with kernel_size=1 for spatial processing
    - Here we use Conv3d with kernel_size=1 for spatial-temporal processing

    Key difference from "add" mode with SimpleAdapter:
    - "add": Global SimpleAdapter encodes (B,24,T,H,W)→(B,dim,T,H',W'), then per-layer Linear projection
    - "direct_raymap_add": NO global SimpleAdapter, each layer directly encodes (B,6,T,H',W')→(B,dim,T,H',W')

    This allows each layer to learn custom camera feature extraction instead of sharing global features.

    Initialization:
    - cam_encoder: ZERO initialization (following ReCamMaster approach)
      - Even though patchify Conv2d produces non-zero features, zero-init per-layer encoder
        ensures camera has zero influence at training start
    - camera_projector: IDENTITY initialization (following ReCamMaster approach)
      - Makes model identical to base model at init, gradually learns projection during training

    Args:
        model: DiT model (dit or dit2) with blocks attribute
        plucker_dim: Input dimension for Plücker rays (default: 6 for d×o, d)
        init_scale: Standard deviation for normal initialization (default: 0.02, not used with zero/identity init)
        add_projector: If True, also add identity-initialized camera_projector (ReCamMaster style)

    Returns:
        The model with per-layer camera encoders (and optionally projectors) initialized (for chaining)
    """
    if model is None:
        return None

    # Check if model has blocks
    if not hasattr(model, "blocks") or model.blocks is None:
        return model

    dim = int(getattr(model, "dim"))

    # Add cam_encoder to each block
    for block in model.blocks:
        if not hasattr(block, "cam_encoder"):
            # Create encoder: Conv3d(plucker_dim, dim, kernel_size=1)
            # Pointwise convolution preserves spatial-temporal structure
            cam_encoder = nn.Conv3d(
                plucker_dim,
                dim,
                kernel_size=1,  # Pointwise convolution (no spatial/temporal mixing)
                stride=1,
                padding=0,
                bias=True
            )

            # Zero initialization (following ReCamMaster approach)
            # Even though patchify Conv2d produces non-zero features,
            # zero-init per-layer encoder ensures camera has zero influence at training start
            # Model starts identical to base model, gradually learns camera control
            with torch.no_grad():
                nn.init.zeros_(cam_encoder.weight)
                if cam_encoder.bias is not None:
                    nn.init.zeros_(cam_encoder.bias)

            # Move to same device/dtype as block
            ref_param = next(block.parameters(), None)
            if ref_param is not None:
                cam_encoder = cam_encoder.to(
                    device=ref_param.device,
                    dtype=ref_param.dtype
                )

            block.cam_encoder = cam_encoder

        # Optionally add camera_projector (ReCamMaster style)
        if add_projector and not hasattr(block, "camera_projector"):
            # Create projector: Linear(dim, dim)
            # Identity initialization makes model identical to base model at init
            camera_projector = nn.Linear(dim, dim, bias=True)

            # Identity initialization (following ReCamMaster approach)
            # projector.weight = eye(dim), projector.bias = zeros(dim)
            with torch.no_grad():
                nn.init.eye_(camera_projector.weight)
                if camera_projector.bias is not None:
                    nn.init.zeros_(camera_projector.bias)

            # Move to same device/dtype as block
            ref_param = next(block.parameters(), None)
            if ref_param is not None:
                camera_projector = camera_projector.to(
                    device=ref_param.device,
                    dtype=ref_param.dtype
                )

            block.camera_projector = camera_projector

    return model


def initialize_vista4d_plucker_camera_encoders(model, plucker_dim=6, add_projector=True):
    """Initialize Vista4D-style per-block raw Plucker camera injection.

    Each DiT block receives raw 6D Plucker tokens with shape ``(B, L, 6)`` and
    applies a zero-initialized ``Linear(6, dim)`` before self-attention. The
    attention output projector is identity-initialized, matching Vista4D's
    no-op-at-step-0 behavior.
    """
    if model is None:
        return None
    if not hasattr(model, "blocks") or model.blocks is None:
        return model

    dim = int(getattr(model, "dim"))
    for block in model.blocks:
        if hasattr(block, "cam_encoder") and not isinstance(block.cam_encoder, nn.Linear):
            raise TypeError(
                "vista4d_plucker mode expects block.cam_encoder to be Linear. "
                f"Found {type(block.cam_encoder).__name__}; use a fresh model or a matching checkpoint."
            )
        if not hasattr(block, "cam_encoder"):
            cam_encoder = nn.Linear(int(plucker_dim), dim, bias=True)
            with torch.no_grad():
                nn.init.zeros_(cam_encoder.weight)
                if cam_encoder.bias is not None:
                    nn.init.zeros_(cam_encoder.bias)

            ref_param = next(block.parameters(), None)
            if ref_param is not None:
                cam_encoder = cam_encoder.to(device=ref_param.device, dtype=ref_param.dtype)
            block.cam_encoder = cam_encoder

        if add_projector and not hasattr(block, "camera_projector"):
            camera_projector = nn.Linear(dim, dim, bias=True)
            with torch.no_grad():
                nn.init.eye_(camera_projector.weight)
                if camera_projector.bias is not None:
                    nn.init.zeros_(camera_projector.bias)

            ref_param = next(block.parameters(), None)
            if ref_param is not None:
                camera_projector = camera_projector.to(device=ref_param.device, dtype=ref_param.dtype)
            block.camera_projector = camera_projector

    return model


def initialize_vista4d_latent_stream_patch_embeddings(model, mask_in_channels: int = 512):
    """Initialize Vista4D-style source / point-cloud stream patch embeddings."""
    if model is None:
        return None
    if not hasattr(model, "patch_embedding"):
        return model
    wan_patch = model.patch_embedding
    out_channels, rgb_in_channels, p1, p2, p3 = wan_patch.weight.shape

    def _clone_rgb_embedding():
        emb = nn.Conv3d(
            rgb_in_channels,
            out_channels,
            kernel_size=(p1, p2, p3),
            stride=(p1, p2, p3),
            bias=True,
        )
        with torch.no_grad():
            emb.weight.copy_(wan_patch.weight.detach())
            emb.bias.copy_(wan_patch.bias.detach())
        return emb

    def _zero_mask_embedding():
        emb = nn.Conv3d(
            int(mask_in_channels),
            out_channels,
            kernel_size=(p1, p2, p3),
            stride=(p1, p2, p3),
            bias=True,
        )
        with torch.no_grad():
            emb.weight.zero_()
            emb.bias.zero_()
        return emb

    ref_param = next(model.parameters(), None)
    device = ref_param.device if ref_param is not None else wan_patch.weight.device
    dtype = ref_param.dtype if ref_param is not None else wan_patch.weight.dtype

    for name, factory in (
        ("vista4d_source_patch_embedding", _clone_rgb_embedding),
        ("vista4d_pointcloud_patch_embedding", _clone_rgb_embedding),
        ("vista4d_source_mask_patch_embedding", _zero_mask_embedding),
        ("vista4d_pointcloud_mask_patch_embedding", _zero_mask_embedding),
    ):
        if not hasattr(model, name):
            setattr(model, name, factory().to(device=device, dtype=dtype))

    return model


def initialize_gen3c_warp_condition_patch_embedding(model, in_channels: int = 32):
    """Zero-init GEN3C-style warped RGB/mask latent channel patch embedding."""
    if model is None or not hasattr(model, "patch_embedding"):
        return model
    existing = getattr(model, "gen3c_warp_condition_patch_embedding", None)
    if existing is not None and int(getattr(existing, "in_channels", -1)) == int(in_channels):
        return model

    wan_patch = model.patch_embedding
    out_channels, _, p1, p2, p3 = wan_patch.weight.shape
    emb = nn.Conv3d(
        int(in_channels),
        out_channels,
        kernel_size=(p1, p2, p3),
        stride=(p1, p2, p3),
        bias=wan_patch.bias is not None,
    )
    ref_param = next(model.parameters(), None)
    device = ref_param.device if ref_param is not None else wan_patch.weight.device
    dtype = ref_param.dtype if ref_param is not None else wan_patch.weight.dtype
    emb = emb.to(device=device, dtype=dtype)
    with torch.no_grad():
        emb.weight.zero_()
        if emb.bias is not None:
            emb.bias.zero_()
    model.gen3c_warp_condition_patch_embedding = emb
    return model


def initialize_pre_dit_camera_encoder(model, init_scale=0.02):
    """
    Initialize pre-DiT camera encoder for camera-control runtime style camera injection.

    This encoder processes camera features ONCE after patch embedding and domain tokens,
    BEFORE the DiT blocks. It's used in combination with per-layer camera control:
    1. control_adapter (SimpleAdapter) encodes: (B, 24, T, H, W) → (B, dim, T, H', W')
    2. pre_dit_cam_encoder processes: (B, dim, T, H', W') → (B, dim, T, H', W')
    3. Features are added to x: x = x + pre_dit_cam_encoder(control_adapter(camera))
    4. Then per-layer cam_encoder in each DiT block further processes camera features

    This is ADDITIVE with per-layer camera control - both can be enabled simultaneously.

    Architecture:
    - pre_dit_cam_encoder: Conv3d(dim, dim, kernel_size=1)
    - Pointwise 3D convolution to preserve spatial-temporal structure
    - Zero initialization ensures no effect at training start

    Args:
        model: DiT model (dit or dit2)
        init_scale: Standard deviation for normal initialization (default: 0.02, not used with zero init)

    Returns:
        The model with pre_dit_cam_encoder initialized (for chaining)
    """
    if model is None:
        return None

    dim = int(getattr(model, "dim", 3072))  # Default to 3072 for Wan2.2-5B

    # Create pre-DiT cam_encoder: Conv3d(dim, dim, kernel_size=1)
    pre_dit_cam_encoder = nn.Conv3d(
        dim,
        dim,
        kernel_size=1,  # Pointwise convolution (no spatial/temporal mixing)
        stride=1,
        padding=0,
        bias=True
    )

    # Zero initialization (following ReCamMaster approach)
    # Model starts identical to base model, gradually learns pre-DiT camera control
    with torch.no_grad():
        nn.init.zeros_(pre_dit_cam_encoder.weight)
        if pre_dit_cam_encoder.bias is not None:
            nn.init.zeros_(pre_dit_cam_encoder.bias)

    # Move to same device/dtype as model
    ref_param = next(model.parameters(), None)
    if ref_param is not None:
        pre_dit_cam_encoder = pre_dit_cam_encoder.to(
            device=ref_param.device,
            dtype=ref_param.dtype
        )

    model.pre_dit_cam_encoder = pre_dit_cam_encoder

    return model


def initialize_noise_camera_encoder(model, init_scale=0.02):
    """
    Initialize a camera encoder used to inject camera information into the "noisy" token feature map.

    This is similar to camera-control runtime's control path that adds camera-conditioned features right after
    patch embedding (before DiT blocks). In this repo, we keep it as a separate module so it can be
    enabled/disabled independently from per-layer and pre-DiT injections.

    Expected usage (when enabled):
      cam = control_adapter(camera_latents)      # (B, 24, T, H, W) -> (B, dim, T, H', W')
      cam = noise_cam_encoder(cam)               # (B, dim, T, H', W') -> (B, dim, T, H', W')
      x = patchify(latents) + cam                # before domain tokens / DiT blocks

    Architecture:
    - noise_cam_encoder: Conv3d(dim, dim, kernel_size=1)
    - Zero initialization ensures no effect at training start.
    """
    if model is None:
        return None

    dim = int(getattr(model, "dim", 3072))  # Default to 3072 for Wan2.2-5B

    noise_cam_encoder = nn.Conv3d(
        dim,
        dim,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )

    with torch.no_grad():
        nn.init.zeros_(noise_cam_encoder.weight)
        if noise_cam_encoder.bias is not None:
            nn.init.zeros_(noise_cam_encoder.bias)

    ref_param = next(model.parameters(), None)
    if ref_param is not None:
        noise_cam_encoder = noise_cam_encoder.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )

    model.noise_cam_encoder = noise_cam_encoder
    return model


def initialize_camera_patchify_layers(model, vae=None, patch_size=(2, 2)):
    """
    Initialize camera patchify layers for direct_raymap_add mode.

    In direct_raymap_add mode, 24D Plücker rays need to be converted to 6D before
    being passed to per-layer cam_encoder modules. This is done via:
    1. PixelUnshuffle: (B*T, 24, H, W) → (B*T, 24*downscale^2, H', W')
    2. Conv2d patchify: (B*T, 24*downscale^2, H', W') → (B*T, 6, H'', W'')

    These layers must be initialized BEFORE loading checkpoint, otherwise checkpoint
    keys (_camera_patchify_conv.*) will be "unexpected keys".

    Args:
        model: DiT model (dit or dit2)
        vae: VAE model to detect spatial compression ratio (default: 16 for Wan2.2)
        patch_size: Patch size tuple (ph, pw) for Conv2d (default: (2, 2))

    Returns:
        The model with camera patchify layers initialized
    """
    if model is None:
        return None

    # Detect VAE spatial compression
    if hasattr(model, 'control_adapter') and hasattr(model.control_adapter, 'pixel_unshuffle'):
        downscale_factor = model.control_adapter.pixel_unshuffle.downscale_factor
    elif vae is not None and hasattr(vae, 'spatial_compression_ratio'):
        downscale_factor = vae.spatial_compression_ratio
    else:
        downscale_factor = 16  # Default for Wan2.2 VAE

    ph, pw = patch_size
    C = 24  # Plücker rays have 24 channels (6 components × 4 for patchification)

    # Get reference device/dtype
    ref_param = next(model.parameters(), None)
    device = ref_param.device if ref_param is not None else "cpu"
    dtype = ref_param.dtype if ref_param is not None else torch.float32

    # Create PixelUnshuffle layer if not exist
    if not hasattr(model, "_camera_patchify_unshuffle"):
        model._camera_patchify_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        model._camera_patchify_unshuffle = model._camera_patchify_unshuffle.to(device=device, dtype=dtype)

    # Create Conv2d patchify layer if not exist
    if not hasattr(model, "_camera_patchify_conv"):
        model._camera_patchify_conv = nn.Conv2d(
            C * (downscale_factor ** 2),  # 24 * 16^2 = 6144
            6,  # Extract 6D Plücker rays
            kernel_size=(ph, pw),
            stride=(ph, pw),
            padding=0,
            bias=True
        ).to(device=device, dtype=dtype)

        # Initialize like nn.Linear / patch_embed (xavier_uniform for weight, zeros for bias)
        # This layer extracts features - normal init is fine
        # Zero influence at step=0 is ensured by zero-init cam_encoder (not this layer)
        with torch.no_grad():
            w = model._camera_patchify_conv.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(model._camera_patchify_conv.bias, 0)

    return model


def initialize_camera_patchify_plucker_encoder(model, init_zero: bool = True):
    """
    Initialize a lightweight Plücker encoder that operates on patchified 6D raymaps.

    This is an alternative to `SimpleAdapter` (control_adapter) for Plücker-based conditioning.
    It reuses the existing camera patchify layers:
      - `_camera_patchify_unshuffle`: PixelUnshuffle(downscale_factor)
      - `_camera_patchify_conv`: Conv2d(24*downscale^2 -> 6, kernel=patch_size, stride=patch_size)

    Then it adds a global pointwise Conv3d encoder:
      - `_camera_plucker_patchify_encoder`: Conv3d(6 -> dim, kernel_size=1)

    Output shape matches `control_adapter(x)`:
      (B, dim, F, H', W') where (F,H',W') match the patchified token grid.

    Notes:
    - Disabled by default via the pipeline flag `camera_plucker_encoder_mode="control_adapter"`.
    - If `init_zero=True`, the encoder starts as a no-op (zero weights/bias), preserving base-model behavior
      until trained (ReCamMaster-style safe initialization).
    """
    if model is None:
        return None
    if not hasattr(model, "dim"):
        return model

    dim = int(getattr(model, "dim"))

    if not hasattr(model, "_camera_plucker_patchify_encoder"):
        enc = nn.Conv3d(6, dim, kernel_size=1, stride=1, padding=0, bias=True)
        with torch.no_grad():
            if init_zero:
                nn.init.zeros_(enc.weight)
                if enc.bias is not None:
                    nn.init.zeros_(enc.bias)
            else:
                nn.init.kaiming_uniform_(enc.weight, a=np.sqrt(5))
                if enc.bias is not None:
                    nn.init.zeros_(enc.bias)

        ref_param = next(model.parameters(), None)
        if ref_param is not None:
            enc = enc.to(device=ref_param.device, dtype=ref_param.dtype)
        model._camera_plucker_patchify_encoder = enc

    return model


def initialize_camera_raymap_token_projection(model, vae=None, init_zero: bool = True):
    """
    Initialize the opt-in Plücker raymap token-conditioning path.

    This path follows the camera-token formulation:
      1. packed camera latents contain patchified Plücker raymaps
      2. `_camera_patchify_*` recovers a 6D raymap token grid
      3. `_camera_plucker_patchify_encoder` is Ecam: Conv3d(6 -> dim)
      4. each DiT block adds Ecam(raymaps) to video tokens before self-attention

    Ecam is zero-initialized and the per-block attention projector is identity-
    initialized, matching the ReCamMaster no-op-at-step-0 convention.
    """
    initialize_camera_patchify_layers(model, vae)
    initialize_camera_patchify_plucker_encoder(model, init_zero=init_zero)
    initialize_perlayer_camera_projectors(model)
    return model


def initialize_perlayer_camera_adapter_projectors(model, init_scale=0.02):
    """
    Initialize per-layer camera adapter projectors for the "add" mode.
    
    This is aligned with video4spatial_ttt/wan_model_wrapper.py approach where:
    1. Uses control_adapter to process Plücker rays globally: (B, 24, T, H, W) → (B, dim, T, H, W)
    2. Each block has a simple Linear(dim, dim) projector
    3. Camera features are projected and ADDED to x: x = x + projector(cam_feat)
    
    Reference (wan_model_wrapper.py line 88-91, 113-124):
        self.cam_encoder = nn.Sequential(nn.Linear(24, dim))
        
        if cam_emb is not None:
            cam_emb = self.cam_encoder(cam_emb)
            x = x + cam_emb
    
    Initialization:
    - Uses small scale initialization (default PyTorch-like) instead of zeros
    - This allows camera signal to flow from the start of training
    - Reference uses default PyTorch init, not zero init
    
    NOTE: Changed from zero init to small scale init (0.02 std) to match
    reference behavior where camera information flows from training start.
    
    Args:
        model: DiT model (dit or dit2) with blocks attribute
        init_scale: Standard deviation for normal initialization (default: 0.02)
    
    Returns:
        The model with per-layer camera adapter projectors initialized (for chaining)
    """
    if model is None:
        return None
    
    # Check if model has blocks
    if not hasattr(model, "blocks") or model.blocks is None:
        return model
    
    dim = int(getattr(model, "dim"))
    
    # Add cam_add_projector to each block (named differently from camera_projector to avoid conflict)
    for block in model.blocks:
        if not hasattr(block, "cam_add_projector"):
            # Create projector: Linear(dim, dim)
            cam_add_projector = nn.Linear(dim, dim, bias=True)
            
            # Small scale initialization (like transformer default)
            # This allows camera signal to flow from training start
            # Reference uses default PyTorch init, not zero init
            with torch.no_grad():
                nn.init.normal_(cam_add_projector.weight, std=init_scale)
                nn.init.zeros_(cam_add_projector.bias)
            
            # Move to same device/dtype as block
            ref_param = next(block.parameters(), None)
            if ref_param is not None:
                cam_add_projector = cam_add_projector.to(
                    device=ref_param.device,
                    dtype=ref_param.dtype
                )
            
            block.cam_add_projector = cam_add_projector
    
    return model
