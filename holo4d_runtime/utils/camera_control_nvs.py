"""
Camera control utilities for novel view synthesis (NVS) training.

This module provides functions to compute relative camera poses and Plücker ray embeddings
for video-to-video training with camera conditioning.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from einops import rearrange


def compute_relative_camera_pose(
    source_c2w: np.ndarray,
    target_c2w: np.ndarray
) -> np.ndarray:
    """
    Compute relative camera pose from source to target.

    Args:
        source_c2w: (4, 4) camera-to-world matrix for source camera
        target_c2w: (4, 4) camera-to-world matrix for target camera

    Returns:
        (4, 4) relative pose matrix that transforms from source camera frame to target camera frame
    """
    # Compute world-to-camera for source
    source_w2c = np.linalg.inv(source_c2w)

    # Relative pose: target_c2w @ source_w2c
    # This gives the transformation from source camera frame to target camera frame
    relative_pose = target_c2w @ source_w2c

    return relative_pose.astype(np.float32)


def custom_meshgrid(*args):
    """Helper for creating meshgrid with 'ij' indexing (torch>=2.0.0)."""
    return torch.meshgrid(*args, indexing='ij')


def compute_plucker_rays(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    device: str = "cpu",
    use_ray_origin: bool = False,
) -> torch.Tensor:
    """
    Compute Plücker ray coordinates for a camera.

    Args:
        c2w: (B, T, 4, 4) camera-to-world matrices
        intrinsics: (B, T, 4) camera intrinsics [fx, fy, cx, cy]
        height: Image height
        width: Image width
        device: Device for computation
        use_ray_origin: If True, use [rays_o, rays_d] (Lyra-2 convention).
                        If False (default), use [rays_o x rays_d, rays_d] (cross-product).

    Returns:
        (B, T, H, W, 6) Plücker coordinates
    """
    calc_dtype = torch.float32
    device_type = "cuda" if "cuda" in device else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        # Force computations to float32 because torch.linalg.cross does not support bfloat16.
        c2w = c2w.to(device=device, dtype=calc_dtype)
        intrinsics = intrinsics.to(device=device, dtype=calc_dtype)

        B = c2w.shape[0]
        T = c2w.shape[1]

        # Create pixel grid
        j, i = custom_meshgrid(
            torch.linspace(0, height - 1, height, device=device, dtype=calc_dtype),
            torch.linspace(0, width - 1, width, device=device, dtype=calc_dtype),
        )
        i = i.reshape([1, 1, height * width]).expand([B, T, height * width]) + 0.5  # [B, T, HxW]
        j = j.reshape([1, 1, height * width]).expand([B, T, height * width]) + 0.5  # [B, T, HxW]

        # Extract intrinsic parameters
        fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)  # B, T, 1

        # Compute ray directions in camera space
        zs = torch.ones_like(i, dtype=calc_dtype)  # [B, T, HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)

        directions = torch.stack((xs, ys, zs), dim=-1)  # B, T, HW, 3
        directions = directions / directions.norm(dim=-1, keepdim=True)  # B, T, HW, 3

        # Transform ray directions to world space
        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, T, HW, 3

        # Ray origins in world space
        rays_o = c2w[..., :3, 3]  # B, T, 3
        rays_o = rays_o[..., None, :].expand_as(rays_d)  # B, T, HW, 3

        # Compute ray representation
        if use_ray_origin:
            # Lyra-2 convention: [rays_o, rays_d] — direct origin + direction
            plucker = torch.cat([rays_o, rays_d], dim=-1)  # B, T, HW, 6
        else:
            # Cross-product convention: [rays_o × rays_d, rays_d]
            rays_dxo = torch.linalg.cross(rays_o, rays_d)
            plucker = torch.cat([rays_dxo, rays_d], dim=-1)  # B, T, HW, 6
        plucker = plucker.reshape(B, T, height, width, 6)  # B, T, H, W, 6

    return plucker


def compute_plucker_rays_dit_resolution(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    height_dit: int,
    width_dit: int,
    device: str = "cpu",
    use_ray_origin: bool = False,
) -> torch.Tensor:
    """Compute Plucker rays directly on the DiT token grid.

    This mirrors Vista4D's camera embedding path: the pixel grid is sampled at
    DiT-token resolution while the coordinates are mapped back to the original
    image coordinate system before applying intrinsics.
    """
    calc_dtype = torch.float32
    device_type = "cuda" if "cuda" in device else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        c2w = c2w.to(device=device, dtype=calc_dtype)
        intrinsics = intrinsics.to(device=device, dtype=calc_dtype)

        B = c2w.shape[0]
        T = c2w.shape[1]
        patch_height = float(height) / float(height_dit)
        patch_width = float(width) / float(width_dit)

        j, i = custom_meshgrid(
            torch.linspace(0, height_dit - 1, height_dit, device=device, dtype=calc_dtype),
            torch.linspace(0, width_dit - 1, width_dit, device=device, dtype=calc_dtype),
        )
        i = i.reshape([1, 1, height_dit * width_dit]).expand([B, T, height_dit * width_dit]) + 0.5
        j = j.reshape([1, 1, height_dit * width_dit]).expand([B, T, height_dit * width_dit]) + 0.5

        # Match Vista4D's token-grid-to-image-grid convention.
        i = i * patch_width + (patch_width / 2.0)
        j = j * patch_height + (patch_height / 2.0)

        fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
        zs = torch.ones_like(i, dtype=calc_dtype)
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)

        directions = torch.stack((xs, ys, zs), dim=-1)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
        rays_o = c2w[..., :3, 3]
        rays_o = rays_o[..., None, :].expand_as(rays_d)

        if use_ray_origin:
            plucker = torch.cat([rays_o, rays_d], dim=-1)
        else:
            rays_dxo = torch.linalg.cross(rays_o, rays_d)
            plucker = torch.cat([rays_dxo, rays_d], dim=-1)
        plucker = plucker.reshape(B, T, height_dit, width_dit, 6)

    return plucker


def _camera_intrinsics_to_batched_fx_fy_cx_cy(
    target_intrinsics: torch.Tensor,
    T: int,
) -> torch.Tensor:
    if target_intrinsics.ndim == 2:
        assert target_intrinsics.shape == (3, 3), f"Expected (3,3), got {target_intrinsics.shape}"
        fx = target_intrinsics[0, 0]
        fy = target_intrinsics[1, 1]
        cx = target_intrinsics[0, 2]
        cy = target_intrinsics[1, 2]
        intrinsics_array = torch.stack([fx, fy, cx, cy]).to(dtype=torch.float32)
        return intrinsics_array.unsqueeze(0).repeat(T, 1).unsqueeze(0)
    if target_intrinsics.ndim == 3:
        assert target_intrinsics.shape == (T, 3, 3), \
            f"Expected ({T},3,3) per-frame K, got {target_intrinsics.shape}"
        fx_t = target_intrinsics[:, 0, 0]
        fy_t = target_intrinsics[:, 1, 1]
        cx_t = target_intrinsics[:, 0, 2]
        cy_t = target_intrinsics[:, 1, 2]
        return torch.stack([fx_t, fy_t, cx_t, cy_t], dim=-1).unsqueeze(0)
    raise AssertionError(f"target_intrinsics must be (3,3) or (T,3,3); got {target_intrinsics.shape}")


def _relative_target_c2w_from_source0(
    source_camera: Dict,
    target_camera: Dict,
    num_frames: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_c2w = torch.as_tensor(source_camera["c2w_opencv"], dtype=torch.float32, device=device)
    target_c2w = torch.as_tensor(target_camera["c2w_opencv"], dtype=torch.float32, device=device)
    target_intrinsics = torch.as_tensor(target_camera["intrinsics"], dtype=torch.float32, device=device)

    T = source_c2w.shape[0]
    assert T == num_frames, f"Camera frames {T} must match video frames {num_frames}"
    assert len(source_c2w.shape) == 3, f"Expected c2w_opencv shape (T, 4, 4), got {source_c2w.shape}"

    intrinsics_batch = _camera_intrinsics_to_batched_fx_fy_cx_cy(target_intrinsics, T)
    source0_w2c = torch.linalg.inv(source_c2w[0])
    relative_c2w = source0_w2c.unsqueeze(0) @ target_c2w
    return relative_c2w.unsqueeze(0), intrinsics_batch


def process_camera_pair_to_plucker(
    source_camera: Dict,
    target_camera: Dict,
    height: int,
    width: int,
    num_frames: int,
    device: str = "cpu",
    use_ray_origin: bool = False,
) -> torch.Tensor:
    """
    Process a pair of source and target cameras to generate relative Plücker ray embeddings.

    This function:
    1. Computes relative pose from the *source reference frame* to target camera
    2. Generates Plücker ray coordinates for the relative camera view
    3. Formats the output for use with SimpleAdapter

    Args:
        source_camera: Dict with 'c2w_opencv' (T, 4, 4) and 'intrinsics' (3, 3)
        target_camera: Dict with 'c2w_opencv' (T, 4, 4) and 'intrinsics' (3, 3)
        height: Video frame height
        width: Video frame width
        num_frames: Number of frames in the video
        device: Device for computation

    Returns:
        (T, H, W, 6) Plücker embedding for the relative camera motion
    """
    relative_c2w_batch, intrinsics_batch = _relative_target_c2w_from_source0(
        source_camera, target_camera, num_frames, device
    )

    # Compute Plücker rays
    plucker_embedding = compute_plucker_rays(
        relative_c2w_batch,
        intrinsics_batch,
        height,
        width,
        device=device,
        use_ray_origin=use_ray_origin,
    )  # (1, T, H, W, 6)

    # Remove batch dimension
    plucker_embedding = plucker_embedding[0]  # (T, H, W, 6)

    return plucker_embedding


def process_camera_pair_to_plucker_dit_resolution(
    source_camera: Dict,
    target_camera: Dict,
    height: int,
    width: int,
    num_frames: int,
    height_dit: int,
    width_dit: int,
    time_stride: int = 4,
    device: str = "cpu",
    use_ray_origin: bool = False,
) -> torch.Tensor:
    """Generate Vista4D-style raw 6D Plucker features on the DiT token grid."""
    relative_c2w_batch, intrinsics_batch = _relative_target_c2w_from_source0(
        source_camera, target_camera, num_frames, device
    )
    plucker = compute_plucker_rays_dit_resolution(
        relative_c2w_batch,
        intrinsics_batch,
        height,
        width,
        height_dit,
        width_dit,
        device=device,
        use_ray_origin=use_ray_origin,
    )
    return plucker[:, :: int(time_stride)].contiguous()


def prepare_camera_control_latents(
    plucker_embedding: torch.Tensor,
    num_frames: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Prepare camera control latents for input to SimpleAdapter.

    This function converts Plücker embeddings to the format expected by SimpleAdapter:
    - Rearranges dimensions from (T, H, W, 6) to (1, 24, T, H, W)
    - The factor of 4 (6->24) comes from unpacking to match VAE latent resolution

    Args:
        plucker_embedding: (T, H, W, 6) Plücker coordinates
        num_frames: Number of frames (should match T)
        dtype: Target dtype for the latents

    Returns:
        (1, 24, T, H, W) control latents ready for SimpleAdapter
    """
    assert plucker_embedding.ndim == 4, f"Expected plucker_embedding shape (T, H, W, 6), got {plucker_embedding.shape}"

    # Rearrange from (T, H, W, 6) to (1, 6, T, H, W)
    control_camera_video = plucker_embedding.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 6, T, H, W)

    # Prepare for VAE latent matching (repeat first frame 4x + rest of frames)
    # This matches the temporal packing used in Wan Video (4x first frame for reference)
    control_camera_latents = torch.cat(
        [
            torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
            control_camera_video[:, :, 1:]
        ],
        dim=2
    ).transpose(1, 2)  # (1, T+3, 6, H, W)

    # Reshape to pack temporal dimension (matching VAE latent packing)
    b, f, c, h, w = control_camera_latents.shape
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
    # Result: (1, 24, num_frames, H, W)

    return control_camera_latents.to(dtype=dtype)


def prepare_camera_control_pose_latents(
    relative_pose_rt: torch.Tensor,
    num_frames: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Pack relative pose (R|t) per original frame into latent-time features.

    We follow the same temporal packing used for video latents and Plücker control:
    - Repeat the first frame 4x, then append the remaining frames (T+3 frames total)
    - Group by 4 to match VAE temporal compression (length = (T-1)//4 + 1)

    Input:
        relative_pose_rt: (T, 12) where 12 = [R(3x3) flattened | t(3)]
        num_frames: T

    Output:
        (1, 48, F, 1, 1) where 48 = 12 * 4 and F = (T-1)//4 + 1
        This is intended for ReCamMaster-style per-layer pose encoders (Linear(48, dim)).
    """
    if not isinstance(relative_pose_rt, torch.Tensor):
        raise TypeError(f"relative_pose_rt must be a torch.Tensor, got {type(relative_pose_rt)}")
    if relative_pose_rt.ndim != 2 or int(relative_pose_rt.shape[1]) != 12:
        raise ValueError(f"Expected relative_pose_rt shape (T, 12), got {tuple(relative_pose_rt.shape)}")
    if int(relative_pose_rt.shape[0]) != int(num_frames):
        raise ValueError(f"Pose T={int(relative_pose_rt.shape[0])} does not match num_frames={int(num_frames)}")

    # (T+3, 12): repeat first frame 4x, then append remaining frames 1..T-1
    packed_seq = torch.cat(
        [relative_pose_rt[0:1].repeat(4, 1), relative_pose_rt[1:]],
        dim=0,
    )
    if packed_seq.shape[0] % 4 != 0:
        raise ValueError(f"Packed pose length must be divisible by 4, got {int(packed_seq.shape[0])}")

    f = int(packed_seq.shape[0] // 4)
    packed = packed_seq.view(f, 4, 12).reshape(f, 48)  # (F, 48)

    # (1, 48, F, 1, 1)
    out = packed.transpose(0, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return out.to(dtype=dtype)


def prepare_camera_control_pose_latents_raw(
    relative_pose_rt: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Prepare raw RT pose features for ReCamMaster-style conditioning without temporal packing.

    Input:
        relative_pose_rt: (F, 12) where F is the latent-time length (e.g., 21 for 81-frame videos)

    Output:
        (1, 12, F, 1, 1)
    """
    if not isinstance(relative_pose_rt, torch.Tensor):
        raise TypeError(f"relative_pose_rt must be a torch.Tensor, got {type(relative_pose_rt)}")
    if relative_pose_rt.ndim != 2 or int(relative_pose_rt.shape[1]) != 12:
        raise ValueError(f"Expected relative_pose_rt shape (F, 12), got {tuple(relative_pose_rt.shape)}")

    out = relative_pose_rt.transpose(0, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return out.to(dtype=dtype)
