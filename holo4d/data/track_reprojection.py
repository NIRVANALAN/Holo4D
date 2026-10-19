"""
Track reprojection: project 3D world-space tracks into 2D image coordinates
under arbitrary camera poses.

Convention (matches Kubric/DiffSynth datasets):
  - camera_pose: (4, 4) cam2world transform
  - intrinsics:  (3, 3) standard K matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
  - points_world: (..., 3) in world coordinates
"""

import torch
from typing import Optional, Tuple


def world_to_camera(
    points_world: torch.Tensor,
    cam2world: torch.Tensor,
) -> torch.Tensor:
    """Transform world-space points to camera-space.

    Args:
        points_world: (..., 3) world coordinates.
        cam2world: (..., 4, 4) camera-to-world transform.
            Broadcastable with points_world batch dims.

    Returns:
        points_cam: (..., 3) camera-space coordinates.
    """
    # world2cam = inv(cam2world)
    R = cam2world[..., :3, :3]  # (..., 3, 3)
    t = cam2world[..., :3, 3]   # (..., 3)
    # world2cam: R_inv = R^T, t_inv = -R^T @ t
    R_inv = R.transpose(-1, -2)  # (..., 3, 3)
    # points_cam = R_inv @ (points_world - t)
    points_shifted = points_world - t  # (..., 3)
    points_cam = torch.einsum("...ij,...j->...i", R_inv, points_shifted)
    return points_cam


def camera_to_pixel(
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project camera-space points to pixel coordinates.

    Args:
        points_cam: (..., 3) camera-space points (x_cam, y_cam, z_cam).
        intrinsics: (..., 3, 3) camera intrinsic matrix K.

    Returns:
        uv: (..., 2) pixel coordinates (u, v) = (col, row).
        depth: (...,) z-depth in camera space.
    """
    z = points_cam[..., 2]  # (...,)
    # Avoid division by zero; mark z<=0 as invalid later
    z_safe = z.clamp(min=1e-6)

    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    u = points_cam[..., 0] / z_safe * fx + cx
    v = points_cam[..., 1] / z_safe * fy + cy

    uv = torch.stack([u, v], dim=-1)  # (..., 2)
    return uv, z


def compute_visibility(
    uv: torch.Tensor,
    depth: torch.Tensor,
    image_h: int,
    image_w: int,
    depth_min: float = 1e-3,
) -> torch.Tensor:
    """Compute visibility mask: in front of camera AND inside image bounds.

    Args:
        uv: (..., 2) pixel coordinates (u=col, v=row).
        depth: (...,) camera-space z-depth.
        image_h: image height in pixels.
        image_w: image width in pixels.
        depth_min: minimum valid depth.

    Returns:
        vis: (...,) bool tensor. True = visible.
    """
    in_front = depth > depth_min
    u, v = uv[..., 0], uv[..., 1]
    in_bounds = (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
    return in_front & in_bounds


def project_tracks_to_target_views(
    tracks_world: torch.Tensor,
    target_cam2world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_h: int,
    image_w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproject 3D world-space tracks into target camera views.

    This is the main entry point for track reprojection.

    Args:
        tracks_world: (N, T, 3) or (B, N, T, 3) — 3D track positions in world space.
            N = number of tracks, T = number of frames.
        target_cam2world: (T, 4, 4) or (B, T, 4, 4) — target camera poses per frame.
        intrinsics: (3, 3) or (T, 3, 3) or (B, T, 3, 3) — camera intrinsics.
            If (3,3), same intrinsics for all frames.
        image_h: target image height in pixels.
        image_w: target image width in pixels.

    Returns:
        track_uv: (B, N, T, 2) pixel coordinates in target views.
        track_vis: (B, N, T) bool visibility mask.
        track_depth: (B, N, T) z-depth in target camera space.
    """
    # Normalize to batch dims: (B, N, T, 3) and (B, T, 4, 4)
    if tracks_world.ndim == 3:
        tracks_world = tracks_world.unsqueeze(0)  # (1, N, T, 3)
    assert tracks_world.ndim == 4, f"Expected (B,N,T,3), got {tracks_world.shape}"
    B, N, T, _ = tracks_world.shape

    if target_cam2world.ndim == 3:
        target_cam2world = target_cam2world.unsqueeze(0)  # (1, T, 4, 4)
    assert target_cam2world.shape == (B, T, 4, 4) or target_cam2world.shape == (1, T, 4, 4), \
        f"cam2world shape {target_cam2world.shape} vs tracks (B={B}, T={T})"

    # Handle intrinsics broadcasting
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).expand(B, T, 3, 3)  # (B, T, 3, 3)
    elif intrinsics.ndim == 3:
        intrinsics = intrinsics.unsqueeze(0).expand(B, T, 3, 3)  # (B, T, 3, 3)
    assert intrinsics.shape[-2:] == (3, 3)

    # Reshape for per-frame projection:
    # tracks_world: (B, N, T, 3) → need per-frame cam transform
    # target_cam2world: (B, T, 4, 4)
    # We process all frames at once using broadcasting.

    # Expand cam2world to match track shape: (B, 1, T, 4, 4)
    cam2w_expanded = target_cam2world.unsqueeze(1)  # (B, 1, T, 4, 4)
    K_expanded = intrinsics.unsqueeze(1)             # (B, 1, T, 3, 3)

    # world_to_camera: (B, N, T, 3)
    points_cam = world_to_camera(tracks_world, cam2w_expanded)

    # camera_to_pixel: (B, N, T, 2), (B, N, T)
    track_uv, track_depth = camera_to_pixel(points_cam, K_expanded)

    # visibility: (B, N, T)
    track_vis = compute_visibility(track_uv, track_depth, image_h, image_w)

    return track_uv, track_vis, track_depth


def sample_tracks_from_pointmap(
    pts3d_world: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    num_tracks: int = 1024,
    strategy: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample sparse tracks from dense per-frame pointmaps.

    For Kubric/Syn4D, the dataset provides dense (H, W, 3) pts3d per frame.
    This function samples N pixel locations from frame 0 and gathers
    their 3D positions across all frames to form track trajectories.

    Args:
        pts3d_world: (T, H, W, 3) dense 3D points in world space, per frame.
        valid_mask: (T, H, W) optional bool mask. If None, all points valid.
        num_tracks: number of tracks to sample.
        strategy: "random" (uniform random from valid frame-0 pixels).

    Returns:
        tracks: (N, T, 3) sampled 3D track trajectories.
        pixel_coords: (N, 2) the (row, col) pixel coordinates in frame 0.
    """
    T, H, W, _ = pts3d_world.shape

    if valid_mask is not None:
        # Use frame-0 valid mask to determine sample locations
        frame0_valid = valid_mask[0]  # (H, W)
    else:
        frame0_valid = torch.ones(H, W, dtype=torch.bool, device=pts3d_world.device)

    valid_indices = frame0_valid.nonzero(as_tuple=False)  # (M, 2) — (row, col)
    M = valid_indices.shape[0]

    if M == 0:
        # No valid points — return empty tracks
        tracks = torch.zeros(0, T, 3, device=pts3d_world.device, dtype=pts3d_world.dtype)
        pixel_coords = torch.zeros(0, 2, device=pts3d_world.device, dtype=torch.long)
        return tracks, pixel_coords

    actual_n = min(num_tracks, M)

    if strategy == "random":
        perm = torch.randperm(M, device=pts3d_world.device)[:actual_n]
        selected = valid_indices[perm]  # (actual_n, 2)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    rows = selected[:, 0]  # (actual_n,)
    cols = selected[:, 1]  # (actual_n,)

    # Gather tracks across all frames: (T, actual_n, 3)
    tracks = pts3d_world[:, rows, cols, :]  # (T, actual_n, 3)
    tracks = tracks.permute(1, 0, 2)  # (actual_n, T, 3)

    return tracks, selected
