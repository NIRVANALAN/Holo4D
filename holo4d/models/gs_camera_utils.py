"""
Camera utilities for 3D Gaussian Splatting output.

Ported from FlashWorld/utils.py — quaternion conversions, ray creation,
camera format conversion between Track4D-360 and FlashWorld conventions.

FlashWorld camera format (11 params per frame):
  [quat_r, quat_i, quat_j, quat_k, tx, ty, tz, fx/W, fy/H, cx/W, cy/H]
  - Quaternion: real part first (r, i, j, k)
  - Intrinsics: normalized by image resolution
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quaternion ↔ rotation matrix conversions (from pytorch3d via FlashWorld)
# ---------------------------------------------------------------------------

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (real-first) to rotation matrices.

    Args:
        quaternions: (..., 4) with layout (r, i, j, k).
    Returns:
        Rotation matrices (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def _standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Ensure real part is non-negative."""
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (real-first).

    Args:
        matrix: (..., 3, 3) rotation matrices.
    Returns:
        Quaternions (..., 4) with layout (r, i, j, k).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return _standardize_quaternion(out)


# ---------------------------------------------------------------------------
# Ray creation (from FlashWorld)
# ---------------------------------------------------------------------------

@torch.amp.autocast(device_type="cuda", enabled=False)
def create_rays(
    cameras: torch.Tensor,
    h: int,
    w: int,
    uv_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create camera rays for each pixel.

    Args:
        cameras: (..., 11) FlashWorld-format cameras.
        h, w: spatial resolution of the feature map (NOT full image).
        uv_offset: optional (..., h, w, 2) sub-pixel offsets.
    Returns:
        rays_o: (..., h, w, 3) ray origins.
        rays_d: (..., h, w, 3) ray directions (normalized).
    """
    prefix_shape = cameras.shape[:-1]
    cameras = cameras.flatten(0, -2).float()
    device = cameras.device
    N = cameras.shape[0]

    c2w = torch.eye(4, device=device)[None].repeat(N, 1, 1)
    c2w[:, :3, :3] = quaternion_to_matrix(cameras[:, :4])
    c2w[:, :3, 3] = cameras[:, 4:7]

    fx, fy, cx, cy = cameras[:, 7:].chunk(4, -1)
    fx, cx = fx * w, cx * w
    fy, cy = fy * h, cy * h

    inds = torch.arange(0, h * w, device=device).expand(N, h * w)
    i = inds % w + 0.5
    j = torch.div(inds, w, rounding_mode="floor") + 0.5

    u = i / cx + (uv_offset[..., 0].reshape(N, h * w) if uv_offset is not None else 0)
    v = j / cy + (uv_offset[..., 1].reshape(N, h * w) if uv_offset is not None else 0)

    zs = -torch.ones_like(i)
    xs = -(u - 1) * cx / fx * zs
    ys = (v - 1) * cy / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)
    rays_o = c2w[..., :3, 3][:, None, :].expand_as(rays_d)

    rays_o = rays_o.reshape(*prefix_shape, h, w, 3)
    rays_d = rays_d.reshape(*prefix_shape, h, w, 3)
    return rays_o, rays_d


# ---------------------------------------------------------------------------
# Camera format conversion: Track4D-360 → FlashWorld
# ---------------------------------------------------------------------------

def _matrix_to_square(mat: torch.Tensor) -> torch.Tensor:
    """Pad (..., 3, 4) → (..., 4, 4) with [0, 0, 0, 1] bottom row."""
    bottom = torch.zeros_like(mat[..., :1, :])
    bottom[..., 0, 3] = 1.0
    return torch.cat([mat, bottom], dim=-2)


@torch.amp.autocast(device_type="cuda", enabled=False)
def normalize_cameras(
    cameras: torch.Tensor,
    ref_w2c: torch.Tensor | None = None,
    T_norm: torch.Tensor | None = None,
    return_meta: bool = False,
):
    """Normalize cameras to a reference frame (first camera) and unit scale.

    Args:
        cameras: (B, N, 11) FlashWorld-format cameras.
        ref_w2c: optional (B, 1, 4, 4) reference world-to-camera.
        T_norm: optional (B, 1, 1) translation normalization factor.
        return_meta: if True, also return ref_w2c and T_norm.
    Returns:
        Normalized cameras (B, N, 11), and optionally (ref_w2c, T_norm).
    """
    cameras = cameras.float()
    B, N = cameras.shape[:2]
    c2ws = torch.zeros(B, N, 3, 4, device=cameras.device)
    c2ws[..., :3, :3] = quaternion_to_matrix(cameras[..., 0:4])
    c2ws[..., :, 3] = cameras[..., 4:7]

    if ref_w2c is None:
        ref_w2c = torch.inverse(_matrix_to_square(c2ws[:, :1]))
    c2ws = (ref_w2c.repeat(1, N, 1, 1) @ _matrix_to_square(c2ws))[..., :3, :]

    if T_norm is None:
        T_norm = c2ws[..., :3, 3].norm(dim=-1).max(dim=1)[0][..., None, None]
    c2ws[..., :3, 3] = c2ws[..., :3, 3] / (T_norm + 1e-2)

    R = matrix_to_quaternion(c2ws[..., :3, :3])
    T = c2ws[..., :3, 3]
    cameras = torch.cat([R.float(), T.float(), cameras[..., 7:]], dim=-1)

    if return_meta:
        return cameras, ref_w2c, T_norm
    return cameras


@torch.amp.autocast(device_type="cuda", enabled=False)
def c2w_K_to_flashworld_cameras(
    c2w: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """Convert Track4D-360 camera format to FlashWorld 11-param format.

    The dataset provides **OpenCV** c2w matrices (x-right, y-down, z-forward).
    FlashWorld's internal convention is **OpenGL** (x-right, y-up, z-backward).
    Both ``create_rays`` (``zs = -1``) and ``gaussian_render`` (the Y/Z flip
    before gsplat) assume the FlashWorld cameras are OpenGL.  We convert here
    so the FlashWorld format genuinely stores OpenGL c2w.

    Args:
        c2w: (..., 4, 4) cam-to-world matrices in **OpenCV** convention.
        K: (..., 3, 3) or (3, 3) intrinsic matrices.
        H, W: image resolution for normalizing intrinsics.
    Returns:
        cameras: (..., 11) FlashWorld-format cameras (OpenGL convention).
    """
    c2w = c2w.float()
    K = K.float()

    R = c2w[..., :3, :3].clone()
    t = c2w[..., :3, 3]
    # OpenCV → OpenGL: flip Y and Z columns of the rotation.
    # OpenCV c2w columns are (right, down, forward) in world space.
    # OpenGL c2w columns are (right, up, backward) in world space.
    R[..., :, 1:3] *= -1
    quat = matrix_to_quaternion(R)  # (..., 4) — now in OpenGL convention

    # Broadcast K to match c2w leading dims.
    # c2w is (..., 4, 4), K should become (..., 3, 3) with same leading dims.
    # Examples: K=(3,3) + c2w=(B,T,4,4) → K=(B,T,3,3)
    #           K=(B,3,3) + c2w=(B,T,4,4) → K=(B,T,3,3)
    target_batch = c2w.shape[:-2]  # e.g. (B, T)
    while K.dim() - 2 < len(target_batch):
        K = K.unsqueeze(-3)  # add dims before the (3,3): (3,3)→(1,3,3), (B,3,3)→(B,1,3,3)
    K = K.expand(*target_batch, 3, 3)

    fx = K[..., 0, 0] / W
    fy = K[..., 1, 1] / H
    cx = K[..., 0, 2] / W
    cy = K[..., 1, 2] / H

    intrinsics = torch.stack([fx, fy, cx, cy], dim=-1)  # (..., 4)
    cameras = torch.cat([quat, t, intrinsics], dim=-1)   # (..., 11)
    return cameras
