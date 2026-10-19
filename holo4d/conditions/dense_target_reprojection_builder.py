"""
dense_target_reprojection_builder.py — pure geometry utility (no nn.Module).

Batched forward-warp of source RGB + depth into the target camera frame using
z-buffer compositing.  Mirrors the per-frame logic of ``forward_warp_rgbd_to_target``
in ``holo4d/visualization/track_panels.py`` but operates entirely on GPU
tensors with no Python loops over the B or T dimensions.

Geometry recap
--------------
For each source pixel (u_s, v_s) with depth d_s:

  1. Back-project to source camera space:
         P_src = K_src⁻¹ @ [u_s, v_s, 1]^T  * d_s

  2. Lift to world space:
         P_world = src_c2w[:3, :3] @ P_src + src_c2w[:3, 3]

  3. Transform to target camera space:
         P_tgt = tgt_w2c[:3, :3] @ P_world + tgt_w2c[:3, 3]

  4. Project to target image:
         u_t = fx_t * P_tgt[0] / P_tgt[2] + cx_t
         v_t = fy_t * P_tgt[1] / P_tgt[2] + cy_t

  5. Z-buffer via scatter_reduce_('amin') — nearest-z wins per target pixel.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dense_target_reprojection(
    source_rgb: torch.Tensor,           # (B, T, H, W, 3)   float32 [0, 1]
    source_depth: torch.Tensor,         # (B, T, H, W)       float32, metric depth
    source_depth_valid: torch.Tensor,   # (B, T, H, W)       bool
    src_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    tgt_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    intrinsics: torch.Tensor,           # (3, 3) or (T, 3, 3) or (B, T, 3, 3)  — source K
    image_h: int,
    image_w: int,
    *,
    return_confidence: bool = False,
    return_disocclusion: bool = False,
    target_intrinsics: Optional[torch.Tensor] = None,   # K to project INTO target view
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
]:
    """Forward-warp source RGB+depth into the target camera, fully batched.

    No Python loops over B or T — all operations are vectorized tensor ops.

    Args:
        source_rgb: ``(B, T, H, W, 3)`` float32, values in [0, 1].
        source_depth: ``(B, T, H, W)`` float32, depth in world/metric units.
        source_depth_valid: ``(B, T, H, W)`` bool, True where depth is reliable.
        src_cam2world: ``(B, T, 4, 4)`` or ``(T, 4, 4)`` source extrinsics.
        tgt_cam2world: ``(B, T, 4, 4)`` or ``(T, 4, 4)`` target extrinsics.
        intrinsics: ``(3, 3)`` or ``(T, 3, 3)`` or ``(B, T, 3, 3)``.
            Assumed identical for source and target cameras.  Broadcast as needed.
        image_h: target image height.
        image_w: target image width.
        return_confidence: also return per-target-pixel confidence (count of
            source pixels that landed on it, normalised by max over the frame).
        return_disocclusion: also return a per-target-pixel disocclusion mask
            (True where the warp produced no source contribution at all).

    Returns:
        warped_rgb:   ``(B, T, H, W, 3)`` float32 [0, 1]; 0 where invalid.
        warped_depth: ``(B, T, H, W)`` float32, depth in *target camera space*;
                      0 where invalid.
        warp_valid:   ``(B, T, H, W)`` bool; True where reprojection hit.

        When ``return_confidence`` or ``return_disocclusion`` is True, an extra
        ``aux`` dict is appended with keys
        ``"confidence"`` ``(B, T, H, W)`` float32 in ``[0, 1]`` and/or
        ``"disocclusion"`` ``(B, T, H, W)`` bool.
    """
    B, T, H, W, _ = source_rgb.shape
    device = source_rgb.device
    dtype = source_rgb.dtype
    N = H * W  # pixels per frame

    # ------------------------------------------------------------------
    # 1. Broadcast and invert camera matrices  (B, T, 4, 4)
    # ------------------------------------------------------------------
    src_c2w = _broadcast_cameras(src_cam2world, B, T, device, dtype)  # (B, T, 4, 4)
    tgt_c2w = _broadcast_cameras(tgt_cam2world, B, T, device, dtype)
    tgt_w2c = _batch_inv44(tgt_c2w)  # (B, T, 4, 4)

    # ------------------------------------------------------------------
    # 2. Broadcast intrinsics  (B, T, 3, 3) — separate K for source unproject
    #    and target projection (4RC labels need this; non-4RC paths default
    #    target_intrinsics=None and reuse `intrinsics` for both).
    # ------------------------------------------------------------------
    K = _broadcast_intrinsics(intrinsics, B, T, device, dtype)  # (B, T, 3, 3) — source K
    if target_intrinsics is not None:
        K_tgt_full = _broadcast_intrinsics(target_intrinsics, B, T, device, dtype)
    else:
        K_tgt_full = K

    # ------------------------------------------------------------------
    # 3. Build pixel grid (u, v) in source image space  (H, W)
    # ------------------------------------------------------------------
    # v = row index (y), u = col index (x)
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # each (H, W)
    # Expand to (B, T, H*W)
    u_flat = u_grid.reshape(1, 1, N).expand(B, T, N)  # (B, T, N)
    v_flat = v_grid.reshape(1, 1, N).expand(B, T, N)

    # ------------------------------------------------------------------
    # 4. Back-project: source pixel + depth → source camera 3D
    # ------------------------------------------------------------------
    # Extract intrinsic scalars  (B, T, 1) each
    fx_s = K[:, :, 0, 0].unsqueeze(-1)   # (B, T, 1)
    fy_s = K[:, :, 1, 1].unsqueeze(-1)
    cx_s = K[:, :, 0, 2].unsqueeze(-1)
    cy_s = K[:, :, 1, 2].unsqueeze(-1)

    d_flat = source_depth.reshape(B, T, N)           # (B, T, N)
    valid_flat = source_depth_valid.reshape(B, T, N) # (B, T, N)

    # P_src = [(u-cx)*d/fx, (v-cy)*d/fy, d]  — (B, T, N, 3)
    x_src = (u_flat - cx_s) * d_flat / fx_s.clamp(min=1e-6)
    y_src = (v_flat - cy_s) * d_flat / fy_s.clamp(min=1e-6)
    pts_src = torch.stack([x_src, y_src, d_flat], dim=-1)  # (B, T, N, 3)

    # ------------------------------------------------------------------
    # 5. Source camera → world  (using src_c2w)
    # ------------------------------------------------------------------
    R_src = src_c2w[:, :, :3, :3]   # (B, T, 3, 3)
    t_src = src_c2w[:, :, :3, 3]    # (B, T, 3)

    # pts_world = R_src @ pts_src^T + t_src
    # (B, T, 3, 3) @ (B, T, 3, N) → (B, T, 3, N)
    pts_world = (R_src @ pts_src.permute(0, 1, 3, 2)) + t_src.unsqueeze(-1)
    # → (B, T, N, 3)
    pts_world = pts_world.permute(0, 1, 3, 2)

    # ------------------------------------------------------------------
    # 6. World → target camera  (using tgt_w2c)
    # ------------------------------------------------------------------
    R_tgt = tgt_w2c[:, :, :3, :3]   # (B, T, 3, 3)
    t_tgt = tgt_w2c[:, :, :3, 3]    # (B, T, 3)

    pts_tgt = (R_tgt @ pts_world.permute(0, 1, 3, 2)) + t_tgt.unsqueeze(-1)
    pts_tgt = pts_tgt.permute(0, 1, 3, 2)  # (B, T, N, 3)

    z_tgt = pts_tgt[..., 2]  # (B, T, N)

    # ------------------------------------------------------------------
    # 7. Project to target image plane (use K_tgt — different from K_src
    #    for the 4RC labeled path).
    # ------------------------------------------------------------------
    fx_t = K_tgt_full[:, :, 0, 0].unsqueeze(-1)
    fy_t = K_tgt_full[:, :, 1, 1].unsqueeze(-1)
    cx_t = K_tgt_full[:, :, 0, 2].unsqueeze(-1)
    cy_t = K_tgt_full[:, :, 1, 2].unsqueeze(-1)

    z_safe = z_tgt.clamp(min=1e-6)
    u_tgt = fx_t * pts_tgt[..., 0] / z_safe + cx_t  # (B, T, N)
    v_tgt = fy_t * pts_tgt[..., 1] / z_safe + cy_t

    u_round = u_tgt.round().long()  # (B, T, N)
    v_round = v_tgt.round().long()

    # ------------------------------------------------------------------
    # 8. Build validity mask (per source pixel)
    # ------------------------------------------------------------------
    depth_ok   = d_flat > 1e-6                              # (B, T, N) bool
    z_ok       = z_tgt > 1e-6                               # (B, T, N)
    in_bounds  = (
        (u_round >= 0) & (u_round < W) &
        (v_round >= 0) & (v_round < H)
    )                                                        # (B, T, N)
    src_valid  = valid_flat & depth_ok & z_ok & in_bounds   # (B, T, N)

    # ------------------------------------------------------------------
    # 9. Z-buffer forward warp via scatter_reduce_('amin')
    #    Flat target pixel index: pix = v*W + u  ∈ [0, H*W)
    # ------------------------------------------------------------------
    pix_idx = (v_round * W + u_round).clamp(0, N - 1)  # (B, T, N)

    # Depth buffer: initialise with +inf; amin gives nearest-z winner
    z_buf = torch.full((B, T, N), float("inf"), device=device, dtype=dtype)
    z_src_masked = z_tgt.clone()
    z_src_masked[~src_valid] = float("inf")  # invalid pixels pushed to +inf

    z_buf.scatter_reduce_(2, pix_idx, z_src_masked, reduce="amin", include_self=True)
    # z_buf: (B, T, N) — nearest z at each target pixel, inf where no hit

    # Winner mask: a source pixel is the winner if its z equals the buffer value
    is_winner = src_valid & (z_tgt == z_buf.gather(2, pix_idx))  # (B, T, N)

    # ------------------------------------------------------------------
    # 10. Scatter RGB of winning source pixels into target frame
    #
    # We only scatter winner pixels. Non-winners must not scatter zeros back
    # into their projected target pixels, otherwise a farther source pixel can
    # overwrite the nearest z-buffer winner and create false holes. Route every
    # non-winner to a dummy bin at index N, then slice the real N pixels.
    # ------------------------------------------------------------------
    rgb_flat = source_rgb.reshape(B, T, N, 3)  # (B, T, N, 3)

    # Initialize outputs as zeros / False with one extra dummy bin.
    warped_rgb_flat_full   = torch.zeros(B, T, N + 1, 3, device=device, dtype=dtype)
    warped_depth_flat_full = torch.zeros(B, T, N + 1,    device=device, dtype=dtype)
    warp_valid_flat_full   = torch.zeros(B, T, N + 1,    device=device, dtype=torch.bool)

    dummy_idx = torch.full_like(pix_idx, N)
    win_idx = torch.where(is_winner, pix_idx.clamp(0, N - 1), dummy_idx)

    # RGB: zero out non-winner source colours before scattering
    is_winner_3 = is_winner.unsqueeze(-1).expand_as(rgb_flat)   # (B, T, N, 3)
    rgb_to_write = torch.where(is_winner_3, rgb_flat, torch.zeros_like(rgb_flat))
    warped_rgb_flat_full.scatter_(2, win_idx.unsqueeze(-1).expand_as(rgb_flat), rgb_to_write)

    # Depth: zero out non-winner z values
    depth_to_write = torch.where(is_winner, z_tgt, torch.zeros_like(z_tgt))
    warped_depth_flat_full.scatter_(2, win_idx, depth_to_write)

    # Valid: write is_winner directly — True only for genuine winners
    warp_valid_flat_full.scatter_(2, win_idx, is_winner)
    warped_rgb_flat = warped_rgb_flat_full[:, :, :N]
    warped_depth_flat = warped_depth_flat_full[:, :, :N]
    warp_valid_flat = warp_valid_flat_full[:, :, :N]

    # ------------------------------------------------------------------
    # 11. Reshape outputs
    # ------------------------------------------------------------------
    warped_rgb   = warped_rgb_flat.reshape(B, T, H, W, 3)
    warped_depth = warped_depth_flat.reshape(B, T, H, W)
    warp_valid   = warp_valid_flat.reshape(B, T, H, W)

    if not (return_confidence or return_disocclusion):
        return warped_rgb, warped_depth, warp_valid

    # ------------------------------------------------------------------
    # 12. Auxiliary maps (computed only on demand — both reuse the
    #     in_bounds source-pixel set already prepared above).
    # ------------------------------------------------------------------
    aux: dict = {}

    if return_confidence:
        # Per-target-pixel confidence: number of source pixels (winners +
        # losers) that hit each target pixel, divided by the per-frame max.
        # This is a cheap "evidence count" that tells the model how densely
        # supported a target pixel is, beyond the boolean valid mask.
        ones_in_bounds = src_valid.float()  # (B, T, N) — 1 where this src pixel hit a real target slot
        count_flat = torch.zeros(B, T, N, device=device, dtype=dtype)
        count_flat.scatter_add_(2, win_idx, ones_in_bounds)
        # Normalise per (B, T) frame to [0, 1]
        max_per_frame = count_flat.amax(dim=2, keepdim=True).clamp(min=1.0)
        confidence_flat = count_flat / max_per_frame
        # Zero confidence where the warp produced no winner (avoid leaking
        # an unsupported pixel as "max-normalised 1.0").
        confidence_flat = confidence_flat * warp_valid_flat.float()
        aux["confidence"] = confidence_flat.reshape(B, T, H, W)

    if return_disocclusion:
        # Disocclusion = "this target pixel received no source contribution".
        # For a single-camera reprojection every target pixel is in-frame, so
        # disocclusion is exactly ``~warp_valid``.  Kept as an explicit named
        # output so the encoder doesn't have to re-derive it.
        aux["disocclusion"] = ~warp_valid

    return warped_rgb, warped_depth, warp_valid, aux


def build_dense_target_reprojection_softsplat(
    source_rgb: torch.Tensor,           # (B, T, H, W, 3)   float32 [0, 1]
    source_depth: torch.Tensor,         # (B, T, H, W)       float32 metric depth
    source_depth_valid: torch.Tensor,   # (B, T, H, W)       bool
    src_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    tgt_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    intrinsics: torch.Tensor,           # source K: (3, 3) or (T, 3, 3) or (B, T, 3, 3)
    image_h: int,
    image_w: int,
    *,
    depth_weight_scale: float = 50.0,
    target_intrinsics: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-frame GEN3C-style bilinear soft-splat reprojection.

    This keeps the same output contract as ``build_dense_target_reprojection``:
    ``(warped_rgb, warped_depth, warp_valid)`` in ``(B,T,H,W,3)``,
    ``(B,T,H,W)``, ``(B,T,H,W)`` layouts. Unlike
    ``build_dense_target_reprojection_hybrid``, this is strictly the
    source[t] -> target[t] pass, so ablations can use the soft-splat backend
    without also enabling multi-frame cache fusion.
    """
    from .soft_splat_warp import (
        _bilinear_splat_points,
        _ensure_batch,
        _ensure_intrinsics,
        _unproject_to_world,
    )

    device = source_rgb.device
    dtype = torch.float32

    B, T = source_rgb.shape[:2]
    H, W = image_h, image_w
    src_c2w = _ensure_batch(src_cam2world, B).to(device=device, dtype=dtype)
    tgt_c2w = _ensure_batch(tgt_cam2world, B).to(device=device, dtype=dtype)
    K = _ensure_intrinsics(intrinsics, B, T).to(device=device, dtype=dtype)
    K_tgt = (
        _ensure_intrinsics(target_intrinsics, B, T).to(device=device, dtype=dtype)
        if target_intrinsics is not None else K
    )

    src_rgb = source_rgb.to(device=device, dtype=dtype)
    src_depth = torch.clamp(source_depth.to(device=device, dtype=dtype), min=0.0, max=1e4)
    src_valid = source_depth_valid.to(device=device).float()
    src_depth = src_depth * (src_valid > 0).float()

    BT = B * T
    src_rgb_flat = src_rgb.reshape(BT, H * W, 3)
    src_depth_flat = src_depth.reshape(BT, 1, H, W)
    src_valid_flat = (src_valid > 0).reshape(BT, H * W).float()
    src_c2w_flat = src_c2w.reshape(BT, 4, 4)
    tgt_w2c_flat = torch.linalg.inv(tgt_c2w).reshape(BT, 4, 4)
    K_src_flat = K.reshape(BT, 3, 3)
    K_tgt_flat = K_tgt.reshape(BT, 3, 3)

    world_pts = _unproject_to_world(src_depth_flat, src_c2w_flat, K_src_flat, H, W)
    world_pts = world_pts.reshape(BT, H * W, 3)

    R = tgt_w2c_flat[:, :3, :3]
    t_vec = tgt_w2c_flat[:, :3, 3]
    cam_pts = torch.einsum("bij,bnj->bni", R, world_pts) + t_vec.unsqueeze(1)
    z = cam_pts[..., 2]
    z_safe = z.clamp(min=1e-7)
    u = K_tgt_flat[:, 0, 0:1] * cam_pts[..., 0] / z_safe + K_tgt_flat[:, 0, 2:3]
    v = K_tgt_flat[:, 1, 1:2] * cam_pts[..., 1] / z_safe + K_tgt_flat[:, 1, 2:3]
    front = z > 1e-6
    mask = src_valid_flat * front.to(dtype)
    uv = torch.stack([u, v], dim=-1)

    values = torch.cat([src_rgb_flat, z.unsqueeze(-1)], dim=-1)
    splat_points = _bilinear_splat_points
    splat_backend = os.environ.get("TRACK4D360_DENSE_HYBRID_SPLAT_BACKEND", "triton").lower()
    if splat_backend in {"triton", "auto"} and source_rgb.is_cuda:
        try:
            from .triton_point_splat import bilinear_splat_points_triton, triton_available
            if triton_available():
                splat_points = bilinear_splat_points_triton
        except Exception:
            if splat_backend == "triton":
                raise

    rgbd, valid = splat_points(
        values,
        mask,
        z,
        uv,
        image_h=H,
        image_w=W,
        depth_weight_scale=depth_weight_scale,
        clamp_output=False,
    )
    rgbd = rgbd.reshape(B, T, 4, H, W)
    valid = valid.reshape(B, T, 1, H, W)
    warped_rgb = torch.clamp(rgbd[:, :, :3], 0.0, 1.0).permute(0, 1, 3, 4, 2).contiguous()
    warped_depth = rgbd[:, :, 3]
    warp_valid = valid[:, :, 0] > 0
    return warped_rgb, warped_depth, warp_valid


def _uniform_cache_frame_indices(T: int, max_frames: int, device: torch.device) -> torch.Tensor:
    if max_frames <= 0 or max_frames >= T:
        return torch.arange(T, device=device, dtype=torch.long)
    max_frames = max(1, int(max_frames))
    if max_frames == 1:
        return torch.tensor([0], device=device, dtype=torch.long)

    idx = torch.linspace(0, T - 1, steps=max_frames, device=device).round().long()
    idx = torch.unique(idx, sorted=True)
    if idx.numel() == max_frames:
        return idx

    used = {int(i) for i in idx.detach().cpu().tolist()}
    extra = [i for i in range(T) if i not in used]
    if extra:
        fill = torch.tensor(extra[: max_frames - idx.numel()], device=device, dtype=torch.long)
        idx = torch.cat([idx, fill]).sort().values
    return idx[:max_frames]


def _camera_pose_features(cam2world: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
    """Compact per-frame camera features for cache keyframe selection.

    Uses camera translation plus optical-axis direction from c2w. Features are
    averaged over batch so every sample in a batch uses the same cache indices.
    """
    c2w = cam2world.detach().to(device=device, dtype=torch.float32)
    if c2w.ndim == 3:
        c2w = c2w.unsqueeze(0)
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"cam2world must have shape (T,4,4) or (B,T,4,4), got {tuple(c2w.shape)}")
    if c2w.shape[1] != T:
        raise ValueError(f"cam2world T dimension {c2w.shape[1]} != T={T}")

    trans = c2w[..., :3, 3].mean(dim=0)  # (T, 3)
    trans = trans - trans.mean(dim=0, keepdim=True)
    trans_scale = trans.norm(dim=-1).amax().clamp_min(1.0e-6)
    trans = trans / trans_scale

    # Camera-to-world +Z is the projective optical axis for the datasets used
    # here. Include it so pure rotation/orbit clips do not collapse to the same
    # translation feature.
    forward = c2w[..., :3, 2].mean(dim=0)
    forward = torch.nn.functional.normalize(forward, dim=-1, eps=1.0e-6)
    return torch.cat([trans, 0.5 * forward], dim=-1)


def _farthest_camera_cache_indices(
    features: torch.Tensor,
    max_frames: int,
    device: torch.device,
) -> torch.Tensor:
    T = int(features.shape[0])
    if max_frames <= 0 or max_frames >= T:
        return torch.arange(T, device=device, dtype=torch.long)
    if max_frames == 1:
        center = T // 2
        return torch.tensor([center], device=device, dtype=torch.long)

    pairwise = torch.cdist(features, features, p=2)
    if pairwise.amax() <= 1.0e-7:
        return _uniform_cache_frame_indices(T, max_frames, device)

    selected = [0, T - 1]
    while len(selected) < max_frames:
        selected_t = torch.tensor(selected, device=device, dtype=torch.long)
        min_dist = pairwise[:, selected_t].amin(dim=1)
        min_dist[selected_t] = -1.0
        selected.append(int(torch.argmax(min_dist).item()))
    return torch.tensor(sorted(selected), device=device, dtype=torch.long)


def select_dense_hybrid_cache_frame_indices(
    T: int,
    device: torch.device,
    *,
    max_frames: int | None = None,
    temporal_stride: int = 4,
    cam2world: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Select source frames for the static multi-frame cache.

    The dense hybrid warp still produces all ``T`` target frames. This helper
    only shrinks the static source cloud. By default it selects 7 source frames
    for standard 49-frame clips using camera-pose farthest-point sampling:
    endpoints first, then frames that maximize coverage in translation + view
    direction space. Set ``TRACK4D360_DENSE_HYBRID_CACHE_FRAMES=all`` or ``0``
    to restore the old all-source-frame cache.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    env = os.environ.get("TRACK4D360_DENSE_HYBRID_CACHE_FRAMES")
    if max_frames is None:
        if env is None or env == "":
            max_frames = min(7, int(T))
        elif env.lower() in {"all", "full", "none"}:
            max_frames = int(T)
        else:
            max_frames = int(env)
    if max_frames <= 0 or max_frames >= T:
        return torch.arange(T, device=device, dtype=torch.long)
    max_frames = max(1, int(max_frames))

    policy = os.environ.get("TRACK4D360_DENSE_HYBRID_CACHE_POLICY", "camera").lower()
    if policy in {"uniform", "linspace", "temporal"} or cam2world is None:
        return _uniform_cache_frame_indices(T, max_frames, device)
    if policy not in {"camera", "pose", "fps"}:
        raise ValueError(
            "TRACK4D360_DENSE_HYBRID_CACHE_POLICY must be camera, pose, fps, "
            f"uniform, linspace, or temporal; got {policy!r}"
        )

    features = _camera_pose_features(cam2world, T, device)
    return _farthest_camera_cache_indices(features, max_frames, device)


# ---------------------------------------------------------------------------
# Hybrid soft-splat dense reprojection (Lyra-2 / GEN3C style)
# ---------------------------------------------------------------------------

def build_dense_target_reprojection_hybrid(
    source_rgb: torch.Tensor,           # (B, T, H, W, 3)   float32 [0, 1]
    source_depth: torch.Tensor,         # (B, T, H, W)       float32 metric depth
    source_depth_valid: torch.Tensor,   # (B, T, H, W)       bool
    src_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    tgt_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    intrinsics: torch.Tensor,           # (3, 3) or (T, 3, 3) or (B, T, 3, 3) — source K
    image_h: int,
    image_w: int,
    *,
    source_static_mask: "torch.Tensor | None" = None,   # (B, T, H, W) bool — True = include in multi-frame pass
    target_dynamic_mask: "torch.Tensor | None" = None,  # (B, T, H, W) bool — True = use per-frame here
    dynamic_tracks_world: "torch.Tensor | None" = None,  # (B,N,T,3)/(N,T,3), optional track-guided dynamic fusion
    dynamic_tracks_valid_mask: "torch.Tensor | None" = None,  # (B,N,T)/(N,T), optional
    dynamic_tracks_query_count: "int | torch.Tensor | None" = None,
    static_mask: "torch.Tensor | None" = None,          # legacy alias for source_static_mask
    depth_weight_scale: float = 50.0,
    target_intrinsics: Optional[torch.Tensor] = None,   # 4RC-path: K to project INTO target view
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for :func:`build_dense_target_reprojection` using
    the Lyra-2 / GEN3C bilinear soft-splat (depth-weighted) with multi-frame
    fusion at static target pixels and per-frame fallback at dynamic pixels.

    Same input / output contract as the z-buffer ``build_dense_target_reprojection``:
    returns ``(warped_rgb, warped_depth, warp_valid)`` with shapes
    ``(B, T, H, W, 3)``, ``(B, T, H, W)``, ``(B, T, H, W)`` (bool). The dense
    branch encoder consumes these unchanged — only the contents of the
    tensors differ.

    Improvements over the z-buffer version:
      - Sub-pixel bilinear scatter (4 neighbours per source pixel) closes
        the rounding holes that nearest-cell hard z-buffer leaves in the
        visible region.
      - Depth-weighted blending — closer pixels dominate softly without
        single-winner-takes-all artefacts.
      - Multi-frame fusion at static pixels: a deterministic source-cache
        keyframe set contributes to every target view (default 7 frames for
        T=49, selected by camera-pose coverage). Set
        TRACK4D360_DENSE_HYBRID_CACHE_FRAMES=0 or all to restore the old
        all-source-frame cache.
      - Dynamic regions (Kubric / Syn4D) can use track-guided multi-frame
        fusion when ``dynamic_tracks_world`` is provided: source-frame track
        colours are sampled from RGB and splatted at the same identities'
        target-time positions. Pixels without track support fall back to the
        per-frame ``s=t→t`` warp.

    See ``holo4d/conditions/soft_splat_warp.py`` for the underlying
    primitives and ``visualize_multiframe_warp.py`` for a panel-by-panel
    debug renderer.
    """
    from .soft_splat_warp import (
        _bilinear_splat_points,
        _unproject_to_world, _ensure_batch, _ensure_intrinsics,
    )

    device = source_rgb.device
    dtype = torch.float32

    B, T = source_rgb.shape[:2]
    src_c2w = _ensure_batch(src_cam2world, B).to(device=device, dtype=dtype)
    tgt_c2w = _ensure_batch(tgt_cam2world, B).to(device=device, dtype=dtype)
    K = _ensure_intrinsics(intrinsics, B, T).to(device=device, dtype=dtype)
    # Source K is used for the unprojection step (line ~495 below); when
    # the 4RC path passes a separate `target_intrinsics`, K_tgt_full is used
    # instead for the per-chunk target projection (line ~578). Non-4RC
    # paths leave target_intrinsics=None and reuse K for both.
    if target_intrinsics is not None:
        K_tgt_full = _ensure_intrinsics(target_intrinsics, B, T).to(device=device, dtype=dtype)
    else:
        K_tgt_full = K

    src_rgb = source_rgb.to(device=device, dtype=dtype)              # (B, T, H, W, 3)
    src_depth = torch.clamp(source_depth.to(device=device, dtype=dtype),
                            min=0.0, max=1e4)                         # (B, T, H, W)
    src_valid = source_depth_valid.to(device=device).float()          # (B, T, H, W)
    src_depth = src_depth * (src_valid > 0).float()

    if source_static_mask is None and static_mask is not None:
        source_static_mask = static_mask
    if source_static_mask is not None:
        source_static = source_static_mask.to(device=device).bool()
        if source_static.shape != src_valid.shape:
            raise ValueError(
                f"source_static_mask shape {tuple(source_static.shape)} != "
                f"source shape {tuple(src_valid.shape)}"
            )
        src_valid_mf = src_valid * source_static.float()
    else:
        src_valid_mf = src_valid

    H, W = image_h, image_w

    # --- Pre-compute world points for every source pixel ----------------
    # (B*T, H, W, 3) — used by both per-frame and multi-frame passes.
    src_rgb_flat = src_rgb.reshape(B * T, H, W, 3).permute(0, 3, 1, 2)  # (B*T, 3, H, W)
    src_depth_flat = src_depth.reshape(B * T, 1, H, W)                  # (B*T, 1, H, W)
    src_valid_flat = (src_valid > 0).reshape(B * T, 1, H, W).float()    # (B*T, 1, H, W)
    src_valid_mf_flat = (src_valid_mf > 0).reshape(B * T, 1, H, W).float()
    src_c2w_flat = src_c2w.reshape(B * T, 4, 4)
    K_src_flat = K.reshape(B * T, 3, 3)
    world_pts = _unproject_to_world(src_depth_flat, src_c2w_flat, K_src_flat, H, W)
    # world_pts: (B*T, H, W, 3)

    # `src_valid_mf` already encodes the source-side filter via
    # `source_static_mask` above (dynamic source pixels zeroed out before
    # we ever splat them). Together with the target-side `target_dynamic_mask`
    # gate in the compose step, the multi-frame fill is double-protected:
    # dynamic source pixels never enter the cloud, AND any residual
    # contamination at dynamic target pixels is overridden by the per-frame
    # warp.

    # Source-frame depth is consumed by `_unproject_to_world` above; from
    # here on the cloud carries RGB only and we splat target-camera z as the
    # 4th value channel below (one (B, Ct, N) z field per chunk for the
    # multi-frame pass, plus a (BCt, HW) z_pf for the per-frame pass). This
    # makes `out_depth` actually mean "target-camera depth at the warped
    # pixel" and matches the legacy z-buffer builder's contract — so the
    # dense control unit's `_normalize_warped_depth_per_frame` sees the same
    # quantity under both `--dense_warp_mode zbuffer` and
    # `--dense_warp_mode hybrid_softsplat`.


    # ------------------------------------------------------------------
    # Cloud-based fast path (2026-04-28). Builds a static source cache once
    # per sample (selected valid + static-filtered source keyframes lifted to
    # world space) and projects+splats that cache into each target chunk. By
    # default the cache is 7 camera-coverage keyframes and the point splat uses
    # Triton on CUDA. Set TRACK4D360_DENSE_HYBRID_CACHE_FRAMES=0/all to restore
    # the old all-frame cache for ablations.
    # ------------------------------------------------------------------
    chunk_env = os.environ.get("TRACK4D360_DENSE_HYBRID_TGT_CHUNK")
    # Default 4 keeps projection tensors bounded. With the default source-cache
    # keyframe selection, N is usually 7*H*W rather than T*H*W.
    chunk_size = int(chunk_env) if chunk_env else 4
    chunk_size = max(1, min(chunk_size, int(T)))

    # Build the static world cloud once.
    # cloud_world: (B, N, 3) where N = T_cache · H · W
    # cloud_rgb:   (B, N, 3) — source RGB only; target-camera z is splatted
    #   as the 4th value channel per chunk below.
    # cloud_valid_mf / cloud_valid_pf: (B, N) — multi-frame (static-only)
    #   and per-frame (all valid) variants.
    # We keep the (T_src, H, W) layout as a "frame_offset" lookup so the
    # per-frame pass can index source[t] cheaply.
    src_rgb_BTHW = src_rgb_flat.reshape(B, T, 3, H, W)
    world_pts_BTHW = world_pts.reshape(B, T, H, W, 3)
    src_valid_mf_BT = src_valid_mf_flat.reshape(B, T, 1, H, W)
    src_valid_BT = src_valid_flat.reshape(B, T, 1, H, W)

    cache_idx = select_dense_hybrid_cache_frame_indices(T, device=device, cam2world=src_c2w)
    T_cache = int(cache_idx.numel())
    world_pts_cache = world_pts_BTHW.index_select(1, cache_idx)
    src_rgb_cache = src_rgb_BTHW.index_select(1, cache_idx)
    src_valid_mf_cache = src_valid_mf_BT.index_select(1, cache_idx)

    N = T_cache * H * W
    # (B, N, 3): flatten selected source keyframes over (T_cache, H, W).
    cloud_world = world_pts_cache.reshape(B, N, 3)
    # (B, N, 3): move channel dim to the end before flattening.
    cloud_rgb = src_rgb_cache.permute(0, 1, 3, 4, 2).reshape(B, N, 3)
    cloud_valid_mf = src_valid_mf_cache.reshape(B, N)
    tgt_w2c_full = torch.linalg.inv(tgt_c2w)                               # (B, T, 4, 4)

    dyn_cache = None
    if dynamic_tracks_world is not None and target_dynamic_mask is not None:
        dyn_cache = _prepare_dynamic_track_cache(
            dynamic_tracks_world=dynamic_tracks_world,
            dynamic_tracks_valid_mask=dynamic_tracks_valid_mask,
            dynamic_tracks_query_count=dynamic_tracks_query_count,
            source_rgb=src_rgb,
            source_static_mask=source_static_mask,
            src_cam2world=src_c2w,
            intrinsics=K,
            cache_idx=cache_idx,
            image_h=H,
            image_w=W,
            batch_size=B,
            num_frames=T,
            device=device,
            dtype=dtype,
        )

    out_rgb = torch.empty(B, T, 3, H, W, dtype=dtype, device=device)
    out_depth = torch.empty(B, T, H, W, dtype=dtype, device=device)
    out_valid = torch.empty(B, T, H, W, dtype=torch.bool, device=device)

    splat_backend = os.environ.get("TRACK4D360_DENSE_HYBRID_SPLAT_BACKEND", "triton").lower()
    splat_points = _bilinear_splat_points
    if splat_backend in {"triton", "auto"} and source_rgb.is_cuda:
        try:
            from .triton_point_splat import bilinear_splat_points_triton, triton_available
            if triton_available():
                splat_points = bilinear_splat_points_triton
        except Exception:
            if splat_backend == "triton":
                raise

    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)
        Ct = t1 - t0

        tgt_w2c_chunk = tgt_w2c_full[:, t0:t1]                             # (B, Ct, 4, 4)
        K_tgt_chunk = K_tgt_full[:, t0:t1]                                 # (B, Ct, 3, 3) — target K

        # ---- Multi-frame pass: project cloud to each of Ct targets ----
        # cam_pts: (B, Ct, N, 3) — transform cloud into target camera space.
        R = tgt_w2c_chunk[..., :3, :3]                                     # (B, Ct, 3, 3)
        t_vec = tgt_w2c_chunk[..., :3, 3]                                  # (B, Ct, 3)
        cam_pts = torch.einsum("bcij,bnj->bcni", R, cloud_world) + t_vec.unsqueeze(2)
        z_mf = cam_pts[..., 2]                                             # (B, Ct, N)

        fx = K_tgt_chunk[..., 0, 0].unsqueeze(-1)                          # (B, Ct, 1)
        fy = K_tgt_chunk[..., 1, 1].unsqueeze(-1)
        cx = K_tgt_chunk[..., 0, 2].unsqueeze(-1)
        cy = K_tgt_chunk[..., 1, 2].unsqueeze(-1)
        z_safe = z_mf.clamp(min=1e-7)
        u_mf = fx * cam_pts[..., 0] / z_safe + cx                          # (B, Ct, N)
        v_mf = fy * cam_pts[..., 1] / z_safe + cy
        front_mf = z_mf > 1e-6
        mask_mf = cloud_valid_mf.unsqueeze(1).expand(B, Ct, N) * front_mf.to(dtype)

        # Reshape to (B*Ct, N, ...) for the cloud splat. Pack target-camera
        # z as the 4th value channel so the splat returns target-frame depth
        # at each warped pixel (depth-weighted bilinear average of the
        # contributing source points' target-frame z values).
        BCt = B * Ct
        rgb_exp = cloud_rgb.unsqueeze(1).expand(B, Ct, N, 3)
        values_mf = torch.cat(
            [rgb_exp, z_mf.unsqueeze(-1)], dim=-1
        ).reshape(BCt, N, 4)
        uv_mf = torch.stack([u_mf, v_mf], dim=-1).reshape(BCt, N, 2)
        depth_mf = z_mf.reshape(BCt, N)
        mask_mf_flat = mask_mf.reshape(BCt, N)

        rgbd_mf_chunk, mask_mf_out = splat_points(
            values_mf, mask_mf_flat, depth_mf, uv_mf,
            image_h=H, image_w=W,
            depth_weight_scale=depth_weight_scale,
            clamp_output=False,
        )
        rgbd_mf_chunk = rgbd_mf_chunk.reshape(B, Ct, 4, H, W)
        mask_mf_chunk = mask_mf_out.reshape(B, Ct, 1, H, W)

        # ---- Per-frame pass: source[t] → target[t] (one frame each) ---
        # Uses the SAME cloud subset, but only the (t0..t1) source slices.
        per_frame_world = world_pts_BTHW[:, t0:t1].reshape(B * Ct, H * W, 3)
        per_frame_rgb   = src_rgb_BTHW[:, t0:t1].permute(0, 1, 3, 4, 2).reshape(B * Ct, H * W, 3)
        per_frame_valid = src_valid_BT[:, t0:t1].reshape(B * Ct, H * W)
        tgt_w2c_pf = tgt_w2c_chunk.reshape(B * Ct, 4, 4)
        K_tgt_pf   = K_tgt_chunk.reshape(B * Ct, 3, 3)
        cam_pf = torch.einsum(
            "bij,bnj->bni", tgt_w2c_pf[:, :3, :3], per_frame_world,
        ) + tgt_w2c_pf[:, :3, 3].unsqueeze(1)
        z_pf = cam_pf[..., 2]                                              # (BCt, HW)
        z_pf_safe = z_pf.clamp(min=1e-7)
        fx_pf = K_tgt_pf[:, 0, 0:1]
        fy_pf = K_tgt_pf[:, 1, 1:2]
        cx_pf = K_tgt_pf[:, 0, 2:3]
        cy_pf = K_tgt_pf[:, 1, 2:3]
        u_pf = fx_pf * cam_pf[..., 0] / z_pf_safe + cx_pf
        v_pf = fy_pf * cam_pf[..., 1] / z_pf_safe + cy_pf
        front_pf = z_pf > 1e-6
        mask_pf_flat = per_frame_valid * front_pf.to(dtype)
        uv_pf = torch.stack([u_pf, v_pf], dim=-1)

        # 4th channel = target-camera z so out_depth is target-frame depth.
        values_pf = torch.cat([per_frame_rgb, z_pf.unsqueeze(-1)], dim=-1)
        rgbd_pf_chunk, mask_pf_out = splat_points(
            values_pf, mask_pf_flat, z_pf, uv_pf,
            image_h=H, image_w=W,
            depth_weight_scale=depth_weight_scale,
            clamp_output=False,
        )
        rgbd_pf_chunk = rgbd_pf_chunk.reshape(B, Ct, 4, H, W)
        mask_pf_chunk = mask_pf_out.reshape(B, Ct, 1, H, W)

        rgbd_dyn_chunk = None
        mask_dyn_chunk = None
        if dyn_cache is not None:
            rgbd_dyn_chunk, mask_dyn_chunk = _splat_dynamic_track_chunk(
                dyn_cache=dyn_cache,
                t0=t0,
                t1=t1,
                tgt_w2c_chunk=tgt_w2c_chunk,
                K_tgt_chunk=K_tgt_chunk,
                splat_points=splat_points,
                image_h=H,
                image_w=W,
                depth_weight_scale=depth_weight_scale,
                dtype=dtype,
            )

        # ---- Compose (vectorised across the chunk) -----------------------
        mf_has = mask_mf_chunk > 0
        pf_has = mask_pf_chunk > 0
        if target_dynamic_mask is not None:
            td_chunk = target_dynamic_mask[:, t0:t1].to(device=device).bool().unsqueeze(2)  # (B, Ct, 1, H, W)
        else:
            td_chunk = torch.zeros_like(mf_has)
        if rgbd_dyn_chunk is not None and mask_dyn_chunk is not None:
            dyn_has = mask_dyn_chunk > 0
            use_dyn = td_chunk & dyn_has
            use_pf = (td_chunk & ~dyn_has) | (~td_chunk & ~mf_has)
            rgbd_chunk = torch.where(
                use_dyn.expand_as(rgbd_mf_chunk),
                rgbd_dyn_chunk,
                torch.where(use_pf.expand_as(rgbd_mf_chunk), rgbd_pf_chunk, rgbd_mf_chunk),
            )
            valid_chunk = torch.where(use_dyn, dyn_has, torch.where(use_pf, pf_has, mf_has))
        else:
            use_pf = td_chunk | (~mf_has)
            rgbd_chunk = torch.where(use_pf.expand_as(rgbd_mf_chunk), rgbd_pf_chunk, rgbd_mf_chunk)
            valid_chunk = torch.where(use_pf, pf_has, mf_has)

        out_rgb[:, t0:t1] = torch.clamp(rgbd_chunk[:, :, :3], 0.0, 1.0)
        out_depth[:, t0:t1] = rgbd_chunk[:, :, 3]
        out_valid[:, t0:t1] = valid_chunk.squeeze(2)

    # warped_rgb in (B, T, H, W, 3) layout to match the z-buffer builder.
    warped_rgb = out_rgb.permute(0, 1, 3, 4, 2).contiguous()
    return warped_rgb, out_depth, out_valid


def _coerce_dynamic_tracks(
    dynamic_tracks_world: torch.Tensor,
    dynamic_tracks_valid_mask: "torch.Tensor | None",
    *,
    batch_size: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    query_count=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tracks = dynamic_tracks_world if torch.is_tensor(dynamic_tracks_world) else torch.as_tensor(dynamic_tracks_world)
    tracks = tracks.to(device=device, dtype=dtype)
    if tracks.ndim == 3:
        tracks = tracks.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if tracks.ndim != 4 or tracks.shape[0] != batch_size or tracks.shape[2] != num_frames or tracks.shape[-1] != 3:
        raise ValueError(
            "dynamic_tracks_world must have shape (N,T,3) or (B,N,T,3), "
            f"got {tuple(tracks.shape)} for B={batch_size}, T={num_frames}"
        )
    if dynamic_tracks_valid_mask is None:
        valid = torch.isfinite(tracks).all(dim=-1) & (tracks.abs().sum(dim=-1) > 1.0e-8)
    else:
        valid = dynamic_tracks_valid_mask if torch.is_tensor(dynamic_tracks_valid_mask) else torch.as_tensor(dynamic_tracks_valid_mask)
        valid = valid.to(device=device).bool()
        if valid.ndim == 2:
            valid = valid.unsqueeze(0).expand(batch_size, -1, -1)
        if valid.shape != tracks.shape[:3]:
            raise ValueError(
                f"dynamic_tracks_valid_mask shape {tuple(valid.shape)} != tracks shape {tuple(tracks.shape[:3])}"
            )
    tracks, valid = _maybe_compact_dynamic_tracks_for_hybrid_cache(tracks, valid, query_count=query_count)
    return tracks, valid


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def _maybe_compact_dynamic_tracks_for_hybrid_cache(
    tracks: torch.Tensor,
    valid: torch.Tensor,
    *,
    query_count=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact/pad dynamic tracks before dense hybrid fusion.

    This is disabled by default for backward compatibility. When enabled via
    ``TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_COMPACT=1`` or a positive
    ``TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_FIXED_N``, identities with no valid
    timestep under the selected mask are removed before source/target
    projection. With ``FIXED_N>0`` every batch is then deterministically
    padded/truncated to that identity count, making q/s multi-query fusion
    workspace shape stable.
    """
    fixed_n = max(_env_int("TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_FIXED_N", 0), 0)
    min_query = max(_env_int("TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_FIXED_N_MIN_QUERY", 2), 1)
    if fixed_n > 0 and _dynamic_track_query_count(query_count) < min_query:
        fixed_n = 0
    if fixed_n <= 0 and not _env_flag("TRACK4D360_DENSE_HYBRID_DYNAMIC_TRACK_COMPACT", False):
        return tracks, valid

    B, _N, T, C = tracks.shape
    keep_masks = valid.any(dim=-1)
    keep_counts = keep_masks.sum(dim=1)
    if fixed_n > 0:
        out_n = int(fixed_n)
    else:
        out_n = int(keep_counts.max().item()) if keep_counts.numel() else 0
        out_n = max(out_n, 1)

    compact_tracks = tracks.new_zeros((B, out_n, T, C))
    compact_valid = torch.zeros((B, out_n, T), dtype=torch.bool, device=valid.device)
    for b in range(B):
        idx = torch.nonzero(keep_masks[b], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if idx.numel() > out_n:
            # Deterministic uniform subsample preserves temporal/query coverage
            # without introducing rank-local randomness into workspace shape.
            pos = torch.linspace(0, idx.numel() - 1, out_n, device=idx.device)
            idx = idx[pos.round().long()]
        n = min(int(idx.numel()), out_n)
        if n > 0:
            idx_n = idx[:n]
            compact_tracks[b, :n] = tracks[b, idx_n]
            compact_valid[b, :n] = valid[b, idx_n]
    return compact_tracks, compact_valid


def _dynamic_track_query_count(query_count) -> int:
    if query_count is None:
        return 1
    if torch.is_tensor(query_count):
        if query_count.numel() == 0:
            return 1
        return max(int(query_count.detach().flatten().max().item()), 1)
    if isinstance(query_count, (list, tuple)):
        if not query_count:
            return 1
        return max(int(max(query_count)), 1)
    return max(int(query_count), 1)


def _project_track_points(
    points_world: torch.Tensor,   # (B, S, N, 3) or (B, C, N, 3)
    cam2world: torch.Tensor,      # (B, S, 4, 4) or (B, C, 4, 4)
    intrinsics: torch.Tensor,     # (B, S, 3, 3) or (B, C, 3, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w2c = torch.linalg.inv(cam2world)
    cam = (
        torch.einsum("bsij,bsnj->bsni", w2c[..., :3, :3], points_world)
        + w2c[..., :3, 3].unsqueeze(2)
    )
    z = cam[..., 2]
    z_safe = z.clamp(min=1.0e-7)
    fx = intrinsics[..., 0, 0].unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)
    u = fx * cam[..., 0] / z_safe + cx
    v = fy * cam[..., 1] / z_safe + cy
    return torch.stack([u, v], dim=-1), z, z > 1.0e-6


def _sample_track_values_from_source(
    source_rgb: torch.Tensor,      # (B, T, H, W, 3)
    source_static_mask: "torch.Tensor | None",
    uv: torch.Tensor,              # (B, S, N, 2)
    cache_idx: torch.Tensor,
    image_h: int,
    image_w: int,
) -> tuple[torch.Tensor, "torch.Tensor | None"]:
    B, S, N, _ = uv.shape
    frames = source_rgb.index_select(1, cache_idx).permute(0, 1, 4, 2, 3).reshape(B * S, 3, image_h, image_w)
    x = (uv[..., 0] / max(image_w - 1, 1)) * 2.0 - 1.0
    y = (uv[..., 1] / max(image_h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).reshape(B * S, N, 1, 2)
    sampled = F.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(B, S, N, 3)

    sampled_dynamic = None
    if source_static_mask is not None:
        dyn = (~source_static_mask.to(device=source_rgb.device).bool()).float()
        dyn_frames = dyn.index_select(1, cache_idx).reshape(B * S, 1, image_h, image_w)
        sampled_dynamic = F.grid_sample(
            dyn_frames, grid, mode="nearest", padding_mode="zeros", align_corners=True
        ).squeeze(1).squeeze(-1).reshape(B, S, N) > 0.5
    return sampled, sampled_dynamic


def _prepare_dynamic_track_cache(
    *,
    dynamic_tracks_world: torch.Tensor,
    dynamic_tracks_valid_mask: "torch.Tensor | None",
    dynamic_tracks_query_count=None,
    source_rgb: torch.Tensor,
    source_static_mask: "torch.Tensor | None",
    src_cam2world: torch.Tensor,
    intrinsics: torch.Tensor,
    cache_idx: torch.Tensor,
    image_h: int,
    image_w: int,
    batch_size: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    tracks, valid = _coerce_dynamic_tracks(
        dynamic_tracks_world,
        dynamic_tracks_valid_mask,
        batch_size=batch_size,
        num_frames=num_frames,
        device=device,
        dtype=dtype,
        query_count=dynamic_tracks_query_count,
    )
    tracks_src = tracks.index_select(2, cache_idx).permute(0, 2, 1, 3).contiguous()  # (B, S, N, 3)
    valid_src = valid.index_select(2, cache_idx).permute(0, 2, 1).contiguous()
    src_c2w_cache = src_cam2world.index_select(1, cache_idx)
    K_cache = intrinsics.index_select(1, cache_idx)
    src_uv, src_z, src_front = _project_track_points(tracks_src, src_c2w_cache, K_cache)
    src_in_bounds = (
        (src_uv[..., 0] >= 0.0) & (src_uv[..., 0] <= float(image_w - 1))
        & (src_uv[..., 1] >= 0.0) & (src_uv[..., 1] <= float(image_h - 1))
    )
    colors, sampled_dynamic = _sample_track_values_from_source(
        source_rgb, source_static_mask, src_uv, cache_idx, image_h, image_w
    )
    source_ok = valid_src & src_front & src_in_bounds & (src_z > 1.0e-6)
    if sampled_dynamic is not None:
        source_ok = source_ok & sampled_dynamic
    return {
        "tracks": tracks,
        "valid": valid,
        "cache_colors": colors,
        "cache_source_ok": source_ok,
    }


def _splat_dynamic_track_chunk(
    *,
    dyn_cache: dict,
    t0: int,
    t1: int,
    tgt_w2c_chunk: torch.Tensor,
    K_tgt_chunk: torch.Tensor,
    splat_points,
    image_h: int,
    image_w: int,
    depth_weight_scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    tracks = dyn_cache["tracks"]
    valid = dyn_cache["valid"]
    colors = dyn_cache["cache_colors"]
    source_ok = dyn_cache["cache_source_ok"]
    B, S, N, _ = colors.shape
    Ct = int(t1 - t0)

    target_tracks = tracks[:, :, t0:t1, :].permute(0, 2, 1, 3).contiguous()
    target_valid = valid[:, :, t0:t1].permute(0, 2, 1).contiguous()
    tgt_c2w_chunk = torch.linalg.inv(tgt_w2c_chunk)
    tgt_uv, tgt_z, tgt_front = _project_track_points(target_tracks, tgt_c2w_chunk, K_tgt_chunk)
    tgt_in_bounds = (
        (tgt_uv[..., 0] >= 0.0) & (tgt_uv[..., 0] <= float(image_w - 1))
        & (tgt_uv[..., 1] >= 0.0) & (tgt_uv[..., 1] <= float(image_h - 1))
    )
    target_ok = target_valid & tgt_front & tgt_in_bounds & (tgt_z > 1.0e-6)

    BCt = B * Ct
    values_rgb = colors.unsqueeze(1).expand(B, Ct, S, N, 3)
    target_z = tgt_z.unsqueeze(2).expand(B, Ct, S, N)
    values = torch.cat([values_rgb, target_z.unsqueeze(-1)], dim=-1).reshape(BCt, S * N, 4)
    uv = tgt_uv.unsqueeze(2).expand(B, Ct, S, N, 2).reshape(BCt, S * N, 2)
    depth = target_z.reshape(BCt, S * N)
    mask = (
        source_ok.unsqueeze(1).expand(B, Ct, S, N)
        & target_ok.unsqueeze(2).expand(B, Ct, S, N)
    ).reshape(BCt, S * N).to(dtype)

    rgbd_dyn, mask_dyn = splat_points(
        values,
        mask,
        depth,
        uv,
        image_h=image_h,
        image_w=image_w,
        depth_weight_scale=depth_weight_scale,
        clamp_output=False,
    )
    return (
        rgbd_dyn.reshape(B, Ct, 4, image_h, image_w),
        mask_dyn.reshape(B, Ct, 1, image_h, image_w),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _broadcast_cameras(
    c2w: torch.Tensor,
    B: int,
    T: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast camera matrices to (B, T, 4, 4).

    Accepts (B, T, 4, 4), (T, 4, 4), or (4, 4).
    """
    c2w = c2w.to(device=device, dtype=dtype)
    if c2w.ndim == 2:
        # (4, 4) → (B, T, 4, 4)
        c2w = c2w.unsqueeze(0).unsqueeze(0).expand(B, T, 4, 4)
    elif c2w.ndim == 3:
        # (T, 4, 4) → (B, T, 4, 4)
        c2w = c2w.unsqueeze(0).expand(B, T, 4, 4)
    # already (B, T, 4, 4) — contiguous copy to allow in-place ops downstream
    return c2w.contiguous()


def _broadcast_intrinsics(
    K: torch.Tensor,
    B: int,
    T: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast intrinsics to (B, T, 3, 3).

    Accepts (3, 3), (T, 3, 3), or (B, T, 3, 3).
    """
    K = K.to(device=device, dtype=dtype)
    if K.ndim == 2:
        K = K.unsqueeze(0).unsqueeze(0).expand(B, T, 3, 3)
    elif K.ndim == 3:
        K = K.unsqueeze(0).expand(B, T, 3, 3)
    return K.contiguous()


def _batch_inv44(m: torch.Tensor) -> torch.Tensor:
    """Batch-invert (B, T, 4, 4) rigid transforms.

    For SE(3) matrices (R | t; 0 | 1):
        inv = [R^T | -R^T t; 0 | 1]

    Falls back to ``torch.linalg.inv`` for non-rigid inputs, but the
    SE(3) path is faster and more numerically stable.
    """
    B, T, _, _ = m.shape
    R = m[..., :3, :3]   # (B, T, 3, 3)
    t = m[..., :3, 3]    # (B, T, 3)

    Rt = R.transpose(-1, -2)                         # (B, T, 3, 3)
    neg_Rt_t = -(Rt @ t.unsqueeze(-1)).squeeze(-1)   # (B, T, 3)

    inv = torch.zeros_like(m)
    inv[..., :3, :3] = Rt
    inv[..., :3, 3]  = neg_Rt_t
    inv[..., 3, 3]   = 1.0
    return inv
