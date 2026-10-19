"""
soft_splat_warp.py — Bilinear splatting forward warp for RGB appearance.

Adapted from Lyra-2's forward_warp_utils_pytorch.py (bilinear_splatting).
Unlike the z-buffer warp in dense_target_reprojection_builder.py which uses
hard nearest-depth compositing, this produces smoother warped RGB with:
  - bilinear weight distribution to 4 neighboring target pixels
  - depth-weighted blending (closer pixels contribute more)
  - fewer holes and less aliasing

This is intended for **warped RGB appearance conditioning** (Lyra-2's Path A).
For geometry-aware dense control, the z-buffer warp remains the default.

Interface matches build_dense_target_reprojection() for easy swapping.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def build_soft_splat_warped_rgb(
    source_rgb: torch.Tensor,           # (B, T, H, W, 3)   float32 [0, 1]
    source_depth: torch.Tensor,         # (B, T, H, W)       float32, metric depth
    source_depth_valid: torch.Tensor,   # (B, T, H, W)       bool
    src_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    tgt_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    intrinsics: torch.Tensor,           # source K: (3, 3) or (T, 3, 3) or (B, T, 3, 3)
    image_h: int,
    image_w: int,
    *,
    target_intrinsics: Optional[torch.Tensor] = None,
    depth_weight_scale: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Soft-splat forward warp of source RGB into the target camera frame.

    Returns:
        warped_rgb:   (B, T, 3, H, W)  float, warped source appearance in target view
        warp_valid:   (B, T, 1, H, W)  float, per-pixel validity mask (>0 where splat landed)
        warp_weights: (B, T, 1, H, W)  float, accumulated splat weight (usable as confidence)
    """
    device = source_rgb.device
    dtype = torch.float32  # geometry must be float32

    # Normalize shapes
    B, T = source_rgb.shape[:2]
    src_c2w = _ensure_batch(src_cam2world, B).to(device=device, dtype=dtype)   # (B, T, 4, 4)
    tgt_c2w = _ensure_batch(tgt_cam2world, B).to(device=device, dtype=dtype)   # (B, T, 4, 4)
    K = _ensure_intrinsics(intrinsics, B, T).to(device=device, dtype=dtype)    # (B, T, 3, 3)
    K_tgt = (
        _ensure_intrinsics(target_intrinsics, B, T).to(device=device, dtype=dtype)
        if target_intrinsics is not None else K
    )

    src_rgb = source_rgb.to(device=device, dtype=dtype)        # (B, T, H, W, 3)
    src_depth = source_depth.to(device=device, dtype=dtype)    # (B, T, H, W)
    src_valid = source_depth_valid.to(device=device)           # (B, T, H, W) bool

    # Clamp and mask depth
    src_depth = torch.clamp(src_depth, min=0, max=1e4)
    src_depth = src_depth * src_valid.float()

    # Compute target w2c from target c2w
    tgt_w2c = torch.linalg.inv(tgt_c2w)  # (B, T, 4, 4)

    # Process all (B, T) frames in a flat batch
    BT = B * T
    src_rgb_flat = src_rgb.reshape(BT, image_h, image_w, 3).permute(0, 3, 1, 2)  # (BT, 3, H, W)
    src_depth_flat = src_depth.reshape(BT, 1, image_h, image_w)                   # (BT, 1, H, W)
    src_valid_flat = src_valid.reshape(BT, 1, image_h, image_w).float()           # (BT, 1, H, W)
    src_c2w_flat = src_c2w.reshape(BT, 4, 4)
    tgt_w2c_flat = tgt_w2c.reshape(BT, 4, 4)
    K_flat = K.reshape(BT, 3, 3)
    K_tgt_flat = K_tgt.reshape(BT, 3, 3)

    # Step 1: Back-project source pixels to 3D world points
    world_pts = _unproject_to_world(src_depth_flat, src_c2w_flat, K_flat, image_h, image_w)
    # world_pts: (BT, H, W, 3)

    # Step 2: Project world points to target pixel coordinates
    tgt_pts = _project_to_target(world_pts, tgt_w2c_flat, K_tgt_flat)
    # tgt_pts: (BT, H, W, 3, 1) — [u, v, z] in target image

    # Mask: valid source + positive target depth
    tgt_z = tgt_pts[:, :, :, 2, 0]  # (BT, H, W)
    mask = src_valid_flat * (tgt_z.unsqueeze(1) > 0).float()  # (BT, 1, H, W)

    # Compute target UV and flow
    tgt_uv = tgt_pts[:, :, :, :2, 0] / (tgt_pts[:, :, :, 2:3, 0] + 1e-7)  # (BT, H, W, 2)
    tgt_uv = tgt_uv.permute(0, 3, 1, 2)  # (BT, 2, H, W)

    grid = _create_grid(BT, image_h, image_w, device=device, dtype=dtype)
    flow = tgt_uv - grid  # (BT, 2, H, W)

    tgt_depth = tgt_z.unsqueeze(1)  # (BT, 1, H, W)

    # Step 3: Bilinear splatting
    warped_rgb_flat, warped_mask_flat, warped_weights_flat = _bilinear_splat(
        src_rgb_flat, mask, tgt_depth, flow,
        depth_weight_scale=depth_weight_scale,
    )
    # warped_rgb_flat: (BT, 3, H, W), warped_mask_flat: (BT, 1, H, W)

    # Reshape back to (B, T, C, H, W)
    warped_rgb = warped_rgb_flat.reshape(B, T, 3, image_h, image_w)
    warp_valid = warped_mask_flat.reshape(B, T, 1, image_h, image_w)
    warp_weights = warped_weights_flat.reshape(B, T, 1, image_h, image_w)

    return warped_rgb, warp_valid, warp_weights


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_batch(cam: torch.Tensor, B: int) -> torch.Tensor:
    """Ensure camera tensor has batch dimension (B, T, 4, 4)."""
    if cam.dim() == 3:  # (T, 4, 4)
        return cam.unsqueeze(0).expand(B, -1, -1, -1)
    return cam


def _ensure_intrinsics(K: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """Normalize intrinsics to (B, T, 3, 3)."""
    if K.dim() == 2:  # (3, 3)
        return K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    elif K.dim() == 3:  # (T, 3, 3)
        return K.unsqueeze(0).expand(B, -1, -1, -1)
    return K  # already (B, T, 3, 3)


def _create_grid(b: int, h: int, w: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Create dense (x, y) pixel grid: (b, 2, h, w)."""
    x = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    y = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([x, y], dim=1)


def _unproject_to_world(
    depth: torch.Tensor,   # (B, 1, H, W)
    c2w: torch.Tensor,     # (B, 4, 4)
    K: torch.Tensor,       # (B, 3, 3)
    H: int, W: int,
) -> torch.Tensor:
    """Back-project depth to 3D world points. Returns (B, H, W, 3)."""
    B = depth.shape[0]
    device = depth.device
    dtype = depth.dtype

    # Pixel grid
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    ones = torch.ones_like(uu)
    pixel_coords = torch.stack([uu, vv, ones], dim=-1)  # (H, W, 3)
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)

    # Unproject: K_inv @ pixel_coords * depth
    K_inv = torch.linalg.inv(K)  # (B, 3, 3)
    pixel_flat = pixel_coords.reshape(B, H * W, 3).transpose(1, 2)  # (B, 3, HW)
    cam_pts = K_inv @ pixel_flat  # (B, 3, HW)
    cam_pts = cam_pts.transpose(1, 2).reshape(B, H, W, 3)  # (B, H, W, 3)

    d = depth[:, 0, :, :].unsqueeze(-1)  # (B, H, W, 1)
    cam_pts = cam_pts * d  # (B, H, W, 3)

    # Camera to world: R @ pts + t
    R = c2w[:, :3, :3]  # (B, 3, 3)
    t = c2w[:, :3, 3]   # (B, 3)
    world_pts = torch.einsum('bij,bhwj->bhwi', R, cam_pts) + t[:, None, None, :]
    return world_pts  # (B, H, W, 3)


def _project_to_target(
    world_pts: torch.Tensor,  # (B, H, W, 3)
    w2c: torch.Tensor,        # (B, 4, 4)
    K: torch.Tensor,          # (B, 3, 3)
) -> torch.Tensor:
    """Project world points to target image coordinates. Returns (B, H, W, 3, 1)."""
    B, H, W, _ = world_pts.shape

    # World to camera
    ones = torch.ones(B, H, W, 1, device=world_pts.device, dtype=world_pts.dtype)
    pts_homo = torch.cat([world_pts, ones], dim=-1).unsqueeze(-1)  # (B, H, W, 4, 1)
    w2c_4d = w2c[:, None, None]  # (B, 1, 1, 4, 4)
    cam_pts = (w2c_4d @ pts_homo)[:, :, :, :3]  # (B, H, W, 3, 1)

    # Camera to pixel
    K_4d = K[:, None, None]  # (B, 1, 1, 3, 3)
    projected = K_4d @ cam_pts  # (B, H, W, 3, 1)
    return projected


def _bilinear_splat_points(
    values: torch.Tensor,     # (B, N, C) values to splat
    mask: torch.Tensor,       # (B, N)    bool/float — 0 = drop point
    depth: torch.Tensor,      # (B, N)    target-camera depth (for depth weighting)
    uv: torch.Tensor,         # (B, N, 2) sub-pixel target coordinates [u, v]
    image_h: int,
    image_w: int,
    *,
    depth_weight_scale: float = 50.0,
    clamp_output: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Direct point-cloud variant of ``_bilinear_splat``.

    Unlike the grid-based ``_bilinear_splat`` (which expects (H, W)
    source layouts and uses the Lyra-2 ``n_views`` trick to fuse multi-
    view sources), this takes a flat ``(B, N, C)`` point cloud, projects
    each point to a sub-pixel target coordinate, and bilinear-splats
    onto a ``(B, image_h+2, image_w+2, C)`` padded canvas. No n_views
    blowup → no per-source-frame canvas replication → ~T× lower peak
    memory and ~T× fewer scatter writes for the dense-warp's multi-frame
    pass when the cloud is built once per sample (see
    ``dense_target_reprojection_builder.build_dense_target_reprojection_hybrid``).

    Returns ``(warped, mask_out)`` of shapes ``(B, C, H, W)`` and
    ``(B, 1, H, W)`` respectively.
    """
    B, N, C = values.shape
    H, W = image_h, image_w
    device = values.device
    dtype = values.dtype

    u_raw = uv[..., 0] + 1.0    # +1 for the padded canvas
    v_raw = uv[..., 1] + 1.0
    in_bounds = (u_raw >= 1.0) & (u_raw <= float(W)) & (v_raw >= 1.0) & (v_raw <= float(H))
    u = u_raw.clamp(0.0, float(W + 1))
    v = v_raw.clamp(0.0, float(H + 1))
    u_floor = torch.floor(u).long().clamp_(0, W + 1)
    v_floor = torch.floor(v).long().clamp_(0, H + 1)
    u_ceil = (u_floor + 1).clamp_(0, W + 1)
    v_ceil = (v_floor + 1).clamp_(0, H + 1)

    du = u - u_floor.to(dtype)
    dv = v - v_floor.to(dtype)
    prox_nw = (1.0 - dv) * (1.0 - du)
    prox_sw = dv * (1.0 - du)
    prox_ne = (1.0 - dv) * du
    prox_se = dv * du

    z = torch.clamp(depth, min=0)
    log_z = torch.log1p(z)
    # Per-batch normalisation of log_z for numerical stability.
    log_z_max = log_z.amax(dim=1, keepdim=True).clamp_min(1e-7)
    exponent = (log_z / log_z_max) * float(depth_weight_scale)
    exponent = exponent.clamp_max(80.0)
    depth_w = torch.exp(exponent) + 1e-7
    inv_dw = (1.0 / depth_w) * mask.to(dtype) * in_bounds.to(dtype)

    w_nw = prox_nw * inv_dw
    w_sw = prox_sw * inv_dw
    w_ne = prox_ne * inv_dw
    w_se = prox_se * inv_dw

    canvas = torch.zeros(B, H + 2, W + 2, C, dtype=dtype, device=device)
    weights = torch.zeros(B, H + 2, W + 2, 1, dtype=dtype, device=device)
    batch_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, N)

    for v_idx, u_idx, w in (
        (v_floor, u_floor, w_nw),
        (v_ceil,  u_floor, w_sw),
        (v_floor, u_ceil,  w_ne),
        (v_ceil,  u_ceil,  w_se),
    ):
        canvas.index_put_((batch_idx, v_idx, u_idx), values * w.unsqueeze(-1), accumulate=True)
        weights.index_put_((batch_idx, v_idx, u_idx), w.unsqueeze(-1), accumulate=True)

    canvas_chw = canvas[:, 1:-1, 1:-1, :].permute(0, 3, 1, 2)   # (B, C, H, W)
    weights_chw = weights[:, 1:-1, 1:-1, :].permute(0, 3, 1, 2) # (B, 1, H, W)
    valid_mask = weights_chw > 0
    out = torch.where(valid_mask, canvas_chw / weights_chw,
                      torch.zeros_like(canvas_chw))
    if clamp_output:
        out = torch.clamp(out, 0.0, 1.0)
    return out, valid_mask.to(dtype)


def _bilinear_splat(
    frame: torch.Tensor,       # (B*n_views, C, H, W)
    mask: torch.Tensor,        # (B*n_views, 1, H, W)
    depth: torch.Tensor,       # (B*n_views, 1, H, W)
    flow: torch.Tensor,        # (B*n_views, 2, H, W)
    *,
    depth_weight_scale: float = 50.0,
    n_views: int = 1,
    clamp_output: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bilinear splatting: scatter source pixels to target with bilinear weights.
    Depth-weighted: closer pixels contribute more.

    Adapted from Lyra-2 bilinear_splatting().

    Returns:
        warped_frame: (B, C, H, W)
        warped_mask:  (B, 1, H, W) — 1 where any splat landed
        raw_weights:  (B, 1, H, W) — accumulated splat weights (confidence proxy)
    """
    B, C, H, W = frame.shape
    device = frame.device
    dtype = frame.dtype

    grid = _create_grid(B, H, W, device=device, dtype=dtype)
    trans_pos = flow + grid  # target pixel coordinates (B, 2, H, W)

    # Offset by 1 for the padded canvas
    trans_pos_offset = trans_pos + 1
    trans_pos_floor = torch.floor(trans_pos_offset).long()
    trans_pos_ceil = torch.ceil(trans_pos_offset).long()

    # Clamp to canvas bounds
    trans_pos_offset = torch.stack([
        torch.clamp(trans_pos_offset[:, 0], 0, W + 1),
        torch.clamp(trans_pos_offset[:, 1], 0, H + 1),
    ], dim=1)
    trans_pos_floor = torch.stack([
        torch.clamp(trans_pos_floor[:, 0], 0, W + 1),
        torch.clamp(trans_pos_floor[:, 1], 0, H + 1),
    ], dim=1)
    trans_pos_ceil = torch.stack([
        torch.clamp(trans_pos_ceil[:, 0], 0, W + 1),
        torch.clamp(trans_pos_ceil[:, 1], 0, H + 1),
    ], dim=1)

    # Bilinear proximity weights for 4 neighbors (NW, SW, NE, SE)
    prox_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
              (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
    prox_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
              (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
    prox_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
              (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
    prox_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
              (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

    # Depth weighting: closer pixels get higher weight
    clamped_depth = torch.clamp(depth, min=0)
    log_depth = torch.log1p(clamped_depth)
    max_log = log_depth.max()
    exponent = log_depth / (max_log + 1e-7) * depth_weight_scale
    exponent = torch.clamp(exponent, max=80.0)
    depth_weights = torch.exp(exponent) + 1e-7

    # Combined weight = proximity * mask / depth_weight (closer = higher weight)
    # Rearrange to (B, H, W, 1) for scatter
    def _weight(prox):
        return torch.moveaxis(prox * mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

    w_nw = _weight(prox_nw)
    w_sw = _weight(prox_sw)
    w_ne = _weight(prox_ne)
    w_se = _weight(prox_se)

    # Padded canvas for scatter (H+2, W+2 to handle boundary)
    canvas = torch.zeros(B, H + 2, W + 2, C, dtype=dtype, device=device)
    weight_canvas = torch.zeros(B, H + 2, W + 2, 1, dtype=dtype, device=device)

    frame_hwc = torch.moveaxis(frame, [0, 1, 2, 3], [0, 3, 1, 2])  # (B, H, W, C)
    batch_idx = torch.arange(B, device=device, dtype=torch.long)[:, None, None]

    # Scatter to 4 neighbors
    canvas.index_put_((batch_idx, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), frame_hwc * w_nw, accumulate=True)
    canvas.index_put_((batch_idx, trans_pos_ceil[:, 1],  trans_pos_floor[:, 0]), frame_hwc * w_sw, accumulate=True)
    canvas.index_put_((batch_idx, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),  frame_hwc * w_ne, accumulate=True)
    canvas.index_put_((batch_idx, trans_pos_ceil[:, 1],  trans_pos_ceil[:, 0]),  frame_hwc * w_se, accumulate=True)

    weight_canvas.index_put_((batch_idx, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), w_nw, accumulate=True)
    weight_canvas.index_put_((batch_idx, trans_pos_ceil[:, 1],  trans_pos_floor[:, 0]), w_sw, accumulate=True)
    weight_canvas.index_put_((batch_idx, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),  w_ne, accumulate=True)
    weight_canvas.index_put_((batch_idx, trans_pos_ceil[:, 1],  trans_pos_ceil[:, 0]),  w_se, accumulate=True)

    # Crop padding and normalize
    canvas_chw = torch.moveaxis(canvas, [0, 1, 2, 3], [0, 2, 3, 1])          # (B, C, H+2, W+2)
    weight_chw = torch.moveaxis(weight_canvas, [0, 1, 2, 3], [0, 2, 3, 1])   # (B, 1, H+2, W+2)

    # Multi-source aggregation (Lyra-2 / GEN3C `n_views` trick): when each
    # real batch entry was duplicated `n_views` times along the leading dim
    # to splat several source views into a single target canvas, sum across
    # the views axis before normalizing. The result is one canvas per real
    # batch entry, with contributions from every source view blended via
    # depth-weighted averaging.
    if n_views > 1:
        b_real = canvas_chw.shape[0] // n_views
        if b_real * n_views != canvas_chw.shape[0]:
            raise ValueError(
                f"_bilinear_splat: leading batch ({canvas_chw.shape[0]}) is not "
                f"divisible by n_views={n_views}."
            )
        canvas_chw = canvas_chw.reshape(b_real, n_views, *canvas_chw.shape[1:]).sum(dim=1)
        weight_chw = weight_chw.reshape(b_real, n_views, *weight_chw.shape[1:]).sum(dim=1)

    canvas_cropped = canvas_chw[:, :, 1:-1, 1:-1]   # (B, C, H, W)
    weight_cropped = weight_chw[:, :, 1:-1, 1:-1]   # (B, 1, H, W)

    valid_mask = weight_cropped > 0
    warped_frame = torch.where(valid_mask, canvas_cropped / weight_cropped, torch.zeros_like(canvas_cropped))
    if clamp_output:
        warped_frame = torch.clamp(warped_frame, 0, 1)
    warped_mask = valid_mask.float()

    return warped_frame, warped_mask, weight_cropped


# ---------------------------------------------------------------------------
# Multi-source soft-splat (Lyra-2 forward_warp_multiframes equivalent)
# ---------------------------------------------------------------------------

def build_multi_source_soft_splat_warped_rgb(
    source_rgb: torch.Tensor,           # (B, T, H, W, 3) float32 [0, 1]
    source_depth: torch.Tensor,         # (B, T, H, W)    float32 metric depth
    source_depth_valid: torch.Tensor,   # (B, T, H, W)    bool
    src_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    tgt_cam2world: torch.Tensor,        # (B, T, 4, 4) or (T, 4, 4)
    intrinsics: torch.Tensor,           # source K: (3, 3) or (T, 3, 3) or (B, T, 3, 3)
    image_h: int,
    image_w: int,
    *,
    target_intrinsics: Optional[torch.Tensor] = None,
    static_mask: Optional[torch.Tensor] = None,  # (B, T, H, W) bool — True = include
    depth_weight_scale: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each target frame ``t``, soft-splat ALL ``T`` source frames into
    target ``t`` (depth-weighted bilinear blend). Mirrors Lyra-2's
    ``forward_warp_multiframes`` pattern (``n_views=T``).

    When ``static_mask`` is provided, source pixels marked dynamic are
    dropped before splatting — the multi-frame fill is then pure-static and
    cannot smear moving objects across timestamps. For datasets without
    per-pixel motion data (RE10K / DL3DV / MVS-Synth), pass ``None``.

    Returns:
        warped_rgb:   (B, T, 3, H, W) float, multi-frame fused appearance
        warp_valid:   (B, T, 1, H, W) float, 1 where any splat landed
        warp_weights: (B, T, 1, H, W) float, accumulated weight (confidence)
    """
    device = source_rgb.device
    dtype = torch.float32

    B, T = source_rgb.shape[:2]
    src_c2w = _ensure_batch(src_cam2world, B).to(device=device, dtype=dtype)   # (B, T, 4, 4)
    tgt_c2w = _ensure_batch(tgt_cam2world, B).to(device=device, dtype=dtype)   # (B, T, 4, 4)
    K = _ensure_intrinsics(intrinsics, B, T).to(device=device, dtype=dtype)    # (B, T, 3, 3)
    K_tgt = (
        _ensure_intrinsics(target_intrinsics, B, T).to(device=device, dtype=dtype)
        if target_intrinsics is not None else K
    )

    src_rgb = source_rgb.to(device=device, dtype=dtype)        # (B, T, H, W, 3)
    src_depth = source_depth.to(device=device, dtype=dtype)    # (B, T, H, W)
    src_valid = source_depth_valid.to(device=device).float()   # (B, T, H, W)
    if static_mask is not None:
        sm = static_mask.to(device=device).float()             # (B, T, H, W)
        if sm.shape != src_valid.shape:
            raise ValueError(
                f"static_mask shape {tuple(sm.shape)} != source shape {tuple(src_valid.shape)}"
            )
        src_valid = src_valid * sm

    src_depth = torch.clamp(src_depth, min=0, max=1e4)
    src_depth = src_depth * (src_valid > 0).float()

    # Step 1: precompute world points for every source pixel — (B*T, H, W, 3).
    src_rgb_flat = src_rgb.reshape(B * T, image_h, image_w, 3).permute(0, 3, 1, 2)
    src_depth_flat = src_depth.reshape(B * T, 1, image_h, image_w)
    src_valid_flat = (src_valid > 0).reshape(B * T, 1, image_h, image_w).float()
    src_c2w_flat = src_c2w.reshape(B * T, 4, 4)
    K_flat = K.reshape(B * T, 3, 3)
    world_pts = _unproject_to_world(src_depth_flat, src_c2w_flat, K_flat, image_h, image_w)
    # world_pts: (B*T, H, W, 3)

    # Step 2: for each target frame t, project every world point into target[t]
    # and bilinear-splat with n_views=T (sum across the T source views).
    out_rgb = torch.empty(B, T, 3, image_h, image_w, dtype=dtype, device=device)
    out_mask = torch.empty(B, T, 1, image_h, image_w, dtype=dtype, device=device)
    out_w = torch.empty(B, T, 1, image_h, image_w, dtype=dtype, device=device)

    grid = _create_grid(B * T, image_h, image_w, device=device, dtype=dtype)  # (B*T, 2, H, W)

    for t in range(T):
        # Target c2w / K replicated across the T source views: (B*T, 4, 4).
        tgt_c2w_t = tgt_c2w[:, t]                                      # (B, 4, 4)
        tgt_w2c_t = torch.linalg.inv(tgt_c2w_t)                        # (B, 4, 4)
        tgt_w2c_rep = tgt_w2c_t.unsqueeze(1).expand(B, T, 4, 4).reshape(B * T, 4, 4)
        K_tgt_t = K_tgt[:, t]                                          # (B, 3, 3)
        K_tgt_rep = K_tgt_t.unsqueeze(1).expand(B, T, 3, 3).reshape(B * T, 3, 3)

        proj = _project_to_target(world_pts, tgt_w2c_rep, K_tgt_rep)   # (B*T, H, W, 3, 1)
        z = proj[:, :, :, 2, 0]                                        # (B*T, H, W)
        front = (z > 0).unsqueeze(1).float()                           # (B*T, 1, H, W)
        mask_t = src_valid_flat * front
        uv = proj[:, :, :, :2, 0] / (proj[:, :, :, 2:3, 0] + 1e-7)     # (B*T, H, W, 2)
        uv_chw = uv.permute(0, 3, 1, 2)
        flow = uv_chw - grid                                           # (B*T, 2, H, W)
        depth_t = z.unsqueeze(1)                                       # (B*T, 1, H, W)

        rgb_t, mask_out_t, w_t = _bilinear_splat(
            src_rgb_flat, mask_t, depth_t, flow,
            depth_weight_scale=depth_weight_scale,
            n_views=T,
        )
        # rgb_t / mask_out_t / w_t: (B, ..., H, W) post n_views aggregation.
        out_rgb[:, t] = rgb_t
        out_mask[:, t] = mask_out_t
        out_w[:, t] = w_t

    return out_rgb, out_mask, out_w


def compute_static_mask_from_tracks_world(
    tracks_world: torch.Tensor,      # (B, N=H*W, T, 3) cam0-normalised
    image_h: int,
    image_w: int,
    motion_threshold: float = 0.01,
) -> Optional[torch.Tensor]:
    """Per-pixel static mask from `tracks_world` motion analysis.

    `tracks_world` from ``Kubric_NVS`` / ``Syn4D`` carries per-pixel 3D
    trajectories anchored at frame 0 (every pixel of the static background
    has constant world-space position; dynamic objects move). Threshold the
    max-frame displacement to separate static from dynamic.

    The default 0.01 threshold matches the cam0-normalised scene scale (the
    dataset divides positions by ``common_scale`` so 1.0 ≈ scene depth, and
    1% of scene scale is a tight static cutoff that excludes Kubric balls).
    Stricter than the visualization helper's 0.05 — when training feeds the
    multi-frame fill into a model, false-positive static labels (= dynamic
    bleeding) hurt much more than false-negative ones.

    Returns ``(B, T, H, W)`` bool, broadcast across the T axis (the
    trajectory is a per-pixel property, not time-varying).  Returns ``None``
    if ``tracks_world`` is ``None`` or has an unexpected shape (caller falls
    back to "no mask = every pixel static = pure static dataset").
    """
    if tracks_world is None:
        return None
    if not torch.is_tensor(tracks_world):
        tracks_world = torch.as_tensor(tracks_world)
    if tracks_world.ndim == 3:
        tracks_world = tracks_world.unsqueeze(0)   # (1, N, T, 3)
    if tracks_world.ndim != 4 or tracks_world.shape[1] != image_h * image_w:
        return None
    B, _N, T, _ = tracks_world.shape
    delta = tracks_world - tracks_world[:, :, :1, :]            # (B, N, T, 3)
    motion = torch.linalg.norm(delta.float(), dim=-1).amax(dim=-1)   # (B, N)
    static_flat = motion < float(motion_threshold)               # (B, N)
    static_2d = static_flat.reshape(B, image_h, image_w)         # (B, H, W)
    return static_2d.unsqueeze(1).expand(B, T, image_h, image_w).contiguous()


def compute_projected_dynamic_mask_from_tracks_world(
    tracks_world: torch.Tensor,         # (B, N=H*W, T, 3) cam0-normalised
    cam2world: torch.Tensor,            # (B, T, 4, 4) cam0-normalised
    intrinsics: torch.Tensor,           # (B, T, 3, 3) pixel-space K
    image_h: int,
    image_w: int,
    motion_threshold: float = 0.01,
    splat_radius: int = 1,
    visible_depth: Optional[torch.Tensor] = None,       # optional (B,T,H,W) target/source z-buffer
    visible_depth_valid: Optional[torch.Tensor] = None, # optional (B,T,H,W) bool
    depth_tolerance: float = 0.03,
    depth_visibility_mode: str = "front",
) -> Optional[torch.Tensor]:
    """Project dynamic 3D track points into a camera sequence.

    The returned mask is in the pixel space of ``cam2world[:, t]``. It can be
    used either as a source-frame dynamic mask (exclude those pixels from the
    multi-frame static cloud) or as a target-frame dynamic mask (force those
    pixels to use the per-frame warp).
    """
    if tracks_world is None:
        return None
    if depth_visibility_mode not in ("front", "surface"):
        raise ValueError(
            "depth_visibility_mode must be 'front' or 'surface', "
            f"got {depth_visibility_mode!r}"
        )
    if not torch.is_tensor(tracks_world):
        tracks_world = torch.as_tensor(tracks_world)
    if tracks_world.ndim == 3:
        tracks_world = tracks_world.unsqueeze(0)
    if tracks_world.ndim != 4:
        raise ValueError(
            f"tracks_world must be (B,N,T,3) or (N,T,3); got {tuple(tracks_world.shape)}"
        )

    # Track-grid resolution may differ from the requested image grid (e.g. 4RC
    # predicts tracks at 280x504 while eval runs at 288x512). Auto-resample by
    # finding the (h_t, w_t) whose product == N and aspect closest to (image_h,
    # image_w), then bilinear-resize spatial dims to (image_h, image_w).
    image_grid_n = int(image_h) * int(image_w)
    # Multi-query tracking concatenates K dense H*W query grids along N. Those
    # tracks are already point samples, not a single image grid to be
    # bilinear-resized; project them directly. Keep the old resample path only
    # for genuine resolution mismatches such as 4RC 280x504 → eval 288x512.
    if tracks_world.shape[1] != image_grid_n and tracks_world.shape[1] % image_grid_n != 0:
        N = int(tracks_world.shape[1])
        target_aspect = float(image_h) / float(image_w)
        best = None
        for h_t in range(1, N + 1):
            if N % h_t != 0:
                continue
            w_t = N // h_t
            err = abs((h_t / w_t) - target_aspect)
            if best is None or err < best[0]:
                best = (err, h_t, w_t)
        if best is None or best[1] * best[2] != N:
            raise ValueError(
                f"compute_projected_dynamic_mask_from_tracks_world: cannot factor "
                f"tracks_world.shape[1]={N} into a 2D grid; pass tracks at "
                f"(image_h*image_w)={image_h * image_w}."
            )
        _, h_t, w_t = best
        if not getattr(compute_projected_dynamic_mask_from_tracks_world,
                       "_warned_resample", False):
            print(
                f"[soft_splat_warp] WARN: tracks grid {h_t}x{w_t} (N={N}) "
                f"!= image grid {image_h}x{image_w} (N={image_h*image_w}); "
                f"bilinear-resampling tracks_world to image grid. "
                f"This warning fires once per process."
            )
            compute_projected_dynamic_mask_from_tracks_world._warned_resample = True
        # tracks_world: (B, N, T, 3) -> (B, h_t, w_t, T, 3) -> (B, T, 3, h_t, w_t)
        # so F.interpolate (which expects spatial as last 2 dims) treats T*3 as channels.
        B0, _, T0, _ = tracks_world.shape
        tw = tracks_world.reshape(B0, h_t, w_t, T0, 3).permute(0, 3, 4, 1, 2).contiguous()
        tw = tw.reshape(B0 * T0, 3, h_t, w_t)
        tw = torch.nn.functional.interpolate(
            tw.float(), size=(image_h, image_w), mode="bilinear", align_corners=False
        )
        tw = tw.reshape(B0, T0, 3, image_h, image_w).permute(0, 3, 4, 1, 2).contiguous()
        tracks_world = tw.reshape(B0, image_h * image_w, T0, 3)

    device = tracks_world.device
    B, _N, T, _ = tracks_world.shape

    delta = tracks_world.float() - tracks_world[:, :, :1, :].float()
    motion = torch.linalg.norm(delta, dim=-1).amax(dim=-1)
    dyn_per_b = motion >= float(motion_threshold)

    if not torch.is_tensor(cam2world):
        cam2world = torch.as_tensor(cam2world)
    if not torch.is_tensor(intrinsics):
        intrinsics = torch.as_tensor(intrinsics)

    c2w = cam2world.to(device=device, dtype=torch.float32)
    if c2w.ndim == 3:
        c2w = c2w.unsqueeze(0).expand(B, -1, -1, -1)
    K = intrinsics.to(device=device, dtype=torch.float32)
    if K.ndim == 2:
        K = K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    elif K.ndim == 3:
        K = K.unsqueeze(0).expand(B, -1, -1, -1)

    depth_ref = None
    depth_ref_valid = None
    if visible_depth is not None:
        depth_ref = visible_depth.to(device=device, dtype=torch.float32)
        if depth_ref.ndim == 3:
            depth_ref = depth_ref.unsqueeze(0)
        if depth_ref.shape[0] == 1 and B > 1:
            depth_ref = depth_ref.expand(B, -1, -1, -1)
        if depth_ref.shape != (B, T, image_h, image_w):
            raise ValueError(
                f"visible_depth shape {tuple(depth_ref.shape)} != "
                f"(B,T,H,W)=({B},{T},{image_h},{image_w})"
            )
        if visible_depth_valid is None:
            depth_ref_valid = torch.isfinite(depth_ref) & (depth_ref > 1e-6)
        else:
            depth_ref_valid = visible_depth_valid.to(device=device).bool()
            if depth_ref_valid.ndim == 3:
                depth_ref_valid = depth_ref_valid.unsqueeze(0)
            if depth_ref_valid.shape[0] == 1 and B > 1:
                depth_ref_valid = depth_ref_valid.expand(B, -1, -1, -1)
            if depth_ref_valid.shape != (B, T, image_h, image_w):
                raise ValueError(
                    f"visible_depth_valid shape {tuple(depth_ref_valid.shape)} != "
                    f"(B,T,H,W)=({B},{T},{image_h},{image_w})"
                )

    out = torch.zeros((B, T, image_h, image_w), dtype=torch.bool, device=device)
    for b in range(B):
        dyn = dyn_per_b[b]
        if not bool(dyn.any()):
            continue
        pts_traj = tracks_world[b, dyn].float()
        for t in range(T):
            pt = pts_traj[:, t, :]
            w2c = torch.linalg.inv(c2w[b, t])
            R = w2c[:3, :3]
            tt = w2c[:3, 3]
            cam = pt @ R.T + tt
            z = cam[:, 2]
            front = z > 1e-6
            if not bool(front.any()):
                continue
            cam = cam[front]
            z = z[front]
            K_bt = K[b, t]
            u = (K_bt[0, 0] * cam[:, 0] / z + K_bt[0, 2]).round().long()
            v = (K_bt[1, 1] * cam[:, 1] / z + K_bt[1, 2]).round().long()
            for du in range(-splat_radius, splat_radius + 1):
                for dv in range(-splat_radius, splat_radius + 1):
                    ui = u + du
                    vi = v + dv
                    ok = (ui >= 0) & (ui < image_w) & (vi >= 0) & (vi < image_h)
                    if not bool(ok.any()):
                        continue
                    if depth_ref is not None:
                        ui_ok = ui[ok]
                        vi_ok = vi[ok]
                        z_ok = z[ok]
                        ref = depth_ref[b, t, vi_ok, ui_ok]
                        ref_valid = depth_ref_valid[b, t, vi_ok, ui_ok]
                        if depth_visibility_mode == "surface":
                            depth_ok = torch.abs(z_ok - ref) <= float(depth_tolerance)
                        else:
                            depth_ok = z_ok <= ref + float(depth_tolerance)
                        visible = ref_valid & torch.isfinite(ref) & depth_ok
                        if not bool(visible.any()):
                            continue
                        out[b, t, vi_ok[visible], ui_ok[visible]] = True
                    else:
                        out[b, t, vi[ok], ui[ok]] = True
    return out


def compute_target_dynamic_mask_from_tracks_world(
    tracks_world: torch.Tensor,
    tgt_cam2world: torch.Tensor,
    tgt_intrinsics: torch.Tensor,
    image_h: int,
    image_w: int,
    motion_threshold: float = 0.01,
    splat_radius: int = 1,
    target_depth: Optional[torch.Tensor] = None,
    target_depth_valid: Optional[torch.Tensor] = None,
    depth_tolerance: float = 0.03,
    depth_visibility_mode: str = "front",
) -> Optional[torch.Tensor]:
    """Backward-compatible target-space wrapper around
    :func:`compute_projected_dynamic_mask_from_tracks_world`.
    """
    return compute_projected_dynamic_mask_from_tracks_world(
        tracks_world=tracks_world,
        cam2world=tgt_cam2world,
        intrinsics=tgt_intrinsics,
        image_h=image_h,
        image_w=image_w,
        motion_threshold=motion_threshold,
        splat_radius=splat_radius,
        visible_depth=target_depth,
        visible_depth_valid=target_depth_valid,
        depth_tolerance=depth_tolerance,
        depth_visibility_mode=depth_visibility_mode,
    )


def build_hybrid_warped_rgb(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    source_depth_valid: torch.Tensor,
    src_cam2world: torch.Tensor,
    tgt_cam2world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_h: int,
    image_w: int,
    *,
    target_intrinsics: Optional[torch.Tensor] = None,
    static_mask: Optional[torch.Tensor] = None,   # (B, T, H, W) bool
    depth_weight_scale: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hybrid warp: multi-frame fused at static target pixels, per-frame at
    dynamic target pixels.

    Strategy:
      1. Per-frame baseline (``build_soft_splat_warped_rgb``) — covers every
         target pixel with content from source frame ``s = t`` only. Carries
         dynamic motion correctly (frame-t source matches frame-t target).
      2. Multi-frame fused (``build_multi_source_soft_splat_warped_rgb``)
         with the static-pixel filter — much richer fill, but only valid
         where the source was marked static.
      3. Combine: for each target pixel, prefer the multi-frame result if
         it has any contribution; otherwise fall back to the per-frame
         result. This means dynamic regions (where multi-frame has no fill
         due to filtering) automatically use the per-frame warp, while
         static regions enjoy the multi-frame coverage.

    For datasets without ``static_mask`` (RE10K / DL3DV / MVS-Synth),
    every pixel is static, the multi-frame fill covers everywhere, and the
    per-frame fallback is only used for pure occlusions.

    Returns ``(warped_rgb, warp_valid, warp_weights)`` matching the
    ``build_soft_splat_warped_rgb`` interface so the unit can swap them.
    """
    rgb_pf, valid_pf, w_pf = build_soft_splat_warped_rgb(
        source_rgb=source_rgb,
        source_depth=source_depth,
        source_depth_valid=source_depth_valid,
        src_cam2world=src_cam2world,
        tgt_cam2world=tgt_cam2world,
        intrinsics=intrinsics,
        image_h=image_h,
        image_w=image_w,
        target_intrinsics=target_intrinsics,
        depth_weight_scale=depth_weight_scale,
    )
    rgb_mf, valid_mf, w_mf = build_multi_source_soft_splat_warped_rgb(
        source_rgb=source_rgb,
        source_depth=source_depth,
        source_depth_valid=source_depth_valid,
        src_cam2world=src_cam2world,
        tgt_cam2world=tgt_cam2world,
        intrinsics=intrinsics,
        image_h=image_h,
        image_w=image_w,
        target_intrinsics=target_intrinsics,
        static_mask=static_mask,
        depth_weight_scale=depth_weight_scale,
    )

    use_mf = (valid_mf > 0)                  # (B, T, 1, H, W) bool
    use_mf_rgb = use_mf.expand_as(rgb_mf)
    rgb_out = torch.where(use_mf_rgb, rgb_mf, rgb_pf)
    valid_out = torch.where(use_mf, valid_mf, valid_pf)
    w_out = torch.where(use_mf, w_mf, w_pf)
    return rgb_out, valid_out, w_out
