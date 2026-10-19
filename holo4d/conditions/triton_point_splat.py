"""Triton point-cloud bilinear splat for Track4D-360 dense warps.

This module is intentionally narrow: it accelerates the flat point-cloud
primitive used by ``build_dense_target_reprojection_hybrid`` without changing
that function's RGB/depth/valid contract. The math matches
``soft_splat_warp._bilinear_splat_points``: four bilinear corner writes,
log-depth weighting, padded canvas, then weighted normalization.
"""

from __future__ import annotations

from typing import Tuple

import torch


try:  # Triton is optional; callers fall back to the torch implementation.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only on non-Triton envs
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _point_splat_accum_kernel(
        values, mask, depth, uv, log_z_max, canvas,
        N: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
        depth_weight_scale: tl.constexpr, BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        active = offsets < N

        mask_vals = tl.load(mask + b * N + offsets, mask=active, other=0.0)
        z_raw = tl.load(depth + b * N + offsets, mask=active, other=0.0)
        z = tl.maximum(z_raw, 0.0)
        log_z = tl.log(1.0 + z)
        lz_max = tl.load(log_z_max + b)
        exponent = (log_z / tl.maximum(lz_max, 1.0e-7)) * depth_weight_scale
        exponent = tl.minimum(exponent, 80.0)
        u_raw = tl.load(uv + (b * N + offsets) * 2 + 0, mask=active, other=-2.0) + 1.0
        v_raw = tl.load(uv + (b * N + offsets) * 2 + 1, mask=active, other=-2.0) + 1.0
        in_bounds = (u_raw >= 1.0) & (u_raw <= W * 1.0) & (v_raw >= 1.0) & (v_raw <= H * 1.0)
        inv_dw = (1.0 / (tl.exp(exponent) + 1.0e-7)) * mask_vals * tl.cast(in_bounds, tl.float32)

        u = tl.minimum(tl.maximum(u_raw, 0.0), W + 1.0)
        v = tl.minimum(tl.maximum(v_raw, 0.0), H + 1.0)
        uf0_f = tl.floor(u)
        vf0_f = tl.floor(v)
        uf0 = tl.cast(tl.minimum(tl.maximum(uf0_f, 0.0), W + 1.0), tl.int64)
        vf0 = tl.cast(tl.minimum(tl.maximum(vf0_f, 0.0), H + 1.0), tl.int64)
        uf1 = tl.minimum(uf0 + 1, W + 1)
        vf1 = tl.minimum(vf0 + 1, H + 1)

        du = u - tl.cast(uf0, tl.float32)
        dv = v - tl.cast(vf0, tl.float32)
        w_nw = (1.0 - dv) * (1.0 - du) * inv_dw
        w_sw = dv * (1.0 - du) * inv_dw
        w_ne = (1.0 - dv) * du * inv_dw
        w_se = dv * du * inv_dw

        base_val = (b * N + offsets) * 4
        c0 = tl.load(values + base_val + 0, mask=active, other=0.0)
        c1 = tl.load(values + base_val + 1, mask=active, other=0.0)
        c2 = tl.load(values + base_val + 2, mask=active, other=0.0)
        c3 = tl.load(values + base_val + 3, mask=active, other=0.0)

        # canvas layout: (B, H+2, W+2, 5), channels 0..3 are RGBD sums,
        # channel 4 is accumulated weight.
        stride_b = (H + 2) * (W + 2) * 5
        stride_y = (W + 2) * 5

        pix = b * stride_b + vf0 * stride_y + uf0 * 5
        write_mask = active & (w_nw != 0.0)
        tl.atomic_add(canvas + pix + 0, c0 * w_nw, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 1, c1 * w_nw, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 2, c2 * w_nw, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 3, c3 * w_nw, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 4, w_nw, sem="relaxed", mask=write_mask)

        pix = b * stride_b + vf1 * stride_y + uf0 * 5
        write_mask = active & (w_sw != 0.0)
        tl.atomic_add(canvas + pix + 0, c0 * w_sw, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 1, c1 * w_sw, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 2, c2 * w_sw, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 3, c3 * w_sw, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 4, w_sw, sem="relaxed", mask=write_mask)

        pix = b * stride_b + vf0 * stride_y + uf1 * 5
        write_mask = active & (w_ne != 0.0)
        tl.atomic_add(canvas + pix + 0, c0 * w_ne, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 1, c1 * w_ne, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 2, c2 * w_ne, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 3, c3 * w_ne, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 4, w_ne, sem="relaxed", mask=write_mask)

        pix = b * stride_b + vf1 * stride_y + uf1 * 5
        write_mask = active & (w_se != 0.0)
        tl.atomic_add(canvas + pix + 0, c0 * w_se, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 1, c1 * w_se, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 2, c2 * w_se, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 3, c3 * w_se, sem="relaxed", mask=write_mask)
        tl.atomic_add(canvas + pix + 4, w_se, sem="relaxed", mask=write_mask)


def triton_available() -> bool:
    return triton is not None


def bilinear_splat_points_triton(
    values: torch.Tensor,
    mask: torch.Tensor,
    depth: torch.Tensor,
    uv: torch.Tensor,
    image_h: int,
    image_w: int,
    *,
    depth_weight_scale: float = 50.0,
    clamp_output: bool = True,
    block_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of ``_bilinear_splat_points`` for C=4 values."""
    if triton is None:
        raise RuntimeError("Triton is not available in this environment")
    if not values.is_cuda:
        raise ValueError("Triton point splat requires CUDA tensors")
    if values.dtype != torch.float32:
        values = values.float()
    if mask.dtype != torch.float32:
        mask = mask.float()
    if depth.dtype != torch.float32:
        depth = depth.float()
    if uv.dtype != torch.float32:
        uv = uv.float()

    B, N, C = values.shape
    if C != 4:
        raise ValueError(f"Triton point splat currently supports C=4, got C={C}")
    H, W = int(image_h), int(image_w)
    values = values.contiguous()
    mask = mask.contiguous()
    depth = depth.contiguous()
    uv = uv.contiguous()

    log_z_max = torch.log1p(depth.clamp_min(0.0)).amax(dim=1).clamp_min(1.0e-7).contiguous()
    canvas = torch.zeros(B, H + 2, W + 2, 5, dtype=torch.float32, device=values.device)
    grid = (B, triton.cdiv(N, block_size))
    _point_splat_accum_kernel[grid](
        values, mask, depth, uv, log_z_max, canvas,
        N, H, W, float(depth_weight_scale), BLOCK=block_size,
    )

    cropped = canvas[:, 1:-1, 1:-1, :]
    sums = cropped[..., :4].permute(0, 3, 1, 2).contiguous()
    weights = cropped[..., 4:5].permute(0, 3, 1, 2).contiguous()
    valid_mask = weights > 0
    out = torch.where(valid_mask, sums / weights.clamp_min(1.0e-20), torch.zeros_like(sums))
    if clamp_output:
        out = out.clamp(0.0, 1.0)
    return out, valid_mask.to(values.dtype)
