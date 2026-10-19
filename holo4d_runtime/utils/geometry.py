# normalization borrowed from geo4d, iccv 25
from functools import cache
try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree as KDTree
from typing import Optional
from pdb import set_trace as st


def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias

def to_numpy(x):
    return todevice(x, "numpy")


def to_cpu(x):
    return todevice(x, "cpu")


def to_cuda(x):
    return todevice(x, "cuda")


def normalize_scalar_sequence_bbox2(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    lower_bound_ratio: float = 0.02,
    upper_bound_ratio: float = 0.98,
    lower_margin: float = 0.01,
    upper_margin: float = 0.01,
    invalid_value: float = -1.0,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    return_st: bool = False,
):
    """
    Normalize a sequence of scalar maps to approximately fill [-1, 1] using robust
    percentile bounds, similar to PMDiff's bbox normalization utilities.

    Args:
        values: Tensor of shape (T, H, W) (or any shape) containing scalar values.
        valid_mask: Boolean tensor with the same shape as `values` marking valid elements.
        lower_bound_ratio / upper_bound_ratio: Percentiles for robust min/max.
        lower_margin / upper_margin: Small margins added around percentile bounds.
        invalid_value: Value used where `valid_mask == False` before clamping.
        clamp_min / clamp_max: Clamp output range.
        return_st: If True, return (normalized_values, s, t) where s is scale and t is shift.

    Returns:
        If return_st=False: Tensor with same shape as `values`, mapped to [-1, 1] (clamped).
        If return_st=True: (normalized_tensor, s, t) where s and t are float scalars.
    """
    if values.shape != valid_mask.shape:
        raise ValueError(f"`values` and `valid_mask` must have the same shape, got {values.shape} vs {valid_mask.shape}.")

    valid_values = values[valid_mask]
    if valid_values.numel() == 0:
        result = torch.full_like(values, float(invalid_value)).clamp(clamp_min, clamp_max)
        if return_st:
            return result, 1.0, 0.0
        return result

    sorted_vals, _ = torch.sort(valid_values.reshape(-1))
    total = int(sorted_vals.numel())
    lower_idx = int(total * float(lower_bound_ratio))
    upper_idx = int(total * float(upper_bound_ratio))
    lower_idx = max(0, min(lower_idx, total - 1))
    upper_idx = max(0, min(upper_idx, total - 1))
    if upper_idx < lower_idx:
        lower_idx, upper_idx = upper_idx, lower_idx

    lower = sorted_vals[lower_idx] - float(lower_margin)
    upper = sorted_vals[upper_idx] + float(upper_margin)
    scale = (upper - lower).clamp_min(1e-12)

    out = (values - lower) / scale
    out = out * 2.0 - 1.0
    out = torch.where(valid_mask, out, torch.full_like(out, float(invalid_value)))
    out = out.clamp(clamp_min, clamp_max)

    if return_st:
        # Return (s, t) compatible with bbox2: s = scale, t = shift (lower bound)
        return out, float(scale.item()), float(lower.item())
    return out


def normalize_inverse_depth_bbox2(
    depth_maps: torch.Tensor,
    valid_mask: torch.Tensor,
    lower_bound_ratio: float = 0.02,
    upper_bound_ratio: float = 0.98,
    eps: float = 1e-8,
    invalid_value: float = -1.0,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
):
    """
    Convert depth to inverse depth (disparity) and normalize to [-1, 1] with robust bounds.

    Args:
        depth_maps: Tensor of shape (T, H, W) with depth values.
        valid_mask: Boolean tensor of shape (T, H, W) for depth>0 pixels.
    """
    if depth_maps.shape != valid_mask.shape:
        raise ValueError(
            f"`depth_maps` and `valid_mask` must have the same shape, got {depth_maps.shape} vs {valid_mask.shape}."
        )
    disp = torch.zeros_like(depth_maps)
    disp[valid_mask] = 1.0 / (depth_maps[valid_mask] + eps)
    return normalize_scalar_sequence_bbox2(
        disp,
        valid_mask=valid_mask,
        lower_bound_ratio=lower_bound_ratio,
        upper_bound_ratio=upper_bound_ratio,
        invalid_value=invalid_value,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def normalize_log_depth_bbox2(
    depth_maps: torch.Tensor,
    valid_mask: torch.Tensor,
    lower_bound_ratio: float = 0.02,
    upper_bound_ratio: float = 0.98,
    lower_margin: float = 0.01,
    upper_margin: float = 0.01,
    eps: float = 1e-8,
    invalid_value: float = -1.0,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    return_st: bool = False,
):
    """
    Convert depth to log-space depth and normalize to [-1, 1] with robust bounds.

    This is designed for training models that predict depth in log space (like Depth Anything 3),
    where the model outputs log(depth) and applies exp() to recover actual depth.

    The transformation is:
        log_depth = log(depth + eps)  # Apply log transform
        normalized = percentile_normalize(log_depth)  # Map to [-1, 1]

    Args:
        depth_maps: Tensor of shape (T, H, W) with depth values.
        valid_mask: Boolean tensor of shape (T, H, W) for valid depth pixels (depth > 0).
        lower_bound_ratio: Percentile for robust minimum (default: 0.02 = 2nd percentile).
        upper_bound_ratio: Percentile for robust maximum (default: 0.98 = 98th percentile).
        lower_margin: Small margin subtracted from lower bound (default: 0.01).
        upper_margin: Small margin added to upper bound (default: 0.01).
        eps: Small epsilon to avoid log(0) (default: 1e-8).
        invalid_value: Value assigned to invalid pixels before clamping (default: -1.0).
        clamp_min: Minimum value after normalization (default: -1.0).
        clamp_max: Maximum value after normalization (default: 1.0).
        return_st: If True, return (normalized, s, t) where s and t are normalization parameters.

    Returns:
        If return_st=False: Tensor of shape (T, H, W) with log-depth normalized to [-1, 1].
        If return_st=True: (normalized_tensor, s, t) where s is scale and t is shift.

    Example:
        >>> depth = torch.tensor([[[1.0, 2.0], [5.0, 10.0]]])  # (1, 2, 2)
        >>> valid = torch.ones_like(depth, dtype=torch.bool)
        >>> log_depth_norm, s, t = normalize_log_depth_bbox2(depth, valid, return_st=True)
        >>> # Now log_depth_norm is in [-1, 1] and ready for VAE encoding
        >>> # Use same (s, t) for normalizing points/rays in log-space
    """
    if depth_maps.shape != valid_mask.shape:
        raise ValueError(
            f"`depth_maps` and `valid_mask` must have the same shape, got {depth_maps.shape} vs {valid_mask.shape}."
        )

    # Apply log transform to valid depth values
    log_depth = torch.zeros_like(depth_maps)
    log_depth[valid_mask] = torch.log(depth_maps[valid_mask] + eps)

    # Normalize log-depth to [-1, 1] using percentile-based bounds
    return normalize_scalar_sequence_bbox2(
        log_depth,
        valid_mask=valid_mask,
        lower_bound_ratio=lower_bound_ratio,
        upper_bound_ratio=upper_bound_ratio,
        lower_margin=lower_margin,
        upper_margin=upper_margin,
        invalid_value=invalid_value,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        return_st=return_st,
    )


def compute_log_depth_quantiles(
    depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log-depth quantiles for a positive depth tensor.

    Args:
        depth: (...,) tensor with positive depth values.
        valid_mask: Optional boolean mask with the same shape as ``depth``.
        q_low: Lower quantile.
        q_high: Upper quantile.

    Returns:
        (ql, qh): scalar tensors on the same device as ``depth``.
    """
    depth_f = depth.float()
    if valid_mask is None:
        vals = depth_f[depth_f > 0]
    else:
        valid = valid_mask.to(device=depth.device, dtype=torch.bool)
        vals = depth_f[valid & (depth_f > 0)]

    if vals.numel() == 0:
        dev = depth.device
        return torch.zeros((), device=dev), torch.ones((), device=dev)

    log_vals = torch.log(vals.clamp_min(1e-8))
    q_low_f = float(q_low)
    q_high_f = float(q_high)
    if q_low_f <= 0.0 and q_high_f >= 1.0:
        ql = log_vals.amin()
        qh = log_vals.amax()
    else:
        ql = torch.quantile(log_vals, q_low_f)
        qh = torch.quantile(log_vals, q_high_f)
    return ql, qh


def normalize_depth_log(
    depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    ql: Optional[torch.Tensor] = None,
    qh: Optional[torch.Tensor] = None,
    q_low: float = 0.02,
    q_high: float = 0.98,
    invalid_value: float = -1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Normalize a positive depth tensor in log space to ``[-1, 1]``.

    Args:
        depth: (...,) tensor with positive depth values.
        valid_mask: Optional boolean mask with the same shape as ``depth``.
        ql, qh: Optional precomputed log-depth quantiles. When omitted they are
            computed from ``depth`` and ``valid_mask``.
        q_low, q_high: Quantile bounds used when computing stats.
        invalid_value: Fill value for invalid pixels.

    Returns:
        normalized: same shape as ``depth``, in ``[-1, 1]``.
        params: dict for :func:`denormalize_depth_log`.
    """
    depth_f = depth.float()
    if valid_mask is None:
        valid = depth_f > 0
    else:
        valid = valid_mask.to(device=depth.device, dtype=torch.bool) & (depth_f > 0)

    if ql is None or qh is None:
        ql, qh = compute_log_depth_quantiles(depth_f, valid, q_low=q_low, q_high=q_high)

    ql_t = torch.as_tensor(ql, device=depth.device, dtype=depth_f.dtype)
    qh_t = torch.as_tensor(qh, device=depth.device, dtype=depth_f.dtype)
    denom = (qh_t - ql_t).clamp_min(1e-8)

    out = torch.full_like(depth_f, float(invalid_value))
    if bool(valid.any()):
        log_depth = torch.log(depth_f[valid].clamp_min(1e-8))
        out[valid] = ((log_depth - ql_t) / denom) * 2.0 - 1.0
    out = out.clamp(-1.0, 1.0)
    params = {
        "norm_type": "log_depth_q",
        "ql": float(ql_t.detach().cpu().item()),
        "qh": float(qh_t.detach().cpu().item()),
        "q_low": float(q_low),
        "q_high": float(q_high),
        "invalid_value": float(invalid_value),
    }
    return out, params


def denormalize_depth_log(
    normalized: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """
    Inverse of :func:`normalize_depth_log`.

    Args:
        normalized: normalized depth tensor in ``[-1, 1]``.
        params: dict returned by :func:`normalize_depth_log`.

    Returns:
        depth tensor in the original positive scale.
    """
    ql = torch.as_tensor(params["ql"], device=normalized.device, dtype=normalized.dtype)
    qh = torch.as_tensor(params["qh"], device=normalized.device, dtype=normalized.dtype)
    denom = (qh - ql).clamp_min(1e-8)
    log_depth = ((normalized.clamp(-1.0, 1.0) + 1.0) * 0.5) * denom + ql
    return torch.exp(log_depth)


def normalize_depth_log_fixed_cap(
    depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    max_depth: float = 14.0,
    invalid_value: float = -1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Normalize positive depth with a fixed log-depth cap after common-scale normalization.

    Mapping:
        depth=1        -> 0 after log scaling only if max_depth ~= 1
        depth=max_depth -> +1
        depth<=0/invalid -> invalid_value

    The contract is inference-safe because ``max_depth`` is fixed.
    """
    depth_f = depth.float()
    valid = (depth_f > 0) if valid_mask is None else (valid_mask.to(device=depth.device, dtype=torch.bool) & (depth_f > 0))
    cap = torch.as_tensor(float(max(max_depth, 1e-6)), device=depth.device, dtype=depth_f.dtype)
    log_cap = torch.log(cap)
    out = torch.full_like(depth_f, float(invalid_value))
    if bool(valid.any()):
        log_depth = torch.log(depth_f[valid].clamp_min(1e-8)).clamp(max=log_cap)
        out[valid] = (log_depth / log_cap) * 2.0 - 1.0
    out = out.clamp(-1.0, 1.0)
    return out, {
        "norm_type": "log_depth_fixed_cap",
        "max_depth": float(cap.detach().cpu().item()),
        "invalid_value": float(invalid_value),
    }


def denormalize_depth_log_fixed_cap(
    normalized: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Inverse of :func:`normalize_depth_log_fixed_cap`."""
    max_depth = float(params.get("max_depth", 14.0))
    log_cap = torch.log(torch.as_tensor(max(max_depth, 1e-6), device=normalized.device, dtype=normalized.dtype))
    log_depth = ((normalized.clamp(-1.0, 1.0) + 1.0) * 0.5) * log_cap
    return torch.exp(log_depth)


def normalize_disparity_fixed_cap(
    depth: torch.Tensor,
    valid_mask=None,
    max_disparity: float = 5.0,
    invalid_value: float = 0.0,
):
    """
    Normalize depth as disparity (1/depth), capped and mapped to [-1, 1].

    Mapping:
        depth=inf/far    → disparity≈0 → -1
        depth=1/max_disp → disparity=max_disp → +1
        depth<=0/invalid → invalid_value

    Inference-safe: max_disparity is fixed.
    """
    depth_f = depth.float()
    valid = (depth_f > 0) if valid_mask is None else (valid_mask.to(device=depth.device, dtype=torch.bool) & (depth_f > 0))
    cap = float(max(max_disparity, 1e-6))
    out = torch.full_like(depth_f, float(invalid_value))
    if bool(valid.any()):
        disp = (1.0 / depth_f[valid].clamp_min(1e-8)).clamp(max=cap)
        out[valid] = (disp / cap) * 2.0 - 1.0
    out = out.clamp(-1.0, 1.0)
    return out, {
        "norm_type": "disparity_fixed_cap",
        "max_disparity": cap,
        "invalid_value": float(invalid_value),
    }


def denormalize_disparity_fixed_cap(
    normalized: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Inverse of :func:`normalize_disparity_fixed_cap`. Returns depth (not disparity)."""
    cap = float(params.get("max_disparity", 5.0))
    disp = ((normalized.clamp(-1.0, 1.0) + 1.0) * 0.5) * cap
    return 1.0 / disp.clamp_min(1e-8)


def normalize_pointcloud_logspace_bbox2(
    pts_all: list[torch.Tensor],
    valid_all: list[torch.Tensor],
    s: float,
    t: float,
    *,
    alpha: float = 2.0,
    beta: float = 2.0,
    eps: float = 1e-8,
    invalid_value: float = 1.05,
    clamp_min: float = -2.0,
    clamp_max: float = 2.0,
):
    """
    Normalize point cloud in log-space using provided (s, t) parameters from log-depth.

    This is used when depth_mode == "log_depth" to ensure points are normalized
    consistently with depth in log-space.

    The transformation is:
        log_z = log(z + eps)
        z' = ((log_z - t) / s) * 2 - 1
        x' = (x / s) * alpha
        y' = (y / s) * beta

    Args:
        pts_all: list of tensors, each shape (1, H, W, 3)
        valid_all: list of tensors, each shape (1, H, W) with boolean validity
        s: Scale parameter from log-depth normalization
        t: Shift parameter from log-depth normalization
        alpha: Scaling factor for x-coordinate (default: 2.0)
        beta: Scaling factor for y-coordinate (default: 2.0)
        eps: Small epsilon to avoid log(0) (default: 1e-8)
        invalid_value: Value for invalid points before clamping
        clamp_min: Minimum value after normalization
        clamp_max: Maximum value after normalization

    Returns:
        Tensor of shape (T, H, W, 3) with points normalized in log-space
    """
    if len(pts_all) != len(valid_all):
        raise ValueError("`pts_all` and `valid_all` must have the same length.")

    cated_pts = torch.cat(pts_all, dim=0)  # (T, H, W, 3)
    cated_valid = torch.cat(valid_all, dim=0).bool().unsqueeze(-1)  # (T, H, W, 1)
    cated_valid_pts = cated_valid.expand(-1, -1, -1, 3)

    device = cated_pts.device
    dtype = cated_pts.dtype

    # Apply log transform to z-coordinate
    z = cated_pts[..., 2:3]  # (T, H, W, 1)
    log_z = torch.zeros_like(z)
    valid_z = cated_valid[..., 0]  # (T, H, W)
    log_z[valid_z] = torch.log(z[valid_z] + eps)

    # Normalize using provided (s, t) parameters
    s_tensor = torch.tensor(s, device=device, dtype=dtype).clamp_min(1e-12)
    t_tensor = torch.tensor(t, device=device, dtype=dtype)

    # Apply normalization
    normalized_pts = cated_pts.clone()
    normalized_pts[..., 2] = (log_z[..., 0] - t_tensor) / s_tensor * 2.0 - 1.0
    normalized_pts[..., 0] = normalized_pts[..., 0] / s_tensor * float(alpha)
    normalized_pts[..., 1] = normalized_pts[..., 1] / s_tensor * float(beta)

    # Set invalid points
    normalized_pts[~cated_valid_pts] = float(invalid_value)
    normalized_pts = normalized_pts.clamp(clamp_min, clamp_max)

    return normalized_pts


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """Signed log1p used as the inverse of inv_log activation."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _signed_expm1(x: torch.Tensor) -> torch.Tensor:
    """Inverse of _signed_log1p."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def normalize_pointcloud_logspace_global(
    pts_all: list[torch.Tensor],
    valid_all: list[torch.Tensor],
    s_seq: float,
    q_low: float | list[float] | None = None,
    q_high: float | list[float] | None = None,
    # eps: float = 1e-6,
    *,
    # alpha: float = 1.0,
    # beta: float = 1.0,
    invalid_value: float = 1.0,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    compute_percentiles: bool = False,
    p_low: float = 0.0,
    p_high: float = 100.0,
):
    """
    Normalize point cloud using inverse inv_log and global affine stats.

    This matches VGGT's point head activation ("inv_log") by applying the
    inverse transform (signed log1p) before mapping to [-1, 1].

    Pipeline:
      1) Pre-normalize by scale (s_seq)
      2) Apply inverse inv_log: y = sign(p) * log1p(|p|)
      3) Global affine mapping to [-1, 1] using q_low/q_high

    NOTE: Unlike PMDiff which uses depth-only scaling (requires alpha=2.0, beta=2.0),
    this uses L2 norm-based common_scale which inherently balances all dimensions.
    Therefore, alpha=1.0, beta=1.0 (no additional scaling) is appropriate.

    Percentile offset trick (±0.01):
    - q_low/q_high should be computed with ±0.01 offsets in statistics computation
    - This expands the percentile bounds slightly, so data at percentile boundaries
      maps to ~±0.98 instead of exactly ±1.0
    - Avoids VAE saturation at the [-1, 1] edges while staying within VAE's training range
    - Example: if raw percentile(2%) = -5.0, use q_low = -5.01 in normalization
      → data at -5.0 maps to ~-0.98 instead of -1.0

    Args:
        pts_all: list of tensors, each shape (1, H, W, 3)
        valid_all: list of tensors, each shape (1, H, W) with boolean validity
        s_seq: Scale factor (use common_scale for points)
        q_low: Global lower bound for affine normalization (float or 3-list or None)
               Should be: percentile(p_low) - 0.01
               If None, must set compute_percentiles=True
        q_high: Global upper bound for affine normalization (float or 3-list or None)
                Should be: percentile(p_high) + 0.01
                If None, must set compute_percentiles=True
        eps: Unused (kept for API compatibility)
        invalid_value: Value for invalid points before clamping (default: 1.0)
        clamp_min: Minimum value after normalization (default: -1.0)
        clamp_max: Maximum value after normalization (default: 1.0)
        compute_percentiles: If True, compute q_low/q_high from data (default: False)
        p_low: Lower percentile to compute (default: 0.0, only used if compute_percentiles=True)
        p_high: Upper percentile to compute (default: 100.0, only used if compute_percentiles=True)

    Returns:
        If compute_percentiles=False:
            Tensor of shape (T, H, W, 3) with points normalized to [-1, 1]
        If compute_percentiles=True:
            Tuple of (normalized_pts, q_low, q_high) where q_low/q_high are 3-element lists
    """
    if len(pts_all) != len(valid_all):
        raise ValueError("`pts_all` and `valid_all` must have the same length.")

    # Concatenate all points
    cated_pts = torch.cat(pts_all, dim=0)  # (T, H, W, 3)
    cated_valid = torch.cat(valid_all, dim=0).bool()  # (T, H, W)

    device = cated_pts.device
    dtype = cated_pts.dtype

    scale = torch.tensor(float(s_seq), device=device, dtype=dtype).clamp_min(1e-12)

    # Pre-normalize and apply optional xy scaling
    pts_scaled = cated_pts / scale
    # pts_scaled[..., 0] = pts_scaled[..., 0] * float(alpha)
    # pts_scaled[..., 1] = pts_scaled[..., 1] * float(beta)

    # Inverse of inv_log activation
    pts_log = _signed_log1p(pts_scaled)

    # Compute percentiles if requested (avoids redundant computation)
    if compute_percentiles:
        import numpy as np
        q_low_computed = []
        q_high_computed = []

        for ch_idx in range(3):
            pts_log_ch = pts_log[..., ch_idx]  # (T, H, W)
            valid_values = pts_log_ch[cated_valid].cpu().numpy()

            if valid_values.size > 0:
                q_low_computed.append(float(np.percentile(valid_values, p_low)))
                q_high_computed.append(float(np.percentile(valid_values, p_high)))
            else:
                q_low_computed.append(0.0)
                q_high_computed.append(1.0)

        q_low = q_low_computed
        q_high = q_high_computed

    # Prepare affine bounds (support scalar or per-axis bounds)
    q_low_t = torch.as_tensor(q_low, device=device, dtype=dtype).view(-1)
    q_high_t = torch.as_tensor(q_high, device=device, dtype=dtype).view(-1)
    if q_low_t.numel() == 1:
        q_low_t = q_low_t.repeat(3)
    if q_high_t.numel() == 1:
        q_high_t = q_high_t.repeat(3)
    q_low_t = q_low_t.view(1, 1, 1, 3)
    q_high_t = q_high_t.view(1, 1, 1, 3)

    denom = (q_high_t - q_low_t).clamp_min(1e-8)
    normalized_pts = 2.0 * (pts_log - q_low_t) / denom - 1.0

    # Set invalid points
    invalid_mask = ~cated_valid.unsqueeze(-1).expand(-1, -1, -1, 3)
    normalized_pts[invalid_mask] = float(invalid_value)
    normalized_pts = torch.clamp(normalized_pts, clamp_min, clamp_max)

    if compute_percentiles:
        return normalized_pts, q_low, q_high
    else:
        return normalized_pts


def denormalize_pointcloud_logspace_global(
    normalized_pts: torch.Tensor,
    s_seq: float,
    q_low: float,
    q_high: float,
    # eps: float = 1e-6,
    # *,
    # alpha: float = 1.0,
    # beta: float = 1.0,
):
    """
    Denormalize point cloud from [-1, 1] back to metric scale.

    This is the inverse of normalize_pointcloud_logspace_global.

    Pipeline (inverse order):
      1) Inverse affine mapping from [-1, 1] to [q_low, q_high]
      2) Apply inv_log activation: p_scaled = sign(y) * (exp(|y|) - 1)
      3) Undo pre-normalization and xy scaling

    Args:
        normalized_pts: Tensor of shape (3, T, H, W) or (T, H, W, 3) with normalized points in [-1, 1]
        s_seq: Scale factor (same as used in normalization)
        q_low: Global lower bound used in normalization (float or 3-list)
        q_high: Global upper bound used in normalization (float or 3-list)
        eps: Unused (kept for API compatibility)
        alpha: Scaling factor for x-coordinate (default: 1.0)
        beta: Scaling factor for y-coordinate (default: 1.0)

    Returns:
        Tensor of same shape with points in metric scale
    """
    device = normalized_pts.device
    dtype = normalized_pts.dtype

    # Handle both (3, T, H, W) and (T, H, W, 3) formats
    if normalized_pts.shape[0] == 3 and len(normalized_pts.shape) == 4:
        # (3, T, H, W) -> (T, H, W, 3)
        pts = normalized_pts.permute(1, 2, 3, 0)
        input_format = "CTHW"
    else:
        pts = normalized_pts
        input_format = "THWC"

    # Prepare affine bounds (support scalar or per-axis bounds)
    q_low_t = torch.as_tensor(q_low, device=device, dtype=dtype).view(-1)
    q_high_t = torch.as_tensor(q_high, device=device, dtype=dtype).view(-1)
    if q_low_t.numel() == 1:
        q_low_t = q_low_t.repeat(3)
    if q_high_t.numel() == 1:
        q_high_t = q_high_t.repeat(3)
    q_low_t = q_low_t.view(1, 1, 1, 3)
    q_high_t = q_high_t.view(1, 1, 1, 3)

    # Clamp to valid range
    pts = torch.clamp(pts, -1.0, 1.0)

    # Inverse affine mapping: from [-1, 1] to [q_low, q_high]
    denom = (q_high_t - q_low_t).clamp_min(1e-8)
    pts_log = (pts + 1.0) / 2.0 * denom + q_low_t

    # Inverse of signed log1p (apply inv_log activation)
    pts_scaled = _signed_expm1(pts_log)

    # Undo xy scaling
    # pts_scaled[..., 0] = pts_scaled[..., 0] / float(alpha)
    # pts_scaled[..., 1] = pts_scaled[..., 1] / float(beta)
    pts_scaled[..., 0] = pts_scaled[..., 0]
    pts_scaled[..., 1] = pts_scaled[..., 1]

    # Undo scale normalization
    scale = torch.tensor(float(s_seq), device=device, dtype=dtype).clamp_min(1e-12)
    pts_metric = pts_scaled * scale

    # Return in original format
    if input_format == "CTHW":
        pts_metric = pts_metric.permute(3, 0, 1, 2)  # (T, H, W, 3) -> (3, T, H, W)

    return pts_metric


def normalize_depth_logspace_global(
    depth_all: list[torch.Tensor],
    valid_all: list[torch.Tensor],
    s_seq: float,
    q_low: float | None = None,
    q_high: float | None = None,
    *,
    invalid_value: float = -1.0,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    compute_percentiles: bool = False,
    p_low: float = 0.0,
    p_high: float = 100.0,
):
    """
    Normalize depth maps using log transform and global affine stats.

    This follows the same pipeline as normalize_pointcloud_logspace_global but for depth maps.

    Pipeline:
      1) Pre-normalize by scale (s_seq)
      2) Apply log transform: y = log(d + eps) where eps=1e-6 (depth is always positive)
      3) Global affine mapping to [-1, 1] using q_low/q_high

    Args:
        depth_all: list of tensors, each shape (1, H, W) with depth values
        valid_all: list of tensors, each shape (1, H, W) with boolean validity
        s_seq: Scale factor (use common_scale for depth)
        q_low: Global lower bound for affine normalization (float or None)
               If None, must set compute_percentiles=True
        q_high: Global upper bound for affine normalization (float or None)
                If None, must set compute_percentiles=True
        invalid_value: Value for invalid depth pixels
        clamp_min: Minimum value after normalization
        clamp_max: Maximum value after normalization
        compute_percentiles: If True, compute q_low/q_high from data
        p_low: Lower percentile for computing q_low (default: 0.0)
        p_high: Upper percentile for computing q_high (default: 100.0)

    Returns:
        If compute_percentiles=False: Tensor of shape (1, T, H, W) with normalized depth
        If compute_percentiles=True: (normalized_depth, q_low, q_high)
    """
    if len(depth_all) != len(valid_all):
        raise ValueError("`depth_all` and `valid_all` must have the same length.")

    cated_depth = torch.cat(depth_all, dim=0)  # (T, H, W)
    cated_valid = torch.cat(valid_all, dim=0).bool()  # (T, H, W)

    device = cated_depth.device
    dtype = cated_depth.dtype

    # Pre-normalize by scale
    depth_scaled = cated_depth / float(s_seq)

    # Apply log transform (depth is always positive, so just log(d + eps))
    eps = 1e-6
    log_depth = torch.zeros_like(depth_scaled)
    log_depth[cated_valid] = torch.log(depth_scaled[cated_valid] + eps)

    # Compute percentiles if requested
    if compute_percentiles:
        import numpy as np
        valid_values = log_depth[cated_valid].cpu().numpy()

        if valid_values.size > 0:
            q_low = float(np.percentile(valid_values, p_low))
            q_high = float(np.percentile(valid_values, p_high))
        else:
            q_low = 0.0
            q_high = 1.0

    # Prepare affine bounds
    q_low_t = torch.tensor(q_low, device=device, dtype=dtype)
    q_high_t = torch.tensor(q_high, device=device, dtype=dtype)

    denom = (q_high_t - q_low_t).clamp_min(1e-8)
    normalized_depth = 2.0 * (log_depth - q_low_t) / denom - 1.0

    # Set invalid pixels
    normalized_depth[~cated_valid] = float(invalid_value)
    normalized_depth = torch.clamp(normalized_depth, clamp_min, clamp_max)

    # Reshape to (1, T, H, W) for consistency with VAE input format
    normalized_depth = normalized_depth.unsqueeze(0)

    if compute_percentiles:
        return normalized_depth, q_low, q_high
    else:
        return normalized_depth


def denormalize_depth_logspace_global(
    normalized_depth: torch.Tensor,
    s_seq: float,
    q_low: float,
    q_high: float,
):
    """
    Denormalize depth map from [-1, 1] back to metric scale.

    This is the inverse of normalize_depth_logspace_global.

    Pipeline (inverse order):
      1) Inverse affine mapping from [-1, 1] to [q_low, q_high]
      2) Apply exp: d_scaled = exp(log_d) - eps
      3) Undo pre-normalization: d = d_scaled * s_seq

    Args:
        normalized_depth: Tensor of shape (1, T, H, W) with normalized depth in [-1, 1]
        s_seq: Scale factor (same as used in normalization)
        q_low: Global lower bound used in normalization
        q_high: Global upper bound used in normalization

    Returns:
        Tensor of same shape with depth in metric scale
    """
    device = normalized_depth.device
    dtype = normalized_depth.dtype

    # Clamp to valid range
    depth = torch.clamp(normalized_depth, -1.0, 1.0)

    # Prepare affine bounds
    q_low_t = torch.tensor(q_low, device=device, dtype=dtype)
    q_high_t = torch.tensor(q_high, device=device, dtype=dtype)

    # Inverse affine mapping: from [-1, 1] to [q_low, q_high]
    denom = (q_high_t - q_low_t).clamp_min(1e-8)
    log_depth = (depth + 1.0) / 2.0 * denom + q_low_t

    # Apply exp to recover scaled depth
    eps = 1e-6
    depth_scaled = torch.exp(log_depth) - eps

    # Undo scale normalization
    scale = torch.tensor(float(s_seq), device=device, dtype=dtype).clamp_min(1e-12)
    depth_metric = depth_scaled * scale

    return depth_metric


def normalize_depth_disparity_percentile(
    depth_all: list[torch.Tensor],
    valid_all: list[torch.Tensor],
    *,
    invalid_value: float = -1.0,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    compute_percentiles: bool = False,
    p_low: float = 2.0,
    p_high: float = 98.0,
    disp_min: float | None = None,
    disp_max: float | None = None,
):
    """
    Normalize depth maps to disparity using percentile-based min/max normalization.

    Following "We target relative depth estimation, so we normalize the disparity
    values to the range of [0, 1] by the maximum and minimum disparity values
    in the sequence" (then rescale to [-1, 1] for VAE).

    Pipeline:
      1) Convert depth to disparity: disp = 1 / depth
      2) Compute percentile-based min/max (e.g., 2% and 98% to avoid outliers)
      3) Normalize to [-1, 1]: normalized = 2 * (disp - disp_min) / (disp_max - disp_min) - 1

    Args:
        depth_all: list of tensors, each shape (1, H, W) with depth values
        valid_all: list of tensors, each shape (1, H, W) with boolean validity
        invalid_value: Value for invalid depth pixels (default: -1.0)
        clamp_min: Minimum value after normalization (default: -1.0)
        clamp_max: Maximum value after normalization (default: 1.0)
        compute_percentiles: If True, compute disp_min/disp_max from data
        p_low: Lower percentile for computing disp_min (default: 2.0 to ignore outliers)
        p_high: Upper percentile for computing disp_max (default: 98.0 to ignore outliers)
        disp_min: Minimum disparity value (if None and compute_percentiles=False, error)
        disp_max: Maximum disparity value (if None and compute_percentiles=False, error)

    Returns:
        If compute_percentiles=False: Tensor of shape (1, T, H, W) with normalized disparity
        If compute_percentiles=True: (normalized_disparity, disp_min, disp_max)
    """
    if len(depth_all) != len(valid_all):
        raise ValueError("`depth_all` and `valid_all` must have the same length.")

    cated_depth = torch.cat(depth_all, dim=0)  # (T, H, W)
    cated_valid = torch.cat(valid_all, dim=0).bool()  # (T, H, W)

    device = cated_depth.device
    dtype = cated_depth.dtype

    # Convert depth to disparity (disp = 1 / depth)
    # Avoid division by zero for invalid regions (will be masked later)
    disparity = torch.zeros_like(cated_depth)
    disparity[cated_valid] = 1.0 / (cated_depth[cated_valid].clamp_min(1e-6))

    # Compute percentiles if requested (to ignore outliers)
    if compute_percentiles:
        import numpy as np
        valid_values = disparity[cated_valid].cpu().numpy()

        if valid_values.size > 0:
            disp_min = float(np.percentile(valid_values, p_low))
            disp_max = float(np.percentile(valid_values, p_high))
        else:
            disp_min = 0.0
            disp_max = 1.0
    else:
        if disp_min is None or disp_max is None:
            raise ValueError("Must provide disp_min and disp_max if compute_percentiles=False")

    # Normalize disparity to [-1, 1]
    # Formula: 2 * (disp - disp_min) / (disp_max - disp_min) - 1
    disp_min_t = torch.tensor(disp_min, device=device, dtype=dtype)
    disp_max_t = torch.tensor(disp_max, device=device, dtype=dtype)

    denom = (disp_max_t - disp_min_t).clamp_min(1e-8)
    normalized_disparity = 2.0 * (disparity - disp_min_t) / denom - 1.0

    # Set invalid pixels
    normalized_disparity[~cated_valid] = float(invalid_value)
    normalized_disparity = torch.clamp(normalized_disparity, clamp_min, clamp_max)

    # Reshape to (1, T, H, W) for consistency with VAE input format
    normalized_disparity = normalized_disparity.unsqueeze(0)

    if compute_percentiles:
        return normalized_disparity, disp_min, disp_max
    else:
        return normalized_disparity


def denormalize_depth_disparity_percentile(
    normalized_disparity: torch.Tensor,
    disp_min: float,
    disp_max: float,
):
    """
    Denormalize disparity from [-1, 1] back to depth in metric scale.

    This is the inverse of normalize_depth_disparity_percentile.

    Pipeline (inverse order):
      1) Inverse affine mapping from [-1, 1] to [disp_min, disp_max]
      2) Convert disparity back to depth: depth = 1 / disp

    Args:
        normalized_disparity: Tensor of shape (1, T, H, W) with normalized disparity in [-1, 1]
        disp_min: Minimum disparity used in normalization
        disp_max: Maximum disparity used in normalization

    Returns:
        Tensor of same shape with depth in metric scale
    """
    device = normalized_disparity.device
    dtype = normalized_disparity.dtype

    # Clamp to valid range
    norm_disp = torch.clamp(normalized_disparity, -1.0, 1.0)

    # Prepare affine bounds
    disp_min_t = torch.tensor(disp_min, device=device, dtype=dtype)
    disp_max_t = torch.tensor(disp_max, device=device, dtype=dtype)

    # Inverse affine mapping: from [-1, 1] to [disp_min, disp_max]
    denom = (disp_max_t - disp_min_t).clamp_min(1e-8)
    disparity = (norm_disp + 1.0) / 2.0 * denom + disp_min_t

    # Convert disparity back to depth
    depth_metric = 1.0 / (disparity.clamp_min(1e-6))

    return depth_metric


def normalize_pointcloud_bbox2(
    pts_all: list[torch.Tensor],
    valid_all: list[torch.Tensor],
    *,
    alpha: float = 2.0,
    beta: float = 2.0,
    lower_bound_ratio: float = 0.02,
    upper_bound_ratio: float = 0.98,
    invalid_value: float = 1.05,
    clamp_min: float = -2.0,
    clamp_max: float = 2.0,
    return_st: bool = False,
):
    """
    Repo-local version of PMDiff's `normalize_pointcloud_bbox2` (bbox2).

    Notes:
    - Computes robust z-bounds across all valid points in the sequence.
    - Applies a per-sequence affine transform:
        x' = (x / s) * alpha
        y' = (y / s) * beta
        z' = ((z - t) / s) * 2 - 1
      where t is the robust lower z bound (minus a small margin) and s is the robust z span
      (plus a small margin).
    - Invalid points are filled with `invalid_value` and the output is clamped.

    Args:
        pts_all: list of tensors, each shape (1, H, W, 3)
        valid_all: list of tensors, each shape (1, H, W) with boolean validity
    Returns:
        If `return_st=False`: tensor of shape (T, H, W, 3).
        If `return_st=True`: (tensor, s_ret, t_ret) where s_ret/t_ret are python floats.
    """
    if len(pts_all) != len(valid_all):
        raise ValueError("`pts_all` and `valid_all` must have the same length.")
    cated_pts = torch.cat(pts_all, dim=0)  # (T, H, W, 3)
    cated_valid = torch.cat(valid_all, dim=0).bool().unsqueeze(-1)  # (T, H, W, 1)

    device = cated_pts.device
    dtype = cated_pts.dtype

    cated_valid_pts = cated_valid.expand(-1, -1, -1, 3)
    valid_pts = cated_pts[cated_valid_pts].reshape(-1, 3)
    valid_pts_z = valid_pts[:, 2]

    s_ret = 1.0
    t_ret = 0.0

    if int(valid_pts_z.numel()) > 0:
        sorted_z, _ = torch.sort(valid_pts_z, dim=0)
        total = int(sorted_z.shape[0])
        lower_idx = int(total * float(lower_bound_ratio))
        upper_idx = int(total * float(upper_bound_ratio))
        lower_idx = max(0, min(lower_idx, total - 1))
        upper_idx = max(0, min(upper_idx, total - 1))
        if upper_idx < lower_idx:
            lower_idx, upper_idx = upper_idx, lower_idx

        lower = sorted_z[lower_idx]
        upper = sorted_z[upper_idx]
        t_ret = float((lower - 0.01).item())
        s_ret = float((upper - lower + 0.02).item())

        lower_bound_z = lower.reshape(1, 1, 1, 1) - 0.01
        upper_bound_z = upper.reshape(1, 1, 1, 1) + 0.01
        s = (upper_bound_z - lower_bound_z).clamp_min(1e-12)  # (1,1,1,1)

        lower_xyz = torch.zeros((1, 1, 1, 3), device=device, dtype=dtype)
        lower_xyz[..., 2:3] = lower_bound_z.to(device=device, dtype=dtype)

        cated_pts = (cated_pts - lower_xyz) / s
        cated_pts[..., 2] = cated_pts[..., 2] * 2.0 - 1.0
        cated_pts[..., 0] = cated_pts[..., 0] * float(alpha)
        cated_pts[..., 1] = cated_pts[..., 1] * float(beta)

    cated_pts[~cated_valid_pts] = float(invalid_value)
    cated_pts = cated_pts.clamp(clamp_min, clamp_max)

    if return_st:
        return cated_pts, s_ret, t_ret
    return cated_pts


def plucker_from_origin_dir_bbox2(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    *,
    s: float,
    t: float,
    rescale_moment: float = 1.0,
    clamp_moment: float = 1.2,
) -> torch.Tensor:
    """
    Compute Plücker coordinates (d, o×d) from a ray origin and direction, applying the same bbox2
    (s,t) normalization used for point maps.

    This matches PMDiff's `plucker_embedding_all_s_t`: it normalizes the ray origin only:
        o'[:2] = o[:2] / s
        o'[2]  = (o[2] - t) / s
    then computes cross(o', d) and clamps the moment.

    Args:
        ray_origin: (..., 3)
        ray_dir: (..., 3), typically unit-length
    Returns:
        plucker: (..., 6) with channels [ray_dir(3), cross(3)]
    """
    if ray_origin.shape[-1] != 3 or ray_dir.shape[-1] != 3:
        raise ValueError(f"Expected ray_origin/ray_dir last dim = 3, got {ray_origin.shape} and {ray_dir.shape}.")

    origin = ray_origin.to(dtype=torch.float32)
    direction = ray_dir.to(dtype=torch.float32)

    denom = float(s) if float(s) != 0.0 else 1.0
    origin = origin.clone()
    origin[..., 2] = (origin[..., 2] - float(t)) / denom
    origin[..., 0] = origin[..., 0] / denom
    origin[..., 1] = origin[..., 1] / denom

    moment = torch.cross(origin, direction, dim=-1) * float(rescale_moment)
    moment = moment.clamp(-float(clamp_moment), float(clamp_moment))
    return torch.cat([direction, moment], dim=-1)

def xy_grid(
    W,
    H,
    device=None,
    origin=(0, 0),
    unsqueeze=None,
    cat_dim=-1,
    homogeneous=False,
    **arange_kw,
):
    """Output a (H,W,2) array of int32
    with output[j,i,0] = i + origin[0]
         output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def estimate_focal_knowing_depth_and_confidence_mask(
    pts3d, pp, conf_mask, focal_mode="median", min_focal=0.0, max_focal=np.inf
):
    """Reprojection method for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    This function considers only points where conf_mask is True.
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, H, W, 2) - pp.view(
        -1, 1, 1, 2
    )  # B,H,W,2

    # Apply the confidence mask
    conf_mask = conf_mask.view(B, H, W)  # Ensure conf_mask is of shape (B, H, W)
    valid_indices = conf_mask  # Boolean mask

    # Flatten the valid points
    pts3d_valid = pts3d[valid_indices]      # Shape: (N, 3)
    pixels_valid = pixels[valid_indices]    # Shape: (N, 2)

    if pts3d_valid.numel() == 0:
        # No valid points, return a default focal length
        focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
        return torch.tensor([focal_base])

    if focal_mode == "median":
        with torch.no_grad():
            # Direct estimation of focal
            u, v = pixels_valid.unbind(dim=-1)
            x, y, z = pts3d_valid.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # Assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(-1), fy_votes.view(-1)), dim=-1)
            focal = torch.nanmedian(f_votes).unsqueeze(0)  # Shape: (1,)

    elif focal_mode == "weiszfeld":
        # Initialize focal with L2 closed-form solution
        xy_over_z = (pts3d_valid[..., :2] / pts3d_valid[..., 2:3]).nan_to_num(
            posinf=0, neginf=0
        )  # Shape: (N, 2)

        dot_xy_px = (xy_over_z * pixels_valid).sum(dim=-1)  # Shape: (N,)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)          # Shape: (N,)

        focal = dot_xy_px.mean() / dot_xy_xy.mean()         # Shape: scalar

        # Iterative re-weighted least-squares
        for _ in range(100):
            # Re-weighting by inverse of distance
            dis = (pixels_valid - focal * xy_over_z).norm(dim=-1)  # Shape: (N,)
            w = dis.clip(min=1e-8).reciprocal()                    # Shape: (N,)
            # Update the scaling with the new weights
            focal = (w * dot_xy_px).sum() / (w * dot_xy_xy).sum()
        focal = focal.unsqueeze(0)  # Shape: (1,)
    else:
        raise ValueError(f"bad focal_mode={focal_mode}")

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    return focal


def estimate_focal(pts3d_i, conf_i, pp=None, min_conf_thr_percentile=10):
    B, H, W, THREE = pts3d_i.shape
    assert B == 1  # Since we're processing one sample at a time

    if pp is None:
        pp = torch.tensor((W / 2, H / 2), device=pts3d_i.device).view(1, 2)  # Shape: (1, 2)

    # Flatten the confidence map using reshape instead of view
    conf_flat = conf_i.reshape(-1)

    # Compute the confidence threshold based on the percentile
    percentile = min_conf_thr_percentile / 100.0  # Convert to a fraction
    conf_threshold = torch.quantile(conf_flat, percentile)

    # Create the confidence mask based on the computed threshold
    conf_mask = conf_i >= conf_threshold
    conf_mask = conf_mask.view(B, H, W)  # Ensure shape is (B, H, W)

    # Check if there are enough valid points
    if conf_mask.sum() < 10:  # Adjust the minimum number as needed
        print("Not enough high-confidence points for focal estimation.")
        # Optionally, adjust the percentile or set conf_mask to all True
        # For example:
        # conf_mask = torch.ones_like(conf_mask, dtype=torch.bool)

    focal = estimate_focal_knowing_depth_and_confidence_mask(
        pts3d_i, pp.unsqueeze(0), conf_mask, focal_mode="weiszfeld"
    ).ravel()
    return float(focal)

@cache
def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)

def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10, num_guessed_focals=100):
    # extract camera poses and focals with RANSAC-PnP
    if cv2 is None:
        raise ModuleNotFoundError("Missing dependency 'cv2' (opencv-python). Required for fast_pnp().")
    if msk.sum() < 4:
        return None, None  # we need at least 4 points for PnP
    pts3d, msk = map(to_numpy, (pts3d, msk))

    H, W, THREE = pts3d.shape
    assert THREE == 3
    pixels = pixel_grid(H, W)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S / 2, S * 3, num=num_guessed_focals)
    else:
        tentative_focals = [focal]

    if pp is None:
        pp = (W / 2, H / 2)
    else:
        pp = to_numpy(pp)

    best = (0,)
    for focal in tentative_focals:
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        try:  # solvePnPRansac is not always solvable, especially when the predicted points are not very good
            success, R, T, inliers = cv2.solvePnPRansac(
                pts3d[msk],
                pixels[msk],
                K,
                None,
                iterationsCount=niter_PnP,
                reprojectionError=5,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not success:
                continue
        except cv2.error:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None, None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]  # world to cam
    R, T = map(torch.from_numpy, (R, T))
    return best_focal, inv(sRT_to_4x4(1, R, T, device))  # cam to world



def estimate_cam_pose_one_sample(sample_preds, device='cpu', niter_PnP=10, min_conf_thr_percentile=0):
    poses_c2w = []
    estimated_focals = []

    # Define the function to process each view
    def process_view(view_idx):
        pts3d = sample_preds[view_idx]["pts3d_in_other_view"].cpu().numpy().squeeze()  # (H, W, 3)
        valid_mask = sample_preds[view_idx]["conf"].cpu().numpy().squeeze() > 1.0  # Confidence mask
        # use the confidence map to filter out low-confidence points
        # conf_threshold_value = torch.quantile(sample_preds[view_idx]["conf"].view(-1), min_conf_thr_percentile / 100.0)
        # valid_mask = sample_preds[view_idx]["conf"].cpu().numpy().squeeze() >= float(conf_threshold_value ) # Confidence mask
        focal_length = float(sample_preds[view_idx]["focal_length"]) if "focal_length" in sample_preds[view_idx] else None

        # Call fast_pnp with unflattened pts3d and mask
        focal_length, pose_c2w = fast_pnp(
            torch.tensor(pts3d),
            focal_length,  # Guess focal length
            torch.tensor(valid_mask, dtype=torch.bool),
            "cpu",
            pp=None,  # Use default principal point (center of image)
            niter_PnP=niter_PnP
        )

        if pose_c2w is None or focal_length is None:
            # log.warning(f"Failed to estimate pose for view {view_idx}")
            print(f"Failed to estimate pose for view {view_idx}")
            return np.eye(4), focal_length  # Return identity pose in case of failure

        # Return the results for this view
        return pose_c2w.cpu().numpy(), focal_length

    # Use ThreadPoolExecutor to process views in parallel
    with ThreadPoolExecutor() as executor:
        # Map the process_view function to each view index
        results = list(executor.map(process_view, range(len(sample_preds))))

    # Collect the results
    for pose_c2w_result, focal_length_result in results:
        poses_c2w.append(pose_c2w_result)
        estimated_focals.append(focal_length_result)

    return poses_c2w, estimated_focals


def estimate_camera_poses(preds, views=None, niter_PnP=10, focal_length_estimation_method='individual'):
    """Estimate camera poses and focal lengths using fast_pnp in parallel."""

    batch_size = len(preds[0]["pts3d_in_other_view"])  # Get the batch size

    # Prepare data_for_processing
    data_for_processing = []

    for i in range(batch_size):
        # Collect preds for each sample in the batch
        sample_preds = [{key: value[i].cpu() for key, value in view.items()} for view in preds]

        data_for_processing.append(sample_preds)

    # Estimate the focal length
    def estimate_focal_for_sample(sample_preds):
        if focal_length_estimation_method == 'first_view_from_global_head':
            # Use global head outputs for focal length estimation
            pts3d_i = sample_preds[0]["pts3d_in_other_view"].unsqueeze(0)  # Shape: (1, H, W, 3)
            conf_i = sample_preds[0]["conf"].unsqueeze(0)                  # Shape: (1, H, W)
        elif focal_length_estimation_method == 'first_view_from_local_head':
            # Use local head outputs for focal length estimation
            pts3d_i = sample_preds[0]["pts3d_local_aligned_to_global"].unsqueeze(0)  # Shape: (1, H, W, 3)
            conf_i = sample_preds[0]["conf_local"].unsqueeze(0)                       # Shape: (1, H, W)
        elif focal_length_estimation_method == 'first_view_from_track_head':
            # Use local head outputs for focal length estimation
            pts3d_i = sample_preds[0]["track"].unsqueeze(0)  # Shape: (1, H, W, 3)
            conf_i = sample_preds[0]["conf_track"].unsqueeze(0)
        elif focal_length_estimation_method == 'individual':
            # Focal length will be estimated individually per view
            return sample_preds
        else:
            raise ValueError(f"Unknown focal_length_estimation_method: {focal_length_estimation_method}")

        # Estimate focal length using the provided function and confidence mask
        estimated_focal = estimate_focal(pts3d_i, conf_i, min_conf_thr_percentile=10)

        # Store the estimated focal length in sample_preds
        for view_pred in sample_preds:
            view_pred["focal_length"] = estimated_focal
            # view_pred["focal_length"] = 256.64

        return sample_preds

    with ThreadPoolExecutor() as executor:
        data_for_processing = list(executor.map(estimate_focal_for_sample, data_for_processing))

    # Estimate the camera poses
    # Use ProcessPoolExecutor to parallelize processing across samples in the batch
    poses_c2w_all = []
    estimated_focals_all = []

    # Use partial to fix arguments
    from functools import partial

    estimate_cam_pose_one_sample_partial = partial(estimate_cam_pose_one_sample, niter_PnP=niter_PnP, min_conf_thr_percentile=85)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(estimate_cam_pose_one_sample_partial, data_for_processing))

    # Collect results from all processed samples
    for poses_c2w_sample, estimated_focals_sample in results:
        poses_c2w_all.append(poses_c2w_sample)
        estimated_focals_all.append(estimated_focals_sample)

    return poses_c2w_all, estimated_focals_all


def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (
        isinstance(Trf, torch.Tensor)
        and isinstance(pts, torch.Tensor)
        and Trf.ndim == 3
        and pts.ndim == 4
    ):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = (
                torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                + Trf[:, None, None, :d, d]
            )
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """Invert a torch or numpy matrix"""
    if isinstance(mat, torch.Tensor):
        # for mixed precision training
        if mat.dtype == torch.bfloat16:
            mat = mat.to(torch.float32)
            mat = torch.linalg.inv(mat)
            return mat
        else:
            mat = torch.linalg.inv(mat)
        return mat
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


def sRT_to_4x4(scale, R, T, device):
    """
    Compose a 4x4 transformation matrix from scale, rotation, and translation.
    """
    trf = torch.eye(4, device=device)
    trf[:3, :3] = scale * R
    trf[:3, 3] = T.flatten()
    return trf


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    # set principal point
    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(
    depthmap, camera_intrinsics, camera_pose, **kw
):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    R_cam2world = camera_pose[:3, :3]
    t_cam2world = camera_pose[:3, 3]

    # Express in absolute coordinates (invalid depth values)
    X_world = (
        np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
    )
    return X_world, valid_mask


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def invalid_to_nans(arr, valid_mask, ndim=999):
    """Replace invalid values with NaN"""
    if valid_mask is not None:
        arr = arr.clone() if isinstance(arr, torch.Tensor) else arr.copy()
        mask = ~valid_mask
        if arr.ndim > ndim:
            mask = mask.unsqueeze(-1).expand_as(arr)
        arr[mask] = float('nan')
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    """Replace invalid values with zeros and return count of valid elements"""
    if valid_mask is None:
        nnz = arr.shape[0] * arr.shape[1] if arr.ndim >= 2 else arr.numel()
        return arr, nnz

    arr = arr.clone() if isinstance(arr, torch.Tensor) else arr.copy()
    mask = ~valid_mask
    if arr.ndim > ndim:
        mask = mask.unsqueeze(-1).expand_as(arr)
    arr[mask] = 0
    nnz = valid_mask.sum()
    return arr, nnz


def normalize_pointcloud(pts1, pts2, norm_mode="avg_dis", valid1=None, valid2=None):
    """renorm pointmaps pts1, pts2 with norm_mode"""
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split("_")

    if norm_mode == "avg":
        # gather all points together (joint normalization)
        nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        nan_pts2, nnz2 = (
            invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        )
        all_pts = (
            torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1
        )

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == "dis":
            pass  # do nothing
        elif dis_mode == "log1p":
            all_dis = torch.log1p(all_dis)
        elif dis_mode == "warp-log1p":
            # actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts1.shape[1:-1]
            pts1 = pts1 * warp_factor[:, : W1 * H1].view(-1, H1, W1, 1)
            if pts2 is not None:
                H2, W2 = pts2.shape[1:-1]
                pts2 = pts2 * warp_factor[:, W1 * H1 :].view(-1, H2, W2, 1)
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f"bad {dis_mode=}")

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:
        # gather all points together (joint normalization)
        nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        all_pts = (
            torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1
        )

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)

        if norm_mode == "avg":
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == "median":
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == "sqrt":
            norm_factor = all_dis.sqrt().nanmean(dim=1) ** 2
        else:
            raise ValueError(f"bad {norm_mode=}")

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor.unsqueeze_(-1)

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    return res


@torch.no_grad()
def get_joint_pointcloud_depth(z1, z2, valid_mask1, valid_mask2=None, quantile=0.5):
    # set invalid points to NaN
    _z1 = invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
    _z2 = (
        invalid_to_nans(z2, valid_mask2).reshape(len(z2), -1)
        if z2 is not None
        else None
    )
    _z = torch.cat((_z1, _z2), dim=-1) if z2 is not None else _z1

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(
    pts1, pts2, valid_mask1=None, valid_mask2=None, z_only=False, center=True
):
    # set invalid points to NaN
    _pts1 = invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
    _pts2 = (
        invalid_to_nans(pts2, valid_mask2).reshape(len(pts2), -1, 3)
        if pts2 is not None
        else None
    )
    _pts = torch.cat((_pts1, _pts2), dim=1) if pts2 is not None else _pts1

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


def find_reciprocal_matches(P1, P2):
    """
    returns 3 values:
    1 - reciprocal_in_P2: a boolean array of size P2.shape[0], a "True" value indicates a match
    2 - nn2_in_P1: a int array of size P2.shape[0], it contains the indexes of the closest points in P1
    3 - reciprocal_in_P2.sum(): the number of matches
    """
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2))
    reciprocal_in_P2 = nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist

    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))


# def compute_per_sequence_percentiles(pts_list, valid_list, common_scale, p_low=0.5, p_high=99.5):
#     """
#     DEPRECATED: Use normalize_pointcloud_logspace_global with compute_percentiles=True instead.

#     This function is kept for backward compatibility but results in redundant computation
#     when followed by normalize_pointcloud_logspace_global (both do the same log transform).

#     Compute per-sequence percentiles for pointmap normalization.

#     CRITICAL: Must apply alpha/beta scaling BEFORE computing percentiles to match
#     the normalization pipeline (see normalize_pointcloud_logspace_global).

#     Args:
#         pts_list: List of tensors, each (1, H, W, 3)
#         valid_list: List of tensors, each (1, H, W)
#         common_scale: Scale factor (mean L2 norm)
#         p_low: Lower percentile (default: 0.5)
#         p_high: Upper percentile (default: 99.5)
#         alpha: Scaling factor for x-coordinate (default: 1.0, for L2 norm-based scale)
#         beta: Scaling factor for y-coordinate (default: 1.0, for L2 norm-based scale)

#     Returns:
#         q_low: [x_low, y_low, z_low] per-channel percentiles
#         q_high: [x_high, y_high, z_high] per-channel percentiles

#     Recommended replacement:
#         normed_pts, q_low, q_high = normalize_pointcloud_logspace_global(
#             pts_list, valid_list, s_seq=common_scale,
#             compute_percentiles=True, p_low=p_low, p_high=p_high
#         )
#     """
#     from diffsynth.utils.geometry import _signed_log1p

#     # Collect transformed values per channel
#     collected = {'x': [], 'y': [], 'z': []}

#     for pts, valid in zip(pts_list, valid_list):
#         # Normalize by common scale
#         pts_norm = pts / common_scale

#         # Apply alpha/beta scaling to X/Y (MUST match normalization pipeline!)
#         # pts_norm[..., 0] = pts_norm[..., 0] * float(alpha)
#         # pts_norm[..., 1] = pts_norm[..., 1] * float(beta)

#         # Apply inverse of inv_log transform
#         y = _signed_log1p(pts_norm)

#         # Get valid points
#         y_valid = y[valid]  # (N, 3)

#         if y_valid.numel() == 0:
#             continue

#         # Collect per channel
#         for ch_idx, ch_name in enumerate(['x', 'y', 'z']):
#             collected[ch_name].append(y_valid[:, ch_idx].cpu().numpy())

#     # Concatenate and compute percentiles
#     q_low = []
#     q_high = []
#     # st()
#     for ch_name in ['x', 'y', 'z']:
#         if collected[ch_name]:
#             samples = np.concatenate(collected[ch_name])
#             q_low.append(float(np.percentile(samples, p_low)))
#             q_high.append(float(np.percentile(samples, p_high)))
#         else:
#             q_low.append(0.0)
#             q_high.append(1.0)

#     return q_low, q_high


# =============================================================================
# Track Residual Computation and Normalization
# =============================================================================
# Shared utilities for training/inference/visualization consistency.
# Used by:
# - Training: diffsynth/core/data/stream3r_adapter.py
# - Inference: diffsynth/pipelines/wan_video_v2v.py
# - Visualization: examples/wanvideo/model_inference/visualize_kubric_track_vae_roundtrip_newdata.py
# =============================================================================


def compute_track_residual(
    track: torch.Tensor,
    xyz_video_cond: torch.Tensor,
    query_idx: int,
) -> torch.Tensor:
    """
    Compute track residual: track - xyz_query.

    Track[t] shows where query frame pixels move to at time t.
    xyz_query is the pointmap at the query frame.
    For static scenes: track[t] ≈ xyz_query → residual ≈ 0.

    Args:
        track: (T, H, W, 3) or (B, T, H, W, 3) track positions
        xyz_video_cond: (T, H, W, 3) or (B, T, H, W, 3) per-frame pointmaps
        query_idx: Index of the query frame (typically 0)

    Returns:
        residual: Same shape as track, residual = track - xyz_query
    """
    # Handle both batched and unbatched inputs
    if track.ndim == 4:  # (T, H, W, 3)
        xyz_query = xyz_video_cond[query_idx:query_idx+1]  # (1, H, W, 3)
        residual = track - xyz_query  # Broadcasts to (T, H, W, 3)
    elif track.ndim == 5:  # (B, T, H, W, 3)
        xyz_query = xyz_video_cond[:, query_idx:query_idx+1]  # (B, 1, H, W, 3)
        residual = track - xyz_query  # Broadcasts to (B, T, H, W, 3)
    else:
        raise ValueError(f"track must be 4D (T,H,W,3) or 5D (B,T,H,W,3), got {track.ndim}D")

    return residual


def compute_signed_log_quantiles(
    data: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel quantiles after signed-log transform.

    Args:
        data: (..., 3) tensor
        valid_mask: Optional boolean mask for valid pixels
        q_low: Lower quantile (default 0.02 for 2%)
        q_high: Upper quantile (default 0.98 for 98%)

    Returns:
        ql: (3,) tensor of lower quantiles per channel
        qh: (3,) tensor of upper quantiles per channel
    """
    # Apply signed log transform
    y = _signed_log1p(data)

    # Flatten spatial dimensions, keep channels separate
    if valid_mask is not None:
        # Expand mask to match data shape if needed
        if valid_mask.ndim == data.ndim - 1:
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(data)
        else:
            valid_mask_expanded = valid_mask
        y_flat = y[valid_mask_expanded].reshape(-1, 3)
    else:
        y_flat = y.reshape(-1, 3)

    if y_flat.numel() == 0:
        return torch.zeros(3, device=data.device), torch.ones(3, device=data.device)

    q_low_f = float(q_low)
    q_high_f = float(q_high)
    if q_low_f <= 0.0 and q_high_f >= 1.0:
        # Fast path for signed_log_p0_p100 (no quantile needed): just use min/max.
        ql = y_flat.amin(dim=0)
        qh = y_flat.amax(dim=0)
    else:
        ql = torch.quantile(y_flat.float(), q_low_f, dim=0)
        qh = torch.quantile(y_flat.float(), q_high_f, dim=0)

    return ql, qh


def normalize_track_signed_log(
    track: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    ql: Optional[torch.Tensor] = None,
    qh: Optional[torch.Tensor] = None,
    q_low: float = 0.02,
    q_high: float = 0.98,
    zero_center: bool = False,
    post_scale: float = 1.0,
    compute_stats_from: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Normalize track using signed-log transform with quantile-based scaling.

    Args:
        track: (..., 3) tensor to normalize
        valid_mask: Optional boolean mask for valid pixels
        ql, qh: Pre-computed quantiles (if None, computed from data)
        q_low, q_high: Quantile values (default 2%-98%)
        zero_center: If True, use symmetric normalization (for residual mode)
                    This ensures x=0 → normalized=0
        post_scale: Extra scaling factor applied after normalization (for residual mode)
        compute_stats_from: If provided, compute stats from this tensor instead of track

    Returns:
        normalized: Same shape as track, values in [-1, 1]
        params: dict with normalization parameters for denormalization
    """
    # Compute quantiles if not provided
    if ql is None or qh is None:
        stats_source = compute_stats_from if compute_stats_from is not None else track
        stats_mask = valid_mask
        ql, qh = compute_signed_log_quantiles(stats_source, stats_mask, q_low, q_high)

    # Apply signed log transform
    y = _signed_log1p(track)

    # Reshape quantiles for broadcasting
    ql_bc = ql.view(*([1] * (track.ndim - 1)), 3).to(track.device, track.dtype)
    qh_bc = qh.view(*([1] * (track.ndim - 1)), 3).to(track.device, track.dtype)

    denom = qh_bc - ql_bc
    denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)

    if zero_center:
        # Zero-centered normalization: scale only, no translation
        # This ensures x=0 → y=0 → normalized=0 (important for residual)
        scale = denom / 2.0
        normalized = y / scale
        if post_scale != 1.0:
            normalized = normalized * post_scale
        normalized = normalized.clamp(-1.0, 1.0)
    else:
        # Standard PPD-style normalization: shift to [0, 1] then to [-1.0, 0.5]
        normalized = (y - ql_bc) / denom  # [0, 1]
        normalized = normalized.clamp(-0.5, 1.0) - 0.5  # [-1.0, 0.5]

    # Zero out invalid pixels
    if valid_mask is not None:
        mask_bc = valid_mask.unsqueeze(-1) if valid_mask.ndim == track.ndim - 1 else valid_mask
        normalized = torch.where(mask_bc, normalized, torch.zeros_like(normalized))

    params = {
        "norm_type": ("signed_log_p0_p100" if float(q_low) <= 0.0 and float(q_high) >= 1.0 else "signed_log_p2_p98"),
        "ql": ql.cpu().tolist() if isinstance(ql, torch.Tensor) else ql,
        "qh": qh.cpu().tolist() if isinstance(qh, torch.Tensor) else qh,
        "q_low": q_low,
        "q_high": q_high,
        "zero_center": zero_center,
        "post_scale": post_scale,
    }

    return normalized, params


def denormalize_track_signed_log(
    normalized: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """
    Denormalize track (inverse of normalize_track_signed_log).

    Args:
        normalized: Normalized tensor
        params: dict from normalize_track_signed_log

    Returns:
        track: Denormalized tensor in original space
    """
    ql = torch.tensor(params["ql"], device=normalized.device, dtype=normalized.dtype)
    qh = torch.tensor(params["qh"], device=normalized.device, dtype=normalized.dtype)
    zero_center = params.get("zero_center", False)
    post_scale = float(params.get("post_scale", 1.0))

    # Reshape for broadcasting
    ql_bc = ql.view(*([1] * (normalized.ndim - 1)), 3)
    qh_bc = qh.view(*([1] * (normalized.ndim - 1)), 3)

    denom = qh_bc - ql_bc
    denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)

    if zero_center:
        # Reverse zero-centered normalization
        scale = denom / 2.0
        k = post_scale if post_scale > 0 else 1.0
        y = normalized * scale / k
    else:
        # Reverse PPD-style normalization
        y_norm = (normalized + 0.5).clamp(-0.5, 1.0)
        y = y_norm * denom + ql_bc

    # Inverse signed log transform
    track = _signed_expm1(y)

    return track


# ---------------------------------------------------------------------------
# Dynamic-region mask & scale (Phase A: oracle normalization)
# ---------------------------------------------------------------------------

def compute_dynamic_mask(
    track: torch.Tensor,
    threshold: float = 0.01,
) -> torch.Tensor:
    """
    Compute dynamic mask from GT 3D tracks.

    A point is dynamic if its accumulated frame-to-frame 3D motion exceeds
    *threshold*.  This matches the criterion used by the St4RTrack / WorldTrack
    evaluator (``third_party/st4rtrack_eval/evaluate_st4rtrack.py``).

    Args:
        track: (T, H, W, 3) or (T, N, 3) — 3D track positions over time.
        threshold: Accumulated-motion threshold (default 0.01).

    Returns:
        dynamic_mask: (H, W) or (N,) boolean tensor — True for dynamic points.
    """
    # frame-to-frame displacement
    motion = track[1:] - track[:-1]             # (T-1, ..., 3)
    motion_norm = motion.norm(dim=-1)            # (T-1, ...)
    total_motion = motion_norm.sum(dim=0)        # (...) = (H,W) or (N,)
    return total_motion > threshold


def compute_dynamic_common_scale(
    xyz_query: torch.Tensor,
    valid_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    fallback_stat: str = "mean",
    reduce_stat: str = "mean",
) -> float:
    """
    Compute common_scale using only dynamic (moving) query-frame points.

    ``common_scale_dyn = reduce(||xyz_query[i]||₂)`` for *i* in
    (valid ∩ dynamic), where ``reduce`` is controlled by ``reduce_stat``.
    Falls back to the full valid set if no dynamic points exist.

    Args:
        xyz_query: (H, W, 3) or (1, H, W, 3) — query-frame point cloud.
        valid_mask: (H, W) boolean — valid-pixel mask for the query frame.
        dynamic_mask: (H, W) boolean — dynamic-pixel mask (from
            :func:`compute_dynamic_mask`).
        fallback_stat: Reduction used when falling back to all valid points
            (default ``"mean"``).
        reduce_stat: Reduction over dynamic valid points. Supported:
            ``"mean"`` and ``"query_p60"`` / ``"q60"``.

    Returns:
        float — the common scale factor (≥ 1e-8).
    """
    if xyz_query.ndim == 4:
        xyz_query = xyz_query.squeeze(0)         # (H, W, 3)

    combined = valid_mask & dynamic_mask          # (H, W)
    pts = xyz_query[combined]                     # (M, 3)

    if pts.numel() == 0:
        # fallback: all valid points
        pts = xyz_query[valid_mask]
        if pts.numel() == 0:
            return 1.0

    norms = pts.norm(dim=-1)                     # (M,)
    stat = str(reduce_stat or "mean").strip().lower()
    if stat == "mean":
        return max(float(norms.mean().item()), 1e-8)
    if stat in {"query_p60", "q60", "p60"}:
        return max(float(torch.quantile(norms, 0.60).item()), 1e-8)
    raise ValueError(f"Unsupported dynamic reduce_stat: {reduce_stat!r}")


def compute_foreground_mask_target(
    track: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.01,
) -> torch.Tensor:
    """
    Build a binary foreground / dynamic mask target for training.

    Mapping:
        - dynamic foreground → ``+1``
        - static background / invalid → ``-1``

    The mask is constant across time (a pixel is either dynamic or not for
    the whole sequence).

    Args:
        track: (T, H, W, 3) — GT 3D track positions.
        valid_mask: (H, W) or (T, H, W) — validity mask.  If ``None``,
            all pixels are considered valid.
        threshold: Motion threshold forwarded to
            :func:`compute_dynamic_mask`.

    Returns:
        mask_target: (1, T, H, W) float tensor in {-1, +1}.
    """
    T = track.shape[0]
    dynamic = compute_dynamic_mask(track, threshold)  # (H, W)

    mask = torch.full((1, T, *dynamic.shape), -1.0, dtype=track.dtype,
                      device=track.device)
    mask[:, :, dynamic] = 1.0

    if valid_mask is not None:
        if valid_mask.ndim == 2:                 # (H, W)
            mask[:, :, ~valid_mask] = -1.0
        elif valid_mask.ndim == 3:               # (T, H, W)
            mask[0][~valid_mask] = -1.0

    return mask


def compute_weighted_common_scale(
    xyz: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> float:
    """
    Compute common_scale as a weighted mean of point norms.

    This is the inference-time counterpart of :func:`compute_dynamic_common_scale`:
    instead of a binary GT dynamic mask, a continuous weight map (predicted by
    the model) gates which points contribute to the scale.

    ``scale = sum(w_i * ||xyz_i||) / sum(w_i)``  over valid pixels.

    Args:
        xyz: (H, W, 3) or (1, H, W, 3) — condition point cloud.
        weights: (H, W) — per-pixel weight in [0, 1].
        valid_mask: (H, W) boolean — optional validity mask.
        eps: Minimum denominator / output floor.

    Returns:
        float — the weighted common scale (≥ eps).
    """
    if xyz.ndim == 4:
        xyz = xyz.squeeze(0)

    norms = xyz.norm(dim=-1)                     # (H, W)
    w = weights.to(norms.dtype)

    if valid_mask is not None:
        w = w * valid_mask.float()

    denom = w.sum()
    if denom < eps:
        # fallback: unweighted mean over valid
        if valid_mask is not None:
            pts = norms[valid_mask]
        else:
            pts = norms.reshape(-1)
        return max(float(pts.mean().item()), eps) if pts.numel() > 0 else 1.0

    return max(float((w * norms).sum().item() / denom.item()), eps)
