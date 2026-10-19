"""Target-independent identity selection in a canonical 3D gauge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
import time

import torch

from holo4d.analysis.torch_cluster_fps import (
    resolve_torch_cluster_fps,
    torch_cluster_fps_indices,
)


@dataclass(frozen=True)
class CanonicalObjectSelection:
    """Fixed-width original-row indices and selection diagnostics."""

    indices: torch.Tensor
    valid: torch.Tensor
    foreground: torch.Tensor
    foreground_candidate_counts: torch.Tensor
    background_candidate_counts: torch.Tensor
    selected_foreground_counts: torch.Tensor
    selected_background_counts: torch.Tensor
    foreground_unique_cell_counts: torch.Tensor
    background_unique_cell_counts: torch.Tensor
    foreground_raw_returned_counts: torch.Tensor
    background_raw_returned_counts: torch.Tensor
    selector_wall_time_ms: torch.Tensor
    selector_peak_allocated_bytes: torch.Tensor


def pad_canonical_object_selection(
    selection: CanonicalObjectSelection,
    padded_count: int,
) -> CanonicalObjectSelection:
    """Right-pad selected row slots without changing selection diagnostics."""
    padded_count = int(padded_count)
    current = int(selection.indices.shape[1])
    if padded_count < current:
        raise ValueError(
            f"padded_count={padded_count} cannot be smaller than selected width {current}"
        )
    if padded_count == current:
        return selection
    pad = padded_count - current
    indices = torch.cat(
        [
            selection.indices,
            torch.full(
                (selection.indices.shape[0], pad),
                -1,
                device=selection.indices.device,
                dtype=selection.indices.dtype,
            ),
        ],
        dim=1,
    )
    valid = torch.cat(
        [
            selection.valid,
            torch.zeros(
                (selection.valid.shape[0], pad),
                device=selection.valid.device,
                dtype=torch.bool,
            ),
        ],
        dim=1,
    )
    return replace(selection, indices=indices, valid=valid)


def _as_batched_tracks(tracks_world: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tracks_world.ndim == 3:
        return tracks_world.unsqueeze(0), True
    if tracks_world.ndim == 4:
        return tracks_world, False
    raise ValueError(
        "tracks_world must have shape (N,T,3) or (B,N,T,3), got "
        f"{tuple(tracks_world.shape)}"
    )


def _as_batched_identity_tensor(
    value: torch.Tensor,
    *,
    batch_size: int,
    identity_count: int,
    name: str,
) -> torch.Tensor:
    already_batched = value.ndim >= 2 and value.shape[:2] == (
        batch_size,
        identity_count,
    )
    if (
        not already_batched
        and value.ndim >= 1
        and value.shape[0] == identity_count
        and batch_size == 1
    ):
        value = value.unsqueeze(0)
    if value.ndim < 2 or value.shape[:2] != (batch_size, identity_count):
        raise ValueError(
            f"{name} must start with {(batch_size, identity_count)}, got {tuple(value.shape)}"
        )
    return value


def foreground_at_query_anchors(
    source_uv: torch.Tensor,
    query_indices: torch.Tensor,
    source_foreground_masks: torch.Tensor,
    *,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Classify identities from the dense mask at each identity's own anchor."""
    if source_uv.ndim == 3:
        source_uv = source_uv.unsqueeze(0)
    if source_uv.ndim != 4 or source_uv.shape[-1] != 2:
        raise ValueError(f"source_uv must be (B,N,T,2), got {tuple(source_uv.shape)}")
    batch_size, identity_count, frame_count, _ = source_uv.shape
    query_indices = _as_batched_identity_tensor(
        torch.as_tensor(query_indices, device=source_uv.device, dtype=torch.long),
        batch_size=batch_size,
        identity_count=identity_count,
        name="query_indices",
    )
    if query_indices.ndim != 2:
        raise ValueError(f"query_indices must be (B,N), got {tuple(query_indices.shape)}")
    if bool(((query_indices < 0) | (query_indices >= frame_count)).any()):
        raise ValueError(f"query_indices must be in [0,{frame_count}), got an out-of-range value")

    masks = torch.as_tensor(source_foreground_masks, device=source_uv.device).bool()
    if masks.ndim == 3:
        masks = masks.unsqueeze(0)
    if masks.ndim != 4 or masks.shape[0] not in {1, batch_size}:
        raise ValueError(f"source_foreground_masks must be (B,T,H,W), got {tuple(masks.shape)}")
    if masks.shape[0] == 1 and batch_size > 1:
        masks = masks.expand(batch_size, -1, -1, -1)
    if masks.shape[1] != frame_count:
        raise ValueError(f"mask/video time mismatch: {masks.shape[1]} != {frame_count}")

    height, width = masks.shape[-2:]
    expected_height, expected_width = map(int, image_hw)
    if (height, width) != (expected_height, expected_width):
        raise ValueError(
            f"mask grid {(height, width)} != image grid "
            f"{(expected_height, expected_width)}"
        )
    batch_ids = torch.arange(batch_size, device=source_uv.device)[:, None]
    identity_ids = torch.arange(identity_count, device=source_uv.device)[None, :]
    anchors = source_uv[batch_ids, identity_ids, query_indices]
    finite = torch.isfinite(anchors).all(dim=-1)
    xy = torch.nan_to_num(anchors).round().long()
    inside = finite & (xy[..., 0] >= 0) & (xy[..., 0] < width)
    inside &= (xy[..., 1] >= 0) & (xy[..., 1] < height)
    x = xy[..., 0].clamp(0, width - 1)
    y = xy[..., 1].clamp(0, height - 1)
    return inside & masks[batch_ids, query_indices, y, x]


def foreground_from_row_aligned_mask(
    source_dynamic_mask_at_tracks: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    batch_size: int,
    identity_count: int,
    frame_count: int,
) -> torch.Tensor:
    """Read a row-aligned dynamic mask at each identity's query anchor."""
    masks = _as_batched_identity_tensor(
        torch.as_tensor(source_dynamic_mask_at_tracks).bool(),
        batch_size=batch_size,
        identity_count=identity_count,
        name="source_dynamic_mask_at_tracks",
    )
    if masks.ndim == 2:
        return masks
    if masks.ndim != 3 or masks.shape[2] != frame_count:
        raise ValueError(
            "source_dynamic_mask_at_tracks must be (B,N) or (B,N,T), got "
            f"{tuple(masks.shape)}"
        )
    queries = _as_batched_identity_tensor(
        torch.as_tensor(query_indices, device=masks.device, dtype=torch.long),
        batch_size=batch_size,
        identity_count=identity_count,
        name="query_indices",
    )
    if bool(((queries < 0) | (queries >= frame_count)).any()):
        raise ValueError(f"query_indices must be in [0,{frame_count})")
    return masks.gather(2, queries.unsqueeze(-1)).squeeze(-1)


def _allocate_pool_budgets(
    foreground_count: int,
    background_count: int,
    target_count: int,
    foreground_fraction: float,
) -> tuple[int, int]:
    total = min(max(int(target_count), 0), foreground_count + background_count)
    if foreground_count == 0:
        foreground_budget = 0
    else:
        foreground_budget = min(
            foreground_count,
            max(1, int(round(total * min(max(float(foreground_fraction), 0.0), 1.0)))),
        )
    background_budget = min(background_count, total - foreground_budget)
    missing = total - foreground_budget - background_budget
    foreground_budget += min(max(foreground_count - foreground_budget, 0), missing)
    missing = total - foreground_budget - background_budget
    background_budget += min(max(background_count - background_budget, 0), missing)
    return foreground_budget, background_budget


def _exact_fps(points: torch.Tensor, identity_ids: torch.Tensor, count: int) -> torch.Tensor:
    count = min(max(int(count), 0), int(identity_ids.numel()))
    if count == 0:
        return identity_ids.new_empty((0,))
    xyz = points[identity_ids].float()
    first = torch.sum((xyz - xyz.mean(dim=0, keepdim=True)) ** 2, dim=1).argmax()
    chosen = torch.empty((count,), device=identity_ids.device, dtype=torch.long)
    chosen[0] = first
    minimum = torch.sum((xyz - xyz[first]) ** 2, dim=1)
    minimum[first] = -1
    for output_index in range(1, count):
        next_index = minimum.argmax()
        chosen[output_index] = next_index
        distance = torch.sum((xyz - xyz[next_index]) ** 2, dim=1)
        minimum = torch.minimum(minimum, distance)
        minimum[chosen[: output_index + 1]] = -1
    return identity_ids[chosen]


def _expand_bits_3d(value: torch.Tensor) -> torch.Tensor:
    value = value.to(torch.int64) & 0x3FF
    value = (value | (value << 16)) & 0x30000FF
    value = (value | (value << 8)) & 0x300F00F
    value = (value | (value << 4)) & 0x30C30C3
    value = (value | (value << 2)) & 0x9249249
    return value


def canonical_morton_cell_codes(points: torch.Tensor, *, bits: int = 10) -> torch.Tensor:
    """Quantize one canonical XYZ pool and return its 3D Morton cell codes."""
    xyz = torch.as_tensor(points).float()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {tuple(xyz.shape)}")
    if xyz.shape[0] == 0:
        return torch.empty((0,), device=xyz.device, dtype=torch.int64)
    lo = xyz.amin(dim=0)
    extent = (xyz.amax(dim=0) - lo).clamp_min(torch.finfo(torch.float32).eps)
    levels = (1 << min(max(int(bits), 1), 10)) - 1
    quantized = (((xyz - lo) / extent) * levels).round().long().clamp(0, levels)
    return (
        _expand_bits_3d(quantized[:, 0])
        | (_expand_bits_3d(quantized[:, 1]) << 1)
        | (_expand_bits_3d(quantized[:, 2]) << 2)
    )


def _morton_stratified(
    points: torch.Tensor,
    identity_ids: torch.Tensor,
    count: int,
    *,
    bits: int,
) -> tuple[torch.Tensor, int]:
    """Select spatially spread rows in O(N log N), with duplicate suppression."""
    count = min(max(int(count), 0), int(identity_ids.numel()))
    if identity_ids.numel() == 0:
        return identity_ids.new_empty((0,)), 0
    codes = canonical_morton_cell_codes(points[identity_ids], bits=bits)
    # Stable sorting makes duplicate points deterministic in original-row order.
    order = torch.argsort(codes, stable=True)
    sorted_codes = codes[order]
    first_in_cell = torch.ones_like(sorted_codes, dtype=torch.bool)
    first_in_cell[1:] = sorted_codes[1:] != sorted_codes[:-1]
    representatives = order[first_in_cell]
    if count == 0:
        return identity_ids.new_empty((0,)), int(representatives.numel())
    if representatives.numel() >= count:
        positions = torch.div(
            torch.arange(count, device=points.device) * representatives.numel(),
            count,
            rounding_mode="floor",
        )
        selected_local = representatives[positions]
    else:
        non_representatives = order[~first_in_cell]
        selected_local = torch.cat(
            [representatives, non_representatives[: count - representatives.numel()]]
        )
    return identity_ids[selected_local], int(representatives.numel())


def select_canonical_object_identities(
    tracks_world: torch.Tensor,
    valid_mask: torch.Tensor | None,
    foreground_identity_mask: torch.Tensor | None,
    target_count: int,
    *,
    reference_frame: int = 0,
    foreground_fraction: float = 0.75,
    method: str = "morton_stratified",
    morton_bits: int = 10,
    torch_cluster_fps_fn: Callable | None = None,
    profile: bool = False,
) -> CanonicalObjectSelection:
    """Select fixed-width rows from the source-frame canonical 3D union."""
    tracks, _ = _as_batched_tracks(torch.as_tensor(tracks_world))
    batch_size, identity_count, frame_count, xyz_dim = tracks.shape
    if xyz_dim != 3:
        raise ValueError(f"tracks_world last dimension must be 3, got {xyz_dim}")
    if not 0 <= int(reference_frame) < frame_count:
        raise ValueError(f"reference_frame must be in [0,{frame_count}), got {reference_frame}")
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")
    if valid_mask is None:
        valid = torch.isfinite(tracks).all(dim=-1) & (tracks.abs().sum(dim=-1) > 0)
    else:
        valid = _as_batched_identity_tensor(
            torch.as_tensor(valid_mask, device=tracks.device).bool(),
            batch_size=batch_size,
            identity_count=identity_count,
            name="valid_mask",
        )
        if valid.shape != tracks.shape[:3]:
            raise ValueError(f"valid_mask must be {tuple(tracks.shape[:3])}, got {tuple(valid.shape)}")
    foreground = (
        torch.zeros((batch_size, identity_count), device=tracks.device, dtype=torch.bool)
        if foreground_identity_mask is None
        else _as_batched_identity_tensor(
            torch.as_tensor(foreground_identity_mask, device=tracks.device).bool(),
            batch_size=batch_size,
            identity_count=identity_count,
            name="foreground_identity_mask",
        )
    )
    if foreground.ndim != 2:
        raise ValueError(f"foreground_identity_mask must be (B,N), got {tuple(foreground.shape)}")

    method = str(method).strip().lower()
    if method not in {"exact_fps", "morton_stratified", "torch_cluster_fps"}:
        raise ValueError(f"Unsupported canonical selection method {method!r}")
    if method == "torch_cluster_fps":
        # Resolve once up front. Even an all-retained or empty pool must not
        # make an explicitly requested backend silently skip its dependency.
        torch_cluster_fps_fn = resolve_torch_cluster_fps(torch_cluster_fps_fn)
    indices = torch.full(
        (batch_size, int(target_count)), -1, device=tracks.device, dtype=torch.long
    )
    selected_foreground = torch.zeros(batch_size, device=tracks.device, dtype=torch.long)
    selected_background = torch.zeros_like(selected_foreground)
    foreground_candidates_count = torch.zeros_like(selected_foreground)
    background_candidates_count = torch.zeros_like(selected_foreground)
    foreground_unique_cell_count = torch.zeros_like(selected_foreground)
    background_unique_cell_count = torch.zeros_like(selected_foreground)
    foreground_raw_returned_count = torch.zeros_like(selected_foreground)
    background_raw_returned_count = torch.zeros_like(selected_foreground)
    selector_wall_time_ms = torch.zeros(batch_size, device=tracks.device, dtype=torch.float64)
    selector_peak_allocated_bytes = torch.zeros(
        batch_size, device=tracks.device, dtype=torch.long
    )
    reference_points = tracks[:, :, int(reference_frame)]
    eligible = valid[:, :, int(reference_frame)] & torch.isfinite(reference_points).all(dim=-1)
    for batch_index in range(batch_size):
        fg_ids = torch.nonzero(eligible[batch_index] & foreground[batch_index], as_tuple=False).flatten()
        bg_ids = torch.nonzero(eligible[batch_index] & ~foreground[batch_index], as_tuple=False).flatten()
        fg_budget, bg_budget = _allocate_pool_budgets(
            int(fg_ids.numel()), int(bg_ids.numel()), int(target_count), foreground_fraction
        )
        profile_cuda = bool(profile and reference_points.is_cuda)
        if profile_cuda:
            torch.cuda.synchronize(reference_points.device)
            memory_before = torch.cuda.memory_allocated(reference_points.device)
            torch.cuda.reset_peak_memory_stats(reference_points.device)
        else:
            memory_before = 0
        started = time.perf_counter() if profile else 0.0
        if method == "exact_fps":
            fg_selected = _exact_fps(reference_points[batch_index], fg_ids, fg_budget)
            bg_selected = _exact_fps(reference_points[batch_index], bg_ids, bg_budget)
            fg_unique_cells = int(fg_ids.numel())
            bg_unique_cells = int(bg_ids.numel())
            fg_raw_returned = int(fg_selected.numel())
            bg_raw_returned = int(bg_selected.numel())
        elif method == "torch_cluster_fps":
            fg_selected, fg_raw_returned = torch_cluster_fps_indices(
                reference_points[batch_index],
                fg_ids,
                fg_budget,
                fps_fn=torch_cluster_fps_fn,
            )
            bg_selected, bg_raw_returned = torch_cluster_fps_indices(
                reference_points[batch_index],
                bg_ids,
                bg_budget,
                fps_fn=torch_cluster_fps_fn,
            )
            fg_unique_cells = int(fg_ids.numel())
            bg_unique_cells = int(bg_ids.numel())
        else:
            fg_selected, fg_unique_cells = _morton_stratified(
                reference_points[batch_index], fg_ids, fg_budget, bits=morton_bits
            )
            bg_selected, bg_unique_cells = _morton_stratified(
                reference_points[batch_index], bg_ids, bg_budget, bits=morton_bits
            )
            fg_raw_returned = int(fg_selected.numel())
            bg_raw_returned = int(bg_selected.numel())
        if profile_cuda:
            torch.cuda.synchronize(reference_points.device)
            peak_bytes = max(
                int(torch.cuda.max_memory_allocated(reference_points.device)) - memory_before,
                0,
            )
        else:
            peak_bytes = 0
        elapsed_ms = (time.perf_counter() - started) * 1000.0 if profile else 0.0
        selected = torch.cat([fg_selected, bg_selected])
        indices[batch_index, : selected.numel()] = selected
        selected_foreground[batch_index] = fg_selected.numel()
        selected_background[batch_index] = bg_selected.numel()
        foreground_candidates_count[batch_index] = fg_ids.numel()
        background_candidates_count[batch_index] = bg_ids.numel()
        foreground_unique_cell_count[batch_index] = fg_unique_cells
        background_unique_cell_count[batch_index] = bg_unique_cells
        foreground_raw_returned_count[batch_index] = fg_raw_returned
        background_raw_returned_count[batch_index] = bg_raw_returned
        selector_wall_time_ms[batch_index] = elapsed_ms
        selector_peak_allocated_bytes[batch_index] = peak_bytes
    return CanonicalObjectSelection(
        indices=indices,
        valid=indices >= 0,
        foreground=foreground,
        foreground_candidate_counts=foreground_candidates_count,
        background_candidate_counts=background_candidates_count,
        selected_foreground_counts=selected_foreground,
        selected_background_counts=selected_background,
        foreground_unique_cell_counts=foreground_unique_cell_count,
        background_unique_cell_counts=background_unique_cell_count,
        foreground_raw_returned_counts=foreground_raw_returned_count,
        background_raw_returned_counts=background_raw_returned_count,
        selector_wall_time_ms=selector_wall_time_ms,
        selector_peak_allocated_bytes=selector_peak_allocated_bytes,
    )


def gather_identity_payload(
    payload: torch.Tensor | None,
    indices: torch.Tensor,
    *,
    expected_identity_count: int | None = None,
):
    """Gather a row-aligned payload and zero-fill selection padding."""
    if payload is None:
        return None
    value = torch.as_tensor(payload, device=indices.device)
    batch_size = indices.shape[0]
    if (
        value.ndim >= 1
        and batch_size == 1
        and (
            (expected_identity_count is not None and value.shape[0] == expected_identity_count)
            or (expected_identity_count is None and value.shape[0] != 1)
        )
    ):
        value = value.unsqueeze(0)
    if value.ndim < 2 or value.shape[0] not in {1, batch_size}:
        raise ValueError(
            f"payload must begin with (B,N), got {tuple(value.shape)} for B={batch_size}"
        )
    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size, *value.shape[1:])
    if expected_identity_count is not None and value.shape[1] != int(expected_identity_count):
        raise ValueError(
            "payload identity count mismatch: expected "
            f"{expected_identity_count}, got {value.shape[1]} for shape {tuple(value.shape)}"
        )
    safe_indices = indices.clamp_min(0)
    gather_index = safe_indices.reshape(*safe_indices.shape, *([1] * (value.ndim - 2)))
    gather_index = gather_index.expand(*safe_indices.shape, *value.shape[2:])
    selected = value.gather(1, gather_index)
    padding = (indices < 0).reshape(*indices.shape, *([1] * (value.ndim - 2)))
    return selected.masked_fill(padding, False if selected.dtype == torch.bool else 0)


def select_canonical_object_identities_query_aware(
    tracks_world: torch.Tensor,
    valid_mask: torch.Tensor | None,
    foreground_identity_mask: torch.Tensor | None,
    target_count: int,
    *,
    query_idx_per_identity: torch.Tensor,
    foreground_fraction: float = 0.75,
    method: str = "morton_stratified",
    morton_bits: int = 10,
    torch_cluster_fps_fn: Callable | None = None,
    profile: bool = False,
) -> CanonicalObjectSelection:
    """Per-query eligibility + per-query spatial selection, then a fixed merge.

    Fixes the canonical t_ref=0 eligibility limitation for multi-query (q2+)
    pools: each raw identity is selected against its OWN query anchor frame,
    so q_last-only disocclusion rows (invalid at t0) stay eligible. Queries
    are selected independently and merged under a deterministic water-filled
    fixed budget — query
    anchor XYZ from different anchor times are never mixed into one FPS/Morton
    distance pool (temporal object motion must not masquerade as spatial
    coverage). This selector intentionally performs no cross-query identity
    association or deduplication; it is a provenance-aware selection ablation,
    not the final world-trajectory fusion design. See doc/track4d-360/cross-checks/
    2026-07-22-gwtt-empty-token-ambiguity-and-tref-eligibility-verification.md.

    Contract mirrors ``select_canonical_object_identities``: same padding
    sentinel (-1), same FG-first order inside each query group, and target
    cameras never influence selection.
    """
    tracks, _ = _as_batched_tracks(torch.as_tensor(tracks_world))
    batch_size, identity_count, frame_count, _ = tracks.shape
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")
    query_idx = _as_batched_identity_tensor(
        torch.as_tensor(query_idx_per_identity),
        batch_size=batch_size,
        identity_count=identity_count,
        name="query_idx_per_identity",
    ).to(dtype=torch.long)
    if query_idx.numel() and (
        bool((query_idx < 0).any()) or bool((query_idx >= frame_count).any())
    ):
        query_min = int(query_idx.min().item())
        query_max = int(query_idx.max().item())
        raise ValueError(
            "query_idx_per_identity must contain frame indices in "
            f"[0, {frame_count - 1}], got min={query_min}, max={query_max}"
        )
    if valid_mask is not None:
        valid_mask = _as_batched_identity_tensor(
            torch.as_tensor(valid_mask),
            batch_size=batch_size,
            identity_count=identity_count,
            name="valid_mask",
        )
    if foreground_identity_mask is not None:
        foreground_identity_mask = _as_batched_identity_tensor(
            torch.as_tensor(foreground_identity_mask),
            batch_size=batch_size,
            identity_count=identity_count,
            name="foreground_identity_mask",
        )

    device = tracks.device
    indices = torch.full((batch_size, int(target_count)), -1, device=device, dtype=torch.long)
    # Same contract as the main selector: `.foreground` is the (B, N)
    # per-identity mask passthrough, NOT per selected slot.
    foreground_out = (
        torch.zeros((batch_size, identity_count), device=device, dtype=torch.bool)
        if foreground_identity_mask is None
        else foreground_identity_mask.bool().to(device=device)
    )
    zeros_b = lambda: torch.zeros((batch_size,), device=device, dtype=torch.long)  # noqa: E731
    fg_cand = zeros_b(); bg_cand = zeros_b(); fg_sel = zeros_b(); bg_sel = zeros_b()
    fg_cells = zeros_b(); bg_cells = zeros_b(); fg_raw = zeros_b(); bg_raw = zeros_b()
    wall_ms = torch.zeros((batch_size,), device=device, dtype=torch.float32)
    peak_bytes = zeros_b()

    for b in range(batch_size):
        queries = torch.unique(query_idx[b], sorted=True)
        num_queries = int(queries.numel())
        if num_queries == 0:
            continue
        # Water-filling budgets (Codex review F4): equal split alone cannot
        # fill K when candidate pools are unbalanced (e.g. q0 has 1 eligible
        # row, q48 has 20, K=10). Cap each query at its own-anchor eligible
        # capacity and redistribute the deficit deterministically until K is
        # met or every pool is exhausted. Per-query selection stays
        # independent (anchor-time XYZ are never mixed in one distance pool).
        query_list = queries.tolist()
        rows_by_query = {}
        capacity = {}
        finite_all = torch.isfinite(tracks[b]).all(dim=-1)
        for q in query_list:
            rows_q = torch.nonzero(query_idx[b] == q, as_tuple=False).flatten()
            rows_by_query[q] = rows_q
            anchor_q = int(q)
            if rows_q.numel() == 0:
                capacity[q] = 0
                continue
            eligible_q = finite_all[rows_q, anchor_q]
            if valid_mask is not None:
                eligible_q = eligible_q & valid_mask[b][rows_q, anchor_q].bool()
            else:
                # Match the base selector's derived-validity rule exactly
                # (finite AND nonzero), so capacity never overestimates.
                nonzero_q = tracks[b][rows_q, anchor_q].abs().sum(dim=-1) > 0
                eligible_q = eligible_q & nonzero_q
            capacity[q] = int(eligible_q.sum())
        budgets = {q: 0 for q in query_list}
        remaining = int(target_count)
        while remaining > 0:
            active = [q for q in query_list if budgets[q] < capacity[q]]
            if not active:
                break
            share = max(remaining // len(active), 1)
            for q in active:
                add = min(share, capacity[q] - budgets[q], remaining)
                budgets[q] += add
                remaining -= add
                if remaining == 0:
                    break
        cursor = 0
        for q in query_list:
            budget = budgets[q]
            if budget <= 0:
                continue
            rows = rows_by_query[q]
            if rows.numel() == 0:
                continue
            sub_tracks = tracks[b : b + 1, rows]
            sub_valid = None if valid_mask is None else valid_mask[b : b + 1, rows]
            sub_fg = (
                None
                if foreground_identity_mask is None
                else foreground_identity_mask[b : b + 1, rows]
            )
            # Eligibility anchored at this query's own frame, not global t0.
            anchor = int(q)
            sub_sel = select_canonical_object_identities(
                sub_tracks,
                sub_valid,
                sub_fg,
                min(budget, int(rows.numel())),
                reference_frame=anchor,
                foreground_fraction=foreground_fraction,
                method=method,
                morton_bits=morton_bits,
                torch_cluster_fps_fn=torch_cluster_fps_fn,
                profile=profile,
            )
            sub_indices = sub_sel.indices[0]
            sub_ok = sub_indices >= 0
            mapped = rows[sub_indices[sub_ok]]
            n = int(mapped.numel())
            take = min(n, int(target_count) - cursor)
            if take > 0:
                indices[b, cursor : cursor + take] = mapped[:take]
                cursor += take
            fg_cand[b] += sub_sel.foreground_candidate_counts[0]
            bg_cand[b] += sub_sel.background_candidate_counts[0]
            fg_sel[b] += sub_sel.selected_foreground_counts[0]
            bg_sel[b] += sub_sel.selected_background_counts[0]
            fg_cells[b] += sub_sel.foreground_unique_cell_counts[0]
            bg_cells[b] += sub_sel.background_unique_cell_counts[0]
            fg_raw[b] += sub_sel.foreground_raw_returned_counts[0]
            bg_raw[b] += sub_sel.background_raw_returned_counts[0]
            wall_ms[b] += sub_sel.selector_wall_time_ms[0]
            peak_bytes[b] = torch.maximum(
                peak_bytes[b], sub_sel.selector_peak_allocated_bytes[0]
            )

    return CanonicalObjectSelection(
        indices=indices,
        valid=indices >= 0,
        foreground=foreground_out,
        foreground_candidate_counts=fg_cand,
        background_candidate_counts=bg_cand,
        selected_foreground_counts=fg_sel,
        selected_background_counts=bg_sel,
        foreground_unique_cell_counts=fg_cells,
        background_unique_cell_counts=bg_cells,
        foreground_raw_returned_counts=fg_raw,
        background_raw_returned_counts=bg_raw,
        selector_wall_time_ms=wall_ms,
        selector_peak_allocated_bytes=peak_bytes,
    )
