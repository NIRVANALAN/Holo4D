"""
Build target-view track sequences for the PointTrackEmbedder.

Takes GT 3D tracks + target cameras → produces:
  - track_seq: (B, N, T, 5) with (frame_idx, world_x, world_y, world_z, vis)
    for the PointTrackEmbedder (embeds 3D world coordinates directly)
  - track_uv_pixel: (B, N, T, 2) projected pixel coords for the PointTrackAdapter
    (spatial block assignment in latent space)
  - track_vis: (B, N, T) visibility mask

Why 3D world coords for embedding (not projected 2D):
  - Our input is dense 3D tracks (like DaS), not sparse 2D tracks (like CoTracker)
  - 3D world coordinates are camera-invariant — richer signal for track identity
  - Sinusoidal + MLP preserves high-frequency spatial info better than VAE
    (per Koichi: "RGB encoding & pre-trained VAE discard high-frequency signals")
  - 2D projection still used by adapter for spatial placement in latent grid
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from holo4d.data.track_reprojection import project_tracks_to_target_views


def _sample_dense_at_track_uv(
    dense: torch.Tensor,        # (B, T, H, W, C)
    track_uv: torch.Tensor,     # (B, N, T, 2) pixel (u=col, v=row)
    image_h: int,
    image_w: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Bilinear-sample a dense target-space map at each track's target pixel.

    Returns (B, N, T, C). Used to read the fg/bg-hybrid-fused reprojected RGB (and
    its validity) at each track identity's projected target-UV — the ab28
    "channelized Vista4D reprojection" signal. Out-of-frame samples read 0 via
    grid_sample zero padding (validity handles them downstream).
    """
    B, T, H, W, C = dense.shape
    # (B*T, C, H, W)
    inp = dense.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
    u = track_uv[..., 0]  # (B, N, T)
    v = track_uv[..., 1]
    gx = 2.0 * u / max(float(W - 1), 1.0) - 1.0
    gy = 2.0 * v / max(float(H - 1), 1.0) - 1.0
    # grid per (B,T): (B*T, N, 1, 2), with N as the sampling dimension
    grid = torch.stack([gx, gy], dim=-1)               # (B, N, T, 2)
    grid = grid.permute(0, 2, 1, 3).reshape(B * T, -1, 1, 2)  # (B*T, N, 1, 2)
    out = F.grid_sample(
        inp.to(dtype=grid.dtype), grid,
        mode=mode, padding_mode="zeros", align_corners=True,
    )                                                  # (B*T, C, N, 1)
    out = out.reshape(B, T, C, -1).permute(0, 3, 1, 2)  # (B, N, T, C)
    return out


def compact_track_tensors_by_visibility(
    track_seq: torch.Tensor,
    track_uv_pixel: torch.Tensor,
    track_vis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop track identities that are invisible/invalid for every frame.

    Sparse backendmix samples keep a dense identity dimension for dataloader
    compatibility, but downstream adapters only need identities that are valid
    in at least one target frame.  The keep mask is a batch-union over B and T
    so returned tensors remain rectangular for arbitrary batch size.
    """
    if track_vis.ndim != 3:
        raise ValueError(f"track_vis must be (B,N,T), got {tuple(track_vis.shape)}")
    identity_keep = track_vis.bool().any(dim=(0, 2))
    if identity_keep.numel() == 0 or bool(identity_keep.all()):
        return track_seq, track_uv_pixel, track_vis, identity_keep
    return (
        track_seq[:, identity_keep],
        track_uv_pixel[:, identity_keep],
        track_vis[:, identity_keep],
        identity_keep,
    )


def apply_random_track_point_dropout(
    track_vis: torch.Tensor,
    min_frac: float,
    max_frac: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Randomly zero out a fraction of whole track identities' visibility.

    Robustness augmentation for dense (legacy_uniform_order) tracking: at
    inference the intended use is dropping the bottom-confidence fraction of
    a dense candidate pool (e.g. low-confidence CoWTracker/VGGT points), not
    subsampling to a different fixed density like the K8192 canonical arms
    or the variable-N range. This drops a per-call fraction sampled uniformly
    from [min_frac, max_frac] of *entire* identities (all frames of that
    track set invisible), not scattered per-(identity, frame) cells --
    matching "this point was never confidently tracked", not "this point
    flickered out for a few frames".

    track_vis: (B, N, T) visibility mask, modified out-of-place (a clone is
    returned; the input tensor is not mutated).
    """
    if track_vis.ndim != 3:
        raise ValueError(f"track_vis must be (B,N,T), got {tuple(track_vis.shape)}")
    if not (0.0 <= min_frac <= max_frac <= 1.0):
        raise ValueError(f"require 0 <= min_frac <= max_frac <= 1, got [{min_frac}, {max_frac}]")
    if max_frac == 0.0:
        return track_vis
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    b, n, _t = track_vis.shape
    out = track_vis.clone()
    for bi in range(b):
        drop_frac = float(
            torch.empty((), device="cpu").uniform_(
                min_frac, max_frac, generator=generator
            ).item()
        )
        n_drop = int(round(drop_frac * n))
        if n_drop <= 0:
            continue
        perm = torch.randperm(n, generator=generator, device="cpu")
        drop_idx = perm[:n_drop].to(device=track_vis.device)
        out[bi, drop_idx, :] = False
    return out


def fix_track_identity_count(
    track_seq: torch.Tensor,
    track_uv_pixel: torch.Tensor,
    track_vis: torch.Tensor,
    target_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniformly subsample or pad track identities to a fixed adapter length.

    Keeping ``N`` constant prevents the Conv3D adapter and distributed
    collectives from ratcheting allocator/communication buffers as backendmix
    q/s sampling changes the compacted identity count.
    """
    target_count = int(target_count)
    if target_count <= 0:
        return track_seq, track_uv_pixel, track_vis
    if track_vis.ndim != 3:
        raise ValueError(f"track_vis must be (B,N,T), got {tuple(track_vis.shape)}")
    n = int(track_vis.shape[1])
    if n == target_count:
        return track_seq, track_uv_pixel, track_vis
    if n > target_count:
        # Deterministic uniform coverage across the compacted identity order.
        idx = torch.linspace(0, n - 1, target_count, device=track_vis.device)
        idx = idx.round().to(dtype=torch.long).clamp_(0, n - 1)
        return track_seq[:, idx], track_uv_pixel[:, idx], track_vis[:, idx]

    pad_n = target_count - n
    seq_pad = track_seq.new_zeros((track_seq.shape[0], pad_n, track_seq.shape[2], track_seq.shape[3]))
    uv_pad = track_uv_pixel.new_zeros((track_uv_pixel.shape[0], pad_n, track_uv_pixel.shape[2], track_uv_pixel.shape[3]))
    vis_pad = track_vis.new_zeros((track_vis.shape[0], pad_n, track_vis.shape[2]))
    return (
        torch.cat([track_seq, seq_pad], dim=1),
        torch.cat([track_uv_pixel, uv_pad], dim=1),
        torch.cat([track_vis, vis_pad], dim=1),
    )


def sample_track_identities_query_balanced(
    track_seq: torch.Tensor,
    track_uv_pixel: torch.Tensor,
    track_vis: torch.Tensor,
    target_count: int,
    query_idx_per_identity: Optional[torch.Tensor],
    *,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly select a fixed identity budget with equal per-query quotas.

    Sampling is independent per batch item. Only identities visible in at least
    one frame are candidates. A query that cannot fill its quota contributes all
    candidates, then the deficit is redistributed across the remaining union.
    Missing rows are padded with all-zero/all-invalid identities.

    Returns the selected tensors plus ``(B,K)`` query ids and source row indices.
    Padded rows use ``-1`` for both metadata tensors.
    """
    if track_vis.ndim != 3:
        raise ValueError(f"track_vis must be (B,N,T), got {tuple(track_vis.shape)}")
    if track_seq.ndim != 4 or track_uv_pixel.ndim != 4:
        raise ValueError("track_seq and track_uv_pixel must be batched 4D tensors")
    if track_seq.shape[:3] != track_vis.shape or track_uv_pixel.shape[:3] != track_vis.shape:
        raise ValueError("track_seq/track_uv_pixel/track_vis shapes are not row-aligned")

    target_count = int(target_count)
    if target_count <= 0:
        empty_meta = torch.empty(
            (track_vis.shape[0], 0), device=track_vis.device, dtype=torch.long,
        )
        return track_seq[:, :0], track_uv_pixel[:, :0], track_vis[:, :0], empty_meta, empty_meta

    B, N, _ = track_vis.shape
    if query_idx_per_identity is None:
        query_ids = torch.zeros((B, N), device=track_vis.device, dtype=torch.long)
    else:
        query_ids = query_idx_per_identity.to(device=track_vis.device, dtype=torch.long)
        if query_ids.ndim == 1:
            query_ids = query_ids.unsqueeze(0)
        if query_ids.shape[0] == 1 and B > 1:
            query_ids = query_ids.expand(B, -1)
        if tuple(query_ids.shape) != (B, N):
            raise ValueError(
                "query_idx_per_identity must match tracks as (B,N): "
                f"got {tuple(query_ids.shape)}, expected {(B, N)}"
            )

    def _randperm(n: int, *, batch_idx: int) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0,), device=track_vis.device, dtype=torch.long)
        if seed is None:
            return torch.randperm(n, device=track_vis.device)
        generator = torch.Generator(device=track_vis.device)
        generator.manual_seed(int(seed) + int(batch_idx) * 1_000_003)
        return torch.randperm(n, device=track_vis.device, generator=generator)

    selected_rows = []
    selected_queries = []
    candidate_mask = track_vis.bool().any(dim=2)
    for b in range(B):
        candidates = torch.where(candidate_mask[b])[0]
        if candidates.numel() == 0:
            selected_rows.append(torch.full((target_count,), -1, device=track_vis.device, dtype=torch.long))
            selected_queries.append(torch.full((target_count,), -1, device=track_vis.device, dtype=torch.long))
            continue

        candidate_queries = query_ids[b, candidates]
        active_queries = torch.unique(candidate_queries, sorted=True)
        q_order = active_queries[_randperm(int(active_queries.numel()), batch_idx=b)]
        base_quota, remainder = divmod(target_count, max(int(q_order.numel()), 1))
        chosen_parts = []
        chosen_mask = torch.zeros((N,), device=track_vis.device, dtype=torch.bool)
        for q_pos, q in enumerate(q_order):
            group = candidates[candidate_queries == q]
            quota = base_quota + (1 if q_pos < remainder else 0)
            take = min(int(group.numel()), int(quota))
            if take <= 0:
                continue
            perm = _randperm(int(group.numel()), batch_idx=b + 10_000 + q_pos)
            picked = group[perm[:take]]
            chosen_parts.append(picked)
            chosen_mask[picked] = True

        chosen = (
            torch.cat(chosen_parts, dim=0)
            if chosen_parts
            else torch.empty((0,), device=track_vis.device, dtype=torch.long)
        )
        deficit = target_count - int(chosen.numel())
        if deficit > 0:
            remaining = candidates[~chosen_mask[candidates]]
            take = min(deficit, int(remaining.numel()))
            if take > 0:
                perm = _randperm(int(remaining.numel()), batch_idx=b + 20_000)
                chosen = torch.cat([chosen, remaining[perm[:take]]], dim=0)

        if chosen.numel() > 1:
            chosen = chosen[_randperm(int(chosen.numel()), batch_idx=b + 30_000)]
        pad_n = target_count - int(chosen.numel())
        if pad_n > 0:
            chosen = torch.cat(
                [chosen, torch.full((pad_n,), -1, device=track_vis.device, dtype=torch.long)],
                dim=0,
            )
        selected_rows.append(chosen)
        safe = chosen.clamp_min(0)
        q_selected = query_ids[b, safe]
        q_selected = q_selected.masked_fill(chosen < 0, -1)
        selected_queries.append(q_selected)

    selected_idx = torch.stack(selected_rows, dim=0)
    selected_query_ids = torch.stack(selected_queries, dim=0)
    safe_idx = selected_idx.clamp_min(0)
    seq_idx = safe_idx[:, :, None, None].expand(-1, -1, track_seq.shape[2], track_seq.shape[3])
    uv_idx = safe_idx[:, :, None, None].expand(-1, -1, track_uv_pixel.shape[2], track_uv_pixel.shape[3])
    vis_idx = safe_idx[:, :, None].expand(-1, -1, track_vis.shape[2])
    out_seq = torch.gather(track_seq, 1, seq_idx)
    out_uv = torch.gather(track_uv_pixel, 1, uv_idx)
    out_vis = torch.gather(track_vis, 1, vis_idx)
    padded = selected_idx < 0
    if bool(padded.any()):
        out_seq = out_seq.masked_fill(padded[:, :, None, None], 0)
        out_uv = out_uv.masked_fill(padded[:, :, None, None], 0)
        out_vis = out_vis.masked_fill(padded[:, :, None], False)
    return out_seq, out_uv, out_vis, selected_query_ids, selected_idx


def build_target_view_tracks(
    tracks_world: torch.Tensor,
    target_cam2world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_h: int,
    image_w: int,
    source_depth_at_tracks: torch.Tensor = None,
    source_depth_valid_at_tracks: torch.Tensor = None,
    include_reproj_depth: bool = False,
    source_rgb_at_tracks: torch.Tensor = None,
    include_track_anchor: bool = False,
    tracks_world_valid_mask: torch.Tensor = None,
    track_coordinate_mode: str = "world_xyz",
    source_dynamic_mask_at_tracks: torch.Tensor = None,
    include_source_dynamic_mask: bool = False,
    target_dynamic_mask_at_tracks: torch.Tensor = None,
    include_target_dynamic_mask: bool = False,
    reprojected_rgb_dense: torch.Tensor = None,
    reprojected_rgb_valid: torch.Tensor = None,
    include_reprojected_rgb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build target-view track sequences for embedder + adapter.

    Args:
        tracks_world: (B, N, T, 3) or (N, T, 3) — 3D track trajectories in world space.
        target_cam2world: (T, 4, 4) or (B, T, 4, 4) — target camera poses.
        intrinsics: (3, 3) or (T, 3, 3) or (B, T, 3, 3) — camera intrinsics.
        image_h: target image height.
        image_w: target image width.
        source_depth_at_tracks: (B, N, T) or (N, T) or None — explicit reprojected depth
            from dataset (e.g., Syn4D full source depth reprojection). Overrides auto-compute.
        source_depth_valid_at_tracks: (B, N, T) or (N, T) or None — validity for explicit depth.
        include_reproj_depth: if True and source_depth_at_tracks is None, auto-compute
            target-camera depth from tracks_world + target_cam2world. Set True when using
            6-channel track config (default_conv3d_patchify_srcdepth.yaml).
        source_rgb_at_tracks: (B, N, T, 3) or (N, T, 3) or None — source RGB color at each
            track grid position, normalized to [0, 1]. Provides appearance hints for co-visible
            regions. Appends 3 channels (R, G, B) after all other channels.
        include_track_anchor: if True, append 3 channels for each track's per-track mean world
            position (camera-invariant identity signal, constant across time per track).
            Produces in_channels 11 (anchor+RGB) or 12 (anchor+depth+RGB).
            Channel layout with anchor: (frame_idx, wx, wy, wz, vis, ax, ay, az, [depth], [R,G,B])
        tracks_world_valid_mask: optional (B,N,T) or (N,T) mask. False entries are treated as
            absent tracks even if the coordinates project into the target camera. This lets
            sparse-tracking training keep the fixed dense-grid tensor shape expected by the
            conv3d adapter while using the adapter null token for unobserved blocks.
        track_coordinate_mode: base 3-channel coordinate representation:
            ``world_xyz`` = normalized camera-0 XYZ (legacy default);
            ``target_uvz`` = normalized target-view pixel u/v plus target-camera z.

    Returns:
        track_seq: (B, N, T, C) — C per channel layout:
            5  = base (frame_idx_norm, coord0, coord1, coord2, vis)
            6  = +depth
            8  = +RGB
            9  = +depth+RGB
            11 = +anchor+RGB
            12 = +anchor+depth+RGB
        track_uv_pixel: (B, N, T, 2) — pixel coordinates for adapter.
        track_vis: (B, N, T) — bool visibility mask.

    Note on normalization (TODO-3 fix):
        ``world_xyz`` normalizes XYZ over ALL N*T track points per sample, making
        it target-camera-invariant. ``target_uvz`` instead exposes the projected
        target-view layout directly while keeping the same 5-channel contract.
    """
    # Handle unbatched input
    if tracks_world.ndim == 3:
        tracks_world = tracks_world.unsqueeze(0)
    if source_depth_at_tracks is not None and source_depth_at_tracks.ndim == 2:
        source_depth_at_tracks = source_depth_at_tracks.unsqueeze(0)
    if tracks_world_valid_mask is not None and tracks_world_valid_mask.ndim == 2:
        tracks_world_valid_mask = tracks_world_valid_mask.unsqueeze(0)

    # Reproject to get 2D pixel coords and visibility
    track_uv, track_vis, _ = project_tracks_to_target_views(
        tracks_world, target_cam2world, intrinsics, image_h, image_w
    )
    # track_uv: (B, N, T, 2) — (u=col, v=row) in pixel space
    # track_vis: (B, N, T) — bool

    B, N, T, _ = tracks_world.shape
    if tracks_world_valid_mask is not None:
        explicit_valid = tracks_world_valid_mask.to(device=tracks_world.device, dtype=torch.bool)
        if explicit_valid.shape[0] == 1 and B > 1:
            explicit_valid = explicit_valid.expand(B, -1, -1)
        if explicit_valid.shape != (B, N, T):
            raise ValueError(
                f"tracks_world_valid_mask shape {tuple(explicit_valid.shape)} != "
                f"(B,N,T)=({B},{N},{T})"
            )
        track_vis = track_vis & explicit_valid

    # Frame indices normalized to [0, 1]
    frame_idx = torch.arange(T, device=tracks_world.device, dtype=tracks_world.dtype)
    frame_idx = frame_idx.view(1, 1, T).expand(B, N, T) / max(T - 1, 1)

    world_xyz = tracks_world  # (B, N, T, 3)

    # Normalize world XYZ over ALL N*T track points (target-camera-invariant).
    # Using all tracks (not just target-visible) ensures the same world track gets
    # the same normalized features regardless of which target camera is active.
    xyz_flat = world_xyz.reshape(B, -1, 3)          # (B, N*T, 3)
    xyz_min = xyz_flat.min(dim=1).values             # (B, 3)
    xyz_max = xyz_flat.max(dim=1).values             # (B, 3)
    xyz_range = (xyz_max - xyz_min).clamp(min=1e-6)  # (B, 3)

    world_norm = (world_xyz - xyz_min.view(B, 1, 1, 3)) / xyz_range.view(B, 1, 1, 3)
    world_norm = world_norm.clamp(0, 1)

    vis_float = track_vis.float()  # (B, N, T)
    world_norm = world_norm * vis_float.unsqueeze(-1)  # zero invisible entries

    def _target_camera_z() -> torch.Tensor:
        c2w = target_cam2world
        if c2w.ndim == 3 and c2w.shape[0] == T:
            c2w_b = c2w.unsqueeze(0).expand(B, -1, -1, -1)  # (B, T, 4, 4)
        else:
            c2w_b = c2w
            if c2w_b.ndim == 4 and c2w_b.shape[0] == 1 and B > 1:
                c2w_b = c2w_b.expand(B, -1, -1, -1)
        w2c = torch.linalg.inv(c2w_b)  # (B, T, 4, 4)
        r_tgt = w2c[:, :, :3, :3]  # (B, T, 3, 3)
        t_tgt = w2c[:, :, :3, 3]   # (B, T, 3)
        pts_tgt = torch.einsum("btij,bntj->bnti", r_tgt, world_xyz) + t_tgt[:, None, :, :]
        return pts_tgt[:, :, :, 2].clamp(min=0) * vis_float

    def _normalize_depth(sd: torch.Tensor) -> torch.Tensor:
        sd_flat = sd.reshape(B, -1)
        sd_max = sd_flat.max(dim=1).values.view(B, 1, 1).clamp(min=1e-6)
        return (sd / sd_max).clamp(0, 1)

    track_coordinate_mode = str(track_coordinate_mode or "world_xyz").lower()
    if track_coordinate_mode in {"world", "world_xyz", "cam0", "cam0_xyz"}:
        coord_norm = world_norm
    elif track_coordinate_mode in {"target_uvz", "uvz", "target_camera_uvz"}:
        u_norm = (track_uv[:, :, :, 0] / max(float(image_w - 1), 1.0)).clamp(0, 1)
        v_norm = (track_uv[:, :, :, 1] / max(float(image_h - 1), 1.0)).clamp(0, 1)
        z_norm = _normalize_depth(_target_camera_z())
        coord_norm = torch.stack([u_norm, v_norm, z_norm], dim=-1)
        coord_norm = coord_norm * vis_float.unsqueeze(-1)
    else:
        raise ValueError(
            "track_coordinate_mode must be 'world_xyz' or 'target_uvz', "
            f"got {track_coordinate_mode!r}"
        )

    # Stack base channels: (B, N, T, 5) = (frame_idx, coord0, coord1, coord2, vis)
    channels = [
        frame_idx.unsqueeze(-1),        # (B, N, T, 1)
        coord_norm,                      # (B, N, T, 3)
        vis_float.unsqueeze(-1),         # (B, N, T, 1)
    ]

    # Optional anchor channels: per-track mean world position (camera-invariant identity).
    # Constant across time per track — provides stable identity signal even when
    # instantaneous geometry is locally ambiguous (Koichi instance_id approximation).
    if include_track_anchor:
        vis_count = vis_float.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, N, 1)
        track_anchor = (world_xyz * vis_float.unsqueeze(-1)).sum(dim=2) / vis_count
        # (B, N, 3) — per-track mean world position over visible timesteps
        track_anchor = track_anchor.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, 3)
        anchor_norm = (track_anchor - xyz_min.view(B, 1, 1, 3)) / xyz_range.view(B, 1, 1, 3)
        anchor_norm = anchor_norm.clamp(0, 1)
        anchor_norm = anchor_norm * vis_float.unsqueeze(-1)  # zero invisible
        channels.append(anchor_norm)  # (B, N, T, 3)

    # Optional channel 6: reprojected source depth at target camera (normalized to [0, 1])
    #
    # Two modes:
    # a) Explicit reproj depth from dataset (Syn4D full source depth reprojection)
    # b) Auto-compute from tracks_world + target_cam2world (works for ALL datasets):
    #    project each tracked 3D point to target camera → z = target depth
    if include_reproj_depth and source_depth_at_tracks is not None:
        sd = source_depth_at_tracks.to(device=tracks_world.device, dtype=tracks_world.dtype)
        if sd.ndim == 2:
            sd = sd.unsqueeze(0)
        if sd.shape[0] == 1 and B > 1:
            sd = sd.expand(B, -1, -1)
        if sd.shape != (B, N, T):
            # Dynamic multi-query augmentation may expand tracks_world by
            # concatenating query-specific track grids while leaving stale
            # single-query source_depth_at_tracks in the sample. In that case,
            # explicit depth is no longer row-aligned; use the current tracks
            # and target cameras instead.
            if target_cam2world is None:
                raise ValueError(
                    "source_depth_at_tracks shape does not match tracks_world "
                    f"and target_cam2world is unavailable for fallback: "
                    f"got {tuple(sd.shape)}, expected {(B, N, T)}"
                )
            sd = _target_camera_z()
            source_depth_valid_at_tracks = None
        elif source_depth_valid_at_tracks is not None:
            sd_valid = source_depth_valid_at_tracks.to(device=tracks_world.device, dtype=torch.bool)
            if sd_valid.ndim == 2:
                sd_valid = sd_valid.unsqueeze(0)
            if sd_valid.shape[0] == 1 and B > 1:
                sd_valid = sd_valid.expand(B, -1, -1)
            if sd_valid.shape == (B, N, T):
                sd = sd * sd_valid.float()
    elif target_cam2world is not None and include_reproj_depth:
        # Auto-compute: project tracks_world[n,t] to target camera → z.
        sd = _target_camera_z()
    else:
        sd = None

    if sd is not None:
        # Per-sample normalize to [0, 1]
        sd_norm = _normalize_depth(sd)
        channels.append(sd_norm.unsqueeze(-1))  # (B, N, T, 1)

    # Optional source RGB channels: per-frame source appearance at each track position.
    # Already normalized to [0, 1] by the caller. Masked by visibility.
    if source_rgb_at_tracks is not None:
        src_rgb = source_rgb_at_tracks.to(device=tracks_world.device, dtype=tracks_world.dtype)
        if src_rgb.ndim == 3:  # (N, T, 3)
            src_rgb = src_rgb.unsqueeze(0)
        if src_rgb.shape != (B, N, T, 3):
            raise ValueError(
                "source_rgb_at_tracks must match tracks_world as (B,N,T,3): "
                f"got {tuple(src_rgb.shape)}, expected {(B, N, T, 3)}"
            )
        # ab28: replace the query-anchored track color with the fg/bg-hybrid-fused
        # + boundary-cleaned reprojected RGB, sampled bilinearly at each track's
        # target-UV. Fall back to the query-anchored src_rgb at disoccluded
        # (warp_valid=0) cells (GPT decision) rather than zeroing. Channel count is
        # unchanged (still 3 RGB) — replace in place, no adapter-width change.
        if include_reprojected_rgb and reprojected_rgb_dense is not None:
            dense = reprojected_rgb_dense.to(device=tracks_world.device, dtype=tracks_world.dtype)
            if dense.ndim == 4:  # (T, H, W, 3)
                dense = dense.unsqueeze(0)
            if dense.shape[0] == 1 and B > 1:
                dense = dense.expand(B, -1, -1, -1, -1)
            reproj = _sample_dense_at_track_uv(dense, track_uv, image_h, image_w, mode="bilinear")
            if reprojected_rgb_valid is not None:
                vmask = reprojected_rgb_valid.to(device=tracks_world.device, dtype=tracks_world.dtype)
                if vmask.ndim == 3:  # (T, H, W)
                    vmask = vmask.unsqueeze(0)
                if vmask.shape[0] == 1 and B > 1:
                    vmask = vmask.expand(B, -1, -1, -1)
                vsamp = _sample_dense_at_track_uv(
                    vmask.unsqueeze(-1), track_uv, image_h, image_w, mode="bilinear"
                )  # (B, N, T, 1)
                valid = (vsamp >= 0.5).to(dtype=reproj.dtype)  # conservative threshold
            else:
                valid = torch.ones_like(reproj[..., :1])
            src_rgb = valid * reproj + (1.0 - valid) * src_rgb  # fallback at invalid
        src_rgb = src_rgb * vis_float.unsqueeze(-1)  # zero out invisible
        channels.append(src_rgb)  # (B, N, T, 3)

    # Optional Vista4D-style dynamic-mask channels (motion mask attached per track
    # point). Each is (B,N,T) in {0,1}, masked by visibility, appended AFTER RGB so
    # all legacy channel orderings (<=12) are byte-identical. A missing mask falls
    # back to zeros (with a warning) so the feature is robust on datasets that do
    # not yet emit per-track masks — train can start before the data side lands.
    def _mask_channel(m, name):
        mm = m.to(device=tracks_world.device, dtype=tracks_world.dtype)
        if mm.ndim == 2:  # (N, T)
            mm = mm.unsqueeze(0)
        if mm.shape[0] == 1 and B > 1:
            mm = mm.expand(B, -1, -1)
        if tuple(mm.shape) != (B, N, T):
            raise ValueError(
                f"{name} must match tracks_world as (B,N,T): "
                f"got {tuple(mm.shape)}, expected {(B, N, T)}"
            )
        return (mm * vis_float).unsqueeze(-1)  # (B, N, T, 1), zero invisible

    if include_source_dynamic_mask:
        if source_dynamic_mask_at_tracks is None:
            import warnings
            warnings.warn(
                "include_source_dynamic_mask=True but source_dynamic_mask_at_tracks "
                "is None; source dynamic-mask channel will be zeros.",
                stacklevel=2,
            )
            channels.append(torch.zeros((B, N, T, 1), device=tracks_world.device, dtype=tracks_world.dtype))
        else:
            channels.append(_mask_channel(source_dynamic_mask_at_tracks, "source_dynamic_mask_at_tracks"))

    if include_target_dynamic_mask:
        if target_dynamic_mask_at_tracks is None:
            import warnings
            warnings.warn(
                "include_target_dynamic_mask=True but target_dynamic_mask_at_tracks "
                "is None; target dynamic-mask channel will be zeros.",
                stacklevel=2,
            )
            channels.append(torch.zeros((B, N, T, 1), device=tracks_world.device, dtype=tracks_world.dtype))
        else:
            channels.append(_mask_channel(target_dynamic_mask_at_tracks, "target_dynamic_mask_at_tracks"))

    track_seq = torch.cat(channels, dim=-1)

    return track_seq, track_uv, track_vis
