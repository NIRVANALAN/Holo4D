"""Load a cotracker + VGGT-Omega *source* reconstruction in the 4RC-run contract.

The Vista4D-official eval acquires its source geometry/tracks from 4RC
(`load_4rc_run` -> a dict with frames/pts/tracks/c2w/K/query_idx). 4RC is low
quality on the harder Vista4D clips, so this module provides the same dict shape
from the alternative source backend: cotracker (dense 2D tracks) + VGGT-Omega
(per-frame camera + metric depth). Drop-in for `load_4rc_run`:

    run = load_cow_vggt_run(seq, cow_root=..., vggt_root=..., num_frames=...)
    source_run = transform_run_to_query_camera(run)   # unchanged downstream

On-disk inputs (precomputed by CoWTracker and VGGT-Omega):
  cotracker : <cow_root>/<seq>_pred_trackvideo.npz   (tracks_xy/visibility/confidence)
  vggt pose : <vggt_root>/<seq>/pose/vggt_omega.npz   (cam_c2w, intrinsics, frame_paths)
  vggt depth: <vggt_root>/<seq>/geometry/vggt_omega_depth.npz  (depth, depth_conf, intrinsics)

Metric scale: VGGT recovers its own scale, independent of 4RC. ab12 (and the
other 4RC-trained ablations) expect source geometry in 4RC metric units, so by
default we rescale VGGT depth/cameras/tracks to 4RC scale using the per-seq
frame-0 depth ratio (same estimator as render_cow_vggt_2x4_comparison). Pass
``scale_align="none"`` to skip (e.g. for backendmix ablations ab15/16 that were
trained on VGGT-scale geometry too).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from holo4d.tools.cowtracker_vggt_dense3d_viz import (
    bilinear_sample,
    pixel_center_resize_xy,
    unproject_depth_to_world,
)


def split_multiquery_cow_roots(cow_root: str | list[str] | tuple[str, ...]) -> list[str]:
    """Return one or more CoW cache roots from a comma/colon/list spec."""
    if isinstance(cow_root, (list, tuple)):
        parts = [str(p).strip() for p in cow_root]
    else:
        text = str(cow_root or "")
        sep = "," if "," in text else ":"
        parts = [p.strip() for p in text.split(sep)]
    roots = [p for p in parts if p]
    if not roots:
        raise ValueError("cow_root is empty")
    return roots


def _lift_cow_tracks(
    cow_npz,
    depth: np.ndarray,
    depth_conf: np.ndarray,
    intrinsics: np.ndarray,
    cam2world: np.ndarray,
    *,
    track_stride: int,
    track_samples: int,
    track_sample_mode: str,
    track_sample_seed: int,
    cow_conf_threshold: float,
    depth_conf_threshold: float,
):
    """Lift cotracker 2D tracks to 3D world points via VGGT depth/cameras.

    Ported verbatim from render_cow_vggt_2x4_comparison._lift_cow_tracks so the
    Vista4D-official cow+vggt backend matches the 2x4 diagnostic exactly.
    Returns (world (N,T,3), visibility (N,T), query_xy (N,2), confidence (N,T)).
    """
    cow_xy = np.asarray(cow_npz["tracks_xy"], dtype=np.float32)
    cow_vis = np.asarray(cow_npz["visibility"], dtype=bool)
    cow_conf = np.asarray(cow_npz["confidence"], dtype=np.float32)
    T = min(cow_xy.shape[0], depth.shape[0], intrinsics.shape[0], cam2world.shape[0])
    cow_xy = cow_xy[:T]
    cow_vis = cow_vis[:T]
    cow_conf = cow_conf[:T]
    depth = depth[:T]
    depth_conf = depth_conf[:T]
    intrinsics = intrinsics[:T]
    cam2world = cam2world[:T]
    cow_h, cow_w = cow_xy.shape[1:3]
    vggt_h, vggt_w = depth.shape[1:3]

    if track_sample_mode == "all":
        yy, xx = np.mgrid[0:cow_h, 0:cow_w]
        qy = yy.reshape(-1)
        qx = xx.reshape(-1)
    elif track_sample_mode == "grid":
        yy, xx = np.mgrid[0:cow_h:int(track_stride), 0:cow_w:int(track_stride)]
        qy = yy.reshape(-1)
        qx = xx.reshape(-1)
    elif track_sample_mode == "jittered":
        stride = max(int(track_stride), 1)
        yy, xx = np.mgrid[0:cow_h:stride, 0:cow_w:stride]
        rng = np.random.default_rng(int(track_sample_seed))
        jy = rng.integers(0, stride, size=yy.shape, endpoint=False)
        jx = rng.integers(0, stride, size=xx.shape, endpoint=False)
        qy = np.minimum(yy + jy, cow_h - 1).reshape(-1)
        qx = np.minimum(xx + jx, cow_w - 1).reshape(-1)
    else:
        total = int(cow_h * cow_w)
        n = min(max(int(track_samples), 1), total)
        rng = np.random.default_rng(int(track_sample_seed))
        flat = np.sort(rng.choice(total, size=n, replace=False)).astype(np.int64)
        qy = flat // cow_w
        qx = flat % cow_w
    query_xy = np.stack([qx, qy], axis=-1).astype(np.float32)
    cow_sample_xy = cow_xy[:, qy, qx]
    cow_sample_vis = cow_vis[:, qy, qx]
    cow_sample_conf = cow_conf[:, qy, qx]
    vggt_xy = pixel_center_resize_xy(cow_sample_xy, src_hw=(cow_h, cow_w), dst_hw=(vggt_h, vggt_w))

    world = np.zeros((query_xy.shape[0], T, 3), dtype=np.float32)
    vis = np.zeros((query_xy.shape[0], T), dtype=bool)
    for t in range(T):
        d, in_bounds = bilinear_sample(depth[t], vggt_xy[t])
        dc, conf_bounds = bilinear_sample(depth_conf[t], vggt_xy[t])
        ok = (
            cow_sample_vis[t]
            & (cow_sample_conf[t] >= float(cow_conf_threshold))
            & in_bounds
            & conf_bounds
            & np.isfinite(d)
            & (d > 1e-6)
            & np.isfinite(dc)
            & (dc >= float(depth_conf_threshold))
        )
        world[:, t] = unproject_depth_to_world(vggt_xy[t], d, intrinsics[t], cam2world[t])
        vis[:, t] = ok
    return world.astype(np.float32), vis, query_xy, cow_sample_conf.transpose(1, 0).astype(np.float32)


def _carry_last_valid_track_positions(tracks_world: np.ndarray, track_vis: np.ndarray) -> np.ndarray:
    """Keep invalid track coordinates finite without reverting to frame 0.

    The explicit ``tracks_valid`` mask is the source of truth for model input and
    visualization. This fallback only prevents NaNs in downstream normalization
    and makes accidental mask omissions degrade to "last seen" tracks instead of
    a frame-0 full-grid reset.
    """
    out = np.asarray(tracks_world, dtype=np.float32).copy()
    valid = np.asarray(track_vis, dtype=bool)
    if out.ndim != 3 or valid.shape != out.shape[:2] or out.shape[1] == 0:
        return out
    for t in range(1, out.shape[1]):
        invalid = ~valid[:, t]
        out[invalid, t] = out[invalid, t - 1]
    return out.astype(np.float32, copy=False)


def _load_4rc_query_depth_for_scale(davis_4rc_root: str, seq: str) -> np.ndarray:
    npz_path = Path(davis_4rc_root) / seq / "result.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"4RC result.npz missing for scale alignment: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as ref:
        ref_pts_q = np.asarray(ref["pred_0_pts"], dtype=np.float32)
        ref_ext_q = np.asarray(ref["pred_0_extrinsic"], dtype=np.float32)
    if ref_pts_q.ndim == 4:
        ref_pts_q = ref_pts_q[0]
    if ref_ext_q.ndim == 3 and ref_ext_q.shape[0] == 1:
        ref_ext_q = ref_ext_q[0]
    if ref_ext_q.shape == (4, 4):
        ref_ext_cw = ref_ext_q
    else:
        if ref_ext_q.shape != (3, 4):
            raise ValueError(f'{npz_path}: unexpected c2w shape {ref_ext_q.shape}')
        ref_ext_cw = np.eye(4, dtype=np.float32)
        ref_ext_cw[:3] = ref_ext_q[:3]
    h4, w4 = ref_pts_q.shape[:2]
    ones = np.ones((h4, w4, 1), dtype=np.float32)
    pts_h = np.concatenate([ref_pts_q, ones], axis=-1)
    # 4RC stores c2w; obtaining camera depth requires its inverse.
    ref_pts_cam = np.einsum("ij,hwj->hwi", np.linalg.inv(ref_ext_cw), pts_h)[..., :3]
    return ref_pts_cam[..., 2].astype(np.float32)


def _estimate_depth_scale_to_4rc(*, vggt_depth0: np.ndarray, davis_4rc_root: str, seq: str) -> float:
    ref_depth0 = _load_4rc_query_depth_for_scale(davis_4rc_root, seq)
    cmp_h, cmp_w = 144, 256
    ref_cmp = cv2.resize(ref_depth0.astype(np.float32), (cmp_w, cmp_h), interpolation=cv2.INTER_LINEAR)
    vggt_cmp = cv2.resize(np.asarray(vggt_depth0, dtype=np.float32), (cmp_w, cmp_h), interpolation=cv2.INTER_LINEAR)
    valid = (
        np.isfinite(ref_cmp) & np.isfinite(vggt_cmp) & (ref_cmp > 1e-3) & (vggt_cmp > 1e-3)
    )
    if int(valid.sum()) <= 100:
        print(f"[cow-vggt] {seq}: WARN <100 jointly-valid px for 4RC/VGGT scale align; alpha=1.0")
        return 1.0
    return float(np.median(ref_cmp[valid] / vggt_cmp[valid]))


def _load_external_scale(root: str | Path, seq: str) -> tuple[float, Path]:
    """Load a positive source-to-reference scale for one reconstructed video."""
    scale_path = Path(root) / f"{seq}.json"
    if not scale_path.is_file():
        raise FileNotFoundError(scale_path)
    payload = json.loads(scale_path.read_text())
    alpha = float(payload["scale"])
    if not np.isfinite(alpha) or alpha <= 1e-8:
        raise ValueError(f"{scale_path}: invalid external scale {alpha}")
    return alpha, scale_path


def _per_frame_world_pts(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> list[np.ndarray]:
    """Full-grid world points per frame (matches 4RC run['pts'] semantics)."""
    H, W = depth.shape[1:3]
    yy, xx = np.mgrid[0:H, 0:W]
    xy = np.stack([xx, yy], axis=-1).astype(np.float32)
    return [
        unproject_depth_to_world(xy, depth[t], K[t], c2w[t]).astype(np.float32)
        for t in range(depth.shape[0])
    ]


def load_cow_vggt_run(
    seq: str,
    *,
    cow_root: str,
    vggt_root: str,
    num_frames: int,
    height: int,
    width: int,
    frame_start: int = 0,
    davis_4rc_root: str | None = None,
    scale_align: str = "4rc",
    external_scale_root: str | None = None,
    track_sample_mode: str = "all",
    track_stride: int = 1,
    track_samples: int = 0,
    track_sample_seed: int = 42,
    cow_conf_threshold: float = 0.5,
    depth_conf_threshold: float = 0.0,
) -> dict:
    """Return a 4RC-run-shaped dict from cotracker + VGGT-Omega source outputs.

    Keys match load_4rc_run: frames (list (H,W,3)), pts (list (H,W,3) world),
    tracks ((N,T,3) world), c2w (list (4,4)), K (list (3,3) at HxW), query_idx=0.
    All geometry is in 4RC metric scale when scale_align="4rc".
    """
    cow_path = Path(cow_root) / f"{seq}_pred_trackvideo.npz"
    pose_path = Path(vggt_root) / seq / "pose" / "vggt_omega.npz"
    depth_path = Path(vggt_root) / seq / "geometry" / "vggt_omega_depth.npz"
    for p in (cow_path, pose_path, depth_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    cow = np.load(cow_path, allow_pickle=True)
    pose = np.load(pose_path, allow_pickle=True)
    geom = np.load(depth_path, allow_pickle=True)
    query_idx = 0
    if "query_frame_idx" in cow:
        query_idx = int(np.asarray(cow["query_frame_idx"]).item())

    depth = np.asarray(geom["depth"], dtype=np.float32)
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth_conf = np.asarray(geom["depth_conf"], dtype=np.float32)
    if depth_conf.ndim == 4 and depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]
    intr = np.asarray(geom["intrinsics"], dtype=np.float32)
    c2w_full = np.asarray(pose["cam_c2w"], dtype=np.float32)

    start = int(frame_start)
    if start < 0:
        raise ValueError(f"frame_start must be non-negative, got {start}")
    available = min(depth.shape[0], depth_conf.shape[0], intr.shape[0], c2w_full.shape[0]) - start
    T = min(int(num_frames), int(available))
    if T <= 0:
        raise ValueError(
            f"{seq}: frame_start={start} is outside VGGT cache length "
            f"{c2w_full.shape[0]}"
        )
    if query_idx < 0 or query_idx >= T:
        raise ValueError(f"{cow_path}: query_frame_idx={query_idx} outside loaded T={T}")
    sequence_anchor_c2w = c2w_full[0].copy()
    depth = depth[start:start + T]
    depth_conf = depth_conf[start:start + T]
    intr = intr[start:start + T].copy()
    c2w = c2w_full[start:start + T].copy()

    # Resample geometry to the requested HxW grid (VGGT cache is usually already
    # there; rescale intrinsics if not). Matches render_cow_vggt_2x4_comparison.
    if depth.shape[1:3] != (height, width):
        depth = np.stack(
            [cv2.resize(d, (width, height), interpolation=cv2.INTER_LINEAR) for d in depth], axis=0
        )
        depth_conf = np.stack(
            [cv2.resize(d, (width, height), interpolation=cv2.INTER_LINEAR) for d in depth_conf], axis=0
        )
        sx = float(width) / float(geom["vggt_hw"][1])
        sy = float(height) / float(geom["vggt_hw"][0])
        intr[:, 0, 0] *= sx
        intr[:, 0, 2] *= sx
        intr[:, 1, 1] *= sy
        intr[:, 1, 2] *= sy

    # ---- Metric scale alignment VGGT -> 4RC (default on) ----
    alpha = 1.0
    if scale_align == "4rc":
        if not davis_4rc_root:
            raise ValueError("scale_align='4rc' requires davis_4rc_root for the frame-0 reference")
        alpha = _estimate_depth_scale_to_4rc(
            vggt_depth0=depth[0], davis_4rc_root=davis_4rc_root, seq=seq
        )
        print(f"[cow-vggt] {seq}: scale_align=4rc alpha_vggt_to_4rc={alpha:.4f}")
    elif scale_align == "external":
        if not external_scale_root:
            raise ValueError("scale_align='external' requires external_scale_root")
        alpha, scale_path = _load_external_scale(external_scale_root, seq)
        print(f"[cow-vggt] {seq}: scale_align=external alpha={alpha:.6f} source={scale_path}")
    elif scale_align not in ("none", "", None):
        raise ValueError(
            f"scale_align must be '4rc', 'external', or 'none', got {scale_align!r}"
        )

    depth = (depth * alpha).astype(np.float32)
    c2w[:, :3, 3] *= alpha
    sequence_anchor_c2w[:3, 3] *= alpha
    trajectory_anchor_cam2world = (
        np.linalg.inv(c2w[query_idx]) @ sequence_anchor_c2w
    ).astype(np.float32)

    # ---- Lift tracks (in the now 4RC-scaled space) ----
    tracks_world, track_vis, query_xy, track_conf = _lift_cow_tracks(
        cow, depth, depth_conf, intr, c2w,
        track_stride=track_stride,
        track_samples=track_samples,
        track_sample_mode=track_sample_mode,
        track_sample_seed=track_sample_seed,
        cow_conf_threshold=cow_conf_threshold,
        depth_conf_threshold=depth_conf_threshold,
    )
    # Keep invalid coordinates finite for downstream normalization, but do not
    # synthesize frame-0 evidence. The original visibility mask remains the
    # source of truth and is consumed by Track4D's track builder.
    tracks_world = _carry_last_valid_track_positions(tracks_world, track_vis)

    cow_h, cow_w = np.asarray(cow["tracks_xy"]).shape[1:3]
    query_xy_out = query_xy.astype(np.float32, copy=True)
    query_xy_out[:, 0] *= float(width) / max(float(cow_w), 1.0)
    query_xy_out[:, 1] *= float(height) / max(float(cow_h), 1.0)

    pts_list = _per_frame_world_pts(depth, intr, c2w)
    # frames: only used downstream for (src_h, src_w) to scale K; provide HxW.
    frame0 = np.zeros((height, width, 3), dtype=np.uint8)
    frames = [frame0 for _ in range(T)]

    return {
        "frames": frames,
        "pts": pts_list,
        "tracks": tracks_world,
        "tracks_valid": track_vis.astype(bool, copy=False),
        "tracks_confidence": track_conf.astype(np.float32, copy=False),
        "tracks_source_xy": query_xy_out.astype(np.float32, copy=False),
        "c2w": [c2w[t] for t in range(T)],
        "K": [intr[t] for t in range(T)],
        "query_idx": int(query_idx),
        "trajectory_anchor_cam2world": trajectory_anchor_cam2world,
        "metadata": {
            "source_backend": "cow_vggt",
            "alpha_vggt_to_4rc": float(alpha),
            "geometry_scale_alpha": float(alpha),
            "geometry_scale_mode": str(scale_align),
            "cow_query_frame_idx": int(query_idx),
            "vggt_frame_start": int(start),
        },
    }


def load_cow_vggt_multiquery_run(
    seq: str,
    *,
    cow_roots: list[str] | tuple[str, ...] | str,
    vggt_root: str,
    num_frames: int,
    height: int,
    width: int,
    frame_start: int = 0,
    davis_4rc_root: str | None = None,
    scale_align: str = "4rc",
    external_scale_root: str | None = None,
    track_sample_mode: str = "all",
    track_stride: int = 1,
    track_samples: int = 0,
    track_sample_seed: int = 42,
    cow_conf_threshold: float = 0.5,
    depth_conf_threshold: float = 0.0,
) -> dict:
    """Return a 4RC-run-shaped dict with multiple CoW query caches concatenated.

    ``cow_roots`` is normally ``[frame0_cache, reverse_last_frame_cache]`` for
    q=2. Each root contains a dense CoW track cache with its own
    ``query_frame_idx`` metadata. The returned world geometry/cameras are shared
    from the first root, while track identities are concatenated along ``N``.
    ``query_idx`` remains 0 so downstream camera normalization keeps the DAVIS
    source-frame-0 coordinate convention; per-identity query frames are carried
    separately in ``tracks_query_idx_per_identity`` for RGB/GMOS gathers.
    """
    roots = split_multiquery_cow_roots(cow_roots)
    runs = [
        load_cow_vggt_run(
            seq,
            cow_root=root,
            vggt_root=vggt_root,
            num_frames=num_frames,
            height=height,
            width=width,
            frame_start=frame_start,
            davis_4rc_root=davis_4rc_root,
            scale_align=scale_align,
            external_scale_root=external_scale_root,
            track_sample_mode=track_sample_mode,
            track_stride=track_stride,
            track_samples=track_samples,
            track_sample_seed=track_sample_seed,
            cow_conf_threshold=cow_conf_threshold,
            depth_conf_threshold=depth_conf_threshold,
        )
        for root in roots
    ]
    base = runs[0]
    query_indices = [int(r["query_idx"]) for r in runs]
    T = int(np.asarray(base["tracks"]).shape[1])
    tracks = np.concatenate([np.asarray(r["tracks"], dtype=np.float32) for r in runs], axis=0)
    tracks_valid = np.concatenate([np.asarray(r["tracks_valid"], dtype=bool) for r in runs], axis=0)
    tracks_conf = np.concatenate([np.asarray(r["tracks_confidence"], dtype=np.float32) for r in runs], axis=0)
    tracks_xy = np.concatenate([np.asarray(r["tracks_source_xy"], dtype=np.float32) for r in runs], axis=0)
    q_per_identity = np.concatenate(
        [
            np.full((np.asarray(r["tracks"]).shape[0],), int(r["query_idx"]), dtype=np.int32)
            for r in runs
        ],
        axis=0,
    )
    for idx, run in enumerate(runs[1:], 1):
        if np.asarray(run["tracks"]).shape[1] != T:
            raise ValueError(f"{seq}: root {idx} has T={np.asarray(run['tracks']).shape[1]}, expected {T}")
    metadata = dict(base.get("metadata", {}))
    metadata.update(
        {
            "source_backend": "cow_vggt_multiquery",
            "cow_query_frame_indices": query_indices,
            "cow_roots": roots,
            "cow_query_count": len(roots),
        }
    )
    return {
        **base,
        "tracks": tracks.astype(np.float32, copy=False),
        "tracks_valid": tracks_valid.astype(bool, copy=False),
        "tracks_confidence": tracks_conf.astype(np.float32, copy=False),
        "tracks_source_xy": tracks_xy.astype(np.float32, copy=False),
        "tracks_query_idx_per_identity": q_per_identity,
        "query_idx": int(query_indices[0]),
        "track_query_indices": [int(q) for q in query_indices],
        "track_query_count": len(query_indices),
        "metadata": metadata,
    }


def select_cow_vggt_query_anchor(run: dict, anchor: str = "last") -> dict:
    """Keep exactly one query's identities from a multi-query CoW/VGGT run.

    This is intentionally not a fusion mode.  It lets a q1-trained model be
    evaluated with an anchor other than frame zero while refusing to silently
    feed it the concatenated q>1 identity distribution it was not trained on.
    Camera geometry remains in the common coordinate frame carried by ``run``;
    only query-indexed track/RGB/confidence rows are selected.
    """
    query_ids = np.asarray(run.get("tracks_query_idx_per_identity"))
    if query_ids.ndim != 1:
        raise ValueError("multi-query CoW/VGGT run is missing per-identity query indices")
    available = [int(q) for q in run.get("track_query_indices", [])]
    if not available:
        available = sorted(int(q) for q in np.unique(query_ids).tolist())
    if not available:
        raise ValueError("multi-query CoW/VGGT run has no query anchors")

    normalized_anchor = str(anchor).strip().lower()
    if normalized_anchor == "first":
        selected_query = available[0]
    elif normalized_anchor == "last":
        selected_query = available[-1]
    else:
        raise ValueError(f"unsupported single-query anchor={anchor!r}; expected first or last")
    keep = query_ids == selected_query
    if not bool(keep.any()):
        raise ValueError(f"CoW/VGGT query anchor {selected_query} has no identities")

    result = dict(run)
    for key in (
        "tracks",
        "query_tracks",
        "tracks_valid",
        "tracks_confidence",
        "tracks_source_xy",
    ):
        value = run.get(key)
        if isinstance(value, np.ndarray) and value.shape[0] == query_ids.shape[0]:
            result[key] = value[keep].copy()
    result["tracks_query_idx_per_identity"] = np.full(
        (int(keep.sum()),), selected_query, dtype=np.int32
    )
    result["track_query_indices"] = [selected_query]
    result["track_query_count"] = 1
    metadata = dict(run.get("metadata", {}))
    metadata.update(
        {
            "cow_single_query_anchor": normalized_anchor,
            "cow_selected_query_idx": selected_query,
        }
    )
    result["metadata"] = metadata
    return result
