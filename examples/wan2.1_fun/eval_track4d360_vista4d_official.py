"""Eval Track4D-360 on Vista4D's official eval dataset.

Hybrid pipeline:
  - SOURCE side (frames, depth, source cameras, point tracks) comes from 4RC
    cached under ${davis_4rc_root}/<video>/result.npz, exactly like
    eval_track4d360_davis_4rc_novel_traj.py. Track4D-360 needs point tracks,
    which Vista4D's pi3-only outputs do not provide; 4RC gives us that.
  - TARGET side (camera trajectory) comes from Vista4D's eval set:
      ${vista4d_eval_root}/cameras/<video>/<camera>.npz
    This replaces the synthetic angle/distance/orbit_radius trajectories from
    the older DAVIS benchmark with the paper-faithful hand-designed cameras
    used by Vista4D's evaluator.
  - PROMPT comes from Vista4D's metadata.csv per (video, camera) row.

Output layout (per (video, camera) pair):
  ${output_dir}/<video>/<camera>/<camera>.mp4
  ${output_dir}/<video>/<camera>/pred_rgb.mp4

The exact effective-camera payload is reconstructed from ``args.json`` by
``holo4d/tools/export_vista4d_effective_cameras.py``.  This driver does
not create a separate ``predictions/`` symlink tree.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import PIL.Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from eval_track4d360_hybrid_dense_syn4d import (  # noqa: E402
    NEGATIVE_PROMPT,
    _run_one_pair,
    build_eval_pipeline,
    precompute_eval_prompt_embs,
)
from eval_track4d360_davis_4rc_novel_traj import (  # noqa: E402
    _dilate_dynamic_mask_video,
    _load_gmos_source_dynamic_mask,
    _load_davis_frames,
    _project_track_labels_to_target_mask,
    _scale_intrinsics,
    _source_mask_at_4rc_track_grid,
    _source_mask_at_track_xy,
    _source_rgb_at_tracks,
)
from eval_track4d360_recammaster_val_4rc import (  # noqa: E402
    extract_source_depth_video,
    load_4rc_run,
    resize_depth_video,
    transform_run_to_query_camera,
)
from holo4d.benchmark.davis_utils import (  # noqa: E402
    resolve_davis_sequences,
)
from holo4d.benchmark.vista4d_trajectories import (  # noqa: E402
    anchor_trajectory_to_source_frame0,
    discover_vista4d_pairs_from_metadata,
    discover_vista4d_transfer_pairs_from_metadata,
    lookup_vista4d_source_prompt,
    lookup_vista4d_pair_seed,
    transform_native_target_to_source_gauge,
)
from holo4d.novel_view.static_scene_novel_traj import (  # noqa: E402
    _warp_source_to_target_frames,
)
from holo4d.data.track_reprojection import project_tracks_to_target_views  # noqa: E402
from holo4d.benchmark.cow_vggt_run import load_cow_vggt_run  # noqa: E402
from holo4d.benchmark.vista4d_pointmap_sim3 import (  # noqa: E402
    estimate_pi3_to_source_background_camera_similarity,
    estimate_pi3_to_source_pointmap_similarity,
    transform_cameras_by_similarity,
)
from holo4d.visualization.artifact_io import WAN_SAMPLE_FPS  # noqa: E402


DEFAULT_PROMPT = (
    "A real video clip with natural motion, captured by a moving camera, "
    "high quality, photorealistic detail."
)


def _intrinsics_to_K(intr: np.ndarray) -> np.ndarray:
    """Vista4D stores intrinsics as either [T,4]=[fx,fy,cx,cy] or [T,3,3]."""
    intr = np.asarray(intr, dtype=np.float32)
    if intr.ndim == 2 and intr.shape[-1] == 4:
        T = intr.shape[0]
        K = np.zeros((T, 3, 3), dtype=np.float32)
        K[:, 0, 0] = intr[:, 0]
        K[:, 1, 1] = intr[:, 1]
        K[:, 0, 2] = intr[:, 2]
        K[:, 1, 2] = intr[:, 3]
        K[:, 2, 2] = 1.0
        return K
    if intr.ndim == 3 and intr.shape[-2:] == (3, 3):
        return intr.astype(np.float32)
    raise ValueError(f"unsupported intrinsics shape {intr.shape}")


def _resample_time(arr: np.ndarray, num_frames: int) -> np.ndarray:
    if arr.shape[0] >= num_frames:
        return arr[:num_frames]
    pad = np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _load_vista4d_target_camera(
    npz_path: Path,
    *,
    num_frames: int,
    src_K_single: np.ndarray,
    vista4d_render_w: int = 1280,
    vista4d_render_h: int = 720,
    target_w: int,
    target_h: int,
    prefer_same_first_frame: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Vista4D target camera npz and rescale intrinsics to (target_w, target_h).

    Vista4D's official intrinsics in eval_data/cameras/<seq>/<traj>.npz are stored
    at 1280x720 (verified across all 110 trajectory files: cx≈640, cy≈360). Older
    code used 832x480 which inflated fx by 1.54× → "heavily zoomed-in" warp.
    For our model running at e.g. 288x512, we must rescale fx/fy/cx/cy.

    Returns (tgt_c2w (T,4,4), tgt_K_perframe (T,3,3)) — full per-frame K is
    preserved so 47/110 zoom trajectories (e.g. camel/zoom-out: fx 5049→1999)
    get correct FOV evolution downstream. Callers that need a single (3,3)
    representative for legacy interfaces should index K[0] explicitly.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"missing Vista4D target camera: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    camera_key = "cam_c2w_fsff" if prefer_same_first_frame and "cam_c2w_fsff" in data.files else "cam_c2w"
    intrinsics_key = (
        "intrinsics_fsff"
        if prefer_same_first_frame and "intrinsics_fsff" in data.files
        else "intrinsics"
    )
    if camera_key not in data.files:
        raise ValueError(
            f"target npz {npz_path} missing {camera_key!r} (got {list(data.files)})"
        )
    tgt_c2w = _resample_time(np.asarray(data[camera_key], dtype=np.float32), num_frames)

    if intrinsics_key in data.files:
        K_per_frame = _resample_time(
            _intrinsics_to_K(data[intrinsics_key]), num_frames
        )
        # Rescale from Vista4D's native render res to our target (target_w, target_h).
        K_scaled = np.array([
            _scale_intrinsics(K, vista4d_render_w, vista4d_render_h, target_w, target_h)
            for K in K_per_frame
        ], dtype=np.float32)
        tgt_K_perframe = K_scaled
    else:
        # Fall back to source intrinsics (already at target res), broadcast to (T,3,3).
        T = tgt_c2w.shape[0]
        tgt_K_perframe = np.broadcast_to(src_K_single.astype(np.float32), (T, 3, 3)).copy()
    return tgt_c2w, tgt_K_perframe.astype(np.float32)


def _load_vista4d_reference_depth_scale(
    *,
    recon_root: Path,
    source_depth: np.ndarray,
) -> tuple[float, int]:
    """Estimate Pi3-to-source scale from aligned static frame-0 depths."""
    depth_path = recon_root / "depths" / "00000.exr"
    pi3_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if pi3_depth is None:
        raise FileNotFoundError(f"could not read Vista4D reference depth {depth_path}")
    if pi3_depth.ndim == 3:
        pi3_depth = pi3_depth[..., 0]
    height, width = source_depth.shape
    pi3_depth = cv2.resize(
        pi3_depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    valid = (
        np.isfinite(source_depth)
        & np.isfinite(pi3_depth)
        & (source_depth > 1e-6)
        & (pi3_depth > 1e-6)
    )
    for folder in ("dynamic_mask", "sky_mask"):
        mask_path = recon_root / folder / "00000.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                valid &= mask < 128
    count = int(valid.sum())
    if count < 1024:
        raise RuntimeError(
            f"insufficient aligned static depth pixels under {recon_root}: {count}"
        )
    scale = float(np.median(source_depth[valid] / pi3_depth[valid]))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"invalid Vista4D depth scale {scale} under {recon_root}")
    return scale, count


def _lookup_caption(metadata_csv: Path, video: str, camera: str) -> str:
    return lookup_vista4d_source_prompt(
        metadata_csv,
        source_video=video,
        source_camera=camera,
        default=DEFAULT_PROMPT,
    )


def discover_vista4d_pairs(
    eval_root: Path,
    video_filter: list | None,
    source_filter: str | None,
    max_cameras_per_video: int = 0,
) -> list[tuple[str, str]]:
    return discover_vista4d_pairs_from_metadata(
        eval_root / "metadata.csv",
        video_filter=video_filter,
        source_filter=source_filter,
        max_cameras_per_video=max_cameras_per_video,
    )


def _damp_trajectory(c2w: np.ndarray, K_perframe: np.ndarray, factor: float):
    """Damp Vista4D's hand-designed trajectory toward frame 0 by a scalar factor.

    Vista4D's official trajectories are designed for a much stronger NVS prior;
    on our 4RC source content + 1.3B/14B Track4D-360 they're often too aggressive
    and the model fails to follow. Linearly damping the per-frame delta from
    frame 0 (rotation as axis-angle, translation as Euclidean delta, K as
    additive delta) shrinks the trajectory toward "stay at frame 0" while
    preserving its shape.

    factor=1.0 → identity (no damping). factor=0.0 → freeze at frame 0.
    """
    if factor >= 0.999:
        return c2w.astype(np.float32, copy=True), K_perframe.astype(np.float32, copy=True)
    from scipy.spatial.transform import Rotation as R

    c2w = c2w.astype(np.float32, copy=False)
    K_perframe = K_perframe.astype(np.float32, copy=False)
    T = c2w.shape[0]
    R0 = c2w[0, :3, :3]
    t0 = c2w[0, :3, 3]
    K0 = K_perframe[0]

    out_c2w = np.tile(np.eye(4, dtype=np.float32)[None], (T, 1, 1))
    for k in range(T):
        Rk = c2w[k, :3, :3]
        # Rotation delta from frame 0 in axis-angle, scaled by factor, then re-applied.
        R_rel = R0.T @ Rk
        rotvec = R.from_matrix(R_rel.astype(np.float64)).as_rotvec()
        damped = R0 @ R.from_rotvec(rotvec * float(factor)).as_matrix().astype(np.float32)
        out_c2w[k, :3, :3] = damped
        out_c2w[k, :3, 3] = t0 + float(factor) * (c2w[k, :3, 3] - t0)
    out_K = K0[None] + float(factor) * (K_perframe - K0[None])
    return out_c2w.astype(np.float32), out_K.astype(np.float32)


def load_vista4d_sample(
    *,
    davis_root: str,
    davis_4rc_root: str,
    vista4d_eval_root: Path,
    video: str,
    camera: str,
    num_frames: int,
    height: int,
    width: int,
    trajectory_video: str | None = None,
    trajectory_camera: str | None = None,
    trajectory_damping: float = 1.0,
    trajectory_anchor_mode: str = "vista4d_frame0",
    start_frame: int = 0,
    vista4d_render_w: int = 1280,
    vista4d_render_h: int = 720,
    source_backend: str = "4rc",
    cow_root: str | None = None,
    vggt_root: str | None = None,
    cow_track_sample_mode: str = "grid",
    cow_track_stride: int = 8,
    cow_conf_threshold: float = 0.5,
    cow_depth_conf_threshold: float = 0.0,
    cow_vggt_scale_align: str = "4rc",
    vista4d_target_scale_align: str = "depth",
    vista4d_target_gauge_align: str = "frame0_depth",
    target_translation_depth_fraction: float = 0.6,
    gmos_root: str = "",
    gmos_mask_subdir: str = "mos_frame",
    gmos_dilate_px: int = 0,
    require_gmos_mask: bool = False,
    target_dynamic_mask_mode: str = "track_labels",
    output_fps: float = WAN_SAMPLE_FPS,
    seed: int = 42,
) -> dict:
    """Build a Track4D-360 inference sample for one (video, camera) pair.

    Source side (with tracks): either 4RC (``source_backend='4rc'``,
    ${davis_4rc_root}/<video>/result.npz) or cotracker + VGGT-Omega
    (``source_backend='cow_vggt'``, from ${cow_root}/${vggt_root}); the cow+vggt
    geometry is rescaled to 4RC metric units so 4RC-trained ablations stay
    in-distribution. Both backends produce the same run contract, so everything
    downstream (query-camera transform, pi3->source scale-align, warp, track
    projection) is identical.
    Target side (camera): Vista4D at ${vista4d_eval_root}/cameras/<video>/<camera>.npz.
    """
    trajectory_video = trajectory_video or video
    trajectory_camera = trajectory_camera or camera

    sequences = resolve_davis_sequences(
        davis_root,
        sequence_names=video,
        num_frames=num_frames,
        start_frame=start_frame,
    )
    if len(sequences) != 1:
        raise RuntimeError(f"expected one DAVIS sequence for {video}, got {len(sequences)}")
    sequence = sequences[0]

    # ---- Source from 4RC (gives us tracks) ----
    # Camera convention: 4RC c2w is OpenCV right-handed (X right, Y down, Z fwd),
    # camera-to-world. Verified via _warp_source_to_target_frames pixel→cam math:
    # x_cam=(u-cx)*depth/fx, y_cam=(v-cy)*depth/fy, z_cam=depth (track_panels.py:108-110).
    if source_backend == "cow_vggt":
        if not cow_root or not vggt_root:
            raise ValueError("source_backend='cow_vggt' requires cow_root and vggt_root")
        run = load_cow_vggt_run(
            sequence.name,
            cow_root=cow_root,
            vggt_root=vggt_root,
            num_frames=num_frames,
            height=height,
            width=width,
            davis_4rc_root=davis_4rc_root,   # frame-0 depth reference for 4RC-scale align
            scale_align=cow_vggt_scale_align,
            track_sample_mode=cow_track_sample_mode,
            track_stride=cow_track_stride,
            cow_conf_threshold=cow_conf_threshold,
            depth_conf_threshold=cow_depth_conf_threshold,
            track_sample_seed=seed,
        )
        source_run = transform_run_to_query_camera(run)
    elif source_backend == "4rc":
        run_dir = Path(davis_4rc_root) / sequence.name
        source_run = transform_run_to_query_camera(load_4rc_run(run_dir))
    else:
        raise ValueError(f"source_backend must be '4rc' or 'cow_vggt', got {source_backend!r}")
    source_frames, source_rgb_video = _load_davis_frames(sequence.frame_paths, width, height)

    # Match the canonical DAVIS protocol: GMOS supplies a full-resolution
    # foreground/background mask, while 4RC supplies depth, cameras, and 3D
    # tracks.  Keeping these inputs separate is important: Vista4D's released
    # SAM masks are used by its own reconstruction pipeline and are not a
    # substitute for Holo4D's GMOS fusion condition.
    source_dynamic_mask = _load_gmos_source_dynamic_mask(
        gmos_root=str(gmos_root or ""),
        seq=sequence.name,
        mask_subdir=str(gmos_mask_subdir or "mos_frame"),
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
    )
    if source_dynamic_mask is None and require_gmos_mask:
        raise FileNotFoundError(
            f"required GMOS masks are missing for {sequence.name}: "
            f"root={gmos_root!r}, subdir={gmos_mask_subdir!r}"
        )
    source_dynamic_mask = _dilate_dynamic_mask_video(
        source_dynamic_mask, int(gmos_dilate_px)
    )
    if source_dynamic_mask is not None:
        print(
            f"[Vista4D GMOS] {sequence.name}: source coverage="
            f"{float(source_dynamic_mask.mean()):.4f} root={gmos_root} "
            f"subdir={gmos_mask_subdir} dilate_px={int(gmos_dilate_px)}"
        )

    src_h, src_w = source_run["frames"][0].shape[:2]
    source_intrinsics = _scale_intrinsics(source_run["K"][0], src_w, src_h, width, height)
    # source_cam2world in query-camera frame (camera at query_idx becomes identity).
    query_idx = int(source_run["query_idx"])
    query_c2w = np.asarray(source_run["c2w"][query_idx], dtype=np.float32)
    world_to_query = np.linalg.inv(query_c2w).astype(np.float32)
    src_c2w_world = np.stack([np.asarray(p, dtype=np.float32) for p in source_run["c2w"]], axis=0)
    source_cam2world = (world_to_query[None] @ src_c2w_world).astype(np.float32)
    source_depth_video = resize_depth_video(extract_source_depth_video(source_run), width, height)
    source_tracks_valid = source_run.get("tracks_valid")
    if isinstance(source_tracks_valid, np.ndarray) and source_tracks_valid.shape == source_run["query_tracks"].shape[:2]:
        source_tracks_valid = source_tracks_valid.astype(bool, copy=False)
    else:
        source_tracks_valid = None
    source_tracks_confidence = source_run.get("tracks_confidence")
    if (
        isinstance(source_tracks_confidence, np.ndarray)
        and source_tracks_confidence.shape == source_run["query_tracks"].shape[:2]
    ):
        source_tracks_confidence = source_tracks_confidence.astype(np.float32, copy=False)
    else:
        source_tracks_confidence = None
    source_tracks_xy = source_run.get("tracks_source_xy")
    if not isinstance(source_tracks_xy, np.ndarray):
        source_tracks_xy = None

    # ---- Target from Vista4D NPZ (paper-faithful hand-designed camera) ----
    # Camera convention: Vista4D's pi3 c2w is OpenCV right-handed (verified from
    # utils/recon_and_seg/recon_pi3.py:80). The released Pi3 source frame-0 is
    # generally NOT identity, while 4RC is normalized to the query camera. The
    # native target must therefore be moved into the Pi3 source-frame0 gauge
    # before it can control geometry expressed in the 4RC query-camera gauge.
    #
    # Metric scale, however, is independent: pi3 and 4RC each recover their own
    # scale via separate focal-shift / triangulation pipelines. To make Vista4D's
    # hand-designed target trajectory line up with 4RC source geometry, we
    # rescale target translations from pi3-scale → 4RC-scale below.
    assert query_idx == 0, (
        f"4RC query_idx must be 0 for Vista4D-camera benchmark (got {query_idx}); "
        "pi3 anchors at frame 0 and we don't auto-correct anchoring offsets."
    )

    cam_npz = vista4d_eval_root / "cameras" / trajectory_video / f"{trajectory_camera}.npz"
    use_same_first_frame = trajectory_anchor_mode == "source_frame0"
    target_cam2world, target_intrinsics = _load_vista4d_target_camera(
        cam_npz,
        num_frames=num_frames,
        src_K_single=source_intrinsics,
        vista4d_render_w=vista4d_render_w,
        vista4d_render_h=vista4d_render_h,
        target_w=width,
        target_h=height,
        prefer_same_first_frame=use_same_first_frame,
    )

    # ---- Gauge alignment (Vista4D Pi3 -> source reconstruction) ----
    # ``frame0_depth`` is the historical path: anchor via the Pi3 frame-0
    # camera and estimate only a depth scale. ``pointmap_sim3`` estimates the
    # complete Pi3-world -> source-query Sim(3) from matched source pointmaps.
    # The latter is required when the reconstructors disagree in orientation
    # or origin, not just metric scale.
    pointmap_similarity = None
    pointmap_alignment = None
    if vista4d_target_gauge_align in {
        "pointmap_sim3",
        "background_camera_sim3",
    }:
        if trajectory_anchor_mode != "vista4d_frame0":
            raise ValueError(
                "pointmap_sim3 is defined for the official native trajectory; "
                "use trajectory_anchor_mode='vista4d_frame0'"
            )
        if trajectory_video != video:
            raise ValueError(
                "pointmap_sim3 requires source and trajectory video to match; "
                f"got source={video!r}, trajectory={trajectory_video!r}"
            )
        estimator_kwargs = {
            "source_query_pointmaps": [
                np.asarray(points, dtype=np.float32)
                for points in source_run["query_pts"][:num_frames]
            ],
            "source_confidence_pointmaps": source_run.get("point_confidence"),
            "vista4d_recon_root": vista4d_eval_root / "recon_and_seg" / video,
        }
        if vista4d_target_gauge_align == "background_camera_sim3":
            pointmap_similarity = estimate_pi3_to_source_background_camera_similarity(
                **estimator_kwargs
            )
        else:
            pointmap_similarity = estimate_pi3_to_source_pointmap_similarity(
                **estimator_kwargs,
                num_frames=num_frames,
            )
        pointmap_alignment = pointmap_similarity.as_dict()
        pointmap_alignment.update(
            {
                "mode": vista4d_target_gauge_align,
                "source_video": video,
                "trajectory_video": trajectory_video,
                "trajectory_camera": trajectory_camera,
                "mask_source": "vista4d_dynamic_mask+sky_mask",
                "rotation_constraint": (
                    "source_camera_frame0"
                    if vista4d_target_gauge_align == "background_camera_sim3"
                    else "free_umeyama"
                ),
            }
        )
        print(
            f"[Vista4D {vista4d_target_gauge_align}] {video}/{camera}: "
            f"scale={pointmap_similarity.scale:.6f} "
            f"inliers={pointmap_similarity.inliers}/"
            f"{pointmap_similarity.correspondences} "
            f"median={pointmap_similarity.median_residual:.6f} "
            f"p90={pointmap_similarity.p90_residual:.6f}"
        )
    elif vista4d_target_gauge_align != "frame0_depth":
        raise ValueError(
            "vista4d_target_gauge_align must be frame0_depth, pointmap_sim3, "
            "or background_camera_sim3; "
            f"got {vista4d_target_gauge_align!r}"
        )

    # ---- Metric scale alignment (legacy frame0-depth path) ----
    # Camera-motion norms are ill-conditioned for nearly static source clips.
    # The default therefore aligns the official Pi3 frame-0 depth directly to
    # the source reconstruction depth on static, non-sky pixels.
    pi3_src_npz = vista4d_eval_root / "recon_and_seg" / trajectory_video / "cameras.npz"
    pi3_mean = 0.0
    ours_mean = 0.0
    pi3_source_frame0_c2w = None
    if pi3_src_npz.exists():
        pi3_src = np.load(pi3_src_npz, allow_pickle=True)
        pi3_src_c2w = np.asarray(pi3_src["cam_c2w"], dtype=np.float32)
        pi3_source_frame0_c2w = pi3_src_c2w[0]
        n_align = min(pi3_src_c2w.shape[0], source_cam2world.shape[0])
        pi3_rel = np.linalg.inv(pi3_src_c2w[0])[None] @ pi3_src_c2w[:n_align]
        ours_rel = np.linalg.inv(source_cam2world[0])[None] @ source_cam2world[:n_align]
        pi3_mean = float(np.linalg.norm(pi3_rel[:, :3, 3], axis=-1).mean())
        ours_mean = float(np.linalg.norm(ours_rel[:, :3, 3], axis=-1).mean())

    depth_align_pixels = 0
    if pointmap_similarity is not None:
        scale_pi3_to_4rc = float(pointmap_similarity.scale)
        depth_align_pixels = int(pointmap_similarity.inliers)
    elif vista4d_target_scale_align == "depth":
        scale_pi3_to_4rc, depth_align_pixels = _load_vista4d_reference_depth_scale(
            recon_root=vista4d_eval_root / "recon_and_seg" / trajectory_video,
            source_depth=np.asarray(source_depth_video[0], dtype=np.float32),
        )
    elif vista4d_target_scale_align == "camera_motion":
        if pi3_mean <= 1e-6 or ours_mean <= 1e-6:
            raise RuntimeError(
                "camera-motion target scale is degenerate; use "
                "--vista4d_target_scale_align depth"
            )
        scale_pi3_to_4rc = ours_mean / pi3_mean
    elif vista4d_target_scale_align == "none":
        scale_pi3_to_4rc = 1.0
    else:
        raise ValueError(
            "vista4d_target_scale_align must be depth, camera_motion, or none; "
            f"got {vista4d_target_scale_align!r}"
        )

    # ---- Optional shared-first-frame anchoring ----
    # The official *_fsff arrays preserve the intended relative camera/FOV
    # curve. Translation is converted in the relative frame; focal evolution
    # is transferred as a ratio, so fx/fy cannot cross zero and flip the image.
    if pointmap_similarity is not None:
        target_cam2world = transform_cameras_by_similarity(
            target_cam2world, pointmap_similarity
        )
    elif trajectory_anchor_mode == "source_frame0":
        target_cam2world, target_intrinsics = anchor_trajectory_to_source_frame0(
            target_c2w=target_cam2world,
            target_intrinsics=target_intrinsics,
            source_frame0_c2w=source_cam2world[0],
            source_intrinsics=source_intrinsics,
            translation_scale=scale_pi3_to_4rc,
        )
    elif trajectory_anchor_mode == "vista4d_frame0":
        if pi3_source_frame0_c2w is None:
            raise FileNotFoundError(
                f"native target-camera mode requires Pi3 source cameras: {pi3_src_npz}"
            )
        target_cam2world = transform_native_target_to_source_gauge(
            target_c2w=target_cam2world,
            source_frame0_c2w=pi3_source_frame0_c2w,
            translation_scale=scale_pi3_to_4rc,
        )
    else:
        raise ValueError(
            "trajectory_anchor_mode must be 'vista4d_frame0' or 'source_frame0', "
            f"got {trajectory_anchor_mode!r}"
        )

    # ---- Optional trajectory damping (toward frame 0) ----
    # Vista4D's hand-designed trajectories are aggressive; damping toward frame 0
    # makes them closer to the model's training distribution. Keeps direction,
    # shrinks magnitude. Applied to translation, rotation (axis-angle), and
    # focal-length evolution simultaneously.
    if trajectory_damping is not None and trajectory_damping < 0.999:
        target_cam2world, target_intrinsics = _damp_trajectory(
            target_cam2world, target_intrinsics, float(trajectory_damping)
        )

    # ---- Point-map-scale safety clamp (prevents scene collapse) ----
    # This is a final guard after robust depth alignment. Measure and rescale
    # translation relative to target frame 0; changing absolute camera centers
    # would depend on the arbitrary world origin.
    depth_frac = float(target_translation_depth_fraction)
    tgt_clamp_scale = 1.0
    src_depth_med = 0.0
    if depth_frac > 0.0:
        _valid_depth = source_depth_video[np.isfinite(source_depth_video) & (source_depth_video > 1e-6)]
        if _valid_depth.size > 0:
            src_depth_med = float(np.median(_valid_depth))
            max_safe_t = depth_frac * src_depth_med
            target_rel = np.linalg.inv(target_cam2world[0])[None] @ target_cam2world
            cur_max_t = float(np.linalg.norm(target_rel[:, :3, 3], axis=-1).max())
            if max_safe_t > 1e-6 and cur_max_t > max_safe_t:
                tgt_clamp_scale = max_safe_t / cur_max_t
                target_rel[:, :3, 3] *= tgt_clamp_scale
                target_cam2world = target_cam2world[0][None] @ target_rel
                print(f"[Vista4D scale-align] CLAMP {video}/{camera}: target translation "
                      f"capped to {depth_frac:.2f}x scene depth "
                      f"(med_depth={src_depth_med:.4f}, max_t {cur_max_t:.4f}->{max_safe_t:.4f}, "
                      f"clamp_scale={tgt_clamp_scale:.4f}) -- near-static source / bad scale ratio")

    source_rel = np.linalg.inv(source_cam2world[0])[None] @ source_cam2world
    target_rel = np.linalg.inv(target_cam2world[0])[None] @ target_cam2world
    src_trans_norm = float(np.linalg.norm(source_rel[:, :3, 3], axis=-1).max())
    tgt_trans_norm = float(np.linalg.norm(target_rel[:, :3, 3], axis=-1).max())
    min_fx = float(target_intrinsics[:, 0, 0].min())
    min_fy = float(target_intrinsics[:, 1, 1].min())
    if min_fx <= 0.0 or min_fy <= 0.0 or not np.isfinite(target_intrinsics).all():
        raise ValueError(
            f"invalid aligned target intrinsics for {video}/{camera}: "
            f"min_fx={min_fx}, min_fy={min_fy}"
        )
    print(f"[Vista4D scale-align] {video}/{camera} <- {trajectory_video}/{trajectory_camera}: "
          f"pi3_mean_src={pi3_mean:.4f}  ours_mean_src={ours_mean:.4f}  "
          f"scale_ratio={scale_pi3_to_4rc:.4f}  scale_mode={vista4d_target_scale_align}  "
          f"depth_pixels={depth_align_pixels}  "
          f"src_max_t={src_trans_norm:.4f}  tgt_max_t_after_rescale={tgt_trans_norm:.4f}  "
          f"src_med_depth={src_depth_med:.4f}  tgt_t_clamp={tgt_clamp_scale:.4f}  "
          f"min_fx={min_fx:.2f}  min_fy={min_fy:.2f}  "
          f"damping={trajectory_damping}  anchor={trajectory_anchor_mode}  "
          f"gauge_align={vista4d_target_gauge_align}  fsff={use_same_first_frame}")

    # ---- Caption from Vista4D metadata.csv ----
    prompt = _lookup_caption(vista4d_eval_root / "metadata.csv", video, camera)
    output_fps = float(output_fps)
    if not np.isfinite(output_fps) or output_fps <= 0.0:
        raise ValueError(f"output_fps must be finite and positive, got {output_fps}")

    # ---- Forward warp source RGBD into target frames ----
    # target_intrinsics here is (T,3,3) — per-frame K so zoom trajectories
    # render with correct evolving FOV (was the cause of "warp totally unlike GT").
    target_frames, target_depth, target_depth_valid = _warp_source_to_target_frames(
        source_rgb_video=source_rgb_video,
        source_depth_video=source_depth_video,
        source_cam2world=source_cam2world,
        source_intrinsics=source_intrinsics,
        target_cam2world=target_cam2world,
        target_intrinsics=target_intrinsics,
    )

    # ---- Project 4RC tracks into target views (so the model gets track conditioning) ----
    # project_tracks_to_target_views accepts (T,3,3) directly — same per-frame K.
    _, track_vis, track_depth = project_tracks_to_target_views(
        torch.as_tensor(np.asarray(source_run["query_tracks"]), dtype=torch.float32),
        torch.as_tensor(target_cam2world, dtype=torch.float32),
        torch.as_tensor(target_intrinsics, dtype=torch.float32),
        image_h=int(height),
        image_w=int(width),
    )
    track_vis_np = track_vis[0].detach().cpu().numpy().astype(np.float32)
    track_depth_np = (track_depth[0].detach().cpu().numpy().astype(np.float32) * track_vis_np)

    source_dynamic_mask_at_tracks = None
    target_dynamic_mask = None
    if source_dynamic_mask is not None:
        if source_tracks_xy is not None:
            source_dynamic_mask_at_tracks = _source_mask_at_track_xy(
                source_dynamic_mask=source_dynamic_mask,
                source_xy=source_tracks_xy,
                num_frames=int(np.asarray(source_run["query_tracks"]).shape[1]),
            )
        else:
            source_dynamic_mask_at_tracks = _source_mask_at_4rc_track_grid(
                source_dynamic_mask=source_dynamic_mask,
                tracks_world=np.asarray(source_run["query_tracks"], dtype=np.float32),
                image_h=int(height),
                image_w=int(width),
                track_query_idx=int(query_idx),
            )

        target_mask_mode = str(target_dynamic_mask_mode).strip().lower()
        if target_mask_mode == "track_labels":
            target_dynamic_mask = _project_track_labels_to_target_mask(
                tracks_world=np.asarray(source_run["query_tracks"], dtype=np.float32),
                dynamic_at_tracks=source_dynamic_mask_at_tracks,
                tracks_valid=source_tracks_valid,
                cam2world=np.asarray(target_cam2world, dtype=np.float32),
                intrinsics=np.asarray(target_intrinsics, dtype=np.float32),
                height=int(height),
                width=int(width),
                min_pixels=0,
            )
        elif target_mask_mode == "dense_source_zbuffer":
            from holo4d.novel_view.static_scene_novel_traj import (
                warp_source_dynamic_mask_to_target_views,
            )

            target_dynamic_mask = warp_source_dynamic_mask_to_target_views(
                source_dynamic_mask=source_dynamic_mask,
                source_depth_video=source_depth_video,
                source_cam2world=source_cam2world,
                source_intrinsics=source_intrinsics,
                target_cam2world=np.asarray(target_cam2world, dtype=np.float32),
                target_intrinsics=np.asarray(target_intrinsics, dtype=np.float32),
                min_pixels=0,
            )
        else:
            raise ValueError(
                "target_dynamic_mask_mode must be track_labels or "
                f"dense_source_zbuffer, got {target_dynamic_mask_mode!r}"
            )
        print(
            f"[Vista4D GMOS] {sequence.name}: target coverage="
            f"{float(target_dynamic_mask.mean()):.4f} mode={target_mask_mode}"
        )

    # Downstream pipeline (dense_target_control_unit, wan_video_track4d360)
    # consumes a single (3,3) target K. For non-zoom trajectories all K[t] match;
    # for zoom trajectories we use frame-0 K as the representative (warp/tracks
    # above already saw the per-frame K, so the conditioning input is correct).
    target_K_single = np.asarray(target_intrinsics[0], dtype=np.float32)

    sample = {
        "video": source_frames,
        "tracks_world": source_run["query_tracks"],
        # Source RGB sampled at the 4RC track-grid identities (query frame),
        # in the SAME identity order as tracks_world. ab12/13/15/16 track
        # configs have track_in_channels=9 (xyz/uvz + depth + RGB); without
        # this the pipeline zero-fills the RGB channels and the track tokens
        # lose their appearance attribute (base_pipeline.py:466 warning).
        "source_rgb_at_tracks": _source_rgb_at_tracks(
            source_rgb_video=source_rgb_video,
            tracks_world=source_run["query_tracks"],
            source_xy=source_tracks_xy,
            image_h=int(height),
            image_w=int(width),
            track_query_idx=int(query_idx),
        ),
        "target_intrinsics": target_K_single,
        "target_intrinsics_per_frame": target_intrinsics,
        "source_cam2world": source_cam2world,
        "source_intrinsics": source_intrinsics,
        "source_rgb_video": source_rgb_video,
        "source_depth_video": source_depth_video,
        "common_scale": np.float32(1.0),
        "track_query_idx": query_idx,
        "prompt": prompt,
        "output_fps": output_fps,
        "has_tracks": True,
        "has_dynamic_tracks": True,
        "has_depth": True,
        "track_video_loss_mask": 1.0,
        "sequence_name": video,
        "target_video": target_frames,
        "target_cam2world": target_cam2world,
        "target_depth": target_depth,
        "target_depth_valid": target_depth_valid,
        "reproj_src_depth_at_target": track_depth_np,
        "reproj_src_depth_valid": track_vis_np,
        "trajectory_name": camera,
        "trajectory_donor_video": trajectory_video,
        "trajectory_donor_camera": trajectory_camera,
        "trajectory_kind": "novel_view",
        "src_cam_name": "4rc",
        "tgt_cam_name": camera,
    }
    if source_tracks_valid is not None:
        sample["tracks_world_valid_mask"] = source_tracks_valid.astype(bool, copy=False)
    if source_tracks_confidence is not None:
        sample["tracks_world_confidence"] = source_tracks_confidence.astype(np.float32, copy=False)
    if source_dynamic_mask is not None:
        sample["source_dynamic_mask"] = source_dynamic_mask.astype(bool, copy=False)
        sample["source_track_dynamic_mask"] = source_dynamic_mask.astype(bool, copy=False)
        sample["source_dynamic_mask_at_tracks"] = source_dynamic_mask_at_tracks.astype(
            bool, copy=False
        )
        sample["target_dynamic_mask"] = target_dynamic_mask.astype(bool, copy=False)
    if pointmap_alignment is not None:
        sample["pointmap_sim3_alignment"] = pointmap_alignment
    return sample


def _save_payload(sample: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        target_cam2world=np.asarray(sample["target_cam2world"], dtype=np.float32),
        target_intrinsics=np.asarray(sample["target_intrinsics"], dtype=np.float32),
        target_depth=np.asarray(sample["target_depth"], dtype=np.float32),
        target_depth_valid=np.asarray(sample["target_depth_valid"], dtype=np.float32),
        reproj_src_depth_at_target=np.asarray(sample["reproj_src_depth_at_target"], dtype=np.float32),
        reproj_src_depth_valid=np.asarray(sample["reproj_src_depth_valid"], dtype=np.float32),
        tracks_world=np.asarray(sample["tracks_world"], dtype=np.float32),
        common_scale=np.asarray(sample.get("common_scale", 1.0), dtype=np.float32),
        trajectory_name=np.asarray(sample.get("trajectory_name", ""), dtype=str),
        src_cam_name=np.asarray(sample.get("src_cam_name", ""), dtype=str),
        tgt_cam_name=np.asarray(sample.get("tgt_cam_name", ""), dtype=str),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model_paths", type=str, default=None)
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--gs_checkpoint_path", type=str, default=None)
    parser.add_argument("--lora_base_model", type=str, default=None)
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--camera_control_checkpoint", type=str, default=None)
    parser.add_argument("--fp8_models", type=str, default=None)
    parser.add_argument("--offload_models", type=str, default=None)

    from holo4d.shared_args import (
        add_cond_null_token_fallback_args,
        add_dense_branch_args,
        add_gs_dynamic_args,
        add_gs_output_args,
        add_perlayer_camera_args,
        add_plucker_args,
        add_query_impulse_args,
        add_source_backend_args,
        add_track_config_args,
        add_warped_condition_args,
    )

    add_track_config_args(parser)
    add_dense_branch_args(parser)
    add_cond_null_token_fallback_args(parser)
    add_query_impulse_args(parser)
    add_plucker_args(parser)
    add_perlayer_camera_args(parser)
    add_warped_condition_args(parser)
    add_gs_output_args(parser)
    add_gs_dynamic_args(parser)

    parser.add_argument(
        "--davis_root", type=str,
        default=os.environ.get("FRAME_ROOT", ""),
    )
    parser.add_argument(
        "--davis_4rc_root", type=str,
        default=os.environ.get("FOURRC_ROOT", ""),
        help="Cached 4RC outputs (per-sequence result.npz with tracks + cameras + depth).",
    )
    parser.add_argument(
        "--vista4d_eval_root", type=str,
        default=os.environ.get("VISTA4D_EVAL_ROOT"),
        help="Vista4D eval root (recon_and_seg/, cameras/, metadata.csv).",
    )
    parser.add_argument(
        "--gmos_root",
        type=str,
        default="",
        help="Full-GMOS output root containing mos_frame/<video>/*.png.",
    )
    parser.add_argument("--gmos_mask_subdir", type=str, default="mos_frame")
    parser.add_argument("--gmos_dilate_px", type=int, default=0)
    parser.add_argument(
        "--require_gmos_mask",
        action="store_true",
        help="Fail instead of silently running without the requested GMOS fusion.",
    )
    parser.add_argument(
        "--target_dynamic_mask_mode",
        choices=("track_labels", "dense_source_zbuffer"),
        default="track_labels",
        help="Propagate GMOS foreground into the target view like the DAVIS driver.",
    )
    # ---- Source geometry/tracks backend (shared with the davis driver) ----
    add_source_backend_args(parser)
    parser.add_argument(
        "--cow_vggt_scale_align", type=str, default="4rc", choices=["4rc", "none"],
        help=("Scale VGGT/CoW geometry to a 4RC frame-0 reference (default) or "
              "keep native VGGT metric scale when no 4RC cache is available."),
    )
    parser.add_argument(
        "--vista4d_target_scale_align",
        type=str,
        default="depth",
        choices=["depth", "camera_motion", "none"],
        help=("Map Vista4D target translations into the source reconstruction scale. "
              "The default uses aligned static frame-0 depth; camera_motion is a "
              "legacy diagnostic and is unstable for nearly static clips."),
    )
    parser.add_argument(
        "--vista4d_target_gauge_align",
        type=str,
        default="frame0_depth",
        choices=["frame0_depth", "pointmap_sim3", "background_camera_sim3"],
        help=("Align official Pi3 target cameras to the source reconstruction. "
              "frame0_depth preserves the legacy frame-0 camera + scalar-depth "
              "mapping; pointmap_sim3 robustly fits scale, rotation, and "
              "translation from corresponding source pointmaps; "
              "background_camera_sim3 fixes orientation/origin to source frame 0 "
              "and estimates only scale on Vista4D static-background pixels."),
    )
    parser.add_argument(
        "--source_filter", type=str, default="davis", choices=["davis", "pexels", "all"],
        help="Restrict to videos with metadata.source matching this. Default: davis.",
    )
    parser.add_argument(
        "--video_names", type=str, default=None,
        help="Comma-separated whitelist of video names; overrides default discovery.",
    )
    parser.add_argument(
        "--pair_names",
        type=str,
        default=None,
        help=("Comma-separated exact video/camera pairs. When set, bypass metadata "
              "pair discovery while retaining metadata prompts and seeds."),
    )
    parser.add_argument("--vista4d_render_w", type=int, default=1280,
                        help="Width at which Vista4D's official cameras/<seq>/<traj>.npz "
                             "intrinsics are stored. Verified default: 1280 (cx≈640 across "
                             "all 110 traj files). Older 832 default inflated fx by 1.54×.")
    parser.add_argument("--vista4d_render_h", type=int, default=720,
                        help="Height at which Vista4D's official cameras/<seq>/<traj>.npz "
                             "intrinsics are stored. Verified default: 720 (cy≈360).")
    parser.add_argument("--trajectory_damping", type=float, default=1.0,
                        help="Damp Vista4D's hand-designed trajectory toward frame 0 by "
                             "this scalar. 1.0 = full Vista4D (default), 0.5 = halve "
                             "translation/rotation/focal-change relative to frame 0, "
                             "0.0 = freeze at frame 0. Useful when Vista4D's aggressive "
                             "trajectories exceed the model's training-distribution "
                             "viewpoint range.")
    parser.add_argument(
        "--target_translation_depth_fraction",
        type=float,
        default=float(os.environ.get("HOLO4D_VISTA4D_TGT_T_DEPTH_FRAC", "0.6")),
        help=("Final target-translation safety cap as a fraction of median source "
              "depth. The legacy-compatible default is 0.6; use 0 to disable the "
              "cap for an exact full-trajectory evaluation."),
    )
    parser.add_argument("--trajectory_anchor_mode", type=str, default="vista4d_frame0",
                        choices=["vista4d_frame0", "source_frame0"],
                        help="vista4d_frame0 keeps the official target camera start. "
                             "source_frame0 re-applies Vista4D's relative motion to the "
                             "4RC/source frame-0 camera so the first generated frame "
                             "shares the source view.")
    parser.add_argument("--max_cameras_per_video", type=int, default=0,
                        help="Optional metadata-order cap per source video. 0 = no cap.")
    parser.add_argument("--random_transfer_trajectories_per_video", type=int, default=0,
                        help="If >0, sample this many donor Vista4D trajectories per "
                             "selected source video. Donor cameras may come from other "
                             "videos, but source frames/depth/tracks still come from the "
                             "selected source video.")
    parser.add_argument("--random_transfer_seed", type=int, default=0,
                        help="Seed for random donor trajectory sampling.")
    parser.add_argument("--trajectory_donor_source_filter", type=str, default="all",
                        choices=["davis", "pexels", "all"],
                        help="Source subset from which random donor trajectories are sampled.")
    parser.add_argument("--max_forward_angle_deg", type=float, default=90.0,
                        help="Reject donor trajectories whose camera forward vector rotates "
                             "farther than this from frame 0. This avoids back-view candidates.")
    parser.add_argument("--allow_same_video_trajectory_donors", action="store_true",
                        default=False,
                        help="Allow random donor trajectories from the same source video.")

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument(
        "--output_fps",
        type=float,
        default=float(WAN_SAMPLE_FPS),
        help=(
            "Encoded prediction FPS. Defaults to the official Wan sample_fps=16; "
            "this is intentionally independent of the source MP4 container FPS."
        ),
    )
    parser.add_argument("--track_spatial_stride", type=int, default=1)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed_from_vista4d_metadata",
        action="store_true",
        help=("Use the published seed from each Vista4D metadata row. This is "
              "required for a pair-matched comparison with the official model."),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--capture_final_latents", action="store_true")
    parser.add_argument("--no_vae_tiling", dest="vae_tiling", action="store_false")
    parser.set_defaults(vae_tiling=True)

    parser.add_argument("--drop_4rc_tracks", action="store_true", default=False,
                        help="If set, runs without 4RC tracks (degraded but ok).")
    parser.add_argument("--force_rerun", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gs_feedback_interval", type=int, default=0)

    parser.add_argument("--use_fsdp", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if getattr(args, "use_fsdp", False):
        from holo4d.eval_fsdp_helper import init_fsdp_early
        init_fsdp_early(args)

    torch.manual_seed(args.seed)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    from holo4d.eval_fsdp_helper import is_main_process
    if is_main_process():
        (output_root / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    eval_root = Path(args.vista4d_eval_root)
    source_filter = None if args.source_filter == "all" else args.source_filter
    video_filter = None
    if args.video_names:
        video_filter = [v.strip() for v in args.video_names.split(",") if v.strip()]
    if args.pair_names and args.random_transfer_trajectories_per_video > 0:
        raise ValueError("--pair_names cannot be combined with random trajectory transfer")
    if args.pair_names:
        pairs = []
        for pair_name in args.pair_names.split(","):
            pair_name = pair_name.strip().strip("/")
            if not pair_name:
                continue
            if "/" not in pair_name:
                raise ValueError(
                    f"invalid --pair_names item {pair_name!r}; expected video/camera"
                )
            video, camera = pair_name.split("/", 1)
            camera_path = eval_root / "cameras" / video / f"{camera}.npz"
            if not camera_path.is_file():
                raise FileNotFoundError(camera_path)
            pairs.append((video, camera, video, camera))
        if not pairs:
            raise ValueError("--pair_names did not contain any usable video/camera pairs")
        print(f"[Vista4D] using {len(pairs)} exact pairs: {args.pair_names}")
    elif args.random_transfer_trajectories_per_video > 0:
        donor_source_filter = (
            None if args.trajectory_donor_source_filter == "all"
            else args.trajectory_donor_source_filter
        )
        pairs = discover_vista4d_transfer_pairs_from_metadata(
            eval_root / "metadata.csv",
            camera_root=eval_root / "cameras",
            video_filter=video_filter,
            source_filter=source_filter,
            donor_source_filter=donor_source_filter,
            trajectories_per_video=args.random_transfer_trajectories_per_video,
            seed=args.random_transfer_seed,
            max_forward_angle_deg=args.max_forward_angle_deg,
            exclude_same_video=not args.allow_same_video_trajectory_donors,
        )
        print(f"[Vista4D] discovered {len(pairs)} random-transfer "
              f"(source, output_camera, donor_video, donor_camera) pairs "
              f"(source_filter={source_filter}, donor_source_filter={donor_source_filter}, "
              f"video_filter={video_filter}, "
              f"trajectories_per_video={args.random_transfer_trajectories_per_video}, "
              f"max_forward_angle_deg={args.max_forward_angle_deg})")
    else:
        pairs = [
            (video, camera, video, camera)
            for video, camera in discover_vista4d_pairs(
                eval_root,
                video_filter=video_filter,
                source_filter=source_filter,
                max_cameras_per_video=args.max_cameras_per_video,
            )
        ]
        print(f"[Vista4D] discovered {len(pairs)} (video, camera) pairs "
              f"(source_filter={source_filter}, video_filter={video_filter}, "
              f"max_cameras_per_video={args.max_cameras_per_video})")

    pipe = build_eval_pipeline(args)
    prompts_to_cache = [DEFAULT_PROMPT, NEGATIVE_PROMPT]
    prompts_to_cache.extend(
        _lookup_caption(eval_root / "metadata.csv", video, camera)
        for video, camera, _, _ in pairs
    )
    precompute_eval_prompt_embs(pipe, prompts_to_cache)

    for video, camera, trajectory_video, trajectory_camera in pairs:
        pair_out_dir = output_root / video / camera
        out_mp4 = pair_out_dir / f"{camera}.mp4"
        if out_mp4.exists() and not args.force_rerun:
            print(f"[Vista4D] skip {video}/{camera}: {out_mp4} already exists")
            continue
        pair_args = args
        if args.seed_from_vista4d_metadata:
            pair_args = copy.copy(args)
            pair_args.seed = lookup_vista4d_pair_seed(
                eval_root / "metadata.csv", video=video, camera=camera
            )
        try:
            sample = load_vista4d_sample(
                davis_root=args.davis_root,
                davis_4rc_root=args.davis_4rc_root,
                vista4d_eval_root=eval_root,
                video=video,
                camera=camera,
                trajectory_video=trajectory_video,
                trajectory_camera=trajectory_camera,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                start_frame=args.start_frame,
                vista4d_render_w=args.vista4d_render_w,
                vista4d_render_h=args.vista4d_render_h,
                trajectory_damping=args.trajectory_damping,
                trajectory_anchor_mode=args.trajectory_anchor_mode,
                source_backend=args.source_backend,
                cow_root=args.cow_root,
                vggt_root=args.vggt_root,
                cow_track_sample_mode=args.cow_track_sample_mode,
                cow_track_stride=args.cow_track_stride,
                cow_conf_threshold=args.cow_conf_threshold,
                cow_depth_conf_threshold=args.cow_depth_conf_threshold,
                cow_vggt_scale_align=args.cow_vggt_scale_align,
                vista4d_target_scale_align=args.vista4d_target_scale_align,
                vista4d_target_gauge_align=args.vista4d_target_gauge_align,
                target_translation_depth_fraction=args.target_translation_depth_fraction,
                gmos_root=args.gmos_root,
                gmos_mask_subdir=args.gmos_mask_subdir,
                gmos_dilate_px=args.gmos_dilate_px,
                require_gmos_mask=args.require_gmos_mask,
                target_dynamic_mask_mode=args.target_dynamic_mask_mode,
                output_fps=args.output_fps,
                seed=pair_args.seed,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[Vista4D] skip {video}/{camera}: {exc}")
            continue

        if "pointmap_sim3_alignment" in sample:
            pair_out_dir.mkdir(parents=True, exist_ok=True)
            (pair_out_dir / "pointmap_sim3_alignment.json").write_text(
                json.dumps(sample["pointmap_sim3_alignment"], indent=2, sort_keys=True)
            )

        if args.drop_4rc_tracks:
            sample["tracks_world"] = None
            sample["source_rgb_at_tracks"] = None
            sample["has_tracks"] = False
            sample["has_dynamic_tracks"] = False
            sample["track_video_loss_mask"] = 0.0

        print(f"[Vista4D] running {video}/{camera} "
              f"(donor: {trajectory_video}/{trajectory_camera}; "
              f"seed: {pair_args.seed}; prompt: {sample['prompt'][:80]}...)")
        _run_one_pair(pipe, sample, pair_args, pair_out_dir)
        # novel_view_payload.npz removed — derivable from Vista4D source
        # + trajectory args. See eval_track4d360_davis_4rc_novel_traj.py.

    print(f"[Vista4D] complete -> {output_root}")


if __name__ == "__main__":
    main()
