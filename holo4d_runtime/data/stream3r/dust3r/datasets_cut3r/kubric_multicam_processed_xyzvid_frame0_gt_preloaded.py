"""
KubricMultiCameraProcessed_XYZVidFrame0GT with support for loading preprocessed MP4 files.

This variant can load pre-computed xyzvid_frame0gt.mp4 files to avoid expensive online
KD-tree building and colorization during training.

Usage:
    dataset = KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded(
        ROOT='path/to/dataset',
        preload_root='path/to/preprocessed',  # Optional
        ...
    )

If preload_root is provided and the MP4 file exists, it loads from there.
Otherwise, falls back to the original online computation.
"""

from __future__ import annotations
import os
from typing import List, Tuple, Optional
import numpy as np
import torch
import imageio

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.kubric_multicam_processed_xyzvid_frame0_gt import (
    KubricMultiCameraProcessed_XYZVidFrame0GT,
    _to_torch,
)
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.kubric_multicam_processed import (
    KubricMultiCameraProcessed,
)
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_stereo_view_dataset import BaseStereoViewDataset
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import geotrf as dust3r_geotrf
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import inv as dust3r_inv


class KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded(KubricMultiCameraProcessed_XYZVidFrame0GT):
    """
    Extended version that can load preprocessed xyzvid_frame0gt.mp4 files.
    """

    def __init__(
        self,
        *args,
        preload_root: str = None,
        preload_allow_resize: bool = True,
        preload_verbose: bool = False,
        preload_require_success: bool = False,
        **kwargs,
    ):
        """
        Args:
            preload_root: Optional root directory containing preprocessed MP4 files.
                Structure: {preload_root}/{split}/{scene_id}/camera_{camera_id}/xyzvid_frame0gt.mp4
                If None or file doesn't exist, falls back to online computation.
            preload_allow_resize: If True, allow resizing preloaded MP4 frames when shape
                mismatches runtime target resolution. Uses center-crop + resize to preserve
                alignment with RGB preprocessing. If False, mismatch falls back to online.
            preload_require_success: If True, never allow silent online fallback when
                preload_root is set for forward clips. Reversed clips (when sampled) are
                still allowed to run online because this class only preloads forward exports.
            *args, **kwargs: Arguments passed to KubricMultiCameraProcessed_XYZVidFrame0GT
        """
        super().__init__(*args, **kwargs)
        self.preload_root = preload_root
        # Default behavior: center-crop + resize on mismatch.
        # For Kubric, principal point is centered, so this mirrors RGB path well.
        self.preload_allow_resize = bool(preload_allow_resize)
        self.preload_verbose = bool(preload_verbose)
        self.preload_require_success = bool(preload_require_success)

        self._preload_success_count = 0
        self._preload_fallback_count = 0
        self._last_preload_success = False
        self._last_preload_failure_reason = None
        self._preload_shape_mismatch_warn_count = 0
        self._preload_resized_count = 0

        if self.preload_root:
            if self.preload_verbose:
                print(
                    "[KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded] Will attempt to load "
                    f"preprocessed videos from: {self.preload_root}"
                )
        else:
            if self.preload_verbose:
                print(
                    "[KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded] No preload_root specified, "
                    "will compute online (slower)"
                )

    def _get_preload_video_path(self, scene_idx: int, camera_idx: int) -> str:
        """Get the path to the preprocessed video file."""
        scene_name = f"{scene_idx:05d}"
        return os.path.join(
            self.preload_root,
            self.split,
            scene_name,
            f"camera_{camera_idx}",
            "xyzvid_frame0gt.mp4"
        )

    def _load_preloaded_track(
        self,
        scene_idx: int,
        camera_idx: int,
        frame_indices: List[int],
        target_resolution: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """
        Load preprocessed xyzvid_frame0gt.mp4 if it exists.

        Args:
            scene_idx: Scene index
            camera_idx: Camera index
            frame_indices: List of frame indices to load (supports non-sequential)
            target_resolution: (width, height) to resize to if needed

        Returns:
            tuple: (track_frames, success) where track_frames is (T,H,W,3) in [-1,1] and success is bool
        """
        self._last_preload_failure_reason = None
        if not self.preload_root:
            self._last_preload_failure_reason = "preload_root is not set"
            return None, False

        video_path = self._get_preload_video_path(scene_idx, camera_idx)

        if not os.path.exists(video_path):
            self._last_preload_failure_reason = f"missing file: {video_path}"
            return None, False

        try:
            # Load video using imageio (same as LowMemoryVideo class in diffsynth.utils.data)
            reader = imageio.get_reader(video_path)

            # Read all frames
            all_frames = []
            for frame in reader:
                all_frames.append(np.array(frame))
            reader.close()

            if len(all_frames) == 0:
                print(f"Warning: Video has no frames: {video_path}")
                self._last_preload_failure_reason = f"video has no frames: {video_path}"
                return None, False

            target_w, target_h = target_resolution
            frames = []
            for frame_idx in frame_indices:
                if frame_idx >= len(all_frames):
                    print(f"Warning: Requested frame {frame_idx} but video only has {len(all_frames)} frames")
                    self._last_preload_failure_reason = (
                        f"requested frame {frame_idx} but video has {len(all_frames)} frames: {video_path}"
                    )
                    return None, False

                frame = all_frames[frame_idx]

                # Accept exact shape, or (when enabled) resize via the shared
                # stereo dataset crop+resize pipeline.
                h, w = frame.shape[:2]
                if w != target_w or h != target_h:
                    if not self.preload_allow_resize:
                        self._preload_shape_mismatch_warn_count += 1
                        if self._preload_shape_mismatch_warn_count <= 10:
                            print(
                                "[Preload] Shape mismatch for xyzvid_frame0gt.mp4 "
                                f"(got {w}x{h}, expected {target_w}x{target_h}); "
                                "fallback to online computation to keep exact alignment."
                            )
                        self._last_preload_failure_reason = (
                            f"shape mismatch without resize permission: got {w}x{h}, "
                            f"expected {target_w}x{target_h} ({video_path})"
                        )
                        return None, False
                    # Reuse shared preprocessing path instead of local custom resize logic.
                    # With centered intrinsics this gives the same center-crop+resize behavior.
                    intrinsics = np.array(
                        [[1.0, 0.0, 0.5 * float(w)],
                         [0.0, 1.0, 0.5 * float(h)],
                         [0.0, 0.0, 1.0]],
                        dtype=np.float32,
                    )
                    frame_pil, _, _ = self._crop_resize_if_necessary(
                        frame,
                        None,
                        intrinsics,
                        (int(target_w), int(target_h)),
                        rng=np.random.default_rng(seed=0),
                        info=f"preloaded_xyzvid_frame0gt/{scene_idx:05d}/camera_{camera_idx}",
                        allow_random_transpose=False,
                    )
                    frame = np.asarray(frame_pil)
                    self._preload_resized_count += 1
                    if self.preload_verbose and self._preload_resized_count <= 1:
                        print(
                            "[Preload] Resized xyzvid_frame0gt.mp4 via shared crop+resize pipeline "
                            f"(from {w}x{h} to {target_w}x{target_h})."
                        )

                # Convert from uint8 [0,255] to float [-1,1]
                frame_float = (frame.astype(np.float32) / 255.0) * 2.0 - 1.0
                frames.append(frame_float)

            track_frames = np.stack(frames, axis=0)  # (T,H,W,3)
            return torch.from_numpy(track_frames).float(), True

        except Exception as e:
            import traceback
            print(f"Warning: Failed to load preprocessed video from {video_path}: {e}")
            traceback.print_exc()
            self._last_preload_failure_reason = f"exception while reading {video_path}: {type(e).__name__}: {e}"
            return None, False

    def _get_views(self, idx, resolution, rng):
        """
        Override _get_views to load preprocessed track data if available.

        When preload succeeds, replaces the grandparent's bbox-based tracks with
        preloaded frame0-colorized tracks in each view's camera coordinates.
        BaseStereoViewDataset.__getitem__ will then transform cam_t -> world.
        """
        # Extract resolution info
        (target_w, target_h), num_views = resolution

        # Get metadata
        meta_idx = self.valid_indices[idx]
        scene_idx = self.scene_idxs[meta_idx]
        camera_idx = self.camera_idxs[meta_idx]

        # Determine frame indices using same logic as parent
        total_frames = self._n_frames
        num_views = min(num_views, total_frames)

        if num_views <= 0:
            return []

        # Sequential sampling (matches preprocessing which uses max_interval=1)
        if self.max_interval > 1:
            if self.preload_require_success and self.preload_root:
                raise RuntimeError(
                    "preload_require_success=True but max_interval>1 requires non-sequential frame sampling "
                    "that cannot be served by a single preprocessed xyzvid_frame0gt.mp4 clip."
                )
            # For non-sequential sampling, fall back to online computation
            self._last_preload_success = False
            return super()._get_views(idx, resolution, rng)

        # Per-sample reverse handling:
        # - Forward clips: attempt preloaded mp4
        # - Reversed clips: run online KD-tree fallback (no reversed preload file in this class)
        # Keep the RNG draw here so reverse probability matches parent semantics.
        if self.reverse_prob > 0.0:
            reverse_selected = bool(rng.random() < self.reverse_prob)
            if reverse_selected:
                self._preload_fallback_count += 1
                if self.preload_verbose and self._preload_fallback_count <= 5:
                    print(
                        f"[Preload] Reverse clip selected for scene {scene_idx}, camera {camera_idx}; "
                        "using online KD-tree fallback."
                    )
                self._last_preload_success = False
                # Force reverse for the fallback call so this sample's temporal order matches the
                # sampled reverse branch (instead of re-sampling reversal inside parent).
                original_reverse_prob = self.reverse_prob
                try:
                    self.reverse_prob = 1.0
                    return super()._get_views(idx, resolution, rng)
                finally:
                    self.reverse_prob = original_reverse_prob

            if self.preload_verbose and self._preload_success_count < 1:
                print(
                    f"[Preload] Forward clip selected for scene {scene_idx}, camera {camera_idx}; "
                    "attempting preloaded xyzvid."
                )

        frame_indices = list(range(num_views))

        # Try to load preprocessed data
        preloaded_xyz_cam0, success = self._load_preloaded_track(
            scene_idx, camera_idx, frame_indices, (target_w, target_h)
        )

        if not success:
            # Fall back to parent's online computation
            self._preload_fallback_count += 1
            failure_reason = self._last_preload_failure_reason or "unknown preload failure"
            if self.preload_require_success and self.preload_root:
                raise RuntimeError(
                    "preload_require_success=True but preloaded xyzvid was unavailable. "
                    f"scene={int(scene_idx):05d}, camera={int(camera_idx)}, "
                    f"path={self._get_preload_video_path(scene_idx, camera_idx)}, reason={failure_reason}"
                )
            if self.preload_verbose and self._preload_fallback_count <= 1:
                print(
                    f"[Preload] Fallback to online computation for scene {scene_idx}, camera {camera_idx}; "
                    f"reason: {failure_reason}"
                )
            self._last_preload_success = False
            return super()._get_views(idx, resolution, rng)

        self._preload_success_count += 1
        if self.preload_verbose and self._preload_success_count <= 1:
            print(f"[Preload] Successfully loaded from preprocessed video for scene {scene_idx}, camera {camera_idx}")

        # Load views from grandparent (RGB, depth, poses, bbox-based tracks)
        views = KubricMultiCameraProcessed._get_views(self, idx, resolution, rng)

        if not views:
            self._last_preload_success = False
            return views

        # Replace grandparent's bbox-based tracks with preloaded frame0-colorized tracks.
        # Store in each view's CAMERA coordinates so BaseStereoViewDataset.__getitem__
        # correctly transforms them to world coordinates.
        cam0_pose = torch.as_tensor(views[0]["camera_pose"], dtype=torch.float32)

        for t, view in enumerate(views):
            if t >= preloaded_xyz_cam0.shape[0]:
                break

            xyz_cam0_frame = preloaded_xyz_cam0[t]  # (H,W,3) in cam0 coords

            if self.preserve_original_track and "track" in view:
                # Preserve original dynamic track (camera coords) and promote to world coords in __getitem__
                # after BaseStereoViewDataset has finalized/validated the sample.
                view["_track_dynamic_cam_preload"] = np.asarray(view["track"], dtype=np.float32).copy()
                if "track_valid_mask" in view:
                    view["_track_dynamic_valid_preload"] = np.asarray(view["track_valid_mask"], dtype=bool).copy()

                # Save raw preloaded values (already quantile-normalized to [-1,1] by preprocessing)
                # BEFORE any geometric transforms. This allows the pipeline to skip all
                # normalization and pass directly to VAE.
                view["_track_video_raw_prenorm"] = preloaded_xyz_cam0[t].cpu().numpy().astype(np.float32)

            # Convert cam0 -> cam_t coordinates: inv(cam_t_pose) @ cam0_pose
            cam_t_pose = torch.as_tensor(view["camera_pose"], dtype=torch.float32)
            cam0_to_cam_t = torch.inverse(cam_t_pose) @ cam0_pose
            xyz_cam_t = dust3r_geotrf(cam0_to_cam_t, xyz_cam0_frame)

            view["track"] = xyz_cam_t.cpu().numpy().astype(np.float32)
            # Keep track_valid_mask from grandparent (query frame depth validity)

        self._last_preload_success = True
        return views

    def __getitem__(self, idx):
        """
        Override __getitem__ to bypass XYZVidFrame0GT's expensive KD-tree
        computation when preloaded track data is available.

        Call chain:
          - BaseStereoViewDataset.__getitem__ calls self._get_views (our override)
          - If preload succeeded: track is already in views (cam_t coords),
            BaseStereoViewDataset transforms to world. Done.
          - If preload failed: run KD-tree fallback (same as XYZVidFrame0GT).
        """
        self._last_preload_success = False

        # Call BaseStereoViewDataset.__getitem__ directly, bypassing
        # XYZVidFrame0GT.__getitem__'s KD-tree computation
        views = BaseStereoViewDataset.__getitem__(self, idx)

        if not views:
            return views

        if self._last_preload_success:
            # Track loaded from preprocessed video, correctly transformed
            # to world coords and validated by BaseStereoViewDataset
            if self.preserve_original_track:
                usable_seg_ids = None
                if views and "segm" in views[0] and "valid_mask" in views[0] and "obj_coords" in views[0]:
                    segm0 = np.asarray(views[0]["segm"])
                    valid0 = np.asarray(views[0]["valid_mask"], dtype=bool)
                    obj0 = np.asarray(views[0]["obj_coords"], dtype=np.float32)
                    if segm0.ndim == 3 and segm0.shape[-1] == 1:
                        segm0 = segm0[..., 0]
                    finite_obj0 = np.isfinite(obj0).all(axis=-1)
                    mask0 = valid0 & (segm0 > 0) & finite_obj0
                    seg_ids = np.unique(segm0[mask0]) if np.any(mask0) else np.array([], dtype=segm0.dtype)
                    usable = []
                    for seg_id in seg_ids.tolist():
                        m = mask0 & (segm0 == seg_id)
                        if int(np.count_nonzero(m)) >= 4:
                            usable.append(int(seg_id))
                    if len(usable) > 0:
                        usable_seg_ids = np.asarray(usable, dtype=np.int64)
                for view in views:
                    # Extract raw prenormalized track_video (already in [-1,1] from preprocessing)
                    raw_prenorm = view.pop("_track_video_raw_prenorm", None)
                    if raw_prenorm is not None:
                        raw_prenorm = np.asarray(raw_prenorm, dtype=np.float32)
                        track_world = np.asarray(view.get("track"), dtype=np.float32) if "track" in view else None
                        if track_world is not None and raw_prenorm.shape[:2] != track_world.shape[:2]:
                            if raw_prenorm.shape[0] == track_world.shape[1] and raw_prenorm.shape[1] == track_world.shape[0]:
                                raw_prenorm = raw_prenorm.swapaxes(0, 1)
                        view["track_video_prenormalized"] = raw_prenorm
                        tv_valid = None
                        if "valid_mask" in view and "segm" in view:
                            segm = np.asarray(view["segm"])
                            valid = np.asarray(view["valid_mask"], dtype=bool)
                            if segm.ndim == 3 and segm.shape[-1] == 1:
                                segm = segm[..., 0]
                            if segm.shape == valid.shape:
                                if usable_seg_ids is not None and usable_seg_ids.size > 0:
                                    tv_valid = valid & np.isin(segm, usable_seg_ids)
                                else:
                                    tv_valid = valid & (segm > 0)
                        if tv_valid is None and "track_valid_mask" in view:
                            tv_valid = np.asarray(view["track_valid_mask"], dtype=bool)
                        if tv_valid is not None:
                            if tv_valid.shape != raw_prenorm.shape[:2]:
                                if tv_valid.shape[0] == raw_prenorm.shape[1] and tv_valid.shape[1] == raw_prenorm.shape[0]:
                                    tv_valid = tv_valid.swapaxes(0, 1)
                            if tv_valid.shape != raw_prenorm.shape[:2]:
                                raise ValueError(
                                    f"track_video_valid_mask shape {tv_valid.shape} does not match "
                                    f"track_video_prenormalized spatial {raw_prenorm.shape[:2]}"
                                )
                            view["track_video_valid_mask"] = tv_valid

                    dyn_cam = view.pop("_track_dynamic_cam_preload", None)
                    dyn_valid = view.pop("_track_dynamic_valid_preload", None)
                    if dyn_cam is None:
                        continue

                    dyn_cam = np.asarray(dyn_cam, dtype=np.float32)
                    track_world = np.asarray(view.get("track"), dtype=np.float32) if "track" in view else None
                    if track_world is not None and dyn_cam.shape[:2] != track_world.shape[:2]:
                        if dyn_cam.shape[0] == track_world.shape[1] and dyn_cam.shape[1] == track_world.shape[0]:
                            dyn_cam = dyn_cam.swapaxes(0, 1)
                            if dyn_valid is not None:
                                dyn_valid = np.asarray(dyn_valid, dtype=bool).swapaxes(0, 1)

                    cam_pose = np.asarray(view["camera_pose"], dtype=np.float32)
                    R_cam2world = cam_pose[:3, :3]
                    t_cam2world = cam_pose[:3, 3]
                    dyn_world = (
                        np.einsum("ik,hwk->hwi", R_cam2world, dyn_cam)
                        + t_cam2world[None, None, :]
                    ).astype(np.float32)

                    if dyn_valid is None:
                        if "track_valid_mask" in view:
                            dyn_valid = np.asarray(view["track_valid_mask"], dtype=bool)
                        else:
                            dyn_valid = np.ones(dyn_world.shape[:2], dtype=bool)
                    else:
                        dyn_valid = np.asarray(dyn_valid, dtype=bool)

                    if dyn_valid.shape != dyn_world.shape[:2]:
                        raise ValueError(
                            f"Preserved dynamic-track valid mask shape {dyn_valid.shape} "
                            f"does not match dynamic track shape {dyn_world.shape[:2]}"
                        )

                    view["track_dynamic_world"] = dyn_world
                    view["track_dynamic_valid_mask"] = dyn_valid
            return views

        # Fallback: run KD-tree computation (same as XYZVidFrame0GT.__getitem__)
        ref = views[0]
        if "obj_coords" not in ref or "segm" not in ref:
            raise KeyError("XYZVidFrame0GT requires 'obj_coords' and 'segm' in dataset views.")

        cam0_pose = _to_torch(ref["camera_pose"], dtype=torch.float32)
        inv_cam0 = dust3r_inv(cam0_pose)
        pts0_world = _to_torch(ref["pts3d"], dtype=torch.float32)
        valid0 = _to_torch(ref["valid_mask"], dtype=torch.bool)
        segm0 = _to_torch(ref["segm"], dtype=torch.int64)
        obj0 = _to_torch(ref["obj_coords"], dtype=torch.float32)

        color0_cam0 = dust3r_geotrf(inv_cam0, pts0_world)
        color0_cam0 = self._normalize_color0_cam0(
            color0_cam0,
            valid0=valid0,
            segm0=segm0,
            mode=self.track_color_norm_mode,
            q_low=self.track_color_quantile_low,
            q_high=self.track_color_quantile_high,
        )

        trees = self._build_kdtrees(
            segm0=segm0, obj0=obj0, color0_cam0=color0_cam0, valid0=valid0
        )

        for view in views:
            segm_t = _to_torch(view["segm"], dtype=torch.int64)
            obj_t = _to_torch(view["obj_coords"], dtype=torch.float32)
            valid_t = _to_torch(view["valid_mask"], dtype=torch.bool)

            colored_cam0, colored_valid = self._colorize_frame(
                segm_t=segm_t, obj_t=obj_t, valid_t=valid_t, trees=trees
            )
            colored_world = dust3r_geotrf(cam0_pose, colored_cam0)

            if self.preserve_original_track and "track" in view:
                view["track_dynamic_world"] = view["track"]
                if "track_valid_mask" in view:
                    view["track_dynamic_valid_mask"] = view["track_valid_mask"]
                # Expose frame0-colorized target in cam0 coords as pre-normalized track_video
                # (same semantic contract as preloaded mp4 path: already in [-1,1]).
                _tv_prenorm = colored_cam0.cpu().numpy().astype(np.float32)
                _tv_valid_np = colored_valid.cpu().numpy().astype(bool)
                _tv_prenorm[~_tv_valid_np] = -1.0  # invalid = black, matching MP4 convention
                view["track_video_prenormalized"] = _tv_prenorm
                view["track_video_valid_mask"] = _tv_valid_np

            view["track"] = colored_world.cpu().numpy().astype(np.float32)
            view["track_valid_mask"] = colored_valid.cpu().numpy().astype(bool)
            view["track_query_idx"] = 0
            view["dataset"] = "KubricMultiCameraProcessed_XYZVidFrame0GT"

        return views


__all__ = ["KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded"]
