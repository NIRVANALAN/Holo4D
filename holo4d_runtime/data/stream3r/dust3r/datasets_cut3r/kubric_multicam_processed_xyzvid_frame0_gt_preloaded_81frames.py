"""
Kubric 81/82-frame dataset with preloaded XYZVid(frame0-colorized) track-video.

This avoids the very slow online KD-tree colorization in KubricMultiCameraProcessed_XYZVidFrame0GT
by loading a precomputed MP4 exported for specific window start offsets and (optionally) reversed
temporal order.

Expected export layout (example for W=480,H=256):
  {preload_root_base}/start{start_offset}/res480x256/{split}/{scene_id}/camera_{camera_id}/xyzvid_frame0gt.mp4
  {preload_root_base}/start{start_offset}/res480x256/{split}/{scene_id}/camera_{camera_id}/xyzvid_frame0gt_rev.mp4

For 82-frame sequences trained as 49-frame clips, the "frame0" reference depends on temporal order.
Forward and reversed clips are NOT equivalent; to support reverse augmentation with preloads, you
must also export `xyzvid_frame0gt_rev.mp4` using reverse_prob=1.0 during preprocessing.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import imageio
import numpy as np
import torch

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_stereo_view_dataset import BaseStereoViewDataset
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.kubric_multicam_processed_xyzvid_frame0_gt import (
    KubricMultiCameraProcessed_XYZVidFrame0GT,
    _to_torch,
)
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.kubric_multicam_processed_xyzvid_frame0_gt_81frames import (
    KubricMultiCameraProcessed_XYZVidFrame0GT_81Frames,
)
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import geotrf as dust3r_geotrf
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import inv as dust3r_inv


class KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded_81Frames(KubricMultiCameraProcessed_XYZVidFrame0GT_81Frames):
    def __init__(
        self,
        *args,
        preload_root_base: Optional[str] = None,
        preload_resolution: Sequence[int] | None = None,
        preload_allow_resize: bool = True,
        preload_verbose: bool = False,
        preload_filename: str = "xyzvid_frame0gt.mp4",
        preload_filename_reversed: str = "xyzvid_frame0gt_rev.mp4",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.preload_root_base = preload_root_base
        self.preload_allow_resize = bool(preload_allow_resize)
        self.preload_verbose = bool(preload_verbose)
        self.preload_filename = str(preload_filename)
        self.preload_filename_reversed = str(preload_filename_reversed)

        if preload_resolution is None:
            self.preload_resolution = None
        else:
            pr = tuple(int(x) for x in list(preload_resolution))
            if len(pr) != 2:
                raise ValueError(f"preload_resolution must be (W,H), got: {preload_resolution}")
            self.preload_resolution = (int(pr[0]), int(pr[1]))

        self._preload_success_count = 0
        self._preload_fallback_count = 0
        self._last_preload_success = False
        self._preload_shape_mismatch_warn_count = 0
        self._preload_resized_count = 0

        if self.preload_verbose:
            if self.preload_root_base:
                print(
                    "[KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded_81Frames] "
                    f"Will attempt to load preprocessed videos from: {self.preload_root_base}"
                )
            else:
                print(
                    "[KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded_81Frames] "
                    "No preload_root_base specified, will compute online (slower)."
                )

    def _get_preload_video_path(
        self,
        scene_idx: int,
        camera_idx: int,
        *,
        start_offset: int,
        export_resolution: Tuple[int, int],
        is_reversed: bool,
    ) -> str:
        export_w, export_h = int(export_resolution[0]), int(export_resolution[1])
        scene_name = f"{int(scene_idx):05d}"
        filename = self.preload_filename_reversed if bool(is_reversed) else self.preload_filename
        return os.path.join(
            str(self.preload_root_base),
            f"start{int(start_offset)}",
            f"res{int(export_w)}x{int(export_h)}",
            str(self.split),
            scene_name,
            f"camera_{int(camera_idx)}",
            filename,
        )

    def _load_preloaded_track(
        self,
        scene_idx: int,
        camera_idx: int,
        *,
        start_offset: int,
        is_reversed: bool,
        num_views: int,
        target_resolution: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], bool]:
        if not self.preload_root_base:
            return None, False

        target_w, target_h = int(target_resolution[0]), int(target_resolution[1])

        # Prefer exact resolution match exports to guarantee alignment with RGB crop+resize.
        candidates: List[Tuple[int, int]] = [(target_w, target_h)]

        # Optional fallback: load from a different exported resolution and crop+resize to target.
        # When preload_allow_resize is True, any resolution is accepted (center-crop handles alignment).
        # Otherwise, only same-aspect-ratio fallbacks are allowed (pure resize, no crop).
        if self.preload_resolution is not None:
            fw, fh = self.preload_resolution
            if int(fw) > 0 and int(fh) > 0 and (int(fw), int(fh)) not in candidates:
                if self.preload_allow_resize:
                    candidates.append((int(fw), int(fh)))
                else:
                    ar_t = float(target_w) / float(target_h)
                    ar_f = float(int(fw)) / float(int(fh))
                    if abs(ar_t - ar_f) < 1e-8:
                        candidates.append((int(fw), int(fh)))

        video_path = None
        used_export_res = None
        for exp_w, exp_h in candidates:
            p = self._get_preload_video_path(
                scene_idx,
                camera_idx,
                start_offset=int(start_offset),
                export_resolution=(int(exp_w), int(exp_h)),
                is_reversed=bool(is_reversed),
            )
            if os.path.exists(p):
                video_path = p
                used_export_res = (int(exp_w), int(exp_h))
                break

        if video_path is None:
            return None, False

        try:
            frames: List[np.ndarray] = []
            reader = imageio.get_reader(video_path)
            try:
                for i, frame in enumerate(reader):
                    if i >= int(num_views):
                        break
                    frame = np.asarray(frame)
                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise ValueError(f"Unexpected frame shape {frame.shape} in {video_path}")

                    if frame.shape[1] != target_w or frame.shape[0] != target_h:
                        if not self.preload_allow_resize:
                            self._preload_shape_mismatch_warn_count += 1
                            if self.preload_verbose and self._preload_shape_mismatch_warn_count <= 5:
                                print(
                                    "[Preload] Shape mismatch for xyzvid_frame0gt.mp4 "
                                    f"{frame.shape[1]}x{frame.shape[0]} vs target {target_w}x{target_h}; "
                                    "set preload_allow_resize=True or re-export."
                                )
                            return None, False

                        # Use the same crop+resize pipeline as RGB preprocessing (principal-point centered).
                        intr = np.array(
                            [
                                [1.0, 0.0, float(frame.shape[1]) / 2.0],
                                [0.0, 1.0, float(frame.shape[0]) / 2.0],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        )
                        frame_pil, _, _ = self._crop_resize_if_necessary(
                            frame,
                            depthmap=None,
                            intrinsics=intr,
                            resolution=(target_w, target_h),
                            rng=np.random.default_rng(seed=0),
                            info=f"preloaded_xyzvid_frame0gt/{int(scene_idx):05d}/camera_{int(camera_idx)}",
                            allow_random_transpose=False,
                        )
                        frame = np.asarray(frame_pil)
                        self._preload_resized_count += 1

                    frame_f = frame.astype(np.float32) / 255.0 * 2.0 - 1.0
                    frames.append(frame_f)
            finally:
                reader.close()

            if len(frames) != int(num_views):
                return None, False

            track_frames = np.stack(frames, axis=0)  # (T,H,W,3) in [-1,1]
            return torch.from_numpy(track_frames).float(), True
        except Exception as e:
            if self.preload_verbose:
                import traceback

                print(f"Warning: Failed to load preprocessed video from {video_path}: {e}")
                traceback.print_exc()
            return None, False

    def _get_views(self, idx, resolution, rng):
        # Let 81Frames choose the actual clip (start_offset + optional reverse) and load views.
        views = super()._get_views(idx, resolution, rng)
        if not views:
            self._last_preload_success = False
            return views

        (target_resolution, num_views_cfg) = resolution
        target_w, target_h = int(target_resolution[0]), int(target_resolution[1])
        num_views = min(int(num_views_cfg), int(len(views)))

        start_offset = int(getattr(self, "_last_window_start_offset", 0))
        is_reversed = bool(getattr(self, "_last_window_is_reversed", False))

        meta_idx = self.valid_indices[idx]
        scene_idx = int(self.scene_idxs[meta_idx])
        camera_idx = int(self.camera_idxs[meta_idx])

        preloaded_xyz_cam0, success = self._load_preloaded_track(
            scene_idx,
            camera_idx,
            start_offset=start_offset,
            is_reversed=is_reversed,
            num_views=num_views,
            target_resolution=(target_w, target_h),
        )
        if not success or preloaded_xyz_cam0 is None:
            self._preload_fallback_count += 1
            self._last_preload_success = False
            return views

        cam0_pose = torch.as_tensor(views[0]["camera_pose"], dtype=torch.float32)
        for t in range(num_views):
            view = views[t]
            xyz_cam0_frame = preloaded_xyz_cam0[t]  # (H,W,3) in cam0 coords, already [-1,1]

            if getattr(self, "preserve_original_track", False) and "track" in view:
                view["_track_dynamic_cam_preload"] = np.asarray(view["track"], dtype=np.float32).copy()
                if "track_valid_mask" in view:
                    view["_track_dynamic_valid_preload"] = np.asarray(view["track_valid_mask"], dtype=bool).copy()
                view["_track_video_raw_prenorm"] = xyz_cam0_frame.cpu().numpy().astype(np.float32)

            cam_t_pose = torch.as_tensor(view["camera_pose"], dtype=torch.float32)
            cam0_to_cam_t = torch.inverse(cam_t_pose) @ cam0_pose
            xyz_cam_t = dust3r_geotrf(cam0_to_cam_t, xyz_cam0_frame)
            view["track"] = xyz_cam_t.cpu().numpy().astype(np.float32)
            view["track_query_idx"] = 0
            view["dataset"] = "KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded_81Frames"

        self._preload_success_count += 1
        self._last_preload_success = True
        return views

    def __getitem__(self, idx):
        # Bypass XYZVidFrame0GT.__getitem__ KD-tree when preload succeeds.
        self._last_preload_success = False
        views = BaseStereoViewDataset.__getitem__(self, idx)
        if not views:
            return views

        if self._last_preload_success and getattr(self, "preserve_original_track", False):
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
                cam_pose = np.asarray(view["camera_pose"], dtype=np.float32)
                R_cam2world = cam_pose[:3, :3]
                t_cam2world = cam_pose[:3, 3]
                dyn_world = (np.einsum("ik,hwk->hwi", R_cam2world, dyn_cam) + t_cam2world[None, None, :]).astype(
                    np.float32
                )

                if dyn_valid is None:
                    if "track_valid_mask" in view:
                        dyn_valid = np.asarray(view["track_valid_mask"], dtype=bool)
                    else:
                        dyn_valid = np.ones(dyn_world.shape[:2], dtype=bool)
                else:
                    dyn_valid = np.asarray(dyn_valid, dtype=bool)

                view["track_dynamic_world"] = dyn_world
                view["track_dynamic_valid_mask"] = dyn_valid

            return views

        # Fallback: run KD-tree computation on the *already loaded* views.
        # Do NOT call KubricMultiCameraProcessed_XYZVidFrame0GT.__getitem__, because it would
        # resample a different random window and break alignment.
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

        trees = self._build_kdtrees(segm0=segm0, obj0=obj0, color0_cam0=color0_cam0, valid0=valid0)

        for view in views:
            segm_t = _to_torch(view["segm"], dtype=torch.int64)
            obj_t = _to_torch(view["obj_coords"], dtype=torch.float32)
            valid_t = _to_torch(view["valid_mask"], dtype=torch.bool)

            colored_cam0, colored_valid = self._colorize_frame(segm_t=segm_t, obj_t=obj_t, valid_t=valid_t, trees=trees)
            colored_world = dust3r_geotrf(cam0_pose, colored_cam0)

            if self.preserve_original_track and "track" in view:
                view["track_dynamic_world"] = view["track"]
                if "track_valid_mask" in view:
                    view["track_dynamic_valid_mask"] = view["track_valid_mask"]
                # Expose frame0-colorized target in cam0 coords as pre-normalized track_video
                # (same semantic contract as preloaded mp4 path: already in [-1,1]).
                # Invalid pixels are set to -1.0 to match the MP4 convention (0 uint8 = -1.0 float).
                _tv_prenorm = colored_cam0.cpu().numpy().astype(np.float32)
                _tv_valid_np = colored_valid.cpu().numpy().astype(bool)
                _tv_prenorm[~_tv_valid_np] = -1.0
                view["track_video_prenormalized"] = _tv_prenorm
                view["track_video_valid_mask"] = _tv_valid_np

            view["track"] = colored_world.cpu().numpy().astype(np.float32)
            view["track_valid_mask"] = colored_valid.cpu().numpy().astype(bool)
            view["track_query_idx"] = 0
            view["dataset"] = "KubricMultiCameraProcessed_XYZVidFrame0GT"

        return views


__all__ = ["KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded_81Frames"]
