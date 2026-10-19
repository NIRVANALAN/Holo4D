"""
KubricMultiCameraProcessed variant for 81-frame (actually 82 frames) datasets.

Inherits from KubricMultiCameraProcessed and overrides frame sampling to randomly
select a contiguous window of `num_views` frames (e.g. 49) from the full 82-frame
sequence. This effectively increases dataset diversity by varying the temporal
start offset.

Data structure is identical to the 49-frame version:
ROOT/
    train/
        00001/
            camera_0/
                rgb_00.jpg ... rgb_81.jpg
                depth_00.png ... depth_81.png
                coords_00.png ... coords_81.png
                segm_00.png ... segm_81.png
            ...
    seq_data.train.npz
"""
from __future__ import annotations

import os.path as osp
import numpy as np
from PIL import Image
import cv2

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.kubric_multicam_processed import (
    KubricMultiCameraProcessed,
    get_intrinsics,
    get_transform,
)
from holo4d_runtime.data.stream3r.dust3r.utils.image import imread_cv2


class KubricMultiCameraProcessed_81Frames(KubricMultiCameraProcessed):
    """
    Kubric 81-frame dataset that randomly samples a contiguous sub-window
    of `num_views` frames from the full sequence at each __getitem__ call.

    The full sequence has `_n_frames` frames (82 for the 81-frame preprocessed
    dataset). At training time, a random start offset in [0, _n_frames - num_views]
    is chosen, then `num_views` consecutive frames are loaded. Temporal reversal
    augmentation (reverse_prob) is applied on top of this sub-window.

    All other behavior (track computation, crop/resize, etc.) is inherited from
    KubricMultiCameraProcessed unchanged.
    """

    def __init__(
        self,
        *args,
        fixed_start_offset: int | None = None,
        allowed_start_offsets: list[int] | tuple[int, ...] | None = None,
        allowed_start_offsets_by_resolution: dict | None = None,
        **kwargs,
    ):
        self.fixed_start_offset = None if fixed_start_offset is None else int(fixed_start_offset)
        self.allowed_start_offsets = None
        if allowed_start_offsets is not None:
            offs = [int(x) for x in list(allowed_start_offsets)]
            self.allowed_start_offsets = offs
        self.allowed_start_offsets_by_resolution = None
        if allowed_start_offsets_by_resolution is not None:
            # Expect keys like (W,H) and values like (0,16,32).
            by_res = {}
            for k, v in dict(allowed_start_offsets_by_resolution).items():
                if not isinstance(k, (list, tuple)) or len(k) != 2:
                    raise ValueError(f"allowed_start_offsets_by_resolution keys must be (W,H), got: {k!r}")
                kk = (int(k[0]), int(k[1]))
                vv = [int(x) for x in list(v)]
                by_res[kk] = vv
            self.allowed_start_offsets_by_resolution = by_res
        super().__init__(*args, **kwargs)
        # `num_views` is not a stable attribute on the base dataset; it is provided via
        # the `resolution=[((W,H), num_views), ...]` config. Report it from `_resolutions`.
        num_views_list = []
        try:
            for _wh, nv in getattr(self, "_resolutions", []) or []:
                num_views_list.append(int(nv))
        except Exception:
            num_views_list = []
        num_views_str = "?"
        if num_views_list:
            uniq = sorted(set(num_views_list))
            num_views_str = str(uniq[0]) if len(uniq) == 1 else str(uniq)
        print(
            f"[KubricMultiCameraProcessed_81Frames] "
            f"Total frames per sequence: {self._n_frames}, "
            f"will sample {num_views_str} contiguous frames "
            f"with {'fixed' if self.fixed_start_offset is not None else 'random'} start offset"
        )

    def _get_views(self, idx, resolution, rng):
        """
        Override to randomly pick a contiguous window of num_views frames
        from the full _n_frames sequence, then load exactly those frames.

        This is essentially the parent's _get_views but with frame indices
        shifted by a random start offset so that e.g. instead of always
        loading frames [0..48], we might load [5..53] or [33..81].
        """
        (target_resolution, num_views) = resolution
        total_frames = self._n_frames  # e.g. 82

        # Clamp num_views
        num_views = min(num_views, total_frames)
        if num_views <= 0:
            return []

        # Random (or fixed/allowed) start offset for contiguous window
        max_start = total_frames - num_views
        if max_start <= 0:
            start_offset = 0
        else:
            if self.fixed_start_offset is not None:
                start_offset = max(0, min(int(self.fixed_start_offset), int(max_start)))
            else:
                # Optional resolution-specific restriction (useful when only certain offsets are pre-exported).
                choices = None
                if self.allowed_start_offsets_by_resolution is not None:
                    key = (int(target_resolution[0]), int(target_resolution[1]))
                    offs = self.allowed_start_offsets_by_resolution.get(key)
                    if offs:
                        choices = [o for o in offs if 0 <= int(o) <= int(max_start)]
                if choices is None and self.allowed_start_offsets is not None and len(self.allowed_start_offsets) > 0:
                    choices = [o for o in self.allowed_start_offsets if 0 <= int(o) <= int(max_start)]

                if choices:
                    start_offset = int(rng.choice(choices))
                else:
                    start_offset = int(rng.integers(0, max_start + 1))

        # Build frame indices: contiguous from start_offset
        if self.max_interval <= 1:
            idxs = list(range(start_offset, start_offset + num_views))
        else:
            # Random sampling with interval within the window
            all_ids = list(range(start_offset, start_offset + total_frames - start_offset))
            first = all_ids[0]
            rest = all_ids[1:]
            if rng.random() < 0.8 and len(rest) >= num_views - 1:
                sampled = rng.choice(rest, size=num_views - 1, replace=False)
                sampled_sorted = sorted(sampled)
            else:
                sampled_sorted = rest[:num_views - 1]
            idxs = [first] + list(sampled_sorted)

        # Optional temporal reversal
        is_reversed = False
        if self.reverse_prob > 0.0 and rng.random() < self.reverse_prob:
            idxs = list(reversed(idxs))
            is_reversed = True

        # Expose last window selection for preloaded variants (best-effort; for debugging only).
        self._last_window_start_offset = int(start_offset)
        self._last_window_is_reversed = bool(is_reversed)

        # Query frame selection (within the selected window)
        if self.query_frame < 0:
            query_view_idx = int(rng.integers(0, len(idxs)))
        else:
            query_view_idx = max(0, min(int(self.query_frame), len(idxs) - 1))

        query_frame_id = int(idxs[query_view_idx])

        # Get metadata
        meta_idx = self.valid_indices[idx]
        scene_idx = self.scene_idxs[meta_idx]
        camera_idx = self.camera_idxs[meta_idx]
        min_depth, max_depth = self.min_max_depth[meta_idx]
        cam_positions = self.cam_positions[meta_idx]    # [T_full, 3]
        cam_quaternions = self.cam_quaternions[meta_idx]  # [T_full, 4]
        focal_length = self.focal_length[meta_idx]
        bboxes = self._get_bboxes(meta_idx)  # [n_instances, T_full, 8, 3]

        # Build paths
        scene_name = f"{scene_idx:05d}"
        camera_dir = osp.join(self.data_dir, scene_name, f"camera_{camera_idx}")

        # Load frames
        uint16_max = np.iinfo(np.uint16).max
        frames = {}

        for frame_id in idxs:
            # Load RGB
            rgb_path = osp.join(camera_dir, f"rgb_{frame_id:02d}.{self.img_ext}")
            rgb_image = imread_cv2(rgb_path)
            if rgb_image is None:
                raise FileNotFoundError(f"Could not load RGB: {rgb_path}")

            # Load depth
            depth_path = osp.join(camera_dir, f"depth_{frame_id:02d}.png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Could not load depth: {depth_path}")

            # Load object coordinates
            coords_path = osp.join(camera_dir, f"coords_{frame_id:02d}.png")
            obj_coords = cv2.imread(coords_path, cv2.IMREAD_UNCHANGED)
            if obj_coords is None:
                raise FileNotFoundError(f"Could not load coords: {coords_path}")

            # Load segmentation
            segm_path = osp.join(camera_dir, f"segm_{frame_id:02d}.png")
            segm = cv2.imread(segm_path, cv2.IMREAD_UNCHANGED)
            if segm is None:
                raise FileNotFoundError(f"Could not load segm: {segm_path}")
            if segm.ndim == 3 and segm.shape[2] == 1:
                segm = segm[:, :, 0]

            # Rescale depth and coords
            depth = depth.astype(np.float32) / uint16_max * (max_depth - min_depth) + min_depth
            obj_coords = obj_coords.astype(np.float32) / uint16_max
            segm = segm.astype(np.int32)

            frames[frame_id] = {
                "rgb": rgb_image,
                "depth": depth,
                "obj_coords": obj_coords,
                "segm": segm,
            }

        # Get image dimensions
        h, w = frames[idxs[0]]["rgb"].shape[:2]

        # Calculate intrinsics
        fx, fy, cx, cy = get_intrinsics(focal_length, h, w)
        intrinsics = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)

        # Convert Kubric depth (Euclidean distance) to z-depth
        for frame_id in idxs:
            depth = frames[frame_id]["depth"].astype(np.float32)
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            depthmap = depth / np.sqrt(1 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
            frames[frame_id]["depthmap"] = depthmap.astype(np.float32)

        # Build views
        views = []
        for vi, frame_i in enumerate(idxs):
            frame = frames[frame_i]

            t, rot = get_transform(cam_positions, cam_quaternions, frame_i)
            matrix_world = np.eye(4, dtype=np.float32)
            matrix_world[:3, :3] = rot
            matrix_world[:3, 3] = t
            camera_pose = matrix_world @ self.M_Blender2Opencv

            view = dict(
                img=Image.fromarray(frame["rgb"]),
                track=np.zeros((h, w, 3), dtype=np.float32),
                track_valid_mask=np.zeros((h, w), dtype=bool),
                depthmap=frame["depthmap"].astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.copy().astype(np.float32),
                obj_coords=frame["obj_coords"].astype(np.float32),
                segm=frame["segm"],
                dataset='KubricMultiCameraProcessed_81Frames',
                label=f'{scene_name}_cam{camera_idx}',
                instance=f'{scene_name}_cam{camera_idx}_{frame_i:05d}',
                is_metric=self.is_metric,
                is_video=self.video,
                quantile=np.array(1.0, dtype=np.float32),
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
                track_query_idx=query_view_idx,
            )
            views.append(view)

        # Compute dense 3D tracks: static background
        K_q = views[query_view_idx]["camera_intrinsics"].astype(np.float32)
        depth_q = views[query_view_idx]["depthmap"].astype(np.float32)
        pose_q = views[query_view_idx]["camera_pose"].astype(np.float32)

        fx_q = float(K_q[0, 0])
        fy_q = float(K_q[1, 1])
        cx_q = float(K_q[0, 2])
        cy_q = float(K_q[1, 2])

        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        valid_q = np.isfinite(depth_q) & (depth_q > 0)

        Xq = (uu - cx_q) / fx_q * depth_q
        Yq = (vv - cy_q) / fy_q * depth_q
        Zq = depth_q
        pts_cam_q = np.stack([Xq, Yq, Zq], axis=-1).reshape(-1, 3).astype(np.float32)

        R_q = pose_q[:3, :3].astype(np.float32)
        t_q = pose_q[:3, 3].astype(np.float32)
        pts_world_static = np.zeros_like(pts_cam_q, dtype=np.float32)
        valid_q_flat = valid_q.reshape(-1)
        if np.any(valid_q_flat):
            pts_world_static[valid_q_flat] = (pts_cam_q[valid_q_flat] @ R_q.T + t_q[None, :]).astype(np.float32)

        Rt_by_frame = {}
        for vi, frame_i in enumerate(idxs):
            pose_i = views[vi]["camera_pose"].astype(np.float32)
            R_i = pose_i[:3, :3].astype(np.float32)
            t_i = pose_i[:3, 3].astype(np.float32)
            Rt_by_frame[int(frame_i)] = (R_i, t_i)

        for vi, frame_i in enumerate(idxs):
            R_i, t_i = Rt_by_frame[int(frame_i)]
            track_flat = np.zeros_like(pts_cam_q, dtype=np.float32)
            if np.any(valid_q_flat):
                track_flat[valid_q_flat] = ((pts_world_static[valid_q_flat] - t_i[None, :]) @ R_i).astype(np.float32)
            views[vi]["track"] = track_flat.reshape(h, w, 3).astype(np.float32)
            views[vi]["track_valid_mask"] = valid_q

        # Overlay dynamic objects using bboxes
        query_obj_coords = frames[query_frame_id]["obj_coords"]
        query_segm = frames[query_frame_id]["segm"]
        n_instances = bboxes.shape[0]

        for seg_id in range(1, n_instances + 1):
            mask = (query_segm == seg_id).squeeze()
            if not np.any(mask):
                continue

            pix_idx = np.flatnonzero(mask.reshape(-1))
            pix_idx = pix_idx[valid_q_flat[pix_idx]]
            if pix_idx.size == 0:
                continue

            obj_local = query_obj_coords.reshape(-1, 3)[pix_idx]

            for vi, frame_i in enumerate(idxs):
                bbox3d = bboxes[seg_id - 1, frame_i]  # [8, 3]
                p0 = bbox3d[0].astype(np.float32)
                ax = (bbox3d[4] - bbox3d[0]).astype(np.float32)
                ay = (bbox3d[2] - bbox3d[0]).astype(np.float32)
                az = (bbox3d[1] - bbox3d[0]).astype(np.float32)

                pts_world = (
                    p0[None, :]
                    + obj_local[:, 0:1] * ax[None, :]
                    + obj_local[:, 1:2] * ay[None, :]
                    + obj_local[:, 2:3] * az[None, :]
                ).astype(np.float32)

                R_i, t_i = Rt_by_frame[int(frame_i)]
                pts_cam = ((pts_world - t_i[None, :]) @ R_i).astype(np.float32)

                track_flat = views[vi]["track"].reshape(-1, 3)
                track_flat[pix_idx] = pts_cam
                views[vi]["track"] = track_flat.reshape(h, w, 3).astype(np.float32)

        # Crop/resize
        for vi, view in enumerate(views):
            rgb_image, depthmap, intrinsics_out, track, track_valid_mask, obj_coords, segm = self._crop_resize_if_necessary(
                view["img"], view["depthmap"], view["camera_intrinsics"], target_resolution,
                rng=rng, info=idx,
                track=view["track"].astype(np.float32), track_valid_mask=view["track_valid_mask"],
                obj_coords=view.get("obj_coords", None),
                segm=view.get("segm", None),
                allow_random_transpose=False
            )
            view["img"] = rgb_image
            view["depthmap"] = depthmap.astype(np.float32)
            view["camera_intrinsics"] = intrinsics_out.astype(np.float32)
            view["track"] = track.astype(np.float32)
            view["track_valid_mask"] = track_valid_mask
            if obj_coords is not None:
                view["obj_coords"] = obj_coords.astype(np.float32)
            if segm is not None:
                view["segm"] = segm.astype(np.int32)

        return views


__all__ = ["KubricMultiCameraProcessed_81Frames"]
