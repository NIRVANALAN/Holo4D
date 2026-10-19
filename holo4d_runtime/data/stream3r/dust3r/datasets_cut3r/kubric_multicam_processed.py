"""
Kubric Multi-Camera Preprocessed dataset for multi-view video with dynamic objects.
Loads data from preprocessed multi-camera format (like dpm-next kubric.py).

Data structure:
ROOT/
    train/  (or validation/)
        00001/
            camera_0/
                rgb_00.jpg ... rgb_48.jpg
                depth_00.png ... depth_48.png
                coords_00.png ... coords_48.png
                segm_00.png ... segm_48.png
            camera_1/
            camera_2/
            camera_3/
        00002/
        ...
    seq_data.train.npz  (metadata)
    seq_data.validation.npz
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import cv2
import torch

# Use BaseStereoViewDataset like KubricWithTracks for consistency
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_stereo_view_dataset import BaseStereoViewDataset
from holo4d_runtime.data.stream3r.dust3r.utils.image import imread_cv2


def get_intrinsics(focal_length, H, W):
    """Calculate intrinsics from focal length (same as dpm-next loader)."""
    fx = focal_length * W
    fy = fx * (H / W)
    cx = 0.5 * W
    cy = 0.5 * H
    return fx, fy, cx, cy


def get_transform(cam_positions, cam_quaternions, frame_idx):
    """Get camera transform for a frame (same as dpm-next loader)."""
    t = cam_positions[frame_idx]
    quat = cam_quaternions[frame_idx]
    # Kubric quaternions are in wxyz format, scipy expects xyzw
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    return t, rot


def _coerce_resolution_with_num_views(resolution, num_views: int):
    """
    Allow Stream3R YAMLs to specify:
      resolution: [[W,H], ...] + num_views: N
    and convert to BaseStereoViewDataset style:
      resolution: [([W,H], N), ...]
    """
    if resolution is None:
        return None
    if not isinstance(resolution, list):
        return resolution
    coerced = []
    for res in resolution:
        if isinstance(res, (list, tuple)) and len(res) == 2 and all(isinstance(x, int) for x in res):
            w, h = int(res[0]), int(res[1])
            coerced.append(([w, h], int(num_views)))
        else:
            coerced.append(res)
    return coerced


class KubricMultiCameraProcessed(BaseStereoViewDataset):
    """
    Kubric Multi-Camera Preprocessed dataset.

    Loads from preprocessed data format compatible with dpm-next kubric.py:
    - Images preprocessed and normalized
    - Metadata in seq_data.{split}.npz
    - Supports multi-camera (camera_idx field in metadata)
    - Supports random query frame selection

    Compatible with existing dataloading pipeline (not the vis_dataset_multicam_pretty.py API).
    """

    def __init__(
        self,
        ROOT,
        split="train",
        camera_id=None, # for random
        num_cameras=4,
        max_interval=1,
        query_frame=0,
        reverse_prob=0.0,
        num_views=None,
        img_ext="jpg",
        *args,
        **kwargs,
    ):
        """
        Args:
            ROOT: Root directory containing split folders and seq_data files
            split: "train" or "validation"
            camera_id: Which camera trajectory to use (0-3), or None for random
            num_cameras: Total number of cameras per scene
            max_interval: Max frame interval for sampling
            query_frame: Query frame index (0 = first frame, -1 = random)
            reverse_prob: Probability of reversing temporal order (0.0 disables; 0.5 matches Kubric_Tracks behavior)
            num_views: Optional number of views/frames to return (used to coerce `resolution` entries like [W,H])
            img_ext: Image file extension ("jpg" or "png")
            *args, **kwargs: Arguments passed to BaseStereoViewDataset
        """
        if num_views is not None and "resolution" in kwargs:
            kwargs["resolution"] = _coerce_resolution_with_num_views(kwargs["resolution"], int(num_views))

        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.split = split
        self.camera_id = camera_id
        self.num_cameras = num_cameras
        self.video = True
        self.is_metric = True
        self.max_interval = int(max_interval)
        self.query_frame = int(query_frame)
        self.reverse_prob = float(reverse_prob)
        self.img_ext = img_ext

        # Stream3R samplers may consult `dataset.num_views`; keep it consistent with configs.
        if num_views is not None:
            self.num_views = int(num_views)

        # Blender to OpenCV coordinate conversion
        self.M_Blender2Opencv = np.array(
            [[1,  0,  0,  0],
             [0, -1,  0,  0],
             [0,  0, -1,  0],
             [0,  0,  0,  1]], dtype=np.float32
        )

        self._load_seq_data()

    def set_camera_id(self, camera_id):
        """Change the camera trajectory to use."""
        if camera_id is not None:
            assert 0 <= camera_id < self.num_cameras, f"camera_id must be in [0, {self.num_cameras})"
        self.camera_id = camera_id
        self._filter_scenes()

    def set_query_frame(self, query_frame):
        """Change the query frame index."""
        self.query_frame = int(query_frame)

    def set_reverse_prob(self, reverse_prob):
        """Change probability of reversing temporal order."""
        self.reverse_prob = float(reverse_prob)

    def _load_seq_data(self):
        """Load preprocessed metadata from seq_data.{split}.npz."""
        seq_data_path = osp.join(self.ROOT, f"seq_data.{self.split}.npz")
        if not osp.exists(seq_data_path):
            raise FileNotFoundError(f"Metadata file not found: {seq_data_path}")

        seq_data = np.load(seq_data_path)

        # Check if this is multi-camera data (has camera_idx field)
        if "camera_idx" not in seq_data:
            raise ValueError(
                f"seq_data does not contain 'camera_idx' field. "
                f"This loader requires preprocessed multi-camera data. "
                f"Available keys: {list(seq_data.keys())}"
            )

        # Store metadata
        self.scene_idxs = seq_data["scene_idx"]  # [N]
        self.camera_idxs = seq_data["camera_idx"]  # [N]
        self.min_max_depth = seq_data["min_max_depth"]  # [N, 2]
        self.cam_positions = seq_data["cam_positions"]  # [N, T, 3]
        self.cam_quaternions = seq_data["cam_quaternions"]  # [N, T, 4]
        self.focal_length = seq_data["focal_length"]  # [N]
        self.bboxes = seq_data["bboxes"]  # [total_instances, T, 8, 3]
        self.bboxes_cumsum = seq_data["bboxes_cumsum"]  # [N+1]

        self._n_frames = self.cam_positions.shape[1]
        self.data_dir = osp.join(self.ROOT, self.split)

        self._filter_scenes()

        print(f"[KubricMultiCameraProcessed] Loaded {len(self.valid_indices)} camera views "
              f"from {len(np.unique(self.scene_idxs))} scenes, {self._n_frames} frames each")

    def _filter_scenes(self):
        """Filter scenes by camera_id if specified."""
        if self.camera_id is None:
            # Use all cameras
            self.valid_indices = np.arange(len(self.scene_idxs))
        else:
            # Filter by camera_id
            self.valid_indices = np.where(self.camera_idxs == self.camera_id)[0]

        if len(self.valid_indices) == 0:
            raise ValueError(f"No scenes found for camera_id={self.camera_id}")

        # Sort valid_indices by scene_idx for consistent ordering
        scene_order = np.argsort(self.scene_idxs[self.valid_indices])
        self.valid_indices = self.valid_indices[scene_order]

    def __len__(self):
        return len(self.valid_indices)

    def get_image_num(self):
        """Return total number of images."""
        return len(self.valid_indices) * self._n_frames

    def _get_bboxes(self, meta_idx):
        """Get bboxes for a metadata index."""
        start_idx = self.bboxes_cumsum[meta_idx]
        end_idx = self.bboxes_cumsum[meta_idx + 1]
        return self.bboxes[start_idx:end_idx]

    def _get_views(self, idx, resolution, rng):
        """
        Load views for a given index.

        Args:
            idx: Index into self.valid_indices
            resolution: Tuple of ((width, height), num_views)
            rng: Random number generator

        Returns:
            List of view dictionaries
        """
        resolution, num_views = resolution

        # Get actual metadata index
        meta_idx = self.valid_indices[idx]

        scene_idx = self.scene_idxs[meta_idx]
        camera_idx = self.camera_idxs[meta_idx]
        min_depth, max_depth = self.min_max_depth[meta_idx]
        cam_positions = self.cam_positions[meta_idx]  # [T, 3]
        cam_quaternions = self.cam_quaternions[meta_idx]  # [T, 4]
        focal_length = self.focal_length[meta_idx]
        bboxes = self._get_bboxes(meta_idx)  # [n_instances, T, 8, 3]

        # Build paths
        scene_name = f"{scene_idx:05d}"
        camera_dir = osp.join(self.data_dir, scene_name, f"camera_{camera_idx}")

        # Frame sampling
        total_frames = self._n_frames
        num_views = min(num_views, total_frames)
        if num_views <= 0:
            return []

        # Sequential sampling by default (best for temporal browsing)
        if self.max_interval <= 1:
            idxs = list(range(num_views))
        else:
            # Random sampling with interval
            all_ids = list(range(total_frames))
            first = all_ids[0]
            rest = all_ids[1:]
            if rng.random() < 0.8 and len(rest) >= num_views - 1:
                sampled = rng.choice(rest, size=num_views - 1, replace=False)
                sampled_sorted = sorted(sampled)
            else:
                sampled_sorted = rest[:num_views - 1]
            idxs = [first] + list(sampled_sorted)

        # Optional temporal reversal augmentation.
        # This mirrors the "is_reverse" branch in Kubric_Tracks, but since we construct tracks
        # from per-frame metadata (depth/coords/bboxes), reversing only needs to flip order here.
        if self.reverse_prob > 0.0 and rng.random() < self.reverse_prob:
            idxs = list(reversed(idxs))

        # Query frame selection
        if self.query_frame < 0:
            # Random query frame
            query_view_idx = rng.integers(0, len(idxs))
        else:
            query_view_idx = max(0, min(int(self.query_frame), len(idxs) - 1))

        query_frame_id = int(idxs[query_view_idx])

        # Load frames
        uint16_max = np.iinfo(np.uint16).max
        frames = {}

        # import pdb; pdb.set_trace()

        for frame_id in idxs:
            # Load RGB
            rgb_path = osp.join(camera_dir, f"rgb_{frame_id:02d}.{self.img_ext}")
            rgb_image = imread_cv2(rgb_path)
            if rgb_image is None:
                raise FileNotFoundError(f"Could not load RGB: {rgb_path}")

            # Load depth (uint16 PNG, normalized to [0, 1] range)
            depth_path = osp.join(camera_dir, f"depth_{frame_id:02d}.png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Could not load depth: {depth_path}")

            # Load object coordinates (uint16 PNG, normalized to [0, 1] range)
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

            # Rescale depth and coords back to original range
            depth = depth.astype(np.float32) / uint16_max * (max_depth - min_depth) + min_depth
            obj_coords = obj_coords.astype(np.float32) / uint16_max
            segm = segm.astype(np.int32)

            frames[frame_id] = {
                "rgb": rgb_image,
                "depth": depth,
                "obj_coords": obj_coords,
                "segm": segm
            }

        # Get image dimensions from first frame
        h, w = frames[idxs[0]]["rgb"].shape[:2]

        # Calculate intrinsics
        fx, fy, cx, cy = get_intrinsics(focal_length, h, w)
        intrinsics = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)

        # Convert Kubric depth (Euclidean distance) to z-depth for each frame
        for frame_id in idxs:
            depth = frames[frame_id]["depth"].astype(np.float32)
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            depthmap = depth / np.sqrt(1 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
            frames[frame_id]["depthmap"] = depthmap.astype(np.float32)

        # Build views with camera poses
        views = []
        for vi, frame_i in enumerate(idxs):
            frame = frames[frame_i]

            # Get camera pose
            t, rot = get_transform(cam_positions, cam_quaternions, frame_i)

            # Build 4x4 pose matrix (cam to world)
            matrix_world = np.eye(4, dtype=np.float32)
            matrix_world[:3, :3] = rot
            matrix_world[:3, 3] = t

            # Convert from Blender to OpenCV
            camera_pose = matrix_world @ self.M_Blender2Opencv

            view = dict(
                img=Image.fromarray(frame["rgb"]),
                track=np.zeros((h, w, 3), dtype=np.float32),  # Will be filled later
                track_valid_mask=np.zeros((h, w), dtype=bool),  # Will be filled later
                depthmap=frame["depthmap"].astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.copy().astype(np.float32),
                obj_coords=frame["obj_coords"].astype(np.float32),  # Object coordinates for correspondence
                segm=frame["segm"],  # Segmentation mask
                dataset='KubricMultiCameraProcessed',
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

        # Compute dense 3D tracks using object coordinates and bboxes
        # (Similar to kubric_multicam.py logic)

        # Query frame backprojection for static background
        K_q = views[query_view_idx]["camera_intrinsics"].astype(np.float32)
        depth_q = views[query_view_idx]["depthmap"].astype(np.float32)
        pose_q = views[query_view_idx]["camera_pose"].astype(np.float32)

        fx_q = float(K_q[0, 0])
        fy_q = float(K_q[1, 1])
        cx_q = float(K_q[0, 2])
        cy_q = float(K_q[1, 2])

        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        valid_q = np.isfinite(depth_q) & (depth_q > 0)

        # Unproject query frame to world coords (static background assumption)
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

        # Compute per-frame camera transforms
        Rt_by_frame = {}
        for vi, frame_i in enumerate(idxs):
            pose_i = views[vi]["camera_pose"].astype(np.float32)
            R_i = pose_i[:3, :3].astype(np.float32)
            t_i = pose_i[:3, 3].astype(np.float32)
            Rt_by_frame[int(frame_i)] = (R_i, t_i)

        # Initialize tracks with static background
        # Note: pts3d and valid_mask are computed by the base class from depthmap and intrinsics
        for vi, frame_i in enumerate(idxs):
            R_i, t_i = Rt_by_frame[int(frame_i)]
            track_flat = np.zeros_like(pts_cam_q, dtype=np.float32)
            if np.any(valid_q_flat):
                track_flat[valid_q_flat] = ((pts_world_static[valid_q_flat] - t_i[None, :]) @ R_i).astype(np.float32)
            views[vi]["track"] = track_flat.reshape(h, w, 3).astype(np.float32)
            views[vi]["track_valid_mask"] = valid_q

        # Overlay dynamic objects using object coordinates and 3D bboxes
        query_obj_coords = frames[query_frame_id]["obj_coords"]
        query_segm = frames[query_frame_id]["segm"]

        n_instances = bboxes.shape[0]

        for seg_id in range(1, n_instances + 1):  # Skip background (seg_id=0)
            mask = (query_segm == seg_id).squeeze()
            if not np.any(mask):
                continue

            # Get object coordinates for this instance
            obj_coords_inst = query_obj_coords[mask]  # [N, 3] in [0, 1]
            pix_idx = np.flatnonzero(mask.reshape(-1))
            pix_idx = pix_idx[valid_q_flat[pix_idx]]

            if pix_idx.size == 0:
                continue

            obj_local = query_obj_coords.reshape(-1, 3)[pix_idx]  # [N, 3]

            # For each timestep, compute 3D positions using bbox
            for vi, frame_i in enumerate(idxs):
                bbox3d = bboxes[seg_id - 1, frame_i]  # [8, 3]

                # Compute world positions using bbox corners
                # Same convention as dpm-next: corners 0, 4, 2, 1 define axes
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

                # Transform to camera coords
                R_i, t_i = Rt_by_frame[int(frame_i)]
                pts_cam = ((pts_world - t_i[None, :]) @ R_i).astype(np.float32)

                # Update track
                track_flat = views[vi]["track"].reshape(-1, 3)
                track_flat[pix_idx] = pts_cam
                views[vi]["track"] = track_flat.reshape(h, w, 3).astype(np.float32)

        # Apply crop/resize to all views consistently
        for vi, view in enumerate(views):
            rgb_image, depthmap, intrinsics, track, track_valid_mask, obj_coords, segm = self._crop_resize_if_necessary(
                view["img"], view["depthmap"], view["camera_intrinsics"], resolution,
                rng=rng, info=idx,
                track=view["track"].astype(np.float32), track_valid_mask=view["track_valid_mask"],
                obj_coords=view.get("obj_coords", None),
                segm=view.get("segm", None),
                allow_random_transpose=False
            )
            view["img"] = rgb_image
            view["depthmap"] = depthmap.astype(np.float32)
            view["camera_intrinsics"] = intrinsics.astype(np.float32)
            view["track"] = track.astype(np.float32)
            view["track_valid_mask"] = track_valid_mask
            if obj_coords is not None:
                view["obj_coords"] = obj_coords.astype(np.float32)
            if segm is not None:
                view["segm"] = segm.astype(np.int32)

        # Note: pts3d and valid_mask will be computed by base class from depthmap + intrinsics
        return views
