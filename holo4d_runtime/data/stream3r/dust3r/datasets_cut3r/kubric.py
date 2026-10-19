"""
Kubric dataset for multi-view video with dynamic objects.
Adapted from dpm-next/dpm/datasets/kubric.py to work with BaseMultiViewDataset.
"""
import os
import os.path as osp
import pickle
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from holo4d_runtime.data.stream3r.dust3r.utils.image import imread_cv2
import holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.utils.cropping as cropping


from pdb import set_trace as st

def get_intrinsics(focal_length, H, W):
    """Get camera intrinsics from focal length and image dimensions."""
    fx = focal_length * W
    fy = fx * (H / W)
    cx = 0.5 * W
    cy = 0.5 * H
    return fx, fy, cx, cy


def get_transform(cam_positions, cam_quaternions, frame_idx):
    """Get camera rotation and translation for a given frame."""
    t = cam_positions[frame_idx]
    quat = cam_quaternions[frame_idx]
    # Convert quaternion to rotation matrix
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    return t, rot


def get_3d_tracks(cam_positions, cam_quaternions, obj_coords, segmentation,
                  bboxes, frame_id, time_id):
    """
    Compute 3D tracks from object coordinates and segmentation.
    Adapted from dpm-next/dpm/datasets/kubric.py:70-121

    Args:
        cam_positions: Camera positions [T, 3]
        cam_quaternions: Camera quaternions [T, 4]
        obj_coords: Object coordinates [H, W, 3] in [0, 1]
        segmentation: Segmentation mask [H, W]
        bboxes: Bounding boxes [N_instances, T, 8, 3]
        frame_id: Source frame index
        time_id: Target frame index

    Returns:
        tracks: [2, N_pts, 3] - 3D positions at [frame_id, time_id]
    """
    dtype = obj_coords.dtype
    num_instances = bboxes.shape[0]

    tracks_list = []
    # Skip background id=0
    for seg_id in range(1, num_instances + 1):
        mask = (segmentation == seg_id).squeeze()

        if not mask.any():
            continue

        t, rot = get_transform(cam_positions, cam_quaternions, frame_id)
        rot_inv = rot.T.astype(dtype)
        t_inv = (-rot_inv @ t).astype(dtype)

        # Compute tracks
        obj_coords_inst = obj_coords[mask]  # [N_pts, 3]
        bbox = bboxes[seg_id - 1, [frame_id, time_id], ...]  # [2, 8, 3]
        XYZ = bbox[:, [4, 2, 1]] - bbox[:, [0]]  # [2, 3, 3]

        # Transform object coords to world coords
        pc_W = (
            bbox[:, 0, None] +
            (obj_coords_inst[None, ..., None] * XYZ[:, None, :]).sum(axis=-2)
        )  # [2, N_pts, 3]

        # Transform to camera coords
        pc_C = np.einsum("ji,sni->snj", rot_inv, pc_W) + t_inv[None, None, :]  # [2, N_pts, 3]
        tracks_list.append(pc_C)

    if len(tracks_list) > 0:
        tracks = np.concatenate(tracks_list, axis=1)
    else:
        # No foreground objects - return empty tracks
        tracks = np.zeros((2, 0, 3), dtype=dtype)

    return tracks


def get_uv_grid_half(h, w, dtype=np.float32):
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    grid = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], axis=-1)
    return grid.astype(dtype)


def inside_image(ji, image_size):
    h, w = image_size
    return (
        (ji[:, 0] >= 0) & (ji[:, 0] < w) &
        (ji[:, 1] >= 0) & (ji[:, 1] < h)
    )


def project_pts3d_blender(K, pts3d, dtype):
    T_ref = np.diag([1.0, -1.0, -1.0]).astype(dtype)
    X_ = (T_ref @ pts3d.T).T
    depth = np.linalg.norm(X_, axis=-1, keepdims=True)
    X__ = X_ / np.clip(depth, 1e-8, None)
    x = X__ / np.clip(X__[:, [-1]], 1e-8, None)
    uv = (K @ x.T).T
    return uv[:, :2], depth


def unproject_depth_blender(K, uv, depth, dtype):
    T_ref = np.diag([1.0, -1.0, -1.0]).astype(dtype)
    K_inv = np.linalg.inv(K).astype(dtype)
    if uv.shape[1] == 3:
        uv_h = uv.astype(dtype)
    else:
        uv_h = np.concatenate([uv, np.ones((uv.shape[0], 1), dtype=dtype)], axis=-1)
    xyz = (T_ref @ (K_inv @ uv_h.T)).T
    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz = xyz / np.clip(norms, 1e-8, None)
    return xyz * depth


def construct_pointmap(depth, bg_mask, tracks_3d_C, K, uv_grid):
    """
    DPM-next style construct_pointmap using Euclidean depth.
    Returns pointmap in Blender camera coordinates.
    """
    dtype = tracks_3d_C.dtype
    image_size = depth.shape[:2]

    # Project source tracks to pixel grid
    uv_orig, z_orig = project_pts3d_blender(K, tracks_3d_C[0], dtype)
    uv = np.round(uv_orig + 0.5) - 0.5
    ji = np.round(uv - 0.5).astype(np.int32)

    mask = inside_image(ji, image_size)
    tracks_3d_C = tracks_3d_C[:, mask, :]
    z_orig = z_orig[mask, :]
    uv = uv[mask, :]
    ji = ji[mask, :]

    # Sample depth at projected pixels
    z = depth[ji[:, 1], ji[:, 0]][..., None]
    z_eps = 1e-1
    mask_mismatch = (np.abs(z - z_orig) < z_eps)[:, 0]
    tracks_3d_C = tracks_3d_C[:, mask_mismatch, :]
    uv = uv[mask_mismatch, :]
    ji = ji[mask_mismatch, :]
    z = z[mask_mismatch, :]

    # Unproject depth to Blender camera coords
    pc_C = unproject_depth_blender(K, uv, z, dtype)
    correction = pc_C - tracks_3d_C[0]
    tgt_pts3d = tracks_3d_C[-1] + correction

    # Background points (static)
    uv_static = uv_grid[bg_mask]
    z_static = depth[bg_mask][:, None]
    pc_C_static = unproject_depth_blender(K, uv_static, z_static, dtype)
    ji_static = np.round(uv_static - 0.5).astype(np.int32)

    pointmap = np.zeros((*image_size, 3), dtype=dtype)
    valid_mask = np.zeros(image_size, dtype=bool)
    fg_mask = np.zeros_like(valid_mask)
    pointmap[ji[:, 1], ji[:, 0]] = tgt_pts3d
    valid_mask[ji[:, 1], ji[:, 0]] = True
    fg_mask[ji[:, 1], ji[:, 0]] = True
    pointmap[ji_static[:, 1], ji_static[:, 0]] = pc_C_static
    valid_mask[ji_static[:, 1], ji_static[:, 0]] = True

    return pointmap, fg_mask, valid_mask


def sparse_to_dense_tracks(obj_coords, segmentation, bboxes, cam_positions,
                           cam_quaternions, query_frame_idx, target_frame_idx,
                           depthmap_euclid, intrinsics):
    """
    Convert sparse 3D tracks to dense [H, W, 3] track map.

    Note: get_3d_tracks already returns coordinates in the QUERY frame's
    Blender camera coordinates, so no extra world->camera transform is needed here.

    Args:
        obj_coords: Object coordinates [H, W, 3] from query frame in [0, 1]
        segmentation: Segmentation mask [H, W] from query frame
        bboxes: Bounding boxes [N_instances, T, 8, 3]
        cam_positions: Camera positions [T, 3]
        cam_quaternions: Camera quaternions [T, 4]
        query_frame_idx: Query frame index (where pixels are sampled)
        target_frame_idx: Target frame index (where we compute 3D positions)
        depthmap_euclid: Query frame depthmap in Euclidean depth (H, W)
        intrinsics: Query frame intrinsics (3, 3)

    Returns:
        dense_tracks: [H, W, 3] - Dense track map in TARGET frame's OpenCV camera coordinates
        valid_mask: [H, W] - Valid track mask
    """
    dtype = obj_coords.dtype
    H, W = segmentation.shape[:2]
    num_instances = bboxes.shape[0]

    # Initialize dense track map with zeros
    dense_tracks = np.zeros((H, W, 3), dtype=dtype)
    valid_mask = np.zeros((H, W), dtype=bool)

    # Blender -> OpenCV camera conversion
    rot_x = Rotation.from_euler('x', 180, degrees=True).as_matrix().astype(dtype)

    intrinsics_track = intrinsics.copy().astype(dtype)

    tracks_3d_C = get_3d_tracks(
        cam_positions, cam_quaternions,
        obj_coords, segmentation, bboxes,
        query_frame_idx, target_frame_idx
    )

    if tracks_3d_C.shape[1] == 0:
        return dense_tracks, valid_mask

    bg_mask = (segmentation == 0).squeeze()
    uv_grid = get_uv_grid_half(H, W, dtype=dtype)
    pointmap, fg_mask, valid = construct_pointmap(
        depthmap_euclid, bg_mask, tracks_3d_C, intrinsics_track, uv_grid
    )

    # Convert Blender camera coords to OpenCV camera coords (query frame)
    pointmap_cv = (rot_x @ pointmap.reshape(-1, 3).T).T.reshape(pointmap.shape)
    dense_tracks = pointmap_cv
    # Match dpm-next visualization: use foreground mask for track validity.
    valid_mask = fg_mask

    return dense_tracks, valid_mask


class Kubric(BaseMultiViewDataset):
    """
    Kubric synthetic dataset with dynamic objects.

    Data structure:
    ROOT/
        train/ or val/
            scene_00000/
                rgb_{00..24}.{img_ext}
                depth_{00..24}.png
                segm_{00..24}.png (optional, for object segmentation)
            scene_00001/
                ...
        seq_data.train.npz or seq_data.val.npz
            - cam_positions: [N, T, 3]
            - cam_quaternions: [N, T, 4]
            - focal_length: [N]
            - min_max_depth: [N, 2]
            - scene_idx or inferred scene indices
    """

    def __init__(self, ROOT, img_ext="png", max_interval=10, *args, **kwargs):
        self.ROOT = ROOT
        self.img_ext = img_ext
        self.video = True
        self.is_metric = True
        self.max_interval = max_interval
        super().__init__(*args, **kwargs)
        self._load_data()

    def _crop_resize_with_target(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """Match BaseMultiViewDataset crop/resize while exposing target_resolution."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)

        image, depthmap, intrinsics = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            if self.seq_aug_crop:
                target_resolution += self.delta_target_resolution
            else:
                target_resolution += rng.integers(0, self.aug_crop)

        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2, target_resolution

    def _crop_resize_track(self, track, track_valid_mask, intrinsics, resolution, target_resolution, info=None):
        """Apply the same crop/resize pipeline to dense tracks."""
        H, W = track.shape[:2]
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y

        track = track[t:b, l:r]
        track_valid_mask = track_valid_mask[t:b, l:r]

        intrinsics_crop = intrinsics.copy()
        intrinsics_crop[0, 2] -= l
        intrinsics_crop[1, 2] -= t

        input_resolution = np.array([track.shape[1], track.shape[0]])
        target_resolution = np.array(target_resolution)
        scale_final = max(target_resolution / input_resolution) + 1e-8
        output_resolution = np.floor(input_resolution * scale_final).astype(int)

        track = cv2.resize(track, tuple(output_resolution), interpolation=cv2.INTER_NEAREST)
        track_valid_mask = cv2.resize(
            track_valid_mask.astype(np.uint8),
            tuple(output_resolution),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        intrinsics_rescaled = cropping.camera_matrix_of_crop(
            intrinsics_crop, input_resolution, output_resolution, scaling=scale_final
        )
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics_rescaled, output_resolution, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics_rescaled, intrinsics2, resolution
        )
        l, t, r, b = crop_bbox
        track = track[t:b, l:r]
        track_valid_mask = track_valid_mask[t:b, l:r]

        return track, track_valid_mask

    def __getitem__(self, idx):
        """Override to add track field after base class computes pts3d."""
        views = super().__getitem__(idx)

        # Add track field for visualization
        #
        # Strategy:
        # 1. If _dense_track is available (computed from obj_coords/segm), use it for foreground
        # 2. For background (or if no tracking data), use pts3d[0] as static tracks
        #
        # Note: Both tracks and pts3d should be in the same coordinate system
        # since sparse_to_dense_tracks uses the same camera transform as _get_views

        # Reference pts3d for pseudo-static tracking (background)
        # pts3d is already in world coordinates (OpenCV) from the base class.
        ref_pts3d = views[0]['pts3d'].copy()  # [H, W, 3] in world coords
        ref_valid_mask = views[0]['valid_mask'].copy()
        ref_camera_pose = views[0]['camera_pose']
        if isinstance(ref_camera_pose, torch.Tensor):
            ref_camera_pose = ref_camera_pose.numpy()
        ref_R = ref_camera_pose[:3, :3]
        ref_t = ref_camera_pose[:3, 3]
        for i, view in enumerate(views):
            # Start with pseudo-static tracks (background) - just use frame 0's pts3d.
            # If dense tracks are available, we'll override the valid mask to foreground only.
            track = ref_pts3d.copy()
            track_valid_mask = ref_valid_mask.copy()

            # If dense tracking is available, use it for foreground objects
            if '_dense_track' in view and '_dense_track_valid' in view:
                dense_track = view['_dense_track']  # In camera 0 coordinates
                dense_track_valid_orig = view['_dense_track_valid']

                # Dense tracks are already cropped/resized in _get_views; keep a fallback just in case.
                target_h, target_w = view['pts3d'].shape[:2]
                orig_h, orig_w = dense_track.shape[:2]
                if (orig_h, orig_w) != (target_h, target_w):
                    dense_track_resized = cv2.resize(
                        dense_track, (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    dense_track_valid_resized = cv2.resize(
                        dense_track_valid_orig.astype(np.uint8), (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    dense_track_resized = dense_track
                    dense_track_valid_resized = dense_track_valid_orig

                # Dense tracks are in QUERY frame OpenCV camera coordinates.
                # Convert to world coordinates using the reference camera pose.
                dense_track_world = (
                    np.einsum("ij,hwj->hwi", ref_R, dense_track_resized) +
                    ref_t[None, None, :]
                )

                # Replace foreground (tracked) regions with real tracks in world coords.
                track[dense_track_valid_resized] = dense_track_world[dense_track_valid_resized]
                # Match dpm-next vis: only foreground tracked points are valid.
                track_valid_mask = dense_track_valid_resized.astype(bool)

                # Clean up temporary fields
                del view['_dense_track']
                del view['_dense_track_valid']

            view['track'] = track.astype(np.float32)
            view['track_valid_mask'] = track_valid_mask

        return views

    def _load_seq_data(self):
        """Load sequence metadata from preprocessed npz file."""
        seq_data_path = osp.join(self.ROOT, f"seq_data.{self.split}.npz")
        if not osp.exists(seq_data_path):
            raise FileNotFoundError(
                f"Sequence data file not found: {seq_data_path}\n"
                f"Please ensure you have the preprocessed Kubric dataset with seq_data files."
            )

        seq_data = np.load(seq_data_path)
        num_seqs = seq_data["min_max_depth"].shape[0]

        # Determine scene keys
        if "scene_idx" in seq_data:
            # New Kubric-G data structure
            seq_idxs = seq_data["scene_idx"]
            seq_keys = [f"{k:05n}" for k in seq_idxs]
        else:
            # Old Kubric-F data structure
            seq_idxs = range(num_seqs)
            seq_keys = [f"scene_{k:06n}" for k in seq_idxs]

        n_frames = seq_data["cam_positions"].shape[1]

        # Store as pytorch tensors to reduce memory usage in multi-process loading
        self._min_max_depth = torch.from_numpy(seq_data["min_max_depth"])  # [N, 2]
        self._cam_positions = torch.from_numpy(seq_data["cam_positions"])  # [N, T, 3]
        self._cam_quaternions = torch.from_numpy(seq_data["cam_quaternions"])  # [N, T, 4]
        self._focal_length = torch.from_numpy(seq_data["focal_length"])  # [N]
        self._n_frames = n_frames

        # Load bboxes for tracking (if available)
        if "bboxes" in seq_data and "bboxes_cumsum" in seq_data:
            self._bboxes = torch.from_numpy(seq_data["bboxes"])
            self._bboxes_cumsum = torch.from_numpy(seq_data["bboxes_cumsum"])
            self.has_tracking = True
        else:
            self.has_tracking = False

        return seq_keys, n_frames

    def _load_data(self):
        """Load dataset metadata and build scene/frame index."""
        # Load sequence metadata
        seq_keys, n_frames = self._load_seq_data()

        # Build scene and frame lists (fast, no file system checks)
        data_dir = osp.join(self.ROOT, self.split)

        scenes = []
        sceneids = []
        scene_frame_list = []
        start_frame_ids = []
        offset = 0

        # Determine cut-off for start frames
        cut_off = (
            self.num_views
            if not self.allow_repeat
            else max(self.num_views // 3, 3)
        )

        if n_frames < cut_off:
            raise ValueError(f"Dataset has only {n_frames} frames but requires at least {cut_off}")

        for scene_idx, scene_kea in enumerate(seq_keys):
            scene_dir = osp.join(data_dir, scene_key)

            # Build frame IDs for this scene (no verification, trust the data)
            frame_ids = list(range(offset, offset + n_frames))
            start_frame_ids_ = frame_ids[:n_frames - cut_off + 1]

            scenes.append(scene_dir)
            scene_frame_list.append(frame_ids)
            sceneids.extend([len(scenes) - 1] * n_frames)
            start_frame_ids.extend(start_frame_ids_)
            offset += n_frames

        self.scenes = scenes
        self.sceneids = sceneids
        self.scene_keys = seq_keys
        self.start_frame_ids = start_frame_ids
        self.scene_frame_list = scene_frame_list

        # NOTE: Caching disabled to avoid permission issues and for faster startup
        # If you want to enable caching, uncomment the code below and ensure write permissions
        # cache_path = osp.join(self.ROOT, f'kubric_loaddata_{self.split}_{self.num_views}.pkl')
        # with open(cache_path, 'wb') as f:
        #     pickle.dump({
        #         'scenes': scenes,
        #         'sceneids': sceneids,
        #         'scene_keys': seq_keys,
        #         'start_frame_ids': start_frame_ids,
        #         'scene_frame_list': scene_frame_list,
        #     }, f)

    def __len__(self):
        return len(self.start_frame_ids)

    def get_image_num(self):
        """Return total number of frames."""
        return sum(len(frames) for frames in self.scene_frame_list)

    def seq_id_from_key(self, seq_key):
        """Get sequence index from sequence key."""
        return self.scene_keys.index(seq_key)

    def data_idx_from_seq_frame(self, seq_key, frame_idx):
        """
        Get dataset index from sequence key and frame index.
        This matches the dpm-next VideoDataset interface.
        """
        seq_idx = self.seq_id_from_key(seq_key)
        # For Kubric, all scenes have the same number of frames
        # Calculate the data index based on scene and frame offset
        cut_off = (
            self.num_views
            if not self.allow_repeat
            else max(self.num_views // 3, 3)
        )
        margin = (self.num_views - 1) * 1  # Simplified margin calculation

        # Find the start index for this scene
        scene_start_idx = seq_idx * (self._n_frames - cut_off + 1)
        # Add the frame offset (accounting for margin)
        data_idx = scene_start_idx + frame_idx - margin

        # Clamp to valid range
        data_idx = max(0, min(data_idx, len(self) - 1))
        return data_idx

    def _get_views(self, idx, resolution, rng, num_views):
        """Load a sequence of views for the given index."""
        start_id = self.start_frame_ids[idx]
        scene_id = self.sceneids[start_id]
        all_frame_ids = self.scene_frame_list[scene_id]

        # Sample frame sequence
        pos, is_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_frame_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=1.0 if self.video else 0.0,  # Always video for Kubric
            fix_interval_prob=0.8,
            block_shuffle=None,  # No shuffling for video
        )

        frame_idxs = np.array(all_frame_ids)[pos]
        scene_dir = self.scenes[scene_id]

        # Get camera parameters for this scene
        focal_length = self._focal_length[scene_id].item()
        cam_positions = self._cam_positions[scene_id]
        cam_quaternions = self._cam_quaternions[scene_id]
        min_depth, max_depth = self._min_max_depth[scene_id].tolist()

        # Get bboxes for tracking (if available)
        if self.has_tracking:
            # Extract bboxes for this scene
            if scene_id > 0:
                start_idx = self._bboxes_cumsum[scene_id - 1].item()
            else:
                start_idx = 0

            if scene_id < len(self._bboxes_cumsum) - 1:
                end_idx = self._bboxes_cumsum[scene_id].item()
            else:
                end_idx = len(self._bboxes)

            bboxes = self._bboxes[start_idx:end_idx].numpy()  # [N_instances, T, 8, 3]
        else:
            bboxes = None

        # First pass: load all frame data
        frames_data = []
        for v, frame_idx in enumerate(frame_idxs):
            # Compute local frame index within the scene
            local_frame_idx = frame_idx - all_frame_ids[0]

            # Load RGB image
            rgb_path = osp.join(scene_dir, f"rgb_{local_frame_idx:02n}.{self.img_ext}")
            rgb_image = imread_cv2(rgb_path)

            # Load depth map
            depth_path = osp.join(scene_dir, f"depth_{local_frame_idx:02n}.png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            # Rescale depth from uint16 to metric depth
            uint16_max = np.iinfo(np.uint16).max
            depth = depth / uint16_max * (max_depth - min_depth) + min_depth
            depth = depth.astype(np.float32)
            depth_euclid = depth.copy()

            # Get intrinsics for depth conversion
            orig_H, orig_W = rgb_image.shape[:2]
            fx_temp, fy_temp, cx_temp, cy_temp = get_intrinsics(focal_length, orig_H, orig_W)

            # Convert from Euclidean depth to Z-depth
            # Kubric stores distance from camera center, but we need distance along Z-axis
            # (matching kubric_movif_track.py line 114-115)
            u_grid, v_grid = np.meshgrid(np.arange(orig_W), np.arange(orig_H))
            depth /= np.sqrt(1 + ((u_grid - cx_temp) / fx_temp) ** 2 + ((v_grid - cy_temp) / fy_temp) ** 2)
            depth_orig = depth.copy()

            # Load tracking data (coords and segm) if available
            if self.has_tracking:
                coords_path = osp.join(scene_dir, f"coords_{local_frame_idx:02n}.png")
                obj_coords = cv2.imread(coords_path, cv2.IMREAD_UNCHANGED)
                if obj_coords is not None:
                    # Keep channel order to match dpm-next raw Kubric handling.
                    if obj_coords.ndim == 3 and obj_coords.shape[2] >= 3:
                        obj_coords = obj_coords[..., :3]
                    obj_coords = obj_coords / uint16_max  # Normalize to [0, 1]
                    obj_coords = obj_coords.astype(np.float32)
                else:
                    obj_coords = None

                segm_path = osp.join(scene_dir, f"segm_{local_frame_idx:02n}.png")
                segm = cv2.imread(segm_path, cv2.IMREAD_UNCHANGED)
            else:
                obj_coords = None
                segm = None

            # Get camera pose
            t, rot = get_transform(cam_positions, cam_quaternions, local_frame_idx)

            # Apply coordinate system transform (Kubric uses Blender, we use OpenCV)
            # The correct transformation is: rot @ rot_x.T (not rot_x @ rot)
            rot_x = Rotation.from_euler('x', 180, degrees=True).as_matrix()
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = rot @ rot_x.T
            camera_pose[:3, 3] = t

            # Build intrinsics matrix (using values already computed above)
            intrinsics = np.array(
                [[fx_temp, 0, cx_temp],
                 [0, fy_temp, cy_temp],
                 [0, 0, 1]],
                dtype=np.float32
            )

            # Store original coords/segm before crop/resize for track computation
            # Note: We keep original resolution coords/segm for 3D track computation
            orig_coords = obj_coords
            orig_segm = segm

            # Crop and resize RGB and depth only
            # (coords/segm are used in original resolution for track computation)
            orig_intrinsics = intrinsics.copy()
            rgb_image, depth, intrinsics, target_resolution = self._crop_resize_with_target(
                rgb_image, depth, intrinsics, resolution, rng=rng,
                info=(scene_dir, f"frame_{local_frame_idx:02n}")
            )

            # Generate img mask and raymap mask
            # st()
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.8, 0.15, 0.05]
            )

            # Store frame data
            frames_data.append({
                'rgb_image': rgb_image,
                'depth': depth,
                'depth_euclid': depth_euclid,
                'camera_pose': camera_pose,
                'intrinsics': intrinsics,
                'orig_intrinsics': orig_intrinsics,
                'depth_orig': depth_orig,
                'target_resolution': target_resolution,
                'local_frame_idx': local_frame_idx,
                'img_mask': img_mask,
                'ray_mask': ray_mask,
                'obj_coords': orig_coords,  # Keep original for track computation
                'segm': orig_segm,
                'v': v
            })

        # Second pass: compute dense tracks from query frame (frame 0) to all other frames
        # For each frame i, compute where frame 0's pixels appear in frame i
        dense_tracks_dict = {}
        if self.has_tracking and bboxes is not None:
            # Convert torch tensors to numpy for track computation
            cam_positions_np = cam_positions.numpy() if isinstance(cam_positions, torch.Tensor) else cam_positions
            cam_quaternions_np = cam_quaternions.numpy() if isinstance(cam_quaternions, torch.Tensor) else cam_quaternions

            # Use frame 0 as the query frame (where we sample pixels to track)
            query_frame = frames_data[0]

            # For all frames (including frame 0), compute where query frame's pixels appear
            for i, frame_target in enumerate(frames_data):
                # Only compute if query frame has tracking data
                if query_frame['obj_coords'] is not None and query_frame['segm'] is not None:
                    try:
                        # Compute dense track map: [H, W, 3]
                        # This gives us where each pixel in frame 0 appears in frame i
                        dense_track, track_valid = sparse_to_dense_tracks(
                            query_frame['obj_coords'], query_frame['segm'],
                            bboxes, cam_positions_np, cam_quaternions_np,
                            query_frame['local_frame_idx'], frame_target['local_frame_idx'],
                            query_frame['depth_euclid'], query_frame['orig_intrinsics']
                        )
                        dense_tracks_dict[i] = (dense_track, track_valid)
                    except Exception as e:
                        print(f"Warning: Failed to compute dense tracks for frames 0->{i}: {e}")
                        dense_tracks_dict[i] = None
                else:
                    dense_tracks_dict[i] = None

        # Third pass: construct view dictionaries
        # Note: pts3d and valid_mask will be computed by BaseMultiViewDataset from depthmap
        views = []
        for i, frame_data in enumerate(frames_data):
            # Check if we have computed dense tracks for this frame
            has_dense_tracking = (i in dense_tracks_dict and dense_tracks_dict[i] is not None)

            view = dict(
                img=frame_data['rgb_image'],  # PIL Image
                depthmap=frame_data['depth'],
                # pts3d will be computed by base class from depthmap + camera_pose + intrinsics
                camera_pose=frame_data['camera_pose'],
                camera_intrinsics=frame_data['intrinsics'],
                dataset='Kubric',
                label=osp.basename(scene_dir),
                instance=f"{osp.basename(scene_dir)}_{frame_data['local_frame_idx']:02n}",
                is_metric=self.is_metric,
                is_video=is_video,
                quantile=np.array(1.0, dtype=np.float32),
                img_mask=frame_data['img_mask'],
                ray_mask=frame_data['ray_mask'],
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=(frame_data['v'] == 0),
            )

            # Store dense tracking data if available (will be used in __getitem__)
            if has_dense_tracking:
                dense_track, track_valid = dense_tracks_dict[i]
                dense_track, track_valid = self._crop_resize_track(
                    dense_track,
                    track_valid,
                    frame_data['orig_intrinsics'],
                    resolution,
                    frame_data['target_resolution'],
                    info=(scene_dir, f"frame_{frame_data['local_frame_idx']:02n}")
                )
                view['_dense_track'] = dense_track
                view['_dense_track_valid'] = track_valid

            views.append(view)

        assert len(views) == num_views
        return views


if __name__ == "__main__":
    # Simple test
    dataset = Kubric(
        ROOT="/path/to/kubric",
        split="train",
        num_views=5,
        resolution=(512, 384),
        allow_repeat=False,
        img_ext="png",
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Total frames: {dataset.get_image_num()}")

    # Test loading one sample
    views = dataset[0]
    print(f"Loaded {len(views)} views")
    for i, view in enumerate(views):
        print(f"View {i}: {view['label']}, shape={view['img'].size}")
