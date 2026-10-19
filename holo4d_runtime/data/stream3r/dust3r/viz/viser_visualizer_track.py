# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import datetime
import threading
from typing import Callable, Optional, Dict, Any
import numpy as np
from tqdm.auto import tqdm
import imageio.v3 as iio
from matplotlib import cm
import cv2
from scipy import ndimage
import torch

import viser
import viser.transforms as tf
from ..utils.device import to_numpy
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import inv
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import estimate_camera_poses
# from diffsynth.models.multiview_dust3r_module import MultiViewDUSt3RLitModule  # Not needed for dataset visualization

# ----------------- Helper Functions -----------------

def detect_sky_mask(img_rgb):
    """
    Detect sky pixels using HSV color space and morphological operations.
    Args:
        img_rgb: RGB image normalized to [-1, 1]
    Returns:
        Boolean mask (as int8) where True indicates non-sky pixels.
    """
    img = ((img_rgb + 1) * 127.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([105, 50, 140])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_light_blue = np.array([95, 5, 150])
    upper_light_blue = np.array([145, 100, 255])
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    lower_white = np.array([0, 0, 235])
    upper_white = np.array([180, 10, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = mask_blue | mask_light_blue | mask_white

    height = mask.shape[0]
    upper_third = int(height * 0.4)
    upper_region = hsv[:upper_third, :, :]
    mask[:upper_third, :] |= ((upper_region[:, :, 1] < 50) & (upper_region[:, :, 2] > 150))

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = mask.astype(bool)
    labels, num_labels = ndimage.label(mask)
    if num_labels > 0:
        top_row_labels = set(labels[0, :])
        top_row_labels.discard(0)
        if top_row_labels:
            mask = np.isin(labels, list(top_row_labels))
            labels, num_labels = ndimage.label(mask)
            if num_labels > 0:
                sizes = ndimage.sum(mask, labels, range(1, num_labels + 1))
                mask_size = mask.size
                big_enough = sizes > mask_size * 0.01
                mask = np.isin(labels, np.where(big_enough)[0] + 1)
    return (~mask).astype(np.int8)

def is_outdoor_scene(frame_data_list):
    sky_ratios = []
    for fd in frame_data_list:
        mask = fd.get('sorted_not_sky_global', np.ones(1))
        sky_ratio = 1.0 - np.mean(mask)
        sky_ratios.append(float(sky_ratio))
    significant = sum(1 for ratio in sky_ratios if ratio > 0.2)
    return significant >= len(sky_ratios) / 4

# ----------------- Update Handlers -----------------
# These functions are lightweight and respond only to their respective events.
def update_frame_visibility(server, frame_data_list, gui_timestep, num_frames, gui_show_global, gui_show_local, gui_show_track, gui_show_traj, gui_show_high_conf, gui_show_low_conf):
    current = int(gui_timestep.value)
    with server.atomic():
        for i in range(num_frames):
            fd = frame_data_list[i]
            # For simplicity, we show the frame if i = current.
            show_frame = (i == current)
            fd['frame_node'].visible = show_frame and show_frame
            fd['frustum_node'].visible = show_frame and show_frame
            fd['point_node_global'].visible = show_frame and gui_show_global.value
            fd['point_node_local'].visible = show_frame and gui_show_local.value
            fd['point_node_track'].visible = show_frame and gui_show_track.value
            fd['point_node_traj'].visible = show_frame and gui_show_traj.value
        server.flush()

def update_point_cloud_colors(server, frame_data_list, gui_timestep, gui_show_confidence_color, gui_rainbow_color_option, gui_show_global, gui_show_local, gui_show_track):
    with server.atomic():
        for i in range(len(frame_data_list)):
            fd = frame_data_list[i]
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global']
                colors_local = fd['colors_confidence_local']
                colors_track = fd['colors_confidence_track']
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global']
                colors_local = fd['colors_rainbow_local']
                colors_track = fd['colors_rainbow_track']
            else:
                colors_global = fd['colors_rgb_global']
                colors_local = fd['colors_rgb_local']
                colors_track = fd['colors_rgb_track']
            fd['point_node_global'].colors = colors_global if gui_show_global.value else []
            fd['point_node_local'].colors = colors_local if gui_show_local.value else []
            fd['point_node_track'].colors = colors_track if gui_show_track.value else []
            server.flush()

def update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option):
    for i in range(len(frame_data_list)):
        fd = frame_data_list[i]
        total_global = len(fd['sorted_pts3d_global'])
        total_local = len(fd['sorted_pts3d_local'])
        total_track = len(fd['sorted_pts3d_track'])

        # Calculate number of points to show based on the percentile
        num_global = max(1, int(total_global * (100 - gui_min_conf_percentile.value) / 100))
        num_local = max(1, int(total_local * (100 - gui_min_conf_percentile.value) / 100))
        num_track = max(1, int(total_track * (100 - gui_min_conf_percentile.value) / 100))

        # Apply sky masking if enabled
        if gui_mask_sky.value:
            mask_global = fd['sorted_not_sky_global'][:num_global]
            mask_local = fd['sorted_not_sky_local'][:num_local]
            mask_track = fd['sorted_not_sky_track'][:num_track]
            # Filter points based on the mask
            pts3d_global = fd['sorted_pts3d_global'][:num_global][mask_global > 0]
            pts3d_local = fd['sorted_pts3d_local'][:num_local][mask_local > 0]
            pts3d_track = fd['sorted_pts3d_track'][:num_track][mask_track > 0]
            
            # Select the appropriate colors based on the active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_confidence_local'][:num_local][mask_local > 0]
                colors_track = fd['colors_confidence_track'][:num_track][mask_track > 0]
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_rainbow_local'][:num_local][mask_local > 0]
                colors_track = fd['colors_rainbow_track'][:num_track][mask_track > 0]
            else:
                colors_global = fd['colors_rgb_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_rgb_local'][:num_local][mask_local > 0]
                colors_track = fd['colors_rgb_track'][:num_track][mask_track > 0]
        else:
            pts3d_global = fd['sorted_pts3d_global'][:num_global]
            pts3d_local = fd['sorted_pts3d_local'][:num_local]
            pts3d_track = fd['sorted_pts3d_track'][:num_track]

            # Select the appropriate colors based on the active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global'][:num_global]
                colors_local = fd['colors_confidence_local'][:num_local]
                colors_track = fd['colors_confidence_track'][:num_track]
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global'][:num_global]
                colors_local = fd['colors_rainbow_local'][:num_local]
                colors_track = fd['colors_rainbow_track'][:num_track]
            else:
                colors_global = fd['colors_rgb_global'][:num_global]
                colors_local = fd['colors_rgb_local'][:num_local]
                colors_track = fd['colors_rgb_track'][:num_track]

        # Update point clouds
        fd['point_node_global'].points = pts3d_global
        fd['point_node_local'].points = pts3d_local
        fd['point_node_track'].points = pts3d_track
        # update colors
        fd['point_node_global'].colors = colors_global
        fd['point_node_local'].colors = colors_local
        fd['point_node_track'].colors = colors_track
    server.flush()

# ----------------- Playback Loop -----------------
def playback_loop(gui_playing, gui_timestep, num_frames, gui_framerate):
    while True:
        if gui_playing.value:
            gui_timestep.value = (int(gui_timestep.value) + 1) % num_frames
        time.sleep(1.0 / float(gui_framerate.value))

def bind_update(widget, update_func):
    widget.on_update(lambda _: update_func())

# ----------------- Main Visualization Function -----------------
def start_visualization(output, min_conf_thr_percentile=10, global_conf_thr_value_to_drop_view=1.5, port=8020, point_size=0.0004,
                        traj_subsample_rate=1, traj_moving_percentile=0, traj_history_frames=10,
                        views=None, query_frame=0, dataset=None, dataset_idx=None, num_cameras=4,
                        reload_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
    """
    Start interactive 3D visualization with tracking support.
    
    Args:
        output: Dict containing 'preds' and 'views' from convert_input_to_pred_format
        min_conf_thr_percentile: Minimum confidence percentile for filtering points
        global_conf_thr_value_to_drop_view: Threshold for high/low confidence views
        port: Port for viser server
        point_size: Size of points in visualization
        traj_subsample_rate: Subsample rate for trajectories (1=all points, 10=every 10th point, etc.)
        traj_moving_percentile: Percentile threshold for filtering moving points (0=show all, 90=only top 10% moving)
        traj_history_frames: Number of history frames to show in trajectory (default 10)
        views: Original views list (optional, for query frame coloring and dynamic query frame support)
        query_frame: Which frame to use as query for tracking coloring (default 0)
        dataset: Optional dataset object for camera switching (multi-camera support)
        dataset_idx: Current dataset index being visualized
        num_cameras: Number of cameras available (for multi-camera datasets)
        reload_callback: Optional callback invoked to request a full reload (e.g. restart the script)
                        when camera/query settings change. If provided, the GUI "Apply" buttons
                        will call this instead of doing in-place updates.
    """
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.gui.set_panel_label("Show Controls")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)

    @server.on_client_connect
    def on_client_connect(client: viser.ClientHandle) -> None:
        with client.atomic():
            client.camera.position = (-0.00141163, -0.01910395, -0.06794288)
            client.camera.look_at = (-0.00352821, -0.01143425, 0.0154939)
        client.flush()

    poses_c2w_batch, estimated_focals = estimate_camera_poses(
        output['preds'], niter_PnP=100, focal_length_estimation_method='first_view_from_global_head'
    )
    poses_c2w = poses_c2w_batch[0]

    def _estimate_point_normalization_scale(anchor_idx: int) -> float:
        """
        Estimate the scalar normalization factor applied by convert_input_to_pred_format().

        That function:
          1) transforms all pts3d into the anchor camera frame (via inv(anchor_pose))
          2) then divides all point clouds by a single scalar (mean distance).

        We recover that scalar by comparing an unnormalized anchor-frame pointmap computed
        from `views` against the already-normalized pointmap stored in `output`.
        """
        if views is None:
            return 1.0
        if anchor_idx < 0 or anchor_idx >= len(views):
            return 1.0
        if "camera_pose" not in views[anchor_idx] or "pts3d" not in views[anchor_idx]:
            return 1.0
        try:
            c2w_anchor = to_numpy(views[anchor_idx]["camera_pose"])
            if c2w_anchor is None or c2w_anchor.shape != (4, 4) or (not np.isfinite(c2w_anchor).all()):
                return 1.0
            w2c_anchor = np.linalg.inv(c2w_anchor)
        except Exception:
            return 1.0

        try:
            pts3d_world = to_numpy(views[anchor_idx]["pts3d"])
            pts3d_world = np.asarray(pts3d_world, dtype=np.float32).reshape(-1, 3)
            pts3d_norm = to_numpy(output["preds"][anchor_idx]["pts3d_in_other_view"].cpu().squeeze())
            pts3d_norm = np.asarray(pts3d_norm, dtype=np.float32).reshape(-1, 3)
        except Exception:
            return 1.0

        if pts3d_world.shape != pts3d_norm.shape or pts3d_world.shape[1] != 3:
            return 1.0

        valid = np.isfinite(pts3d_world).all(axis=1) & np.isfinite(pts3d_norm).all(axis=1)
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return 1.0

        # Subsample for speed.
        max_samples = 8192
        if idx.size > max_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=max_samples, replace=False)

        pts_h = np.concatenate([pts3d_world[idx], np.ones((idx.size, 1), dtype=np.float32)], axis=1)
        pts_anchor = (pts_h @ w2c_anchor.T)[:, :3]

        n_unnorm = np.linalg.norm(pts_anchor, axis=1)
        n_norm = np.linalg.norm(pts3d_norm[idx], axis=1)
        good = np.isfinite(n_unnorm) & np.isfinite(n_norm) & (n_norm > 1e-6)
        if not np.any(good):
            return 1.0

        ratios = n_unnorm[good] / n_norm[good]
        scale = float(np.median(ratios))
        if not np.isfinite(scale) or scale <= 0:
            return 1.0
        return scale

    # If `output` was created by convert_input_to_pred_format(), points live in the
    # anchor camera frame and are globally scaled; convert dataset camera poses to
    # the same coordinate system so frustums align with the point cloud.
    dataset_pose_transform = None  # (w2c_anchor, scale)
    if views is not None and len(views) > 0:
        anchor_idx = min(int(query_frame), len(views) - 1)
        try:
            c2w_anchor = to_numpy(views[anchor_idx].get("camera_pose", None))
            if c2w_anchor is not None and np.asarray(c2w_anchor).shape == (4, 4) and np.isfinite(c2w_anchor).all():
                w2c_anchor = np.linalg.inv(np.asarray(c2w_anchor, dtype=np.float32))
                scale = _estimate_point_normalization_scale(anchor_idx)
                dataset_pose_transform = (w2c_anchor, float(scale))
        except Exception:
            dataset_pose_transform = None

    def _get_dataset_pose_and_focal(i: int):
        if views is None:
            return None, None
        if i < 0 or i >= len(views):
            return None, None
        v = views[i]
        if "camera_pose" not in v:
            return None, None
        pose = to_numpy(v["camera_pose"])
        if pose is None or not np.isfinite(pose).all():
            return None, None
        # Convert to the same coordinate system as the point cloud (anchor camera frame + scale).
        if dataset_pose_transform is not None:
            w2c_anchor, scale = dataset_pose_transform
            try:
                pose = (w2c_anchor @ np.asarray(pose, dtype=np.float32)).astype(np.float32)
                if np.isfinite(scale) and scale > 0:
                    pose[:3, 3] /= float(scale)
            except Exception:
                pass
        focal = None
        if "camera_intrinsics" in v:
            K = to_numpy(v["camera_intrinsics"])
            if K is not None and np.isfinite(K).all() and K.shape == (3, 3):
                focal = float(abs(K[1, 1]))
                if not np.isfinite(focal) or focal <= 0:
                    focal = None
        return pose, focal

    server.scene.set_up_direction((0.0, -1.0, 0.0))
    server.scene.world_axes.visible = False

    num_frames = len(output['preds'])
    frame_data_list = []
    cumulative_pts = []

    # ----------------- Grouped GUI Controls -----------------
    with server.gui.add_folder("Point and Camera Options", expand_by_default=False):
        gui_point_size = server.gui.add_slider("Point Size", min=1e-6, max=0.005, step=1e-5, initial_value=point_size)
        gui_frustum_size_percent = server.gui.add_slider("Camera Size (%)", min=0.1, max=10.0, step=0.1, initial_value=2.0)
        gui_show_camera = server.gui.add_checkbox("Show Camera", False)
        gui_mask_sky = server.gui.add_checkbox("Mask Sky", False)
        gui_show_confidence_color = server.gui.add_checkbox("Show Confidence", False)
        gui_rainbow_color_option = server.gui.add_checkbox("Color by View", False)

    with server.gui.add_folder("Playback Options", expand_by_default=False):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0)
        gui_next_frame = server.gui.add_button("Next Frame")
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_displayall = server.gui.add_checkbox("Display All", False)
        gui_framerate = server.gui.add_slider("FPS", min=0.25, max=60, step=0.25, initial_value=10)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("0.5", "1", "10", "20", "30", "60"))

    with server.gui.add_folder("Pointmap Head Options", expand_by_default=False):
        gui_show_global = server.gui.add_checkbox("Global", False)
        gui_show_local = server.gui.add_checkbox("Local", True)
        gui_show_track = server.gui.add_checkbox("Track", False)
        gui_show_traj = server.gui.add_checkbox("Traj", False)
    
    with server.gui.add_folder("Track/Traj Options", expand_by_default=False):
        gui_query_frame = server.gui.add_slider(
            "Query Frame", 
            min=0, max=num_frames - 1, step=1, initial_value=query_frame
        )
        gui_apply_query_frame = server.gui.add_button("Apply Query Frame")
        gui_query_frame_info = server.gui.add_text(
            "Info", 
            initial_value=f"Current query: frame {query_frame}",
            disabled=True
        )
    
    # Camera selection (for multi-camera datasets)
    with server.gui.add_folder("Camera Selection", expand_by_default=False):
        current_camera = dataset.camera_id if (dataset is not None and hasattr(dataset, 'camera_id')) else 0
        gui_camera_id = server.gui.add_dropdown(
            "Camera ID",
            options=[str(i) for i in range(num_cameras)],
            initial_value=str(current_camera)
        )
        gui_apply_camera = server.gui.add_button("Apply Camera (Reload)")
        gui_camera_info = server.gui.add_text(
            "Camera Info",
            initial_value=f"Current camera: {current_camera}",
            disabled=True
        )

    with server.gui.add_folder("Confidence Options", expand_by_default=False):
        gui_show_high_conf = server.gui.add_checkbox("Show High-Conf Views", True)
        # Default to True: many dataset/model confidence maps are in [0, 1],
        # so the high/low split can otherwise hide all camera frustums.
        gui_show_low_conf = server.gui.add_checkbox("Show Low-Conf Views", True)
        gui_global_conf_threshold = server.gui.add_slider("High/Low Conf Threshold", min=1.0, max=12.0, step=0.1, initial_value=global_conf_thr_value_to_drop_view)
        gui_min_conf_percentile = server.gui.add_slider("Per-View Conf Percentile", min=0, max=100, step=1, initial_value=min_conf_thr_percentile)

    button_render_gif = server.gui.add_button("Render a GIF")
    button_render_dynamic = server.gui.add_button("Render Dynamic")
    button_render_static = server.gui.add_button("Render Static")
    button_save_png = server.gui.add_button("Save Current View as PNG")

    @gui_next_frame.on_click
    def next_frame(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def prev_frame(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def playing_update(_):
        state = gui_playing.value
        gui_timestep.disabled = state
        gui_next_frame.disabled = state
        gui_prev_frame.disabled = state

    @gui_displayall.on_update
    def playall_update(_):
        state = gui_displayall.value

        gui_playing.value = False
        gui_playing.disabled = state
        gui_timestep.disabled = state
        gui_next_frame.disabled = state
        gui_prev_frame.disabled = state

        current = int(gui_timestep.value)

        with server.atomic():
            for i in range(num_frames):
                cur_shortcut = (state or i==current)

                fd = frame_data_list[i]
                fd['frame_node'].visible = True and cur_shortcut
                # Always show camera frustum for visible frames.
                fd['frustum_node'].visible = bool(gui_show_camera.value) and cur_shortcut
                fd['point_node_global'].visible = gui_show_global.value and cur_shortcut
                fd['point_node_local'].visible = gui_show_local.value and cur_shortcut
                fd['point_node_track'].visible = gui_show_track.value and cur_shortcut
        server.flush()

    @gui_framerate_options.on_click
    def fps_options(_):
        gui_framerate.value = float(gui_framerate_options.value)
    
    @gui_query_frame.on_update
    def query_frame_update(_):
        gui_query_frame_info.value = f"Query: {int(gui_query_frame.value)} (click Apply)"

    server.scene.add_frame("/cams", show_axes=False)
    
    # Store raw data for dynamic query frame support
    # We'll populate this after frame processing
    raw_frame_data = {
        'pts3d_track_all': [],  # Track positions for each frame (HW, 3)
        'img_rgb_all': [],       # RGB images for each frame (HW, 3)
        'views': views,          # Original views for recomputation
    }

    # ----------------- Frame Processing -----------------
    # Use query_frame for track coloring (which frame's RGB to use for track colors)
    query_frame_idx = min(query_frame, num_frames - 1)
    img_rgb_query_hw = to_numpy(output['views'][query_frame_idx]['img'].cpu().squeeze().permute(1, 2, 0))
    img_rgb_query_flat_full = img_rgb_query_hw.reshape(-1, 3)
    not_sky_mask_query_flat_full = detect_sky_mask(img_rgb_query_hw).flatten().astype(np.int8)

    # Track tensors are indexed by query-frame pixels; only keep query pixels that are valid.
    # Otherwise many invalid query pixels appear at/near the origin and make track coloring look wrong.
    track_valid_query_flat = None
    if views is not None and len(views) > query_frame_idx:
        try:
            tv = views[query_frame_idx].get("track_valid_mask", None)
            if tv is None:
                tv = views[query_frame_idx].get("valid_mask", None)
            if tv is not None:
                tv = np.asarray(tv, dtype=bool).reshape(-1)
                if tv.shape[0] == img_rgb_query_flat_full.shape[0]:
                    track_valid_query_flat = tv
        except Exception:
            track_valid_query_flat = None

    if track_valid_query_flat is not None:
        img_rgb_query_flat = img_rgb_query_flat_full[track_valid_query_flat]
        not_sky_mask_query_flat = not_sky_mask_query_flat_full[track_valid_query_flat]
    else:
        img_rgb_query_flat = img_rgb_query_flat_full
        not_sky_mask_query_flat = not_sky_mask_query_flat_full

    for i in tqdm(range(num_frames)):
        pred = output['preds'][i]
        view = output['views'][i]

        img_rgb_orig = to_numpy(view['img'].cpu().squeeze().permute(1,2,0))
        not_sky_mask = detect_sky_mask(img_rgb_orig).flatten().astype(np.int8)

        pts3d_global = to_numpy(pred['pts3d_in_other_view'].cpu().squeeze()).reshape(-1, 3)
        conf_global = to_numpy(pred['conf'].cpu().squeeze()).flatten()
        pts3d_local = to_numpy(pred['pts3d_local_aligned_to_global_for_causalvggt'].cpu().squeeze()).reshape(-1, 3) # TODO: hard code only for causevggt
        conf_local = to_numpy(pred['conf_local'].cpu().squeeze()).flatten()
        pts3d_track = to_numpy(pred['track_aligned_to_global'].cpu().squeeze()).reshape(-1, 3)
        conf_track = to_numpy(pred['conf_track'].cpu().squeeze()).flatten()
        # Track in world coordinates for trajectory visualization (so background stays fixed)
        pts3d_track_world = to_numpy(pred['track_in_world'].cpu().squeeze()).reshape(-1, 3)

        img_rgb = to_numpy(view['img'].cpu().squeeze().permute(1,2,0))
        img_rgb_flat = img_rgb.reshape(-1, 3)
        
        # Store raw data for dynamic query frame support
        raw_frame_data['pts3d_track_all'].append(pts3d_track.copy())
        raw_frame_data['img_rgb_all'].append(img_rgb_flat.copy())
        if track_valid_query_flat is not None and pts3d_track.shape[0] == track_valid_query_flat.shape[0]:
            pts3d_track = pts3d_track[track_valid_query_flat]
            conf_track = conf_track[track_valid_query_flat]
            pts3d_track_world = pts3d_track_world[track_valid_query_flat]
        img_rgb_flat_track = img_rgb_query_flat

        cumulative_pts.append(pts3d_global)

        sort_idx_global = np.argsort(-conf_global)
        sorted_conf_global = conf_global[sort_idx_global]
        sorted_pts3d_global = pts3d_global[sort_idx_global]
        sorted_img_rgb_global = img_rgb_flat[sort_idx_global]
        sorted_not_sky_global = not_sky_mask[sort_idx_global]

        sort_idx_local = np.argsort(-conf_local)
        sorted_conf_local = conf_local[sort_idx_local]
        sorted_pts3d_local = pts3d_local[sort_idx_local]
        # Local pointmap: color by the current frame RGB (track is the one colored by query RGB).
        sorted_img_rgb_local = img_rgb_flat[sort_idx_local]
        sorted_not_sky_local = not_sky_mask[sort_idx_local]

        # Keep query-pixel order for track (needed for consistent trajectories and stable coloring).
        sort_idx_track = np.arange(pts3d_track.shape[0])
        sorted_conf_track = conf_track[sort_idx_track]
        sorted_pts3d_track = pts3d_track[sort_idx_track]
        sorted_pts3d_track_world = pts3d_track_world[sort_idx_track]  # World coords for trajectory
        sorted_img_rgb_track = img_rgb_flat_track[sort_idx_track]
        sorted_not_sky_track = not_sky_mask_query_flat[sort_idx_track]

        colors_rgb_global = ((sorted_img_rgb_global + 1) * 127.5).astype(np.uint8) / 255.0
        colors_rgb_local = ((sorted_img_rgb_local + 1) * 127.5).astype(np.uint8) / 255.0
        colors_rgb_track = ((sorted_img_rgb_track + 1) * 127.5).astype(np.uint8) / 255.0

        conf_norm_global = (sorted_conf_global - sorted_conf_global.min()) / (sorted_conf_global.max() - sorted_conf_global.min() + 1e-8)
        conf_norm_local = (sorted_conf_local - sorted_conf_local.min()) / (sorted_conf_local.max() - sorted_conf_local.min() + 1e-8)
        conf_norm_track = (sorted_conf_track - sorted_conf_track.min()) / (sorted_conf_track.max() - sorted_conf_track.min() + 1e-8)
        colormap = cm.turbo
        colors_confidence_global = colormap(conf_norm_global)[:, :3]
        colors_confidence_local = colormap(conf_norm_local)[:, :3]
        colors_confidence_track = colormap(conf_norm_track)[:, :3]

        def rainbow_color(n, total):
            import colorsys
            hue = n / total
            return colorsys.hsv_to_rgb(hue, 1.0, 1.0)

        rainbow_color_for_frame = rainbow_color(i, num_frames)
        colors_rainbow_global = np.tile(rainbow_color_for_frame, (sorted_pts3d_global.shape[0], 1))
        colors_rainbow_local = np.tile(rainbow_color_for_frame, (sorted_pts3d_local.shape[0], 1))
        colors_rainbow_track = np.tile(rainbow_color_for_frame, (sorted_pts3d_track.shape[0], 1))

        max_conf_global = conf_global.max()
        is_high_confidence = max_conf_global >= gui_global_conf_threshold.value

        c2w_ds, focal_ds = _get_dataset_pose_and_focal(i)
        c2w = c2w_ds if c2w_ds is not None else poses_c2w[i]
        height, width = view['img'].shape[2], view['img'].shape[3]
        if focal_ds is not None:
            focal_length = focal_ds
        else:
            # Convert focal_length to float, handle None and tensor types
            fl = estimated_focals[0][i]
            if fl is None:
                focal_length = None
            elif hasattr(fl, 'item'):  # torch.Tensor
                focal_length = float(fl.item())
            else:
                focal_length = float(fl) if fl is not None else None
        img_rgb_reshaped = img_rgb.reshape(height, width, 3)
        img_rgb_normalized = ((img_rgb_reshaped + 1) * 127.5).astype(np.uint8)
        img_downsampled = img_rgb_normalized[::4, ::4]

        frame_data = {
            'sorted_pts3d_global': sorted_pts3d_global,
            'colors_rgb_global': colors_rgb_global,
            'colors_confidence_global': colors_confidence_global,
            'colors_rainbow_global': colors_rainbow_global,
            'sorted_pts3d_local': sorted_pts3d_local,
            'colors_rgb_local': colors_rgb_local,
            'colors_confidence_local': colors_confidence_local,
            'colors_rainbow_local': colors_rainbow_local,
            'sorted_not_sky_global': sorted_not_sky_global,
            'sorted_not_sky_local': sorted_not_sky_local,
            'max_conf_global': float(max_conf_global),
            'sorted_pts3d_track': sorted_pts3d_track,
            'sorted_pts3d_track_world': sorted_pts3d_track_world,  # World coords for trajectory
            'colors_rgb_track': colors_rgb_track,
            'colors_confidence_track': colors_confidence_track,
            'colors_rainbow_track': colors_rainbow_track,
            'sorted_not_sky_track': sorted_not_sky_track,
            'is_high_confidence': is_high_confidence,
            # Needed for correct query-frame recoloring (colors must follow the same point order).
            'sort_idx_local': sort_idx_local,
            'sort_idx_track': sort_idx_track,
            'c2w': c2w,
            'height': height,
            'width': width,
            'focal_length': focal_length,
            'img_downsampled': img_downsampled,
            'rainbow_color': rainbow_color_for_frame,
        }
        frame_data_list.append(frame_data)

    # Percentile for scene extent calculation (10th to 90th percentile by default)
    extent_percentile = 80
    cumulative_pts_combined = np.concatenate(cumulative_pts, axis=0)
    # Calculate percentiles for each coordinate
    min_coords = np.percentile(cumulative_pts_combined, 100 - extent_percentile, axis=0)
    max_coords = np.percentile(cumulative_pts_combined, extent_percentile, axis=0)
    scene_extent = max_coords - min_coords
    max_extent = np.max(scene_extent)

    # ----------------- Create Visualization Nodes -----------------
    # Filter invalid track points (points with all coordinates < 0.1 are considered invalid)
    # Use world coordinates for trajectory so background stays fixed when camera moves
    track_valid_mask = ~(np.all(frame_data_list[0]['sorted_pts3d_track_world'] < 0.1, axis=1))

    track_points_first_world = frame_data_list[0]['sorted_pts3d_track_world'][track_valid_mask]

    # Compute movement for each point across all frames (in world coordinates)
    delta = np.zeros_like(track_points_first_world)
    for i in tqdm(range(num_frames), desc="Computing point movements"):
        track_points_cur_world = frame_data_list[i]['sorted_pts3d_track_world'][track_valid_mask]
        delta += np.abs(track_points_first_world - track_points_cur_world)
    norms = np.linalg.norm(delta, axis=1)
    
    # Apply moving filter based on percentile (0 = show all points, 90 = only top 10% moving)
    if traj_moving_percentile > 0:
        threshold = np.percentile(norms, traj_moving_percentile)
        moving_mask = norms >= threshold
        print(f"Trajectory filter: showing {moving_mask.sum()} / {len(moving_mask)} points (top {100-traj_moving_percentile}% moving)")
    else:
        moving_mask = np.ones(len(norms), dtype=bool)  # Show all points
        print(f"Trajectory filter: showing all {moving_mask.sum()} valid points (dense tracking mode)")

    for i in tqdm(range(num_frames), desc="Creating visualization nodes"):
        fd = frame_data_list[i]
        frame_node = server.scene.add_frame(f"/cams/t{i}", show_axes=False, visible=False)

        # Build trajectory lines from history frames using world coordinates
        # This ensures background stays fixed and only moving objects show trajectories
        lines = []
        for j in range(max(i - traj_history_frames, 0), i + 1):
            track_points_prev = frame_data_list[max(j-1, 0)]['sorted_pts3d_track_world'][track_valid_mask][moving_mask]
            track_points_next = frame_data_list[j]['sorted_pts3d_track_world'][track_valid_mask][moving_mask]
            line = np.stack((track_points_prev, track_points_next), axis=1)  # N, 2, 3
            lines.append(line)

        lines = np.stack(lines, axis=1)  # N, history_len, 2, 3
        
        # Apply subsampling based on traj_subsample_rate (1 = all points, 10 = every 10th)
        if traj_subsample_rate > 1:
            lines = lines[::traj_subsample_rate]
        num_lines, len_t = lines.shape[:2]

        import matplotlib.colors as mcolors
        def generate_uniform_colors(N):
            hues = np.linspace(0, 1, N, endpoint=False)  # 色相均匀分布
            hsv = np.stack([hues, np.ones(N), np.ones(N)], axis=1)  # 饱和度和明度都是1
            rgb = mcolors.hsv_to_rgb(hsv)  # 转成RGB，0~1
            rgb_255 = (rgb * 255).astype(int)  # 乘255并转成整数
            return rgb_255

        traj_colors = generate_uniform_colors(num_lines)
        # traj_colors = traj_colors[np.random.permutation(num_lines)]

        traj_colors = traj_colors[:, np.newaxis, np.newaxis, :]
        traj_colors = np.tile(traj_colors, (1, len_t, 2, 1))
        lines = lines.reshape(-1, 2, 3)
        traj_colors = traj_colors.reshape(-1, 2, 3)

        point_node_traj = server.scene.add_line_segments(
            name = f"/traj/t{i}",
            points = lines,
            colors = traj_colors,
            line_width=4,
            visible=False,
        )

        point_node_global = server.scene.add_point_cloud(
            name=f"/pts3d_global/t{i}",
            points=fd['sorted_pts3d_global'],
            colors=fd['colors_rgb_global'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )
        point_node_local = server.scene.add_point_cloud(
            name=f"/pts3d_local/t{i}",
            points=fd['sorted_pts3d_local'],
            colors=fd['colors_rgb_local'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )
        point_node_track = server.scene.add_point_cloud(
            name=f"/pts3d_track/t{i}",
            points=fd['sorted_pts3d_track'],
            colors=fd['colors_rgb_track'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )

        rotation_matrix = fd['c2w'][:3, :3]
        position = fd['c2w'][:3, 3]
        rotation_quaternion = tf.SO3.from_matrix(rotation_matrix).wxyz
        # Calculate FOV with fallback for failed pose estimation
        fl = fd['focal_length']
        h = fd['height']
        try:
            # Ensure all values are Python floats (handle torch.Tensor)
            h_float = float(h.item()) if hasattr(h, 'item') else float(h)
            if fl is None:
                fov = np.deg2rad(60)
            else:
                fl_float = float(fl.item()) if hasattr(fl, 'item') else float(fl)
                if not np.isfinite(fl_float) or fl_float <= 0:
                    fov = np.deg2rad(60)
                else:
                    fov = 2.0 * np.arctan2(h_float / 2.0, fl_float)
        except Exception as e:
            print(f"Error calculating FOV: {e}")
            fov = np.deg2rad(60)
        aspect_ratio = fd['width'] / fd['height']
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)

        frustum_node = server.scene.add_camera_frustum(
            name=f"/cams/t{i}/frustum",
            fov=fov,
            aspect=aspect_ratio,
            scale=frustum_scale,
            color=fd['rainbow_color'],
            image=fd['img_downsampled'],
            wxyz=rotation_quaternion,
            position=position,
            visible=False,
        )

        fd['frame_node'] = frame_node
        fd['point_node_global'] = point_node_global
        fd['point_node_local'] = point_node_local
        fd['point_node_track'] = point_node_track
        fd['point_node_traj'] = point_node_traj
        fd['frustum_node'] = frustum_node

    # Initially set all nodes hidden
    for fd in frame_data_list:
        fd['frame_node'].visible = False
        fd['point_node_global'].visible = False
        fd['point_node_local'].visible = False
        fd['point_node_track'].visible = False
        fd['point_node_traj'].visible = False
        fd['frustum_node'].visible = False
    server.flush()

    # Initialize timestep to show first frames and disable playing
    gui_timestep.value = 0
    gui_playing.value = False

    # Scene type detection and sky masking initialization
    is_outdoor = is_outdoor_scene(frame_data_list)
    gui_mask_sky.value = False # we dont mask sky for tracking

    print("\nScene type detection:")
    sky_ratios = [1.0 - np.mean(fd['sorted_not_sky_global']) for fd in frame_data_list]
    significant = sum(1 for r in sky_ratios if r > 0.2)
    print(f"- Found {significant}/{len(sky_ratios)} frames with significant sky presence (>20% sky pixels)")
    print(f"- Scene classified as: {'outdoor' if is_outdoor else 'indoor'}, setting mask_sky to {is_outdoor}")

    # Initial visibility setup
    with server.atomic():
        for i in range(num_frames):
            fd = frame_data_list[i]
            if i == gui_timestep.value:
                fd['frame_node'].visible = True
                # Always show camera frustum for the current frame.
                fd['frustum_node'].visible = bool(gui_show_camera.value)
            else:
                fd['frame_node'].visible = False
                fd['frustum_node'].visible = False
            
            # Set up initial points with sky masking if needed
            pts3d_global = fd['sorted_pts3d_global']
            pts3d_local = fd['sorted_pts3d_local']
            pts3d_track = fd['sorted_pts3d_track']
            
            # Select appropriate colors based on active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global']
                colors_local = fd['colors_confidence_local']
                colors_track = fd['colors_confidence_track']
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global']
                colors_local = fd['colors_rainbow_local']
                colors_track = fd['colors_rainbow_track']
            else:
                colors_global = fd['colors_rgb_global']
                colors_local = fd['colors_rgb_local']
                colors_track = fd['colors_rgb_track']
            
            if is_outdoor and gui_mask_sky.value:  # Apply sky masking if outdoor scene
                mask_global = fd['sorted_not_sky_global']
                mask_local = fd['sorted_not_sky_local']
                mask_track = fd['sorted_not_sky_track']
                pts3d_global = pts3d_global[mask_global > 0]
                pts3d_local = pts3d_local[mask_local > 0]
                pts3d_track = pts3d_track[mask_track > 0]
                colors_global = colors_global[mask_global > 0]
                colors_local = colors_local[mask_local > 0]
                colors_track = colors_track[mask_track > 0]
            
            # Update point clouds
            fd['point_node_global'].points = pts3d_global
            fd['point_node_local'].points = pts3d_local
            fd['point_node_track'].points = pts3d_track
            fd['point_node_global'].colors = colors_global
            fd['point_node_local'].colors = colors_local
            fd['point_node_track'].colors = colors_track
            if i == gui_timestep.value:
                fd['point_node_global'].visible = gui_show_global.value
                fd['point_node_local'].visible = gui_show_local.value
                fd['point_node_track'].visible = gui_show_track.value
                fd['point_node_traj'].visible = gui_show_traj.value
            else:
                fd['point_node_global'].visible = False
                fd['point_node_local'].visible = False
                fd['point_node_track'].visible = False
                fd['point_node_traj'].visible = False

    server.flush()

    # ----------------- GUI Callback Updates -----------------
    @gui_timestep.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                # we only show current frame
                if i == current:
                    fd['frame_node'].visible = True
                    # Always show camera frustum for the current frame.
                    fd['frustum_node'].visible = bool(gui_show_camera.value)
                    fd['point_node_global'].visible = gui_show_global.value
                    fd['point_node_local'].visible = gui_show_local.value
                    fd['point_node_track'].visible = gui_show_track.value
                    fd['point_node_traj'].visible = gui_show_traj.value
                else:
                    fd['frame_node'].visible = False
                    fd['frustum_node'].visible = False
                    fd['point_node_global'].visible = False
                    fd['point_node_local'].visible = False
                    fd['point_node_track'].visible = False
                    fd['point_node_traj'].visible = False
        server.flush()

    @gui_point_size.on_update
    def _(_):
        with server.atomic():
            for fd in frame_data_list:
                fd['point_node_global'].point_size = gui_point_size.value
                fd['point_node_local'].point_size = gui_point_size.value
                fd['point_node_track'].point_size = gui_point_size.value
        server.flush()

    @gui_frustum_size_percent.on_update
    def _(_):
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)
        with server.atomic():
            for fd in frame_data_list:
                fd['frustum_node'].scale = frustum_scale
        server.flush()

    @gui_show_camera.on_update
    def _(_):
        current = int(gui_timestep.value)
        state = bool(gui_show_camera.value)
        display_all = bool(gui_displayall.value)
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if display_all:
                    fd['frustum_node'].visible = state
                else:
                    fd['frustum_node'].visible = state and (i == current)
        server.flush()

    @gui_show_confidence_color.on_update
    def _(_):
        # Make options mutually exclusive
        if gui_show_confidence_color.value and gui_rainbow_color_option.value:
            gui_rainbow_color_option.value = False
        
        # Update colors for all visible points
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_rainbow_color_option.on_update
    def _(_):
        # Make options mutually exclusive
        if gui_rainbow_color_option.value and gui_show_confidence_color.value:
            gui_show_confidence_color.value = False
            
        # Update colors for all visible points
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_min_conf_percentile.on_update
    def _(_):
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_mask_sky.on_update
    def _(_):
        # For each visible frame, update filtering if mask sky changes.
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_show_global.on_update
    def _(_):
        with server.atomic():
            current = int(gui_timestep.value)
            for i in range(num_frames):
                if i == current:
                    frame_data_list[i]['point_node_global'].visible = gui_show_global.value
                else:
                    frame_data_list[i]['point_node_global'].visible = False
        server.flush()

    @gui_show_local.on_update
    def _(_):
        with server.atomic():
            current = int(gui_timestep.value)
            for i in range(num_frames):
                if i == current:
                    frame_data_list[i]['point_node_local'].visible = gui_show_local.value
                else:
                    frame_data_list[i]['point_node_local'].visible = False
        server.flush()

    @gui_show_track.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            for i in range(num_frames):
                if i == current:
                    frame_data_list[i]['point_node_track'].visible = gui_show_track.value
                else:
                    frame_data_list[i]['point_node_track'].visible = False
        server.flush()
    
    @gui_show_traj.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            for i in range(num_frames):
                if i == current:
                    frame_data_list[i]['point_node_traj'].visible = gui_show_traj.value
                else:
                    frame_data_list[i]['point_node_traj'].visible = False
        server.flush()

    @gui_show_high_conf.on_update
    def _(_):
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if i == int(gui_timestep.value):
                    # Hide frustum and points if high confidence views are disabled
                    if fd['is_high_confidence'] and gui_show_high_conf.value:
                        fd['point_node_global'].visible = gui_show_global.value and gui_show_high_conf.value
                        fd['point_node_local'].visible = gui_show_local.value and gui_show_high_conf.value
                    else:
                        fd['point_node_global'].visible = False  # Hide if not high confidence
                        fd['point_node_local'].visible = False  # Hide if not high confidence
        server.flush()

    @gui_show_low_conf.on_update
    def _(_):
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if i == int(gui_timestep.value):
                    # Hide frustum and points if low confidence views are disabled
                    if (not fd['is_high_confidence']) and gui_show_low_conf.value:
                        fd['point_node_global'].visible = gui_show_global.value and gui_show_low_conf.value
                        fd['point_node_local'].visible = gui_show_local.value and gui_show_low_conf.value
                    else:
                        fd['point_node_global'].visible = False  # Hide if high confidence
                        fd['point_node_local'].visible = False  # Hide if high confidence
        server.flush()

    @gui_global_conf_threshold.on_update
    def _(_):
        for fd in frame_data_list:
            fd['is_high_confidence'] = fd['max_conf_global'] >= gui_global_conf_threshold.value
        server.flush()

    # ----------------- Start Playback Loop -----------------
    def local_playback_loop():
        while True:
            if gui_playing.value:
                gui_timestep.value = (int(gui_timestep.value) + 1) % num_frames
            time.sleep(1.0 / float(gui_framerate.value))
    playback_thread = threading.Thread(target=local_playback_loop)
    playback_thread.start()
    
    # ----------------- Query Frame and Camera Selection Callbacks -----------------
    @gui_apply_query_frame.on_click
    def _(_):
        """Apply new query frame."""
        new_query = int(gui_query_frame.value)
        if new_query < 0 or new_query >= num_frames:
            gui_query_frame_info.value = f"Invalid query frame: {new_query}"
            return

        if reload_callback is not None:
            reload_callback(
                {
                    "query_frame": new_query,
                    "camera_id": int(gui_camera_id.value),
                    "dataset_idx": dataset_idx,
                }
            )
            return

        # Dataset-backed visualization: fully reload the sample for the new query frame so that
        # Track geometry + colors actually correspond to the selected query frame.
        if dataset is not None and dataset_idx is not None and hasattr(dataset, "set_query_frame"):
            nonlocal views, output, poses_c2w, estimated_focals, dataset_pose_transform, max_extent, track_valid_query_flat
            try:
                dataset.set_query_frame(new_query)
                new_views = dataset[dataset_idx]
                from holo4d_runtime.data.stream3r.dust3r.datasets.utils.transforms import convert_input_to_pred_format

                new_output = convert_input_to_pred_format(new_views, query_frame=new_query)
                poses_c2w_batch, estimated_focals_new = estimate_camera_poses(
                    new_output["preds"], niter_PnP=100, focal_length_estimation_method="first_view_from_global_head"
                )
                poses_c2w_new = poses_c2w_batch[0]
            except Exception as e:
                gui_query_frame_info.value = f"Failed to reload query frame {new_query}: {e}"
                return

            views = new_views
            output = new_output
            poses_c2w = poses_c2w_new
            estimated_focals = estimated_focals_new

            # Recompute dataset pose transform for the new anchor + scaling.
            dataset_pose_transform = None
            if views is not None and len(views) > 0:
                anchor_idx = min(int(new_query), len(views) - 1)
                try:
                    c2w_anchor = to_numpy(views[anchor_idx].get("camera_pose", None))
                    if c2w_anchor is not None and np.asarray(c2w_anchor).shape == (4, 4) and np.isfinite(c2w_anchor).all():
                        w2c_anchor = np.linalg.inv(np.asarray(c2w_anchor, dtype=np.float32))
                        scale = _estimate_point_normalization_scale(anchor_idx)
                        dataset_pose_transform = (w2c_anchor, float(scale))
                except Exception:
                    dataset_pose_transform = None

            # Rebuild query RGB + validity mask for track.
            img_rgb_query_hw = to_numpy(output["views"][new_query]["img"].cpu().squeeze().permute(1, 2, 0))
            img_rgb_query_flat_full = img_rgb_query_hw.reshape(-1, 3)
            not_sky_mask_query_flat_full = detect_sky_mask(img_rgb_query_hw).flatten().astype(np.int8)
            track_valid_query_flat = None
            try:
                tv = views[new_query].get("track_valid_mask", None)
                if tv is None:
                    tv = views[new_query].get("valid_mask", None)
                if tv is not None:
                    tv = np.asarray(tv, dtype=bool).reshape(-1)
                    if tv.shape[0] == img_rgb_query_flat_full.shape[0]:
                        track_valid_query_flat = tv
            except Exception:
                track_valid_query_flat = None

            if track_valid_query_flat is not None:
                img_rgb_query_flat = img_rgb_query_flat_full[track_valid_query_flat]
                not_sky_mask_query_flat = not_sky_mask_query_flat_full[track_valid_query_flat]
            else:
                img_rgb_query_flat = img_rgb_query_flat_full
                not_sky_mask_query_flat = not_sky_mask_query_flat_full

            # Reset dynamic data stores.
            raw_frame_data["pts3d_track_all"] = []
            raw_frame_data["img_rgb_all"] = []

            # Update per-frame data buffers in-place.
            cumulative_pts_new = []
            for i in range(num_frames):
                pred_i = output["preds"][i]
                view_i = output["views"][i]

                img_rgb_i = to_numpy(view_i["img"].cpu().squeeze().permute(1, 2, 0))
                img_rgb_flat_i = img_rgb_i.reshape(-1, 3)
                not_sky_i = detect_sky_mask(img_rgb_i).flatten().astype(np.int8)

                pts3d_global_i = to_numpy(pred_i["pts3d_in_other_view"].cpu().squeeze()).reshape(-1, 3)
                conf_global_i = to_numpy(pred_i["conf"].cpu().squeeze()).flatten()
                pts3d_local_i = to_numpy(pred_i["pts3d_local_aligned_to_global_for_causalvggt"].cpu().squeeze()).reshape(-1, 3)
                conf_local_i = to_numpy(pred_i["conf_local"].cpu().squeeze()).flatten()

                pts3d_track_i = to_numpy(pred_i["track_aligned_to_global"].cpu().squeeze()).reshape(-1, 3)
                conf_track_i = to_numpy(pred_i["conf_track"].cpu().squeeze()).flatten()
                pts3d_track_world_i = to_numpy(pred_i["track_in_world"].cpu().squeeze()).reshape(-1, 3)

                raw_frame_data["pts3d_track_all"].append(pts3d_track_i.copy())
                raw_frame_data["img_rgb_all"].append(img_rgb_flat_i.copy())

                if track_valid_query_flat is not None and pts3d_track_i.shape[0] == track_valid_query_flat.shape[0]:
                    pts3d_track_i = pts3d_track_i[track_valid_query_flat]
                    conf_track_i = conf_track_i[track_valid_query_flat]
                    pts3d_track_world_i = pts3d_track_world_i[track_valid_query_flat]

                cumulative_pts_new.append(pts3d_global_i)

                sort_idx_global_i = np.argsort(-conf_global_i)
                sort_idx_local_i = np.argsort(-conf_local_i)
                sort_idx_track_i = np.arange(pts3d_track_i.shape[0])

                sorted_pts3d_global_i = pts3d_global_i[sort_idx_global_i]
                sorted_pts3d_local_i = pts3d_local_i[sort_idx_local_i]
                sorted_pts3d_track_i = pts3d_track_i[sort_idx_track_i]
                sorted_pts3d_track_world_i = pts3d_track_world_i[sort_idx_track_i]

                sorted_img_rgb_global_i = img_rgb_flat_i[sort_idx_global_i]
                sorted_img_rgb_local_i = img_rgb_flat_i[sort_idx_local_i]
                sorted_img_rgb_track_i = img_rgb_query_flat[sort_idx_track_i]

                colors_rgb_global_i = ((sorted_img_rgb_global_i + 1) * 127.5).astype(np.uint8) / 255.0
                colors_rgb_local_i = ((sorted_img_rgb_local_i + 1) * 127.5).astype(np.uint8) / 255.0
                colors_rgb_track_i = ((sorted_img_rgb_track_i + 1) * 127.5).astype(np.uint8) / 255.0

                sorted_conf_global_i = conf_global_i[sort_idx_global_i]
                sorted_conf_local_i = conf_local_i[sort_idx_local_i]
                sorted_conf_track_i = conf_track_i[sort_idx_track_i]
                conf_norm_global_i = (sorted_conf_global_i - sorted_conf_global_i.min()) / (sorted_conf_global_i.max() - sorted_conf_global_i.min() + 1e-8)
                conf_norm_local_i = (sorted_conf_local_i - sorted_conf_local_i.min()) / (sorted_conf_local_i.max() - sorted_conf_local_i.min() + 1e-8)
                conf_norm_track_i = (sorted_conf_track_i - sorted_conf_track_i.min()) / (sorted_conf_track_i.max() - sorted_conf_track_i.min() + 1e-8)
                colormap = cm.turbo
                colors_confidence_global_i = colormap(conf_norm_global_i)[:, :3]
                colors_confidence_local_i = colormap(conf_norm_local_i)[:, :3]
                colors_confidence_track_i = colormap(conf_norm_track_i)[:, :3]

                rainbow_color_for_frame = frame_data_list[i]["rainbow_color"]
                colors_rainbow_global_i = np.tile(rainbow_color_for_frame, (sorted_pts3d_global_i.shape[0], 1))
                colors_rainbow_local_i = np.tile(rainbow_color_for_frame, (sorted_pts3d_local_i.shape[0], 1))
                colors_rainbow_track_i = np.tile(rainbow_color_for_frame, (sorted_pts3d_track_i.shape[0], 1))

                max_conf_global_i = float(conf_global_i.max()) if conf_global_i.size > 0 else 0.0
                is_high_confidence_i = max_conf_global_i >= float(gui_global_conf_threshold.value)

                c2w_ds, focal_ds = _get_dataset_pose_and_focal(i)
                c2w_i = c2w_ds if c2w_ds is not None else poses_c2w[i]
                height_i, width_i = view_i["img"].shape[2], view_i["img"].shape[3]
                if focal_ds is not None:
                    focal_length_i = focal_ds
                else:
                    fl = estimated_focals[0][i]
                    if fl is None:
                        focal_length_i = None
                    elif hasattr(fl, "item"):
                        focal_length_i = float(fl.item())
                    else:
                        focal_length_i = float(fl) if fl is not None else None

                img_rgb_normalized = ((img_rgb_i + 1) * 127.5).astype(np.uint8)
                img_downsampled_i = img_rgb_normalized[::4, ::4]

                fd = frame_data_list[i]
                fd["sorted_pts3d_global"] = sorted_pts3d_global_i
                fd["sorted_pts3d_local"] = sorted_pts3d_local_i
                fd["sorted_pts3d_track"] = sorted_pts3d_track_i
                fd["sorted_pts3d_track_world"] = sorted_pts3d_track_world_i

                fd["colors_rgb_global"] = colors_rgb_global_i
                fd["colors_rgb_local"] = colors_rgb_local_i
                fd["colors_rgb_track"] = colors_rgb_track_i
                fd["colors_confidence_global"] = colors_confidence_global_i
                fd["colors_confidence_local"] = colors_confidence_local_i
                fd["colors_confidence_track"] = colors_confidence_track_i
                fd["colors_rainbow_global"] = colors_rainbow_global_i
                fd["colors_rainbow_local"] = colors_rainbow_local_i
                fd["colors_rainbow_track"] = colors_rainbow_track_i

                fd["sorted_not_sky_global"] = not_sky_i[sort_idx_global_i]
                fd["sorted_not_sky_local"] = not_sky_i[sort_idx_local_i]
                fd["sorted_not_sky_track"] = not_sky_mask_query_flat[sort_idx_track_i]

                fd["max_conf_global"] = max_conf_global_i
                fd["is_high_confidence"] = is_high_confidence_i
                fd["sort_idx_local"] = sort_idx_local_i
                fd["sort_idx_track"] = sort_idx_track_i
                fd["c2w"] = c2w_i
                fd["height"] = height_i
                fd["width"] = width_i
                fd["focal_length"] = focal_length_i
                fd["img_downsampled"] = img_downsampled_i

                try:
                    fd["frustum_node"].position = c2w_i[:3, 3]
                    fd["frustum_node"].wxyz = tf.SO3.from_matrix(c2w_i[:3, :3]).wxyz
                except Exception:
                    pass

            # Recompute scene extent for camera scaling.
            cumulative_pts_combined = np.concatenate(cumulative_pts_new, axis=0)
            min_coords = np.percentile(cumulative_pts_combined, 20, axis=0)
            max_coords = np.percentile(cumulative_pts_combined, 80, axis=0)
            max_extent = float(np.max(max_coords - min_coords))
            frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)
            with server.atomic():
                for fd in frame_data_list:
                    fd["frustum_node"].scale = frustum_scale
            server.flush()

            # Update trajectories to match the new track set.
            try:
                track_valid_mask = ~(np.all(frame_data_list[0]["sorted_pts3d_track_world"] < 0.1, axis=1))
                track_points_first_world = frame_data_list[0]["sorted_pts3d_track_world"][track_valid_mask]
                delta = np.zeros_like(track_points_first_world)
                for i in range(num_frames):
                    track_points_cur_world = frame_data_list[i]["sorted_pts3d_track_world"][track_valid_mask]
                    delta += np.abs(track_points_first_world - track_points_cur_world)
                norms = np.linalg.norm(delta, axis=1)
                if traj_moving_percentile > 0:
                    threshold = np.percentile(norms, traj_moving_percentile)
                    moving_mask = norms >= threshold
                else:
                    moving_mask = np.ones(len(norms), dtype=bool)

                import matplotlib.colors as mcolors

                def generate_uniform_colors(N):
                    hues = np.linspace(0, 1, N, endpoint=False)
                    hsv = np.stack([hues, np.ones(N), np.ones(N)], axis=1)
                    rgb = mcolors.hsv_to_rgb(hsv)
                    return (rgb * 255).astype(int)

                for i in range(num_frames):
                    lines = []
                    for j in range(max(i - traj_history_frames, 0), i + 1):
                        prev = frame_data_list[max(j - 1, 0)]["sorted_pts3d_track_world"][track_valid_mask][moving_mask]
                        nxt = frame_data_list[j]["sorted_pts3d_track_world"][track_valid_mask][moving_mask]
                        lines.append(np.stack((prev, nxt), axis=1))
                    lines = np.stack(lines, axis=1)  # N, history_len, 2, 3
                    if traj_subsample_rate > 1:
                        lines = lines[::traj_subsample_rate]
                    num_lines, len_t = lines.shape[:2]
                    traj_colors = generate_uniform_colors(num_lines)[:, np.newaxis, np.newaxis, :]
                    traj_colors = np.tile(traj_colors, (1, len_t, 2, 1))
                    lines = lines.reshape(-1, 2, 3)
                    traj_colors = traj_colors.reshape(-1, 2, 3)

                    fd = frame_data_list[i]
                    fd["point_node_traj"].points = lines
                    fd["point_node_traj"].colors = traj_colors
            except Exception:
                pass

            # Refresh points/colors according to current filtering + color mode.
            update_points_filtering(
                server,
                frame_data_list,
                gui_timestep,
                gui_min_conf_percentile,
                gui_mask_sky,
                gui_show_confidence_color,
                gui_rainbow_color_option,
            )
            gui_query_frame_info.value = f"Applied query frame: {new_query}"
            print(f"Reloaded sample for query frame {new_query}")
            return
        
        print(f"Applying query frame: {new_query}")

        new_query_rgb_flat_full = raw_frame_data['img_rgb_all'][new_query]
        if not (
            isinstance(new_query_rgb_flat_full, np.ndarray)
            and new_query_rgb_flat_full.ndim == 2
            and new_query_rgb_flat_full.shape[1] == 3
        ):
            gui_query_frame_info.value = f"Query frame image has unexpected shape: {getattr(new_query_rgb_flat_full, 'shape', None)}"
            return
        if track_valid_query_flat is not None and new_query_rgb_flat_full.shape[0] == track_valid_query_flat.shape[0]:
            new_query_rgb_flat = new_query_rgb_flat_full[track_valid_query_flat]
        else:
            new_query_rgb_flat = new_query_rgb_flat_full

        # Update track colors for all frames so they correspond to the query frame RGB.
        # Respect track's query-pixel ordering.
        with server.atomic():
            # Sky mask for track should follow the query frame if sky masking is enabled.
            try:
                h0 = int(frame_data_list[0]['height'])
                w0 = int(frame_data_list[0]['width'])
                if new_query_rgb_flat_full.shape[0] == h0 * w0:
                    new_query_rgb_hw = new_query_rgb_flat_full.reshape(h0, w0, 3)
                    new_not_sky_query_flat_full = detect_sky_mask(new_query_rgb_hw).flatten().astype(np.int8)
                    if track_valid_query_flat is not None and new_not_sky_query_flat_full.shape[0] == track_valid_query_flat.shape[0]:
                        new_not_sky_query_flat = new_not_sky_query_flat_full[track_valid_query_flat]
                    else:
                        new_not_sky_query_flat = new_not_sky_query_flat_full
                else:
                    new_not_sky_query_flat = None
            except Exception:
                new_not_sky_query_flat = None

            for fd in frame_data_list:
                sort_idx_track = fd.get('sort_idx_track', None)
                if sort_idx_track is None:
                    continue
                if len(sort_idx_track) != new_query_rgb_flat.shape[0]:
                    continue

                fd['colors_rgb_track'] = ((new_query_rgb_flat[sort_idx_track] + 1) * 127.5).astype(np.uint8) / 255.0

                if new_not_sky_query_flat is not None:
                    fd['sorted_not_sky_track'] = new_not_sky_query_flat[sort_idx_track]

        # Refresh points/colors according to current filtering + color mode.
        update_points_filtering(
            server,
            frame_data_list,
            gui_timestep,
            gui_min_conf_percentile,
            gui_mask_sky,
            gui_show_confidence_color,
            gui_rainbow_color_option,
        )
        gui_query_frame_info.value = f"Applied query frame: {new_query}"
        print(f"Track colors updated from query frame {new_query}")
    
    @gui_apply_camera.on_click
    def _(_):
        """Apply camera selection - shows info about reloading."""
        new_camera = int(gui_camera_id.value)
        current_camera = dataset.camera_id if (dataset is not None and hasattr(dataset, 'camera_id')) else 0

        if reload_callback is not None:
            reload_callback(
                {
                    "camera_id": new_camera,
                    "query_frame": int(gui_query_frame.value),
                    "dataset_idx": dataset_idx,
                }
            )
            return
        
        if dataset is None or not hasattr(dataset, 'set_camera_id'):
            gui_camera_info.value = f"Camera switching requires kubric_multicam dataset"
            return
        
        if new_camera == current_camera:
            gui_camera_info.value = f"Already using camera {new_camera}"
            return
        
        # Update the dataset's camera_id
        dataset.set_camera_id(new_camera)
        gui_camera_info.value = f"Switched to camera {new_camera}. Restart visualization to see changes."
        print(f"Camera switched to {new_camera}. Restart the visualization script to reload data.")

    @button_render_gif.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        fps = gui_framerate.value
        if client is None:
            print("Error: No client connected.")
            return
        try:
            # 获取当前相机参数
            cam_pos = np.array(client.camera.position)
            look_at = np.array(client.camera.look_at)
            up = np.array(client.camera.up_direction)
            # 计算场景中心
            all_pts = np.concatenate([fd['sorted_pts3d_global'] for fd in frame_data_list], axis=0)
            center = np.mean(all_pts, axis=0)
            # 计算半径和轨迹
            radius = np.linalg.norm(cam_pos - center)
            circle_radius = radius * 0.1  # 小圆，半径为当前视角到中心距离的10%
            n_frames = 100
            images = []
            for i in range(n_frames):
                theta = 2 * np.pi * i / n_frames
                # 在当前视角附近做圆周运动
                # 以当前相机的up方向和cam_pos->center方向确定圆的平面
                forward = (center - cam_pos)
                forward = forward / np.linalg.norm(forward)
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                # 圆周点
                offset = np.cos(theta) * right * circle_radius + np.sin(theta) * up * circle_radius
                new_pos = cam_pos + offset
                client.camera.position = tuple(new_pos)
                client.camera.look_at = tuple(center)
                client.camera.up_direction = tuple(up)
                client.flush()
                time.sleep(0.05)
                img = client.get_render(height=720, width=1280)
                images.append(img)

            # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # save_dir = f"visualization_circle_{now}"
            # os.makedirs(save_dir, exist_ok=True)
            # for i, img in enumerate(images):
            #     iio.imwrite(os.path.join(save_dir, f"frame_{i:04d}.png"), img)

            gif_bytes = iio.imwrite("<bytes>", images, extension=".gif", fps=fps, loop=0)
            client.send_file_download("visualization.gif", gif_bytes)
        except Exception as e:
            print(f"Error while rendering trajectory GIF: {e}")

    @button_render_dynamic.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            print("Error: No client connected.")
            return
        try:
            images = []
            original_timestep = gui_timestep.value
            original_playing = gui_playing.value
            gui_playing.value = False
            fps = gui_framerate.value
            for i in range(num_frames):
                gui_timestep.value = i
                time.sleep(0.1)
                images.append(client.get_render(height=720, width=1280))

            # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # save_dir = f"visualization_dynamic_{now}"
            # os.makedirs(save_dir, exist_ok=True)
            # for i, img in enumerate(images):
            #     iio.imwrite(os.path.join(save_dir, f"frame_{i:04d}.png"), img)

            # Create and save GIF
            gif_bytes = iio.imwrite("<bytes>", images, extension=".gif", fps=fps, loop=0)
            client.send_file_download("visualization.gif", gif_bytes)
            gui_timestep.value = original_timestep
            gui_playing.value = original_playing
        except Exception as e:
            print(f"Error while rendering GIF: {e}")

    @button_render_static.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            print("Error: No client connected.")
            return
        try:
            images = []
            original_timestep = gui_timestep.value
            original_playing = gui_playing.value
            gui_playing.value = False
            fps = gui_framerate.value
            for i in range(num_frames):
                gui_timestep.value = i
                # Show all previous pointmaps and frustums up to and including i
                with server.atomic():
                    for j in range(num_frames):
                        fd = frame_data_list[j]
                        if j <= i:
                            fd['point_node_global'].visible = gui_show_global.value
                            fd['point_node_local'].visible = gui_show_local.value
                            fd['point_node_track'].visible = gui_show_track.value
                            fd['point_node_traj'].visible = gui_show_traj.value
                            fd['frame_node'].visible = True
                            fd['frustum_node'].visible = True  # Show all previous frustums
                        else:
                            fd['point_node_global'].visible = False
                            fd['point_node_local'].visible = False
                            fd['point_node_track'].visible = False
                            fd['point_node_traj'].visible = False
                            fd['frame_node'].visible = False
                            fd['frustum_node'].visible = False
                server.flush()
                time.sleep(0.1)
                images.append(client.get_render(height=720, width=1280))
            # Restore visibility to only the current timestep
            with server.atomic():
                for j in range(num_frames):
                    fd = frame_data_list[j]
                    if j == original_timestep:
                        fd['point_node_global'].visible = gui_show_global.value
                        fd['point_node_local'].visible = gui_show_local.value
                        fd['point_node_track'].visible = gui_show_track.value
                        fd['point_node_traj'].visible = gui_show_traj.value
                        # Restore frustum visibility based on confidence settings
                        if fd['is_high_confidence']:
                            fd['frustum_node'].visible = gui_show_high_conf.value
                        else:
                            fd['frustum_node'].visible = gui_show_low_conf.value
                    else:
                        fd['point_node_global'].visible = False
                        fd['point_node_local'].visible = False
                        fd['point_node_track'].visible = False
                        fd['point_node_traj'].visible = False
                        fd['frustum_node'].visible = False
            server.flush()

            # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # save_dir = f"visualization_static_{now}"
            # os.makedirs(save_dir, exist_ok=True)
            # for i, img in enumerate(images):
            #     iio.imwrite(os.path.join(save_dir, f"frame_{i:04d}.png"), img)

            gif_bytes = iio.imwrite("<bytes>", images, extension=".gif", fps=fps, loop=0)
            client.send_file_download("visualization.gif", gif_bytes)
            gui_timestep.value = original_timestep
            gui_playing.value = original_playing
        except Exception as e:
            import traceback
            print(f"Error while rendering static GIF: {e}")
            print("Traceback:")
            print(traceback.format_exc())

    @button_save_png.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            print("Error: No client connected.")
            return
        try:
            # Get current render from client
            image = client.get_render(height=720, width=1280)

            # Generate filename with timestamp
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualization_{now}.png"

            # Convert to bytes
            img_bytes = iio.imwrite("<bytes>", image, extension=".png")

            # Send file to client for download
            client.send_file_download(filename, img_bytes)

        except Exception as e:
            import traceback
            print(f"Error while saving PNG: {e}")
            print("Traceback:")
            print(traceback.format_exc())

    # public_url = server.request_share_url()
    return server

# Example usage:
# server = start_visualization(output=your_output_dict, min_conf_thr_percentile=10, global_conf_thr_value_to_drop_view=1.5, port=8020)
