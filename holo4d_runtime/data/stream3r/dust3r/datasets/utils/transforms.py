# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
import torch
import torchvision.transforms as tvf

from holo4d_runtime.data.stream3r.dust3r.utils.image import ImgNorm

from holo4d_runtime.data.stream3r.dust3r.utils.image import ImgNorm
from holo4d_runtime.data.stream3r.dust3r.utils.device import todevice
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import ( geotrf, inv,)
from holo4d_runtime.data.stream3r.dust3r.utils.visualizer import Visualizer

# define the standard image transforms
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])

# from CoTracker and DELTA
TrackAug = tvf.Compose([
    tvf.RandomApply([tvf.ColorJitter(0.2, 0.2, 0.2, 0.25/3.14)], p=0.25),
    tvf.RandomApply([tvf.GaussianBlur(11, sigma=(0.1, 2.0))], p=0.05),
    ImgNorm
])


def normalize_pointcloud_from_views(pts_list, norm_mode="avg_dis", valid_list=None):
    """Normalize point clouds from multiple views, excluding invalid points from normalization."""
    assert all(pts.ndim >= 3 and pts.shape[-1] == 3 for pts in pts_list)

    norm_mode, dis_mode = norm_mode.split("_")

    # Concatenate all point clouds and valid masks if provided
    all_pts = torch.cat(pts_list, dim=1)
    if valid_list is not None:
        all_valid = torch.cat(valid_list, dim=1)
        valid_pts = all_pts[all_valid]  # Keep only valid points for norm calculation
    else:
        valid_pts = all_pts

    # Compute the distance to the origin for valid points
    dis = valid_pts.norm(dim=-1)

    # Apply distance transformation based on dis_mode
    if dis_mode == "dis":
        pass  # Do nothing
    elif dis_mode == "log1p":
        dis = torch.log1p(dis)
    elif dis_mode == "warp-log1p":
        log_dis = torch.log1p(dis)
        warp_factor = log_dis / dis.clip(min=1e-8)
        all_pts = all_pts * warp_factor.view(-1, 1)  # Warp the points with the warp factor
        dis = log_dis  # The final distance is now the log-transformed distance
    else:
        raise ValueError(f"Unsupported distance mode: {dis_mode}")

    # Apply different normalization modes
    if norm_mode == "avg":
        norm_factor = dis.mean()  # Compute mean distance of valid points
    elif norm_mode == "median":
        norm_factor = dis.median()  # Compute median distance of valid points
    else:
        raise ValueError(f"Unsupported normalization mode: {norm_mode}")

    norm_factor = norm_factor.clip(min=1e-8)  # Prevent division by zero

    # Normalize all point clouds
    if valid_list is not None:
        normalized_pts = [torch.where(valid.unsqueeze(-1), pts / norm_factor, pts)
                        for pts, valid in zip(pts_list, valid_list)]
    else:
        normalized_pts = [pts / norm_factor for pts in pts_list]

    return normalized_pts


def convert_input_to_pred_format(views, query_frame=0):
    """
    Convert dataset views to prediction format for visualization.

    Args:
        views: List of view dictionaries from dataset
        query_frame: Which frame to use as the anchor/query for tracking (default: 0)
                     This determines which frame's pixel positions are tracked across time.
    """
    views = todevice(views, "cpu")
    n = len(views)
    output = dict()

    output["preds"] = [dict() for _ in range(n)]
    output["views"] = [dict() for _ in range(n)]
    output["query_frame"] = query_frame  # Store for reference

    # Use query_frame as anchor instead of hardcoded frame 0
    anchor_idx = min(query_frame, n - 1)
    inv_matrix_anchor = inv(views[anchor_idx]["camera_pose"].float())

    for i in range(n):
        # track from dataset shows where reference frame's points are at each frame
        # Transform track to anchor frame's coordinate system
        if "track" in views[i]:
            output["preds"][i]["track_aligned_to_global"] = geotrf(inv_matrix_anchor, views[i]["track"].unsqueeze(0))
        else:
            # Fallback: use pts3d directly
            output["preds"][i]["track_aligned_to_global"] = geotrf(inv_matrix_anchor, views[i]["pts3d"].unsqueeze(0))

        output["preds"][i]["pts3d_in_other_view"] = geotrf(inv_matrix_anchor, views[i]["pts3d"].unsqueeze(0))
        output["preds"][i]["pts3d_local_aligned_to_global"] = geotrf(inv_matrix_anchor, views[i]["pts3d"].unsqueeze(0))

        output["preds"][i]["conf"] = 10 * torch.ones(1, *views[i]["pts3d"].shape[:-1])
        output["preds"][i]["conf_local"] = 10 * torch.ones(1, *views[i]["pts3d"].shape[:-1])
        output["preds"][i]["conf_track"] = 10 * torch.ones(1, *views[i]["pts3d"].shape[:-1])

        output["views"][i]["img"] = views[i]["img"].unsqueeze(0)

    # Normalize all point clouds together so they have consistent scale
    pts3ds_all = normalize_pointcloud_from_views(
        [output["preds"][i]["pts3d_in_other_view"] for i in range(n)] +
        [output["preds"][i]["pts3d_local_aligned_to_global"] for i in range(n)] +
        [output["preds"][i]["track_aligned_to_global"] for i in range(n)]
    )
    pts3ds_in_other_view = pts3ds_all[:n]
    pts3d_local_aligned_to_global = pts3ds_all[n:2*n]
    track = pts3ds_all[2*n:3*n]

    for i in range(n):
        output["preds"][i]["pts3d_in_other_view"] = pts3ds_in_other_view[i]
        output["preds"][i]["pts3d_local_aligned_to_global"] = pts3d_local_aligned_to_global[i]
        output["preds"][i]["track_aligned_to_global"] = track[i]
        # Local visualization uses pts3d (per-frame depth-based point cloud)
        # This shows the same positions as Global but allows different confidence filtering
        output["preds"][i]["pts3d_local_aligned_to_global_for_causalvggt"] = pts3d_local_aligned_to_global[i]
        # For trajectory visualization - use same data as track (camera_0 coords) so lines align with Track point cloud
        # Clone to ensure no reference issues
        output["preds"][i]["track_in_world"] = track[i].clone()

    return output


# the very hard code vis_track, assume image is not transposed in dataset (because of K)
def vis_track(views):
    views = todevice(views, "cpu")

    def convert_3d_tracks_to_2d(tracks_3d, camera_intrinsics, visibilitys):
        n = len(tracks_3d)
        ret = []
        ret_vis = []
        for i in range(n):
            point_map = tracks_3d[i]

            if visibilitys is None:
                visibility = torch.ones(*point_map.shape[:-1], 1, dtype=torch.bool)
            else:
                visibility = visibilitys[i]

            K = camera_intrinsics[i]

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            X = point_map[..., 0]
            Y = point_map[..., 1]
            Z = point_map[..., 2].clamp(min=1e-6)

            u = fx * X / Z + cx
            v = fy * Y / Z + cy

            projected_2d = torch.stack((u, v), dim=-1)
            sampled_projected_2d = projected_2d[::20, ::20, :].reshape(-1, 2) # sample points

            # hard code rescale
            import torch.nn.functional as F
            visibility = visibility.squeeze(-1).float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 24, 512, 512]
            visibility = F.interpolate(visibility, size=(384, 512), mode='nearest')
            visibility = visibility.unsqueeze(-1).bool().squeeze(0).squeeze(0)

            sampled_visibility = visibility[::20, ::20, :].reshape(-1, 1) # sample points
            ret.append(sampled_projected_2d)
            ret_vis.append(sampled_visibility)

        return ret, ret_vis

    tracks_2d, visibilitys = convert_3d_tracks_to_2d(
        [view["track"] for view in views],
        [view["camera_intrinsics"] for view in views],
        [view["visibility"] for view in views] if "visibility" in views[0] else None,
    )
    tracks_2d = torch.stack([track_2d for track_2d in tracks_2d]).unsqueeze(0)
    visibility = torch.stack([visibility for visibility in visibilitys]).unsqueeze(0)

    video = torch.stack([view["img"] for view in views]).unsqueeze(0)
    video = ((video + 1.) / 2.) * 255.

    vis = Visualizer(save_dir="./develop/saved_track_videos", pad_value=120, linewidth=1, fps=5)
    vis.visualize(video, tracks_2d, visibility)