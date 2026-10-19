import ast
import csv
import hashlib
import json
import math
import os
import os.path as osp
import pickle
import struct
import tarfile
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL.Image
import torch

try:
    import cv2
except Exception:  # pragma: no cover - cv2 should be present in training env
    cv2 = None
try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - optional until new-format data is used
    safe_open = None

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.bedlam import hdri_scenes, invalid_seqs
import holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.utils.cropping as cropping
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from holo4d_runtime.data.stream3r.dust3r.utils.image import imread_cv2


SYN4D_INDEX_CACHE_VERSION = 2
SYN4D_SEQUENCE_CACHE_VERSION = 1


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def _safe_cache_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _is_background_mask_name(mask_name: str) -> bool:
    return "env" in str(mask_name).lower()


@dataclass
class ActorPose:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float


def _parse_comment_config(comment: str | None) -> dict[str, str]:
    config = {}
    if not comment:
        return config
    for value in str(comment).split(";"):
        if not value or "=" not in value:
            continue
        key, val = value.split("=", 1)
        config[key] = val
    return config


def read_seq_csv(csv_path: str) -> list[dict]:
    with open(csv_path, mode="r", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))

    group_info_list = []
    current_group = None
    for row in csv_rows:
        row_type = row.get("Type")
        if row_type == "Comment":
            continue

        if row_type == "Group":
            if current_group is not None:
                group_info_list.append(current_group)

            group_config = _parse_comment_config(row.get("Comment", ""))
            cameraroot_location = None
            if (
                "cameraroot_x" in group_config
                and "cameraroot_y" in group_config
                and "cameraroot_z" in group_config
            ):
                cameraroot_location = {
                    "x": float(group_config["cameraroot_x"]),
                    "y": float(group_config["cameraroot_y"]),
                    "z": float(group_config["cameraroot_z"]),
                }

            current_group = {
                "group_config": group_config,
                "camera_pose": ActorPose(
                    float(row["X"]),
                    float(row["Y"]),
                    float(row["Z"]),
                    float(row["Yaw"]),
                    float(row["Pitch"]),
                    float(row["Roll"]),
                ),
                "sequence_name": group_config.get("sequence_name"),
                "sequence_frames": int(group_config["frames"]) if "frames" in group_config else None,
                "hdri": group_config.get("hdri"),
                "camera_hfov": float(group_config["camera_hfov"]) if "camera_hfov" in group_config else None,
                "time": float(group_config["time"]) if "time" in group_config else None,
                "cameraroot_yaw": float(group_config["cameraroot_yaw"]) if "cameraroot_yaw" in group_config else None,
                "cameraroot_location": cameraroot_location,
                "bodies": [],
                "objects": [],
            }
            continue

        if current_group is None:
            continue

        if row_type == "Body":
            body = row["Body"]
            body_config = _parse_comment_config(row.get("Comment", ""))
            subject = "undefined"
            animation_id = "undefined"
            if body.startswith("moyo"):
                subject = body.split("_", maxsplit=1)[0]
                animation_id = body
            elif "_" in body:
                subject = body.rsplit("_", maxsplit=1)[0]
                animation_id = body.rsplit("_", maxsplit=1)[1]
            current_group["bodies"].append(
                {
                    "index": int(row["Index"]) if row.get("Index") else None,
                    "body": body,
                    "subject": subject,
                    "animation_id": animation_id,
                    "pose": ActorPose(
                        float(row["X"]),
                        float(row["Y"]),
                        float(row["Z"]),
                        float(row["Yaw"]),
                        float(row["Pitch"]),
                        float(row["Roll"]),
                    ),
                    "start_frame": int(body_config.get("start_frame", 0)),
                    "texture_body": body_config.get("texture_body"),
                    "texture_clothing": body_config.get("texture_clothing"),
                    "texture_clothing_overlay": body_config.get("texture_clothing_overlay"),
                    "hair": body_config.get("hair"),
                    "haircolor": body_config.get("haircolor"),
                    "shoe": body_config.get("shoe"),
                    "body_config": body_config,
                }
            )
            continue

        if row_type == "Object":
            object_config = _parse_comment_config(row.get("Comment", ""))
            current_group["objects"].append(
                {
                    "index": int(row["Index"]) if row.get("Index") else None,
                    "body": row.get("Body"),
                    "object_id": object_config.get("object_id"),
                    "object_group": object_config.get("object_group"),
                    "anim_name": object_config.get("anim_name"),
                    "pose": ActorPose(
                        float(row["X"]),
                        float(row["Y"]),
                        float(row["Z"]),
                        float(row["Yaw"]),
                        float(row["Pitch"]),
                        float(row["Roll"]),
                    ),
                    "object_config": object_config,
                }
            )

    if current_group is not None:
        group_info_list.append(current_group)
    return group_info_list


def read_depth_exr(path: str) -> np.ndarray:
    try:
        import Imath
        import OpenEXR

        exr_file = OpenEXR.InputFile(path)
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_map = header.get("channels", {})
        for channel_name in ("Depth", "Z", "R", "Y"):
            if channel_name not in channel_map:
                continue
            channel = exr_file.channel(channel_name, pixel_type)
            depth = np.frombuffer(channel, dtype=np.float32).reshape(height, width)
            return np.asarray(depth, dtype=np.float32)
        raise IOError(f"No supported depth channel found in EXR {path}. Available channels: {list(channel_map.keys())}")
    except Exception:
        depth = imread_cv2(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return np.asarray(depth, dtype=np.float32)


def _parse_npz_npy_header(npz_path: str, array_name: str):
    """Parse the .npy header inside a STORED .npz to get shape/dtype/offset without loading data.

    Returns (shape, dtype, data_byte_offset) or None if the array is compressed or malformed.
    """
    npy_member = f"{array_name}.npy"
    try:
        zf = zipfile.ZipFile(npz_path, "r")
        info = zf.getinfo(npy_member)
        zf.close()
        if info.compress_type != zipfile.ZIP_STORED:
            return None

        with open(npz_path, "rb") as f:
            f.seek(info.header_offset)
            sig = f.read(4)
            if sig != b"PK\x03\x04":
                return None
            hdr = f.read(26)
            fname_len = struct.unpack("<H", hdr[22:24])[0]
            extra_len = struct.unpack("<H", hdr[24:26])[0]
            npy_start = info.header_offset + 30 + fname_len + extra_len

            f.seek(npy_start)
            magic = f.read(8)
            if magic[:6] != b"\x93NUMPY":
                return None
            ver_major = magic[6]
            if ver_major == 1:
                header_len = struct.unpack("<H", f.read(2))[0]
            elif ver_major in (2, 3):
                header_len = struct.unpack("<I", f.read(4))[0]
            else:
                return None
            header_str = f.read(header_len).decode("latin1").strip()
            header_dict = ast.literal_eval(header_str)
            if header_dict.get("fortran_order", False):
                return None
            return tuple(header_dict["shape"]), np.dtype(header_dict["descr"]), f.tell()
    except Exception:
        return None


def _mmap_npz_array_selective(npz_path: str, array_name: str, frame_indices: np.ndarray) -> np.ndarray:
    """Read only selected frames from a STORED (uncompressed) .npy inside a .npz via mmap.

    For a 11GB vertices_sequence.npz, this reads 49 frames in ~0.5s vs 38s for full load.
    Falls back to normal np.load if the array is compressed or the format is unexpected.
    """
    parsed = _parse_npz_npy_header(npz_path, array_name)
    if parsed is not None:
        try:
            shape, dtype, data_offset = parsed
            mm = np.memmap(npz_path, dtype=dtype, mode="r", offset=data_offset, shape=shape)
            result = np.array(mm[frame_indices])
            del mm
            return result
        except Exception:
            pass  # fall through to normal load
    # Fallback: normal load
    with np.load(npz_path, allow_pickle=True) as data:
        return np.array(data[array_name][frame_indices])


def get_wrapped_pts3d(
    target_pts3d,
    target_tracking_npz_dict,
    vertices_seq_d_list,
    faces_list,
    source_frame_idx,
    depth_query_valid_map,
    static_background_valid_map,
    wrapped_face_distance_thresh=0.1,
):
    pts3d_target = torch.as_tensor(target_pts3d, dtype=torch.float32).clone()
    h, w, _ = pts3d_target.shape
    wrapped_valid_mask = torch.zeros((h, w), dtype=torch.bool)
    static_background_valid_map = torch.as_tensor(static_background_valid_map, dtype=torch.bool)
    if static_background_valid_map.ndim == 3 and static_background_valid_map.shape[-1] == 1:
        static_background_valid_map = static_background_valid_map[:, :, 0]
    wrapped_valid_mask |= static_background_valid_map

    for mask_type, target_tracking_npz_data in target_tracking_npz_dict.items():
        masked_valid = torch.as_tensor(target_tracking_npz_data["masked_valid"], dtype=torch.bool)
        if masked_valid.ndim == 3 and masked_valid.shape[-1] == 1:
            masked_valid = masked_valid[:, :, 0]
        face_index_map_np = target_tracking_npz_data.get("face_index_map", target_tracking_npz_data.get("face_indices"))
        face_index_map = torch.as_tensor(face_index_map_np, dtype=torch.long)
        if face_index_map.ndim == 3 and face_index_map.shape[-1] == 1:
            face_index_map = face_index_map[:, :, 0]
        direct_points = target_tracking_npz_data.get("corresponding_points_all")
        if direct_points is not None:
            direct_points = torch.as_tensor(direct_points, dtype=torch.float32)
            valid_y, valid_x = torch.where(masked_valid)
            valid_mask_with_vertex = (face_index_map >= 0) & masked_valid
            if valid_mask_with_vertex.any() and source_frame_idx < direct_points.shape[0]:
                valid_y2, valid_x2 = torch.where(valid_mask_with_vertex)
                pixel_to_n = torch.full((h, w), -1, dtype=torch.long)
                pixel_to_n[valid_y, valid_x] = torch.arange(valid_y.numel(), dtype=torch.long)
                n_indices = pixel_to_n[valid_y2, valid_x2]
                pts3d_target[valid_y2, valid_x2, :] = direct_points[source_frame_idx, n_indices, :]
                wrapped_valid_mask[valid_y2, valid_x2] = True
            continue
        if mask_type not in faces_list or mask_type not in vertices_seq_d_list:
            continue

        weights = torch.as_tensor(target_tracking_npz_data["weights"], dtype=torch.float32)
        orientation_sign = torch.as_tensor(target_tracking_npz_data["orientation_sign"], dtype=torch.float32)
        distances = torch.as_tensor(target_tracking_npz_data["distances"], dtype=torch.float32)

        faces = torch.as_tensor(faces_list[mask_type], dtype=torch.long)
        original_mesh_sequence = torch.as_tensor(vertices_seq_d_list[mask_type], dtype=torch.float32)

        nearest_indices = face_index_map[masked_valid]
        n = nearest_indices.numel()
        if n == 0:
            continue

        distances_flat = distances.reshape(-1)[:n]
        valid_y, valid_x = torch.where(masked_valid)
        if float(wrapped_face_distance_thresh) < 1e5:
            valid_distance_mask = distances_flat <= float(wrapped_face_distance_thresh)
            face_index_map = face_index_map.clone()
            if valid_y.numel() == valid_distance_mask.numel():
                invalid_mask = ~valid_distance_mask
                if invalid_mask.any():
                    face_index_map[valid_y[invalid_mask], valid_x[invalid_mask]] = -1

        projected_faces = faces[nearest_indices.flatten()]
        projected_faces_point = original_mesh_sequence[:, projected_faces.flatten()].reshape(-1, n, 3, 3)
        face_normal_all = torch.cross(
            projected_faces_point[:, :, 1] - projected_faces_point[:, :, 0],
            projected_faces_point[:, :, 2] - projected_faces_point[:, :, 0],
            dim=-1,
        )
        face_normal_all = torch.nn.functional.normalize(face_normal_all, eps=1e-6, dim=-1)

        bary_w = weights[0].reshape(-1, 3)[:n]
        projected_points_bary_all = torch.sum(
            projected_faces_point * bary_w.unsqueeze(0).unsqueeze(-1),
            dim=2,
        )
        face_normal_all = face_normal_all * orientation_sign.reshape(-1)[:n].view(1, n, 1)
        corresponding_points_all = projected_points_bary_all + torch.abs(distances_flat).view(1, n, 1) * face_normal_all

        vertex_track_map = torch.zeros_like(pts3d_target)
        vertex_track_map[valid_y, valid_x, :] = corresponding_points_all[source_frame_idx]

        valid_mask_with_vertex = (face_index_map >= 0) & masked_valid
        if valid_mask_with_vertex.any():
            valid_y2, valid_x2 = torch.where(valid_mask_with_vertex)
            pts3d_target[valid_y2, valid_x2, :] = vertex_track_map[valid_y2, valid_x2, :]
            wrapped_valid_mask[valid_y2, valid_x2] = True

    depth_query_valid_map = torch.as_tensor(depth_query_valid_map, dtype=torch.bool)
    if depth_query_valid_map.ndim == 3 and depth_query_valid_map.shape[-1] == 1:
        depth_query_valid_map = depth_query_valid_map[:, :, 0]
    # All depth-valid pixels are valid (see precompute_all_wrapped_pts3d comment).
    wrapped_valid_mask = depth_query_valid_map
    pts3d_target[~wrapped_valid_mask] = 0.0
    return pts3d_target.cpu().numpy(), wrapped_valid_mask.cpu().numpy()


def get_wrapped_pts3d_compact_faceid_np(
    target_pts3d,
    target_tracking_npz_dict,
    vertices_seq_d_list,
    faces_list,
    source_frame_idx,
    target_frame_idx,
    depth_query_valid_map,
    static_background_valid_map,
    wrapped_face_distance_thresh=0.1,
    mask_bg=False,
):
    """Numpy reconstruction for the compact_faceid Syn4D tracking schema.

    The compact ``tracking_safetensors`` shard stores only
    (``valid_flat_idx``, ``face_ids``) for the ``target_frame_idx`` pixel grid.
    Barycentric weights, orientation sign and signed distance are RE-DERIVED
    here by projecting each query point onto its face triangle at
    ``target_frame_idx``, then the wrapped position is evaluated on the same
    face at ``source_frame_idx``. This reproduces (to float precision) the
    values the old dense ``vertex_indices_*.npz`` stored precomputed.
    """
    pts3d_target_np = np.asarray(target_pts3d, dtype=np.float32).copy()
    h, w, _ = pts3d_target_np.shape
    pts3d_target_flat = pts3d_target_np.reshape(-1, 3)
    wrapped_valid_mask = np.zeros((h, w), dtype=bool)
    foreground_valid_mask = np.zeros((h, w), dtype=bool)

    static_background_valid = np.asarray(static_background_valid_map, dtype=bool)
    if static_background_valid.ndim == 3 and static_background_valid.shape[-1] == 1:
        static_background_valid = static_background_valid[:, :, 0]
    wrapped_valid_mask |= static_background_valid

    source_frame_idx = int(source_frame_idx)
    target_frame_idx = int(target_frame_idx)
    distance_thresh = float(wrapped_face_distance_thresh)

    for mask_type, target_tracking_npz_data in target_tracking_npz_dict.items():
        if mask_type not in faces_list or mask_type not in vertices_seq_d_list:
            continue

        faces = np.asarray(faces_list[mask_type], dtype=np.int64)
        original_mesh_sequence = np.asarray(vertices_seq_d_list[mask_type], dtype=np.float32)
        if (
            source_frame_idx < 0
            or target_frame_idx < 0
            or source_frame_idx >= original_mesh_sequence.shape[0]
            or target_frame_idx >= original_mesh_sequence.shape[0]
            or faces.size == 0
        ):
            continue

        valid_flat_idx = np.asarray(target_tracking_npz_data["valid_flat_idx"], dtype=np.int64).reshape(-1)
        nearest_indices = np.asarray(target_tracking_npz_data["face_ids"], dtype=np.int64).reshape(-1)
        stored_h = int(target_tracking_npz_data.get("height", h))
        stored_w = int(target_tracking_npz_data.get("width", w))
        if stored_h != h or stored_w != w:
            raise ValueError(
                "Compact Syn4D tracking image size does not match loaded "
                f"RGB/depth grid for mask={mask_type}: shard=({stored_h}, {stored_w}) "
                f"loaded=({h}, {w}) path={target_tracking_npz_data.get('safetensor_path') or target_tracking_npz_data.get('npz_path')}"
            )
        n0 = min(valid_flat_idx.size, nearest_indices.size)
        if n0 == 0:
            continue

        valid_flat_idx = valid_flat_idx[:n0]
        nearest_indices = nearest_indices[:n0]
        in_bounds = (
            (valid_flat_idx >= 0)
            & (valid_flat_idx < h * w)
            & (nearest_indices >= 0)
            & (nearest_indices < faces.shape[0])
        )
        if not np.any(in_bounds):
            continue

        valid_flat_idx = valid_flat_idx[in_bounds]
        nearest_indices = nearest_indices[in_bounds]
        valid_y = valid_flat_idx // w
        valid_x = valid_flat_idx % w

        query_points = pts3d_target_flat[valid_flat_idx]
        projected_faces = faces[nearest_indices]
        tri_target = original_mesh_sequence[target_frame_idx, projected_faces]

        a = tri_target[:, 0]
        b = tri_target[:, 1]
        c = tri_target[:, 2]
        v0 = b - a
        v1 = c - a
        v2 = query_points - a
        d00 = np.sum(v0 * v0, axis=-1)
        d01 = np.sum(v0 * v1, axis=-1)
        d11 = np.sum(v1 * v1, axis=-1)
        d20 = np.sum(v2 * v0, axis=-1)
        d21 = np.sum(v2 * v1, axis=-1)
        denom = d00 * d11 - d01 * d01
        denom_valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)
        denom_safe = np.maximum(denom, 1e-12)

        bary_1 = (d11 * d20 - d01 * d21) / denom_safe
        bary_2 = (d00 * d21 - d01 * d20) / denom_safe
        bary_0 = 1.0 - bary_1 - bary_2
        bary_w = np.stack((bary_0, bary_1, bary_2), axis=-1).astype(np.float32, copy=False)

        face_normal_target = np.cross(b - a, c - a)
        norm_target = np.linalg.norm(face_normal_target, axis=-1, keepdims=True)
        face_normal_target = face_normal_target / np.maximum(norm_target, 1e-6)
        projected_target = np.sum(tri_target * bary_w[:, :, None], axis=1)
        signed_distance = np.sum((query_points - projected_target) * face_normal_target, axis=-1)
        orientation_sign = np.sign(signed_distance).astype(np.float32, copy=False)
        orientation_sign[orientation_sign == 0] = 1.0
        distances_flat = np.abs(signed_distance).astype(np.float32, copy=False)

        finite_valid = (
            denom_valid
            & np.isfinite(query_points).all(axis=-1)
            & np.isfinite(bary_w).all(axis=-1)
            & np.isfinite(distances_flat)
            & np.isfinite(orientation_sign)
        )
        if distance_thresh < 1e5:
            finite_valid &= distances_flat <= distance_thresh
        if not np.any(finite_valid):
            continue

        valid_y = valid_y[finite_valid]
        valid_x = valid_x[finite_valid]
        bary_w = bary_w[finite_valid]
        orientation_sign = orientation_sign[finite_valid]
        distances_flat = distances_flat[finite_valid]
        projected_faces = projected_faces[finite_valid]

        tri_source = original_mesh_sequence[source_frame_idx, projected_faces]
        face_normal_source = np.cross(
            tri_source[:, 1] - tri_source[:, 0],
            tri_source[:, 2] - tri_source[:, 0],
        )
        norm_source = np.linalg.norm(face_normal_source, axis=-1, keepdims=True)
        face_normal_source = face_normal_source / np.maximum(norm_source, 1e-6)
        projected_points_bary_source = np.sum(tri_source * bary_w[:, :, None], axis=1)
        corresponding_points_source = (
            projected_points_bary_source
            + distances_flat[:, None] * face_normal_source * orientation_sign[:, None]
        )

        pts3d_target_np[valid_y, valid_x, :] = corresponding_points_source.astype(np.float32, copy=False)
        wrapped_valid_mask[valid_y, valid_x] = True
        if not _is_background_mask_name(mask_type):
            foreground_valid_mask[valid_y, valid_x] = True

    depth_query_valid = np.asarray(depth_query_valid_map, dtype=bool)
    if depth_query_valid.ndim == 3 and depth_query_valid.shape[-1] == 1:
        depth_query_valid = depth_query_valid[:, :, 0]
    if bool(mask_bg):
        wrapped_valid_mask = foreground_valid_mask & depth_query_valid
    else:
        wrapped_valid_mask = depth_query_valid
    pts3d_target_np[~wrapped_valid_mask] = 0.0
    return pts3d_target_np, wrapped_valid_mask


def precompute_all_wrapped_pts3d(
    target_pts3d,
    target_tracking_npz_dict,
    vertices_seq_d_list,
    faces_list,
    num_frames,
    depth_query_valid_map,
    static_background_valid_map,
    wrapped_face_distance_thresh=0.1,
    track_query_idx=0,
):
    """Batch version: compute wrapped pts3d for ALL frames at once.

    Returns (all_frames_pts3d, wrapped_valid_mask) where all_frames_pts3d has
    shape (num_frames, H, W, 3) and wrapped_valid_mask has shape (H, W).
    This avoids recomputing the expensive mesh interpolation 50 times.

    Compact (new safetensor) tracking does not precompute corresponding points,
    so it is reconstructed per frame via get_wrapped_pts3d_compact_faceid_np with
    the query frame (track_query_idx) as the fixed reference grid.
    """
    if any(
        isinstance(d, dict) and d.get("format") == "compact_faceid"
        for d in target_tracking_npz_dict.values()
    ):
        if not all(
            isinstance(d, dict) and d.get("format") == "compact_faceid"
            for d in target_tracking_npz_dict.values()
        ):
            raise ValueError(
                "Mixed compact_faceid and dense Syn4D tracking schemas are not "
                "supported within one source frame."
            )
        pts3d_template = np.asarray(target_pts3d, dtype=np.float32)
        h, w, _ = pts3d_template.shape
        all_frames = np.broadcast_to(pts3d_template, (int(num_frames), h, w, 3)).copy()
        wrapped_valid_mask = None
        for t in range(int(num_frames)):
            frame_pts3d, frame_valid = get_wrapped_pts3d_compact_faceid_np(
                target_pts3d=pts3d_template,
                target_tracking_npz_dict=target_tracking_npz_dict,
                vertices_seq_d_list=vertices_seq_d_list,
                faces_list=faces_list,
                source_frame_idx=t,
                target_frame_idx=int(track_query_idx),
                depth_query_valid_map=depth_query_valid_map,
                static_background_valid_map=static_background_valid_map,
                wrapped_face_distance_thresh=wrapped_face_distance_thresh,
                mask_bg=False,
            )
            all_frames[t] = frame_pts3d
            wrapped_valid_mask = frame_valid
        if wrapped_valid_mask is None:
            depth_query_valid = np.asarray(depth_query_valid_map, dtype=bool)
            if depth_query_valid.ndim == 3 and depth_query_valid.shape[-1] == 1:
                depth_query_valid = depth_query_valid[:, :, 0]
            wrapped_valid_mask = depth_query_valid
        return all_frames, wrapped_valid_mask

    pts3d_template = torch.as_tensor(target_pts3d, dtype=torch.float32)
    h, w, _ = pts3d_template.shape
    wrapped_valid_mask = torch.zeros((h, w), dtype=torch.bool)
    static_background_valid_map = torch.as_tensor(static_background_valid_map, dtype=torch.bool)
    if static_background_valid_map.ndim == 3 and static_background_valid_map.shape[-1] == 1:
        static_background_valid_map = static_background_valid_map[:, :, 0]
    wrapped_valid_mask |= static_background_valid_map

    # Pre-allocate all-frames output: (num_frames, H, W, 3)
    all_frames = pts3d_template.unsqueeze(0).expand(num_frames, -1, -1, -1).clone()

    for mask_type, target_tracking_npz_data in target_tracking_npz_dict.items():
        masked_valid = torch.as_tensor(target_tracking_npz_data["masked_valid"], dtype=torch.bool)
        if masked_valid.ndim == 3 and masked_valid.shape[-1] == 1:
            masked_valid = masked_valid[:, :, 0]
        face_index_map_np = target_tracking_npz_data.get("face_index_map", target_tracking_npz_data.get("face_indices"))
        face_index_map = torch.as_tensor(face_index_map_np, dtype=torch.long)
        if face_index_map.ndim == 3 and face_index_map.shape[-1] == 1:
            face_index_map = face_index_map[:, :, 0]
        direct_points = target_tracking_npz_data.get("corresponding_points_all")
        if direct_points is not None:
            direct_points = torch.as_tensor(direct_points, dtype=torch.float32)
            valid_y, valid_x = torch.where(masked_valid)
            valid_mask_with_vertex = (face_index_map >= 0) & masked_valid
            if valid_mask_with_vertex.any():
                valid_y2, valid_x2 = torch.where(valid_mask_with_vertex)
                pixel_to_n = torch.full((h, w), -1, dtype=torch.long)
                pixel_to_n[valid_y, valid_x] = torch.arange(valid_y.numel(), dtype=torch.long)
                n_indices = pixel_to_n[valid_y2, valid_x2]
                T = min(num_frames, direct_points.shape[0])
                all_frames[:T, valid_y2, valid_x2, :] = direct_points[:T, n_indices, :]
                wrapped_valid_mask[valid_y2, valid_x2] = True
            continue
        if mask_type not in faces_list or mask_type not in vertices_seq_d_list:
            continue

        weights = torch.as_tensor(target_tracking_npz_data["weights"], dtype=torch.float32)
        orientation_sign = torch.as_tensor(target_tracking_npz_data["orientation_sign"], dtype=torch.float32)
        distances = torch.as_tensor(target_tracking_npz_data["distances"], dtype=torch.float32)

        faces = torch.as_tensor(faces_list[mask_type], dtype=torch.long)
        original_mesh_sequence = torch.as_tensor(vertices_seq_d_list[mask_type], dtype=torch.float32)

        nearest_indices = face_index_map[masked_valid]
        n = nearest_indices.numel()
        if n == 0:
            continue

        distances_flat = distances.reshape(-1)[:n]
        valid_y, valid_x = torch.where(masked_valid)
        if float(wrapped_face_distance_thresh) < 1e5:
            valid_distance_mask = distances_flat <= float(wrapped_face_distance_thresh)
            face_index_map = face_index_map.clone()
            if valid_y.numel() == valid_distance_mask.numel():
                invalid_mask = ~valid_distance_mask
                if invalid_mask.any():
                    face_index_map[valid_y[invalid_mask], valid_x[invalid_mask]] = -1

        projected_faces = faces[nearest_indices.flatten()]
        # corresponding_points_all: (num_source_frames, n, 3)
        projected_faces_point = original_mesh_sequence[:, projected_faces.flatten()].reshape(-1, n, 3, 3)
        face_normal_all = torch.cross(
            projected_faces_point[:, :, 1] - projected_faces_point[:, :, 0],
            projected_faces_point[:, :, 2] - projected_faces_point[:, :, 0],
            dim=-1,
        )
        face_normal_all = torch.nn.functional.normalize(face_normal_all, eps=1e-6, dim=-1)

        bary_w = weights[0].reshape(-1, 3)[:n]
        projected_points_bary_all = torch.sum(
            projected_faces_point * bary_w.unsqueeze(0).unsqueeze(-1),
            dim=2,
        )
        face_normal_all = face_normal_all * orientation_sign.reshape(-1)[:n].view(1, n, 1)
        corresponding_points_all = projected_points_bary_all + torch.abs(distances_flat).view(1, n, 1) * face_normal_all

        valid_mask_with_vertex = (face_index_map >= 0) & masked_valid
        if valid_mask_with_vertex.any():
            valid_y2, valid_x2 = torch.where(valid_mask_with_vertex)
            # Build index map: valid_y2/valid_x2 -> position in corresponding_points_all
            # valid_y, valid_x = where(masked_valid), corresponding_points_all indexed by n
            # valid_y2, valid_x2 = where(valid_mask_with_vertex) is a subset
            pixel_to_n = torch.full((h, w), -1, dtype=torch.long)
            pixel_to_n[valid_y, valid_x] = torch.arange(n, dtype=torch.long)
            n_indices = pixel_to_n[valid_y2, valid_x2]  # which n-index for each valid_y2/x2
            # Write all frames at once: all_frames[t, valid_y2, valid_x2] = corresponding_points_all[t, n_indices]
            n_valid = valid_y2.shape[0]
            T = min(num_frames, corresponding_points_all.shape[0])
            all_frames[:T, valid_y2, valid_x2, :] = corresponding_points_all[:T, n_indices, :]
            wrapped_valid_mask[valid_y2, valid_x2] = True

    depth_query_valid_map = torch.as_tensor(depth_query_valid_map, dtype=torch.bool)
    if depth_query_valid_map.ndim == 3 and depth_query_valid_map.shape[-1] == 1:
        depth_query_valid_map = depth_query_valid_map[:, :, 0]
    # All depth-valid pixels are valid: tracked pixels have vertex positions,
    # untracked pixels keep pts3d_query (initialized at line 380) → zero residual.
    # This avoids "gap" pixels (valid depth but no env/object mask) getting zeroed
    # out, which would create incorrect non-zero residual = (0 - xyz_query).
    wrapped_valid_mask = depth_query_valid_map
    all_frames[:, ~wrapped_valid_mask] = 0.0
    return all_frames.cpu().numpy(), wrapped_valid_mask.cpu().numpy()


class Syn4D_Multi(BaseMultiViewDataset):
    @staticmethod
    def _parse_valid_sequences_file(path):
        """Parse a valid_sequences.txt file into a set of (scene_name, base_seq) tuples."""
        valid_set = set()
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    valid_set.add((parts[0], parts[1]))
        if not valid_set:
            raise ValueError(f"valid_sequences_file={path!r} parsed to empty set — check the file format.")
        return valid_set

    def __init__(
        self,
        *args,
        ROOT,
        stride: int = 5,
        stride_mode: str = "all",
        trackable_scene_suffixes: tuple[str, ...] | list[str] | None = ("_bald", "_group"),
        scene_exclude: tuple[str, ...] | list[str] | None = None,
        scene_include: tuple[str, ...] | list[str] | None = None,
        return_trackvid: bool = False,
        trackvid_fill_value: float = -1.0,
        invalid_depth_max: float = 300.0,
        object_distance_std_thresh: float = 999999.0,
        non_object_distance_thresh: float = 999999.0,
        query_inconsistent_distance_thresh: float = 99999.0,
        wrapped_face_distance_thresh: float = 99999.0,
        valid_sequences_file: str | None = None,
        enable_sequence_disk_cache: bool | None = None,
        sequence_disk_cache_dir: str | None = None,
        sequence_mem_cache_size: int = 2,
        tracking_format: str = "auto",
        metadata_root: str | None = None,
        **kwargs,
    ):
        self.ROOT = ROOT
        if not osp.isdir(self.ROOT):
            raise FileNotFoundError(
                f"Syn4D root does not exist: {self.ROOT!r}. "
                "Set SYN4D_ROOT or pass the correct ROOT path."
            )
        # tracking_format selects the on-disk 3D-track schema:
        #   "auto"          -> dense vertex_indices_*.npz if present, else compact
        #                      tracking_safetensors/ shard (new format).
        #   "dense"         -> require old dense npz only.
        #   "compact_faceid"-> require compact (valid_flat_idx + face_ids).
        #   "safetensor"    -> require the safetensor shard only (skip npz).
        if tracking_format not in ("auto", "dense", "compact_faceid", "safetensor"):
            raise ValueError(
                f"Unsupported tracking_format={tracking_format!r}. "
                "Use 'auto', 'dense', 'compact_faceid', or 'safetensor'."
            )
        self.tracking_format = tracking_format
        # The new safetensor archives do not ship the shared mesh metadata
        # (bedlam2_smpl_npz / new_weight_bone); allow pointing metadata_root at an
        # existing Syn4D tree (e.g. via SYN4D_METADATA_ROOT) while RGB/depth/tracking
        # come from the new root.
        if metadata_root is None:
            metadata_root = os.environ.get("SYN4D_METADATA_ROOT")
        self.metadata_root = str(metadata_root) if metadata_root else osp.join(self.ROOT, "metadata")
        self.root_smpl_npz_path = osp.join(self.metadata_root, "bedlam2_smpl_npz")
        self.root_obj_npz_path = osp.join(self.metadata_root, "new_weight_bone")
        self.cloth_tar_root = osp.join(self.metadata_root, "b2_assetdata_download", "clothing", "npz")
        self.stride_choices = self._parse_stride_choices(stride)
        self.stride = int(self.stride_choices[0])
        self.stride_mode = self._parse_stride_mode(stride_mode)
        self.cache_root = self._resolve_cache_root()
        if trackable_scene_suffixes is None:
            self.trackable_scene_suffixes = tuple()
        else:
            self.trackable_scene_suffixes = tuple(str(x) for x in trackable_scene_suffixes if str(x))
        self.scene_exclude = tuple(str(x) for x in scene_exclude) if scene_exclude else ()
        self.scene_include = tuple(str(x) for x in scene_include) if scene_include else ()
        self.valid_sequences_file = str(valid_sequences_file) if valid_sequences_file else None
        self._valid_seq_set = (
            self._parse_valid_sequences_file(self.valid_sequences_file)
            if self.valid_sequences_file
            else None
        )
        self._valid_scene_names = (
            sorted({scene for scene, _ in self._valid_seq_set})
            if self._valid_seq_set is not None
            else None
        )
        self.return_trackvid = bool(return_trackvid)
        self.trackvid_fill_value = float(trackvid_fill_value)
        self.invalid_depth_max = float(invalid_depth_max)
        self.object_distance_std_thresh = float(object_distance_std_thresh)
        self.non_object_distance_thresh = float(non_object_distance_thresh)
        self.query_inconsistent_distance_thresh = float(query_inconsistent_distance_thresh)
        self.wrapped_face_distance_thresh = float(wrapped_face_distance_thresh)
        if enable_sequence_disk_cache is None:
            enable_sequence_disk_cache = _env_flag("SYN4D_ENABLE_SEQUENCE_DISK_CACHE", False)
        self.enable_sequence_disk_cache = bool(enable_sequence_disk_cache)
        if sequence_disk_cache_dir is None:
            sequence_disk_cache_dir = os.environ.get("SYN4D_SEQUENCE_DISK_CACHE_DIR")
        env_mem_cache_size = os.environ.get("SYN4D_SEQUENCE_MEM_CACHE_SIZE")
        if env_mem_cache_size is not None:
            sequence_mem_cache_size = int(env_mem_cache_size)
        self.sequence_mem_cache_size = max(0, int(sequence_mem_cache_size))
        self.video = True
        self.is_metric = True
        self.dataset_label = "Syn4D"
        self._blacklisted_sample_indices = set()
        self._camera_csv_cache = {}
        self._npz_cache = OrderedDict()
        self._npz_cache_size = max(0, int(os.environ.get("SYN4D_NPZ_CACHE_SIZE", "256")))
        # Per-(scene, camera_seq) open safetensor handles for the new format.
        self._tracking_safetensor_cache = {}
        self._env_mask_safetensor_cache = {}
        self._cloth_npz_cache = OrderedDict()
        self._cloth_npz_cache_size = max(0, int(os.environ.get("SYN4D_CLOTH_NPZ_CACHE_SIZE", "16")))
        self._sequence_vertices_cache = OrderedDict()
        self.sequence_disk_cache_root = self._resolve_sequence_cache_root(sequence_disk_cache_dir) if self.enable_sequence_disk_cache else None

        cloth_json_path = osp.join(self.cloth_tar_root, "archive_map.json")
        self.cloth_npz_to_tar = {}
        if osp.isfile(cloth_json_path):
            with open(cloth_json_path, "r") as f:
                cloth_json = json.load(f)
            for tar_name, npz_entries in cloth_json.items():
                for entry in npz_entries:
                    self.cloth_npz_to_tar[entry] = tar_name

        super().__init__(*args, **kwargs)
        self.seq_info_by_scene = {}
        self.sequence_scales_by_scene = {}
        self.samples = []
        self._build_index()

    def __len__(self):
        return len(self.samples)

    def get_stats(self):
        if self.stride_mode == "max_available":
            return f"{len(self)} fixed-start clips with max-available stride from {list(self.stride_choices)}"
        if len(self.stride_choices) == 1:
            return f"{len(self)} fixed-start clips"
        return f"{len(self)} fixed-start clips across stride choices {list(self.stride_choices)}"

    @staticmethod
    def _parse_stride_choices(stride) -> tuple[int, ...]:
        if isinstance(stride, str):
            tokens = [token.strip() for token in stride.split(",") if token.strip()]
            values = [int(token) for token in tokens] if tokens else [5]
        elif isinstance(stride, (list, tuple)):
            values = [int(token) for token in stride]
        else:
            values = [int(stride)]
        values = sorted(set(v for v in values if v > 0))
        if not values:
            raise ValueError(f"Invalid Syn4D stride spec: {stride!r}")
        return tuple(values)

    @staticmethod
    def _parse_stride_mode(stride_mode: str) -> str:
        mode = str(stride_mode).strip().lower()
        if mode in ("all", "max_available"):
            return mode
        raise ValueError(f"Invalid Syn4D stride_mode: {stride_mode!r}")

    def _resolve_cache_root(self) -> Path:
        repo_root = Path(__file__).resolve().parents[5]
        cache_root = repo_root / "outputs" / "cache" / "syn4d_index"
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _resolve_sequence_cache_root(self, override_path: str | None) -> Path:
        if override_path:
            cache_root = Path(override_path).expanduser()
        else:
            repo_root = Path(__file__).resolve().parents[5]
            cache_root = repo_root / "outputs" / "cache" / "syn4d_sequence_vertices"
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _index_cache_path(self) -> Path:
        cache_key = {
            "version": int(SYN4D_INDEX_CACHE_VERSION),
            "root": str(Path(self.ROOT).resolve()),
            "num_views": int(self.num_views),
            "stride_choices": [int(v) for v in self.stride_choices],
            "stride_mode": str(self.stride_mode),
            "trackable_scene_suffixes": list(self.trackable_scene_suffixes),
            "scene_exclude": sorted(self.scene_exclude),
            "scene_include": sorted(self.scene_include),
            "valid_sequences_file": str(self.valid_sequences_file) if self.valid_sequences_file else None,
            "valid_sequences": sorted(list(self._valid_seq_set)) if self._valid_seq_set is not None else None,
        }
        payload = json.dumps(cache_key, sort_keys=True)
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return self.cache_root / f"syn4d_index_{digest}.pkl"

    def _load_cached_index(self) -> bool:
        cache_path = self._index_cache_path()
        if not cache_path.exists():
            return False
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            self.seq_info_by_scene = payload["seq_info_by_scene"]
            self.sequence_scales_by_scene = payload["sequence_scales_by_scene"]
            self.samples = payload["samples"]
            print(f"[Syn4D_Multi] loaded index cache: {cache_path} ({len(self.samples)} clips)")
            return True
        except Exception as exc:
            print(f"[Syn4D_Multi] failed to load index cache {cache_path}: {exc}")
            return False

    def _save_cached_index(self):
        cache_path = self._index_cache_path()
        payload = {
            "seq_info_by_scene": self.seq_info_by_scene,
            "sequence_scales_by_scene": self.sequence_scales_by_scene,
            "samples": self.samples,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
        try:
            with tmp_path.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            print(f"[Syn4D_Multi] saved index cache: {cache_path} ({len(self.samples)} clips)")
        except OSError as exc:
            # Race with another rank — if the final cache exists, that's fine
            if cache_path.exists():
                print(f"[Syn4D_Multi] index cache already written by another rank: {cache_path}")
            else:
                raise exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _base_sequence_name(camera_seq_name: str) -> str:
        parts = str(camera_seq_name).split("_")
        if parts and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return str(camera_seq_name)

    def _sequence_cache_key(self, scene_name: str, sequence_name: str):
        return (str(scene_name), str(sequence_name))

    def _sequence_cache_path(self, scene_name: str, sequence_name: str) -> Path:
        assert self.sequence_disk_cache_root is not None
        cache_key = {
            "version": int(SYN4D_SEQUENCE_CACHE_VERSION),
            "root": str(Path(self.ROOT).resolve()),
            "scene_name": str(scene_name),
            "sequence_name": str(sequence_name),
        }
        payload = json.dumps(cache_key, sort_keys=True)
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        safe_scene = _safe_cache_name(scene_name)
        safe_seq = _safe_cache_name(sequence_name)
        return self.sequence_disk_cache_root / f"{safe_scene}__{safe_seq}__{digest}.pkl"

    def _sequence_mem_cache_get(self, key):
        cached = self._sequence_vertices_cache.get(key)
        if cached is None:
            return None
        self._sequence_vertices_cache.move_to_end(key)
        return cached

    def _sequence_mem_cache_put(self, key, value):
        if self.sequence_mem_cache_size <= 0:
            return
        self._sequence_vertices_cache[key] = value
        self._sequence_vertices_cache.move_to_end(key)
        while len(self._sequence_vertices_cache) > self.sequence_mem_cache_size:
            self._sequence_vertices_cache.popitem(last=False)

    def _load_sequence_vertices_from_disk_cache(self, scene_name: str, sequence_name: str):
        if not self.enable_sequence_disk_cache or self.sequence_disk_cache_root is None:
            return None
        cache_path = self._sequence_cache_path(scene_name, sequence_name)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("rb") as f:
                return pickle.load(f)
        except Exception as exc:
            print(f"[Syn4D_Multi] failed to load sequence cache {cache_path}: {exc}")
            return None

    def _save_sequence_vertices_to_disk_cache(self, scene_name: str, sequence_name: str, payload):
        if not self.enable_sequence_disk_cache or self.sequence_disk_cache_root is None:
            return
        cache_path = self._sequence_cache_path(scene_name, sequence_name)
        if cache_path.exists():
            return
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)

    def _cloth_cache_get(self, key):
        cached = self._cloth_npz_cache.get(key)
        if cached is None:
            return None
        self._cloth_npz_cache.move_to_end(key)
        return cached

    def _cloth_cache_put(self, key, value):
        if self._cloth_npz_cache_size <= 0:
            return
        self._cloth_npz_cache[key] = value
        self._cloth_npz_cache.move_to_end(key)
        while len(self._cloth_npz_cache) > self._cloth_npz_cache_size:
            self._cloth_npz_cache.popitem(last=False)

    def _resolve_sequence_name(self, scene_name: str, camera_seq_name: str):
        seq_info_map = self.seq_info_by_scene.get(scene_name, {})
        candidates = []
        camera_seq_name = str(camera_seq_name)
        base_sequence_name = self._base_sequence_name(camera_seq_name)
        for candidate in (camera_seq_name, base_sequence_name):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates:
            if candidate in seq_info_map:
                return candidate
        return None

    def _build_index(self):
        if self._load_cached_index():
            return

        filter_info = ""
        if self.scene_include:
            filter_info += f", scene_include={list(self.scene_include)}"
        if self.scene_exclude:
            filter_info += f", scene_exclude={list(self.scene_exclude)}"
        if self.valid_sequences_file:
            filter_info += f", valid_sequences_file={self.valid_sequences_file}"
        print(
            "[Syn4D_Multi] building index "
            f"(root={self.ROOT}, num_views={self.num_views}, strides={list(self.stride_choices)}{filter_info})"
        )
        scenes = []
        for scene_name in sorted(os.listdir(self.ROOT)):
            scene_dir = osp.join(self.ROOT, scene_name)
            if not osp.isdir(scene_dir) or scene_name == "metadata":
                continue
            if self._valid_scene_names is not None and scene_name not in self._valid_scene_names:
                continue
            # When scene_include is set, it takes priority over suffix-based filtering
            if self.scene_include:
                if scene_name not in self.scene_include:
                    continue
            else:
                if self.trackable_scene_suffixes and not any(scene_name.endswith(suffix) for suffix in self.trackable_scene_suffixes):
                    continue
            if self.scene_exclude and scene_name in self.scene_exclude:
                continue
            if any(scene_name.startswith(x) for x in hdri_scenes):
                continue
            if "closeup" in scene_name or "orbit_stadium" in scene_name or "pitchUp52" in scene_name:
                continue
            be_seq_path = osp.join(scene_dir, "be_seq.csv")
            transform_json_path = osp.join(scene_dir, "be_seq_transform_channels.json")
            png_root = osp.join(scene_dir, "png")
            mp4_root = osp.join(scene_dir, "mp4")
            has_png = osp.isdir(png_root)
            has_mp4 = osp.isdir(mp4_root)
            # New format ships mp4/ only (no png/); accept either as the RGB source.
            if not (osp.isfile(be_seq_path) and osp.isfile(transform_json_path) and (has_png or has_mp4)):
                continue
            scenes.append((scene_name, scene_dir, be_seq_path, transform_json_path, png_root, mp4_root, has_png))

        n_scenes = len(scenes)
        t_start = time.monotonic()
        print(f"[Syn4D_Multi] scanning {n_scenes} scenes ...", flush=True)
        for scene_idx, (scene_name, scene_dir, be_seq_path, transform_json_path, png_root, mp4_root, has_png) in enumerate(scenes, 1):
            scene_t0 = time.monotonic()
            scene_samples_before = len(self.samples)
            seq_info_list = read_seq_csv(be_seq_path)
            self.seq_info_by_scene[scene_name] = {
                item["sequence_name"]: item for item in seq_info_list if item.get("sequence_name")
            }
            with open(transform_json_path, "r") as f:
                transformation_json = json.load(f)

            scale_map = {}
            for item in transformation_json:
                sequence_id = str(item.get("sequence_id", ""))
                sequence_body_index = item.get("sequence_body_index", -1)
                transform_channels = item.get("transform_channels", {})
                global_translation = np.array(
                    [
                        float(item.get("map_camera_target_actor_location_x", 0.0)),
                        float(item.get("map_camera_target_actor_location_y", 0.0)),
                        float(item.get("map_camera_target_actor_location_z", 0.0)),
                    ],
                    dtype=np.float32,
                ) / 100.0
                scale_x = float(transform_channels.get("scale_x", 1.0))
                scale_map.setdefault(sequence_id, []).append((sequence_body_index, scale_x, global_translation))
            for sequence_id, scale_items in scale_map.items():
                scale_items.sort(key=lambda x: x[0] if isinstance(x[0], int) else 10**9)
                scale_map[sequence_id] = [(scale, translation) for _, scale, translation in scale_items]
            self.sequence_scales_by_scene[scene_name] = scale_map

            # Enumerate camera sequences from png/ (old) or mp4/ (new format).
            if has_png:
                camera_seq_entries = [
                    (name, osp.join(png_root, name))
                    for name in sorted(os.listdir(png_root))
                    if osp.isdir(osp.join(png_root, name))
                ]
            else:
                camera_seq_entries = [
                    (osp.splitext(name)[0], mp4_root)
                    for name in sorted(os.listdir(mp4_root))
                    if name.endswith(".mp4")
                ]

            for camera_seq_name, view_dir in camera_seq_entries:
                sequence_name_base = self._base_sequence_name(camera_seq_name)
                if self._valid_seq_set is not None and (scene_name, sequence_name_base) not in self._valid_seq_set:
                    continue
                invalid_key = f"{scene_name}_{sequence_name_base}"
                if invalid_key in invalid_seqs:
                    continue
                sequence_name = self._resolve_sequence_name(scene_name, camera_seq_name)
                if sequence_name is None:
                    continue

                camera_csv_path = osp.join(scene_dir, "ground_truth", "meta_exr_csv", f"{camera_seq_name}_camera.csv")
                if not osp.isfile(camera_csv_path):
                    continue

                # Frame names: png listing (old) or CSV-derived (new mp4-only).
                if has_png:
                    frame_names_all = sorted([x for x in os.listdir(view_dir) if x.endswith(".png")])
                else:
                    frame_names_all = self._list_frame_names_from_camera_csv(camera_csv_path)
                if len(frame_names_all) == 0:
                    continue

                mask_root = osp.join(scene_dir, "exr_layers", "masks", camera_seq_name)
                mask_types = self._discover_mask_types(mask_root, frame_names_all[0]) if osp.isdir(mask_root) else []

                # The set of trackable frames is exactly listdir(<scene>/npz/<cam_seq>/)
                # — one readdir gives us every frame_stem that has a tracking subdir,
                # instead of stat'ing each PNG frame individually. On Lustre this cuts
                # the tracking-filter phase from hundreds of thousands of round-trips
                # to ~O(num_cameras) (dataset invariant: every npz stem subdir is
                # non-empty and contains vertex_indices_*/tracking_*.npz).
                npz_cam_dir = osp.join(self.ROOT, scene_name, "npz", camera_seq_name)
                try:
                    with os.scandir(npz_cam_dir) as it:
                        trackable_stems = {
                            e.name for e in it if e.is_dir(follow_symlinks=False)
                        }
                except OSError:
                    trackable_stems = set()
                # New format may not ship the npz/ stem dirs; fall back to the set
                # of frame stems present in the tracking safetensor index.
                if not trackable_stems and self.tracking_format in ("auto", "safetensor", "compact_faceid"):
                    cached = self._get_tracking_safetensor(scene_name, camera_seq_name)
                    if cached is not None:
                        record_key_to_id = cached["index"].get("record_key_to_id", {})
                        trackable_stems = {
                            key.split("/", 1)[0] for key in record_key_to_id.keys() if "/" in key
                        }
                if not trackable_stems:
                    continue
                trackable_start_indices = [
                    i for i, fn in enumerate(frame_names_all)
                    if osp.splitext(fn)[0] in trackable_stems
                ]
                if not trackable_start_indices:
                    continue

                total_frames = len(frame_names_all)
                for start_idx in trackable_start_indices:
                    remaining_frames = total_frames - start_idx
                    max_stride_for_start = (remaining_frames - 1) // max(int(self.num_views) - 1, 1)
                    if max_stride_for_start <= 0:
                        continue
                    valid_strides = [int(v) for v in self.stride_choices if int(v) <= max_stride_for_start]
                    if not valid_strides:
                        continue
                    if self.stride_mode == "max_available":
                        stride_values = [max(valid_strides)]
                    else:
                        stride_values = valid_strides
                    for stride_value in stride_values:
                        end_idx = start_idx + int(stride_value) * (int(self.num_views) - 1) + 1
                        if end_idx > total_frames:
                            continue
                        self.samples.append(
                            {
                                "scene_name": scene_name,
                                "camera_seq_name": camera_seq_name,
                                "sequence_name": sequence_name,
                                "view_dir": view_dir,
                                "camera_csv_path": camera_csv_path,
                                "mask_root": mask_root,
                                "mask_types": mask_types,
                                "frame_names": frame_names_all[start_idx:end_idx:int(stride_value)],
                                "frame_indices_abs": list(range(start_idx, end_idx, int(stride_value))),
                                "fps": max(1, 24 // int(stride_value)),
                                "stride": int(stride_value),
                            }
                        )

            scene_elapsed = time.monotonic() - scene_t0
            scene_samples = len(self.samples) - scene_samples_before
            total_elapsed = time.monotonic() - t_start
            print(
                f"[Syn4D_Multi] [{scene_idx}/{n_scenes}] {scene_name:<24} "
                f"+{scene_samples:>5} clips  "
                f"(scene {scene_elapsed:5.1f}s | total {total_elapsed:6.1f}s | "
                f"running {len(self.samples)} clips)",
                flush=True,
            )

        if not self.samples:
            raise ValueError(
                f"Syn4D_Multi found no valid {self.num_views}-frame clips under ROOT={self.ROOT!r}. "
                f"Current trackable_scene_suffixes={self.trackable_scene_suffixes!r}. "
                "Check the dataset root, supported scene suffixes, and/or reduce num_views."
            )
        total_elapsed = time.monotonic() - t_start
        print(
            f"[Syn4D_Multi] scan complete: {len(self.samples)} clips across {n_scenes} scenes "
            f"in {total_elapsed:.1f}s",
            flush=True,
        )
        self._save_cached_index()

    def _discover_mask_types(self, mask_root: str, first_frame_name: str) -> list[str]:
        prefix = osp.splitext(first_frame_name)[0]
        mask_types = []
        for filename in sorted(os.listdir(mask_root)):
            if filename.startswith(prefix) and filename.endswith(".png"):
                suffix = filename[len(prefix) : -4]
                if suffix:
                    mask_types.append(suffix)
        return mask_types

    @staticmethod
    def _list_frame_names_from_camera_csv(camera_csv_path: str) -> list[str]:
        """Derive frame names from the camera CSV 'name' column.

        Used by the new mp4-only format where there is no png/ dir to list.
        The CSV 'name' values keep the nominal .png suffix (e.g.
        'seq_000000_1_0000.png'); the stem matches the safetensor record keys.
        """
        frame_names = []
        with open(camera_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name")
                if not name:
                    continue
                basename = osp.basename(str(name))
                if basename.endswith(".png"):
                    frame_names.append(basename)
        return sorted(frame_names)

    def _build_cloth_npz_rel_path(self, subject: str | None, animation_id: str | None):
        if not subject:
            return None
        subject_str = str(subject)
        anim_str = None
        if animation_id and animation_id != "undefined":
            anim_str = str(animation_id)
        else:
            parts = subject_str.split("_")
            if len(parts) >= 3 and parts[-1].isdigit():
                anim_str = parts[-1]
                subject_str = "_".join(parts[:-1])
        if not anim_str:
            return None
        return f"{subject_str}/{anim_str}/{anim_str}.npz"

    def _find_cloth_member(self, tar_file, npz_rel_path: str, tar_name: str):
        for candidate in (npz_rel_path, f"{tar_name}/{npz_rel_path}", f"{tar_name}.tar/{npz_rel_path}"):
            try:
                return tar_file.getmember(candidate)
            except KeyError:
                continue
        for member in tar_file.getmembers():
            if member.name.endswith(npz_rel_path):
                return member
        return None

    def _load_cloth_npz(self, subject: str | None, animation_id: str | None):
        npz_rel_path = self._build_cloth_npz_rel_path(subject, animation_id)
        if not npz_rel_path:
            return None
        cached = self._cloth_cache_get(npz_rel_path)
        if cached is not None:
            return cached
        tar_name = self.cloth_npz_to_tar.get(npz_rel_path)
        if not tar_name:
            return None
        tar_path = osp.join(self.cloth_tar_root, f"{tar_name}.tar")
        if not osp.exists(tar_path):
            return None
        try:
            with tarfile.open(tar_path, "r") as tar_file:
                member = self._find_cloth_member(tar_file, npz_rel_path, tar_name)
                if member is None:
                    return None
                extracted = tar_file.extractfile(member)
                if extracted is None:
                    return None
                with extracted:
                    npz_data = np.load(extracted, allow_pickle=True)
                    out = {key: npz_data[key] for key in npz_data.files}
                    self._cloth_cache_put(npz_rel_path, out)
                    return out
        except Exception:
            return None

    def _load_npz_as_dict(self, npz_path: str):
        cached = self._npz_cache.get(npz_path)
        if cached is not None:
            self._npz_cache.move_to_end(npz_path)
            return cached
        with np.load(npz_path, allow_pickle=True) as data:
            out = {k: data[k] for k in data.files}
        self._npz_cache[npz_path] = out
        self._npz_cache.move_to_end(npz_path)
        while len(self._npz_cache) > self._npz_cache_size:
            self._npz_cache.popitem(last=False)
        return out

    # ---- new-format (compact_faceid) tracking + env-mask safetensor I/O -----

    @staticmethod
    def _slice_safetensor_tensor(handle, name: str, start: int, end: int):
        start = int(start)
        end = int(end)
        try:
            return np.asarray(handle.get_slice(name)[start:end])
        except Exception:
            return np.asarray(handle.get_tensor(name)[start:end])

    def _tracking_safetensor_paths(self, scene_name: str, camera_seq_name: str) -> tuple[str, str]:
        root = osp.join(self.ROOT, scene_name, "tracking_safetensors")
        return (
            osp.join(root, f"{camera_seq_name}.safetensors"),
            osp.join(root, f"{camera_seq_name}_index.json"),
        )

    def _get_tracking_safetensor(self, scene_name: str, camera_seq_name: str):
        key = (str(scene_name), str(camera_seq_name))
        if key in self._tracking_safetensor_cache:
            return self._tracking_safetensor_cache[key]
        shard_path, index_path = self._tracking_safetensor_paths(scene_name, camera_seq_name)
        if not (osp.isfile(shard_path) and osp.isfile(index_path)):
            self._tracking_safetensor_cache[key] = None
            return None
        if safe_open is None:
            raise ImportError(
                "Syn4D tracking safetensor shard exists but the safetensors package "
                f"is not installed: {shard_path}"
            )
        with open(index_path, "r") as f:
            index = json.load(f)
        handle = safe_open(shard_path, framework="np")
        cached = {
            "handle": handle,
            "index": index,
            "offsets": np.asarray(handle.get_tensor("offsets"), dtype=np.int64),
            "record_hw": np.asarray(handle.get_tensor("record_hw"), dtype=np.int32),
            "shard_path": shard_path,
        }
        self._tracking_safetensor_cache[key] = cached
        return cached

    def _env_mask_safetensor_paths(self, scene_name: str, camera_seq_name: str) -> tuple[str, str]:
        root = osp.join(self.ROOT, scene_name, "env_mask_safetensors")
        return (
            osp.join(root, f"{camera_seq_name}.safetensors"),
            osp.join(root, f"{camera_seq_name}_index.json"),
        )

    def _get_env_mask_safetensor(self, scene_name: str, camera_seq_name: str):
        key = (str(scene_name), str(camera_seq_name))
        if key in self._env_mask_safetensor_cache:
            return self._env_mask_safetensor_cache[key]
        shard_path, index_path = self._env_mask_safetensor_paths(scene_name, camera_seq_name)
        if not (osp.isfile(shard_path) and osp.isfile(index_path)):
            self._env_mask_safetensor_cache[key] = None
            return None
        if safe_open is None:
            raise ImportError(
                "Syn4D env-mask safetensor shard exists but the safetensors package "
                f"is not installed: {shard_path}"
            )
        with open(index_path, "r") as f:
            index = json.load(f)
        handle = safe_open(shard_path, framework="np")
        cached = {
            "handle": handle,
            "index": index,
            "offsets": np.asarray(handle.get_tensor("offsets"), dtype=np.int64),
            "shard_path": shard_path,
        }
        self._env_mask_safetensor_cache[key] = cached
        return cached

    def _env_mask_safetensor_index_exists(self, scene_name: str, camera_seq_name: str) -> bool:
        shard_path, index_path = self._env_mask_safetensor_paths(scene_name, camera_seq_name)
        return osp.isfile(shard_path) and osp.isfile(index_path)

    def _safetensor_tracking_mask_names(self, scene_name: str, camera_seq_name: str, frame_stem: str) -> set[str]:
        """Mask names that have a compact tracking record for this frame."""
        cached = self._get_tracking_safetensor(scene_name, camera_seq_name)
        if cached is None:
            return set()
        prefix = f"{frame_stem}/"
        record_key_to_id = cached["index"].get("record_key_to_id", {})
        return {
            key[len(prefix):]
            for key in record_key_to_id.keys()
            if key.startswith(prefix) and len(key) > len(prefix)
        }

    def _load_vertex_indices_safetensor(self, scene_name: str, camera_seq_name: str, frame_stem: str, mask_name: str):
        cached = self._get_tracking_safetensor(scene_name, camera_seq_name)
        if cached is None:
            return None
        record_key = f"{frame_stem}/{mask_name}"
        record_key_to_id = cached["index"].get("record_key_to_id", {})
        if record_key not in record_key_to_id:
            return None
        record_id = int(record_key_to_id[record_key])
        offsets = cached["offsets"]
        if record_id < 0 or record_id + 1 >= offsets.shape[0]:
            raise IndexError(
                f"Invalid Syn4D tracking safetensor record id={record_id} "
                f"for {scene_name}/{camera_seq_name}/{record_key}"
            )
        start = int(offsets[record_id])
        end = int(offsets[record_id + 1])
        hw = cached["record_hw"][record_id]
        handle = cached["handle"]
        return {
            "format": "compact_faceid",
            "height": int(hw[0]),
            "width": int(hw[1]),
            "valid_flat_idx": self._slice_safetensor_tensor(
                handle, "valid_flat_idx_values", start, end
            ).astype(np.int64, copy=False).reshape(-1),
            "face_ids": self._slice_safetensor_tensor(
                handle, "face_ids_values", start, end
            ).astype(np.int64, copy=False).reshape(-1),
            "safetensor_path": cached["shard_path"],
            "record_key": record_key,
        }

    def _load_env_mask_safetensor(self, scene_name: str, camera_seq_name: str, frame_stem: str):
        cached = self._get_env_mask_safetensor(scene_name, camera_seq_name)
        if cached is None:
            return None
        frame_to_id = cached["index"].get("frame_to_id", {})
        if frame_stem not in frame_to_id:
            return None
        record_id = int(frame_to_id[frame_stem])
        offsets = cached["offsets"]
        if record_id < 0 or record_id + 1 >= offsets.shape[0]:
            raise IndexError(
                f"Invalid Syn4D env-mask safetensor record id={record_id} "
                f"for {scene_name}/{camera_seq_name}/{frame_stem}"
            )
        start = int(offsets[record_id])
        end = int(offsets[record_id + 1])
        png_bytes = self._slice_safetensor_tensor(
            cached["handle"], "png_bytes_values", start, end
        ).astype(np.uint8, copy=False).reshape(-1)
        if cv2 is None:
            raise ImportError("cv2 is required to decode Syn4D env-mask PNG bytes from safetensor")
        env_mask = cv2.imdecode(png_bytes, cv2.IMREAD_UNCHANGED)
        if env_mask is None:
            raise IOError(
                "Failed to decode Syn4D env-mask PNG bytes from safetensor: "
                f"{cached['shard_path']} record={frame_stem}"
            )
        return env_mask

    def _load_tracking_dict(self, scene_name: str, camera_seq_name: str, frame_stem: str, mask_name: str):
        """Return the per-mask tracking dict (dense npz or compact safetensor).

        Honours self.tracking_format: dense npz is tried first (unless
        tracking_format == 'safetensor'), then the compact safetensor shard.
        Returns None if neither source has a record for this (frame, mask).
        """
        track_dir = osp.join(self.ROOT, scene_name, "npz", camera_seq_name, frame_stem)
        if self.tracking_format != "safetensor":
            npz_path = self._resolve_tracking_npz_path(track_dir, mask_name)
            if npz_path is not None:
                return self._load_npz_as_dict(npz_path)
            if self.tracking_format == "dense":
                return None
        if self.tracking_format in ("auto", "safetensor", "compact_faceid"):
            return self._load_vertex_indices_safetensor(scene_name, camera_seq_name, frame_stem, mask_name)
        return None

    def _load_camera_rows(self, csv_path: str) -> dict[str, dict[str, str]]:
        cached = self._camera_csv_cache.get(csv_path)
        if cached is not None:
            return cached
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        mapping = {str(row["name"]): row for row in rows if row.get("name")}
        self._camera_csv_cache[csv_path] = mapping
        return mapping

    def euler_to_rotation_matrix(self, yaw, pitch, roll):
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)
        r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        r_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        r_roll = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        zxy_xyz = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        return r_yaw @ r_pitch @ r_roll @ zxy_xyz

    def euler_to_rotation_matrix_obj_v1(self, yaw, pitch, roll):
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)
        r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        r_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        r_roll = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        zxy_xyz = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        return r_yaw @ r_pitch @ r_roll @ zxy_xyz

    def euler_to_rotation_matrix_obj_objverse(self, yaw, pitch, roll):
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)
        r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        r_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        r_roll = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        return r_yaw @ r_pitch @ r_roll

    def _world_to_camera(self, points_world, camera_pose):
        world_to_cam = np.linalg.inv(camera_pose)
        return points_world @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]

    def _depth_filtering_enabled(self) -> bool:
        return (
            self.object_distance_std_thresh < 1e5
            or self.non_object_distance_thresh < 1e5
        )

    def _query_consistency_filter_enabled(self) -> bool:
        return self.query_inconsistent_distance_thresh < 1e5

    def _resolve_sequence_info(self, scene_name: str, sample: dict):
        seq_info_map = self.seq_info_by_scene[scene_name]
        resolved_sequence_name = sample.get("sequence_name")
        if resolved_sequence_name not in seq_info_map:
            resolved_sequence_name = self._resolve_sequence_name(scene_name, sample["camera_seq_name"])
        if resolved_sequence_name is None or resolved_sequence_name not in seq_info_map:
            available_examples = sorted(seq_info_map.keys())[:8]
            raise KeyError(
                f"Failed to resolve Syn4D sequence name for scene={scene_name!r}, "
                f"camera_seq_name={sample['camera_seq_name']!r}, stored_sequence_name={sample.get('sequence_name')!r}. "
                f"Example available keys: {available_examples}"
            )
        return resolved_sequence_name, seq_info_map[resolved_sequence_name]

    def _build_selected_vertices_seq(
        self,
        scene_name: str,
        resolved_sequence_name: str,
        seq_info: dict,
        frame_indices_abs: np.ndarray,
        required_mask_names: set[str] | None = None,
    ):
        sequence_frames = int(seq_info["sequence_frames"])
        sequence_scale_map = self.sequence_scales_by_scene.get(scene_name, {})
        sequence_scales = sequence_scale_map.get(resolved_sequence_name, [])
        if not sequence_scales:
            prefix = f"{resolved_sequence_name}_"
            matched_ids = sorted([k for k in sequence_scale_map.keys() if k.startswith(prefix)])
            for matched_id in matched_ids:
                sequence_scales.extend(sequence_scale_map[matched_id])

        frame_indices_abs = np.asarray(frame_indices_abs, dtype=np.int64)
        if frame_indices_abs.size == 0:
            raise ValueError("Syn4D frame_indices_abs is empty")
        if np.any(frame_indices_abs < 0) or np.any(frame_indices_abs >= sequence_frames):
            raise IndexError(
                f"Syn4D frame_indices_abs out of range for scene={scene_name!r}, "
                f"sequence={resolved_sequence_name!r}: min={int(frame_indices_abs.min())}, "
                f"max={int(frame_indices_abs.max())}, sequence_frames={sequence_frames}"
            )

        vertices_seq_d_selected = {}
        faces_list = {}

        if required_mask_names is not None:
            required_mask_names = {str(name) for name in required_mask_names}

        bodies = seq_info.get("bodies", [])
        if bodies:
            body_info = bodies[0]
            if required_mask_names is None or "_00_clothing" in required_mask_names:
                cloth_npz = self._load_cloth_npz(body_info.get("subject"), body_info.get("animation_id"))
            else:
                cloth_npz = None
            if cloth_npz is not None:
                vertices_seq = cloth_npz["vertices_seq"][1:-2]
                if len(vertices_seq) == sequence_frames:
                    vertices_seq = np.asarray(vertices_seq[frame_indices_abs], dtype=np.float32)
                    body_pose = body_info["pose"]
                    body_pose_matrix = self.euler_to_rotation_matrix_obj_v1(body_pose.yaw, body_pose.pitch, body_pose.roll)
                    body_translation = np.array([body_pose.x, body_pose.y, body_pose.z], dtype=np.float32) / 100.0
                    if sequence_scales:
                        _, global_translation = sequence_scales[0]
                    else:
                        global_translation = np.zeros(3, dtype=np.float32)
                    body_translation = body_translation + global_translation
                    vertices_seq_d = vertices_seq @ body_pose_matrix.T + body_translation
                    vertices_seq_d_selected["_00_clothing"] = vertices_seq_d.astype(np.float32)
                    faces_list["_00_clothing"] = cloth_npz["faces"]

            body_npz_path = osp.join(
                self.root_smpl_npz_path,
                body_info.get("subject"),
                f"{body_info.get('subject')}_{body_info.get('animation_id')}.npz",
            )
            if (required_mask_names is None or "_00_body" in required_mask_names) and osp.exists(body_npz_path):
                # Try mmap to read only needed frames (body NPZs up to 239MB)
                body_header = _parse_npz_npy_header(body_npz_path, "vertices_seq")
                if body_header is not None:
                    body_full_frames = body_header[0][0]
                    body_trimmed_frames = body_full_frames - 3  # [1:-2] trims 3
                    body_use_mmap = body_trimmed_frames == sequence_frames
                else:
                    body_use_mmap = False
                if body_use_mmap:
                    # Indices into full array: frame_indices_abs + 1 (skip first frame of [1:-2])
                    body_actual_indices = frame_indices_abs + 1
                    vertices_seq = _mmap_npz_array_selective(body_npz_path, "vertices_seq", body_actual_indices).astype(np.float32)
                    body_pose = body_info["pose"]
                    body_pose_matrix = self.euler_to_rotation_matrix_obj_v1(body_pose.yaw, body_pose.pitch, body_pose.roll)
                    body_translation = np.array([body_pose.x, body_pose.y, body_pose.z], dtype=np.float32) / 100.0
                    if sequence_scales:
                        _, global_translation = sequence_scales[0]
                    else:
                        global_translation = np.zeros(3, dtype=np.float32)
                    trans = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
                    vertices_seq_d = vertices_seq @ trans.T @ body_pose_matrix.T + (body_translation + global_translation)
                    vertices_seq_d_selected["_00_body"] = vertices_seq_d.astype(np.float32)
                    with np.load(body_npz_path, allow_pickle=True) as body_npz_f:
                        faces_list["_00_body"] = body_npz_f["faces"]
                else:
                    # Fallback: full load (compressed NPZ, header parse failure, or frame count mismatch)
                    with np.load(body_npz_path, allow_pickle=True) as body_npz_f:
                        body_npz = {k: body_npz_f[k] for k in body_npz_f.files}
                    vertices_seq = body_npz["vertices_seq"][1:-2]
                    if len(vertices_seq) == sequence_frames:
                        vertices_seq = np.asarray(vertices_seq[frame_indices_abs], dtype=np.float32)
                        body_pose = body_info["pose"]
                        body_pose_matrix = self.euler_to_rotation_matrix_obj_v1(body_pose.yaw, body_pose.pitch, body_pose.roll)
                        body_translation = np.array([body_pose.x, body_pose.y, body_pose.z], dtype=np.float32) / 100.0
                        if sequence_scales:
                            _, global_translation = sequence_scales[0]
                        else:
                            global_translation = np.zeros(3, dtype=np.float32)
                        trans = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
                        vertices_seq_d = vertices_seq @ trans.T @ body_pose_matrix.T + (body_translation + global_translation)
                        vertices_seq_d_selected["_00_body"] = vertices_seq_d.astype(np.float32)
                        faces_list["_00_body"] = body_npz["faces"]

        for obj_id, obj_info in enumerate(seq_info.get("objects", [])):
            mask_name = f"obj_{(obj_id + 1):02d}"
            if required_mask_names is not None and mask_name not in required_mask_names:
                continue
            npz_path = osp.join(self.root_obj_npz_path, obj_info["object_group"], obj_info["object_id"], "vertices_sequence.npz")
            if not osp.exists(npz_path):
                continue
            # Use mmap for selective frame reading (object NPZs up to 11GB)
            obj_header = _parse_npz_npy_header(npz_path, "vertices_seq")
            if obj_header is not None:
                obj_frames = obj_header[0][0]
            else:
                # Fallback: must load to get shape
                with np.load(npz_path, allow_pickle=True) as fallback_data:
                    obj_frames = fallback_data["vertices_seq"].shape[0]
            selected_indices = (frame_indices_abs + 1) % obj_frames
            vertices_seq = _mmap_npz_array_selective(npz_path, "vertices_seq", selected_indices).astype(np.float32)
            vertices_seq[..., 1] = -vertices_seq[..., 1]

            body_pose = obj_info["pose"]
            body_pose_matrix = self.euler_to_rotation_matrix_obj_objverse(body_pose.yaw, body_pose.pitch, body_pose.roll)
            body_translation = np.array([body_pose.x, body_pose.y, body_pose.z], dtype=np.float32) / 100.0
            if obj_id < len(sequence_scales):
                body_scale, global_translation = sequence_scales[obj_id]
            else:
                body_scale, global_translation = 1.0, np.zeros(3, dtype=np.float32)
            vertices_seq_d = (vertices_seq * body_scale) @ body_pose_matrix.T + (body_translation + global_translation)
            vertices_seq_d_selected[mask_name] = vertices_seq_d.astype(np.float32)
            with np.load(npz_path, allow_pickle=True) as obj_npz_f:
                faces_list[mask_name] = obj_npz_f["faces"]

        return vertices_seq_d_selected, faces_list

    def _build_vertices_seq(self, scene_name: str, sample: dict, required_mask_names: set[str] | None = None):
        resolved_sequence_name, seq_info = self._resolve_sequence_info(scene_name, sample)
        frame_indices_abs = np.asarray(sample["frame_indices_abs"], dtype=np.int64)
        return self._build_selected_vertices_seq(
            scene_name,
            resolved_sequence_name,
            seq_info,
            frame_indices_abs,
            required_mask_names=required_mask_names,
        )

    @staticmethod
    def _discover_tracking_mask_names(track_dir: str) -> set[str]:
        if not osp.isdir(track_dir):
            return set()
        mask_names = set()
        for filename in os.listdir(track_dir):
            if filename.endswith(".npz"):
                if filename.startswith("vertex_indices_"):
                    mask_names.add(filename[len("vertex_indices_") : -len(".npz")])
                elif filename.startswith("tracking_"):
                    mask_names.add(filename[len("tracking_") : -len(".npz")])
        return mask_names

    @staticmethod
    def _resolve_tracking_npz_path(track_dir: str, mask_name: str) -> str | None:
        for prefix in ("vertex_indices_", "tracking_"):
            npz_path = osp.join(track_dir, f"{prefix}{mask_name}.npz")
            if osp.exists(npz_path):
                return npz_path
        return None

    def _discover_tracking_mask_names_any(self, scene_name: str, camera_seq_name: str, frame_stem: str) -> set[str]:
        """Mask names with a tracking record for this frame, from npz OR safetensor.

        Honours self.tracking_format. In 'auto' the dense npz dir is scanned
        first and the compact safetensor shard is consulted only if no npz
        record is present (the new archives ship empty npz dirs + safetensors).
        """
        track_dir = osp.join(self.ROOT, scene_name, "npz", camera_seq_name, frame_stem)
        names: set[str] = set()
        if self.tracking_format != "safetensor":
            names |= self._discover_tracking_mask_names(track_dir)
        if self.tracking_format in ("auto", "safetensor", "compact_faceid") and not names:
            names |= self._safetensor_tracking_mask_names(scene_name, camera_seq_name, frame_stem)
        return names

    def _frame_has_tracking_npz(self, scene_name: str, camera_seq_name: str, frame_name: str) -> bool:
        frame_stem = osp.splitext(str(frame_name))[0]
        return bool(self._discover_tracking_mask_names_any(scene_name, camera_seq_name, frame_stem))

    def _crop_resize_if_necessary_with_track(
        self,
        image,
        depthmap,
        intrinsics,
        resolution,
        *,
        rng=None,
        info=None,
        track=None,
        track_valid_mask=None,
        allow_random_transpose=False,
    ):
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        width, height = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, width - cx)
        min_margin_y = min(cy, height - cy)
        assert min_margin_x > width / 5, f"Bad principal point in view={info}"
        assert min_margin_y > height / 5, f"Bad principal point in view={info}"
        crop_bbox = (cx - min_margin_x, cy - min_margin_y, cx + min_margin_x, cy + min_margin_y)

        image, depthmap, intrinsics, track, track_valid_mask = cropping.crop_image_depthmap(
            image,
            depthmap,
            intrinsics,
            crop_bbox,
            track=track,
            track_valid_mask=track_valid_mask,
        )

        width, height = image.size
        assert resolution[0] >= resolution[1]
        if height > 1.1 * width:
            resolution = resolution[::-1]
        elif 0.9 < height / width < 1.1 and resolution[0] != resolution[1]:
            if allow_random_transpose and rng is not None and rng.integers(2):
                resolution = resolution[::-1]

        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            delta = rng.integers(0, self.aug_crop) if rng is not None else 0
            target_resolution += delta

        image, depthmap, intrinsics, track, track_valid_mask = cropping.rescale_image_depthmap(
            image,
            depthmap,
            intrinsics,
            target_resolution,
            track=track,
            track_valid_mask=track_valid_mask,
        )

        intrinsics_out = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics_out, resolution)
        image, depthmap, intrinsics_out, track, track_valid_mask = cropping.crop_image_depthmap(
            image,
            depthmap,
            intrinsics,
            crop_bbox,
            track=track,
            track_valid_mask=track_valid_mask,
        )

        return image, depthmap, intrinsics_out, track, track_valid_mask

    def _load_frame_geometry(
        self,
        sample: dict,
        local_view_idx: int,
        per_frame_tracking: list[dict] | None,
        *,
        compute_pts3d_world: bool = True,
    ):
        scene_name = sample["scene_name"]
        frame_name = sample["frame_names"][local_view_idx]
        frame_stem = osp.splitext(frame_name)[0]
        impath = osp.join(sample["view_dir"], frame_name)
        row = self._load_camera_rows(sample["camera_csv_path"]).get(frame_name)
        if row is None:
            raise ValueError(f"camera row not found for {frame_name}")

        raw_rgb_frames = sample.get("_raw_rgb_frames_by_name") or {}
        if frame_name in raw_rgb_frames:
            rgb_image = np.asarray(raw_rgb_frames[frame_name])
            if rgb_image.ndim == 2:
                rgb_image = np.repeat(rgb_image[..., None], 3, axis=-1)
            if rgb_image.ndim == 3 and rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[..., :3]
            rgb_image = np.ascontiguousarray(rgb_image.astype(np.uint8, copy=False))
        else:
            rgb_image = imread_cv2(impath)
        if rgb_image is None:
            raise ValueError(f"Failed to load RGB image: {impath}")
        if not np.any(rgb_image):
            raise ValueError(f"Purely black RGB image: {impath}")
        depthmap = read_depth_exr(
            osp.join(self.ROOT, scene_name, "exr_layers", "depth", sample["camera_seq_name"], f"{frame_stem}_depth.exr")
        ) / 100.0
        invalid_depth_mask = np.zeros(depthmap.shape, dtype=bool)
        if self._depth_filtering_enabled() and per_frame_tracking is not None:
            frame_tracking = per_frame_tracking[local_view_idx]
            for mask_type, tracking_data in frame_tracking.items():
                masked_valid = tracking_data.get("masked_valid")
                distances = tracking_data.get("distances")
                if masked_valid is None or distances is None:
                    continue
                masked_valid = np.asarray(masked_valid, dtype=bool)
                if masked_valid.ndim == 3 and masked_valid.shape[-1] == 1:
                    masked_valid = masked_valid[..., 0]
                if masked_valid.shape != depthmap.shape:
                    continue
                valid_y, valid_x = np.where(masked_valid)
                if len(valid_y) == 0:
                    continue
                distances = np.asarray(distances, dtype=np.float32).reshape(-1)
                n = min(len(valid_y), len(distances))
                valid_y = valid_y[:n]
                valid_x = valid_x[:n]
                distances = distances[:n]
                if mask_type.startswith("obj_"):
                    if float(np.std(distances)) >= self.object_distance_std_thresh:
                        invalid_depth_mask[valid_y, valid_x] = True
                else:
                    bad = distances >= self.non_object_distance_thresh
                    if np.any(bad):
                        invalid_depth_mask[valid_y[bad], valid_x[bad]] = True

        invalid_depth_mask |= depthmap > self.invalid_depth_max
        invalid_depth_mask |= ~np.isfinite(depthmap)
        depthmap[invalid_depth_mask] = 0.0

        focal_length = float(row["focal_length"])
        sensor_width = float(row["sensor_width"])
        sensor_height = float(row["sensor_height"])
        image_width = rgb_image.shape[1]
        image_height = rgb_image.shape[0]
        fx = (focal_length / sensor_width) * image_width
        fy = (focal_length / sensor_height) * image_height
        cx = image_width / 2.0
        cy = image_height / 2.0
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = self.euler_to_rotation_matrix(float(row["yaw"]), float(row["pitch"]), float(row["roll"])).astype(np.float32)
        camera_pose[:3, 3] = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float32) / 100.0

        if compute_pts3d_world:
            pts3d_world, valid_mask = depthmap_to_absolute_camera_coordinates(
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                camera_pose=camera_pose,
            )
            valid_mask = valid_mask & np.isfinite(pts3d_world).all(axis=-1)
        else:
            pts3d_world = None
            valid_mask = None
        return impath, rgb_image, depthmap, intrinsics, camera_pose, pts3d_world, valid_mask

    def _get_views(self, idx, resolution, rng, num_views, track_query_idx=0):
        if int(idx) in self._blacklisted_sample_indices:
            raise ValueError(f"blacklisted Syn4D clip index={idx}")
        sample = self.samples[idx]
        if int(num_views) > len(sample["frame_names"]):
            raise ValueError(f"Requested num_views={num_views} but indexed clip has only {len(sample['frame_names'])} frames")

        sample = dict(sample)
        sample["frame_names"] = sample["frame_names"][: int(num_views)]
        sample["frame_indices_abs"] = sample["frame_indices_abs"][: int(num_views)]

        track_query_idx = int(track_query_idx)
        if track_query_idx < 0 or track_query_idx >= len(sample["frame_names"]):
            raise ValueError(
                f"track_query_idx={track_query_idx} out of range for clip with "
                f"{len(sample['frame_names'])} frames"
            )
        query_frame_name = sample["frame_names"][track_query_idx]
        if not self._frame_has_tracking_npz(
            sample["scene_name"], sample["camera_seq_name"], query_frame_name
        ):
            raise ValueError(
                f"No NPZ data at track_query_idx={track_query_idx} "
                f"(frame={osp.splitext(query_frame_name)[0]}). "
                "NPZ only exists every 5 absolute frames."
            )
        query_frame_stem = osp.splitext(sample["frame_names"][track_query_idx])[0]
        query_track_dir = osp.join(self.ROOT, sample["scene_name"], "npz", sample["camera_seq_name"], query_frame_stem)
        query_mask_names = self._discover_tracking_mask_names_any(
            sample["scene_name"], sample["camera_seq_name"], query_frame_stem
        )
        vertices_seq_d_list, faces_list = self._build_vertices_seq(
            sample["scene_name"],
            sample,
            required_mask_names=query_mask_names if query_mask_names else None,
        )
        mesh_backed_masks = set(vertices_seq_d_list.keys()) & set(query_mask_names)
        if query_mask_names and not mesh_backed_masks:
            raise ValueError(
                "Syn4D tracking NPZ exists but no matching mesh vertex sequence "
                f"could be loaded for scene={sample['scene_name']!r}, "
                f"camera_seq={sample['camera_seq_name']!r}, "
                f"frame={query_frame_stem!r}, masks={sorted(query_mask_names)}. "
                "Returning this sample would produce a static repeated query "
                "pointmap, so it is rejected. Check Syn4D metadata paths and "
                "permissions."
            )
        if self._depth_filtering_enabled():
            per_frame_tracking = []
            for frame_name in sample["frame_names"]:
                frame_stem = osp.splitext(frame_name)[0]
                track_dir = osp.join(self.ROOT, sample["scene_name"], "npz", sample["camera_seq_name"], frame_stem)
                frame_data = {}
                for mask_name in vertices_seq_d_list.keys():
                    tracking_data = self._load_tracking_dict(
                        sample["scene_name"], sample["camera_seq_name"], frame_stem, mask_name
                    )
                    if tracking_data is not None:
                        frame_data[mask_name] = tracking_data
                per_frame_tracking.append(frame_data)
        else:
            per_frame_tracking = [None] * int(num_views)

        query_tracking = {}
        for mask_name in vertices_seq_d_list.keys():
            tracking_data = self._load_tracking_dict(
                sample["scene_name"], sample["camera_seq_name"], query_frame_stem, mask_name
            )
            if tracking_data is not None:
                query_tracking[mask_name] = tracking_data
        per_frame_tracking[track_query_idx] = query_tracking

        try:
            _, _, _, _, _, pts3d_world_query, depth_query_valid_map = self._load_frame_geometry(
                sample,
                track_query_idx,
                per_frame_tracking,
                compute_pts3d_world=True,
            )
        except ValueError as e:
            if "Purely black RGB image" in str(e):
                self._blacklisted_sample_indices.add(int(idx))
            raise

        static_background_valid_map = np.zeros_like(depth_query_valid_map, dtype=bool)
        query_frame_stem_for_mask = osp.splitext(sample['frame_names'][track_query_idx])[0]
        got_png_env_mask = False
        for mask_type in sample["mask_types"]:
            if "env" not in mask_type:
                continue
            mask_path = osp.join(sample["mask_root"], f"{query_frame_stem_for_mask}{mask_type}.png")
            if not osp.isfile(mask_path):
                continue
            env_mask = imread_cv2(mask_path)
            env_valid = np.any(env_mask > 0, axis=-1) if env_mask.ndim == 3 else (env_mask > 0)
            if env_valid.shape == static_background_valid_map.shape:
                static_background_valid_map |= env_valid
                got_png_env_mask = True
        # New format: env masks live in env_mask_safetensors/ (PNG bytes), not as
        # per-frame PNGs. Fall back to the safetensor shard when no PNG was found.
        if not got_png_env_mask:
            env_mask = self._load_env_mask_safetensor(
                sample["scene_name"], sample["camera_seq_name"], query_frame_stem_for_mask
            )
            if env_mask is not None:
                env_valid = np.any(env_mask > 0, axis=-1) if env_mask.ndim == 3 else (env_mask > 0)
                if env_valid.shape == static_background_valid_map.shape:
                    static_background_valid_map |= env_valid

        # Batch-compute wrapped pts3d for ALL frames at once (avoids 50x redundant computation)
        all_wrapped, wrapped_valid_mask_common = precompute_all_wrapped_pts3d(
            target_pts3d=pts3d_world_query,
            target_tracking_npz_dict=query_tracking,
            vertices_seq_d_list=vertices_seq_d_list,
            faces_list=faces_list,
            num_frames=int(num_views),
            depth_query_valid_map=depth_query_valid_map,
            static_background_valid_map=static_background_valid_map,
            wrapped_face_distance_thresh=self.wrapped_face_distance_thresh,
            track_query_idx=int(track_query_idx),
        )

        if self._query_consistency_filter_enabled():
            query_wrapped_world = all_wrapped[track_query_idx]
            query_track_valid_mask = wrapped_valid_mask_common
            query_check_mask = (
                query_track_valid_mask
                & np.isfinite(query_wrapped_world).all(axis=-1)
                & np.isfinite(pts3d_world_query).all(axis=-1)
            )
            query_track_l2 = np.linalg.norm(query_wrapped_world.astype(np.float32) - pts3d_world_query.astype(np.float32), axis=-1)
            query_inconsistent_mask = query_check_mask & (query_track_l2 > self.query_inconsistent_distance_thresh)
        else:
            query_inconsistent_mask = np.zeros_like(depth_query_valid_map, dtype=bool)

        views = []
        for local_view_idx in range(int(num_views)):
            try:
                impath, rgb_image, depthmap, intrinsics, camera_pose, _, _ = self._load_frame_geometry(
                    sample,
                    local_view_idx,
                    per_frame_tracking,
                    compute_pts3d_world=False,
                )
            except ValueError as e:
                if "Purely black RGB image" in str(e):
                    self._blacklisted_sample_indices.add(int(idx))
                raise
            wrapped_world = all_wrapped[local_view_idx].copy()
            track_valid_mask = wrapped_valid_mask_common.copy()

            wrapped_world = wrapped_world.astype(np.float32)
            track_valid_mask = track_valid_mask & np.isfinite(wrapped_world).all(axis=-1) & (~query_inconsistent_mask)
            wrapped_world[~track_valid_mask] = 0.0

            rgb_image, depthmap, intrinsics, wrapped_world, track_valid_mask = self._crop_resize_if_necessary_with_track(
                rgb_image,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=impath,
                track=wrapped_world,
                track_valid_mask=track_valid_mask.astype(np.bool_),
                allow_random_transpose=False,
            )

            view = dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                track=wrapped_world.astype(np.float32),
                track_valid_mask=track_valid_mask.astype(np.bool_),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                track_query_idx=track_query_idx,
                dataset=self.dataset_label,
                label=f"{sample['scene_name']}_{sample['camera_seq_name']}",
                instance=f"{osp.splitext(sample['frame_names'][0])[0]}_{local_view_idx:03d}_q{track_query_idx}",
                fps=sample["fps"],
                stride=int(sample.get("stride", self.stride)),
                is_metric=self.is_metric,
                is_video=True,
                quantile=np.array(1.0, dtype=np.float32),
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
            )
            if self.return_trackvid:
                dummy_trackvid = np.full_like(wrapped_world, self.trackvid_fill_value, dtype=np.float32)
                dummy_valid = np.zeros(track_valid_mask.shape, dtype=np.bool_)
                view["track_video_prenormalized"] = dummy_trackvid
                view["track_video_valid_mask"] = dummy_valid
            views.append(view)

        return views


__all__ = ["Syn4D_Multi"]
