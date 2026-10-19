"""Portable, pickle-free dense NVS example contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "holo4d-dense-example-v1"
REQUIRED = ("source_video", "tracks_world", "tracks_world_valid_mask",
            "source_cam2world", "source_intrinsics", "target_cam2world",
            "target_intrinsics", "source_depth_video", "source_rgb_at_tracks")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Bundle member escapes its root: {name}")
    return path


def validate_condition(data: dict) -> dict:
    missing = set(REQUIRED) - data.keys()
    if missing:
        raise ValueError(f"Missing condition fields: {sorted(missing)}")
    video = np.asarray(data["source_video"])
    if video.ndim != 4 or video.shape[0] != 49 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError("source_video must be uint8 (49,H,W,3)")
    tracks = np.asarray(data["tracks_world"])
    if tracks.ndim != 3 or tracks.shape[1:] != (49, 3):
        raise ValueError("tracks_world must be (N,49,3)")
    if data["tracks_world_valid_mask"].shape != tracks.shape[:2]:
        raise ValueError("Track validity shape mismatch")
    for key in ("source_cam2world", "target_cam2world"):
        if data[key].shape != (49, 4, 4):
            raise ValueError(f"{key} must contain all49 camera poses")
    for key in ("source_intrinsics", "target_intrinsics"):
        if data[key].shape not in ((3, 3), (49, 3, 3)):
            raise ValueError(f"{key} has invalid intrinsics shape")
    for key, value in data.items():
        if np.asarray(value).dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite values in {key}")
    return {"frames": 49, "height": int(video.shape[1]), "width": int(video.shape[2]),
            "identities": int(tracks.shape[0]),
            "valid_ratio": float(data["tracks_world_valid_mask"].mean()),
            "target_K_per_frame": data["target_intrinsics"].shape == (49, 3, 3)}


def load_example(manifest_path: Path, example_id: str, *, verify: bool = True):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Unknown example bundle format")
    entries = [entry for entry in manifest["examples"] if entry["id"] == example_id]
    if len(entries) != 1:
        raise ValueError(f"Unknown or duplicated example: {example_id}")
    entry = entries[0]
    if not entry.get("caption", "").strip():
        raise ValueError("Example caption must not be empty")
    path = resolve_member(manifest_path.parent, entry["condition"])
    if verify and file_sha256(path) != entry["condition_sha256"]:
        raise ValueError(f"Condition checksum mismatch: {example_id}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    report = validate_condition(data)
    return entry, data, report


def condition_to_sample(data: dict, entry: dict) -> dict:
    from PIL import Image

    sample = {key: value for key, value in data.items() if key != "source_video"}
    sample["video"] = [Image.fromarray(frame) for frame in data["source_video"]]
    sample["prompt"] = entry["caption"]
    sample["track_query_idx"] = int(np.asarray(data.get("track_query_idx", 0)))
    sample["has_tracks"] = True
    sample["trajectory_name"] = entry.get("trajectory", "example")
    sample["trajectory_kind"] = "novel_view"
    sample["output_fps"] = 16
    # Neither paired target RGB nor archived predictions enter the model.
    sample["target_video"] = None
    sample["inference_target_video"] = None
    if data["target_intrinsics"].ndim == 3:
        sample["target_intrinsics_per_frame"] = data["target_intrinsics"]
    return sample


def resize_condition(data: dict, height: int, width: int) -> dict:
    """Resize the image canvas, not the world gauge or identity population.

    RGB/depth use bilinear interpolation; masks use nearest. Both source and
    all target intrinsics are rescaled. This is not a new geometry estimate.
    """
    import torch
    import torch.nn.functional as F

    old_h, old_w = data["source_video"].shape[1:3]
    result = dict(data)
    video = torch.from_numpy(data["source_video"].copy()).permute(0, 3, 1, 2).float()
    result["source_video"] = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    for key in ("source_intrinsics", "target_intrinsics", "target_intrinsics_per_frame"):
        if key in data:
            k = data[key].copy()
            k[..., 0, :] *= width / old_w
            k[..., 1, :] *= height / old_h
            result[key] = k
    for key in ("source_depth_video", "source_depth_valid", "source_dynamic_mask", "target_dynamic_mask", "reproj_src_depth_at_target", "reproj_src_depth_valid"):
        if key not in data:
            continue
        value = data[key]
        # Canonical loader stores these payloads per identity, not as images.
        # Camera-space z and identity validity do not change with image resize.
        if key in ("reproj_src_depth_at_target", "reproj_src_depth_valid", "source_depth_valid") and value.ndim == 2:
            continue
        if value.ndim not in (3, 4) or value.shape[-2:] != (old_h, old_w):
            raise ValueError(f"Unsupported image-shaped condition: {key} {value.shape}")
        tensor = torch.from_numpy(value.copy()).float()
        if value.ndim == 3:
            tensor = tensor[:, None]
        is_mask = "mask" in key or "valid" in key
        resized = F.interpolate(tensor, size=(height, width), mode="nearest" if is_mask else "bilinear", **({} if is_mask else {"align_corners": False}))
        if value.ndim == 3:
            resized = resized[:, 0]
        result[key] = resized.numpy().astype(value.dtype)
    validate_condition(result)
    return result
