import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base.easy_dataset import EasyDataset


class DavisStream3RPred(EasyDataset):
    """
    DAVIS real-world sequences backed by Stream3R exports.

    Expects the output structure produced by:
      `scripts/stream3r/export_stream3r_xyz_davis.py`

    ROOT/
      <sequence_name>/
        stream3r_pred.npz

    Each NPZ contains (at least):
      - images: (T,3,H,W) float32 [0,1]
      - world_points_recon: (T,H,W,3) float32 (metric)
      - valid_mask: (T,H,W) bool
      - cam2world: (T,4,4) float32 (optional)
    """

    def __init__(
        self,
        *,
        ROOT: str,
        split: str | None = None,
        resolution=None,
        num_views: int = 49,
        allow_repeat: bool = True,
        npz_name: str = "stream3r_pred.npz",
        pad_to_num_views: bool = True,
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        self.ROOT = str(ROOT)
        self.split = split
        self.num_views = int(num_views)
        self.allow_repeat = bool(allow_repeat)
        self.npz_name = str(npz_name)
        self.pad_to_num_views = bool(pad_to_num_views)

        # Stream3R datasets use a list of resolutions like [[W, H]]
        if resolution is None:
            self._resolutions = [[256, 256]]
        else:
            self._resolutions = resolution

        # Normalize resolution format to (W, H)
        w, h = self._infer_first_resolution_wh(self._resolutions)
        self._target_w = int(w)
        self._target_h = int(h)

        root = Path(self.ROOT).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"DavisStream3RPred ROOT not found: {root}")

        seq_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
        start_idx = int(start_idx)
        if end_idx is None:
            end_idx = len(seq_names)
        end_idx = int(end_idx)
        self.seq_names = seq_names[start_idx:end_idx]
        if not self.seq_names:
            raise RuntimeError(f"No sequence folders found under: {root}")

        # Pre-build NPZ paths (fail fast if missing)
        self._npz_paths: list[Path] = []
        for name in self.seq_names:
            npz_path = root / name / self.npz_name
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing Stream3R NPZ: {npz_path}")
            self._npz_paths.append(npz_path)

    @staticmethod
    def _infer_first_resolution_wh(resolutions):
        # Resolutions commonly look like: [[W, H]] or [[[W, H]]] in some configs
        r = resolutions
        while isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], (list, tuple)):
            r = r[0]
        if not (isinstance(r, (list, tuple)) and len(r) == 2):
            raise ValueError(f"Invalid resolution format: {resolutions!r} (expected [[W,H]])")
        return int(r[0]), int(r[1])

    @staticmethod
    def _to_pil_from_float_chw(img_chw: np.ndarray) -> Image.Image:
        if img_chw.ndim != 3 or img_chw.shape[0] != 3:
            raise ValueError(f"Expected image CHW with 3 channels, got {img_chw.shape}")
        img = np.clip(img_chw, 0.0, 1.0)
        img_hwc = (img.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(img_hwc, mode="RGB")

    @staticmethod
    def _resize_points_thw3(points_thw3: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
        if points_thw3.ndim != 4 or points_thw3.shape[-1] != 3:
            raise ValueError(f"Expected points (T,H,W,3), got {points_thw3.shape}")
        t, h, w, _ = points_thw3.shape
        if (h, w) == (out_h, out_w):
            return points_thw3.astype(np.float32, copy=False)
        pts = torch.from_numpy(points_thw3.astype(np.float32)).permute(0, 3, 1, 2)  # (T,3,H,W)
        pts_rs = F.interpolate(pts, size=(int(out_h), int(out_w)), mode="bilinear", align_corners=False)
        return pts_rs.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    @staticmethod
    def _resize_mask_thw(mask_thw: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
        if mask_thw.ndim != 3:
            raise ValueError(f"Expected mask (T,H,W), got {mask_thw.shape}")
        t, h, w = mask_thw.shape
        if (h, w) == (out_h, out_w):
            return mask_thw.astype(bool, copy=False)
        m = torch.from_numpy(mask_thw.astype(np.float32)).unsqueeze(1)  # (T,1,H,W)
        m_rs = F.interpolate(m, size=(int(out_h), int(out_w)), mode="nearest")[:, 0] > 0.5
        return m_rs.cpu().numpy().astype(bool)

    @staticmethod
    def _coerce_intrinsics_to_k(intrinsics: np.ndarray, *, width: int, height: int) -> np.ndarray:
        intr = np.asarray(intrinsics, dtype=np.float32)
        if intr.shape == (3, 3):
            return intr
        if intr.shape == (4,):
            fx, fy, cx, cy = [float(v) for v in intr]
            return np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        # Fallback: principal point at image center, focal ~= max side length
        f = float(max(int(width), int(height)))
        cx = float(width) * 0.5
        cy = float(height) * 0.5
        return np.array(
            [
                [f, 0.0, cx],
                [0.0, f, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self._npz_paths)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # Some wrappers may pass tuple indices; only the base idx matters here.
            idx = idx[0]
        idx = int(idx) % len(self._npz_paths)

        npz_path = self._npz_paths[idx]
        seq_name = self.seq_names[idx]

        data = np.load(str(npz_path), allow_pickle=True)
        if "images" not in data:
            raise KeyError(f"NPZ missing 'images': {npz_path}")
        if "world_points_recon" not in data:
            raise KeyError(f"NPZ missing 'world_points_recon': {npz_path}")
        if "valid_mask" not in data:
            raise KeyError(f"NPZ missing 'valid_mask': {npz_path}")

        images_tchw = np.asarray(data["images"], dtype=np.float32)  # (T,3,H,W)
        points_thw3 = np.asarray(data["world_points_recon"], dtype=np.float32)  # (T,H,W,3)
        valid_thw = np.asarray(data["valid_mask"]).astype(bool)  # (T,H,W)

        # Camera pose (optional). Stream3R exports cam2world if available.
        cam2world_t44 = None
        if "cam2world" in data and data["cam2world"] is not None:
            cam2world_t44 = np.asarray(data["cam2world"], dtype=np.float32)
            if cam2world_t44.ndim == 3 and cam2world_t44.shape[1:] == (4, 4):
                pass
            else:
                cam2world_t44 = None

        intrinsics_t33 = None
        for key in ("intrinsics", "intrinsics_pred", "camera_intrinsics"):
            if key in data and data[key] is not None:
                intr_raw = np.asarray(data[key], dtype=np.float32)
                break
        else:
            intr_raw = None

        if intr_raw is None:
            intrinsics_t33 = np.repeat(
                self._coerce_intrinsics_to_k(np.zeros((0,), dtype=np.float32), width=self._target_w, height=self._target_h)[None],
                int(images_tchw.shape[0]),
                axis=0,
            )
        elif intr_raw.ndim == 2 and intr_raw.shape == (3, 3):
            intrinsics_t33 = np.repeat(intr_raw[None], int(images_tchw.shape[0]), axis=0)
        elif intr_raw.ndim == 2 and intr_raw.shape[1] == 4:
            intrinsics_t33 = np.stack(
                [self._coerce_intrinsics_to_k(v, width=self._target_w, height=self._target_h) for v in intr_raw],
                axis=0,
            )
        elif intr_raw.ndim == 3 and intr_raw.shape[1:] == (3, 3):
            intrinsics_t33 = intr_raw
        else:
            intrinsics_t33 = np.repeat(
                self._coerce_intrinsics_to_k(intr_raw, width=self._target_w, height=self._target_h)[None],
                int(images_tchw.shape[0]),
                axis=0,
            )
        if int(intrinsics_t33.shape[0]) < int(images_tchw.shape[0]):
            intrinsics_t33 = np.concatenate(
                [
                    intrinsics_t33,
                    np.repeat(intrinsics_t33[-1:,:,:], int(images_tchw.shape[0]) - int(intrinsics_t33.shape[0]), axis=0),
                ],
                axis=0,
            )
        elif int(intrinsics_t33.shape[0]) > int(images_tchw.shape[0]):
            intrinsics_t33 = intrinsics_t33[: int(images_tchw.shape[0])]

        T = int(images_tchw.shape[0])
        if T <= 0:
            raise ValueError(f"Empty sequence in NPZ: {npz_path}")

        # Take / pad to num_views (Wan VAE expects 4k+1 like 49)
        if T >= self.num_views:
            sel = list(range(self.num_views))
        else:
            if not self.pad_to_num_views:
                raise ValueError(f"Sequence {seq_name} has only {T} frames, need {self.num_views}: {npz_path}")
            sel = list(range(T)) + [T - 1] * (self.num_views - T)

        images_tchw = images_tchw[sel]
        points_thw3 = points_thw3[sel]
        valid_thw = valid_thw[sel]
        cam2world_t44 = cam2world_t44[sel] if cam2world_t44 is not None else None
        intrinsics_t33 = intrinsics_t33[sel]

        # Resize to target resolution (W,H) provided by config
        out_h, out_w = int(self._target_h), int(self._target_w)
        points_thw3 = self._resize_points_thw3(points_thw3, out_h=out_h, out_w=out_w)
        valid_thw = self._resize_mask_thw(valid_thw, out_h=out_h, out_w=out_w)

        views = []
        for t in range(self.num_views):
            pil = self._to_pil_from_float_chw(images_tchw[t])
            if pil.size != (out_w, out_h):
                pil = pil.resize((out_w, out_h), Image.BILINEAR)

            pts = points_thw3[t]  # (H,W,3)
            valid = valid_thw[t]  # (H,W)

            # A positive depth-like map (used only for optional visualization paths in the adapter).
            depthmap = np.linalg.norm(pts, axis=-1).astype(np.float32)

            # Provide camera pose for cam1 alignment (fallback to identity if unavailable)
            if cam2world_t44 is not None:
                pose = cam2world_t44[t]
            else:
                pose = np.eye(4, dtype=np.float32)
            K = intrinsics_t33[t] if intrinsics_t33 is not None else self._coerce_intrinsics_to_k(
                np.zeros((0,), dtype=np.float32),
                width=out_w,
                height=out_h,
            )

            views.append(
                {
                    "img": pil,
                    "depthmap": depthmap,
                    "pts3d": pts.astype(np.float32, copy=False),
                    "valid_mask": valid.astype(bool, copy=False),
                    "camera_pose": pose.astype(np.float32, copy=False),
                    "camera_intrinsics": np.asarray(K, dtype=np.float32),
                    "intrinsics": np.asarray(K, dtype=np.float32),
                    # Metadata required by Stream3RDatasetAdapter output formatting.
                    "dataset": "davis_stream3r",
                    "label": str(seq_name),
                    "instance": 0,
                }
            )

        return views
