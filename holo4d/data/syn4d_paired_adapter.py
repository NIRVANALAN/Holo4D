"""
Syn4D Paired Camera Dataset for Track4D-360.

Produces paired (source, target) samples from two different cameras
in a Syn4D scene, with world-space 3D tracks for track conditioning.

Tracks are dense by default: every pixel on a uniform grid (controlled by
``track_spatial_stride``) becomes a track.  stride=1 → full H*W dense tracks.

Syn4D directory layout:
  <scene_root>/png/<seq_name>/  → RGB frames (PNG)
  <scene_root>/exr_layers/depth/<seq_name>/  → depth EXR files
  <scene_root>/ground_truth/meta_exr_csv/<seq_name>_camera.csv  → per-frame camera params
  <syn4d_root>/caption/<scene_name>_per_video_chunks.json  → per-chunk captions

Sequence naming: seq_XXXXXX_N where XXXXXX is the sequence index and N is camera index (0-7).
Each scene has multiple sequences, each with 8 cameras (N=0..7).

Frame selection: each __getitem__ picks a random caption chunk, a random contiguous
window of ``num_frames`` frames within that chunk, and random source/target cameras.
This matches the recammaster training data pipeline.

Each sample returns:
  - video: list of PIL images (source camera RGB)
  - target_video: list of PIL images (target camera RGB)
  - tracks_world: (N, T, 3) world-space 3D tracks
  - target_cam2world: (T, 4, 4) target camera extrinsics
  - target_intrinsics: (3, 3) target camera K matrix
  - prompt: text prompt (from Syn4D captions)
"""

import csv
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from typing import Optional, List, Dict, Tuple


_PNG_FRAME_RE = re.compile(r"^(?P<seq>seq_\d+_\d+)_(?P<frame>\d+)\.png$")

DEFAULT_PROMPT = "A dynamic multi-view video scene."


def _euler_to_rotation_matrix(yaw, pitch, roll):
    """Syn4D camera coordinate convention: yaw-pitch-roll → rotation matrix.

    Matches the euler_to_rotation_matrix in diffsynth/data/stream3r/dust3r/datasets_cut3r/syn4d.py
    and recammaster-official/inference_recammaster_syn4d_multimodal.py.
    """
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
    r_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                         [0, 1, 0],
                         [np.sin(pitch), 0, np.cos(pitch)]])
    r_roll = np.array([[1, 0, 0],
                        [0, np.cos(roll), np.sin(roll)],
                        [0, -np.sin(roll), np.cos(roll)]])
    zxy_xyz = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    return r_yaw @ r_pitch @ r_roll @ zxy_xyz


def _depthmap_to_pts3d_world(depthmap, intrinsics, camera_pose):
    """Unproject depth map to world-space 3D points.

    Args:
        depthmap: (H, W) float32
        intrinsics: (3, 3) camera intrinsics
        camera_pose: (4, 4) cam2world

    Returns:
        pts3d_world: (H, W, 3)
        valid_mask: (H, W) bool
    """
    H, W = depthmap.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    yy, xx = np.mgrid[0:H, 0:W]
    x_cam = (xx - cx) / fx * depthmap
    y_cam = (yy - cy) / fy * depthmap
    z_cam = depthmap

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    pts_world = pts_cam @ R.T + t  # (H, W, 3)

    valid = (depthmap > 0) & np.isfinite(depthmap) & np.all(np.isfinite(pts_world), axis=-1)
    return pts_world.astype(np.float32), valid


def _sanitize_caption_text(raw, default_prompt=None):
    """Extract clean caption text from Syn4D caption strings.

    Matches recammaster caption_utils.sanitize_caption_text.
    """
    if raw is None:
        return default_prompt
    text = str(raw).strip()
    if len(text) == 0:
        return default_prompt
    prediction_match = re.search(r"Prediction:\s*(.*)", text, flags=re.DOTALL)
    if prediction_match is not None:
        prediction = prediction_match.group(1).strip()
        if len(prediction) > 0:
            return prediction
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith("Prompt:"):
            continue
        if line.startswith("Describe the video in detail."):
            continue
        cleaned_lines.append(line)
    if len(cleaned_lines) == 0:
        return default_prompt
    return cleaned_lines[-1]


def _parse_chunk_tag(tag):
    """Parse chunk tag "start-end" → (start, end)."""
    start_str, end_str = str(tag).split("-")
    start = int(start_str)
    end = int(end_str)
    if end < start:
        raise ValueError(f"Invalid chunk tag: {tag}")
    return start, end


def _load_depth_exr(depth_path):
    """Load depth from EXR file. Returns (H, W) float32 in metres, or None."""
    if not os.path.isfile(depth_path):
        return None
    try:
        import OpenEXR
        import Imath
        exr_file = OpenEXR.InputFile(depth_path)
        header = exr_file.header()
        dw = header["displayWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        channels = list(header["channels"].keys())
        ch = "Depth" if "Depth" in channels else channels[0]
        data = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(data, dtype=np.float32).reshape(h, w)
        return depth / 100.0  # cm → m
    except ImportError:
        pass
    except Exception:
        pass
    try:
        import cv2
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[..., 0]
        return np.asarray(depth, dtype=np.float32) / 100.0
    except Exception:
        return None


class Syn4DPairedCameraDataset(torch.utils.data.Dataset):
    """
    Paired camera dataset for a single Syn4D scene.

    Lazily loads RGB frames, camera poses, intrinsics, and 3D tracks from a
    Syn4D scene directory. Each __getitem__ call randomly selects:
      - a caption chunk and random start frame within it (continuous 49 frames)
      - source and target cameras

    This matches the recammaster training data pipeline.

    Args:
        scene_root: Root of the Syn4D scene (e.g., /path/to/Syn4D/arena)
        syn4d_root: Root of the Syn4D dataset (parent of scene dirs, contains caption/)
        num_cameras: Number of cameras per sequence (default 8, i.e. cam 0-7)
        num_frames: Number of temporal frames to use (default 49)
        height: Output image height (default 288)
        width: Output image width (default 512)
        track_spatial_stride: Spatial stride for dense track grid (default 1).
        steps_per_epoch: Synthetic epoch length (default 100)
        sequence_idx: Which sequence to use (default 0 = seq_000000_*)
        caption_chunk_size: Minimum chunk size for valid captions (default 81)
    """

    def __init__(
        self,
        scene_root: str,
        num_cameras: int = 8,
        num_frames: int = 49,
        height: int = 288,
        width: int = 512,
        track_spatial_stride: int = 1,
        steps_per_epoch: int = 100,
        sequence_idx: int = 0,
        syn4d_root: Optional[str] = None,
        # Legacy compat (ignored)
        stride: int = 1,
        caption_chunk_size: int = 81,
    ):
        self.scene_root = scene_root
        self.scene_name = os.path.basename(scene_root)
        self.syn4d_root = syn4d_root or os.path.dirname(scene_root)
        self.num_cameras = num_cameras
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.track_spatial_stride = track_spatial_stride
        self.steps_per_epoch = steps_per_epoch
        self.sequence_idx = sequence_idx
        self.seq_root = f"seq_{sequence_idx:06d}"

        # Directory roots
        self.png_root = os.path.join(scene_root, "png")
        self.depth_root = os.path.join(scene_root, "exr_layers", "depth")
        self.csv_root = os.path.join(scene_root, "ground_truth", "meta_exr_csv")

        # Discover cameras
        self.camera_views = self._find_camera_views()
        if len(self.camera_views) < 2:
            raise ValueError(
                f"Need at least 2 cameras in {scene_root} "
                f"(sequence {sequence_idx}), found {len(self.camera_views)}"
            )

        # Pre-load camera CSVs (small, ~477 rows each)
        self._camera_csv_cache: Dict[int, Dict[str, dict]] = {}
        for view in self.camera_views:
            self._camera_csv_cache[view["cam_id"]] = self._load_camera_csv(view["csv_path"])

        # Scan frame counts per view
        self._frame_maps: Dict[int, Dict[int, str]] = {}  # cam_id → {frame_id → path}
        min_num_frames = float("inf")
        for view in self.camera_views:
            frame_map = self._scan_png_frames(view["png_dir"])
            self._frame_maps[view["cam_id"]] = frame_map
            min_num_frames = min(min_num_frames, len(frame_map))
        self._min_num_frames = int(min_num_frames)

        # Load captions
        self._caption_cache = self._load_caption_json()

        # Max start frame for random window selection
        self._max_start = max(0, self._min_num_frames - self.num_frames)

        print(
            f"[Syn4DPaired] Scene '{self.scene_name}' seq {sequence_idx}: "
            f"{len(self.camera_views)} cameras, {self._min_num_frames} frames, "
            f"random start 0–{self._max_start}, "
            f"resolution {height}×{width}"
        )

    def _find_camera_views(self) -> List[dict]:
        """Find camera view directories for the given sequence index."""
        if not os.path.isdir(self.png_root):
            raise FileNotFoundError(f"No png/ directory in {self.scene_root}")

        seq_prefix = f"{self.seq_root}_"
        views = []
        for d in sorted(os.listdir(self.png_root)):
            if d.startswith(seq_prefix):
                cam_id = int(d.split("_")[-1])
                if cam_id >= self.num_cameras:
                    continue
                views.append({
                    "seq_name": d,
                    "cam_id": cam_id,
                    "png_dir": os.path.join(self.png_root, d),
                    "csv_path": os.path.join(self.csv_root, f"{d}_camera.csv"),
                    "depth_dir": os.path.join(self.depth_root, d),
                })
        return sorted(views, key=lambda x: x["cam_id"])

    def _load_camera_csv(self, csv_path) -> Dict[str, dict]:
        """Load per-frame camera params from CSV. Returns {frame_name → row_dict}."""
        if not os.path.isfile(csv_path):
            return {}
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        return {str(row["name"]): row for row in rows if row.get("name")}

    def _scan_png_frames(self, png_dir) -> Dict[int, str]:
        """Scan PNG directory. Returns {frame_id → full_path}."""
        frame_map = {}
        if not os.path.isdir(png_dir):
            return frame_map
        for name in os.listdir(png_dir):
            if not name.endswith(".png"):
                continue
            m = _PNG_FRAME_RE.match(name)
            if m is None:
                continue
            frame_id = int(m.group("frame"))
            frame_map[frame_id] = os.path.join(png_dir, name)
        return frame_map

    def _load_caption_json(self) -> dict:
        """Load per-video-chunks caption JSON for this scene."""
        caption_path = os.path.join(
            self.syn4d_root, "caption", f"{self.scene_name}_per_video_chunks.json"
        )
        if not os.path.isfile(caption_path):
            print(f"  [Syn4DPaired] WARN: No caption file at {caption_path}")
            return {}
        with open(caption_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_caption_chunks_for_view(self, cam_id) -> Dict[str, str]:
        """Get {chunk_tag → caption_text} for a camera view."""
        key = f"mp4/{self.seq_root}_{cam_id}.mp4"
        chunks = self._caption_cache.get(key, {})
        if not isinstance(chunks, dict):
            return {}
        return {str(k): str(v) for k, v in chunks.items()}

    def _select_frames(self) -> List[int]:
        """Pick a random contiguous frame window from the full available range.

        Uniformly samples start from [0, min_num_frames - num_frames].
        """
        max_start = max(0, self._min_num_frames - self.num_frames)
        start = random.randint(0, max_start)
        return list(range(start, start + self.num_frames))

    def _find_caption_for_frames(self, cam_id, frame_ids) -> str:
        """Find the best-matching caption chunk for a frame window.

        Picks the chunk with the most overlap with the selected frames.
        Falls back to DEFAULT_PROMPT if no captions available.
        """
        chunks = self._get_caption_chunks_for_view(cam_id)
        if not chunks:
            return DEFAULT_PROMPT

        frame_start = frame_ids[0]
        frame_end = frame_ids[-1]
        best_text = ""
        best_overlap = -1
        for tag, text in chunks.items():
            try:
                cs, ce = _parse_chunk_tag(tag)
            except Exception:
                continue
            overlap_start = max(frame_start, cs)
            overlap_end = min(frame_end, ce)
            overlap = max(0, overlap_end - overlap_start + 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_text = text
        return _sanitize_caption_text(best_text, default_prompt=DEFAULT_PROMPT)

    def _frame_name(self, cam_id, frame_id) -> str:
        """Construct the PNG filename for a given camera and frame."""
        # Syn4D frame names: seq_XXXXXX_N_YYYY.png (4-digit frame id)
        return f"{self.seq_root}_{cam_id}_{frame_id:04d}.png"

    def _load_frames(self, cam_id, frame_ids) -> List[Image.Image]:
        """Load and resize RGB frames for a camera view."""
        view = self.camera_views[cam_id]
        png_dir = view["png_dir"]
        frames = []
        for fid in frame_ids:
            fname = self._frame_name(cam_id, fid)
            path = os.path.join(png_dir, fname)
            img = Image.open(path).convert("RGB")
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), Image.LANCZOS)
            frames.append(img)
        return frames

    def _load_poses_and_intrinsics(self, cam_id, frame_ids):
        """Load camera poses and intrinsics for selected frames.

        Returns: cam2world (T, 4, 4) tensor, intrinsics (3, 3) tensor
        """
        cam_rows = self._camera_csv_cache[cam_id]

        poses = []
        intrinsics_computed = None
        for fid in frame_ids:
            fname = self._frame_name(cam_id, fid)
            row = cam_rows.get(fname)
            if row is not None:
                cam2world = np.eye(4, dtype=np.float32)
                cam2world[:3, :3] = _euler_to_rotation_matrix(
                    float(row["yaw"]), float(row["pitch"]), float(row["roll"])
                ).astype(np.float32)
                cam2world[:3, 3] = np.array(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype=np.float32,
                ) / 100.0
                poses.append(cam2world)

                if intrinsics_computed is None:
                    # Compute intrinsics at original resolution, scale to output
                    # Syn4D images are 1280×720; use sensor params for generality
                    focal = float(row["focal_length"])
                    sw = float(row["sensor_width"])
                    sh = float(row["sensor_height"])
                    # Original image resolution (constant per sequence)
                    orig_w, orig_h = 1280, 720
                    fx = (focal / sw) * orig_w
                    fy = (focal / sh) * orig_h
                    cx = orig_w / 2.0
                    cy = orig_h / 2.0
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                    K[0, :] *= self.width / orig_w
                    K[1, :] *= self.height / orig_h
                    intrinsics_computed = K
            else:
                poses.append(np.eye(4, dtype=np.float32))

        if intrinsics_computed is None:
            fx = fy = float(self.width)
            cx, cy = self.width / 2.0, self.height / 2.0
            intrinsics_computed = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )

        cam2world = torch.from_numpy(np.stack(poses)).float()
        intrinsics = torch.from_numpy(intrinsics_computed).float()
        return cam2world, intrinsics

    def _build_tracks(self, cam_id, frame_ids, cam2world, intrinsics):
        """Build dense world-space 3D tracks from depth maps for selected frames.

        Returns (N, T, 3) tensor.
        """
        view = self.camera_views[cam_id]
        depth_dir = view["depth_dir"]
        intrinsics_np = intrinsics.numpy()
        cam2world_np = cam2world.numpy()

        pts3d_frames = []
        for t, fid in enumerate(frame_ids):
            fname = self._frame_name(cam_id, fid)
            stem = os.path.splitext(fname)[0]
            depth_path = os.path.join(depth_dir, f"{stem}_depth.exr")
            depth = _load_depth_exr(depth_path)
            if depth is None:
                pts3d_frames = []
                break

            if depth.shape != (self.height, self.width):
                depth_pil = Image.fromarray(depth)
                depth_pil = depth_pil.resize((self.width, self.height), Image.NEAREST)
                depth = np.array(depth_pil)

            pts3d, valid = _depthmap_to_pts3d_world(
                depth, intrinsics_np, cam2world_np[t]
            )
            pts3d_frames.append(pts3d)

        if pts3d_frames:
            pts3d_all = np.stack(pts3d_frames, axis=0)  # (T, H, W, 3)
            pts3d_tensor = torch.from_numpy(pts3d_all)

            T, H, W, _ = pts3d_tensor.shape
            s = self.track_spatial_stride
            rows = torch.arange(0, H, s)
            cols = torch.arange(0, W, s)
            grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
            grid_r = grid_r.reshape(-1)
            grid_c = grid_c.reshape(-1)

            tracks = pts3d_tensor[:, grid_r, grid_c, :]  # (T, N, 3)
            tracks = tracks.permute(1, 0, 2).contiguous()  # (N, T, 3)
            tracks = torch.nan_to_num(tracks, nan=0.0, posinf=0.0, neginf=0.0)
            return tracks

        # Fallback: random dummy tracks
        T = self.num_frames
        s = self.track_spatial_stride
        Hg = (self.height + s - 1) // s
        Wg = (self.width + s - 1) // s
        N = Hg * Wg
        base = torch.randn(N, 1, 3) * 3.0
        return base + torch.randn(N, T, 3) * 0.1

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        """Return a paired (source, target) camera sample with random frame window."""
        num_cams = len(self.camera_views)

        # Random source and target cameras
        src_idx = random.randint(0, num_cams - 1)
        tgt_idx = random.randint(0, num_cams - 2)
        if tgt_idx >= src_idx:
            tgt_idx += 1

        src_cam_id = self.camera_views[src_idx]["cam_id"]
        tgt_cam_id = self.camera_views[tgt_idx]["cam_id"]

        # Random contiguous frame window from full range
        frame_ids = self._select_frames()

        # Load data for source and target
        src_frames = self._load_frames(src_cam_id, frame_ids)
        tgt_frames = self._load_frames(tgt_cam_id, frame_ids)

        src_cam2world, src_intrinsics = self._load_poses_and_intrinsics(src_cam_id, frame_ids)
        tgt_cam2world, tgt_intrinsics = self._load_poses_and_intrinsics(tgt_cam_id, frame_ids)

        # Build tracks from source camera
        tracks_world = self._build_tracks(src_cam_id, frame_ids, src_cam2world, src_intrinsics)

        # Find best-matching caption for the selected frame window
        prompt = self._find_caption_for_frames(src_cam_id, frame_ids)

        return {
            "video": src_frames,              # list of PIL Images
            "target_video": tgt_frames,        # list of PIL Images
            "tracks_world": tracks_world,      # (N, T, 3)
            "target_cam2world": tgt_cam2world, # (T, 4, 4)
            "target_intrinsics": tgt_intrinsics, # (3, 3)
            "prompt": prompt,
        }


class DummySyn4DPairedDataset(torch.utils.data.Dataset):
    """
    Dummy dataset for CPU testing. Generates random paired camera data.

    Track density matches the real dataset: dense grid with
    ``track_spatial_stride`` (default 1 → H*W tracks).
    """

    def __init__(
        self,
        num_frames=49,
        height=288,
        width=512,
        track_spatial_stride=1,
        steps_per_epoch=100,
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.track_spatial_stride = track_spatial_stride
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        T = self.num_frames
        H, W = self.height, self.width
        s = self.track_spatial_stride
        Hg = (H + s - 1) // s
        Wg = (W + s - 1) // s
        N = Hg * Wg

        # Random RGB frames as PIL images
        src_frames = [
            Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
            for _ in range(T)
        ]
        tgt_frames = [
            Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
            for _ in range(T)
        ]

        # Random camera poses (identity + small translation)
        cam2world = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
        cam2world[:, :3, 3] = torch.randn(T, 3) * 0.5

        # Simple pinhole intrinsics
        fx = fy = float(W)
        cx, cy = W / 2.0, H / 2.0
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

        # Dense grid of 3D tracks (world space)
        base_pts = torch.randn(N, 1, 3) * 2.0
        tracks = base_pts + torch.randn(N, T, 3) * 0.05

        return {
            "video": src_frames,
            "target_video": tgt_frames,
            "tracks_world": tracks,
            "target_cam2world": cam2world,
            "target_intrinsics": K,
            "prompt": "Generate the target view video.",
        }
