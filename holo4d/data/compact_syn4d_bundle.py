"""Metadata-free Syn4D bundle adapter for compact smoke-test runs.

The full Syn4D loader reconstructs compact face-id tracks from the shared mesh
metadata tree.  The frozen evaluation bundle already contains the resulting
source tracks, RGB clips, and target camera arrays, so this adapter deliberately
loads only those audited artifacts.  It is restricted to the 49-frame source-0
bundle contract and is not a replacement for the training loader.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


class CompactSyn4DBundle:
    """Load one frozen flying_group source-0 pair without mesh metadata."""

    def __init__(self, bundle_root, *, sequence_idx=0, num_frames=49):
        self.bundle_root = Path(bundle_root)
        self.sequence_idx = int(sequence_idx)
        self.num_frames = int(num_frames)
        if self.num_frames != 49:
            raise ValueError("compact Syn4D bundle currently requires num_frames=49")
        self.base = f"seq_{self.sequence_idx:06d}"
        self.pair_prefix = f"flying_group_{self.base}_src0_tgt"
        self.source_clip = self.bundle_root / "assets" / "source_clips" / f"{self.base}_src0_f5_49f.mp4"
        if not self.source_clip.is_file():
            raise FileNotFoundError(self.source_clip)
        self._nvs_groups = [{
            "scene_name": "flying_group",
            "base_seq": self.base,
            "camera_names": [f"{self.base}_{i}" for i in range(8)],
            "common_starts": [5],
        }]
        self._source_frames = None

    def _read(self, path: Path):
        frames = [np.asarray(frame)[..., :3].astype(np.uint8, copy=False) for frame in iio.imiter(path)]
        if len(frames) != self.num_frames:
            raise ValueError(f"{path}: expected {self.num_frames} frames, got {len(frames)}")
        return frames

    def _condition(self, backend):
        if backend:
            path = self.bundle_root / "ours_geometry_backends" / "canonical" / backend / self.base / "source_condition.npz"
            with np.load(path, allow_pickle=False) as data:
                return {key: np.asarray(data[key]) for key in data.files}, path
        path = self.bundle_root / "geometry" / self.base / "ours_source_tracks_49f.npz"
        with np.load(path, allow_pickle=False) as data:
            result = {key: np.asarray(data[key]) for key in data.files}
        return result, path

    def load_pair(self, src_cam_idx, tgt_cam_idx, start_frame, group_idx=0, track_query_idx=0, window_start_frame=None):
        if int(src_cam_idx) != 0:
            raise ValueError("compact Syn4D bundle contains audited source camera 0 only")
        if int(group_idx) != 0 or int(start_frame) != 5 or int(track_query_idx) != 0:
            raise ValueError("compact bundle contract is group=0, start_frame=5, query=0")
        tgt = int(tgt_cam_idx)
        if not 1 <= tgt <= 7:
            raise ValueError(f"target camera must be 1..7, got {tgt}")
        if self._source_frames is None:
            self._source_frames = self._read(self.source_clip)
        pair = self.bundle_root / "gt" / f"{self.pair_prefix}{tgt}" / "target" / "gt.mp4"
        target_frames = self._read(pair)
        camera_path = self.bundle_root / "cameras" / f"{self.pair_prefix}{tgt}" / "target.npz"
        with np.load(camera_path, allow_pickle=False) as camera:
            target_c2w = np.asarray(camera["cam_c2w"], dtype=np.float32)
            target_k = np.asarray(camera["intrinsics"], dtype=np.float32)
        gt, _ = self._condition(None)
        tracks = np.asarray(gt["tracks_world"], dtype=np.float32)
        valid = np.asarray(gt["tracks_world_valid_mask"], dtype=bool)
        source_c2w = np.asarray(gt["source_cam2world"], dtype=np.float32)
        source_k = np.asarray(gt["source_intrinsics"], dtype=np.float32)
        if source_k.ndim == 2:
            source_k = np.repeat(source_k[None], self.num_frames, axis=0)
        depth = np.asarray(gt["source_depth"], dtype=np.float32)
        if depth.ndim == 2:
            depth = np.repeat(depth[None], self.num_frames, axis=0)
        n, h, w = tracks.shape[0], self._source_frames[0].shape[0], self._source_frames[0].shape[1]
        if n != h * w:
            raise ValueError(f"GT track grid {n} does not match source frame {h}x{w}")
        first = self._source_frames[0].reshape(-1, 3).astype(np.float32) / 255.0
        source_rgb_at_tracks = np.broadcast_to(first[:, None, :], (n, self.num_frames, 3)).copy()
        caption_path = self.bundle_root / "baselines" / "iphone_captions.json"
        captions = json.loads(caption_path.read_text()) if caption_path.is_file() else {}
        pair_id = f"{self.pair_prefix}{tgt}"
        return {
            "video": self._source_frames,
            "source_rgb_video": np.asarray(self._source_frames, dtype=np.float32) / 255.0,
            "target_video": target_frames,
            "target_video_is_ground_truth": True,
            "tracks_world": tracks,
            "tracks_world_valid_mask": valid,
            "tracks_query_idx_per_identity": np.zeros((n,), dtype=np.int32),
            "track_query_idx": 0,
            "track_query_indices": [0],
            "track_query_count": 1,
            "target_cam2world": target_c2w,
            "target_intrinsics": target_k,
            "source_cam2world": source_c2w,
            "source_intrinsics": source_k,
            "source_depth": depth[0],
            "source_depth_video": depth,
            "source_rgb_at_tracks": source_rgb_at_tracks,
            "prompt": captions.get(pair_id, "Generate the requested target-view video from the source video."),
            "common_scale": float(np.asarray(gt.get("common_scale", 1.0)).item()),
            "has_tracks": True,
            "has_dynamic_tracks": True,
            "has_depth": True,
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "src_cam_name": f"{self.base}_0",
            "tgt_cam_name": f"{self.base}_{tgt}",
        }
