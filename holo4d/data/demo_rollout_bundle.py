"""Load the self-contained two-window Syn4D demo bundle."""
from __future__ import annotations

from pathlib import Path

import numpy as np


FORMAT = "holo4d_syn4d_flying_group_2chunk_demo_v1"


class DemoRolloutBundle:
    """Expose one audited 49 + 48 frame pair through the Syn4D loader contract."""

    def __init__(self, bundle_path, *, sequence_idx=0, num_frames=49):
        self.bundle_path = Path(bundle_path)
        self.sequence_idx = int(sequence_idx)
        self.num_frames = int(num_frames)
        if self.sequence_idx != 0 or self.num_frames != 49:
            raise ValueError("demo bundle supports only sequence 0 with 49-frame chunks")
        if not self.bundle_path.is_file():
            raise FileNotFoundError(self.bundle_path)
        self._archive = np.load(self.bundle_path, allow_pickle=False)
        actual = str(np.asarray(self._archive["format"]).item())
        if actual != FORMAT:
            raise ValueError(f"{self.bundle_path}: unsupported format {actual!r}")
        self._nvs_groups = [{
            "scene_name": "flying_group",
            "base_seq": "seq_000000",
            "camera_names": ["seq_000000_0", "seq_000000_1"],
            # The seed starts at 5. The continuation window begins at 53 and
            # uses its first available tracking anchor, absolute frame 55.
            "common_starts": [5, 55],
        }]

    def _value(self, prefix: str, name: str, *, dtype=None):
        result = np.asarray(self._archive[f"{prefix}_{name}"])
        return result.astype(dtype, copy=False) if dtype is not None else result

    def load_pair(
        self,
        src_cam_idx,
        tgt_cam_idx,
        start_frame,
        group_idx=0,
        track_query_idx=0,
        window_start_frame=None,
    ):
        if (int(src_cam_idx), int(tgt_cam_idx), int(group_idx)) != (0, 1, 0):
            raise ValueError("demo bundle contains only flying_group camera pair 0->1")
        request = (int(start_frame), int(track_query_idx), window_start_frame)
        if request == (5, 0, None):
            prefix = "seed"
        elif request == (55, 2, 53):
            prefix = "continuation"
        else:
            raise ValueError(
                "demo bundle supports seed(start=5, query=0) or "
                "continuation(window_start=53, anchor=55, query=2); "
                f"got {request}"
            )
        video = self._value(prefix, "source_video", dtype=np.uint8)
        target = self._value(prefix, "target_video", dtype=np.uint8)
        depth = self._value(prefix, "source_depth_video", dtype=np.float32)
        tracks = self._value(prefix, "tracks_world", dtype=np.float32)
        n = tracks.shape[0]
        return {
            "video": [frame for frame in video],
            "target_video": [frame for frame in target],
            "target_video_is_ground_truth": True,
            "tracks_world": tracks,
            "target_cam2world": self._value(prefix, "target_cam2world", dtype=np.float32),
            "target_intrinsics": self._value(prefix, "target_intrinsics", dtype=np.float32),
            "common_scale": float(self._value(prefix, "common_scale").item()),
            "track_query_idx": int(track_query_idx),
            "track_query_indices": [int(track_query_idx)],
            "track_query_count": 1,
            "prompt": str(self._value(prefix, "prompt").item()),
            "target_prompt": str(self._value(prefix, "target_prompt").item()),
            "src_cam_name": "seq_000000_0",
            "tgt_cam_name": "seq_000000_1",
            "reproj_src_depth_at_target": self._value(
                prefix, "reproj_src_depth_at_target", dtype=np.float32
            ),
            "reproj_src_depth_valid": self._value(
                prefix, "reproj_src_depth_valid", dtype=bool
            ),
            "has_tracks": True,
            "has_dynamic_tracks": True,
            "has_depth": False,
            "target_depth": None,
            "source_depth": depth[0],
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "source_rgb_at_tracks": self._value(
                prefix, "source_rgb_at_tracks", dtype=np.float32
            ),
            "source_cam2world": self._value(prefix, "source_cam2world", dtype=np.float32),
            "source_intrinsics": self._value(prefix, "source_intrinsics", dtype=np.float32),
            "source_depth_video": depth,
            "tracks_query_idx_per_identity": np.full(
                (n,), int(track_query_idx), dtype=np.int32
            ),
        }
