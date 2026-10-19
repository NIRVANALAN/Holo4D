"""The opt-in 4RC mask fusion reuses CoW's static/dynamic track semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from holo4d.benchmark.cow_vggt_static_fusion import fuse_static_background_valid_mask


EVAL_DIR = Path(__file__).resolve().parents[2] / "examples" / "wan2.1_fun"
sys.path.insert(0, str(EVAL_DIR))
from eval_track4d360_davis_4rc_novel_traj import _freeze_static_track_identities  # noqa: E402


def test_query_background_is_frozen_and_persistent_while_foreground_moves() -> None:
    tracks = np.array(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    dynamic = np.array([[False] * 3, [True] * 3])
    valid = np.array([[True, False, False], [True, True, False]])
    query = np.zeros((2,), dtype=np.int32)

    fused_tracks = _freeze_static_track_identities(tracks, dynamic, query)
    fused_valid = fuse_static_background_valid_mask(
        valid, dynamic, tracks.shape[:2], query, tracks_world=fused_tracks
    )

    np.testing.assert_array_equal(fused_tracks[0], np.repeat(tracks[0, :1], 3, axis=0))
    np.testing.assert_array_equal(fused_tracks[1], tracks[1])
    np.testing.assert_array_equal(fused_valid[0], [True, True, True])
    np.testing.assert_array_equal(fused_valid[1], valid[1])
