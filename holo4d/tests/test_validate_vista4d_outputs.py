import csv

import imageio.v2 as imageio
import numpy as np
import pytest

from holo4d.tools.validate_vista4d_outputs import decodable_frames, requested_pairs


def _write_metadata(path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "camera"])
        writer.writeheader()
        writer.writerows(
            [
                {"video": "scene-b", "camera": "orbit-2"},
                {"video": "scene-a", "camera": "orbit-1"},
                {"video": "scene-a", "camera": "orbit-2"},
                {"video": "scene-a", "camera": "orbit-3"},
            ]
        )


def test_requested_pairs_are_filtered_limited_and_stable(tmp_path):
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata)

    assert requested_pairs(metadata, {"scene-a"}, max_cameras=2) == [
        ("scene-a", "orbit-1"),
        ("scene-a", "orbit-2"),
    ]
    with pytest.raises(ValueError, match="absent from metadata"):
        requested_pairs(metadata, {"missing-scene"}, max_cameras=2)


def test_decodable_frames_counts_the_complete_video(tmp_path):
    video = tmp_path / "sample.mp4"
    writer = imageio.get_writer(video, fps=8, macro_block_size=None)
    try:
        for value in range(4):
            writer.append_data(np.full((16, 16, 3), value * 32, dtype=np.uint8))
    finally:
        writer.close()

    assert decodable_frames(video) == 4
