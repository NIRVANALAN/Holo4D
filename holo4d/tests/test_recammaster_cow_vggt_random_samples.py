from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holo4d.data.recammaster_cache_names import (
    all10_rel_frames_tag,
    parse_cow_track_npz_path,
)


def test_parse_cow_track_npz_path_extracts_labeled_sample_identity():
    root = Path("/cache/cow/source_tracks")
    path = root / "train/f18_aperture10/scene100/cam07_start0029_pred_trackvideo.npz"

    sample = parse_cow_track_npz_path(path, root)

    assert sample.rel_dir == "train/f18_aperture10/scene100"
    assert sample.source_cam == "cam07"
    assert sample.start_frame == 29


def test_all10_rel_frames_tag_matches_production_cache_names():
    assert all10_rel_frames_tag("0:49:8") == "rel000-048s8"
    assert all10_rel_frames_tag("0,8,16,24,32,40,48") == "rel000-048s8"
