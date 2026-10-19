"""Parse ReCamMaster CoWTracker and VGGT-Omega cache names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class LabeledReCamSample:
    rel_dir: str
    source_cam: str
    start_frame: int
    cow_npz: str
    source_json: str
    all10_json: str


def parse_cow_track_npz_path(path: Path, source_tracks_root: Path) -> LabeledReCamSample:
    rel = path.relative_to(source_tracks_root)
    match = re.fullmatch(r"(cam\d+)_start(\d{4})_pred_trackvideo\.npz", rel.name)
    if match is None:
        raise ValueError(f"unexpected CoWTracker filename: {path}")
    rel_dir = rel.parent.as_posix()
    source_cam = match.group(1)
    start_frame = int(match.group(2))
    return LabeledReCamSample(
        rel_dir=rel_dir,
        source_cam=source_cam,
        start_frame=start_frame,
        cow_npz=str(path),
        source_json="",
        all10_json="",
    )


def _parse_rel_frame_values(value: str) -> list[int]:
    raw = str(value).strip()
    if ":" in raw:
        parts = [int(p) for p in raw.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"bad rel-frame range: {value}")
        return list(range(start, stop, step))
    return [int(p) for p in re.split(r"[,| ]+", raw) if p]


def all10_rel_frames_tag(value: str) -> str:
    frames = _parse_rel_frame_values(value)
    if not frames:
        raise ValueError("empty all10 rel-frame list")
    if len(frames) >= 2:
        diffs = [b - a for a, b in zip(frames[:-1], frames[1:])]
        if len(set(diffs)) == 1:
            return f"rel{frames[0]:03d}-{frames[-1]:03d}s{diffs[0]}"
    return "rel" + "-".join(f"{f:03d}" for f in frames)
