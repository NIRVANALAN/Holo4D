from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DAVIS_TRAJECTORIES_13579_ZERO_BASED = (
    "pan_right",
    "pan_down",
    "translate_right",
    "zoom_out",
    "orbit_right",
)


@dataclass(frozen=True)
class DavisSequence:
    name: str
    frame_paths: tuple[Path, ...]


def split_csv(spec: str | None) -> tuple[str, ...]:
    if spec is None:
        return ()
    return tuple(item.strip() for item in str(spec).split(",") if item.strip())


def _sorted_image_paths(sequence_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(sequence_dir.glob(suffix))
    return sorted(paths, key=lambda path: path.name)


def discover_davis_sequences(
    davis_root: str | Path,
    *,
    num_frames: int = 49,
    start_frame: int = 0,
) -> list[DavisSequence]:
    root = Path(davis_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"DAVIS root not found: {root}")

    needed = int(start_frame) + int(num_frames)
    sequences: list[DavisSequence] = []
    for sequence_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        frames = _sorted_image_paths(sequence_dir)
        if len(frames) < needed:
            continue
        window = tuple(frames[int(start_frame):needed])
        sequences.append(DavisSequence(sequence_dir.name, window))
    return sequences


def resolve_davis_sequences(
    davis_root: str | Path,
    *,
    sequence_names: str | None = None,
    num_frames: int = 49,
    start_frame: int = 0,
) -> list[DavisSequence]:
    sequences = discover_davis_sequences(
        davis_root,
        num_frames=num_frames,
        start_frame=start_frame,
    )
    requested = set(split_csv(sequence_names))
    if not requested:
        return sequences

    selected = [sequence for sequence in sequences if sequence.name in requested]
    missing = sorted(requested - {sequence.name for sequence in selected})
    if missing:
        raise ValueError(
            f"Requested DAVIS sequences are missing or shorter than {num_frames} frames: {missing}"
        )
    return selected


def davis_sequence_names_csv(
    davis_root: str | Path,
    *,
    sequence_names: str | None = None,
    num_frames: int = 49,
    start_frame: int = 0,
) -> str:
    return ",".join(
        sequence.name
        for sequence in resolve_davis_sequences(
            davis_root,
            sequence_names=sequence_names,
            num_frames=num_frames,
            start_frame=start_frame,
        )
    )
