"""Small helpers for resumable NPZ cache producers."""

from __future__ import annotations

import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def is_readable_npz(path: str | Path) -> bool:
    """Return true only when *path* has a readable, non-empty ZIP directory."""

    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=True) as archive:
            return bool(archive.files)
    except (EOFError, OSError, ValueError, pickle.UnpicklingError, zipfile.BadZipFile):
        return False


def atomic_savez_compressed(path: str | Path, **arrays: Any) -> None:
    """Write a compressed NPZ beside its destination, then atomically replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        if not is_readable_npz(temporary):
            raise RuntimeError(f"temporary NPZ failed validation: {temporary}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
