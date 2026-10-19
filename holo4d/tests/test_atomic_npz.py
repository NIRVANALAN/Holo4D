from __future__ import annotations

import numpy as np

from holo4d.utils.atomic_npz import atomic_savez_compressed, is_readable_npz


def test_readable_npz_rejects_corrupt_archive(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    path.write_bytes(b"interrupted write")
    assert not is_readable_npz(path)


def test_atomic_savez_replaces_corrupt_archive(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    path.write_bytes(b"interrupted write")
    atomic_savez_compressed(path, values=np.arange(6, dtype=np.float32))
    assert is_readable_npz(path)
    with np.load(path) as archive:
        np.testing.assert_array_equal(
            archive["values"], np.arange(6, dtype=np.float32)
        )
    assert not list(tmp_path.glob("*.tmp.npz"))
