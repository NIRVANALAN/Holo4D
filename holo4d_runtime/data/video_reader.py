"""
decord/imageio compatibility shim.

This repo depends on decord for video I/O in dataloaders, but decord is hard
to install on some platforms (e.g. new GLIBC, ARM). We try decord first and
fall back to an imageio-based wrapper with the same API.

Public API — the returned VideoReader class supports:
    vr = VideoReader(path, num_threads=...)   # num_threads ignored by fallback
    len(vr)                                    # total frames
    vr.get_avg_fps()                           # float fps
    vr.get_batch(indices)                      # object with .asnumpy() → (T,H,W,3) uint8

Both backends must return byte-identical frame data given the same path and
indices. imageio uses ffmpeg under the hood (same as decord), so decoded
pixels match up to the usual ffmpeg seek-keyframe precision.
"""

from __future__ import annotations

import os
import numpy as np

__all__ = ["VideoReader", "VIDEO_BACKEND"]


try:
    from decord import VideoReader as _DecordVideoReader  # type: ignore

    VideoReader = _DecordVideoReader
    VIDEO_BACKEND = "decord"

except Exception as _decord_err:  # pragma: no cover — platform-dependent
    # --- imageio fallback ---------------------------------------------------
    import imageio.v2 as imageio  # v2 has get_reader + count_frames

    class _BatchResult:
        """Mimics decord's NDArray with an .asnumpy() method."""

        __slots__ = ("_arr",)

        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def asnumpy(self) -> np.ndarray:
            return self._arr

        def __array__(self, dtype=None):  # allow np.asarray(batch)
            return self._arr if dtype is None else self._arr.astype(dtype)

    class _ImageioVideoReader:
        """imageio-backed VideoReader that exposes decord's API subset.

        Only implements the methods actually used in this repo:
        __len__, get_batch, get_avg_fps. Extra kwargs (num_threads, ctx, ...)
        are accepted and silently ignored to stay call-compatible with decord.
        """

        def __init__(self, uri, *args, num_threads: int = 1, ctx=None, **kwargs):
            # imageio accepts str paths, file-like objects, or bytes.
            self._uri = uri
            self._reader = imageio.get_reader(uri, "ffmpeg")
            # Cache frame count + fps since imageio probes these lazily.
            try:
                self._nframes = int(self._reader.count_frames())
            except Exception:
                meta = self._reader.get_meta_data()
                self._nframes = int(meta.get("nframes", 0))
            meta = self._reader.get_meta_data()
            self._fps = float(meta.get("fps", 0.0))

        def __len__(self) -> int:
            return self._nframes

        def get_avg_fps(self) -> float:
            return self._fps

        def get_batch(self, indices) -> _BatchResult:
            # Accept list/tuple/ndarray/torch.Tensor-like — anything iterable.
            if hasattr(indices, "tolist"):
                indices = indices.tolist()
            frames = []
            for idx in indices:
                idx = int(idx)
                if idx < 0:
                    idx += self._nframes
                frames.append(self._reader.get_data(idx))  # (H, W, 3) uint8 RGB
            return _BatchResult(np.stack(frames, axis=0))

        # Let GC release the reader deterministically when wrapper is deleted.
        def __del__(self):
            try:
                self._reader.close()
            except Exception:
                pass

    VideoReader = _ImageioVideoReader
    VIDEO_BACKEND = "imageio"
    # Surface one warning on import so users know decord is missing.
    import warnings as _warnings
    _warnings.warn(
        f"decord unavailable ({_decord_err.__class__.__name__}: {_decord_err}); "
        "falling back to imageio for video I/O. Install decord for faster loading.",
        stacklevel=2,
    )
