"""Small camera-matrix conversion helpers used by benchmark loaders."""

from __future__ import annotations

import numpy as np

def k4_from_k33(K):
    K = np.asarray(K, dtype=np.float32)
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)


def make_relative_c2w(c2w_abs):
    c2w_abs = np.asarray(c2w_abs, dtype=np.float32)
    if c2w_abs.ndim != 3 or c2w_abs.shape[1:] != (4, 4):
        raise ValueError(f"Expected (T,4,4) camera poses, got {tuple(c2w_abs.shape)}")
    w2c0 = np.linalg.inv(c2w_abs[0]).astype(np.float32)
    return np.stack([w2c0 @ c2w_abs[t] for t in range(c2w_abs.shape[0])], axis=0).astype(np.float32)
