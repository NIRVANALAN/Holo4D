"""
TartanAir_Multi with static tracking data for training alongside dynamic datasets.

Uses StaticTrackMixin to add track + trackvid to TartanAir_Multi.
See static_track_mixin.py for details.

Usage:
    dataset = TartanAir_Multi_WithStaticTrack(
        ROOT='path/to/processed_tartanair/',
        num_views=49,
        resolution=[[256, 256]],
        transform=ImgNorm,
        return_trackvid=True,
    )
"""

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.static_track_mixin import StaticTrackMixin
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.tartanair import TartanAir_Multi


class TartanAir_Multi_WithStaticTrack(StaticTrackMixin, TartanAir_Multi):
    """TartanAir with static tracking: track = frame-0 pts3d, residual = 0."""
    def __init__(self, *args, **kwargs):
        # Keep XYZ colorization physically consistent across channels:
        # one shared scale for X/Y/Z from frame-0 (no per-channel quantile normalization).
        kwargs.setdefault("trackvid_norm_mode", "shared_absmax")
        super().__init__(*args, **kwargs)


__all__ = ["TartanAir_Multi_WithStaticTrack"]
