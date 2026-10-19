"""
ScanNet_Multi with static tracking data for training alongside dynamic datasets.

Uses StaticTrackMixin to add track + trackvid to ScanNet_Multi.
See static_track_mixin.py for details.

Usage:
    dataset = ScanNet_Multi_WithStaticTrack(
        ROOT='path/to/processed_scannet/',
        split='train',
        num_views=49,
        resolution=[[256, 256]],
        transform=ImgNorm,
        return_trackvid=True,
    )
"""

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.static_track_mixin import StaticTrackMixin
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.scannet import ScanNet_Multi


class ScanNet_Multi_WithStaticTrack(StaticTrackMixin, ScanNet_Multi):
    """ScanNet with static tracking: track = frame-0 pts3d, residual = 0."""
    pass


__all__ = ["ScanNet_Multi_WithStaticTrack"]
