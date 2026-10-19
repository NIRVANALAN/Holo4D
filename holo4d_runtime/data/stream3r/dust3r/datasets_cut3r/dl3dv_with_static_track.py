"""
DL3DV_Multi with static tracking data for training alongside dynamic datasets.

Uses StaticTrackMixin to add track + trackvid to DL3DV_Multi.
See static_track_mixin.py for details.

Usage:
    dataset = DL3DV_Multi_WithStaticTrack(
        ROOT='path/to/processed_dl3dv/',
        split='train',
        num_views=49,
        resolution=[[256, 256]],
        transform=ImgNorm,
        return_trackvid=True,
    )
"""

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.static_track_mixin import StaticTrackMixin
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.dl3dv import DL3DV_Multi


class DL3DV_Multi_WithStaticTrack(StaticTrackMixin, DL3DV_Multi):
    """DL3DV with static tracking: track = frame-0 pts3d, residual = 0."""
    pass


__all__ = ["DL3DV_Multi_WithStaticTrack"]
