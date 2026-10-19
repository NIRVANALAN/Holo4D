"""
MVS_Synth_Multi with static tracking data for residual-only training.

Uses StaticTrackMixin to add track + optional trackvid to MVS_Synth_Multi.
See static_track_mixin.py for details.
"""

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.mvs_synth import MVS_Synth_Multi
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.static_track_mixin import StaticTrackMixin


class MVS_Synth_Multi_WithStaticTrack(StaticTrackMixin, MVS_Synth_Multi):
    """MVS-Synth with static tracking: track = frame-0 pts3d, residual = 0."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("trackvid_norm_mode", "shared_absmax")
        super().__init__(*args, **kwargs)


__all__ = ["MVS_Synth_Multi_WithStaticTrack"]
