"""
Kubric 81-frame XYZVid(frame0-colorized) dataset with random 49-frame window sampling.

This class combines:
  - KubricMultiCameraProcessed_81Frames: random contiguous sub-window sampling from 82-frame sequences
  - KubricMultiCameraProcessed_XYZVidFrame0GT: frame0-colorized track target generation

Result:
  - Source sequences come from 81/82-frame preprocessed Kubric data.
  - Each sample returns `num_views` contiguous frames (e.g. 49) with random start offset.
  - `track` is replaced by XYZVid(frame0-colorized), and optional dynamic track is preserved.
"""

from __future__ import annotations

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.kubric_multicam_processed_81frames import (
    KubricMultiCameraProcessed_81Frames,
)
from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.kubric_multicam_processed_xyzvid_frame0_gt import (
    KubricMultiCameraProcessed_XYZVidFrame0GT,
)


class KubricMultiCameraProcessed_XYZVidFrame0GT_81Frames(
    KubricMultiCameraProcessed_XYZVidFrame0GT,
    KubricMultiCameraProcessed_81Frames,
):
    """
    Use 81-frame random window sampling + XYZVid(frame0-colorized) targets.

    MRO intentionally places XYZVid class first so its __getitem__ post-processing runs,
    while 81Frames contributes the random-window `_get_views` implementation.
    """

    pass


__all__ = ["KubricMultiCameraProcessed_XYZVidFrame0GT_81Frames"]
