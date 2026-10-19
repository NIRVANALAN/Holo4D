"""
Compatibility wrapper for Kubric tracking datasets.

Historically this repo has used two variants for Kubric dense tracking:
- `Kubric_Tracks` (in kubric_movif_track.py): the original, working dataset for MoVi-F style tracks.
- `KubricWithTracks`: a name referenced by configs / scripts.

This file keeps the `KubricWithTracks` name, but implements it by subclassing
`Kubric_Tracks` and only adding small compatibility glue:
- Accept `allow_repeat` and `num_views` arguments (as used by Stream3R dataset strings)
- Accept `resolution=[[W,H], ...]` + `num_views=N` and convert it to the stereo-style
  `resolution=[([W,H], N), ...]` expected by BaseStereoViewDataset.
- Map `split="validation"` to `split="test"` for datasets that only define train/test.
"""

from __future__ import annotations

from typing import Any

from .kubric_movif_track import Kubric_Tracks


def _coerce_resolution_with_num_views(resolution, num_views: int):
    if resolution is None:
        return None
    if not isinstance(resolution, list):
        return resolution
    coerced = []
    for res in resolution:
        # Common Stream3R config style: [W, H]
        if isinstance(res, (list, tuple)) and len(res) == 2 and all(isinstance(x, int) for x in res):
            w, h = int(res[0]), int(res[1])
            coerced.append(([w, h], int(num_views)))
        else:
            coerced.append(res)
    return coerced


class KubricWithTracks(Kubric_Tracks):
    def __init__(
        self,
        *args: Any,
        ROOT: str,
        split: str = "train",
        resolution=None,
        num_views: int | None = None,
        allow_repeat: bool = False,
        overfitting: bool = False,
        **kwargs: Any,
    ):
        if split == "validation":
            split = "test"

        if num_views is not None:
            resolution = _coerce_resolution_with_num_views(resolution, int(num_views))

        super().__init__(
            *args,
            ROOT=ROOT,
            split=split,
            resolution=resolution,
            allow_repeat=allow_repeat,
            **kwargs,
        )

        if num_views is not None:
            self.num_views = int(num_views)

        print(f"[KubricWithTracks] has {len(self.sceneids)} scenes")

        if overfitting and hasattr(self, "sceneids") and len(self.sceneids) > 0:
            self.sceneids = self.sceneids[:1]
        else:
            self.sceneids = self.sceneids[:-5] # leave some for validation
        
        print(f"[KubricWithTracks] will use {len(self.sceneids)} scenes")


__all__ = ["KubricWithTracks"]

