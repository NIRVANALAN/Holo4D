"""
KubricMultiCameraProcessed wrapper that replaces the training target with the
"XYZ video (frame0-colorized)" used by:

  bash_scripts/setup_pipeline/v_dpm/visualize_script/visualize_kubric_xyzVid-frame0_vae_roundtrip_newdata.sh

Target definition (per-frame, per-pixel):
  - Each frame shows the current visible surface (so it changes over time),
  - But each visible object point is colored by its frame-0 XYZ value (in frame-0 camera coords),
    using canonical object coordinates (`obj_coords`) + instance segmentation (`segm`) for correspondence.

Implementation notes:
  - This class does NOT modify KubricMultiCameraProcessed; it subclasses and post-processes its outputs.
  - We write the target into `view["track"]` / `view["track_valid_mask"]` so the existing V2V training
    pipeline can keep using `--extra_inputs track` as the diffusion target.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.kubric_multicam_processed import KubricMultiCameraProcessed
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import geotrf as dust3r_geotrf
from holo4d_runtime.data.stream3r.dust3r.utils.geometry import inv as dust3r_inv


def _to_torch(x, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


class KubricMultiCameraProcessed_XYZVidFrame0GT(KubricMultiCameraProcessed):
    """
    Produces standard Stream3R views, but replaces `track` with the frame0-colorized XYZ video GT.
    """

    def __init__(
        self,
        *args,
        preserve_original_track: bool = False,
        track_color_norm_mode: str = "quantile",
        track_color_quantile_low: float = 0.02,
        track_color_quantile_high: float = 0.98,
        track_color_knn_backend: str = "ckdtree",
        cupy_query_block: int = 4096,
        cupy_ref_block: int = 4096,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.preserve_original_track = bool(preserve_original_track)
        self.track_color_norm_mode = str(track_color_norm_mode)
        self.track_color_quantile_low = float(track_color_quantile_low)
        self.track_color_quantile_high = float(track_color_quantile_high)
        self.track_color_knn_backend = str(track_color_knn_backend or "ckdtree").lower()
        self.cupy_query_block = int(cupy_query_block)
        self.cupy_ref_block = int(cupy_ref_block)
        self._cupy_backend_warns = 0
        print(
            "[KubricMultiCameraProcessed_XYZVidFrame0GT] Using XYZVid(frame0-colorized) as `track` target. "
            f"track_color_norm_mode={self.track_color_norm_mode!r}, "
            f"q=({self.track_color_quantile_low:g},{self.track_color_quantile_high:g}), "
            f"knn_backend={self.track_color_knn_backend!r}"
        )

    @staticmethod
    def _import_cupy():
        try:
            import cupy as cp
            # Some environments have a namespace-stub `cupy` package without CUDA bindings.
            if getattr(cp, "__file__", None) is None or not hasattr(cp, "ndarray") or not hasattr(cp, "asarray"):
                return None
            return cp
        except Exception:
            return None

    @staticmethod
    def _import_cupy_kdtree():
        # Note: As of cupy-cuda12x 13.6.0 in our environment, cupyx.scipy.spatial does not
        # expose KDTree/cKDTree. Keep this helper for forward-compatibility, but it will
        # currently return (None, None) and we will fall back to cupy_bf/ckdtree.
        cp = KubricMultiCameraProcessed_XYZVidFrame0GT._import_cupy()
        if cp is None:
            return None, None
        try:
            from cupyx.scipy.spatial import KDTree as CuPyKDTree  # type: ignore
            return cp, CuPyKDTree
        except Exception:
            return None, None

    @staticmethod
    def _normalize_color0_cam0(
        color0_cam0: torch.Tensor,  # (H,W,3)
        *,
        valid0: torch.Tensor,  # (H,W) bool
        segm0: torch.Tensor,  # (H,W) int64
        mode: str,
        q_low: float,
        q_high: float,
    ) -> torch.Tensor:
        """
        Normalize the per-pixel frame-0 XYZ "color" used for colorization.

        This is *not* the VDPM track normalization; it only affects the target values we
        store into `track` for training (and visualization).

        Normalization uses foreground-valid pixels only: (valid0 & segm0>0).
        """
        mode = str(mode or "none").lower()
        if mode in {"off", "none", "raw"}:
            return color0_cam0
        if mode not in {"minmax", "quantile"}:
            raise ValueError(f"Unknown track_color_norm_mode={mode!r}. Expected one of: none|minmax|quantile.")

        mask = valid0.bool() & (segm0 > 0) & torch.isfinite(color0_cam0).all(dim=-1)
        out = color0_cam0.float().clone()
        for c in range(3):
            vals = out[..., c][mask]
            if vals.numel() == 0:
                mn, mx = 0.0, 1.0
            else:
                if mode == "minmax":
                    mn = float(vals.min().item())
                    mx = float(vals.max().item())
                else:
                    ql = float(q_low)
                    qh = float(q_high)
                    if not (0.0 <= ql < qh <= 1.0):
                        raise ValueError(
                            f"Invalid quantiles for track_color_norm_mode='quantile': "
                            f"q_low={q_low}, q_high={q_high} (expected 0<=low<high<=1)."
                        )
                    mn = float(torch.quantile(vals, ql).item())
                    mx = float(torch.quantile(vals, qh).item())
            if mx - mn < 1e-6:
                mx = mn + 1e-6
            out[..., c] = (out[..., c] - mn) / (mx - mn) * 2.0 - 1.0
        out[~mask] = 0.0
        return out.clamp(-1.0, 1.0)

    def _build_kdtrees(
        self,
        *,
        segm0: torch.Tensor,  # (H,W) int64
        obj0: torch.Tensor,  # (H,W,3) float32
        color0_cam0: torch.Tensor,  # (H,W,3) float32 (frame0 XYZ in cam0 coords)
        valid0: torch.Tensor,  # (H,W) bool
    ) -> Dict[int, Any]:
        backend = str(self.track_color_knn_backend or "ckdtree").lower()
        if backend == "auto":
            cp = self._import_cupy()
            backend = "cupy_bf" if (cp is not None) else "ckdtree"
        if backend == "cupy_kdtree":
            cp, kdt = self._import_cupy_kdtree()
            if cp is None or kdt is None:
                # In current env, this is expected; treat cupy_kdtree as an alias for cupy_bf.
                cp2 = self._import_cupy()
                if cp2 is not None:
                    if self._cupy_backend_warns < 5:
                        print("[KubricMultiCameraProcessed_XYZVidFrame0GT] cupy_kdtree unavailable; using cupy_bf instead.")
                        self._cupy_backend_warns += 1
                    backend = "cupy_bf"
                else:
                    if self._cupy_backend_warns < 5:
                        print("[KubricMultiCameraProcessed_XYZVidFrame0GT] cupy_kdtree unavailable; falling back to ckdtree.")
                        self._cupy_backend_warns += 1
                    backend = "ckdtree"
        if backend == "cupy_bf":
            cp = self._import_cupy()
            if cp is None:
                if self._cupy_backend_warns < 5:
                    print("[KubricMultiCameraProcessed_XYZVidFrame0GT] cupy_bf unavailable; falling back to ckdtree.")
                    self._cupy_backend_warns += 1
                backend = "ckdtree"

        trees: Dict[int, Any] = {}
        mask0 = valid0 & (segm0 > 0) & torch.isfinite(obj0).all(dim=-1) & torch.isfinite(color0_cam0).all(dim=-1)
        if not mask0.any():
            return trees
        seg_ids = torch.unique(segm0[mask0]).tolist()
        for seg_id in seg_ids:
            seg_id = int(seg_id)
            m = mask0 & (segm0 == seg_id)
            if not m.any():
                continue
            coords_np = obj0[m].to(torch.float32).cpu().numpy()
            if coords_np.shape[0] < 4:
                continue
            colors_np = color0_cam0[m].to(torch.float32).cpu().numpy()
            if backend == "cupy_kdtree":
                cp, CuPyKDTree = self._import_cupy_kdtree()
                if cp is None or CuPyKDTree is None:
                    continue

                coords_cp = cp.asarray(coords_np, dtype=cp.float32)
                colors_cp = cp.asarray(colors_np, dtype=cp.float32)
                tree = CuPyKDTree(coords_cp)
                trees[seg_id] = ("cupy_kdtree", tree, colors_cp)
            elif backend == "cupy_bf":
                cp = self._import_cupy()
                if cp is None:
                    continue

                coords_cp = cp.asarray(coords_np, dtype=cp.float32)
                colors_cp = cp.asarray(colors_np, dtype=cp.float32)
                coords_norm = (coords_cp * coords_cp).sum(axis=1)
                trees[seg_id] = ("cupy_bf", coords_cp, colors_cp, coords_norm)
            else:
                trees[seg_id] = ("ckdtree", cKDTree(coords_np), colors_np)
        return trees

    def _colorize_frame(
        self,
        *,
        segm_t: torch.Tensor,  # (H,W) int64
        obj_t: torch.Tensor,  # (H,W,3) float32
        valid_t: torch.Tensor,  # (H,W) bool
        trees: Dict[int, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = valid_t.shape
        out_cam0 = torch.zeros((h, w, 3), dtype=torch.float32)
        out_valid = torch.zeros((h, w), dtype=torch.bool)
        base = valid_t & (segm_t > 0) & torch.isfinite(obj_t).all(dim=-1)
        if not base.any() or not trees:
            return out_cam0, out_valid
        seg_ids = torch.unique(segm_t[base]).tolist()
        for seg_id in seg_ids:
            seg_id = int(seg_id)
            if seg_id not in trees:
                continue
            m = base & (segm_t == seg_id)
            if not m.any():
                continue
            rec = trees[seg_id]
            kind = rec[0]
            if kind == "ckdtree":
                _k, tree, colors_np = rec
                q_np = obj_t[m].to(torch.float32).cpu().numpy()
                _, nn_idx = tree.query(q_np, k=1)
                out_cam0[m] = torch.from_numpy(colors_np[nn_idx]).to(torch.float32)
                out_valid[m] = True
            elif kind == "cupy_kdtree":
                import cupy as cp

                _k, tree, colors_cp = rec
                q_np = obj_t[m].to(torch.float32).cpu().numpy()
                q_cp = cp.asarray(q_np, dtype=cp.float32)
                _dist, nn_idx = tree.query(q_cp, k=1)
                colored = colors_cp[nn_idx.astype(cp.int64)]
                out_cam0[m] = torch.from_numpy(cp.asnumpy(colored)).to(torch.float32)
                out_valid[m] = True
            elif kind == "cupy_bf":
                import cupy as cp

                _k, coords_cp, colors_cp, coords_norm = rec
                q_np = obj_t[m].to(torch.float32).cpu().numpy()
                q_cp = cp.asarray(q_np, dtype=cp.float32)
                q_norm = (q_cp * q_cp).sum(axis=1)

                qn = int(q_cp.shape[0])
                rn = int(coords_cp.shape[0])
                q_block = max(256, int(self.cupy_query_block))
                r_block = max(256, int(self.cupy_ref_block))

                best_d = cp.full((qn,), cp.inf, dtype=cp.float32)
                best_i = cp.zeros((qn,), dtype=cp.int64)

                # Blockwise reduction to avoid materializing huge (Q,R) matrices.
                for r0 in range(0, rn, r_block):
                    r1 = min(rn, r0 + r_block)
                    r = coords_cp[r0:r1]
                    rnrm = coords_norm[r0:r1]
                    rt = r.T
                    for q0 in range(0, qn, q_block):
                        q1 = min(qn, q0 + q_block)
                        qq = q_cp[q0:q1]
                        qnrm = q_norm[q0:q1]
                        # dist^2 = ||q||^2 + ||r||^2 - 2 q·r
                        d = qnrm[:, None] + rnrm[None, :] - 2.0 * (qq @ rt)
                        idx_local = d.argmin(axis=1).astype(cp.int64)
                        dmin = d[cp.arange(d.shape[0]), idx_local]
                        cur = best_d[q0:q1]
                        upd = dmin < cur
                        best_d[q0:q1] = cp.where(upd, dmin, cur)
                        best_i[q0:q1] = cp.where(upd, idx_local + r0, best_i[q0:q1])

                colored = colors_cp[best_i]
                out_cam0[m] = torch.from_numpy(cp.asnumpy(colored)).to(torch.float32)
                out_valid[m] = True
            else:
                raise ValueError(f"Unknown KNN backend record kind={kind!r}")
        return out_cam0, out_valid

    def __getitem__(self, idx):
        views: List[dict] = super().__getitem__(idx)
        if not views:
            return views

        # Reference is view 0 (matches the visualize script, and matches Stream3RDatasetAdapter's cam1 canonical frame).
        ref = views[0]
        if "obj_coords" not in ref or "segm" not in ref:
            raise KeyError("XYZVidFrame0GT requires 'obj_coords' and 'segm' in dataset views.")

        cam0_pose = _to_torch(ref["camera_pose"], dtype=torch.float32)  # cam0->world
        inv_cam0 = dust3r_inv(cam0_pose)  # world->cam0

        pts0_world = _to_torch(ref["pts3d"], dtype=torch.float32)  # (H,W,3) world coords
        valid0 = _to_torch(ref["valid_mask"], dtype=torch.bool)  # (H,W)
        segm0 = _to_torch(ref["segm"], dtype=torch.int64)  # (H,W)
        obj0 = _to_torch(ref["obj_coords"], dtype=torch.float32)  # (H,W,3) in [0,1]

        # "Color": frame-0 XYZ expressed in the frame-0 camera coords (cam0).
        color0_cam0 = dust3r_geotrf(inv_cam0, pts0_world)
        color0_cam0 = self._normalize_color0_cam0(
            color0_cam0,
            valid0=valid0,
            segm0=segm0,
            mode=self.track_color_norm_mode,
            q_low=self.track_color_quantile_low,
            q_high=self.track_color_quantile_high,
        )

        trees = self._build_kdtrees(segm0=segm0, obj0=obj0, color0_cam0=color0_cam0, valid0=valid0)

        for view in views:
            segm_t = _to_torch(view["segm"], dtype=torch.int64)
            obj_t = _to_torch(view["obj_coords"], dtype=torch.float32)
            valid_t = _to_torch(view["valid_mask"], dtype=torch.bool)

            colored_cam0, colored_valid = self._colorize_frame(
                segm_t=segm_t, obj_t=obj_t, valid_t=valid_t, trees=trees
            )

            # Store target in WORLD coordinates so downstream logic (which converts track->cam1) remains consistent.
            colored_world = dust3r_geotrf(cam0_pose, colored_cam0)

            if self.preserve_original_track and "track" in view:
                # Preserve dynamic track as primary target for residual mode.
                view["track_dynamic_world"] = view["track"]
                if "track_valid_mask" in view:
                    view["track_dynamic_valid_mask"] = view["track_valid_mask"]
                # Expose frame0-colorized target in cam0 coords as pre-normalized track_video
                # (same semantic contract as preloaded mp4 path: already in [-1,1]).
                # Invalid pixels are set to -1.0 to match the MP4 convention (0 uint8 = -1.0 float).
                _tv_prenorm = colored_cam0.cpu().numpy().astype(np.float32)
                _tv_valid_np = colored_valid.cpu().numpy().astype(bool)
                _tv_prenorm[~_tv_valid_np] = -1.0
                view["track_video_prenormalized"] = _tv_prenorm
                view["track_video_valid_mask"] = _tv_valid_np

            view["track"] = colored_world.cpu().numpy().astype(np.float32)
            view["track_valid_mask"] = colored_valid.cpu().numpy().astype(bool)
            view["track_query_idx"] = 0
            view["dataset"] = "KubricMultiCameraProcessed_XYZVidFrame0GT"

        return views


__all__ = ["KubricMultiCameraProcessed_XYZVidFrame0GT"]
