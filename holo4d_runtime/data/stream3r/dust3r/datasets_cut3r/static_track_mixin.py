"""
Mixin for adding static tracking data to any BaseMultiViewDataset subclass.

For static scenes, the track is the frame-0 pointmap repeated for all frames
(nothing moves, residual = 0). The trackvid uses depth-based visibility checking
and configurable frame-0-based normalization.

Usage:
    class MyDataset_WithStaticTrack(StaticTrackMixin, MyDataset):
        pass

    dataset = MyDataset_WithStaticTrack(ROOT=..., return_trackvid=True)
"""

import numpy as np


class StaticTrackMixin:
    """
    Mixin that adds static tracking data to BaseMultiViewDataset subclasses.

    Must be listed BEFORE the dataset class in MRO so its __init__/__getitem__
    and _get_views are called first.

    Adds to each view:
      - track: frame-0's pts3d (world coords), same for ALL frames
      - track_valid_mask: frame-0's depth validity
      - track_query_idx: 0

    When return_trackvid=True, also adds:
      - trackvid: frame-0-colorized XYZ with visibility checking (normalized [-1,1])
      - trackvid_valid_mask: visibility mask
    """

    def __init__(
        self,
        *args,
        return_trackvid=False,
        trackvid_norm_mode="quantile",
        trackvid_q_low=0.02,
        trackvid_q_high=0.98,
        trackvid_depth_rel_tol=0.1,
        trackvid_depth_abs_tol=0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_trackvid = return_trackvid
        self.trackvid_norm_mode = trackvid_norm_mode
        self.trackvid_q_low = trackvid_q_low
        self.trackvid_q_high = trackvid_q_high
        self.trackvid_depth_rel_tol = trackvid_depth_rel_tol
        self.trackvid_depth_abs_tol = trackvid_depth_abs_tol

    # ------------------------------------------------------------------
    # Force video_prob=1.0 in _get_views without duplicating loading code
    # ------------------------------------------------------------------

    def _get_views(self, idx, resolution, rng, num_views):
        """Wrap parent's _get_views to force sequential frame ordering."""
        orig_fn = self.get_seq_from_start_id

        def _forced_video_seq(*args, **kwargs):
            kwargs["video_prob"] = 1.0
            return orig_fn(*args, **kwargs)

        self.get_seq_from_start_id = _forced_video_seq
        try:
            return super()._get_views(idx, resolution, rng, num_views)
        finally:
            self.get_seq_from_start_id = orig_fn

    # ------------------------------------------------------------------
    # __getitem__: add track + optional trackvid after parent computes pts3d
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        views = super().__getitem__(idx)
        if not views:
            return views

        # Static track: frame-0 world pointmap is the track for all frames
        anchor_pts3d = views[0]["pts3d"].copy()    # (H,W,3) world
        anchor_valid = views[0]["valid_mask"].copy()  # (H,W)

        for view in views:
            view["track"] = anchor_pts3d.copy()
            view["track_valid_mask"] = anchor_valid.copy()
            view["track_query_idx"] = 0

        if self.return_trackvid:
            self._compute_trackvid(views)

        return views

    # ------------------------------------------------------------------
    # Trackvid: frame-0-colorized XYZ with visibility + frame-0 norm
    # ------------------------------------------------------------------

    def _compute_trackvid(self, views):
        cam0_pose = views[0]["camera_pose"]        # (4,4) cam0 -> world
        cam0_intr = views[0]["camera_intrinsics"]   # (3,3)
        inv_cam0 = np.linalg.inv(cam0_pose)         # world -> cam0
        R = inv_cam0[:3, :3]
        t_vec = inv_cam0[:3, 3]

        depth0 = views[0]["depthmap"]    # (H,W) z in cam0
        valid0 = views[0]["valid_mask"]  # (H,W)

        # Frame-0 cam0 XYZ
        pts0_cam0 = (
            np.einsum("ik,hwk->hwi", R, views[0]["pts3d"]) + t_vec[None, None, :]
        ).astype(np.float32)

        norm_mode = str(self.trackvid_norm_mode or "quantile").lower()
        if norm_mode in {"shared_absmax", "absmax", "global_absmax"}:
            vals = np.abs(pts0_cam0[valid0])
            shared_absmax = float(vals.max()) if vals.size > 0 else 1.0
            if shared_absmax < 1e-6:
                shared_absmax = 1.0
            mn, mx = None, None
        else:
            # Per-channel bounds from frame-0 only.
            mn, mx = self._normalize_cam0_xyz(
                pts0_cam0, valid0,
                mode=norm_mode,
                q_low=self.trackvid_q_low,
                q_high=self.trackvid_q_high,
            )
            shared_absmax = None

        for view in views:
            pts_cam0 = (
                np.einsum("ik,hwk->hwi", R, view["pts3d"]) + t_vec[None, None, :]
            ).astype(np.float32)

            visible = self._check_visibility_in_frame0(
                pts_cam0, view["valid_mask"],
                depth0, valid0, cam0_intr,
                depth_rel_tol=self.trackvid_depth_rel_tol,
                depth_abs_tol=self.trackvid_depth_abs_tol,
            )

            if shared_absmax is not None:
                # One shared scale for X/Y/Z (no per-channel normalization, no quantile clipping).
                trackvid = pts_cam0 / shared_absmax
            else:
                trackvid = np.zeros_like(pts_cam0)
                for c in range(3):
                    trackvid[..., c] = (pts_cam0[..., c] - mn[c]) / (mx[c] - mn[c]) * 2.0 - 1.0
            trackvid = np.clip(trackvid, -1.0, 1.0)
            trackvid[~visible] = -1.0  # invalid = black, matching MP4 convention

            view["trackvid"] = trackvid.astype(np.float32)
            view["trackvid_valid_mask"] = visible
            # Adapter multimodality path expects this key for already-normalized track videos.
            view["track_video_prenormalized"] = trackvid.astype(np.float32)
            view["track_video_valid_mask"] = visible

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_visibility_in_frame0(
        pts_cam0, valid_t, depth0, valid0, intrinsics0,
        depth_rel_tol=0.1, depth_abs_tol=0.05,
    ):
        """Check which frame-t points were visible in frame 0 via depth reprojection."""
        H, W = pts_cam0.shape[:2]
        z_cam0 = pts_cam0[..., 2]
        finite_pts = np.isfinite(pts_cam0).all(axis=-1)

        fx, fy = intrinsics0[0, 0], intrinsics0[1, 1]
        cx, cy = intrinsics0[0, 2], intrinsics0[1, 2]

        z_safe = np.where(z_cam0 > 1e-6, z_cam0, 1e-6)
        u0 = fx * pts_cam0[..., 0] / z_safe + cx
        v0 = fy * pts_cam0[..., 1] / z_safe + cy

        finite_uv = np.isfinite(u0) & np.isfinite(v0)
        in_front = (z_cam0 > 1e-6) & np.isfinite(z_cam0)
        in_bounds = (u0 >= -0.5) & (u0 < W - 0.5) & (v0 >= -0.5) & (v0 < H - 0.5)

        # Replace non-finite projections and hard-clip huge finite values before int cast
        # (avoids RuntimeWarning: invalid value encountered in cast).
        i32 = np.iinfo(np.int32)
        lo = float(i32.min + 1)
        hi = float(i32.max - 1)
        u0_safe = np.asarray(u0, dtype=np.float64)
        v0_safe = np.asarray(v0, dtype=np.float64)
        u0_safe = np.nan_to_num(u0_safe, nan=lo, posinf=hi, neginf=lo)
        v0_safe = np.nan_to_num(v0_safe, nan=lo, posinf=hi, neginf=lo)
        u0_safe = np.clip(u0_safe, lo, hi)
        v0_safe = np.clip(v0_safe, lo, hi)
        u0_int = np.clip(np.rint(u0_safe).astype(np.int32), 0, W - 1)
        v0_int = np.clip(np.rint(v0_safe).astype(np.int32), 0, H - 1)

        valid0_at_proj = valid0[v0_int, u0_int]
        depth0_at_proj = depth0[v0_int, u0_int]
        finite_depth0 = np.isfinite(depth0_at_proj)
        depth_diff = np.abs(z_cam0 - depth0_at_proj)
        depth_threshold = depth_rel_tol * np.abs(depth0_at_proj) + depth_abs_tol
        depth_consistent = finite_depth0 & np.isfinite(depth_diff) & np.isfinite(depth_threshold) & (depth_diff < depth_threshold)

        return valid_t & finite_pts & finite_uv & in_front & in_bounds & valid0_at_proj & depth_consistent

    @staticmethod
    def _normalize_cam0_xyz(color0_cam0, valid0, mode="quantile", q_low=0.02, q_high=0.98):
        """Per-channel normalization bounds from frame-0 cam0 XYZ only."""
        mn_list, mx_list = [], []
        for c in range(3):
            vals = color0_cam0[..., c][valid0]
            if vals.size == 0:
                mn, mx = 0.0, 1.0
            elif mode == "quantile":
                mn = float(np.percentile(vals, q_low * 100))
                mx = float(np.percentile(vals, q_high * 100))
            elif mode == "minmax":
                mn, mx = float(vals.min()), float(vals.max())
            elif mode in {"off", "none", "raw"}:
                mn, mx = -1.0, 1.0
            else:
                raise ValueError(
                    f"Unknown trackvid_norm_mode={mode!r}. "
                    "Expected one of: quantile|minmax|none|shared_absmax"
                )
            if mx - mn < 1e-6:
                mx = mn + 1e-6
            mn_list.append(mn)
            mx_list.append(mx)
        return np.array(mn_list, dtype=np.float32), np.array(mx_list, dtype=np.float32)
