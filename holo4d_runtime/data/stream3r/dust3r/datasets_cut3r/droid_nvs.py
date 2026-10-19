# PointWorld-DROID cross-camera NVS dataset for track4d-360.
#
# A DROID episode has two static EXTERNAL ZED cameras (ext1, ext2) watching one
# DYNAMIC robot-manipulation scene. We treat them as a novel-view pair: source =
# one external camera's video, target = the other's. The 3D conditioning tracks
# are the released PointWorld scene-flow point tracks (world frame, visible from
# both cameras), chained from the released overlapping 11-frame clips into a
# long (num_views) window WITHOUT re-running CoTracker. Depth comes from the
# packaged per-frame *_depth.h5; camera poses + intrinsics come from the flows
# file's per-clip `extrinsic` (world2cam) / `intrinsic` (at 320x180).
#
# Emits the same dict contract as Syn4D_Multi_NVS / Kubric_NVS / StaticScene_NVS
# (see datasets_cut3r spec): cam0-normalized tracks, cam2cam0 OpenCV poses with
# translations / common_scale, pixel intrinsics at output res, depth / common_scale.
#
# Requires a prebuilt index (tools/build_droid_nvs_index.py) listing valid
# (uuid, scene_dir, flows, depth, src_serial, tgt_serial, run_start, run_clips).

from __future__ import annotations

import json
import os
import random

import cv2
import h5py
import numpy as np
from PIL import Image

from .base.easy_dataset import EasyDataset

DEFAULT_PROMPT = "Generate the novel view trajectory video given the input video."


# --------------------------------------------------------------------------- #
#  Track chaining (vendored from DROID_PW/tools/extract_long_tracks.py)
# --------------------------------------------------------------------------- #
def _voxel_join(prev_pts, prev_ok, cur_pts, cur_ok, voxel):
    cidx = np.nonzero(cur_ok)[0]
    table = {}
    if cidx.size:
        keys = np.round(cur_pts[cidx] / voxel).astype(np.int64)
        for n, k in zip(cidx.tolist(), map(tuple, keys)):
            table.setdefault(k, n)
    pk = np.round(prev_pts / voxel).astype(np.int64)
    matched = np.full(prev_pts.shape[0], -1, dtype=np.int64)
    for i in np.nonzero(prev_ok)[0]:
        matched[i] = table.get(tuple(pk[i]), -1)
    return matched, matched >= 0


# Robust clip join (replaces single-frame voxel-EXACT _voxel_join, which drifts:
# grid-brittle matching + single overlap frame + compounding wrong picks). MUTUAL
# nearest-neighbor on the MEAN position across the full overlap window, max_dist
# gated. scipy-free. Mirrors DROID_PW/tools/extract_long_tracks.py:robust_join.
_NB27 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def _masked_mean(P, V):
    c = V.sum(0).astype(np.float32)
    s = (P * V[..., None]).sum(0)
    out = np.full((P.shape[1], 3), np.nan, np.float32)
    nz = c > 0
    out[nz] = s[nz] / c[nz, None]
    return out


def _bucket(pts, ok, vox):
    table = {}
    for i in np.nonzero(ok)[0]:
        table.setdefault(tuple((pts[i] / vox).round().astype(np.int64)), []).append(int(i))
    return table


def _nn1(A, B, max_dist):
    """NN index in B for each row of A within max_dist (-1 if none). cKDTree if
    scipy present (fast), else voxel-bucket fallback."""
    out = np.full(A.shape[0], -1, np.int64)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return out
    try:
        from scipy.spatial import cKDTree
        d, j = cKDTree(B).query(A, distance_upper_bound=max_dist)
        ok = j < B.shape[0]
        out[ok] = j[ok]
        return out
    except ImportError:
        vox = max(float(max_dist), 1e-6)
        tab = _bucket(B, np.ones(B.shape[0], bool), vox)
        for i in range(A.shape[0]):
            kx, ky, kz = (A[i] / vox).round().astype(np.int64)
            best, bd, ai = -1, max_dist, A[i]
            for dx, dy, dz in _NB27:
                for n in tab.get((kx + dx, ky + dy, kz + dz), ()):
                    dd = float(np.linalg.norm(ai - B[n]))
                    if dd < bd:
                        bd, best = dd, n
            out[i] = best
        return out


def robust_join(prev_sf, prev_vis, cur_idx, alive, cc_sf, cc_vis, skip, fpc, max_dist):
    overlap = fpc - skip
    P = prev_sf[skip:fpc, cur_idx]
    Pv = prev_vis[skip:fpc, cur_idx] & alive[None, :]
    Q = cc_sf[0:overlap]
    Qv = cc_vis[0:overlap]
    Pm = _masked_mean(P, Pv); Qm = _masked_mean(Q, Qv)
    Pok = (~np.isnan(Pm).any(1)) & alive
    Qok = ~np.isnan(Qm).any(1)
    matched = np.full(P.shape[1], -1, np.int64)
    pi = np.nonzero(Pok)[0]; qi = np.nonzero(Qok)[0]
    if pi.size == 0 or qi.size == 0:
        return matched, matched >= 0
    pq = _nn1(Pm[pi], Qm[qi], max_dist)
    qp = _nn1(Qm[qi], Pm[pi], max_dist)
    a = np.nonzero(pq >= 0)[0]
    mutual = a[qp[pq[a]] == a]
    matched[pi[mutual]] = qi[pq[mutual]]
    return matched, matched >= 0


def _chain(run_clips, skip, fpc, length, voxel):
    """run_clips: list of dict(sf,vis,col) consecutive clips spaced by `skip`.
    Returns (L, pos (L,M,3), vis (L,M), col (L,M,3))."""
    m = len(run_clips)
    overlap = fpc - skip
    cov = (m - 1) * skip + fpc
    L = int(min(length, cov))
    c0 = run_clips[0]
    M0 = c0["sf"].shape[1]
    pos = np.full((L, M0, 3), np.nan, np.float32)
    vis = np.zeros((L, M0), bool)
    col = np.zeros((L, M0, 3), np.uint8)
    f0 = min(fpc, L)
    pos[:f0], vis[:f0], col[:f0] = c0["sf"][:f0], c0["vis"][:f0], c0["col"][:f0]
    cur = np.arange(M0)
    alive = np.ones(M0, bool)
    prev = c0
    for ci in range(1, m):
        off = ci * skip
        if off + overlap >= L:   # no new frames beyond what's filled -> stop joining
            break
        cc = run_clips[ci]
        matched, good = robust_join(prev["sf"], prev["vis"], cur, alive,
                                    cc["sf"], cc["vis"], skip, fpc, voxel)
        alive &= good
        cur = np.where(good, matched, 0)
        lo, hi = off + overlap, min(off + fpc, L)
        if hi > lo:
            blk = slice(overlap, overlap + (hi - lo))
            pos[lo:hi] = np.where(alive[None, :, None], cc["sf"][blk][:, cur], np.nan)
            vis[lo:hi] = alive[None, :] & cc["vis"][blk][:, cur]
            col[lo:hi] = cc["col"][blk][:, cur] * alive[None, :, None]
        prev = cc
    surv = alive & ~np.isnan(pos).any(axis=(0, 2))
    return L, pos[:, surv], vis[:, surv], col[:, surv]


def _read_camera_clips(h5, serial):
    cam = f"camera_{serial}_ext"
    clips = {}
    for k in h5:
        if ":" not in k or cam not in h5[k]:
            continue
        g = h5[k][cam]
        sf = np.asarray(g["scene_flows"][()], np.float32)
        vis = np.asarray(g["scene_visibility"][()]).astype(bool)
        if "scene_depth_valid_mask" in g:
            vis &= np.asarray(g["scene_depth_valid_mask"][()]).astype(bool)
        col = np.asarray(g["scene_colors"][()]).astype(np.uint8)
        if col.ndim == 2:
            col = np.broadcast_to(col[None], (sf.shape[0], col.shape[0], 3)).copy()
        clips[int(k.split(":")[0])] = dict(sf=sf, vis=vis, col=col)
    return clips


def _nearest_idx(query_ts, table_ts):
    order = np.argsort(table_ts)
    s = table_ts[order]
    p = np.clip(np.searchsorted(s, query_ts), 1, len(s) - 1)
    pick_left = (query_ts - s[p - 1]) <= (s[p] - query_ts)
    return order[np.where(pick_left, p - 1, p)]


def _resize_K(K, sw, sh):
    K = K.astype(np.float32).copy()
    K[0, 0] *= sw; K[0, 2] *= sw
    K[1, 1] *= sh; K[1, 2] *= sh
    return K


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class DROID_NVS(EasyDataset):
    def __init__(self, ROOT, index_file, num_views=49, resolution=(256, 144),
                 steps_per_epoch=10000, track_spatial_stride=1, voxel=0.02,
                 max_tracks=4096, seed=None, swap_src_tgt_prob=0.5,
                 min_num_cameras=1, max_num_cameras=2, z_min=0.1,
                 return_source_depth_video=False, return_source_rgb_video=False,
                 return_source_cameras=False, return_target_depth=False,
                 return_target_tracks=False, return_dynamic_masks=False,
                 multi_query_prob=0.0, multi_query_choices="1", fixed_start_frame=None,
                 **_ignored):
        self.ROOT = ROOT
        self._num_views = int(num_views)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.W, self.H = int(resolution[0]), int(resolution[1])
        self.steps_per_epoch = int(steps_per_epoch)
        # Track grid stride (the released CoTracker tracks are already sparse, so
        # this is 1); exposed as an attribute so GeometryRobustnessWrapper_NVS can
        # read it for sparse-track / multi-query grid inference.
        self.track_spatial_stride = int(track_spatial_stride)
        self.voxel = float(voxel)
        self.max_tracks = int(max_tracks)
        self.seed = seed
        self.swap_prob = float(swap_src_tgt_prob)
        self.min_cams = int(min_num_cameras); self.max_cams = int(max_num_cameras)
        self.z_min = float(z_min)   # stable-reprojection gate (metres, see _sample)
        self.r_src_depth = bool(return_source_depth_video)
        self.r_src_rgb = bool(return_source_rgb_video)
        self.r_src_cam = bool(return_source_cameras)
        self.r_tgt_depth = bool(return_target_depth)
        self.r_tgt_tracks = bool(return_target_tracks)
        self.fixed_start_frame = fixed_start_frame
        with open(index_file) as f:
            self.index = json.load(f)
        if not self.index:
            raise RuntimeError(f"empty DROID index: {index_file}")

    # ---- EasyDataset contract -------------------------------------------- #
    @property
    def _resolutions(self):
        return [(self.W, self.H)]

    @property
    def num_views(self):
        return self._num_views

    def set_epoch(self, epoch):
        pass

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        rng = random.Random(None if self.seed is None else self.seed + int(idx))
        last = None
        for _ in range(10):
            try:
                return self._sample(rng)
            except Exception as e:  # skip bad scene, resample
                last = e
        raise RuntimeError(f"DROID_NVS: 10 retries failed; last={last!r}")

    # ---- core ------------------------------------------------------------ #
    def _sample(self, rng):
        T, W, H = self._num_views, self.W, self.H
        ent = self.index[rng.randrange(len(self.index))]
        src_serial, tgt_serial = ent["src_serial"], ent["tgt_serial"]
        if rng.random() < self.swap_prob:
            src_serial, tgt_serial = tgt_serial, src_serial

        def _cam_KE(f, serial):
            """intrinsic (320x180) + extrinsic (world2cam) from any clip with this cam."""
            cam = f"camera_{serial}_ext"
            ck = next(k for k in f if ":" in k and cam in f[k])
            return (np.asarray(f[ck][cam]["intrinsic"][()], np.float32),
                    np.asarray(f[ck][cam]["extrinsic"][()], np.float32))

        with h5py.File(ent["flows"], "r") as f:
            canon_ts_all = np.array(json.loads(f.attrs["canonical_timestamps"]), np.int64)
            src_clips = _read_camera_clips(f, src_serial)
            tgt_clips = _read_camera_clips(f, tgt_serial)
            Ksrc, Esrc = _cam_KE(f, src_serial)   # Esrc = world2cam
            Ktgt, Etgt = _cam_KE(f, tgt_serial)

        # ---- PointWorld sample_cameras convention ----
        # ext1/ext2 share the same clip keys (clipping + motion-filter are camera-
        # independent), so chain BOTH over the same run, randomly pick a camera
        # subset, and CONCATENATE their world-frame tracks (union, no correspondence).
        skip, run_start, nclips = ent["skip"], ent["run_start"], ent["run_clips"]
        def _chain_cam(clips_dict):
            run = [clips_dict[run_start + i * skip] for i in range(nclips)]
            fpc = run[0]["sf"].shape[0]
            return _chain(run, skip, fpc, 10_000, self.voxel)     # (L,M,3) world, vis, col
        cam_tracks = {"src": _chain_cam(src_clips), "tgt": _chain_cam(tgt_clips)}
        Lc = min(cam_tracks["src"][0], cam_tracks["tgt"][0])
        if Lc < T:
            raise RuntimeError(f"chained window {Lc} < num_views {T}")
        avail = ["src", "tgt"]
        k = rng.randint(max(1, self.min_cams), min(self.max_cams, len(avail)))   # inclusive
        chosen = rng.sample(avail, k)
        o = 0 if self.fixed_start_frame is not None else rng.randrange(0, Lc - T + 1)
        canon_ts = canon_ts_all[run_start + o: run_start + o + T]
        pos_l, vis_l, col_l = [], [], []
        for c in chosen:
            _, p, v, cc = cam_tracks[c]
            p, v, cc = p[o:o + T], v[o:o + T], cc[o:o + T]
            keep = v.all(0)
            if keep.sum() < 8:
                keep = v.any(0)
            if keep.sum():
                pos_l.append(p[:, keep]); vis_l.append(v[:, keep]); col_l.append(cc[:, keep])
        if not pos_l:
            raise RuntimeError("no surviving tracks")
        pos = np.concatenate(pos_l, axis=1)          # (T, Mtot, 3) world (selected cameras)
        vis = np.concatenate(vis_l, axis=1)
        col = np.concatenate(col_l, axis=1)

        # ---- stable-reprojection gate (matches viz Z_MIN, see DROID_PW doc §7) ----
        # A track at tiny +depth in a camera is projection-degenerate: z is the
        # projection denominator, so mm of 3D noise -> 100+px of 2D motion (worst
        # for near-orthogonal camera pairs, e.g. optical axes 103deg apart). Mask
        # frames where a point is behind/grazing the source(cam0) OR target camera;
        # this keeps out-of-FOV-but-in-front points (still valid 3D cond) and drops
        # only the degenerate ones. Gate is metric (pos & extrinsics pre-normalize).
        def _infront(E):
            zc = (pos.reshape(-1, 3) @ E[:3, :3].T + E[:3, 3])[:, 2].reshape(T, -1)
            return zc > self.z_min                       # (T, Mtot)
        vis = vis & _infront(Esrc) & _infront(Etgt)
        keep = vis.any(0)
        if keep.sum() < 8:
            raise RuntimeError("no stable tracks after z-gate")
        pos, vis, col = pos[:, keep], vis[:, keep], col[:, keep]

        if pos.shape[1] > self.max_tracks:
            sel = np.asarray(rng.sample(range(pos.shape[1]), self.max_tracks))
            pos, vis, col = pos[:, sel], vis[:, sel], col[:, sel]

        # ---- cam0 normalization (cam0 = source cam at window frame 0; static) ----
        # Esrc maps world -> src cam. common_scale from cam0-space track norms.
        pw = pos.reshape(-1, 3)
        pcam0 = (Esrc[:3, :3] @ pw.T).T + Esrc[:3, 3]             # world -> src cam (cam0)
        valid = vis.reshape(-1)
        common_scale = float(np.linalg.norm(pcam0[valid], axis=1).mean()) if valid.any() else 1.0
        common_scale = max(common_scale, 1e-6)
        tracks_cam0 = (pcam0 / common_scale).reshape(T, -1, 3).astype(np.float32)  # (T,M,3)
        tracks_world = np.transpose(tracks_cam0, (1, 0, 2)).copy()  # (M,T,3) per contract

        c2w_src = np.linalg.inv(Esrc)
        c2w_tgt = np.linalg.inv(Etgt)
        w2c_cam0 = Esrc                                            # inv(c2w_src)
        def norm_pose(c2w):
            p = (w2c_cam0 @ c2w).astype(np.float32)
            p[:3, 3] /= common_scale
            return p
        src_pose = norm_pose(c2w_src)                              # ~identity
        tgt_pose = norm_pose(c2w_tgt)
        target_cam2world = np.broadcast_to(tgt_pose, (T, 4, 4)).astype(np.float32).copy()
        source_cam2world = np.broadcast_to(src_pose, (T, 4, 4)).astype(np.float32).copy()
        Ksrc_o = _resize_K(Ksrc, W / 320.0, H / 180.0)
        Ktgt_o = _resize_K(Ktgt, W / 320.0, H / 180.0)

        # ---- RGB videos (source + target) from MP4, timestamp aligned ----
        with h5py.File(os.path.join(ent["scene_dir"], "trajectory.h5"), "r") as t:
            src_cts = t[f"observation/timestamp/cameras/{src_serial}_estimated_capture"][()].astype(np.int64)
            tgt_cts = t[f"observation/timestamp/cameras/{tgt_serial}_estimated_capture"][()].astype(np.int64)
        src_video = self._load_mp4(ent["scene_dir"], src_serial, _nearest_idx(canon_ts, src_cts), W, H)
        tgt_video = self._load_mp4(ent["scene_dir"], tgt_serial, _nearest_idx(canon_ts, tgt_cts), W, H)

        # ---- depth (source frame0 + optional videos) from depth.h5 ----
        src_depth_vid = self._load_depth(ent["depth"], src_serial, canon_ts, W, H) / common_scale
        tgt_depth_vid = self._load_depth(ent["depth"], tgt_serial, canon_ts, W, H) / common_scale

        # ---- source RGB sampled at the track grid (contract: (M,T,3) in [0,1]) ----
        src_rgb01 = np.stack([np.asarray(im, np.float32) / 255.0 for im in src_video])  # (T,H,W,3)
        # project cam0 tracks back into source frame0 to sample colors — but we already
        # carry per-point colors from the flow; use those (T,M,3)->(M,T,3) in [0,1]
        source_rgb_at_tracks = (np.transpose(col, (1, 0, 2)).astype(np.float32) / 255.0)

        sample = {
            "video": src_video,
            "target_video": tgt_video,
            "tracks_world": tracks_world,                 # (M,T,3) cam0-normalized XYZ
            "target_cam2world": target_cam2world,         # (T,4,4) cam2cam0 OpenCV
            "target_intrinsics": Ktgt_o,                  # (3,3) pixel K @ output res
            "common_scale": common_scale,
            "track_query_idx": 0,
            "prompt": ent.get("prompt") or DEFAULT_PROMPT,
            "prompt_emb": None,
            "has_tracks": True,
            "has_dynamic_tracks": True,
            "has_depth": True,
            "target_depth": tgt_depth_vid.astype(np.float32),       # (T,H,W)
            "source_depth": src_depth_vid[0].astype(np.float32),    # (H,W)
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "source_rgb_at_tracks": source_rgb_at_tracks.astype(np.float32),
        }
        if self.r_src_rgb:
            sample["source_rgb_video"] = src_rgb01.astype(np.float32)
        if self.r_src_cam:
            sample["source_cam2world"] = source_cam2world
            sample["source_intrinsics"] = Ksrc_o
        if self.r_src_depth:
            sample["source_depth_video"] = src_depth_vid.astype(np.float32)
        if self.r_tgt_depth:
            sample["target_depth_video"] = tgt_depth_vid.astype(np.float32)
            sample["target_depth_valid_mask"] = (tgt_depth_vid > 0)
            sample["target_depth_valid"] = (tgt_depth_vid > 0)
        return sample

    # ---- loaders --------------------------------------------------------- #
    def _load_mp4(self, scene_dir, serial, frame_idx, W, H):
        cap = cv2.VideoCapture(os.path.join(scene_dir, "recordings", "MP4", f"{serial}.mp4"))
        out = []
        for j in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
            ok, bgr = cap.read()
            if not ok:
                bgr = np.zeros((H, W, 3), np.uint8)
            rgb = cv2.resize(bgr[..., ::-1], (W, H), interpolation=cv2.INTER_AREA)
            out.append(Image.fromarray(rgb))
        cap.release()
        return out

    def _load_depth(self, depth_path, serial, canon_ts, W, H):
        with h5py.File(depth_path, "r") as d:
            g = d[f"{serial}+ext"]
            didx = _nearest_idx(canon_ts, g["timestamps"][()].astype(np.int64))
            depth = g["depth"][:][didx].astype(np.float32) / 1000.0   # (T,180,320) m
        return np.stack([cv2.resize(z, (W, H), interpolation=cv2.INTER_NEAREST) for z in depth])
