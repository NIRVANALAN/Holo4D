# OmniWorld-DROID cross-camera NVS dataloader.
#
# Same unified contract as DROID_NVS (ext1<->ext2 source/target, cam0-normalized
# 3D tracks + depth + OpenCV cameras), but the 3D tracks come from OmniWorld's
# per-frame OPTICAL FLOW chained into long tracks and lifted via OmniWorld's
# 720p FoundationStereo depth + camera intrinsics + DROID 6-DoF extrinsics.
# Advantages over DROID_NVS: full-frame dense tracks, arbitrary length (no 11f
# clip limit / voxel-join), 720p depth. Weakness: flow-integration drift on
# fast/long motion (mostly-static DROID scenes are fine).
#
# Inputs are EXTRACTED OmniWorld annotation dirs (read the index built by
# bash_scripts/holo4d/droid_io/build_omni_droid_index.py):
#   <ann>/flow/<serial>/<f>/flow_u_16.png, flow_v_16.png   (uint16, raw/65535*600-300 px)
#   <ann>/foundation_stereo/<serial>/<f>.png               (uint16, raw/65535*10 m, 720p)
#   <ann>/recordings/camera_info_dict.npy                  ({serial:{cam_matrix:K@1280x720}})
#   <ann>/meta_info.json                                   (serials + 6-DoF extrinsics)
# RGB from the raw DROID MP4 (recordings/MP4/<serial>.mp4).
from __future__ import annotations

import json
import os
import random

import cv2
import h5py  # noqa: F401  (kept for parity / optional future use)
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from .base.easy_dataset import EasyDataset

_FLOW_MIN, _FLOW_MAX = -300.0, 300.0
_MAX_DEPTH = 10.0
# Unified NVS task prompt — MUST match one of the trainer's precomputed default
# prompts (scripts/wan2.1_fun/train_track4d360.py:_DEFAULT_PROMPTS_TO_PRECOMPUTE),
# because T5 is offloaded/deleted after precompute and strict-cache mode raises on
# any uncached prompt. OmniWorld's per-scene task captions are NOT in that cache,
# so we use the unified NVS default (text is near-null for this cfg=1 geometric
# NVS setup). To use per-scene captions instead, precompute their T5 embeddings
# and ship them as sample['prompt_emb'].
DEFAULT_PROMPT = "Generate the novel view trajectory video given the input video."


def _sixdof_to_c2w(v):
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R.from_euler("xyz", v[3:]).as_matrix()
    M[:3, 3] = v[:3]
    return M


def _resize_K(K, sx, sy):
    K = K.astype(np.float32).copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def _viewmatrix(lookdir, up, position):
    """mip-NeRF360 / AnySplat lookat view matrix (traj.py). CONVENTION-AGNOSTIC:
    it derives the basis from the pose's own look/up vectors, so feeding OpenCV
    poses (y-down) round-trips back to OpenCV — no manual handedness. Returns 3x4."""
    vec2 = lookdir / (np.linalg.norm(lookdir) + 1e-12)
    vec0 = np.cross(up, vec2); vec0 = vec0 / (np.linalg.norm(vec0) + 1e-12)
    vec1 = np.cross(vec2, vec0); vec1 = vec1 / (np.linalg.norm(vec1) + 1e-12)
    return np.stack([vec0, vec1, vec2, position], axis=1)


def _interp_pose(A, B, alpha, dist=0.1):
    """Interpolate between two c2w poses at fraction alpha using AnySplat's
    keyframe-pose scheme: linearly interpolate the (position, lookat-point,
    up-point) triplet then rebuild with _viewmatrix. alpha=0 -> A, 1 -> B. Any
    convention preserved at the endpoints; scene-facing throughout."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)

    def pts(c):
        pos = c[:3, 3]
        lookat = c[:3, 3] - dist * c[:3, 2]
        up = c[:3, 3] + dist * c[:3, 1]
        return pos, lookat, up

    (pa, la, ua), (pb, lb, ub) = pts(A), pts(B)
    a = float(alpha)
    pos = (1 - a) * pa + a * pb
    lookat = (1 - a) * la + a * lb
    up = (1 - a) * ua + a * ub
    m = _viewmatrix(pos - lookat, up - pos, pos)   # 3x4
    M = np.eye(4, dtype=np.float64); M[:3, :] = m
    return M


def _smooth_alpha_schedule(num_frames, start=0.0, end=1.0, schedule="cosine"):
    """AnySplat-style temporal interpolation schedule with exact endpoints."""
    if int(num_frames) < 2:
        raise ValueError(f"smooth camera path needs at least 2 frames, got {num_frames}")
    t = np.linspace(0.0, 1.0, int(num_frames), dtype=np.float64)
    if schedule == "cosine":
        # Matches AnySplat model_wrapper.render_video_generic smooth=True.
        t = (np.cos(np.pi * (t + 1.0)) + 1.0) * 0.5
    elif schedule != "linear":
        raise ValueError(f"unknown smooth camera schedule: {schedule}")
    return (float(start) + (float(end) - float(start)) * t).astype(np.float32)


def _load_flow(ann, cam, t):
    u = cv2.imread(f"{ann}/flow/{cam}/{t}/flow_u_16.png", cv2.IMREAD_UNCHANGED)
    v = cv2.imread(f"{ann}/flow/{cam}/{t}/flow_v_16.png", cv2.IMREAD_UNCHANGED)
    if u is None or v is None:
        return None
    u = u.astype(np.float32) / 65535.0 * (_FLOW_MAX - _FLOW_MIN) + _FLOW_MIN
    v = v.astype(np.float32) / 65535.0 * (_FLOW_MAX - _FLOW_MIN) + _FLOW_MIN
    return np.stack([u, v], -1)


def _load_depth(ann, cam, t):
    d = cv2.imread(f"{ann}/foundation_stereo/{cam}/{t}.png", cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float32) / 65535.0 * _MAX_DEPTH


def _hold_fill(world, valid):
    """Forward-fill invalid (T,M) frames in (T,M,3) by holding the most-recent
    valid position. The caller guarantees frame 0 is valid, so no leading gap
    remains and the result is fully finite — the valid mask still tells the
    model which frames to ignore."""
    T, M = valid.shape
    src = np.where(valid, np.arange(T)[:, None], 0)
    np.maximum.accumulate(src, axis=0, out=src)
    cols = np.arange(M)[None, :]
    return world[src, cols, :]


class OMNI_DROID_NVS(EasyDataset):
    def __init__(self, ROOT, index_file, num_views=49, resolution=(256, 144),
                 steps_per_epoch=10000, track_spatial_stride=1, grid_step=36,
                 frame_stride=2, max_tracks=4096, seed=None, swap_src_tgt_prob=0.5,
                 same_view_prob=0.0,
                 z_min=0.1, pw_index_file=None, require_optimized_extrinsics=False,
                 detect_occlusion=True, min_vis_frames=4,
                 occ_depth_rel=0.5, occ_depth_abs=0.15, cache_manifest=None,
                 caption_dir=None,
                 return_source_depth_video=False, return_source_rgb_video=False,
                 return_source_cameras=False, return_target_depth=False,
                 return_target_tracks=False, return_dynamic_masks=False, **_ignored):
        self.ROOT = ROOT
        self._num_views = int(num_views)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.W, self.H = int(resolution[0]), int(resolution[1])
        self.steps_per_epoch = int(steps_per_epoch)
        self.track_spatial_stride = int(track_spatial_stride)
        self.grid_step = int(grid_step)
        self.frame_stride = int(frame_stride)
        self.max_tracks = int(max_tracks)
        self.seed = seed
        self.swap_prob = float(swap_src_tgt_prob)
        self.same_view_prob = float(same_view_prob)
        if not 0.0 <= self.same_view_prob <= 1.0:
            raise ValueError(f"same_view_prob must be in [0,1], got {self.same_view_prob}")
        self.z_min = float(z_min)
        # Occlusion / visibility: flow-chaining from a frame-0 grid leaks
        # background points onto entering occluders (e.g. a gripper that is not
        # in frame 0). Detect that via out-of-frame + depth-jump latching, mask
        # the captured frames, and emit a per-frame validity mask.
        self.detect_occlusion = bool(detect_occlusion)
        self.min_vis_frames = int(min_vis_frames)
        self.occ_depth_rel = float(occ_depth_rel)
        self.occ_depth_abs = float(occ_depth_abs)
        # Per-scene RICH caption T5 embeddings (canonical cut3r sharded cache).
        # captions_all.json keys "{uuid}/{start}-{end}" -> totalcaption+chunk text;
        # precompute_omni_droid_t5.py -> captions_t5_emb shards + shard_index.json,
        # read via CachedT5EmbStore.get_by_key(key). Per sample we pick the chunk
        # whose frame range best overlaps the sampled window. Miss (no caption /
        # cache absent) falls back to the unified DEFAULT_PROMPT (in the trainer's
        # pipe.precomputed_prompt_embs), so it is safe before the store exists.
        self.cap_store = None
        self.cap_text = {}
        self.cap_chunks = {}      # uuid -> list[(start, end, key)]
        if caption_dir:
            try:
                from .mixed_nvs_captions import CachedT5EmbStore
                store = CachedT5EmbStore(caption_dir)
                self.cap_store = store if store.available else None
            except Exception:
                self.cap_store = None
            cap_json = os.path.join(caption_dir, "captions_all.json")
            if self.cap_store is not None and os.path.exists(cap_json):
                try:
                    self.cap_text = json.load(open(cap_json))
                    for key in self.cap_text:
                        uu, _, rng = key.rpartition("/")     # uuid has no "/"
                        a, _, b = rng.partition("-")
                        self.cap_chunks.setdefault(uu, []).append((int(a), int(b), key))
                except Exception:
                    self.cap_text, self.cap_chunks = {}, {}
        self.r_src_depth = bool(return_source_depth_video)
        self.r_src_rgb = bool(return_source_rgb_video)
        self.r_src_cam = bool(return_source_cameras)
        self.r_tgt_depth = bool(return_target_depth)
        # PointWorld OPTIMIZED extrinsics (bundle-adjusted) for accurate cross-camera
        # poses. OmniWorld's own meta extrinsics are RAW DROID (cross-cam reproj ~cm
        # off). Map uuid -> PointWorld flows file; used when the scene overlaps.
        self.pw_flows = {}
        if pw_index_file and os.path.exists(pw_index_file):
            for e in json.load(open(pw_index_file)):
                if e.get("uuid") and e.get("flows"):
                    self.pw_flows[e["uuid"]] = e["flows"]
        self.require_optimized = bool(require_optimized_extrinsics)
        # Offline flow->track cache (built by build_omni_track_cache.py). When a
        # scene has a cache entry, _sample reads a pre-chained clip (uv,z,vis)
        # instead of re-decoding flow PNGs — the ~9s chaining cost disappears.
        # uuid -> npz path. World lift + cam0 norm + extrinsics still done here.
        self.cache = {}
        if cache_manifest and os.path.exists(cache_manifest):
            man = json.load(open(cache_manifest))
            for u, rec in man.items():
                if u == "_meta":
                    continue
                p = rec.get("npz") if isinstance(rec, dict) else rec
                if p and os.path.exists(p):
                    self.cache[u] = p
        with open(index_file) as f:
            self.index = json.load(f)
        if self.require_optimized and self.pw_flows:
            self.index = [e for e in self.index if e.get("uuid") in self.pw_flows]
        if not self.index:
            raise RuntimeError(f"empty OmniWorld DROID index: {index_file}")

    @staticmethod
    def _optimized_c2w(pw_flows_path, s1, s2):
        """Read PointWorld optimized world2cam for two serials -> (c2w1, c2w2)."""
        with h5py.File(pw_flows_path, "r") as f:
            def E(s):
                cam = f"camera_{s}_ext"
                ck = next(k for k in f if ":" in k and cam in f[k])
                return np.asarray(f[ck][cam]["extrinsic"][()], np.float32)
            return np.linalg.inv(E(s1)), np.linalg.inv(E(s2))

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
        rng = random.Random(None if self.seed is None else self.seed + int(idx if isinstance(idx, int) else idx[0]))
        for _ in range(32):
            try:
                return self._sample(rng)
            except Exception:
                continue
        raise RuntimeError("OMNI_DROID_NVS: too many failed samples")

    # ---- core ------------------------------------------------------------ #
    def _chain_lift(self, ann, cam, K, c2w, start, T, stride, W, H):
        """Chain flow over native frames [start, start+stride*(T-1)] integrating
        EVERY native frame, sample at stride -> (T,N,2) uv; lift to world (T,N,3).

        Also returns a per-frame visibility mask `vis (T,N)`. A point is invisible
        at frame t if its depth is out of range, it has left the image, or it has
        been latched off by occlusion detection (leaving the frame, or an abrupt
        depth jump = captured by an entering occluder such as the gripper)."""
        s = self.grid_step
        gx, gy = np.meshgrid(np.arange(s // 2, W, s), np.arange(s // 2, H, s))
        P0 = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
        N = len(P0)
        last = start + stride * (T - 1)
        cur = P0.copy()
        pos_disp = {}   # di -> (N,2) chained position at displayed frame di
        inb_disp = {}   # di -> (N,) bool in-bounds (computed on the un-clipped pos)
        for t in range(start, last + 1):
            if (t - start) % stride == 0:
                di = (t - start) // stride
                pos_disp[di] = cur.copy()
                inb_disp[di] = ((cur[:, 0] >= 0) & (cur[:, 0] < W) &
                                (cur[:, 1] >= 0) & (cur[:, 1] < H))
            if t == last:
                break
            fl = _load_flow(ann, cam, t)
            if fl is None:
                raise RuntimeError(f"missing flow {cam}/{t}")
            d = cv2.remap(fl, np.clip(cur[:, 0], 0, W - 1).reshape(1, -1),
                          np.clip(cur[:, 1], 0, H - 1).reshape(1, -1), cv2.INTER_LINEAR)[0]
            cur = cur + d
        # lift each displayed frame via its own depth + track visibility
        UV = np.zeros((T, N, 2), np.float32)
        world = np.full((T, N, 3), np.nan, np.float32)
        vis = np.zeros((T, N), bool)
        Kinv = np.linalg.inv(K)
        alive = np.ones(N, bool)              # latched once OOB / occluded
        z_prev = np.full(N, np.nan, np.float32)   # last trusted depth per point
        for di in range(T):
            uv = pos_disp[di]; UV[di] = uv
            inb = inb_disp[di]
            if self.detect_occlusion:
                alive &= inb                  # leaving the frame kills the track
            native_t = start + di * stride
            dep = _load_depth(ann, cam, native_t)
            if dep is None:
                continue                      # vis stays False this frame
            u = np.clip(uv[:, 0], 0, W - 1); v = np.clip(uv[:, 1], 0, H - 1)
            z = dep[v.astype(int), u.astype(int)]
            good = (z > self.z_min) & (z < _MAX_DEPTH)
            if self.detect_occlusion:
                # depth-jump latch: a background point captured by an entering
                # occluder shows an abrupt change vs the last trusted depth.
                seen = np.isfinite(z_prev)
                jump = (seen & good
                        & (np.abs(z - z_prev) > self.occ_depth_abs)
                        & (np.abs(z - z_prev) > self.occ_depth_rel * np.maximum(z_prev, 1e-3)))
                alive &= ~jump
            campt = (Kinv @ np.stack([u, v, np.ones(N)], 0)) * z[None, :]
            wp = (c2w[:3, :3] @ campt).T + c2w[:3, 3]
            frame_valid = good & inb & alive
            wp[~frame_valid] = np.nan
            world[di] = wp
            vis[di] = frame_valid
            z_prev = np.where(good, z, z_prev)   # update reference on good depths
        return UV, world, P0, vis

    @staticmethod
    def _lift_cached(npz, serial, clip_i, K, c2w, Wn, Hn):
        """Lift a cached clip's (uv, z, vis) to world (T,N,3) + vis (T,N).

        Mirrors the lift in _chain_lift; the cache stored camera-space uv/z so
        the world lift, extrinsics and cam0 norm stay deferred to here."""
        uv = npz[f"uv__{serial}"][clip_i].astype(np.float32)   # (T,N,2)
        z = npz[f"z__{serial}"][clip_i].astype(np.float32)     # (T,N)
        vis = np.asarray(npz[f"vis__{serial}"][clip_i], bool)  # (T,N)
        Kinv = np.linalg.inv(K)
        u = np.clip(uv[..., 0], 0, Wn - 1); v = np.clip(uv[..., 1], 0, Hn - 1)
        pix = np.stack([u, v, np.ones_like(u)], -1)            # (T,N,3)
        campt = pix @ Kinv.T * z[..., None]                    # (T,N,3)
        world = campt @ c2w[:3, :3].T + c2w[:3, 3]
        world[~vis] = np.nan
        return world.astype(np.float32), vis

    def _load_rgb(self, scene_dir, serial, start, T, stride, W, H):
        cap = cv2.VideoCapture(os.path.join(scene_dir, "recordings", "MP4", f"{serial}.mp4"))
        want = set(start + i * stride for i in range(T))
        last = start + stride * (T - 1)
        out = {}
        t = 0
        while t <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if t in want:
                out[t] = Image.fromarray(cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2RGB))
            t += 1
        cap.release()
        if len(out) < T:
            raise RuntimeError("too few RGB frames")
        return [out[start + i * stride] for i in range(T)]

    def _load_depth_video(self, ann, serial, start, T, stride, W, H):
        out = []
        for i in range(T):
            d = _load_depth(ann, serial, start + i * stride)
            if d is None:
                d = np.zeros((H, W), np.float32)
            else:
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            out.append(d)
        return np.stack(out)

    def _sample(self, rng):
        ent = self.index[rng.randrange(len(self.index))]
        return self._build_sample(ent, rng)

    def _raw_scene_geo(self, ent):
        """Read a scene's raw src/tgt cameras (PointWorld-optimized when the scene
        overlaps, else RAW DROID). Used to transplant a donor camera rig for the
        borrowed-view eval. Returns (c2w_src, c2w_tgt, Ksrc, Ktgt) float64/float32."""
        ann = ent["ann_dir"]
        if not os.path.isabs(ann):
            ann = os.path.join(self.ROOT, ann)
        meta = json.load(open(os.path.join(ann, "meta_info.json")))
        s1, s2 = str(meta["ext1_cam_serial"]), str(meta["ext2_cam_serial"])
        ekey = {s1: "ext1_cam_extrinsics", s2: "ext2_cam_extrinsics"}
        ci = np.load(os.path.join(ann, "recordings", "camera_info_dict.npy"),
                     allow_pickle=True).item()
        Ksrc = np.asarray(ci[s1]["cam_matrix"], np.float32)
        Ktgt = np.asarray(ci[s2]["cam_matrix"], np.float32)
        c2w_src = _sixdof_to_c2w(np.asarray(meta[ekey[s1]], np.float64))
        c2w_tgt = _sixdof_to_c2w(np.asarray(meta[ekey[s2]], np.float64))
        pwf = self.pw_flows.get(meta.get("uuid"))
        if pwf and os.path.exists(pwf):
            try:
                c2w_s_opt, c2w_t_opt = self._optimized_c2w(pwf, s1, s2)
                c2w_src, c2w_tgt = c2w_s_opt.astype(np.float64), c2w_t_opt.astype(np.float64)
            except Exception:
                pass
        return c2w_src, c2w_tgt, Ksrc, Ktgt

    def load_borrowed_view_sample(self, scene_i, donor_ent, seed=42):
        """DETERMINISTIC eval sample: render held-out scene ``self.index[scene_i]``
        (src=ext1) to a STATIC novel camera transplanted from ``donor_ent``'s
        ext1->ext2 rig. Frame-correct in the scene's own metric world:
            c2w_tgt = c2w_src_scene @ (inv(c2w_src_donor) @ c2w_tgt_donor)
        intrinsics borrowed from the donor's target cam. There is no ground-truth
        image for the borrowed pose, so this is a perceptual / novel-trajectory
        eval (the returned ``target_video`` is the scene's real ext2 view, kept
        only as a reference panel — NOT a paired GT)."""
        ent = self.index[scene_i]
        c2w_src_A, _c2w_tgt_A, _Ks_A, _Kt_A = self._raw_scene_geo(ent)
        c2w_src_D, c2w_tgt_D, _Ks_D, Ktgt_D = self._raw_scene_geo(donor_ent)
        M_rel = np.linalg.inv(c2w_src_D) @ c2w_tgt_D          # tgt in donor's src frame
        borrow_c2w_tgt = (c2w_src_A @ M_rel).astype(np.float64)
        rng = random.Random(seed)
        return self._build_sample(ent, rng, no_swap=True, det_clip=True,
                                  borrow_c2w_tgt=borrow_c2w_tgt, borrow_Ktgt=Ktgt_D)

    def load_interpolated_view_sample(self, scene_i, alpha=0.5, seed=42):
        """DETERMINISTIC eval sample rendering held-out scene ``self.index[scene_i]``
        (src=ext1) to a STATIC virtual camera whose center is interpolated between
        the source (ext1) and target (ext2) cameras at fraction ``alpha`` and which
        LOOKS AT the scene centroid (the shared center both real cameras observe).
        alpha=0 == source view, alpha=1 == target view, alpha=0.5 == midpoint. Unlike
        the donor-borrow variant this always yields a meaningful, scene-facing view."""
        ent = self.index[scene_i]
        rng = random.Random(seed)
        return self._build_sample(ent, rng, no_swap=True, det_clip=True,
                                  interp_alpha=float(alpha))

    def load_same_view_sample(self, scene_i, seed=42):
        """Deterministic identity-camera eval without target-video leakage.

        Source and target are both ext1, but ``inference_target_video`` is
        explicitly ``None`` so the generated target still starts from noise.
        The ext1 target video is retained only as paired reconstruction GT.
        """
        ent = self.index[scene_i]
        rng = random.Random(seed)
        return self._build_sample(
            ent,
            rng,
            no_swap=True,
            det_clip=True,
            force_same_view=True,
        )

    def load_smooth_interpolated_view_sample(
        self,
        scene_i,
        alpha_start=0.0,
        alpha_end=1.0,
        schedule="cosine",
        seed=42,
    ):
        """Render a per-frame camera path from ext1 to ext2.

        Camera pose and intrinsics follow the same interpolation family as the
        static-alpha path. The default cosine alpha schedule matches AnySplat's
        smooth video rendering schedule and preserves both endpoints exactly.
        """
        ent = self.index[scene_i]
        rng = random.Random(seed)
        interp_alphas = _smooth_alpha_schedule(
            self._num_views, start=alpha_start, end=alpha_end, schedule=schedule
        )
        return self._build_sample(
            ent,
            rng,
            no_swap=True,
            det_clip=True,
            interp_alphas=interp_alphas,
        )

    def _build_sample(self, ent, rng, *, no_swap=False, det_clip=False,
                      force_same_view=False,
                      borrow_c2w_tgt=None, borrow_Ktgt=None, interp_alpha=None,
                      interp_alphas=None):
        T, W, H = self._num_views, self.W, self.H
        ann, scene_dir, nfr = ent["ann_dir"], ent["raw_scene_dir"], int(ent["n_frames"])
        # ann_dir may be relative to ROOT (SquashFS-mounted layout); raw_scene_dir
        # (DROID MP4s) stays absolute. Absolute ann_dir keeps the plain-extracted
        # layout working unchanged.
        if not os.path.isabs(ann):
            ann = os.path.join(self.ROOT, ann)
        meta = json.load(open(os.path.join(ann, "meta_info.json")))
        s1, s2 = str(meta["ext1_cam_serial"]), str(meta["ext2_cam_serial"])
        src, tgt = (s1, s2)
        ekey = {s1: "ext1_cam_extrinsics", s2: "ext2_cam_extrinsics"}
        if (not no_swap) and rng.random() < self.swap_prob:
            src, tgt = tgt, src
        if force_same_view:
            if any(x is not None for x in (
                borrow_c2w_tgt, borrow_Ktgt, interp_alpha, interp_alphas,
            )):
                raise ValueError("force_same_view cannot be combined with a novel target camera")
            tgt = src
        # Identity-camera augmentation is training-only. Keep all deterministic
        # novel-view/eval helpers (`no_swap=True`) on the real ext1->ext2 pair.
        # The explicit >0 guard preserves the old RNG stream at the default.
        if (not no_swap) and self.same_view_prob > 0.0 and rng.random() < self.same_view_prob:
            tgt = src
        ci = np.load(os.path.join(ann, "recordings", "camera_info_dict.npy"), allow_pickle=True).item()
        Ksrc = np.asarray(ci[src]["cam_matrix"], np.float32)
        Ktgt = np.asarray(ci[tgt]["cam_matrix"], np.float32)
        # Extrinsics: prefer PointWorld OPTIMIZED (accurate cross-camera) when the
        # scene overlaps; else fall back to OmniWorld RAW DROID extrinsics (~cm off).
        c2w_src = _sixdof_to_c2w(np.asarray(meta[ekey[src]], np.float64))
        c2w_tgt = _sixdof_to_c2w(np.asarray(meta[ekey[tgt]], np.float64))
        pwf = self.pw_flows.get(meta.get("uuid"))
        if pwf and os.path.exists(pwf):
            try:
                c2w_s_opt, c2w_t_opt = self._optimized_c2w(pwf, src, tgt)
                c2w_src, c2w_tgt = c2w_s_opt.astype(np.float64), c2w_t_opt.astype(np.float64)
            except Exception:
                pass
        Wn, Hn = 1280, 720

        # Fast path: read a pre-chained clip from the offline cache (skips the
        # ~9s flow-PNG decode + chaining). Falls back to live chaining if the
        # scene is not cached. The cached clip fixes (start, stride).
        npz = None
        cpath = self.cache.get(meta.get("uuid"))
        if cpath is not None:
            try:
                npz = np.load(cpath, allow_pickle=True)
            except Exception:
                npz = None
        if npz is not None:
            stride = int(npz["frame_stride"])
            starts = np.asarray(npz["clip_starts"]).tolist()
            clip_i = 0 if det_clip else rng.randrange(len(starts))
            start = int(starts[clip_i])
            Wn, Hn = int(npz["W"]), int(npz["H"])
            world_s, vis_s = self._lift_cached(npz, src, clip_i, Ksrc, c2w_src, Wn, Hn)
            world_t, vis_t = self._lift_cached(npz, tgt, clip_i, Ktgt, c2w_tgt, Wn, Hn)
            if world_s.shape[0] < T or world_t.shape[0] < T:
                raise RuntimeError(
                    f"cached clip shorter than requested views: "
                    f"source={world_s.shape[0]} target={world_t.shape[0]} requested={T}"
                )
            # Eval inspection utilities may request a short prefix (for example
            # a 9-frame comparison-format smoke) from the canonical 49f cache.
            world_s, vis_s = world_s[:T], vis_s[:T]
            world_t, vis_t = world_t[:T], vis_t[:T]
        else:
            stride = max(1, min(self.frame_stride, (nfr - 1) // (T - 1)))
            if (T - 1) * stride + 1 > nfr:
                raise RuntimeError("scene too short")
            start = 0 if det_clip else rng.randrange(0, nfr - (T - 1) * stride)
            # chain + lift BOTH external cameras (PointWorld-style union)
            uv_s, world_s, _P0s, vis_s = self._chain_lift(ann, src, Ksrc, c2w_src, start, T, stride, Wn, Hn)
            uv_t, world_t, _P0t, vis_t = self._chain_lift(ann, tgt, Ktgt, c2w_tgt, start, T, stride, Wn, Hn)
        keep_w, keep_v, keep_origin = [], [], []
        camera_track_sets = ((world_s, vis_s),) if src == tgt else (
            (world_s, vis_s),
            (world_t, vis_t),
        )
        for origin, (w, vmask) in enumerate(camera_track_sets):
            # keep tracks anchored (valid) at the query frame (frame 0) with at
            # least min_vis_frames visible frames; occluded frames are kept but
            # masked out, and their world position is held so no NaN leaks into
            # the cam0-scale / embedder arithmetic.
            anchor = vmask[0] & np.isfinite(w[0]).all(-1)
            enough = vmask.sum(0) >= self.min_vis_frames
            m = anchor & enough
            if m.sum():
                vk = vmask[:, m].copy()
                wk = _hold_fill(w[:, m].copy(), vk)
                keep_w.append(wk); keep_v.append(vk)
                keep_origin.append(np.full(wk.shape[1], origin, dtype=np.int8))
        if not keep_w:
            raise RuntimeError("no valid lifted tracks")
        world = np.concatenate(keep_w, axis=1)       # (T, M, 3) world, union
        valid = np.concatenate(keep_v, axis=1)       # (T, M) per-frame validity
        track_origin_camera = np.concatenate(keep_origin, axis=0)  # 0=source, 1=real target
        if world.shape[1] > self.max_tracks:
            if det_clip:
                sel = np.arange(self.max_tracks)
            else:
                sel = np.asarray(rng.sample(range(world.shape[1]), self.max_tracks))
            world = world[:, sel]; valid = valid[:, sel]
            track_origin_camera = track_origin_camera[sel]

        # ---- cam0 normalization (cam0 = src cam, static external) ----
        Esrc = np.linalg.inv(c2w_src).astype(np.float32)     # world2cam(src)
        pw = world.reshape(-1, 3)
        pcam0 = (Esrc[:3, :3] @ pw.T).T + Esrc[:3, 3]
        common_scale = max(float(np.linalg.norm(pcam0, axis=1).mean()), 1e-6)
        tracks_cam0 = (pcam0 / common_scale).reshape(T, -1, 3).astype(np.float32)
        tracks_world = np.transpose(tracks_cam0, (1, 0, 2)).copy()   # (M,T,3)
        tracks_world_valid_mask = np.transpose(valid, (1, 0)).copy()  # (M,T) bool

        w2c_cam0 = Esrc
        def norm_pose(c2w):
            p = (w2c_cam0 @ c2w).astype(np.float32); p[:3, 3] /= common_scale; return p
        _c2w_tgt_render = c2w_tgt if borrow_c2w_tgt is None else borrow_c2w_tgt.astype(np.float64)
        src_pose = norm_pose(c2w_src)
        real_tgt_pose = norm_pose(c2w_tgt)
        tgt_pose = norm_pose(_c2w_tgt_render)
        target_cam2world = np.broadcast_to(tgt_pose, (T, 4, 4)).astype(np.float32).copy()
        source_cam2world = np.broadcast_to(src_pose, (T, 4, 4)).astype(np.float32).copy()
        real_target_cam2world = np.broadcast_to(real_tgt_pose, (T, 4, 4)).astype(np.float32).copy()
        sx, sy = W / Wn, H / Hn
        _Ktgt_render = Ktgt if borrow_Ktgt is None else np.asarray(borrow_Ktgt, np.float32)
        Ksrc_o = _resize_K(Ksrc, sx, sy)
        real_Ktgt_o = _resize_K(Ktgt, sx, sy)
        Ktgt_o = _resize_K(_Ktgt_render, sx, sy)

        # Interpolated virtual camera (normalized cam0 frame): center lerped between
        # source & target, oriented to look at the scene centroid (shared center).
        # Overrides the target pose + intrinsics; always a meaningful scene-facing view.
        if interp_alpha is not None:
            a = float(interp_alpha)
            tgt_pose = _interp_pose(src_pose, tgt_pose, a).astype(np.float32)
            target_cam2world = np.broadcast_to(tgt_pose, (T, 4, 4)).astype(np.float32).copy()
            Ktgt_o = ((1.0 - a) * Ksrc_o + a * Ktgt_o).astype(np.float32)
        elif interp_alphas is not None:
            alpha_path = np.asarray(interp_alphas, dtype=np.float32)
            if alpha_path.shape != (T,):
                raise ValueError(f"interp_alphas must have shape ({T},), got {alpha_path.shape}")
            target_cam2world = np.stack(
                [_interp_pose(src_pose, tgt_pose, float(a)) for a in alpha_path], axis=0
            ).astype(np.float32)
            target_intrinsics_per_frame = np.stack(
                [(1.0 - float(a)) * Ksrc_o + float(a) * Ktgt_o for a in alpha_path], axis=0
            ).astype(np.float32)
            Ktgt_o = target_intrinsics_per_frame[0]

        src_video = self._load_rgb(scene_dir, src, start, T, stride, W, H)
        tgt_video = self._load_rgb(scene_dir, tgt, start, T, stride, W, H)
        src_depth_vid = self._load_depth_video(ann, src, start, T, stride, W, H) / common_scale
        tgt_depth_vid = self._load_depth_video(ann, tgt, start, T, stride, W, H) / common_scale

        src_rgb01 = np.stack([np.asarray(im, np.float32) / 255.0 for im in src_video])

        # source RGB at each track, sampled at the QUERY frame (q=0) on the source
        # image and broadcast over T — matches _source_rgb_at_track_grid used by
        # the grid datasets, so DROID feeds real per-track color into the Conv3D
        # adapter's RGB channels (not zeros). cam0 == source camera, so the cam0
        # tracks (un-normalized) are source-camera 3D; project with Ksrc_o.
        q = 0
        cam3d = tracks_cam0[q] * common_scale                       # (M,3) src-cam coords
        zc = cam3d[:, 2]
        zsafe = np.clip(zc, 1e-6, None)
        u = Ksrc_o[0, 0] * cam3d[:, 0] / zsafe + Ksrc_o[0, 2]
        v = Ksrc_o[1, 1] * cam3d[:, 1] / zsafe + Ksrc_o[1, 2]
        ui = np.clip(np.round(u).astype(int), 0, W - 1)
        vi = np.clip(np.round(v).astype(int), 0, H - 1)
        rgb_q = src_rgb01[q][vi, ui, :].astype(np.float32)          # (M,3)
        inb = (zc > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid[q]
        rgb_q[~inb] = 0.0
        source_rgb_at_tracks = np.broadcast_to(rgb_q[:, None, :], (rgb_q.shape[0], T, 3)).copy()

        # rich per-scene caption: pick the chunk whose frame range best overlaps
        # the sampled window [start, start+span]; miss -> unified cached prompt.
        prompt, prompt_emb = DEFAULT_PROMPT, None
        cand = self.cap_chunks.get(ent.get("uuid")) if self.cap_store is not None else None
        if cand:
            wlo, whi = start, start + stride * (T - 1)
            _a, _b, key = max(cand, key=lambda c: min(c[1], whi) - max(c[0], wlo))
            emb = self.cap_store.get_by_key(key)
            txt = self.cap_text.get(key)
            if emb is not None and isinstance(txt, str) and txt.strip():
                prompt, prompt_emb = txt.strip(), emb

        sample = {
            "video": src_video,
            "target_video": tgt_video,
            "tracks_world": tracks_world,
            "tracks_world_valid_mask": tracks_world_valid_mask,
            "target_cam2world": target_cam2world,
            "target_intrinsics": Ktgt_o,
            "common_scale": common_scale,
            "track_query_idx": 0,
            "prompt": prompt,          # per-scene caption if cached, else unified default
            "prompt_emb": prompt_emb,  # bf16 (L,dim) from CachedT5EmbStore, or None
            "has_tracks": True,
            "has_dynamic_tracks": True,
            "has_depth": True,
            "target_depth": tgt_depth_vid.astype(np.float32),
            "source_depth": src_depth_vid[0].astype(np.float32),
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "source_rgb_at_tracks": source_rgb_at_tracks,
            # OmniDROID novel-view visualization metadata. The model ignores
            # these keys; they keep the real ext2 reference distinct from the
            # virtual render camera and preserve source-track provenance.
            "track_origin_camera": track_origin_camera,
            "real_target_cam2world": real_target_cam2world,
            "real_target_intrinsics": real_Ktgt_o,
            "source_camera_name": src,
            "real_target_camera_name": tgt,
            "source_target_same_view": bool(src == tgt),
            "same_view_sampling_prob": float(self.same_view_prob),
        }
        novel_view_mode = None
        novel_view_alpha = None
        if interp_alpha is not None:
            novel_view_mode = "interp"
            novel_view_alpha = float(interp_alpha)
        elif interp_alphas is not None:
            novel_view_mode = "smooth_interp"
        elif borrow_c2w_tgt is not None:
            novel_view_mode = "borrow"
        if force_same_view:
            # Match training's source-to-source task while preventing eval GT
            # from entering the target latent. This is a reconstruction test,
            # not target-video-conditioned denoising.
            sample["omni_droid_novel_view_mode"] = "same_view"
            sample["inference_target_video"] = None
            sample["target_video_is_ground_truth"] = True
        elif novel_view_mode is not None:
            # target_video remains useful as a real ext2 reference panel, but
            # it is not the image at the requested virtual camera. Explicitly
            # tell eval to initialize the target domain from noise.
            sample["omni_droid_novel_view_mode"] = novel_view_mode
            sample["omni_droid_interp_alpha"] = novel_view_alpha
            if interp_alphas is not None:
                sample["omni_droid_interp_alphas"] = np.asarray(interp_alphas, dtype=np.float32)
                sample["target_intrinsics_per_frame"] = target_intrinsics_per_frame
            sample["inference_target_video"] = None
            sample["target_video_is_ground_truth"] = bool(
                novel_view_mode == "interp" and np.isclose(novel_view_alpha, 1.0)
            )
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
