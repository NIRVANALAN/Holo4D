"""
Syn4D NVS dataset for Track4D-360 training.

Returns paired (source, target) camera data for novel view synthesis.
Wraps Syn4D_Multi internally, reusing all existing Syn4D data loading
infrastructure (camera CSV, depth EXR, mesh-based vertex tracking).

YAML-compatible — use in configs like:
    "500 @ Syn4D_Multi_NVS(ROOT='/path/to/Syn4D', num_views=49)"

Each __getitem__ returns:
  - video:              list of PIL Images (source camera RGB)
  - target_video:       list of PIL Images (target camera RGB)
  - tracks_world:       (N, T, 3) float32  — world-space 3D tracks
  - target_cam2world:   (T, 4, 4) float32  — target camera extrinsics
  - target_intrinsics:  (3, 3) float32      — target camera K matrix
  - prompt:             str                 — scene caption

With return_target_tracks=True (Stage 2 joint tracking):
  - target_track_video: (T, H, W, 3) float32 — target-view dense 3D positions (cam0-normalized)
  - target_track_valid_mask: (T, H, W) bool  — per-pixel validity mask

With return_target_depth=True:
  - target_depth_video: (T, H, W) float32 — target-view depth in target-camera coordinates,
    scaled by the same source-derived common_scale as tracks/cameras
  - target_depth_valid_mask: (T, H, W) bool — valid positive depth pixels
"""

import json
import os
import os.path as osp
import random
import re
from collections import defaultdict

import cv2
import imageio.v3 as iio
import numpy as np
import PIL.Image
import torch

from .base.easy_dataset import EasyDataset
from .syn4d import Syn4D_Multi, precompute_all_wrapped_pts3d
from .track_query_sampling import (
    choose_canonical_random_query_indices as _choose_canonical_random_query_indices,
    choose_track_query_indices as _choose_track_query_indices,
)


# Updated 2026-04-30: unified with track4d360_mixed_nvs.DEFAULT_PROMPT — one
# task-meaningful NVS fallback for every dataset. Must be in
# scripts/wan2.1_fun/train_track4d360.py:_DEFAULT_PROMPTS_TO_PRECOMPUTE so
# the strict-cache check finds a precomputed embedding for the fallback.
DEFAULT_PROMPT = "Generate the novel view trajectory video given the input video."


def compact_tracking_if_available(loader, scene, camera):
    getter = getattr(loader, '_get_tracking_safetensor', None)
    if getattr(loader, 'tracking_format', 'npz') in ('auto', 'safetensor', 'compact_faceid') and callable(getter):
        return getter(scene, camera)
    return None


def _parse_int_choices(value, default=(1,)):
    if isinstance(value, (tuple, list)):
        raw = value
    elif isinstance(value, str):
        raw = [p for p in re.split(r"[,| ]+", value.strip()) if p]
    else:
        raw = [value]
    out = []
    for item in raw:
        v = int(item)
        if v > 0:
            out.append(v)
    return tuple(out or tuple(default))


class Syn4D_Multi_NVS(EasyDataset):
    """Syn4D paired-camera dataset for Track4D-360 NVS training.

    Inherits from ``EasyDataset`` for YAML compatibility (``"N @ Syn4D_Multi_NVS(...)"``)
    and compositional loading via ``CatDataset``.

    Wraps ``Syn4D_Multi`` via composition to reuse all existing data loading
    (camera CSV, depth EXR, mesh-based vertex tracking, index caching, etc.).

    Each sample randomly picks:
      - A multi-camera group (scene + base sequence)
      - Two different cameras (source, target)
      - A random start frame (uniform over [0, total - num_views])

    Source camera provides RGB + mesh-tracked 3D world-space tracks.
    Target camera provides RGB + camera pose + intrinsics.
    """

    def __init__(
        self,
        ROOT,
        num_views=49,
        resolution=(512, 288),
        steps_per_epoch=500,
        track_spatial_stride=1,
        max_temporal_stride=3,
        caption_dir=None,
        default_prompt=DEFAULT_PROMPT,
        scene=None,
        sequence_idx=None,
        fixed_start_frame=None,
        return_target_tracks=False,
        return_target_depth=False,
        random_query_frame=False,
        trackable_scene_suffixes=None,
        exclude_scenes=None,
        group_camera_pairs=None,
        group_camera_sample_prob=0.0,
        valid_sequences_file=None,
        use_mp4_rgb=False,
        return_source_depth_video=False,
        return_source_rgb_video=False,
        return_source_cameras=False,
        return_dynamic_masks=False,
        multi_query_prob=0.0,
        multi_query_choices="1",
        multi_query_schedule="legacy_fixed",
        seed=None,
        **syn4d_kwargs,
    ):
        """
        Args:
            ROOT: Syn4D dataset root (contains scene directories).
            num_views: Number of consecutive frames per sample (default 49).
            resolution: (width, height) output resolution (default 512x288 for 16:9).
            steps_per_epoch: Virtual epoch length.
            track_spatial_stride: Spatial stride for track grid. 1 = every pixel.
            max_temporal_stride: Max temporal stride for frame sampling (default 3).
                Each sample randomly picks stride in [1, max_temporal_stride].
                stride=1 has 50% probability, others split the remaining 50%.
            caption_dir: Directory with ``{scene}_per_video_chunks.json``.
                         Defaults to ``{ROOT}/caption``.
            default_prompt: Fallback caption when no caption found.
            scene: Optional scene name filter (partial match). For overfit training.
            sequence_idx: Optional index to select N-th sequence within scene.
            fixed_start_frame: If set, always use this start frame instead of
                random sampling. Use "first" to auto-select the first available
                trackable start frame. Useful for true single-example overfitting.
            return_target_tracks: If True, also load target-view dense 3D
                position maps via mesh tracking (Stage 2: joint NVS + 3D tracking).
                Returns additional keys: target_track_video (T, H, W, 3) and
                target_track_valid_mask (T, H, W). Default False preserves
                Stage 1 behavior.
            return_target_depth: If True, also load target-view depth maps for
                the target camera. Depth is divided by the same source-derived
                common_scale as the 3D tracks/cameras, but is otherwise left in
                raw target-camera depth units for later log-space normalization.
            trackable_scene_suffixes: Passed to Syn4D_Multi.
                None (default) includes ALL scenes with tracking data.
            exclude_scenes: Optional list of scene names to exclude (exact match).
                E.g. ['train_group', 'flying_group'] to hold out eval scenes.
            group_camera_pairs: Camera suffix pairs to oversample, e.g. "0-1,2-3,4-5,6-7".
                These camera pairs share the same first-frame pose in Syn4D, making them
                easier (larger overlap). Matches recammaster's group_scene_pairs.
            group_camera_sample_prob: Probability of sampling from group_camera_pairs
                instead of uniformly from all 8 cameras. E.g. 0.25 = 25% of the time
                the (src, tgt) pair is drawn from within a camera pair.
            valid_sequences_file: Path to a text file listing valid (scene, sequence) pairs,
                one per line in the format "<scene_name> <seq_id>" (e.g. "arena seq_000001").
                Only groups whose (scene_name, base_seq) appear in this file are kept.
                Also used to restrict Syn4D_Multi indexing to the valid scenes.
            use_mp4_rgb: If True, load Syn4D RGB from scene/mp4/<camera_seq>.mp4
                with Decord batch decoding when available. Missing or unreadable
                MP4s fall back to the original PNG frame path; depth/tracking still
                load from the existing EXR/NPZ path.
            **syn4d_kwargs: Additional keyword args forwarded to Syn4D_Multi.
        """
        self.ROOT = ROOT
        self._num_views = int(num_views)
        self.steps_per_epoch = int(steps_per_epoch)
        self.track_spatial_stride = int(track_spatial_stride)
        self.max_temporal_stride = max(1, int(max_temporal_stride))
        self.default_prompt = str(default_prompt)
        self.scene_filter = scene
        self.sequence_idx = sequence_idx
        self.fixed_start_frame = fixed_start_frame
        self.return_target_tracks = bool(return_target_tracks)
        self.return_target_depth = bool(return_target_depth)
        self.return_source_depth_video = bool(return_source_depth_video)
        self.return_source_rgb_video = bool(return_source_rgb_video)
        self.return_source_cameras = bool(return_source_cameras)
        self.return_dynamic_masks = bool(return_dynamic_masks)
        self.multi_query_prob = float(multi_query_prob)
        self.multi_query_choices = _parse_int_choices(multi_query_choices, default=(1,))
        self.multi_query_schedule = str(multi_query_schedule).strip().lower()
        if self.multi_query_schedule not in {"legacy_fixed", "canonical_random_subset"}:
            raise ValueError(f"unsupported multi_query_schedule={multi_query_schedule!r}")
        # q2-realism opt-in (default OFF → legacy behavior unchanged).
        # TRACK4D360_SYN4D_PER_FRAME_VALIDITY=1: emit real per-frame source-track
        #   visibility (tracks_world_valid_mask) instead of the legacy "valid only
        #   if valid in ALL frames" collapse. This is what real CoW/VGGT q2 eval
        #   feeds the model; without it, synthetic multi-query is persistent and
        #   the model never learns per-frame occlusion on q>1 identities.
        # TRACK4D360_MULTI_QUERY_VALIDITY_DROPOUT: extra independent per-frame
        #   visibility dropout applied to EXTRA-query (non-primary) identities, so
        #   q0/q48 disagree structurally (not just via additive geometry noise).
        self.per_frame_track_validity = bool(
            int(os.environ.get("TRACK4D360_SYN4D_PER_FRAME_VALIDITY", "0") or 0)
        )
        self.multi_query_validity_dropout = max(
            0.0,
            min(1.0, float(os.environ.get("TRACK4D360_MULTI_QUERY_VALIDITY_DROPOUT", "0.0") or 0.0)),
        )
        self.seed = None if seed is None else int(seed)
        self.use_mp4_rgb = bool(use_mp4_rgb)
        self._mp4_rgb_fallback_warned = False
        self.random_query_frame = bool(random_query_frame)
        self.exclude_scenes = set(exclude_scenes) if exclude_scenes else set()

        # Valid sequence filtering
        self._valid_seq_set = None
        if valid_sequences_file:
            self._valid_seq_set = self._parse_valid_sequences_file(valid_sequences_file)

        # Group camera pair oversampling (matches recammaster group_scene_pairs)
        self.group_camera_sample_prob = float(group_camera_sample_prob)
        self._group_camera_pairs = []
        if group_camera_pairs:
            for pair_str in str(group_camera_pairs).split(","):
                a, b = pair_str.strip().split("-")
                self._group_camera_pairs.append((int(a), int(b)))

        # Parse resolution
        if isinstance(resolution, int):
            self.output_width = self.output_height = resolution
        elif isinstance(resolution, (tuple, list)):
            self.output_width, self.output_height = int(resolution[0]), int(resolution[1])
        else:
            self.output_width = self.output_height = int(resolution)
        self.resolution = (self.output_width, self.output_height)

        # Caption directory + parallel T5 embedding cache (one safetensors per
        # scene: {caption_dir}/{scene_name}_per_video_chunks_t5_emb.safetensors,
        # written by holo4d/precompute_t5_cache.py). Both the JSON and
        # safetensors handles are lazily opened per scene the first time the
        # scene is sampled, then reused for the rest of the run.
        self.caption_dir = str(caption_dir) if caption_dir else os.path.join(ROOT, "caption")
        self._caption_json_cache = {}
        # Maps scene_name -> safe_open handle (or False sentinel if file absent).
        self._caption_emb_handle_cache = {}

        # Create internal Syn4D_Multi for data loading.
        # Uses num_views directly (stride=1). Each sample stores exactly num_views
        # consecutive frames. At sample time, we pick a temporal stride and construct
        # the strided frame list from the FULL frame list stored per camera in the
        # NVS group, capping stride by (available_frames // num_views).
        # This maximizes valid start positions — unlike the old approach of
        # internal_num_views = num_views * max_stride, which excluded starts
        # near the end of sequences.
        # NPZ tracking data only exists every 5 frames — Syn4D_Multi already
        # filters starts to NPZ-available frames.
        syn4d_kwargs["stride"] = 1
        # If a valid_sequences_file is given, restrict indexing to those scenes only
        if self._valid_seq_set is not None and "scene_include" not in syn4d_kwargs:
            syn4d_kwargs["scene_include"] = sorted({scene for scene, _ in self._valid_seq_set})
        # Forward the sequence filter to the raw loader as well.  Without this,
        # the NVS wrapper filters groups only *after* Syn4D_Multi scans every
        # camera sequence in a scene, which is prohibitively expensive for
        # one-sequence export/evaluation jobs.
        if valid_sequences_file and "valid_sequences_file" not in syn4d_kwargs:
            syn4d_kwargs["valid_sequences_file"] = valid_sequences_file
        self._syn4d = Syn4D_Multi(
            ROOT=ROOT,
            num_views=num_views,
            resolution=resolution,
            trackable_scene_suffixes=trackable_scene_suffixes,
            **syn4d_kwargs,
        )

        # Build NVS groups from internal dataset's samples
        self._nvs_groups = self._build_nvs_groups()
        if not self._nvs_groups:
            raise ValueError(
                f"No multi-camera NVS groups found in Syn4D root: {ROOT} "
                f"(scene={scene!r}, sequence_idx={sequence_idx!r})"
            )

        total_cameras = sum(len(g["camera_names"]) for g in self._nvs_groups)
        total_starts = sum(len(g["common_starts"]) for g in self._nvs_groups)
        fixed_info = ""
        if self.fixed_start_frame is not None:
            if str(self.fixed_start_frame).lower() == "first":
                resolved = self._nvs_groups[0]["common_starts"][0]
                fixed_info = f", fixed_start=first→{resolved}"
            else:
                fixed_info = f", fixed_start={self.fixed_start_frame}"
        tgt_track_info = ", return_target_tracks=True" if self.return_target_tracks else ""
        tgt_depth_info = ", return_target_depth=True" if self.return_target_depth else ""
        query_info = ", random_query_frame=True" if self.random_query_frame else ""
        pair_info = ""
        if self._group_camera_pairs and self.group_camera_sample_prob > 0:
            pair_strs = [f"{a}-{b}" for a, b in self._group_camera_pairs]
            pair_info = f", group_pairs={','.join(pair_strs)} @ {self.group_camera_sample_prob:.0%}"
        print(
            f"[Syn4D_Multi_NVS] {len(self._nvs_groups)} groups, "
            f"{total_cameras} total cameras, {total_starts} trackable start positions "
            f"(temporal_stride=1..{self.max_temporal_stride}, 50% stride=1, NPZ-filtered"
            f"{fixed_info}{tgt_track_info}{tgt_depth_info}{query_info}{pair_info})"
        )

    # ------------------------------------------------------------------
    # EasyDataset / CatDataset compatibility
    # ------------------------------------------------------------------

    @property
    def _resolutions(self):
        """Required by CatDataset for resolution consistency checks."""
        return [(self.output_width, self.output_height)]

    @property
    def num_views(self):
        """Required by CatDataset for view count consistency checks."""
        return self._num_views

    def set_epoch(self, epoch):
        """Called by ResizedDataset at each epoch for reproducible shuffling."""
        pass  # Syn4D_Multi_NVS uses random sampling, no epoch-dependent state needed

    @staticmethod
    def _parse_valid_sequences_file(path):
        """Parse a valid_sequences.txt file into a set of (scene_name, base_seq) tuples.

        Expected format: one entry per line, ``<scene_name> <seq_id>``,
        e.g. ``arena seq_000001``. Lines with fewer than two whitespace-separated
        tokens are ignored.
        """
        valid_set = set()
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    valid_set.add((parts[0], parts[1]))
        if not valid_set:
            raise ValueError(f"valid_sequences_file={path!r} parsed to empty set — check the file format.")
        return valid_set

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_nvs_groups(self):
        """Group internal Syn4D_Multi samples by (scene, base_seq) for multi-camera NVS."""
        # Group by (scene, base_seq) → cam_name → {start_frame: internal_idx}
        cam_starts_by_group = defaultdict(lambda: defaultdict(dict))

        for internal_idx, sample in enumerate(self._syn4d.samples):
            scene = sample["scene_name"]
            base_seq = Syn4D_Multi._base_sequence_name(sample["camera_seq_name"])
            cam_name = sample["camera_seq_name"]
            start = sample["frame_indices_abs"][0]
            cam_starts_by_group[(scene, base_seq)][cam_name][start] = internal_idx

        # Optional scene filter
        if self.scene_filter:
            cam_starts_by_group = {
                k: v
                for k, v in cam_starts_by_group.items()
                if self.scene_filter in k[0]
            }

        # Exclude eval scenes
        if self.exclude_scenes:
            cam_starts_by_group = {
                k: v
                for k, v in cam_starts_by_group.items()
                if k[0] not in self.exclude_scenes
            }

        # Restrict to valid (scene, base_seq) pairs from valid_sequences_file
        if self._valid_seq_set is not None:
            cam_starts_by_group = {
                k: v
                for k, v in cam_starts_by_group.items()
                if (k[0], k[1]) in self._valid_seq_set
            }

        # Optional sequence_idx filter: keep only N-th base_seq per scene
        if self.sequence_idx is not None:
            seqs_by_scene = defaultdict(list)
            for scene, base_seq in cam_starts_by_group.keys():
                seqs_by_scene[scene].append(base_seq)
            for scene in seqs_by_scene:
                seqs_by_scene[scene] = sorted(set(seqs_by_scene[scene]))

            kept_keys = set()
            for scene, seqs in seqs_by_scene.items():
                idx = int(self.sequence_idx)
                if idx < len(seqs):
                    kept_keys.add((scene, seqs[idx]))
            cam_starts_by_group = {
                k: v for k, v in cam_starts_by_group.items() if k in kept_keys
            }

        # Build groups with >= 2 cameras and common start frames
        nvs_groups = []
        for (scene, base_seq), cam_dict in cam_starts_by_group.items():
            if len(cam_dict) < 2:
                continue

            # Common start frames across ALL cameras
            start_sets = [set(starts.keys()) for starts in cam_dict.values()]
            common = start_sets[0]
            for s in start_sets[1:]:
                common &= s
            if not common:
                continue

            camera_names = sorted(cam_dict.keys())

            # Filter to starts where the query frame (frame 0) has NPZ tracking data.
            # NPZ vertex-to-pixel mapping exists every 5 frames in Syn4D.
            # Check the first camera to discover which starts have NPZ dirs.
            ref_cam = camera_names[0]
            ref_idx = cam_dict[ref_cam]
            # The raw Syn4D loader accepts both the legacy per-frame NPZ layout
            # and the compact tracking_safetensors layout. Keep this wrapper's
            # trackable-start filter consistent: the new format stores frame
            # stems in the camera shard index rather than <scene>/npz/ folders.
            trackable_stems_from_safetensors = set()
            cached_tracking = compact_tracking_if_available(self._syn4d, scene, ref_cam)
            if cached_tracking is not None:
                trackable_stems_from_safetensors = {
                    key.split("/", 1)[0]
                    for key in cached_tracking["index"].get("record_key_to_id", {})
                    if "/" in key
                }
            trackable_starts = set()
            for s in common:
                if s not in ref_idx:
                    continue
                sample = self._syn4d.samples[ref_idx[s]]
                frame0_stem = os.path.splitext(sample["frame_names"][0])[0]
                npz_dir = os.path.join(
                    self.ROOT, sample["scene_name"], "npz",
                    sample["camera_seq_name"], frame0_stem,
                )
                if os.path.isdir(npz_dir) or frame0_stem in trackable_stems_from_safetensors:
                    trackable_starts.add(s)
            if not trackable_starts:
                continue

            # Cache full frame list per camera for stride construction at sample time.
            # frame_names_all[i] is the i-th frame filename; start indices index into this.
            # Format-aware (must match the base Syn4D_Multi enumeration so that the
            # frame_indices_abs / common_starts computed there index correctly here):
            #   old png format -> list the png/ dir;
            #   new mp4-only format -> view_dir is mp4/ (no PNGs), so derive the full
            #   frame list from the camera CSV exactly as the base loader does
            #   (csv 'name' stems match the tracking-safetensor record keys).
            all_frame_names_by_cam = {}
            for cam_name in camera_names:
                any_idx = next(iter(cam_dict[cam_name].values()))
                sample_any = self._syn4d.samples[any_idx]
                view_dir = sample_any["view_dir"]
                png_names = sorted(
                    x for x in os.listdir(view_dir) if x.endswith(".png")
                )
                if png_names:
                    all_frame_names_by_cam[cam_name] = png_names
                else:
                    camera_csv_path = os.path.join(
                        self.ROOT, sample_any["scene_name"], "ground_truth",
                        "meta_exr_csv", f"{sample_any['camera_seq_name']}_camera.csv",
                    )
                    all_frame_names_by_cam[cam_name] = (
                        self._syn4d._list_frame_names_from_camera_csv(camera_csv_path)
                    )

            nvs_groups.append(
                {
                    "scene_name": scene,
                    "base_seq": base_seq,
                    "camera_names": camera_names,
                    "cam_starts": dict(cam_dict),
                    "common_starts": sorted(trackable_starts),
                    "all_frame_names_by_cam": all_frame_names_by_cam,
                }
            )

        return nvs_groups

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        # Support tuple indexing from BatchedRandomSampler: (idx, ar_idx, nview)
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        seed = None if self.seed is None else self.seed + int(idx)
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if seed is None:
                    return self._getitem_impl()
                attempt_seed = seed + attempt * 1000003
                py_state = random.getstate()
                np_state = np.random.get_state()
                try:
                    random.seed(attempt_seed)
                    np.random.seed(attempt_seed % (2**32 - 1))
                    return self._getitem_impl()
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Syn4D_Multi_NVS: failed after {max_retries} retries: {e}"
                    ) from e

    def _pick_temporal_stride(self):
        """Pick temporal stride: 50% stride=1, rest uniform over [2, max_temporal_stride]."""
        if self.max_temporal_stride <= 1:
            return 1
        if random.random() < 0.5:
            return 1
        return random.randint(2, self.max_temporal_stride)

    def _pick_camera_pair(self, group):
        """Pick (src_cam, tgt_cam) names, optionally oversampling paired cameras."""
        cam_names = group["camera_names"]
        if (self._group_camera_pairs
                and self.group_camera_sample_prob > 0
                and random.random() < self.group_camera_sample_prob):
            # Oversample: pick a camera pair by suffix index, map to actual names
            cam_suffix_to_name = {}
            for name in cam_names:
                suffix = int(name.rsplit("_", 1)[-1])
                cam_suffix_to_name[suffix] = name
            # Filter to pairs that exist in this group
            valid_pairs = [
                (a, b) for a, b in self._group_camera_pairs
                if a in cam_suffix_to_name and b in cam_suffix_to_name
            ]
            if valid_pairs:
                a, b = random.choice(valid_pairs)
                pair = [cam_suffix_to_name[a], cam_suffix_to_name[b]]
                random.shuffle(pair)
                return pair[0], pair[1]
        # Default: uniform random 2-from-all
        return tuple(random.sample(cam_names, 2))

    def _get_valid_query_indices(self, sample):
        valid_query_indices = []
        for i, frame_name in enumerate(sample["frame_names"]):
            if self._syn4d._frame_has_tracking_npz(
                sample["scene_name"], sample["camera_seq_name"], frame_name
            ):
                valid_query_indices.append(i)
        return valid_query_indices

    def _load_per_frame_dynamic_masks(self, mask_root, mask_types, frame_names,
                                      scene_name=None, camera_seq_name=None):
        """Per-frame foreground (= dynamic) mask for Syn4D.

        Foreground = NOT (union of all ``_env*`` masks). This is the
        canonical Syn4D foreground indicator (the ``static_background_valid_map``
        in ``syn4d.py:1524-1534``), generalised to per-frame.

        Two on-disk sources, picked per frame:
          - old png format: ``<frame_stem>_env*.png`` under ``mask_root``;
          - new mp4-only format: the per-camera env-mask safetensor shard
            (``env_mask_safetensors/<seq>.safetensors``), one combined env mask
            per frame, decoded via ``Syn4D_Multi._load_env_mask_safetensor``.
            Without this, maskon configs on the new format would silently get
            all-static masks (no _env*.png exist) — i.e. a broken ablation.

        Resize strategy: center-crop to the output aspect ratio, then
        nearest-resize to (output_w, output_h). This approximates the
        Syn4D RGB pipeline (``_crop_resize_if_necessary_with_track``) which
        center-crops around the principal point — for symmetric Syn4D
        intrinsics that's image center, so this matches.

        Returns ``(T, H, W)`` bool, ``True`` = dynamic. Returns all-False when
        neither source has a record (= treat as pure-static).
        """
        H_out, W_out = self.output_height, self.output_width
        out = np.zeros((len(frame_names), H_out, W_out), dtype=bool)
        env_types = [m for m in (mask_types or []) if "env" in m]
        use_safetensor = (
            scene_name is not None and camera_seq_name is not None
            and self._syn4d._get_env_mask_safetensor(scene_name, camera_seq_name) is not None
        )
        if (not mask_root or not env_types) and not use_safetensor:
            return out

        target_aspect = W_out / max(H_out, 1)

        for t, fn in enumerate(frame_names):
            stem = osp.splitext(fn)[0]
            bg = None
            if mask_root:
                for mtype in env_types:
                    mp = osp.join(mask_root, f"{stem}{mtype}.png")
                    if not osp.isfile(mp):
                        continue
                    m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
                    if m is None:
                        continue
                    mb = (m > 0).any(axis=-1) if m.ndim == 3 else (m > 0)
                    bg = mb if bg is None else (bg | mb)
            if bg is None and use_safetensor:
                env = self._syn4d._load_env_mask_safetensor(scene_name, camera_seq_name, stem)
                if env is not None:
                    bg = (env > 0).any(axis=-1) if env.ndim == 3 else (env > 0)
            if bg is None:
                continue
            fg = ~bg
            h, w = fg.shape
            if w / max(h, 1) > target_aspect:
                new_w = int(round(h * target_aspect))
                x0 = (w - new_w) // 2
                cropped = fg[:, x0:x0 + new_w]
            else:
                new_h = int(round(w / target_aspect))
                y0 = (h - new_h) // 2
                cropped = fg[y0:y0 + new_h, :]
            fg_small = cv2.resize(
                cropped.astype(np.uint8), (W_out, H_out),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            out[t] = fg_small
        return out

    def _getitem_impl(self):
        # 1. Pick random group
        group = random.choice(self._nvs_groups)

        # 2. Pick 2 random cameras (source, target), with optional pair oversampling
        src_cam, tgt_cam = self._pick_camera_pair(group)

        # 3. Pick start frame (fixed or random)
        if self.fixed_start_frame is not None:
            if str(self.fixed_start_frame).lower() == "first":
                start = group["common_starts"][0]
            else:
                start = int(self.fixed_start_frame)
                if start not in group["common_starts"]:
                    raise ValueError(
                        f"fixed_start_frame={start} not in common_starts "
                        f"{group['common_starts'][:5]}... for {group['scene_name']}"
                    )
        else:
            start = random.choice(group["common_starts"])

        # 4. Look up internal indices — samples have num_views consecutive frames (stride=1)
        src_idx = group["cam_starts"][src_cam][start]
        tgt_idx = group["cam_starts"][tgt_cam][start]
        src_orig = self._syn4d.samples[src_idx]
        tgt_orig = self._syn4d.samples[tgt_idx]

        # 5. Pick temporal stride, capped by available frames from start.
        #    A clip of N frames at stride s spans 1 + s*(N-1) absolute frames,
        #    so max_stride = (remaining - 1) // (N - 1).
        src_all_names = group["all_frame_names_by_cam"][src_cam]
        remaining = len(src_all_names) - start
        max_feasible_stride = max(1, (remaining - 1) // max(self._num_views - 1, 1))
        effective_max = min(self.max_temporal_stride, max_feasible_stride)
        temporal_stride = self._pick_temporal_stride()
        temporal_stride = min(temporal_stride, effective_max)

        # 6. Construct strided frame list from the FULL frame list.
        #    E.g., stride=2 from start=100 → frames [100, 102, 104, ...] (49 frames)
        strided_names = src_all_names[start::temporal_stride][:self._num_views]
        strided_indices = list(range(start, len(src_all_names), temporal_stride))[:self._num_views]

        # Patch sample with strided frames and load via _get_views.
        src_patched = dict(src_orig)
        src_patched["frame_names"] = strided_names
        src_patched["frame_indices_abs"] = strided_indices
        src_patched = self._maybe_attach_mp4_rgb_to_sample(src_patched)

        valid_query_indices = self._get_valid_query_indices(src_patched)
        if not valid_query_indices:
            raise ValueError(
                f"No NPZ-aligned query frames found for "
                f"{src_patched['scene_name']} / {src_patched['camera_seq_name']} "
                f"start={start} stride={temporal_stride}"
            )
        if np.random.random() < max(0.0, min(1.0, self.multi_query_prob)):
            if self.multi_query_schedule == "canonical_random_subset":
                q_count = int(np.random.choice(self.multi_query_choices))
                track_query_indices = _choose_canonical_random_query_indices(
                    self._num_views,
                    q_count,
                    np.random,
                    available_indices=valid_query_indices,
                )
            else:
                q_count = int(random.choice(self.multi_query_choices))
                track_query_indices = _choose_track_query_indices(
                    self._num_views, q_count, available_indices=valid_query_indices,
                )
        elif self.random_query_frame and len(valid_query_indices) > 1:
            track_query_indices = [int(random.choice(valid_query_indices))]
        else:
            track_query_indices = [int(valid_query_indices[0])]
        track_query_idx = int(track_query_indices[0])

        self._syn4d.samples[src_idx] = src_patched
        try:
            query_view_sets = [
                self._syn4d._get_views(
                    src_idx,
                    self.resolution,
                    np.random.default_rng(np.random.randint(0, 2**32 - 1)),
                    self._num_views,
                    track_query_idx=int(q_idx),
                )
                for q_idx in track_query_indices
            ]
            src_views = query_view_sets[0]
        finally:
            self._syn4d.samples[src_idx] = src_orig

        # 7. Extract source frames (PIL Images)
        src_frames = [v["img"] for v in src_views]

        # 8. Extract tracks (world space) and source RGB appearance at track positions.
        # Keep the primary query chunk separate so common_scale does not drift
        # when training/visualization concatenates extra query-frame grids.
        track_outs = [self._extract_tracks(vs) for vs in query_view_sets]
        track_chunks = [t for t, _ in track_outs]
        valid_chunks = [v for _, v in track_outs]
        # Independent per-frame visibility dropout on extra-query identities so
        # q0 vs q48 disagree structurally, not just via additive geometry noise.
        # No-op when the knob is 0 or only one query is active.
        if self.multi_query_validity_dropout > 0.0 and len(valid_chunks) > 1:
            for qi in range(1, len(valid_chunks)):
                drop = np.random.random(valid_chunks[qi].shape) < self.multi_query_validity_dropout
                valid_chunks[qi] = valid_chunks[qi] & ~drop
                # Keep tracks/valid consistent: zeroed observations must be masked.
                track_chunks[qi] = track_chunks[qi].copy()
                track_chunks[qi][~valid_chunks[qi]] = 0.0
        tracks_world = np.concatenate(track_chunks, axis=0)
        tracks_world_valid_mask = np.concatenate(valid_chunks, axis=0).astype(bool, copy=False)
        tracks_query_idx_per_identity = np.concatenate(
            [
                np.full((chunk.shape[0],), int(q), dtype=np.int32)
                for q, chunk in zip(track_query_indices, track_chunks)
            ],
            axis=0,
        )
        source_rgb_at_tracks = np.concatenate(
            [self._extract_source_rgb_at_tracks(vs) for vs in query_view_sets], axis=0,
        )

        # 9. Load target camera data (RGB + poses + intrinsics, no depth/tracking needed)
        #    Construct strided target frames from full frame list.
        tgt_all_names = group["all_frame_names_by_cam"][tgt_cam]
        tgt_strided_names = tgt_all_names[start::temporal_stride][:self._num_views]
        tgt_strided_indices = list(range(start, len(tgt_all_names), temporal_stride))[:self._num_views]
        tgt_patched = dict(tgt_orig)
        tgt_patched["frame_names"] = tgt_strided_names
        tgt_patched["frame_indices_abs"] = tgt_strided_indices
        tgt_patched = self._maybe_attach_mp4_rgb_to_sample(tgt_patched)
        tgt_frames, tgt_cam2world, tgt_intrinsics = self._load_target_camera(tgt_patched)

        # 9b. (Stage 2) Load target-view dense 3D tracks via mesh tracking
        tgt_track_video = None
        tgt_track_valid_mask = None
        tgt_depth_video = None
        tgt_depth_valid_mask = None
        if self.return_target_tracks or self.return_target_depth:
            (
                tgt_track_video,
                tgt_track_valid_mask,
                tgt_depth_video,
                tgt_depth_valid_mask,
            ) = self._load_target_geometry(
                tgt_idx, tgt_orig, tgt_patched, track_query_idx=track_query_idx,
            )

        # 10. Normalize to frame-0 source camera space + common_scale
        #    (matches ECCV pipeline: Stream3RDatasetAdapter's in_camera1 + common_scale)
        src_cam0_pose = src_views[0]["camera_pose"].astype(np.float32)  # (4,4) cam2world
        _, _, common_scale = self._normalize_to_cam0(
            track_chunks[0], tgt_cam2world, src_cam0_pose,
        )
        tracks_world, tgt_cam2world, common_scale = self._normalize_to_cam0(
            tracks_world, tgt_cam2world, src_cam0_pose, common_scale=common_scale,
        )

        # 10b. (Stage 2) Normalize target tracks to cam0 space + common_scale
        if tgt_track_video is not None:
            tgt_track_video = self._normalize_target_tracks_to_cam0(
                tgt_track_video, tgt_track_valid_mask, src_cam0_pose, common_scale,
            )
        if tgt_depth_video is not None:
            tgt_depth_video = self._normalize_target_depth_to_common_scale(
                tgt_depth_video, tgt_depth_valid_mask, common_scale,
            )

        # 10c. Compute reprojected source depth → target camera (in cam0-normalized space)
        reproj_depth, reproj_valid = self._compute_reprojected_depth_at_target(
            src_views, tgt_cam2world, tgt_intrinsics, src_cam0_pose, common_scale,
        )

        # 11. Find caption — use TARGET camera caption for training
        #     (matches recammaster: train on target description, infer with source)
        prompt, prompt_emb = self._find_caption_with_emb(
            group["scene_name"], tgt_cam, tgt_strided_indices
        )

        src_depth_frame0 = src_views[0]["depthmap"].astype(np.float32) / max(common_scale, 1e-6)

        result = {
            "video": src_frames,
            "target_video": tgt_frames,
            "scene_id": str(group["scene_name"]),
            "source_camera_id": str(src_cam),
            "target_camera_id": str(tgt_cam),
            "frame_ids": [int(frame_index) for frame_index in strided_indices],
            "temporal_stride": int(temporal_stride),
            "tracks_world": tracks_world,
            "target_cam2world": tgt_cam2world,
            "target_intrinsics": tgt_intrinsics,
            "common_scale": common_scale,
            "track_query_idx": track_query_idx,
            "track_query_indices": [int(q) for q in track_query_indices],
            "track_query_count": len(track_query_indices),
            # Always retain row provenance. Single-query and legacy schedules
            # simply emit one repeated anchor; canonical 3D fusion needs this
            # field to classify each identity at its own query frame.
            "tracks_query_idx_per_identity": tracks_query_idx_per_identity,
            # Explicit per-frame validity is emitted only under the q2-realism
            # opt-in; legacy default leaves it absent so downstream nonzero
            # derivation is unchanged (see _extract_tracks / __init__ flags).
            **(
                {"tracks_world_valid_mask": tracks_world_valid_mask}
                if (self.per_frame_track_validity or self.multi_query_validity_dropout > 0.0)
                else {}
            ),
            "prompt": prompt,
            "prompt_emb": prompt_emb,
            "reproj_src_depth_at_target": reproj_depth,
            "reproj_src_depth_valid": reproj_valid,
            # --- unified format keys ---
            "has_tracks": True,
            "has_dynamic_tracks": True,
            "has_depth": tgt_depth_video is not None,
            "target_depth": tgt_depth_video,
            "source_depth": src_depth_frame0,
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "source_rgb_at_tracks": source_rgb_at_tracks,
        }

        # Stage 2: add target-view dense 3D tracks
        if tgt_track_video is not None:
            result["target_track_video"] = tgt_track_video
            result["target_track_valid_mask"] = tgt_track_valid_mask
        if tgt_depth_video is not None:
            result["target_depth_video"] = tgt_depth_video
            result["target_depth_valid_mask"] = tgt_depth_valid_mask

        # Per-frame source fields for the dense GEN3C-style target-space
        # control branch (and legacy source_depth_video).  All cam0-normalized
        # + common_scale-rescaled for consistency with tracks / target cams.
        self._attach_source_dense_fields(result, src_views, src_cam0_pose, common_scale)

        # Per-frame foreground masks for the multi-frame dense warp's
        # static / dynamic separation. Syn4D doesn't carry per-frame
        # segmentation PNGs the way Kubric does, but it has per-frame
        # `_env*` masks for the static background — foreground = NOT
        # (union of env masks). This is the canonical Syn4D foreground
        # indicator (mirrors `static_background_valid_map` in
        # `syn4d.py:1524-1534`, generalised per-frame). Bypasses the
        # threshold-based `tracks_world` motion fallback, which
        # under-classifies Syn4D actor motion (their amplitude is
        # smaller than Kubric balls so even 0.005 leaks).
        if self.return_dynamic_masks:
            result["source_dynamic_mask"] = self._load_per_frame_dynamic_masks(
                src_orig.get("mask_root"), src_orig.get("mask_types"), strided_names,
                scene_name=src_orig.get("scene_name"), camera_seq_name=src_orig.get("camera_seq_name"),
            )
            result["target_dynamic_mask"] = self._load_per_frame_dynamic_masks(
                tgt_orig.get("mask_root"), tgt_orig.get("mask_types"), tgt_strided_names,
                scene_name=tgt_orig.get("scene_name"), camera_seq_name=tgt_orig.get("camera_seq_name"),
            )

        return result

    # ------------------------------------------------------------------
    # Normalization: frame-0 camera space + common_scale
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_to_cam0(tracks_world, tgt_cam2world, src_cam0_pose, common_scale=None):
        """Transform tracks and cameras to frame-0 source camera space, then apply common_scale.

        Matches the ECCV pipeline (Stream3RDatasetAdapter):
        1. in_camera1 = inv(src_cam0_pose)  → world-to-cam0 transform
        2. tracks_cam0 = world_to_cam0 @ tracks_world
        3. tgt_cam2cam0 = world_to_cam0 @ tgt_cam2world
        4. common_scale = mean(||tracks_cam0||_2) over valid points
        5. Divide all positions (tracks + camera translations) by common_scale

        Args:
            tracks_world: (N, T, 3) float32 world-space 3D tracks
            tgt_cam2world: (T, 4, 4) float32 target camera extrinsics (world space)
            src_cam0_pose: (4, 4) float32 source camera frame-0 cam2world

        Returns:
            tracks_cam0: (N, T, 3) normalized tracks in cam0 space
            tgt_cam2cam0: (T, 4, 4) normalized target cameras in cam0 space
            common_scale: float, the scale factor used
        """
        # 1. World → camera-0 space
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)  # (4, 4)
        R = world_to_cam0[:3, :3]  # (3, 3)
        t = world_to_cam0[:3, 3]   # (3,)

        N, T, _ = tracks_world.shape
        # tracks_cam0[n, t] = R @ tracks_world[n, t] + t
        tracks_cam0 = np.einsum("ij,ntj->nti", R, tracks_world) + t[None, None, :]

        # Target camera: tgt_cam2cam0 = world_to_cam0 @ tgt_cam2world
        tgt_cam2cam0 = world_to_cam0[None, :, :] @ tgt_cam2world  # (T, 4, 4)

        # 2. Common scale = mean L2 norm of valid (non-zero) cam0-space track
        # points unless the caller supplies a primary-query scale. Multi-query
        # training concatenates several query grids; the scene scale must stay
        # independent of query count.
        if common_scale is None:
            valid_mask = np.any(tracks_cam0 != 0, axis=-1)  # (N, T)
            valid_pts = tracks_cam0[valid_mask]  # (M, 3)
            if valid_pts.shape[0] > 0:
                common_scale = float(np.linalg.norm(valid_pts, axis=-1).mean())
            else:
                common_scale = 1.0
        else:
            common_scale = float(common_scale)
        common_scale = max(common_scale, 1e-6)  # prevent division by zero

        # 3. Apply common_scale: divide all 3D positions and camera translations
        tracks_cam0 = tracks_cam0 / common_scale
        tgt_cam2cam0[:, :3, 3] /= common_scale  # scale translations only; rotations unchanged

        return tracks_cam0.astype(np.float32), tgt_cam2cam0.astype(np.float32), common_scale

    # ------------------------------------------------------------------
    # Track extraction
    # ------------------------------------------------------------------

    def _compute_reprojected_depth_at_target(
        self, src_views, tgt_cam2world, tgt_intrinsics, src_cam0_pose, common_scale,
    ):
        """Compute per-frame source depth reprojected to the target camera.

        For each source pixel n at frame t:
          1. Look up source depth d_src[t, y_n, x_n]
          2. Unproject to 3D in source camera space
          3. Transform to cam0-normalized space
          4. Project to target camera → get target-camera depth (z)

        Returns:
            reproj_depth: (N, T) float32 — target-camera depth for each source pixel,
                in cam0-normalized space (/ common_scale). 0 where invalid.
            reproj_valid: (N, T) bool — True where the projection is valid (d_src > 0,
                z_tgt > 0, i.e., visible from target).
        """
        T = len(src_views)
        H, W = src_views[0]["depthmap"].shape[:2]
        s = self.track_spatial_stride

        yy = np.arange(0, H, s)
        xx = np.arange(0, W, s)
        grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
        grid_y = grid_y.ravel().astype(np.float32)
        grid_x = grid_x.ravel().astype(np.float32)
        N = len(grid_y)

        # Source and target intrinsics
        src_K = src_views[0]["camera_intrinsics"].astype(np.float32)
        fx_s, fy_s, cx_s, cy_s = src_K[0, 0], src_K[1, 1], src_K[0, 2], src_K[1, 2]

        # World → cam0 transform
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)

        reproj_depth = np.zeros((N, T), dtype=np.float32)
        reproj_valid = np.zeros((N, T), dtype=bool)

        for t in range(T):
            d_src = src_views[t]["depthmap"][grid_y.astype(int), grid_x.astype(int)]  # (N,)
            valid_src = d_src > 1e-6

            if not valid_src.any():
                continue

            # Unproject to 3D in source camera space at frame t
            x_cam = (grid_x - cx_s) * d_src / fx_s  # (N,)
            y_cam = (grid_y - cy_s) * d_src / fy_s
            z_cam = d_src
            pts_src_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

            # Source cam → world: p_world = src_c2w[t] @ p_cam
            src_c2w_t = src_views[t]["camera_pose"].astype(np.float32)
            R_src = src_c2w_t[:3, :3]
            t_src = src_c2w_t[:3, 3]
            pts_world = (R_src @ pts_src_cam.T).T + t_src  # (N, 3)

            # World → cam0 space
            R_w2c0 = world_to_cam0[:3, :3]
            t_w2c0 = world_to_cam0[:3, 3]
            pts_cam0 = (R_w2c0 @ pts_world.T).T + t_w2c0  # (N, 3)

            # Apply common_scale
            pts_cam0 = pts_cam0 / common_scale

            # Project to target camera at frame t (already in cam0-normalized space)
            tgt_w2c_t = np.linalg.inv(tgt_cam2world[t]).astype(np.float32)
            R_tgt = tgt_w2c_t[:3, :3]
            t_tgt = tgt_w2c_t[:3, 3]
            pts_tgt_cam = (R_tgt @ pts_cam0.T).T + t_tgt  # (N, 3)

            z_tgt = pts_tgt_cam[:, 2]
            valid = valid_src & (z_tgt > 1e-3)

            reproj_depth[:, t] = np.where(valid, z_tgt, 0.0)
            reproj_valid[:, t] = valid

        return reproj_depth.astype(np.float32), reproj_valid

    def _extract_tracks(self, src_views):
        """Extract world-space tracks + validity from source view dicts.

        Samples a spatial grid from the per-frame (H, W, 3) track maps.

        Returns ``(tracks, valid)`` where ``tracks`` is ``(N, T, 3)`` and
        ``valid`` is ``(N, T)`` bool.

        - Legacy (``per_frame_track_validity=False``): a track is valid only if
          it is valid in ALL frames; identities failing that are zeroed and the
          returned mask is that all-frames flag broadcast over T. Preserves the
          historical persistent-track distribution.
        - q2-realism (``per_frame_track_validity=True``): keep the real
          per-frame visibility; only the invalid (identity, frame) entries are
          zeroed. This lets multi-query identities carry genuine per-frame
          occlusion, matching real CoW/VGGT q2 eval.
        """
        T = len(src_views)
        H, W = src_views[0]["track"].shape[:2]
        s = self.track_spatial_stride

        yy = np.arange(0, H, s)
        xx = np.arange(0, W, s)
        grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
        grid_y = grid_y.ravel()
        grid_x = grid_x.ravel()
        N = len(grid_y)

        tracks = np.zeros((N, T, 3), dtype=np.float32)
        valid_pf = np.zeros((N, T), dtype=bool)

        for t in range(T):
            track_t = src_views[t]["track"]  # (H, W, 3) float32
            valid_t = src_views[t]["track_valid_mask"]  # (H, W) bool

            tracks[:, t, :] = track_t[grid_y, grid_x, :]
            valid_pf[:, t] = valid_t[grid_y, grid_x]

        if self.per_frame_track_validity:
            tracks[~valid_pf] = 0.0
            return tracks, valid_pf

        # Legacy: valid only if valid in ALL frames.
        valid_all = valid_pf.all(axis=1)
        tracks[~valid_all] = 0.0
        return tracks, np.repeat(valid_all[:, None], T, axis=1)

    def _extract_source_rgb_at_tracks(self, src_views):
        """Sample frame-0 source RGB at the track grid positions.

        Returns (N, T, 3) float32 in [0, 1], broadcast from frame-0 appearance.
        The same spatial grid as _extract_tracks is used so indices align exactly.
        """
        T = len(src_views)
        H, W = src_views[0]["track"].shape[:2]
        s = self.track_spatial_stride

        yy = np.arange(0, H, s)
        xx = np.arange(0, W, s)
        grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
        grid_y = grid_y.ravel()
        grid_x = grid_x.ravel()

        img0 = np.array(src_views[0]["img"], dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]
        rgb0 = img0[grid_y, grid_x, :]                                   # (N, 3)
        return np.broadcast_to(rgb0[:, None, :], (len(grid_y), T, 3)).copy()  # (N, T, 3)

    # ------------------------------------------------------------------
    # Source-side per-frame tensors for the dense GEN3C-style branch
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_source_rgb_video(src_views):
        """Stack per-frame source RGB as ``(T, H, W, 3)`` float32 in ``[0, 1]``.

        Used by the dense target-space control branch, which forward-warps
        source RGB into the target camera frame.
        """
        arr = np.stack(
            [np.array(v["img"], dtype=np.float32) / 255.0 for v in src_views],
            axis=0,
        )  # (T, H, W, 3)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _extract_source_cam2cam0(src_views, src_cam0_pose, common_scale):
        """Return ``(T, 4, 4)`` source cam2world in cam0-normalized space.

        Mirrors the tgt-side logic inside ``_normalize_to_cam0``: applies
        ``world_to_cam0`` and rescales translations by ``common_scale`` so
        the dense branch's reprojection uses the SAME coordinate frame +
        scale as the persistent tracks / target cameras / target depths.

        By construction frame 0 maps to the identity.
        """
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)  # (4, 4)
        src_c2w_world = np.stack(
            [v["camera_pose"].astype(np.float32) for v in src_views], axis=0
        )  # (T, 4, 4)
        src_c2cam0 = world_to_cam0[None, :, :] @ src_c2w_world            # (T, 4, 4)
        src_c2cam0 = src_c2cam0.copy()
        src_c2cam0[:, :3, 3] /= max(float(common_scale), 1e-6)
        return src_c2cam0.astype(np.float32, copy=False)

    @staticmethod
    def _extract_source_intrinsics(src_views):
        """Return the source pinhole intrinsics as a ``(3, 3)`` float32 array.

        The dataset uses one shared intrinsics matrix per source camera
        (unchanged across time), so we just take frame 0.
        """
        return src_views[0]["camera_intrinsics"].astype(np.float32, copy=False)

    def _attach_source_dense_fields(
        self,
        result: dict,
        src_views,
        src_cam0_pose: np.ndarray,
        common_scale: float,
    ) -> None:
        """Populate ``result`` with the per-frame source tensors consumed by
        the dense GEN3C-style target-space control branch, gated by the
        constructor flags ``return_source_rgb_video`` /
        ``return_source_cameras`` (and the existing
        ``return_source_depth_video`` which owns the depth stream).

        All tensors are in the cam0-normalized + common_scale frame so the
        dense branch's forward-warp stays consistent with the tracks and
        target cameras produced elsewhere in this dataset.
        """
        if self.return_source_rgb_video:
            result["source_rgb_video"] = self._extract_source_rgb_video(src_views)
        if self.return_source_cameras:
            result["source_cam2world"] = self._extract_source_cam2cam0(
                src_views, src_cam0_pose, common_scale,
            )
            result["source_intrinsics"] = self._extract_source_intrinsics(src_views)
        if self.return_source_depth_video and "source_depth_video" not in result:
            result["source_depth_video"] = np.stack(
                [v["depthmap"].astype(np.float32) for v in src_views], axis=0
            ) / max(float(common_scale), 1e-6)

    # ------------------------------------------------------------------
    # Optional Syn4D MP4 RGB loading
    # ------------------------------------------------------------------

    def _syn4d_mp4_path(self, sample):
        return os.path.join(
            self.ROOT,
            str(sample["scene_name"]),
            "mp4",
            f"{sample['camera_seq_name']}.mp4",
        )

    @staticmethod
    def _normalize_raw_rgb_frame(frame):
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA frame, got shape={frame.shape}")
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return np.ascontiguousarray(frame.astype(np.uint8, copy=False))

    def _read_rgb_frames_from_mp4_decord(self, mp4_path, selected_ids):
        from decord import VideoReader, cpu  # type: ignore

        reader = VideoReader(str(mp4_path), ctx=cpu(0))
        max_frame_id = max(selected_ids)
        if max_frame_id >= len(reader):
            raise FileNotFoundError(
                f"Frame id {max_frame_id} exceeds Syn4D mp4 length {len(reader)}: {mp4_path}"
            )
        return reader.get_batch(selected_ids).asnumpy()

    def _read_rgb_frames_from_mp4_imageio(self, mp4_path, selected_ids):
        wanted = set(int(frame_id) for frame_id in selected_ids)
        max_frame_id = max(wanted)
        frames_by_id = {}
        try:
            for frame_id, frame in enumerate(iio.imiter(mp4_path)):
                if frame_id > max_frame_id:
                    break
                if frame_id in wanted:
                    frames_by_id[frame_id] = frame
        except Exception as exc:
            raise RuntimeError(f"Failed to read Syn4D mp4 video: {mp4_path}") from exc

        missing = [frame_id for frame_id in selected_ids if frame_id not in frames_by_id]
        if missing:
            raise FileNotFoundError(f"Missing frame ids {missing} in Syn4D mp4 video: {mp4_path}")
        return np.stack([np.asarray(frames_by_id[frame_id]) for frame_id in selected_ids], axis=0)

    def _load_raw_rgb_frames_for_sample(self, sample):
        if not self.use_mp4_rgb:
            return None
        mp4_path = self._syn4d_mp4_path(sample)
        if not os.path.isfile(mp4_path):
            return None

        frame_names = list(sample["frame_names"])
        selected_ids = [int(frame_id) for frame_id in sample["frame_indices_abs"][: len(frame_names)]]
        try:
            frame_array = self._read_rgb_frames_from_mp4_decord(mp4_path, selected_ids)
        except Exception:
            try:
                frame_array = self._read_rgb_frames_from_mp4_imageio(mp4_path, selected_ids)
            except Exception as exc:
                if not self._mp4_rgb_fallback_warned:
                    print(
                        "[Syn4D_Multi_NVS] WARNING: failed to read Syn4D MP4 RGB; "
                        f"falling back to PNG frames. First failure: {mp4_path}: {exc}"
                    )
                    self._mp4_rgb_fallback_warned = True
                return None

        if len(frame_array) != len(frame_names):
            raise ValueError(
                f"Syn4D mp4 decoded {len(frame_array)} frames for {len(frame_names)} requested: {mp4_path}"
            )
        return {
            frame_name: self._normalize_raw_rgb_frame(frame)
            for frame_name, frame in zip(frame_names, frame_array)
        }

    def _maybe_attach_mp4_rgb_to_sample(self, sample):
        raw_frames = self._load_raw_rgb_frames_for_sample(sample)
        if raw_frames is None:
            return sample
        sample = dict(sample)
        sample["_raw_rgb_frames_by_name"] = raw_frames
        return sample

    # ------------------------------------------------------------------
    # Target camera loading
    # ------------------------------------------------------------------

    def _load_target_camera(self, tgt_sample):
        """Load target camera RGB frames, cam2world poses, and intrinsics.

        Only loads RGB + camera CSV — no depth or mesh tracking (fast).
        """
        tgt_frames = []
        tgt_cam2world_list = []
        tgt_intrinsics = None

        camera_rows = self._syn4d._load_camera_rows(tgt_sample["camera_csv_path"])
        raw_rgb_frames = tgt_sample.get("_raw_rgb_frames_by_name") or {}

        for frame_name in tgt_sample["frame_names"]:
            # --- RGB (OpenCV for speed, ~30% faster than PIL on network FS) ---
            if frame_name in raw_rgb_frames:
                rgb = self._normalize_raw_rgb_frame(raw_rgb_frames[frame_name])
            else:
                impath = os.path.join(tgt_sample["view_dir"], frame_name)
                bgr = cv2.imread(impath, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Failed to load image: {impath}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
            img = PIL.Image.fromarray(rgb)
            tgt_frames.append(img)

            # --- Camera pose from CSV ---
            row = camera_rows.get(frame_name)
            if row is None:
                raise ValueError(f"Camera row not found for {frame_name}")

            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = self._syn4d.euler_to_rotation_matrix(
                float(row["yaw"]), float(row["pitch"]), float(row["roll"])
            ).astype(np.float32)
            camera_pose[:3, 3] = (
                np.array(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype=np.float32,
                )
                / 100.0
            )
            tgt_cam2world_list.append(camera_pose)

            # --- Intrinsics at output resolution (same for all frames) ---
            if tgt_intrinsics is None:
                focal_length = float(row["focal_length"])
                sensor_width = float(row["sensor_width"])
                sensor_height = float(row["sensor_height"])
                fx = (focal_length / sensor_width) * self.output_width
                fy = (focal_length / sensor_height) * self.output_height
                cx = self.output_width / 2.0
                cy = self.output_height / 2.0
                tgt_intrinsics = np.array(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
                )

        tgt_cam2world = np.stack(tgt_cam2world_list, axis=0)  # (T, 4, 4)
        return tgt_frames, tgt_cam2world, tgt_intrinsics

    # ------------------------------------------------------------------
    # Target-view dense 3D track loading (Stage 2)
    # ------------------------------------------------------------------

    def _load_target_geometry(self, tgt_idx, tgt_orig, tgt_patched, track_query_idx=0):
        """Load target-view dense 3D positions and depth maps.

        Uses the same _get_views() infrastructure as source tracks — Syn4D_Multi
        loads depth + mesh NPZ → world-space (H, W, 3) per frame.

        Args:
            tgt_idx: Internal Syn4D_Multi sample index for target camera.
            tgt_orig: Original (unpatched) sample dict.
            tgt_patched: Patched sample dict with strided frame names/indices.

        Returns:
            tgt_track_video: (T, H, W, 3) float32 world-space 3D positions.
            tgt_track_valid_mask: (T, H, W) bool per-pixel validity.
            tgt_depth_video: (T, H, W) float32 raw target-camera depth.
            tgt_depth_valid_mask: (T, H, W) bool valid depth mask.
        """
        # Temporarily patch the sample to use strided frames, then load via _get_views
        self._syn4d.samples[tgt_idx] = tgt_patched
        try:
            rng = np.random.default_rng()
            tgt_views = self._syn4d._get_views(
                tgt_idx, self.resolution, rng, self._num_views,
                track_query_idx=track_query_idx,
            )
        finally:
            self._syn4d.samples[tgt_idx] = tgt_orig

        T = len(tgt_views)
        H, W = tgt_views[0]["track"].shape[:2]
        tgt_track_video = np.zeros((T, H, W, 3), dtype=np.float32)
        tgt_track_valid_mask = np.zeros((T, H, W), dtype=bool)
        tgt_depth_video = np.zeros((T, H, W), dtype=np.float32)
        tgt_depth_valid_mask = np.zeros((T, H, W), dtype=bool)

        for t in range(T):
            tgt_track_video[t] = tgt_views[t]["track"]          # (H, W, 3)
            tgt_track_valid_mask[t] = tgt_views[t]["track_valid_mask"]  # (H, W)
            tgt_depth_video[t] = tgt_views[t]["depthmap"]
            tgt_depth_valid_mask[t] = np.isfinite(tgt_views[t]["depthmap"]) & (tgt_views[t]["depthmap"] > 0)

        return tgt_track_video, tgt_track_valid_mask, tgt_depth_video, tgt_depth_valid_mask

    @staticmethod
    def _normalize_target_tracks_to_cam0(tgt_track_video, tgt_track_valid_mask,
                                          src_cam0_pose, common_scale):
        """Transform target-view 3D tracks to cam0 space and apply common_scale.

        Uses the SAME normalization as source tracks (world_to_cam0 + common_scale)
        to ensure consistency. common_scale is computed from SOURCE tracks only
        (already computed in _normalize_to_cam0).

        Args:
            tgt_track_video: (T, H, W, 3) float32 world-space positions.
            tgt_track_valid_mask: (T, H, W) bool validity.
            src_cam0_pose: (4, 4) source camera frame-0 cam2world.
            common_scale: float, scale from source track normalization.

        Returns:
            tgt_track_cam0: (T, H, W, 3) float32 cam0-normalized positions.
        """
        world_to_cam0 = np.linalg.inv(src_cam0_pose).astype(np.float32)
        R = world_to_cam0[:3, :3]  # (3, 3)
        t = world_to_cam0[:3, 3]   # (3,)

        T, H, W, _ = tgt_track_video.shape
        # Reshape to (T*H*W, 3) for batch transform
        pts = tgt_track_video.reshape(-1, 3)
        pts_cam0 = (R @ pts.T).T + t[None, :]  # (T*H*W, 3)
        pts_cam0 = pts_cam0.reshape(T, H, W, 3)

        # Apply common_scale (same as source)
        pts_cam0 = pts_cam0 / common_scale

        # Zero out invalid pixels
        pts_cam0[~tgt_track_valid_mask] = 0.0

        return pts_cam0.astype(np.float32)

    @staticmethod
    def _normalize_target_depth_to_common_scale(tgt_depth_video, tgt_depth_valid_mask, common_scale):
        """Scale target-camera depth by the same source-derived common_scale."""
        depth_scaled = tgt_depth_video.astype(np.float32) / float(max(common_scale, 1e-6))
        depth_scaled[~tgt_depth_valid_mask] = 0.0
        return depth_scaled.astype(np.float32)

    # ------------------------------------------------------------------
    # Caption loading
    # ------------------------------------------------------------------

    def _find_caption(self, scene_name, cam_seq_name, frame_indices_abs):
        """Legacy text-only caption lookup. Prefer _find_caption_with_emb for
        new callsites that also want the precomputed T5 embedding when it's
        available."""
        text, _ = self._find_caption_with_emb(scene_name, cam_seq_name, frame_indices_abs)
        return text

    def _get_caption_emb_handle(self, scene_name):
        """Return a ``safe_open`` handle for the per-scene T5 embedding cache,
        or ``None`` when the safetensors file isn't present.

        Cache file path mirrors the source caption JSON layout:
        ``{caption_dir}/{scene_name}_per_video_chunks_t5_emb.safetensors``.
        Per-scene handles are kept open for the lifetime of the dataset to
        avoid per-call ``open()`` syscalls on beegfs/Lustre.
        """
        cached = self._caption_emb_handle_cache.get(scene_name)
        if cached is False:
            return None
        if cached is not None:
            return cached
        emb_path = os.path.join(
            self.caption_dir, f"{scene_name}_per_video_chunks_t5_emb.safetensors"
        )
        if not os.path.isfile(emb_path):
            self._caption_emb_handle_cache[scene_name] = False
            return None
        try:
            from safetensors import safe_open  # local import: keep optional
            handle = safe_open(emb_path, framework="pt", device="cpu")
        except Exception:
            self._caption_emb_handle_cache[scene_name] = False
            return None
        self._caption_emb_handle_cache[scene_name] = handle
        return handle

    def _find_caption_with_emb(self, scene_name, cam_seq_name, frame_indices_abs):
        """Find the best-matching caption for the selected frame window AND
        return the precomputed T5 embedding for that exact chunk if cached.

        Returns ``(text, emb_or_none)``. ``emb`` is ``None`` when:
          - the per-scene safetensors cache is absent,
          - no chunk overlapped the frame window (default-prompt fallback),
          - or the matched chunk text was sanitized to the default prompt
            (precompute filters those out before encoding).

        Loads ``{caption_dir}/{scene_name}_per_video_chunks.json`` for the text
        path; the embedding is keyed identically inside the safetensors file
        as ``mp4/{cam_seq_name}.mp4/{chunk_tag}``.
        """
        json_path = os.path.join(
            self.caption_dir, f"{scene_name}_per_video_chunks.json"
        )

        if json_path not in self._caption_json_cache:
            if os.path.isfile(json_path):
                with open(json_path, "r") as f:
                    self._caption_json_cache[json_path] = json.load(f)
            else:
                self._caption_json_cache[json_path] = {}

        caption_data = self._caption_json_cache[json_path]
        if not caption_data:
            from .mixed_nvs_captions import report_cache_miss as _report
            _report(
                "Syn4D_Multi_NVS",
                f"scene={scene_name!r} cam={cam_seq_name!r} "
                f"(captions JSON missing/empty: {json_path})",
            )
            return self.default_prompt, None

        # Find entry for this camera sequence
        video_key = f"mp4/{cam_seq_name}.mp4"
        chunks = caption_data.get(video_key, {})
        if not chunks:
            from .mixed_nvs_captions import report_cache_miss as _report
            _report(
                "Syn4D_Multi_NVS",
                f"scene={scene_name!r} cam={cam_seq_name!r} "
                f"(no chunks for video_key={video_key!r})",
            )
            return self.default_prompt, None

        # Find chunk with most overlap
        frame_set = set(int(f) for f in frame_indices_abs)
        best_caption = self.default_prompt
        best_chunk_tag = None
        best_overlap = 0

        for chunk_tag, raw_caption in chunks.items():
            try:
                parts = chunk_tag.split("-")
                chunk_start, chunk_end = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue

            chunk_frames = set(range(chunk_start, chunk_end + 1))
            overlap = len(frame_set & chunk_frames)
            if overlap > best_overlap:
                best_overlap = overlap
                best_caption = _sanitize_caption(raw_caption)
                best_chunk_tag = chunk_tag

        text = best_caption if best_caption else self.default_prompt
        if best_chunk_tag is None:
            return text, None

        emb_handle = self._get_caption_emb_handle(scene_name)
        if emb_handle is None:
            return text, None
        emb_key = f"{video_key}/{best_chunk_tag}"
        try:
            emb = emb_handle.get_tensor(emb_key)
        except Exception:
            # Key may be missing if precompute filtered out this caption
            # (e.g. it sanitized to the default prompt) — fall back to None.
            emb = None
        return text, emb

    # ------------------------------------------------------------------
    # Public API for deterministic eval access
    # ------------------------------------------------------------------

    @property
    def num_groups(self):
        """Number of multi-camera NVS groups."""
        return len(self._nvs_groups)

    def get_group_info(self, group_idx=0):
        """Return public metadata for one NVS group.

        Returns:
            dict with keys:
              - camera_names: list[str] — sorted camera sequence names
              - common_starts: list[int] — trackable start frames
              - scene_name: str
        """
        g = self._nvs_groups[group_idx]
        return {
            "camera_names": list(g["camera_names"]),
            "common_starts": list(g["common_starts"]),
            "scene_name": g["scene_name"],
        }

    def load_pair(self, src_cam_idx, tgt_cam_idx, start_frame,
                  group_idx=0, track_query_idx=0, window_start_frame=None):
        """Load a deterministic (source, target) pair.

        Same data pipeline as ``__getitem__`` but with explicit camera/frame
        selection instead of random sampling. Returns the same dict format.

        Args:
            src_cam_idx: index into group's camera_names list.
            tgt_cam_idx: index into group's camera_names list.
            start_frame: absolute tracking-anchor frame (must be in
                ``common_starts``). In the usual case this is also the first
                frame of the returned window.
            group_idx: which NVS group (default 0).
            track_query_idx: local frame in the returned window containing the
                GT tracking anchor.
            window_start_frame: optional absolute first frame of the returned
                window. This permits a window whose first frame is not itself
                trackable, provided ``start_frame == window_start_frame +
                track_query_idx`` is a valid tracking anchor. It is required
                for an exactly contiguous I2V rollout when Syn4D stores GT
                tracking only every five frames.

        Returns:
            dict with keys: video, target_video, tracks_world,
            target_cam2world, target_intrinsics, common_scale, prompt,
            src_cam_name, tgt_cam_name.
        """
        group = self._nvs_groups[group_idx]
        start_frame = int(start_frame)
        track_query_idx = int(track_query_idx)
        window_start_frame = (
            start_frame if window_start_frame is None else int(window_start_frame)
        )
        if not 0 <= track_query_idx < self._num_views:
            raise ValueError(
                f"track_query_idx={track_query_idx} outside [0, {self._num_views})"
            )
        if start_frame != window_start_frame + track_query_idx:
            raise ValueError(
                "Syn4D deterministic window requires start_frame to equal "
                "window_start_frame + track_query_idx; got "
                f"{start_frame} != {window_start_frame} + {track_query_idx}"
            )
        src_cam = group["camera_names"][src_cam_idx]
        tgt_cam = group["camera_names"][tgt_cam_idx]
        if start_frame not in group["cam_starts"][src_cam] or start_frame not in group["cam_starts"][tgt_cam]:
            raise ValueError(
                f"tracking anchor {start_frame} is unavailable for {src_cam}/{tgt_cam}"
            )

        src_all_names = group["all_frame_names_by_cam"][src_cam]
        tgt_all_names = group["all_frame_names_by_cam"][tgt_cam]
        window_end = window_start_frame + self._num_views
        if window_start_frame < 0 or window_end > len(src_all_names) or window_end > len(tgt_all_names):
            raise ValueError(
                f"window [{window_start_frame}, {window_end}) is outside available "
                f"source/target frame ranges ({len(src_all_names)}, {len(tgt_all_names)})"
            )

        src_idx = group["cam_starts"][src_cam][start_frame]
        tgt_idx = group["cam_starts"][tgt_cam][start_frame]
        src_orig = self._syn4d.samples[src_idx]
        tgt_orig = self._syn4d.samples[tgt_idx]

        # Load source views (mesh-tracked 3D correspondences)
        src_patched = dict(src_orig)
        src_patched["frame_names"] = src_all_names[window_start_frame:window_end]
        src_patched["frame_indices_abs"] = list(range(window_start_frame, window_end))
        src_for_views = self._maybe_attach_mp4_rgb_to_sample(src_patched)
        self._syn4d.samples[src_idx] = src_for_views
        try:
            rng = np.random.default_rng(42)
            src_views = self._syn4d._get_views(
                src_idx, self.resolution, rng, self._num_views,
                track_query_idx=track_query_idx,
            )
        finally:
            self._syn4d.samples[src_idx] = src_orig

        src_frames = [v["img"] for v in src_views]
        tracks_world, tracks_world_valid_mask = self._extract_tracks(src_views)
        source_rgb_at_tracks = self._extract_source_rgb_at_tracks(src_views)
        # NOTE: `_extract_source_depths` used to be called here but the method
        # does not exist and the result was never consumed. The persistent-
        # branch depth signal the pipeline actually uses is
        # `reproj_src_depth_at_target` from `_compute_reprojected_depth_at_target`
        # below (identical to `__getitem__`'s path).

        # Load target camera
        tgt_patched = dict(tgt_orig)
        tgt_patched["frame_names"] = tgt_all_names[window_start_frame:window_end]
        tgt_patched["frame_indices_abs"] = list(range(window_start_frame, window_end))
        tgt_for_rgb = self._maybe_attach_mp4_rgb_to_sample(tgt_patched)
        tgt_frames, tgt_cam2world, tgt_intrinsics = self._load_target_camera(tgt_for_rgb)

        # (Stage 2) Load target-view dense 3D tracks
        tgt_track_video = None
        tgt_track_valid_mask = None
        tgt_depth_video = None
        tgt_depth_valid_mask = None
        if self.return_target_tracks or self.return_target_depth:
            (
                tgt_track_video,
                tgt_track_valid_mask,
                tgt_depth_video,
                tgt_depth_valid_mask,
            ) = self._load_target_geometry(
                tgt_idx, tgt_orig, tgt_for_rgb,
                track_query_idx=track_query_idx,
            )

        # Normalize to frame-0 source camera space + common_scale
        src_cam0_pose = src_views[0]["camera_pose"].astype(np.float32)
        tracks_world, tgt_cam2world, common_scale = self._normalize_to_cam0(
            tracks_world, tgt_cam2world, src_cam0_pose,
        )
        reproj_depth, reproj_valid = self._compute_reprojected_depth_at_target(
            src_views, tgt_cam2world, tgt_intrinsics, src_cam0_pose, common_scale,
        )

        # (Stage 2) Normalize target tracks
        if tgt_track_video is not None:
            tgt_track_video = self._normalize_target_tracks_to_cam0(
                tgt_track_video, tgt_track_valid_mask, src_cam0_pose, common_scale,
            )
        if tgt_depth_video is not None:
            tgt_depth_video = self._normalize_target_depth_to_common_scale(
                tgt_depth_video, tgt_depth_valid_mask, common_scale,
            )

        # Captions — source (default for inference) and target (for training)
        src_frame_indices = src_patched["frame_indices_abs"]
        tgt_frame_indices = tgt_patched["frame_indices_abs"]
        source_prompt, source_prompt_emb = self._find_caption_with_emb(
            group["scene_name"], src_cam, src_frame_indices,
        )
        target_prompt, target_prompt_emb = self._find_caption_with_emb(
            group["scene_name"], tgt_cam, tgt_frame_indices,
        )

        src_depth_frame0 = src_views[0]["depthmap"].astype(np.float32) / max(common_scale, 1e-6)

        result = {
            "video": src_frames,
            "target_video": tgt_frames,
            "tracks_world": tracks_world,
            "target_cam2world": tgt_cam2world,
            "target_intrinsics": tgt_intrinsics,
            "common_scale": common_scale,
            "track_query_idx": track_query_idx,
            "prompt": source_prompt,
            "prompt_emb": source_prompt_emb,
            "target_prompt": target_prompt,
            "target_prompt_emb": target_prompt_emb,
            "src_cam_name": src_cam,
            "tgt_cam_name": tgt_cam,
            "reproj_src_depth_at_target": reproj_depth,
            "reproj_src_depth_valid": reproj_valid,
            # --- unified format keys ---
            "has_tracks": True,
            "has_dynamic_tracks": True,
            "has_depth": tgt_depth_video is not None,
            "target_depth": tgt_depth_video,
            "source_depth": src_depth_frame0,
            "track_video_loss_mask": 1.0,
            "drop_source_cond": False,
            "source_rgb_at_tracks": source_rgb_at_tracks,
            # q2-realism opt-in only (see __getitem__); legacy default omits it.
            **(
                {"tracks_world_valid_mask": tracks_world_valid_mask}
                if self.per_frame_track_validity
                else {}
            ),
        }
        if tgt_track_video is not None:
            result["target_track_video"] = tgt_track_video
            result["target_track_valid_mask"] = tgt_track_valid_mask
        if tgt_depth_video is not None:
            result["target_depth_video"] = tgt_depth_video
            result["target_depth_valid_mask"] = tgt_depth_valid_mask

        # Per-frame source fields for the dense GEN3C-style target-space
        # control branch (and legacy source_depth_video).  All cam0-normalized
        # + common_scale-rescaled for consistency with tracks / target cams.
        self._attach_source_dense_fields(result, src_views, src_cam0_pose, common_scale)

        # See Syn4D_Multi_NVS._getitem_impl: env-derived per-frame foreground
        # masks override the threshold-based dynamic detection.
        if self.return_dynamic_masks:
            result["source_dynamic_mask"] = self._load_per_frame_dynamic_masks(
                src_patched.get("mask_root"),
                src_patched.get("mask_types"),
                list(src_patched.get("frame_names", [])),
                scene_name=src_patched.get("scene_name"),
                camera_seq_name=src_patched.get("camera_seq_name"),
            )
            result["target_dynamic_mask"] = self._load_per_frame_dynamic_masks(
                tgt_patched.get("mask_root"),
                tgt_patched.get("mask_types"),
                list(tgt_patched.get("frame_names", [])),
                scene_name=tgt_patched.get("scene_name"),
                camera_seq_name=tgt_patched.get("camera_seq_name"),
            )

        return result

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_stats(self):
        return (
            f"{len(self._nvs_groups)} multi-camera groups, "
            f"{self.steps_per_epoch} steps/epoch"
        )

    def __repr__(self):
        return (
            f"Syn4D_Multi_NVS({self.get_stats()}, "
            f"resolution={self.output_width}x{self.output_height}, "
            f"num_views={self._num_views}, "
            f"track_stride={self.track_spatial_stride})"
        )


class Syn4D_Multi_NVS_FixedPair(Syn4D_Multi_NVS):
    """Ordered-pair wrapper around ``Syn4D_Multi_NVS``.

    Reuses ``load_pair(...)`` so the underlying sample construction stays
    identical to the normal NVS path, but fixes the source/target camera
    ordering instead of sampling from all P(8,2) camera pairs.
    """

    def __init__(
        self,
        *args,
        src_cam_idx=0,
        tgt_cam_idx=1,
        group_idx=0,
        fixed_track_query_idx=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.src_cam_idx = int(src_cam_idx)
        self.tgt_cam_idx = int(tgt_cam_idx)
        self.group_idx = int(group_idx)
        self.fixed_track_query_idx = int(fixed_track_query_idx)

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                group = self._nvs_groups[self.group_idx]
                if self.fixed_start_frame is not None:
                    if str(self.fixed_start_frame).lower() == "first":
                        start = group["common_starts"][0]
                    else:
                        start = int(self.fixed_start_frame)
                else:
                    start = random.choice(group["common_starts"])
                return self.load_pair(
                    src_cam_idx=self.src_cam_idx,
                    tgt_cam_idx=self.tgt_cam_idx,
                    start_frame=start,
                    group_idx=self.group_idx,
                    track_query_idx=self.fixed_track_query_idx,
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Syn4D_Multi_NVS_FixedPair: failed after {max_retries} retries: {e}"
                    ) from e


def _sanitize_caption(raw_text):
    """Extract 'Prediction:' section from raw Syn4D caption text.

    Matches recammaster's ``caption_utils.sanitize_caption_text``.
    """
    if not raw_text or not raw_text.strip():
        return ""

    lines = raw_text.strip().split("\n")
    pred_idx = None
    for i, line in enumerate(lines):
        if "prediction" in line.lower() and ":" in line:
            pred_idx = i
            break

    if pred_idx is not None:
        text = "\n".join(lines[pred_idx + 1 :]).strip()
        if text:
            return text

    # Fallback: return raw text, stripping common headers
    text = raw_text.strip()
    for prefix in ("Prompt:", "Caption:", "Description:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


__all__ = ["Syn4D_Multi_NVS", "Syn4D_Multi_NVS_FixedPair"]
