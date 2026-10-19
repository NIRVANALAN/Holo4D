import os.path as osp
from pdb import set_trace as st
import json
import itertools
from collections import deque
import sys
from pathlib import Path
from typing import Dict, List, Optional
import random

import cv2
import numpy as np
import time
from tqdm import tqdm

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from holo4d_runtime.data.stream3r.dust3r.utils.image import imread_cv2


class Objaverse_Multi(BaseMultiViewDataset):
    """
    Objaverse dataloader adapted from RandomViewVideoDataset format.

    This dataset:
    - Loads source views (orthogonal views) from source_root folder
    - Loads target views (random views) from random_views_root folder
    - By default returns target views (random views)
    - With return_source_views=True, returns source views + additional target views
    - Supports multiple source and target views per instance

    Data structure:
    - source_root/category/instance_id/view_0/frame{idx}.png (source views)
    - source_root/category/instance_id/view_0/view.json (source camera)
    - random_views_root/category/instance_id/random_{N}_views/view{idx}.png (target views)
    - random_views_root/category/instance_id/random_{N}_views/view{idx}.json (target camera)
    - random_views_root/category/instance_id/random_{N}_views/timestep_info.json
    """

    def __init__(self,
                 ROOT=None,  # For compatibility with dataset instantiation
                 mask_bg=True,
                 overfitting=False,
                 source_root="data/objaverse_rendered",
                 random_views_root="data/objaverse_random_128_views",
                 return_source_views=False,  # NEW: option to return source views
                 source_view_indices=None,  # Optional: specify which source views to use
                 cache_dir=None,  # Optional: override location for cached scene metadata
                 *args, **kwargs):
        """
        Args:
            ROOT: Root directory (for compatibility, will use random_views_root if not provided)
            mask_bg: Whether to mask background (True, False, or "rand")
            overfitting: Whether to overfit on a single object
            source_root: Path to source views folder (orthogonal views)
            random_views_root: Path to random target views folder
            return_source_views: If True, return source views + target views
            source_view_indices: Specific source view indices to use (e.g., [0, 1, 2, 3, 4, 5])
            cache_dir: Directory for cached scene metadata (defaults to shared target_view_cache)
        """
        # Set paths - use ROOT if provided, otherwise use default random_views_root
        if ROOT is None:
            ROOT = random_views_root

        self.source_root = Path(source_root)
        self.random_views_root = Path(ROOT)
        self.cache_dir = self._resolve_cache_dir(cache_dir)

        # Initialize parent class without ROOT parameter
        super().__init__(*args, **kwargs)
        self.mask_bg = mask_bg
        assert self.mask_bg is True, 'objaverse bg has depth=1e10'
        # self.bg_depth_value = 2.75 # set to a emperical number
        self.bg_depth_value = 0 # ! for calculating valid mask.
        self.is_metric = True  # Objaverse has metric scale
        self.dataset_label = "Objaverse-11K"

        self.return_source_views = return_source_views
        self.source_view_indices = source_view_indices or list(range(6))

        # Auto-detect number of views from folder name
        self.num_rendered_views = self._detect_num_views_from_path()

        # Load all scenes from random_views_root directory
        self.scenes = self._load_scenes(overfitting)
        self.scene_list = list(self.scenes.keys())

        # Build reference image list
        self.all_ref_imgs = list(self.scenes.keys())
        self.invalidate = {scene: {} for scene in self.scene_list}
        self.invalid_scenes = {scene: False for scene in self.scene_list}

        print(f"🎯 Objaverse_Multi initialized:")
        print(f"   📁 Source data: {self.source_root}")
        print(f"   🎲 Random views: {self.random_views_root}")
        print(f"   📊 Detected {self.num_rendered_views} rendered views per instance")
        print(f"   🔄 Return source views: {self.return_source_views}")
        print(f"   📊 Total views per sample: {self.num_views}")
        if self.return_source_views:
            print(f"   📊 Source view indices: {self.source_view_indices}")
            print(f"   📊 Composition: {len(self.source_view_indices)} source + {self.num_views - len(self.source_view_indices)} target views")
        print(f"   📊 Total scenes: {len(self.scenes)}")

        self.is_synthetic = True

    def _detect_num_views_from_path(self) -> int:
        """Auto-detect number of views from the random_views_root path name."""
        import re
        path_str = str(self.random_views_root)
        match = re.search(r'random[_-](\d+)[_-]views', path_str)

        if match:
            num_views = int(match.group(1))
            print(f"🔍 Detected {num_views} views from path: {path_str}")
        else:
            num_views = 128
            print(f"⚠️ Could not detect view count from path, defaulting to {num_views}")

        return num_views

    def _resolve_cache_dir(self, cache_dir: Optional[str]) -> Path:
        """Determine where cached metadata should live."""
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            return cache_path

        repo_root = Path(__file__).resolve().parents[3]
        repo_cache_dir = repo_root / "train_list" / "target_view_cache"
        if repo_cache_dir.parent.exists():
            repo_cache_dir.mkdir(parents=True, exist_ok=True)
            return repo_cache_dir

        fallback = self.random_views_root.parent / "target_view_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _load_scenes(self, overfitting=False):
        """
        Load all scenes from random_views_root directory with JSON caching.

        This discovers available instances by checking random_views_root,
        not the source_root, because we need instances with random views.

        Uses a cache file to speed up repeated loading.

        Split handling:
        - split='train': All instances except the final 50
        - split='val' or 'test': Final 50 instances for evaluation
        """
        # Auto-detect folder name pattern based on number of views
        folder_pattern = f"random_{self.num_rendered_views}_views"

        # Try to load from cache first
        cache_filename = f"objaverse_cut3r_scenes_{self.split}_{folder_pattern}.json"
        cache_path = self.cache_dir / cache_filename
        scenes = None

        if cache_path.exists():
            print(f"📦 Loading scenes from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                scenes = json.load(f)
            print(f"✅ Loaded {len(scenes)} scenes from cache")
        else:
            shared_cache, shared_cache_path = self._load_shared_target_cache()
            if shared_cache:
                scenes = self._build_scenes_from_target_cache(shared_cache, folder_pattern)
                if scenes:
                    print(f"📦 Loaded {len(scenes)} scenes from shared cache: {shared_cache_path}")
                    self._save_scene_cache(cache_path, scenes)
                else:
                    print(f"⚠️ Shared cache {shared_cache_path} had no scenes matching {folder_pattern}")

        if scenes is None:
            print(f"🔍 Building scene index (this may take a while)...")
            scenes = self._discover_scenes(folder_pattern)
            self._save_scene_cache(cache_path, scenes)

        # Apply train/val split BEFORE overfitting mode
        # Reserve final 50 instances for validation
        if scenes and not overfitting:
            scenes = self._apply_train_val_split(scenes)

        if overfitting and scenes:
            # Take first scene for overfitting
            first_key = list(scenes.keys())[0]
            scenes = {first_key: scenes[first_key]}
            print(f"🔧 Overfitting mode: using single scene {first_key}")

        if not scenes:
            raise RuntimeError(
                f"No valid scenes found!\n"
                f"Checked: {self.random_views_root}\n"
                f"Pattern: {folder_pattern}\n"
                f"Split: {self.split}\n"
                f"Make sure:\n"
                f"  1. Random views exist in random_views_root\n"
                f"  2. Corresponding source views exist in source_root\n"
                f"  3. Folder structure matches: category/instance_id/random_N_views/"
            )

        return scenes

    def _apply_train_val_split(self, scenes: Dict) -> Dict:
        """
        Split scenes into train and validation sets.

        Strategy:
        - Sort all scene keys deterministically (alphabetically)
        - Reserve final 50 instances for validation
        - All other instances for training

        Args:
            scenes: Dictionary of all available scenes

        Returns:
            Filtered dictionary based on self.split
        """
        if not scenes:
            return scenes

        # Sort scene keys deterministically (alphabetically)
        all_scene_keys = sorted(scenes.keys())
        total_scenes = len(all_scene_keys)

        # Reserve final 50 instances for validation
        val_size = 50
        if total_scenes <= val_size:
            # Not enough scenes to split
            print(f"⚠️ Only {total_scenes} scenes available, cannot reserve {val_size} for validation")
            if self.split in ['val', 'test']:
                # Use all scenes for validation if requested
                print(f"   Using all {total_scenes} scenes for validation")
                return scenes
            else:
                # Use all scenes for training
                print(f"   Using all {total_scenes} scenes for training")
                return scenes

        # Split: final 50 for val, rest for train
        train_keys = all_scene_keys[:-val_size]
        val_keys = all_scene_keys[-val_size:]

        if self.split in ['val', 'test']:
            # Return validation set (final 50 instances)
            filtered_scenes = {k: scenes[k] for k in val_keys}
            print(f"🎯 Validation split: {len(filtered_scenes)} scenes (final 50 instances)")
            print(f"   First val scene: {val_keys[0]}")
            print(f"   Last val scene: {val_keys[-1]}")
        else:
            # Return training set (all except final 50)
            filtered_scenes = {k: scenes[k] for k in train_keys}
            print(f"🎯 Training split: {len(filtered_scenes)} scenes (all except final 50)")
            print(f"   First train scene: {train_keys[0]}")
            print(f"   Last train scene: {train_keys[-1]}")

        return filtered_scenes

    def _save_scene_cache(self, cache_path: Path, scenes: Dict):
        """Persist computed scene metadata."""
        if not scenes:
            return
        try:
            print(f"💾 Saving scene cache to: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(scenes, f, indent=2)
            print(f"✅ Cached {len(scenes)} scenes")
        except Exception as exc:
            print(f"Warning: Could not save scene cache to {cache_path}: {exc}")

    def _load_shared_target_cache(self):
        """Load shared target metadata generated by RandomView datasets, if present."""
        cache_candidates = [
            "RandomViewVaeLatentDataset_target_views.json",
            "RandomViewVideoDataset_target_views.json",
        ]
        for filename in cache_candidates:
            cache_path = self.cache_dir / filename
            if not cache_path.exists():
                continue
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                return cache, cache_path
            except Exception as exc:
                print(f"Warning: Failed to read shared cache {cache_path}: {exc}")
        return None, None

    def _build_scenes_from_target_cache(self, target_cache: Dict, folder_pattern: str) -> Dict:
        """Convert target_view_cache metadata into Objaverse scene records."""
        if not target_cache:
            return {}

        scenes = {}
        preferred_split = 'train' if self.split in (None, 'train') else 'test'
        fallback_splits = ['train', 'test']

        for scene_key, split_metadata in tqdm(target_cache.items()):
            metadata = split_metadata.get(preferred_split)
            if metadata is None:
                for split_name in fallback_splits:
                    metadata = split_metadata.get(split_name)
                    if metadata:
                        break
            if metadata is None:
                continue

            random_views_path = metadata.get('random_views_path')
            if not random_views_path:
                continue
            random_views_path = Path(random_views_path)

            if random_views_path.name != folder_pattern:
                continue

            num_views = metadata.get('num_views')
            if num_views and num_views != self.num_rendered_views:
                continue

            if not random_views_path.exists():
                continue

            source_instance_path = self.source_root / scene_key
            if not source_instance_path.exists():
                continue

            if not self._has_required_source_views(source_instance_path):
                continue

            scenes[scene_key] = {
                'source_instance_path': str(source_instance_path),
                'random_views_path': str(random_views_path),
                'random_state_idx': metadata.get('random_state_idx', 0)
            }

        return scenes

    def _discover_scenes(self, folder_pattern: str) -> Dict:
        """
        Discover all valid scenes by traversing filesystem.
        This is slow and should be cached.
        """
        scenes = {}

        if not self.random_views_root.exists():
            raise FileNotFoundError(f"random_views_root directory not found: {self.random_views_root}")

        # Traverse random_views_root to find all instances with random views
        for category_path in self.random_views_root.iterdir():
            if not category_path.is_dir():
                continue

            for instance_path in category_path.iterdir():
                if not instance_path.is_dir():
                    continue

                # Check if this instance has random views folder
                random_views_path = instance_path / folder_pattern
                if not random_views_path.exists():
                    continue

                # Check if at least one view exists
                first_view_file = random_views_path / "view0.png"
                if not first_view_file.exists():
                    continue

                # Check if corresponding source folder exists
                source_instance_path = self.source_root / category_path.name / instance_path.name
                if not source_instance_path.exists():
                    continue

                # Check if source views are available
                if not self._has_required_source_views(source_instance_path):
                    continue

                scene_key = f"{category_path.name}/{instance_path.name}"
                scenes[scene_key] = {
                    'source_instance_path': str(source_instance_path),
                    'random_views_path': str(random_views_path),
                    'random_state_idx': self._get_random_state_idx(random_views_path)
                }

        return scenes

    def _has_required_source_views(self, source_instance_path: Path) -> bool:
        """Check if source instance has all required source views."""
        if not self.return_source_views:
            # If not using source views, no requirement
            return True

        # Check if all required source view indices exist
        max_required_idx = max(self.source_view_indices)
        for view_idx in range(max_required_idx + 1):
            view_dir = source_instance_path / f'view_{view_idx}'
            if not view_dir.exists():
                return False
        return True

    def _get_random_state_idx(self, random_views_path: Path) -> int:
        """Get random_state_idx from timestep_info.json."""
        timestep_info_path = random_views_path / "timestep_info.json"
        if not timestep_info_path.exists():
            return 0

        try:
            with open(timestep_info_path, 'r') as f:
                timestep_info = json.load(f)
            return timestep_info.get('random_state_idx', 0)
        except Exception as e:
            print(f"Warning: Could not read timestep_info.json: {e}")
            return 0

    def __len__(self):
        return len(self.all_ref_imgs)

    def _get_source_impath(self, source_instance_path: str, view_idx: int, frame_idx: int):
        """Get path to source image."""
        return osp.join(source_instance_path, f'view_{view_idx}', f'frame{frame_idx}.png')

    def _get_source_depthpath(self, source_instance_path: str, view_idx: int, frame_idx: int):
        """Get path to source depth map."""
        return osp.join(source_instance_path, f'view_{view_idx}', f'frame{frame_idx}_depth{frame_idx:04d}.exr')

    def _get_source_camera_path(self, source_instance_path: str, view_idx: int):
        """Get path to source camera metadata."""
        return osp.join(source_instance_path, f'view_{view_idx}', 'view.json')

    def _get_target_impath(self, random_views_path: str, view_idx: int):
        """Get path to target image."""
        return osp.join(random_views_path, f'view{view_idx}.png')

    def _get_target_camera_path(self, random_views_path: str, view_idx: int):
        """Get path to target camera metadata."""
        return osp.join(random_views_path, f'view{view_idx}.json')

    def _get_target_depthpath(self, random_views_path: str, view_idx: int, frame_idx: int):
        """Get path to target depth map."""
        return osp.join(random_views_path, f'view{view_idx}_depth{frame_idx:04d}.exr')

    def _read_depthmap_exr(self, depthpath):
        """Read depth map from EXR file."""
        try:
            import OpenEXR
            import Imath

            if not osp.exists(depthpath):
                return None

            exr_file = OpenEXR.InputFile(depthpath)
            header = exr_file.header()

            dw = header['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            # Read R channel as depth
            channel_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_array = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], size[0])

            return depth_array
        except Exception as e:
            print(f"Warning: Could not load depth from {depthpath}: {e}")
            return None

    def _load_camera_from_json(self, camera_path: str, image_height: int, image_width: int):
        """Load camera metadata from JSON file and convert to OpenCV format.

        Args:
            camera_path: Path to view.json file.
            image_height/image_width: Native resolution of the loaded image before any resizing.
        """
        image_height = int(image_height)
        image_width = int(image_width)
        with open(camera_path, 'r') as f:
            camera_data = json.load(f)

        # Extract camera pose (c2w matrix) in Blender space
        camera_pose_blender = np.array(camera_data['matrix_world'], dtype=np.float32)

        # Convert from Blender to OpenCV coordinate system
        # Blender: +X right, +Y forward, +Z up
        # OpenCV: +X right, +Y down, +Z forward
        blender_to_opencv = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float32)

        # Apply conversion: c2w_opencv = c2w_blender @ inv(blender_to_opencv)
        blender_to_opencv_inv = np.linalg.inv(blender_to_opencv)
        camera_pose = camera_pose_blender @ blender_to_opencv_inv

        # Extract FOV and compute intrinsics
        x_fov = camera_data.get('x_fov', camera_data.get('fov', 0.8575560450553894))
        y_fov = camera_data.get('y_fov', x_fov)

        # Compute focal lengths from FOV
        fx = image_width / (2 * np.tan(x_fov / 2))
        fy = image_height / (2 * np.tan(y_fov / 2))
        cx = image_width / 2
        cy = image_height / 2

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return camera_pose, intrinsics

    def _get_views(self, idx, resolution, rng, num_views):
        """
        Load views for the given index.

        When return_source_views=True:
        - First load all source views from self.source_view_indices
        - Then load (num_views - len(source_views)) from target views

        When return_source_views=False:
        - Load num_views from target views only

        Returns:
            List of view dictionaries with required format (total length = num_views)
        """
        scene_key = self.all_ref_imgs[idx]
        scene_info = self.scenes[scene_key]

        source_instance_path = scene_info['source_instance_path']
        random_views_path = scene_info['random_views_path']
        random_state_idx = scene_info['random_state_idx']

        views = []

        if self.return_source_views:
            # First load all source views specified in source_view_indices
            source_views = self._load_source_views(
                source_instance_path, random_state_idx, resolution, rng,
                num_views=len(self.source_view_indices)
            )
            views.extend(source_views)

            # Then load remaining views from target to reach num_views total
            num_remaining = num_views - len(source_views)
            if num_remaining > 0:
                target_views = self._load_target_views(
                    random_views_path, random_state_idx, resolution, rng, num_remaining
                )
                views.extend(target_views)
        else:
            # Return only target views (random views)
            views = self._load_target_views(
                random_views_path, random_state_idx, resolution, rng, num_views
            )

        return views

    def _load_source_views(self, source_instance_path: str, random_state_idx: int,
                          resolution, rng, num_views):
        """Load source views (orthogonal views) from self.source_view_indices."""
        views = []

        # Use source_view_indices to determine which views to load
        # Only load up to num_views requested
        selected_indices = self.source_view_indices[:num_views]

        for view_idx in selected_indices:
            impath = self._get_source_impath(source_instance_path, view_idx, random_state_idx)

            if not osp.exists(impath):
                # Try frame 0 as fallback
                impath = self._get_source_impath(source_instance_path, view_idx, 0)

            if not osp.exists(impath):
                print(f"Warning: Source image not found: {impath}")
                continue

            # Load image
            rgb_image = imread_cv2(impath)

            # Load camera at native resolution
            camera_path = self._get_source_camera_path(source_instance_path, view_idx)
            if not osp.exists(camera_path):
                print(f"Warning: Source camera not found: {camera_path}")
                continue

            native_h, native_w = rgb_image.shape[:2]
            camera_pose, intrinsics = self._load_camera_from_json(camera_path, native_h, native_w)

            # Load depth map
            depthpath = self._get_source_depthpath(source_instance_path, view_idx, random_state_idx)
            depthmap = self._read_depthmap_exr(depthpath)

            assert depthmap is not None
                # Create dummy depth map at native resolution
                # depthmap = np.ones((native_h, native_w), dtype=np.float32) * 5.0

            # Resize if necessary
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                print(f"Warning: No valid depth in {impath}")
                # Add small valid region to avoid complete failure
                depthmap[0, 0] = 1.0

            # Generate img mask and ray mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, len(views), rng)
            
            if self.mask_bg:
                bg_mask = np.isclose(depthmap, 1e10)
                depthmap[bg_mask] = self.bg_depth_value # background has depth=1e10. set to a emperical number here.

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=impath.split('/')[-4] + '/' + impath.split('/')[-3],  # category/instance
                instance=f"source_view_{view_idx}",
                is_metric=self.is_metric,
                is_video=False,
                quantile=np.array(0.9, dtype=np.float32),
                img_mask=img_mask,
                ray_mask=ray_mask,
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
            ))

        return views

    def _load_target_views(self, random_views_path: str, random_state_idx: int,
                          resolution, rng, num_views):
        """Load target views (random views)."""
        views = []

        # Sample random view indices
        selected_view_indices = rng.choice(self.num_rendered_views,
                                          size=min(num_views, self.num_rendered_views),
                                          replace=False).tolist()

        for view_idx in selected_view_indices:
            impath = self._get_target_impath(random_views_path, view_idx)

            if not osp.exists(impath):
                print(f"Warning: Target image not found: {impath}")
                continue

            # Load image
            rgb_image = imread_cv2(impath)

            # Load camera at native resolution
            camera_path = self._get_target_camera_path(random_views_path, view_idx)
            if not osp.exists(camera_path):
                print(f"Warning: Target camera not found: {camera_path}")
                continue

            native_h, native_w = rgb_image.shape[:2]
            camera_pose, intrinsics = self._load_camera_from_json(camera_path, native_h, native_w)

            # Load depth map
            depthpath = self._get_target_depthpath(random_views_path, view_idx, random_state_idx)
            depthmap = self._read_depthmap_exr(depthpath)

            if depthmap is None:
                # Create dummy depth map at native resolution
                depthmap = np.ones((native_h, native_w), dtype=np.float32) * 5.0

            # Resize if necessary
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                print(f"Warning: No valid depth in {impath}")
                depthmap[0, 0] = 1.0

            # Generate img mask and ray mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, len(views), rng)

            # Apply background masking (same as source views)
            if self.mask_bg:
                bg_mask = np.isclose(depthmap, 1e10)
                depthmap[bg_mask] = self.bg_depth_value  # background has depth=1e10. set to an empirical number here.

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=impath.split('/')[-4] + '/' + impath.split('/')[-3],  # category/instance
                instance=f"random_view_{view_idx}",
                is_metric=self.is_metric,
                is_video=False,
                quantile=np.array(0.9, dtype=np.float32),
                img_mask=img_mask,
                ray_mask=ray_mask,
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
            ))

        return views
