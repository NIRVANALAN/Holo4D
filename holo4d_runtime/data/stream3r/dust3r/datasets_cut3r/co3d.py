import os.path as osp
from pdb import set_trace as st
import json
import itertools
from collections import deque
import sys

import cv2
import numpy as np
import time

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from holo4d_runtime.data.stream3r.dust3r.utils.image import imread_cv2

from safetensors.numpy import save_file, load_file


class Co3d_Multi(BaseMultiViewDataset):

    def __init__(self, mask_bg="rand", load_uniform_context=False, overfitting=False,
                 use_bounded_video_sampling=False, max_distance_between_context_views=100,
                 initial_max_distance_between_context_views=25,
                 *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, "rand")
        self.mask_bg = mask_bg
        self.is_metric = False
        self.dataset_label = "Co3d_v2"
        self.load_uniform_context = load_uniform_context
        self.is_synthetic = False

        # Bounded video sampling parameters (PixelSplat-style)
        self.use_bounded_video_sampling = use_bounded_video_sampling
        self.max_distance_between_context_views = max_distance_between_context_views
        self.initial_max_distance_between_context_views = initial_max_distance_between_context_views

        overfitting_id = ("apple", "608_95658_192033")
        # overfitting_id = ("apple", "272_29064_56962") # ! align with the chunk dataset? Not in here.
        # overfitting = False # whether working on a single object
        # overfitting = True # whether working on a single object
        
        # ! since co3d is a video, uniform sampling encourages better coverage
        # self.uniform_view_sample = False
        # self.uniform_view_sample = True

        # load all scenes
        with open(osp.join(self.ROOT, f"selected_seqs_{self.split}.json"),
                  "r") as f:
            self.scenes = json.load(f)
            self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
            self.scenes = {
                (k, k2): v2
                for k, v in self.scenes.items()
                for k2, v2 in v.items()
            }

            if overfitting:
                self.scenes = {overfitting_id: self.scenes[overfitting_id]}

        self.scene_list = list(self.scenes.keys())
        cut_off = (self.num_views if not self.allow_repeat else max(
            self.num_views // 3, 3))
        self.cut_off = cut_off
        self.all_ref_imgs = [(key, value)
                             for key, values in self.scenes.items()
                             for value in values[:len(values) - cut_off + 1]]
        self.invalidate = {scene: {} for scene in self.scene_list}
        self.invalid_scenes = {scene: False for scene in self.scene_list}

    def __len__(self):
        return len(self.all_ref_imgs)

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images",
                        f"frame{view_idx:06n}.safetensor")

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images",
                        f"frame{view_idx:06n}.jpg")

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "depths",
                        f"frame{view_idx:06n}.jpg.geometric.png")

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "masks",
                        f"frame{view_idx:06n}.png")

    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(
            input_metadata["maximum_depth"])
        return depthmap
    
    def _get_sequential_ctx_views(self, num_views, ref_img_idx, image_pool, rng):
        """
        Load uniformly sampled image IDs from the input.

        Args:
            num_views: number of views to return
            ref_img_idx: the reference image index (last image)
            image_pool: all available image indices
            rng: random number generator

        Returns:
            imgs_idxs: list of uniformly sampled image indices
            ordered_video: False (as specified)
        """
        pos_ref = image_pool.index(ref_img_idx)
        total_frames = len(image_pool)

        # Calculate uniform spacing
        if num_views == 1:
            imgs_idxs = [pos_ref]
        elif num_views >= total_frames:
            # If requesting more views than available, return all
            imgs_idxs = list(range(total_frames))
        else:
            # Uniformly sample num_views from the available frames
            # Use linspace to get uniform indices
            uniform_positions = np.linspace(pos_ref, total_frames - 1, num_views, dtype=int)
            imgs_idxs = uniform_positions.tolist()

        return imgs_idxs, False

    def _get_bounded_video_views(self, num_views, image_pool, rng):
        """
        Sample video views with bounded spacing (PixelSplat-style).

        Randomly samples a video interval where the distance between first and last view
        is bounded by max_distance_between_context_views. Returns sequential views
        within that interval.

        Args:
            num_views: number of views to return
            image_pool: all available image indices
            rng: random number generator

        Returns:
            imgs_idxs: list of sequential image indices (positions in image_pool)
            ordered_video: True (views are sequential)
        """
        total_frames = len(image_pool)

        if num_views == 1:
            start_pos = rng.integers(0, total_frames)
            return [start_pos], True

        if num_views >= total_frames:
            return list(range(total_frames)), True

        max_span = min(self.max_distance_between_context_views, total_frames - 1)
        max_interval_length = min(max_span + 1, total_frames)

        if max_interval_length < num_views:
            interval_length = num_views
        else:
            preferred_max_length = min(
                self.initial_max_distance_between_context_views + 1,
                max_interval_length,
            )
            if rng.random() < 0.7 and preferred_max_length >= num_views:
                interval_length = rng.integers(num_views, preferred_max_length + 1)
            else:
                interval_length = rng.integers(num_views, max_interval_length + 1)

        max_start_pos = total_frames - interval_length
        start_pos = rng.integers(0, max_start_pos + 1)
        end_pos = start_pos + interval_length - 1

        imgs_idxs_float = np.linspace(start_pos, end_pos, num_views)
        imgs_idxs = np.round(imgs_idxs_float).astype(int).tolist()

        # Enforce uniqueness by filling missing indices from the interval
        unique_idxs = []
        seen = set()
        for idx in imgs_idxs:
            if idx not in seen:
                unique_idxs.append(idx)
                seen.add(idx)

        if len(unique_idxs) < num_views:
            for candidate in range(start_pos, end_pos + 1):
                if candidate in seen:
                    continue
                unique_idxs.append(candidate)
                seen.add(candidate)
                if len(unique_idxs) == num_views:
                    break

        return unique_idxs, True

    def _get_views(self, idx, resolution, rng, num_views):
        invalid_seq = True
        scene_info, ref_img_idx = self.all_ref_imgs[idx]
        # st()

        while invalid_seq:
            while self.invalid_scenes[scene_info]:
                idx = rng.integers(low=0, high=len(self.all_ref_imgs))
                scene_info, ref_img_idx = self.all_ref_imgs[idx]

            obj, instance = scene_info

            image_pool = self.scenes[obj, instance]
            if len(image_pool) < self.cut_off: # if the scene does not contain enough views
                print("Invalid scene!")
                self.invalid_scenes[scene_info] = True
                continue

            if self.use_bounded_video_sampling:
                # Bounded video sampling (PixelSplat-style)
                imgs_idxs, ordered_video = self._get_bounded_video_views(
                    num_views, image_pool, rng)
            elif self.load_uniform_context:
                # Uniform sampling from ref_img to end
                imgs_idxs, ordered_video = self._get_sequential_ctx_views(
                    num_views, image_pool[0], image_pool, rng)
            else:
                # Default: sequential from random start
                imgs_idxs, ordered_video = self.get_seq_from_start_id(
                    num_views, ref_img_idx, image_pool, rng)

            # print('imgs_idxs:', imgs_idxs)

            if resolution not in self.invalidate[
                    obj, instance]:  # flag invalid images
                self.invalidate[obj, instance][resolution] = [
                    False for _ in range(len(image_pool))
                ]
            # decide now if we mask the bg
            mask_bg = (self.mask_bg
                       == True) or (self.mask_bg == "rand"
                                    and rng.choice(2, p=[0.9, 0.1]))
            views = []

            imgs_idxs = deque(imgs_idxs)

            while len(imgs_idxs) > 0:  # some images (few) have zero depth
                if (len(image_pool) -
                        sum(self.invalidate[obj, instance][resolution])
                        < self.cut_off):
                    print("Invalid scene!")
                    invalid_seq = True
                    self.invalid_scenes[scene_info] = True
                    break

                im_idx = imgs_idxs.pop()
                if self.invalidate[obj, instance][resolution][im_idx]:
                    # search for a valid image
                    random_direction = 2 * rng.choice(2) - 1
                    for offset in range(1, len(image_pool)):
                        tentative_im_idx = (
                            im_idx +
                            (random_direction * offset)) % len(image_pool)
                        # Check candidate validity
                        if (not self.invalidate[obj, instance][resolution][tentative_im_idx]):
                            im_idx = tentative_im_idx
                            break
                    else:
                        # No valid frame found, mark scene as invalid
                        print(f"Invalid scene! Could not find replacement for frame {im_idx}")
                        invalid_seq = True
                        self.invalid_scenes[scene_info] = True
                        break

                view_idx = image_pool[im_idx]
                impath = self._get_impath(obj, instance, view_idx)
                depthpath = self._get_depthpath(obj, instance, view_idx)

                # load camera params
                metadata_path = self._get_metadatapath(obj, instance, view_idx)
                # input_metadata = np.load(metadata_path)
                # st()
                input_metadata = load_file(metadata_path)
                camera_pose = input_metadata["camera_pose"].astype(np.float32)
                intrinsics = input_metadata["camera_intrinsics"].astype(
                    np.float32)

                # load image and depth
                rgb_image = imread_cv2(impath)
                depthmap = self._read_depthmap(depthpath, input_metadata)

                if mask_bg:
                    # load object mask
                    maskpath = self._get_maskpath(obj, instance, view_idx)
                    maskmap = imread_cv2(
                        maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    maskmap = (maskmap / 255.0) > 0.1

                    # update the depthmap with mask
                    depthmap *= maskmap

                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image,
                    depthmap,
                    intrinsics,
                    resolution,
                    rng=rng,
                    info=impath)
                num_valid = (depthmap > 0.0).sum()
                if num_valid == 0:
                    # problem, invalidate image and retry
                    self.invalidate[obj, instance][resolution][im_idx] = True
                    imgs_idxs.append(im_idx)
                    continue

                # generate img mask and raymap mask
                img_mask, ray_mask = self.get_img_and_ray_masks(
                    self.is_metric, len(views), rng)

                views.append(
                    dict(
                        img=rgb_image,
                        depthmap=depthmap,
                        camera_pose=camera_pose,
                        camera_intrinsics=intrinsics,
                        dataset=self.dataset_label,
                        label=osp.join(obj, instance),
                        instance=osp.split(impath)[1],
                        is_metric=self.is_metric,
                        is_video=ordered_video,
                        quantile=np.array(0.9, dtype=np.float32),
                        img_mask=img_mask,
                        ray_mask=ray_mask,
                        camera_only=False,
                        depth_only=False,
                        single_view=False,
                        reset=False,
                    ))

            if len(views) == num_views and not all(
                [view["instance"] == views[0]["instance"] for view in views]):
                invalid_seq = False
        return views
