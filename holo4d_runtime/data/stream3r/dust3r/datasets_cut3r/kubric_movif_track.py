import cv2
import os.path as osp
import numpy as np
import os
from safetensors.numpy import load_file
from tqdm import tqdm
from pdb import set_trace as st

from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_stereo_view_dataset import BaseStereoDynamicViewDataset
from holo4d_runtime.data.stream3r.dust3r.utils.image import imread_cv2


class Kubric_Tracks(BaseStereoDynamicViewDataset):
    def __init__(self, debug_return_visibility=False, *args, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        self.debug_return_visibility = debug_return_visibility
        assert self.split in ["train", "test"]
        assert self.aug_crop in [0, False]
        # if self.split == "train":
        #     self.ROOT = os.path.join(self.ROOT, "trackings")
        # elif self.split == "test":
        #     self.ROOT = os.path.join(self.ROOT, "trackings_val")
        self.ROOT = ROOT # only one folder available
        
        self.M_Blender2Opencv = np.array(
            [[1,  0,  0,  0],
              [0, -1,  0,  0],
              [0,  0, -1,  0],
              [0,  0,  0,  1]]
        )

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.sceneids = os.listdir(self.ROOT)

    def __len__(self):
        return len(self.sceneids)

    def get_image_num(self):
        return len(self.sceneids * self.num_views)

    def _get_views(self, idx, resolution, rng):
        resolution, num_views = resolution
        scene_id = self.sceneids[idx]
        scene_dir = osp.join(self.ROOT, scene_id)
        depth_range = np.load(osp.join(scene_dir, scene_id + ".npy"), allow_pickle=True).item()["depth_range"].astype(np.float32)

        is_reverse = self._rng.choice([True, False])
        if is_reverse:
            cam = np.load(osp.join(scene_dir, scene_id + "_dense_reverse.npy"), allow_pickle=True).item()
        else:
            cam = np.load(osp.join(scene_dir, scene_id + "_dense.npy"), allow_pickle=True).item()

        # resort the tracking trajectory points, as queries is not in order
        dense_queries = cam["queries"].astype(np.float32) # N, 3
        dense_queries = np.round(dense_queries[:, 1:] - 0.5).astype(int)
        dense_queries_indices =  512 * dense_queries[:, 0] + dense_queries[:, 1]
        sort_indices = np.argsort(dense_queries_indices)
        cam["cam_pos"] = cam["cam_pos"][sort_indices]
        if self.debug_return_visibility:
            cam["visibility"] = ~(cam["visibility"][sort_indices].astype(bool)) # so that true is visible

        # the camera system in kubric is blender, while we use opencv here
        cam["cam_pos"][:, :, 1] *= -1
        cam["cam_pos"][:, :, 2] *= -1

        views = []

        total_frames = cam["intrinsics"].shape[0]
        idxs = [i for i in range(total_frames)]
        if is_reverse:
            idxs = idxs[::-1]
        first = idxs[0]
        rest = idxs[1:]
        if self._rng.random() < 0.8:
            sampled = self._rng.choice(rest, size=num_views-1, replace=False)
            sampled_sorted = sorted(sampled, key=lambda x: idxs.index(x))
        else:
            sampled_sorted = rest[:num_views-1]
        idxs = [first] + sampled_sorted

        for i in idxs:
            basename = str(i).zfill(3)
            rgb_dir = osp.join(scene_dir, "frames")
            depth_dir = osp.join(scene_dir, "depths")

            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
            depth = cv2.imread(osp.join(depth_dir, basename + ".png"), cv2.IMREAD_ANYDEPTH).astype(np.float32)

            depth_min = depth_range[0]
            depth_max = depth_range[1]
            
            depthmap = depth_min + depth * (depth_max-depth_min) / 65535.0

            intrinsics = cam["intrinsics"][i]
            h, w = rgb_image.shape[:2]
            intrinsics = np.array([
                [ intrinsics[0][0] * w,  0.   , w / 2  ],
                [ 0.   , -intrinsics[1][1] * h, h / 2  ],
                [ 0.   ,  0.   , 1.   ]
            ])

            # the depth map in kubric the distance from point to optical center
            # but we need to use the depth where the distance is along z axis
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            depthmap /= np.sqrt(1 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)

            # the camera system in kubric is blender, while we use opencv here
            camera_pose = cam["matrix_world"][i] @ self.M_Blender2Opencv
            track = cam["cam_pos"][:, i]

            # this use 16% time, we skip the check
            # if np.any(np.isinf(track)):
            #     print(f"warning: [kubric] {self.split} {scene_id} track contains inf")
            #     return self._get_views((idx+1)%self.__len__(), resolution, rng)

            track = track.reshape(h, w, 3)
            track_valid_mask = np.ones((h, w), dtype=bool)

            if self.debug_return_visibility:
                visibility = cam["visibility"][:, i]
                visibility = visibility.reshape(h, w, 1)

            # we do not allow random transpose as tracking should be sequential
            rgb_image, depthmap, intrinsics, track, track_valid_mask = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=idx, track=track, track_valid_mask=track_valid_mask, allow_random_transpose=False
            )

            views.append(
                dict(
                    img=rgb_image,
                    track=track.astype(np.float32),
                    track_valid_mask=track_valid_mask,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="Kubric_Tracks",
                    label=scene_id + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".png"),
                    is_metric=self.is_metric,
                    is_video=True,
                    quantile=np.array(1.0, dtype=np.float32),
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
            if self.debug_return_visibility:
                views[-1]['visibility'] = visibility

        assert len(views) == num_views
        return views


if __name__ == "__main__":
    import torch
    import pause
    from torchvision.transforms import ToPILImage
    from holo4d_runtime.data.stream3r.dust3r.datasets_cut3r.base.base_stereo_view_dataset import view_name
    from holo4d_runtime.data.stream3r.dust3r.utils.image import rgb
    from IPython.display import display
    from holo4d_runtime.data.stream3r.dust3r.datasets.utils.transforms import ImgNorm, convert_input_to_pred_format, vis_track, TrackAug


    def main():
        from holo4d_runtime.data.stream3r.dust3r.viz.viser_visualizer_track import start_visualization

        num_views = 20
        dataset = Kubric_Tracks(
            split="train", 
            # num_views=20, 
            ROOT="data/cut3r_processed/3dscene_kubric_movif",
            aug_crop=0, 
            # resolution=[(512, 384)], 
            resolution=[((504, 378), num_views)],
            transform=ImgNorm
        )

        select_idx = 300
        views = dataset[select_idx]
        output = convert_input_to_pred_format(views)

        save_path = os.path.join("develop/2d_compare/test_data", views[0]['dataset'] + str(select_idx))
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(views)):
            print(view_name(views[i]))
            ToPILImage()(rgb(views[i]["img"])).save(f"{save_path}/{i}.png")

        server = start_visualization(
            output=output,
            min_conf_thr_percentile=0,
            global_conf_thr_value_to_drop_view=1,
            point_size=0.0016,
        )

        share_url = server.request_share_url()
        print(share_url)

        pause.days(1)

    main()