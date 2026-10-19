import torch
from torch.utils.data.distributed import DistributedSampler

from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .base.easy_dataset import TakeDataset  # noqa
from holo4d_runtime.data.stream3r.dust3r.utils.dist_utils import get_world_size, get_rank
from .arkitscenes import ARKitScenes_Multi  # noqa
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .bedlam import BEDLAM_Multi
from .blendedmvs import BlendedMVS_Multi  # noqa
from .co3d import Co3d_Multi  # noqa
from .cop3d import Cop3D_Multi
from .dl3dv import DL3DV_Multi
from .dl3dv_with_static_track import DL3DV_Multi_WithStaticTrack
from .dynamic_replica import DynamicReplica
from .eden import EDEN_Multi
from .hypersim import HyperSim_Multi
from .irs import IRS
from .hoi4d import HOI4D_Multi
from .kubric import Kubric  # noqa
from .kubric_with_tracks import KubricWithTracks  # noqa
from .kubric_multicam_processed import KubricMultiCameraProcessed  # noqa
from .kubric_multicam_processed_xyzvid_frame0_gt import KubricMultiCameraProcessed_XYZVidFrame0GT  # noqa
from .kubric_multicam_processed_xyzvid_frame0_gt_81frames import KubricMultiCameraProcessed_XYZVidFrame0GT_81Frames  # noqa
from .kubric_multicam_processed_xyzvid_frame0_gt_preloaded import KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded  # noqa
from .kubric_multicam_processed_xyzvid_frame0_gt_preloaded_81frames import KubricMultiCameraProcessed_XYZVidFrame0GT_Preloaded_81Frames  # noqa
from .kubric_multicam_processed_81frames import KubricMultiCameraProcessed_81Frames  # noqa
from .mapfree import MapFree_Multi
from .megadepth import MegaDepth_Multi  # noqa
from .mp3d import MP3D_Multi
from .mvs_synth_with_static_track import MVS_Synth_Multi_WithStaticTrack
from .mvimgnet import MVImgNet_Multi
from .mvs_synth import MVS_Synth_Multi
from .omniobject3d import OmniObject3D_Multi
from .pointodyssey import PointOdyssey_Multi
from .realestate10k import RE10K_Multi
from .scannet import ScanNet_Multi
from .scannet_with_static_track import ScanNet_Multi_WithStaticTrack
from .scannetpp import ScanNetpp_Multi  # noqa
from .smartportraits import SmartPortraits_Multi
from .spring import Spring
from .syn4d import Syn4D_Multi
from .syn4d_nvs import Syn4D_Multi_NVS, Syn4D_Multi_NVS_FixedPair
from .track4d360_mixed_nvs import Kubric_NVS, StaticScene_NVS, StaticScene_NVS_FixedScene, StaticScene_NVS_FixedWindows, RE10K_NVS, Objaverse_NVS, HomographyNVS, ReCamMaster_NVS, ReCamMasterCowVGGT_NVS, SynCamMaster_NVS, GeometryRobustnessWrapper_NVS  # noqa
from .droid_nvs import DROID_NVS  # noqa
from .omni_droid_nvs import OMNI_DROID_NVS  # noqa
from .synscapes import SynScapes
from .tartanair import TartanAir_Multi
from .tartanair_with_static_track import TartanAir_Multi_WithStaticTrack
from .threedkb import ThreeDKenBurns
from .uasol import UASOL_Multi
from .urbansyn import UrbanSyn
from .unreal4k import UnReal4K_Multi
from .vkitti2 import VirtualKITTI2_Multi  # noqa
from .waymo import Waymo_Multi  # noqa
from .wildrgbd import WildRGBD_Multi  # noqa
from .objaverse import Objaverse_Multi  # noqa
from .davis_stream3r_pred import DavisStream3RPred  # noqa

# from spann3r, slam3r
from .habitat import Habitat
from .project_aria_seq import Aria_Seq


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    persistent_workers=False,
    multiprocessing_context=None,
    fixed_length=False,
):
    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            fixed_length=fixed_length,
        )
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    # Check if sampler is a BatchSampler (like BatchedRandomSampler)
    # If so, use batch_sampler parameter instead of sampler + batch_size
    from torch.utils.data import BatchSampler
    if isinstance(sampler, BatchSampler):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,  # Use batch_sampler for BatchedRandomSampler
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )

    return data_loader
