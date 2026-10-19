from pathlib import Path

import numpy as np

from holo4d.benchmark.vista4d_trajectories import (
    anchor_trajectory_to_source_frame0,
    discover_vista4d_transfer_pairs_from_metadata,
    discover_vista4d_pairs_from_metadata,
    lookup_vista4d_source_prompt,
)


def _rot_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_anchor_trajectory_to_source_frame0_keeps_first_target_pose_and_k_shared():
    target = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    target[0, :3, :3] = _rot_z(0.4)
    target[0, :3, 3] = np.array([3.0, -2.0, 1.0], dtype=np.float32)
    target[1, :3, 3] = np.array([4.0, -2.0, 1.0], dtype=np.float32)
    target[2, :3, :3] = _rot_z(0.7)
    target[2, :3, 3] = np.array([5.0, -1.5, 1.2], dtype=np.float32)

    target_k = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
    target_k[:, 0, 0] = np.array([1000.0, 900.0, 800.0], dtype=np.float32)
    target_k[:, 1, 1] = np.array([1000.0, 900.0, 800.0], dtype=np.float32)
    target_k[:, 0, 2] = 640.0
    target_k[:, 1, 2] = 360.0

    source0 = np.eye(4, dtype=np.float32)
    source0[:3, 3] = np.array([0.25, 0.5, -0.75], dtype=np.float32)
    source_k = np.array(
        [[512.0, 0.0, 256.0], [0.0, 512.0, 144.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    anchored, anchored_k = anchor_trajectory_to_source_frame0(
        target_c2w=target,
        target_intrinsics=target_k,
        source_frame0_c2w=source0,
        source_intrinsics=source_k,
    )

    expected_relative = np.linalg.inv(target[0])[None] @ target
    expected_anchored = source0[None] @ expected_relative
    expected_anchored[0] = source0
    assert np.allclose(anchored, expected_anchored)
    assert np.allclose(anchored[0], source0)
    assert np.allclose(anchored_k[0], source_k)
    assert np.isclose(anchored_k[-1, 0, 0], source_k[0, 0] * 800.0 / 1000.0)
    assert np.isclose(anchored_k[-1, 1, 1], source_k[1, 1] * 800.0 / 1000.0)


def test_discover_vista4d_pairs_can_cap_two_cameras_per_selected_video(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "video,camera,source,prompt\n"
        "avocado-slice,close-crane-above,pexels,p\n"
        "avocado-slice,left-front-zoom,pexels,p\n"
        "avocado-slice,third-extra,pexels,p\n"
        "cows,crane-below-right,davis,p\n"
        "cows,front-dolly-zoom,davis,p\n"
        "goat,side-truck,davis,p\n",
        encoding="utf-8",
    )

    pairs = discover_vista4d_pairs_from_metadata(
        metadata,
        video_filter=["avocado-slice", "cows"],
        source_filter=None,
        max_cameras_per_video=2,
    )

    assert pairs == [
        ("avocado-slice", "close-crane-above"),
        ("avocado-slice", "left-front-zoom"),
        ("cows", "crane-below-right"),
        ("cows", "front-dolly-zoom"),
    ]


def test_discover_vista4d_transfer_pairs_samples_cross_video_donors(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    rows = ["video,camera,source,prompt\n"]
    for video in ["source-a", "source-b", "source-c"]:
        for i in range(5):
            rows.append(f"{video},cam-{i},pexels,p\n")
    metadata.write_text("".join(rows), encoding="utf-8")

    camera_root = tmp_path / "cameras"
    for video in ["source-a", "source-b", "source-c"]:
        for i in range(5):
            cam_dir = camera_root / video
            cam_dir.mkdir(parents=True, exist_ok=True)
            c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
            np.savez(cam_dir / f"cam-{i}.npz", cam_c2w=c2w)

    pairs = discover_vista4d_transfer_pairs_from_metadata(
        metadata,
        camera_root=camera_root,
        video_filter=["source-a", "source-b"],
        source_filter="pexels",
        donor_source_filter="pexels",
        trajectories_per_video=4,
        seed=7,
        max_forward_angle_deg=90.0,
    )

    assert len(pairs) == 8
    for source_video, output_camera, donor_video, donor_camera in pairs:
        assert source_video in {"source-a", "source-b"}
        assert donor_video != source_video
        assert output_camera == f"{donor_video}__{donor_camera}"


def test_discover_vista4d_transfer_pairs_filters_back_view_donors(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "video,camera,source,prompt\n"
        "source-a,self,pexels,p\n"
        "donor-front,front,pexels,p\n"
        "donor-back,back,pexels,p\n",
        encoding="utf-8",
    )

    camera_root = tmp_path / "cameras"
    front_dir = camera_root / "donor-front"
    front_dir.mkdir(parents=True, exist_ok=True)
    front_c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    np.savez(front_dir / "front.npz", cam_c2w=front_c2w)

    back_dir = camera_root / "donor-back"
    back_dir.mkdir(parents=True, exist_ok=True)
    back_c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    back_c2w[1, :3, :3] = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    np.savez(back_dir / "back.npz", cam_c2w=back_c2w)

    source_dir = camera_root / "source-a"
    source_dir.mkdir(parents=True, exist_ok=True)
    np.savez(source_dir / "self.npz", cam_c2w=front_c2w)

    pairs = discover_vista4d_transfer_pairs_from_metadata(
        metadata,
        camera_root=camera_root,
        video_filter=["source-a"],
        source_filter="pexels",
        donor_source_filter="pexels",
        trajectories_per_video=2,
        seed=0,
        max_forward_angle_deg=90.0,
    )

    assert pairs == [("source-a", "donor-front__front", "donor-front", "front")]


def test_lookup_vista4d_source_prompt_uses_source_video_not_donor_camera(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "video,camera,source,prompt\n"
        "source-a,self,pexels,caption for source\n"
        "donor-front,front,pexels,caption for donor\n",
        encoding="utf-8",
    )

    prompt = lookup_vista4d_source_prompt(
        metadata,
        source_video="source-a",
        source_camera="donor-front__front",
        default="fallback",
    )

    assert prompt == "caption for source"
