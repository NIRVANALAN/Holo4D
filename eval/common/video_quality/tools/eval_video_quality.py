#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import pickle
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMON_METRICS_ROOT = REPO_ROOT / "common_metrics_on_video_quality"

_FLAT_VIDEO_RE = re.compile(r"video(?P<index>\d+)\.mp4$")
_SYN4D_PRED_RE = re.compile(
    r"(?P<scene>[^/]+)/(?P<seq>[^/]+)/pred(?:_(?P<modality>rgb|pointmap))?_src(?P<src>\d+)_(?:to_tgt(?P<tgt>\d+)|official_camtype(?P<camtype>\d+))_chunk(?P<chunk>[^/]+)\.mp4$"
)
_KUBRIC_PRED_RE = re.compile(
    r"scene_(?P<scene>\d+)/pred(?:_rgb)?_src(?P<src>\d+)_(?:to_tgt(?P<tgt>\d+)|official_camtype(?P<camtype>\d+))\.mp4$"
)


@dataclass
class EvalSample:
    sample_id: str
    pred_path: Path
    prompt: Optional[str]
    source_path: Optional[Path]
    gt_path: Optional[Path]
    group_id: Optional[str]
    view_id: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Track4D-360 video quality metrics.")
    parser.add_argument("--mode", type=str, required=True, choices=["flat", "paired_csv", "syn4d", "kubric", "official_nvs"])
    parser.add_argument("--pred_root", type=str, default=None, help="Directory containing prediction mp4 files.")
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metrics. Defaults: flat=clip_t,clip_f,clip_v; syn4d/kubric=fid,fvd,clip_t,clip_f,clip_v,fvd_v.",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Flat mode dataset root containing metadata.csv and videos/.")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Flat mode metadata csv. Defaults to <dataset_path>/metadata.csv.")
    parser.add_argument("--paired_csv", type=str, default=None, help="CSV with sample_id,pred_path,source_path,gt_path,text for matched NVS evaluation.")
    parser.add_argument("--syn4d_dataset_root", type=str, default=None, help="Syn4D dataset root for prompt lookup.")
    parser.add_argument(
        "--syn4d_modality",
        type=str,
        default="rgb",
        choices=["rgb", "pointmap", "legacy", "all"],
        help="Which Syn4D prediction modality to evaluate.",
    )
    parser.add_argument("--kubric_dataset_root", type=str, default=None, help="Kubric dataset root for prompt lookup.")
    parser.add_argument(
        "--shared_eval_root",
        type=str,
        default=None,
        help="Official shared eval asset root containing source_cam01.mp4 and camXX/{caption.txt,gt.mp4}.",
    )
    parser.add_argument(
        "--official_modality",
        type=str,
        default="rgb",
        choices=["rgb", "pointmap"],
        help="Official NVS modality to evaluate.",
    )
    parser.add_argument(
        "--official_pointmap_source_mode",
        type=str,
        default="run",
        choices=["run", "shared"],
        help="Source pointmap path in official_nvs pointmap mode: run=per-cam source_pointmap.mp4, shared=shared_eval_root/source_cam01_pointmap.mp4.",
    )
    parser.add_argument("--kubric_split", type=str, default="train")
    parser.add_argument("--kubric_seq_data_file", type=str, default=None)
    parser.add_argument("--prompt_from", type=str, default="source", choices=["source", "target"])
    parser.add_argument("--source_video_name", type=str, default="source_cam01.mp4")
    parser.add_argument("--output_json", type=str, default=None, help="Where to write the aggregated metrics json.")
    parser.add_argument("--per_sample_csv", type=str, default=None, help="Where to write per-sample CLIP metrics.")
    parser.add_argument(
        "--clip_detail_dir",
        type=str,
        default=None,
        help="Optional directory for per-sequence frame-level CLIP inspection CSVs.",
    )
    parser.add_argument("--pred_traj", type=str, default=None, help="Predicted trajectory file, typically BA-Track's batrack_traj.txt.")
    parser.add_argument("--pred_traj_format", type=str, default="auto", choices=["auto", "tum", "batrack_pkl"])
    parser.add_argument("--gt_traj", type=str, default=None, help="Ground-truth trajectory path or directory.")
    parser.add_argument("--gt_traj_format", type=str, default="tum", choices=["tum", "sintel", "tartanair", "official_camjson"])
    parser.add_argument("--camera_extrinsics_path", type=str, default=None, help="Official camera_extrinsics.json path when gt_traj_format=official_camjson.")
    parser.add_argument("--official_cam_type", type=int, default=None, help="Camera type id for official camera trajectory evaluation.")
    parser.add_argument("--traj_skip", type=int, default=0)
    parser.add_argument("--traj_end", type=int, default=-1)
    parser.add_argument("--traj_stride", type=int, default=1)
    parser.add_argument("--pose_report", type=str, default=None, help="Optional text report path for pose metrics.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on frames per video.")
    parser.add_argument("--eval_height", type=int, default=None, help="Optional resize height for FID/FVD/FVD-V.")
    parser.add_argument("--eval_width", type=int, default=None, help="Optional resize width for FID/FVD/FVD-V.")
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="OpenCLIP model name. HF hub ids should be prefixed with hf-hub:.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default=None,
        help="Optional OpenCLIP pretrained tag or local checkpoint path. Leave empty when using hf-hub: model ids.",
    )
    parser.add_argument("--clip_bpe_path", type=str, default=None, help="Optional BPE file for local OpenCLIP checkpoints.")
    parser.add_argument(
        "--clip_backend", choices=["diffsynth", "standalone"], default="diffsynth",
        help="OpenCLIP implementation. 'standalone' avoids loading diffsynth and its FA3 extension in metric-only jobs.",
    )
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--fvd_method", type=str, default="styleganv", choices=["styleganv", "videogpt"])
    parser.add_argument(
        "--common_metrics_root",
        type=str,
        default=str(DEFAULT_COMMON_METRICS_ROOT),
        help="Path to the vendored common_metrics_on_video_quality directory.",
    )
    return parser.parse_args()


def parse_metrics(raw: Optional[str], mode: str) -> List[str]:
    if raw is None:
        if mode == "flat":
            raw = "clip_t,clip_f,clip_v"
        else:
            raw = "fvd,clip_t,clip_f,clip_v"
    metrics = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = {"fvd", "clip_t", "clip_f", "clip_v", "fvd_v", "rot_err", "trans_err"}
    unknown = sorted(set(metrics) - valid)
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")
    return metrics


def load_video_tensor(video_path: Path) -> torch.Tensor:
    reader = imageio.get_reader(str(video_path))
    frames = []
    try:
        for frame in reader:
            frames.append(torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float() / 255.0)
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames found in {video_path}")
    return torch.stack(frames, dim=0)


def resize_video(video_tchw: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if video_tchw.shape[1] == 1:
        video_tchw = video_tchw.repeat(1, 3, 1, 1)
    if tuple(video_tchw.shape[-2:]) == tuple(size_hw):
        return video_tchw
    return F.interpolate(video_tchw, size=size_hw, mode="bilinear", align_corners=False)


def truncate_video(video_tchw: torch.Tensor, max_frames: Optional[int]) -> torch.Tensor:
    if max_frames is None:
        return video_tchw
    return video_tchw[: max_frames]


def load_caption_json(caption_path: Path) -> Dict[str, Dict[str, str]]:
    if not caption_path.is_file():
        return {}
    with open(caption_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[str(key)] = {str(k): str(v) for k, v in value.items()}
    return out


def read_text_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def load_syn4d_prompt(
    syn4d_root: Path,
    scene_name: str,
    seq_root: str,
    view_idx: int,
    chunk_tag: str,
    caption_cache: Dict[Path, Dict[str, Dict[str, str]]],
) -> Optional[str]:
    caption_path = syn4d_root / "caption" / f"{scene_name}_per_video_chunks.json"
    if caption_path not in caption_cache:
        caption_cache[caption_path] = load_caption_json(caption_path)
    key = f"mp4/{seq_root}_{int(view_idx)}.mp4"
    prompt = caption_cache[caption_path].get(key, {}).get(str(chunk_tag))
    if prompt is None:
        return None
    prompt = prompt.strip()
    return prompt if prompt else None


def build_kubric_prompt_index(dataset_root: Path, split: str, seq_data_file: Optional[str]) -> Dict[Tuple[int, int], str]:
    from diffsynth.data.kubric_recammaster import KubricTextVideoDataset

    dataset = KubricTextVideoDataset(
        base_path=str(dataset_root),
        split=split,
        seq_data_file=seq_data_file,
        max_num_frames=1,
        num_frames=1,
        height=256,
        width=256,
        is_i2v=False,
    )
    return {(int(sample["scene_idx"]), int(sample["camera_idx"])): str(sample["text"]) for sample in dataset.samples}


def build_flat_samples(pred_root: Path, dataset_path: Path, metadata_csv: Path) -> List[EvalSample]:
    with open(metadata_csv, "r", encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    if not metadata:
        raise ValueError(f"{metadata_csv} is empty.")
    if "file_name" not in metadata[0] or "text" not in metadata[0]:
        raise ValueError(f"{metadata_csv} must contain file_name,text columns.")
    samples = []
    pred_files = []
    for path in pred_root.glob("video*.mp4"):
        match = _FLAT_VIDEO_RE.match(path.name)
        if match is not None:
            pred_files.append((int(match.group("index")), path))
    pred_files.sort(key=lambda item: item[0])
    if not pred_files:
        raise ValueError(f"No flat prediction files like video0.mp4 found under {pred_root}")

    for video_idx, pred_path in pred_files:
        if video_idx >= len(metadata):
            raise ValueError(f"{pred_path.name} exceeds metadata rows in {metadata_csv}")
        row = metadata[video_idx]
        source_path = dataset_path / "videos" / str(row["file_name"])
        samples.append(
            EvalSample(
                sample_id=f"video{video_idx}",
                pred_path=pred_path,
                prompt=str(row["text"]),
                source_path=source_path,
                # Track4D-360 patch: use source_path as gt_path so FVD/FID
                # compute distribution distance pred-vs-source (a camera-controlled
                # NVS metric — same scene under a different viewpoint, scored
                # against the original viewpoint as proxy).
                gt_path=source_path if source_path.is_file() else None,
                group_id=f"video{video_idx}",
                view_id=str(video_idx),
            )
        )
    return samples


def build_paired_csv_samples(manifest: Path) -> List[EvalSample]:
    """Read explicit prediction/source/target triplets; never substitute source for GT."""
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty paired manifest: {manifest}")
    required = {"sample_id", "pred_path", "source_path", "gt_path", "text"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing paired manifest columns: {sorted(missing)}")
    samples = []
    seen = set()
    for row in rows:
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate sample_id: {sample_id!r}")
        seen.add(sample_id)
        paths = {key: Path(row[key].strip()) for key in ("pred_path", "source_path", "gt_path")}
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{sample_id}: missing {key}: {path}")
        samples.append(EvalSample(
            sample_id=sample_id,
            pred_path=paths["pred_path"],
            source_path=paths["source_path"],
            gt_path=paths["gt_path"],
            prompt=row["text"].strip() or None,
            group_id=sample_id,
            view_id=row.get("target_camera") or None,
        ))
    return samples


def build_syn4d_samples(pred_root: Path, syn4d_root: Path, prompt_from: str, modality: str) -> List[EvalSample]:
    samples: List[EvalSample] = []
    caption_cache: Dict[Path, Dict[str, Dict[str, str]]] = {}
    for pred_path in sorted(pred_root.rglob("*.mp4")):
        if "compare_" in pred_path.name:
            continue
        if "pointmap" in pred_path.name and modality not in {"pointmap", "all"}:
            continue
        rel = pred_path.relative_to(pred_root).as_posix()
        match = _SYN4D_PRED_RE.match(rel)
        if match is None:
            continue
        file_modality = match.group("modality") or "legacy"
        if modality == "rgb" and file_modality not in {"rgb", "legacy"}:
            continue
        if modality == "pointmap" and file_modality != "pointmap":
            continue
        if modality == "legacy" and file_modality != "legacy":
            continue
        scene_name = match.group("scene")
        seq_root = match.group("seq")
        src = int(match.group("src"))
        tgt = match.group("tgt")
        chunk = match.group("chunk")
        if file_modality == "rgb":
            source_name = f"source_rgb_cam{src}_chunk{chunk}.mp4"
            gt_template = "gt_rgb_tgt{target}_chunk{chunk}.mp4"
        elif file_modality == "pointmap":
            source_name = f"source_pointmap_cam{src}_chunk{chunk}.mp4"
            gt_template = "gt_pointmap_tgt{target}_chunk{chunk}.mp4"
        else:
            source_name = f"source_view{src}_chunk{chunk}.mp4"
            gt_template = "gt_tgt{target}_chunk{chunk}.mp4"
        source_path = pred_path.parent / source_name
        gt_path = None
        view_id = None
        prompt_view = src
        sample_id = f"{scene_name}/{seq_root}/{pred_path.stem}"
        if tgt is not None:
            target_view = int(tgt)
            gt_name = gt_template.format(target=target_view, chunk=chunk)
            gt_path = pred_path.parent / gt_name
            view_id = str(target_view)
            if prompt_from == "target":
                prompt_view = target_view
        else:
            camtype = match.group("camtype")
            view_id = f"official_camtype{camtype}"
        prompt = load_syn4d_prompt(syn4d_root, scene_name, seq_root, prompt_view, chunk, caption_cache)
        samples.append(
            EvalSample(
                sample_id=sample_id,
                pred_path=pred_path,
                prompt=prompt,
                source_path=source_path if source_path.is_file() else None,
                gt_path=gt_path if gt_path is not None and gt_path.is_file() else None,
                group_id=f"{scene_name}/{seq_root}/{chunk}",
                view_id=view_id,
            )
        )
    if not samples:
        raise ValueError(f"No Syn4D prediction files found under {pred_root}")
    return samples


def build_kubric_samples(
    pred_root: Path,
    kubric_root: Path,
    split: str,
    seq_data_file: Optional[str],
    prompt_from: str,
) -> List[EvalSample]:
    prompt_index = build_kubric_prompt_index(kubric_root, split, seq_data_file)
    samples: List[EvalSample] = []
    for pred_path in sorted(pred_root.rglob("*.mp4")):
        if "compare_" in pred_path.name or "pointmap" in pred_path.name:
            continue
        rel = pred_path.relative_to(pred_root).as_posix()
        match = _KUBRIC_PRED_RE.match(rel)
        if match is None:
            continue
        scene_idx = int(match.group("scene"))
        src = int(match.group("src"))
        tgt = match.group("tgt")
        rgb_prefix = "pred_rgb_" if pred_path.name.startswith("pred_rgb_") else "pred_"
        source_name = f"source_rgb_cam{src}.mp4" if rgb_prefix == "pred_rgb_" else f"source_cam{src}.mp4"
        source_path = pred_path.parent / source_name
        gt_path = None
        view_id = None
        prompt_camera = src
        sample_id = f"scene_{scene_idx:05d}/{pred_path.stem}"
        if tgt is not None:
            target_camera = int(tgt)
            gt_name = f"gt_rgb_tgt{target_camera}.mp4" if rgb_prefix == "pred_rgb_" else f"gt_tgt{target_camera}.mp4"
            gt_path = pred_path.parent / gt_name
            view_id = str(target_camera)
            if prompt_from == "target":
                prompt_camera = target_camera
        else:
            camtype = match.group("camtype")
            view_id = f"official_camtype{camtype}"
        prompt = prompt_index.get((scene_idx, prompt_camera))
        samples.append(
            EvalSample(
                sample_id=sample_id,
                pred_path=pred_path,
                prompt=prompt,
                source_path=source_path if source_path.is_file() else None,
                gt_path=gt_path if gt_path is not None and gt_path.is_file() else None,
                group_id=f"scene_{scene_idx:05d}",
                view_id=view_id,
            )
        )
    if not samples:
        raise ValueError(f"No Kubric prediction files found under {pred_root}")
    return samples


def build_official_nvs_samples(
    pred_root: Path,
    shared_eval_root: Path,
    prompt_from: str,
    source_video_name: str,
    modality: str = "rgb",
    pointmap_source_mode: str = "run",
) -> List[EvalSample]:
    if modality == "rgb":
        source_path = shared_eval_root / source_video_name
        pred_name = "pred.mp4"
        gt_name = "gt.mp4"
    elif modality == "pointmap":
        source_path = shared_eval_root / "source_cam01_pointmap.mp4"
        pred_name = "pred_pointmap.mp4"
        gt_name = "gt_pointmap.mp4"
    else:
        raise ValueError(f"Unsupported official modality: {modality}")

    if not source_path.is_file() and modality == "rgb":
        raise ValueError(f"Missing official shared source video: {source_path}")
    source_prompt = read_text_file(shared_eval_root / "source_caption.txt")

    samples: List[EvalSample] = []
    for cam_dir in sorted(pred_root.glob("cam*")):
        if not cam_dir.is_dir():
            continue
        pred_path = cam_dir / pred_name
        if not pred_path.is_file():
            continue
        shared_cam_dir = shared_eval_root / cam_dir.name
        prompt = read_text_file(shared_cam_dir / "caption.txt") if prompt_from == "target" else source_prompt
        gt_path = shared_cam_dir / gt_name
        sample_source_path = source_path
        if modality == "pointmap":
            run_source = cam_dir / "source_pointmap.mp4"
            if pointmap_source_mode == "run" and run_source.is_file():
                sample_source_path = run_source
            elif not sample_source_path.is_file():
                sample_source_path = None

        cam_match = re.fullmatch(r"cam(\d+)", cam_dir.name)
        view_id = str(int(cam_match.group(1))) if cam_match else cam_dir.name
        samples.append(
            EvalSample(
                sample_id=f"{cam_dir.name}_{modality}",
                pred_path=pred_path,
                prompt=prompt,
                source_path=sample_source_path,
                gt_path=gt_path if gt_path.is_file() else None,
                group_id=pred_root.name,
                view_id=view_id,
            )
        )
    if not samples:
        raise ValueError(f"No official NVS prediction files found under {pred_root}")
    return samples


def build_samples(args: argparse.Namespace) -> List[EvalSample]:
    if args.mode == "paired_csv":
        if args.paired_csv is None:
            raise ValueError("--paired_csv is required in paired_csv mode.")
        return build_paired_csv_samples(Path(args.paired_csv))
    pred_root = Path(args.pred_root)
    if args.mode == "flat":
        if args.dataset_path is None:
            raise ValueError("--dataset_path is required in flat mode.")
        dataset_path = Path(args.dataset_path)
        metadata_csv = Path(args.metadata_csv) if args.metadata_csv else dataset_path / "metadata.csv"
        return build_flat_samples(pred_root, dataset_path, metadata_csv)
    if args.mode == "syn4d":
        if args.syn4d_dataset_root is None:
            raise ValueError("--syn4d_dataset_root is required in syn4d mode.")
        return build_syn4d_samples(pred_root, Path(args.syn4d_dataset_root), args.prompt_from, args.syn4d_modality)
    if args.mode == "official_nvs":
        if args.shared_eval_root is None:
            raise ValueError("--shared_eval_root is required in official_nvs mode.")
        return build_official_nvs_samples(
            pred_root=pred_root,
            shared_eval_root=Path(args.shared_eval_root),
            prompt_from=args.prompt_from,
            source_video_name=args.source_video_name,
            modality=args.official_modality,
            pointmap_source_mode=args.official_pointmap_source_mode,
        )
    if args.kubric_dataset_root is None:
        raise ValueError("--kubric_dataset_root is required in kubric mode.")
    return build_kubric_samples(
        pred_root=pred_root,
        kubric_root=Path(args.kubric_dataset_root),
        split=args.kubric_split,
        seq_data_file=args.kubric_seq_data_file,
        prompt_from=args.prompt_from,
    )


class VideoCache:
    def __init__(self) -> None:
        self._cache: Dict[Path, torch.Tensor] = {}

    def get(self, path: Optional[Path]) -> Optional[torch.Tensor]:
        if path is None:
            return None
        if path not in self._cache:
            self._cache[path] = load_video_tensor(path)
        return self._cache[path]


class OpenClipEmbedder:
    def __init__(
        self,
        device: torch.device,
        model_name: str,
        pretrained: Optional[str],
        clip_bpe_path: Optional[str],
        backend: str = "diffsynth",
    ) -> None:
        if backend == "standalone":
            if clip_bpe_path is not None:
                raise ValueError("--clip_bpe_path requires --clip_backend diffsynth")
            from open_clip import create_model_and_transforms, get_tokenizer
        elif backend == "diffsynth":
            from diffsynth.extensions.ImageQualityMetric.open_clip import create_model_and_transforms, get_tokenizer
        else:
            raise ValueError(f"Unknown CLIP backend: {backend}")

        self.device = device
        self.to_pil = transforms.ToPILImage()
        create_kwargs = {
            "model_name": model_name,
            "pretrained": pretrained,
            "precision": "fp32",
            "device": device,
            "jit": False,
            "output_dict": True,
        }
        if model_name.startswith("hf-hub:"):
            create_kwargs["pretrained"] = None
        self.model, _, self.preprocess = create_model_and_transforms(**create_kwargs)
        self.model.eval()
        self.tokenizer = (
            get_tokenizer(model_name)
            if backend == "standalone"
            else get_tokenizer(model_name, clip_bpe_path)
        )
        self.text_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        if prompt not in self.text_cache:
            text = self.tokenizer([prompt]).to(self.device)
            feat = self.model.encode_text(text, normalize=True).squeeze(0).detach().cpu()
            self.text_cache[prompt] = feat
        return self.text_cache[prompt]

    @torch.no_grad()
    def encode_images(self, frames_tchw: torch.Tensor, batch_size: int) -> torch.Tensor:
        feats = []
        for start in range(0, frames_tchw.shape[0], batch_size):
            batch_frames = frames_tchw[start : start + batch_size]
            processed = [self.preprocess(self.to_pil(frame.cpu())) for frame in batch_frames]
            batch = torch.stack(processed, dim=0).to(self.device)
            feat = self.model.encode_image(batch, normalize=True).detach().cpu()
            feats.append(feat)
        return torch.cat(feats, dim=0)


def make_safe_sample_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", sample_id).strip("_") or "sample"


def write_clip_detail_csv(
    output_dir: Optional[Path],
    sample: EvalSample,
    prompt: Optional[str],
    clip_t_scores: Optional[torch.Tensor],
    clip_f_scores: Optional[torch.Tensor],
    clip_v_scores: Optional[torch.Tensor],
) -> Optional[str]:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{make_safe_sample_name(sample.sample_id)}.csv"
    num_rows = 0
    for tensor in [clip_t_scores, clip_v_scores]:
        if tensor is not None:
            num_rows = max(num_rows, int(tensor.shape[0]))
    if clip_f_scores is not None:
        num_rows = max(num_rows, int(clip_f_scores.shape[0]) + 1)
    if num_rows == 0:
        num_rows = 1

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "frame_idx",
                "next_frame_idx",
                "clip_t_score",
                "clip_f_score_to_next",
                "clip_v_score",
                "prompt",
                "pred_path",
                "source_path",
                "gt_path",
            ],
        )
        writer.writeheader()
        for frame_idx in range(num_rows):
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "frame_idx": frame_idx,
                    "next_frame_idx": frame_idx + 1 if clip_f_scores is not None and frame_idx < clip_f_scores.shape[0] else "",
                    "clip_t_score": float(clip_t_scores[frame_idx].item()) if clip_t_scores is not None and frame_idx < clip_t_scores.shape[0] else "",
                    "clip_f_score_to_next": float(clip_f_scores[frame_idx].item()) if clip_f_scores is not None and frame_idx < clip_f_scores.shape[0] else "",
                    "clip_v_score": float(clip_v_scores[frame_idx].item()) if clip_v_scores is not None and frame_idx < clip_v_scores.shape[0] else "",
                    "prompt": prompt or "",
                    "pred_path": str(sample.pred_path),
                    "source_path": str(sample.source_path) if sample.source_path is not None else "",
                    "gt_path": str(sample.gt_path) if sample.gt_path is not None else "",
                }
            )
    return str(output_path)


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=self.weights, transform_input=False).to(device)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.device = device
        preprocess = self.weights.transforms()
        self.mean = torch.tensor(preprocess.mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(preprocess.std).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def forward(self, frames_nchw: torch.Tensor) -> torch.Tensor:
        frames_nchw = F.interpolate(frames_nchw, size=(299, 299), mode="bilinear", align_corners=False)
        frames_nchw = (frames_nchw - self.mean) / self.std
        out = self.model(frames_nchw)
        if hasattr(out, "logits"):
            out = out.logits
        return out


def compute_feature_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance_from_stats(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    cov_prod = sigma1 @ sigma2
    covmean, _ = scipy.linalg.sqrtm(cov_prod, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def ensure_metric_imports(common_metrics_root: Path):
    helper_path = common_metrics_root / "calculate_fvd.py"
    if not helper_path.is_file():
        raise FileNotFoundError(f"Missing vendored FVD helper: {helper_path}")

    package_name = "_track4d_vendor_common_metrics"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__file__ = str(common_metrics_root / "__init__.py")
        package.__path__ = [str(common_metrics_root)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package

    fvd_package = sys.modules.get("fvd")
    if fvd_package is None:
        fvd_package = types.ModuleType("fvd")
        fvd_package.__file__ = str(common_metrics_root / "fvd" / "__init__.py")
        fvd_package.__path__ = [str(common_metrics_root / "fvd")]  # type: ignore[attr-defined]
        fvd_package.__package__ = "fvd"
        sys.modules["fvd"] = fvd_package

    module = importlib.import_module(f"{package_name}.calculate_fvd")
    return module.calculate_fvd


def ensure_evo_imports():
    from evo.core import sync  # type: ignore
    from evo.core.metrics import PoseRelation, Unit  # type: ignore
    from evo.core.trajectory import PosePath3D, PoseTrajectory3D  # type: ignore
    import evo.main_ape as main_ape  # type: ignore
    import evo.main_rpe as main_rpe  # type: ignore
    from evo.tools import file_interface  # type: ignore
    from scipy.spatial.transform import Rotation  # type: ignore

    return {
        "sync": sync,
        "PoseRelation": PoseRelation,
        "Unit": Unit,
        "PosePath3D": PosePath3D,
        "PoseTrajectory3D": PoseTrajectory3D,
        "main_ape": main_ape,
        "main_rpe": main_rpe,
        "file_interface": file_interface,
        "Rotation": Rotation,
    }


def read_sintel_camera_file(filename: Path) -> np.ndarray:
    tag_float = 202021.25
    with open(filename, "rb") as handle:
        check = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if check != tag_float:
            raise ValueError(f"Wrong Sintel camera tag in {filename}: expected {tag_float}, got {check}")
        _intrinsics = np.fromfile(handle, dtype=np.float64, count=9).reshape((3, 3))
        extrinsics = np.fromfile(handle, dtype=np.float64, count=12).reshape((3, 4))
    return extrinsics


def load_sintel_traj_local(gt_traj_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    evo = ensure_evo_imports()
    rotation_cls = evo["Rotation"]

    gt_pose_files = sorted(gt_traj_dir.iterdir())
    timestamps = [float(path.stem.split("_")[-1]) for path in gt_pose_files]
    tum_poses = []
    for pose_file in gt_pose_files:
        gt_pose = read_sintel_camera_file(pose_file)
        gt_pose = np.concatenate([gt_pose, np.array([[0, 0, 0, 1]], dtype=np.float64)], axis=0)
        gt_pose_inv = np.linalg.inv(gt_pose)
        xyz = gt_pose_inv[:3, 3]
        xyzw = rotation_cls.from_matrix(gt_pose_inv[:3, :3]).as_quat()
        wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
        tum_poses.append(np.concatenate([xyz, wxyz], axis=0))

    traj = np.stack(tum_poses, axis=0)
    traj[:, :3] = traj[:, :3] - np.mean(traj[:, :3], axis=0, keepdims=True)
    return traj, np.asarray(timestamps, dtype=np.float64)


def load_tum_like_traj_local(
    traj_file: Path,
    traj_format: str,
    skip: int = 0,
    end: int = -1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    evo = ensure_evo_imports()
    file_interface = evo["file_interface"]

    traj = file_interface.read_tum_trajectory_file(str(traj_file))
    xyz = traj.positions_xyz
    quat = traj.orientations_quat_wxyz
    timestamps = traj.timestamps

    if traj_format == "tartanair":
        xyz = xyz[:, [1, 2, 0]]
        quat = quat[:, [0, 2, 3, 1]]
    elif traj_format != "tum":
        raise NotImplementedError(f"Unsupported trajectory format: {traj_format}")

    traj_wxyz = np.column_stack((xyz, quat))
    if end == -1:
        end = traj_wxyz.shape[0]
    return traj_wxyz[skip:end:stride], timestamps[skip:end:stride]


def make_pose_trajectory(traj_wxyz: np.ndarray, timestamps: np.ndarray):
    evo = ensure_evo_imports()
    pose_trajectory_cls = evo["PoseTrajectory3D"]
    return pose_trajectory_cls(
        positions_xyz=traj_wxyz[:, :3],
        orientations_quat_wxyz=traj_wxyz[:, 3:],
        timestamps=timestamps,
    )


def compute_pose_errors(
    pred_traj_path: Path,
    gt_traj_path: Path,
    gt_traj_format: str,
    skip: int,
    end: int,
    stride: int,
    report_path: Optional[Path],
) -> Dict[str, float]:
    evo = ensure_evo_imports()
    sync = evo["sync"]
    pose_relation = evo["PoseRelation"]
    unit = evo["Unit"]
    main_ape = evo["main_ape"]
    main_rpe = evo["main_rpe"]

    pred_traj_arr, pred_timestamps = load_tum_like_traj_local(pred_traj_path, "tum")
    if gt_traj_format == "sintel":
        gt_traj_arr, gt_timestamps = load_sintel_traj_local(gt_traj_path)
        if end != -1:
            gt_traj_arr = gt_traj_arr[skip:end:stride]
            gt_timestamps = gt_timestamps[skip:end:stride]
        else:
            gt_traj_arr = gt_traj_arr[skip::stride]
            gt_timestamps = gt_timestamps[skip::stride]
    else:
        gt_traj_arr, gt_timestamps = load_tum_like_traj_local(
            gt_traj_path,
            gt_traj_format,
            skip=skip,
            end=end,
            stride=stride,
        )

    pred_traj = make_pose_trajectory(pred_traj_arr, pred_timestamps)
    gt_traj = make_pose_trajectory(gt_traj_arr, gt_timestamps)

    if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
        pred_traj.timestamps = gt_traj.timestamps

    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    ate_result = main_ape.ape(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=pose_relation.translation_part,
        align=True,
        correct_scale=True,
    )
    rpe_rot_result = main_rpe.rpe(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=pose_relation.rotation_angle_deg,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    rpe_trans_result = main_rpe.rpe(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=pose_relation.translation_part,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(f"{ate_result}")
            handle.write(f"{rpe_rot_result}")
            handle.write(f"{rpe_trans_result}")

    return {
        "ate": float(ate_result.stats["rmse"]),
        "trans_err": float(rpe_trans_result.stats["rmse"]),
        "rot_err": float(rpe_rot_result.stats["rmse"]),
        "num_aligned_poses": int(gt_traj.num_poses),
    }


def resolve_eval_size(samples: Sequence[EvalSample], cache: VideoCache, args: argparse.Namespace) -> Tuple[int, int]:
    if args.eval_height is not None and args.eval_width is not None:
        return int(args.eval_height), int(args.eval_width)
    first_video = cache.get(samples[0].pred_path)
    if first_video is None:
        raise ValueError("Unable to resolve evaluation size from the first prediction.")
    return int(first_video.shape[-2]), int(first_video.shape[-1])


def resolve_global_pair_frames(samples: Sequence[EvalSample], cache: VideoCache, max_frames: Optional[int]) -> int:
    frame_counts = []
    for sample in samples:
        pred = cache.get(sample.pred_path)
        gt = cache.get(sample.gt_path)
        if pred is None or gt is None:
            continue
        frame_counts.append(min(pred.shape[0], gt.shape[0]))
    if not frame_counts:
        raise ValueError("No samples with both prediction and GT videos were found.")
    frames = min(frame_counts)
    if max_frames is not None:
        frames = min(frames, max_frames)
    return frames


def build_pair_batches(
    samples: Sequence[EvalSample],
    cache: VideoCache,
    size_hw: Tuple[int, int],
    max_frames: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    usable = [sample for sample in samples if sample.gt_path is not None]
    if not usable:
        raise ValueError("FID/FVD need prediction and GT videos, but none were found.")
    frames = resolve_global_pair_frames(usable, cache, max_frames)
    pred_videos = []
    gt_videos = []
    ids = []
    for sample in usable:
        pred = resize_video(truncate_video(cache.get(sample.pred_path), frames), size_hw)
        gt = resize_video(truncate_video(cache.get(sample.gt_path), frames), size_hw)
        pred_videos.append(pred)
        gt_videos.append(gt)
        ids.append(sample.sample_id)
    return torch.stack(pred_videos, dim=0), torch.stack(gt_videos, dim=0), ids


def compute_fid(
    pred_videos_btchw: torch.Tensor,
    gt_videos_btchw: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    extractor = InceptionFeatureExtractor(device)
    pred_frames = pred_videos_btchw.reshape(-1, *pred_videos_btchw.shape[2:])
    gt_frames = gt_videos_btchw.reshape(-1, *gt_videos_btchw.shape[2:])

    def collect(frames_nchw: torch.Tensor) -> np.ndarray:
        feats = []
        for start in range(0, frames_nchw.shape[0], batch_size):
            batch = frames_nchw[start : start + batch_size].to(device)
            feats.append(extractor(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    pred_feats = collect(pred_frames)
    gt_feats = collect(gt_frames)
    mu1, sigma1 = compute_feature_stats(pred_feats)
    mu2, sigma2 = compute_feature_stats(gt_feats)
    value = frechet_distance_from_stats(mu1, sigma1, mu2, sigma2)
    return {
        "value": value,
        "num_videos": int(pred_videos_btchw.shape[0]),
        "num_frames_per_video": int(pred_videos_btchw.shape[1]),
        "num_total_frames": int(pred_feats.shape[0]),
    }


def compute_clip_metrics(
    samples: Sequence[EvalSample],
    cache: VideoCache,
    embedder: OpenClipEmbedder,
    batch_size: int,
    clip_detail_dir: Optional[Path],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    per_sample_rows = []
    clip_t_scores: List[float] = []
    clip_f_scores: List[float] = []
    clip_v_scores: List[float] = []

    for sample in samples:
        pred = cache.get(sample.pred_path)
        if pred is None:
            continue
        pred_feats = embedder.encode_images(pred, batch_size=batch_size)
        row: Dict[str, object] = {
            "sample_id": sample.sample_id,
            "pred_path": str(sample.pred_path),
            "source_path": str(sample.source_path) if sample.source_path is not None else None,
            "gt_path": str(sample.gt_path) if sample.gt_path is not None else None,
            "group_id": sample.group_id,
            "view_id": sample.view_id,
            "prompt": sample.prompt,
            "num_frames": int(pred.shape[0]),
            "prompt_available": sample.prompt is not None,
            "source_available": sample.source_path is not None,
            "gt_available": sample.gt_path is not None,
        }

        text_scores: Optional[torch.Tensor] = None
        frame_scores: Optional[torch.Tensor] = None
        view_scores: Optional[torch.Tensor] = None

        if sample.prompt:
            text_feat = embedder.encode_text(sample.prompt)
            text_scores = torch.matmul(pred_feats, text_feat)
            clip_t_scores.extend(text_scores.tolist())
            row["clip_t"] = float(text_scores.mean().item())
        else:
            row["clip_t"] = None

        if pred_feats.shape[0] >= 2:
            frame_scores = torch.sum(pred_feats[:-1] * pred_feats[1:], dim=-1)
            clip_f_scores.extend(frame_scores.tolist())
            row["clip_f"] = float(frame_scores.mean().item())
        else:
            row["clip_f"] = None

        source = cache.get(sample.source_path)
        if source is not None:
            paired_frames = min(pred.shape[0], source.shape[0])
            source_feats = embedder.encode_images(source[:paired_frames], batch_size=batch_size)
            pred_feats_trimmed = pred_feats[:paired_frames]
            view_scores = torch.sum(source_feats * pred_feats_trimmed, dim=-1)
            clip_v_scores.extend(view_scores.tolist())
            row["clip_v"] = float(view_scores.mean().item())
        else:
            row["clip_v"] = None

        row["clip_detail_csv"] = write_clip_detail_csv(
            output_dir=clip_detail_dir,
            sample=sample,
            prompt=sample.prompt,
            clip_t_scores=text_scores,
            clip_f_scores=frame_scores,
            clip_v_scores=view_scores,
        )

        per_sample_rows.append(row)

    def summarize(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {
            "value": float(arr.mean()),
            "value_std": float(arr.std()),
            "count": int(arr.size),
        }

    metrics = {
        "clip_t": summarize(clip_t_scores),
        "clip_f": summarize(clip_f_scores),
        "clip_v": summarize(clip_v_scores),
    }
    return metrics, per_sample_rows


def compute_fvd(
    pred_videos_btchw: torch.Tensor,
    gt_videos_btchw: torch.Tensor,
    device: torch.device,
    method: str,
    common_metrics_root: Path,
) -> Dict[str, float]:
    calculate_fvd = ensure_metric_imports(common_metrics_root)
    result = calculate_fvd(pred_videos_btchw, gt_videos_btchw, device=device, method=method, only_final=True)
    return {
        "value": float(result["value"][0]),
        "num_videos": int(pred_videos_btchw.shape[0]),
        "num_frames_per_video": int(pred_videos_btchw.shape[1]),
    }


def compute_fvd_v(
    samples: Sequence[EvalSample],
    cache: VideoCache,
    size_hw: Tuple[int, int],
    max_frames: Optional[int],
    device: torch.device,
    method: str,
    common_metrics_root: Path,
) -> Optional[Dict[str, float]]:
    groups: Dict[str, List[EvalSample]] = defaultdict(list)
    for sample in samples:
        if sample.group_id is not None and sample.gt_path is not None and sample.source_path is not None:
            groups[sample.group_id].append(sample)
    if not groups:
        return None

    common_view_ids: Optional[set] = None
    for group_samples in groups.values():
        view_ids = {sample.view_id for sample in group_samples if sample.view_id is not None and not sample.view_id.startswith("official_camtype")}
        common_view_ids = view_ids if common_view_ids is None else (common_view_ids & view_ids)
    if not common_view_ids:
        return None
    ordered_views = sorted(common_view_ids, key=lambda value: int(value))
    num_views = len(ordered_views) + 1
    if num_views < 10:
        return {
            "value": None,
            "error": f"FVD-V needs at least 10 synchronized views for the borrowed FVD implementation; found {num_views}.",
        }

    pred_view_videos = []
    gt_view_videos = []
    for group_id, group_samples in groups.items():
        by_view = {sample.view_id: sample for sample in group_samples}
        if any(view_id not in by_view for view_id in ordered_views):
            continue
        source = cache.get(group_samples[0].source_path)
        if source is None:
            continue
        pred_views = [source]
        gt_views = [source]
        for view_id in ordered_views:
            sample = by_view[view_id]
            pred = cache.get(sample.pred_path)
            gt = cache.get(sample.gt_path)
            if pred is None or gt is None:
                pred_views = []
                gt_views = []
                break
            pred_views.append(pred)
            gt_views.append(gt)
        if not pred_views or not gt_views:
            continue
        group_frames = min(video.shape[0] for video in pred_views + gt_views)
        if max_frames is not None:
            group_frames = min(group_frames, max_frames)
        resized_pred_views = [resize_video(truncate_video(video, group_frames), size_hw) for video in pred_views]
        resized_gt_views = [resize_video(truncate_video(video, group_frames), size_hw) for video in gt_views]
        for frame_idx in range(group_frames):
            pred_view_videos.append(torch.stack([video[frame_idx] for video in resized_pred_views], dim=0))
            gt_view_videos.append(torch.stack([video[frame_idx] for video in resized_gt_views], dim=0))

    if not pred_view_videos:
        return None
    pred_batch = torch.stack(pred_view_videos, dim=0)
    gt_batch = torch.stack(gt_view_videos, dim=0)
    calculate_fvd = ensure_metric_imports(common_metrics_root)
    result = calculate_fvd(pred_batch, gt_batch, device=device, method=method, only_final=True)
    return {
        "value": float(result["value"][0]),
        "num_groups": int(len(groups)),
        "num_view_videos": int(pred_batch.shape[0]),
        "num_views_per_clip": int(pred_batch.shape[1]),
    }


def write_json(path: Optional[str], payload: Dict[str, object]) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_per_sample_csv(path: Optional[str], rows: Sequence[Dict[str, object]]) -> None:
    if path is None or not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    metrics = parse_metrics(args.metrics, args.mode)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    samples = build_samples(args)
    cache = VideoCache()

    summary: Dict[str, object] = {
        "mode": args.mode,
        "pred_root": str(Path(args.pred_root).resolve()) if args.pred_root else None,
        "num_samples": len(samples),
        "metrics_requested": metrics,
        "device": str(device),
        "clip_backend": args.clip_backend,
    }

    per_sample_rows: List[Dict[str, object]] = []
    size_hw = resolve_eval_size(samples, cache, args)
    common_metrics_root = Path(args.common_metrics_root)

    if any(metric in metrics for metric in ["clip_t", "clip_f", "clip_v"]):
        embedder = OpenClipEmbedder(
            device=device,
            model_name=args.clip_model_name,
            pretrained=args.clip_pretrained,
            clip_bpe_path=args.clip_bpe_path,
            backend=args.clip_backend,
        )
        clip_detail_dir = Path(args.clip_detail_dir) if args.clip_detail_dir else None
        clip_metrics, per_sample_rows = compute_clip_metrics(
            samples,
            cache,
            embedder,
            batch_size=args.batch_size,
            clip_detail_dir=clip_detail_dir,
        )
        for metric_name in ["clip_t", "clip_f", "clip_v"]:
            if metric_name in metrics:
                if clip_metrics[metric_name] is None:
                    raise ValueError(f"{metric_name} was requested but required inputs were missing.")
                summary[metric_name] = clip_metrics[metric_name]

    if "fvd" in metrics:
        pred_videos, gt_videos, used_ids = build_pair_batches(samples, cache, size_hw=size_hw, max_frames=args.max_frames)
        summary["paired_eval_samples"] = used_ids
        if pred_videos.shape[1] < 10:
            raise ValueError(f"FVD requires at least 10 frames per video; found {pred_videos.shape[1]}.")
        summary["fvd"] = compute_fvd(
            pred_videos,
            gt_videos,
            device=device,
            method=args.fvd_method,
            common_metrics_root=common_metrics_root,
        )

    if "fvd_v" in metrics:
        result = compute_fvd_v(
            samples=samples,
            cache=cache,
            size_hw=size_hw,
            max_frames=args.max_frames,
            device=device,
            method=args.fvd_method,
            common_metrics_root=common_metrics_root,
        )
        if result is None:
            raise ValueError("FVD-V was requested but synchronized multiview prediction/GT groups were not found.")
        summary["fvd_v"] = result

    if any(metric in metrics for metric in ["rot_err", "trans_err"]):
        if args.pred_traj is None or args.gt_traj is None:
            raise ValueError("rot_err/trans_err require --pred_traj and --gt_traj.")
        pose_metrics = compute_pose_errors(
            pred_traj_path=Path(args.pred_traj),
            gt_traj_path=Path(args.gt_traj),
            gt_traj_format=args.gt_traj_format,
            skip=args.traj_skip,
            end=args.traj_end,
            stride=args.traj_stride,
            report_path=Path(args.pose_report) if args.pose_report else None,
        )
        summary["pose_alignment"] = {
            "pred_traj": str(Path(args.pred_traj).resolve()),
            "gt_traj": str(Path(args.gt_traj).resolve()),
            "gt_traj_format": args.gt_traj_format,
            "traj_skip": int(args.traj_skip),
            "traj_end": int(args.traj_end),
            "traj_stride": int(args.traj_stride),
            "ate": pose_metrics["ate"],
            "num_aligned_poses": pose_metrics["num_aligned_poses"],
        }
        if "trans_err" in metrics:
            summary["trans_err"] = {
                "value": pose_metrics["trans_err"],
                "unit": "translation RMSE after align+scale correction, delta=1 frame",
            }
        if "rot_err" in metrics:
            summary["rot_err"] = {
                "value": pose_metrics["rot_err"],
                "unit": "degrees RMSE after align+scale correction, delta=1 frame",
            }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    write_json(args.output_json, summary)
    write_per_sample_csv(args.per_sample_csv, per_sample_rows)


if __name__ == "__main__":
    main()
