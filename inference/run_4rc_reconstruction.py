#!/usr/bin/env python3
"""Run upstream 4RC on one staged image sequence and save its dense 3D outputs.

This small adapter calls the public Arc / dust3r inference API and writes the
NPZ/metadata contract consumed by Holo4D's source-condition loader. It does not
copy or modify the separately installed 4RC checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def flatten_output_dict(output_dict: dict) -> dict[str, np.ndarray]:
    flattened: dict[str, np.ndarray] = {"n_frames": np.asarray(len(output_dict["preds"]), dtype=np.int32)}
    for group_name in ("preds", "views"):
        for index, item in enumerate(output_dict[group_name]):
            for key, value in item.items():
                array = to_numpy(value)
                if array is not None:
                    flattened[f"{group_name[:-1]}_{index}_{key}"] = array
    query = output_dict.get("track_query_img_uint8")
    if query is not None:
        flattened["track_query_img_uint8"] = np.asarray(query)
    refine = output_dict.get("refine_track_visual", False)
    flattened["refine_track_visual"] = np.asarray(bool(refine))
    return flattened


def atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_default(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def save_reconstruction_outputs(output_dir: Path, payload: dict, metadata: dict) -> None:
    # Serialize first; unsupported metadata must never leave a published NPZ.
    encoded = json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + '\n'
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite existing 4RC output under {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fourrc-transaction-', dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        (staging/'metadata.json').write_text(encoded)
        atomic_save_npz(staging/'result.npz', payload)
        staging.rename(output_dir)


def git_revision(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def checkpoint_fingerprint(checkpoint: str) -> dict:
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        candidate = path / "model.safetensors"
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        return {"reference": checkpoint}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fourrc-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", default="Luo-Yihang/4RC")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--dtype", choices=("32", "16-mixed", "bf16-mixed", "bf16-mixed-no-grad-scaling"), default="bf16-mixed")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frames_dir = args.frames_dir.expanduser().resolve()
    fourrc_repo = args.fourrc_repo.expanduser().resolve()
    paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != 49:
        raise ValueError(f"4RC adapter requires exactly 49 staged frames, found {len(paths)} under {frames_dir}")
    if not (0 <= args.query_index < len(paths)):
        raise ValueError(f"query index {args.query_index} is outside [0, 48]")
    if not fourrc_repo.is_dir():
        raise FileNotFoundError(f"4RC repository not found: {fourrc_repo}")
    if str(fourrc_repo) not in sys.path:
        sys.path.insert(0, str(fourrc_repo))

    import torch
    from arc.dust3r.inference_multiview import inference
    from arc.dust3r.utils.image import load_images
    from arc.models.arc.arc import Arc

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    images = load_images([str(path) for path in paths], size=args.size, patch_size=14, verbose=True)
    query_tensor = torch.tensor([args.query_index])
    for image in images:
        image["track_query_idx"] = query_tensor

    model = Arc.from_pretrained(args.checkpoint).to(device).eval()
    started = time.perf_counter()
    with torch.no_grad():
        output_dict, profiling = inference(
            images,
            model,
            device,
            dtype=args.dtype,
            profiling=True,
            verbose=True,
            use_center_as_anchor=False,
        )
    elapsed = time.perf_counter() - started
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / "result.npz"
    metadata_path = output_dir / "metadata.json"
    if result_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing 4RC output under {output_dir}")
    metadata = {
        "dataset": "Holo4D raw-video source",
        "sequence_name": frames_dir.name,
        "num_frames": len(paths),
        "track_query_idx": args.query_index,
        "checkpoint": checkpoint_fingerprint(args.checkpoint),
        "fourrc_repo": str(fourrc_repo),
        "fourrc_repo_revision": git_revision(fourrc_repo),
        "size": args.size,
        "dtype": args.dtype,
        "device": str(device),
        "profiling": {**profiling, "holo_adapter_elapsed_seconds": elapsed},
        "source_frame_paths": [str(path) for path in paths],
    }
    save_reconstruction_outputs(output_dir, flatten_output_dict(output_dict), metadata)
    print(f"[4RC] wrote {result_path} and {metadata_path}; inference={elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
