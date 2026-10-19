"""Disk-backed prompt-embedding cache used by Track4D-360 evaluation.

The cache deliberately maps the *exact prompt string* to its Wan UMT5
embedding.  It is dataset-agnostic: Vista4D, DAVIS, and future evaluation
suites can share the same loader while keeping separate provenance manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors


CACHE_FILENAME = "prompt_embeddings.safetensors"
MANIFEST_FILENAME = "prompt_embeddings_manifest.json"


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def tensor_key(prompt: str) -> str:
    return f"prompt__{prompt_sha256(prompt)}"


def save_eval_prompt_cache(
    output_dir: str | Path,
    prompt_to_embedding: dict[str, torch.Tensor],
    *,
    provenance: dict | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    records = []
    for prompt in sorted(prompt_to_embedding):
        embedding = prompt_to_embedding[prompt]
        if embedding.ndim == 3 and embedding.shape[0] == 1:
            embedding = embedding[0]
        if embedding.ndim != 2:
            raise ValueError(
                f"prompt embedding must be [tokens, dim], got {tuple(embedding.shape)}"
            )
        key = tensor_key(prompt)
        tensors[key] = embedding.to(dtype=torch.bfloat16, device="cpu").contiguous()
        records.append(
            {
                "prompt": prompt,
                "prompt_sha256": prompt_sha256(prompt),
                "tensor_key": key,
                "shape": list(tensors[key].shape),
                "dtype": str(tensors[key].dtype),
            }
        )

    cache_path = output_dir / CACHE_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    save_safetensors(tensors, str(cache_path))
    manifest = {
        "format": "track4d360_eval_prompt_t5_v1",
        "num_prompts": len(records),
        "records": records,
        "provenance": provenance or {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return cache_path, manifest_path


def load_eval_prompt_cache(
    cache_dir: str | Path,
    *,
    expected_prompts: list[str] | tuple[str, ...] | None = None,
    strict: bool = True,
) -> dict[str, torch.Tensor]:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / CACHE_FILENAME
    manifest_path = cache_dir / MANIFEST_FILENAME
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"eval T5 cache requires {cache_path} and {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "track4d360_eval_prompt_t5_v1":
        raise ValueError(f"unsupported eval T5 cache format in {manifest_path}")
    tensors = load_safetensors(str(cache_path), device="cpu")
    table = {}
    for record in manifest.get("records", []):
        prompt = str(record["prompt"])
        if record.get("prompt_sha256") != prompt_sha256(prompt):
            raise ValueError(f"prompt hash mismatch in {manifest_path}: {prompt!r}")
        key = str(record["tensor_key"])
        if key not in tensors:
            raise KeyError(f"missing tensor {key!r} in {cache_path}")
        table[prompt] = tensors[key]

    expected = {str(prompt) for prompt in (expected_prompts or [])}
    missing = sorted(expected - set(table))
    if strict and missing:
        preview = "\n".join(f"  - {prompt}" for prompt in missing[:10])
        raise RuntimeError(
            f"eval T5 cache misses {len(missing)} required prompt(s):\n{preview}"
        )
    return table
