"""Pre-compute Wan umt5-xxl T5 prompt embeddings for Track4D-360 caption stores.

Walks every caption directory passed via --caption_dirs, reads its
`captions_all.json` or merged `captions_rank*.json`, encodes each unique
caption once with the Wan text encoder, and writes a single
`captions_t5_emb.safetensors` (plus `captions_t5_emb_meta.json`) next to
the JSON files. Training-time loader is in
`holo4d_runtime/data/stream3r/dust3r/datasets_cut3r/mixed_nvs_captions.py`
(class `CachedT5EmbStore`).

Why this layout:
- Wan 1.3B and 14B share the SAME `models_t5_umt5-xxl-enc-bf16.safetensors`
  (verified: weights/wan21-14b-controlCam/ ships only the DiT). One cache
  serves both runs.
- `ChunkedCaptionStore` already merges all rank shards in-memory, so a
  single per-dir embedding file matches the lookup pattern (sequence_id
  + frame_indices -> key -> tensor).
- Tensors are bf16, trimmed to the actual non-pad token count. Disk size
  scales with mean caption length, not max_seq_len.

Launch:
  torchrun --nproc_per_node=2 \
      holo4d/precompute_t5_cache.py \
      --caption_dirs <dir1> <dir2> ...

Two-GPU sharding splits unique caption texts across ranks; each rank writes
a partial; rank 0 merges into the final per-dir file and removes partials.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save


# --- import from this repo without invoking the full pipeline ---------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# `holo4d_runtime.models.wan_video_text_encoder` pulls diffusers / transformers
# at import time which can fail when CUDA_HOME isn't set (e.g. in unit-test
# environments that exercise the helpers below without ever loading T5).
# Defer that import to ``load_text_encoder`` so this module is importable
# without a CUDA toolchain.
#
# `mixed_nvs_captions` is tiny and pure-Python (json + path utilities only),
# so we import its helpers directly via importlib to avoid going through
# `holo4d_runtime/data/__init__.py` which has its own optional-dep chain.
import importlib.util as _importlib_util  # noqa: E402

_CAPTIONS_PATH = (
    _REPO / "holo4d_runtime" / "data" / "stream3r" / "dust3r" / "datasets_cut3r"
    / "mixed_nvs_captions.py"
)
_caption_spec = _importlib_util.spec_from_file_location(
    "_track4d_caption_utils", _CAPTIONS_PATH
)
_caption_mod = _importlib_util.module_from_spec(_caption_spec)
_caption_spec.loader.exec_module(_caption_mod)
load_chunked_captions = _caption_mod.load_chunked_captions
sanitize_mixed_nvs_caption = _caption_mod.sanitize_mixed_nvs_caption


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--caption_dirs",
        nargs="+",
        required=True,
        help="One or more caption directories containing captions_all.json or "
             "captions_rank*.json. One safetensors will be written per dir.",
    )
    p.add_argument(
        "--t5_path",
        default=str(_REPO / "weights/wan21-1p3b-run/models_t5_umt5-xxl-enc-bf16.safetensors"),
        help="Path to the Wan T5 safetensors. Default: shared 1.3B/14B umt5-xxl bf16.",
    )
    p.add_argument(
        "--tokenizer_path",
        default=str(_REPO / "weights/wan21-1p3b-run/google/umt5-xxl"),
        help="Path to the umt5-xxl tokenizer directory.",
    )
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if captions_t5_emb.safetensors already exists.",
    )
    p.add_argument(
        "--default_prompt",
        default="A multi-view video scene.",
        help="The DEFAULT_PROMPT used by datasets when no chunk caption matches. "
             "We FILTER this out of the encoding set: any caption equal to this "
             "string after sanitization is skipped. Pass empty string to disable "
             "the filter.",
    )
    p.add_argument(
        "--max_print_chars",
        type=int,
        default=120,
        help="Truncate per-caption stdout printout to this many characters.",
    )
    p.add_argument(
        "--limit_captions",
        type=int,
        default=0,
        help="Dry-run guard: take only the first N json keys per caption_dir. "
             "Output filename is rewritten with a `.dryrun` infix "
             "(captions_t5_emb.dryrun.safetensors) so the real cache slot "
             "is NOT clobbered. Use this to validate the full encode→shard→"
             "merge→write→load roundtrip on a tiny sample (e.g. "
             "--limit_captions 100) before kicking off a multi-hour DL3DV "
             "run. Set to 0 (default) for a real run.",
    )
    return p.parse_args()


def init_dist():
    """Initialize torch.distributed if launched via torchrun; else single-rank fallback."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def load_text_encoder(t5_path, tokenizer_path, device):
    """Load only the T5 encoder + tokenizer (no VAE/DiT).

    Imports holo4d_runtime.models.* lazily here so this script's top-level import
    works in unit-test environments without CUDA_HOME (the diffusers/PEFT
    chain compiled by transformers' modeling_utils requires CUDA_HOME).
    """
    from holo4d_runtime.models.wan_video_text_encoder import (  # noqa: E402
        HuggingfaceTokenizer,
        WanTextEncoder,
    )

    print(f"[rank {os.environ.get('RANK', '0')}] Loading T5 from {t5_path}", flush=True)
    encoder = WanTextEncoder()
    state = safetensors_load(t5_path)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Wan T5 safetensors are exact-match; warn loudly if not.
        print(f"[T5 load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean="whitespace")
    return encoder, tokenizer


def encode_batch(encoder, tokenizer, texts, device):
    """Tokenize + encode a list of caption strings. Returns list of [L, dim] bf16 CPU tensors,
    one per input, where L is the actual non-pad token count for that caption."""
    ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        emb = encoder(ids, mask)  # [B, max_len, dim]
    out = []
    seq_lens_cpu = seq_lens.cpu().tolist()
    for i, L in enumerate(seq_lens_cpu):
        L = max(int(L), 1)  # guard against pathological all-pad input
        out.append(emb[i, :L].to(dtype=torch.bfloat16, device="cpu").contiguous().clone())
    return out


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def collect_jobs(captions_raw, default_prompt):
    """Build:
      key_to_text: dict {json_key: sanitized_text}
      unique_texts: sorted list of unique sanitized texts (excludes default_prompt)
      text_to_keys: dict {sanitized_text: [json_key, ...]}
      skipped_default: int — count of json keys whose caption sanitized to default_prompt

    `captions_raw` is the merged dict from `load_chunked_captions(caption_dir)`.
    Captions equal to `default_prompt` are dropped — they don't need encoding.
    Sample fallback at training time will route those through whatever default-emb
    mechanism the trainer provides (or back to online T5 for that small fraction).
    """
    default_norm = (default_prompt or "").strip()
    key_to_text = {}
    skipped_default = 0
    skipped_empty = 0
    for k, raw in captions_raw.items():
        text = sanitize_mixed_nvs_caption(raw)
        if not text:
            skipped_empty += 1
            continue
        if default_norm and text.strip() == default_norm:
            skipped_default += 1
            continue
        key_to_text[k] = text

    text_to_keys = {}
    for k, t in key_to_text.items():
        text_to_keys.setdefault(t, []).append(k)
    unique_texts = sorted(text_to_keys.keys())
    return key_to_text, unique_texts, text_to_keys, skipped_default, skipped_empty


def discover_caption_jobs(caption_dir):
    """Yield (output_safetensors_path, label, captions_dict) for every caption
    JSON file in this directory. One job = one output file, mirroring source layout.

    Two layouts are supported:

    A) cut3r format — captions_all.json (or captions_rank*.json shards merged).
       Flat dict {chunk_key: caption_text}. Emits ONE job for the whole dir,
       writing to ``captions_t5_emb.safetensors``.
    B) Syn4D format — one ``{scene}_per_video_chunks.json`` per scene.
       Nested dict {video_path: {chunk_range: caption_text}} or
       {video_path: caption_text}. Emits ONE job PER scene file, writing to
       ``{scene}_per_video_chunks_t5_emb.safetensors``. Keys in the output
       safetensors are flattened to ``{video_path}/{chunk_range}`` (or just
       ``{video_path}`` when the value is a string).
    """
    caption_dir = Path(caption_dir)

    cut3r_files = sorted(caption_dir.glob("captions_all.json")) + sorted(
        caption_dir.glob("captions_rank*.json")
    )
    syn4d_files = sorted(caption_dir.glob("*_per_video_chunks.json"))

    if cut3r_files:
        merged = {}
        for f in cut3r_files:
            with f.open("r", encoding="utf-8") as fh:
                merged.update(json.load(fh))
        yield (
            caption_dir / "captions_t5_emb.safetensors",
            f"cut3r:{caption_dir.name}",
            merged,
        )
        return

    if syn4d_files:
        for f in syn4d_files:
            scene = f.stem[: -len("_per_video_chunks")] if f.stem.endswith(
                "_per_video_chunks"
            ) else f.stem
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            flat = {}
            for video_path, chunks in data.items():
                if isinstance(chunks, dict):
                    for chunk_range, text in chunks.items():
                        flat[f"{video_path}/{chunk_range}"] = text
                elif isinstance(chunks, str):
                    flat[video_path] = chunks
            out_path = caption_dir / f"{scene}_per_video_chunks_t5_emb.safetensors"
            yield (out_path, f"syn4d:{scene}", flat)
        return


def process_caption_dir(args, caption_dir, encoder, tokenizer, rank, world_size, device):
    caption_dir = Path(caption_dir)
    if rank == 0:
        print(f"\n=== {caption_dir} ===", flush=True)

    jobs = list(discover_caption_jobs(caption_dir))
    if not jobs:
        if rank == 0:
            print(f"[warn] no captions found under {caption_dir}; skipping", flush=True)
        return

    if rank == 0:
        print(f"[layout] {len(jobs)} output safetensors will be written", flush=True)

    for final_path, label, captions_raw in jobs:
        process_one_job(
            args, caption_dir, final_path, label, captions_raw,
            encoder, tokenizer, rank, world_size, device,
        )


def process_one_job(
    args, caption_dir, final_path, label, captions_raw,
    encoder, tokenizer, rank, world_size, device,
):
    """Encode + save one safetensors file, sharded across ranks via partials."""
    final_path = Path(final_path)

    # Dry-run truncation: keep only the first N json keys per job, AND retarget
    # the output to `<stem>.dryrun.safetensors` so the real cache slot isn't
    # taken by a partial encode. Same code path as the real run — exercises
    # collect_jobs / shard / encode / barrier / merge / write / meta.
    limit = int(getattr(args, "limit_captions", 0) or 0)
    if limit > 0 and len(captions_raw) > limit:
        # `dict` preserves insertion order on py3.7+, so taking the first N
        # items is deterministic across ranks (each rank reads the same json
        # off the same filesystem and sees identical ordering).
        captions_raw = dict(list(captions_raw.items())[:limit])
        # Retarget output: foo.safetensors -> foo.dryrun.safetensors.
        final_path = final_path.with_name(f"{final_path.stem}.dryrun.safetensors")
        if rank == 0:
            print(
                f"[dryrun:{label}] limit_captions={limit} → "
                f"writing to {final_path.name} (NOT the real cache slot)",
                flush=True,
            )

    stem = final_path.stem  # e.g. "captions_t5_emb" or "arena_per_video_chunks_t5_emb"
    # Loader (CachedT5EmbStore) expects "<stem>_meta.json" (underscore, not dot)
    # next to the safetensors. Stay consistent with that convention.
    meta_path = final_path.parent / f"{stem}_meta.json"
    partial_path = caption_dir / f".{stem}.part{rank}.safetensors"
    partial_keys_path = caption_dir / f".{stem}.part{rank}.json"

    if final_path.exists() and not args.overwrite:
        if rank == 0:
            print(f"[skip:{label}] {final_path.name} exists (use --overwrite to recompute)", flush=True)
        return

    if not captions_raw:
        if rank == 0:
            print(f"[warn:{label}] empty caption set; skipping", flush=True)
        return

    key_to_text, unique_texts, text_to_keys, skipped_default, skipped_empty = collect_jobs(
        captions_raw, args.default_prompt
    )
    if rank == 0:
        print(
            f"[stats] {len(captions_raw)} json entries, "
            f"{len(key_to_text)} kept, "
            f"{skipped_default} skipped == default '{args.default_prompt}', "
            f"{skipped_empty} skipped empty, "
            f"{len(unique_texts)} unique non-default texts to encode",
            flush=True,
        )

    # Hard-fail safety: the default prompt MUST NOT enter the encoding set.
    default_norm = (args.default_prompt or "").strip()
    if default_norm:
        for t in unique_texts:
            assert t.strip() != default_norm, (
                f"BUG: default prompt leaked into encoding set: {t!r}"
            )

    # Shard unique texts across ranks. Index-mod sharding keeps batches roughly balanced.
    my_text_indices = list(range(rank, len(unique_texts), world_size))
    my_texts = [unique_texts[i] for i in my_text_indices]

    # Encode in batches and print each caption as it goes through.
    encoded = {}  # text_hash -> tensor
    t0 = time.perf_counter()
    for batch_start in range(0, len(my_texts), args.batch_size):
        batch_texts = my_texts[batch_start : batch_start + args.batch_size]
        tensors = encode_batch(encoder, tokenizer, batch_texts, device)
        for i, (t, tensor) in enumerate(zip(batch_texts, tensors)):
            assert default_norm == "" or t.strip() != default_norm, (
                "BUG: default prompt would have been encoded"
            )
            encoded[text_hash(t)] = tensor
            global_idx = my_text_indices[batch_start + i]
            preview = t.replace("\n", " ")
            if len(preview) > args.max_print_chars:
                preview = preview[: args.max_print_chars] + "..."
            print(
                f"[rank {rank}] [{global_idx + 1}/{len(unique_texts)}] {preview}",
                flush=True,
            )
        if rank == 0 and (batch_start // args.batch_size) % 50 == 0:
            done = batch_start + len(batch_texts)
            elapsed = time.perf_counter() - t0
            rate = done / max(elapsed, 1e-6)
            print(
                f"[rank 0] [progress] {done}/{len(my_texts)} unique texts encoded "
                f"({rate:.1f}/s, {elapsed:.0f}s elapsed)",
                flush=True,
            )

    # Each rank writes its hashed-emb partial + the json_key->text_hash mapping it owns.
    my_keys_to_hash = {
        k: text_hash(t)
        for t, keys in text_to_keys.items()
        if text_hash(t) in encoded
        for k in keys
    }
    if encoded:
        safetensors_save(encoded, str(partial_path))
        partial_keys_path.write_text(json.dumps(my_keys_to_hash))
        print(
            f"[rank {rank}] wrote {partial_path.name} ({len(encoded)} unique embs, "
            f"{len(my_keys_to_hash)} keys)",
            flush=True,
        )

    if world_size > 1:
        dist.barrier()

    # Rank 0: merge partials -> single per-job file. Layout: keys are the ORIGINAL
    # json keys (e.g. "f859d5f2/0-48"), values are bf16 [L, dim] embedding tensors.
    if rank == 0:
        merged_emb = {}  # text_hash -> tensor
        merged_key_map = {}  # json_key -> text_hash
        for r in range(world_size):
            p_emb = caption_dir / f".{stem}.part{r}.safetensors"
            p_keys = caption_dir / f".{stem}.part{r}.json"
            if p_emb.exists():
                shard = safetensors_load(str(p_emb))
                merged_emb.update(shard)
            if p_keys.exists():
                merged_key_map.update(json.loads(p_keys.read_text()))

        # Expand: for every json_key, write the embedding tensor under that key.
        # Same key as the source JSON -> same key in safetensors. Lookup by
        # `safe_open(...).get_tensor("f859d5f2/0-48")`.
        #
        # Memory-conscious merge: don't double-allocate. For each unique hash,
        # transfer ownership of its tensor from `merged_emb` to `final` (one
        # `dict.pop` instead of one `.clone()`). Only when N json keys share
        # one caption text (Syn4D has many boilerplate "completely dark screen"
        # duplicates) do we clone — safetensors refuses to save aliased
        # tensors, so each output key must own its own storage. This keeps
        # peak host RAM at ~`sizeof(unique_embs)` instead of ~2x that, which
        # matters on hosts where the merge of a 35 GB DL3DV cache OOM-killed.
        already_owned = set()  # hashes whose tensor has been moved into `final`
        final = {}
        missing = 0
        for k, h in merged_key_map.items():
            if h not in merged_emb:
                missing += 1
                continue
            if h in already_owned:
                # Duplicate caption text -> need an independent storage.
                final[k] = merged_emb[h].clone()
            else:
                # First occurrence -> move the tensor (no copy).
                final[k] = merged_emb.pop(h)
                already_owned.add(h)
        if missing:
            print(f"[warn:{label}] {missing} keys had no encoded tensor (rank gap?); skipped", flush=True)

        # Capture the unique-text count BEFORE we drop merged_emb. After the
        # pop loop above, merged_emb only holds orphan embeddings (encoded
        # hashes never referenced by merged_key_map), so its len() isn't the
        # right scalar for `num_unique_texts` anyway — already_owned is.
        n_unique_texts = len(already_owned)

        # Drop merged_emb before the safetensors write so we aren't holding
        # two ~35 GB dicts at the same time. Most entries are already empty
        # (popped) but the dict bookkeeping itself can be non-trivial.
        merged_emb.clear()
        del merged_emb
        gc.collect()

        tmp_path = caption_dir / f".{stem}.tmp.safetensors"
        safetensors_save(final, str(tmp_path))
        os.replace(tmp_path, final_path)

        meta = {
            "format": "safetensors",
            "dim": final[next(iter(final))].shape[-1] if final else 4096,
            "dtype": "bfloat16",
            "max_seq_len": args.max_seq_len,
            "trim_to_actual_seq_len": True,
            "t5_path": args.t5_path,
            "tokenizer_path": args.tokenizer_path,
            "default_prompt_filtered_out": args.default_prompt,
            "num_keys": len(final),
            "num_unique_texts": n_unique_texts,
            "label": label,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        # Cleanup partials.
        for r in range(world_size):
            for path in (
                caption_dir / f".{stem}.part{r}.safetensors",
                caption_dir / f".{stem}.part{r}.json",
            ):
                if path.exists():
                    path.unlink()
        print(
            f"[rank 0] wrote {final_path.name} "
            f"({len(final)} keys, {n_unique_texts} unique embs) [{label}]",
            flush=True,
        )

    if world_size > 1:
        dist.barrier()


def main():
    args = parse_args()
    rank, world_size, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[init] world_size={world_size} device={device}", flush=True)
        print(f"[init] t5={args.t5_path}", flush=True)
        print(f"[init] tokenizer={args.tokenizer_path}", flush=True)

    encoder, tokenizer = load_text_encoder(args.t5_path, args.tokenizer_path, device)

    for caption_dir in args.caption_dirs:
        process_caption_dir(args, caption_dir, encoder, tokenizer, rank, world_size, device)
        # Free CUDA cache between dirs to keep working set tight.
        torch.cuda.empty_cache()
        gc.collect()

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print("[done]", flush=True)


if __name__ == "__main__":
    main()
