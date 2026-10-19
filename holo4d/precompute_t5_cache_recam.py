"""Pre-compute Wan umt5-xxl T5 embeddings for ReCamMaster / SynCamMaster
(SynCamVideo-Dataset / MultiCamVideo-Dataset).

Layout (per the local SynCamVideo-Dataset on clifton):
    ${ROOT}/${split}/<aperture>/<scene>/videos/cam<NN>_caption.txt
        # train: split=train, aperture=f24_aperture5, scene=scene1..scene100X
        # val:   split=val,   aperture=basic,         (no scene level)

Format: Tarsier "Prompt: ...\\nPrediction: <text>" — sanitized by
`sanitize_mixed_nvs_caption`.

Output:
    <output_dir>/captions_all.json           {key -> sanitized text}
    <output_dir>/captions_t5_emb.safetensors {key -> bf16 [L, dim] tensor}
    <output_dir>/captions_t5_emb_meta.json

Key format: f"{split}/{aperture}/{scene}/{cam}" (scene="" when val has
no scene level). Adjust to whatever the ReCamMaster_NVS loader uses.

Launch:
    torchrun --nproc_per_node=2 -m holo4d.precompute_t5_cache_recam \\
        --root data/SynCamVideo-Dataset \\
        --output_dir data/captions/syncamvideo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import importlib.util as _importlib_util  # noqa: E402

_CAPTIONS_PATH = (
    _REPO / "holo4d_runtime" / "data" / "stream3r" / "dust3r" / "datasets_cut3r"
    / "mixed_nvs_captions.py"
)
_spec = _importlib_util.spec_from_file_location("_track4d_caption_utils", _CAPTIONS_PATH)
_mod = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sanitize_mixed_nvs_caption = _mod.sanitize_mixed_nvs_caption


_CAM_RE = re.compile(r"^cam(\d+)_caption\.txt$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="SynCamVideo / MultiCamVideo dataset root "
                        "(parent of train/ and val/).")
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   help="Which splits to walk.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--t5_path",
                   default=str(_REPO / "weights/wan21-1p3b-run/models_t5_umt5-xxl-enc-bf16.safetensors"))
    p.add_argument("--tokenizer_path",
                   default=str(_REPO / "weights/wan21-1p3b-run/google/umt5-xxl"))
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--default_prompt", default="A multi-view video scene.")
    p.add_argument("--flush_every", type=int, default=2000,
                   help="Flush in-memory embeddings to a part-shard every N keys "
                        "to bound CPU memory. Lower = less RAM.")
    p.add_argument("--keep_shards", action="store_true",
                   help="Skip the final merge into one safetensors file; leave "
                        "captions_t5_emb.partXXXXX.safetensors in place. Useful "
                        "when the merged file would exceed CPU RAM.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def init_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        torch.cuda.set_device(local)
        if world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
        return rank, world, local
    return 0, 1, 0


def load_text_encoder(t5_path, tokenizer_path, device):
    from holo4d_runtime.models.wan_video_text_encoder import (  # noqa: E402
        HuggingfaceTokenizer, WanTextEncoder,
    )
    print(f"[rank {os.environ.get('RANK', '0')}] Loading T5 from {t5_path}", flush=True)
    enc = WanTextEncoder()
    state = safetensors_load(t5_path)
    miss, unex = enc.load_state_dict(state, strict=False)
    if miss or unex:
        print(f"[T5] missing={len(miss)} unexpected={len(unex)}", flush=True)
    enc = enc.to(device=device, dtype=torch.bfloat16).eval()
    for p_ in enc.parameters():
        p_.requires_grad_(False)
    tok = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean="whitespace")
    return enc, tok


def encode_batch(encoder, tokenizer, texts, device):
    ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device); mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        emb = encoder(ids, mask)
    out = []
    for i, L in enumerate(seq_lens.cpu().tolist()):
        out.append(emb[i, :max(int(L), 1)].to(dtype=torch.bfloat16, device="cpu").contiguous().clone())
    return out


def collect_captions(root: Path, splits, default_prompt: str):
    """Walk the SynCamVideo / MultiCamVideo tree and collect per-(split,
    aperture, scene, cam) captions.

    Layout has TWO variants:
      train/<aperture>/<scene>/videos/cam<NN>_caption.txt
      val/<aperture>/videos/cam<NN>_caption.txt   (no scene level)

    Returns dict {key: sanitized_text} where key is
    f"{split}/{aperture}/{scene}/cam{nn}" (scene="" for val).
    """
    out: dict[str, str] = {}
    skipped_default = 0
    skipped_empty = 0
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[skip] split dir not found: {split_dir}")
            continue
        for aperture_dir in sorted(split_dir.iterdir()):
            if not aperture_dir.is_dir():
                continue
            # Detect val-style (videos/ directly under aperture) vs
            # train-style (scenes intermediate).
            videos_direct = aperture_dir / "videos"
            scene_dirs = (
                [aperture_dir]
                if videos_direct.is_dir()
                else sorted(p for p in aperture_dir.iterdir() if p.is_dir())
            )
            for scene_dir in scene_dirs:
                vid_dir = scene_dir / "videos"
                if not vid_dir.is_dir():
                    continue
                scene_name = "" if scene_dir == aperture_dir else scene_dir.name
                for cap_path in sorted(vid_dir.glob("cam*_caption.txt")):
                    m = _CAM_RE.match(cap_path.name)
                    if not m:
                        continue
                    cam_id = int(m.group(1))
                    try:
                        raw = cap_path.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    text = sanitize_mixed_nvs_caption(raw)
                    if not text:
                        skipped_empty += 1
                        continue
                    if default_prompt and text == default_prompt:
                        skipped_default += 1
                        continue
                    key = (
                        f"{split}/{aperture_dir.name}/{scene_name}/cam{cam_id:02d}"
                        if scene_name
                        else f"{split}/{aperture_dir.name}/cam{cam_id:02d}"
                    )
                    out[key] = text
    print(f"[collect] {len(out)} captions kept, {skipped_default} == default, "
          f"{skipped_empty} empty across splits {splits}", flush=True)
    return out


def main():
    args = parse_args()
    rank, world, local = init_dist()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.output_dir / "captions_all.json"
    sft_out = args.output_dir / "captions_t5_emb.safetensors"
    meta_out = args.output_dir / "captions_t5_emb_meta.json"

    if sft_out.exists() and not args.overwrite:
        print(f"[skip] {sft_out} exists (--overwrite to redo)", flush=True)
        return

    captions = collect_captions(args.root, args.splits, args.default_prompt)
    if rank == 0:
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(captions, f, indent=2)
        print(f"[write] {json_out}  ({len(captions)} keys)", flush=True)

    keys_sorted = sorted(captions.keys())
    my_keys = keys_sorted[rank::world]
    print(f"[rank {rank}] encoding {len(my_keys)}/{len(keys_sorted)}", flush=True)

    encoder, tokenizer = load_text_encoder(args.t5_path, args.tokenizer_path, device)

    def _shard_path(part_idx: int) -> Path:
        return args.output_dir / f"captions_t5_emb.rank{rank}.part{part_idx:05d}.safetensors"

    embs: dict[str, torch.Tensor] = {}
    part_idx = 0
    written_parts: list[Path] = []
    rank_total = 0

    def _flush(force: bool):
        nonlocal embs, part_idx, rank_total
        if not embs:
            return
        if not force and len(embs) < args.flush_every:
            return
        sp = _shard_path(part_idx)
        safetensors_save(embs, str(sp))
        written_parts.append(sp)
        rank_total += len(embs)
        print(f"[rank {rank}] wrote part {sp.name} ({len(embs)} keys, "
              f"cumulative {rank_total}/{len(my_keys)})", flush=True)
        embs = {}
        part_idx += 1

    for i in range(0, len(my_keys), args.batch_size):
        chunk = my_keys[i : i + args.batch_size]
        texts = [captions[k] for k in chunk]
        out_e = encode_batch(encoder, tokenizer, texts, device)
        for k, e in zip(chunk, out_e):
            embs[k] = e
        _flush(force=False)
        if i % (args.batch_size * 16) == 0:
            print(f"[rank {rank}] {i + len(chunk)}/{len(my_keys)}", flush=True)
    _flush(force=True)
    print(f"[rank {rank}] done — {rank_total} keys across {len(written_parts)} part(s)",
          flush=True)

    # Free encoder memory before the merge step.
    del encoder, tokenizer
    torch.cuda.empty_cache()

    if world > 1:
        dist.barrier()

    if rank == 0:
        all_parts = sorted(args.output_dir.glob("captions_t5_emb.rank*.part*.safetensors"))
        total_keys = 0
        if args.keep_shards:
            # Build key->shard index AND enumerate keys in one pass.
            # The index lets the runtime loader resolve key -> tensor in
            # O(1) without re-reading every shard header at startup.
            key_to_shard: dict[str, str] = {}
            for sp in all_parts:
                with safe_open(str(sp), framework="pt") as f:
                    for k in f.keys():
                        key_to_shard[k] = sp.name
            total_keys = len(key_to_shard)
            index_out = args.output_dir / "captions_t5_emb_shard_index.json"
            with index_out.open("w", encoding="utf-8") as f:
                json.dump(key_to_shard, f)
            with meta_out.open("w", encoding="utf-8") as f:
                json.dump({
                    "keys": sorted(key_to_shard.keys()),
                    "shards": [p.name for p in all_parts],
                    "shard_index": index_out.name,
                    "max_seq_len": int(args.max_seq_len),
                    "default_prompt": args.default_prompt,
                    "num_keys": total_keys,
                    "splits": args.splits,
                    "root": str(args.root),
                }, f, indent=2)
            print(f"[done] {total_keys} keys across {len(all_parts)} shards "
                  f"(no merge — --keep_shards), index -> {index_out.name}",
                  flush=True)
        else:
            # Stream-merge: load tensors one at a time so we never hold a full
            # shard plus the merged dict simultaneously. Peak memory == size
            # of merged dict only.
            merged: dict[str, torch.Tensor] = {}
            for sp in all_parts:
                with safe_open(str(sp), framework="pt") as f:
                    for k in f.keys():
                        merged[k] = f.get_tensor(k)
                sp.unlink()
            safetensors_save(merged, str(sft_out))
            total_keys = len(merged)
            with meta_out.open("w", encoding="utf-8") as f:
                json.dump({
                    "keys": sorted(merged.keys()),
                    "max_seq_len": int(args.max_seq_len),
                    "default_prompt": args.default_prompt,
                    "num_keys": total_keys,
                    "splits": args.splits,
                    "root": str(args.root),
                }, f, indent=2)
            print(f"[done] merged {total_keys} keys -> {sft_out}", flush=True)


if __name__ == "__main__":
    main()
