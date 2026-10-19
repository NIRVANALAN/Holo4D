import json
import os
from pathlib import Path


DEFAULT_CAPTION_OUTPUT_ROOT = Path(
    os.environ.get(
        "TRACK4D360_CAPTION_OUTPUT_ROOT",
        "data/captions",
    )
)


# ---- Rate-limited cache-miss reporter --------------------------------------
# When a dataset has to fall back to DEFAULT_PROMPT because its T5 emb cache
# doesn't cover a particular sample, we want to know — but at the same time
# we don't want 100k+ identical log lines spamming stdout.
#
# This caches the first ``max_unique`` unique (dataset, key) misses per
# dataset name and prints each exactly once. After the cap, prints one
# summary "suppressing further" line per dataset and goes silent.
#
# Usage: from any dataset's miss-fallback branch:
#     from .mixed_nvs_captions import report_cache_miss
#     report_cache_miss("Syn4D_Multi_NVS", f"{scene_name}/{cam}/frames=[{a}..{b}]")
_CACHE_MISS_REGISTRY: dict[str, set[str]] = {}
_CACHE_MISS_SUPPRESSED: set[str] = set()


def report_cache_miss(dataset_name: str, key: str, max_unique: int = 20) -> None:
    """Log a T5 cache miss once per unique (dataset, key) pair.

    Set TRACK4D360_VERBOSE_CACHE_MISS=1 to override the cap and see every miss.
    """
    cap_override = os.environ.get("TRACK4D360_VERBOSE_CACHE_MISS", "0") == "1"
    seen = _CACHE_MISS_REGISTRY.setdefault(dataset_name, set())
    if not cap_override and len(seen) >= max_unique:
        if dataset_name not in _CACHE_MISS_SUPPRESSED:
            _CACHE_MISS_SUPPRESSED.add(dataset_name)
            print(
                f"[T5 cache MISS] {dataset_name}: hit {max_unique} unique fallback keys, "
                f"suppressing further messages. "
                f"Set TRACK4D360_VERBOSE_CACHE_MISS=1 to see every miss.",
                flush=True,
            )
        return
    if key in seen:
        return
    seen.add(key)
    print(
        f"[T5 cache MISS] {dataset_name}: key={key!r} → falling back to "
        f"DEFAULT_PROMPT precomputed embedding (unique miss #{len(seen)})",
        flush=True,
    )


def sanitize_mixed_nvs_caption(raw_text):
    """Extract the useful caption text from Tarsier-style outputs."""
    if not raw_text or not str(raw_text).strip():
        return ""

    lines = str(raw_text).strip().split("\n")
    pred_idx = None
    for idx, line in enumerate(lines):
        if "prediction" in line.lower() and ":" in line:
            pred_idx = idx
            break

    if pred_idx is not None:
        text = "\n".join(lines[pred_idx + 1 :]).strip()
        if text:
            return text

    text = str(raw_text).strip()
    for prefix in ("Prompt:", "Caption:", "Description:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def load_chunked_captions(caption_dir):
    """Load Tarsier chunk captions from captions_all.json or rank shards."""
    caption_dir = Path(caption_dir)
    captions_all_path = caption_dir / "captions_all.json"
    if captions_all_path.is_file():
        with captions_all_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    merged = {}
    for shard_path in sorted(caption_dir.glob("captions_rank*.json")):
        with shard_path.open("r", encoding="utf-8") as f:
            merged.update(json.load(f))
    return merged


def select_caption_by_chunk_overlap(captions, sequence_id, frame_indices, default_prompt):
    """Pick the caption chunk with the highest overlap with the target frames."""
    if not captions:
        return default_prompt

    prefix = f"{sequence_id}/"
    frame_set = {int(frame_idx) for frame_idx in frame_indices}
    best_caption = default_prompt
    best_overlap = 0

    for key, raw_caption in captions.items():
        if not key.startswith(prefix):
            continue
        chunk_tag = key[len(prefix) :]
        try:
            chunk_start_str, chunk_end_str = chunk_tag.split("-", 1)
            chunk_start = int(chunk_start_str)
            chunk_end = int(chunk_end_str)
        except (ValueError, TypeError):
            continue

        chunk_frames = set(range(chunk_start, chunk_end + 1))
        overlap = len(frame_set & chunk_frames)
        if overlap > best_overlap:
            best_overlap = overlap
            best_caption = sanitize_mixed_nvs_caption(raw_caption)

    return best_caption if best_caption else default_prompt


def default_caption_dir_for_dataset(dataset_type, caption_root=None):
    caption_root = Path(caption_root) if caption_root else DEFAULT_CAPTION_OUTPUT_ROOT
    if caption_root.is_dir():
        if (caption_root / "captions_all.json").is_file():
            return caption_root
        if any(caption_root.glob("captions_rank*.json")):
            return caption_root
    dataset_map = {
        "tartanair": "tartanair",
        "dl3dv": "dl3dv",
        "mvs_synth": "mvs_synth",
        "wildrgbd": "wildrgbd",
        "re10k": "re10k",
    }
    subdir = dataset_map.get(dataset_type)
    if subdir is None:
        return None
    return caption_root / subdir


def _select_caption_key_by_chunk_overlap(captions, sequence_id, frame_indices):
    """Variant of select_caption_by_chunk_overlap that returns the *json key*
    (e.g. ``f859d5f2/0-48``) of the best-overlap chunk, plus the raw caption
    text. Used by the T5 embedding cache lookup so we can index a parallel
    safetensors file by exactly the same key as the source captions JSON.

    Returns ``(key_or_none, raw_caption_or_none)``.
    """
    if not captions:
        return None, None

    prefix = f"{sequence_id}/"
    frame_set = {int(frame_idx) for frame_idx in frame_indices}
    best_key = None
    best_caption = None
    best_overlap = 0

    for key, raw_caption in captions.items():
        if not key.startswith(prefix):
            continue
        chunk_tag = key[len(prefix):]
        try:
            chunk_start_str, chunk_end_str = chunk_tag.split("-", 1)
            chunk_start = int(chunk_start_str)
            chunk_end = int(chunk_end_str)
        except (ValueError, TypeError):
            continue
        chunk_frames = set(range(chunk_start, chunk_end + 1))
        overlap = len(frame_set & chunk_frames)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = key
            best_caption = raw_caption
    return best_key, best_caption


class CachedT5EmbStore:
    """Lazy-loaded T5 embedding cache for one dataset's caption directory.

    The companion file to ``captions_all.json`` (or merged ``captions_rank*.json``)
    is ``captions_t5_emb.safetensors``, written by
    ``holo4d/precompute_t5_cache.py``. Keys mirror the JSON keys exactly
    (e.g. ``f859d5f2/0-48``). Tensors are bf16, trimmed to actual non-pad token
    count.

    The default-prompt fallback is NOT stored under any sentinel key here —
    the precompute script filters DEFAULT_PROMPT out of the encoding set. The
    training-time pipeline serves default-prompt samples from
    ``pipe.precomputed_prompt_embs`` (set up by Track4D-360 training init in
    ``scripts/wan2.1_fun/train_track4d360.py:_setup_prompt_emb_cache``), not
    from this safetensors file. ``get_default()`` therefore always returns
    None and is kept only as a stable no-op API surface.
    """

    # Sentinel preserved for back-compat callers; the safetensors will not
    # contain this key, so ``get_default()`` returns None.
    DEFAULT_PROMPT_KEY = "__default__"

    def __init__(self, caption_dir):
        self.caption_dir = Path(caption_dir)
        self._safetensors_path = self.caption_dir / "captions_t5_emb.safetensors"
        self._meta_path = self.caption_dir / "captions_t5_emb_meta.json"
        self._meta = None
        # Single-file mode: one handle. Multi-shard mode (KEEP_SHARDS=1 in
        # precompute_t5_cache_recam.py for ReCam/SynCam): {shard_path: handle}
        # plus a key->shard_path index built once at first _ensure_open.
        self._handle = None
        self._shard_handles = None
        self._key_to_shard = None
        self._available_keys = None  # set[str], lazily populated
        # Discover shard layout up-front so .available reflects either form.
        self._shard_paths = sorted(
            self.caption_dir.glob("captions_t5_emb.rank*.part*.safetensors")
        )
        self._available = (
            self._safetensors_path.is_file() or len(self._shard_paths) > 0
        )

    @property
    def available(self):
        return self._available

    def _ensure_open(self):
        if not self._available:
            return
        if self._handle is not None or self._shard_handles is not None:
            return
        # Lazy-import safetensors so non-cached paths don't pay the cost.
        from safetensors import safe_open  # type: ignore

        if self._safetensors_path.is_file():
            # Single-file path (existing datasets: dl3dv, re10k, kubric_*, etc.).
            self._handle = safe_open(
                str(self._safetensors_path), framework="pt", device="cpu"
            )
            self._available_keys = set(self._handle.keys())
        else:
            # Multi-shard path (ReCam / SynCam, where the merge step would OOM).
            # Fast path: if precompute wrote captions_t5_emb_shard_index.json
            # ({key: shard_basename}), load that JSON instead of scanning all
            # 70 shard headers. ~5–10× faster startup per worker.
            self._shard_handles = {}
            self._key_to_shard = {}
            index_path = self.caption_dir / "captions_t5_emb_shard_index.json"
            if index_path.is_file():
                try:
                    with index_path.open() as fh:
                        name_to_path = {p.name: str(p) for p in self._shard_paths}
                        idx_raw = json.load(fh)
                    self._key_to_shard = {
                        k: name_to_path[v] for k, v in idx_raw.items()
                        if v in name_to_path
                    }
                except Exception:
                    self._key_to_shard = {}
            if not self._key_to_shard:
                # Fallback: scan each shard's keys to build the index in-memory.
                for sp in self._shard_paths:
                    h = safe_open(str(sp), framework="pt", device="cpu")
                    self._shard_handles[str(sp)] = h
                    for k in h.keys():
                        self._key_to_shard[k] = str(sp)
            self._available_keys = set(self._key_to_shard.keys())
        if self._meta_path.is_file():
            try:
                self._meta = json.loads(self._meta_path.read_text())
            except Exception:
                self._meta = None

    def get_by_key(self, key):
        """Return [L, dim] bf16 tensor for ``key`` or None if absent / disabled."""
        if not self._available or not key:
            return None
        self._ensure_open()
        if self._available_keys is None or key not in self._available_keys:
            return None
        if self._handle is not None:
            return self._handle.get_tensor(key)
        # Multi-shard: route to the shard that owns this key. Lazy-open the
        # shard handle on first use (the JSON-index fast path defers all
        # safetensors opens to actual key fetches, not startup).
        sp = self._key_to_shard[key]
        h = self._shard_handles.get(sp)
        if h is None:
            from safetensors import safe_open  # type: ignore
            h = safe_open(sp, framework="pt", device="cpu")
            self._shard_handles[sp] = h
        return h.get_tensor(key)

    def get_default(self):
        return self.get_by_key(self.DEFAULT_PROMPT_KEY)


class ChunkedCaptionStore:
    """Lazy-loaded chunk-caption lookup for one dataset.

    Two parallel resources (loaded only when first queried):
      1. The captions JSON shards (``captions_all.json`` / ``captions_rank*.json``)
         that map chunk keys to raw Tarsier text.
      2. ``captions_t5_emb.safetensors`` (if present in the same directory):
         pre-computed Wan umt5-xxl embeddings, keyed identically to the JSON.

    Use ``get(...)`` for the legacy text-only API, ``get_with_emb(...)`` for the
    new (text, emb_or_none) API consumed by training when the cache exists.
    """

    def __init__(self, caption_dir, default_prompt):
        self.caption_dir = Path(caption_dir)
        self.default_prompt = str(default_prompt)
        self._captions = None
        self._emb_store = CachedT5EmbStore(self.caption_dir)

    @property
    def emb_available(self):
        return self._emb_store.available

    def _ensure_captions(self):
        if self._captions is None:
            self._captions = load_chunked_captions(self.caption_dir)

    def get(self, sequence_id, frame_indices):
        self._ensure_captions()
        return select_caption_by_chunk_overlap(
            captions=self._captions,
            sequence_id=sequence_id,
            frame_indices=frame_indices,
            default_prompt=self.default_prompt,
        )

    def get_with_emb(self, sequence_id, frame_indices):
        """Return ``(prompt_text, prompt_emb_or_none)``. The embedding is None when
        the cache file does not exist or the matched chunk key is not present.
        Callers that consume only one of the two should still prefer this
        method so a single chunk-overlap selection serves both lookups."""
        self._ensure_captions()
        key, raw = _select_caption_key_by_chunk_overlap(
            self._captions, sequence_id, frame_indices
        )
        if key is None:
            # No chunk in the captions JSON overlaps the sampled frame range.
            # Report once per unique (dataset, sequence_id, frame-range) so
            # the user can see WHICH samples are falling through.
            try:
                fr_lo = min(int(f) for f in frame_indices)
                fr_hi = max(int(f) for f in frame_indices)
                fr = f"[{fr_lo}..{fr_hi}]"
            except (ValueError, TypeError):
                fr = "?"
            report_cache_miss(
                f"ChunkedCaptionStore[{self.caption_dir.name}]",
                f"sequence_id={sequence_id!r} frames={fr}",
            )
            text = self.default_prompt
            emb = self._emb_store.get_default() if self._emb_store.available else None
            return text, emb
        text = sanitize_mixed_nvs_caption(raw) or self.default_prompt
        emb = self._emb_store.get_by_key(key) if self._emb_store.available else None
        if emb is None:
            # Key was found in the captions JSON but no matching emb in the
            # safetensors. This is the classic "text exists, T5 emb missing"
            # gap — exactly what audit_t5_cache_strict checks.
            report_cache_miss(
                f"ChunkedCaptionStore[{self.caption_dir.name}]",
                f"key={key!r} (text present, emb absent)",
            )
        # Note: the default prompt is NOT encoded in the safetensors cache
        # (precompute filters it out — every cached entry is a real Tarsier
        # caption). When emb is None, the training side falls back to whatever
        # default-emb mechanism it provides (or back to online T5 for that
        # small fraction of samples).
        return text, emb
