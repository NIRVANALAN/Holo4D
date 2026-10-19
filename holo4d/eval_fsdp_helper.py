"""Eval-side FSDP wrapping helper for Track4D-360 inference.

Mirrors the training-time FSDP setup at
``scripts/wan2.1_fun/train_track4d360.py:1052+`` so the same DiTBlock-based
sharding works for inference launchers. Backward-compatible: when not
launched via `accelerate launch ...`, all helpers are no-ops and the
existing single-GPU flow runs unchanged.

Use cases
---------
- 14B inference on a 2-GPU box (e.g. 2x RTX A6000 48GB) where 14B bf16
  (~28GB DiT + T5 + activations) does not fit on a single 48 GB card.
  Sharding the DiT across both GPUs drops per-GPU usage to ~14 GB plus
  the (small) auxiliary modules.
- Future 32B+ models that won't fit on any single GPU at all.

Usage
-----
1. Launch via:
       accelerate launch \\
         --config_file examples/accelerate_configs/accelerate_config_fsdp_2gpu.yaml \\
         examples/wan2.1_fun/eval_track4d360_*.py \\
         --use_fsdp <other args>

2. Inside the eval entry's ``build_eval_pipeline`` call:
       from holo4d.eval_fsdp_helper import maybe_init_fsdp
       ...
       pipe = WanVideoTrack4D360Pipeline.from_pretrained(...)
       # after camera-control base initialization / track init / checkpoint load
       maybe_init_fsdp(pipe)

3. Gate file-writes with ``is_main_process()``::
       from holo4d.eval_fsdp_helper import is_main_process
       if is_main_process():
           imageio.mimwrite(out_path, frames)

The accelerate config wraps ``DiTBlock`` (and ``T5SelfAttention`` if T5
is also prepared) via ``TRANSFORMER_BASED_WRAP``. Auxiliary modules
(``track_adapter``, ``track_block_injector``, ``dense_target_control_encoder``,
``control_adapter``, ``control_patch_embedding``, ``domain_embedding``,
``gs_decoder``) stay outside FSDP and are gathered every forward via the
existing pipeline call paths.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


# Module-global Accelerator. Initialized lazily in ``maybe_init_fsdp`` so
# importing this helper has zero runtime cost when FSDP is not active.
_ACCELERATOR = None  # type: Optional["accelerate.Accelerator"]  # noqa: F821


def fsdp_active() -> bool:
    """True when the process is part of a multi-rank ``accelerate launch``.

    Detected via ``WORLD_SIZE>1`` env var, which accelerate sets at launch
    time. Single-GPU runs (``python script.py``) always return False, so
    every helper in this module becomes a no-op.
    """
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    """True if running single-process (no FSDP) or as rank 0 under FSDP.

    Use this to gate file writes, progress prints, and other
    side-effects. Compute (e.g. ``pipe(...)``) MUST run on every rank
    so FSDP all-gather/reduce-scatter calls are synchronous.
    """
    if not fsdp_active():
        return True
    return int(os.environ.get("RANK", "0")) == 0


def get_accelerator():
    """Return the singleton Accelerator (or None when FSDP is not active)."""
    return _ACCELERATOR


def init_fsdp_early(args) -> Optional["accelerate.Accelerator"]:  # noqa: F821
    """Initialize the Accelerator BEFORE model loading and pin
    ``args.device`` to this rank's CUDA device.

    Why this is necessary (per fpt review 2026-04-28):
        ``WanVideoTrack4D360Pipeline.from_pretrained`` allocates the full
        ~28 GB DiT, ~10 GB T5, and ~1 GB VAE onto ``args.device`` BEFORE
        any FSDP wrap can fire. If ``args.device == "cuda"`` (the default)
        on both ranks of an ``accelerate launch``, each rank silently
        defaults to ``cuda:0`` because no ``set_device(local_rank)`` has
        run yet. Both ranks then OOM on ``cuda:0`` (or one rank
        succeeds while the other steals memory on the same device).

        Solution: instantiate Accelerator early so ``accelerate.prepare``'s
        process-group init runs and ``accelerator.device`` reflects this
        rank's local device (``cuda:0`` on rank 0, ``cuda:1`` on rank 1,
        etc.). Then point ``args.device`` at that device so subsequent
        ``from_pretrained`` calls allocate to the correct GPU.

    No-op when not under ``accelerate launch`` (``WORLD_SIZE<=1``).

    Should be called from each eval entry's ``main()`` BEFORE
    ``build_eval_pipeline(args)``.
    """
    global _ACCELERATOR
    if not fsdp_active():
        return None

    import accelerate

    if _ACCELERATOR is None:
        # mixed_precision="no" is CRITICAL for inference. The yaml has
        # mixed_precision=bf16 (correct for training), which makes
        # accelerator.prepare(model) upcast every param to fp32 at
        # accelerator.py:1716 BEFORE FSDP can shard. The upcast doubles
        # bf16 (28 GB) -> fp32 (56 GB) transiently per-param, OOMing on
        # 48 GB A6000s right at the prepare call (verified empirically
        # 2026-04-29: rank 0/1 fail with "Tried to allocate 772.00 MiB,
        # 46.28 GB already allocated"). Inference doesn't need the fp32
        # master copy (no optimizer); bf16 params + bf16 compute is
        # correct. Override the yaml's mixed_precision here.
        _ACCELERATOR = accelerate.Accelerator(mixed_precision="no")
    accelerator = _ACCELERATOR

    # Pin args.device to this rank's local CUDA device so from_pretrained
    # allocates the DiT/T5/VAE to the right GPU on each rank.
    new_device = str(accelerator.device)
    if hasattr(args, "device"):
        args.device = new_device

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if is_main_process():
        print(
            f"[Eval/FSDP] Early accelerator init: world_size={world_size}, "
            f"rank={rank}, device={new_device}",
            flush=True,
        )

    return accelerator


def maybe_init_fsdp(pipe) -> Optional["accelerate.Accelerator"]:  # noqa: F821
    """Wrap ``pipe.dit`` with FSDP if running under ``accelerate launch``.

    No-op when not multi-process. The wrap exclusion logic mirrors
    ``train_track4d360.py:1052-1090`` exactly so the same forward path
    that the training pipeline exercises will work at inference.

    Returns the Accelerator instance (or None if not under FSDP).
    """
    global _ACCELERATOR
    if not fsdp_active():
        return None

    import accelerate

    if _ACCELERATOR is None:
        # mixed_precision="no" is CRITICAL for inference. The yaml has
        # mixed_precision=bf16 (correct for training), which makes
        # accelerator.prepare(model) upcast every param to fp32 at
        # accelerator.py:1716 BEFORE FSDP can shard. The upcast doubles
        # bf16 (28 GB) -> fp32 (56 GB) transiently per-param, OOMing on
        # 48 GB A6000s right at the prepare call (verified empirically
        # 2026-04-29: rank 0/1 fail with "Tried to allocate 772.00 MiB,
        # 46.28 GB already allocated"). Inference doesn't need the fp32
        # master copy (no optimizer); bf16 params + bf16 compute is
        # correct. Override the yaml's mixed_precision here.
        _ACCELERATOR = accelerate.Accelerator(mixed_precision="no")
    accelerator = _ACCELERATOR

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # ---------------------------------------------------------------
    # 1. Hide pipe.vae from FSDP module tree.
    #    pipe.vae is accessed via .encode()/.decode() (plain Python
    #    methods), never pipe.vae(...). With SIZE_BASED_WRAP the VAE
    #    would get sharded but no __call__ ever fires the FSDP
    #    pre-forward gather hook -> params stay at shape (0,) on
    #    some ranks. The fix: detach vae from pipe._modules and
    #    re-attach as a plain instance attribute so FSDP traversal
    #    can't see it. See train_track4d360.py:1057-1067.
    # ---------------------------------------------------------------
    if hasattr(pipe, "vae") and isinstance(pipe.vae, torch.nn.Module):
        _vae = pipe.vae
        # Use try/except in case the attribute is already detached
        try:
            del pipe.vae
        except AttributeError:
            pass
        object.__setattr__(pipe, "vae", _vae)

    # ---------------------------------------------------------------
    # 2. Hide DiT root submodules / parameters accessed via dit.X(...) or
    #    dit.X[...] that bypass dit.__call__.
    #
    #    Why broader than training: at TRAINING, `model(data)` ->
    #    `model.forward()` eventually invokes `dit(...)` (dit.__call__),
    #    which fires the root FSDP unit's pre-forward gather hook, so
    #    every submodule and parameter on `dit` is gathered before any
    #    `dit.X(...)` access in patchify/etc. — see
    #    `train_track4d360.py:1083-1087`.
    #
    #    At EVAL, `pipe(...)` calls `model_fn_wan_video_v2v(dit, ...)`
    #    which calls `dit.patchify(...)`, `dit.time_embedding(...)`,
    #    `dit.text_embedding(...)`, `dit.head(...)`, `dit.img_emb(...)`,
    #    `dit.control_patch_embedding(...)`, `dit.domain_embedding[...]`
    #    DIRECTLY without ever invoking `dit.__call__`. So the root
    #    FSDP gather hook NEVER fires, and any sharded submodule/param
    #    sits at shape (0,) -> conv3d/linear raises
    #    "weight should have at least three dimensions" or
    #    "size of tensor a (X) must match tensor b (0)".
    #
    #    Fix: hide every submodule/param the eval path touches via
    #    `dit.X(...)`. They become plain instance attributes,
    #    fully replicated on each rank. Tiny memory cost (<300 MB at
    #    bf16); the 28 GB DiTBlock weights stay sharded.
    #
    #    NOT hidden: `dit.blocks` (the workhorse, MUST be sharded;
    #    each DiTBlock IS wrapped as its own FSDP unit so its
    #    pre-forward hook fires per-block during dit.patchify->blocks
    #    iteration), `dit.control_adapter` (invoked via __call__, its
    #    own FSDP unit hook fires correctly).
    # ---------------------------------------------------------------
    _dit = pipe.dit
    for _name in (
        "time_embedding",
        "time_projection",
        "text_embedding",
        "head",
        "img_emb",
        "patch_embedding",
        "ref_conv",
        "pre_dit_cam_encoder",
        "control_patch_embedding",   # 14B eval bug 2026-04-29 (a): dit.patchify calls
                                     # this via .__call__ but dit.__call__ never fires
                                     # -> conv3d weight at shape (0,). Hide it.
        "control_adapter",           # 14B eval bug 2026-04-29 (b): same root cause —
                                     # dit.patchify calls dit.control_adapter(x) but
                                     # control_adapter isn't a DiTBlock so it's part
                                     # of the root FSDP unit, not its own unit. The
                                     # training comment claimed it fires via __call__
                                     # but at eval dit.__call__ never fires (model_fn
                                     # uses dit.method(...) directly), so the root
                                     # gather never runs and control_adapter sits at
                                     # shape (0,). Hide it like the other 8.
    ):
        _mod = getattr(_dit, _name, None)
        if _mod is None or not isinstance(_mod, torch.nn.Module):
            continue
        try:
            delattr(_dit, _name)
        except AttributeError:
            continue
        object.__setattr__(_dit, _name, _mod)

    # `domain_embedding` is a Parameter, not a Module — needs slightly
    # different setattr trick. Same root cause: accessed via
    # `dit.domain_embedding[domain_ids]` in model_fn_wan_video_v2v
    # (wan_video_v2v.py:3029) without dit.__call__ ever firing.
    _dom = getattr(_dit, "domain_embedding", None)
    if isinstance(_dom, torch.nn.Parameter):
        try:
            delattr(_dit, "domain_embedding")
        except AttributeError:
            pass
        object.__setattr__(_dit, "domain_embedding", _dom)

    # ---------------------------------------------------------------
    # 3. Wrap pipe.dit with FSDP via accelerator.prepare. The auto-wrap
    #    policy in the accelerate yaml (TRANSFORMER_BASED_WRAP on
    #    DiTBlock) shards just the blocks; everything else stays
    #    replicated on each rank.
    # ---------------------------------------------------------------
    pipe.dit = accelerator.prepare(pipe.dit)

    if is_main_process():
        print(
            f"[Eval/FSDP] Wrapped pipe.dit with FSDP across {world_size} ranks "
            f"(this rank={rank}, device={accelerator.device})",
            flush=True,
        )

    return accelerator
