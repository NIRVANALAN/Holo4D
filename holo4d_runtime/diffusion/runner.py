import torch
import os
import re
import time
import traceback
import torch.distributed as dist
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from .logging import create_logger
from .model_printing import print_trainable_parameters_verification
from typing import Optional


def _debug_print_missing_grads(accelerator: Accelerator, model) -> None:
    """When TRACK4D_DEBUG_MISSING_GRADS=1 or train_track4d360 --debug_missing_grads."""
    if os.environ.get("TRACK4D_DEBUG_MISSING_GRADS", "0") != "1":
        return
    unwrapped = accelerator.unwrap_model(model)
    missing = []
    for name, param in unwrapped.named_parameters():
        if param.requires_grad and param.grad is None:
            missing.append(name)
    if not missing:
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    print(f"[rank{rank}] missing grads ({len(missing)}):", flush=True)
    for name in missing:
        print(f"  {name}", flush=True)


def _normalize_wandb_tags(raw):
    if raw is None:
        return None
    parts = [raw] if isinstance(raw, str) else list(raw)
    tags = []
    for part in parts:
        if part is None:
            continue
        for token in re.split(r"[,\s]+", str(part).strip()):
            if token:
                tags.append(token)
    return tags or None


def _set_epoch_for_dataloader(dataloader, epoch: int) -> None:
    """
    Best-effort epoch propagation for DDP shuffling.

    Stream3R datasets/samplers require calling `set_epoch(epoch)` to reshuffle both:
    - `dataset` (e.g., "N @ dataset" ResizedDataset mapping)
    - `sampler` / `batch_sampler` (e.g., BatchedRandomSampler / DistributedSampler)
    """
    if dataloader is None:
        return

    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        try:
            dataset.set_epoch(epoch)
        except TypeError:
            pass

    for attr in ("sampler", "batch_sampler"):
        sampler = getattr(dataloader, attr, None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(epoch)
            except TypeError:
                pass

        inner = getattr(sampler, "sampler", None)
        if inner is not None and hasattr(inner, "set_epoch"):
            try:
                inner.set_epoch(epoch)
            except TypeError:
                pass


def _reduce_optional_scalar_metrics(
    accelerator: Accelerator,
    metric_sums: dict,
    metric_counts: dict,
):
    """Reduce scalar metrics even when some ranks did not emit every key.

    Some training paths only produce a metric on the ranks whose local sample
    exercised that branch. A naive per-rank dict iteration deadlocks because
    different ranks call accelerator.gather() for different keys. We first
    agree on the union of keys, then gather both value and presence so missing
    metrics do not skew the average.
    """
    if len(metric_sums) == 0:
        return None

    local_metrics = {}
    for key, value_sum in metric_sums.items():
        count = max(int(metric_counts.get(key, 1)), 1)
        local_metrics[key] = value_sum / float(count)

    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        gathered_key_lists = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(gathered_key_lists, sorted(local_metrics.keys()))
        ordered_keys = []
        seen = set()
        for key_list in gathered_key_lists:
            for key in key_list or ():
                if key not in seen:
                    seen.add(key)
                    ordered_keys.append(key)
    else:
        ordered_keys = sorted(local_metrics.keys())

    reduced_metrics = {}
    zero = torch.zeros(1, device=accelerator.device, dtype=torch.float32)
    for key in ordered_keys:
        local_value = local_metrics.get(key, None)
        value_tensor = (
            local_value.reshape(1).to(device=accelerator.device, dtype=torch.float32)
            if local_value is not None
            else zero
        )
        present_tensor = torch.tensor(
            [1.0 if local_value is not None else 0.0],
            device=accelerator.device,
            dtype=torch.float32,
        )
        gathered_value = accelerator.gather(value_tensor)
        gathered_present = accelerator.gather(present_tensor)
        denom = max(float(gathered_present.sum().item()), 1.0)
        reduced_metrics[key] = float(gathered_value.sum().item() / denom)

    return reduced_metrics


def _maybe_create_torch_profiler(accelerator: Accelerator, args, output_path: str | None):
    if not accelerator.is_main_process or args is None:
        return None

    active_steps = int(getattr(args, "torch_profile_active_steps", 0) or 0)
    if active_steps <= 0:
        return None

    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    trace_dir = getattr(args, "torch_profile_dir", None) or os.path.join(
        output_path or ".",
        "torch_profiler",
    )
    os.makedirs(trace_dir, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    wait_steps = max(int(getattr(args, "torch_profile_wait_steps", 1) or 0), 0)
    warmup_steps = max(int(getattr(args, "torch_profile_warmup_steps", 1) or 0), 0)
    repeat = max(int(getattr(args, "torch_profile_repeat", 1) or 1), 1)
    rank = os.environ.get("RANK", "0")

    profiler = profile(
        activities=activities,
        schedule=schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=repeat,
        ),
        on_trace_ready=tensorboard_trace_handler(trace_dir, worker_name=f"rank{rank}"),
        record_shapes=bool(getattr(args, "torch_profile_record_shapes", True)),
        profile_memory=bool(getattr(args, "torch_profile_memory", True)),
        with_stack=bool(getattr(args, "torch_profile_with_stack", False)),
    )
    print(
        "[torch-profiler] rank0 enabled "
        f"(wait={wait_steps}, warmup={warmup_steps}, active={active_steps}, repeat={repeat}) "
        f"-> {trace_dir}",
        flush=True,
    )
    return profiler


def _print_cuda_oom_snapshot(accelerator: Accelerator, where: str, step: int, epoch_id: int) -> None:
    """Emit enough process/GPU state to distinguish true model OOM from GPU sharing."""
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", str(getattr(accelerator, "local_process_index", "?")))
    host = os.uname().nodename
    prefix = f"[cuda-oom] rank={rank} local_rank={local_rank} host={host} where={where} step={step} epoch={epoch_id + 1}"

    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            free_b, total_b = torch.cuda.mem_get_info(device)
            gib = 1024 ** 3
            print(
                f"{prefix} device={device} "
                f"mem_get_info_free={free_b / gib:.2f}GiB total={total_b / gib:.2f}GiB "
                f"allocated={torch.cuda.memory_allocated(device) / gib:.2f}GiB "
                f"reserved={torch.cuda.memory_reserved(device) / gib:.2f}GiB "
                f"max_allocated={torch.cuda.max_memory_allocated(device) / gib:.2f}GiB "
                f"max_reserved={torch.cuda.max_memory_reserved(device) / gib:.2f}GiB",
                flush=True,
            )
    except Exception as exc:
        print(f"{prefix} failed to read torch CUDA counters: {exc}", flush=True)

    try:
        import subprocess

        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=8,
        )
        if gpu_query.stdout.strip():
            print(f"{prefix} nvidia-smi gpu memory:\n{gpu_query.stdout.rstrip()}", flush=True)
        if gpu_query.stderr.strip():
            print(f"{prefix} nvidia-smi gpu query stderr:\n{gpu_query.stderr.rstrip()}", flush=True)

        app_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=8,
        )
        if app_query.stdout.strip():
            print(f"{prefix} nvidia-smi compute apps:\n{app_query.stdout.rstrip()}", flush=True)
        if app_query.stderr.strip():
            print(f"{prefix} nvidia-smi app query stderr:\n{app_query.stderr.rstrip()}", flush=True)
    except Exception as exc:
        print(f"{prefix} failed to run nvidia-smi snapshot: {exc}", flush=True)

    # Out-of-process / non-PyTorch HBM forensics. Per-GPU `used` above can far
    # exceed THIS process's allocation when (a) inter-node comm buffers (NCCL
    # net-proxy / libfabric-CXI / GDR) pile up on the NIC-local GPU (local_rank
    # 0), or (b) orphaned ranks from a prior crashed job still hold HBM that
    # --query-compute-apps fails to attribute. Default-format nvidia-smi prints
    # the FULL process table (every PID, not just this job's), and ps lists this
    # user's python procs on the host so a count > GPUs-per-node = zombies.
    try:
        import subprocess

        full_smi = subprocess.run(
            ["nvidia-smi"], check=False, text=True, capture_output=True, timeout=8,
        )
        if full_smi.stdout.strip():
            print(f"{prefix} nvidia-smi full:\n{full_smi.stdout.rstrip()}", flush=True)

        ps_query = subprocess.run(
            ["ps", "-o", "pid,user,etimes,rss,cmd", "-u", os.environ.get("USER", "")],
            check=False, text=True, capture_output=True, timeout=8,
        )
        if ps_query.stdout.strip():
            py_lines = "\n".join(
                ln for ln in ps_query.stdout.splitlines() if "python" in ln.lower()
            )
            print(f"{prefix} host python procs:\n{py_lines}", flush=True)
    except Exception as exc:
        print(f"{prefix} failed to run process-table snapshot: {exc}", flush=True)


def _print_mem_probe(accelerator: Accelerator, where: str, step: int, epoch_id: int) -> None:
    """Lightweight per-rank HBM probe (NO subprocess, safe to call periodically).

    `mem_get_info` free/total is GPU-wide and therefore includes out-of-process
    comm buffers (NCCL net-proxy / libfabric-CXI / GDR) and any orphaned ranks.
    A shrinking `free` while torch `reserved` stays flat localizes the pressure
    OUTSIDE the PyTorch allocator — which is the 288p backendmix OOM signature
    on local_rank 0. Gated by TRACK4D360_MEM_PROBE in the caller.
    """
    if not torch.cuda.is_available():
        return
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get(
        "LOCAL_RANK", str(getattr(accelerator, "local_process_index", "?"))
    )
    host = os.uname().nodename
    device = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    print(
        f"[mem-probe] rank={rank} local_rank={local_rank} host={host} where={where} "
        f"step={step} epoch={epoch_id + 1} device={device} "
        f"gpu_used={(total_b - free_b) / gib:.2f}GiB gpu_free={free_b / gib:.2f}GiB "
        f"torch_reserved={torch.cuda.memory_reserved(device) / gib:.2f}GiB "
        f"torch_max_reserved={torch.cuda.max_memory_reserved(device) / gib:.2f}GiB",
        flush=True,
    )


def _validate_graph_signature(accelerator: Accelerator, signature: dict | None) -> None:
    """Fail fast on per-rank graph-path divergence before backward hangs.

    Track4D-360 has rank-local conditioning dropout / camera-only samples.
    Those are safe only if every trainable branch still executes with the same
    forward graph shape across ranks. We gather a tiny int signature before
    backward and print it if ranks diverge, so the run aborts loudly instead of
    hanging inside DDP/FSDP gradient sync.
    """
    if not signature or accelerator.num_processes <= 1:
        return
    if not dist.is_available() or not dist.is_initialized():
        return

    ordered_keys = [
        "camera_only_tier",
        "drop_tracks",
        "drop_dense",
        "track_path",
        "dense_path",
    ]
    local = torch.tensor(
        [int(signature.get(key, 0)) for key in ordered_keys],
        device=accelerator.device,
        dtype=torch.int32,
    )
    gathered = accelerator.gather(local).view(accelerator.num_processes, len(ordered_keys))
    path_key_indices = [
        key_idx for key_idx, key in enumerate(ordered_keys)
        if key.endswith("_path")
    ]
    path_gathered = gathered[:, path_key_indices]
    ref = path_gathered[0]
    if torch.all(path_gathered.eq(ref.unsqueeze(0))):
        return

    if accelerator.is_main_process:
        print("[graph-signature] graph-path mismatch before backward; per-rank signatures:", flush=True)
        for rank_idx in range(accelerator.num_processes):
            parts = [
                f"{key}={int(gathered[rank_idx, key_idx].item())}"
                for key_idx, key in enumerate(ordered_keys)
            ]
            print(f"  rank{rank_idx}: " + ", ".join(parts), flush=True)
        print(
            "[graph-signature] legend: track_path/dense_path "
            "0=off, 1=normal encoder path, 2=null-fallback path, 3=missing-input no-op path. "
            "camera_only_tier/drop_* are data labels and may differ across ranks.",
            flush=True,
        )
    raise RuntimeError(
        "Per-rank graph signature mismatch before backward; "
        "see [graph-signature] lines above."
    )


def launch_training_task(
    accelerator: Accelerator,
    dataset: Optional[torch.utils.data.Dataset],
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    prefetch_factor: int = 2,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        prefetch_factor = args.dataloader_prefetch_factor
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_train_steps = getattr(args, "max_train_steps", None)

        # Auto-calculate save_steps if not explicitly set
        if save_steps is None and getattr(args, 'save_every_n_epochs', 0) > 0:
            # Calculate steps per epoch (optimizer steps, not iterations)
            if train_dataloader is not None:
                dataset_size = len(train_dataloader.dataset)
            elif dataset is not None:
                dataset_size = len(dataset)
            else:
                dataset_size = None

            if dataset_size is not None:
                grad_accum = getattr(args, 'gradient_accumulation_steps', 1)
                num_gpus = accelerator.num_processes
                global_batch_size = num_gpus * grad_accum  # per_device_batch_size is always 1 for Wan
                steps_per_epoch = dataset_size // global_batch_size
                save_steps = steps_per_epoch * args.save_every_n_epochs

                if accelerator.is_main_process:
                    print(f"[Auto-calculated save_steps] Dataset size: {dataset_size}, Global batch size: {global_batch_size}")
                    print(f"[Auto-calculated save_steps] Steps per epoch: {steps_per_epoch}, Save every {args.save_every_n_epochs} epochs -> save_steps: {save_steps}")
    else:
        max_train_steps = None

    # Initialize WandB logger (only on main process)
    wandb_logger = None
    if accelerator.is_main_process and args is not None and getattr(args, 'use_wandb', False):
        # Reuse logger if the caller already provided one (avoids double-init).
        wandb_logger = getattr(model_logger, "wandb_logger", None)

    if accelerator.is_main_process and args is not None and getattr(args, 'use_wandb', False) and wandb_logger is None:
        wandb_config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'save_steps': save_steps,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
        }
        # Add all args to config
        if args is not None:
            wandb_config.update({k: v for k, v in vars(args).items() if not k.startswith('_')})

        wandb_tags = _normalize_wandb_tags(getattr(args, 'wandb_tags', None))
        wandb_dir = os.environ.get("WANDB_DIR") or getattr(args, "output_path", None) or getattr(model_logger, "output_path", None)

        wandb_logger = create_logger(
            enabled=True,
            project=getattr(args, 'wandb_project', 'wanvideo-training'),
            name=getattr(args, 'wandb_name', None),
            entity=getattr(args, 'wandb_entity', None),
            tags=wandb_tags,
            notes=getattr(args, 'wandb_notes', None),
            config=wandb_config,
            dir=wandb_dir,
        )

        model_logger.wandb_logger = wandb_logger

    # Watch model if requested
    if wandb_logger is not None and getattr(args, 'wandb_watch_model', False):
        wandb_logger.watch_model(
            model,
            log='all' if getattr(args, 'wandb_log_gradients', False) else 'parameters',
            log_freq=getattr(args, 'wandb_log_freq', 100)
        )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    if train_dataloader is None:
        if dataset is None:
            raise ValueError("Either `dataset` or `train_dataloader` must be provided.")
        prefetch_factor = prefetch_factor if num_workers > 0 else None
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            collate_fn=lambda x: x[0],
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    raw_dataloader = train_dataloader
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

    print_trainable_parameters_verification(accelerator, model, args)

    # Startup HBM snapshot (full nvidia-smi + process table, every rank), once,
    # right after FSDP/NCCL init and before any training step. If the NIC-local
    # GPU (local_rank 0) already carries tens of GiB of out-of-process memory
    # here, the OOM driver is inter-node comm buffers (or a zombie rank), not a
    # training/checkpoint tensor leak. Gated by TRACK4D360_MEM_PROBE.
    if os.environ.get("TRACK4D360_MEM_PROBE", "0") == "1":
        _print_cuda_oom_snapshot(
            accelerator, where="startup", step=model_logger.num_steps, epoch_id=-1
        )

    torch_profiler = _maybe_create_torch_profiler(accelerator, args, getattr(model_logger, "output_path", None))

    try:
        if torch_profiler is not None:
            torch_profiler.start()
        optimizer.zero_grad()
        reached_max_steps = False
        for epoch_id in range(num_epochs):
            _set_epoch_for_dataloader(raw_dataloader, epoch_id)
            _set_epoch_for_dataloader(dataloader, epoch_id)

            # Create progress bar that tracks optimizer steps (not iterations)
            # Calculate expected optimizer steps for this epoch
            grad_accum = getattr(accelerator, "gradient_accumulation_steps", 1)
            total_batches = len(dataloader)
            expected_opt_steps = total_batches // grad_accum

            # Only render a progress bar on global rank 0. Using local main process
            # prints one bar per node in multi-node runs.
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch_id+1}/{num_epochs}",
                postfix={"opt_step": model_logger.num_steps, "exp_steps_this_epoch": expected_opt_steps},
                disable=not accelerator.is_main_process,
            )

            # Track gradient accumulation manually for DeepSpeed
            is_deepspeed = str(accelerator.state.distributed_type) == "DistributedType.DEEPSPEED"
            deepspeed_accum_counter = 0
            # Track mean loss over micro-steps so the progress bar reflects per-optimizer-step loss.
            micro_loss_sum = None
            micro_loss_count = 0
            micro_metric_sums = {}
            micro_metric_counts = {}
            micro_data_wait_s = 0.0
            micro_forward_s = 0.0
            micro_backward_s = 0.0
            prev_iteration_end_time = time.perf_counter()
            step_total_start_time = None
            micro_step_id = 0

            for data in pbar:
                iteration_start_time = time.perf_counter()
                micro_data_wait_s += max(0.0, iteration_start_time - prev_iteration_end_time)
                if step_total_start_time is None:
                    step_total_start_time = iteration_start_time
                # DeepSpeed handles gradient accumulation internally via config,
                # so skip accelerator.accumulate() context to avoid no_sync conflicts
                use_accumulate_context = not is_deepspeed

                if use_accumulate_context:
                    context_manager = accelerator.accumulate(model)
                else:
                    # No-op context manager for DeepSpeed
                    from contextlib import nullcontext
                    context_manager = nullcontext()

                with context_manager:
                    forward_start_time = time.perf_counter()
                    unwrapped_model = accelerator.unwrap_model(model)
                    if hasattr(unwrapped_model, "set_training_step_context"):
                        unwrapped_model.set_training_step_context(
                            global_step=model_logger.num_steps,
                            micro_step=micro_step_id,
                            process_index=accelerator.process_index,
                        )
                    try:
                        if dataset is not None and getattr(dataset, "load_from_cache", False):
                            loss = model({}, inputs=data)
                        else:
                            loss = model(data)
                    except torch.OutOfMemoryError:
                        _print_cuda_oom_snapshot(
                            accelerator,
                            where="forward",
                            step=model_logger.num_steps,
                            epoch_id=epoch_id,
                        )
                        raise
                    micro_forward_s += time.perf_counter() - forward_start_time
                    micro_step_id += 1
                    if hasattr(unwrapped_model, "consume_latest_graph_signature"):
                        try:
                            graph_signature = unwrapped_model.consume_latest_graph_signature()
                        except Exception:
                            graph_signature = None
                        _validate_graph_signature(accelerator, graph_signature)
                    backward_start_time = time.perf_counter()
                    try:
                        accelerator.backward(loss)
                    except torch.OutOfMemoryError:
                        _print_cuda_oom_snapshot(
                            accelerator,
                            where="backward",
                            step=model_logger.num_steps,
                            epoch_id=epoch_id,
                        )
                        raise
                    micro_backward_s += time.perf_counter() - backward_start_time
                    _debug_print_missing_grads(accelerator, model)
                    if loss is not None:
                        # Accumulate loss in fp32 for stability (per-rank). We reduce across ranks at optimizer step.
                        if micro_loss_sum is None:
                            micro_loss_sum = loss.detach().float()
                        else:
                            micro_loss_sum = micro_loss_sum + loss.detach().float()
                        micro_loss_count += 1
                    # Accumulate optional auxiliary metrics emitted by the training module.
                    if hasattr(unwrapped_model, "consume_latest_log_metrics"):
                        try:
                            micro_metrics = unwrapped_model.consume_latest_log_metrics()
                        except Exception:
                            micro_metrics = None
                        if isinstance(micro_metrics, dict):
                            for key, value in micro_metrics.items():
                                if value is None:
                                    continue
                                if torch.is_tensor(value):
                                    if value.numel() != 1:
                                        continue
                                    v = value.detach().float().to(device=accelerator.device)
                                else:
                                    v = torch.tensor(float(value), device=accelerator.device, dtype=torch.float32)
                                if key not in micro_metric_sums:
                                    micro_metric_sums[key] = v
                                    micro_metric_counts[key] = 1
                                else:
                                    micro_metric_sums[key] = micro_metric_sums[key] + v
                                    micro_metric_counts[key] = micro_metric_counts[key] + 1

                    # Determine if we should step the optimizer
                    # For DDP: use accelerator.sync_gradients (set by accelerator.accumulate())
                    # For DeepSpeed: manually track gradient accumulation steps
                    if is_deepspeed:
                        deepspeed_accum_counter += 1
                        should_step = (deepspeed_accum_counter % grad_accum == 0)
                    else:
                        should_step = accelerator.sync_gradients

                    # Only step optimizer/scheduler when gradients should be synced
                    # (i.e., not during intermediate gradient accumulation micro-steps)
                    if should_step:
                        reduction_start_time = time.perf_counter()
                        # Compute mean loss for this optimizer step:
                        # - average across micro-steps (grad accumulation)
                        # - then average across ranks (data parallel)
                        step_loss = None
                        if micro_loss_sum is not None and micro_loss_count > 0:
                            local_step_loss = micro_loss_sum / float(micro_loss_count)
                            gathered = accelerator.gather(local_step_loss.reshape(1))
                            step_loss = gathered.mean().item()
                        step_extra_metrics = _reduce_optional_scalar_metrics(
                            accelerator,
                            micro_metric_sums,
                            micro_metric_counts,
                        )
                        reduction_s = time.perf_counter() - reduction_start_time

                        optimizer_start_time = time.perf_counter()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        optimizer_s = time.perf_counter() - optimizer_start_time

                        step_total_s = 0.0
                        if step_total_start_time is not None:
                            step_total_s = time.perf_counter() - step_total_start_time
                        timing_metrics = {
                            "timing/data_wait_s": micro_data_wait_s,
                            "timing/forward_s": micro_forward_s,
                            "timing/backward_s": micro_backward_s,
                            "timing/reduction_s": reduction_s,
                            "timing/optimizer_s": optimizer_s,
                            "timing/step_total_s": step_total_s,
                            "timing/microsteps": float(micro_loss_count),
                        }
                        timing_metrics["timing/other_s"] = max(
                            0.0,
                            step_total_s
                            - micro_forward_s
                            - micro_backward_s
                            - reduction_s
                            - optimizer_s,
                        )
                        if accelerator.is_main_process:
                            if step_extra_metrics is None:
                                step_extra_metrics = {}
                            step_extra_metrics = {**step_extra_metrics, **timing_metrics}

                        # Update progress bar with current optimizer step count
                        pbar.set_postfix({
                            "opt_step": model_logger.num_steps,
                            "loss": f"{step_loss:.4f}" if step_loss is not None else "N/A"
                        })

                        # Periodic lightweight HBM probe (every rank). Tracks
                        # whether the out-of-process ~30GiB on local_rank 0 is
                        # present from startup (comm buffers) or grows at the
                        # first checkpoint. Gated; default cadence every 50 steps.
                        if os.environ.get("TRACK4D360_MEM_PROBE", "0") == "1":
                            _probe_every = int(os.environ.get("TRACK4D360_MEM_PROBE_EVERY", "50"))
                            if _probe_every > 0 and model_logger.num_steps % _probe_every == 0:
                                _print_mem_probe(
                                    accelerator,
                                    where="probe",
                                    step=model_logger.num_steps,
                                    epoch_id=epoch_id,
                                )

                        # Optional: latent debug logging (V2V flow-matching).
                        # This is no-grad, but FSDP still requires collective-safe
                        # full-param materialization and rank synchronization around
                        # the rank-0-only decode/write work.
                        debug_latents_due = (
                            args is not None
                            and getattr(args, "debug_latents", False)
                            and int(getattr(args, "debug_latents_every", 1000)) > 0
                            and model_logger.num_steps % int(getattr(args, "debug_latents_every", 1000)) == 0
                            and getattr(args, "task", None) == "sft:v2v"
                        )
                        # FSDP2 (torch 2.6): the rank0-only debug forward is not
                        # hang-safe. FSDP1's summon_full_params no-ops on FSDP2
                        # (modules are FSDPModule, not FullyShardedDataParallel), and
                        # manually unshard()-ing then running the forward on rank0 only
                        # desyncs per-block shard state vs the modules' own
                        # forward hooks (set_reshard_after_forward, which would let us
                        # suppress that, only exists in torch>=2.7). The result is a
                        # deadlock on the next training collective. Skip the diagnostic
                        # under FSDP2 — it still works under FSDP1/DDP, and train +
                        # checkpoint are unaffected. See
                        # doc/track4d-360/2026-05-29-fsdp-control-module-sharding.md.
                        if debug_latents_due and bool(getattr(accelerator, "is_fsdp2", False)):
                            if accelerator.is_main_process:
                                print(
                                    "[debug_latents] Skipped under FSDP2 (not hang-safe on "
                                    "torch 2.6); use FSDP1/DDP for latent debug. Training/"
                                    "checkpointing are unaffected."
                                )
                            debug_latents_due = False

                        if debug_latents_due:
                            from contextlib import nullcontext
                            is_fsdp = str(accelerator.state.distributed_type) == "DistributedType.FSDP"
                            if is_fsdp:
                                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                                summon_ctx = FSDP.summon_full_params(
                                    model, recurse=True, writeback=False, rank0_only=True,
                                )
                            else:
                                summon_ctx = nullcontext()
                            try:
                                with summon_ctx:
                                    if accelerator.is_main_process:
                                        try:
                                            # Free up memory before VAE decoding
                                            torch.cuda.empty_cache()

                                            from .debug_latents import log_flowmatch_v2v_latents

                                            unwrapped = accelerator.unwrap_model(model)
                                            output_root = os.path.join(
                                                model_logger.output_path,
                                                str(getattr(args, "debug_latents_outdir", "debug_latents")),
                                            )
                                            # Handle both list and dict data formats
                                            debug_data = data[0] if isinstance(data, list) else data
                                            with accelerator.autocast():
                                                debug_meta = log_flowmatch_v2v_latents(
                                                    unwrapped,
                                                    debug_data,
                                                    output_root=output_root,
                                                    global_step=model_logger.num_steps,
                                                    fps=int(getattr(args, "debug_latents_fps", 16)),
                                                    quality=int(getattr(args, "debug_latents_quality", 5)),
                                                    max_frames=getattr(args, "debug_latents_max_frames", None),
                                                    tiled_decode=bool(getattr(args, "debug_latents_vae_tiled", True)),
                                                    tile_size=tuple(getattr(args, "debug_latents_vae_tile_size", (30, 52))),
                                                    tile_stride=tuple(getattr(args, "debug_latents_vae_tile_stride", (15, 26))),
                                                )
                                            step_dir = os.path.join(output_root, f"step-{int(model_logger.num_steps):06d}")
                                            combined_rel = None
                                            if isinstance(debug_meta, dict):
                                                combined_rel = ((debug_meta.get("paths") or {}).get("combined_mp4"))
                                            if combined_rel:
                                                print(f"[debug_latents] Wrote {os.path.join(output_root, combined_rel)}")
                                            else:
                                                print(f"[debug_latents] Wrote {step_dir}")
                                            torch.cuda.empty_cache()

                                            # Custom per-module debug visualization hook
                                            if hasattr(unwrapped, "log_debug_visualization"):
                                                try:
                                                    with accelerator.autocast():
                                                        unwrapped.log_debug_visualization(
                                                            debug_data, output_root, model_logger.num_steps,
                                                            fps=int(getattr(args, "debug_latents_fps", 16)),
                                                            quality=int(getattr(args, "debug_latents_quality", 5)),
                                                        )
                                                except Exception as e:
                                                    print(f"[debug_vis] Failed at step {model_logger.num_steps}: {e}")
                                                    traceback.print_exc()
                                        except Exception as e:
                                            print(f"[debug_latents] Failed at step {model_logger.num_steps}: {e}")
                                            traceback.print_exc()
                                            torch.cuda.empty_cache()
                            finally:
                                # In FSDP rank0_only mode, non-main ranks leave the
                                # summon context before rank 0 finishes video I/O.
                                # Hold them here so the next training collective starts
                                # from all ranks together.
                                if is_fsdp:
                                    accelerator.wait_for_everyone()
                                    torch.cuda.empty_cache()
                                else:
                                    torch.cuda.empty_cache()
                        elif (
                            args is not None
                            and getattr(args, "debug_latents", False)
                            and int(getattr(args, "debug_latents_every", 1000)) > 0
                            and model_logger.num_steps % int(getattr(args, "debug_latents_every", 1000)) == 0
                            and accelerator.is_main_process
                        ):
                            print("[debug_latents] Ignoring: --debug_latents currently supports --task sft:v2v only.")

                        # Log metrics and gradients via model_logger (includes WandB if configured)
                        # Use reduced per-step loss if available (matches tqdm), otherwise fall back to local last-micro loss.
                        log_loss = None
                        if step_loss is not None:
                            log_loss = torch.tensor(step_loss, device=accelerator.device)
                        else:
                            log_loss = loss
                        model_logger.on_step_end(
                            accelerator,
                            model,
                            save_steps,
                            loss=log_loss,
                            epoch_id=epoch_id,
                            extra_metrics=step_extra_metrics,
                        )
                        if torch_profiler is not None:
                            torch_profiler.step()

                        if max_train_steps is not None and model_logger.num_steps >= int(max_train_steps):
                            reached_max_steps = True
                            if accelerator.is_main_process:
                                print(f"\n[Training] Reached max_train_steps={int(max_train_steps)}. Stopping training.")
                            break

                        # Reset micro-step loss accumulator after the optimizer step.
                        micro_loss_sum = None
                        micro_loss_count = 0
                        micro_metric_sums = {}
                        micro_metric_counts = {}
                        micro_data_wait_s = 0.0
                        micro_forward_s = 0.0
                        micro_backward_s = 0.0
                        step_total_start_time = None
                        micro_step_id = 0

                prev_iteration_end_time = time.perf_counter()

            if reached_max_steps:
                break

            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)

        model_logger.on_training_end(accelerator, model, save_steps)
    finally:
        if torch_profiler is not None:
            torch_profiler.stop()
        # Finish WandB logging
        if wandb_logger:
            wandb_logger.finish()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
