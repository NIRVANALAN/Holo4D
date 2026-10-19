import os, torch
from accelerate import Accelerator
import gc


def _cleanup_cuda_after_checkpoint():
    if not torch.cuda.is_available():
        gc.collect()
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    gc.collect()


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        wandb_logger=None,
        log_freq=100,
        wandb_log_gradients: bool = False,
        wandb_log_parameters: bool = False,
        max_checkpoints_to_keep: int = 5,
        allow_existing_checkpoints: bool = False,
        initial_step: int = 0,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = initial_step
        self.wandb_logger = wandb_logger
        self.log_freq = log_freq
        self.wandb_log_gradients = wandb_log_gradients
        self.wandb_log_parameters = wandb_log_parameters
        self.max_checkpoints_to_keep = max_checkpoints_to_keep

        # Validate output directory for existing checkpoints
        if not allow_existing_checkpoints:
            self._validate_output_directory()

    def _validate_output_directory(self):
        """
        Validate that the output directory doesn't contain existing checkpoints.
        This prevents accidentally mixing old and new training runs.
        """
        if not os.path.exists(self.output_path):
            return

        import glob

        # Check for checkpoint files
        checkpoint_patterns = [
            os.path.join(self.output_path, "*.safetensors"),
            os.path.join(self.output_path, "*.pt"),
            os.path.join(self.output_path, "*.pth"),
            os.path.join(self.output_path, "*.ckpt"),
        ]

        existing_checkpoints = []
        for pattern in checkpoint_patterns:
            existing_checkpoints.extend(glob.glob(pattern))

        if existing_checkpoints:
            checkpoint_list = "\n  • ".join([os.path.basename(f) for f in existing_checkpoints[:10]])
            if len(existing_checkpoints) > 10:
                checkpoint_list += f"\n  ... and {len(existing_checkpoints) - 10} more"

            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Output directory contains existing checkpoint files!\n"
                f"{'='*80}\n"
                f"Directory: {self.output_path}\n"
                f"Found {len(existing_checkpoints)} checkpoint file(s):\n"
                f"  • {checkpoint_list}\n\n"
                f"This likely means you're accidentally running into a folder with old checkpoints.\n"
                f"To prevent mixing old and new training runs, please:\n"
                f"  1. Use a new output directory, OR\n"
                f"  2. Clean the existing directory, OR\n"
                f"  3. Set --allow_existing_checkpoints if you're intentionally resuming\n"
                f"{'='*80}\n"
            )
            raise RuntimeError(error_msg)

    def _check_v2v_parameters(self, unwrapped_model, state_dict):
        """
        Check that V2V-specific parameters like domain_embedding are saved in checkpoints.
        This is crucial for LoRA training where these parameters are trainable but not LoRA weights.
        """
        # Check if the model has a V2V dit (with domain_embedding)
        model_to_check = None
        if hasattr(unwrapped_model, 'pipe') and hasattr(unwrapped_model.pipe, 'dit'):
            model_to_check = unwrapped_model.pipe.dit
        elif hasattr(unwrapped_model, 'dit'):
            model_to_check = unwrapped_model.dit

        if model_to_check is None:
            return

        # Check if domain_embedding exists in the original model
        has_domain_embedding = False
        for name, _ in model_to_check.named_parameters():
            if 'domain_embedding' in name:
                has_domain_embedding = True
                break

        if not has_domain_embedding:
            return

        # If domain_embedding exists in model, assert it's in the saved state dict
        domain_embedding_found = False
        for key in state_dict.keys():
            if 'domain_embedding' in key:
                domain_embedding_found = True
                print(f"  ✓ V2V parameter saved: {key}")
                break

        assert domain_embedding_found, (
            "CRITICAL: domain_embedding exists in V2V model but was NOT saved in checkpoint! "
            "This will cause inference to fail with random initialization. "
            "Ensure domain_embedding is in trainable_models list."
        )


    def log_to_wandb(self, accelerator: Accelerator, model: torch.nn.Module, loss=None, epoch_id=None, extra_metrics=None):
        """Log metrics and gradients to WandB"""
        if self.wandb_logger is None or not accelerator.is_main_process:
            return

        # Log loss if provided
        if loss is not None:
            self.wandb_logger.log_metrics({
                "train/loss": loss,
            }, step=self.num_steps, commit=False)

        # Log epoch if provided
        if epoch_id is not None:
            self.wandb_logger.log_metrics({
                "train/epoch": epoch_id,
            }, step=self.num_steps, commit=False)

        if extra_metrics:
            metrics_to_log = {}
            for key, value in extra_metrics.items():
                if value is None:
                    continue
                metric_key = key if str(key).startswith("train/") else f"train/{key}"
                metrics_to_log[metric_key] = float(value)
            if metrics_to_log:
                self.wandb_logger.log_metrics(metrics_to_log, step=self.num_steps, commit=False)

        # Log gradients/parameters at specified frequency
        if self.num_steps % self.log_freq == 0 and (self.wandb_log_gradients or self.wandb_log_parameters):
            # Try to get the actual model (unwrap if needed)
            unwrapped_model = accelerator.unwrap_model(model)

            target_model = unwrapped_model.pipe.dit if hasattr(unwrapped_model, 'pipe') and hasattr(unwrapped_model.pipe, 'dit') else unwrapped_model

            if self.wandb_log_gradients:
                self.wandb_logger.log_gradients(
                    target_model,
                    step=self.num_steps,
                    prefix="gradients"
                )

            if self.wandb_log_parameters:
                self.wandb_logger.log_model_parameters(
                    target_model,
                    step=self.num_steps,
                    prefix="parameters"
                )

        # Commit all metrics for this step
        self.wandb_logger.log_metrics({}, step=self.num_steps, commit=True)

    def _cleanup_old_checkpoints(self, accelerator: Accelerator, pattern_prefix: str = "step-"):
        """
        Remove old checkpoints, keeping only the most recent N checkpoints.

        Step checkpoints on milestone boundaries are never deleted by this
        rolling cleanup. The milestone interval defaults to 5000 steps and can
        be overridden with TRACK4D360_CHECKPOINT_MILESTONE_STEPS. For example,
        with max_checkpoints_to_keep=5, save_steps=500, and a 2500-step
        milestone interval, cleanup keeps every step-2500/5000/... checkpoint
        plus the latest five non-milestone step checkpoints.

        Args:
            accelerator: Accelerator instance
            pattern_prefix: Prefix of checkpoint files to clean up (e.g., "step-" or "epoch-")
        """
        if not accelerator.is_main_process:
            return

        if self.max_checkpoints_to_keep <= 0:
            return

        import glob
        import re

        # Find all checkpoint files matching the pattern
        pattern = os.path.join(self.output_path, f"{pattern_prefix}*.safetensors")
        checkpoint_files = glob.glob(pattern)

        if len(checkpoint_files) <= self.max_checkpoints_to_keep:
            return

        # Extract step/epoch numbers and sort
        checkpoints_with_nums = []
        for ckpt_path in checkpoint_files:
            basename = os.path.basename(ckpt_path)
            # Extract number from "step-12345.safetensors" or "epoch-5.safetensors"
            match = re.search(rf"{pattern_prefix}(\d+)\.safetensors", basename)
            if match:
                num = int(match.group(1))
                checkpoints_with_nums.append((num, ckpt_path))

        # Sort by number (ascending)
        checkpoints_with_nums.sort(key=lambda x: x[0])

        if pattern_prefix == "step-":
            milestone_steps = int(os.environ.get("TRACK4D360_CHECKPOINT_MILESTONE_STEPS", "5000"))
            checkpoints_with_nums = [
                (num, path)
                for num, path in checkpoints_with_nums
                if milestone_steps <= 0 or num % milestone_steps != 0
            ]

        # Delete all but the most recent N
        num_to_delete = len(checkpoints_with_nums) - self.max_checkpoints_to_keep
        if num_to_delete > 0:
            for i in range(num_to_delete):
                ckpt_num, ckpt_path = checkpoints_with_nums[i]
                try:
                    os.remove(ckpt_path)
                    print(f"[ModelLogger] Removed old checkpoint: {os.path.basename(ckpt_path)}")
                except Exception as e:
                    print(f"[ModelLogger] Warning: Failed to remove {ckpt_path}: {e}")

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, loss=None, epoch_id=None, extra_metrics=None):
        """
        Called at the end of each training step.

        Args:
            accelerator: Accelerator instance
            model: Training model
            save_steps: Save checkpoint every N steps (None = save per epoch)
            loss: Current step loss for logging (optional)
            epoch_id: Current epoch ID for logging (optional)
            extra_metrics: Additional scalar metrics for logging (optional)
        """
        # Log to WandB before incrementing step counter (so logs align with current step)
        self.log_to_wandb(
            accelerator,
            model,
            loss=loss.item() if loss is not None else None,
            epoch_id=epoch_id,
            extra_metrics=extra_metrics,
        )

        self.num_steps += 1
        # One-shot smoke save at step 2 to validate the save-OOM fix without
        # burning ~200 steps of compute. Remove the `self.num_steps == 2` clause
        # once a green save has been observed in the run logs.
        smoke_save_step = 2
        if save_steps is not None and (
            self.num_steps % save_steps == 0 or self.num_steps == smoke_save_step
        ):
            if accelerator.is_main_process:
                epoch_str = f" (Epoch {epoch_id+1})" if epoch_id is not None else ""
                tag = " [SMOKE]" if self.num_steps == smoke_save_step and self.num_steps % save_steps != 0 else ""
                print(f"\n[Checkpoint] Saving at optimizer step {self.num_steps}{epoch_str}{tag}")
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            # Clean up old checkpoints after saving
            self._cleanup_old_checkpoints(accelerator, pattern_prefix="step-")
            # Keep all ranks aligned before the next forward/backward collectives.
            accelerator.wait_for_everyone()


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        # torch.cuda.empty_cache()
        # gc.collect()

        # Set device to avoid NCCL warning
        if torch.cuda.is_available():
            torch.cuda.set_device(accelerator.local_process_index)

        # Save only model weights (no optimizer states) with DeepSpeed ZeRO
        if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin is not None:
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # Get model state dict using accelerator (handles ZeRO gathering)
                unwrapped_model = accelerator.unwrap_model(model)

                # Use DeepSpeed's state_dict method to gather ZeRO shards (model only, no optimizer)
                state_dict = unwrapped_model.module.state_dict() if hasattr(unwrapped_model, 'module') else unwrapped_model.state_dict()

                # Apply transformations
                state_dict = unwrapped_model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                state_dict = self.state_dict_converter(state_dict)
                # Cast fp32 -> bf16 for compact safetensors (~2x smaller). FSDP
                # mixed_precision keeps fp32 master params; the saved DiT runs
                # in bf16 anyway, so the fp32 master is wasted bytes on disk.
                state_dict = {
                    k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.bfloat16 else v)
                    for k, v in state_dict.items()
                }

                # Assert domain_embedding is saved for V2V models
                # self._check_v2v_parameters(unwrapped_model, state_dict)

                # Save as safetensors
                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
                accelerator.save(state_dict, path, safe_serialization=True)

                del state_dict
                # gc.collect()

                # Clean up old epoch checkpoints after saving
                self._cleanup_old_checkpoints(accelerator, pattern_prefix="epoch-")
        else:
            # Non-DeepSpeed fallback
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            # accelerator.get_state_dict() unconditionally enters FSDP
            # FULL_STATE_DICT mode (rank0_only=True, offload_to_cpu=True)
            # regardless of `fsdp_state_dict_type` yaml — see accelerate
            # accelerator.py:3631-3637. Collective: every rank must enter;
            # rank>0 gets an empty dict.
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                state_dict = unwrapped_model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                state_dict = self.state_dict_converter(state_dict)
                # Cast fp32 -> bf16 for compact safetensors (~2x smaller). FSDP
                # mixed_precision keeps fp32 master params; the saved DiT runs
                # in bf16 anyway, so the fp32 master is wasted bytes on disk.
                state_dict = {
                    k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.bfloat16 else v)
                    for k, v in state_dict.items()
                }

                # Assert domain_embedding is saved for V2V models
                # self._check_v2v_parameters(unwrapped_model, state_dict)

                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
                accelerator.save(state_dict, path, safe_serialization=True)

                # Clean up old epoch checkpoints after saving
                self._cleanup_old_checkpoints(accelerator, pattern_prefix="epoch-")
            del state_dict
            gc.collect()

        # Keep non-main ranks from entering the next collective while rank 0 is
        # still flushing checkpoint data to the parallel filesystem. The process
        # group timeout is configured by the launcher for long multi-node saves.
        accelerator.wait_for_everyone()


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        # Free cached HBM blocks before the entry barrier. Without this, NCCL's
        # barrier workspace alloc fails with "Cuda failure 2 'out of memory'"
        # at the first save when HBM is ~96/96 GB (training peak + activation
        # tail under FSDP FULL_SHARD + grad-ckpt at 14B). The caching allocator
        # holds blocks that are reusable but not free-to-driver; empty_cache()
        # returns them so cudaMalloc can hand a fresh slab to NCCL.
        _cleanup_cuda_after_checkpoint()

        # Set device to avoid NCCL warning
        if torch.cuda.is_available():
            torch.cuda.set_device(accelerator.local_process_index)

        # Save only model weights (no optimizer states) with DeepSpeed ZeRO
        if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin is not None:
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # Get model state dict using accelerator (handles ZeRO gathering)
                unwrapped_model = accelerator.unwrap_model(model)

                # Use DeepSpeed's state_dict method to gather ZeRO shards (model only, no optimizer)
                state_dict = unwrapped_model.module.state_dict() if hasattr(unwrapped_model, 'module') else unwrapped_model.state_dict()

                # Apply transformations
                state_dict = unwrapped_model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                state_dict = self.state_dict_converter(state_dict)
                # Cast fp32 -> bf16 for compact safetensors (~2x smaller). FSDP
                # mixed_precision keeps fp32 master params; the saved DiT runs
                # in bf16 anyway, so the fp32 master is wasted bytes on disk.
                state_dict = {
                    k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.bfloat16 else v)
                    for k, v in state_dict.items()
                }

                # Assert domain_embedding is saved for V2V models
                # self._check_v2v_parameters(unwrapped_model, state_dict)

                # Save as safetensors
                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, file_name)
                accelerator.save(state_dict, path, safe_serialization=True)

                del state_dict
                # gc.collect()
        else:
            # Non-DeepSpeed fallback
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            # See on_epoch_end(): get_state_dict already enters FSDP
            # FULL_STATE_DICT collective; every rank must enter.
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                state_dict = unwrapped_model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                state_dict = self.state_dict_converter(state_dict)
                # Cast fp32 -> bf16 for compact safetensors (~2x smaller). FSDP
                # mixed_precision keeps fp32 master params; the saved DiT runs
                # in bf16 anyway, so the fp32 master is wasted bytes on disk.
                state_dict = {
                    k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.bfloat16 else v)
                    for k, v in state_dict.items()
                }

                # Assert domain_embedding is saved for V2V models
                # self._check_v2v_parameters(unwrapped_model, state_dict)

                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, file_name)
                accelerator.save(state_dict, path, safe_serialization=True)
            del state_dict
            _cleanup_cuda_after_checkpoint()

        _cleanup_cuda_after_checkpoint()

        # Keep non-main ranks from entering the next collective while rank 0 is
        # still flushing checkpoint data to the parallel filesystem. The process
        # group timeout is configured by the launcher for long multi-node saves.
        accelerator.wait_for_everyone()
        _cleanup_cuda_after_checkpoint()
