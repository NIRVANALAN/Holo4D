import glob
import json
import os
import torch
from ..core import ModelConfig, load_state_dict
from ..utils.controlnet import ControlNetInput
from peft import LoraConfig, inject_adapter_in_model


def _is_log_process():
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    except Exception:
        return True


def _rank0_print(*args, **kwargs):
    if _is_log_process():
        print(*args, **kwargs)


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._first_checkpoint_save = True
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        # Normalize FSDP/DDP wrapper segments so the names match what
        # `accelerator.get_state_dict(model)` returns for the saved checkpoint.
        # Why: under FSDP1 with TRANSFORMER_BASED_WRAP (each DiTBlock is its
        # own FSDP unit), `self.named_parameters()` traverses INTO the FSDP
        # wrapper and emits names with the `_fsdp_wrapped_module.` segment
        # baked in (e.g. `pipe.dit.blocks.0._fsdp_wrapped_module.self_attn.q.weight`).
        # Meanwhile `accelerator.get_state_dict(model)` always uses
        # FULL_STATE_DICT for FSDP and STRIPS that segment, returning clean
        # `pipe.dit.blocks.0.self_attn.q.weight`. The two name spaces don't
        # match, so the `name in trainable_param_names` filter in
        # `export_trainable_state_dict` silently drops EVERY DiTBlock param
        # (self_attn / ffn), and you end up with a checkpoint that has only
        # the un-wrapped track / control_adapter / domain_embedding /
        # control_patch_embedding keys. Loss looks fine during training (the
        # gradients flow), but the persisted file is missing the entire DiT
        # fine-tune. Same idea applies to DDP's `module.` prefix and
        # torch.compile's `_orig_mod.` prefix; strip them all.
        _SEGMENTS_TO_STRIP = ("._fsdp_wrapped_module", "._checkpoint_wrapped_module", "._orig_mod", ".module")
        names = set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            for seg in _SEGMENTS_TO_STRIP:
                name = name.replace(seg + ".", ".")
                if name.endswith(seg):
                    name = name[: -len(seg)]
            names.add(name)
        return names

    def filter_exported_trainable_state_dict(self, state_dict):
        return state_dict
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def _resolve_model_state_key(self, model_state_keys, key):
        """Best-effort key resolver for checkpoint entries saved with optional prefixes."""
        if key in model_state_keys:
            return key
        candidates = [key]
        for prefix in ("pipe.dit.", "dit.", "module.", "_forward_module.", "model.", "base_model.model."):
            if key.startswith(prefix):
                candidates.append(key[len(prefix):])
        if "." in key:
            candidates.append(key.split(".", 1)[1])
        for cand in candidates:
            if cand in model_state_keys:
                return cand
        return key


    def _adapt_non_lora_params_for_model(self, model, non_lora_params):
        """
        Adapt non-LoRA checkpoint parameters for loading into the current model.

        Special handling for shape mismatches:
        - domain_embedding row-count mismatch:
        - copy overlapping rows from checkpoint (e.g., source_rgb + target_track),
        - keep newly-added modality rows (e.g., target_track_video) as current init.
        - control_patch_embedding / patch_embedding mismatch:
        - copy overlapping tensor region,
        - keep newly-added channels initialized by current model.
        """
        if not non_lora_params:
            return non_lora_params

        model_state = model.state_dict()
        model_state_keys = set(model_state.keys())

        remapped = {}
        for key, value in non_lora_params.items():
            target_key = self._resolve_model_state_key(model_state_keys, key)
            remapped[target_key] = value

        ckpt_domain_key = None
        for key, value in remapped.items():
            if key.endswith("domain_embedding") and torch.is_tensor(value):
                ckpt_domain_key = key
                break
        model_domain_key = None
        for key in model_state_keys:
            if key.endswith("domain_embedding") and torch.is_tensor(model_state[key]):
                model_domain_key = key
                break

        if ckpt_domain_key is not None and model_domain_key is not None:
            ckpt_domain = remapped[ckpt_domain_key]
            model_domain = model_state[model_domain_key]
            if tuple(ckpt_domain.shape) != tuple(model_domain.shape):
                if (
                    ckpt_domain.ndim == 2
                    and model_domain.ndim == 2
                    and int(ckpt_domain.shape[1]) == int(model_domain.shape[1])
                ):
                    merged = model_domain.detach().clone().to(dtype=ckpt_domain.dtype)
                    rows_to_copy = min(int(ckpt_domain.shape[0]), int(model_domain.shape[0]))
                    merged[:rows_to_copy] = ckpt_domain[:rows_to_copy].to(dtype=merged.dtype)
                    if ckpt_domain_key != model_domain_key and ckpt_domain_key in remapped:
                        remapped.pop(ckpt_domain_key, None)
                    remapped[model_domain_key] = merged
                    print(
                        f"  ℹ️ Partial domain_embedding load: ckpt={tuple(ckpt_domain.shape)} -> "
                        f"model={tuple(model_domain.shape)}, copied first {rows_to_copy} row(s), "
                        "kept newly-added rows initialized."
                    )
                else:
                    remapped.pop(ckpt_domain_key, None)
                    print(
                        f"  ⚠ Skipping incompatible domain_embedding from checkpoint: "
                        f"ckpt={tuple(ckpt_domain.shape)} vs model={tuple(model_domain.shape)}"
                    )

        # Partial-load control_patch_embedding / patch_embedding tensors when shape differs.
        # This keeps compatibility across different control input channel configs and
        # feedback-training patch-embedding widening.
        for key, value in list(remapped.items()):
            if not torch.is_tensor(value):
                continue
            if ("control_patch_embedding" not in key and "patch_embedding" not in key):
                continue
            if key not in model_state or not torch.is_tensor(model_state[key]):
                continue

            model_tensor = model_state[key]
            ckpt_tensor = value
            if tuple(ckpt_tensor.shape) == tuple(model_tensor.shape):
                continue

            if ckpt_tensor.ndim != model_tensor.ndim:
                remapped.pop(key, None)
                print(
                    f"  ⚠ Skipping incompatible {key} from checkpoint: "
                    f"ckpt={tuple(ckpt_tensor.shape)} vs model={tuple(model_tensor.shape)}"
                )
                continue

            merged = model_tensor.detach().clone().to(dtype=ckpt_tensor.dtype)
            copy_shape = [min(int(a), int(b)) for a, b in zip(ckpt_tensor.shape, model_tensor.shape)]
            copy_slices = tuple(slice(0, n) for n in copy_shape)
            merged[copy_slices] = ckpt_tensor[copy_slices].to(dtype=merged.dtype)
            remapped[key] = merged
            print(
                f"  ℹ️ Partial {key} load: ckpt={tuple(ckpt_tensor.shape)} -> "
                f"model={tuple(model_tensor.shape)}, copied overlap={tuple(copy_shape)}, "
                "kept remaining model init."
            )

        return remapped


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict_keys = set(state_dict.keys())
        missing_trainable = sorted(trainable_param_names - state_dict_keys)
        if missing_trainable:
            preview = ", ".join(missing_trainable[:20])
            if len(missing_trainable) > 20:
                preview += f", ... (+{len(missing_trainable) - 20} more)"
            raise RuntimeError(
                "Refusing to save an incomplete training checkpoint: "
                f"{len(missing_trainable)} trainable parameter(s) are missing from "
                f"accelerator state_dict. Missing trainables: {preview}. "
                "This usually means wrapper prefixes such as FSDP/DDP names were "
                "not normalized consistently before checkpoint export."
            )

        # Add V2V-specific parameters that should always be saved (whether trainable or not)
        # These are critical for V2V inference and may be trainable (when in trainable_models)
        # or non-trainable (when fine-tuning other parts). This ensures they're always saved.
        v2v_params_to_save = []
        for name in state_dict.keys():
            if 'domain_embedding' in name or 'control_patch_embedding' in name:
                v2v_params_to_save.append(name)

        # Include both trainable and V2V-specific params
        all_params_to_save = trainable_param_names.union(set(v2v_params_to_save))
        state_dict = {name: param for name, param in state_dict.items() if name in all_params_to_save}
        state_dict = self.filter_exported_trainable_state_dict(state_dict)

        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_

        # Print summary only on first checkpoint save
        if self._first_checkpoint_save:
            self._first_checkpoint_save = False
            self._print_checkpoint_summary(state_dict)

        return state_dict

    def _print_checkpoint_summary(self, state_dict):
        """Print a compact summary of checkpoint parameters.

        Mirrors the grouping format of
        ``holo4d_runtime/diffusion/model_printing.py::print_trainable_parameters_verification``:
        block-prefixed keys are collapsed into a single ``*.blocks.* — N blocks
        [a..b]`` group with a per-block param count and submodule breakdown
        from one example block; non-block keys are grouped by their submodule
        prefix and shown with a top-K leaf list.
        """
        keys = sorted(state_dict.keys())
        total_params = 0

        # First pass: collect per-key numel + group by leaf prefix.
        # block_roots[root_key] = {"blocks": {bidx: [(rel_leaf_name, numel)]},
        #                          "modules": {bidx: {submod: numel_sum}}}
        # non_block[group_prefix] = [(leaf_name, numel), ...]
        block_roots: dict[str, dict[str, dict]] = {}
        non_block: dict[str, list[tuple[str, int]]] = {}

        for key in keys:
            tensor = state_dict[key]
            try:
                numel = int(tensor.numel()) if hasattr(tensor, "numel") else 0
            except Exception:
                numel = 0
            total_params += numel

            parts = key.split(".")
            if "blocks" in parts:
                idx = parts.index("blocks")
                if len(parts) > idx + 2 and parts[idx + 1].lstrip("-").isdigit():
                    block_idx = parts[idx + 1]
                    submod = parts[idx + 2]
                    root_key = ".".join(parts[: idx + 1]) + ".*"  # e.g. blocks.* or dit.blocks.*
                    rb = block_roots.setdefault(root_key, {"blocks": {}, "modules": {}})
                    rel_leaf = ".".join(parts[idx + 2:])  # e.g. self_attn.q.weight
                    rb["blocks"].setdefault(block_idx, []).append((rel_leaf, numel))
                    rb["modules"].setdefault(block_idx, {})
                    rb["modules"][block_idx][submod] = rb["modules"][block_idx].get(submod, 0) + numel
                    continue
            # Non-block grouping: take first ≤3 dot-segments as the prefix.
            if len(parts) >= 3:
                prefix = ".".join(parts[:3])
            else:
                prefix = ".".join(parts[:2]) if len(parts) >= 2 else key
            non_block.setdefault(prefix, []).append((key, numel))

        _rank0_print(
            f"\n[Checkpoint] Saving {len(keys)} parameters "
            f"(total {total_params:,} ≈ {total_params / 1e6:.2f}M):"
        )
        _rank0_print(
            f"[Checkpoint] Parameter groups: "
            f"{len(block_roots)} block group(s), {len(non_block)} non-block group(s)"
        )

        # Block roots — one summary line per *.blocks.* group + example submodule breakdown.
        for root_key in sorted(block_roots.keys()):
            blocks_map = block_roots[root_key]["blocks"]
            modules_map = block_roots[root_key]["modules"]
            block_idx_strs = list(blocks_map.keys())
            numeric = sorted({int(b) for b in block_idx_strs})
            if numeric and numeric == list(range(numeric[0], numeric[-1] + 1)):
                range_str = f"[{numeric[0]}..{numeric[-1]}]"
            elif len(numeric) <= 5:
                range_str = f"[{', '.join(map(str, numeric))}]"
            else:
                range_str = f"[{numeric[0]}..{numeric[-1]}]"

            example_idx = str(numeric[0]) if numeric else block_idx_strs[0]
            example_total = sum(n for _, n in blocks_map[example_idx])
            num_blocks = len(block_idx_strs)
            total_across = sum(sum(n for _, n in v) for v in blocks_map.values())

            _rank0_print(f"\n  📦 {root_key} — {num_blocks} blocks {range_str}:")
            _rank0_print(f"     Params per block: {example_total:,} × {num_blocks} = {total_across:,}")

            module_totals = modules_map.get(example_idx, {})
            if module_totals:
                _rank0_print(f"     Submodules (example from block {example_idx}):")
                for submod, m_total in sorted(module_totals.items(), key=lambda kv: (-kv[1], kv[0])):
                    _rank0_print(f"       • {submod}: {m_total:,}")

        # Non-block groups — one summary block per group, top-K leaves.
        TOP_K = 8
        for prefix in sorted(non_block.keys()):
            entries = sorted(non_block[prefix], key=lambda kv: (-kv[1], kv[0]))
            group_total = sum(n for _, n in entries)
            _rank0_print(f"\n  📦 {prefix}:")
            _rank0_print(f"     Total params: {group_total:,}")
            shown = 0
            for full_name, numel in entries:
                if shown >= TOP_K:
                    break
                short = full_name.replace(prefix + ".", "├─ ", 1) if full_name.startswith(prefix + ".") else full_name
                _rank0_print(f"         {short}: {numel:,}")
                shown += 1
            if len(entries) > shown:
                _rank0_print(f"         … ({len(entries) - shown} more)")

        if os.environ.get("TRACK4D360_CHECKPOINT_VERBOSE_KEYS", "0") == "1":
            _rank0_print("\n[Checkpoint] Full saved parameter key list:")
            for key in keys:
                _rank0_print(f"  • {key}")
        else:
            _rank0_print(
                "\n[Checkpoint] Full key list suppressed "
                "(set TRACK4D360_CHECKPOINT_VERBOSE_KEYS=1 to print every saved tensor key)."
            )
        _rank0_print()
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data
    
    def parse_vram_config(self, fp8=False, offload=False, device="cpu"):
        if fp8:
            return {
                "offload_dtype": torch.float8_e4m3fn,
                "offload_device": device,
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": device,
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
            }
        elif offload:
            return {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
                "clear_parameters": True,
            }
        else:
            return {}
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, fp8_models=None, offload_models=None, device="cpu"):
        fp8_models = [] if fp8_models is None else fp8_models.split(",")
        offload_models = [] if offload_models is None else offload_models.split(",")
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            for path in model_paths:
                vram_config = self.parse_vram_config(
                    fp8=path in fp8_models,
                    offload=path in offload_models,
                    device=device
                )
                model_configs.append(ModelConfig(path=path, **vram_config))
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for model_id_with_origin_path in model_id_with_origin_paths:
                model_id, origin_file_pattern = model_id_with_origin_path.split(":", 1)
                vram_config = self.parse_vram_config(
                    fp8=model_id_with_origin_path in fp8_models,
                    offload=model_id_with_origin_path in offload_models,
                    device=device
                )
                if os.path.isabs(origin_file_pattern):
                    resolved_paths = glob.glob(origin_file_pattern)
                    if len(resolved_paths) == 0 and os.path.exists(origin_file_pattern):
                        resolved_paths = [origin_file_pattern]
                    if len(resolved_paths) == 0:
                        raise FileNotFoundError(
                            f"Local model path pattern did not match any files: {origin_file_pattern}"
                        )
                    model_path = resolved_paths if len(resolved_paths) > 1 else resolved_paths[0]
                    model_configs.append(ModelConfig(path=model_path, **vram_config))
                else:
                    model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern, **vram_config))
        return model_configs
    
    
    def expand_trainable_patterns(self, pipe, trainable_models):
        """
        Expand wildcard patterns in trainable_models.

        Supports patterns like:
        - "dit.blocks.*.self_attn" -> expands to all self_attn layers
        - "dit.domain_embedding" -> single parameter
        - "dit.control_adapter" -> single module
        """
        if trainable_models is None:
            return []

        patterns = [pattern.strip() for pattern in trainable_models.split(",") if pattern.strip()]
        expanded = []

        for pattern in patterns:
            # Check if pattern contains wildcard
            if "*" in pattern:
                # Handle dit.blocks.*.module_name pattern
                if ".blocks.*." in pattern:
                    # Extract the base and suffix
                    parts = pattern.split(".blocks.*.")
                    if len(parts) == 2:
                        base_model, suffix = parts
                        # Get the model (e.g., "dit")
                        if hasattr(pipe, base_model):
                            model = getattr(pipe, base_model)
                            if hasattr(model, "blocks"):
                                num_blocks = len(model.blocks)
                                for i in range(num_blocks):
                                    expanded.append(f"{base_model}.blocks.{i}.{suffix}")
                                print(f"Expanded '{pattern}' to {num_blocks} modules")
                            else:
                                print(f"Warning: {base_model} has no 'blocks' attribute")
                        else:
                            print(f"Warning: pipe has no '{base_model}' attribute")
                else:
                    print(f"Warning: Unsupported wildcard pattern: {pattern}")
                    expanded.append(pattern)
            else:
                expanded.append(pattern)

        return expanded

    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None, lora_alpha=None,
        preset_lora_path=None, preset_lora_model=None,
        freeze_lora_after_load=False,
        task="sft",
        train_sigma_shift=None,
    ):
        # Scheduler
        scheduler_kwargs = {}
        if train_sigma_shift is not None:
            scheduler_kwargs["shift"] = train_sigma_shift
        pipe.scheduler.set_timesteps(1000, training=True, **scheduler_kwargs)

        # Freeze untrainable models
        # Expand wildcard patterns (e.g., dit.blocks.*.self_attn)
        trainable_model_list = self.expand_trainable_patterns(pipe, trainable_models)
        pipe.freeze_except(trainable_model_list)

        # Preset LoRA
        if preset_lora_path is not None:
            pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)
        
        # FP8
        # FP8 relies on a model-specific memory management scheme.
        # It is delegated to the subclass.
        
        # Add LoRA to the base models (supports comma-separated list for multiple models)
        if lora_base_model is not None and not task.endswith(":data_process"):
            lora_base_models = [m.strip() for m in lora_base_model.split(",")]
            for base_model_name in lora_base_models:
                if (not hasattr(pipe, base_model_name)) or getattr(pipe, base_model_name) is None:
                    print(f"No {base_model_name} model in the pipeline. Skipping LoRA patch for this model.")
                    continue
                model = self.add_lora_to_model(
                    getattr(pipe, base_model_name),
                    target_modules=lora_target_modules.split(","),
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    upcast_dtype=pipe.torch_dtype,
                )
                if lora_checkpoint is not None:
                    state_dict = load_state_dict(lora_checkpoint)

                    # Separate LoRA weights from non-LoRA parameters (e.g., domain_embedding, control_patch_embedding)
                    # Non-LoRA params need to be loaded separately to ensure they're restored from checkpoint
                    non_lora_params = {}
                    for key, value in state_dict.items():
                        if 'lora_A' not in key and 'lora_B' not in key:
                            non_lora_params[key] = value

                    # Load LoRA weights (with key mapping for PEFT format)
                    lora_state_dict = self.mapping_lora_state_dict(state_dict)
                    load_result = model.load_state_dict(lora_state_dict, strict=False)
                    _rank0_print(f"LoRA checkpoint loaded for {base_model_name}: {lora_checkpoint}, total {len(lora_state_dict)} keys")
                    if len(load_result[1]) > 0:
                        _rank0_print(f"Warning, LoRA key mismatch for {base_model_name}! Unexpected keys: {load_result[1]}")

                    # Load non-LoRA parameters separately (critical for V2V models)
                    if len(non_lora_params) > 0:
                        non_lora_params = self._adapt_non_lora_params_for_model(model, non_lora_params)
                        # Get device from model's first parameter
                        model_device = next(model.parameters()).device
                        non_lora_params = {k: v.to(device=model_device, dtype=pipe.torch_dtype) for k, v in non_lora_params.items()}
                        missing_keys, unexpected_keys = model.load_state_dict(non_lora_params, strict=False)

                        # Report loaded parameters
                        loaded_params = [k for k in non_lora_params.keys() if k not in missing_keys]
                        if len(loaded_params) > 0:
                            _rank0_print(
                                f"  ✓ Loaded {len(loaded_params)} non-LoRA parameters from checkpoint: "
                                + ", ".join(loaded_params[:6])
                                + (" ..." if len(loaded_params) > 6 else "")
                            )
                if freeze_lora_after_load:
                    frozen_lora_params = 0
                    for full_param_name, param in model.named_parameters():
                        if ".lora_A." in full_param_name or ".lora_B." in full_param_name:
                            param.requires_grad_(False)
                            frozen_lora_params += 1
                    _rank0_print(
                        f"  ✓ Froze {frozen_lora_params} LoRA parameter tensors after load for {base_model_name}"
                    )
                setattr(pipe, base_model_name, model)
                enabled_trainables = []

                # Re-enable gradients for trainable parameters within this model
                # (LoRA injection freezes all base parameters by default)
                wrapped_model = getattr(pipe, base_model_name)

                for trainable_name in trainable_model_list:
                    if trainable_name.startswith(base_model_name + "."):
                        param_name = trainable_name[len(base_model_name) + 1:]

                        # Try to find parameter in wrapped model's named_parameters
                        param_found = False
                        for full_param_name, param in wrapped_model.named_parameters():
                            # Check if this is the parameter we're looking for
                            # It might be nested like: base_model.model.domain_embedding or just domain_embedding
                            if full_param_name.endswith(param_name) or full_param_name == param_name:
                                param.requires_grad_(True)
                                enabled_trainables.append(trainable_name)
                                param_found = True
                                break

                        # If not found as a direct parameter, check if it's a module (e.g., control_patch_embedding is a Conv3d)
                        if not param_found:
                            for full_module_name, module in wrapped_model.named_modules():
                                if full_module_name.endswith(param_name) or full_module_name == param_name:
                                    # Enable gradients for all parameters in this module
                                    num_params_enabled = 0
                                    for p in module.parameters():
                                        p.requires_grad_(True)
                                        num_params_enabled += 1
                                    if num_params_enabled > 0:
                                        enabled_trainables.append(trainable_name)
                                        param_found = True
                                    break

                        if not param_found:
                            _rank0_print(f"  ✗ Warning: Parameter/module {param_name} not found in wrapped model")
                _rank0_print(
                    f"✓ LoRA applied to {base_model_name}"
                    + (
                        f" | extra trainables: {', '.join(enabled_trainables)}"
                        if enabled_trainables else ""
                    )
                )


    def split_pipeline_units(self, task, pipe, trainable_models=None, lora_base_model=None):
        models_require_backward = []
        if trainable_models is not None:
            models_require_backward += [m.strip() for m in trainable_models.split(",") if m.strip()]
        if lora_base_model is not None:
            # Support comma-separated list of lora base models
            models_require_backward += [m.strip() for m in lora_base_model.split(",") if m.strip()]
        if task.endswith(":data_process"):
            _, pipe.units = pipe.split_pipeline_units(models_require_backward)
        elif task.endswith(":train"):
            pipe.units, _ = pipe.split_pipeline_units(models_require_backward)
        return pipe
    
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        controlnet_keys_map = (
            ("blockwise_controlnet_", "blockwise_controlnet_inputs",),
            ("controlnet_", "controlnet_inputs"),
        )
        controlnet_inputs = {}
        for extra_input in extra_inputs:
            for prefix, name in controlnet_keys_map:
                if extra_input.startswith(prefix):
                    if name not in controlnet_inputs:
                        controlnet_inputs[name] = {}
                    controlnet_inputs[name][extra_input.replace(prefix, "")] = data[extra_input]
                    break
            else:
                inputs_shared[extra_input] = data[extra_input]
        for name, params in controlnet_inputs.items():
            inputs_shared[name] = [ControlNetInput(**params)]
        return inputs_shared
