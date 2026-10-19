from __future__ import annotations

from typing import Any


def print_trainable_parameters_verification(accelerator: Any, model: Any, args: Any | None) -> None:
    """
    Print a compact summary of trainable parameters (main-process only).

    This is used by `diffsynth/diffusion/runner.py` to keep the training loop readable.
    """
    if not getattr(accelerator, "is_main_process", False):
        return

    print("\n" + "=" * 80)
    print("TRAINABLE PARAMETERS VERIFICATION")
    print("=" * 80)

    unwrapped_model = accelerator.unwrap_model(model)

    target_modules: list[str] = []
    if args is not None and hasattr(args, "trainable_models") and args.trainable_models:
        for pattern in str(args.trainable_models).split(","):
            pattern = pattern.strip()
            if not pattern:
                continue
            parts = pattern.split(".")
            if not parts:
                continue
            top_level = parts[0]
            if top_level == "pipe" and len(parts) > 1:
                target_modules.append(parts[1])
            else:
                target_modules.append(top_level)
        target_modules = sorted(set(target_modules))

    trainable_params: dict[str, tuple[int, tuple[int, ...]]] = {}
    total_trainable = 0
    total_frozen = 0

    target_module_trainable = 0
    target_module_frozen = 0

    is_fsdp = "FSDP" in str(getattr(getattr(accelerator, "state", None), "distributed_type", ""))

    # NOTE on FSDP-wrapped submodules: their child params may show as 0-numel
    # shard stubs and/or rank-local flat-param views here. This makes the
    # post-prepare percentage useful as a sanity check for "some params are
    # trainable", but not as global 14B parameter accounting. Tried wrapping with
    # FSDP.summon_full_params(..., recurse=True) but that issued 40+
    # collective all-gathers (one per wrapped DiT block) and stalled
    # 14B-32-rank startup for tens of seconds (observed 2026-04-30 on the
    # 8node 288x512 launch). Reverted to plain named_parameters() —
    # cosmetic local-shard counts are preferable to a collective stall.
    param_snapshots: list[tuple[str, int, tuple[int, ...], bool]] = [
        (
            name,
            int(param.numel()),
            tuple(int(x) for x in param.shape),
            bool(getattr(param, "requires_grad", False)),
        )
        for name, param in unwrapped_model.named_parameters()
    ]

    for name, num_params, shape, requires_grad in param_snapshots:
        in_target_module = False
        if target_modules:
            for target in target_modules:
                if f".{target}." in name or name.startswith(f"{target}.") or name.startswith(f"pipe.{target}."):
                    in_target_module = True
                    break

        if requires_grad:
            trainable_params[name] = (num_params, shape)
            total_trainable += num_params
            if in_target_module or not target_modules:
                target_module_trainable += num_params
        else:
            total_frozen += num_params
            if in_target_module or not target_modules:
                target_module_frozen += num_params

    grouped_params: dict[str, list[tuple[str, int, tuple[int, ...]]]] = {}
    for name, (count, shape) in trainable_params.items():
        parts = name.split(".")
        if len(parts) >= 3:
            if "blocks" in parts:
                idx = parts.index("blocks")
                prefix = ".".join(parts[: idx + 3])  # e.g., pipe.dit.blocks.0.self_attn
            else:
                prefix = ".".join(parts[:3])
        else:
            prefix = name

        grouped_params.setdefault(prefix, []).append((name, count, shape))

    print(f"\nTrainable modules ({len(grouped_params)} module groups):")

    block_roots: dict[str, dict[str, Any]] = {}
    non_block_params: dict[str, list[tuple[str, int, tuple[int, ...]]]] = {}

    for prefix in grouped_params.keys():
        parts = prefix.split(".")
        if "blocks" in parts:
            idx = parts.index("blocks")
            if len(parts) > idx + 2:
                block_idx = parts[idx + 1]
                module_name = parts[idx + 2]

                root_key = ".".join(parts[: idx + 1]) + ".*"  # e.g., pipe.dit.blocks.*
                if root_key not in block_roots:
                    block_roots[root_key] = {"blocks": {}, "modules": {}}

                block_roots[root_key]["blocks"].setdefault(block_idx, []).extend(grouped_params[prefix])
                block_roots[root_key]["modules"].setdefault(block_idx, {})
                block_roots[root_key]["modules"][block_idx][module_name] = (
                    block_roots[root_key]["modules"][block_idx].get(module_name, 0)
                    + sum(p[1] for p in grouped_params[prefix])
                )
            else:
                non_block_params[prefix] = grouped_params[prefix]
        else:
            non_block_params[prefix] = grouped_params[prefix]

    def _as_int_or_none(x: str) -> int | None:
        try:
            return int(x)
        except Exception:
            return None

    for root_key in sorted(block_roots.keys()):
        blocks_map = block_roots[root_key]["blocks"]
        modules_map = block_roots[root_key]["modules"]
        block_indices = list(blocks_map.keys())
        if not block_indices:
            continue

        numeric_block_indices = [b for b in block_indices if _as_int_or_none(b) is not None]
        if numeric_block_indices:
            example_block_idx = min(numeric_block_indices, key=lambda b: int(b))
        else:
            example_block_idx = sorted(block_indices)[0]

        example_params = blocks_map[example_block_idx]
        example_total = sum(p[1] for p in example_params)

        def _relative_spec_for_block(block_idx: str, params_: list[tuple[str, int, tuple[int, ...]]]):
            prefix_ = root_key.replace("*", str(block_idx))
            rel = []
            for full_name, numel, shape in params_:
                rel_name = full_name.replace(prefix_ + ".", "", 1)
                rel.append((rel_name, numel, shape))
            rel.sort()
            return rel

        example_spec = _relative_spec_for_block(example_block_idx, example_params)
        mismatched_blocks: list[str] = []
        per_block_totals: dict[str, int] = {}
        for bidx, params in blocks_map.items():
            per_block_totals[bidx] = sum(p[1] for p in params)
            if _relative_spec_for_block(bidx, params) != example_spec:
                mismatched_blocks.append(bidx)

        num_blocks = len(block_indices)
        total_across_blocks = sum(per_block_totals.values())

        # Format block range nicely
        block_range_str = ""
        if numeric_block_indices:
            sorted_blocks = sorted([int(b) for b in numeric_block_indices])
            if len(sorted_blocks) > 1 and sorted_blocks == list(range(sorted_blocks[0], sorted_blocks[-1] + 1)):
                block_range_str = f"[{sorted_blocks[0]}..{sorted_blocks[-1]}]"
            elif len(sorted_blocks) <= 5:
                block_range_str = f"[{', '.join(map(str, sorted_blocks))}]"
            else:
                block_range_str = f"[{sorted_blocks[0]}..{sorted_blocks[-1]}]"
        else:
            block_range_str = f"[{', '.join(block_indices[:5])}{'...' if len(block_indices) > 5 else ''}]"

        print(f"\n  📦 {root_key} — {num_blocks} blocks {block_range_str}:")
        print(f"     Params per block: {example_total:,} × {num_blocks} = {total_across_blocks:,}")

        if mismatched_blocks:
            mismatch_preview = ", ".join(map(str, mismatched_blocks[:5]))
            suffix = "..." if len(mismatched_blocks) > 5 else ""
            print(f"     ⚠ Structure varies across blocks (mismatch: {mismatch_preview}{suffix})")

        module_totals = modules_map.get(example_block_idx, {})
        if module_totals:
            print(f"     Submodules (example from block {example_block_idx}):")
            for module_name, module_total in sorted(module_totals.items(), key=lambda x: (-x[1], x[0])):
                print(f"       • {module_name}: {module_total:,}")

    def _truncate_and_print_params(
        *,
        prefix: str,
        params_list: list[tuple[str, int, tuple[int, ...]]],
        indent: str,
        max_lines: int,
    ) -> None:
        shown = 0
        for param_name, param_count, _shape in params_list:
            if shown >= max_lines:
                break
            short_name = param_name.replace(prefix + ".", indent)
            print(f"     {short_name}: {param_count:,}")
            shown += 1
        omitted = len(params_list) - shown
        if omitted > 0:
            print(f"     {indent}… ({omitted} more)")

    for prefix in sorted(non_block_params.keys()):
        params_list = non_block_params[prefix]
        group_total = sum(p[1] for p in params_list)
        print(f"\n  📦 {prefix}:")
        print(f"     Total params: {group_total:,}")
        top_k = 25
        params_sorted = sorted(params_list, key=lambda x: (-x[1], x[0]))
        _truncate_and_print_params(prefix=prefix, params_list=params_sorted, indent="    ├─ ", max_lines=top_k)

    print(f"\n{'─' * 80}")
    if is_fsdp:
        print(
            "NOTE: FSDP is active; counts below are from the rank-local "
            "post-wrap parameter view, not a full global parameter census."
        )
    print(f"Total trainable parameters: {total_trainable:,} ({total_trainable / 1e6:.2f}M)")
    print(f"Total frozen parameters:    {total_frozen:,} ({total_frozen / 1e6:.2f}M)")
    denom = total_trainable + total_frozen
    overall_pct = 100 * total_trainable / denom if denom else 0.0
    print(f"Percentage trainable (overall): {overall_pct:.2f}%")

    if target_modules:
        target_total = target_module_trainable + target_module_frozen
        if target_total > 0:
            print(
                f"Percentage trainable (in requested modules: {', '.join(target_modules)}): "
                f"{100 * target_module_trainable / target_total:.2f}%"
            )
        else:
            print(f"Percentage trainable (in requested modules: {', '.join(target_modules)}): N/A (no parameters found)")

    print("=" * 80 + "\n")
