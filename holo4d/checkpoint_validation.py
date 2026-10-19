"""Strict coverage of released dense-track fine-tuning deltas.

Frozen base weights are intentionally absent from deltas. Required learned
conditioning and self-attention parameters must not be silently missing.
This validates load coverage without changing tensors, RNG or load order.
"""
from __future__ import annotations


TRACK_MODULES = ('track_embedder', 'track_adapter', 'track_identity_adapter',
                 'track_projector', 'track_block_injector')


def required_dit_parameter(name):
    return ('.self_attn.' in name or '.camera_projector.' in name
            or name == 'domain_embedding'
            or name.startswith(('control_patch_embedding.', '_camera_')))


def validate_dense_checkpoint(pipe, raw_state, dit_state, base_state):
    """Fail on missing/shape-mismatched active parameters in released deltas."""
    missing, mismatches, routed, counts = [], [], set(), {}
    required = [(name, value) for name, value in pipe.dit.named_parameters()
                if required_dit_parameter(name)]
    for name, value in required:
        if name not in dit_state:
            missing.append('dit.' + name)
        elif tuple(dit_state[name].shape) != tuple(value.shape):
            mismatches.append('dit.' + name)
    counts['dit_active_parameters'] = len(required)
    assembled_missing = [name for name, _ in pipe.dit.named_parameters()
                         if not name.startswith('control_adapter.')
                         and name not in base_state and name not in dit_state]
    counts['assembled_dit_parameters'] = sum(1 for name, _ in pipe.dit.named_parameters()
                                            if not name.startswith('control_adapter.'))
    for attribute in TRACK_MODULES:
        module = getattr(pipe, attribute, None)
        prefixes = ('pipe.' + attribute + '.', attribute + '.')
        supplied = {key: value for key, value in raw_state.items()
                    if key.startswith(prefixes)}
        if module is None:
            if supplied:
                missing.append(attribute + ' (checkpoint module not instantiated)')
            continue
        parameters = dict(module.named_parameters())
        counts[attribute] = len(parameters)
        for name, value in parameters.items():
            key = next((p + name for p in prefixes if p + name in raw_state), None)
            if key is None:
                missing.append(attribute + '.' + name)
            elif tuple(raw_state[key].shape) != tuple(value.shape):
                mismatches.append(attribute + '.' + name)
        # Buffers such as sin_enc.freqs are derived, not learned parameters;
        # present buffers are still required to have valid names and shapes.
        state = module.state_dict()
        for key, value in supplied.items():
            prefix = next(p for p in prefixes if key.startswith(p))
            local = key[len(prefix):]
            if local not in state:
                mismatches.append(key + ' (unknown track key)')
            elif tuple(value.shape) != tuple(state[local].shape):
                mismatches.append(key)
            routed.add(key)
    unknown = [key for key in raw_state if key.startswith(('pipe.track_', 'track_'))
               and key not in routed]
    report = dict(required_parameter_counts=counts, missing_parameters=missing,
                  incompatible_parameters=mismatches, unrouted_track_keys=unknown,
                  assembled_dit_missing_parameters=assembled_missing,
                  passed=not (missing or mismatches or unknown or assembled_missing))
    if not report['passed']:
        raise ValueError('Dense checkpoint coverage failed: ' + str(report))
    return report
