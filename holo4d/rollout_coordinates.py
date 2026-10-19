"""Explicit Sim(3) metadata for comparing window-local camera poses.

Metadata is for validation; these helpers never rewrite the sampler's input
coordinates or fit a camera path to make a boundary test pass.
"""
import numpy as np


def poses_in_rollout_gauge(poses, gauge):
    scale = float(gauge['scale'])
    rotation = np.asarray(gauge['rotation'], dtype=np.float64)
    translation = np.asarray(gauge['translation'], dtype=np.float64)
    if (not np.isfinite(scale) or scale <= 0 or rotation.shape != (3, 3)
            or translation.shape != (3,) or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()):
        raise ValueError('Invalid prepared-to-rollout similarity')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or np.linalg.det(rotation) <= 0:
        raise ValueError('Camera gauge rotation must be a proper rotation')
    result = np.asarray(poses, dtype=np.float64).copy()
    if result.ndim != 3 or result.shape[-2:] != (4, 4) or not np.isfinite(result).all():
        raise ValueError('Camera poses must be finite (T,4,4) matrices')
    result[:, :3, :3] = rotation @ result[:, :3, :3]
    result[:, :3, 3] = scale * (result[:, :3, 3] @ rotation.T) + translation
    return result


def target_boundary_report(previous, current, previous_k, current_k, *, atol=2e-5):
    pose_error = float(np.abs(previous[-1] - current[0]).max())
    left = previous_k[-1] if np.asarray(previous_k).ndim == 3 else previous_k
    right = current_k[0] if np.asarray(current_k).ndim == 3 else current_k
    intrinsic_error = float(np.abs(left - right).max())
    report = dict(target_camera_matrix_max_abs=pose_error,
                  target_intrinsics_max_abs=intrinsic_error,
                  passed=pose_error <= atol and intrinsic_error <= 2e-4)
    if not report['passed']:
        raise ValueError('Target camera discontinuity in common rollout gauge: ' + str(report))
    return report


def validate_rollout_camera_metadata(root, entry):
    """Validate explicit gauges when present; legacy bundles remain readable."""
    from holo4d.examples_io import resolve_member

    present = ['prepared_to_rollout' in window for window in entry['windows']]
    if not any(present):
        return dict(status='legacy_window_local_gauges_not_certified', boundaries=[])
    if not all(present):
        raise ValueError('Partial rollout camera-gauge metadata')
    previous, rows = None, []
    for window in entry['windows']:
        with np.load(resolve_member(root, window['condition']), allow_pickle=False) as data:
            target = poses_in_rollout_gauge(data['target_cam2world'], window['prepared_to_rollout'])
            intrinsic = np.asarray(data['target_intrinsics'])
        if previous is not None:
            rows.append(dict(chunk=window['chunk_index'],
                             **target_boundary_report(previous[0], target, previous[1], intrinsic)))
        previous = target, intrinsic
    return dict(status='passed', boundaries=rows)
