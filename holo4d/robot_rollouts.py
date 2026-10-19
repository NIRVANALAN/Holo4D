"""Portable, checksummed source-only robot long-rollout inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from holo4d.examples_io import condition_to_sample, file_sha256, resolve_member, validate_condition

FORMAT = 'holo4d-prepared-robot-rollouts-v1'


def load_rollout(manifest_path: Path, example_id: str):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError('Unknown robot rollout bundle format')
    entries = [e for e in manifest['examples'] if e['id'] == example_id]
    if len(entries) != 1:
        raise ValueError(f'Unknown or duplicate robot example: {example_id}')
    entry = entries[0]
    if not entry.get('caption', '').strip():
        raise ValueError('Empty source caption')
    windows = entry['windows']
    if len(windows) < 2 or [w['chunk_index'] for w in windows] != list(range(len(windows))):
        raise ValueError('Expected ordered multi-window rollout')
    if [w['start'] for w in windows] != [48*i for i in range(len(windows))]:
        raise ValueError('Expected one-frame overlap and48-frame increments')
    for window in windows:
        expected = min(49, entry['real_frames'] - window['start'])
        if expected < 2 or window['valid_frames'] != expected:
            raise ValueError('Invalid real-frame/padding contract')
        resolve_member(manifest_path.parent, window['condition'])
    from holo4d.rollout_coordinates import validate_rollout_camera_metadata
    entry = dict(entry, camera_coordinate_audit=validate_rollout_camera_metadata(manifest_path.parent, entry))
    return manifest, entry


def load_window(root: Path, entry: dict, index: int):
    window = entry['windows'][index]
    path = resolve_member(root, window['condition'])
    if file_sha256(path) != window['condition_sha256']:
        raise ValueError(f'Condition checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    report = validate_condition(data)
    if (report['height'], report['width']) != (entry['height'], entry['width']):
        raise ValueError('Source canvas differs from manifest')
    sample = condition_to_sample(data, entry)
    return sample, report


def attach_generated_boundary(sample: dict, previous_frames):
    """Only a freshly generated previous target is eligible as image boundary."""
    from PIL import Image

    if len(previous_frames) != 49:
        raise ValueError('Previous generated window must contain49frames')
    image = np.asarray(previous_frames[-1], dtype=np.uint8)
    if image.shape != np.asarray(sample['video'][0]).shape:
        raise ValueError('Generated boundary/source canvas mismatch')
    sample['inference_input_image'] = Image.fromarray(image)
    sample['use_target_image_condition'] = True
    return sample
