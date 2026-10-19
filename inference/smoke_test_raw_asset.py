from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from holo4d.visualization.artifact_io import save_mp4


def main():
    p = argparse.ArgumentParser(description='Validate and render the bundled Holo4D raw smoke pair.')
    p.add_argument('--asset', type=Path, default=ROOT / 'assets/smoke/raw_pair.npz')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/smoke/raw_pair')
    args = p.parse_args()
    raw = np.load(args.asset, allow_pickle=True)
    required = {
        'source_video', 'target_video', 'source_cam2world', 'target_cam2world',
        'source_intrinsics', 'target_intrinsics', 'source_depth_video',
        'tracks_world', 'tracks_world_valid_mask', 'prompt', 'fps',
    }
    missing = sorted(required.difference(raw.files))
    if missing:
        raise SystemExit(f'missing raw-pair keys: {missing}')
    source = np.asarray(raw['source_video'])
    target = np.asarray(raw['target_video'])
    if source.ndim != 4 or source.shape[-1] != 3 or target.shape != source.shape:
        raise SystemExit(f'bad video shapes: source={source.shape}, target={target.shape}')
    tracks = np.asarray(raw['tracks_world'])
    valid = np.asarray(raw['tracks_world_valid_mask'])
    if tracks.ndim != 3 or tracks.shape[1] != source.shape[0] or tracks.shape[2] != 3:
        raise SystemExit(f'bad tracks shape: {tracks.shape}; expected (N,{source.shape[0]},3)')
    if valid.shape != tracks.shape[:2]:
        raise SystemExit(f'bad valid-mask shape: {valid.shape} vs {tracks.shape[:2]}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fps = int(np.asarray(raw['fps']).reshape(()))
    save_mp4(args.output_dir / 'source.mp4', source, fps=fps)
    save_mp4(args.output_dir / 'target.mp4', target, fps=fps)
    report = {
        'format': 'holo4d_raw_pair_smoke_v1',
        'asset': str(args.asset),
        'asset_sha256': hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        'source_shape': list(source.shape),
        'target_shape': list(target.shape),
        'tracks_shape': list(tracks.shape),
        'valid_ratio': float(valid.mean()),
        'fps': fps,
        'outputs': ['source.mp4', 'target.mp4'],
        'status': 'passed',
    }
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
