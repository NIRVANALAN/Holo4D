"""Reproduce the released TRI/Figure0 multi-window robot demonstrations."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def model_args(options, entry, stage):
    from holo4d import hub
    from eval_track4d360_davis_4rc_novel_traj import parse_args

    kind = 't2v' if stage == 'seed' else 'i2v'
    env = 'T2V' if stage == 'seed' else 'I2V'
    wan = Path(os.environ[f'{env}_ROOT']) if os.environ.get(f'{env}_ROOT') else hub.download_wan(kind, root=hub.cache_root())
    checkpoint = Path(os.environ[f'{env}_CHECKPOINT']) if os.environ.get(f'{env}_CHECKPOINT') else hub.download_checkpoint(kind, root=hub.cache_root())
    dit = str(wan / 'diffusion_pytorch_model*.safetensors')
    prefix = hub.WAN_REPOS[kind]
    bundle = f'{prefix}:{dit},{prefix}:{wan / "Wan2.1_VAE.pth"},{prefix}:{wan / "models_t5_umt5-xxl-enc-bf16.pth"}'
    if stage != 'seed':
        bundle += f',{prefix}:{wan / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"}'
    sys.argv = [sys.argv[0], '--checkpoint_path', str(checkpoint),
        '--camera_control_checkpoint', dit, '--model_id_with_origin_paths', bundle,
        '--tokenizer_path', str(wan / 'google/umt5-xxl'),
        '--track_config', str(ROOT / 'config/holo4d/default_conv3d_patchify_srcdepth_srcrgb.yaml'),
        '--output_dir', str(options.output), '--height', str(entry['height']), '--width', str(entry['width']),
        '--num_frames', '49', '--no_dense_branch', '--use_query_frame_impulse_condition',
        '--use_plucker_camera_control', '--enable_v2v_plucker_camera_control',
        '--use_perlayer_camera_control', '--camera_injection_mode', 'projector',
        '--camera_plucker_encoder_mode', 'patchify', '--use_plucker_raymap_tokens',
        '--track_injection_mode', 'per_block', '--track_injection_block_mode', 'concat_project',
        '--track_spatial_stride', '1', '--track_sample_mode', 'all', '--track_stride', '1',
        '--num_inference_steps', str(options.steps), '--cfg_scale', '5', '--sigma_shift', '5', '--seed', '42']
    args = parse_args()
    args.skip_comparison = True
    args.vae_tiling = True
    args.capture_final_latents = False
    return args


def run_stage(options, entry, indices):
    import torch
    from holo4d.examples_io import file_sha256
    from holo4d.robot_rollouts import attach_generated_boundary, load_window
    from holo4d.tools.i2v_continuation_long_video import read_video
    from eval_track4d360_hybrid_dense_syn4d import build_eval_pipeline, _run_one_pair, precompute_eval_prompt_embs, NEGATIVE_PROMPT

    stage = 'seed' if indices == [0] else 'continuation'
    args = model_args(options, entry, stage)
    print(f'[robot-rollout] stage={stage} checkpoint={args.checkpoint_path} caption={entry["caption"]}', flush=True)
    checkpoint_hash = file_sha256(Path(args.checkpoint_path))
    pipe = None
    for index in indices:
        folder = options.output / entry['id'] / f'chunk_{index:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        previous = options.output / entry['id'] / f'chunk_{index-1:03d}/pred_rgb.mp4'
        contract = dict(example=entry['id'], chunk=index, condition_sha256=entry['windows'][index]['condition_sha256'],
                        checkpoint_sha256=checkpoint_hash, backbone=stage, steps=options.steps,
                        cfg=5, shift=5, seed=42, fps=16, mp4_quality=9,
                        previous_generated_sha256=file_sha256(previous) if index else None)
        if 'prepared_to_rollout' in entry['windows'][index]:
            contract['prepared_to_rollout'] = entry['windows'][index]['prepared_to_rollout']
        complete = folder / 'COMPLETE.json'
        if complete.is_file() and json.loads(complete.read_text()) == contract:
            if len(read_video(folder/'pred_rgb.mp4')) != 49:
                raise ValueError(f'Corrupt completed chunk: {folder}')
            print(f'[robot-rollout] skip verified complete {folder}', flush=True)
            continue
        if (folder/'pred_rgb.mp4').exists() or complete.exists():
            raise ValueError(f'Existing output is incomplete or incompatible; choose a new output root: {folder}')
        sample, report = load_window(options.manifest.parent, entry, index)
        if index:
            attach_generated_boundary(sample, read_video(previous))
        else:
            sample['use_target_image_condition'] = False
        if pipe is None:
            pipe = build_eval_pipeline(args)
            precompute_eval_prompt_embs(pipe, [entry['caption'], NEGATIVE_PROMPT])
        (folder/'run_contract.json').write_text(json.dumps(dict(contract, condition_report=report), indent=2)+'\n')
        _run_one_pair(pipe, sample, args, folder)
        if len(read_video(folder/'pred_rgb.mp4')) != 49:
            raise ValueError('Incomplete generated window')
        complete.write_text(json.dumps(contract, indent=2)+'\n')
        print(f'[robot-rollout] complete chunk{index}', flush=True)
        del sample
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def assemble(options, entry):
    from holo4d.conditions.i2v_target_condition import plan_long_horizon_chunks, stitch_long_horizon_chunks
    from holo4d.tools.i2v_continuation_long_video import read_video, write_video

    root = options.output / entry['id']
    frames = [read_video(root/f'chunk_{w["chunk_index"]:03d}/pred_rgb.mp4') for w in entry['windows']]
    result = stitch_long_horizon_chunks(frames, plan_long_horizon_chunks(entry['real_frames'], chunk_frames=49))
    destination = root/'pred_rgb_full.mp4'
    if not destination.exists():
        write_video(destination, result, fps=16)
    if len(read_video(destination)) != entry['real_frames']:
        raise ValueError('Assembled real-frame count mismatch')
    print(f'[robot-rollout] full video {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--example', choices=('tri_static_high', 'figure0_orbit'), default='tri_static_high')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/robot_rollouts')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--limit-chunks', type=int, default=0, help='Smoke-only bound; full video is not assembled if truncated')
    parser.add_argument('--validate-only', action='store_true')
    options = parser.parse_args()
    from holo4d.robot_rollouts import load_rollout, load_window
    _, entry = load_rollout(options.manifest, options.example)
    print('[robot-rollout] camera-coordinate preflight=' + json.dumps(entry['camera_coordinate_audit']), flush=True)
    count = len(entry['windows'])
    if options.steps <= 0 or options.limit_chunks < 0:
        parser.error('steps must be positive and limit-chunks nonnegative')
    if options.validate_only:
        for index in range(count):
            sample, report = load_window(options.manifest.parent, entry, index)
            print(json.dumps(dict(example=entry['id'], chunk=index, caption=entry['caption'], **report)), flush=True)
            del sample
        return
    os.environ.setdefault('DISABLE_FLASH_ATTN_3', '1')
    os.environ.setdefault('HOLO4D_FLASH_ATTN3', '0')
    os.environ.setdefault('HOLO4D_ALLOW_ONLINE_T5_ENCODE', '1')
    from holo4d.fa3_compat import configure_flash_attention_imports
    configure_flash_attention_imports()
    sys.path.insert(0, str(ROOT/'examples/wan2.1_fun'))
    count = min(count, options.limit_chunks or count)
    run_stage(options, entry, [0])
    if count > 1:
        run_stage(options, entry, list(range(1, count)))
    if count == len(entry['windows']):
        assemble(options, entry)


if __name__ == '__main__':
    main()
