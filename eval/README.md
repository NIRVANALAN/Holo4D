# Evaluation entrypoints

The public tree keeps metric code separate from model/data assets. Nothing in
this directory contains the DAVIS, iPhone/DyCheck, or Syn4D datasets.

## Frozen predictions and prepared inputs

[Five benchmark downloads and verification](../doc/release/2026-10-19-benchmark-downloads.md)
provide original baseline/Holo4D predictions, minimal conditions, scoring
caches and evaluation inputs at a pinned HF revision. Historical protocols
and reconstructed-only input fields are explicitly distinguished; no new
generation result is substituted for a paper result.

## Shared video and camera metrics

`eval/run_eval_video.sh` and `eval/run_eval_camera.sh` are the
shared aggregators used by all benchmarks. They expect a prediction tree and a
benchmark-specific manifest, and write per-pair JSON/TSV plus aggregate files.
The video-quality runner reports CLIP-T/F/V by default and optional FVD;
camera runners support VGGT-Omega, ViPE, and DA3 backends. The launcher prints
the two required I3D weight paths when FVD is requested.

## Final-paper protocols

Use the table to choose a scorer; these protocols are not interchangeable.
Install the camera metric dependencies with `pip install -e '.[eval]'`.
The released scorer pins evo1.37.0, the version used for the verified archive
replays; evo1.38 changed the `ape` API and rejects the scorer's naming arguments.
VBench0.1.5 requires NumPy<2, whereas inference uses NumPy>=2: do **not** install
VBench in the inference environment. Its dependency group is declared only in
`pyproject.toml`. From the repository root, create a separate environment:

```bash
conda create -n holo4d-vbench python=3.12 pip -y
conda activate holo4d-vbench
python -m pip install --upgrade 'pip>=25.1'
python -m pip install --group vbench
python -m pip check
```

Run `bash eval/common/eval_vbench.sh` from the repository root in that environment
with the required input/output variables below. No editable Holo4D installation
is needed for the lightweight scoring helpers. VBench still requires its
upstream model weights; ensure its PyTorch build supports the selected GPU.
Do not install `requirements/cu130.txt` in this scorer environment because it
contains the inference NumPy2 requirement.

| Metric | Public entrypoint | Frozen input / aggregation |
| --- | --- | --- |
| VGGT-Omega camera | `common/eval_camera_metrics_vggt_omega.sh` | First 49 frames, 672×384 canvas, balanced-512, `all_norm`; DAVIS ATE/RPE-T 54 translation-bearing pairs, RPE-R 100 |
| Pi3 source-aligned camera | `common/eval_camera_pi3.sh` | Joint source + generated 49+49; source-only `pose_stabilized` alignment; TransErr² / RotErr over all 100 DAVIS pairs |
| VBench classic five | `common/eval_vbench.sh` | v0.1.5; normalized 672×384×49 at 16 FPS, imageio H.264 quality 9, source-specific DAVIS captions; each dimension n=100 |
| DAVIS source-proxy FVD / CLIP | `common/eval_video_metrics.sh` | Explicitly select metrics; 288×512×49; source proxy is not novel-view ground truth |
| Syn4D paired-target FVD | `common/eval_paired_fvd.sh` | 288×512×49; StyleGAN-V I3D; prediction vs paired target GT, fixed pair-ID manifest |
| iPhone RGB / masked RGB | `iphone/recompute_baseline_metrics.py` | Five scenes; first 49 frames, 288×512; 16:9 GT crop; valid × covisible masks |
| iPhone pseudo-depth | `iphone/eval_iphone_4rc_recon_depth.py`, `iphone/eval_iphone_vggt_omega_recon_depth.py` | Same target crop/masks; use `holo4d.tools.summarize_iphone_depth_scene_mean` for equal five-scene means |

The Pi3 wrapper requires `METHOD_PRED_ROOT`, `PAIRS_CSV`, `SOURCE_ROOT`
(each scene has `video.mp4` and `cameras.npz`), `CAMERAS_ROOT`, `VISTA4D_REPO`,
`POSE_CACHE_ROOT` and `OUTPUT_JSON`. Explicitly choose `TARGET_CAMERA_GAUGE`:
`shared_world` for released native target poses, `source_frame0_local` only for
effective-camera exports in that frame. Do not align generated target cameras
to target GT in the source-aligned protocol. The optional legacy ATE fields
computed from Pi3 poses are not independent VGGT-Omega estimates.

Prepare VBench inputs with `python -m holo4d.tools.prepare_davis_vbench_inputs`;
it rejects missing/empty captions and incomplete/duplicate frozen pair IDs.
The scorer wrapper requires `VIDEOS_PATH` (including `prompts.json`) and
`RESULT_DIR`. Set `EXPECTED_VIDEOS` for a non-DAVIS cohort. Prepare paired-GT
FVD manifests with `holo4d.tools.prepare_syn4d_paired_fvd_manifest` or
`holo4d.tools.prepare_bundle32_fvd_manifest`; never compare two different
cohorts as if they were a matched backend experiment.

Reconstructor repositories/checkpoints, Pi3/VBench/I3D/CLIP/LPIPS weights,
datasets, masks, frozen camera manifests and prediction MP4s remain external
assets. Pseudo-depth also requires the external geometry scorer documented in
the depth scripts. Including an entrypoint does not bundle these assets.

### What has actually been verified?

The public camera scorers are
also replayed against the frozen DAVIS pose caches; this verifies scoring math,
not fresh reconstruction inference on every paper video. Use
`python -m holo4d.tools.verify_metric_replay --expected reference.json
--actual replay.json --section all_norm` to check a VGGT summary without
rounding away differences (`--section pose_stabilized` for Pi3).

## DAVIS

`eval/davis/eval_track4d360_davis_4rc_novel_traj.py` runs the 49-frame DAVIS
source-geometry protocol (4RC/CoWTracker/VGGT-Omega caches are external).
`inference/sample_19500_davis.sh` is a thin convenience wrapper. Use
`eval/run_eval_video.sh` or `eval/common/eval_video_metrics.sh` on standardized
`pred.mp4` outputs.

## iPhone / DyCheck

`eval/iphone/eval_iphone_nvs.py` implements the masked and unmasked DyCheck
PSNR/SSIM/LPIPS metrics. `flow3d_metrics.py` is self-contained and does not
import Shape-of-Motion. The 4RC and VGGT-Omega depth evaluators are provided
as `eval_iphone_4rc_recon_depth.py` and
`eval_iphone_vggt_omega_recon_depth.py`; they require the corresponding
external geometry repository/cache and the iPhone manifest.

For a strict five-scene comparison, `eval/iphone/recompute_baseline_metrics.py`
scores the fixed GEN3C, Vista4D, TrajectoryCrafter, and ReCamMaster prediction
layouts, plus any standardized `<scene>/pred.mp4` method (for example ViDAR or
Holo4D). It fails if any of the five predictions is missing; it evaluates the
first 49 frames at 288×512 after the 16:9 GT center crop, with unmasked and
valid×covisible masked metrics. Example:

```bash
python eval/iphone/recompute_baseline_metrics.py \
  --gt-root /data/iphone_nvs/gt \
  --fixed-root /data/iphone_nvs/external_predictions_fixed \
  --standardized-root vidar=/data/iphone_nvs/vidar \
  --standardized-root holo4d_4rc=/data/iphone_nvs/holo4d_4rc \
  --output-root /data/iphone_nvs/recomputed_metrics
```

The script records each source MP4 path and per-scene scores in JSON; no
benchmark videos or checkpoints are included in this repository.
The verified five-baseline table and exact protocol are in
[`iphone/README.md`](iphone/README.md).

## Two-model long rollout

`../inference/sample_mixed_t2v_i2v_syn4d.sh` evaluates a generated-boundary
rollout: T2V seed (step-19500) followed by native I2V continuation
(step-22500). The first stage uses source RGB/video and geometry only; the
second stage receives the generated seed target endpoint as its image
condition. See [`LONG_ROLLOUT.md`](LONG_ROLLOUT.md) for the artifact contract.

## Syn4D

`eval/syn4d/` contains the flying-group depth evaluators and prediction
standardization helpers. The model sampler itself is
`inference/sample_19500_syn4d.sh`.

All scripts accept absolute paths or environment overrides. Before reporting
numbers, record the dataset revision, resolution, frame count, metric mask,
geometry backend, and checkpoint in the output manifest.
