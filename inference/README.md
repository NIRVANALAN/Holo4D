# Inference and raw-video preprocessing

## Scope and compatibility notes (2026-10-19)

- Raw preprocessing creates **one 49-frame window only**, not an automatic
  window-by-window long-video preprocessing pipeline. Prepared long-video
  bundles use the separate `sample_robot_rollouts.py` route.
- A strongly mismatched input/output aspect now emits a warning before crop.
  Portrait robots normally require `--width 384 --height 672`.
- `--gpu` selects the visible CUDA device for 4RC (`cuda:<index>`), as well as
  being forwarded to the other backend tools. With one Slurm-visible GPU use
  `--gpu 0`; select the allocation/visibility before launch.
- CoWTracker defaults to PyTorch SDPA `auto`: flash when supported, **math
  fallback enabled**. `COWTRACKER_SDPA_BACKEND=flash` explicitly opts into
  flash-only; `math` is also available. New caches use this native SDPA route,
  **not the custom-FA3 kernel used to create the paper's historical caches**.
  Weights are unchanged, but caches are not promised bit-identical.
- Translation/orbit-radius values are in each backend's reconstruction units
  before common-scale normalization. CoW/VGGT uses `cow_vggt_scale_align=none`;
  the same `--distance` or `--orbit-radius` does **not** define matched physical
  camera amplitude across CoW/VGGT and 4RC. Angles are degrees. Inspect the
  condition first; this release does not silently change camera amplitudes.
- Adding `prepared_to_rollout` metadata changes a long-window resume contract.
  Old `COMPLETE.json` files without that field are deliberately incompatible
  with the newly certified bundle. Keep the old outputs and use a new output
  root; the sampler will not overwrite them or silently bless an old contract.

This folder has two entry paths:

1. **Prepared examples (recommended first run):** download a released condition bundle and sample it with `sample_examples.py`. This avoids installing any external reconstructor.
2. **Your own raw video:** `preprocess_raw_video.py` extracts a 49-frame source window, runs a source-geometry backend, optionally runs GMOS, writes a pickle-free dense condition bundle, and can optionally invoke Holo4D sampling.

The raw-video runner has passed a real **4RC → portable condition → Holo4D 14B sampling** GPU gate (A100, 49 frames, 50 denoising steps). A fresh **CoWTracker + VGGT-Omega + GMOS → portable condition** GPU gate also passed; it verifies preprocessing, not a fresh diffusion result from that condition. External installations still need compatible dependencies and weights.

## 1. Prepared example quick start

For the newly GPU-verified 14B 49-frame, 14B 246-frame and commercial 5B
examples, see [paired source/target bundles](../doc/release/2026-10-18-verified-input-target-examples.md).
Their HF download includes the exact dense conditions, captions and generated
reference outputs; generated targets are not ground truth.

Follow the repository [installation and model-download instructions](../README.md#quick-start-prepared-examples). Validate one downloaded example before loading the diffusion model:

```bash
EXAMPLE=robot_tri bash inference/demo_examples.sh --validate-only
```

Then sample it (the example caption is stored in the manifest and is printed before inference):

```bash
CHECKPOINT=/path/to/holo4d-14b.safetensors \
WAN_ROOT=/path/to/wan2.1-t2v-14b \
EXAMPLE=robot_tri OUTPUT_DIR=outputs/my-example \
bash inference/demo_examples.sh
```

The wrapper resolves/downloads the pinned prepared bundle and supplies its manifest automatically; `robot_tri` is the actual example ID. The small manifest in `assets/examples/` is an inventory, not a bundled copy of the large condition arrays. Use the checkpoint/config pair documented in [Models](../README.md#models). The 5B model recipes have different resolutions and track contracts; do not substitute one for another. `sample_examples.py` does not reconstruct geometry and never consumes the archived reference prediction as an RGB condition.

## 2. What the raw-video command does

```text
raw MP4
  └─ ffmpeg: first 49 frames sampled at 16 FPS, center-cropped/resized once
       ├─ 4RC: per-frame camera + depth/point map + direct persistent 3D tracks
       └─ CoWTracker 2D tracks + VGGT-Omega depth/cameras → lifted 3D tracks
            └─ optional GMOS/GMOS-S masks and explicit backend-specific fusion
                 └─ source RGB + dense tracks + source depth/cameras + target camera
                      └─ condition.npz + hash-checked manifest + provenance.json
                           └─ optional released Holo4D sampler
```

The command requires an explicit nonempty caption and a target-camera path recipe (default `orbit_right`). It uses the same 49 staged RGB frames for geometry, track/mask prediction and Holo4D input. Intrinsics are produced/rescaled by the backend and bound to that exact crop. The condition exporter validates the released `holo4d-dense-example-v1` shape contract before writing. Outputs are never overwritten; choose a new output directory for each run.

The default output canvas is **672×384 (width×height)**. Use `--width 384 --height 672` for portrait. Both are center-cropped, not letterboxed. The runner rejects input clips that cannot produce 49 frames at the requested FPS; it does not silently duplicate frames. Source time is sampled from the start of the clip.

### 4RC direct 3D tracks (default)

Install [4RC](https://github.com/Luo-Yihang/4RC) in its own environment and download/use its checkpoint according to its upstream terms. The runner calls the upstream `Arc` / DUSt3R multiview API through `run_4rc_reconstruction.py`; it does not rely on the research-only DAVIS wrapper. Example (adjust environment names and paths for your machine):

```bash
python inference/preprocess_raw_video.py \
  --video /data/robot.mp4 \
  --caption "A robot arm places an object on the work surface." \
  --output-dir outputs/raw/robot_orbit_4rc \
  --backend 4rc \
  --fourrc-repo /opt/4RC --fourrc-env 4rc \
  --holo-env holo4d \
  --trajectory orbit_right --angle-deg 30
```

The backend produces dense 4RC track identities directly, alongside cameras and point maps. The condition builder consumes the resulting per-frame prediction arrays, derives camera-space source depth with the matching camera poses, and exports the direct 3D track condition; it does not approximate 4RC tracks from a point-map-only warp.

### CoWTracker + VGGT-Omega (alternative geometry backend)

Install [CoWTracker](https://github.com/facebookresearch/cowtracker) with its submodules and [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) in their own environments. Download the VGGT-Omega-1B-512 checkpoint from its model card; CoWTracker can use its upstream automatic checkpoint retrieval or an explicit local checkpoint. Then run:

```bash
python inference/preprocess_raw_video.py \
  --video /data/robot.mp4 \
  --caption "A robot arm places an object on the work surface." \
  --output-dir outputs/raw/robot_orbit_cow_vggt \
  --backend cow_vggt \
  --cowtracker-repo /opt/cowtracker \
  --cowtracker-checkpoint /models/cow_tracker_model.pth \
  --cow-env cowtracker \
  --vggt-repo /opt/vggt-omega \
  --vggt-checkpoint /models/vggt_omega_1b_512.pt \
  --vggt-env vggt-omega --holo-env holo4d \
  --trajectory orbit_right --angle-deg 30
```

This is **2D CoW tracks lifted using VGGT-Omega depth and camera poses**; it is not 4RC 3D tracking. The runner uses a single VGGT scale consistently for depths, camera translations and lifted tracks, then the Holo4D loader's standard common-scale normalization. It does not silently align the VGGT reconstruction to 4RC.

### Optional GMOS or GMOS-S masks

Install [GMOS](https://github.com/Jyxarthur/gmos) and its Pi3, SAM2 and GMOS checkpoints in a separate compatible environment. The runner calls `inference_gmos.py` or `inference_gmos_s.py` on the exact staged frame folder at stride 1 and passes the resulting frame masks to condition construction.

Add to either backend command:

```bash
  --gmos gmos --gmos-repo /opt/gmos --gmos-env gmos \
  --pi3-checkpoint /models/pi3.safetensors \
  --sam2-checkpoint /models/sam2.1_hiera_large.pt \
  --gmos-checkpoint /models/gmos_proposer.pth
```

For GMOS-S, change `--gmos gmos` to `--gmos gmos-s` and provide its foreground/background checkpoint with `--gmos-checkpoint`. GMOS mask loading and track fusion are different operations:

- On **CoWTracker + VGGT-Omega**, loaded GMOS labels drive the existing static/dynamic track fusion path.
- On **4RC**, the mask is available for dynamic conditioning, but persistent-track fusion is off unless `--fuse-gmos-static-tracks` is also passed. That flag is rejected unless GMOS masks are being produced. Do not infer fusion from a directory name.

## 3. Run Holo4D in the same command

After selecting the matching released model and downloading its weights, set its path and append `--sample`:

```bash
CHECKPOINT=/models/holo4d-14b.safetensors \
WAN_ROOT=/models/wan2.1-t2v-14b \
python inference/preprocess_raw_video.py \
  --video /data/robot.mp4 \
  --caption "A robot arm places an object on the work surface." \
  --output-dir outputs/raw/robot_orbit_4rc_sampled \
  --backend 4rc --fourrc-repo /opt/4RC --fourrc-env 4rc \
  --holo-env holo4d --sample --model 14b \
  --trajectory orbit_right --angle-deg 30
```

Set `WAN_ROOT`, `VAE_PATH`, `T5_PATH`, `TOKENIZER_PATH`, and other model asset variables as required by the [model loader](../README.md#quick-start-prepared-examples). Without `--sample`, only preprocessing and condition export run. If `--sample` is used, the checkpoint and model's training canvas/config must match; the example loader resizes only where its documented model recipe supports it.

## 4. Output layout and inspection

```text
<output-dir>/
  input/<sequence>/frame_00000.png ... frame_00048.png
  geometry_4rc/<sequence>/result.npz                  # 4RC backend
  tracks_cow/<sequence>_pred_trackvideo.npz            # CoW backend
  geometry_vggt_omega/<sequence>/{pose,geometry}/...   # CoW/VGGT backend
  masks_gmos/{mos_frame|frame}/<sequence>/...          # optional
  condition/{condition.npz,manifest.json,provenance.json}
  run.json
  sampling/                                             # only with --sample
```

The bundle stores all dense track identities, their validity/confidence when provided, source and target camera poses/intrinsics, source camera-z depth, source RGB at queried identities, caption, trajectory/backend metadata and a SHA-256. A paired target RGB or archived generated prediction is never placed in the model condition. `condition/manifest.json` is directly consumable by `inference/sample_examples.py`.

Inspect the returned `condition.npz` and `provenance.json`, then render the canonical track/camera previews described in [visualization](../doc/visualization.md) before trusting a new reconstruction. The preprocessing tool exports geometry but does not certify camera calibration or guarantee a useful target view.

For a self-contained 2×3 inspection of the exact portable bundle, including the backend depth, CoWTracker-style source tracks, dense target-camera pointmap reprojection and relative camera path, run:

```bash
python inference/visualize_raw_condition.py \
  --condition outputs/raw-run/condition/condition.npz \
  --output-dir outputs/raw-run/visualization
```

After sampling, add `--generated-video outputs/raw-run/sampling/14b/<sequence>/pred_rgb.mp4` to also write a separately labelled `postgen_<backend>_condition_track_overlay.mp4`. The input is the condition bundle, not the reconstructor cache, so this view is tied to the exact geometry actually sent to Holo4D. It projects source tracks onto generated RGB; it does not independently track the generated video.

## 5. Environment and dependency notes

- The released Holo4D environment, model files, and inference commands are documented in the top-level [README](../README.md).
- 4RC, CoWTracker, VGGT-Omega and GMOS have separate dependencies, checkpoints, licenses and upstream installation requirements. Use separate conda environments when their PyTorch/CUDA requirements conflict. Pass each environment through the corresponding `--*-env` option; the runner invokes `conda run` and does not mutate or install into an environment.
- On clusters, allocate one GPU and set `CUDA_VISIBLE_DEVICES` before invoking the script. Backend commands use the visible device index (default `0`). Load the cluster's supported CUDA runtime before launching each environment.
- CoWTracker and VGGT-Omega use 512-class inference canvases; this differs from the final Holo4D canvas. The source camera, depth, tracks, RGB and all intrinsics are rescaled/mapped by the existing loader to the chosen Holo4D canvas.
- Network downloads or gated VGGT-Omega weights may require separate acceptance/authentication. The preprocessing command does not bypass those terms.

## 6. Verification status

CPU contract and renderer tests pass. A real fresh 4RC 49-frame preprocessing run produced a portable condition bundle; `sample_examples.py --validate-only` accepted it, and the matching AB12 step-19500 14B sampler completed all 50 steps and wrote a 49-frame 672×384, 16-FPS MP4. The later clone audit also ran fresh CoW/VGGT + GMOS through validated condition export. These checks use installed upstream environments, rather than establishing a clean-machine install; GMOS-S is not covered. The raw 4RC output uses static source K from frame 0, matching the established inference loader; compatibility of changing that convention remains unverified. Its target-view pointmap support is spatially limited, so that example is an execution smoke rather than a recommended visual demo.
