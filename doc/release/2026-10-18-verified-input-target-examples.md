# Verified source, generated target and condition bundles

The October 18 A100 release-validation outputs are available in the model Hub,
alongside the exact model-ready inputs used to produce them:

[Browse the pinned bundle](https://huggingface.co/yslan/Holo4D-2026/tree/457c61c322756835758ccc3478bba74f7dbcaf86/examples/release_validation_20261018).

HF revision: `457c61c322756835758ccc3478bba74f7dbcaf86`.
Prefix: `examples/release_validation_20261018/`.

| Relative folder | Video | Source backend | Model |
| --- | --- | --- | --- |
| `single49/14b_figure1/` | 49 frames, 384×672 portrait | CoWTracker + VGGT-Ω, GMOS mask loaded | AB12 T2V-14B step-19500 |
| `long/tri_static_high/` | 246 frames, 672×384 | 4RC q0, no GMOS | T2V-14B step-19500 → I2V-14B step-22500 |
| `single49/5b_commercial_davis_bear/` | 49 frames, 672×384 | 4RC q0, no GMOS | Commercial-mixture 5B step-29000, dense/no identity embedding |

Dimensions above are width×height. Every video is 16 FPS. Each folder contains
`source.mp4`, `target.mp4` and a portable `manifest.json`; the single windows
contain `condition.npz`, and the long case contains six lossless condition
windows under `chunks/`. **Targets are generated predictions, not GT.**
The Bear clip is a software-validation input, not a selected paper highlight.

## Download and reproduce

Download from inside your Holo4D repository:

```bash
hf download yslan/Holo4D-2026 \
  --revision 457c61c322756835758ccc3478bba74f7dbcaf86 \
  --include 'examples/release_validation_20261018/**' \
  --local-dir ./holo4d-hub-data
```

The bundle's `README.md` contains exact 49-frame and long-video commands,
checkpoint hashes, captions and settings. Validate the 14B single window
without loading diffusion weights:

```bash
EXAMPLE_MANIFEST="$PWD/holo4d-hub-data/examples/release_validation_20261018/single49/14b_figure1/manifest.json" \
MODEL=14b EXAMPLE=robot_figure1 \
bash inference/demo_examples.sh --validate-only
```

Validate the long sequence:

```bash
python inference/sample_robot_rollouts.py \
  --manifest holo4d-hub-data/examples/release_validation_20261018/long/tri_static_high/manifest.json \
  --example tri_static_high --validate-only
```

For generation, follow the installed-model/environment requirements in the
[inference guide](../../inference/README.md) and remove `--validate-only`.
Single-window generation uses 50 steps, CFG 5, shift 5, seed 42. The long runner
uses 50 steps per window and attaches the **previous generated target frame**
to each continuation, not a real target frame.

Window starts are 0/48/96/144/192/240. Their valid lengths are 49/49/49/49/49/6;
the last condition repeats its final real frame to reach 49 model frames.
Stitching removes each later window's first overlapping frame and trims the
padding, producing 246 real frames, not 294. Both complete source/target videos
and all six per-window source/generated pairs are included for inspection.

## Verification and access boundary

The original bundle has 38 files totaling 866,219,238 bytes. All 18 MP4s passed
full decoding and exact frame/raster/FPS checks; all 18 anonymous range requests
returned `206 video/mp4`, and full anonymous video downloads matched local
SHA-256. Remote sizes match every original file. An independent CPU check also
verified source/condition temporal alignment, all five exact chunk overlaps,
final padding, and the three public CLI validation paths.

The additional `ACCESS_AND_PROVENANCE.md` makes 39 files at the pinned revision.
It corrects an unverified code URL in the original `provenance.json`: the actual
tested checkout is `NIRVANALAN/Holo4D_dev`, commit
`d3e177d938e6bbc67f559dd29c992e315a35cd80`. Anonymous GitHub access returned 404
at verification time. **HF assets are public, but end-to-end reproduction still
requires access to the code until the owner publishes that repository.** No
repository visibility, Git commit/push, website pin or demo publication was
changed by this asset upload.

The original 38 files and their `SHA256SUMS` are unchanged; the later access note
is outside that original checksum list. Local validation logs are not uploaded.
This is execution/provenance validation, not new benchmark metrics, convergence
evidence or a guarantee of byte-identical results across CUDA/attention kernels.
Source footage retains its upstream rights; see the bundle's source-media note.
