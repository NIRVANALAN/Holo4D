# Visualization and preview guide

## Newly verified inputs and outputs

[The October 18 reproduction bundle](release/2026-10-18-verified-input-target-examples.md)
contains input/generated-target pairs for 14B 49 frames, 14B 246 frames and
commercial 5B 49 frames, with exact conditions and captions. Targets are model
predictions. These are separate from the older gallery references below.

Use `inference/visualize_raw_condition.py` to inspect a downloaded or newly
reconstructed `condition.npz`; the [inference guide](../inference/README.md#4-output-layout-and-inspection)
provides the command. Its 2×3 review includes source RGB/depth/tracks, target
pointmap reprojection and relative cameras. Source tracks projected onto
generated RGB are labelled as visualization; this is not an independent
measurement of the generated video's tracking accuracy.

## Start here

Two approved full robot previews and all source-side replay conditions are now
indexed in [the robot rollout release](release/2026-10-11-prepared-robot-long-rollouts.md).
Repository previews: `assets/demo/robot_long/tri_static_high.mp4` (246frames)
and `figure0_orbit.mp4` (81frames). They are Source | Generated reviews, not
continuous tracking overlays. Original-resolution paper PNGs are separate.

| Preview | Location | Purpose |
|---|---|---|
| README results | `media/docs/results.gif` | Project-page interleaved players (source + 3D tracks, condition, synthesized view; clean and overlay loops) for a Vista4D, a DAVIS and a DROID video |
| README comparison | `media/docs/comparison.gif` | Source / Vista4D / Holo4D on two Vista4D-benchmark trajectories |
| README teaser and overview | `media/docs/teaser.jpg`, `media/docs/overview.jpg` | Exports of the paper's Figures 1 and 2 |
| Ten-example video browser | `media/docs/gallery.html` | Source/result, caption, backend, trajectory and copyable command |
| Preview provenance | `media/docs/manifest.json` | Pinned HF revision and fresh versus archived output labels |

The gallery does not fetch all ten videos at startup. It displays first-frame
posters, has `preload="none"`, and attaches media URLs only when the pair's
play button is pressed. Selecting another pair releases the previous pair's
video URLs. No account, JavaScript library or model is needed for playback.
Videos hosted on HF need internet access; the two fresh presentation copies
are local files. The viewer does not claim frame-perfect synchronized playback.

Run from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/media/docs/gallery.html`. On a remote cluster,
forward port8000 with your normal SSH tunnel. GitHub's README renderer shows
the GIFs and images, but a repository HTML link is not a hosted website.

## Provenance, not just attractive pictures

The bear and Figure0 previews use fresh50step public14B validation outputs.
The other eight use archived example-pack predictions and are labelled as
such: they are not newly reproduced50step results, GT targets or paper metrics.
The source caption and input contract come from the pinned prepared manifest.

First-frame posters and contact sheets retain the aspect ratio. Scaling is
only for display, not a change to model inputs. Smaller H.264 presentation
copies retain49frames and16FPS; the original inference MP4s are unchanged.
Source-video copyrights remain upstream, including DAVIS/DROID/Figure/Tesla.

## Regenerate the README assets

README GIFs are rebuilt from the pinned project-page videos on the Hub (CPU only, FFmpeg + Pillow):

```bash
python scripts/build_readme_media.py --output-dir media/docs
```

The example gallery, posters and manifest come from the prepared example pack and a release-validation root:

```bash
python scripts/build_readme_visuals.py \
  --bundle-root /path/to/robot_davis10 \
  --validation-root /path/to/validation \
  --output-dir media/docs
```

If the validation root has fresh `14b/<example>/pred_rgb.mp4` files, they replace archived references in the gallery.
`teaser.jpg` and `overview.jpg` are exported from the paper's Figure 1 and Figure 2 PDFs.

## Inspect geometry before expensive sampling

1. Compare source RGB and tracks projected with the **source** camera. A
   stationary camera does not make moving-object tracks stationary.
2. Inspect target reprojection with **every target frame's K** and pose. RGB,
   depth, masks and intrinsics must share the same crop and canvas.
3. Compare source and target camera centers and viewing directions in one
   world gauge. Orbiting needs camera translation around a center, not yaw only.
4. Show a spatially balanced track subset with CoWTracker-style colors and
   visible trails using `holo4d/visualization/cowtracker_tracks.py`. Keep the
   subset/colors stable across panels; it is not the full inference population.
5. Label pointmaps/tracks as diagnostic geometry, never as model-generated RGB.
   A pointmap render is not an extra Holo4D input branch.

The public preview gallery is for easy example browsing. Historical research
camera-comparison galleries and Viser scripts are separate; they are not
silently advertised here as a validated interactive camera-design UI.

See [release contracts](release/2026-10-08-inference-refresh.md) and
[configuration categories](../config/holo4d/README.md).
