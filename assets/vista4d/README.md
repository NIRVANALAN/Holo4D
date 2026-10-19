# Vista4D public-code samples

These ten 384 × 672, 49-frame videos were generated with the public `inference/sample_vista4d.sh` pipeline and `t2v_49_chunk.safetensors`. The run used 50 denoising steps, CFG 5, sigma shift 5, seed 42, source-frame-0 anchoring, camera damping 0.5, static-depth scale alignment, and focal-ratio transfer.

| Source | Official trajectory | Video |
|---|---|---|
| avocado-slice | close-crane-above | [MP4](avocado-slice__close-crane-above.mp4) |
| avocado-slice | left-front-zoom | [MP4](avocado-slice__left-front-zoom.mp4) |
| basketball-four | right-left | [MP4](basketball-four__right-left.mp4) |
| basketball-four | shoulder-out | [MP4](basketball-four__shoulder-out.mp4) |
| bed-shopping | dolly-out | [MP4](bed-shopping__dolly-out.mp4) |
| bed-shopping | top-arc | [MP4](bed-shopping__top-arc.mp4) |
| bmx-bumps | crane-follow-zoom | [MP4](bmx-bumps__crane-follow-zoom.mp4) |
| bmx-bumps | front-follow | [MP4](bmx-bumps__front-follow.mp4) |
| breakdance | crane-dolly-zoom | [MP4](breakdance__crane-dolly-zoom.mp4) |
| breakdance | left-right | [MP4](breakdance__left-right.mp4) |

Machine-readable frame counts, byte sizes, and SHA-256 digests are recorded in [`manifest.json`](manifest.json).
