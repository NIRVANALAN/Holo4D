# Ready-to-run demo outputs

These are the checked reference outputs for the two no-data-path inference demos.
Both were generated at 672 x 384 with seed 42, 50 denoising steps, CFG 5, and
sigma shift 5.

| Demo | Frames | Reference output | Command |
|---|---:|---|---|
| First chunk | 49 | [t2v_49_chunk.mp4](t2v_49_chunk.mp4) | `bash inference/demo_t2v_49.sh` |
| Generated-boundary rollout | 97 (49 + 48) | [i2v_49_plus_48_rollout.mp4](i2v_49_plus_48_rollout.mp4) | `bash inference/demo_i2v_97.sh` |

The rollout's second chunk is conditioned on the first chunk's generated final
frame. Its repeated frame 0 is removed during stitching. The scripts download
the compact input bundle, pinned Wan bases, and released Holo4D checkpoints on
first use; no external dataset path is required.
