# Prepared robot long-rollout release — 2026-10-11

Two author-approved long-video demonstrations accompany the existing49-frame
examples. Full source-side preprocessing is distributed separately from the
small repository MP4s; neither the original 4RC environment nor the original dataset
paths are needed to consume the prepared conditions.

| Example | Real length | Window count | Source backend |
|---|---:|---:|---|
| TRI static-high |246frames|6|4RC q0 throughout, no GMOS|
| Figure0 orbit |81frames|2|Historical CoW/VGGT+GMOS seed, then4RC q0 without GMOS|

Entry point: [demo_robot_rollouts.sh](../../inference/demo_robot_rollouts.sh).

```bash
EXAMPLE=tri_static_high bash inference/demo_robot_rollouts.sh --validate-only
EXAMPLE=tri_static_high bash inference/demo_robot_rollouts.sh
EXAMPLE=figure0_orbit bash inference/demo_robot_rollouts.sh
```

The script downloads only `examples/robot_long2_20261011` from
`yslan/Holo4D-2026`, pinned to`8c203a6d6ce07908fb94e6c0e5cc8ae258a997be`.
Each condition has source49-frame RGB, dense3D tracks, validity, source depth,
camera poses, query RGB/depth payloads and all target K. The manifest carries
per-condition SHA256 and the exact source caption. The conditions are already
in the sampling gauge; do not rescale them again.

Seed model:14B AB12 T2V step19500. Continuation:14B native I2V step22500.
Defaults:50steps,CFG5,shift5,seed42,49frame windows,16FPS/quality9.
Later windows use the freshly generated previous target's last frame.
Input padding is trimmed and one-frame overlaps deduplicated for the full
output. Archived reference videos are not model inputs or byte-exact fixtures.
`T2V_ROOT`, `I2V_ROOT`, `T2V_CHECKPOINT`, `I2V_CHECKPOINT` can point to already
downloaded weights. Completed chunks are reused only when their checksum,
checkpoint, boundary and sampling contracts match; incompatible outputs fail.

Tests:8 focused CPU tests pass. Eight exported window payloads agree with
archived tracks/target poses/K exactly; projected-depth differences are under
4.8e-7. A1step/two-window-per-example public-H200 real-model smoke is separate
from these CPU checks; it is not a new50step quality result.

Both small previews and posters are under `assets/demo/robot_long/`, totalling
about1.5MB. Long previews contain Source | Generated RGB only. Source-media
rights are upstream; the code license does not grant commercial rights to
Figure or DROID footage. These are qualitative examples, not motion-fidelity
or continuous-camera-orbit benchmarks.
