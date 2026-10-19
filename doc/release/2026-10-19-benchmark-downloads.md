# Frozen benchmark predictions, conditions and metric replays

Published add-only to [yslan/Holo4D-2026](https://huggingface.co/yslan/Holo4D-2026/tree/d01da5570d116b7e8994bc4916b0d45a58a75b42/evaluation/benchmarks_20261019_v1),
revision `d01da5570d116b7e8994bc4916b0d45a58a75b42`.
Existing model weights, example assets and earlier uploads were not replaced.

| Download | Bytes | Original prediction videos |
|---|---:|---:|
| `davis.zip` | 39,330,402,461 | 1,200 |
| `vista4d110.zip` | 34,681,086,850 | 858 |
| `syn4d32.zip` | 7,157,974,316 | 572 |
| `iphone.zip` | 1,732,585,781 | 40 |
| `robot.zip` | 2,690,106,721 | 565 |

Exactly five benchmark archives, no nested ZIP shards. Historical methods,
checkpoints and protocols remain in separately named collections: these counts
are not the sample counts of a pooled evaluation. Total: 3,235 predictions,
85,592,156,129 bytes. Model weights are downloaded separately.

## Download and check

From the repository root, download only this release, not the whole model repo:

```bash
hf download yslan/Holo4D-2026 \
  --revision d01da5570d116b7e8994bc4916b0d45a58a75b42 \
  --include 'evaluation/benchmarks_20261019_v1/*' \
  --local-dir eval-results
cd eval-results/evaluation/benchmarks_20261019_v1
sha256sum -c SHA256SUMS
unzip -q davis.zip
```

To download one benchmark only, supply its exact filename to `hf download`.
Each ZIP extracts into its own benchmark directory. The shared README,
manifest and SHA256SUMS list all five archives; a full `sha256sum -c` requires
all five to be present.

## Contents and provenance

- `results/`: original prediction MP4s, baseline outputs and original run
  metadata; not re-encoded or replaced by new generations.
- `conditions/`: prepared RGB, tracks, depth, cameras, masks, captions and
  checksums. Final AB12 inputs include DAVIS three arms × 100, iPhone two
  arms × five, and the nonadjacent Syn4D 32 pairs. Native-110 has 110 inputs
  per backend (4RC and CoW/VGGT). Robot conditions cover two approved long
  examples / eight windows, not every historical robot candidate.
- `metrics/`: original full-precision scores, cached camera replays and
  separate fresh A100 comparisons. Do not mix `all_norm` with Sim(3)-subset
  values, or source-proxy FVD with paired-target FVD.
- `evaluation/`: paired Syn4D target images/cameras and iPhone GT/masks/
  source calibration. Target RGB is for scoring, not generation conditioning.

DAVIS saved tracks, masks, target cameras and normalizers are copied exactly
from the actual sampling snapshots. Source-only fields absent from those
snapshots are rebuilt from original caches and explicitly marked as rebuilt.
Native-110 and iPhone historical runs have no complete sampling snapshot;
their prepared inputs are original-cache recipe rebuilds, not certified
whole-input bit copies. The historical Syn4D32 arm retains its original
generic NVS prompt and 12 FPS, not later semantic annotations. Its pairs are
0→7 / 2→6; older GT/oracle cohorts are separate and cannot be compared as
if they used the same pair IDs.

A legacy `gmos_static_fusion` flag records the explicit 4RC switch, not
CoW's automatic mask-based fusion. See original args and per-entry provenance.
The five model recipes do not share interchangeable conditioning contracts.
Use [prepared-example inference](../../inference/README.md) and the original
checkpoint, seed, steps, CFG and shift for each entry. Source reconstruction
errors and cross-hardware numerical differences are not hidden by this export.

## Verification and limits

All five ZIPs passed local CRC/SHA256, matching remote LFS hashes and sizes,
and anonymous HTTP Range 206 checks. The three metadata files were downloaded
at the pinned revision and matched local bytes exactly. All 342 final input
IDs passed the frozen-cohort gate; 30 export/publisher/replay tests passed.

An actual ZIP-extracted native-110 CoW cached-camera replay was run with this
repository's scorer: all 110 IDs match; maximum per-pair numeric difference
is 6.106226635438361e-15, and `all_norm` passes absolute tolerance 1e-10.
This verifies scoring from cached poses, not new neural camera inference.
Earlier 19 cached-camera and 122 score-aggregation checks remain separately
recorded. Archived FVD results are not claimed as new feature-extraction runs.

The paper's sampling reference hardware is H200. Fixed seeds reproduce outputs on
the same GPU type; they are not bit-identical across GPU architectures.

Upstream dataset, model and baseline licenses still apply. The code license
does not relicense their RGB/depth assets. Temporary neural features, decoded
debug frames and redundant giant pointmap dumps are intentionally omitted.
