# Training data

This document records the data mixture used by the two public 384p training
recipes:

- [`config/holo4d/train_384p.yaml`](../config/holo4d/train_384p.yaml) for the
  49-frame V2V model.
- [`config/holo4d/train_i2v_384p.yaml`](../config/holo4d/train_i2v_384p.yaml)
  for the 49-frame I2V continuation model.

Both recipes train at 384 x 672 and have a logical epoch size of 250,000
samples. In the YAML syntax, `N @ Dataset(...)` resizes that dataset to `N`
logical samples before concatenation. The percentages below are therefore the
configured sampling proportions for one logical epoch; they are not raw file,
frame, or unique-scene percentages. Shuffling and distributed sampling preserve
these proportions apart from sampler padding or failed-sample retries.

## V2V mixture

| Training source | Logical samples | Share | Role |
| --- | ---: | ---: | --- |
| ReCamMaster / MultiCamVideo, dense one-query tracks | 52,500 | 21% | Dynamic |
| ReCamMaster / MultiCamVideo, sparse stride-2 tracks | 22,500 | 9% | Dynamic |
| Syn4D | 100,000 | 40% | Dynamic |
| DL3DV | 50,000 | 20% | Static |
| MVS-Synth | 25,000 | 10% | Static |
| **Total** | **250,000** | **100%** | **70% dynamic / 30% static** |

The two ReCamMaster rows together contribute 30%. In the Syn4D row,
`sparse_track_prob=0.5` and `multi_query_prob=0.5` are augmentations within
that row, not additional dataset weights. `flying_group` is excluded through
the recipe, and the accepted Syn4D sequence list is
[`config/syn4d_valid_sequences.txt`](../config/syn4d_valid_sequences.txt).
MVS-Synth sequence `0000` is held out.

## I2V continuation mixture

| Training source | Logical samples | Share | Role |
| --- | ---: | ---: | --- |
| OmniDROID | 62,500 | 25% | Dynamic real-world robot data |
| ReCamMaster / MultiCamVideo, dense one-query tracks | 39,375 | 15.75% | Dynamic |
| ReCamMaster / MultiCamVideo, sparse stride-2 tracks | 16,875 | 6.75% | Dynamic |
| Syn4D | 75,000 | 30% | Dynamic |
| DL3DV | 37,500 | 15% | Static |
| MVS-Synth | 18,750 | 7.5% | Static |
| **Total** | **250,000** | **100%** | **77.5% dynamic / 22.5% static** |

The two ReCamMaster rows together contribute 22.5%. Every I2V sample exposes
one truthful frame-0 query, matching the boundary frame used for rollout
continuation. OmniDROID uses `same_view_prob=0.1` inside its 25% allocation,
so identity-camera source-to-source targets account for an expected 2.5% of all
I2V samples. Its `swap_src_tgt_prob=0.5` changes direction within OmniDROID
but does not change the top-level mixture. The I2V recipe also excludes Syn4D
`flying_group` and MVS-Synth sequence `0000`.

## Public data sources

| Config name | Public source | Access and license note |
| --- | --- | --- |
| Syn4D | [Hugging Face](https://huggingface.co/datasets/Syn4D/Syn4D), [project](https://jzr99.github.io/Syn4D/), [code](https://github.com/jzr99/Syn4D) | CC BY 4.0. Read the included third-party notices before redistribution. The released stride-1 and stride-5 archives contain the RGB, depth, masks, tracks, and metadata used by the loader. |
| ReCamMaster / MultiCamVideo | [Hugging Face](https://huggingface.co/datasets/KlingTeam/MultiCamVideo-Dataset), [project](https://jianhongbai.github.io/ReCamMaster/), [code](https://github.com/KlingAIResearch/ReCamMaster) | The dataset card lists Apache 2.0. It provides synchronized multi-camera videos and camera trajectories; Holo4D additionally consumes the derived caches listed below. |
| OmniDROID | [OmniWorld annotations](https://huggingface.co/datasets/InternRobotics/OmniWorld), [OmniWorld code](https://github.com/yangzhou24/OmniWorld), and [raw DROID videos](https://droid-dataset.github.io/droid/the-droid-dataset) | Holo4D's `OmniDROID` name means OmniWorld-DROID annotations aligned to original DROID RGB videos; it is not a separate upstream dataset. OmniWorld is CC BY-NC-SA 4.0. Follow the DROID download terms for the raw MP4 files. |
| DL3DV | [DL3DV-ALL-480P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P), [download tools](https://github.com/DL3DV-10K/Dataset) | The 480p repository contains frames and camera poses. It is gated: users must sign in, accept the dataset conditions, and follow the DL3DV license. |
| MVS-Synth | [upstream data](https://huggingface.co/datasets/phuang17/MVS-Synth), [project page](https://phuang17.github.io/DeepMVS/mvs-synth.html), [Holo4D-friendly processed copy](https://huggingface.co/datasets/yslan/processed_mvs_synth) | The upstream card labels the license as `other` and describes research/educational use. The processed copy does not replace the upstream terms. |

Dataset licenses are independent of the Holo4D code license. Check every
upstream license and terms-of-use page before downloading, training, or
redistributing derived data.

## What the configs expect

The public URLs above are the canonical raw-data sources. Several rows also
require Holo4D-specific geometry, tracking, caption, or index caches; downloading
only the raw videos is not sufficient for full training.

| Source | Environment variables | Expected content |
| --- | --- | --- |
| Syn4D | `SYN4D_ROOT`, `SYN4D_CAPTION_DIR` | Extracted Syn4D scene archives plus the corresponding caption/T5 cache directory. |
| ReCamMaster | `RECAMMASTER_ROOT`, `RECAMMASTER_CAPTION_CACHE_DIR`, `MULTICAM_FG_MASK_ROOT`, `RECAMMASTER_SOURCE49_ROOT`, `RECAMMASTER_PAIR1X49_LABEL_ROOT`, `RECAMMASTER_COW_ROOT` | Raw MultiCamVideo scenes, captions, foreground masks, 49-frame VGGT source geometry, pair scale labels, and CoTracker/CoW-style tracks. |
| DL3DV | `DL3DV_ROOT` | Holo4D/CUT3R-style DL3DV scenes with poses and generated DA3 depth used by `use_da3_depth=True`. |
| MVS-Synth | `MVS_SYNTH_ROOT` | Processed MVS-Synth scene layout accepted by `StaticScene_NVS`; the processed Hugging Face copy above is the convenient starting point. |
| OmniDROID | `OMNI_DROID_ROOT`, `OMNI_DROID_INDEX`, `OMNI_DROID_CACHE_MANIFEST`, `DROID_NVS_INDEX`, `OMNI_CAPTION_DIR` | Extracted OmniWorld-DROID annotations, matching raw DROID MP4 recordings, an episode index, a track-cache manifest, optional optimized-extrinsics entries, and captions. |

Relevant in-repository cache tools include
[`holo4d/precompute_t5_cache.py`](../holo4d/precompute_t5_cache.py),
[`holo4d/precompute_t5_cache_recam.py`](../holo4d/precompute_t5_cache_recam.py),
[`holo4d/tools/cowtracker_source_track_batch.py`](../holo4d/tools/cowtracker_source_track_batch.py),
and
[`holo4d/tools/label_recammaster_foreground_sam3.py`](../holo4d/tools/label_recammaster_foreground_sam3.py).
The JSON indices and geometry/scale caches are derived metadata, not additional
public datasets. Generate them in the exact layouts consumed by the loaders and
do not redistribute upstream content unless its license permits it.

After preparing the data, export the variables shown above and use:

```bash
# 49-frame V2V
bash train/train_384p.sh

# 49-frame I2V continuation
bash train/train_i2v_384p.sh
```

The launchers use these two YAML files by default. If `DATA_CONFIG` is
overridden, the mixture in that replacement file, rather than the percentages
documented here, becomes authoritative.
