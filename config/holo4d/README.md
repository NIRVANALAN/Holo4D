# Configuration index

Choose configurations by the **checkpoint's training contract**, not by a
similar filename. Data configs, track encoders and historical ablations are
not interchangeable. Public entrypoints select the compatible defaults;
consult [the model table](../../README.md#models) before overriding them.

## Released inference configurations

| Model | Track configuration | Public entrypoint | Canvas, height × width |
|---|---|---|---|
| First-window 14B | `default_conv3d_patchify_srcdepth_srcrgb.yaml` | `inference/demo_examples.sh`, `inference/sample_19500_syn4d.sh` | 384×672; prepared native portrait examples also validated |
| Continuation 14B | `default_conv3d_patchify_srcdepth_srcrgb.yaml` | `inference/sample_mixed_t2v_i2v_syn4d.sh` | 384×672, generated boundary image |
| Commercial-mixture 5B | `default_conv3d_patchify_srcdepth_srcrgb.yaml` | `MODEL=5b-commercial bash inference/demo_examples.sh` | 384×672, no persistent-identity embedding |
| Full-mixture 5B | `persistent_identity_target_srcdepth_srcrgb.yaml` | `MODEL=5b-full bash inference/demo_examples.sh` | 704×1280, identity-target; landscape only |

An encoder YAML is only one part of the contract: backbone initialization,
camera injection, track injection, rendered-branch enablement and image
boundary also matter. Use the documented launcher rather than constructing a
new recipe from the YAML alone. `persistent_identity_shared_srcdepth_srcrgb.yaml`
is a different identity variant; it is not the released full-mixture recipe.

## Training configurations

| File | Meaning |
|---|---|
| `train_384p.yaml` | Holo4D-14B data recipe, used by `train/train_384p.sh` |
| `train_i2v_384p.yaml` | 14B continuation-model data recipe, used by `train/train_i2v_384p.sh` |
| `default*.yaml` | Track-encoder configurations inherited by the released ones above |

Set your dataset roots in the data recipes before training.
