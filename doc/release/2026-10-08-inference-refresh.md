# Inference release refresh — 2026-10-08

## Included updates

- Persistent identities receive every target-frame K, rather than silently reusing K0 during zoom.
- Current dense-track projection and static/dynamic fusion helpers are included, along with an explicit 4RC fusion flag. Merely loading a GMOS mask is not equivalent to enabling fusion.
- FlashAttention compatibility is configured before model imports to avoid duplicate standalone/xFormers FA3 registration.
- The 14B I2V driver preserves continuation-only resume and previous-generated-frame conditioning.
- Video artifacts use the shared imageio/H.264 helper, 16 FPS and quality 9.
- Native Vista4D cameras and historical shared-first-frame half-viewpoint cameras are explicitly different launcher modes.
- 5B commercial/no-identity 384p and full-mixture/identity-target 704p are separate model recipes; known 5B I2V boundary defects remain documented, not declared solved.

## Portable example format

`holo4d-dense-example-v1` uses a relative-path JSON manifest and pickle-free compressed NPZ. Each entry has its own caption, source backend, crop/canvas, trajectory, identity count and SHA-256. The source video is stored losslessly in the NPZ; its MP4 is for viewing. An archived prediction is a qualitative reference only and never enters model conditioning.

| Field | Shape / meaning |
|---|---|
| `source_video` | uint8 `(49,H,W,3)` |
| `tracks_world` | float `(N,49,3)`, all dense identities |
| `tracks_world_valid_mask` | bool `(N,49)`; absent upstream mask becomes explicit all-true |
| `source_cam2world`, `target_cam2world` | `(49,4,4)`, same world gauge as tracks |
| `source_intrinsics`, `target_intrinsics` | `(3,3)` or `(49,3,3)`, canvas pixel coordinates |
| `source_depth_video` | all 49 depth frames in the same reconstructed scale |
| `source_rgb_at_tracks` | query-sampled appearance used by the canonical loader |
| Optional masks and query indices | retained without reducing the identity count |

No GT target RGB or reference prediction becomes an image condition. This first-window driver is not an I2V target-image demo. Visualization may display fewer identities, but sampling uses the full NPZ population.

Set `EVAL_T5_CACHE` to an existing matching UMT5 prompt-feature cache to omit T5 loading. The existing prompt-cache layer must match the actual caption and negative prompt; missing cache files fail rather than silently switching text encoders. Without a cache, captions are printed and encoded online by UMT5.

For 704p 5B, the example driver resizes RGB and depth bilinearly, masks with nearest-neighbor, and both source/target K with the image scale. World coordinates, camera extrinsics and identity count are unchanged. Landscape-only full-mixture support is intentional until a portrait recipe is validated.

## External backend integration

For a new video, prepare fields matching this contract using 4RC or VGGT-Omega/CoWTracker. Reconstructed points and tracks must share a gauge; camera-to-world matrices and intrinsics must match the RGB crop and resize. GMOS masks must match that same crop. The prepared-cache example driver is self-contained; upstream reconstructors, their packages and their weights are not all vendored or installed by this release.

## Maintenance and validation

Research synchronization tools are not part of the public release. When porting
an upstream change, review the diff and run the release tests before accepting
it; research files do not contain every public-specific adaptation. Do not
blindly synchronize the complete training repository.

See [configuration categories](../../config/holo4d/README.md) and the [visualization guide](../visualization.md).

Real 50-step generation gates passed for **14B 384p**, **commercial 5B 384p** and **full-mixture 5B 704p**. Each output has 49 frames, H.264 and 16 FPS; contact sheets were inspected. The **native portrait 14B** and **97-frame 14B continuation** gates also completed successfully. The continuation's condition PNG is byte-identical to the seed's generated last frame; no GT boundary is supplied. Sampled frames from both were inspected, but execution success does not establish uniformly good framing or benchmark quality.

The fresh continuation exposed a lost compact-NPZ routing hook in the port; both dataset entrypoints are restored and regression-tested. It also exposed old stitch defaults of12FPS/quality8. The public stitch launcher/renderer now use16FPS and the shared quality9 setting. Corrected artifacts were exported separately without diffusion resampling; original outputs were preserved. Historical parity records remain historical evidence.

## Publication and media rights

HF prefix: [yslan/Holo4D-2026/examples/robot_davis10](https://huggingface.co/yslan/Holo4D-2026/tree/main/examples/robot_davis10), published in commit `974e77018656957a6baf784cbfe9f9788186b5f3`. The downloaded manifest matches local byte-for-byte. The code is Apache-2.0; source video copyrights and dataset-specific restrictions remain upstream. Do not assign the model repository's license to Figure/Tesla or DROID/DAVIS source footage. Absolute cluster provenance is maintained outside the public pack.
