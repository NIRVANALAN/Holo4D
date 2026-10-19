"""
Arbitrary-feature Gaussian Splatting renderer for Track4D-360.

A thin gsplat wrapper that mirrors the API of
:func:`holo4d.models.gs_renderer.gaussian_render`, but rasterises
**arbitrary D-channel per-Gaussian features** instead of spherical
harmonics. The intended use is the SoM-style "render 3D positions as colors"
trick used by the dynamic GS loss to compare per-frame Gaussian positions
against ``target_track_video[t]``.

This module is intentionally **separate** from ``gs_renderer.py`` and does
not import from it. The existing static GS rendering path stays byte-
identical.

gsplat support
--------------

``gsplat.rasterization(..., colors=(N, D), sh_degree=None, render_mode="RGB")``
returns a ``(C, H, W, D)`` feature image with no SH evaluation, just alpha-
composited features. We render one view at a time so that we can re-use the
existing per-view checkpointing pattern from ``gs_renderer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gsplat import rasterization


# ---------------------------------------------------------------------------
# Single-view feature renderer
# ---------------------------------------------------------------------------


def _render_single_view_features(
    xyz: torch.Tensor,
    features: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    test_c2w: torch.Tensor,
    test_intr: torch.Tensor,
    W: int,
    H: int,
    near_plane: float,
    far_plane: float,
    bg_features: torch.Tensor,
):
    """Rasterise a single view to a (1, H, W, D) feature map and (1, H, W, 1) alpha.

    The bg_features tensor is composited manually after gsplat's alpha so we
    do not depend on gsplat's evolving ``backgrounds`` API.
    """
    test_w2c = test_c2w.float().inverse().unsqueeze(0)  # (1, 4, 4)
    test_intr_i = torch.zeros(3, 3, device=test_intr.device)
    test_intr_i[0, 0] = test_intr[0]
    test_intr_i[1, 1] = test_intr[1]
    test_intr_i[0, 2] = test_intr[2]
    test_intr_i[1, 2] = test_intr[3]
    test_intr_i[2, 2] = 1
    test_intr_i = test_intr_i.unsqueeze(0)  # (1, 3, 3)

    rendering, alpha, _ = rasterization(
        xyz,
        rotation,
        scale,
        opacity,
        features,            # (N, D) per-Gaussian features
        test_w2c,
        test_intr_i,
        W,
        H,
        sh_degree=None,      # disables SH evaluation, treats colors as raw features
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode="RGB",   # alpha-compositing on the feature channels
        rasterize_mode="classic",
    )
    # rendering: (1, H, W, D), alpha: (1, H, W, 1)
    composed = rendering + (1 - alpha) * bg_features[None, None, None, :]
    return composed, alpha


# ---------------------------------------------------------------------------
# Per-view checkpointing autograd Function (mirrors gs_renderer.py pattern)
# ---------------------------------------------------------------------------


class GaussianFeatureRendererWithCheckpoint(torch.autograd.Function):
    """Memory-efficient feature rasterisation via per-view gradient checkpointing.

    Forward pass renders all views without storing intermediate activations.
    Backward pass re-computes each view's forward pass one at a time.
    """

    @staticmethod
    def forward(
        ctx,
        xyz,
        features,
        scale,
        rotation,
        opacity,
        test_c2ws,
        test_intr,
        W,
        H,
        near_plane,
        far_plane,
        bg_features,
    ):
        ctx.save_for_backward(
            xyz, features, scale, rotation, opacity, test_c2ws, test_intr, bg_features
        )
        ctx.W = W
        ctx.H = H
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        D = features.shape[-1]
        with torch.no_grad():
            V, _ = test_intr.shape
            renderings = torch.zeros(V, H, W, D, device=xyz.device, dtype=features.dtype)
            for iv in range(V):
                rendering, _ = _render_single_view_features(
                    xyz, features, scale, rotation, opacity,
                    test_c2ws[iv], test_intr[iv], W, H,
                    near_plane, far_plane, bg_features[iv],
                )
                renderings[iv : iv + 1] = rendering
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, features, scale, rotation, opacity, test_c2ws, test_intr, bg_features = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        features = features.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W, H = ctx.W, ctx.H
        near_plane, far_plane = ctx.near_plane, ctx.far_plane
        with torch.enable_grad():
            V, _ = test_intr.shape
            for iv in range(V):
                rendering, _ = _render_single_view_features(
                    xyz, features, scale, rotation, opacity,
                    test_c2ws[iv], test_intr[iv], W, H,
                    near_plane, far_plane, bg_features[iv],
                )
                rendering.backward(grad_output[iv : iv + 1])
        return (
            xyz.grad,
            features.grad,
            scale.grad,
            rotation.grad,
            opacity.grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gaussian_render_features(
    xyz: torch.Tensor,
    features: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    test_c2ws: torch.Tensor,
    test_intr: torch.Tensor,
    W: int,
    H: int,
    near_plane: float = 0.01,
    far_plane: float = 1000.0,
    use_checkpoint: bool = False,
    bg_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render 3D Gaussians with arbitrary D-channel features per primitive.

    Args:
        xyz: ``(B, V, N_pts, 3)`` per-Gaussian world-space positions, where
            ``B`` is batch and ``V`` is the number of views to render. The
            same Gaussians are rendered into all V views (caller broadcasts).
        features: ``(B, V, N_pts, D)`` per-Gaussian features. Pass per-
            Gaussian xyz here for the position-map trick.
        scale: ``(B, V, N_pts, 3)`` Gaussian scales.
        rotation: ``(B, V, N_pts, 4)`` Gaussian quaternions (real-first).
        opacity: ``(B, V, N_pts)`` Gaussian opacities in ``[0, 1]``.
        test_c2ws: ``(B, V, 4, 4)`` camera-to-world matrices.
        test_intr: ``(B, V, 4)`` intrinsics ``[fx_px, fy_px, cx_px, cy_px]``.
        W, H: render resolution.
        near_plane, far_plane: clipping planes.
        use_checkpoint: use per-view gradient checkpointing for memory.
        bg_value: scalar background value used for every feature channel.
            Use ``0.0`` for the position-map trick (so masked-L1 ignores
            unrendered regions).

    Returns:
        features_image: ``(B, V, D, H, W)`` rendered feature image.
        alpha: ``(B, V, 1, H, W)`` accumulated alpha mask (1 where Gaussians
            covered the pixel, 0 elsewhere). Useful for masking against the
            ground-truth valid mask.
    """
    if not torch.is_grad_enabled():
        use_checkpoint = False

    if xyz.dim() != 4 or features.dim() != 4:
        raise ValueError(
            f"xyz and features must be 4D (B, V, N, *); got xyz={tuple(xyz.shape)}, "
            f"features={tuple(features.shape)}"
        )

    # OpenGL → COLMAP convention (flip Y and Z axes) — matches gs_renderer.py
    test_c2ws = test_c2ws.clone()
    test_c2ws[:, :, :3, 1:3] *= -1

    device = test_intr.device
    B, V, _ = test_intr.shape
    D = features.shape[-1]

    # Per-view background feature vector — same value across all D channels.
    bg_features = torch.full((V, D), float(bg_value), device=device, dtype=features.dtype)

    feature_imgs = []
    alphas = []
    for ib in range(B):
        xyz_i = xyz[ib].reshape(-1, 3).float()
        features_i = features[ib].reshape(-1, D)
        scale_i = scale[ib].reshape(-1, 3).float()
        rotation_i = rotation[ib].reshape(-1, 4).float()
        opacity_i = opacity[ib].reshape(-1).float()

        if use_checkpoint:
            rendering = GaussianFeatureRendererWithCheckpoint.apply(
                xyz_i, features_i, scale_i, rotation_i, opacity_i,
                test_c2ws[ib], test_intr[ib], W, H,
                near_plane, far_plane, bg_features,
            )
            # rerun once without grad to also collect alpha (cheap, just for masking)
            with torch.no_grad():
                alpha_views = torch.zeros(V, H, W, 1, device=device, dtype=features.dtype)
                for iv in range(V):
                    _, alpha_i = _render_single_view_features(
                        xyz_i, features_i, scale_i, rotation_i, opacity_i,
                        test_c2ws[ib][iv], test_intr[ib][iv], W, H,
                        near_plane, far_plane, bg_features[iv],
                    )
                    alpha_views[iv : iv + 1] = alpha_i
        else:
            rendering = torch.zeros(V, H, W, D, device=device, dtype=features.dtype)
            alpha_views = torch.zeros(V, H, W, 1, device=device, dtype=features.dtype)
            for iv in range(V):
                r, a = _render_single_view_features(
                    xyz_i, features_i, scale_i, rotation_i, opacity_i,
                    test_c2ws[ib][iv], test_intr[ib][iv], W, H,
                    near_plane, far_plane, bg_features[iv],
                )
                rendering[iv : iv + 1] = r
                alpha_views[iv : iv + 1] = a

        feature_imgs.append(rendering)
        alphas.append(alpha_views)

    feature_imgs = torch.stack(feature_imgs, dim=0)  # (B, V, H, W, D)
    alphas = torch.stack(alphas, dim=0)              # (B, V, H, W, 1)

    feature_imgs = feature_imgs.permute(0, 1, 4, 2, 3).contiguous()  # (B, V, D, H, W)
    alphas = alphas.permute(0, 1, 4, 2, 3).contiguous()              # (B, V, 1, H, W)
    return feature_imgs, alphas
