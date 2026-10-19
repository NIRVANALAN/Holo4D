"""
Gaussian Splatting renderer — wraps gsplat rasterization.

Ported from FlashWorld/models/render.py.
Provides differentiable rendering of pixel-aligned 3D Gaussians with
optional gradient checkpointing for memory efficiency.

Background compositing is done manually (not via gsplat's backgrounds param)
for compatibility with gsplat >= 1.5.2 which changed the backgrounds API.

Requires: pip install gsplat
"""

import torch
import torch.nn as nn

from gsplat import rasterization


def _render_single_view(xyz, feature, scale, rotation, opacity, test_c2w, test_intr,
                        W, H, sh_degree, near_plane, far_plane, bg_color):
    """Render a single view. Returns (1, H, W, 4) RGBD and (1, H, W, 1) alpha."""
    test_w2c = test_c2w.float().inverse().unsqueeze(0)  # (1, 4, 4)
    test_intr_i = torch.zeros(3, 3, device=test_intr.device)
    test_intr_i[0, 0] = test_intr[0]
    test_intr_i[1, 1] = test_intr[1]
    test_intr_i[0, 2] = test_intr[2]
    test_intr_i[1, 2] = test_intr[3]
    test_intr_i[2, 2] = 1
    test_intr_i = test_intr_i.unsqueeze(0)  # (1, 3, 3)
    rendering, alpha, _ = rasterization(
        xyz, rotation, scale, opacity, feature,
        test_w2c, test_intr_i, W, H,
        sh_degree=sh_degree,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB+D",
        rasterize_mode="classic",
    )  # rendering: (1, H, W, 4), alpha: (1, H, W, 1)
    # Manual background compositing on RGB channels
    rgb = rendering[..., :3] + (1 - alpha) * bg_color[None, None, None, :]
    depth = rendering[..., 3:]
    return torch.cat([rgb, depth], dim=-1), alpha


def _render_single_view_patch(xyz, feature, scale, rotation, opacity,
                               test_c2w, test_intr, W, H, sh_degree,
                               near_plane, far_plane, bg_color, patch_h, patch_w):
    """Render a single view using patch-by-patch tiling for memory efficiency.

    Each patch is rendered with a shifted principal point so the tile looks
    like a full frame to gsplat. Results are assembled into a full (H, W, 4)
    RGBD tensor. Background compositing is applied per-patch.

    Ported from Lyra-1 DeferredBPPatch (src/rendering/gs_deferred_patch.py).
    The principal-point shift trick: rendering patch (row=n, col=m) at size
    (patch_h, patch_w) with cx' = cx - m*patch_w, cy' = cy - n*patch_h
    makes gsplat project only the Gaussians that fall in that tile.

    Args:
        test_intr: (4,) [fx, fy, cx, cy] in pixel units.
        test_c2w: (4, 4) camera-to-world matrix.
        patch_h, patch_w: tile dimensions; must divide H and W exactly.
    Returns:
        (H, W, 4) RGBD tensor (RGB in [0,1] after bg composite, D raw depth).
    """
    fx = test_intr[0]
    fy = test_intr[1]
    cx = test_intr[2]
    cy = test_intr[3]

    test_w2c = test_c2w.float().inverse().unsqueeze(0)  # (1, 4, 4)
    rendering = torch.zeros(H, W, 4, device=xyz.device, dtype=feature.dtype)

    for n in range(H // patch_h):       # row-wise patches
        for m in range(W // patch_w):   # column-wise patches
            # Shift principal point so this tile appears as a full frame
            patch_intr = torch.zeros(3, 3, device=xyz.device, dtype=torch.float32)
            patch_intr[0, 0] = fx
            patch_intr[1, 1] = fy
            patch_intr[0, 2] = cx - m * patch_w
            patch_intr[1, 2] = cy - n * patch_h
            patch_intr[2, 2] = 1.0
            patch_intr = patch_intr.unsqueeze(0)  # (1, 3, 3)

            rgbd, alpha, _ = rasterization(
                xyz, rotation, scale, opacity, feature,
                test_w2c, patch_intr, patch_w, patch_h,
                sh_degree=sh_degree,
                near_plane=near_plane, far_plane=far_plane,
                render_mode="RGB+D",
                rasterize_mode="classic",
            )  # rgbd: (1, ph, pw, 4), alpha: (1, ph, pw, 1)

            rgb = rgbd[..., :3] + (1 - alpha) * bg_color[None, None, None, :]
            tile = torch.cat([rgb, rgbd[..., 3:]], dim=-1)  # (1, ph, pw, 4)

            r0, r1 = n * patch_h, (n + 1) * patch_h
            c0, c1 = m * patch_w, (m + 1) * patch_w
            rendering[r0:r1, c0:c1] = tile[0]

    return rendering


class GaussianRendererWithCheckpoint(torch.autograd.Function):
    """Memory-efficient Gaussian rendering via per-view gradient checkpointing.

    Forward: renders all views without storing intermediate activations.
    Backward: re-computes each view's forward pass one at a time.
    """

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane, bg_colors):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, bg_colors)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            V, _ = test_intr.shape
            renderings = torch.zeros(V, H, W, 4, device=xyz.device)
            for iv in range(V):
                rendering, _ = _render_single_view(
                    xyz, feature, scale, rotation, opacity,
                    test_c2ws[iv], test_intr[iv], W, H,
                    sh_degree, near_plane, far_plane, bg_colors[iv],
                )
                renderings[iv:iv + 1] = rendering
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, bg_colors = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W, H = ctx.W, ctx.H
        sh_degree = ctx.sh_degree
        near_plane, far_plane = ctx.near_plane, ctx.far_plane
        with torch.enable_grad():
            V, _ = test_intr.shape
            for iv in range(V):
                rendering, _ = _render_single_view(
                    xyz, feature, scale, rotation, opacity,
                    test_c2ws[iv], test_intr[iv], W, H,
                    sh_degree, near_plane, far_plane, bg_colors[iv],
                )
                rendering.backward(grad_output[iv:iv + 1])
        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None, None


class GaussianRendererPatch(torch.autograd.Function):
    """Patch-tiled deferred-backprop Gaussian rendering (Lyra-1 DeferredBPPatch style).

    Same deferred pattern as GaussianRendererWithCheckpoint but each view is
    rendered as (H//patch_h × W//patch_w) tiles. gsplat's internal
    alpha-compositing buffers scale with tile area instead of full-frame area,
    reducing peak GPU memory when tiles are small.

    Backward: re-renders each patch individually and backprops the corresponding
    gradient slice, accumulating into xyz/feature/scale/rotation/opacity.
    """

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane, bg_colors, patch_h, patch_w):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity,
                               test_c2ws, test_intr, bg_colors)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        ctx.patch_h = patch_h
        ctx.patch_w = patch_w
        with torch.no_grad():
            V, _ = test_intr.shape
            renderings = torch.zeros(V, H, W, 4, device=xyz.device)
            for iv in range(V):
                renderings[iv] = _render_single_view_patch(
                    xyz, feature, scale, rotation, opacity,
                    test_c2ws[iv], test_intr[iv], W, H,
                    sh_degree, near_plane, far_plane, bg_colors[iv],
                    patch_h, patch_w,
                )
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, bg_colors = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        patch_h, patch_w = ctx.patch_h, ctx.patch_w
        with torch.enable_grad():
            V, _ = test_intr.shape
            for iv in range(V):
                # Re-render each patch and backprop the corresponding gradient slice
                test_w2c = test_c2ws[iv].float().inverse().unsqueeze(0)
                fx = test_intr[iv][0]
                fy = test_intr[iv][1]
                cx = test_intr[iv][2]
                cy = test_intr[iv][3]
                for n in range(ctx.H // patch_h):
                    for m in range(ctx.W // patch_w):
                        patch_intr = torch.zeros(3, 3, device=xyz.device,
                                                 dtype=torch.float32)
                        patch_intr[0, 0] = fx
                        patch_intr[1, 1] = fy
                        patch_intr[0, 2] = cx - m * patch_w
                        patch_intr[1, 2] = cy - n * patch_h
                        patch_intr[2, 2] = 1.0
                        patch_intr = patch_intr.unsqueeze(0)

                        rgbd, alpha, _ = rasterization(
                            xyz, rotation, scale, opacity, feature,
                            test_w2c, patch_intr, patch_w, patch_h,
                            sh_degree=ctx.sh_degree,
                            near_plane=ctx.near_plane, far_plane=ctx.far_plane,
                            render_mode="RGB+D", rasterize_mode="classic",
                        )
                        rgb = rgbd[..., :3] + (1 - alpha) * bg_colors[iv][None, None, None, :]
                        tile = torch.cat([rgb, rgbd[..., 3:]], dim=-1)  # (1, ph, pw, 4)

                        r0, r1 = n * patch_h, (n + 1) * patch_h
                        c0, c1 = m * patch_w, (m + 1) * patch_w
                        grad_tile = grad_output[iv, r0:r1, c0:c1, :]
                        tile[0].backward(grad_tile)

        return (xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad,
                None, None, None, None, None, None, None, None, None, None)


def gaussian_render(
    gaussian_params: torch.Tensor,
    test_c2ws: torch.Tensor,
    test_intr: torch.Tensor,
    W: int,
    H: int,
    near_plane: float = 0.01,
    far_plane: float = 1000.0,
    use_checkpoint: bool = False,
    sh_degree: int = 0,
    bg_mode: str = "random",
    patch_size: "tuple[int, int] | None" = None,
    per_frame_gaussians: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Render 3D Gaussians to RGB + depth images.

    Args:
        gaussian_params: (B, V, N_pts, D) Gaussian parameters.
            D = 3 (xyz) + 1 (opacity) + 3 (scale) + 4 (rotation) + (sh_degree+1)^2*3 (SH).
        test_c2ws: (B, V, 4, 4) camera-to-world matrices.
        test_intr: (B, V, 4) intrinsics [fx_px, fy_px, cx_px, cy_px].
        W, H: render resolution.
        near_plane, far_plane: clipping planes.
        use_checkpoint: use gradient checkpointing for memory efficiency.
        sh_degree: spherical harmonics degree.
        bg_mode: 'random', 'white', or 'black'.
        per_frame_gaussians: if True, render view ``iv`` from
            ``gaussian_params[ib, iv]`` only (per-frame Gaussian set, suitable
            for video where per-frame Gaussians encode dynamic scene content).
            If False (default, FlashWorld/LRM convention), all V Gaussian sets
            are flattened into a single cloud and each view renders from the
            union — correct for multi-view LRM fusion where input views feed a
            shared static scene, INCORRECT for dynamic per-frame video.
    Returns:
        rgb: (B, V, 3, H, W) rendered RGB in [-1, 1].
        depth: (B, V, 1, H, W) rendered depth.
    """
    if not torch.is_grad_enabled():
        use_checkpoint = False

    # OpenGL → COLMAP convention (flip Y and Z axes)
    test_c2ws = test_c2ws.clone()
    test_c2ws[:, :, :3, 1:3] *= -1

    device = test_intr.device
    B, V, _ = test_intr.shape
    sh_channels = (sh_degree + 1) ** 2 * 3

    if per_frame_gaussians and gaussian_params.shape[1] != V:
        raise ValueError(
            f"per_frame_gaussians=True requires gaussian_params.shape[1] == V, "
            f"got params V={gaussian_params.shape[1]} vs cameras V={V}"
        )

    def _split_params_1d(params_1d: torch.Tensor):
        """params_1d: (N_pts, D). Returns the 5-tuple gsplat expects."""
        xyz_, opa_, scl_, rot_, feat_ = params_1d.split(
            [3, 1, 3, 4, sh_channels], dim=-1
        )
        opa_ = opa_.squeeze(-1)
        feat_ = feat_.reshape(-1, (sh_degree + 1) ** 2, 3)
        return xyz_, feat_, scl_, rot_, opa_

    renderings = []
    for ib in range(B):
        if bg_mode == "random":
            bg_colors = torch.rand(V, 3, device=device)
        elif bg_mode == "white":
            bg_colors = torch.ones(V, 3, device=device)
        elif bg_mode == "black":
            bg_colors = torch.zeros(V, 3, device=device)
        else:
            raise ValueError(f"Invalid background mode: {bg_mode}")

        if per_frame_gaussians:
            # One Gaussian set per view: render view iv from gaussian_params[ib, iv].
            # The checkpointed apply functions take a fixed Gaussian set + V cameras
            # and loop internally; for per-frame we call them with V=1 in a loop here.
            rendering = torch.zeros(V, H, W, 4, device=device)
            for iv in range(V):
                xyz_i, feature_i, scale_i, rotation_i, opacity_i = _split_params_1d(
                    gaussian_params[ib, iv].float()
                )
                c2w_iv = test_c2ws[ib, iv:iv + 1]
                intr_iv = test_intr[ib, iv:iv + 1]
                bg_iv = bg_colors[iv:iv + 1]

                if use_checkpoint:
                    if patch_size is not None:
                        ph, pw = patch_size
                        assert H % ph == 0 and W % pw == 0, (
                            f"patch_size ({ph},{pw}) must divide H={H}, W={W} exactly"
                        )
                        r1 = GaussianRendererPatch.apply(
                            xyz_i, feature_i, scale_i, rotation_i, opacity_i,
                            c2w_iv, intr_iv, W, H,
                            sh_degree, near_plane, far_plane, bg_iv, ph, pw,
                        )
                    else:
                        r1 = GaussianRendererWithCheckpoint.apply(
                            xyz_i, feature_i, scale_i, rotation_i, opacity_i,
                            c2w_iv, intr_iv, W, H,
                            sh_degree, near_plane, far_plane, bg_iv,
                        )
                    rendering[iv:iv + 1] = r1
                else:
                    r, _ = _render_single_view(
                        xyz_i, feature_i, scale_i, rotation_i, opacity_i,
                        test_c2ws[ib, iv], test_intr[ib, iv], W, H,
                        sh_degree, near_plane, far_plane, bg_colors[iv],
                    )
                    rendering[iv:iv + 1] = r
            renderings.append(rendering)
            continue

        # Legacy/LRM path: flatten (V, N_pts, D) → (V*N_pts, D), render union from each view.
        params_flat = gaussian_params[ib].float().reshape(-1, gaussian_params.shape[-1])
        xyz_i, feature_i, scale_i, rotation_i, opacity_i = _split_params_1d(params_flat)

        if use_checkpoint:
            if patch_size is not None:
                ph, pw = patch_size
                assert H % ph == 0 and W % pw == 0, (
                    f"patch_size ({ph},{pw}) must divide H={H}, W={W} exactly"
                )
                renderings.append(
                    GaussianRendererPatch.apply(
                        xyz_i, feature_i, scale_i, rotation_i, opacity_i,
                        test_c2ws[ib], test_intr[ib], W, H,
                        sh_degree, near_plane, far_plane, bg_colors, ph, pw,
                    )
                )
            else:
                renderings.append(
                    GaussianRendererWithCheckpoint.apply(
                        xyz_i, feature_i, scale_i, rotation_i, opacity_i,
                        test_c2ws[ib], test_intr[ib], W, H,
                        sh_degree, near_plane, far_plane, bg_colors,
                    )
                )
        else:
            rendering = torch.zeros(V, H, W, 4, device=device)
            for iv in range(V):
                r, _ = _render_single_view(
                    xyz_i, feature_i, scale_i, rotation_i, opacity_i,
                    test_c2ws[ib][iv], test_intr[ib][iv], W, H,
                    sh_degree, near_plane, far_plane, bg_colors[iv],
                )
                rendering[iv:iv + 1] = r
            renderings.append(rendering)

    # (B, V, H, W, 4) → (B, V, 4, H, W)
    renderings = torch.stack(renderings, dim=0).permute(0, 1, 4, 2, 3).contiguous()
    rgb = renderings[:, :, :3].mul_(2).add_(-1).clamp(-1, 1)
    depth = renderings[:, :, 3:]
    return rgb, depth
