"""
Visualization utilities for holo4d conditioning pipeline.

Provides:
  - Target-view track overlay on RGB frames
  - Block occupancy heatmap
  - Source vs target comparison panels
  - Training progress composites
"""

import os
import numpy as np
import torch
from typing import Optional


def _to_numpy_img(tensor_or_np):
    """Convert image tensor (C,H,W) or (H,W,C) float [-1,1] or [0,1] to uint8 (H,W,3)."""
    if isinstance(tensor_or_np, torch.Tensor):
        x = tensor_or_np.detach().cpu().float().numpy()
    else:
        x = np.array(tensor_or_np, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.transpose(1, 2, 0)  # CHW → HWC
    if x.min() < 0:
        x = (x + 1) / 2  # [-1,1] → [0,1]
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x


def _track_colors(N: int) -> np.ndarray:
    """Generate N distinct colors using HSV spacing. Returns (N, 3) uint8 RGB."""
    import colorsys
    colors = []
    for i in range(N):
        h = i / max(N, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.9)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.uint8)


def draw_tracks_on_frame(
    frame: np.ndarray,
    track_uv: np.ndarray,
    track_vis: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radius: int = 2,
) -> np.ndarray:
    """Draw track positions on an RGB frame.

    Args:
        frame: (H, W, 3) uint8 RGB image.
        track_uv: (N, 2) pixel coordinates (u=col, v=row) for this frame.
        track_vis: (N,) bool visibility.
        colors: (N, 3) uint8 colors per track. If None, auto-generated.
        radius: dot radius in pixels.

    Returns:
        frame_out: (H, W, 3) uint8 with tracks drawn.
    """
    frame_out = frame.copy()
    H, W = frame_out.shape[:2]
    N = track_uv.shape[0]
    if colors is None:
        colors = _track_colors(N)

    for i in range(N):
        if not track_vis[i]:
            continue
        u, v = int(round(track_uv[i, 0])), int(round(track_uv[i, 1]))
        if 0 <= u < W and 0 <= v < H:
            r = radius
            v_lo, v_hi = max(0, v - r), min(H, v + r + 1)
            u_lo, u_hi = max(0, u - r), min(W, u + r + 1)
            frame_out[v_lo:v_hi, u_lo:u_hi] = colors[i]

    return frame_out


def visualize_track_overlay(
    frames: torch.Tensor,
    track_uv: torch.Tensor,
    track_vis: torch.Tensor,
    output_dir: str,
    prefix: str = "track_overlay",
    max_frames: int = 8,
    max_tracks: int = 256,
):
    """Save track overlay images for a sequence of frames.

    Args:
        frames: (T, C, H, W) or (T, H, W, C) float images.
        track_uv: (N, T, 2) pixel coords.
        track_vis: (N, T) bool visibility.
        output_dir: directory to save images.
        prefix: filename prefix.
        max_frames: max frames to visualize.
        max_tracks: max tracks to draw.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(frames, torch.Tensor):
        frames_np = frames.detach().cpu()
    else:
        frames_np = torch.from_numpy(frames)

    T = frames_np.shape[0]
    uv_np = track_uv[:max_tracks].detach().cpu().numpy() if isinstance(track_uv, torch.Tensor) else track_uv[:max_tracks]
    vis_np = track_vis[:max_tracks].detach().cpu().numpy() if isinstance(track_vis, torch.Tensor) else track_vis[:max_tracks]
    N = uv_np.shape[0]
    colors = _track_colors(N)

    frame_indices = np.linspace(0, T - 1, min(max_frames, T), dtype=int)

    try:
        from PIL import Image
    except ImportError:
        print("[vis] PIL not available, skipping track overlay")
        return

    for t in frame_indices:
        frame_img = _to_numpy_img(frames_np[t])
        frame_with_tracks = draw_tracks_on_frame(frame_img, uv_np[:, t], vis_np[:, t], colors)
        img = Image.fromarray(frame_with_tracks)
        img.save(os.path.join(output_dir, f"{prefix}_t{t:03d}.png"))


def visualize_block_occupancy(
    block_mask: torch.Tensor,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    block_t: int,
    block_h: int,
    block_w: int,
    output_dir: str,
    prefix: str = "block_occupancy",
):
    """Save block occupancy heatmap (spatial, summed over temporal blocks).

    Args:
        block_mask: (num_blocks,) bool mask.
        latent_t, latent_h, latent_w: latent grid dims.
        block_t, block_h, block_w: block sizes.
        output_dir: save directory.
        prefix: filename prefix.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_bt = (latent_t + block_t - 1) // block_t
    num_bh = (latent_h + block_h - 1) // block_h
    num_bw = (latent_w + block_w - 1) // block_w

    mask_np = block_mask.detach().cpu().numpy().astype(np.float32)
    # Reshape to (num_bt, num_bh, num_bw)
    grid = mask_np.reshape(num_bt, num_bh, num_bw)
    # Sum over temporal dimension → (num_bh, num_bw)
    spatial_occ = grid.sum(axis=0)

    # Normalize to [0, 255]
    if spatial_occ.max() > 0:
        spatial_occ = spatial_occ / spatial_occ.max() * 255
    spatial_occ = spatial_occ.astype(np.uint8)

    # Upscale for visibility
    scale = max(1, 256 // max(num_bh, num_bw))
    heatmap = np.repeat(np.repeat(spatial_occ, scale, axis=0), scale, axis=1)

    try:
        from PIL import Image
        img = Image.fromarray(heatmap, mode="L")
        img.save(os.path.join(output_dir, f"{prefix}.png"))
    except ImportError:
        np.save(os.path.join(output_dir, f"{prefix}.npy"), spatial_occ)


def visualize_source_target_comparison(
    source_frames: torch.Tensor,
    target_frames: torch.Tensor,
    predicted_frames: Optional[torch.Tensor],
    track_uv: torch.Tensor,
    track_vis: torch.Tensor,
    output_dir: str,
    step: int = 0,
    max_frames: int = 4,
    max_tracks: int = 256,
):
    """Save side-by-side: source | target GT | predicted (if available) | target+tracks.

    Args:
        source_frames: (T, C, H, W) source view.
        target_frames: (T, C, H, W) target GT view.
        predicted_frames: (T, C, H, W) model prediction, or None.
        track_uv: (N, T, 2) tracks in target view.
        track_vis: (N, T) bool.
        output_dir: save dir.
        step: training step number.
        max_frames: frames to visualize.
        max_tracks: max tracks to draw.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        print("[vis] PIL not available, skipping comparison")
        return

    T = source_frames.shape[0]
    frame_indices = np.linspace(0, T - 1, min(max_frames, T), dtype=int)

    uv_np = track_uv[:max_tracks].detach().cpu().numpy() if isinstance(track_uv, torch.Tensor) else track_uv[:max_tracks]
    vis_np = track_vis[:max_tracks].detach().cpu().numpy() if isinstance(track_vis, torch.Tensor) else track_vis[:max_tracks]
    N = uv_np.shape[0]
    colors = _track_colors(N)

    for t in frame_indices:
        src_img = _to_numpy_img(source_frames[t])
        tgt_img = _to_numpy_img(target_frames[t])
        tgt_with_tracks = draw_tracks_on_frame(tgt_img, uv_np[:, t], vis_np[:, t], colors)

        panels = [src_img, tgt_img]
        if predicted_frames is not None:
            pred_img = _to_numpy_img(predicted_frames[t])
            panels.append(pred_img)
        panels.append(tgt_with_tracks)

        # Resize all panels to same height
        H = min(p.shape[0] for p in panels)
        W = min(p.shape[1] for p in panels)
        panels_resized = [p[:H, :W] for p in panels]

        composite = np.concatenate(panels_resized, axis=1)
        img = Image.fromarray(composite)
        img.save(os.path.join(output_dir, f"step{step:06d}_t{t:03d}.png"))


def save_loss_curve(losses: list, output_dir: str, filename: str = "loss_curve.png"):
    """Save training loss curve."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(losses, linewidth=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)
    except ImportError:
        # Save raw data if matplotlib not available
        np.save(os.path.join(output_dir, "losses.npy"), np.array(losses))
