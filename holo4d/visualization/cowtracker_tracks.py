"""Shared CoWTracker-style 2D trajectory rendering helpers.

The visual convention is extracted from
``bash_scripts/holo4d/droid_io/inspect_droid_scene.py``: spatial Bremm
identity colors, a darkened RGB background, visible trajectory segments, and a
white-outlined current point.  This module accepts already-projected UV tracks
so it can be reused by synthetic and real-data visualizers.
"""

from __future__ import annotations

import cv2
import numpy as np


def bremm_rgb(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return CoWTracker-style spatial identity colors in RGB order."""
    xy = np.asarray(xy, dtype=np.float32)
    nx = np.clip(xy[:, 0] / max(width - 1, 1), 0.0, 1.0)
    ny = np.clip(xy[:, 1] / max(height - 1, 1), 0.0, 1.0)
    hue = (nx * 0.8 + ny * 0.2) % 1.0
    saturation = 0.6 + 0.4 * (1.0 - ny)
    value = 0.7 + 0.3 * nx
    hsv = np.stack(
        [
            (hue * 179).astype(np.uint8),
            (saturation * 255).astype(np.uint8),
            (value * 255).astype(np.uint8),
        ],
        axis=1,
    ).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return bgr[:, ::-1].copy()


def uniform_visible_selection(visible: np.ndarray, max_tracks: int) -> np.ndarray:
    """Deterministically thin all ever-visible identities in grid order."""
    visible = np.asarray(visible, dtype=bool)
    candidates = np.flatnonzero(visible.any(axis=1))
    if candidates.size <= max_tracks:
        return candidates
    positions = np.linspace(0, candidates.size - 1, max_tracks).astype(np.int64)
    return candidates[positions]


def _spatial_fps_indices(points: np.ndarray, target_count: int) -> np.ndarray:
    count = min(max(int(target_count), 0), points.shape[0])
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    centroid = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    selected = np.empty((count,), dtype=np.int64)
    selected[0] = first
    nearest = np.sum((points - points[first]) ** 2, axis=1)
    nearest[first] = -1.0
    for output_index in range(1, count):
        next_index = int(np.argmax(nearest))
        selected[output_index] = next_index
        distance = np.sum((points - points[next_index]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
        nearest[selected[: output_index + 1]] = -1.0
    return selected


def query_balanced_spatial_selection(
    uv: np.ndarray,
    visible: np.ndarray,
    query_ids: np.ndarray,
    max_tracks: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Select a query-balanced, spatially uniform set of displayed tracks.

    Spatial FPS is performed independently for each query using its own query-
    frame projection, with the median visible projection as a fallback. This
    removes flattened-grid selection artifacts without changing analysis data.
    """
    uv = np.asarray(uv, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    query = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if uv.shape[:2] != visible.shape:
        raise ValueError(f"uv/visible shape mismatch: {uv.shape} vs {visible.shape}")
    if query.shape != (uv.shape[0],):
        raise ValueError(f"query_ids must be {(uv.shape[0],)}, got {query.shape}")
    candidates = np.flatnonzero(visible.any(axis=1))
    count = min(max(int(max_tracks), 0), candidates.size)
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    groups = np.unique(query[candidates])
    base, remainder = divmod(count, max(groups.size, 1))
    chosen = []
    unused = []
    for group_index, query_id in enumerate(groups):
        members = candidates[query[candidates] == query_id]
        anchors = np.full((members.size, 2), np.nan, dtype=np.float32)
        query_frame = int(np.clip(query_id, 0, uv.shape[1] - 1))
        query_visible = visible[members, query_frame]
        anchors[query_visible] = uv[members[query_visible], query_frame]
        for local_index in np.flatnonzero(~query_visible):
            identity = members[local_index]
            identity_uv = uv[identity, visible[identity]]
            if identity_uv.size:
                anchors[local_index] = np.median(identity_uv, axis=0)
        finite = np.isfinite(anchors).all(axis=1)
        eligible_members = members[finite]
        eligible_anchors = anchors[finite]
        normalized = eligible_anchors / np.asarray(
            [max(width - 1, 1), max(height - 1, 1)], dtype=np.float32
        )
        budget = min(eligible_members.size, base + int(group_index < remainder))
        local_selected = _spatial_fps_indices(normalized, budget)
        selected_mask = np.zeros((eligible_members.size,), dtype=bool)
        selected_mask[local_selected] = True
        chosen.extend(eligible_members[local_selected].tolist())
        unused.extend(eligible_members[~selected_mask].tolist())
    if len(chosen) < count and unused:
        unused_array = np.asarray(unused, dtype=np.int64)
        fallback_uv = np.nanmedian(
            np.where(visible[unused_array, :, None], uv[unused_array], np.nan), axis=1
        )
        finite = np.isfinite(fallback_uv).all(axis=1)
        normalized = fallback_uv[finite] / np.asarray(
            [max(width - 1, 1), max(height - 1, 1)], dtype=np.float32
        )
        local_selected = _spatial_fps_indices(normalized, count - len(chosen))
        chosen.extend(unused_array[finite][local_selected].tolist())
    return np.sort(np.asarray(chosen[:count], dtype=np.int64))


def candidate_spatial_selection(
    uv: np.ndarray,
    visible: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    max_tracks: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Spatially thin only the requested candidate identities for display."""
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    candidate_visible = np.asarray(visible, dtype=bool).copy()
    candidate_mask = np.zeros((candidate_visible.shape[0],), dtype=bool)
    candidate_mask[candidate_ids] = True
    candidate_visible[~candidate_mask] = False
    return query_balanced_spatial_selection(
        uv,
        candidate_visible,
        query_ids,
        min(max(int(max_tracks), 0), candidate_ids.size),
        width,
        height,
    )


def query_anchor_colors(
    uv: np.ndarray,
    query_frames: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Color each identity by its projected position at its own query frame."""
    uv = np.asarray(uv, dtype=np.float32)
    query_frames = np.asarray(query_frames, dtype=np.int64)
    anchors = np.full((uv.shape[0], 2), np.nan, dtype=np.float32)
    for identity_idx in range(uv.shape[0]):
        query_frame = int(np.clip(query_frames[identity_idx], 0, uv.shape[1] - 1))
        if np.isfinite(uv[identity_idx, query_frame]).all():
            anchors[identity_idx] = uv[identity_idx, query_frame]
            continue
        finite_times = np.flatnonzero(np.isfinite(uv[identity_idx]).all(axis=1))
        if finite_times.size:
            anchors[identity_idx] = uv[identity_idx, finite_times[0]]
    anchors = np.nan_to_num(anchors, nan=0.0, posinf=0.0, neginf=0.0)
    return bremm_rgb(anchors, width, height)


def _trail_start_time(time_idx: int, trail_length: int) -> int:
    """Start index for a trail containing at most ``trail_length`` frames."""
    frame_count = max(int(trail_length), 1)
    return max(0, int(time_idx) - frame_count + 1)


def _magma_rgb_lut() -> np.ndarray:
    """Return OpenCV's perceptually uniform magma table in RGB order."""
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_MAGMA).reshape(256, 3)
    return bgr[:, ::-1].copy()


def _magma_color(lut: np.ndarray, value: float) -> tuple[int, int, int]:
    """Look up one RGB magma color from a normalized scalar."""
    index = int(round(np.clip(value, 0.0, 1.0) * 255.0))
    return tuple(int(channel) for channel in lut[index])


def _normalized_projected_arclength(
    uv: np.ndarray,
    visible: np.ndarray,
) -> np.ndarray:
    """Normalize each identity's cumulative visible image-space arc length."""
    finite = np.isfinite(uv).all(axis=2)
    segment_valid = visible[:, :-1] & visible[:, 1:] & finite[:, :-1] & finite[:, 1:]
    segment_length = np.linalg.norm(np.diff(uv, axis=1), axis=2)
    segment_length = np.where(segment_valid, segment_length, 0.0)
    cumulative = np.concatenate(
        [np.zeros((uv.shape[0], 1), dtype=np.float32), np.cumsum(segment_length, axis=1)],
        axis=1,
    )
    return cumulative / np.maximum(cumulative[:, -1:], 1e-6)


def render_cowtracker_tracks(
    frames_rgb: np.ndarray,
    uv: np.ndarray,
    visible: np.ndarray,
    selection: np.ndarray,
    colors_rgb: np.ndarray,
    trail_selection: np.ndarray | None = None,
    trail_length: int = 14,
    background_scale: float | np.ndarray = 0.4,
    overlay_alpha: float | np.ndarray = 1.0,
    line_thickness: int = 1,
    point_radius: int = 2,
    oldest_trail_intensity: float = 0.22,
    newest_trail_intensity: float = 0.82,
    trail_colormap: str = "identity",
    gradient_floor: float = 0.15,
    gradient_top: float = 0.88,
    white_outline: bool = True,
    endpoint_colors_rgb: np.ndarray | None = None,
    endpoint_outline_colors_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Render dark-background CoWTracker trails from projected UV tracks.

    The identity mode age-fades stable Bremm identity colors for analysis.
    The magma_arclength mode uses the MolmoMotion visual convention:
    old-to-new color is parameterized by each trajectory's total projected
    arc length, and the current endpoint is bright magma with a white locator
    ring. The latter is intended for presentation assets where motion
    direction matters more than assigning a separate hue to every ID.
    """
    frames_rgb = np.asarray(frames_rgb, dtype=np.uint8)
    uv = np.asarray(uv, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    selection = np.asarray(selection, dtype=np.int64)
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)
    endpoint_colors = (
        None
        if endpoint_colors_rgb is None
        else np.asarray(endpoint_colors_rgb, dtype=np.uint8)
    )
    endpoint_outline_colors = (
        None
        if endpoint_outline_colors_rgb is None
        else np.asarray(endpoint_outline_colors_rgb, dtype=np.uint8)
    )
    if uv.shape[:2] != visible.shape:
        raise ValueError(f"uv/visible shape mismatch: {uv.shape} vs {visible.shape}")
    if frames_rgb.shape[0] != uv.shape[1]:
        raise ValueError(f"frame/track time mismatch: {frames_rgb.shape[0]} vs {uv.shape[1]}")
    for name, endpoint_palette in (
        ("endpoint_colors_rgb", endpoint_colors),
        ("endpoint_outline_colors_rgb", endpoint_outline_colors),
    ):
        if endpoint_palette is not None and endpoint_palette.shape != colors_rgb.shape:
            raise ValueError(
                f"{name} shape mismatch: {endpoint_palette.shape} vs {colors_rgb.shape}"
            )
    if trail_colormap not in {"identity", "magma_arclength"}:
        raise ValueError(
            "trail_colormap must be 'identity' or 'magma_arclength', got "
            f"{trail_colormap!r}"
        )

    magma = _magma_rgb_lut() if trail_colormap == "magma_arclength" else None
    arc_fraction = None
    arc_row = None
    if trail_colormap == "magma_arclength":
        arc_ids = np.unique(selection)
        arc_fraction = _normalized_projected_arclength(
            uv[arc_ids], visible[arc_ids]
        )
        arc_row = np.full((uv.shape[0],), -1, dtype=np.int64)
        arc_row[arc_ids] = np.arange(arc_ids.size, dtype=np.int64)
    trail_mask = np.zeros((uv.shape[0],), dtype=bool)
    if trail_selection is None:
        trail_mask[selection] = True
    else:
        trail_indices = np.asarray(trail_selection, dtype=np.int64).reshape(-1)
        trail_mask[trail_indices] = True

    background_scales = np.broadcast_to(
        np.asarray(background_scale, dtype=np.float32), (frames_rgb.shape[0],)
    )
    overlay_alphas = np.broadcast_to(
        np.asarray(overlay_alpha, dtype=np.float32), (frames_rgb.shape[0],)
    )
    if not np.isfinite(background_scales).all() or np.any(background_scales < 0.0):
        raise ValueError("background_scale must be finite and non-negative")
    if not np.isfinite(overlay_alphas).all() or np.any(
        (overlay_alphas < 0.0) | (overlay_alphas > 1.0)
    ):
        raise ValueError("overlay_alpha must be finite and in [0, 1]")

    rendered = []
    for time_idx, frame in enumerate(frames_rgb):
        base = np.clip(
            frame.astype(np.float32) * background_scales[time_idx], 0, 255
        ).astype(np.uint8)
        canvas = base.copy()
        first_time = _trail_start_time(time_idx, trail_length)
        for identity_idx in selection:
            color = tuple(int(channel) for channel in colors_rgb[identity_idx])
            trail_times = (
                range(first_time + 1, time_idx + 1)
                if trail_mask[identity_idx]
                else ()
            )
            for trail_time in trail_times:
                previous_time = trail_time - 1
                if not (visible[identity_idx, previous_time] and visible[identity_idx, trail_time]):
                    continue
                if not (
                    np.isfinite(uv[identity_idx, previous_time]).all()
                    and np.isfinite(uv[identity_idx, trail_time]).all()
                ):
                    continue
                if trail_colormap == "magma_arclength":
                    identity_arc = arc_fraction[arc_row[identity_idx]]
                    segment_fraction = 0.5 * (
                        identity_arc[previous_time]
                        + identity_arc[trail_time]
                    )
                    color_value = gradient_floor + segment_fraction * (
                        gradient_top - gradient_floor
                    )
                    faded_color = _magma_color(magma, float(color_value))
                else:
                    age_fraction = (trail_time - first_time) / max(time_idx - first_time, 1)
                    intensity = oldest_trail_intensity + age_fraction * (
                        newest_trail_intensity - oldest_trail_intensity
                    )
                    faded_color = tuple(int(round(channel * intensity)) for channel in color)
                point0 = tuple(np.rint(uv[identity_idx, previous_time]).astype(int))
                point1 = tuple(np.rint(uv[identity_idx, trail_time]).astype(int))
                cv2.line(canvas, point0, point1, faded_color, line_thickness, cv2.LINE_AA)
            if visible[identity_idx, time_idx] and np.isfinite(uv[identity_idx, time_idx]).all():
                point = tuple(np.rint(uv[identity_idx, time_idx]).astype(int))
                default_tip_color = (
                    _magma_color(magma, 0.95)
                    if trail_colormap == "magma_arclength"
                    else color
                )
                tip_color = (
                    tuple(int(channel) for channel in endpoint_colors[identity_idx])
                    if endpoint_colors is not None
                    else default_tip_color
                )
                outline_color = None
                if endpoint_outline_colors is not None:
                    outline_color = tuple(
                        int(channel)
                        for channel in endpoint_outline_colors[identity_idx]
                    )
                elif white_outline:
                    outline_color = (255, 255, 255)
                if outline_color is not None:
                    cv2.circle(canvas, point, point_radius, outline_color, 1, cv2.LINE_AA)
                    cv2.circle(
                        canvas,
                        point,
                        max(point_radius - 1, 1),
                        tip_color,
                        -1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.circle(canvas, point, point_radius, tip_color, -1, cv2.LINE_AA)
        alpha = float(overlay_alphas[time_idx])
        if alpha < 1.0:
            canvas = np.clip(
                base.astype(np.float32) * (1.0 - alpha)
                + canvas.astype(np.float32) * alpha,
                0.0,
                255.0,
            ).astype(np.uint8)
        rendered.append(canvas)
    return np.stack(rendered, axis=0)
