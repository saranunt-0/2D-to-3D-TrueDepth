"""Depth-map interpretation: turning model output into distances.

Depth models output one of three kinds of map:

``disparity``
    Relative *inverse* depth (bigger = nearer), known only up to an unknown
    scale and shift. Depth Anything V2, MiDaS, and "white = near" depth maps.
``depth``
    Relative depth (bigger = farther), again up to scale and shift.
    "White = far" depth maps exported from other tools.
``metric``
    Absolute planar depth in metres (MoGe-2, metric models).
``layers``
    Illustrator's depth: a per-pixel *layer index* (bigger = drawn on top),
    i.e. an ordering, not a distance. Equal index steps are equal depth
    steps, so the natural mapping is LINEAR.

Everything is first converted to a normalised *nearness* ``t`` in [0, 1]
(1 = nearest pixel, 0 = farthest), then mapped to distances with one of the
TrueDepth-style mappings:

``INVERSE``  1/z = t/near + (1 - t)/far   (correct for disparity input)
``LINEAR``   z   = t*near + (1 - t)*far   (correct for relative depth input)
``METRIC``   z   = metric_depth * scale   (only for metric input)

For relative maps the near/far ratio stands in for the unknown shift, so it
also decides how straight walls and floors come out.
"""

import numpy as np

from . import imaging

KINDS = ("disparity", "depth", "metric", "layers")
MAPPINGS = ("INVERSE", "LINEAR", "METRIC")


def _finite_range(values, mask=None):
    sel = np.isfinite(values)
    if mask is not None:
        sel &= mask
    if not np.any(sel):
        return 0.0, 1.0
    vals = values[sel]
    return float(vals.min()), float(vals.max())


def normalize(values, mask=None):
    """Min/max normalise finite values to [0, 1]; non-finite become 0."""
    values = np.asarray(values, dtype=np.float32)
    lo, hi = _finite_range(values, mask)
    span = hi - lo if hi > lo else 1.0
    out = (values - lo) / span
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def nearness(depth_map, kind, mask=None):
    """Convert a raw depth map to normalised nearness (1 = near, 0 = far)."""
    depth_map = np.asarray(depth_map, dtype=np.float32)
    if kind in ("disparity", "layers"):
        return normalize(depth_map, mask)
    if kind == "depth":
        return 1.0 - normalize(depth_map, mask)
    if kind == "metric":
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = np.isfinite(depth_map) & (depth_map > 0)
            inv = np.where(ok, 1.0 / depth_map, np.nan).astype(np.float32)
        return normalize(inv, mask)
    raise ValueError(f"unknown depth kind {kind!r}")


def default_mapping(kind):
    return {"disparity": "INVERSE", "depth": "LINEAR", "metric": "METRIC", "layers": "LINEAR"}[kind]


def quantize(t, steps):
    """Snap nearness to ``steps`` evenly spaced levels (0/1 = off).

    Gives the flat, stacked "paper cut-out / diorama" look; with layer-index
    maps each level is one illustration layer.
    """
    steps = int(steps)
    if steps < 2:
        return t
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return (np.round(t * (steps - 1)) / (steps - 1)).astype(np.float32)


def robust_clip(values, low_pct=0.5, high_pct=99.5):
    """Clip outliers to the given percentiles of the finite values."""
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    return np.clip(values, lo, hi).astype(np.float32)


def map_to_distance(t, mapping, near, far):
    """Map nearness ``t`` to planar distance using INVERSE or LINEAR mapping."""
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    near = max(float(near), 1e-4)
    far = max(float(far), near + 1e-4)
    if mapping == "INVERSE":
        return (1.0 / (t / near + (1.0 - t) / far)).astype(np.float32)
    if mapping == "LINEAR":
        return (t * near + (1.0 - t) * far).astype(np.float32)
    raise ValueError(f"mapping {mapping!r} needs metric input")


def metric_distance(depth_map, scale, mask=None):
    """Scale a metric depth map; invalid pixels are pushed to the far value."""
    depth_map = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth_map) & (depth_map > 0)
    if mask is not None:
        valid &= mask
    out = depth_map.copy()
    fill = float(depth_map[valid].max()) if np.any(valid) else 1.0
    out[~valid] = fill
    return (out * float(scale)).astype(np.float32)


def detail_signal(gray, radius_px):
    """High-pass luminance, centred on 0, roughly in [-0.5, 0.5].

    This is the 'detail displacement' source: the depth model cannot resolve
    brush strokes, bark or carvings, but image luminance can.
    """
    gray = np.asarray(gray, dtype=np.float32)
    if radius_px <= 0:
        return np.zeros_like(gray)
    return gray - imaging.gaussian_blur(gray, radius_px)
