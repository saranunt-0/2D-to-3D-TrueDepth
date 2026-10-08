"""MoGe-2 post-processing in NumPy (no SciPy / PyTorch).

The official MoGe-2 ONNX export only contains the network's forward pass,
which returns an *affine-invariant* point map: the true camera-space points
are ``(x, y, z + shift) * metric_scale`` for an unknown ``shift``, and the
focal length is implicit. MoGe's ``infer()`` recovers focal and shift with
``recover_focal_shift`` (SciPy Levenberg-Marquardt). This module ports that
step: for a given shift the optimal focal has a closed form, so the problem
is 1-D and is solved here with a log-spaced scan plus golden-section refine.

Reference: microsoft/MoGe, moge/utils/geometry_numpy.py (MIT licence).
Conventions (OpenCV camera): x right, y down, z forward. ``uv`` is the image
plane normalised by half the image diagonal, so focal is "relative to half
the diagonal" exactly as in MoGe.
"""

import math

import numpy as np


def normalized_view_plane_uv(width, height):
    aspect = width / height
    span_x = aspect / (1 + aspect ** 2) ** 0.5
    span_y = 1 / (1 + aspect ** 2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=np.float32)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu, vv], axis=-1)


def _masked_nearest_downsample(points, uv, mask, size):
    """Nearest-neighbour downsample keeping only valid samples (flattened)."""
    h, w = points.shape[:2]
    sw, sh = size
    ys = np.clip(((np.arange(sh) + 0.5) * h / sh).astype(np.int64), 0, h - 1)
    xs = np.clip(((np.arange(sw) + 0.5) * w / sw).astype(np.int64), 0, w - 1)
    p = points[ys][:, xs].reshape(-1, 3)
    q = uv[ys][:, xs].reshape(-1, 2)
    if mask is not None:
        m = mask[ys][:, xs].reshape(-1)
        p, q = p[m], q[m]
    ok = np.all(np.isfinite(p), axis=1)
    return p[ok].astype(np.float64), q[ok].astype(np.float64)


def _residual(shift, uv, xy, z, focal=None):
    denom = z + shift
    xy_proj = xy / denom[:, None]
    if focal is None:
        focal = (xy_proj * uv).sum() / max(np.square(xy_proj).sum(), 1e-12)
    err = focal * xy_proj - uv
    return float(np.square(err).sum()), float(focal)


def _solve_shift(uv, xy, z, focal=None):
    z_min = float(z.min())
    span = max(float(z.max()) - z_min, 1e-6)
    # Search over w = z_min + shift > 0 (all points must stay in front).
    ws = span * np.logspace(-4, 3, 160)
    costs = [_residual(w - z_min, uv, xy, z, focal)[0] for w in ws]
    i = int(np.argmin(costs))
    lo = ws[max(i - 1, 0)]
    hi = ws[min(i + 1, len(ws) - 1)]
    # Golden-section refinement in log space.
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc = _residual(math.exp(c) - z_min, uv, xy, z, focal)[0]
    fd = _residual(math.exp(d) - z_min, uv, xy, z, focal)[0]
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _residual(math.exp(c) - z_min, uv, xy, z, focal)[0]
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _residual(math.exp(d) - z_min, uv, xy, z, focal)[0]
    shift = math.exp((a + b) / 2) - z_min
    _, f = _residual(shift, uv, xy, z, focal)
    return shift, f


def recover_focal_shift(points, mask=None, focal=None, downsample_size=(64, 64)):
    """Return (focal, shift) for an affine point map (H, W, 3)."""
    h, w = points.shape[:2]
    uv = normalized_view_plane_uv(w, h)
    p, q = _masked_nearest_downsample(points, uv, mask, downsample_size)
    if len(p) < 2:
        return 1.0, 0.0
    shift, f = _solve_shift(q, p[:, :2], p[:, 2], focal)
    return f, shift


def focal_to_fov(focal, width, height):
    """MoGe focal (relative to half diagonal) -> (fov_x, fov_y) in radians."""
    aspect = width / height
    fx = focal / 2 * (1 + aspect ** 2) ** 0.5 / aspect   # normalised by width
    fy = focal / 2 * (1 + aspect ** 2) ** 0.5            # normalised by height
    return 2 * math.atan(0.5 / fx), 2 * math.atan(0.5 / fy)


def postprocess(points, mask_prob, metric_scale=None, mask_threshold=0.5):
    """Raw MoGe outputs -> (depth (H, W) metres or relative, valid mask, fov_long).

    ``fov_long`` is the field of view across the longer image side, the
    convention used by Blender cameras (sensor fit AUTO).
    """
    points = np.asarray(points, dtype=np.float32)
    h, w = points.shape[:2]
    valid = None if mask_prob is None else (np.asarray(mask_prob) > mask_threshold)
    focal, shift = recover_focal_shift(points, valid)
    depth = points[..., 2] + shift
    ok = np.isfinite(depth) & (depth > 0)
    valid = ok if valid is None else (valid & ok)
    if metric_scale is not None:
        depth = depth * float(np.asarray(metric_scale).reshape(-1)[0])
    depth = np.where(valid, depth, np.inf).astype(np.float32)
    fov_x, fov_y = focal_to_fov(focal, w, h)
    return depth, valid, (fov_x if w >= h else fov_y)
