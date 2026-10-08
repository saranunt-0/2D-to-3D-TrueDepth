"""Pure-NumPy image helpers.

Blender's bundled Python ships NumPy but not Pillow, OpenCV or SciPy, so every
resampling / filtering step the add-on needs is implemented here with NumPy
only. Arrays are float32, row 0 is the *top* of the image (NumPy/OpenCV
convention, not Blender's bottom-up pixel buffer).
"""

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _resample_matrix(n_in, n_out):
    """Return a dense (n_out, n_in) matrix doing antialiased linear resampling.

    Pixel-centre convention (like PIL/OpenCV ``align_corners=False``). When
    downsampling, the triangle filter is widened by the scale factor so the
    result is properly low-passed (no aliasing).
    """
    scale = n_in / n_out
    support = max(scale, 1.0)
    centers = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.arange(n_in, dtype=np.float64)
    dist = np.abs(src[None, :] - centers[:, None]) / support
    weights = np.clip(1.0 - dist, 0.0, None)
    sums = weights.sum(axis=1, keepdims=True)
    # Can only be zero if n_in == 0; guard anyway.
    sums[sums == 0] = 1.0
    return (weights / sums).astype(np.float32)


def resize(img, out_h, out_w):
    """Antialiased bilinear resize of an (H, W) or (H, W, C) float array."""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    out_h, out_w = int(out_h), int(out_w)
    if (h, w) == (out_h, out_w):
        return img.copy()
    ry = _resample_matrix(h, out_h)
    rx = _resample_matrix(w, out_w)
    if img.ndim == 2:
        return ry @ img @ rx.T
    out = np.tensordot(ry, img, axes=(1, 0))            # (out_h, w, C)
    out = np.tensordot(out, rx, axes=(1, 1))            # (out_h, C, out_w)
    return np.ascontiguousarray(out.transpose(0, 2, 1))


def _box_blur_axis(img, radius, axis):
    """Box blur of half-width ``radius`` along ``axis`` with edge clamping."""
    if radius <= 0:
        return img
    pad = [(0, 0)] * img.ndim
    pad[axis] = (radius + 1, radius)
    padded = np.pad(img, pad, mode="edge")
    csum = np.cumsum(padded, axis=axis, dtype=np.float64)
    n = img.shape[axis]
    hi = [slice(None)] * img.ndim
    lo = [slice(None)] * img.ndim
    hi[axis] = slice(2 * radius + 1, 2 * radius + 1 + n)
    lo[axis] = slice(0, n)
    return ((csum[tuple(hi)] - csum[tuple(lo)]) * (1.0 / (2 * radius + 1))).astype(np.float32)


def gaussian_blur(img, sigma):
    """Approximate Gaussian blur (three box passes, O(N) for any sigma)."""
    img = np.asarray(img, dtype=np.float32)
    if sigma <= 0.3:
        return img.copy()
    # Box widths for 3 passes approximating a Gaussian (Kovesi 2010).
    n = 3
    w_ideal = np.sqrt(12.0 * sigma * sigma / n + 1.0)
    wl = int(np.floor(w_ideal))
    if wl % 2 == 0:
        wl -= 1
    wu = wl + 2
    m = round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4))
    out = img
    for i in range(n):
        radius = ((wl if i < m else wu) - 1) // 2
        out = _box_blur_axis(out, radius, 0)
        out = _box_blur_axis(out, radius, 1)
    return out


def luminance(rgb):
    """Rec.709 luminance of an (H, W, 3+) array."""
    return np.asarray(rgb, dtype=np.float32)[..., :3] @ _LUMA


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def sample_bilinear(img, u, v):
    """Sample ``img`` at normalised coordinates.

    ``u`` runs 0 (left edge) .. 1 (right edge), ``v`` runs 0 (top edge) ..
    1 (bottom edge). Edge coordinates sample the outermost pixel centres, so a
    grid spanning [0, 1] covers the full image without running off it.
    """
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    x = np.clip(np.asarray(u, dtype=np.float32) * w - 0.5, 0.0, w - 1.0)
    y = np.clip(np.asarray(v, dtype=np.float32) * h - 0.5, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def sample_nearest(img, u, v):
    """Nearest-pixel sampling with the same coordinate convention as above."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    x = np.clip(np.floor(np.asarray(u) * w).astype(np.int64), 0, w - 1)
    y = np.clip(np.floor(np.asarray(v) * h).astype(np.int64), 0, h - 1)
    return img[y, x]
