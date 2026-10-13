"""Marigold normals / intrinsics -> Blender-ready material maps (NumPy only).

All maps keep the processing resolution of the prediction: the mesh samples
them through its 0..1 image UVs, so they need no resampling to the image size.

Conventions of the returned maps:

* ``normal``    [h,w,3] in [0,1]: camera-space unit normal n encoded as
                n * 0.5 + 0.5 (x right, y up, z towards the camera). The
                mesh's local frame *is* the projector camera's frame, so this
                is an Object-space normal map for Blender's Normal Map node.
* ``albedo``    [h,w,3] scene-linear (IID-Appearance predicts sRGB, converted;
                IID-Lighting predicts linear).
* ``roughness``, ``metallic``  [h,w] in [0,1] (linear data).
* ``shading``, ``residual``    [h,w,3] scene-linear, calibrated to the input
                image so that  image ~= albedo * shading + residual  (the
                Hypersim decomposition IID-Lighting is trained on; the model
                only predicts them up to scale).
"""

import numpy as np

from . import imaging

RENAME = {"metallicity": "metallic"}


def normal_to_color(n):
    return (np.asarray(n, np.float32) * 0.5 + 0.5).astype(np.float32)


def calibrate_lighting(image_lin, albedo, shading, residual):
    """Least-squares a, b >= 0 with image ~= a * albedo * shading + b * residual.

    Returns (a * shading, b * residual, info) where info has the coefficients
    and the relative RMS error of the fit.
    """
    i = np.asarray(image_lin, np.float64).reshape(-1)
    x1 = (np.asarray(albedo, np.float64) * shading).reshape(-1)
    x2 = np.asarray(residual, np.float64).reshape(-1)
    a = b = 0.0
    m = np.stack([x1, x2], axis=1)
    coef, *_ = np.linalg.lstsq(m, i, rcond=None)
    if coef[0] >= 0 and coef[1] >= 0:
        a, b = coef
    else:                       # constrained: refit with the column that is allowed to stay
        a1 = max(float(x1 @ i / max(x1 @ x1, 1e-12)), 0.0)
        b1 = max(float(x2 @ i / max(x2 @ x2, 1e-12)), 0.0)
        e1 = np.sum((i - a1 * x1) ** 2)
        e2 = np.sum((i - b1 * x2) ** 2)
        a, b = (a1, 0.0) if e1 <= e2 else (0.0, b1)
    err = np.sqrt(np.mean((i - a * x1 - b * x2) ** 2)) / max(np.sqrt(np.mean(i ** 2)), 1e-12)
    info = {"shading_scale": float(a), "residual_scale": float(b), "fit_error": float(err)}
    return (shading * a).astype(np.float32), (residual * b).astype(np.float32), info


def intrinsics_to_maps(targets, image_srgb=None):
    """``marigold.split_targets`` output -> {pass name: array}, plus info.

    ``image_srgb`` (any size, display sRGB [0,1]) is needed to calibrate
    shading / residual; without it they are only normalised (99th percentile
    of the luminance -> 1).
    """
    maps, info = {}, {}
    for name, (arr, space, _up_to_scale) in targets.items():
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 3 and space == "srgb":
            arr = imaging.srgb_to_linear(arr)
        elif arr.ndim == 2:
            arr = np.clip(arr, 0.0, 1.0)
        maps[RENAME.get(name, name)] = arr
    if "shading" in maps and "residual" in maps and "albedo" in maps:
        if image_srgb is not None:
            h, w = maps["albedo"].shape[:2]
            img = imaging.srgb_to_linear(imaging.resize(np.asarray(image_srgb, np.float32)[..., :3], h, w))
            maps["shading"], maps["residual"], info = calibrate_lighting(
                img, maps["albedo"], maps["shading"], maps["residual"])
        else:
            for key in ("shading", "residual"):
                ref = float(np.percentile(imaging.luminance(maps[key]), 99))
                maps[key] = maps[key] / max(ref, 1e-6)
    return maps, info
