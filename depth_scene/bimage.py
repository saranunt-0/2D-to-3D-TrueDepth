"""Blender Image <-> NumPy, and depth-map files."""

import os
import tempfile
import uuid

import bpy
import numpy as np

from .core import imaging

# (image name, token/size) -> array; avoids re-reading pixels on every slider move.
_cache = {}


def _key(img):
    return (img.name_full, tuple(img.size), img.get("depth_scene_token", ""), img.filepath)


def to_array(img):
    """(H, W, 4) float32, row 0 = top. Cached until the image changes."""
    key = _key(img)
    arr = _cache.get(key)
    if arr is not None:
        return arr
    w, h = img.size
    if w == 0 or h == 0:
        img.reload()
        w, h = img.size
        if w == 0:
            raise RuntimeError(f"Could not load image '{img.name}'")
    buf = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(h, w, img.channels)[::-1]
    if img.channels < 4:
        pad = np.ones((h, w, 4), np.float32)
        pad[..., : img.channels] = arr[..., :4]
        if img.channels == 1:
            pad[..., 1] = pad[..., 2] = arr[..., 0]
        arr = pad
    arr = np.ascontiguousarray(arr)
    for k in [k for k in _cache if k[0] == key[0]]:
        del _cache[k]
    _cache[key] = arr
    return arr


def display_rgb(img):
    """RGB in display-referred sRGB [0, 1] - what depth networks were trained on."""
    arr = to_array(img)[..., :3]
    # Float images (EXR/HDR) are scene-linear in Blender; 8/16-bit sRGB images
    # come back encoded as-is.
    if img.is_float and img.colorspace_settings.name not in ("sRGB",):
        arr = imaging.linear_to_srgb(arr)
    return np.clip(arr, 0.0, 1.0)


def has_alpha(img):
    """True only if the image really contains transparent pixels."""
    if img.channels < 4 or img.alpha_mode == "NONE":
        return False
    return bool(np.any(to_array(img)[..., 3] < 0.99))


def clear_cache():
    _cache.clear()


def output_dir(save_location):
    path = bpy.path.abspath(save_location) if save_location else ""
    if not path or (save_location.startswith("//") and not bpy.data.filepath):
        path = os.path.join(tempfile.gettempdir(), "depth_scene")
    os.makedirs(path, exist_ok=True)
    return path


def _write_exr(name, rgba, folder, non_color):
    """Save an (h, w, 4) float array as EXR next to the .blend and load it as an Image."""
    h, w = rgba.shape[:2]
    path = os.path.join(folder, bpy.path.clean_name(name) + ".exr")
    img = bpy.data.images.get(name)
    if img is not None and tuple(img.size) != (w, h):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    img.alpha_mode = "STRAIGHT"
    img.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).ravel())
    img.filepath_raw = path
    img.file_format = "OPEN_EXR"
    img.save()
    img["depth_scene_token"] = uuid.uuid4().hex
    img.update()
    _cache[_key(img)] = rgba
    return img


def write_depth(name, depth, valid, folder):
    """Save depth as a float EXR (R = depth, A = valid) and load it as an Image."""
    h, w = depth.shape
    rgba = np.zeros((h, w, 4), np.float32)
    finite = np.isfinite(depth)
    fill = float(depth[finite].max()) if finite.any() else 1.0
    d = np.where(finite, depth, fill)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = d
    rgba[..., 3] = 1.0 if valid is None else valid.astype(np.float32)
    return _write_exr(name, rgba, folder, non_color=True)


def write_map(name, arr, folder, non_color):
    """Save a material pass ((h, w) grey or (h, w, 3) colour, float) as EXR and load it.

    ``non_color`` for data (normal, roughness, metallic, shading); colour maps
    (albedo, residual) keep Blender's scene-linear default for float images.
    """
    arr = np.asarray(arr, np.float32)
    h, w = arr.shape[:2]
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = arr[..., None] if arr.ndim == 2 else arr[..., :3]
    return _write_exr(name, rgba, folder, non_color)


def depth_arrays(img):
    """(depth (h, w), valid (h, w) bool or None) from a depth Image."""
    arr = to_array(img)
    alpha = arr[..., 3]
    valid = None if np.all(alpha > 0.5) else alpha > 0.5
    # Greyscale maps have R = G = B; our EXRs store depth in R.
    return arr[..., 0], valid
