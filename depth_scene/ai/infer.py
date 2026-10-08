"""Model-specific pre/post-processing.

Input: an RGB float image (H, W, 3) in display-referred sRGB [0, 1], row 0 at
the top. Output: ``DepthResult`` with a depth map (any resolution, same
aspect), its kind, an optional validity mask and an optional estimated FOV.
"""

import time

import numpy as np

from ..core import imaging, moge_post
from . import runtime

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthResult:
    def __init__(self, depth, kind, valid=None, fov=None, seconds=0.0, provider=""):
        self.depth = depth
        self.kind = kind          # "disparity" | "metric"
        self.valid = valid        # bool (h, w) or None
        self.fov = fov            # radians across the longer side, or None
        self.seconds = seconds
        self.provider = provider


def _multiple_of(x, m=14):
    return max(m, int(round(x / m)) * m)


def _input_dtype(sess, index=0):
    return np.float16 if "float16" in sess.get_inputs()[index].type else np.float32


def _run(sess, feeds):
    outputs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outputs))


def depth_anything_v2(sess, rgb, input_size=518):
    """Depth Anything V2: shorter side -> ``input_size`` (multiple of 14).

    Matches the official ``infer_image`` resize ('lower_bound', keep aspect).
    Works with the onnx-community export (pixel_values -> predicted_depth)
    and with other single-input exports by using positional names.
    """
    h, w = rgb.shape[:2]
    scale = input_size / min(h, w)
    th, tw = _multiple_of(h * scale), _multiple_of(w * scale)
    x = imaging.resize(rgb[..., :3], th, tw)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x.transpose(2, 0, 1)[None].astype(_input_dtype(sess))
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    disp = np.asarray(out, dtype=np.float32).reshape(out.shape[-2], out.shape[-1])
    disp = np.maximum(disp, 0.0)   # the head ends in ReLU; keep it explicit
    return DepthResult(disp, "disparity")


def moge2(sess, rgb, input_size=518):
    """MoGe-2: image in [0, 1] (normalised inside the graph) + token budget."""
    h, w = rgb.shape[:2]
    num_tokens = int(np.clip((input_size / 14.0) ** 2, 1200, 3600))
    # The exported graph resizes without antialiasing, so pre-resize to the
    # token grid ourselves (as MoGe's own ONNX notes recommend).
    aspect = w / h
    base_h = round((num_tokens / aspect) ** 0.5)
    base_w = round((num_tokens * aspect) ** 0.5)
    x = imaging.resize(rgb[..., :3], base_h * 14, base_w * 14)
    x = x.transpose(2, 0, 1)[None].astype(_input_dtype(sess))
    feeds = {}
    for inp in sess.get_inputs():
        if inp.name == "num_tokens" or "int" in inp.type:
            feeds[inp.name] = np.array(num_tokens, dtype=np.int64)
        else:
            feeds[inp.name] = x
    out = _run(sess, feeds)
    points = out["points"][0]
    mask = out.get("mask")
    mask = None if mask is None else mask[0]
    scale = out.get("metric_scale")
    depth, valid, fov = moge_post.postprocess(points, mask, scale)
    return DepthResult(depth, "metric", valid=valid, fov=fov)


FAMILIES = {"depth_anything_v2": depth_anything_v2, "moge2": moge2}


def estimate(model_info, model_path, rgb, device="AUTO", input_size=518):
    sess = runtime.get_session(model_path, device)
    t0 = time.perf_counter()
    result = FAMILIES[model_info.family](sess, rgb, input_size=input_size)
    result.seconds = time.perf_counter() - t0
    result.provider = sess.get_providers()[0]
    return result
