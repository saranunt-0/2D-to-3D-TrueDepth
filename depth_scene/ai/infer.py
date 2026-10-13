"""Inference entry points (Marigold only on this branch).

Input: an RGB float image (H, W, 3) in display-referred sRGB [0, 1], row 0 at
the top. ``estimate`` returns a ``DepthResult`` (depth map at the processing
resolution, same aspect as the image); ``estimate_passes`` returns material
maps (normal / albedo / roughness / metallic / shading / residual) from the
normals and intrinsics models, see ``core.passes`` for their conventions.
"""

import os
import time

from ..core import passes
from . import marigold, runtime


class DepthResult:
    def __init__(self, depth, kind, valid=None, seconds=0.0, provider=""):
        self.depth = depth
        self.kind = kind          # "depth": 0 = near, 1 = far (affine-invariant)
        self.valid = valid
        self.seconds = seconds
        self.provider = provider


class PassResult:
    def __init__(self, maps, info=None, seconds=0.0, provider=""):
        self.maps = maps          # pass name -> float32 array (h, w[, 3])
        self.info = info or {}
        self.seconds = seconds
        self.provider = provider


def _load(model_info, model_dir, device, task):
    missing = marigold.missing_files(model_dir)
    if missing:
        raise RuntimeError(f"Marigold model folder {model_dir} is missing {', '.join(missing)} - "
                           f"{model_info.manual}")
    cfg = marigold.load_config(model_dir)
    if marigold.task_of(cfg) != task:
        raise RuntimeError(f"{model_dir} holds a '{marigold.task_of(cfg)}' model, expected '{task}' - "
                           f"{model_info.manual}")
    sessions = runtime.get_sessions([os.path.join(model_dir, f) for f in marigold.FILES[:3]], device)
    return sessions, cfg


def estimate(model_info, model_dir, rgb, device="AUTO", processing_resolution=768, steps=0,
             ensemble_size=1, seed=0, progress=None):
    sessions, cfg = _load(model_info, model_dir, device, "depth")
    t0 = time.perf_counter()
    depth = marigold.predict(sessions, cfg, rgb, steps=steps, processing_resolution=processing_resolution,
                             ensemble_size=ensemble_size, seed=seed, progress=progress)
    return DepthResult(depth, "depth", seconds=time.perf_counter() - t0,
                       provider=sessions[1].get_providers()[0])


def estimate_passes(model_info, model_dir, rgb, device="AUTO", processing_resolution=768, steps=0,
                    ensemble_size=1, seed=0, progress=None):
    """Normals or intrinsics model -> ``PassResult`` with ``model_info.outputs`` maps."""
    sessions, cfg = _load(model_info, model_dir, device, model_info.task)
    t0 = time.perf_counter()
    out = marigold.predict(sessions, cfg, rgb, steps=steps, processing_resolution=processing_resolution,
                           ensemble_size=ensemble_size, seed=seed, progress=progress)
    info = {}
    if model_info.task == "normals":
        maps = {"normal": passes.normal_to_color(out)}
    else:
        maps, info = passes.intrinsics_to_maps(marigold.split_targets(cfg, out), rgb)
    missing = [k for k in model_info.outputs if k not in maps]
    if missing:
        raise RuntimeError(f"{model_dir} did not produce {', '.join(missing)} - is it the right checkpoint?")
    return PassResult({k: maps[k] for k in model_info.outputs}, info, seconds=time.perf_counter() - t0,
                      provider=sessions[1].get_providers()[0])
