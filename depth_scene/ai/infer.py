"""Depth inference entry point (Marigold only on this branch).

Input: an RGB float image (H, W, 3) in display-referred sRGB [0, 1], row 0 at
the top. Output: ``DepthResult`` with a depth map at the processing
resolution (same aspect as the image) and its kind.
"""

import os
import time

from . import marigold, runtime


class DepthResult:
    def __init__(self, depth, kind, valid=None, seconds=0.0, provider=""):
        self.depth = depth
        self.kind = kind          # "depth": 0 = near, 1 = far (affine-invariant)
        self.valid = valid
        self.seconds = seconds
        self.provider = provider


def estimate(model_info, model_dir, rgb, device="AUTO", processing_resolution=768, steps=0,
             ensemble_size=1, seed=0, progress=None):
    missing = marigold.missing_files(model_dir)
    if missing:
        raise RuntimeError(f"Marigold model folder {model_dir} is missing {', '.join(missing)} - "
                           f"{model_info.manual}")
    sessions = runtime.get_sessions([os.path.join(model_dir, f) for f in marigold.FILES[:3]], device)
    cfg = marigold.load_config(model_dir)
    t0 = time.perf_counter()
    depth = marigold.predict(sessions, cfg, rgb, steps=steps, processing_resolution=processing_resolution,
                             ensemble_size=ensemble_size, seed=seed, progress=progress)
    return DepthResult(depth, "depth", seconds=time.perf_counter() - t0,
                       provider=sessions[1].get_providers()[0])
