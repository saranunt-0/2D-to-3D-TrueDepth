"""Marigold depth (prs-eth/marigold-depth-v1-1) with ONNX Runtime + NumPy.

Marigold is Stable Diffusion 2 fine-tuned to *generate* a depth map
conditioned on the image. ``tools/export_marigold_onnx.py`` splits the
diffusers pipeline into three ONNX graphs plus a JSON config; this module
re-implements ``MarigoldDepthPipeline.__call__`` around them:

    vae_encoder.onnx   image [B,3,H,W] in [-1,1]   -> latent [B,4,H/8,W/8]
                       (posterior mode x VAE scaling factor)
    unet.onnx          sample [B,8,h,w] (image latent ++ depth latent),
                       timestep int64 [1]            -> noise [B,4,h,w]
                       (the empty-prompt text embedding is baked in)
    vae_decoder.onnx   latent [B,4,h,w]             -> depth [B,1,H,W] in [0,1]
                       (decode, channel mean, clip, (x+1)/2)
    marigold.json      scheduler numbers, defaults, licence

Output is affine-invariant *depth* (0 = near, 1 = far), i.e. the add-on's
``depth`` kind with LINEAR mapping.
"""

import json
import os

import numpy as np

from ..core import imaging
from ..core.ddim import DDIM

FILES = ("vae_encoder.onnx", "unet.onnx", "vae_decoder.onnx", "marigold.json")


def missing_files(model_dir):
    return [f for f in FILES if not os.path.exists(os.path.join(model_dir, f))]


def load_config(model_dir):
    with open(os.path.join(model_dir, "marigold.json"), encoding="utf-8") as fh:
        return json.load(fh)


def preprocess(rgb, processing_resolution):
    """[0,1] RGB (H,W,3) -> padded [-1,1] NCHW, padding, processed size.

    Mirrors MarigoldImageProcessor.preprocess: normalise, resize so the
    longer edge is ``processing_resolution`` (integer floor, antialiased),
    replicate-pad bottom/right to a multiple of 8.
    """
    x = np.asarray(rgb, dtype=np.float32)[..., :3] * 2.0 - 1.0
    h, w = x.shape[:2]
    if processing_resolution and processing_resolution > 0:
        long_side = max(h, w)
        nh = h * int(processing_resolution) // long_side
        nw = w * int(processing_resolution) // long_side
        if nh == 0 or nw == 0:
            raise ValueError(f"extreme aspect ratio {w}x{h}")
        x = imaging.resize(x, nh, nw)
    ph, pw = -x.shape[0] % 8, -x.shape[1] % 8
    x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return x.transpose(2, 0, 1)[None].astype(np.float32), (ph, pw)


def ensemble(preds):
    """Combine ensemble members (each [H,W] in [0,1]).

    diffusers aligns members with a SciPy scale/shift optimisation; Blender
    has no SciPy, so each member is least-squares aligned (scale + shift) to
    the running median twice, then the median is renormalised to [0, 1].
    """
    if len(preds) == 1:
        return preds[0]
    stack = np.stack(preds).astype(np.float64)
    ref = np.median(stack, axis=0)
    for _ in range(2):
        for i in range(len(stack)):
            a = np.stack([stack[i].ravel(), np.ones(stack[i].size)], axis=1)
            coef, *_ = np.linalg.lstsq(a, ref.ravel(), rcond=None)
            stack[i] = stack[i] * coef[0] + coef[1]
        ref = np.median(stack, axis=0)
    lo, hi = ref.min(), ref.max()
    return ((ref - lo) / (hi - lo if hi > lo else 1.0)).astype(np.float32)


def predict(sessions, cfg, rgb, steps=0, processing_resolution=None, ensemble_size=1,
            seed=0, latents=None, progress=None):
    """Run Marigold. ``sessions`` = (vae_encoder, unet, vae_decoder) ORT sessions.

    ``steps`` / ``processing_resolution`` of 0/None use the checkpoint
    defaults stored in ``cfg``. ``latents`` (ensemble, 4, h, w) overrides the
    seeded starting noise (used to compare against diffusers exactly).
    Returns depth [h, w] in [0, 1] at the processing resolution.
    """
    enc, unet, dec = sessions
    steps = int(steps) or int(cfg.get("default_denoising_steps") or 1)
    if processing_resolution is None:
        processing_resolution = int(cfg.get("default_processing_resolution") or 768)
    image, (ph, pw) = preprocess(rgb, processing_resolution)
    image_latent = enc.run(None, {"image": image})[0].astype(np.float32)   # [1,4,h,w]

    sched = DDIM.from_config(cfg)
    timesteps = sched.timesteps(steps)
    rng = np.random.default_rng(seed)
    total = ensemble_size * (len(timesteps) + 1)
    done = 0
    preds = []
    for e in range(ensemble_size):
        if latents is not None:
            lat = np.asarray(latents[e:e + 1], dtype=np.float32)
        else:
            lat = rng.standard_normal(image_latent.shape).astype(np.float32)
        for t in timesteps:
            sample = np.concatenate([image_latent, lat], axis=1)
            noise = unet.run(None, {"sample": sample, "timestep": np.array([t], dtype=np.int64)})[0]
            lat = sched.step(noise, t, lat)
            done += 1
            if progress:
                progress(done / total)
        depth = dec.run(None, {"latent": lat})[0][0, 0].astype(np.float32)   # [H,W] in [0,1]
        depth = depth[: depth.shape[0] - ph or None, : depth.shape[1] - pw or None]
        preds.append(depth)
        done += 1
        if progress:
            progress(done / total)
    return ensemble(preds)
