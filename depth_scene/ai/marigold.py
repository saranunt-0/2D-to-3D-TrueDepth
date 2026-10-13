"""Marigold (prs-eth/marigold-*-v1-1) with ONNX Runtime + NumPy.

Marigold is Stable Diffusion 2 fine-tuned to *generate* a dense map
conditioned on the image. ``tools/export_marigold_onnx.py`` splits a diffusers
Marigold pipeline into three ONNX graphs plus a JSON config; this module
re-implements the pipelines' ``__call__`` around them:

    vae_encoder.onnx   image [B,3,H,W] in [-1,1]   -> latent [B,4,H/8,W/8]
                       (posterior mode x VAE scaling factor)
    unet.onnx          sample [B,4+4T,h,w] (image latent ++ T target latents),
                       timestep int64 [1]            -> noise [B,4T,h,w]
                       (the empty-prompt text embedding is baked in)
    vae_decoder.onnx   latent [B,4,h,w] -> depth [B,1,H,W] in [0,1]      (depth: decode,
                                           channel mean, clip, (x+1)/2)
                                     or -> decoded [B,3,H,W] in [-1,1]  (normals / intrinsics:
                                           decode, clip)
    marigold.json      task, targets, scheduler numbers, defaults, licence

Tasks (``cfg["task"]``, "depth" when absent - folders exported before normals
support):

* depth       -> [h, w] affine-invariant depth (0 = near, 1 = far)
* normals     -> [h, w, 3] unit camera-space normals (x right, y up, z towards
                 the viewer)
* intrinsics  -> {target name: [h, w, 3] in [0, 1]} in each target's
                 ``prediction_space`` (see ``split_targets``)
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


def task_of(cfg):
    return cfg.get("task", "depth")


def n_targets(cfg):
    return int(cfg.get("n_targets", 1))


def target_names(cfg):
    if task_of(cfg) != "intrinsics":
        return [task_of(cfg)]
    return list(cfg["target_properties"]["target_names"])


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


# ----------------------------------------------------------------------------- ensembling

def ensemble(preds):
    """Combine depth ensemble members (each [H,W] in [0,1]).

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


def normalize_normals(n, eps=1e-6):
    """Unit length along axis 0 of a [3, ...] array (MarigoldNormalsPipeline.normalize_normals)."""
    return n / np.maximum(np.linalg.norm(n, axis=0, keepdims=True), eps)


def ensemble_normals(preds):
    """diffusers' default "closest" reduction: per pixel, keep the member closest to the mean direction."""
    if len(preds) == 1:
        return preds[0]
    stack = np.stack(preds)                                        # [E,3,H,W]
    mean = normalize_normals(stack.mean(axis=0))                   # [3,H,W]
    cos = (stack * mean[None]).sum(axis=1)                         # [E,H,W]
    best = cos.argmax(axis=0)                                      # [H,W]
    return np.take_along_axis(stack, best[None, None], axis=0)[0].astype(np.float32)


def ensemble_intrinsics(preds):
    """diffusers' default reduction for intrinsics: per-pixel median ([T,3,H,W] members)."""
    if len(preds) == 1:
        return preds[0]
    return np.median(np.stack(preds), axis=0).astype(np.float32)


# ----------------------------------------------------------------------------- decoding

def _unpad(x, ph, pw):
    return x[..., : x.shape[-2] - ph or None, : x.shape[-1] - pw or None]


def _decode(dec, cfg, lat, ph, pw):
    """One ensemble member's latent [1,4T,h,w] -> task-specific array (still channels-first)."""
    task = task_of(cfg)
    if task == "depth":
        return _unpad(dec.run(None, {"latent": lat})[0][0, 0].astype(np.float32), ph, pw)
    # One target at a time keeps the decoder's full-resolution activations to batch 1.
    decoded = [_unpad(dec.run(None, {"latent": np.ascontiguousarray(lat[:, 4 * i: 4 * i + 4])})[0][0], ph, pw)
               for i in range(n_targets(cfg))]
    x = np.clip(np.stack(decoded).astype(np.float32), -1.0, 1.0)  # [T,3,H,W]
    if task == "normals":
        n = x[0]
        if not cfg.get("use_full_z_range", True):
            n[2] = n[2] * 0.5 + 0.5
        return normalize_normals(n)                                # [3,H,W]
    return (x + 1.0) / 2.0                                         # [T,3,H,W] in [0, 1]


def predict(sessions, cfg, rgb, steps=0, processing_resolution=None, ensemble_size=1,
            seed=0, latents=None, progress=None):
    """Run Marigold. ``sessions`` = (vae_encoder, unet, vae_decoder) ORT sessions.

    ``steps`` / ``processing_resolution`` of 0/None use the checkpoint
    defaults stored in ``cfg``. ``latents`` (ensemble, 4T, h, w) overrides the
    seeded starting noise (used to compare against diffusers exactly).
    Returns, at the processing resolution: depth [h, w]; normals [h, w, 3];
    intrinsics {target name: [h, w, 3]}.
    """
    enc, unet, dec = sessions
    task = task_of(cfg)
    if task not in ("depth", "normals", "intrinsics"):
        raise RuntimeError(f"Unsupported Marigold task '{task}' - update the add-on")
    steps = int(steps) or int(cfg.get("default_denoising_steps") or 1)
    if processing_resolution is None:
        processing_resolution = int(cfg.get("default_processing_resolution") or 768)
    image, (ph, pw) = preprocess(rgb, processing_resolution)
    image_latent = enc.run(None, {"image": image})[0].astype(np.float32)   # [1,4,h,w]
    b, c, h, w = image_latent.shape
    lat_shape = (b, c * n_targets(cfg), h, w)

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
            lat = rng.standard_normal(lat_shape).astype(np.float32)
        for t in timesteps:
            sample = np.concatenate([image_latent, lat], axis=1)
            noise = unet.run(None, {"sample": sample, "timestep": np.array([t], dtype=np.int64)})[0]
            lat = sched.step(noise, t, lat)
            done += 1
            if progress:
                progress(done / total)
        preds.append(_decode(dec, cfg, lat, ph, pw))
        done += 1
        if progress:
            progress(done / total)

    if task == "depth":
        return ensemble(preds)
    if task == "normals":
        return np.ascontiguousarray(ensemble_normals(preds).transpose(1, 2, 0))
    out = ensemble_intrinsics(preds)                               # [T,3,H,W]
    return {name: np.ascontiguousarray(out[i].transpose(1, 2, 0)) for i, name in enumerate(target_names(cfg))}


def split_targets(cfg, maps):
    """Intrinsics dict -> {name: (array, prediction_space, up_to_scale)}.

    "stack" targets (IID-Appearance's ``material``) are split into their
    single-channel sub-targets (roughness = R, metallicity = G).
    """
    props = cfg["target_properties"]
    out = {}
    for name in target_names(cfg):
        p = props.get(name, {})
        space = p.get("prediction_space", "srgb")
        if space == "stack":
            for i, sub in enumerate(p.get("sub_target_names", [])):
                if sub is None:
                    continue
                sp = props.get(sub, {})
                out[sub] = (maps[name][..., i], sp.get("prediction_space", "linear"), bool(sp.get("up_to_scale")))
        else:
            out[name] = (maps[name], space, bool(p.get("up_to_scale")))
    return out
