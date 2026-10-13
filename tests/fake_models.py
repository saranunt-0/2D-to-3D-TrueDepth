"""Tiny ONNX graphs with the same I/O contract as the exported Marigold graphs.

Real weights cannot be downloaded in CI, so these stand-ins exercise the
add-on's denoising loop end to end (feeds, timestep dtype, latent shapes,
concatenation order, unpadding, ensembling). Each fake is simple enough that
tests can recompute the expected depth in NumPy.
"""

import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

UNET_IMAGE_GAIN = 0.1     # noise = 0.1 * image_latent + 0.5 * depth_latent + 1e-4 * t
UNET_DEPTH_GAIN = 0.5
UNET_T_GAIN = 1e-4


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _const(name, arr):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.asarray(arr), name + "_v"))


def zero_snr_trailing_config():
    """marigold.json for an SD2-style zero-SNR, v-prediction, trailing DDIM (computed in NumPy)."""
    betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
    ac = np.cumprod(1.0 - betas)
    s = np.sqrt(ac)
    s0, sT = s[0], s[-1]
    s = (s - sT) * s0 / (s0 - sT)
    ac = s ** 2
    alphas = np.concatenate([ac[:1], ac[1:] / ac[:-1]])
    ac = np.cumprod(alphas)
    return {
        "format": 1, "model": "fake", "scaling_factor": 0.18215, "num_train_timesteps": 1000,
        "timestep_spacing": "trailing", "steps_offset": 1, "prediction_type": "v_prediction",
        "alphas_cumprod": ac.tolist(), "final_alpha_cumprod": float(ac[0]),
        "default_denoising_steps": 1, "default_processing_resolution": 64, "license": "test",
    }


APPEARANCE_PROPS = {
    "target_names": ["albedo", "material"],
    "albedo": {"prediction_space": "srgb"},
    "material": {"prediction_space": "stack", "sub_target_names": ["roughness", "metallicity", None]},
    "roughness": {"prediction_space": "linear", "up_to_scale": False},
    "metallicity": {"prediction_space": "linear", "up_to_scale": False},
}
LIGHTING_PROPS = {
    "target_names": ["albedo", "shading", "residual"],
    "albedo": {"prediction_space": "linear", "up_to_scale": False},
    "shading": {"prediction_space": "linear", "up_to_scale": True},
    "residual": {"prediction_space": "linear", "up_to_scale": True},
}
DECODER_GAIN = 1.5         # rgb decoder: clip(1.5 * tanh(first 3 latent channels)) -> exercises clipping


def task_config(task, cfg=None):
    """marigold.json for a fake "depth" | "normals" | "iid-appearance" | "iid-lighting" model."""
    cfg = dict(cfg or zero_snr_trailing_config())
    if task == "normals":
        cfg.update(task="normals", n_targets=1, use_full_z_range=True, decoder_output="rgb")
    elif task == "iid-appearance":
        cfg.update(task="intrinsics", n_targets=2, target_properties=APPEARANCE_PROPS, decoder_output="rgb")
    elif task == "iid-lighting":
        cfg.update(task="intrinsics", n_targets=3, target_properties=LIGHTING_PROPS, decoder_output="rgb")
    return cfg


def make_marigold_like(folder, cfg=None, task="depth"):
    """Write vae_encoder/unet/vae_decoder.onnx + marigold.json into ``folder``."""
    os.makedirs(folder, exist_ok=True)
    cfg = task_config(task, cfg)
    n_t = int(cfg.get("n_targets", 1))

    # encoder: 8x8 average pool -> RGB means + their channel mean = 4 latent channels
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [None, 3, None, None])
    latent = helper.make_tensor_value_info("latent", TensorProto.FLOAT, [None, 4, None, None])
    nodes = [
        helper.make_node("AveragePool", ["image"], ["p"], kernel_shape=[8, 8], strides=[8, 8]),
        helper.make_node("ReduceMean", ["p"], ["m"], axes=[1], keepdims=1),
        helper.make_node("Concat", ["p", "m"], ["latent"], axis=1),
    ]
    _save(helper.make_graph(nodes, "fake_vae_encoder", [image], [latent]),
          os.path.join(folder, "vae_encoder.onnx"))

    # unet: linear in both latents and the timestep, so tests can replay it exactly
    sample = helper.make_tensor_value_info("sample", TensorProto.FLOAT, [None, 4 + 4 * n_t, None, None])
    timestep = helper.make_tensor_value_info("timestep", TensorProto.INT64, [1])
    noise = helper.make_tensor_value_info("noise", TensorProto.FLOAT, [None, 4 * n_t, None, None])
    nodes = [
        _const("split", np.array([4, 4 * n_t], np.int64)),
        helper.make_node("Split", ["sample", "split"], ["img", "dep"], axis=1),
        _const("reps", np.array([1, n_t, 1, 1], np.int64)),
        helper.make_node("Tile", ["img", "reps"], ["imgt"]),
        _const("gi", np.float32(UNET_IMAGE_GAIN)), _const("gd", np.float32(UNET_DEPTH_GAIN)),
        _const("gt", np.float32(UNET_T_GAIN)),
        helper.make_node("Mul", ["imgt", "gi"], ["a"]),
        helper.make_node("Mul", ["dep", "gd"], ["b"]),
        helper.make_node("Cast", ["timestep"], ["tf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["tf", "gt"], ["c"]),
        helper.make_node("Add", ["a", "b"], ["ab"]),
        helper.make_node("Add", ["ab", "c"], ["noise"]),
    ]
    _save(helper.make_graph(nodes, "fake_unet", [sample, timestep], [noise]),
          os.path.join(folder, "unet.onnx"))

    latent_in = helper.make_tensor_value_info("latent", TensorProto.FLOAT, [None, 4, None, None])
    if cfg.get("task", "depth") == "depth":
        # decoder: channel mean -> x8 nearest upsample -> sigmoid (always in [0, 1])
        out = helper.make_tensor_value_info("depth", TensorProto.FLOAT, [None, 1, None, None])
        nodes = [
            helper.make_node("ReduceMean", ["latent"], ["m"], axes=[1], keepdims=1),
            _const("roi", np.array([], np.float32)),
            _const("scales", np.array([1, 1, 8, 8], np.float32)),
            helper.make_node("Resize", ["m", "roi", "scales"], ["u"], mode="nearest"),
            helper.make_node("Sigmoid", ["u"], ["depth"]),
        ]
    else:
        # decoder: first 3 channels -> x8 nearest -> clip(1.5 * tanh) in [-1, 1]
        out = helper.make_tensor_value_info("decoded", TensorProto.FLOAT, [None, 3, None, None])
        nodes = [
            _const("s0", np.array([0], np.int64)), _const("s3", np.array([3], np.int64)),
            _const("ax", np.array([1], np.int64)),
            helper.make_node("Slice", ["latent", "s0", "s3", "ax"], ["rgb"]),
            _const("roi", np.array([], np.float32)),
            _const("scales", np.array([1, 1, 8, 8], np.float32)),
            helper.make_node("Resize", ["rgb", "roi", "scales"], ["u"], mode="nearest"),
            helper.make_node("Tanh", ["u"], ["th"]),
            _const("g", np.float32(DECODER_GAIN)),
            helper.make_node("Mul", ["th", "g"], ["thg"]),
            _const("lo", np.float32(-1.0)), _const("hi", np.float32(1.0)),
            helper.make_node("Clip", ["thg", "lo", "hi"], ["decoded"]),
        ]
    _save(helper.make_graph(nodes, "fake_vae_decoder", [latent_in], [out]),
          os.path.join(folder, "vae_decoder.onnx"))

    with open(os.path.join(folder, "marigold.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return folder


def fake_denoise(image_nchw, lat, steps, cfg):
    """NumPy replay of the fake encoder + U-Net + DDIM: final latent [1,4T,h,w]."""
    from depth_scene.core.ddim import DDIM

    b, c, h, w = image_nchw.shape
    p = image_nchw.reshape(b, c, h // 8, 8, w // 8, 8).mean(axis=(3, 5))
    img_lat = np.concatenate([p, p.mean(axis=1, keepdims=True)], axis=1)
    img_lat = np.tile(img_lat, (1, lat.shape[1] // 4, 1, 1))
    sched = DDIM.from_config(cfg)
    for t in sched.timesteps(steps):
        noise = UNET_IMAGE_GAIN * img_lat + UNET_DEPTH_GAIN * lat + UNET_T_GAIN * t
        lat = sched.step(noise, t, lat)
    return lat


def fake_reference(image_nchw, lat, steps, cfg):
    """NumPy replay of the fake depth graphs + DDIM for one ensemble member."""
    lat = fake_denoise(image_nchw, lat, steps, cfg)
    m = lat.mean(axis=1)[0]
    up = np.repeat(np.repeat(m, 8, axis=0), 8, axis=1)
    return 1.0 / (1.0 + np.exp(-up))


def fake_decoded(image_nchw, lat, steps, cfg):
    """NumPy replay for the rgb-decoder tasks: [T,3,H,W] in [-1,1] (before task post-processing)."""
    lat = fake_denoise(image_nchw, lat, steps, cfg)
    out = []
    for i in range(lat.shape[1] // 4):
        x = lat[0, 4 * i: 4 * i + 3]
        x = np.repeat(np.repeat(x, 8, axis=1), 8, axis=2)
        out.append(np.clip(DECODER_GAIN * np.tanh(x), -1.0, 1.0))
    return np.stack(out)
