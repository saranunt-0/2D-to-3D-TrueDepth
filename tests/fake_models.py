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


def make_marigold_like(folder, cfg=None):
    """Write vae_encoder/unet/vae_decoder.onnx + marigold.json into ``folder``."""
    os.makedirs(folder, exist_ok=True)

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
    sample = helper.make_tensor_value_info("sample", TensorProto.FLOAT, [None, 8, None, None])
    timestep = helper.make_tensor_value_info("timestep", TensorProto.INT64, [1])
    noise = helper.make_tensor_value_info("noise", TensorProto.FLOAT, [None, 4, None, None])
    nodes = [
        _const("split", np.array([4, 4], np.int64)),
        helper.make_node("Split", ["sample", "split"], ["img", "dep"], axis=1),
        _const("gi", np.float32(UNET_IMAGE_GAIN)), _const("gd", np.float32(UNET_DEPTH_GAIN)),
        _const("gt", np.float32(UNET_T_GAIN)),
        helper.make_node("Mul", ["img", "gi"], ["a"]),
        helper.make_node("Mul", ["dep", "gd"], ["b"]),
        helper.make_node("Cast", ["timestep"], ["tf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["tf", "gt"], ["c"]),
        helper.make_node("Add", ["a", "b"], ["ab"]),
        helper.make_node("Add", ["ab", "c"], ["noise"]),
    ]
    _save(helper.make_graph(nodes, "fake_unet", [sample, timestep], [noise]),
          os.path.join(folder, "unet.onnx"))

    # decoder: channel mean -> x8 nearest upsample -> sigmoid (always in [0, 1])
    latent_in = helper.make_tensor_value_info("latent", TensorProto.FLOAT, [None, 4, None, None])
    depth = helper.make_tensor_value_info("depth", TensorProto.FLOAT, [None, 1, None, None])
    nodes = [
        helper.make_node("ReduceMean", ["latent"], ["m"], axes=[1], keepdims=1),
        _const("roi", np.array([], np.float32)),
        _const("scales", np.array([1, 1, 8, 8], np.float32)),
        helper.make_node("Resize", ["m", "roi", "scales"], ["u"], mode="nearest"),
        helper.make_node("Sigmoid", ["u"], ["depth"]),
    ]
    _save(helper.make_graph(nodes, "fake_vae_decoder", [latent_in], [depth]),
          os.path.join(folder, "vae_decoder.onnx"))

    with open(os.path.join(folder, "marigold.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg or zero_snr_trailing_config(), fh)
    return folder


def fake_reference(image_nchw, lat, steps, cfg):
    """NumPy replay of the fake graphs + DDIM for one ensemble member."""
    from depth_scene.core.ddim import DDIM

    b, c, h, w = image_nchw.shape
    p = image_nchw.reshape(b, c, h // 8, 8, w // 8, 8).mean(axis=(3, 5))
    img_lat = np.concatenate([p, p.mean(axis=1, keepdims=True)], axis=1)
    sched = DDIM.from_config(cfg)
    for t in sched.timesteps(steps):
        noise = UNET_IMAGE_GAIN * img_lat + UNET_DEPTH_GAIN * lat + UNET_T_GAIN * t
        lat = sched.step(noise, t, lat)
    m = lat.mean(axis=1)[0]
    up = np.repeat(np.repeat(m, 8, axis=0), 8, axis=1)
    return 1.0 / (1.0 + np.exp(-up))
