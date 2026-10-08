"""Tiny ONNX graphs with the same I/O signature as the real depth models.

Real weights cannot be downloaded in CI, so these stand-ins exercise the
whole inference path (feeds, dtypes, dynamic shapes, post-processing).
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_depth_anything_like(path, fp16=False):
    """pixel_values [1,3,H,W] -> predicted_depth [1,H,W] (= channel mean, ReLU)."""
    dt = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    inp = helper.make_tensor_value_info("pixel_values", dt, [1, 3, "h", "w"])
    out = helper.make_tensor_value_info("predicted_depth", dt, [1, "h", "w"])
    nodes = [
        # opset 13: ReduceMean takes axes as an attribute
        helper.make_node("ReduceMean", ["pixel_values"], ["m"], axes=[1], keepdims=0),
        helper.make_node("Relu", ["m"], ["predicted_depth"]),
    ]
    graph = helper.make_graph(nodes, "fake_da2", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_moge_like(path, affine_points, mask, metric_scale):
    """image [1,3,H,W] + num_tokens -> constant points/normal/mask/metric_scale.

    The outputs ignore the input and return a precomputed affine point map,
    so the test can check focal/shift recovery end to end.
    """
    h, w = affine_points.shape[:2]
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "h", "w"])
    tokens = helper.make_tensor_value_info("num_tokens", TensorProto.INT64, [])
    outs = [
        helper.make_tensor_value_info("points", TensorProto.FLOAT, [1, h, w, 3]),
        helper.make_tensor_value_info("normal", TensorProto.FLOAT, [1, h, w, 3]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, h, w]),
        helper.make_tensor_value_info("metric_scale", TensorProto.FLOAT, [1]),
    ]
    consts = {
        "points": affine_points[None].astype(np.float32),
        "normal": np.zeros((1, h, w, 3), np.float32),
        "mask": mask[None].astype(np.float32),
        "metric_scale": np.array([metric_scale], np.float32),
    }
    nodes = [
        helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr, name + "_v"))
        for name, arr in consts.items()
    ]
    graph = helper.make_graph(nodes, "fake_moge2", [image, tokens], outs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
