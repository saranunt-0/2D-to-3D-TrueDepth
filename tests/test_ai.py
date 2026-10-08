import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from depth_scene.ai import infer, models, runtime  # noqa: E402
from fake_models import make_depth_anything_like, make_moge_like  # noqa: E402
from synthetic import camera_points, corridor_depth  # noqa: E402


def test_registry_licences_are_flagged():
    assert models.MODELS[models.DEFAULT_MODEL].commercial_ok
    for key in ("da2_base", "da2_large"):
        assert not models.MODELS[key].commercial_ok
        assert models.MODELS[key].license_id == "CC-BY-NC-4.0"
    for m in models.MODELS.values():
        assert m.url("fp32").startswith("https://huggingface.co/")
        assert m.url("fp16")  # falls back to fp32 when no fp16 file exists


@pytest.mark.parametrize("fp16", [False, True])
def test_depth_anything_preprocess_shapes(tmp_path, fp16):
    path = str(tmp_path / "da.onnx")
    make_depth_anything_like(path, fp16=fp16)
    rgb = np.random.default_rng(0).random((300, 450, 3)).astype(np.float32)
    res = infer.estimate(models.get("da2_small"), path, rgb, device="CPU", input_size=518)
    h, w = res.depth.shape
    assert h % 14 == 0 and w % 14 == 0
    assert min(h, w) == 518                         # shorter side, official resize
    assert abs(w / h - 450 / 300) < 0.05            # aspect preserved
    assert res.kind == "disparity" and res.depth.dtype == np.float32
    assert res.provider == "CPUExecutionProvider"


def test_moge_pipeline_recovers_fov_and_metric_depth(tmp_path):
    fov = math.radians(70.0)
    w, h = 112, 84
    depth, _ = corridor_depth(w, h, fov)
    gt = camera_points(depth, fov)
    cv = np.stack([gt[..., 0], -gt[..., 1], -gt[..., 2]], axis=-1)
    scale, shift = 2.5, 0.6
    affine = cv / scale
    affine[..., 2] -= shift
    mask = np.ones((h, w), np.float32)
    mask[:3] = 0.0                                  # pretend the top rows are sky
    path = str(tmp_path / "moge.onnx")
    make_moge_like(path, affine, mask, scale)
    rgb = np.zeros((h * 4, w * 4, 3), np.float32)
    res = infer.estimate(models.get("moge2_vits"), path, rgb, device="CPU")
    assert res.kind == "metric"
    assert math.degrees(res.fov) == pytest.approx(70.0, abs=0.3)
    assert not res.valid[:3].any() and res.valid[3:].all()
    assert np.allclose(res.depth[3:], depth[3:], rtol=5e-3)


def test_gpu_request_without_gpu_provider_is_explicit():
    if any(p in runtime.available_providers() for p in runtime.GPU_PROVIDERS):
        pytest.skip("machine has a GPU provider")
    with pytest.raises(RuntimeError, match="No GPU execution provider"):
        runtime.choose_providers("GPU")
    assert runtime.choose_providers("AUTO") == ["CPUExecutionProvider"]
