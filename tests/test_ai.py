import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from depth_scene.ai import infer, marigold, models, runtime  # noqa: E402
from depth_scene.core.ddim import DDIM  # noqa: E402
from fake_models import fake_reference, make_marigold_like, zero_snr_trailing_config  # noqa: E402


def test_registry_is_marigold_only_and_licence_flagged():
    assert list(models.MODELS) == ["marigold_depth_v1_1"]
    info = models.get(models.DEFAULT_MODEL)
    assert info.family == "marigold" and info.output_kind == "depth"
    assert info.commercial_ok and "RAIL++-M" in info.license_id
    assert info.filename("fp32") == "marigold_depth_v1_1_fp32"
    assert info.filename("fp16") == "marigold_depth_v1_1_fp16"
    assert "export_marigold_onnx.py" in info.manual


# ---------------------------------------------------------------- DDIM port (no diffusers needed)

def test_trailing_timesteps_match_marigold_v1_1_schedule():
    sched = DDIM.from_config(zero_snr_trailing_config())
    assert sched.timesteps(1).tolist() == [999]
    assert sched.timesteps(4).tolist() == [999, 749, 499, 249]


def test_leading_timesteps_with_offset():
    cfg = dict(zero_snr_trailing_config(), timestep_spacing="leading")
    assert DDIM.from_config(cfg).timesteps(4).tolist() == [751, 501, 251, 1]


def test_single_trailing_step_is_the_x0_prediction():
    """At t=999 zero-SNR alpha is 0, so a v-prediction step returns -v scaled by sqrt(final alpha)."""
    cfg = zero_snr_trailing_config()
    sched = DDIM.from_config(cfg)
    sched.timesteps(1)
    x = np.random.default_rng(0).standard_normal((1, 4, 3, 5)).astype(np.float32)
    v = np.random.default_rng(1).standard_normal(x.shape).astype(np.float32)
    out = sched.step(v, 999, x)
    a_prev = cfg["final_alpha_cumprod"]
    expected = a_prev ** 0.5 * (-v) + (1 - a_prev) ** 0.5 * x     # x0 = -v, eps = x when alpha_t = 0
    assert np.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------- pipeline plumbing with fake graphs

def _sessions(folder):
    return runtime.get_sessions([os.path.join(folder, f) for f in marigold.FILES[:3]], "CPU")


@pytest.mark.parametrize("steps", [1, 3])
def test_predict_matches_numpy_replay(tmp_path, steps):
    folder = make_marigold_like(str(tmp_path / "m"))
    cfg = marigold.load_config(folder)
    rgb = np.random.default_rng(2).random((48, 72, 3)).astype(np.float32)   # multiples of 8: no pad
    lat = np.random.default_rng(3).standard_normal((1, 4, 6, 9)).astype(np.float32)
    got = marigold.predict(_sessions(folder), cfg, rgb, steps=steps, processing_resolution=0, latents=lat)
    image = (rgb * 2 - 1).transpose(2, 0, 1)[None]
    ref = fake_reference(image, lat, steps, cfg)
    assert got.shape == (48, 72)
    assert np.abs(got - ref).max() < 1e-5


def test_preprocess_resizes_longest_edge_and_pads_to_8():
    rgb = np.full((300, 451, 3), 0.25, np.float32)
    x, (ph, pw) = marigold.preprocess(rgb, 100)
    # diffusers: new = side * R // longest  -> 66 x 100, then pad to 72 x 104
    assert (ph, pw) == (6, 4)
    assert x.shape == (1, 3, 72, 104)
    assert np.allclose(x, -0.5)                                   # [0,1] -> [-1,1], replicate pad


def test_predict_unpads_and_uses_defaults(tmp_path):
    folder = make_marigold_like(str(tmp_path / "m"))
    cfg = marigold.load_config(folder)                            # default resolution 64, 1 step
    rgb = np.random.default_rng(4).random((50, 30, 3)).astype(np.float32)
    seen = []
    depth = marigold.predict(_sessions(folder), cfg, rgb, progress=seen.append)
    assert depth.shape == (64, 38)                                # 50x30 -> 64x38 (padded to 64x40, unpadded)
    assert 0.0 <= depth.min() and depth.max() <= 1.0
    assert seen[-1] == pytest.approx(1.0)


def test_ensemble_realigns_affine_copies():
    base = np.random.default_rng(5).random((20, 30)).astype(np.float32)
    members = [base, base * 0.5 + 0.2, base * 1.5 - 0.1]
    out = marigold.ensemble(members)
    norm = (base - base.min()) / (base.max() - base.min())
    assert np.abs(out - norm).max() < 1e-4


def test_estimate_reports_missing_files(tmp_path):
    folder = make_marigold_like(str(tmp_path / "m"))
    os.remove(os.path.join(folder, "unet.onnx"))
    with pytest.raises(RuntimeError, match="unet.onnx"):
        infer.estimate(models.get(models.DEFAULT_MODEL), folder, np.zeros((16, 16, 3), np.float32), device="CPU")


def test_estimate_end_to_end_with_seed(tmp_path):
    folder = make_marigold_like(str(tmp_path / "m"))
    rgb = np.random.default_rng(6).random((40, 64, 3)).astype(np.float32)
    info = models.get(models.DEFAULT_MODEL)
    a = infer.estimate(info, folder, rgb, device="CPU", processing_resolution=64, steps=2, seed=7)
    b = infer.estimate(info, folder, rgb, device="CPU", processing_resolution=64, steps=2, seed=7)
    c = infer.estimate(info, folder, rgb, device="CPU", processing_resolution=64, steps=2, seed=8)
    assert a.kind == "depth" and a.depth.shape == (40, 64)
    assert np.array_equal(a.depth, b.depth) and not np.array_equal(a.depth, c.depth)
    assert a.provider == "CPUExecutionProvider"


def test_gpu_request_without_gpu_provider_is_explicit():
    if any(p in runtime.available_providers() for p in runtime.GPU_PROVIDERS):
        pytest.skip("machine has a GPU provider")
    with pytest.raises(RuntimeError, match="No GPU execution provider"):
        runtime.choose_providers("GPU")
    assert runtime.choose_providers("AUTO") == ["CPUExecutionProvider"]


# ---------------------------------------------------------------- exact parity with diffusers (optional)

def test_ddim_port_matches_diffusers_scheduler():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    worst = 0.0
    for pred in ("epsilon", "v_prediction", "sample"):
        for spacing in ("leading", "trailing", "linspace"):
            zsnr = pred != "epsilon"
            s = diffusers.DDIMScheduler(
                num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                clip_sample=False, set_alpha_to_one=False, steps_offset=1, prediction_type=pred,
                timestep_spacing=spacing, rescale_betas_zero_snr=zsnr)
            mine = DDIM([float(a) for a in s.alphas_cumprod.double()], float(s.final_alpha_cumprod),
                        1000, spacing, 1, pred)
            for n in (1, 2, 4, 10):
                s.set_timesteps(n)
                assert np.array_equal(mine.timesteps(n), s.timesteps.numpy())
                g = torch.Generator().manual_seed(n)
                x = torch.randn(1, 4, 8, 12, generator=g)
                xm = x.numpy().copy()
                for t in s.timesteps:
                    out = torch.randn(1, 4, 8, 12, generator=g)
                    x = s.step(out, t, x).prev_sample
                    xm = mine.step(out.numpy(), int(t), xm)
                worst = max(worst, float(np.abs(xm - x.numpy()).max() / np.abs(x.numpy()).max()))
    assert worst < 1e-5
