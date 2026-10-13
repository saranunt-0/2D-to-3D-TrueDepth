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
from fake_models import (fake_decoded, fake_reference, make_marigold_like,  # noqa: E402
                         zero_snr_trailing_config)


def test_registry_is_marigold_only_and_licence_flagged():
    assert list(models.MODELS) == ["marigold_depth_v1_1", "marigold_normals_v1_1",
                                   "marigold_iid_appearance_v1_1", "marigold_iid_lighting_v1_1"]
    assert list(models.DEPTH_MODELS) == ["marigold_depth_v1_1"]
    info = models.get(models.DEFAULT_MODEL)
    assert info.family == "marigold" and info.output_kind == "depth" and info.task == "depth"
    assert info.filename("fp32") == "marigold_depth_v1_1_fp32"
    assert info.filename("fp16") == "marigold_depth_v1_1_fp16"
    for m in models.MODELS.values():
        assert m.commercial_ok and "RAIL++-M" in m.license_id
        assert "export_marigold_onnx.py" in m.manual and m.checkpoint in m.manual
    assert set(models.PASS_MODELS) == {"normals", "appearance", "lighting"}
    assert models.get(models.PASS_MODELS["appearance"]).outputs == ("albedo", "roughness", "metallic")


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


# ---------------------------------------------------------------- normals / intrinsics

def _rgb_and_latents(n_t, seed=10, h=48, w=72):
    rgb = np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)
    lat = np.random.default_rng(seed + 1).standard_normal((3, 4 * n_t, h // 8, w // 8)).astype(np.float32)
    return rgb, lat


@pytest.mark.parametrize("full_z", [True, False])
def test_normals_match_numpy_replay(tmp_path, full_z):
    folder = make_marigold_like(str(tmp_path / "n"), cfg=None, task="normals")
    cfg = dict(marigold.load_config(folder), use_full_z_range=full_z)
    rgb, lat = _rgb_and_latents(1)
    got = marigold.predict(_sessions(folder), cfg, rgb, steps=2, processing_resolution=0, latents=lat[:1])
    assert got.shape == (48, 72, 3)
    n = fake_decoded((rgb * 2 - 1).transpose(2, 0, 1)[None], lat[:1], 2, cfg)[0]
    if not full_z:
        n[2] = n[2] * 0.5 + 0.5
    n = n / np.maximum(np.linalg.norm(n, axis=0, keepdims=True), 1e-6)
    assert np.abs(got - n.transpose(1, 2, 0)).max() < 1e-5
    assert np.allclose(np.linalg.norm(got, axis=-1), 1.0, atol=1e-5)


def test_normals_ensemble_picks_member_closest_to_mean():
    up = np.zeros((3, 2, 2), np.float32)
    up[1] = 1.0
    tilted = marigold.normalize_normals(up + np.array([0.3, 0, 0], np.float32)[:, None, None])
    away = np.zeros_like(up)
    away[0] = 1.0
    out = marigold.ensemble_normals([away, up, tilted])
    # mean direction ~ between x and y; tilted (x 0.3, y 1) is closer than pure y or pure x
    assert np.allclose(out, tilted)


@pytest.mark.parametrize("task,names", [("iid-appearance", ["albedo", "material"]),
                                        ("iid-lighting", ["albedo", "shading", "residual"])])
def test_intrinsics_match_numpy_replay_and_median(tmp_path, task, names):
    folder = make_marigold_like(str(tmp_path / task), task=task)
    cfg = marigold.load_config(folder)
    n_t = len(names)
    rgb, lat = _rgb_and_latents(n_t)
    got = marigold.predict(_sessions(folder), cfg, rgb, steps=1, processing_resolution=0,
                           ensemble_size=3, latents=lat)
    assert list(got) == names
    image = (rgb * 2 - 1).transpose(2, 0, 1)[None]
    members = [(fake_decoded(image, lat[e:e + 1], 1, cfg) + 1) / 2 for e in range(3)]
    ref = np.median(np.stack(members), axis=0)
    for i, name in enumerate(names):
        assert got[name].shape == (48, 72, 3)
        assert np.abs(got[name] - ref[i].transpose(1, 2, 0)).max() < 1e-5


def test_split_targets_unstacks_material():
    from fake_models import APPEARANCE_PROPS

    cfg = {"task": "intrinsics", "target_properties": APPEARANCE_PROPS}
    mat = np.random.default_rng(0).random((4, 5, 3)).astype(np.float32)
    alb = np.full((4, 5, 3), 0.5, np.float32)
    out = marigold.split_targets(cfg, {"albedo": alb, "material": mat})
    assert set(out) == {"albedo", "roughness", "metallicity"}
    assert out["albedo"][1] == "srgb"
    assert np.array_equal(out["roughness"][0], mat[..., 0]) and out["roughness"][1] == "linear"
    assert np.array_equal(out["metallicity"][0], mat[..., 1])


def test_estimate_passes_appearance_and_lighting(tmp_path):
    rgb = np.random.default_rng(12).random((40, 64, 3)).astype(np.float32)
    kw = dict(device="CPU", processing_resolution=64, steps=2, seed=1)
    app = models.get(models.PASS_MODELS["appearance"])
    res = infer.estimate_passes(app, make_marigold_like(str(tmp_path / "a"), task="iid-appearance"), rgb, **kw)
    assert list(res.maps) == ["albedo", "roughness", "metallic"]
    assert res.maps["albedo"].shape == (40, 64, 3) and res.maps["roughness"].shape == (40, 64)
    assert 0.0 <= res.maps["metallic"].min() and res.maps["metallic"].max() <= 1.0
    lit = models.get(models.PASS_MODELS["lighting"])
    res = infer.estimate_passes(lit, make_marigold_like(str(tmp_path / "l"), task="iid-lighting"), rgb, **kw)
    assert list(res.maps) == ["albedo", "shading", "residual"]
    assert {"shading_scale", "residual_scale", "fit_error"} <= set(res.info)
    nrm = models.get(models.PASS_MODELS["normals"])
    res = infer.estimate_passes(nrm, make_marigold_like(str(tmp_path / "n"), task="normals"), rgb, **kw)
    n = res.maps["normal"] * 2 - 1
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-5)


def test_wrong_model_folder_is_rejected(tmp_path):
    folder = make_marigold_like(str(tmp_path / "d"))                  # a depth export
    nrm = models.get(models.PASS_MODELS["normals"])
    with pytest.raises(RuntimeError, match="'depth' model, expected 'normals'"):
        infer.estimate_passes(nrm, folder, np.zeros((16, 16, 3), np.float32), device="CPU")


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
