import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_scene.core import depth as dm  # noqa: E402
from depth_scene.core import geometry as geo  # noqa: E402
from depth_scene.core import imaging  # noqa: E402
from synthetic import camera_points, corridor_depth  # noqa: E402

FOV = math.radians(65.0)


# ---------------------------------------------------------------- imaging

def test_resize_constant_and_shape():
    img = np.full((37, 53, 3), 0.25, np.float32)
    out = imaging.resize(img, 11, 17)
    assert out.shape == (11, 17, 3)
    assert np.allclose(out, 0.25, atol=1e-6)


def test_resize_downsample_antialiases_checkerboard():
    yy, xx = np.mgrid[0:64, 0:64]
    checker = ((xx + yy) % 2).astype(np.float32)
    out = imaging.resize(checker, 8, 8)
    assert np.allclose(out, 0.5, atol=0.02)


def test_resize_upsample_keeps_linear_ramp():
    ramp = np.tile(np.arange(10, dtype=np.float32), (4, 1))
    out = imaging.resize(ramp, 4, 40)
    interior = out[0, 4:-4]
    assert np.all(np.diff(interior) > 0)
    assert np.allclose(np.diff(interior), 0.25, atol=1e-4)


def test_gaussian_blur_variance_and_mass():
    img = np.zeros((201, 201), np.float32)
    img[100, 100] = 1.0
    sigma = 6.0
    out = imaging.gaussian_blur(img, sigma)
    assert out.sum() == pytest.approx(1.0, abs=1e-3)
    yy, xx = np.mgrid[0:201, 0:201]
    var_x = (out * (xx - 100) ** 2).sum()
    assert math.sqrt(var_x) == pytest.approx(sigma, rel=0.08)


def test_sample_bilinear_edges_and_centre():
    img = np.array([[0.0, 1.0], [2.0, 3.0]], np.float32)
    assert imaging.sample_bilinear(img, 0.0, 0.0) == pytest.approx(0.0)
    assert imaging.sample_bilinear(img, 1.0, 1.0) == pytest.approx(3.0)
    assert imaging.sample_bilinear(img, 0.5, 0.5) == pytest.approx(1.5)


# ---------------------------------------------------------------- depth mapping

def test_inverse_mapping_recovers_depth_from_affine_disparity():
    z = np.linspace(1.5, 12.0, 50, dtype=np.float32)
    disparity = 3.7 / z + 0.42          # unknown scale + shift, as from DA2
    t = dm.nearness(disparity, "disparity")
    rec = dm.map_to_distance(t, "INVERSE", near=1.5, far=12.0)
    assert np.allclose(rec, z, rtol=1e-4)


def test_linear_mapping_recovers_relative_depth():
    z = np.linspace(2.0, 9.0, 30, dtype=np.float32)
    t = dm.nearness(z * 0.1 + 5.0, "depth")
    rec = dm.map_to_distance(t, "LINEAR", near=2.0, far=9.0)
    assert np.allclose(rec, z, rtol=1e-5)


def test_metric_nearness_and_invalid_fill():
    z = np.array([[1.0, 2.0], [np.inf, 4.0]], np.float32)
    t = dm.nearness(z, "metric")
    assert t[0, 0] == pytest.approx(1.0) and t[1, 1] == pytest.approx(0.0)
    filled = dm.metric_distance(z, 2.0)
    assert filled[1, 0] == pytest.approx(8.0)


# ---------------------------------------------------------------- geometry

def test_grid_shape_follows_aspect():
    assert geo.grid_shape(2000, 1000, 200) == (200, 100)
    assert geo.grid_shape(500, 1000, 64) == (32, 64)


def test_face_winding_points_at_camera():
    depth = np.full((40, 60), 5.0, np.float32)
    p = geo.MeshParams(resolution=12, mapping="LINEAR", near=5.0, far=5.0001, fov=FOV)
    verts, faces, _, _ = geo.build(p, depth, "depth", (60, 40))
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    assert np.all(normals[:, 2] > 0)


def _corridor_mesh(res, pillar=None, **kw):
    w, h = 320, 240
    depth, _ = corridor_depth(w, h, FOV, pillar=pillar)
    disparity = 2.3 / depth + 0.17
    p = geo.MeshParams(resolution=res, fov=FOV, mapping="INVERSE",
                       near=float(depth.min()), far=float(depth.max()), **kw)
    return geo.build(p, disparity, "disparity", (w, h)), depth


def test_corridor_walls_come_out_planar():
    (verts, faces, uv, info), depth = _corridor_mesh(160, remove_stretched=False)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    # distance of every vertex to the nearest of the 5 corridor planes
    d = np.min(np.stack([np.abs(np.abs(x) - 1.0), np.abs(np.abs(y) - 1.2), np.abs(z + 12.0)]), axis=0)
    # sampling a depth map between pixel centres blurs only the plane seams
    assert np.median(d) < 1e-3
    assert np.percentile(d, 95) < 0.05


def test_stretched_faces_spare_walls_but_cut_occlusions():
    (_, faces_clean, _, info_clean), _ = _corridor_mesh(200)
    total = 199 * 149
    # grazing walls survive, apart from a few faces at the vanishing point
    assert info_clean["stretched_removed"] / total < 0.01

    (_, _, _, info_occ), _ = _corridor_mesh(200, pillar=(-0.9, -0.3, 3.0))
    # the pillar's two silhouette edges each cut ~1 column of faces top to bottom
    assert info_occ["stretched_removed"] - info_clean["stretched_removed"] > 100


def test_stretch_test_is_resolution_independent_for_walls():
    for res in (64, 128, 512):
        (_, _, _, info), _ = _corridor_mesh(res)
        nx, ny = info["grid"]
        assert info["stretched_removed"] / ((nx - 1) * (ny - 1)) < 0.02


def test_metric_mapping_unprojects_exactly():
    w, h = 160, 120
    depth, _ = corridor_depth(w, h, FOV)
    p = geo.MeshParams(resolution=w, fov=FOV, mapping="METRIC", metric_scale=1.0,
                       remove_stretched=False)
    verts, _, _, _ = geo.build(p, depth, "metric", (w, h))
    gt = camera_points(depth, FOV)
    # vertex grid spans pixel edges, GT is at pixel centres: compare the far wall
    far = verts[:, 2] < -11.99
    assert far.sum() > 10
    # only the vertices sitting exactly on a wall seam get a blended depth
    assert np.mean(np.abs(verts[far, 2] + 12.0) < 1e-4) >= 0.85
    assert np.abs(verts[:, 0]).max() <= np.abs(gt[..., 0]).max() * 1.05


def test_relief_displacement_range_and_invert():
    disparity = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    p = geo.MeshParams(mode="RELIEF", resolution=10, base_displacement=0.5, size=2.0)
    verts, _, _, _ = geo.build(p, disparity, "disparity", (10, 10))
    assert verts[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert verts[:, 2].max() == pytest.approx(0.5, abs=1e-6)
    assert np.abs(verts[:, 0]).max() == pytest.approx(1.0)
    p.invert = True
    inv, _, _, _ = geo.build(p, disparity, "disparity", (10, 10))
    assert np.allclose(inv[:, 2], 0.5 - verts[:, 2], atol=1e-6)


def test_alpha_removes_faces_and_compacts():
    depth = np.ones((20, 20), np.float32)
    alpha = np.ones((20, 20), np.float32)
    alpha[:, :10] = 0.0
    p = geo.MeshParams(mode="RELIEF", resolution=20, use_alpha=True)
    verts, faces, uv, info = geo.build(p, depth, "depth", (20, 20), alpha=alpha)
    assert info["faces"] < 19 * 19 / 2
    assert faces.max() == len(verts) - 1
    assert len(uv) == len(verts)
    assert np.all(uv[:, 0] > 0.45)


def test_detail_displacement_moves_bright_pixels_closer():
    depth = np.full((64, 64), 4.0, np.float32)
    gray = np.zeros((64, 64), np.float32)
    gray[28:36, 28:36] = 1.0
    p = geo.MeshParams(resolution=64, fov=FOV, mapping="LINEAR", near=4.0, far=4.001,
                       detail_strength=0.2, detail_radius=10.0, remove_stretched=False)
    verts, _, _, _ = geo.build(p, depth, "depth", (64, 64), gray=gray)
    zs = -verts[:, 2].reshape(64, 64)
    assert zs[32, 32] < zs[2, 2]


# ---------------------------------------------------------------- quantize (diorama look)

def test_quantize_levels_and_off():
    t = np.linspace(0, 1, 101, dtype=np.float32)
    q = dm.quantize(t, 5)
    assert sorted(set(np.round(q, 6))) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert np.array_equal(dm.quantize(t, 0), t) and np.array_equal(dm.quantize(t, 1), t)


def test_quantized_relief_gives_flat_layers():
    ramp = np.tile(np.linspace(0, 7, 64, dtype=np.float32), (48, 1))
    p = geo.MeshParams(mode="RELIEF", resolution=64, base_displacement=1.0, layer_steps=4)
    verts, _, _, _ = geo.build(p, ramp, "disparity", (64, 48))
    levels = sorted(set(np.round(verts[:, 2], 5)))
    assert levels == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0], abs=1e-5)


# ---------------------------------------------------------------- material passes

def test_srgb_linear_roundtrip():
    from depth_scene.core import imaging

    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.abs(imaging.linear_to_srgb(imaging.srgb_to_linear(x)) - x).max() < 1e-5


def test_lighting_calibration_recovers_scales():
    from depth_scene.core import passes

    rng = np.random.default_rng(0)
    albedo = rng.random((20, 30, 3)).astype(np.float32)
    shading = rng.random((20, 30, 3)).astype(np.float32)
    residual = rng.random((20, 30, 3)).astype(np.float32) * 0.2
    image = albedo * shading * 2.5 + residual * 0.4
    s, r, info = passes.calibrate_lighting(image, albedo, shading, residual)
    assert info["shading_scale"] == pytest.approx(2.5, rel=1e-4)
    assert info["residual_scale"] == pytest.approx(0.4, rel=1e-4)
    assert info["fit_error"] < 1e-5
    assert np.allclose(albedo * s + r, image, atol=1e-4)


def test_lighting_calibration_never_negative():
    from depth_scene.core import passes

    rng = np.random.default_rng(1)
    albedo = rng.random((10, 10, 3)).astype(np.float32)
    shading = rng.random((10, 10, 3)).astype(np.float32)
    residual = 1.0 - albedo * shading                 # anti-correlated: unconstrained fit gives b < 0
    image = albedo * shading * 1.5
    _, _, info = passes.calibrate_lighting(image, albedo, shading, residual)
    assert info["shading_scale"] >= 0 and info["residual_scale"] >= 0


def test_intrinsics_to_maps_converts_srgb_albedo_and_renames():
    from depth_scene.core import imaging, passes

    alb = np.full((4, 6, 3), 0.5, np.float32)
    rough = np.full((4, 6), 1.2, np.float32)
    maps, info = passes.intrinsics_to_maps({"albedo": (alb, "srgb", False), "roughness": (rough, "linear", False),
                                            "metallicity": (rough * 0, "linear", False)})
    assert set(maps) == {"albedo", "roughness", "metallic"}
    assert np.allclose(maps["albedo"], imaging.srgb_to_linear(alb))
    assert maps["roughness"].max() == 1.0                     # clipped to [0, 1]
    assert info == {}


# ---------------------------------------------------------------- Marigold V2 log depth

def test_log_mapping_inverts_affine_log_depth():
    """Log depth d = a*log z + b, min/max normalised: LOG mapping with the true near/far is exact."""
    z = np.geomspace(1.5, 24.0, 50).astype(np.float32)
    d = 0.37 * np.log(z) - 1.1                                  # unknown scale and shift
    t = dm.nearness(d, "log_depth")
    assert t[0] == pytest.approx(1.0) and t[-1] == pytest.approx(0.0)
    out = dm.map_to_distance(t, "LOG", 1.5, 24.0)
    assert np.allclose(out, z, rtol=1e-4)
    assert dm.default_mapping("log_depth") == "LOG"
    lin = dm.map_to_distance(t, "LINEAR", 1.5, 24.0)            # the wrong mapping bends the geometry
    assert np.abs(lin - z).max() > 3.0
