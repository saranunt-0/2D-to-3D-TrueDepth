"""Integration tests inside Blender (``pip install bpy`` or run with Blender's Python).

    python -m pytest tests/test_blender.py
"""

import math
import os
import sys

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from synthetic import camera_points, corridor_image  # noqa: E402

FOV = math.radians(65.0)
W, H = 320, 240


@pytest.fixture(scope="module")
def addon(tmp_path_factory):
    import addon_utils

    tmp = tmp_path_factory.mktemp("depth_scene")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mod = addon_utils.enable("depth_scene", default_set=True, handle_error=None)
    assert mod is not None, "add-on failed to enable"
    prefs = bpy.context.preferences.addons["depth_scene"].preferences
    prefs.models_dir = str(tmp / "models")
    scene = bpy.context.scene
    scene.depth_scene.save_location = str(tmp / "depth")

    rgb, depth = corridor_image(W, H, FOV)
    img = bpy.data.images.new("corridor.png", W, H, alpha=True)
    rgba = np.ones((H, W, 4), np.float32)
    rgba[..., :3] = rgb
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = str(tmp / "corridor.png")
    img.file_format = "PNG"
    img.save()
    scene.depth_scene.image = img
    yield {"tmp": tmp, "image": img, "depth": depth, "rgb": rgb}
    addon_utils.disable("depth_scene", default_set=True)


def _world_verts(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(obj.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


def _depth_image_from(depth, name, tmp):
    from depth_scene import bimage

    return bimage.write_depth(name, depth.astype(np.float32), None, str(tmp))


def test_png_pixels_are_display_encoded(addon):
    """8-bit images come back sRGB-encoded (what the networks expect)."""
    from depth_scene import bimage

    img = bpy.data.images.load(str(addon["tmp"] / "corridor.png"), check_existing=False)
    rgb = bimage.display_rgb(img)
    assert np.abs(rgb - addon["rgb"]).max() < 1.5 / 255


def test_perspective_from_existing_metric_depth_is_exact(addon):
    s = bpy.context.scene.depth_scene
    s.use_existing_depth = True
    s.existing_depth = _depth_image_from(addon["depth"], "gt_depth", addon["tmp"])
    s.existing_kind = "metric"
    s.mode = "PERSPECTIVE"
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    obj = bpy.context.active_object
    o = obj.depth_scene
    assert o.is_depth_scene and o.mapping == "METRIC"
    o.live_update = False
    o.fov = FOV
    o.remove_stretched = False
    o.resolution = W
    o.live_update = True
    bpy.ops.depth_scene.rebuild()

    # Rig maps camera space to world: forward = +Y, up = +Z.
    v = _world_verts(obj)
    x, fwd, up = v[:, 0], v[:, 1], v[:, 2]
    walls = np.abs(np.abs(x) - 1.0)
    floor_ceiling = np.abs(np.abs(up) - 1.2)
    end = np.abs(fwd - 12.0)
    pillar = np.abs(fwd - 3.2)
    dist = np.min(np.stack([walls, floor_ceiling, end, pillar]), axis=0)
    assert np.median(dist) < 1e-3
    assert np.percentile(dist, 90) < 0.02

    proj = o.projector
    assert proj.data.angle == pytest.approx(FOV, rel=1e-5)
    assert bpy.context.scene.camera == o.shot_camera
    assert bpy.context.scene.render.resolution_x == W
    assert bpy.context.scene.view_settings.view_transform == "Standard"


def test_live_update_and_stretch_removal(addon):
    obj = bpy.context.active_object
    o = obj.depth_scene
    n_before = len(obj.data.polygons)
    o.resolution = 128                      # live update triggers rebuild
    assert len(obj.data.polygons) < n_before
    o.remove_stretched = True
    assert o.stat_removed > 50              # pillar silhouettes are cut
    assert o.stat_removed < 0.05 * (127 * 95)
    assert len(obj.data.uv_layers) == 1


def test_inverse_mapping_from_disparity_matches_metric(addon):
    s = bpy.context.scene.depth_scene
    disparity = (3.0 / addon["depth"] + 0.25).astype(np.float32)
    s.existing_depth = _depth_image_from(disparity, "gt_disparity", addon["tmp"])
    s.existing_kind = "disparity"
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    obj = bpy.context.active_object
    o = obj.depth_scene
    assert o.mapping == "INVERSE"
    o.live_update = False
    o.fov, o.near, o.far = FOV, float(addon["depth"].min()), float(addon["depth"].max())
    o.remove_stretched = False
    o.live_update = True
    bpy.ops.depth_scene.rebuild()
    v = _world_verts(obj)
    end = np.abs(v[:, 1] - 12.0) < 0.05
    assert end.sum() > 20


def test_ai_generation_with_depth_anything_signature(addon):
    from depth_scene import paths
    from depth_scene.ai import models
    from fake_models import make_depth_anything_like

    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    make_depth_anything_like(paths.model_path(models.get("da2_small"), "fp32"))
    s = bpy.context.scene.depth_scene
    s.use_existing_depth = False
    s.model = "da2_small"
    s.precision = "fp32"
    s.device = "CPU"
    s.input_size = 266
    assert bpy.ops.depth_scene.generate_depth() == {"FINISHED"}
    assert s.existing_depth is not None and s.existing_kind == "disparity"
    assert os.path.exists(bpy.path.abspath(s.existing_depth.filepath_raw))
    # Build reuses the depth generated above (same signature).
    name = s.existing_depth.name
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    assert bpy.context.active_object.depth_scene.depth_image.name == name


def test_ai_generation_with_moge_signature_sets_fov(addon):
    from depth_scene import paths
    from depth_scene.ai import models
    from fake_models import make_moge_like

    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    fov = math.radians(72.0)
    w, h = 96, 72
    from synthetic import corridor_depth
    depth, _ = corridor_depth(w, h, fov)
    gt = camera_points(depth, fov)
    cv = np.stack([gt[..., 0], -gt[..., 1], -gt[..., 2]], axis=-1) / 2.0
    cv[..., 2] -= 0.5
    make_moge_like(paths.model_path(models.get("moge2_vits"), "fp32"), cv, np.ones((h, w)), 2.0)
    s = bpy.context.scene.depth_scene
    s.model = "moge2_vits"
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    o = bpy.context.active_object.depth_scene
    assert o.depth_kind == "metric" and o.mapping == "METRIC"
    assert math.degrees(o.estimated_fov) == pytest.approx(72.0, abs=0.5)
    assert math.degrees(o.projector.data.angle) == pytest.approx(72.0, abs=0.5)


def test_relief_mode(addon):
    s = bpy.context.scene.depth_scene
    s.use_existing_depth = True
    s.existing_depth = bpy.data.images["gt_disparity"]
    s.existing_kind = "disparity"
    s.mode = "RELIEF"
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    obj = bpy.context.active_object
    o = obj.depth_scene
    co = np.empty(len(obj.data.vertices) * 3, np.float32)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    assert co[:, 2].max() == pytest.approx(o.base_displacement, rel=1e-3)
    assert o.projector.data.type == "ORTHO"
    n_before = len(obj.data.vertices)
    o.base_displacement = 1.0
    assert len(obj.data.vertices) == n_before
    co = np.empty(len(obj.data.vertices) * 3, np.float32)
    obj.data.vertices.foreach_get("co", co)
    assert co.reshape(-1, 3)[:, 2].max() == pytest.approx(1.0, rel=1e-3)
    o.shading = "PRINCIPLED"
    o.bump = 0.4
    nodes = {n.bl_idname for n in o.material.node_tree.nodes}
    assert {"ShaderNodeBsdfPrincipled", "ShaderNodeBump"} <= nodes
    o.detail_strength = 0.05                  # luminance detail path
    assert len(obj.data.vertices) > 0


def test_camera_move_keyframes(addon):
    obj = bpy.context.active_object
    o = obj.depth_scene
    o.mode = "PERSPECTIVE"
    bpy.ops.depth_scene.camera_move(move="DOLLY_IN", distance=0.5, frames=48)
    cam = o.shot_camera
    fc = cam.animation_data.action
    assert fc is not None
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    p0 = cam.matrix_world.translation.copy()
    scene.frame_set(scene.frame_start + 47)
    p1 = cam.matrix_world.translation.copy()
    # dolly in moves forward = world +Y for the default rig orientation
    assert (p1 - p0).y == pytest.approx(0.5, abs=1e-4)
    bpy.ops.depth_scene.reset_shot()
    assert cam.animation_data is None or cam.animation_data.action is None


def test_render_through_projector_reproduces_image(addon):
    """The decisive alignment check: projector render == source image."""
    for ob in list(bpy.data.objects):          # earlier builds would occlude the view
        bpy.data.objects.remove(ob)
    s = bpy.context.scene.depth_scene
    s.mode = "PERSPECTIVE"
    s.existing_depth = bpy.data.images["gt_depth"]
    s.existing_kind = "metric"
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    obj = bpy.context.active_object
    o = obj.depth_scene
    o.live_update = False
    o.fov = FOV
    o.resolution = 320
    o.live_update = True
    bpy.ops.depth_scene.rebuild()

    scene = bpy.context.scene
    scene.camera = o.projector
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.cycles.device = "CPU"
    scene.render.resolution_percentage = 50
    out = str(addon["tmp"] / "projector.png")
    scene.render.filepath = out
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    rendered = bpy.data.images.load(out)
    rw, rh = rendered.size
    buf = np.empty(rw * rh * rendered.channels, np.float32)
    rendered.pixels.foreach_get(buf)
    got = buf.reshape(rh, rw, rendered.channels)[::-1, :, :3]
    from depth_scene.core import imaging
    ref = imaging.resize(addon["rgb"], rh, rw)
    err = np.abs(got - ref).mean(axis=2)
    # holes from removed stretched faces show the (black) world: ignore them
    assert np.median(err) < 0.02
    assert np.mean(err < 0.06) > 0.9


class _Layout:
    """Fake UILayout that validates property names and operator ids."""

    def __init__(self):
        self.calls = 0

    def _sub(self, *a, **k):
        return _Layout()

    row = column = box = split = _sub

    def prop(self, data, name, **kw):
        assert name in data.bl_rna.properties, f"missing property {name!r}"
        self.calls += 1

    def template_ID(self, data, name, **kw):
        assert name in data.bl_rna.properties, f"missing property {name!r}"

    def operator(self, idname, **kw):
        mod, op = idname.split(".")
        getattr(getattr(bpy.ops, mod), op).get_rna_type()   # raises if unknown
        return type("OpProps", (), {})()

    def label(self, **kw):
        pass

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)


def test_panels_draw_without_errors(addon):
    from depth_scene import preferences, ui

    obj = bpy.context.active_object
    assert obj.depth_scene.is_depth_scene
    for mode in ("PERSPECTIVE", "RELIEF"):
        obj.depth_scene.live_update = False
        obj.depth_scene.mode = mode
        for cls in ui.classes:
            fake = type("P", (), {"layout": _Layout()})()
            if hasattr(cls, "poll"):
                assert cls.poll(bpy.context)
            cls.draw(fake, bpy.context)
    for use_existing in (False, True):
        bpy.context.scene.depth_scene.use_existing_depth = use_existing
        ui.DEPTHSCENE_PT_main.draw(type("P", (), {"layout": _Layout()})(), bpy.context)
    prefs = bpy.context.preferences.addons["depth_scene"].preferences
    preferences.DepthScenePreferences.draw(type("P", (), {
        "layout": _Layout(), "ort_variant": prefs.ort_variant, "models_dir": prefs.models_dir,
        "bl_rna": prefs.bl_rna})(), bpy.context)


def test_download_class_with_file_url(tmp_path):
    from depth_scene.ai import download

    src = tmp_path / "src.bin"
    src.write_bytes(os.urandom(3 * 1024 * 1024 + 17))
    dest = tmp_path / "out" / "model.onnx"
    job = download.Download(src.as_uri(), str(dest))
    job.run()
    assert job.error is None and job.done and job.progress == 1.0
    assert dest.read_bytes() == src.read_bytes()
    bad = download.Download((tmp_path / "missing.bin").as_uri(), str(tmp_path / "x.onnx"))
    bad.run()
    assert bad.error and not os.path.exists(tmp_path / "x.onnx")
