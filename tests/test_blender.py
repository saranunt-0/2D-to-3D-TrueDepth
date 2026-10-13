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

from synthetic import corridor_image  # noqa: E402

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


def test_marigold_generation_and_linear_build(addon):
    from depth_scene import paths
    from depth_scene.ai import models
    from fake_models import make_marigold_like

    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    info = models.get(models.DEFAULT_MODEL)
    make_marigold_like(paths.model_path(info, "fp32"))
    s = bpy.context.scene.depth_scene
    s.use_existing_depth = False
    s.model = info.key
    s.precision = "fp32"
    s.device = "CPU"
    s.input_size = 128
    s.denoise_steps = 2
    s.ensemble_size = 2
    s.seed = 3
    assert bpy.ops.depth_scene.generate_depth() == {"FINISHED"}
    assert s.existing_depth is not None and s.existing_kind == "depth"
    assert tuple(s.existing_depth.size) == (128, 96)            # 320x240 -> longest edge 128
    assert os.path.exists(bpy.path.abspath(s.existing_depth.filepath_raw))
    # Build reuses the depth generated above (same signature) and maps it LINEAR (0 = near).
    name = s.existing_depth.name
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    o = bpy.context.active_object.depth_scene
    assert o.depth_image.name == name
    assert o.depth_kind == "depth" and o.mapping == "LINEAR"
    # Changing a generation setting invalidates the reuse signature.
    s.seed = 4
    from depth_scene import builder
    assert s.existing_depth.get("depth_scene_source") != builder.depth_signature(s, s.image)


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


def _render(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * img.channels, np.float32)
    img.pixels.foreach_get(buf)
    out = buf.reshape(h, w, img.channels)[::-1, :, :3].copy()
    bpy.data.images.remove(img)
    return out


def test_camera_space_normal_map_matches_geometry_under_a_light(addon):
    """Decisive convention check: a ground-truth camera-space normal map, applied as the add-on
    wires it (Object space), shades like the true geometry; flipping X does not."""
    from depth_scene import bimage
    from depth_scene.core import passes
    from synthetic import corridor_depth, corridor_normals

    obj = bpy.context.active_object            # projector-alignment scene from the previous test
    o = obj.depth_scene
    _, plane = corridor_depth(W, H, FOV, pillar=(-0.9, -0.45, 3.2))
    gt = bimage.write_map("gt_normal", passes.normal_to_color(corridor_normals(plane)), str(addon["tmp"]), True)
    o.live_update = False
    o.remove_stretched = True
    o.live_update = True
    o.shading = "PRINCIPLED"
    o.use_albedo = False
    assert bpy.ops.depth_scene.add_light(kind="POINT") == {"FINISHED"}
    light = [ob for ob in bpy.data.objects if ob.type == "LIGHT"][-1]
    assert light.parent == obj.parent and light.location.z < 0          # in front of the projector
    assert light.data.energy > 0

    scene = bpy.context.scene
    scene.camera = o.projector
    scene.cycles.samples = 16
    scene.cycles.seed = 0
    scene.render.resolution_percentage = 50
    o.use_normal_map = False
    mesh = _render(scene, str(addon["tmp"] / "lit_mesh.png"))
    o.normal_map = gt
    o.use_normal_map = True
    nodes = {n.bl_idname: n for n in o.material.node_tree.nodes}
    assert nodes["ShaderNodeNormalMap"].space == "OBJECT"
    mapped = _render(scene, str(addon["tmp"] / "lit_normal_map.png"))
    o.normal_flip_x = True
    flipped = _render(scene, str(addon["tmp"] / "lit_flipped.png"))
    o.normal_flip_x = False
    lit = mesh.mean(axis=2) > 0.02                                      # ignore holes / unlit pixels
    good = np.abs(mapped - mesh).mean(axis=2)[lit].mean()
    bad = np.abs(flipped - mesh).mean(axis=2)[lit].mean()
    assert lit.mean() > 0.5
    print(f"normal map vs mesh shading: {good:.4f}; flipped X: {bad:.4f}")
    assert good < 0.35 * bad, (good, bad)
    for ob in [ob for ob in bpy.data.objects if ob.type == "LIGHT"]:
        bpy.data.objects.remove(ob)


def test_material_passes_generate_wire_and_toggle(addon):
    from depth_scene import builder, paths
    from depth_scene.ai import models
    from fake_models import make_marigold_like

    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    s = bpy.context.scene.depth_scene
    for group in builder.PASS_GROUPS:
        info = models.get(models.PASS_MODELS[group])
        make_marigold_like(paths.model_path(info, "fp32"), task=info.export_task)
    s.precision, s.device, s.input_size, s.denoise_steps, s.ensemble_size, s.seed = "fp32", "CPU", 96, 1, 1, 0
    s.pass_normals = s.pass_appearance = s.pass_lighting = True
    s.albedo_from = "APPEARANCE"
    obj = bpy.context.active_object
    o = obj.depth_scene
    o.shading = "EMISSION"
    o.use_albedo = True                                                 # previous test switched it off
    assert bpy.ops.depth_scene.generate_passes() == {"FINISHED"}
    assert o.shading == "PRINCIPLED"
    for prop in ("normal_map", "albedo_map", "roughness_map", "metallic_map", "shading_map", "residual_map"):
        img = getattr(o, prop)
        assert img is not None, prop
        assert os.path.exists(bpy.path.abspath(img.filepath_raw))
        assert tuple(img.size) == (96, 72)                                # processing resolution, image aspect
    assert o.albedo_map["depth_scene_pass_group"] == "appearance"
    assert o.normal_map.colorspace_settings.name == "Non-Color"
    assert o.albedo_map.colorspace_settings.name != "Non-Color"

    nt = o.material.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")

    def src(socket_name):
        links = bsdf.inputs[socket_name].links
        return links[0].from_node.bl_idname if links else None

    assert src("Base Color") == "ShaderNodeMix"                         # image -> albedo mix
    assert src("Roughness") == "ShaderNodeMapRange"
    assert src("Metallic") == "ShaderNodeMath"
    assert src("Normal") == "ShaderNodeNormalMap"
    assert src("Emission Color") is None                               # highlights off by default

    # Toggles and parameters rewire / update the tree live.
    o.use_residual = True
    o.residual_strength = 2.0
    o.use_shading = True
    o.bump = 0.3
    o.original_light = 0.25
    o.normal_space = "TANGENT"
    nt = o.material.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    assert src("Emission Color") == "ShaderNodeTexImage"
    assert bsdf.inputs["Emission Strength"].default_value == pytest.approx(2.0)
    assert src("Normal") == "ShaderNodeBump"                            # normal map -> bump -> BSDF
    mixes = [n for n in nt.nodes if n.bl_idname == "ShaderNodeMix"]
    assert {m.blend_type for m in mixes} == {"MIX", "MULTIPLY"}
    assert any(n.bl_idname == "ShaderNodeMixShader" for n in nt.nodes)  # original lighting blend
    assert next(n for n in nt.nodes if n.bl_idname == "ShaderNodeNormalMap").space == "TANGENT"
    o.use_normal_map = o.use_roughness_map = o.use_metallic_map = o.use_albedo = False
    o.roughness, o.metallic = 0.3, 0.6
    nt = o.material.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    assert not any(n.bl_idname == "ShaderNodeNormalMap" for n in nt.nodes)
    assert bsdf.inputs["Roughness"].default_value == pytest.approx(0.3)
    assert bsdf.inputs["Metallic"].default_value == pytest.approx(0.6)
    o.cast_shadows = False
    assert obj.visible_shadow is False
    o.cast_shadows = True

    # Same image + settings: maps are reused (not recomputed); albedo source follows the preference.
    token = o.normal_map["depth_scene_token"]
    s.albedo_from = "LIGHTING"
    assert bpy.ops.depth_scene.generate_passes() == {"FINISHED"}
    assert o.normal_map["depth_scene_token"] == token
    assert o.albedo_map["depth_scene_pass_group"] == "lighting"

    # A missing model folder only skips that pass.
    import shutil
    shutil.rmtree(paths.model_path(models.get(models.PASS_MODELS["normals"]), "fp32"))
    s.seed = 5                                                          # force regeneration
    assert bpy.ops.depth_scene.generate_passes() == {"FINISHED"}
    assert o.roughness_map["depth_scene_source"].endswith("|5")
    assert not o.normal_map["depth_scene_source"].endswith("|5")
    s.pass_normals = s.pass_appearance = s.pass_lighting = False


def test_marigold_v2_backend_end_to_end(addon, monkeypatch):
    """V2 backend with a fake marigold-v2 install: Build runs depth + passes out of process, then builds."""
    from fake_marigold_v2 import make_fake_repo

    s = bpy.context.scene.depth_scene
    prefs = bpy.context.preferences.addons["depth_scene"].preferences
    repo = make_fake_repo(str(addon["tmp"] / "v2"))
    prefs.v2_python, prefs.v2_repo, prefs.v2_assets = "", repo, ""
    s.backend = "V2"
    s.use_existing_depth = False
    s.image = addon["image"]
    with pytest.raises(RuntimeError, match="Marigold V2 is not set up: Python"):   # clear error
        bpy.ops.depth_scene.generate_depth()
    prefs.v2_python = sys.executable
    s.v2_checkpoint, s.v2_resolution, s.v2_seed = "Log-stage2", 160, 3
    s.pass_normals = s.pass_albedo_v2 = True
    s.pass_appearance = s.pass_lighting = False
    s.albedo_from = "V2"
    s.mode = "PERSPECTIVE"
    assert bpy.ops.depth_scene.build() == {"FINISHED"}                    # blocking without a window
    obj = bpy.context.active_object
    o = obj.depth_scene
    assert o.is_depth_scene and o.depth_kind == "log_depth" and o.mapping == "LOG"
    assert tuple(o.depth_image.size) == (160, 128)                        # 320x240 -> 160 x 120 -> multiples of 16
    assert s.existing_depth == o.depth_image and s.existing_kind == "log_depth"
    calls = open(os.path.join(bpy.path.abspath(s.save_location), "marigold_v2", "corridor", "out_depth",
                              "calls.txt")).read()
    assert "--width 160 --height 128" in calls and "--seed 3" in calls and "Log-stage2" in calls

    # Passes imported in the add-on's conventions and wired into a lit material.
    from depth_scene import bimage
    assert o.shading == "PRINCIPLED"
    alb = bimage.to_array(o.albedo_map)
    assert np.allclose(alb[5, 5, :3], 0.5, atol=1e-4)                      # gamma 2.2 undone -> linear
    nrm = bimage.to_array(o.normal_map)
    assert np.allclose(nrm[5, 5, :3], [0.8, 0.5, 0.9], atol=1e-4)           # (0.6, 0, 0.8) encoded
    assert o.albedo_map["depth_scene_pass_group"] == "v2"

    # Log depth -> geometry: the fake depth grows left to right, so the right edge is farther.
    co = np.empty(len(obj.data.vertices) * 3, np.float32)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    left = co[co[:, 0] < np.percentile(co[:, 0], 5)]
    right = co[co[:, 0] > np.percentile(co[:, 0], 95)]
    assert -right[:, 2].mean() > -left[:, 2].mean()

    # Same settings: nothing is re-run.
    n_calls = calls.count("\n")
    assert bpy.ops.depth_scene.generate_depth() == {"FINISHED"}
    calls = open(os.path.join(bpy.path.abspath(s.save_location), "marigold_v2", "corridor", "out_depth",
                              "calls.txt")).read()
    assert calls.count("\n") == n_calls

    # A failing run (e.g. CUDA out of memory) is reported, nothing is half-imported.
    monkeypatch.setenv("FAKE_MV2_FAIL", "normals")
    s.v2_seed = 4
    token = o.normal_map["depth_scene_token"]
    with pytest.raises(RuntimeError, match="normals failed .exit 1.*out of memory"):
        bpy.ops.depth_scene.generate_passes()
    assert o.normal_map["depth_scene_token"] == token
    log = open(os.path.join(bpy.path.abspath(s.save_location), "marigold_v2", "corridor", "marigold_v2.log")).read()
    assert "out of memory" in log
    s.backend = "V1"
    s.pass_normals = s.pass_albedo_v2 = False


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

    def separator(self, **kw):
        pass

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)


def test_panels_draw_without_errors(addon):
    from depth_scene import preferences, ui

    obj = bpy.context.active_object
    assert obj.depth_scene.is_depth_scene
    for mode, shading in (("PERSPECTIVE", "PRINCIPLED"), ("RELIEF", "EMISSION"), ("PERSPECTIVE", "EMISSION")):
        obj.depth_scene.live_update = False
        obj.depth_scene.mode = mode
        obj.depth_scene.shading = shading
        bpy.context.scene.depth_scene.pass_appearance = shading == "PRINCIPLED"
        bpy.context.scene.depth_scene.pass_lighting = shading == "PRINCIPLED"
        for cls in ui.classes:
            fake = type("P", (), {"layout": _Layout()})()
            if hasattr(cls, "poll"):
                assert cls.poll(bpy.context)
            cls.draw(fake, bpy.context)
    s = bpy.context.scene.depth_scene
    for backend in ("V1", "V2"):
        s.backend = backend
        s.pass_normals = s.pass_albedo_v2 = backend == "V2"
        for use_existing in (False, True):
            s.use_existing_depth = use_existing
            ui.DEPTHSCENE_PT_main.draw(type("P", (), {"layout": _Layout()})(), bpy.context)
    s.backend = "V1"
    s.pass_normals = s.pass_albedo_v2 = False
    prefs = bpy.context.preferences.addons["depth_scene"].preferences
    preferences.DepthScenePreferences.draw(type("P", (), {
        "layout": _Layout(), "ort_variant": prefs.ort_variant, "models_dir": prefs.models_dir,
        "v2_python": prefs.v2_python, "v2_repo": prefs.v2_repo, "v2_assets": prefs.v2_assets,
        "bl_rna": prefs.bl_rna})(), bpy.context)
