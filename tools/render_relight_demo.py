"""Render the README relighting demo (docs/images/demo_relight.png).

    python tools/render_relight_demo.py OUTPUT_DIR      (needs `pip install bpy`)

Builds the synthetic corridor from its exact depth, assigns its ground-truth
camera-space normal map the way the Normals pass would, then renders three
panels: unlit (Emission), Principled lit by a warm spot light, and a cool
light further down the corridor with Original Lighting 0.3.
"""

import math
import os
import sys

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

import addon_utils  # noqa: E402

from synthetic import corridor_depth, corridor_image, corridor_normals  # noqa: E402

OUT = os.path.abspath(sys.argv[-1] if len(sys.argv) > 1 and not sys.argv[-1].endswith(".py") else "demo_out")
W, H = 480, 360
FOV = math.radians(65.0)


def render(scene, name):
    path = os.path.join(OUT, name)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path, check_existing=False)
    buf = np.empty(W * H * img.channels, np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(H, W, img.channels)[..., :3]


def main():
    os.makedirs(OUT, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("depth_scene", default_set=True, handle_error=None)
    from depth_scene import bimage
    from depth_scene.core import passes

    scene = bpy.context.scene
    s = scene.depth_scene
    s.save_location = OUT
    rgb, depth = corridor_image(W, H, FOV)
    img = bpy.data.images.new("corridor.png", W, H, alpha=True)
    rgba = np.ones((H, W, 4), np.float32)
    rgba[..., :3] = rgb
    img.pixels.foreach_set(rgba[::-1].ravel())
    s.image = img
    s.use_existing_depth = True
    s.existing_depth = bimage.write_depth("gt_depth", depth, None, OUT)
    s.existing_kind = "metric"
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    obj = bpy.context.active_object
    o = obj.depth_scene
    o.live_update = False
    o.fov, o.resolution = FOV, 480
    o.live_update = True
    bpy.ops.depth_scene.rebuild()
    _, plane = corridor_depth(W, H, FOV, pillar=(-0.9, -0.45, 3.2))
    o.normal_map = bimage.write_map("gt_normal", passes.normal_to_color(corridor_normals(plane)), OUT, True)

    scene.camera = o.projector
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.device = "CPU"
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("Ambient")
    scene.world.color = (0.02, 0.02, 0.025)

    panels = [render(scene, "relight_a_emission.png")]
    o.shading = "PRINCIPLED"
    o.roughness = 0.6
    bpy.ops.depth_scene.add_light(kind="SPOT")
    light = [ob for ob in bpy.data.objects if ob.type == "LIGHT"][-1]
    light.location = (0.55, 0.9, -2.0)          # rig = camera space: right, high, in front of the pillar
    light.rotation_euler = (math.radians(-10), math.radians(35), 0)
    light.data.energy = 300
    light.data.color = (1.0, 0.85, 0.65)
    light.data.spot_size = math.radians(110)
    panels.append(render(scene, "relight_b_spot.png"))
    light.location = (-0.6, 0.9, -5.5)
    light.rotation_euler = (math.radians(-10), math.radians(-30), 0)
    light.data.color = (0.6, 0.75, 1.0)
    o.original_light = 0.3
    panels.append(render(scene, "relight_c_moved.png"))

    gap = np.ones((H, 8, 3), np.float32)
    strip = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)
    h, w = strip.shape[:2]
    out = bpy.data.images.new("demo_relight", w, h)
    px = np.ones((h, w, 4), np.float32)
    px[..., :3] = strip
    out.pixels.foreach_set(px.ravel())
    out.filepath_raw = os.path.join(OUT, "demo_relight.png")
    out.file_format = "PNG"
    out.save()
    print("saved", out.filepath_raw)


if __name__ == "__main__":
    main()
