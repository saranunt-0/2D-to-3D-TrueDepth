"""Render the README demo images from the synthetic corridor.

    python tools/render_demo.py OUTPUT_DIR          (needs `pip install bpy`)

Simulates a Depth Anything-style output (affine inverse depth at 518 px,
slightly blurred) from the exact corridor depth, builds the scene with the
add-on's operators, then renders: projector view, a moved shot camera, an
overview of the geometry and a dolly-in frame sequence.
"""

import math
import os
import sys

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

import addon_utils  # noqa: E402

from synthetic import corridor_image  # noqa: E402

OUT = os.path.abspath(sys.argv[-1] if len(sys.argv) > 1 and not sys.argv[-1].endswith(".py") else "demo_out")
W, H = 960, 720
FOV = math.radians(65.0)


def save_rgb(rgb, path):
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(os.path.basename(path), w, h, alpha=True)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = rgb
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return img


def render(scene, camera, path, pct=50, samples=16):
    scene.camera = camera
    scene.render.resolution_percentage = pct
    scene.cycles.samples = samples
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("depth_scene", default_set=True)
    from depth_scene import bimage
    from depth_scene.core import imaging

    rgb, depth = corridor_image(W, H, FOV)
    image = save_rgb(rgb, os.path.join(OUT, "demo_source.png"))

    # "Model output": affine inverse depth at 518 px, softened edges.
    disparity = 2.0 / depth + 0.05
    lo = imaging.gaussian_blur(imaging.resize(disparity, 518, 690), 0.8)
    depth_img = bimage.write_depth("demo_disparity", lo, None, OUT)

    scene = bpy.context.scene
    s = scene.depth_scene
    s.image = image
    s.use_existing_depth = True
    s.existing_depth = depth_img
    s.existing_kind = "disparity"
    s.mode = "PERSPECTIVE"
    assert bpy.ops.depth_scene.build() == {"FINISHED"}
    obj = bpy.context.active_object
    o = obj.depth_scene
    o.live_update = False
    o.fov = FOV
    o.near, o.far = float(depth.min()), float(depth.max())
    o.resolution = 512
    o.live_update = True
    bpy.ops.depth_scene.rebuild()
    print(f"mesh: {o.stat_faces} faces, {o.stat_removed} stretched faces removed")

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    world = bpy.data.worlds.new("demo") if scene.world is None else scene.world
    scene.world = world
    world.color = (0.02, 0.02, 0.025)

    render(scene, o.projector, os.path.join(OUT, "demo_projector.png"))

    shot = o.shot_camera
    shot.location = (0.25, 0.1, -1.6)          # rig space: right, up, forward
    shot.rotation_euler = (0.0, math.radians(8), 0.0)
    render(scene, shot, os.path.join(OUT, "demo_shot.png"))

    over = bpy.data.objects.new("Overview", bpy.data.cameras.new("Overview"))
    scene.collection.objects.link(over)
    over.data.angle = math.radians(60)
    over.data.clip_end = 200
    over.location = (3.2, -4.2, 2.4)
    direction = np.array([-0.2, 5.0, -0.4]) - np.array(over.location)
    from mathutils import Vector
    over.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    render(scene, over, os.path.join(OUT, "demo_overview.png"))

    # Dolly-in frames for the GIF (shot camera pushes 2 m down the corridor).
    bpy.ops.depth_scene.reset_shot()
    frames = os.path.join(OUT, "frames")
    os.makedirs(frames, exist_ok=True)
    n = 24
    for i in range(n):
        f = i / (n - 1)
        ease = 0.5 - 0.5 * math.cos(math.pi * f)
        shot.location = (0.0, 0.0, -2.0 * ease)
        shot.rotation_euler = (0.0, math.radians(6) * math.sin(math.pi * f), 0.0)
        render(scene, shot, os.path.join(frames, f"f{i:03d}.png"), pct=40, samples=8)
    print("wrote", OUT)


main()
