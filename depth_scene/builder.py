"""Create and live-update the 3D scene: rig, mesh, cameras, material."""

import math
import os

import bpy
import numpy as np
from mathutils import Vector

from . import bimage, paths
from .ai import infer, models
from .core import depth as depth_mod
from .core import geometry, imaging

RIG_ROTATION = (math.pi / 2, 0.0, 0.0)   # camera space (-Z forward) -> world (+Y forward, +Z up)


# ----------------------------------------------------------------------------- depth

def depth_signature(s, image):
    return f"{image.name_full}|{s.model}|{s.custom_model_path}|{s.precision}|{s.input_size}|{s.input_blur}"


def generate_depth(scene_settings, image):
    """Run the selected model on ``image``; returns (depth Image, kind, fov or None, result)."""
    s = scene_settings
    if s.model == "CUSTOM":
        model_file = bpy.path.abspath(s.custom_model_path)
        info = models.ModelInfo("custom", "Custom", "depth_anything_v2", "", "", "", True, {}, "disparity", "")
    else:
        info = models.get(s.model)
        model_file = paths.model_path(info, s.precision)
    if not model_file or not os.path.exists(model_file):
        hint = info.manual or "download it in the add-on preferences"
        raise RuntimeError(f"Model file not found: {model_file or '(none)'} - {hint}")

    rgb = bimage.display_rgb(image)
    result = infer.estimate(info, model_file, rgb, device=s.device, input_size=s.input_size,
                            input_blur=s.input_blur)
    folder = bimage.output_dir(s.save_location)
    name = f"{os.path.splitext(image.name)[0]}_depth"
    depth_img = bimage.write_depth(name, result.depth, result.valid, folder)
    depth_img["depth_scene_kind"] = result.kind
    depth_img["depth_scene_source"] = depth_signature(s, image)
    if result.fov:
        depth_img["depth_scene_fov"] = float(result.fov)
    return depth_img, result.kind, result.fov, result


def suggest_near_far(depth, kind, valid):
    """Initial near/far. For disparity, assume zero shift: far/near = d_max/d_min."""
    sel = np.isfinite(depth) if valid is None else (valid & np.isfinite(depth))
    vals = depth[sel] if sel.any() else depth.reshape(-1)
    if kind == "metric":
        return float(vals.min()), float(vals.max())
    if kind == "disparity":
        hi = float(np.percentile(vals, 99.5))
        lo = max(float(np.percentile(vals, 0.5)), hi / 100.0, 1e-6)
        ratio = float(np.clip(hi / lo, 1.5, 100.0))
        return 1.0, ratio
    if kind == "layers":
        return 1.0, 2.0       # shallow stack of layers; widen Far to separate them
    return 1.0, 10.0


# ----------------------------------------------------------------------------- mesh

def _params(o):
    return geometry.MeshParams(
        mode=o.mode, resolution=o.resolution, fov=o.fov, mapping=o.mapping, near=o.near,
        far=max(o.far, o.near + 1e-3), metric_scale=o.metric_scale, remove_stretched=o.remove_stretched,
        stretch_angle=o.stretch_angle, size=o.size, base_displacement=o.base_displacement,
        invert=o.invert, smooth=o.smooth, detail_strength=o.detail_strength,
        detail_radius=o.detail_radius, use_alpha=o.use_alpha, remove_masked=o.remove_masked,
        layer_steps=o.layer_steps,
    )


def write_mesh(mesh, verts, faces, uv):
    mesh.clear_geometry()
    n_loops = faces.size
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.loops.add(n_loops)
    mesh.loops.foreach_set("vertex_index", faces.ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, n_loops, 4, dtype=np.int32))
    uv_layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", uv[faces.ravel()].ravel())
    mesh.update(calc_edges=True)
    if hasattr(mesh, "shade_smooth"):
        mesh.shade_smooth()


def rebuild(obj, quiet=False):
    """Recompute the mesh of a Depth Scene object from its settings."""
    o = obj.depth_scene
    if o.image is None or o.depth_image is None:
        if quiet:
            return None
        raise RuntimeError("Object has no image or depth map")
    depth, valid = bimage.depth_arrays(o.depth_image)
    w, h = o.image.size
    gray = alpha = None
    if o.detail_strength != 0.0:
        gray = imaging.luminance(bimage.display_rgb(o.image))
    if o.use_alpha and bimage.has_alpha(o.image):
        alpha = bimage.to_array(o.image)[..., 3]
    params = _params(o)
    if params.mode == "PERSPECTIVE" and params.mapping == "METRIC" and o.depth_kind != "metric":
        params.mapping = "INVERSE"
    verts, faces, uv, info = geometry.build(params, depth, o.depth_kind, (w, h), gray=gray, alpha=alpha, valid=valid)
    write_mesh(obj.data, verts, faces, uv)
    o.stat_faces = info["faces"]
    o.stat_removed = info["stretched_removed"]
    sync_cameras(obj, lens_changed=False, z_range=info.get("z_range"))
    return info


# ----------------------------------------------------------------------------- material

def update_material(obj):
    o = obj.depth_scene
    mat = o.material
    if mat is None:
        mat = bpy.data.materials.new(f"{obj.name} Material")
        o.material = mat
        obj.data.materials.append(mat)
    if bpy.app.version < (5, 0, 0):      # always on (and deprecated) since 5.0
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.location = (-400, 0)
    tex.image = o.image
    tex.extension = "EXTEND"
    tex.interpolation = "Cubic"
    if o.shading == "EMISSION":
        shader = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(tex.outputs["Color"], shader.inputs["Color"])
        color_out = shader.outputs["Emission"]
    else:
        shader = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(tex.outputs["Color"], shader.inputs["Base Color"])
        shader.inputs["Roughness"].default_value = 0.8
        if o.bump > 0:
            bw = nt.nodes.new("ShaderNodeRGBToBW")
            bw.location = (-150, -250)
            bump = nt.nodes.new("ShaderNodeBump")
            bump.location = (50, -250)
            bump.inputs["Strength"].default_value = o.bump
            nt.links.new(tex.outputs["Color"], bw.inputs["Color"])
            nt.links.new(bw.outputs["Val"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
        color_out = shader.outputs["BSDF"]
    shader.location = (150, 0)
    if o.use_alpha and o.image is not None and bimage.has_alpha(o.image):
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        transp.location, mix.location = (150, 200), (400, 0)
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
        nt.links.new(color_out, mix.inputs[2])
        color_out = mix.outputs["Shader"]
    nt.links.new(color_out, out.inputs["Surface"])
    return mat


# ----------------------------------------------------------------------------- cameras

def _camera_object(name, rig, collection):
    cam = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, cam)
    collection.objects.link(obj)
    obj.parent = rig
    return obj


def sync_cameras(obj, lens_changed=False, z_range=None):
    """Keep the projector camera matched to the reconstruction settings."""
    o = obj.depth_scene
    cams = [c for c in (o.projector, o.shot_camera) if c is not None and c.type == "CAMERA"]
    if not cams:
        return
    far_clip = None
    if z_range is not None:
        far_clip = max(z_range[1] * 2.0, 10.0)
    for cam in cams:
        is_projector = cam == o.projector
        data = cam.data
        data.sensor_fit = "AUTO"
        if o.mode == "RELIEF":
            data.type = "ORTHO"
            data.ortho_scale = o.size        # AUTO sensor fit: longer side
            if is_projector:
                cam.location = (0.0, 0.0, o.size * 1.5)
            data.clip_end = max(o.size * 10, 10.0)
        else:
            data.type = "PERSP"
            if is_projector or lens_changed:
                data.angle = o.fov
            if is_projector:
                cam.location = (0.0, 0.0, 0.0)
            if far_clip:
                data.clip_end = max(data.clip_end, far_clip)
            data.clip_start = 0.01


def reset_shot_camera(obj):
    o = obj.depth_scene
    if o.shot_camera is None or o.projector is None:
        return
    o.shot_camera.animation_data_clear()
    o.shot_camera.matrix_basis = o.projector.matrix_basis.copy()
    o.shot_camera.data.angle = o.projector.data.angle
    o.shot_camera.data.type = o.projector.data.type
    o.shot_camera.data.ortho_scale = o.projector.data.ortho_scale


def add_camera_move(obj, move, distance, frame_start, frame_end):
    """Keyframe a simple parallax move on the shot camera (in rig space)."""
    o = obj.depth_scene
    cam = o.shot_camera
    if cam is None:
        raise RuntimeError("No shot camera")
    reset_shot_camera(obj)
    start = cam.location.copy()
    rot = cam.matrix_basis.to_3x3()
    local = {
        "DOLLY_IN": Vector((0, 0, -distance)),
        "DOLLY_OUT": Vector((0, 0, distance)),
        "TRUCK_LEFT": Vector((-distance, 0, 0)),
        "TRUCK_RIGHT": Vector((distance, 0, 0)),
        "PEDESTAL_UP": Vector((0, distance, 0)),
    }[move]
    cam.location = start
    cam.keyframe_insert("location", frame=frame_start)
    cam.location = start + rot @ local
    cam.keyframe_insert("location", frame=frame_end)
    cam.location = start


# ----------------------------------------------------------------------------- scene

def build_scene(context, image, depth_img, kind, mode, est_fov=None):
    """Create rig + mesh + cameras + material for ``image``/``depth_img``."""
    scene = context.scene
    s = scene.depth_scene
    base = os.path.splitext(image.name)[0]

    collection = bpy.data.collections.new(f"Depth Scene - {base}")
    scene.collection.children.link(collection)

    rig = bpy.data.objects.new(f"{base} Rig", None)
    rig.empty_display_type = "PLAIN_AXES"
    rig.rotation_euler = RIG_ROTATION
    rig.location = scene.cursor.location      # repeated builds don't pile up at the origin
    collection.objects.link(rig)

    mesh = bpy.data.meshes.new(f"{base} Mesh")
    obj = bpy.data.objects.new(f"{base} 3D", mesh)
    collection.objects.link(obj)
    obj.parent = rig

    projector = _camera_object(f"{base} Projector", rig, collection)
    projector.hide_select = True
    projector.lock_location = projector.lock_rotation = (True, True, True)
    shot = _camera_object(f"{base} Shot Camera", rig, collection)

    o = obj.depth_scene
    o.live_update = False                 # batch-assign without rebuilding each time
    o.is_depth_scene = True
    o.mode = mode
    o.image = image
    o.depth_image = depth_img
    o.depth_kind = kind
    o.projector = projector
    o.shot_camera = shot
    if est_fov:
        o.estimated_fov = est_fov
        o.fov = est_fov
    depth, valid = bimage.depth_arrays(depth_img)
    near, far = suggest_near_far(depth, kind, valid)
    o.near, o.far = near, far
    o.mapping = depth_mod.default_mapping(kind)
    o.shading = "EMISSION" if mode == "PERSPECTIVE" else "PRINCIPLED"
    o.use_alpha = bimage.has_alpha(image)
    if mode == "RELIEF":
        o.base_displacement = o.size * 0.15

    update_material(obj)
    rebuild(obj)
    sync_cameras(obj, lens_changed=True)
    reset_shot_camera(obj)
    o.live_update = True

    if s.set_render_size:
        scene.render.resolution_x, scene.render.resolution_y = image.size
        scene.render.resolution_percentage = 100
    if s.set_standard_view:
        scene.view_settings.view_transform = "Standard"
    scene.camera = shot

    for ob in context.view_layer.objects.selected:
        ob.select_set(False)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj
