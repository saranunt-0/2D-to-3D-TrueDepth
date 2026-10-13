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

def depth_signature(s, image, model_key=None):
    """Identifies a generated map: reused while image + model settings are unchanged."""
    if model_key is None and s.backend == "V2":
        return v2_signature(s, image, "depth")
    return (f"{image.name_full}|{model_key or s.model}|{s.precision}|{s.input_size}|{s.denoise_steps}|"
            f"{s.ensemble_size}|{s.seed}")


def v2_signature(s, image, modality):
    ckpt = s.v2_checkpoint if modality == "depth" else modality
    return f"{image.name_full}|marigold-v2:{ckpt}|{s.v2_resolution}|{s.v2_seed}"


def generate_depth(scene_settings, image, progress=None):
    """Run Marigold on ``image``; returns (depth Image, kind, result)."""
    s = scene_settings
    info = models.get(s.model)
    model_dir = paths.model_path(info, s.precision)
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Model folder not found: {model_dir} - {info.manual}")

    rgb = bimage.display_rgb(image)
    result = infer.estimate(info, model_dir, rgb, device=s.device, processing_resolution=s.input_size,
                            steps=s.denoise_steps, ensemble_size=s.ensemble_size, seed=s.seed,
                            progress=progress)
    folder = bimage.output_dir(s.save_location)
    name = f"{os.path.splitext(image.name)[0]}_depth"
    depth_img = bimage.write_depth(name, result.depth, result.valid, folder)
    depth_img["depth_scene_kind"] = result.kind
    depth_img["depth_scene_source"] = depth_signature(s, image)
    return depth_img, result.kind, result


# ----------------------------------------------------------------------------- material passes

PASS_GROUPS = ("normals", "appearance", "lighting")
NON_COLOR_PASSES = {"normal", "roughness", "metallic", "shading"}


def enabled_pass_groups(s):
    """Material-pass groups run with the in-Blender (v1.1 ONNX) models.

    With the Marigold V2 backend the normals come from V2 (``enabled_v2_passes``);
    IID-Appearance / IID-Lighting have no V2 counterpart and still run as v1.1.
    """
    groups = [g for g in PASS_GROUPS if getattr(s, f"pass_{g}")]
    if s.backend == "V2":
        groups = [g for g in groups if g != "normals"]
    return groups


def enabled_v2_passes(s):
    """Marigold V2 modalities to run besides depth (V2 backend only)."""
    if s.backend != "V2":
        return []
    return [m for m, on in (("normals", s.pass_normals), ("albedo", s.pass_albedo_v2)) if on]


def pass_image_name(image, group, pass_name):
    base = os.path.splitext(image.name)[0]
    return f"{base}_{pass_name}_{group}" if pass_name == "albedo" else f"{base}_{pass_name}"


def generate_passes(scene_settings, image, groups=None, progress=None):
    """Run the normals / intrinsics models for ``image``.

    Returns ({group: {pass name: Image}}, messages, errors). Groups whose maps
    already exist with the same image + settings are reused, not recomputed;
    a failing group (e.g. model not converted yet) is reported in ``errors``
    and the others still run.
    """
    s = scene_settings
    groups = list(groups if groups is not None else enabled_pass_groups(s))
    results, messages, errors = {}, [], []
    rgb = None
    for gi, group in enumerate(groups):
        info = models.get(models.PASS_MODELS[group])
        sig = depth_signature(s, image, info.key)
        names = {p: pass_image_name(image, group, p) for p in info.outputs}
        imgs = {p: bpy.data.images.get(n) for p, n in names.items()}
        if all(i is not None and i.get("depth_scene_source") == sig for i in imgs.values()):
            results[group] = imgs
            messages.append(f"{info.label}: reused")
            continue
        try:
            model_dir = paths.model_path(info, s.precision)
            if not os.path.isdir(model_dir):
                raise RuntimeError(f"model folder {info.filename(s.precision)} not found - {info.manual}")
            if rgb is None:
                rgb = bimage.display_rgb(image)
            sub = None
            if progress:
                def sub(f, gi=gi):
                    progress((gi + f) / len(groups))
            res = infer.estimate_passes(info, model_dir, rgb, device=s.device, processing_resolution=s.input_size,
                                        steps=s.denoise_steps, ensemble_size=s.ensemble_size, seed=s.seed,
                                        progress=sub)
        except Exception as exc:          # keep going: other passes may still work
            errors.append(f"{info.label}: {exc}")
            continue
        folder = bimage.output_dir(s.save_location)
        imgs = {}
        for pass_name, arr in res.maps.items():
            img = bimage.write_map(names[pass_name], arr, folder, non_color=pass_name in NON_COLOR_PASSES)
            img["depth_scene_source"] = sig
            img["depth_scene_pass_group"] = group
            imgs[pass_name] = img
        results[group] = imgs
        extra = ""
        if res.info.get("fit_error") is not None:
            extra = f", image = albedo x shading + residual fits to {res.info['fit_error']:.0%} RMS"
        messages.append(f"{info.label}: {res.seconds:.1f}s on {res.provider}{extra}")
    return results, messages, errors


def assign_passes(obj, results, albedo_from="APPEARANCE"):
    """Point the object's material maps at freshly generated passes and switch to lit shading."""
    o = obj.depth_scene
    if not results:
        return
    if "normals" in results:
        o.normal_map = results["normals"]["normal"]
    if "appearance" in results:
        o.roughness_map = results["appearance"]["roughness"]
        o.metallic_map = results["appearance"]["metallic"]
    if "lighting" in results:
        o.shading_map = results["lighting"]["shading"]
        o.residual_map = results["lighting"]["residual"]
    albedos = {g: results[g]["albedo"] for g in ("appearance", "lighting", "v2") if "albedo" in results.get(g, {})}
    if "v2" in results and "normal" in results["v2"]:
        o.normal_map = results["v2"]["normal"]
    if albedos:
        pref = {"LIGHTING": "lighting", "V2": "v2"}.get(albedo_from, "appearance")
        current = o.albedo_map.get("depth_scene_pass_group") if o.albedo_map is not None else None
        if pref in albedos:
            o.albedo_map = albedos[pref]
        elif current != pref:
            o.albedo_map = next(iter(albedos.values()))
    o.shading = "PRINCIPLED"
    update_material(obj)


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
    return 1.0, 10.0


# ----------------------------------------------------------------------------- mesh

def _params(o):
    return geometry.MeshParams(
        mode=o.mode, resolution=o.resolution, fov=o.fov, mapping=o.mapping, near=o.near,
        far=max(o.far, o.near + 1e-3), metric_scale=o.metric_scale, remove_stretched=o.remove_stretched,
        stretch_angle=o.stretch_angle, size=o.size, base_displacement=o.base_displacement,
        invert=o.invert, smooth=o.smooth, detail_strength=o.detail_strength,
        detail_radius=o.detail_radius, use_alpha=o.use_alpha, layer_steps=o.layer_steps,
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

def _tex(nt, image, location, data=False):
    """Image Texture node sampling ``image`` through the mesh's image UVs."""
    node = nt.nodes.new("ShaderNodeTexImage")
    node.location = location
    node.image = image
    node.extension = "EXTEND"
    node.interpolation = "Cubic"
    if data and image is not None and image.colorspace_settings.name != "Non-Color":
        image.colorspace_settings.name = "Non-Color"      # normal / roughness / metallic / shading are data
    return node


def _mix_color(nt, blend, factor, a, b, location):
    """Mix node (RGBA). Socket indices are stable across Blender 4.2 - 5.x: 0 Factor, 6 A, 7 B, out 2."""
    node = nt.nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.blend_type = blend
    node.location = location
    node.inputs[0].default_value = factor
    nt.links.new(a, node.inputs[6])
    nt.links.new(b, node.inputs[7])
    return node.outputs[2]


def _principled(nt, o, tex):
    """Lit material: Principled BSDF fed by the material passes. Returns the surface shader socket."""
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    link = nt.links.new

    # base colour: image -> albedo (mix, gain) -> x baked shading
    color = tex.outputs["Color"]
    if o.use_albedo and o.albedo_map is not None:
        alb = _tex(nt, o.albedo_map, (-1000, 0))
        alb_out = alb.outputs["Color"]
        if o.albedo_gain != 1.0:
            hsv = nt.nodes.new("ShaderNodeHueSaturation")
            hsv.location = (-700, 0)
            hsv.inputs[2].default_value = o.albedo_gain       # Value: scales RGB
            link(alb_out, hsv.inputs[4])
            alb_out = hsv.outputs[0]
        color = _mix_color(nt, "MIX", o.albedo_mix, color, alb_out, (-450, 150))
    if o.use_shading and o.shading_map is not None:
        sh = _tex(nt, o.shading_map, (-1000, -300), data=True)
        color = _mix_color(nt, "MULTIPLY", o.shading_strength, color, sh.outputs["Color"], (-200, 150))
    link(color, bsdf.inputs["Base Color"])

    # roughness: map remapped to [min, max], else constant
    if o.use_roughness_map and o.roughness_map is not None:
        rt = _tex(nt, o.roughness_map, (-1000, -600), data=True)
        rng = nt.nodes.new("ShaderNodeMapRange")
        rng.location = (-450, -450)
        rng.inputs["To Min"].default_value = o.roughness_min
        rng.inputs["To Max"].default_value = o.roughness_max
        link(rt.outputs["Color"], rng.inputs["Value"])
        link(rng.outputs["Result"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = o.roughness

    # metallic: map x strength (clamped), else constant
    if o.use_metallic_map and o.metallic_map is not None:
        mt = _tex(nt, o.metallic_map, (-1000, -900), data=True)
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.use_clamp = True
        mul.location = (-450, -750)
        mul.inputs[1].default_value = o.metallic_strength
        link(mt.outputs["Color"], mul.inputs[0])
        link(mul.outputs["Value"], bsdf.inputs["Metallic"])
    else:
        bsdf.inputs["Metallic"].default_value = o.metallic

    # normal: normal map (camera = Object space, see core.passes) -> luminance bump on top
    normal = None
    if o.use_normal_map and o.normal_map is not None:
        nt_tex = _tex(nt, o.normal_map, (-1000, -1200), data=True)
        col = nt_tex.outputs["Color"]
        if o.normal_flip_x or o.normal_flip_y:
            flip = nt.nodes.new("ShaderNodeVectorMath")      # c' = c * s + t (flip: s = -1, t = 1)
            flip.operation = "MULTIPLY_ADD"
            flip.location = (-700, -1200)
            flip.inputs[1].default_value = (-1.0 if o.normal_flip_x else 1.0, -1.0 if o.normal_flip_y else 1.0, 1.0)
            flip.inputs[2].default_value = (1.0 if o.normal_flip_x else 0.0, 1.0 if o.normal_flip_y else 0.0, 0.0)
            link(col, flip.inputs[0])
            col = flip.outputs[0]
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.location = (-450, -1100)
        nmap.space = "OBJECT" if o.normal_space == "CAMERA" else "TANGENT"
        if nmap.space == "TANGENT":
            nmap.uv_map = "UVMap"
        nmap.inputs["Strength"].default_value = o.normal_strength
        link(col, nmap.inputs["Color"])
        normal = nmap.outputs["Normal"]
    if o.bump > 0:
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        bw.location = (-450, -1350)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.location = (-200, -1200)
        bump.inputs["Strength"].default_value = o.bump
        link(tex.outputs["Color"], bw.inputs["Color"])
        link(bw.outputs["Val"], bump.inputs["Height"])
        if normal is not None:
            link(normal, bump.inputs["Normal"])
        normal = bump.outputs["Normal"]
    if normal is not None:
        link(normal, bsdf.inputs["Normal"])

    # baked highlights: non-diffuse residual as emission
    if o.use_residual and o.residual_map is not None:
        rs = _tex(nt, o.residual_map, (-1000, 300))
        link(rs.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = o.residual_strength

    surface = bsdf.outputs["BSDF"]
    if o.original_light > 0:
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.location = (500, 300)
        link(tex.outputs["Color"], emit.inputs["Color"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.location = (800, 100)
        mix.inputs["Fac"].default_value = o.original_light
        link(surface, mix.inputs[1])
        link(emit.outputs["Emission"], mix.inputs[2])
        surface = mix.outputs["Shader"]
    return surface


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
    out.location = (1300, 0)
    tex = _tex(nt, o.image, (-1000, 600))
    if o.shading == "EMISSION":
        shader = nt.nodes.new("ShaderNodeEmission")
        shader.location = (500, 0)
        nt.links.new(tex.outputs["Color"], shader.inputs["Color"])
        surface = shader.outputs["Emission"]
    else:
        surface = _principled(nt, o, tex)
    if o.use_alpha and o.image is not None and bimage.has_alpha(o.image):
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        transp.location, mix.location = (800, 300), (1050, 0)
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
        nt.links.new(surface, mix.inputs[2])
        surface = mix.outputs["Shader"]
    nt.links.new(surface, out.inputs["Surface"])
    return mat


# ----------------------------------------------------------------------------- lights

def add_light(obj, kind="POINT"):
    """Add a light to the rig, in front of the projector, scaled to the scene.

    Placed at half the Near distance (perspective), i.e. in front of every
    reconstructed surface, slightly up and to the right of the camera.
    """
    o = obj.depth_scene
    rig = obj.parent
    if o.mode == "PERSPECTIVE":
        mid = math.sqrt(o.near * max(o.far, o.near))      # typical distance to the lit surfaces
        loc = Vector((0.15 * o.near, 0.2 * o.near, -0.5 * o.near))
    else:
        mid = o.size * 1.5
        loc = Vector((0.25 * o.size, 0.25 * o.size, o.size))
    power = 1000.0 * (mid / 3.5) ** 2                       # ~1 kW at 3.5 m: a lit room
    data = bpy.data.lights.new(f"{obj.name} {kind.title()} Light", kind)
    if kind == "SUN":
        data.energy = 3.0
    elif kind == "AREA":
        data.energy = power * 1.5
        data.size = max(0.25 * mid, 0.05)
    else:
        data.energy = power
    if kind == "SPOT":
        data.spot_size = math.radians(70.0)
        data.spot_blend = 0.4
    light = bpy.data.objects.new(data.name, data)
    collection = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection
    collection.objects.link(light)
    light.parent = rig
    light.location = loc
    if kind == "SUN":
        light.rotation_euler = (math.radians(-35.0), math.radians(-25.0), 0.0)   # from upper left, into the scene
    return light


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

def build_scene(context, image, depth_img, kind, mode):
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
    depth, valid = bimage.depth_arrays(depth_img)
    near, far = suggest_near_far(depth, kind, valid)
    o.near, o.far = near, far
    o.mapping = depth_mod.default_mapping(kind)
    o.shading = "EMISSION" if mode == "PERSPECTIVE" else "PRINCIPLED"
    o.use_alpha = bimage.has_alpha(image)
    obj.visible_shadow = o.cast_shadows
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
