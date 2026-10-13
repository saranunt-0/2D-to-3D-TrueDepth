"""Sidebar panels (3D Viewport > N panel > Depth Scene)."""

import bpy

from . import builder, paths, v2_job
from .ai import marigold, marigold_v2, models, runtime
from .operators import _active_depth_object
from .properties import KIND_ITEMS

KIND_LABELS = {k: n for k, n, _ in KIND_ITEMS}
PASS_PROPS = (("normals", "pass_normals"), ("appearance", "pass_appearance"), ("lighting", "pass_lighting"))


def _missing_model_warning(layout, info, precision):
    if marigold.missing_files(paths.model_path(info, precision)):
        col = layout.column(align=True)
        col.alert = True
        col.label(text=f"Needs folder {info.filename(precision)} in the models folder", icon="ERROR")
        col.label(text=f"Convert with tools/export_marigold_onnx.py --task {info.export_task}")
        return True
    return False


def _draw_v2_depth(box, s):
    """Marigold V2 (external) settings + setup status."""
    python, repo, assets = v2_job.settings_from_prefs()
    modalities = ["depth"] + builder.enabled_v2_passes(s)
    problems = marigold_v2.check_setup(python, repo, assets, modalities=modalities, depth_checkpoint=s.v2_checkpoint)
    if problems:
        col = box.column(align=True)
        col.alert = True
        col.label(text=problems[0], icon="ERROR")
        col.label(text="Set it up in Preferences > Add-ons > Depth Scene (see README)")
    col = box.column(align=True)
    col.prop(s, "v2_checkpoint")
    col.prop(s, "v2_resolution")
    col.prop(s, "v2_seed")
    col = box.column(align=True)
    col.label(text="Separate process on a CUDA GPU: ~17 GB VRAM at 1024 px.", icon="INFO")
    col.label(text="Each run reloads the 20 B backbone (minutes). Apache-2.0.")
    box.prop(s, "save_location")
    job = v2_job.ACTIVE.get("job")
    if job is not None:
        box.label(text=job.status or "Marigold V2 running...", icon="SORTTIME")
        box.label(text="Esc in the 3D Viewport cancels")
    else:
        box.operator("depth_scene.generate_depth", text="Generate Depth (Marigold V2)", icon="SHADERFX")


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Depth Scene"


class DEPTHSCENE_PT_main(_Base, bpy.types.Panel):
    bl_label = "Depth Scene"

    def draw(self, context):
        layout = self.layout
        s = context.scene.depth_scene

        box = layout.box()
        box.label(text="1. Image", icon="IMAGE_DATA")
        box.template_ID(s, "image", open="image.open")

        box = layout.box()
        box.label(text="2. Depth", icon="MOD_DISPLACE")
        box.prop(s, "use_existing_depth")
        if s.use_existing_depth:
            box.template_ID(s, "existing_depth", open="image.open")
            box.prop(s, "existing_kind")
        elif s.backend == "V2":
            box.prop(s, "backend", text="")
            _draw_v2_depth(box, s)
        else:
            box.prop(s, "backend", text="")
            box.prop(s, "model")
            info = models.get(s.model)
            _missing_model_warning(box, info, s.precision)
            row = box.row(align=True)
            row.prop(s, "device", text="")
            row.prop(s, "precision", text="")
            col = box.column(align=True)
            col.prop(s, "input_size")
            col.prop(s, "denoise_steps")
            col.prop(s, "ensemble_size")
            col.prop(s, "seed")
            box.label(text=f"{info.license_id}: commercial OK, use restrictions apply", icon="INFO")
            box.prop(s, "save_location")
            if runtime.import_ort() is None:
                box.label(text="Install ONNX Runtime in Preferences > Add-ons", icon="ERROR")
            box.operator("depth_scene.generate_depth", icon="SHADERFX")
        if not s.use_existing_depth and s.existing_depth is not None:
            box.label(text=f"Last depth: {s.existing_depth.name} ({KIND_LABELS.get(s.existing_kind, '')})",
                      icon="CHECKMARK")

        box = layout.box()
        box.label(text="3. Material Passes (optional)", icon="MATERIAL")
        col = box.column(align=True)
        v2 = s.backend == "V2"
        if v2:
            col.prop(s, "pass_normals", text="Normal Map (Marigold V2)")
            col.prop(s, "pass_albedo_v2")
            col.separator()
        for group, prop in PASS_PROPS:
            if v2 and group == "normals":
                continue
            col.prop(s, prop, text=f"{s.bl_rna.properties[prop].name}{' (v1.1 ONNX)' if v2 else ''}")
            if getattr(s, prop):
                _missing_model_warning(col, models.get(models.PASS_MODELS[group]), s.precision)
        if sum((s.pass_appearance, s.pass_lighting, v2 and s.pass_albedo_v2)) > 1:
            box.prop(s, "albedo_from")
        if builder.enabled_pass_groups(s) or builder.enabled_v2_passes(s):
            col = box.column(align=True)
            col.label(text="Uses the depth settings above (resolution, seed...);", icon="INFO")
            col.label(text="each pass ~ one depth run. Build runs them too.")
            box.operator("depth_scene.generate_passes", text="Generate for Active Object", icon="NODE_MATERIAL")

        box = layout.box()
        box.label(text="4. Build", icon="SCENE_DATA")
        box.row().prop(s, "mode", expand=True)
        col = box.column(align=True)
        col.prop(s, "set_render_size")
        col.prop(s, "set_standard_view")
        row = box.row()
        row.scale_y = 1.5
        row.operator("depth_scene.build", icon="OUTLINER_OB_MESH")


class DEPTHSCENE_PT_object(_Base, bpy.types.Panel):
    bl_label = "Reconstruction"
    bl_parent_id = "DEPTHSCENE_PT_main"

    @classmethod
    def poll(cls, context):
        return _active_depth_object(context) is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        obj = _active_depth_object(context)
        o = obj.depth_scene
        row = layout.row(align=True)
        row.prop(o, "live_update", toggle=True, icon="FILE_REFRESH")
        row.operator("depth_scene.rebuild", text="Rebuild")
        layout.prop(o, "mode")
        layout.prop(o, "depth_image")
        layout.prop(o, "depth_kind")

        col = layout.column(align=True)
        col.prop(o, "resolution")
        col.prop(o, "smooth")
        if not (o.mode == "PERSPECTIVE" and o.mapping == "METRIC"):
            col.prop(o, "layer_steps")
        if o.mode == "PERSPECTIVE":
            col = layout.column(align=True)
            col.prop(o, "fov")
            layout.prop(o, "mapping")
            col = layout.column(align=True)
            if o.mapping == "METRIC":
                col.prop(o, "metric_scale")
            else:
                col.prop(o, "near")
                col.prop(o, "far")
            col = layout.column(align=True)
            col.prop(o, "remove_stretched")
            sub = col.column()
            sub.active = o.remove_stretched
            sub.prop(o, "stretch_angle")
        else:
            col = layout.column(align=True)
            col.prop(o, "size")
            col.prop(o, "base_displacement")
            col.prop(o, "invert")
        col = layout.column(align=True)
        col.prop(o, "detail_strength")
        sub = col.column()
        sub.active = o.detail_strength != 0
        sub.prop(o, "detail_radius")
        col = layout.column(align=True)
        col.prop(o, "use_alpha")
        layout.label(text=f"{o.stat_faces:,} faces, {o.stat_removed:,} stretched removed", icon="INFO")


def _map_section(layout, o, toggle, image_prop, hint):
    """Header toggle + map picker; returns a column for the parameters (greyed out when off)."""
    box = layout.box()
    box.prop(o, toggle)
    col = box.column()
    col.active = getattr(o, toggle)
    col.prop(o, image_prop, text="Map")
    if getattr(o, image_prop) is None:
        col.label(text=hint, icon="INFO")
    return col


class DEPTHSCENE_PT_material(_Base, bpy.types.Panel):
    bl_label = "Material"
    bl_parent_id = "DEPTHSCENE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _active_depth_object(context) is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        o = _active_depth_object(context).depth_scene
        layout.prop(o, "shading")
        if o.shading != "PRINCIPLED":
            layout.label(text="Switch to Principled to use lights, shadows and passes", icon="INFO")
            return

        col = _map_section(layout, o, "use_albedo", "albedo_map", "Generate Albedo / Roughness / Metallic")
        col.prop(o, "albedo_mix")
        col.prop(o, "albedo_gain")
        col = _map_section(layout, o, "use_shading", "shading_map", "Generate Shading / Highlights")
        col.prop(o, "shading_strength")

        col = _map_section(layout, o, "use_normal_map", "normal_map", "Generate the Normal Map pass")
        col.prop(o, "normal_space")
        col.prop(o, "normal_strength")
        row = col.row(align=True)
        row.prop(o, "normal_flip_x", toggle=True)
        row.prop(o, "normal_flip_y", toggle=True)
        layout.prop(o, "bump")

        col = _map_section(layout, o, "use_roughness_map", "roughness_map", "Generate Albedo / Roughness / Metallic")
        if o.use_roughness_map and o.roughness_map is not None:
            sub = col.column(align=True)
            sub.prop(o, "roughness_min")
            sub.prop(o, "roughness_max")
        else:
            layout.prop(o, "roughness")
        col = _map_section(layout, o, "use_metallic_map", "metallic_map", "Generate Albedo / Roughness / Metallic")
        if o.use_metallic_map and o.metallic_map is not None:
            col.prop(o, "metallic_strength")
        else:
            layout.prop(o, "metallic")

        col = _map_section(layout, o, "use_residual", "residual_map", "Generate Shading / Highlights")
        col.prop(o, "residual_strength")


class DEPTHSCENE_PT_lighting(_Base, bpy.types.Panel):
    bl_label = "Lights & Shadows"
    bl_parent_id = "DEPTHSCENE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _active_depth_object(context) is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        o = _active_depth_object(context).depth_scene
        layout.prop(o, "cast_shadows")
        row = layout.row()
        row.active = o.shading == "PRINCIPLED"
        row.prop(o, "original_light")
        layout.use_property_split = False
        row = layout.row(align=True)
        for kind, icon in (("POINT", "LIGHT_POINT"), ("AREA", "LIGHT_AREA"), ("SPOT", "LIGHT_SPOT"),
                           ("SUN", "LIGHT_SUN")):
            row.operator("depth_scene.add_light", text=kind.title(), icon=icon).kind = kind
        layout.label(text="Lights are parented to the rig, in front of the projector.", icon="INFO")


class DEPTHSCENE_PT_camera(_Base, bpy.types.Panel):
    bl_label = "Camera / 2.5D"
    bl_parent_id = "DEPTHSCENE_PT_main"

    @classmethod
    def poll(cls, context):
        return _active_depth_object(context) is not None

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("depth_scene.look_through", text="Projector", icon="OUTLINER_OB_CAMERA").which = "PROJECTOR"
        row.operator("depth_scene.look_through", text="Shot", icon="VIEW_CAMERA").which = "SHOT"
        layout.operator("depth_scene.camera_move", icon="ANIM")
        layout.operator("depth_scene.reset_shot", icon="LOOP_BACK")
        layout.label(text="Animate the Shot camera; keep the Projector fixed.", icon="INFO")


classes = (DEPTHSCENE_PT_main, DEPTHSCENE_PT_object, DEPTHSCENE_PT_material, DEPTHSCENE_PT_lighting,
           DEPTHSCENE_PT_camera)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
