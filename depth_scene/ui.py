"""Sidebar panels (3D Viewport > N panel > Depth Scene)."""

import math
import os

import bpy

from . import paths
from .ai import models, runtime
from .operators import _active_depth_object
from .properties import KIND_ITEMS

KIND_LABELS = {k: n for k, n, _ in KIND_ITEMS}


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
        else:
            box.prop(s, "model")
            if s.model == "CUSTOM":
                box.prop(s, "custom_model_path")
            else:
                info = models.get(s.model)
                if not info.commercial_ok:
                    box.label(text=f"{info.license_id}: non-commercial use only", icon="ERROR")
                path = paths.model_path(info, s.precision)
                if not os.path.exists(path):
                    row = box.row()
                    row.alert = True
                    op = row.operator("depth_scene.download_model", text=f"Download (~{info.size_mb(s.precision)} MB)", icon="URL")
                    op.model, op.precision = info.key, s.precision if s.precision in info.files else "fp32"
            row = box.row(align=True)
            row.prop(s, "device", text="")
            row.prop(s, "precision", text="")
            box.prop(s, "input_size")
            box.prop(s, "save_location")
            if runtime.import_ort() is None:
                box.label(text="Install ONNX Runtime in Preferences > Add-ons", icon="ERROR")
            box.operator("depth_scene.generate_depth", icon="SHADERFX")
            if s.existing_depth is not None:
                box.label(text=f"Last depth: {s.existing_depth.name} ({KIND_LABELS.get(s.existing_kind, '')})",
                          icon="CHECKMARK")

        box = layout.box()
        box.label(text="3. Build", icon="SCENE_DATA")
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
        if o.mode == "PERSPECTIVE":
            col = layout.column(align=True)
            col.prop(o, "fov")
            if o.estimated_fov > 0:
                col.operator("depth_scene.apply_estimated_fov",
                             text=f"Use Estimated FOV ({math.degrees(o.estimated_fov):.1f} deg)")
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
        col.prop(o, "remove_masked")
        layout.label(text=f"{o.stat_faces:,} faces, {o.stat_removed:,} stretched removed", icon="INFO")


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
        o = _active_depth_object(context).depth_scene
        layout.prop(o, "shading")
        row = layout.row()
        row.active = o.shading == "PRINCIPLED"
        row.prop(o, "bump")


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


classes = (DEPTHSCENE_PT_main, DEPTHSCENE_PT_object, DEPTHSCENE_PT_material, DEPTHSCENE_PT_camera)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
