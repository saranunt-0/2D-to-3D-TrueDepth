"""Scene-level (generation) and object-level (live reconstruction) settings."""

import math

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

from .ai import models

MODEL_ITEMS = [(m.key, m.label, m.description) for m in models.MODELS.values()]
MODEL_ITEMS.append(("CUSTOM", "Custom ONNX file", "Any Depth-Anything-style ONNX (image -> relative inverse depth)"))

MODE_ITEMS = [
    ("PERSPECTIVE", "Perspective", "Unproject pixels along camera rays - true 3D for corridors, rooms, streets"),
    ("RELIEF", "Flat Relief", "Displace a flat plane toward the camera - bas-relief look"),
]
KIND_ITEMS = [
    ("disparity", "White = Near", "Relative inverse depth (Depth Anything, MiDaS, most AI depth maps)"),
    ("depth", "White = Far", "Relative depth (z-buffer style maps)"),
    ("metric", "Metric (metres)", "Absolute planar depth, e.g. EXR from MoGe or a renderer"),
]
MAPPING_ITEMS = [
    ("INVERSE", "Inverse", "1/z interpolated between near and far - correct for disparity maps"),
    ("LINEAR", "Linear", "z interpolated between near and far - correct for 'white = far' maps"),
    ("METRIC", "Metric", "Use the model's metric depth directly (MoGe / metric EXR)"),
]
SHADING_ITEMS = [
    ("EMISSION", "Emission (exact colours)", "Unlit: renders exactly like the source image from the projector camera"),
    ("PRINCIPLED", "Principled (lit)", "Reacts to scene lights; enables the Bump control"),
]


def _rebuild(self, context):
    if not self.live_update or not self.is_depth_scene:
        return
    from . import builder
    builder.rebuild(self.id_data, quiet=True)


def _material(self, context):
    if self.is_depth_scene:
        from . import builder
        builder.update_material(self.id_data)


def _fov(self, context):
    from . import builder
    builder.sync_cameras(self.id_data, lens_changed=True)
    _rebuild(self, context)


class DepthSceneSettings(bpy.types.PropertyGroup):
    """Generation settings, stored on the Scene."""

    image: PointerProperty(name="Image", type=bpy.types.Image)
    model: EnumProperty(name="Model", items=MODEL_ITEMS, default=0)
    custom_model_path: StringProperty(name="ONNX File", subtype="FILE_PATH")
    device: EnumProperty(name="Device", items=[
        ("AUTO", "Auto", "GPU if available, else CPU"),
        ("GPU", "GPU", "CUDA / DirectML / CoreML"),
        ("CPU", "CPU", "Always CPU")], default="AUTO")
    precision: EnumProperty(name="Precision", items=[
        ("fp32", "Float32", "Full precision (recommended on CPU)"),
        ("fp16", "Float16", "Half the download/memory; fastest on GPU")], default="fp32")
    input_size: IntProperty(name="Inference Size", default=518, min=266, max=1792, step=14,
                            description="Model input size (shorter side for Depth Anything, token budget for MoGe). "
                                        "Higher = more detail, slower")
    use_existing_depth: BoolProperty(name="Use Existing Depth Map", default=False)
    existing_depth: PointerProperty(name="Depth Map", type=bpy.types.Image)
    existing_kind: EnumProperty(name="Depth Map Type", items=KIND_ITEMS, default="disparity")
    save_location: StringProperty(name="Save Location", subtype="DIR_PATH", default="//depth_scene/",
                                  description="Folder for generated depth maps ('//' = next to the .blend; "
                                              "temp folder if the file is unsaved)")
    mode: EnumProperty(name="Mode", items=MODE_ITEMS, default="PERSPECTIVE")
    set_render_size: BoolProperty(name="Match Render Resolution", default=True,
                                  description="Set the render resolution to the image size")
    set_standard_view: BoolProperty(name="Standard View Transform", default=True,
                                    description="Switch colour management to 'Standard' so the unlit image "
                                                "renders with its original colours (AgX/Filmic would alter them)")


class DepthSceneObject(bpy.types.PropertyGroup):
    """Reconstruction settings, stored on each generated mesh object (live)."""

    is_depth_scene: BoolProperty(default=False, options={"HIDDEN"})
    live_update: BoolProperty(name="Live Update", default=True,
                              description="Rebuild the mesh whenever a setting changes")
    mode: EnumProperty(name="Mode", items=MODE_ITEMS, default="PERSPECTIVE", update=_rebuild)
    image: PointerProperty(name="Image", type=bpy.types.Image, update=_material)
    depth_image: PointerProperty(name="Depth", type=bpy.types.Image, update=_rebuild)
    depth_kind: EnumProperty(name="Depth Type", items=KIND_ITEMS, default="disparity", update=_rebuild)
    estimated_fov: FloatProperty(name="Estimated FOV", subtype="ANGLE", default=0.0)
    projector: PointerProperty(name="Projector Camera", type=bpy.types.Object)
    shot_camera: PointerProperty(name="Shot Camera", type=bpy.types.Object)
    material: PointerProperty(name="Material", type=bpy.types.Material)

    # ---- geometry (shared)
    resolution: IntProperty(name="Resolution", default=384, min=8, soft_max=1024, max=4096, update=_rebuild,
                            description="Grid vertices along the longer image side (TrueDepth: Resolution / Grid Resolution)")
    smooth: FloatProperty(name="Smooth", default=0.0, min=0.0, soft_max=3.0, max=20.0, precision=2,
                          subtype="PERCENTAGE", update=_rebuild,
                          description="Blur the depth map (radius in % of the image size) to reduce noise")
    detail_strength: FloatProperty(name="Detail Displacement", default=0.0, soft_min=-0.3, soft_max=0.3,
                                   precision=3, update=_rebuild,
                                   description="Physical high-frequency detail from image luminance "
                                               "(bright = closer). Fraction of distance (Perspective) or plane size (Relief)")
    detail_radius: FloatProperty(name="Detail Radius", default=0.5, min=0.05, soft_max=3.0, max=10.0,
                                 subtype="PERCENTAGE", precision=2, update=_rebuild,
                                 description="Size of the details kept by Detail Displacement, % of image size")
    use_alpha: BoolProperty(name="Use Alpha Channel", default=False, update=_rebuild,
                            description="Delete faces where the image is transparent")
    remove_masked: BoolProperty(name="Remove Sky / Invalid", default=False, update=_rebuild,
                                description="Delete faces the model marked invalid (MoGe sky mask)")

    # ---- perspective
    fov: FloatProperty(name="Field of View", subtype="ANGLE", default=math.radians(60.0),
                       min=math.radians(5.0), max=math.radians(170.0), update=_fov,
                       description="Camera FOV across the longer image side, used to unproject the depth")
    mapping: EnumProperty(name="Depth Mapping", items=MAPPING_ITEMS, default="INVERSE", update=_rebuild)
    near: FloatProperty(name="Near", default=1.0, min=0.001, soft_max=100.0, unit="LENGTH", update=_rebuild,
                        description="Distance of the nearest pixel")
    far: FloatProperty(name="Far", default=10.0, min=0.002, soft_max=1000.0, unit="LENGTH", update=_rebuild,
                       description="Distance of the farthest pixel. The near/far ratio also controls how straight walls are")
    metric_scale: FloatProperty(name="Scale", default=1.0, min=0.001, soft_max=10.0, update=_rebuild,
                                description="Multiplier on metric depth")
    remove_stretched: BoolProperty(name="Remove Stretched Faces", default=True, update=_rebuild,
                                   description="Cut the rubber-sheet faces at depth discontinuities")
    stretch_angle: FloatProperty(name="Stretch Threshold", subtype="ANGLE", default=math.radians(86.0),
                                 min=math.radians(45.0), max=math.radians(89.9), update=_rebuild,
                                 description="Delete faces seen at a steeper angle than this. Lower = cut more")

    # ---- relief
    size: FloatProperty(name="Size", default=2.0, min=0.01, soft_max=20.0, unit="LENGTH", update=_rebuild,
                        description="Width of the longer side of the relief plane")
    base_displacement: FloatProperty(name="Base Displacement", default=0.3, soft_min=-2.0, soft_max=2.0,
                                     unit="LENGTH", update=_rebuild,
                                     description="Depth of the relief (large-scale displacement)")
    invert: BoolProperty(name="Invert", default=False, update=_rebuild)

    # ---- shading
    shading: EnumProperty(name="Shading", items=SHADING_ITEMS, default="EMISSION", update=_material)
    bump: FloatProperty(name="Bump", default=0.0, min=0.0, max=1.0, update=_material,
                        description="Shader-only fine detail from image luminance (Principled shading)")

    # ---- info (read-only for UI)
    stat_faces: IntProperty(default=0, options={"HIDDEN"})
    stat_removed: IntProperty(default=0, options={"HIDDEN"})


classes = (DepthSceneSettings, DepthSceneObject)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.depth_scene = PointerProperty(type=DepthSceneSettings)
    bpy.types.Object.depth_scene = PointerProperty(type=DepthSceneObject)


def unregister():
    del bpy.types.Object.depth_scene
    del bpy.types.Scene.depth_scene
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
