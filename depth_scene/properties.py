"""Scene-level (generation) and object-level (live reconstruction) settings."""

import math

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

from .ai import models

MODEL_ITEMS = [(m.key, m.label, m.description) for m in models.DEPTH_MODELS.values()]

MODE_ITEMS = [
    ("PERSPECTIVE", "Perspective", "Unproject pixels along camera rays - true 3D for corridors, rooms, streets"),
    ("RELIEF", "Flat Relief", "Displace a flat plane toward the camera - bas-relief look"),
]
KIND_ITEMS = [
    ("disparity", "White = Near", "Relative inverse depth (most AI depth maps from other tools)"),
    ("depth", "White = Far", "Relative depth (Marigold output, z-buffer style maps)"),
    ("metric", "Metric (metres)", "Absolute planar depth, e.g. EXR from a renderer"),
]
MAPPING_ITEMS = [
    ("INVERSE", "Inverse", "1/z interpolated between near and far - correct for disparity maps"),
    ("LINEAR", "Linear", "z interpolated between near and far - correct for 'white = far' maps"),
    ("METRIC", "Metric", "Use metric depth directly (metric EXR)"),
]
SHADING_ITEMS = [
    ("EMISSION", "Emission (exact colours)", "Unlit: renders exactly like the source image from the projector camera"),
    ("PRINCIPLED", "Principled (lit)", "Reacts to scene lights and shadows; uses the material passes "
                                       "(normal, albedo, roughness, metallic, shading, highlights)"),
]
NORMAL_SPACE_ITEMS = [
    ("CAMERA", "Camera (Marigold)", "Normals in the projector camera's frame (x right, y up, z towards the "
                                    "camera) - what Marigold predicts. Applied in the mesh's Object space"),
    ("TANGENT", "Tangent", "Standard tangent-space normal map (from other tools), using the image UVs"),
]
ALBEDO_FROM_ITEMS = [
    ("APPEARANCE", "Appearance", "Albedo from IID-Appearance (sRGB-trained, pairs with roughness/metallic)"),
    ("LIGHTING", "Lighting", "Albedo from IID-Lighting (linear, consistent with its shading/highlights)"),
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


def _shadows(self, context):
    if self.is_depth_scene:
        self.id_data.visible_shadow = self.cast_shadows


def _fov(self, context):
    from . import builder
    builder.sync_cameras(self.id_data, lens_changed=True)
    _rebuild(self, context)


class DepthSceneSettings(bpy.types.PropertyGroup):
    """Generation settings, stored on the Scene."""

    image: PointerProperty(name="Image", type=bpy.types.Image)
    model: EnumProperty(name="Model", items=MODEL_ITEMS, default=0)
    device: EnumProperty(name="Device", items=[
        ("AUTO", "Auto", "GPU if available, else CPU"),
        ("GPU", "GPU", "CUDA / DirectML / CoreML"),
        ("CPU", "CPU", "Always CPU")], default="AUTO")
    precision: EnumProperty(name="Precision", items=[
        ("fp32", "Float32", "Full precision (recommended on CPU)"),
        ("fp16", "Float16", "fp16 U-Net: ~half the memory and faster; tiny depth differences")], default="fp32")
    input_size: IntProperty(name="Processing Resolution", default=768, min=0, soft_min=256, max=2048, step=8,
                            description="Longer image side fed to Marigold (768 = what Stable Diffusion 2 was "
                                        "trained at; 0 = native size). Lower = faster, less detail")
    denoise_steps: IntProperty(name="Denoising Steps", default=0, min=0, soft_max=10, max=50,
                               description="Diffusion steps (0 = checkpoint default). v1.1 is trained for 1-4; "
                                           "more steps cost time and rarely help")
    ensemble_size: IntProperty(name="Ensemble", default=1, min=1, soft_max=10, max=20,
                               description="Average several predictions with different noise. "
                                           "Smoother, more reliable, but N times slower")
    seed: IntProperty(name="Seed", default=0, min=0,
                      description="Noise seed - same seed + settings = same depth map")
    # ---- material passes (extra Marigold models; same resolution / steps / ensemble / seed)
    pass_normals: BoolProperty(name="Normal Map", default=False,
                               description="Marigold Normals: per-pixel surface normals so lights shade "
                                           "fine detail the mesh is too coarse to show")
    pass_appearance: BoolProperty(name="Albedo / Roughness / Metallic", default=False,
                                  description="Marigold IID-Appearance: de-lit colour plus PBR roughness and "
                                              "metallic maps")
    pass_lighting: BoolProperty(name="Shading / Highlights", default=False,
                                description="Marigold IID-Lighting: albedo, the painted diffuse shading and "
                                            "the non-diffuse residual (highlights, glows), calibrated so "
                                            "image = albedo x shading + residual")
    albedo_from: EnumProperty(name="Albedo From", items=ALBEDO_FROM_ITEMS, default="APPEARANCE",
                              description="Which model's albedo the material uses when both passes are generated")
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
    layer_steps: IntProperty(name="Quantize Layers", default=0, min=0, soft_max=32, max=256, update=_rebuild,
                             description="Snap depth to this many flat layers (0 = continuous). "
                                         "Gives a paper cut-out / diorama look; ideal with Illustrator's Depth")
    use_alpha: BoolProperty(name="Use Alpha Channel", default=False, update=_rebuild,
                            description="Delete faces where the image is transparent")

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

    # ---- material passes: maps (any image can be assigned) + toggles + parameters
    normal_map: PointerProperty(name="Normal Map", type=bpy.types.Image, update=_material)
    albedo_map: PointerProperty(name="Albedo", type=bpy.types.Image, update=_material)
    roughness_map: PointerProperty(name="Roughness Map", type=bpy.types.Image, update=_material)
    metallic_map: PointerProperty(name="Metallic Map", type=bpy.types.Image, update=_material)
    shading_map: PointerProperty(name="Shading Map", type=bpy.types.Image, update=_material)
    residual_map: PointerProperty(name="Highlights Map", type=bpy.types.Image, update=_material)

    use_normal_map: BoolProperty(name="Normal Map", default=True, update=_material,
                                 description="Shade with the normal map instead of the mesh normals only")
    normal_space: EnumProperty(name="Space", items=NORMAL_SPACE_ITEMS, default="CAMERA", update=_material)
    normal_strength: FloatProperty(name="Strength", default=1.0, min=0.0, soft_max=2.0, max=10.0, update=_material,
                                   description="0 = mesh normals, 1 = normal map, >1 exaggerates")
    normal_flip_x: BoolProperty(name="Flip X", default=False, update=_material,
                                description="Invert the red channel (maps with a left-pointing X axis)")
    normal_flip_y: BoolProperty(name="Flip Y", default=False, update=_material,
                                description="Invert the green channel (DirectX-style maps)")

    use_albedo: BoolProperty(name="Albedo", default=True, update=_material,
                             description="Use the de-lit albedo as base colour, so painted shadows and "
                                         "highlights are not lit a second time")
    albedo_mix: FloatProperty(name="Albedo Mix", default=1.0, min=0.0, max=1.0, subtype="FACTOR", update=_material,
                              description="0 = original image colours, 1 = albedo")
    albedo_gain: FloatProperty(name="Albedo Gain", default=1.0, min=0.0, soft_max=2.0, max=10.0, update=_material,
                               description="Brightness multiplier on the albedo")

    use_shading: BoolProperty(name="Baked Shading", default=False, update=_material,
                              description="Multiply the base colour by the painted diffuse shading "
                                          "(keeps soft shadows / ambient occlusion the mesh cannot cast)")
    shading_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                                    update=_material)

    use_roughness_map: BoolProperty(name="Roughness Map", default=True, update=_material)
    roughness: FloatProperty(name="Roughness", default=0.8, min=0.0, max=1.0, subtype="FACTOR", update=_material,
                             description="Constant roughness when no map is used")
    roughness_min: FloatProperty(name="Min", default=0.0, min=0.0, max=1.0, subtype="FACTOR", update=_material,
                                 description="Roughness for map value 0")
    roughness_max: FloatProperty(name="Max", default=1.0, min=0.0, max=1.0, subtype="FACTOR", update=_material,
                                 description="Roughness for map value 1")

    use_metallic_map: BoolProperty(name="Metallic Map", default=True, update=_material)
    metallic: FloatProperty(name="Metallic", default=0.0, min=0.0, max=1.0, subtype="FACTOR", update=_material,
                            description="Constant metallic when no map is used")
    metallic_strength: FloatProperty(name="Strength", default=1.0, min=0.0, soft_max=1.0, max=4.0,
                                     update=_material, description="Multiplier on the metallic map (clamped to 1)")

    use_residual: BoolProperty(name="Baked Highlights", default=False, update=_material,
                               description="Add the painted non-diffuse light (specular highlights, lamps, "
                                           "glows) as emission")
    residual_strength: FloatProperty(name="Strength", default=1.0, min=0.0, soft_max=2.0, max=20.0,
                                     update=_material)

    original_light: FloatProperty(name="Original Lighting", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                                  update=_material,
                                  description="Blend towards the unlit source image (1 = looks exactly like "
                                              "the painting, ignores scene lights)")
    cast_shadows: BoolProperty(name="Cast Shadows", default=True, update=_shadows,
                               description="The reconstructed geometry casts shadows from scene lights")

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
