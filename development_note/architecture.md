# Architecture

## Pipeline (three independent stages, as in TrueDepth)
1. **Depth** (`depth_scene/ai/`) - ONNX Runtime runs a registry model (`models.py`) with
   family-specific pre/post-processing (`infer.py`). Output: depth map + kind
   (`disparity` | `depth` | `metric`) + optional validity mask + optional FOV.
   Saved as float EXR (R = depth, A = valid) by `bimage.write_depth`.
2. **Geometry** (`depth_scene/core/`) - NumPy only, unit-tested without Blender.
   `geometry.build(MeshParams, depth, kind, image_size, ...)` returns verts/faces/uv
   in *projector-camera space* (x right, y up, -z forward).
3. **Shading** (`builder.update_material`) - image texture on matching UVs; Emission
   (exact colours) or Principled (+ Bump).

## Blender objects per build
`<name> Rig` (empty, rot X +90 deg, at 3D cursor)
 ├─ `<name> 3D` mesh  - settings in `Object.depth_scene` (live rebuild on update)
 ├─ `<name> Projector` camera (locked, FOV = settings.fov)
 └─ `<name> Shot Camera` (user-animated)

## Key decisions
- ONNX Runtime instead of PyTorch: 14-24 MB vs ~2 GB; Blender ships NumPy only.
- pip `--target` into a per-Python-version, per-variant user folder; strip numpy so
  Blender's own build is used.
- Numpy mesh rebuild (not Geometry Nodes): version-proof 4.2 -> 5.2, unit-testable,
  ~0.1 s at default resolution.
- Stretched faces = angle test on |grad log z|*|r| (resolution independent).
- Default model DA2-Small (Apache); MoGe-2 ViT-S recommended for perspective (FOV + metric).
