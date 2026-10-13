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

### Branch `marigold` additions
- Depth = Marigold (3 ONNX graphs + NumPy DDIM, `ai/marigold.py`, `core/ddim.py`).
- **1b. Material passes** (`builder.generate_passes` -> `infer.estimate_passes`):
  Marigold normals / IID-Appearance / IID-Lighting, same `marigold.predict` loop with
  `task` + `n_targets` from `marigold.json`; `core/passes.py` turns outputs into maps
  (normal encoded n*0.5+0.5, albedo linear, roughness/metallic, shading/residual
  calibrated so image = albedo*shading + residual). Stored as EXR, assigned to
  `Object.depth_scene.*_map`, reused by signature (image + model + settings).
- Principled tree (`_principled`): base = image -> albedo mix/gain -> x shading;
  roughness MapRange; metallic Math; NormalMap (OBJECT space = projector camera frame)
  -> Bump; residual -> Emission; optional MixShader to unlit image (Original Lighting).
  Rebuilt on every property change (managed material).
- **Marigold V2 backend** (`v2_job.V2Job`, `ai/marigold_v2.py`): external process running the
  user's marigold-v2 `scripts/infer.py` per modality (depth / normals / albedo); outputs (.npy)
  converted and saved as the same EXR maps; depth kind `log_depth` -> LOG mapping. Modal
  operator `depth_scene.marigold_v2` keeps Blender responsive.

## Blender objects per build
`<name> Rig` (empty, rot X +90 deg, at 3D cursor)
 ├─ `<name> 3D` mesh  - settings in `Object.depth_scene` (live rebuild on update)
 ├─ `<name> Projector` camera (locked, FOV = settings.fov)
 ├─ `<name> Shot Camera` (user-animated)
 └─ optional `<name> 3D <Type> Light` (Add Light, in front of the projector)

## Key decisions
- ONNX Runtime instead of PyTorch: 14-24 MB vs ~2 GB; Blender ships NumPy only.
- pip `--target` into a per-Python-version, per-variant user folder; strip numpy so
  Blender's own build is used.
- Numpy mesh rebuild (not Geometry Nodes): version-proof 4.2 -> 5.2, unit-testable,
  ~0.1 s at default resolution.
- Stretched faces = angle test on |grad log z|*|r| (resolution independent).
- Default model DA2-Small (Apache); MoGe-2 ViT-S recommended for perspective (FOV + metric).
