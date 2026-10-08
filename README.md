# Depth Scene — turn one 2D image into a 3D scene in Blender

A Blender add-on that estimates depth from a single image with a small AI model
and rebuilds the picture as real 3D geometry. You can then move a camera
through it (2.5D / parallax shots) or use it as a relief. The workflow and
parameters follow the commercial
[TrueDepth](https://superhivemarket.com/products/truedepth) add-on. This is an
independent, open-source implementation and is not affiliated with TrueDepth.

![Dolly into a reconstructed corridor](docs/images/demo_dolly.gif)

*Dolly-in through a corridor rebuilt from one image (the synthetic test scene
in `tests/synthetic.py`). The dark band behind the red pillar is the area the
single photo never saw. See [Limitations](#limitations).*

| Source image | Rebuilt scene, seen from outside | New camera position |
|---|---|---|
| ![](docs/images/demo_source.png) | ![](docs/images/demo_overview.png) | ![](docs/images/demo_shot.png) |

---

## 1. Basic concept

The add-on works in three separate stages. Changing a setting in one stage
does not improve the others:

```
 image ──► [1] AI depth estimation ──► depth map (.exr)
                                          │
                     [2] geometry ◄───────┘   grid mesh, unprojected or displaced
                          │
                     [3] shading                image re-projected from the camera,
                                                optional bump / lighting
```

**Stage 1: depth.** An ONNX model runs locally (CPU or GPU, no PyTorch
needed) and produces a depth map. It is saved as a 32-bit EXR so you can
reuse or edit it.

**Stage 2: geometry.** There are two modes:

* **Perspective** (corridors, rooms, streets). Each vertex of an *N × M* grid
  is pushed along its camera ray to its estimated distance:

  ```
  x = (u − ½)·2·tan(FOV/2)·(W/max(W,H)) · z
  y = (½ − v)·2·tan(FOV/2)·(H/max(W,H)) · z
  position = (x, y, −z)        (Blender camera space, camera looks down −Z)
  ```

  Every vertex sits exactly on the ray of the pixel it came from, so the
  image lines up perfectly from the original viewpoint, and a corridor
  becomes a real corridor. Faces that span a depth jump (the "rubber sheet"
  between a door frame and the room behind it) are removed by **Remove
  Stretched Faces**.
* **Flat Relief.** A plane in the image's aspect ratio, displaced toward the
  camera by how near each pixel is. This gives a bas-relief look, suited to
  paintings and landscapes.

**Stage 3: shading.** The image is mapped with UVs that match the
projection. *Emission* shading renders exactly like the source photo.
*Principled* shading reacts to lights and enables **Bump** for fine,
shader-only detail.

**Cameras (2.5D).** Each build creates two cameras under a rig empty:

* a locked **Projector** camera, at the viewpoint the image was taken from;
* a **Shot** camera that you animate.

Small Shot-camera moves give convincing parallax. Large moves reveal what the
photo never showed. Move the rig empty to place, rotate or scale the whole
scene.

### Why distances need "Near / Far" (and why walls can bend)

Depth Anything V2 outputs *relative inverse depth*. It knows what is in front
of what, but not the true scale or the zero point (shift). The add-on maps it
the way TrueDepth does:

| Mapping | Formula (t = 1 nearest … 0 farthest) | Use for |
|---|---|---|
| **Inverse** | 1/z = t/near + (1−t)/far | AI disparity maps (default) |
| **Linear**  | z = t·near + (1−t)·far | "white = far" depth maps |
| **Metric**  | z = model depth × scale | MoGe-2 or metric EXRs |

With *Inverse*, the **far/near ratio stands in for the unknown shift**. Only
one ratio makes the corridor walls perfectly straight. The add-on starts
from a "zero shift" guess (`far/near = max/min disparity`). If walls bow
outward or inward, adjust **Far**. **MoGe-2** avoids this problem because it
estimates metric depth *and* the camera FOV.

---

## 2. Which AI model (short version)

The full comparison and sources are in [docs/RESEARCH.md](docs/RESEARCH.md).

| Model (ONNX) | Params | Download | Licence | Output | Best for |
|---|---|---|---|---|---|
| **Depth Anything V2 Small** *(default)* | 24.8 M | ~99 MB (fp16 ~50 MB) | Apache-2.0 ✅ | relative inverse depth | fast, sharp depth; relief; general use |
| **MoGe-2 ViT-S** | 35 M | ~141 MB | MIT ✅ | metric depth **+ estimated FOV** | **corridors / perspective mode** |
| Depth Anything V2 Base / Large | 97.5 M / 335 M | 390 MB / 1.34 GB | CC-BY-NC-4.0 ⚠️ | relative inverse depth | more detail, **non-commercial only** |
| MoGe-2 ViT-B | 104 M | ~420 MB | MIT ✅ | metric + FOV | better geometry, slower |

Recommendation for corridor shots: use **MoGe-2 ViT-S** in Perspective mode,
because it sets the FOV and gives straight walls without tuning. Use
**Depth Anything V2 Small** when you want the fastest, crispest depth map or
are working in relief mode.

---

## 3. Installation

Requires **Blender 4.2 LTS or newer**. Tested with Blender 4.2.23 LTS and
5.2.2 LTS.

1. **Get the add-on zip.** Either
   * compress the `depth_scene` folder of this repo into `depth_scene.zip`, or
   * run `python tools/build_zip.py`, which writes `dist/depth_scene-0.1.0.zip`.
2. **Install it.** In Blender, open *Edit → Preferences → Get Extensions*,
   then the **⌄** menu (top right) → **Install from Disk…** and pick the zip.
   (Blender 4.2+ also accepts drag-and-drop of the zip onto the window.)
3. **Install the AI runtime.** Open *Preferences → Add-ons → Depth Scene*,
   choose the runtime and click **Install**:
   * **CPU (all platforms)**. Works everywhere and uses the GPU via CoreML on
     Apple Silicon. Download is 14–24 MB. Current ONNX Runtime macOS wheels
     need Apple Silicon + macOS 14; Intel Macs get an older compatible
     release automatically.
   * **NVIDIA CUDA**. Needs CUDA 12 + cuDNN 9 installed on the system.
   * **DirectML (Windows)**. Works with any DirectX 12 GPU (NVIDIA, AMD,
     Intel).

   ONNX Runtime is installed into the add-on's own user folder, and
   Blender's NumPy is left untouched. Restart Blender if you switch runtimes.
4. **Download a model.** In the same preferences panel, click **Download**
   next to *Depth Anything V2 Small* and/or *MoGe-2 ViT-S*. Files go to the
   add-on's user folder, or to a folder you choose in *Models Folder*.
   * Offline or behind a firewall: download the `.onnx` file yourself (the
     URL appears in the error message) and save it in the models folder under
     the name shown in the list, e.g. `da2_small_fp32.onnx` or
     `moge2_vits_fp32.onnx`.

> You can use the add-on **without AI**: tick *Use Existing Depth Map* and
> load a depth map made elsewhere (white = near, white = far, or a metric
> EXR).

---

## 4. Quick start (corridor shot)

1. 3D Viewport → press **N** → **Depth Scene** tab.
2. **1. Image**: open your picture (JPG/PNG/WebP/EXR).
3. **2. Depth**: pick a model (MoGe-2 ViT-S for corridors), *Device: Auto*,
   *Inference Size 518*. Optionally click **Generate Depth Map** to preview.
4. **3. Build**: choose **Perspective** → **Build 3D Scene**. This creates a
   new collection with the rig, mesh, Projector and Shot cameras. The render
   resolution is set to the image size, the view transform to *Standard*,
   and the Shot camera becomes active.
5. Tune in **Reconstruction** (live; each change rebuilds the mesh):
   * walls bowed → adjust **Far** (Depth Anything) or use the estimated
     **FOV** (MoGe);
   * jagged edges around doors → lower **Stretch Threshold** (e.g. 84°);
   * noisy surfaces → **Smooth** 0.5–1 %.
6. **Camera / 2.5D** → **Add Camera Move** → *Dolly In*, with a distance of
   about 30–50 % of *Near* → render.

---

## 5. Parameters (and their TrueDepth counterparts)

| Depth Scene | TrueDepth | What it does |
|---|---|---|
| Model (S/B/L, MoGe) | Model Size ViT-S/B/L | Depth network and size |
| Device Auto/GPU/CPU | Processing Device | Execution provider |
| Precision fp32/fp16 | Precision | fp16 halves download and memory and is fastest on GPU; use fp32 on CPU |
| Inference Size | (implicit) | Model input resolution (shorter side), a multiple of 14 |
| Use Existing Depth Map | (import) | Skip AI and use your own depth map |
| Save Location | Save Location | Where depth EXRs go (`//` = next to the .blend) |
| Mode Perspective / Flat Relief | Perspective Mode / Flat Relief | Unprojection vs plane displacement |
| Resolution | Resolution / Grid Resolution | Vertices along the longer side (default 384) |
| Field of View | FOV | Camera FOV across the longer side, used for unprojection |
| Depth Mapping Inverse/Linear/Metric | Depth Mapping Linear/Inverse | See the table above |
| Near / Far | Near / Far Distance | Distance of the nearest and farthest pixel |
| Remove Stretched Faces + Threshold | Remove Stretched Faces | Deletes faces seen steeper than the threshold angle |
| Base Displacement | Base Displacement | Relief depth |
| Detail Displacement + Radius | Detail Displacement | Real geometry from image-luminance high-pass (brush strokes, bark) |
| Bump | Bump / Detail Bump | Shader-only detail (Principled shading) |
| Smooth | Smooth | Blurs the depth map before meshing |
| Use Alpha Channel | Include Alpha Channel | Deletes transparent areas (PNG cut-outs) |
| Remove Sky / Invalid | — | Deletes pixels MoGe flags as invalid (sky) |
| Projector / Shot camera | Projector / Shot Camera | 2.5D camera projection |

**Not implemented in v0.1:** HDRI Studio / 360° depth sphere, floor
reflections and puddles, shadow-catcher "invisible mode", seamless textures,
surface presets, 3D-printing tools (lithophane, STL frame), video and batch
processing, tilt correction. The geometry is a normal mesh, so Blender's own
STL exporter works.

---

## Limitations

* **One image cannot show what is behind things.** Moving the Shot camera
  reveals holes behind foreground objects (the dark band in the GIF). Keep
  moves small, or paint or inpaint those areas.
* **Relative-depth models have an unknown scale and shift.** With Depth
  Anything, set the FOV yourself and tune Near/Far. MoGe-2 solves both, but
  its FOV estimate can be off for paintings with non-photographic
  perspective.
* **Wrong FOV = skewed room.** If you know the real lens, enter it (e.g.
  phone main camera ≈ 65–75° across the long side).
* Rebuilds run in Python: about 0.1 s at resolution 384 and about 0.8 s at
  1024 on a 2000 px image. Turn off *Live Update* while dragging if the
  mesh is very dense.

---

## Development

```
depth_scene/          the add-on (Blender extension, manifest + bl_info)
  core/               NumPy-only maths: resampling, depth mapping, unprojection,
                      stretched faces, MoGe focal/shift recovery  (no bpy)
  ai/                 ONNX Runtime: model registry, inference, downloads (no bpy)
  builder.py          rig / mesh / cameras / material in Blender
  properties.py       live per-object settings  ·  operators.py  ·  ui.py
  preferences.py      ONNX Runtime install + model downloads
tests/                pytest: core maths, ONNX plumbing (fake models), and Blender
                      integration (needs `pip install bpy`)
tools/build_zip.py    package the extension
tools/render_demo.py  render the README images (needs bpy)
docs/RESEARCH.md      model survey, maths, validation
```

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy onnx onnxruntime pytest bpy     # bpy = Blender as a Python module
python -m pytest tests/
```

## Licence

Add-on code: GPL-3.0-or-later, as required for Blender add-ons and declared
in `blender_manifest.toml`. Model weights are downloaded from their authors
and keep their own licences: DA2-Small Apache-2.0, DA2-Base/Large
CC-BY-NC-4.0, MoGe-2 MIT. The MoGe focal/shift recovery in
`core/moge_post.py` is a NumPy port of MIT-licensed code from
[microsoft/MoGe](https://github.com/microsoft/MoGe).
