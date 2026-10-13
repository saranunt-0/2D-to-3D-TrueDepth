# Depth Scene (Marigold edition) — turn one 2D image into a 3D scene in Blender

> **Branch `marigold`.** This variant ships **only**
> [Marigold](https://github.com/prs-eth/Marigold) models: depth
> (`prs-eth/marigold-depth-v1-1`) builds the mesh, and the optional normals
> and intrinsic-image models (`marigold-normals-v1-1`,
> `marigold-iid-appearance-v1-1`, `marigold-iid-lighting-v1-1`) turn the image
> into a lit PBR material. The `main` branch has the lighter models
> (Depth Anything V2, MoGe-2) and Illustrator's Depth instead.

A Blender add-on that estimates depth from a single image with Marigold, a
diffusion-based AI depth model, and rebuilds the picture as real 3D geometry. You can then move a camera
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
   │                                      │
   │                 [2] geometry ◄───────┘   grid mesh, unprojected or displaced
   │                      │
   └─► [1b] material  ─► [3] shading            image re-projected from the camera;
       passes (optional)                        unlit, or lit with normal / albedo /
       normal, albedo, roughness,               roughness / metallic maps, lights
       metallic, shading, highlights            and cast shadows
```

**Stage 1: depth.** Marigold runs locally through ONNX Runtime (CPU or GPU,
no PyTorch in Blender) and produces a depth map. It is saved as a 32-bit EXR
so you can reuse or edit it.

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
*Principled* shading reacts to lights and casts shadows; with the optional
**material passes** (stage 1b, [below](#material-passes-lighting-and-shadows))
it uses an AI normal map, a de-lit albedo and roughness / metallic maps.

**Cameras (2.5D).** Each build creates two cameras under a rig empty:

* a locked **Projector** camera, at the viewpoint the image was taken from;
* a **Shot** camera that you animate.

Small Shot-camera moves give convincing parallax. Large moves reveal what the
photo never showed. Move the rig empty to place, rotate or scale the whole
scene.

### How Marigold works, and what that means for the settings

Marigold ([Ke et al., CVPR 2024](https://arxiv.org/abs/2312.02145)) is Stable
Diffusion 2 fine-tuned to *generate* a depth map for the input image. One
estimate runs in four stages:

1. The VAE encoder turns the image into a latent.
2. The U-Net denoises a random "depth latent" over a few **Denoising Steps**,
   conditioned on that image latent.
3. The VAE decoder turns the result back into a depth map.
4. Optionally, an **Ensemble** of runs with different noise is aligned and
   combined.

The add-on re-implements the diffusers pipeline around three ONNX graphs. The
DDIM scheduler is ported to NumPy and matches diffusers to 1e-6.

* **Output:** *affine-invariant depth*, 0 = near … 1 = far, with unknown
  scale and shift. The add-on maps it with **Linear** depth mapping:
  `z = near + (far − near) · depth`. Only the near/far *ratio* changes the
  shape. If walls bow, adjust **Far**.
* **Denoising Steps:** v1.1 is trained with zero-SNR, trailing timesteps for
  1–4 steps; 1 is usually enough. 0 uses the checkpoint default.
* **Processing Resolution:** default 768 px on the longer side (what SD2 was
  trained at). Lower is faster but less detailed.
* **Ensemble / Seed:** results vary slightly with the noise seed. An
  ensemble of N averages N runs and is N× slower.

| Mapping | Formula (t = 1 nearest … 0 farthest) | Use for |
|---|---|---|
| **Linear**  | z = t·near + (1−t)·far | Marigold (default), "white = far" maps |
| **Inverse** | 1/z = t/near + (1−t)/far | disparity maps from other tools |
| **Metric**  | z = depth × scale | metric EXRs from a renderer |

### Material passes: lighting and shadows

A mesh rebuilt from one image is coarse, and the image already contains the
painter's (or the sun's) lighting. Lighting it again in Blender works much
better with three more Marigold models, each toggled in **3. Material
Passes**:

| Toggle | Model | Maps | Used for |
|---|---|---|---|
| **Normal Map** | `marigold-normals-v1-1` | camera-space normals | lights shade detail the mesh is too coarse to show |
| **Albedo / Roughness / Metallic** | `marigold-iid-appearance-v1-1` | albedo (sRGB→linear), roughness, metallicity | de-lit base colour + PBR surface response |
| **Shading / Highlights** | `marigold-iid-lighting-v1-1` | albedo, diffuse shading, non-diffuse residual | keep painted soft shadows / glows on purpose |

![Relighting the rebuilt corridor](docs/images/demo_relight.png)

*Left: unlit (Emission) = the source image. Middle: Principled shading, one
warm spot light; the pillar and walls cast real shadows. Right: a cool light
moved down the corridor, with Original Lighting 0.3. Synthetic test scene
with its ground-truth normal map; the real Marigold models could not be
downloaded in the development sandbox.*

How the maps are wired (Principled BSDF, rebuilt live on every change):

* **Normal map → Object space.** Marigold predicts normals in the camera's
  frame (x right, y up, z towards the viewer). The mesh is built in exactly
  that frame (it sits under the rig like the projector camera), so the map
  plugs into a Normal Map node in *Object* space with no conversion.
  Verified by rendering: shading with a ground-truth normal map differs from
  the true geometry by 0.002 (mean abs), while a flipped-X map differs by 0.50.
  **Strength** blends mesh normals (0) and map normals (1); >1 exaggerates.
  *Space: Tangent* and **Flip X / Y** are there for normal maps from other
  tools. **Bump** (image luminance) stacks on top.
* **Base colour** = image → albedo (**Albedo Mix**, **Albedo Gain**) →
  × **Baked Shading** (multiply). The albedo has the lighting removed, so
  painted shadows are not lit twice. Baked Shading puts back the soft
  shadows / ambient occlusion that the coarse mesh cannot cast.
* **Roughness** map remapped to **Min…Max**; **Metallic** map × **Strength**.
  Each falls back to a constant when its map is off.
* **Baked Highlights**: the IID-Lighting *residual* (specular highlights,
  lamps, glows) is added as emission × **Strength**.
* **Original Lighting** blends the whole material towards the unlit image
  (1 = looks exactly like the source, ignores lights).
* **Cast Shadows** toggles the object's shadow visibility. **Add Light**
  (Point / Area / Spot / Sun) drops a light in front of the projector, scaled
  to the scene's Near / Far.

IID-Lighting predicts its shading and residual only up to scale. The add-on
calibrates them with a 2-parameter, non-negative least-squares fit of
`image ≈ albedo × shading + residual` (the decomposition the model is trained
on). The fit error is shown in the status bar, and a large error means the
decomposition does not explain this image well. All maps stay at the
processing resolution; the mesh samples them through its image UVs, so they
need no resampling.

---

## 2. Licence: can I use Marigold commercially?

**Yes, with conditions.** Checked on 2026-10-12 (depth) and 2026-10-13
(normals, IID) against the [repo](https://github.com/prs-eth/Marigold), its
README ("The models are licensed under RAIL++-M") and `LICENSE-MODEL.txt`:

| Part | Licence | Commercial use |
|---|---|---|
| Marigold code (`prs-eth/Marigold`) | Apache-2.0 | ✅ |
| Weights `prs-eth/marigold-depth-v1-1` | **CreativeML Open RAIL++-M** | ✅ with use-based restrictions |
| Weights `marigold-normals-v1-1`, `marigold-iid-appearance-v1-1`, `marigold-iid-lighting-v1-1` | **CreativeML Open RAIL++-M** (repo README; a search snippet of the iid-lighting Hugging Face card says the same; the cards themselves could not be opened from the sandbox) | ✅ with use-based restrictions |
| Base model it was fine-tuned from (Stable Diffusion 2) | CreativeML Open RAIL++-M | ✅ with use-based restrictions |
| This add-on's code | GPL-3.0-or-later | ✅ |

What Open RAIL++-M asks of you:

* **No non-commercial clause.** It grants a royalty-free licence to use,
  reproduce and distribute, including as a hosted service. The licensor
  claims no rights in your outputs (the depth maps, meshes and renders).
* **Use-based restrictions** (§5, Attachment A) must be respected. If you
  *redistribute* the weights or the converted ONNX files (a "Derivative of
  the Model"), you must:
  * include the licence;
  * pass the restrictions on as an enforceable term;
  * mark modified files.
* ⚠️ Marigold's own `LICENSE-MODEL.txt` leaves Attachment A as a template
  placeholder ("[insert use restrictions]"). The concrete list therefore
  comes from Stable Diffusion 2's Open RAIL++-M, which Marigold derives from.
  It prohibits illegal use, harming minors, disinformation, harassment,
  discriminatory or medical-advice uses, and so on.
* The add-on never ships Marigold weights. You convert them yourself, so
  just *using* the add-on is unaffected. If you **sell** a product that
  bundles the converted ONNX files, the redistribution conditions above
  apply. Have a lawyer confirm for a commercial product.
* Don't confuse it with **Marigold V2** (Huawei, Sept 2026). That is a
  different, ~20 B-parameter model built on Qwen-Image-Edit, and far too
  large to run inside Blender.

---

## 3. Installation

Requires **Blender 4.2 LTS or newer**. Tested with Blender 4.2.23 LTS and
5.2.2 LTS.

1. **Get the add-on zip.** Either compress the `depth_scene` folder of this
   branch, or run `python tools/build_zip.py`.
2. **Install it.** In Blender, open *Edit → Preferences → Get Extensions*,
   then the **⌄** menu → **Install from Disk…** and pick the zip.
3. **Install the AI runtime.** Open *Preferences → Add-ons → Depth Scene*,
   choose the runtime and click **Install**:
   * **CPU** works everywhere;
   * **CUDA** needs an NVIDIA GPU with CUDA 12 + cuDNN 9;
   * **DirectML** works with any DirectX 12 GPU on Windows.

   Marigold is heavy, so a GPU is strongly recommended.
4. **Convert the models (one time, outside Blender).** Marigold is
   published as diffusers/PyTorch weights, so convert it with any Python
   3.10+:

   ```bash
   pip install torch diffusers transformers accelerate onnx onnxruntime onnxconverter-common
   python tools/export_marigold_onnx.py --task depth --check --fp16
   # optional material passes:
   python tools/export_marigold_onnx.py --task normals --check --fp16
   python tools/export_marigold_onnx.py --task iid-appearance --check --fp16
   python tools/export_marigold_onnx.py --task iid-lighting --check --fp16
   ```

   * Each run downloads the task's `prs-eth/marigold-*-v1-1` checkpoint from
     Hugging Face and writes `marigold_<task>_v1_1_fp32/`:
     `vae_encoder.onnx`, `unet.onnx` + `unet.onnx.data`, `vae_decoder.onnx`
     and `marigold.json`, about 3.6 GB.
   * `--fp16` also writes the `_fp16/` folder (about 2.0 GB) with a
     half-precision U-Net. It is smaller and was faster in our CPU test too.
   * `--check` compares the add-on's NumPy denoising loop against the
     official diffusers pipeline.

   Copy the folder(s) into the add-on's models folder (shown in the
   preferences; you can also set your own). Only the depth model is
   required; each pass needs only its own folder.

> You can use the add-on **without AI**: tick *Use Existing Depth Map* and
> load a depth map made elsewhere.

---

## 4. Quick start (corridor shot)

1. 3D Viewport → press **N** → **Depth Scene** tab.
2. **1. Image**: open your picture (JPG/PNG/WebP/EXR).
3. **2. Depth**: *Marigold Depth v1.1*, *Device: Auto*. Use *Precision:
   Float16* on a GPU and Float32 on CPU. Set *Processing Resolution 768*,
   *Denoising Steps 0* (the default, 1 step) and *Ensemble 1*.
   Optionally click **Generate Depth Map** to preview.
4. **3. Material Passes** (optional): tick **Normal Map** and **Albedo /
   Roughness / Metallic** (and **Shading / Highlights** if you want to keep
   painted shadows).
5. **4. Build**: choose **Perspective** → **Build 3D Scene**. Build also
   runs the ticked passes and switches the material to *Principled*. For an
   existing scene, use **Generate for Active Object**.
6. **Lights & Shadows** → **Add Light** (Point / Area / Spot / Sun) and move
   it. Tune **Material** (live): Normal **Strength**, **Albedo Mix**,
   roughness **Min / Max**, **Baked Shading**, **Original Lighting**…
7. Tune in **Reconstruction** (live):
   * walls bowed → adjust **Far** (Linear mapping);
   * wrong perspective → set the **FOV** of the original lens or painting;
   * jagged edges around doors → lower **Stretch Threshold**;
   * noisy surfaces → **Smooth** 0.5–1 % or Ensemble 3–5.
8. **Camera / 2.5D** → **Add Camera Move** → *Dolly In* → render.

---

## 5. Parameters (and their TrueDepth counterparts)

| Depth Scene | TrueDepth | What it does |
|---|---|---|
| Model (Marigold v1.1) | Model Size | Depth network |
| Device Auto/GPU/CPU | Processing Device | Execution provider |
| Precision fp32/fp16 | Precision | fp16 halves the U-Net (2.0 vs 3.6 GB) and was faster even on CPU; max depth difference 0.008 |
| Processing Resolution | (implicit) | Longer side fed to Marigold (768 default, 0 = native) |
| Denoising Steps | — | Diffusion steps (0 = checkpoint default; 1–4 for v1.1) |
| Ensemble / Seed | — | Average N noisy runs / make results repeatable |
| Use Existing Depth Map | (import) | Skip AI and use your own depth map |
| Save Location | Save Location | Where depth EXRs go (`//` = next to the .blend) |
| Mode Perspective / Flat Relief | Perspective Mode / Flat Relief | Unprojection vs plane displacement |
| Resolution | Resolution / Grid Resolution | Vertices along the longer side (default 384) |
| Field of View | FOV | Camera FOV across the longer side, used for unprojection |
| Depth Mapping Linear/Inverse/Metric | Depth Mapping Linear/Inverse | See the table above |
| Near / Far | Near / Far Distance | Distance of the nearest / farthest pixel |
| Remove Stretched Faces + Threshold | Remove Stretched Faces | Deletes faces seen steeper than the threshold angle |
| Base Displacement | Base Displacement | Relief depth |
| Detail Displacement + Radius | Detail Displacement | Real geometry from image-luminance high-pass |
| Bump | Bump / Detail Bump | Shader-only detail (Principled shading) |
| Material Passes: Normal Map / Albedo-Roughness-Metallic / Shading-Highlights | — | Extra Marigold models → material maps (same resolution / steps / seed as depth) |
| Normal Map: Strength, Space, Flip X/Y | — | Mesh (0) ↔ AI normals (1); camera (Object) or tangent space |
| Albedo: Mix, Gain | — | Original image ↔ de-lit albedo; brightness |
| Baked Shading: Strength | — | Multiply painted diffuse shading back in |
| Roughness: Min/Max (or constant) · Metallic: Strength (or constant) | — | PBR surface response |
| Baked Highlights: Strength | — | Painted specular / glow (IID residual) as emission |
| Original Lighting | — | Blend towards the unlit image |
| Cast Shadows · Add Light | — | Shadow visibility; quick light rig |
| Smooth | Smooth | Blurs the depth map before meshing |
| Quantize Layers | — | Snaps depth to N flat layers (paper cut-out / diorama) |
| Use Alpha Channel | Include Alpha Channel | Deletes transparent areas (PNG cut-outs) |
| Projector / Shot camera | Projector / Shot Camera | 2.5D camera projection |

**Not implemented:** HDRI Studio / 360°, floor reflections, shadow catcher,
seamless textures, surface presets, 3D-printing tools, video/batch, tilt
correction.

---

## Limitations

* **Heavy.** ~950 M parameters (SD2 U-Net 866 M + VAE 84 M).
  * The model folder is 3.6 GB at fp32 and 2.0 GB at fp16.
  * Measured on a 4-core CPU with 1 step: about **32 s at 768 px** and
    9 s at 384 px, with roughly 9 GB peak RAM. Each extra step or ensemble
    member adds U-Net time.
  * Each **material pass** is another model run. At 768 px on 4 CPU cores,
    one VAE decode is ~19 s and dominates: normals ≈ depth (~35 s),
    IID-Appearance ~55 s (2 maps decoded), IID-Lighting ~75–100 s (3 maps),
    ~10 GB peak RAM. Use 384 px to try settings, or a GPU.
  * On a GPU it takes seconds.
  * Details are in [docs/RESEARCH.md](docs/RESEARCH.md).
* **Intrinsics were trained indoors.** IID-Appearance (InteriorVerse) and
  IID-Lighting (Hypersim) learned from synthetic indoor renders. Expect weaker
  albedo / roughness on paintings and outdoor scenes. The **Albedo Mix** and
  **Original Lighting** sliders let you back off.
* **No FOV estimate.** Marigold predicts depth only, so set the FOV
  yourself for perspective scenes.
* **Ensemble alignment** uses a least-squares approximation, because Blender
  has no SciPy and diffusers uses a SciPy optimiser. With Ensemble = 1
  (the default) results match diffusers.
* **One image cannot show what is behind things.** Keep camera moves small.

---

## Development

```
depth_scene/          the add-on (Blender extension, manifest + bl_info)
  core/               NumPy-only maths: resampling, depth mapping, unprojection,
                      stretched faces, DDIM scheduler port, material-pass maps (no bpy)
  ai/                 ONNX Runtime: Marigold pipeline (depth / normals / intrinsics)
                      + registry   (no bpy)
  builder.py · properties.py · operators.py · ui.py · preferences.py
tests/                pytest: core maths, Marigold loop with fake ONNX graphs (all tasks),
                      DDIM + export parity vs diffusers (if installed), Blender integration
                      (incl. a normal-map convention render test)
tools/export_marigold_onnx.py   convert a checkpoint (--task, --check, --fp16, --random-init [--tiny])
tools/build_zip.py              package the extension
```

```bash
pip install numpy onnx onnxruntime pytest bpy
python -m pytest tests/          # + torch diffusers for the parity test
```

## Licence

Add-on code: GPL-3.0-or-later (see `blender_manifest.toml`). Marigold code:
Apache-2.0. Marigold weights: CreativeML Open RAIL++-M. You convert them
yourself; nothing from them ships here. See [section 2](#2-licence-can-i-use-marigold-commercially).
