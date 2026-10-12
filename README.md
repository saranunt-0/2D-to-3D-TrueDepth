# Depth Scene (Marigold edition) — turn one 2D image into a 3D scene in Blender

> **Branch `marigold`.** This variant ships **only** the
> [Marigold](https://github.com/prs-eth/Marigold) depth model
> (`prs-eth/marigold-depth-v1-1`). The `main` branch has the lighter models
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
                                          │
                     [2] geometry ◄───────┘   grid mesh, unprojected or displaced
                          │
                     [3] shading                image re-projected from the camera,
                                                optional bump / lighting
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
*Principled* shading reacts to lights and enables **Bump** for fine,
shader-only detail.

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

---

## 2. Licence: can I use Marigold commercially?

**Yes, with conditions.** Checked on 2026-10-12 against the
[repo](https://github.com/prs-eth/Marigold) and its `LICENSE-MODEL.txt`:

| Part | Licence | Commercial use |
|---|---|---|
| Marigold code (`prs-eth/Marigold`) | Apache-2.0 | ✅ |
| Weights `prs-eth/marigold-depth-v1-1` | **CreativeML Open RAIL++-M** | ✅ with use-based restrictions |
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
4. **Convert the model (one time, outside Blender).** Marigold is published
   as diffusers/PyTorch weights, so convert it with any Python 3.10+:

   ```bash
   pip install torch diffusers transformers accelerate onnx onnxruntime onnxconverter-common
   python tools/export_marigold_onnx.py --check --fp16
   ```

   * It downloads `prs-eth/marigold-depth-v1-1` from Hugging Face and writes
     `marigold_depth_v1_1_fp32/`: `vae_encoder.onnx`, `unet.onnx` +
     `unet.onnx.data`, `vae_decoder.onnx` and `marigold.json`, about 3.6 GB.
   * `--fp16` also writes `marigold_depth_v1_1_fp16/` (about 2.0 GB) with a
     half-precision U-Net. It is smaller and was faster in our CPU test too.
   * `--check` compares the add-on's NumPy denoising loop against the
     official diffusers pipeline.

   Copy the folder(s) into the add-on's models folder (shown in the
   preferences; you can also set your own).

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
4. **3. Build**: choose **Perspective** → **Build 3D Scene**.
5. Tune in **Reconstruction** (live):
   * walls bowed → adjust **Far** (Linear mapping);
   * wrong perspective → set the **FOV** of the original lens or painting;
   * jagged edges around doors → lower **Stretch Threshold**;
   * noisy surfaces → **Smooth** 0.5–1 % or Ensemble 3–5.
6. **Camera / 2.5D** → **Add Camera Move** → *Dolly In* → render.

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
  * On a GPU it takes seconds.
  * Details are in [docs/RESEARCH.md](docs/RESEARCH.md).
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
                      stretched faces, DDIM scheduler port   (no bpy)
  ai/                 ONNX Runtime: Marigold pipeline + registry   (no bpy)
  builder.py · properties.py · operators.py · ui.py · preferences.py
tests/                pytest: core maths, Marigold loop with fake ONNX graphs,
                      DDIM parity vs diffusers (if installed), Blender integration
tools/export_marigold_onnx.py   convert the checkpoint (--check, --fp16, --random-init)
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
