# Research notes — model choice, reconstruction maths, validation

Date: 2026-10-08. Evidence labels: **[fact]** checked against a primary
source · **[strong]** consistent across several secondary sources ·
**[inference]** my reasoning · **[unverified]** could not be tested in the
development sandbox.

## 0. Branch `marigold` (2026-10-12)

This branch ships **only** Marigold depth
([prs-eth/Marigold](https://github.com/prs-eth/Marigold), Ke et al., CVPR 2024).
Depth Anything V2, MoGe-2 and Illustrator's Depth were removed. Sections 1–4
below record the original model survey from `main`, for context.

### 0.1 Licence [fact, primary sources]

| Item | Licence | Source |
|---|---|---|
| Code | Apache-2.0 | `LICENSE.txt`; README "🎫 License" |
| Weights (`marigold-depth-v1-1`) | Open RAIL++-M (dated 2024-11-22, "based on the CreativeML Open RAIL-M") | `LICENSE-MODEL.txt`; HF card licence field `openrail++` |
| Base model (Stable Diffusion 2) | CreativeML Open RAIL++-M | Marigold README (fine-tuned from SD2) |

* §2 grants a perpetual, royalty-free licence to reproduce, distribute and
  sublicense. There is no non-commercial clause, and §4 explicitly allows
  hosting as a service. **Commercial use is allowed.**
* §4 conditions apply on redistribution of the Model or Derivatives (ONNX
  conversions count): pass on the §5 use restrictions as an enforceable
  term, include the licence, mark changed files, and keep notices.
* §6: the licensor claims no rights in outputs.
* ⚠️ Attachment A (the list of restrictions) in Marigold's
  `LICENSE-MODEL.txt` is an **unfilled template** ("[insert use
  restrictions]"). The SD2 Open RAIL++-M it derives from does list them, and
  a derivative of SD2 must carry them, so treat SD2's list as binding
  [inference]. This needs a legal check before shipping weights in a
  product.
* The HF card for `marigold-depth-hr-v1-1` says Apache-2.0 while the repo
  says RAIL++-M, which is inconsistent [fact]. Assume RAIL++-M.
* "Marigold V2" (Huawei, arXiv 2609.08084, Sept 2026) is a different
  ~20 B-parameter model built on Qwen-Image-Edit. It is out of scope for
  in-Blender use.

### 0.2 How it runs in the add-on

The diffusers `MarigoldDepthPipeline` (read from source) is:

1. image × 2 − 1;
2. resize the longest edge to the processing resolution (antialiased);
3. replicate-pad bottom/right to a multiple of 8;
4. VAE-encode, taking the posterior mode × 0.18215;
5. take a noise latent; for each DDIM step, the U-Net receives
   [image latent ‖ depth latent] (8 channels), the timestep, and a 2-token
   empty-prompt CLIP embedding;
6. VAE-decode, channel mean, clip to [−1, 1], then (x+1)/2;
7. unpad;
8. if Ensemble > 1, align the runs (SciPy) and take the median.

The output is affine-invariant depth: the paper normalises *depth* between
its 2nd and 98th percentiles, so 0 = near and 1 = far.

`tools/export_marigold_onnx.py` exports `vae_encoder`, `unet` (with the empty
embedding baked in) and `vae_decoder` as ONNX, plus `marigold.json` with the
checkpoint's own scheduler numbers. `core/ddim.py` ports `DDIMScheduler`
(`set_timesteps` + `step`, eta 0). `ai/marigold.py` re-implements the loop.
Ensembling uses an iterative least-squares scale/shift alignment because
Blender has no SciPy [approximation].

### 0.3 Validation [measured; real architecture, random weights]

* **DDIM port vs diffusers `DDIMScheduler`:**
  * covers epsilon / v / sample prediction × leading / trailing / linspace
    spacing, with and without zero-SNR, at 1–50 steps;
  * timesteps are identical;
  * worst relative state error is 1.0e-6.
* **Export:** U-Net 866 M + VAE 84 M parameters. The fp32 folder is
  3.6 GB (U-Net in `unet.onnx.data`) and exports in ~2 min. The fp16 folder
  is 2.0 GB, using the same >2 GB shape-inference and Cast fixes as the
  Depth Pro export.
* **End to end:** the add-on's NumPy/ONNX loop vs the diffusers pipeline,
  from identical noise (2 steps, 384×256): max |diff| **5.6e-6**, with the
  same runtime (8.3 s vs 8.4 s).
* **CPU cost** (4 cores, ORT 1.30 inside Blender 5.2's Python, 1 step):
  * 768 px: 32 s;
  * 384 px: 8.9 s fp32, 5.4 s fp16 (max depth difference 0.008);
  * peak RSS ≈ 9 GB.

### 0.4 Unverified

- [ ] Real checkpoint conversion (Hugging Face is blocked in the sandbox):
  `python tools/export_marigold_onnx.py --check --fp16`. Expect max |diff|
  < 1e-4 vs diffusers.
- [ ] Depth quality on the corridor painting and the illustration.
- [ ] GPU providers (CUDA / DirectML / CoreML) with the fp16 U-Net.

## 1. Requirements

1. Runs **inside Blender's bundled Python**, which has NumPy but no
   PyTorch, Pillow or SciPy. A dependency of ~2 GB (PyTorch + CUDA) is not
   "lightweight".
2. Runs **locally on CPU**, using the GPU when present.
3. Licence usable for commercial work by default (TrueDepth is a paid
   product, and the user may want the same freedom).
4. Good **geometry**, not just pretty depth maps: for corridor shots, walls
   and floors must come out planar, and the camera FOV must be right.

Requirement 1 decides the runtime: **ONNX Runtime**. The 1.30 wheels are
14–24 MB, it has CPU, CUDA, DirectML (Windows) and CoreML (macOS) providers,
and it only needs NumPy. Caveat [fact, PyPI]: 1.30 publishes macOS wheels
only for arm64 / macOS 14+, so Intel Macs resolve to an older release.

## 2. Candidates

| Model | Params | Licence | Output | FOV? | ONNX | Notes |
|---|---|---|---|---|---|---|
| Depth Anything V2 Small | 24.8 M [fact] | Apache-2.0 [fact] | affine-invariant inverse depth | no | community (`onnx-community/depth-anything-v2-small`, dynamic H/W, fp32/fp16/int8) [strong] | Mature and fast with sharp edges. Same S/B/L ladder as TrueDepth's "Model Size" [inference] |
| Depth Anything V2 Base / Large | 97.5 M / 335.3 M [fact] | **CC-BY-NC-4.0** [fact] | same | no | community, Large fp32 1.34 GB, fp16 669 MB [strong] | Non-commercial only |
| **MoGe-2 ViT-S** (normal) | 35 M [fact] | MIT code [fact]; weights per HF card | affine point map + metric scale + mask + normals → **metric depth + FOV** | **yes** | **official** (`Ruicheng/moge-2-vits-normal-onnx`, dynamic H/W, variable tokens) [fact] | Built for monocular *geometry*. Focal/shift recovery is not in the ONNX graph (ported here) |
| MoGe-2 ViT-B | 104 M [fact] | MIT | same | yes | official | Slower, better |
| Depth Anything 3 Small / Base | 0.08 B / 0.12 B [fact] | Apache-2.0 [fact] | depth + ray maps (intrinsics) | yes | no official ONNX; community exports with differing I/O [strong] | Promising. "Any-view" model; the monocular-specialised DA3Mono is Large only |
| DA3Mono-Large / DA3Metric-Large | 0.35 B [fact] | Apache-2.0 [fact] | relative / metric depth | no | community | Too heavy for "lightweight" |
| MoGe-3 (Aug 2026) | 370 M+ [fact] | MIT | point maps | yes | none; needs FlexGEMM/Triton, no macOS [fact] | Not deployable in Blender |
| Apple Depth Pro | two ViT-L encoders | Apple licence | metric + focal | yes | community | Heavy |

Independent benchmark data for "lightweight" depth is thin. The one
cross-model study found (a wildlife-monitoring benchmark) placed Depth
Anything V2 at the best accuracy/latency trade-off among the models it
tested [strong]. Papers on tiny KITTI-trained models (LMDepth, Lite-Mono
derivatives) are narrow-domain and self-reported, so they are not suitable
for arbitrary paintings and photos [inference].

### Decision

* **Default: Depth Anything V2 Small.** It is the smallest well-supported
  model, Apache-2.0, with a stable ONNX export and dynamic shapes.
* **Recommended for corridors: MoGe-2 ViT-S.** Only 10 M parameters more,
  MIT, official ONNX. It solves the two things relative-depth models cannot:
  the unknown **shift**, which bends walls, and the unknown **FOV**, which
  skews rooms.
* DA2 Base/Large and MoGe-2 ViT-B are offered as options. The UI flags the
  NC licence.
* **Not now:** DA3. There is no official ONNX, community exports disagree
  on inputs/outputs, and the light variants are "any-view" models. Revisit
  when an official ONNX export appears; it would slot into `ai/infer.py`
  as a third family.

### 2.1 Addendum (2026-10-10): Illustrator's Depth

Added at the user's request as an optional, manually converted model.

* **What it predicts** [fact, from the repo source]: a per-pixel *layer
  index*. Ground truth is each SVG element's document order (later-drawn =
  higher index = on top). Training uses an L1 loss on median/mean-deviation
  normalised maps of the same sign, so the output grows toward the front. It
  is an ordering, not a distance. The add-on stores it as kind `layers`
  (normalised like disparity, mapped LINEAR).
* **Architecture** [fact]: Apple Depth Pro's encoder (patch + image DINOv2
  ViT-L), decoder and head. The FOV head is dropped. Forward returns
  `-1/clamp(head, 1e-4, 1e4)`. Input is a plain resize to 1536×1536 with
  Normalize(0.5, 0.5). The add-on clips the 0.5/99.5 percentiles because
  the clamp can saturate.
* **Licence** [fact]: Adobe Research License, noncommercial research only
  ("academic research and teaching"; excludes "development of commercial
  products"). That is stricter than CC-BY-NC. The add-on ships no Adobe
  code or weights. Users run `tools/export_illustrators_depth_onnx.py`
  inside the authors' environment, and the resulting ONNX inherits the
  restriction.
* **Weights** are a PyTorch Lightning checkpoint on Google Drive (no ONNX
  release, no Hugging Face mirror). They were unreachable from the
  development sandbox, so the checkpoint-loading step is **unverified**.

## 3. Reconstruction maths

### 3.1 Unprojection (Perspective mode)

Blender cameras with sensor fit AUTO define the FOV across the longer image
side. For a grid vertex at normalised image position (u, v) (u: 0 left → 1
right, v: 0 top → 1 bottom) and planar distance z:

```
hw = tan(FOV/2)·W/max(W,H),  hh = tan(FOV/2)·H/max(W,H)
P  = ( (u−½)·2hw·z,  (½−v)·2hh·z,  −z )        # camera space, looking down −Z
```

Each vertex lies on its pixel's ray, so from the projector camera the UV
mapping is exact at the vertices. The mesh and projector camera share one
parent (the rig empty, rotated +90° about X so camera-space −Z becomes world
+Y and the scene "stands up").

### 3.2 Depth mapping and the shift problem

Depth Anything predicts disparity d ≈ a/z + b with unknown a (scale) and
b (shift). After min/max normalisation to t ∈ [0, 1], the *Inverse* mapping
`1/z = t/near + (1−t)/far` is affine in d. It reproduces the true geometry up
to scale only when far/near matches the true depth ratio, i.e. when the
implied shift equals b. A wrong ratio makes planes curve. The add-on
initialises `far/near = d_max/d_min` (the "b = 0" guess, using 0.5/99.5
percentiles, clamped to [1.5, 100]) and exposes Near/Far for tuning.

MoGe-2 predicts an affine point map p with true point = (p + (0, 0, shift))·s.
For a given shift the best focal has a closed form,
`f = Σ(xy/(z+shift)·uv) / Σ(xy/(z+shift))²`, so recovery is a **1-D
minimisation over shift**. `core/moge_post.py` solves it with a log-spaced
scan plus golden-section refinement. That is NumPy only, replacing MoGe's
SciPy Levenberg–Marquardt solver, and does not depend on a starting guess.

### 3.3 Remove Stretched Faces as an angle test

For a plane with unit normal n, viewed along the (unnormalised) ray
r = (x, y, 1):

```
z = c/(n·r)   ⇒   |∇_xy log z| = |n_xy| / |n·r|
cos φ = |n·r| / |r|  (φ = angle between ray and surface normal)
⇒ |∇ log z|·|r| = 1/cos φ   (exact when n ⟂ optical axis, a lower bound otherwise)
```

A face is deleted when `|∇ log z|·|r| > 1/cos(threshold)`. Because it is
an *angle*, the test gives the same answer at any grid resolution for real
surfaces. Occlusion edges are depth jumps across a single face, which appear
as near-90° surfaces and get removed. Real grazing surfaces (corridor walls)
survive except within a few faces of the vanishing point.

Calibration on the synthetic corridor, with the depth blurred like a 518 px
model output:

| threshold | false removals, clean corridor (128 / 256 / 512 grid) | removals at the pillar edges (256 grid) |
|---|---|---|
| 80° | 900 / 3792 / 15086 ❌ | 3121 |
| 85° | 4 / 0 / 0 | 532 |
| **86° (default)** | **0 / 0 / 0** | 359 |
| 88° | 0 / 0 / 0 | 228 |

### 3.4 Detail and bump

* *Detail Displacement* is a high-pass of image luminance
  (`gray − gaussian(gray, r)`), moved along the view ray (Perspective) or +Z
  (Relief). The luminance is first downsampled to 2× the grid resolution,
  because finer detail would only alias.
* *Bump* is a shader Bump node fed by image luminance. It changes shading
  only and has no geometry cost.

## 4. Validation

All on Blender **4.2.23 LTS** (Python 3.11, NumPy 1.26) and **5.2.2 LTS**
(Python 3.13), via `pip install bpy`. 35 tests pass on both.

* Analytic corridor: perspective reconstruction from metric depth puts the
  median vertex < 1 mm from the true planes (90th percentile < 2 cm, at
  seams).
* Affine disparity (`3/z + 0.25`) with the correct near/far reproduces depth
  to 1e-4 relative error.
* MoGe post-processing on synthetic affine point maps recovers the FOV to
  within 0.2–0.5° and depth to 0.5 %.
* **Projector-render check:** a Cycles render through the Projector camera
  reproduces the source image (median per-pixel error < 2 %). This verifies
  UVs, FOV, sensor fit and colour management end to end.
* The real `pip install onnxruntime` into the add-on's private folder works
  (≈6 s) and leaves the host NumPy untouched.
* The extension zip passes Blender's own `blender_ext.py validate` (4.2 and
  5.2) and installs and enables through `extensions.package_install_files`.

### Unverified items (sandbox had no Hugging Face access)

- [ ] **Real Depth Anything V2 Small weights** via *Download* → Generate on
  `1.webp` (corridor painting). Expect a 518×710 disparity map in about
  0.3–1 s on CPU.
- [ ] **Real MoGe-2 ViT-S weights**: check output tensor names
  (`points/normal/mask/metric_scale`), that the FOV estimate is plausible
  (≈50–70° for the corridor painting), and that the walls look straight.
- [ ] fp16 DA2 file on CPU and GPU, plus the CUDA / DirectML / CoreML
  providers.
- [ ] UI interaction in a real Blender window (panels were checked with a
  validating fake layout, not clicked).

### Unverified items (Illustrator's Depth)

- [ ] Real checkpoint → `export_illustrators_depth_onnx.py --check` (expected:
  ONNX vs PyTorch relative difference < 1e-3).
- [ ] Output direction on a real image: front shapes should get *higher*
  values (`kind = layers`). If the relief comes out inverted, tick *Invert*
  and report it.
- [ ] fp16 export accuracy and speed on a GPU.

## Sources

- Marigold code, licences, news: https://github.com/prs-eth/Marigold · weights: https://huggingface.co/prs-eth/marigold-depth-v1-1 · paper: https://arxiv.org/abs/2312.02145
- diffusers Marigold pipeline (source read for the port): https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/marigold
- Marigold V2 (Huawei, different model): https://github.com/huawei-bayerlab/marigold-v2

- Depth Anything V2 README (params, licence, input size 518): https://github.com/DepthAnything/Depth-Anything-V2
- onnx-community DA2 exports: https://huggingface.co/onnx-community/depth-anything-v2-small · https://huggingface.co/onnx-community/depth-anything-v2-large/tree/main/onnx
- DA2 ONNX preprocessing notes (mirror card): https://huggingface.co/inference4j/depth-anything-v2-small
- MoGe (code, model table, licence, MoGe-3 news): https://github.com/microsoft/MoGe · ONNX notes: https://github.com/microsoft/MoGe/blob/main/docs/onnx.md
- MoGe-2 paper: https://arxiv.org/abs/2507.02546 · ONNX weights: https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx
- Depth Anything 3 (model table, licences): https://github.com/ByteDance-Seed/Depth-Anything-3 · https://huggingface.co/depth-anything/DA3-BASE
- Community DA3 ONNX examples: https://huggingface.co/Heliosoph/da3-base-onnx · https://github.com/devin-lai/Depth-Anything-3-Onnx · https://huggingface.co/qualcomm/Depth-Anything-V3
- Cross-model benchmark (wildlife): https://arxiv.org/html/2510.04723v1
- Illustrator's Depth (code, licence, model): https://github.com/nissmar/illustrators_depth · paper: https://arxiv.org/abs/2511.17454
- Apple Depth Pro (backbone): https://github.com/apple/ml-depth-pro
- TrueDepth: https://superhivemarket.com/products/truedepth · https://www.creativeshrimp.com/true-depth-blender-add-on-the-complete-guide-v3.html
