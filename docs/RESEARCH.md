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

### 0.5 Material passes: normals and intrinsics (2026-10-13)

Goal: let the rebuilt scene react to Blender lights and shadows, with every
map toggleable and adjustable.

**Models [fact, prs-eth/Marigold repo + diffusers 0.41 source]:**

| Checkpoint | diffusers pipeline | U-Net in/out channels | Targets (decode → post-processing) |
|---|---|---|---|
| `marigold-normals-v1-1` | `MarigoldNormalsPipeline` | 8 / 4 | normals: clip [−1,1] → (z·½+½ if not `use_full_z_range`) → unit length |
| `marigold-iid-appearance-v1-1` | `MarigoldIntrinsicsPipeline` | 12 / 8 | `albedo` (srgb), `material` = stack [roughness, metallicity, –] (linear) |
| `marigold-iid-lighting-v1-1` | `MarigoldIntrinsicsPipeline` | 16 / 12 | `albedo` (linear), `shading` (linear, up to scale), `residual` (linear, up to scale) |

* Target properties come from `config/train_marigold_iid_{appearance,lighting}.yaml`.
  All three use 4 default steps and 768 px; normals and IID decode each
  target's 4-channel latent separately, then clip and map (x+1)/2 (IID).
* Normals frame: x right, y up, z towards the observer
  (`visualize_normals` docstring). Ensemble: "closest" (the member nearest
  the normalised mean, per pixel). IID ensemble: per-pixel median.
* Licence: the repo README says "The models are licensed under RAIL++-M" for
  all checkpoints [fact]. A search snippet of the iid-lighting HF card
  shows Open RAIL++-M [strong]. The normals / appearance HF cards could not be
  opened (HF blocked) [unverified].
* Training domain: IID-Appearance on InteriorVerse, IID-Lighting on
  Hypersim, both synthetic indoor renders [fact, README]. Quality on
  paintings / outdoor scenes is [unverified].

**Design decisions:**

* **Normal map in Object space, no conversion [inference, then measured].**
  The mesh lives in the rig in the projector camera's own frame (camera at
  the origin looking down −Z, +Y up), so camera-space normals *are*
  object-space normals. Tangent space would be wrong away from
  camera-facing planes. Measured in Cycles: ground-truth normal map vs true
  geometry, mean abs pixel difference 0.0024; X-flipped map 0.50
  (`test_camera_space_normal_map_matches_geometry_under_a_light`, both
  Blender versions).
* **IID-Lighting calibration [inference].** Hypersim's decomposition is
  linear `I = A·S + R`. Shading and residual are predicted up to scale, so a
  2-parameter non-negative least-squares fit against the (linearised) input
  image restores their scale. That makes *Baked Shading* (multiply) and
  *Baked Highlights* (emission) physically meaningful. The relative RMS of
  the fit is reported to the user.
* **Maps stay at processing resolution**; the mesh samples them through
  image UVs, so no upsampling step is needed.
* **One pass at a time, one session set cached**: peak memory stays at
  ~1 U-Net. Each target is decoded at batch 1 for the same reason.

**Validation [measured]:**

* Export parity vs diffusers, identical noise:
  * tiny random-init models, all four tasks: max |diff| ≤ 5.3e-5 (normals:
    normalisation amplifies float error where the vector is short; others
    ≤ 3.4e-6);
  * full-size IID-Lighting (U-Net 16→12 channels, random weights,
    384×256, 1 step): **9.6e-6**.
* CPU cost, 4 cores, 768 px, 1 step:
  * encoder ~9 s, U-Net ~10 s per step, VAE decoder **~19 s per target**;
  * IID-Lighting: 101 s at 768 px (incl. first-run warm-up) and 22.5 s at
    384 px, peak RSS 10.4 GB;
  * normals ≈ depth, IID-Appearance ≈ +1 decode.
* `tests/test_export.py` runs the four-task parity in CI when torch +
  diffusers are installed. Blender tests (4.2.23, 5.2.2) cover pass
  generation with fake graphs, reuse, per-pass failure isolation, node
  wiring for every toggle, shadows and Add Light.

**Unverified / follow-ups:**

- [ ] Real normals / IID checkpoints (HF blocked): convert with
  `--task normals|iid-appearance|iid-lighting --check`.
- [ ] Quality of IID on the user's corridor painting, NYC illustration and
  aspen landscape (outside the indoor training domain).
- [ ] Sharing the VAE-encoder latent across passes would save ~9 s per
  extra pass at 768 px. This is only valid if the four checkpoints ship
  byte-identical VAEs (expected, since Marigold freezes the VAE, but not
  checked).

### 0.6 Marigold V2 as an external backend (2026-10-13)

**What it is [fact, repo README + code at commit 9926d93]:** Marigold V2
(Pavlovic, Wandel, Obukhov et al., ACM TOG / SIGGRAPH Asia 2026,
arXiv 2609.08084).

* **Architecture:** LoRA adapters (and, for some checkpoints, a fine-tuned
  VAE decoder) on the frozen Qwen-Image-Edit-2509 DiT. It makes one
  rectified-flow step at a fixed t = 0.499. Precomputed prompt embeddings
  replace the 7 B text encoder.
* **Loading:** the DiT is quantised to 4-bit NF4 with bitsandbytes each time
  it loads.
* **Requirements:** Linux, Python 3.10, a CUDA GPU; 17 GB VRAM at 1024² and
  29 GB at 2048². No ONNX export; diffusers integration is still open on
  their checklist.
* **Checkpoints:**
  * depth: `Log-stage2` (default), `Log-stage1`, `Log-layered`
    (see-through), `Uniform-base`, `Disparity-base`, `Disparity-layered`,
    `Uniform-layered`;
  * `normals`;
  * `albedo` (linear).
* **Licence:** code and models Apache-2.0 (`LICENSE`, `NOTICE`). The
  Qwen-Image-Edit-2509 backbone is Apache-2.0 [strong: Hugging Face licence
  tag via search; the card itself was not opened].

**Why external [inference]:** a 20 B DiT with bitsandbytes 4-bit
quantisation cannot be exported to ONNX Runtime in a way that runs inside
Blender's Python. At ~10 GB in 4-bit, CPU inference would be impractical
anyway. The stable interface is the repo's own CLI, `scripts/infer.py`.

**Output contract [fact, `scripts/infer.py` + `marigoldv2/validation/folder_steps.py`]:**

* Files go to `OUT/images/predictions_npy/<stem>.npy`.
* **Depth** `[H,W]` is the channel mean of the decoded image. It is not
  clipped; training depth is normalised to [−1, 1] (`QwenDepthEncode`).
* **Normals** `[3,H,W]` are renormalised to unit length, with zero vectors
  where invalid. They use the V1.1 camera frame (`ReadSurfaceNormalsNpy`
  reads "camera-space normals (Marigold V1.1 format)"; horizontal flips
  negate x).
* **Albedo** `[3,H,W]` is `clamp_min(0) ** (1/2.2)` of the linear
  prediction, because `infer.py` requests the "srgb" output space.
* **Resolution:** at native resolution, predictions are resized back to the
  input size. With `--width/--height` they stay at that size.
* Every `infer.py` call spawns `evaluation.depth.evaluate_pipeline`, which
  loads and quantises the backbone again.

**Add-on design:**

* The backend is selectable; v1.1 stays the in-Blender default. V2
  produces depth, normals and albedo; v1.1 IID still supplies
  roughness / metallic / shading.
* `ai/marigold_v2.py` (pure): command line, child environment (strip
  PYTHONHOME / PYTHONPATH that Blender may set), output paths, conversions
  (depth `(clip(x,−1,1)+1)/2`, normals renormalised, albedo `x^2.2`),
  inference size (aspect kept, multiples of 16), setup check.
* `v2_job.py`: one subprocess per modality, sequential, with a stdout reader
  thread. A modal operator polls it on a 0.5 s timer (progress bar, status
  text, Esc cancels); scripts and tests use a blocking path. Results are
  saved as EXR with signatures, so unchanged settings reuse them.
* New depth kind `log_depth` with **LOG** mapping
  `z = near^t · far^(1−t)`. This is exact for affine log depth:
  `log z = a·d + b`, normalised d. A unit test recovers z to 1e-4, where
  LINEAR is off by >3 m on the same data.

**Validation [measured, without the real model]:**

* A fake `infer.py` (same CLI, asset checks and output layout, known values)
  drives the whole path. Unit tests cover the command line, environment,
  setup check, conversions and failure exit codes.
* The Blender test (4.2.23 and 5.2.2) covers:
  * Build → V2 depth + normals + albedo → LOG-mapped mesh;
  * pass values (linear albedo 0.5; normal (0.6, 0, 0.8) encoded);
  * reuse without new calls;
  * an out-of-memory failure reported with the log;
  * a missing-setup error.

**Unverified:**

- [ ] A real run (no GPU, no 20 B weights in the sandbox): load time, VRAM
  and the exact numeric range of real depth outputs (clipping at ±1
  assumed from training normalisation).
- [ ] Windows / WSL. The repo targets Linux; the add-on passes
  `CREATE_NO_WINDOW` on Windows, but this is untested.

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
* **Export dry run** [measured; real architecture, random weights, torch 2.8,
  4-core CPU]:
  * 647.7 M parameters; fp32 ONNX is a 1 MB graph plus a 2.58 GB `.data`
    file; export takes ~2 min.
  * ONNX Runtime vs PyTorch: per-pixel relative error p99 1.2e-7. CPU time
    is ~65 s in PyTorch and ~75 s in ORT per 1536² image.
  * fp16 (1.29 GB) needed two fixes in the converter path. Shape inference
    must run from the file path because the model is over 2 GB. The 96
    shape-derived `Cast` nodes must target float16.
  * After those fixes fp16 matches fp32 (rank correlation 1.0) but is no
    faster on CPU and peaks at 9.9 GB RSS, so treat it as GPU-only.

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
- Marigold V2 (Huawei Bayer Lab; code, checkpoints table, CLI, licence): https://github.com/huawei-bayerlab/marigold-v2 · weights: https://huggingface.co/huawei-bayerlab/marigold-v2-0 · paper: https://arxiv.org/abs/2609.08084 · backbone: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- Marigold normals / IID checkpoints: https://huggingface.co/prs-eth/marigold-normals-v1-1 · https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1 · https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1 · IID target configs: https://github.com/prs-eth/Marigold/tree/main/config

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
