# Module: marigold
<!-- Branch `marigold`: Marigold-only variant (diffusion depth via 3 ONNX graphs + NumPy DDIM) -->

---

## [2026-10-13] Marigold V2 as an external backend

**Type**: `feature`
**Status**: `resolved` (real V2 run unverified: no GPU in sandbox)

### Context
User: "Continue in the marigold branch. Set up the plugin for marigold v2. Save it as different commit."

### Work Done
- Research (repo at 9926d93): Qwen-Image-Edit-2509 DiT (20 B, 4-bit NF4 on load) + LoRA,
  single step; CUDA 17 GB @1024; CLI scripts/infer.py; outputs .npy (depth mean of decoded,
  normals unit V1.1 frame, albedo gamma 1/2.2); Apache-2.0 code + checkpoints + backbone.
- Decision: cannot run in Blender -> external backend driving the official CLI; V1.1 kept.
- core/depth.py: `log_depth` kind + LOG mapping. ai/marigold_v2.py (pure helpers).
- v2_job.py: sequential subprocess per modality, reader thread, import as EXR, reuse by signature.
- Operator `depth_scene.marigold_v2` (modal: progress, status, Esc; blocking from scripts);
  Generate/Build/Generate Passes route to it with backend V2; nested errors re-reported.
- Prefs: python / repo / assets + setup check. Scene: backend, checkpoint, resolution, seed,
  Albedo (V2) pass; albedo source "V2".

### Checklist
- [x] Unit tests: CLI, env, setup check, conversions, failure (fake infer.py)
- [x] Blender 4.2.23 / 5.2.2: 66 tests pass incl. V2 end-to-end, reuse, failure, LOG geometry
- [ ] Real Marigold V2 run on a CUDA GPU

### Debug notes
- Nested bpy.ops error reports raise RuntimeError inside the calling operator -> caught in
  `_start_v2` and re-reported.
- 320x240 at 160 px -> 160x128 (120 is not a multiple of 16; test expectation fixed).

### Assumptions Made
- Real depth outputs live in about [-1, 1] (training normalisation); clipped there.
- Every infer.py call reloads the backbone (it spawns evaluate_pipeline) -> minutes per modality.

---

## [2026-10-13] Material passes: normals + intrinsics (lighting & shadows)

**Type**: `feature`
**Status**: `resolved` (real normals/IID checkpoints unverified)

### Context
User: Marigold can also predict normals and materials; add them as options so the generated
scene interacts with lighting and shadows, with maximum control (toggle on/off, adjust).

### Work Done
- Research: diffusers 0.41 `MarigoldNormalsPipeline` / `MarigoldIntrinsicsPipeline` source,
  prs-eth/Marigold configs (IID target properties), licence (README: all models RAIL++-M).
- `ai/marigold.py`: generic task loop (depth / normals / intrinsics), T target latents,
  per-target decode at batch 1, normals "closest" ensemble, IID median, `split_targets`.
- `ai/models.py`: normals, iid-appearance, iid-lighting entries; `DEPTH_MODELS`, `PASS_MODELS`.
- `core/passes.py`: normal encoding, sRGB->linear albedo, non-negative LSQ calibration of
  shading/residual (`I = A*S + R`).
- `tools/export_marigold_onnx.py`: `--task`, rgb decoder graph, task fields in marigold.json,
  `--tiny` random-init; `export_folder`/`check` reusable from tests.
- Blender: Scene toggles (3 passes + albedo source), Build runs passes, "Generate for Active
  Object", per-object maps + 20 live material parameters, Lights & Shadows panel (Cast
  Shadows, Original Lighting, Add Light Point/Area/Spot/Sun).

### Checklist
- [x] Export parity vs diffusers, tiny models, 4 tasks: max |diff| <= 5.3e-5
- [x] Full-size IID-Lighting (random weights): 9.6e-6; 768 px 101 s, 384 px 22.5 s, 10.4 GB
- [x] Normal-map convention: Cycles render GT normals vs geometry 0.0024, flipped X 0.50
- [x] 54 tests pass on Blender 4.2.23 and 5.2.2 (+5 torch-only tests in the torch venv)
- [ ] Real checkpoints / quality on the user's images (HF blocked)

### Debug notes
- Blender test failed once on test-order leakage (previous test set `use_albedo=False`);
  reset explicitly in the passes test.

### Assumptions Made
- Normals/appearance HF cards carry the same RAIL++-M licence as the repo README states.
- IID quality outside indoor scenes is unknown; sliders allow backing off.

---

## [2026-10-12] Marigold-only branch

**Type**: `feature`
**Status**: `resolved` (real checkpoint unverified)

### Context
User asked for a new branch dedicated to Marigold (prs-eth/Marigold), a licence check for
commercial use, and permission to remove the other models.

### Work Done
- Licence: code Apache-2.0; weights Open RAIL++-M (commercial OK, use-based restrictions,
  redistribution conditions). Attachment A of Marigold's LICENSE-MODEL.txt is an unfilled
  template -> SD2's Open RAIL++-M restrictions apply as base model. HR-variant card says Apache
  (inconsistent with repo). Marigold V2 (Huawei, 20B, Qwen-Image-Edit) is a different model.
- Removed DA2 / MoGe-2 / Illustrator's Depth: registry, inference, moge_post, download module,
  export_illustrators_depth_onnx, layers kind, input blur, estimated FOV, sky mask, their tests.
- Added core/ddim.py (DDIMScheduler port), ai/marigold.py (pipeline), multi-session runtime,
  tools/export_marigold_onnx.py (--check vs diffusers, --fp16, --random-init), Marigold UI
  (processing resolution, steps, ensemble, seed, progress bar), folder-based model management.

### Checklist
- [x] DDIM port == diffusers (timesteps identical, rel. err 1e-6) across all configs
- [x] Export of real architecture (random weights): fp32 3.6 GB, fp16 2.0 GB
- [x] Add-on loop vs MarigoldDepthPipeline, same noise: max |diff| 5.6e-6
- [x] CPU timing: 32 s @768 px / 9 s @384 px (1 step, 4 cores), ~9 GB RSS
- [x] Tests pass on Blender 4.2.23 and 5.2.2
- [ ] Real checkpoint (HF blocked)

### Debug notes
- `--check` first failed: diffusers' input check reads (H, W) from shape[-2:], which is wrong
  for channels-last NumPy input -> pass a channels-first tensor.

### Assumptions Made
- SD2's Attachment A restrictions bind Marigold weights (derivative of SD2) - legal reading, not verified by counsel.
