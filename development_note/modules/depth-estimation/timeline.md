# Module: depth-estimation
<!-- AI depth: model registry, ONNX inference, MoGe post-processing -->

---

## [2026-10-08] Choose lightweight depth model + implement inference

**Type**: `feature` / `investigation`
**Status**: `resolved` (real-weight runs unverified - see below)

### Context
User asked for the best lightweight depth model for a TrueDepth-like Blender add-on,
corridor shots as the main use case.

### Work Done
- Surveyed DA2 S/B/L, DA3 (S/B/Mono/Metric), MoGe-2 S/B, MoGe-3, Depth Pro (docs/RESEARCH.md).
- Verified DA2 licences/params from the official README; MoGe model table + ONNX notes
  from microsoft/MoGe; DA3 licence table from ByteDance-Seed/Depth-Anything-3.
- Implemented `ai/models.py`, `ai/runtime.py`, `ai/infer.py`, `ai/download.py`,
  `core/moge_post.py` (SciPy-free port of recover_focal_shift).
- Tests use fake ONNX graphs with the real I/O signatures (`tests/fake_models.py`).

### Checklist
- [x] DA2 preprocessing: shorter side -> 518, multiple of 14, ImageNet norm (test)
- [x] fp16 input dtype handled (test)
- [x] MoGe focal/shift recovery: FOV within 0.2-0.5 deg on synthetic affine point maps
- [x] Download class (file:// URL test), pip install of onnxruntime into --target (manual run)
- [ ] Real weights - Hugging Face blocked in the sandbox

### Unverified Items
- [ ] Preferences > Download "Depth Anything V2 Small" then Generate Depth on a corridor image -
      expected: ~518x710 disparity map, < 1-2 s on CPU, plausible near/far.
- [ ] Download "MoGe-2 ViT-S" and Build in Perspective mode - expected: output names
      points/normal/mask/metric_scale accepted, FOV estimate shown (likely 50-70 deg), straight walls.
- [ ] GPU providers (CUDA / DirectML / CoreML) and fp16 DA2 file.

### Root Cause / Outcome
Default DA2-Small (Apache-2.0, 24.8M); MoGe-2 ViT-S (MIT, 35M) recommended for perspective.

### Assumptions Made
- onnx-community DA2 export accepts arbitrary multiples of 14 (dynamic H/W) - strong evidence, untested.
- MoGe ONNX `mask` output is already sigmoid-ed (per v2.py forward) - threshold 0.5.

---

## [2026-10-10] Add Illustrator's Depth (optional, manual ONNX)

**Type**: `feature`
**Status**: `resolved` (real checkpoint unverified - see below)

### Context
User asked to add Illustrator's Depth (github.com/nissmar/illustrators_depth) after the
painting/illustration depth research.

### Work Done
- Read the repo source (model, dataset transforms, loss, SVG depth rasterisation):
  Depth Pro encoder/decoder/head, fixed 1536x1536 stretch, Normalize(0.5, 0.5),
  output -1/clamp(head) growing toward the front (GT = SVG document order).
- Licence check: Adobe Research License = noncommercial research only (stricter than CC-BY-NC).
  Weights = Lightning checkpoint on Google Drive (blocked in sandbox).
- Design: keep the add-on ONNX-only. `tools/export_illustrators_depth_onnx.py` converts the
  checkpoint inside the authors' env (their loader, no Adobe code copied); add-on gets a
  "manual" registry entry (no URL), family `illustrators_depth`, kind `layers` (LINEAR),
  `Input Blur` (their gaussian_sigma) and a generic `Quantize Layers` option.
- Tests: fake ONNX with the exact exported I/O + planted clamp outlier; core tests for
  layers/quantize/robust clip; Blender build with layered relief.

### Checklist
- [x] 41 tests pass on Blender 4.2.23 and 5.2.2
- [x] Export script on the real architecture with random weights (see RESEARCH.md / PR)
- [ ] Real checkpoint conversion + direction check on a real illustration

### Unverified Items
- [ ] `python tools/export_illustrators_depth_onnx.py --ckpt .../id_model.ckpt --check` in the
      authors' env - expected: relative diff < 1e-3, ~2.5 GB .onnx + .data.
- [ ] Generate on an illustration (e.g. 2.png) - front shapes should come out higher/closer.

### Assumptions Made
- Output sign (front = higher) inferred from GT definition + loss; not observed on real weights.
