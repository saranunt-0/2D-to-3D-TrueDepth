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
