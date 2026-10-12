# Module: marigold
<!-- Branch `marigold`: Marigold-only variant (diffusion depth via 3 ONNX graphs + NumPy DDIM) -->

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
