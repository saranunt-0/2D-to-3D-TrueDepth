# Module: reconstruction
<!-- core/: depth mapping, unprojection, relief, stretched faces (NumPy only) -->

---

## [2026-10-08] Perspective/relief mesh generation

**Type**: `feature`
**Status**: `resolved`

### Work Done
- `core/imaging.py`: antialiased resize, O(N) 3-box Gaussian, bilinear sampling (no SciPy/PIL).
- `core/depth.py`: nearness normalisation per kind; INVERSE/LINEAR/METRIC mappings.
- `core/geometry.py`: grid, unprojection, relief, stretched-face angle test, alpha/mask removal, compaction.
- Bug found by tests: infinite depth (sky) counted as valid in metric nearness -> fixed (isfinite).
- Perf: detail high-pass now computed at 2x grid size; box blur uses slices (2x faster).

### Checklist
- [x] Analytic corridor: median vertex error < 1 mm to true planes
- [x] Affine disparity + correct near/far reproduces depth (1e-4)
- [x] Stretch threshold sweep on blurred depth -> default 86 deg (0 false removals at 128/256/512)
- [x] Face winding faces the camera

### Assumptions Made
- Zero-shift initial guess for DA2 near/far ratio (percentiles 0.5/99.5, clamp 1.5-100).
