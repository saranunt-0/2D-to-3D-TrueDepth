# Module: blender-addon
<!-- bpy glue: extension, preferences, properties, builder, operators, UI -->

---

## [2026-10-08] Add-on v0.1.0

**Type**: `feature`
**Status**: `resolved` (UI not clicked in a real window)

### Work Done
- Extension manifest (4.2+) + bl_info; validates with Blender's blender_ext.py (4.2 & 5.2);
  installs via extensions.package_install_files (root-level and folder zips).
- Preferences: ORT variant (cpu/cuda/directml) install, model download/delete with progress.
- Live per-object settings -> numpy rebuild (~0.1 s at res 384 on a 2000 px image).
- Rig at 3D cursor; projector (locked) + shot camera; camera-move operator.
- Debug: projector render looked dark -> root cause was test isolation (other rigs at the
  origin occluding), not colour management; fixed test + rigs now spawn at the 3D cursor.
- Debug: relief test "array length mismatch" -> test passed a 2-D array to foreach_get.

### Checklist
- [x] 35 tests pass on bpy 4.2.23 LTS and 5.2.2 LTS
- [x] Projector render reproduces source image (median error < 2 %)
- [x] Panels draw against a validating fake layout (property names/operator ids)
- [ ] Real UI interaction / modal download progress in a window

### Unverified Items
- [ ] Install zip in desktop Blender, click Install ONNX Runtime, Download, Build on 1.webp.
