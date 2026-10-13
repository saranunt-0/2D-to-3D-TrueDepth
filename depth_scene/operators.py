"""User-facing operators."""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty

from . import bimage, builder, v2_job


def _cursor(context, kind):
    if context.window is not None:      # None when run from a script
        context.window.cursor_set(kind)


class _Progress:
    """Marigold takes seconds to minutes: show progress on the cursor/status bar."""

    def __init__(self, context):
        self.wm = context.window_manager if context.window is not None else None
        if self.wm:
            self.wm.progress_begin(0, 100)

    def __call__(self, fraction):
        if self.wm:
            self.wm.progress_update(int(fraction * 100))

    def end(self):
        if self.wm:
            self.wm.progress_end()
            self.wm = None          # idempotent: a later end() must not stop another operator's bar


def _active_depth_object(context):
    obj = context.active_object
    if obj is not None and obj.type == "MESH" and obj.depth_scene.is_depth_scene:
        return obj
    return None


class DEPTHSCENE_OT_generate_depth(bpy.types.Operator):
    bl_idname = "depth_scene.generate_depth"
    bl_label = "Generate Depth Map"
    bl_description = "Run the selected AI model on the image and save a depth map (EXR)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.scene.depth_scene.image is not None

    def execute(self, context):
        s = context.scene.depth_scene
        if s.backend == "V2":
            return _start_v2(self, context, build=False)
        _cursor(context, "WAIT")
        progress = _Progress(context)
        try:
            depth_img, kind, result = builder.generate_depth(s, s.image, progress=progress)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            progress.end()
            _cursor(context, "DEFAULT")
        s.existing_depth = depth_img
        s.existing_kind = kind
        self.report({"INFO"}, f"Depth {depth_img.size[0]}x{depth_img.size[1]} in {result.seconds:.1f}s "
                              f"on {result.provider}")
        return {"FINISHED"}


def _start_v2(op, context, build=False, passes_only=False):
    """Hand over to the Marigold V2 operator (modal with a window, blocking from scripts)."""
    if "job" in v2_job.ACTIVE:
        op.report({"WARNING"}, "Marigold V2 is already running - wait for it or press Esc to cancel")
        return {"CANCELLED"}
    mode = "INVOKE_DEFAULT" if context.window is not None else "EXEC_DEFAULT"
    try:
        res = bpy.ops.depth_scene.marigold_v2(mode, build=build, passes_only=passes_only)
    except RuntimeError as exc:         # its error report, re-raised by bpy.ops
        op.report({"ERROR"}, str(exc).replace("Error: ", "", 1).strip())
        return {"CANCELLED"}
    return {"FINISHED"} if "RUNNING_MODAL" in res else res


def _run_passes(op, context, obj, progress):
    """Generate the enabled v1.1 material passes for ``obj`` and report; never raises."""
    s = context.scene.depth_scene
    if not builder.enabled_pass_groups(s):
        return {}, []
    _cursor(context, "WAIT")
    results, messages, errors = builder.generate_passes(s, obj.depth_scene.image, progress=progress)
    builder.assign_passes(obj, results, s.albedo_from)
    for msg in errors:
        op.report({"WARNING"}, msg)
    if results:
        op.report({"INFO"}, "Material passes: " + "; ".join(messages) + " - add lights to see them")
    return results, errors


class DEPTHSCENE_OT_build(bpy.types.Operator):
    bl_idname = "depth_scene.build"
    bl_label = "Build 3D Scene"
    bl_description = ("Generate depth if needed, then build the 3D mesh, projector and shot cameras "
                      "(and the enabled material passes)")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.depth_scene.image is not None

    def execute(self, context):
        s = context.scene.depth_scene
        image = s.image
        progress = _Progress(context)
        try:
            if s.use_existing_depth:
                if s.existing_depth is None:
                    raise RuntimeError("Pick a depth map image or disable 'Use Existing Depth Map'")
                depth_img, kind = s.existing_depth, s.existing_kind
                if not depth_img.is_float or depth_img.colorspace_settings.name != "Non-Color":
                    depth_img.colorspace_settings.name = "Non-Color"
            elif (s.existing_depth is not None
                  and s.existing_depth.get("depth_scene_source") == builder.depth_signature(s, image)):
                # Same image + model settings: reuse the depth from 'Generate Depth Map'.
                depth_img, kind = s.existing_depth, s.existing_kind
            elif s.backend == "V2":
                progress.end()
                return _start_v2(self, context, build=True)   # runs V2, then builds when it finishes
            else:
                _cursor(context, "WAIT")
                depth_img, kind, _ = builder.generate_depth(s, image, progress=progress)
                s.existing_depth, s.existing_kind = depth_img, kind
            obj = builder.build_scene(context, image, depth_img, kind, s.mode)
            o = obj.depth_scene
            self.report({"INFO"}, f"Built '{obj.name}': {o.stat_faces} faces, {o.stat_removed} stretched removed")
            if builder.enabled_pass_groups(s):
                _run_passes(self, context, obj, progress)
            if builder.enabled_v2_passes(s):
                progress.end()
                return _start_v2(self, context, passes_only=True)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            progress.end()
            _cursor(context, "DEFAULT")
        return {"FINISHED"}


class DEPTHSCENE_OT_generate_passes(bpy.types.Operator):
    bl_idname = "depth_scene.generate_passes"
    bl_label = "Generate Material Passes"
    bl_description = ("Run the enabled Marigold normals / intrinsics models on the active Depth Scene "
                      "object's image and wire the maps into its material")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = _active_depth_object(context)
        s = context.scene.depth_scene
        return (obj is not None and obj.depth_scene.image is not None
                and bool(builder.enabled_pass_groups(s) or builder.enabled_v2_passes(s)))

    def execute(self, context):
        obj = _active_depth_object(context)
        s = context.scene.depth_scene
        progress = _Progress(context)
        try:
            results, errors = _run_passes(self, context, obj, progress)
        finally:
            progress.end()
            _cursor(context, "DEFAULT")
        if builder.enabled_v2_passes(s):
            return _start_v2(self, context, passes_only=True)
        if not results:
            self.report({"ERROR"}, errors[0] if errors else "No pass generated")
            return {"CANCELLED"}
        return {"FINISHED"}


class DEPTHSCENE_OT_marigold_v2(bpy.types.Operator):
    bl_idname = "depth_scene.marigold_v2"
    bl_label = "Run Marigold V2"
    bl_description = ("Run your Marigold V2 install (separate process, CUDA GPU) for depth and the enabled "
                      "V2 passes, then import the results. Esc cancels")
    bl_options = {"REGISTER", "UNDO"}

    build: BoolProperty(name="Build When Done", default=False,
                        description="Build the 3D scene from the new depth map when it finishes")
    passes_only: BoolProperty(name="Passes Only", default=False,
                              description="Only the V2 material passes, for the active Depth Scene object")

    _timer = None
    _job = None

    @classmethod
    def poll(cls, context):
        return context.scene.depth_scene.image is not None and "job" not in v2_job.ACTIVE

    def _make_job(self, context):
        s = context.scene.depth_scene
        modalities = list(builder.enabled_v2_passes(s))
        image = s.image
        self._target = None
        if self.passes_only:
            obj = _active_depth_object(context)
            if obj is None:
                raise RuntimeError("Select a Depth Scene object first")
            self._target = obj.name
            image = obj.depth_scene.image
        else:
            modalities.insert(0, "depth")
        if not modalities:
            raise RuntimeError("Nothing to run: enable a Marigold V2 pass")
        return v2_job.V2Job(s, image, modalities)

    def _finish(self, context, job):
        s = context.scene.depth_scene
        depth = job.results.get("depth")
        if depth is not None:
            s.existing_depth, s.existing_kind = depth, job.kind
        obj = bpy.data.objects.get(self._target) if self._target else None
        if self.build and depth is not None:
            obj = builder.build_scene(context, job.image, depth, job.kind, s.mode)
            if builder.enabled_pass_groups(s):
                _run_passes(self, context, obj, None)
        if obj is not None and job.pass_results():
            builder.assign_passes(obj, job.pass_results(), s.albedo_from)
        done = ", ".join(sorted(job.results)) or "nothing"
        self.report({"INFO"}, f"Marigold V2 done: {done}" + (f" -> built '{obj.name}'" if self.build and obj else ""))

    def execute(self, context):          # scripts / background mode: blocking
        try:
            job = self._make_job(context)
            state = job.run_blocking()
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if state != "done":
            self.report({"ERROR"}, job.error)
            return {"CANCELLED"}
        self._finish(context, job)
        return {"FINISHED"}

    def invoke(self, context, event):
        try:
            self._job = self._make_job(context)
            self._job.start()
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        v2_job.ACTIVE["job"] = self._job
        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _cleanup(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        if context.workspace is not None:
            context.workspace.status_text_set(None)
        v2_job.ACTIVE.pop("job", None)

    def modal(self, context, event):
        job = self._job
        if event.type == "ESC":
            job.cancel()
            self._cleanup(context)
            self.report({"WARNING"}, "Marigold V2 cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        state = job.poll()
        if state == "running":
            context.window_manager.progress_update(int(job.fraction * 100))
            if context.workspace is not None:
                context.workspace.status_text_set(job.status + "   (Esc to cancel)")
            for area in context.screen.areas if context.screen else ():
                if area.type == "VIEW_3D":
                    area.tag_redraw()
            return {"PASS_THROUGH"}
        self._cleanup(context)
        if state == "error":
            self.report({"ERROR"}, job.error)
            return {"CANCELLED"}
        try:
            self._finish(context, job)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class DEPTHSCENE_OT_add_light(bpy.types.Operator):
    bl_idname = "depth_scene.add_light"
    bl_label = "Add Light"
    bl_description = "Add a light in front of the projector, sized to the scene, to try the lit material"
    bl_options = {"REGISTER", "UNDO"}

    kind: EnumProperty(name="Type", items=[
        ("POINT", "Point", ""), ("AREA", "Area", ""), ("SPOT", "Spot", ""), ("SUN", "Sun", "")], default="POINT")

    @classmethod
    def poll(cls, context):
        obj = _active_depth_object(context)
        return obj is not None and obj.parent is not None

    def execute(self, context):
        obj = _active_depth_object(context)
        light = builder.add_light(obj, self.kind)
        if obj.depth_scene.shading != "PRINCIPLED":
            obj.depth_scene.shading = "PRINCIPLED"
        self.report({"INFO"}, f"Added '{light.name}' (power {light.data.energy:g}) - move it to relight the scene")
        return {"FINISHED"}


class DEPTHSCENE_OT_rebuild(bpy.types.Operator):
    bl_idname = "depth_scene.rebuild"
    bl_label = "Rebuild"
    bl_description = "Re-read the image/depth map and rebuild the active Depth Scene mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _active_depth_object(context) is not None

    def execute(self, context):
        obj = _active_depth_object(context)
        bimage.clear_cache()
        try:
            builder.rebuild(obj)
            builder.update_material(obj)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class DEPTHSCENE_OT_camera_move(bpy.types.Operator):
    bl_idname = "depth_scene.camera_move"
    bl_label = "Add Camera Move"
    bl_description = "Keyframe a parallax move on the shot camera (2.5D photo effect)"
    bl_options = {"REGISTER", "UNDO"}

    move: EnumProperty(name="Move", items=[
        ("DOLLY_IN", "Dolly In", "Push forward into the scene"),
        ("DOLLY_OUT", "Dolly Out", "Pull back"),
        ("TRUCK_LEFT", "Truck Left", "Slide sideways"),
        ("TRUCK_RIGHT", "Truck Right", "Slide sideways"),
        ("PEDESTAL_UP", "Pedestal Up", "Rise vertically"),
    ], default="DOLLY_IN")
    distance: FloatProperty(name="Distance", default=0.5, min=0.0, soft_max=10.0, unit="LENGTH",
                            description="How far the camera travels. Keep it small relative to Near "
                                        "to avoid revealing holes behind objects")
    frames: IntProperty(name="Frames", default=120, min=2)

    @classmethod
    def poll(cls, context):
        obj = _active_depth_object(context)
        return obj is not None and obj.depth_scene.shot_camera is not None

    def invoke(self, context, event):
        o = _active_depth_object(context).depth_scene
        self.distance = round(0.4 * (o.near if o.mode == "PERSPECTIVE" else o.size * 0.3), 3)
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = _active_depth_object(context)
        scene = context.scene
        start = scene.frame_start
        end = start + self.frames - 1
        scene.frame_end = max(scene.frame_end, end)
        builder.add_camera_move(obj, self.move, self.distance, start, end)
        scene.camera = obj.depth_scene.shot_camera
        return {"FINISHED"}


class DEPTHSCENE_OT_reset_shot(bpy.types.Operator):
    bl_idname = "depth_scene.reset_shot"
    bl_label = "Reset Shot Camera"
    bl_description = "Snap the shot camera back onto the projector and remove its animation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = _active_depth_object(context)
        return obj is not None and obj.depth_scene.shot_camera is not None

    def execute(self, context):
        builder.reset_shot_camera(_active_depth_object(context))
        return {"FINISHED"}


class DEPTHSCENE_OT_look_through(bpy.types.Operator):
    bl_idname = "depth_scene.look_through"
    bl_label = "Look Through"
    bl_description = "Make this camera active and view through it"
    bl_options = {"REGISTER"}

    which: EnumProperty(items=[("PROJECTOR", "Projector", ""), ("SHOT", "Shot", "")])

    @classmethod
    def poll(cls, context):
        return _active_depth_object(context) is not None

    def execute(self, context):
        o = _active_depth_object(context).depth_scene
        cam = o.projector if self.which == "PROJECTOR" else o.shot_camera
        if cam is None:
            return {"CANCELLED"}
        context.scene.camera = cam
        area = context.area
        if area and area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
        return {"FINISHED"}


classes = (
    DEPTHSCENE_OT_generate_depth,
    DEPTHSCENE_OT_build,
    DEPTHSCENE_OT_generate_passes,
    DEPTHSCENE_OT_marigold_v2,
    DEPTHSCENE_OT_add_light,
    DEPTHSCENE_OT_rebuild,
    DEPTHSCENE_OT_camera_move,
    DEPTHSCENE_OT_reset_shot,
    DEPTHSCENE_OT_look_through,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
