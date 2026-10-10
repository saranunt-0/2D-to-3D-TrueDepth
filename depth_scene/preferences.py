"""Add-on preferences: ONNX Runtime installation and model downloads."""

import os
import shutil
import subprocess
import sys
import threading

import bpy
from bpy.props import EnumProperty, StringProperty

from . import paths
from .ai import download, models, runtime

ORT_VARIANTS = [
    ("cpu", "CPU (all platforms)", "onnxruntime - CPU everywhere, CoreML GPU on macOS", 0),
    ("cuda", "NVIDIA CUDA", "onnxruntime-gpu - needs CUDA 12 + cuDNN 9 installed", 1),
    ("directml", "DirectML (Windows GPU)", "onnxruntime-directml - any DirectX 12 GPU on Windows", 2),
]
ORT_PACKAGES = {"cpu": "onnxruntime", "cuda": "onnxruntime-gpu", "directml": "onnxruntime-directml"}

# Shared state polled by modal operators and drawn in the preferences.
_jobs = {}


def activate_site_packages():
    p = paths.prefs()
    variant = p.ort_variant if p else "cpu"
    runtime.add_site_dir(paths.site_packages_dir(variant))


class _ThreadJobMixin:
    """Run ``work`` in a thread, keep Blender responsive, report at the end."""

    _timer = None

    def _start(self, context, key, work):
        self._key = key
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        for area in context.screen.areas if context.screen else ():
            area.tag_redraw()
        if self._thread.is_alive():
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        job = _jobs.pop(self._key, {})
        if job.get("error"):
            self.report({"ERROR"}, job["error"])
            return {"CANCELLED"}
        self.report({"INFO"}, job.get("message", "Done"))
        return {"FINISHED"}


def install_onnxruntime(package, target, python=None):
    """pip-install ``package`` into ``target`` without touching Blender's NumPy."""
    py = python or sys.executable
    if subprocess.run([py, "-m", "pip", "--version"], capture_output=True).returncode != 0:
        subprocess.run([py, "-m", "ensurepip", "--upgrade"], check=True, capture_output=True)
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    cmd = [py, "-m", "pip", "install", "--upgrade", "--no-input", "--only-binary=:all:",
           "--disable-pip-version-check", "--target", target, package]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or "pip failed").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else "pip failed")
    # pip pulls NumPy in as a dependency; Blender must keep using its own build.
    for name in os.listdir(target):
        if name == "numpy" or name.startswith("numpy-") or name == "numpy.libs" or name == "bin":
            shutil.rmtree(os.path.join(target, name), ignore_errors=True)


class DEPTHSCENE_OT_install_ort(_ThreadJobMixin, bpy.types.Operator):
    bl_idname = "depth_scene.install_ort"
    bl_label = "Install ONNX Runtime"
    bl_description = "pip-install ONNX Runtime into the add-on's private folder (needs internet)"

    def execute(self, context):
        variant = paths.prefs().ort_variant
        target = paths.site_packages_dir(variant)
        key = "install"
        if key in _jobs:
            self.report({"WARNING"}, "Installation already running")
            return {"CANCELLED"}
        _jobs[key] = {}

        def work():
            try:
                install_onnxruntime(ORT_PACKAGES[variant], target)
                runtime.add_site_dir(target)
                _jobs[key]["message"] = (f"Installed {ORT_PACKAGES[variant]}. Restart Blender if an older "
                                         "ONNX Runtime was already loaded.")
            except Exception as exc:
                _jobs[key]["error"] = f"ONNX Runtime install failed: {exc}"

        return self._start(context, key, work)


class DEPTHSCENE_OT_download_model(_ThreadJobMixin, bpy.types.Operator):
    bl_idname = "depth_scene.download_model"
    bl_label = "Download Model"
    bl_description = "Download the ONNX model file from Hugging Face"

    model: StringProperty()
    precision: StringProperty(default="fp32")

    def execute(self, context):
        info = models.get(self.model)
        dest = paths.model_path(info, self.precision)
        key = f"dl:{info.key}:{self.precision}"
        if key in _jobs:
            return {"CANCELLED"}
        job = download.Download(info.url(self.precision), dest)
        _jobs[key] = {"job": job, "message": f"Downloaded {info.label}"}

        def work():
            job.run()
            if job.error:
                _jobs[key]["error"] = f"Download failed ({job.error}). You can download {info.url(self.precision)} manually into {paths.models_dir()}"

        return self._start(context, key, work)


class DEPTHSCENE_OT_delete_model(bpy.types.Operator):
    bl_idname = "depth_scene.delete_model"
    bl_label = "Delete Model"
    bl_description = "Delete the downloaded model file"
    bl_options = {"INTERNAL"}

    model: StringProperty()
    precision: StringProperty(default="fp32")

    def execute(self, context):
        runtime.clear_sessions()
        path = paths.model_path(models.get(self.model), self.precision)
        for p in (path, path + ".data"):    # big models keep weights in a .data sidecar
            if os.path.exists(p):
                os.remove(p)
        return {"FINISHED"}


def download_progress(key):
    job = _jobs.get(key, {}).get("job")
    return None if job is None else job.progress


class DepthScenePreferences(bpy.types.AddonPreferences):
    bl_idname = paths.ADDON_ID

    ort_variant: EnumProperty(name="ONNX Runtime", items=ORT_VARIANTS, default="cpu",
                              update=lambda self, ctx: activate_site_packages())
    models_dir: StringProperty(name="Models Folder", subtype="DIR_PATH",
                               description="Where model files are stored (empty = add-on user folder)")

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="AI runtime", icon="SETTINGS")
        ver = runtime.ort_version()
        row = box.row()
        row.prop(self, "ort_variant")
        if "install" in _jobs:
            row.label(text="Installing...", icon="SORTTIME")
        else:
            row.operator(DEPTHSCENE_OT_install_ort.bl_idname, text="Reinstall" if ver else "Install", icon="IMPORT")
        if ver:
            box.label(text=f"onnxruntime {ver}: " + ", ".join(p.replace("ExecutionProvider", "") for p in runtime.available_providers()),
                      icon="CHECKMARK")
        else:
            box.label(text="ONNX Runtime not installed - AI depth is disabled (existing depth maps still work)", icon="ERROR")

        box = layout.box()
        box.label(text="Depth models", icon="FILE_3D")
        box.prop(self, "models_dir")
        for info in models.MODELS.values():
            for precision in info.files:
                row = box.row(align=True)
                label = f"{info.label}  [{precision}, ~{info.size_mb(precision)} MB, {info.license_id}]"
                row.label(text=label, icon="CHECKMARK" if os.path.exists(paths.model_path(info, precision)) else "BLANK1")
                key = f"dl:{info.key}:{precision}"
                prog = download_progress(key)
                if prog is not None:
                    row.label(text=f"{prog * 100:.0f}%")
                elif os.path.exists(paths.model_path(info, precision)):
                    op = row.operator(DEPTHSCENE_OT_delete_model.bl_idname, text="", icon="TRASH")
                    op.model, op.precision = info.key, precision
                elif info.url(precision) is None:
                    row.label(text=f"manual: {info.filename(precision)}")
                else:
                    op = row.operator(DEPTHSCENE_OT_download_model.bl_idname, text="Download", icon="URL")
                    op.model, op.precision = info.key, precision
        col = box.column(align=True)
        col.label(text="Depth Anything V2 Base/Large weights are CC-BY-NC-4.0: non-commercial use only.", icon="INFO")
        col.label(text="Small (Apache-2.0) and MoGe-2 (MIT) are fine for commercial work.")
        col.label(text="Illustrator's Depth: Adobe Research licence - academic research/teaching only.", icon="ERROR")
        col.label(text="Convert it with tools/export_illustrators_depth_onnx.py (see README).")


classes = (DEPTHSCENE_OT_install_ort, DEPTHSCENE_OT_download_model, DEPTHSCENE_OT_delete_model, DepthScenePreferences)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    activate_site_packages()


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
