"""Run a Marigold V2 install as a background process and import its outputs.

``V2Job`` runs ``scripts/infer.py`` once per modality (depth, normals,
albedo) in the user's marigold-v2 environment, one after the other, without
blocking Blender: a modal operator calls ``poll()`` on a timer
(``operators.DEPTHSCENE_OT_marigold_v2``), scripts and tests call
``run_blocking()``. Each finished modality is converted
(``ai.marigold_v2.load_prediction``) and saved as EXR like the v1.1 maps, so
the rest of the add-on (mesh, material passes, reuse) does not care which
backend produced them.

Every run of ``infer.py`` loads and 4-bit-quantises the 20 B backbone again,
so each modality costs minutes of start-up on top of the inference itself.
"""

import collections
import os
import shutil
import subprocess
import threading

import bpy

from . import bimage, builder, paths
from .ai import marigold_v2
from .core import passes

ACTIVE = {}          # "job" -> running V2Job (one at a time), read by the UI


def settings_from_prefs():
    p = paths.prefs()
    if p is None:
        return "", "", ""
    return bpy.path.abspath(p.v2_python), bpy.path.abspath(p.v2_repo), bpy.path.abspath(p.v2_assets)


class V2Job:
    """Depth and/or material passes for one image via Marigold V2.

    ``modalities``: subset of ("depth", "normals", "albedo"). Outputs whose
    image already exists with the same signature are reused, not recomputed.
    """

    def __init__(self, scene_settings, image, modalities, python=None, repo=None, assets=None):
        s = scene_settings
        if python is None:
            python, repo, assets = settings_from_prefs()
        self.python, self.repo, self.assets = python, repo, assets
        self.s = s
        self.image = image
        self.base = os.path.splitext(image.name)[0]
        self.results = {}            # modality -> Image
        self.kind = marigold_v2.depth_kind(s.v2_checkpoint)
        self.log = collections.deque(maxlen=400)
        self.status = ""
        self.error = ""
        self._proc = None
        self._reader = None
        self._current = None
        self.todo = []
        for m in modalities:
            img = bpy.data.images.get(self._image_name(m))
            if img is not None and img.get("depth_scene_source") == builder.v2_signature(s, image, m):
                self.results[m] = img
            else:
                self.todo.append(m)
        self.total = len(self.todo)
        self.folder = os.path.join(bimage.output_dir(s.save_location), "marigold_v2",
                                   bpy.path.clean_name(self.base))

    # ------------------------------------------------------------------ setup
    def _image_name(self, modality):
        if modality == "depth":
            return f"{self.base}_depth"
        if modality == "normals":
            return builder.pass_image_name(self.image, "v2", "normal")
        return builder.pass_image_name(self.image, "v2", "albedo")

    def problems(self):
        ckpt = self.s.v2_checkpoint
        return marigold_v2.check_setup(self.python, self.repo, self.assets, modalities=self.todo,
                                       depth_checkpoint=ckpt)

    def start(self):
        """Validate, write the input PNG and launch the first modality. Raises on setup errors."""
        if not self.todo:
            return
        problems = self.problems()
        if problems:
            raise RuntimeError("Marigold V2 is not set up: " + problems[0])
        in_dir = os.path.join(self.folder, "input")
        shutil.rmtree(in_dir, ignore_errors=True)       # infer.py processes every image in the folder
        os.makedirs(in_dir, exist_ok=True)
        rgb = bimage.display_rgb(self.image)
        self.size = marigold_v2.inference_size(rgb.shape[1], rgb.shape[0], self.s.v2_resolution)
        self.input_png = bimage.write_png(rgb, os.path.join(in_dir, bpy.path.clean_name(self.base) + ".png"))
        self._launch(self.todo[0])

    def _launch(self, modality):
        out_dir = os.path.join(self.folder, f"out_{modality}")
        shutil.rmtree(out_dir, ignore_errors=True)
        ckpt = None
        if modality == "depth":
            ckpt = marigold_v2.checkpoint_path(self.repo, self.assets, "depth", self.s.v2_checkpoint)
        cmd = marigold_v2.build_command(self.python, self.repo, modality, os.path.dirname(self.input_png),
                                        out_dir, checkpoint=ckpt, size=self.size, seed=self.s.v2_seed)
        self.log.append("$ " + " ".join(cmd))
        self._current = (modality, out_dir)
        self.status = f"Marigold V2 {modality} ({self.done_count + 1}/{self.total}): starting..."
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(cmd, cwd=self.repo, env=marigold_v2.build_env(self.assets),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace", bufsize=1, **kwargs)
        self._reader = threading.Thread(target=self._read, args=(self._proc, modality), daemon=True)
        self._reader.start()

    def _read(self, proc, modality):
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self.log.append(line)
                self.status = f"Marigold V2 {modality} ({self.done_count + 1}/{self.total}): {line[-90:]}"

    @property
    def done_count(self):
        return self.total - len(self.todo)

    @property
    def fraction(self):
        return self.done_count / self.total if self.total else 1.0

    # ------------------------------------------------------------------ driving
    def poll(self):
        """Advance the job: "running", "done" or "error" (see ``self.error``)."""
        if self.error:
            return "error"
        if not self.todo:
            return "done"
        if self._proc is None or self._proc.poll() is None:
            return "running"
        self._reader.join(timeout=2.0)
        modality, out_dir = self._current
        code = self._proc.returncode
        self._proc = None
        if code != 0:
            tail = [ln for ln in self.log if ln.strip()][-6:]
            self.error = f"Marigold V2 {modality} failed (exit {code}): " + " | ".join(tail)
            self._write_log()
            return "error"
        try:
            self._import(modality, out_dir)
        except Exception as exc:
            self.error = f"Marigold V2 {modality}: could not read the result ({exc})"
            self._write_log()
            return "error"
        self.todo.pop(0)
        if self.todo:
            self._launch(self.todo[0])
            return "running"
        self._write_log()
        return "done"

    def run_blocking(self, timeout=None):
        self.start()
        while True:
            state = self.poll()
            if state != "running":
                return state
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.cancel()
                    self.error = "Marigold V2 timed out"
                    return "error"

    def cancel(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.error = self.error or "Cancelled"

    def _write_log(self):
        try:
            os.makedirs(self.folder, exist_ok=True)
            with open(os.path.join(self.folder, "marigold_v2.log"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.log))
        except OSError:
            pass

    # ------------------------------------------------------------------ results
    def _import(self, modality, out_dir):
        path = marigold_v2.prediction_path(out_dir, self.input_png)
        if not os.path.isfile(path):
            raise RuntimeError(f"no prediction at {path}")
        arr = marigold_v2.load_prediction(path, modality)
        folder = bimage.output_dir(self.s.save_location)
        name = self._image_name(modality)
        if modality == "depth":
            img = bimage.write_depth(name, arr, None, folder)
            img["depth_scene_kind"] = self.kind
        elif modality == "normals":
            img = bimage.write_map(name, passes.normal_to_color(arr), folder, non_color=True)
        else:
            img = bimage.write_map(name, arr, folder, non_color=False)
        img["depth_scene_source"] = builder.v2_signature(self.s, self.image, modality)
        if modality != "depth":
            img["depth_scene_pass_group"] = "v2"
        self.results[modality] = img

    def pass_results(self):
        """{"v2": {"normal": Image, "albedo": Image}} for ``builder.assign_passes``."""
        out = {}
        if "normals" in self.results:
            out["normal"] = self.results["normals"]
        if "albedo" in self.results:
            out["albedo"] = self.results["albedo"]
        return {"v2": out} if out else {}
