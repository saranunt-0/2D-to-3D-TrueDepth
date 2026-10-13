"""Marigold V2 (huawei-bayerlab/marigold-v2) as an external backend.

Marigold V2 (Pavlovic et al., SIGGRAPH Asia 2026) fine-tunes LoRA adapters on
the frozen Qwen-Image-Edit-2509 diffusion transformer (~20 B parameters,
quantised to 4 bit with bitsandbytes when it loads) into single-step dense
predictors: depth, see-through depth, surface normals and albedo.

It cannot run inside Blender: it needs PyTorch + bitsandbytes on a CUDA GPU
(~17 GB VRAM at 1024 px, ~29 GB at 2048 px, Linux). So the add-on drives the
official command line of a separate install,

    <python> <repo>/scripts/infer.py --modality depth|normals|albedo
             --image_dir IN --output_dir OUT [--checkpoint C] [--width W --height H] [--seed S]

and imports what it writes (``scripts/infer.py`` + ``marigoldv2/validation/
folder_steps.py`` at commit 9926d93, read 2026-10-13):

    OUT/images/predictions_npy/<stem>.npy
        depth    [H, W]    mean of the decoded RGB, nominally in [-1, 1];
                           log / uniform checkpoints: bigger = farther,
                           disparity checkpoints: bigger = nearer
        normals  [3, H, W] unit camera-space normals, Marigold V1.1 frame
                           (x right, y up, z towards the viewer)
        albedo   [3, H, W] linear albedo ** (1 / 2.2)  ("srgb" output space)

Predictions are resized back to the input resolution by ``infer.py``.

Licence (checked 2026-10-13): code and checkpoints Apache-2.0 (repo LICENSE,
NOTICE); the Qwen-Image-Edit-2509 backbone is Apache-2.0 (its Hugging Face
licence tag).

This module is NumPy / stdlib only (no bpy) so it is unit-testable.
"""

import os
import sys

import numpy as np

REPO_URL = "https://github.com/huawei-bayerlab/marigold-v2"
WEIGHTS = "huawei-bayerlab/marigold-v2-0"
LICENSE_ID = "Apache-2.0 (code, LoRA checkpoints and Qwen-Image-Edit-2509 backbone)"
MODALITIES = ("depth", "normals", "albedo")
ALBEDO_GAMMA = 2.2          # folder_steps._linear_to_srgb: albedo.clamp_min(0) ** (1 / 2.2)

# Depth checkpoints under <assets>/checkpoints/Marigold-V2/depth/ -> (kind, label, description)
DEPTH_CHECKPOINTS = {
    "Log-stage2": ("log_depth", "Log (default)", "Affine-invariant log depth - the paper model"),
    "Log-layered": ("log_depth", "Log, see-through",
                    "See-through log depth: geometry behind glass / windows"),
    "Log-stage1": ("log_depth", "Log, stage 1", "Stage-1 model (before SinkLoss fine-tuning)"),
    "Uniform-base": ("depth", "Linear", "Affine-invariant linear depth (Marigold V1 style)"),
    "Disparity-base": ("disparity", "Disparity", "Affine-invariant inverse depth"),
    "Disparity-layered": ("disparity", "Disparity, see-through", "See-through inverse depth"),
    "Uniform-layered": ("depth", "Linear, see-through", "See-through linear depth"),
}
DEFAULT_DEPTH_CHECKPOINT = "Log-stage2"


def depth_kind(checkpoint):
    return DEPTH_CHECKPOINTS[checkpoint][0]


def assets_dir(repo, assets=""):
    """``$DEPTH_ASSETS_DIR`` if given, else ``<repo>/assets`` (as infer.py does)."""
    return os.path.abspath(os.path.expanduser(assets)) if assets else os.path.join(os.path.abspath(repo), "assets")


def checkpoint_path(repo, assets, modality, depth_checkpoint=DEFAULT_DEPTH_CHECKPOINT):
    root = os.path.join(assets_dir(repo, assets), "checkpoints", "Marigold-V2")
    if modality == "depth":
        return os.path.join(root, "depth", depth_checkpoint)
    return os.path.join(root, modality)


def check_setup(python, repo, assets="", modalities=("depth",), depth_checkpoint=DEFAULT_DEPTH_CHECKPOINT):
    """List what is missing for a run; empty list = looks ready (files only, no GPU check)."""
    problems = []
    if not python or not os.path.isfile(python):
        problems.append("Python of the marigold-v2 environment not set or not found "
                        "(e.g. ~/miniconda3/envs/marigold-v2/bin/python)")
    if not repo or not os.path.isfile(os.path.join(repo, "scripts", "infer.py")):
        problems.append("marigold-v2 repository not set or scripts/infer.py missing "
                        f"(git clone {REPO_URL})")
        return problems
    root = assets_dir(repo, assets)
    if not os.path.isdir(os.path.join(root, "checkpoints", "Qwen-Image-Edit-2509")):
        problems.append(f"Qwen-Image-Edit-2509 not found in {root}/checkpoints - run "
                        "'python scripts/download_assets.py --skip-datasets' in the repo")
    for m in modalities:
        ckpt = checkpoint_path(repo, assets, m, depth_checkpoint)
        if not os.path.isdir(ckpt):
            problems.append(f"Marigold V2 {m} checkpoint missing: {ckpt}")
    return problems


def inference_size(width, height, long_side):
    """(W, H) multiples of 16 with the image's aspect, or None for native resolution."""
    long_side = int(long_side)
    if long_side <= 0:
        return None
    scale = long_side / max(width, height)

    def r16(v):
        return max(16, int(round(v * scale / 16.0)) * 16)

    return r16(width), r16(height)


def build_command(python, repo, modality, image_dir, output_dir, checkpoint=None, size=None, seed=2025):
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    cmd = [python, os.path.join(repo, "scripts", "infer.py"), "--modality", modality,
           "--image_dir", image_dir, "--output_dir", output_dir, "--seed", str(int(seed))]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if size is not None:
        cmd += ["--width", str(int(size[0])), "--height", str(int(size[1]))]
    return cmd


def build_env(assets="", base=None):
    """Child environment: Blender's own Python settings must not leak into the conda env."""
    env = dict(os.environ if base is None else base)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE"):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"            # stream progress lines as they happen
    if assets:
        env["DEPTH_ASSETS_DIR"] = os.path.abspath(os.path.expanduser(assets))
    return env


def prediction_path(output_dir, image_filename):
    stem = os.path.splitext(os.path.basename(image_filename))[0]
    return os.path.join(output_dir, "images", "predictions_npy", stem + ".npy")


def load_prediction(path, modality):
    """Read an infer.py .npy and convert it to the add-on's conventions.

    depth   -> [H, W] in [0, 1] ((clip(x, -1, 1) + 1) / 2; direction per checkpoint kind)
    normals -> [H, W, 3] unit camera-space normals (invalid pixels: facing the camera)
    albedo  -> [H, W, 3] linear albedo
    """
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if modality == "depth":
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected [H, W] depth, got {arr.shape}")
        arr = np.where(np.isfinite(arr), arr, 1.0)
        return ((np.clip(arr, -1.0, 1.0) + 1.0) * 0.5).astype(np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"{path}: expected [3, H, W] {modality}, got {arr.shape}")
    hwc = np.ascontiguousarray(arr.transpose(1, 2, 0))
    if modality == "normals":
        norm = np.linalg.norm(hwc, axis=-1, keepdims=True)
        ok = norm > 1e-6
        n = np.where(ok, hwc / np.maximum(norm, 1e-6), np.array([0.0, 0.0, 1.0], np.float32))
        return n.astype(np.float32)
    if modality == "albedo":
        return np.power(np.clip(hwc, 0.0, None), ALBEDO_GAMMA).astype(np.float32)
    raise ValueError(f"unknown modality {modality!r}")


def default_python_hint():
    """Typical conda env location, shown as a hint in the preferences."""
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        return os.path.join(home, "miniconda3", "envs", "marigold-v2", "python.exe")
    return os.path.join(home, "miniconda3", "envs", "marigold-v2", "bin", "python")
