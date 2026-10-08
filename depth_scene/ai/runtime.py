"""ONNX Runtime loading, provider selection and session caching."""

import importlib
import os
import sys

_SESSIONS = {}

# Preferred order when device is AUTO / GPU.
GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",       # Windows DirectML (any DX12 GPU)
    "CoreMLExecutionProvider",    # macOS
    "ROCMExecutionProvider",
)


def add_site_dir(path):
    """Make packages installed by the add-on (pip --target) importable."""
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
        importlib.invalidate_caches()


def import_ort():
    """Return the onnxruntime module or None if it is not installed."""
    try:
        import onnxruntime as ort  # noqa: F401
        return ort
    except ImportError:
        return None


def ort_version():
    ort = import_ort()
    return getattr(ort, "__version__", None) if ort else None


def available_providers():
    ort = import_ort()
    return list(ort.get_available_providers()) if ort else []


def choose_providers(device):
    """Provider list for ``device`` in {'AUTO', 'GPU', 'CPU'}."""
    available = available_providers()
    if device == "CPU":
        return ["CPUExecutionProvider"]
    gpus = [p for p in GPU_PROVIDERS if p in available]
    if device == "GPU" and not gpus:
        raise RuntimeError(
            "No GPU execution provider available. Installed ONNX Runtime offers: "
            + ", ".join(available) + ". Install the GPU/DirectML package in the add-on preferences."
        )
    return gpus[:1] + ["CPUExecutionProvider"]


def get_session(model_path, device="AUTO"):
    ort = import_ort()
    if ort is None:
        raise RuntimeError("ONNX Runtime is not installed. Open Preferences > Add-ons > Depth Scene and click 'Install ONNX Runtime'.")
    providers = choose_providers(device)
    key = (os.path.abspath(model_path), os.path.getmtime(model_path), tuple(providers))
    sess = _SESSIONS.get(key)
    if sess is None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        _SESSIONS.clear()          # keep only one model in memory
        _SESSIONS[key] = sess
    return sess


def clear_sessions():
    _SESSIONS.clear()
