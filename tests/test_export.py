"""ONNX export of every Marigold task vs the diffusers pipelines (tiny random weights).

Needs torch + diffusers + onnx (skipped otherwise):

    python -m pytest tests/test_export.py
"""

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("export_marigold_onnx",
                                               os.path.join(ROOT, "tools", "export_marigold_onnx.py"))
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


@pytest.mark.parametrize("task", ["depth", "normals", "iid-appearance", "iid-lighting"])
def test_export_matches_diffusers(tmp_path, task):
    pipe = exporter.random_pipeline(task, tiny=True)
    out = exporter.export_folder(pipe, task, tmp_path / task)
    import json

    cfg = json.loads((out / "marigold.json").read_text())
    assert cfg["task"] == exporter.TASKS[task]["task"]
    assert cfg["n_targets"] == exporter.n_targets_of(exporter.TASKS[task])
    assert exporter.check(pipe, out, steps=2, size=64, task=task) < 2e-4
