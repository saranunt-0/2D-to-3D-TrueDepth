"""Marigold V2 external backend: command line, outputs and conversions (no GPU, no bpy).

A fake ``scripts/infer.py`` (tests/fake_marigold_v2.py) stands in for the real
20 B-parameter model; it mirrors the CLI and output layout of
huawei-bayerlab/marigold-v2 at commit 9926d93.
"""

import os
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_scene.ai import marigold_v2 as v2  # noqa: E402
from fake_marigold_v2 import make_fake_repo  # noqa: E402


def write_png(path, w, h):
    """Minimal 8-bit RGB PNG (Pillow-free)."""
    raw = b"".join(b"\x00" + bytes([128, 64, 32]) * w for _ in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


def test_checkpoint_kinds():
    assert v2.depth_kind("Log-stage2") == "log_depth"
    assert v2.depth_kind("Log-layered") == "log_depth"
    assert v2.depth_kind("Uniform-base") == "depth"
    assert v2.depth_kind("Disparity-base") == "disparity"
    assert v2.DEFAULT_DEPTH_CHECKPOINT == "Log-stage2"


@pytest.mark.parametrize("w,h,side,expected", [(2000, 1459, 1024, (1024, 752)), (447, 447, 1024, (1024, 1024)),
                                               (1263, 680, 768, (768, 416)), (100, 50, 0, None)])
def test_inference_size_keeps_aspect_in_multiples_of_16(w, h, side, expected):
    size = v2.inference_size(w, h, side)
    assert size == expected
    if size:
        assert size[0] % 16 == 0 and size[1] % 16 == 0


def test_command_and_environment():
    cmd = v2.build_command("/env/bin/python", "/repo", "normals", "/in", "/out", size=(1024, 752), seed=7,
                           checkpoint="/ck/normals")
    assert cmd[:4] == ["/env/bin/python", os.path.join("/repo", "scripts", "infer.py"), "--modality", "normals"]
    assert cmd[cmd.index("--width") + 1] == "1024" and cmd[cmd.index("--height") + 1] == "752"
    assert cmd[cmd.index("--seed") + 1] == "7" and cmd[cmd.index("--checkpoint") + 1] == "/ck/normals"
    assert "--width" not in v2.build_command("p", "r", "depth", "i", "o")
    with pytest.raises(ValueError):
        v2.build_command("p", "r", "materials", "i", "o")
    env = v2.build_env("/data/assets", base={"PYTHONHOME": "/blender/python", "PYTHONPATH": "x", "PATH": "/bin"})
    assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
    assert env["DEPTH_ASSETS_DIR"] == os.path.abspath("/data/assets") and env["PATH"] == "/bin"


def test_check_setup_reports_what_is_missing(tmp_path):
    problems = v2.check_setup("", "", "")
    assert any("Python" in p for p in problems) and any("repository" in p for p in problems)
    repo = make_fake_repo(str(tmp_path), with_assets=False)
    problems = v2.check_setup(sys.executable, repo, "", modalities=("depth", "normals"))
    assert any("Qwen-Image-Edit-2509" in p for p in problems)
    repo = make_fake_repo(str(tmp_path))
    assert v2.check_setup(sys.executable, repo, "", modalities=("depth", "normals", "albedo")) == []
    problems = v2.check_setup(sys.executable, repo, "", depth_checkpoint="Log-layered")
    assert problems == [f"Marigold V2 depth checkpoint missing: {v2.checkpoint_path(repo, '', 'depth', 'Log-layered')}"]


@pytest.mark.parametrize("size", [None, (64, 48)])
def test_fake_run_outputs_convert_to_addon_conventions(tmp_path, size):
    repo = make_fake_repo(str(tmp_path))
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    write_png(str(image_dir / "corridor.png"), 40, 30)
    out = {}
    for modality in v2.MODALITIES:
        out_dir = str(tmp_path / f"out_{modality}")
        cmd = v2.build_command(sys.executable, repo, modality, str(image_dir), out_dir, size=size)
        subprocess.run(cmd, check=True, env=v2.build_env(), capture_output=True)
        out[modality] = v2.load_prediction(v2.prediction_path(out_dir, "corridor.png"), modality)
    h, w = (30, 40) if size is None else (48, 64)
    depth = out["depth"]
    assert depth.shape == (h, w)
    assert depth[0, 0] == 1.0                                  # 1.5 clipped to 1 -> (1 + 1) / 2
    assert depth[1, 0] == pytest.approx(0.1) and depth[1, -1] == pytest.approx(0.9)
    n = out["normals"]
    assert n.shape == (h, w, 3)
    assert np.allclose(n[1, 1], [0.6, 0.0, 0.8], atol=1e-6)     # renormalised
    assert np.allclose(n[0, 0], [0.0, 0.0, 1.0])                # zero vector -> faces the camera
    assert np.allclose(out["albedo"], 0.5, atol=1e-6)           # gamma 2.2 undone


def test_failure_exit_code_is_visible(tmp_path):
    repo = make_fake_repo(str(tmp_path))
    image_dir = tmp_path / "in"
    image_dir.mkdir()
    write_png(str(image_dir / "x.png"), 8, 8)
    env = v2.build_env()
    env["FAKE_MV2_FAIL"] = "depth"
    res = subprocess.run(v2.build_command(sys.executable, repo, "depth", str(image_dir), str(tmp_path / "o")),
                         env=env, capture_output=True, text=True)
    assert res.returncode == 1 and "out of memory" in res.stdout
