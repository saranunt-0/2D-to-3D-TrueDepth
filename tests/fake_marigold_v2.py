"""A stand-in for a Marigold V2 install (huawei-bayerlab/marigold-v2).

The real model needs a CUDA GPU with 17+ GB and a 20 B-parameter backbone, so
tests drive this fake ``scripts/infer.py`` instead. It has the same command
line, validates the same asset folders and writes the same output layout
(``OUT/images/predictions_npy/<stem>.npy``) with known values:

* depth    [H, W]    linear ramp left -> right from -0.8 to 0.8 (plus 1.5 at
                     one pixel, to check clipping to [-1, 1])
* normals  [3, H, W] (0.6, 0, 0.8) scaled by 2 (not unit: the add-on renormalises),
                     one zero vector (invalid pixel)
* albedo   [3, H, W] 0.5 ** (1 / 2.2)   -> linear 0.5
"""

import os
import textwrap

INFER = textwrap.dedent('''
    import argparse, os, struct, sys
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="depth", choices=("depth", "normals", "albedo"))
    ap.add_argument("--checkpoint")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets = os.environ.get("DEPTH_ASSETS_DIR", os.path.join(repo, "assets"))
    if not os.path.isdir(os.path.join(assets, "checkpoints", "Qwen-Image-Edit-2509")):
        sys.exit("FileNotFoundError: Qwen checkpoint not found")
    if os.environ.get("FAKE_MV2_FAIL") == a.modality:
        print("torch.OutOfMemoryError: CUDA out of memory", flush=True)
        sys.exit(1)
    log = os.path.join(a.output_dir, "calls.txt")
    os.makedirs(a.output_dir, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(" ".join(sys.argv[1:]) + "\\n")
    for name in sorted(os.listdir(a.image_dir)):
        with open(os.path.join(a.image_dir, name), "rb") as fh:
            head = fh.read(24)
        assert head[:8] == b"\\x89PNG\\r\\n\\x1a\\n", "fake infer reads PNG only"
        w, h = struct.unpack(">II", head[16:24])
        if a.width:                       # fixed resolution: infer.py does not resize back
            w, h = a.width, a.height
        print(f"Modality: {a.modality}; {name} {w}x{h}", flush=True)
        if a.modality == "depth":
            out = np.tile(np.linspace(-0.8, 0.8, w, dtype=np.float32), (h, 1))
            out[0, 0] = 1.5
        elif a.modality == "normals":
            out = np.zeros((3, h, w), np.float32)
            out[0], out[2] = 1.2, 1.6
            out[:, 0, 0] = 0.0
        else:
            out = np.full((3, h, w), 0.5 ** (1 / 2.2), np.float32)
        dst = os.path.join(a.output_dir, "images", "predictions_npy")
        os.makedirs(dst, exist_ok=True)
        np.save(os.path.join(dst, os.path.splitext(name)[0] + ".npy"), out)
''')


def make_fake_repo(root, with_assets=True, depth_checkpoints=("Log-stage2",)):
    """Create <root>/marigold-v2 with scripts/infer.py and (optionally) the asset folders."""
    repo = os.path.join(root, "marigold-v2")
    os.makedirs(os.path.join(repo, "scripts"), exist_ok=True)
    with open(os.path.join(repo, "scripts", "infer.py"), "w", encoding="utf-8") as fh:
        fh.write(INFER)
    if with_assets:
        ck = os.path.join(repo, "assets", "checkpoints")
        os.makedirs(os.path.join(ck, "Qwen-Image-Edit-2509"), exist_ok=True)
        for name in depth_checkpoints:
            os.makedirs(os.path.join(ck, "Marigold-V2", "depth", name), exist_ok=True)
        for name in ("normals", "albedo"):
            os.makedirs(os.path.join(ck, "Marigold-V2", name), exist_ok=True)
    return repo
