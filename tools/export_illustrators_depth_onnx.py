"""Convert the Illustrator's Depth checkpoint to ONNX for the Depth Scene add-on.

Illustrator's Depth (Maruani et al., CVPR 2026, https://github.com/nissmar/illustrators_depth)
predicts a per-pixel *layer index* for illustrations. The authors publish a
PyTorch Lightning checkpoint; Blender's Python has no PyTorch, so convert it
once to ONNX on a machine where the authors' environment is installed:

    git clone --recursive https://github.com/nissmar/illustrators_depth.git
    cd illustrators_depth
    # follow their README: conda env, `pip install -e ml-depth-pro`, gdown the checkpoint
    pip install onnx onnxruntime            # + onnxconverter-common for --fp16
    python /path/to/2D-to-3D-TrueDepth/tools/export_illustrators_depth_onnx.py \
        --ckpt checkpoints/mmsvg_model/checkpoints/id_model.ckpt --check

This writes ``illustrators_depth_fp32.onnx`` + ``illustrators_depth_fp32.onnx.data``
(~2.5 GB of weights; ONNX keeps tensors > 2 GB in an external file). Copy BOTH
files into the add-on's models folder (Preferences > Add-ons > Depth Scene).
``--fp16`` additionally writes ``illustrators_depth_fp16.onnx`` (~1.3 GB, for GPUs).

LICENCE: the weights are under the Adobe Research License - noncommercial
research purposes only (academic research and teaching). The exported ONNX
files are derivative works of those weights and carry the same restriction:
do not use them commercially or redistribute them outside those terms.

``--random-init`` builds the same architecture with random weights (needs only
torch + Apple's depth_pro package). It exists to test the export pipeline and
measure speed without the checkpoint; its output is meaningless.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch

SIZE = 1536          # Depth Pro's encoder is hard-wired to 1536 x 1536
INPUT_NAME = "image"
OUTPUT_NAME = "layer_depth"


class LayerDepthNet(torch.nn.Module):
    """Depth Pro encoder/decoder/head; returns the authors' -1/clamp(inverse) map."""

    def __init__(self, encoder, decoder, head):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

    def forward(self, image):
        features, _ = self.decoder(self.encoder(image))
        inverse = self.head(features)
        return -1.0 / torch.clamp(inverse, min=1e-4, max=1e4)


def load_checkpoint(ckpt, repo):
    """Load with the authors' own loader (needs their repo on sys.path)."""
    sys.path.insert(0, str(Path(repo).resolve()))
    from src.model.illustrators_depth_model import load_illustrators_depth_model

    # Their loader finds hparams.yaml by splitting the path on '/'.
    model, _ = load_illustrators_depth_model(Path(ckpt).resolve().as_posix(), device="cpu")
    return LayerDepthNet(model.encoder, model.decoder, model.head)


def random_init():
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=None,
                              use_fov_head=False, fov_encoder_preset=None)
    dp, _ = depth_pro.create_model_and_transforms(config=cfg)
    return LayerDepthNet(dp.encoder, dp.decoder, dp.head)


def export(net, out_path, opset):
    import onnx

    out_path = Path(out_path).resolve()
    tmp_dir = out_path.parent / (out_path.stem + "_tmp_export")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / "model.onnx"
    dummy = torch.zeros(1, 3, SIZE, SIZE)
    with torch.no_grad():
        torch.onnx.export(net, (dummy,), str(tmp), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    # Models > 2 GB come out as one file per tensor; repack into a single .data file.
    model = onnx.load(str(tmp), load_external_data=True)
    data_name = out_path.name + ".data"
    for p in (out_path, out_path.parent / data_name):
        if p.exists():
            p.unlink()
    onnx.save_model(model, str(out_path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data_name, size_threshold=1024)
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()
    return out_path


def to_fp16(fp32_path, out_path):
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(fp32_path), load_external_data=True)
    model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save_model(model, str(out_path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=Path(out_path).name + ".data", size_threshold=1024)
    return out_path


def check(net, onnx_path):
    import onnxruntime as ort

    x = torch.rand(1, 3, SIZE, SIZE) * 2 - 1
    t0 = time.perf_counter()
    with torch.no_grad():
        ref = net(x).numpy()
    t_torch = time.perf_counter() - t0
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    got = sess.run(None, {INPUT_NAME: x.numpy()})[0]
    t_ort = time.perf_counter() - t0
    err = np.abs(got - ref)
    scale = max(float(np.abs(ref).max()), 1e-6)
    print(f"output {got.shape}  max|diff| {err.max():.3e}  (rel {err.max() / scale:.3e})")
    print(f"torch CPU {t_torch:.1f}s   onnxruntime CPU {t_ort:.1f}s")
    return float(err.max() / scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", help="id_model.ckpt from the authors' Google Drive zip")
    ap.add_argument("--repo", default=".", help="illustrators_depth checkout (default: current dir)")
    ap.add_argument("--out", default="illustrators_depth_fp32.onnx")
    ap.add_argument("--fp16", action="store_true", help="also write illustrators_depth_fp16.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", action="store_true", help="compare ONNX Runtime against PyTorch")
    ap.add_argument("--random-init", action="store_true", help="test the pipeline without weights")
    args = ap.parse_args()
    if not args.random_init and not args.ckpt:
        ap.error("--ckpt is required (or --random-init for a dry run)")

    net = random_init() if args.random_init else load_checkpoint(args.ckpt, args.repo)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"parameters: {n_params / 1e6:.1f} M")

    t0 = time.perf_counter()
    fp32 = export(net, args.out, args.opset)
    size = sum(f.stat().st_size for f in fp32.parent.glob(fp32.name + "*")) / 1e9
    print(f"wrote {fp32} (+ .data), {size:.2f} GB, {time.perf_counter() - t0:.0f}s")
    if args.check:
        check(net, fp32)
    if args.fp16:
        name = fp32.name.replace("fp32", "fp16") if "fp32" in fp32.name else fp32.stem + "_fp16.onnx"
        fp16 = to_fp16(fp32, fp32.with_name(name))
        print(f"wrote {fp16} (+ .data)")
    print("\nCopy the .onnx AND .onnx.data files into the Depth Scene models folder.")
    print("Licence: Adobe Research License - noncommercial research use only.")


if __name__ == "__main__":
    main()
