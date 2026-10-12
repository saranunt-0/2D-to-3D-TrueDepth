"""Registry of supported depth models - Marigold branch.

This branch ships a single model family, Marigold depth (Ke et al., CVPR 2024,
https://github.com/prs-eth/Marigold): Stable Diffusion 2 fine-tuned into an
affine-invariant depth estimator, run with ONNX Runtime (no PyTorch).

Licence (checked 2026-10-12):
* code (prs-eth/Marigold): Apache-2.0;
* weights (prs-eth/marigold-depth-v1-1): CreativeML Open RAIL++-M - commercial
  use is allowed, but use-based restrictions apply and must be passed on when
  the weights (or ONNX conversions) are redistributed. Marigold is a derivative
  of Stable Diffusion 2, whose Open RAIL++-M lists the concrete restrictions.

The authors publish diffusers/PyTorch weights only, so users convert them once
with ``tools/export_marigold_onnx.py`` (needs torch + diffusers).
"""


class ModelInfo:
    def __init__(self, key, label, family, params, license_id, commercial_ok, files,
                 output_kind, description, manual=""):
        self.key = key
        self.label = label
        self.family = family            # "marigold"
        self.params = params
        self.license_id = license_id
        self.commercial_ok = commercial_ok
        self.files = files              # precision -> approx MB (a folder of ONNX files)
        self.output_kind = output_kind  # "depth": 0 = near, 1 = far
        self.description = description
        self.manual = manual

    def filename(self, precision):
        """Folder name holding vae_encoder/unet/vae_decoder.onnx + marigold.json."""
        precision = precision if precision in self.files else "fp32"
        return f"{self.key}_{precision}"

    def size_mb(self, precision):
        precision = precision if precision in self.files else "fp32"
        return self.files[precision]


MARIGOLD_DEPTH_V1_1 = ModelInfo(
    key="marigold_depth_v1_1",
    label="Marigold Depth v1.1",
    family="marigold",
    params="~950M (SD2 U-Net 865M + VAE 84M)",
    license_id="CreativeML Open RAIL++-M",
    commercial_ok=True,
    # Measured export sizes; fp16 halves the U-Net only (VAE stays fp32).
    files={"fp32": 3620, "fp16": 1970},
    output_kind="depth",
    description="Diffusion-based affine-invariant depth, ~950M params, Open RAIL++-M",
    manual="Convert prs-eth/marigold-depth-v1-1 with tools/export_marigold_onnx.py "
           "(see README) and put the output folder in the models folder",
)

MODELS = {m.key: m for m in (MARIGOLD_DEPTH_V1_1,)}

DEFAULT_MODEL = MARIGOLD_DEPTH_V1_1.key


def get(key):
    return MODELS[key]
