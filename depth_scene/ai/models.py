"""Registry of supported depth models.

Three families, all run with ONNX Runtime (no PyTorch needed):

* Depth Anything V2 (S / B / L) - relative inverse depth ("disparity").
  Same ViT-S/B/L sizes TrueDepth exposes. Only Small is Apache-2.0; Base and
  Large weights are CC-BY-NC-4.0 (non-commercial use only).
* MoGe-2 (S / B) - monocular *geometry*: metric depth plus the camera field
  of view, which is what perspective reconstruction of a corridor needs.
  MIT licence.
* Illustrator's Depth (Maruani et al., CVPR 2026) - per-pixel *layer index*
  for illustrations / flat digital art, built on Apple Depth Pro (~0.64 B
  params, fixed 1536 px input). Weights are under the Adobe Research
  Licence (non-commercial research only) and ship as a PyTorch checkpoint,
  so there is no download URL: users convert it once with
  ``tools/export_illustrators_depth_onnx.py``.
"""

HF = "https://huggingface.co"


class ModelInfo:
    def __init__(self, key, label, family, size, params, license_id, commercial_ok,
                 files, output_kind, description, manual=""):
        self.key = key
        self.label = label
        self.family = family            # "depth_anything_v2" | "moge2" | "illustrators_depth"
        self.size = size                # "ViT-S" ...
        self.params = params
        self.license_id = license_id
        self.commercial_ok = commercial_ok
        self.files = files              # precision -> (url or None, approx MB)
        self.output_kind = output_kind  # "disparity" | "metric" | "layers"
        self.description = description
        self.manual = manual            # how to obtain the file when url is None

    def filename(self, precision):
        precision = precision if precision in self.files else "fp32"
        return f"{self.key}_{precision}.onnx"

    def url(self, precision):
        precision = precision if precision in self.files else "fp32"
        return self.files[precision][0]

    def size_mb(self, precision):
        precision = precision if precision in self.files else "fp32"
        return self.files[precision][1]


def _da2(size_word, size, params, lic, ok, mb32, mb16):
    repo = f"{HF}/onnx-community/depth-anything-v2-{size_word}/resolve/main/onnx"
    return ModelInfo(
        key=f"da2_{size_word}",
        label=f"Depth Anything V2 {size_word.title()}",
        family="depth_anything_v2",
        size=size,
        params=params,
        license_id=lic,
        commercial_ok=ok,
        files={"fp32": (f"{repo}/model.onnx", mb32), "fp16": (f"{repo}/model_fp16.onnx", mb16)},
        output_kind="disparity",
        description=f"Relative depth, {params} params, {lic}",
    )


def _moge(size_word, size, params, mb):
    url = f"{HF}/Ruicheng/moge-2-vit{size_word}-normal-onnx/resolve/main/model.onnx"
    return ModelInfo(
        key=f"moge2_vit{size_word}",
        label=f"MoGe-2 ViT-{size_word.upper()} (geometry + FOV)",
        family="moge2",
        size=size,
        params=params,
        license_id="MIT",
        commercial_ok=True,
        files={"fp32": (url, mb)},
        output_kind="metric",
        description=f"Metric depth + estimated FOV, {params} params, MIT",
    )


ILLUSTRATORS_DEPTH = ModelInfo(
    key="illustrators_depth",
    label="Illustrator's Depth (layers, research only)",
    family="illustrators_depth",
    size="Depth Pro (2x ViT-L)",
    params="~640M",
    license_id="Adobe Research (non-commercial research only)",
    commercial_ok=False,
    # No URL: the authors publish a PyTorch checkpoint on Google Drive.
    files={"fp32": (None, 2550), "fp16": (None, 1280)},
    output_kind="layers",
    description="Layer index for illustrations / flat art, ~640M params, Adobe Research licence",
    manual="Convert the checkpoint with tools/export_illustrators_depth_onnx.py "
           "(see README) and put the .onnx (+ .onnx.data) in the models folder",
)

MODELS = {
    m.key: m
    for m in (
        _da2("small", "ViT-S", "24.8M", "Apache-2.0", True, 99, 50),
        _da2("base", "ViT-B", "97.5M", "CC-BY-NC-4.0", False, 390, 195),
        _da2("large", "ViT-L", "335M", "CC-BY-NC-4.0", False, 1340, 670),
        _moge("s", "ViT-S", "35M", 141),
        _moge("b", "ViT-B", "104M", 420),
        ILLUSTRATORS_DEPTH,
    )
}

DEFAULT_MODEL = "da2_small"


def get(key):
    return MODELS[key]
