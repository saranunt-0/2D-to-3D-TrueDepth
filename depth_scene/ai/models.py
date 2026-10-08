"""Registry of supported depth models.

Two families, both run with ONNX Runtime (no PyTorch needed):

* Depth Anything V2 (S / B / L) - relative inverse depth ("disparity").
  Same ViT-S/B/L sizes TrueDepth exposes. Only Small is Apache-2.0; Base and
  Large weights are CC-BY-NC-4.0 (non-commercial use only).
* MoGe-2 (S / B) - monocular *geometry*: metric depth plus the camera field
  of view, which is what perspective reconstruction of a corridor needs.
  MIT licence.
"""

HF = "https://huggingface.co"


class ModelInfo:
    def __init__(self, key, label, family, size, params, license_id, commercial_ok,
                 files, output_kind, description):
        self.key = key
        self.label = label
        self.family = family            # "depth_anything_v2" | "moge2"
        self.size = size                # "ViT-S" ...
        self.params = params
        self.license_id = license_id
        self.commercial_ok = commercial_ok
        self.files = files              # precision -> (url, approx MB)
        self.output_kind = output_kind  # "disparity" | "metric"
        self.description = description

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


MODELS = {
    m.key: m
    for m in (
        _da2("small", "ViT-S", "24.8M", "Apache-2.0", True, 99, 50),
        _da2("base", "ViT-B", "97.5M", "CC-BY-NC-4.0", False, 390, 195),
        _da2("large", "ViT-L", "335M", "CC-BY-NC-4.0", False, 1340, 670),
        _moge("s", "ViT-S", "35M", 141),
        _moge("b", "ViT-B", "104M", 420),
    )
}

DEFAULT_MODEL = "da2_small"


def get(key):
    return MODELS[key]
