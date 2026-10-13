"""Registry of supported models - Marigold branch.

This branch ships one model family, Marigold (Ke et al., CVPR 2024 / TPAMI
2025, https://github.com/prs-eth/Marigold): Stable Diffusion 2 fine-tuned into
dense predictors, run with ONNX Runtime (no PyTorch). Four v1.1 checkpoints:

* depth            - affine-invariant depth: builds the mesh;
* normals          - camera-space surface normals: normal map;
* iid-appearance   - albedo + roughness + metallicity: PBR material;
* iid-lighting     - albedo + diffuse shading + non-diffuse residual
                     (image = albedo * shading + residual): de-lighting.

Licence (checked 2026-10-13): code Apache-2.0; all v1.1 weights CreativeML
Open RAIL++-M (repo README: "The models are licensed under RAIL++-M") -
commercial use is allowed, but use-based restrictions apply and must be passed
on when the weights (or ONNX conversions) are redistributed. Marigold derives
from Stable Diffusion 2, whose Open RAIL++-M lists the concrete restrictions.

The authors publish diffusers/PyTorch weights only, so users convert them once
with ``tools/export_marigold_onnx.py`` (needs torch + diffusers).
"""


class ModelInfo:
    def __init__(self, key, label, family, params, license_id, commercial_ok, files,
                 output_kind, description, manual="", task="depth", outputs=("depth",), checkpoint="",
                 export_task="depth"):
        self.key = key
        self.label = label
        self.family = family            # "marigold"
        self.params = params
        self.license_id = license_id
        self.commercial_ok = commercial_ok
        self.files = files              # precision -> approx MB (a folder of ONNX files)
        self.output_kind = output_kind  # depth task: "depth" (0 = near, 1 = far)
        self.description = description
        self.manual = manual
        self.task = task                # "depth" | "normals" | "intrinsics"
        self.outputs = outputs          # pass names written by builder.generate_passes
        self.checkpoint = checkpoint    # Hugging Face id to convert
        self.export_task = export_task  # tools/export_marigold_onnx.py --task

    def filename(self, precision):
        """Folder name holding vae_encoder/unet/vae_decoder.onnx + marigold.json."""
        precision = precision if precision in self.files else "fp32"
        return f"{self.key}_{precision}"

    def size_mb(self, precision):
        precision = precision if precision in self.files else "fp32"
        return self.files[precision]


def _marigold(key, label, task, outputs, checkpoint, description, export_task):
    return ModelInfo(
        key=key,
        label=label,
        family="marigold",
        params="~950M (SD2 U-Net 865M + VAE 84M)",
        license_id="CreativeML Open RAIL++-M",
        commercial_ok=True,
        # Measured export sizes; fp16 halves the U-Net only (VAE stays fp32).
        # The IID U-Nets differ only in their first/last convolution.
        files={"fp32": 3620, "fp16": 1970},
        output_kind="depth" if task == "depth" else task,
        description=description,
        manual=f"Convert {checkpoint} with tools/export_marigold_onnx.py --task {export_task} "
               "(see README) and put the output folder in the models folder",
        task=task,
        outputs=outputs,
        checkpoint=checkpoint,
        export_task=export_task,
    )


MARIGOLD_DEPTH_V1_1 = _marigold(
    "marigold_depth_v1_1", "Marigold Depth v1.1", "depth", ("depth",), "prs-eth/marigold-depth-v1-1",
    "Diffusion-based affine-invariant depth, ~950M params, Open RAIL++-M", "depth")
MARIGOLD_NORMALS_V1_1 = _marigold(
    "marigold_normals_v1_1", "Marigold Normals v1.1", "normals", ("normal",), "prs-eth/marigold-normals-v1-1",
    "Camera-space surface normals -> normal map", "normals")
MARIGOLD_IID_APPEARANCE_V1_1 = _marigold(
    "marigold_iid_appearance_v1_1", "Marigold IID Appearance v1.1", "intrinsics",
    ("albedo", "roughness", "metallic"), "prs-eth/marigold-iid-appearance-v1-1",
    "Albedo, roughness and metallicity (trained on indoor InteriorVerse renders)", "iid-appearance")
MARIGOLD_IID_LIGHTING_V1_1 = _marigold(
    "marigold_iid_lighting_v1_1", "Marigold IID Lighting v1.1", "intrinsics",
    ("albedo", "shading", "residual"), "prs-eth/marigold-iid-lighting-v1-1",
    "Albedo, diffuse shading and non-diffuse residual (trained on indoor Hypersim)", "iid-lighting")

MODELS = {m.key: m for m in (MARIGOLD_DEPTH_V1_1, MARIGOLD_NORMALS_V1_1, MARIGOLD_IID_APPEARANCE_V1_1,
                             MARIGOLD_IID_LIGHTING_V1_1)}
DEPTH_MODELS = {k: m for k, m in MODELS.items() if m.task == "depth"}

# Material passes (Scene toggles) -> model producing them.
PASS_MODELS = {
    "normals": MARIGOLD_NORMALS_V1_1.key,
    "appearance": MARIGOLD_IID_APPEARANCE_V1_1.key,
    "lighting": MARIGOLD_IID_LIGHTING_V1_1.key,
}

DEFAULT_MODEL = MARIGOLD_DEPTH_V1_1.key


def get(key):
    return MODELS[key]
