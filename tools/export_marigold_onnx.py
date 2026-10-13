"""Convert Marigold v1.1 checkpoints to ONNX for Depth Scene.

Marigold ships as diffusers/PyTorch pipelines. Blender's Python has no
PyTorch, so convert each model once on any machine with Python 3.10+:

    pip install torch diffusers transformers accelerate onnx onnxruntime
    pip install onnxconverter-common            # only for --fp16
    python tools/export_marigold_onnx.py --task depth          --check [--fp16]
    python tools/export_marigold_onnx.py --task normals        --check [--fp16]
    python tools/export_marigold_onnx.py --task iid-appearance --check [--fp16]
    python tools/export_marigold_onnx.py --task iid-lighting   --check [--fp16]

--task       checkpoint                          folder              output
depth        prs-eth/marigold-depth-v1-1         marigold_depth_v1_1  depth (builds the mesh)
normals      prs-eth/marigold-normals-v1-1       marigold_normals_v1_1  normal map
iid-appearance  prs-eth/marigold-iid-appearance-v1-1  marigold_iid_appearance_v1_1
                                                  albedo, roughness, metallicity
iid-lighting prs-eth/marigold-iid-lighting-v1-1  marigold_iid_lighting_v1_1
                                                  albedo, diffuse shading, residual

Each folder (``<folder>_fp32`` / ``_fp16``) holds

    vae_encoder.onnx   image [B,3,H,W] in [-1,1] -> latent (posterior mode x scaling)
    unet.onnx (+ unet.onnx.data, ~3.5 GB)
                       sample [B,4+4T,h,w], timestep int64 [1] -> noise [B,4T,h,w]
                       (T targets; empty-prompt CLIP embedding baked in: no text encoder)
    vae_decoder.onnx   depth: latent -> depth [B,1,H,W] in [0,1] (0 = near)
                       others: latent -> decoded [B,3,H,W] in [-1,1]
    marigold.json      task, targets, DDIM scheduler numbers, defaults, licence

--fp16 also writes ``<folder>_fp16`` (fp16 U-Net, ~2.0 GB; VAE kept fp32).
Copy the folder(s) into the add-on's models folder
(Preferences > Add-ons > Depth Scene > Models Folder).

LICENCE: Marigold weights are under CreativeML Open RAIL++-M (commercial use
allowed with use-based restrictions; Marigold derives from Stable Diffusion 2,
whose Open RAIL++-M lists them). The ONNX files are Derivatives of the Model:
if you redistribute them, include the licence and pass on the restrictions.

``--random-init`` builds the same architecture (SD2 U-Net with 4+4T input
channels, SD VAE, zero-SNR trailing DDIM) with random weights and no download,
to test the export and the add-on's NumPy denoising loop (``--tiny`` shrinks
it to a few MB for quick tests).
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# Marigold v1.1 target properties (prs-eth/Marigold config/train_marigold_iid_*.yaml);
# real checkpoints carry them in model_index.json, --random-init needs them here.
APPEARANCE_PROPS = {
    "target_names": ["albedo", "material"],
    "albedo": {"prediction_space": "srgb"},
    "material": {"prediction_space": "stack", "sub_target_names": ["roughness", "metallicity", None]},
    "roughness": {"prediction_space": "linear", "up_to_scale": False},
    "metallicity": {"prediction_space": "linear", "up_to_scale": False},
}
LIGHTING_PROPS = {
    "target_names": ["albedo", "shading", "residual"],
    "albedo": {"prediction_space": "linear", "up_to_scale": False},
    "shading": {"prediction_space": "linear", "up_to_scale": True},
    "residual": {"prediction_space": "linear", "up_to_scale": True},
}
TASKS = {
    "depth": dict(checkpoint="prs-eth/marigold-depth-v1-1", folder="marigold_depth_v1_1",
                  pipeline="MarigoldDepthPipeline", task="depth", prediction_type="depth"),
    "normals": dict(checkpoint="prs-eth/marigold-normals-v1-1", folder="marigold_normals_v1_1",
                    pipeline="MarigoldNormalsPipeline", task="normals", prediction_type="normals"),
    "iid-appearance": dict(checkpoint="prs-eth/marigold-iid-appearance-v1-1",
                           folder="marigold_iid_appearance_v1_1", pipeline="MarigoldIntrinsicsPipeline",
                           task="intrinsics", prediction_type="intrinsics", target_properties=APPEARANCE_PROPS),
    "iid-lighting": dict(checkpoint="prs-eth/marigold-iid-lighting-v1-1", folder="marigold_iid_lighting_v1_1",
                         pipeline="MarigoldIntrinsicsPipeline", task="intrinsics", prediction_type="intrinsics",
                         target_properties=LIGHTING_PROPS),
}


def n_targets_of(spec):
    return len(spec["target_properties"]["target_names"]) if "target_properties" in spec else 1


class VaeEncoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scale = float(vae.config.scaling_factor)

    def forward(self, image):
        return self.vae.encode(image).latent_dist.mode() * self.scale


class UNetDepth(torch.nn.Module):
    def __init__(self, unet, text_embedding):
        super().__init__()
        self.unet = unet
        self.register_buffer("text", text_embedding.detach().clone())

    def forward(self, sample, timestep):
        text = self.text.expand(sample.shape[0], -1, -1)
        return self.unet(sample, timestep, encoder_hidden_states=text, return_dict=False)[0]


class VaeDecoder(torch.nn.Module):
    """Depth: MarigoldDepthPipeline.decode_prediction (channel mean, clip, to [0, 1])."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scale = float(vae.config.scaling_factor)

    def forward(self, latent):
        x = self.vae.decode(latent / self.scale, return_dict=False)[0]
        x = torch.clip(x.mean(dim=1, keepdim=True), -1.0, 1.0)
        return (x + 1.0) / 2.0


class VaeDecoderRGB(torch.nn.Module):
    """Normals / intrinsics: decode + clip to [-1, 1]; the task-specific rest is NumPy."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scale = float(vae.config.scaling_factor)

    def forward(self, latent):
        return torch.clip(self.vae.decode(latent / self.scale, return_dict=False)[0], -1.0, 1.0)


def empty_text_embedding(pipe):
    """Exactly what MarigoldDepthPipeline computes on first call."""
    if getattr(pipe, "empty_text_embedding", None) is not None:
        return pipe.empty_text_embedding
    ids = pipe.tokenizer("", padding="do_not_pad", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        return pipe.text_encoder(ids)[0]      # [1, 2, 1024]


def random_pipeline(task="depth", tiny=False):
    import diffusers
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

    spec = TASKS[task]
    n_t = n_targets_of(spec)
    torch.manual_seed(0)
    if tiny:        # same graph structure, a few MB: for tests
        text_dim = 32
        unet = UNet2DConditionModel(
            sample_size=32, in_channels=4 + 4 * n_t, out_channels=4 * n_t, layers_per_block=1,
            block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=text_dim,
            attention_head_dim=8, use_linear_projection=True)
        vae = AutoencoderKL(
            in_channels=3, out_channels=3, latent_channels=4, layers_per_block=1,
            block_out_channels=(32, 32, 32, 32), down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4, sample_size=64, scaling_factor=0.18215)
    else:
        text_dim = 1024
        unet = UNet2DConditionModel(
            sample_size=96, in_channels=4 + 4 * n_t, out_channels=4 * n_t, layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
            up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
            cross_attention_dim=1024, attention_head_dim=(5, 10, 20, 20), use_linear_projection=True)
        vae = AutoencoderKL(
            in_channels=3, out_channels=3, latent_channels=4, layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            sample_size=768, scaling_factor=0.18215)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False, steps_offset=1, prediction_type="v_prediction",
        timestep_spacing="trailing", rescale_betas_zero_snr=True)
    kwargs = dict(unet=unet, vae=vae, scheduler=scheduler, text_encoder=None, tokenizer=None,
                  prediction_type=spec["prediction_type"], default_denoising_steps=1,
                  default_processing_resolution=768)
    if task == "normals":
        kwargs["use_full_z_range"] = True
    if "target_properties" in spec:
        kwargs["target_properties"] = spec["target_properties"]
    pipe = getattr(diffusers, spec["pipeline"])(**kwargs)
    pipe.empty_text_embedding = torch.randn(1, 2, text_dim)
    return pipe


def load_pipeline(model, task):
    import diffusers

    spec = TASKS[task]
    pipe = getattr(diffusers, spec["pipeline"]).from_pretrained(model, torch_dtype=torch.float32)
    pred = pipe.config.get("prediction_type")
    if pred != spec["prediction_type"]:
        raise SystemExit(f"{model} predicts '{pred}', but --task {task} expects '{spec['prediction_type']}' "
                         f"(disparity checkpoints are not supported)")
    return pipe


def task_config(pipe, task):
    """marigold.json fields describing what the U-Net predicts and how to decode it."""
    spec = TASKS[task]
    n_t = pipe.unet.config.out_channels // pipe.vae.config.latent_channels
    cfg = {"task": spec["task"], "n_targets": int(n_t),
           "decoder_output": "depth" if spec["task"] == "depth" else "rgb"}
    if spec["task"] == "normals":
        cfg["use_full_z_range"] = bool(getattr(pipe, "use_full_z_range", True))
    if spec["task"] == "intrinsics":
        props = pipe.target_properties
        if len(props["target_names"]) != n_t:
            raise SystemExit(f"target_names {props['target_names']} do not match {n_t} U-Net targets")
        cfg["target_properties"] = props
    return cfg


def scheduler_config(pipe, model):
    s = pipe.scheduler
    c = s.config
    if type(s).__name__ != "DDIMScheduler":
        raise SystemExit(f"{type(s).__name__} is not supported (the add-on implements DDIM only)")
    if c.get("clip_sample") or c.get("thresholding"):
        raise SystemExit("clip_sample / thresholding are not supported by the add-on's DDIM port")
    return {
        "format": 1,
        "model": model,
        "scaling_factor": float(pipe.vae.config.scaling_factor),
        "num_train_timesteps": int(c.num_train_timesteps),
        "timestep_spacing": c.timestep_spacing,
        "steps_offset": int(c.get("steps_offset", 0)),
        "prediction_type": c.prediction_type,
        "alphas_cumprod": [float(a) for a in s.alphas_cumprod.double()],
        "final_alpha_cumprod": float(s.final_alpha_cumprod),
        "default_denoising_steps": pipe.default_denoising_steps,
        "default_processing_resolution": pipe.default_processing_resolution,
        "license": "CreativeML Open RAIL++-M (use-based restrictions apply)",
    }


def export(module, args, path, inputs, outputs, dynamic_axes, opset):
    """torch.onnx.export, then repack > 2 GB weights into one <name>.data file."""
    import onnx

    path = Path(path)
    tmp_dir = path.parent / (path.stem + "_tmp_export")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(module, args, str(tmp), input_names=inputs, output_names=outputs,
                          dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
                          dynamo=False)
    model = onnx.load(str(tmp), load_external_data=True)
    data = path.parent / (path.name + ".data")
    for p in (path, data):
        if p.exists():
            p.unlink()
    big = sum(t.ByteSize() for t in model.graph.initializer) > 1.8e9
    if big:
        onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location=data.name, size_threshold=1024)
    else:
        onnx.save_model(model, str(path))
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


def unet_to_fp16(src, dst):
    """fp16 U-Net with float32 I/O (same fixes as the >2 GB Depth Pro export)."""
    import onnx
    from onnxconverter_common import float16

    src, dst = Path(src), Path(dst)
    shaped = src.with_name(src.stem + "_shapes.onnx")
    onnx.shape_inference.infer_shapes_path(str(src), str(shaped))
    model = onnx.load(str(shaped), load_external_data=True)
    shaped.unlink()
    model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    declared = {v.name: v.type.tensor_type.elem_type for v in model.graph.value_info}
    for node in model.graph.node:
        if node.op_type == "Cast" and declared.get(node.output[0]) == onnx.TensorProto.FLOAT16:
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                    attr.i = onnx.TensorProto.FLOAT16
    data = dst.parent / (dst.name + ".data")
    for p in (dst, data):
        if p.exists():
            p.unlink()
    onnx.save_model(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data.name, size_threshold=1024)


def export_folder(pipe, task, out, opset=17, model_name="random-init"):
    """Write the three graphs + marigold.json for ``pipe`` into ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_t = pipe.unet.config.out_channels // pipe.vae.config.latent_channels
    text = pipe.empty_text_embedding
    img = torch.zeros(1, 3, 64, 96)
    lat = torch.zeros(1, 4, 8, 12)
    export(VaeEncoder(pipe.vae), (img,), out / "vae_encoder.onnx", ["image"], ["latent"],
           {"image": {0: "batch", 2: "height", 3: "width"}, "latent": {0: "batch", 2: "h", 3: "w"}}, opset)
    export(UNetDepth(pipe.unet, text), (torch.zeros(1, 4 + 4 * n_t, 8, 12), torch.tensor([999], dtype=torch.int64)),
           out / "unet.onnx", ["sample", "timestep"], ["noise"],
           {"sample": {0: "batch", 2: "h", 3: "w"}, "noise": {0: "batch", 2: "h", 3: "w"}}, opset)
    if TASKS[task]["task"] == "depth":
        decoder, name = VaeDecoder(pipe.vae), "depth"
    else:
        decoder, name = VaeDecoderRGB(pipe.vae), "decoded"
    export(decoder, (lat,), out / "vae_decoder.onnx", ["latent"], [name],
           {"latent": {0: "batch", 2: "h", 3: "w"}, name: {0: "batch", 2: "height", 3: "width"}}, opset)
    cfg = scheduler_config(pipe, model_name)
    cfg.update(task_config(pipe, task))
    with open(out / "marigold.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    return out


def check(pipe, folder, steps, size=256, task="depth"):
    """Add-on NumPy/ONNX loop vs the diffusers Marigold pipeline, same noise."""
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT))
    from depth_scene.ai import marigold

    folder = Path(folder)
    n_t = pipe.unet.config.out_channels // pipe.vae.config.latent_channels
    rng = np.random.default_rng(1)
    h, w = size, size * 3 // 2                      # multiples of 8: no resize/pad differences
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    rgb = np.stack([np.sin(6 * xx) * 0.5 + 0.5, yy, np.cos(5 * yy + 3 * xx) * 0.5 + 0.5], -1)
    rgb = np.clip(rgb + rng.normal(0, 0.02, rgb.shape), 0, 1).astype(np.float32)
    latents = rng.standard_normal((1, 4 * n_t, h // 8, w // 8)).astype(np.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        # Channels-first tensor: diffusers' input check reads (H, W) from shape[-2:].
        ref = pipe(torch.from_numpy(rgb.transpose(2, 0, 1)[None].copy()), num_inference_steps=steps,
                   ensemble_size=1, processing_resolution=0,
                   match_input_resolution=False, latents=torch.from_numpy(latents),
                   output_type="np").prediction
    t_torch = time.perf_counter() - t0
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sessions = [ort.InferenceSession(str(folder / f), opts, providers=["CPUExecutionProvider"])
                for f in marigold.FILES[:3]]
    cfg = marigold.load_config(folder)
    t0 = time.perf_counter()
    got = marigold.predict(sessions, cfg, rgb, steps=steps, processing_resolution=0, latents=latents)
    t_ort = time.perf_counter() - t0
    kind = marigold.task_of(cfg)
    if kind == "depth":
        ref, got = ref[0, ..., 0], got
    elif kind == "normals":
        ref = ref[0]
    else:                                           # [T,H,W,3] vs {name: [H,W,3]}
        got = np.stack([got[name] for name in marigold.target_names(cfg)])
    err = np.abs(got - ref)
    sat = float(np.mean(np.abs(ref) >= 0.999)) if kind == "normals" else float(np.mean((ref <= 1e-3) | (ref >= 0.999)))
    print(f"{folder.name}: {steps} step(s) at {w}x{h}: max |diff| {err.max():.2e}, mean {err.mean():.2e} "
          f"(range {ref.min():.3f}..{ref.max():.3f}, {sat:.0%} at the clip limits); "
          f"diffusers {t_torch:.1f}s, add-on loop {t_ort:.1f}s")
    return float(err.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="depth", choices=sorted(TASKS), help="which Marigold model to convert")
    ap.add_argument("--model", default=None, help="Hugging Face id or local diffusers folder "
                                                  "(default: the task's prs-eth v1.1 checkpoint)")
    ap.add_argument("--out-dir", default=".", help="where to create the model folder(s)")
    ap.add_argument("--fp16", action="store_true", help="also write the fp16-U-Net variant")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", action="store_true", help="compare against the diffusers pipeline")
    ap.add_argument("--check-steps", type=int, default=2)
    ap.add_argument("--random-init", action="store_true", help="test the pipeline without weights")
    ap.add_argument("--tiny", action="store_true", help="with --random-init: a few-MB model for quick tests")
    args = ap.parse_args()
    spec = TASKS[args.task]
    model = args.model or spec["checkpoint"]
    folder = spec["folder"]

    pipe = random_pipeline(args.task, args.tiny) if args.random_init else load_pipeline(model, args.task)
    pipe.vae.eval()
    pipe.unet.eval()
    text = empty_text_embedding(pipe)
    pipe.empty_text_embedding = text
    n_unet = sum(p.numel() for p in pipe.unet.parameters()) / 1e6
    n_vae = sum(p.numel() for p in pipe.vae.parameters()) / 1e6
    print(f"U-Net {n_unet:.0f} M params, VAE {n_vae:.0f} M params, scheduler {type(pipe.scheduler).__name__} "
          f"({pipe.scheduler.config.prediction_type}, {pipe.scheduler.config.timestep_spacing})")

    t0 = time.perf_counter()
    out = export_folder(pipe, args.task, Path(args.out_dir).resolve() / f"{folder}_fp32", args.opset,
                        "random-init" if args.random_init else model)
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e9
    print(f"wrote {out} ({size:.2f} GB) in {time.perf_counter() - t0:.0f}s")
    if args.check:
        check(pipe, out, args.check_steps, size=64 if args.tiny else 256, task=args.task)
    if args.fp16:
        import shutil

        del pipe.unet
        gc.collect()
        out16 = out.with_name(f"{folder}_fp16")
        out16.mkdir(exist_ok=True)
        for name in ("vae_encoder.onnx", "vae_decoder.onnx", "marigold.json"):
            shutil.copy2(out / name, out16 / name)
        unet_to_fp16(out / "unet.onnx", out16 / "unet.onnx")
        print(f"wrote {out16} (fp16 U-Net; compare a few results against fp32)")
    print("\nCopy the folder(s) into the Depth Scene models folder.")
    print("Licence: CreativeML Open RAIL++-M - keep the use-based restrictions when redistributing.")


if __name__ == "__main__":
    main()
