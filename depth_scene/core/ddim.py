"""Deterministic DDIM scheduler in NumPy (eta = 0).

A line-for-line port of the parts of ``diffusers.DDIMScheduler`` that the
Marigold depth pipeline uses (``set_timesteps`` + ``step``), so the denoising
loop can run inside Blender without PyTorch. The scheduler's numbers
(``alphas_cumprod``, spacing, prediction type, ...) are not hard-coded: the
export script copies them from the checkpoint's own scheduler config, which
keeps zero-SNR / trailing-timestep checkpoints (Marigold v1.1) exact.
"""

import numpy as np


class DDIM:
    def __init__(self, alphas_cumprod, final_alpha_cumprod, num_train_timesteps=1000,
                 timestep_spacing="leading", steps_offset=0, prediction_type="epsilon"):
        self.alphas_cumprod = np.asarray(alphas_cumprod, dtype=np.float64)
        self.final_alpha_cumprod = float(final_alpha_cumprod)
        self.num_train_timesteps = int(num_train_timesteps)
        self.timestep_spacing = timestep_spacing
        self.steps_offset = int(steps_offset)
        if prediction_type not in ("epsilon", "v_prediction", "sample"):
            raise ValueError(f"unsupported prediction_type {prediction_type!r}")
        self.prediction_type = prediction_type
        self.num_inference_steps = None

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg["alphas_cumprod"], cfg["final_alpha_cumprod"], cfg["num_train_timesteps"],
                   cfg["timestep_spacing"], cfg.get("steps_offset", 0), cfg["prediction_type"])

    def timesteps(self, num_inference_steps):
        """Same schedule as ``DDIMScheduler.set_timesteps``."""
        n = int(num_inference_steps)
        T = self.num_train_timesteps
        if n < 1 or n > T:
            raise ValueError(f"num_inference_steps must be in [1, {T}]")
        self.num_inference_steps = n
        if self.timestep_spacing == "linspace":
            ts = np.linspace(0, T - 1, n).round()[::-1].copy().astype(np.int64)
        elif self.timestep_spacing == "leading":
            ratio = T // n
            ts = (np.arange(0, n) * ratio).round()[::-1].copy().astype(np.int64)
            ts += self.steps_offset
        elif self.timestep_spacing == "trailing":
            ratio = T / n
            ts = np.round(np.arange(T, 0, -ratio)).astype(np.int64)
            ts -= 1
        else:
            raise ValueError(f"unsupported timestep_spacing {self.timestep_spacing!r}")
        return ts

    def step(self, model_output, timestep, sample):
        """One deterministic DDIM update (``DDIMScheduler.step`` with eta=0)."""
        if self.num_inference_steps is None:
            raise RuntimeError("call timesteps() first")
        t = int(timestep)
        prev_t = t - self.num_train_timesteps // self.num_inference_steps
        a_t = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        b_t = 1.0 - a_t
        out = np.asarray(model_output, dtype=np.float64)
        x = np.asarray(sample, dtype=np.float64)
        if self.prediction_type == "epsilon":
            x0 = (x - b_t ** 0.5 * out) / a_t ** 0.5
            eps = out
        elif self.prediction_type == "sample":
            x0 = out
            eps = (x - a_t ** 0.5 * x0) / b_t ** 0.5
        else:  # v_prediction
            x0 = a_t ** 0.5 * x - b_t ** 0.5 * out
            eps = a_t ** 0.5 * out + b_t ** 0.5 * x
        prev = a_prev ** 0.5 * x0 + (1.0 - a_prev) ** 0.5 * eps
        return prev.astype(np.float32)
