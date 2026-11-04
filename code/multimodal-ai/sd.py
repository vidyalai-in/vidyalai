"""Stable Diffusion 1.5 helpers for Beyond Text Part 6 (v06, latent diffusion).

Model: stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M, ungated), fp16 variant only,
loaded straight to the GPU. The safety checker that ships with it stays on.

Contents
- load_pipe(): the diffusers StableDiffusionPipeline in fp16 on cuda (safety checker included).
- set_sampler(pipe, name): "ddim", "euler", "dpmpp2m" (DPM-Solver++ 2M), "pndm" (the shipped default).
- encode(pipe, prompts, negative=""): CLIP ViT-L/14 text states (B, 77, 768) for prompt and unconditional branch.
- cfg_sample(...): a plain classifier-free-guidance sampling loop that can also return the decoded
  prediction of the clean latent (x0) at chosen steps. Mirrors what the pipeline does.
- decode(pipe, latents): VAE decode to PIL images.
- safety(pipe, images): run the shipped checker, return (flags, max margin to a concept threshold).
- ClipJudge: openai/clip-vit-base-patch32 (MIT) image/text embeddings for CLIP score and diversity.
- GpuMonitor: samples SM clock and power with nvidia-smi in the background while a block runs.
- cuda_median(fn, n, warmup): median wall time of fn() with torch.cuda.synchronize around each call.
- wilson(k, n), boot_ci(values): 95% intervals.
"""
from __future__ import annotations

import math
import os
import statistics
import subprocess
import time

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/home/amit/.cache/vidyalai-scratch/hf")
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
VAE_SCALE = 0.18215  # vae.config.scaling_factor for SD 1.x


def load_pipe(safety: bool = True):
    from diffusers import StableDiffusionPipeline
    kw = {} if safety else {"safety_checker": None, "requires_safety_checker": False}
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, dtype=torch.float16, variant="fp16", **kw)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def set_sampler(pipe, name: str):
    from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                           PNDMScheduler)
    cfg = pipe.scheduler.config
    if name == "ddim":
        s = DDIMScheduler.from_config(cfg)
    elif name == "euler":
        s = EulerDiscreteScheduler.from_config(cfg)
    elif name == "dpmpp2m":
        s = DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="dpmsolver++", solver_order=2)
    elif name == "pndm":
        s = PNDMScheduler.from_config(cfg)
    else:
        raise ValueError(name)
    pipe.scheduler = s
    return s


@torch.no_grad()
def encode(pipe, prompts, negative=""):
    """Return (cond, uncond), each (B, 77, 768) fp16: the last hidden states of CLIP ViT-L/14's text model."""
    if isinstance(prompts, str):
        prompts = [prompts]
    neg = [negative] * len(prompts) if isinstance(negative, str) else list(negative)
    tok = pipe.tokenizer
    def run(texts):
        ids = tok(texts, padding="max_length", max_length=tok.model_max_length, truncation=True,
                  return_tensors="pt").input_ids.to("cuda")
        return pipe.text_encoder(ids)[0]
    return run(prompts), run(neg)


@torch.no_grad()
def cfg_sample(pipe, cond, uncond, steps=20, guidance=7.5, seed=0, keep_x0_at=(), size=512):
    """Classifier-free guidance loop. Returns final latents and {step_index: x0_latent} for keep_x0_at."""
    sched = pipe.scheduler
    sched.set_timesteps(steps, device="cuda")
    b = cond.shape[0]
    g = torch.Generator("cuda").manual_seed(seed)
    lat = torch.randn((b, 4, size // 8, size // 8), generator=g, device="cuda", dtype=torch.float16)
    lat = lat * sched.init_noise_sigma
    ctx = torch.cat([uncond, cond])
    x0s = {}
    for i, t in enumerate(sched.timesteps):
        inp = sched.scale_model_input(torch.cat([lat, lat]), t)
        eps_u, eps_c = pipe.unet(inp, t, encoder_hidden_states=ctx).sample.chunk(2)
        eps = eps_u + guidance * (eps_c - eps_u)
        out = sched.step(eps, t, lat)
        if i in keep_x0_at:  # DDIM and Euler return their estimate of the clean latent
            x0s[i] = out.pred_original_sample
        lat = out.prev_sample
    return lat, x0s


@torch.no_grad()
def decode(pipe, lat):
    x = pipe.vae.decode(lat / VAE_SCALE).sample
    x = (x / 2 + 0.5).clamp(0, 1)
    arr = (x.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
    from PIL import Image
    return [Image.fromarray(a) for a in arr]


@torch.no_grad()
def safety(pipe, images):
    """Run the shipped safety checker. Returns (flags, margins): margin = max over the 17 concepts of
    (cosine similarity - that concept's threshold); > 0 means flagged."""
    sc = pipe.safety_checker
    inp = pipe.feature_extractor(images, return_tensors="pt").pixel_values.to("cuda", torch.float16)
    emb = sc.visual_projection(sc.vision_model(inp)[1])
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    ce = torch.nn.functional.normalize(sc.concept_embeds.float(), dim=-1)
    se = torch.nn.functional.normalize(sc.special_care_embeds.float(), dim=-1)
    sp = emb @ se.T - sc.special_care_embeds_weights.float()
    adj = (sp > 0).any(dim=1, keepdim=True).float() * 0.01
    cs = emb @ ce.T - sc.concept_embeds_weights.float() + adj
    flags = (cs > 0).any(dim=1).cpu().tolist()
    return flags, cs.max(dim=1).values.cpu().tolist()


class ClipJudge:
    """CLIP ViT-B/32 (openai/clip-vit-base-patch32, MIT). A different CLIP from SD's ViT-L/14 text encoder."""
    NAME = "openai/clip-vit-base-patch32"

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained(self.NAME, dtype=torch.float16).to("cuda").eval()
        self.p = CLIPProcessor.from_pretrained(self.NAME)

    @torch.no_grad()
    def images(self, imgs):
        x = self.p(images=imgs, return_tensors="pt").pixel_values.to("cuda", torch.float16)
        f = self.m.get_image_features(pixel_values=x)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu()

    @torch.no_grad()
    def texts(self, texts):
        t = self.p(text=texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        f = self.m.get_text_features(**t)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu()


def clip_score(img_emb, txt_emb):
    """CLIP score: 100 * max(cos, 0) per image (rows paired); Hessel et al. 2021 use 2.5 instead of 100."""
    return (100 * (img_emb * txt_emb).sum(-1).clamp(min=0)).numpy()


def pairwise_diversity(emb):
    """Mean pairwise cosine distance (1 - cos) among the rows of emb (one prompt's seeds)."""
    s = emb @ emb.T
    n = emb.shape[0]
    iu = torch.triu_indices(n, n, 1)
    return float((1 - s[iu[0], iu[1]]).mean())


class GpuMonitor:
    """Context manager: sample SM clock (MHz) and power (W) every 200 ms with nvidia-smi."""

    def __enter__(self):
        self.p = subprocess.Popen([SMI, "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                                   "--format=csv,noheader,nounits", "-lms", "200"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        time.sleep(0.3)
        return self

    def __exit__(self, *a):
        self.p.terminate()
        out, _ = self.p.communicate(timeout=5)
        rows = []
        for line in out.strip().splitlines():
            try:
                rows.append([float(v) for v in line.split(",")])
            except ValueError:
                pass
        rows = rows[2:] or rows  # skip the first samples taken before the work started
        arr = np.array(rows) if rows else np.zeros((1, 3))
        self.clock = float(np.median(arr[:, 0]))
        self.power = float(np.median(arr[:, 1]))
        self.temp = float(np.median(arr[:, 2]))
        self.n = len(rows)


def cuda_median(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def boot_ci(values, n_boot=5000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
