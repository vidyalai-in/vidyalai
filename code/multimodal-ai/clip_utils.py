"""Shared CLIP / contrastive helpers for the Beyond Text series (Part 3, v03).

Contents
- load_encoder(name): CLIP or SigLIP from Hugging Face in fp16 on the GPU.
- encode_images / encode_texts: L2-normalised embeddings (float32, CPU).
- zeroshot_classifier: one text embedding per class, averaged over prompt templates
  in embedding space (as in the CLIP paper, Sec. 3.1.4).
- clip_loss / siglip_loss: the two contrastive objectives on an N x N logit matrix.
- cifar10(split): CIFAR-10 from the uoft-cs/cifar10 parquet files (measurement only).
- render_shapes(...): synthetic coloured-shape scenes with captions (own data).
- wilson / paired_diff_ci: confidence intervals for rates and paired differences.
"""
from __future__ import annotations

import io
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

HF = os.environ.setdefault("HF_HOME", "/home/amit/.cache/vidyalai-scratch/hf")

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
# The 18 CIFAR-10 templates from github.com/openai/CLIP/blob/main/data/prompts.md
CIFAR10_TEMPLATES = [
    "a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
    "a low contrast photo of a {}.", "a high contrast photo of a {}.", "a bad photo of a {}.",
    "a good photo of a {}.", "a photo of a small {}.", "a photo of a big {}.",
    "a photo of the {}.", "a blurry photo of the {}.", "a black and white photo of the {}.",
    "a low contrast photo of the {}.", "a high contrast photo of the {}.", "a bad photo of the {}.",
    "a good photo of the {}.", "a photo of the small {}.", "a photo of the big {}.",
]


# ---------------------------------------------------------------- pretrained encoders
def load_encoder(name: str, device: str = "cuda"):
    """Return (model, processor, kind) with weights in fp16 on `device`."""
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    proc = AutoProcessor.from_pretrained(name)
    kind = "siglip" if "siglip" in name.lower() else "clip"
    return model, proc, kind


def _feat(out):
    # transformers 5 may return a ModelOutput; take the pooled projection.
    if isinstance(out, torch.Tensor):
        return out
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, k, None) is not None:
            return getattr(out, k)
    raise TypeError(type(out))


@torch.no_grad()
def encode_images(model, proc, images, bs: int = 256, device: str = "cuda"):
    feats = []
    for i in range(0, len(images), bs):
        px = proc(images=images[i:i + bs], return_tensors="pt")["pixel_values"]
        f = _feat(model.get_image_features(pixel_values=px.to(device, torch.float16)))
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats)


@torch.no_grad()
def encode_texts(model, proc, texts, kind: str, bs: int = 512, device: str = "cuda"):
    feats = []
    for i in range(0, len(texts), bs):
        if kind == "siglip":  # SigLIP was trained with texts padded to 64 tokens
            tok = proc(text=texts[i:i + bs], padding="max_length", max_length=64,
                       truncation=True, return_tensors="pt")
        else:
            tok = proc(text=texts[i:i + bs], padding=True, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
        f = _feat(model.get_text_features(**tok))
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats)


def zeroshot_classifier(model, proc, kind, classes, templates):
    """[n_classes, d] weights: per class, mean of normalised template embeddings, renormalised."""
    rows = []
    for c in classes:
        e = encode_texts(model, proc, [t.format(c) for t in templates], kind)
        rows.append(F.normalize(e.mean(0), dim=-1))
    return torch.stack(rows)


# ---------------------------------------------------------------- losses
def clip_loss(img, txt, scale):
    """Symmetric cross-entropy (InfoNCE) over the N x N similarity matrix.
    img, txt: [N, d] L2-normalised; scale = 1 / temperature."""
    logits = scale * img @ txt.t()
    labels = torch.arange(len(img), device=img.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def siglip_loss(img, txt, scale, bias):
    """Sigmoid loss: every (i, j) pair is an independent yes/no question (SigLIP Alg. 1)."""
    logits = scale * img @ txt.t() + bias
    n = len(img)
    sign = 2 * torch.eye(n, device=img.device) - 1  # +1 on the diagonal, -1 elsewhere
    return -F.logsigmoid(sign * logits).sum() / n


# ---------------------------------------------------------------- data
def cifar10(split: str = "test"):
    """(list of PIL images, np.array labels). CIFAR-10 has no stated licence: measure only."""
    import glob
    import pandas as pd
    from PIL import Image
    path = glob.glob(f"{HF}/hub/datasets--uoft-cs--cifar10/snapshots/*/plain_text/{split}-*.parquet")[0]
    df = pd.read_parquet(path)
    imgs = [Image.open(io.BytesIO(b["bytes"])).convert("RGB") for b in df["img"]]
    return imgs, df["label"].to_numpy()


SHAPES = ["circle", "square", "triangle", "diamond", "cross", "ring"]
COLOURS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220),
           "yellow": (235, 210, 40), "purple": (150, 60, 190), "white": (245, 245, 245)}
SIZES = ["small", "large"]
POSITIONS = ["top left", "top right", "bottom left", "bottom right"]


def caption(s, c, z, p):
    return f"a {SIZES[z]} {list(COLOURS)[c]} {SHAPES[s]} at the {POSITIONS[p]}"


def all_captions():
    return [(s, c, z, p) for s in range(len(SHAPES)) for c in range(len(COLOURS))
            for z in range(len(SIZES)) for p in range(len(POSITIONS))]


def _mask(shape, yy, xx, cy, cx, r):
    dy, dx = yy - cy, xx - cx
    if shape == "circle":
        return dy ** 2 + dx ** 2 <= r ** 2
    if shape == "ring":
        d = dy ** 2 + dx ** 2
        return (d <= r ** 2) & (d >= (0.55 * r) ** 2)
    if shape == "square":
        return (abs(dy) <= 0.85 * r) & (abs(dx) <= 0.85 * r)
    if shape == "diamond":
        return abs(dy) + abs(dx) <= r
    if shape == "cross":
        w = 0.35 * r
        return ((abs(dy) <= w) & (abs(dx) <= r)) | ((abs(dx) <= w) & (abs(dy) <= r))
    if shape == "triangle":  # apex up
        return (dy <= 0.8 * r) & (dy >= -r) & (abs(dx) <= (dy + r) * 0.6)
    raise ValueError(shape)


def render_shapes(attrs, size: int = 32, seed: int = 0):
    """attrs: int array [n, 4] of (shape, colour, size, position). Returns uint8 [n, size, size, 3].
    Dark grey background with noise; position jittered inside its quadrant."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cols = np.array(list(COLOURS.values()), dtype=np.float32)
    out = np.empty((len(attrs), size, size, 3), dtype=np.uint8)
    q = size / 4
    for i, (s, c, z, p) in enumerate(attrs):
        r = (0.17 if z == 0 else 0.29) * size * rng.uniform(0.9, 1.1)
        cy = (q if p < 2 else 3 * q) + rng.uniform(-1.5, 1.5) * size / 32
        cx = (q if p % 2 == 0 else 3 * q) + rng.uniform(-1.5, 1.5) * size / 32
        bg = rng.uniform(30, 80)
        img = np.full((size, size, 3), bg, np.float32) + rng.normal(0, 8, (size, size, 3))
        m = _mask(SHAPES[s], yy, xx, cy, cx, r)
        img[m] = cols[c] * rng.uniform(0.85, 1.0)
        out[i] = np.clip(img, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for k successes out of n, as (rate, lo, hi)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def paired_diff_ci(a, b, z: float = 1.96):
    """Mean of (a - b) for per-item 0/1 correctness arrays, with a normal-approx 95% CI,
    plus the discordant counts (a right & b wrong, a wrong & b right)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    return {"diff": m, "lo": m - z * se, "hi": m + z * se,
            "a_only": int(((a == 1) & (b == 0)).sum()), "b_only": int(((a == 0) & (b == 1)).sum())}
