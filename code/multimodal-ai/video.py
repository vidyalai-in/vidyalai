"""Shared code for Beyond Text Part 12 (video generation), writer v12.

Contents
- moving_fmnist(...): synthetic clips of one Fashion-MNIST item (Xiao et al. 2017, MIT) moving in a straight
  line at constant integer velocity on a dark canvas, bouncing off the edges. Made on the fly on the GPU.
- VideoDiT: a small spacetime-patch transformer (DiT-style adaLN-Zero blocks, Peebles and Xie arXiv 2212.09748)
  over per-frame 4x4 patches, in three attention layouts:
      "frames"     spatial attention only: every frame is denoised on its own (an image model run per frame)
      "factorized" spatial attention within each frame, then temporal attention across frames at each patch
                   position, in every block (the VDM / AnimateDiff / Latte pattern)
      "full"       one attention over all frames x patches (CogVideoX / HunyuanVideo / Wan style 3D attention)
- flow_loss / euler_sample: rectified flow (x_t = (1-t) noise + t data, velocity target data - noise).
- clip_metrics(...): per-frame item location, 28x28 crops, Fashion-MNIST classifier labels and features,
  temporal consistency (label agreement across frames, feature cosine to frame 0), motion speed and
  smoothness (second difference of the centre-of-mass track).
- attention_bench(...): SDPA time and peak memory for full vs factorized attention at a given latent grid.
- GpuMonitor, cuda_median, wilson, boot_ci.
"""
from __future__ import annotations

import math
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import fmnist_eval as fe  # noqa: E402

SMI = "/usr/lib/wsl/lib/nvidia-smi"
CANVAS, ITEM = 48, 28
MAXPOS = CANVAS - ITEM  # top-left corner ranges over 0..20


# ------------------------------------------------------------------ data
def bounce(p0, v, frames, lo=0, hi=MAXPOS):
    """Positions (B, T) of a point starting at p0 with velocity v, reflected at lo and hi."""
    t = torch.arange(frames, device=p0.device)
    raw = p0[:, None] + v[:, None] * t[None]
    span = hi - lo
    r = torch.remainder(raw - lo, 2 * span)
    return lo + torch.where(r > span, 2 * span - r, r)


def render(items, py, px):
    """items (B, 28, 28) in [-1, 1]; py, px (B, T) top-left corners. Returns (B, T, 1, 48, 48)."""
    B, T = py.shape
    dev = items.device
    yy = torch.arange(CANVAS, device=dev)[None, None, :, None] - py[:, :, None, None]
    xx = torch.arange(CANVAS, device=dev)[None, None, None, :] - px[:, :, None, None]
    inside = (yy >= 0) & (yy < ITEM) & (xx >= 0) & (xx < ITEM)
    idx = (yy.clamp(0, ITEM - 1) * ITEM + xx.clamp(0, ITEM - 1))
    flat = items.reshape(B, 1, ITEM * ITEM).expand(B, T, -1)
    vals = torch.gather(flat, 2, idx.reshape(B, T, -1)).reshape(B, T, CANVAS, CANVAS)
    return torch.where(inside, vals, torch.full_like(vals, -1.0)).unsqueeze(2)


def moving_fmnist(images, n, frames=8, vmax=3, generator=None, labels=None):
    """Sample n clips. images: (N, 1, 28, 28) tensor on the target device. Returns dict with
    x (n, T, 1, 48, 48), item index, start positions and velocities (pixels per frame)."""
    dev = images.device
    g = generator
    idx = torch.randint(0, len(images), (n,), device=dev, generator=g)
    p0 = torch.randint(0, MAXPOS + 1, (n, 2), device=dev, generator=g)
    v = torch.randint(-vmax, vmax + 1, (n, 2), device=dev, generator=g)
    still = (v == 0).all(1)
    v[still, 0] = 1  # every clip moves
    py, px = bounce(p0[:, 0], v[:, 0], frames), bounce(p0[:, 1], v[:, 1], frames)
    out = {"x": render(images[idx, 0], py, px), "idx": idx, "py": py, "px": px, "v": v}
    if labels is not None:
        out["y"] = labels[idx]
    return out


# ------------------------------------------------------------------ model
def sincos_1d(dim, pos):
    om = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2))
    a = pos.reshape(-1).double()[:, None] * om[None]
    return torch.cat([a.sin(), a.cos()], 1).float()


def sincos_2d(dim, n):
    yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    return torch.cat([sincos_1d(dim // 2, yy), sincos_1d(dim // 2, xx)], 1)


def timestep_embedding(t, dim=256, scale=1000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * scale)[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)


def modulate(x, shift, scale):  # x (B, T, S, D); shift, scale (B, D)
    return x * (1 + scale[:, None, None]) + shift[:, None, None]


class Attn(nn.Module):
    """Multi-head self-attention over one of three axes of a (B, T, S, D) tensor."""

    def __init__(self, d, heads, axis):
        super().__init__()
        self.qkv, self.proj, self.h, self.axis = nn.Linear(d, 3 * d), nn.Linear(d, d), heads, axis

    def forward(self, x):
        B, T, S, D = x.shape
        hd = D // self.h
        if self.axis == "time":       # each patch position on its own: sequences of length T
            # Written out with einsum: for sequences of 8 tokens this is 2.3x faster than SDPA on a
            # transposed (B*S, heads, T, hd) copy, and it computes the same softmax attention.
            q, k, v = self.qkv(x).reshape(B, T, S, 3, self.h, hd).unbind(3)
            s = torch.einsum("btshd,bushd->bshtu", q, k) * hd ** -0.5
            p = s.float().softmax(-1).to(v.dtype)
            return self.proj(torch.einsum("bshtu,bushd->btshd", p, v).reshape(B, T, S, D))
        if self.axis == "space":      # each frame on its own: batch B*T, sequence S
            seq = x.reshape(B * T, S, D)
        else:                         # "full": one sequence of T*S tokens
            seq = x.reshape(B, T * S, D)
        b, n, _ = seq.shape
        q, k, v = self.qkv(seq).reshape(b, n, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        a = self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, D))
        return a.reshape(B, T, S, D)


class Block(nn.Module):
    """adaLN-Zero transformer block. axes: e.g. ("space",), ("space", "time") or ("full",).
    Each attention sub-layer and the MLP get their own shift, scale and zero-initialised gate."""

    def __init__(self, d, heads, axes, mlp=4):
        super().__init__()
        self.axes = axes
        self.attn = nn.ModuleList([Attn(d, heads, a) for a in axes])
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(approximate="tanh"), nn.Linear(mlp * d, d))
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 3 * d * (len(axes) + 1)))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        mods = self.ada(c).chunk(3 * (len(self.axes) + 1), -1)
        for i, layer in enumerate(list(self.attn) + [self.mlp]):
            sh, sc, g = mods[3 * i:3 * i + 3]
            x = x + g[:, None, None] * layer(modulate(self.norm(x), sh, sc))
        return x


class VideoDiT(nn.Module):
    """Spacetime-patch transformer for (B, T, 1, 48, 48) clips: 4x4 patches per frame, 12 x 12 = 144 per frame."""

    def __init__(self, mode="factorized", frames=8, size=CANVAS, patch=4, d=192, depth=6, heads=6):
        super().__init__()
        axes = {"frames": ("space",), "factorized": ("space", "time"), "full": ("full",)}[mode]
        self.mode, self.p, self.g, self.T = mode, patch, size // patch, frames
        self.embed = nn.Conv2d(1, d, patch, stride=patch)
        pos = sincos_2d(d, self.g)[None] + sincos_1d(d, torch.arange(frames))[:, None]  # (T, S, d)
        self.register_buffer("pos", pos[None], persistent=False)
        self.temb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, axes) for _ in range(depth)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.adaf = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final = nn.Linear(d, patch * patch)
        for m in (self.adaf[-1], self.final):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t):
        B, T = x.shape[:2]
        h = self.embed(x.reshape(B * T, 1, *x.shape[-2:])).flatten(2).transpose(1, 2)
        h = h.reshape(B, T, self.g * self.g, -1) + self.pos[:, :T]
        c = self.temb(timestep_embedding(t))
        for blk in self.blocks:
            h = blk(h, c)
        sh, sc = self.adaf(c).chunk(2, -1)
        h = self.final(modulate(self.nf(h), sh, sc))                      # (B, T, S, p*p)
        h = h.reshape(B, T, self.g, self.g, self.p, self.p).permute(0, 1, 2, 4, 3, 5)
        return h.reshape(B, T, 1, self.g * self.p, self.g * self.p)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ objective and sampler
def flow_loss(model, x1):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    tt = t.view(-1, *[1] * (x1.dim() - 1))
    return F.mse_loss(model((1 - tt) * x0 + tt * x1, t).float(), x1 - x0)


@torch.no_grad()
def euler_sample(model, z, steps=32):
    x, ts = z.clone(), torch.linspace(0, 1, steps + 1, device=z.device)
    for k in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, ts[k].expand(len(x))).float()
        x = x + (ts[k + 1] - ts[k]) * v
    return x.clamp(-1, 1)


# ------------------------------------------------------------------ metrics
@torch.no_grad()
def clip_metrics(x, thr=0.25):
    """x: (B, T, 1, 48, 48) in [-1, 1]. Returns per-frame centre of mass, 28x28 crops centred on the
    item's bounding box, classifier labels/features and per-clip consistency and motion numbers."""
    B, T = x.shape[:2]
    inten = ((x[:, :, 0].float() + 1) / 2).clamp(0, 1)                    # (B, T, 48, 48) in [0, 1]
    ar = torch.arange(CANVAS, device=x.device, dtype=torch.float32)
    mass = inten.sum((-1, -2)).clamp(min=1e-6)
    cy = (inten.sum(-1) * ar).sum(-1) / mass
    cx = (inten.sum(-2) * ar).sum(-1) / mass
    on = inten > thr
    rows, cols = on.any(-1), on.any(-2)                                   # (B, T, 48)

    def box_center(m):
        big = torch.where(m, ar, torch.full_like(ar, 1e9)).amin(-1)
        small = torch.where(m, ar, torch.full_like(ar, -1e9)).amax(-1)
        ok = m.any(-1)
        return torch.where(ok, (big + small) / 2, torch.full_like(big, CANVAS / 2))

    by, bx = box_center(rows), box_center(cols)
    ty = (by - (ITEM - 1) / 2).round().clamp(0, MAXPOS).long()
    tx = (bx - (ITEM - 1) / 2).round().clamp(0, MAXPOS).long()
    yy = ty[..., None] + torch.arange(ITEM, device=x.device)            # (B, T, 28)
    xx = tx[..., None] + torch.arange(ITEM, device=x.device)
    flat = x[:, :, 0].reshape(B, T, -1)
    gi = (yy[..., :, None] * CANVAS + xx[..., None, :]).reshape(B, T, -1)
    crops = torch.gather(flat, 2, gi).reshape(B * T, 1, ITEM, ITEM)
    feats, probs = fe.features(crops)
    feats = torch.from_numpy(feats).float().reshape(B, T, -1)
    labels = torch.from_numpy(probs.argmax(1)).reshape(B, T)
    fn = F.normalize(feats, dim=-1)
    cos0 = (fn[:, 1:] * fn[:, :1]).sum(-1).mean(1)                        # feature cosine to frame 0
    cosn = (fn[:, 1:] * fn[:, :-1]).sum(-1).mean(1)                       # to the previous frame
    mode_lab = torch.mode(labels, dim=1).values
    agree = (labels == mode_lab[:, None]).float().mean(1)                 # share of frames with the clip's majority label
    same = (labels == labels[:, :1]).all(1).float()                       # every frame has frame 0's label
    track = torch.stack([cy, cx], -1).cpu()                               # (B, T, 2)
    vel = track[:, 1:] - track[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return {"crops": crops.cpu(), "labels": labels, "same": same, "agree": agree, "cos0": cos0, "cosn": cosn,
            "speed": vel.norm(dim=-1).mean(1), "accel": acc.norm(dim=-1).mean(1), "track": track,
            "mass": mass.cpu()}


# ------------------------------------------------------------------ attention benchmark
def attention_bench(T, H, W, heads=8, hd=64, mode="full", n=7, warmup=3, dtype=torch.bfloat16, backend=None):
    """Time one self-attention call (SDPA, bf16) over a latent grid of T x H x W tokens, either full
    (one sequence of T*H*W) or factorized (spatial attention over H*W per frame, then temporal over T
    per position). Returns median seconds, peak extra memory in bytes and the attention FLOPs counted
    as 4 * L^2 * hd per head per sequence (QK^T and AV, two FLOPs per multiply-add)."""
    S = H * W
    dev = "cuda"
    torch.cuda.empty_cache()
    if mode == "full":
        q = torch.randn(1, heads, T * S, hd, device=dev, dtype=dtype)
        k, v = torch.randn_like(q), torch.randn_like(q)
        fn = lambda: F.scaled_dot_product_attention(q, k, v)
        flops = 4 * (T * S) ** 2 * hd * heads
    else:
        qs = torch.randn(T, heads, S, hd, device=dev, dtype=dtype)
        ks, vs = torch.randn_like(qs), torch.randn_like(qs)
        qt = torch.randn(S, heads, T, hd, device=dev, dtype=dtype)
        kt, vt = torch.randn_like(qt), torch.randn_like(qt)

        def fn():
            F.scaled_dot_product_attention(qs, ks, vs)
            return F.scaled_dot_product_attention(qt, kt, vt)
        flops = 4 * T * S ** 2 * hd * heads + 4 * S * T ** 2 * hd * heads
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if backend is None:
        med, lo, hi = cuda_median(fn, n=n, warmup=warmup)
    else:  # e.g. SDPBackend.MATH: the textbook version that stores the full score matrix
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(backend):
            med, lo, hi = cuda_median(fn, n=n, warmup=warmup)
    peak = torch.cuda.max_memory_allocated() - base
    return {"T": T, "H": H, "W": W, "tokens": T * S, "mode": mode, "median_s": med, "min_s": lo, "max_s": hi,
            "peak_extra_bytes": int(peak), "attn_flops": float(flops), "tflops": flops / med / 1e12, "n": n}


# ------------------------------------------------------------------ timing and stats helpers
class GpuMonitor:
    """Context manager: sample SM clock (MHz), power (W) and memory used (MiB) every 200 ms with nvidia-smi."""

    def __enter__(self):
        self.p = subprocess.Popen([SMI, "--query-gpu=clocks.sm,power.draw,memory.used",
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
        rows = rows[2:] or rows
        arr = np.array(rows) if rows else np.zeros((1, 3))
        self.clock, self.power = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        self.mem_max = float(arr[:, 2].max())
        self.n = len(rows)


def cuda_median(fn, n=7, warmup=3):
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
