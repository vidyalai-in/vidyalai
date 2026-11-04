"""Token ledger helpers (Beyond Text, Part 1): how many tokens or latent values each modality becomes.

Synthetic inputs (no real photos, voices or faces), Claude's documented image-token rule, a GPU timer
that samples nvidia-smi while it runs, and a row format for the ledger table.
"""
import math
import statistics
import subprocess
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

NVSMI = "/usr/lib/wsl/lib/nvidia-smi"


# ---------------------------------------------------------------- synthetic inputs
def scene(w: int, h: int, t: float = 0.0, dots: int = 12) -> Image.Image:
    """A self-made picture drawn in relative coordinates, so every size shows the same scene.
    Sky gradient, sun, hills, a house, a tree and a row of small dots. `t` in [0, 1] slides the
    sun and a ball for video frames. `dots` sets how many small red dots sit in the bottom row
    (default 12, the article's test picture)."""
    s = 4  # supersample, then downsize for smooth edges
    W, H = w * s, h * s
    img = Image.new("RGB", (W, H))
    top, bot = np.array([90, 150, 230]), np.array([200, 225, 250])
    ramp = np.linspace(0, 1, H)[:, None]
    sky = (top * (1 - ramp) + bot * ramp).astype(np.uint8)
    img.paste(Image.fromarray(np.repeat(sky[:, None, :], W, axis=1).reshape(H, W, 3)))
    d = ImageDraw.Draw(img)
    X = lambda f: f * W
    Y = lambda f: f * H
    sx = 0.15 + 0.6 * t
    d.ellipse([X(sx), Y(0.08), X(sx + 0.12), Y(0.08) + X(0.12)], fill=(250, 200, 60))
    d.polygon([(X(0), Y(0.75)), (X(0.3), Y(0.5)), (X(0.6), Y(0.75))], fill=(70, 140, 80))
    d.polygon([(X(0.4), Y(0.75)), (X(0.75), Y(0.45)), (X(1.0), Y(0.75))], fill=(50, 120, 70))
    d.rectangle([X(0), Y(0.75), X(1), Y(1)], fill=(110, 170, 90))
    d.rectangle([X(0.55), Y(0.58), X(0.78), Y(0.82)], fill=(220, 120, 90))
    d.polygon([(X(0.53), Y(0.58)), (X(0.665), Y(0.46)), (X(0.80), Y(0.58))], fill=(150, 60, 50))
    d.rectangle([X(0.64), Y(0.70), X(0.69), Y(0.82)], fill=(90, 60, 40))
    d.rectangle([X(0.57), Y(0.62), X(0.62), Y(0.67)], fill=(250, 240, 180))
    d.rectangle([X(0.71), Y(0.62), X(0.76), Y(0.67)], fill=(250, 240, 180))
    d.rectangle([X(0.20), Y(0.60), X(0.23), Y(0.82)], fill=(100, 70, 40))
    d.ellipse([X(0.13), Y(0.42), X(0.30), Y(0.66)], fill=(40, 110, 50))
    for i in range(dots):  # small objects: the kind of detail patches blur
        cx = 0.05 + i * (0.88 / max(dots - 1, 1)) if dots != 12 else 0.05 + i * 0.08
        d.ellipse([X(cx), Y(0.90), X(cx + 0.02), Y(0.90) + X(0.02)], fill=(200, 40, 40))
    bx = 0.1 + 0.8 * t
    d.ellipse([X(bx), Y(0.80), X(bx + 0.05), Y(0.80) + X(0.05)], fill=(240, 240, 240), outline=(30, 30, 30), width=max(1, s))
    return img.resize((w, h), Image.LANCZOS)


def video(seconds: float, fps: int, w: int, h: int):
    n = int(round(seconds * fps))
    return [scene(w, h, i / max(1, n - 1)) for i in range(n)]


def tones(seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    """A chord, a rising chirp and a little noise, float32 in [-1, 1]. No voice."""
    t = np.arange(int(seconds * sr)) / sr
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 277.2 * t) + 0.12 * np.sin(2 * np.pi * 329.6 * t)
    f = 200 + 1800 * (t / max(seconds, 1e-9))
    x += 0.15 * np.sin(2 * np.pi * np.cumsum(f) / sr)
    x += 0.01 * np.random.default_rng(seed).standard_normal(len(t))
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


# ---------------------------------------------------------------- Claude's image-token rule
def claude_image_tokens(w: int, h: int) -> int:
    """One visual token per 28x28-pixel patch (platform.claude.com/docs/en/build-with-claude/vision)."""
    return math.ceil(w / 28) * math.ceil(h / 28)


def claude_resized_size(w: int, h: int, max_edge: int = 2576, max_tokens: int = 4784):
    """The documented reference implementation (vision-coordinates page), high-resolution tier defaults."""
    def fits(a, b):
        return math.ceil(a / 28) * 28 <= max_edge and math.ceil(b / 28) * 28 <= max_edge and claude_image_tokens(a, b) <= max_tokens
    if fits(w, h):
        return (w, h)
    if h > w:
        rh, rw = claude_resized_size(h, w, max_edge, max_tokens)
        return (rw, rh)
    ar = w / h
    lo, hi = 1, w
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / ar), 1)):
            lo = mid
        else:
            hi = mid
    return (lo, max(round(lo / ar), 1))


def claude_expected(w: int, h: int, **kw) -> int:
    return claude_image_tokens(*claude_resized_size(w, h, **kw))


# ---------------------------------------------------------------- GPU timing with clock and power
class SmiSampler:
    """Samples SM clock (MHz) and power (W) every 0.2 s in a thread while active."""
    def __init__(self, every=0.2):
        self.every, self.rows, self._stop = every, [], threading.Event()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run([NVSMI, "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                c, p, tmp = [float(v) for v in out.split(",")]
                self.rows.append((c, p, tmp))
            except Exception:
                pass
            self._stop.wait(self.every)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def summary(self):
        if not self.rows:
            return {}
        c, p, t = zip(*self.rows)
        return {"sm_clock_mhz_median": statistics.median(c), "power_w_median": statistics.median(p),
                "temp_c_max": max(t), "n_samples": len(c)}


def time_gpu(fn, warmup: int = 3, runs: int = 10, min_seconds: float = 1.0):
    """Median wall time of fn() in ms, synchronizing CUDA around each run. Repeats the timed runs
    until at least min_seconds have elapsed so the nvidia-smi sampler sees the GPU busy."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    with SmiSampler() as smi:
        t_start = time.perf_counter()
        while len(times) < runs or time.perf_counter() - t_start < min_seconds:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            if len(times) >= 200:
                break
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times), "runs": len(times), **smi.summary()}


# ---------------------------------------------------------------- ledger rows
def row(modality, representation, model, kind, positions, dim, raw_values, raw_bits, per, code_bits=None, **extra):
    """One ledger entry. positions x dim = values. Discrete rows give code_bits per code."""
    values = positions * dim
    r = {"modality": modality, "representation": representation, "model": model, "kind": kind, "per": per,
         "positions": positions, "dim": dim, "values": values, "raw_values": raw_values,
         "value_ratio": raw_values / values, **extra}
    if code_bits is not None:
        r["bits"] = values * code_bits
        r["bit_ratio"] = raw_bits / r["bits"]
    return r
