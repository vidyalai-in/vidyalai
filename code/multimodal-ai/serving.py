"""Shared helpers for Beyond Text Part 13 (serving multimodal models).

1. A synthetic report page (`report_page`) drawn in relative coordinates at any size: a title, a bar
   chart of monthly values and a small table, all from a seed, plus `page_questions`, short questions
   whose answers are read off the page (exact match). No people, logos or brands.
2. Claude helpers: `image_block` (base64 PNG, optional cache_control) and `stream_call`, which times
   time to first text token and the full reply through the ledger client (guard before, record after).
3. Statistics: `median_ci` and `quantile_ci` (percentile bootstrap), `wilson` re-exported from vlm.py.
4. Local timing: `cuda_times` (CUDA-synchronized wall times after warm-up) and `GpuSampler` from vlm.py.
"""
from __future__ import annotations

import base64
import io
import math
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
from vlm import GpuSampler, wilson  # noqa: E402,F401

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun")
ROWS = ("North depot", "South depot", "East depot", "West depot", "River depot")
COLS = ("Pallets in", "Pallets out", "Late trucks")
_FD = None


def _font(px, bold=False):
    global _FD
    if _FD is None:
        import matplotlib
        _FD = matplotlib.get_data_path() + "/fonts/ttf/"
    return ImageFont.truetype(_FD + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), max(6, int(px)))


def page_facts(seed: int) -> dict:
    """The numbers on page `seed`: six monthly bar values and a 5 x 3 table."""
    rng = random.Random(1_000 + seed)
    bars = [rng.randrange(20, 96) for _ in MONTHS]
    table = {r: [rng.randrange(100, 990), rng.randrange(100, 990), rng.randrange(0, 40)] for r in ROWS}
    return {"page": seed + 1, "bars": bars, "table": table}


def report_page(seed: int, w: int = 1200, h: int = 800, tag: str | None = None) -> Image.Image:
    """Render page `seed` at w x h. `tag` prints a small reference string in the footer, which makes
    an otherwise identical page a different image (used to force cache misses)."""
    f = page_facts(seed)
    s = 2                                    # supersample for smooth text, then downsize
    W, H = w * s, h * s
    u = W / 1200                             # layout unit: 1 at 1200 px wide
    img = Image.new("RGB", (W, H), (252, 252, 249))
    d = ImageDraw.Draw(img)
    d.text((40 * u, 28 * u), f"Depot operations report, page {f['page']}", font=_font(34 * u, True), fill=(20, 30, 50))
    d.text((40 * u, 76 * u), "Synthetic data drawn for a serving experiment", font=_font(18 * u), fill=(90, 100, 120))
    # bar chart, left half
    x0, x1, y0, y1 = 90 * u, 560 * u, 150 * H / 800, 640 * H / 800
    d.text((x0, y0 - 34 * u), "Trucks unloaded per day", font=_font(20 * u, True), fill=(30, 40, 60))
    for v in range(0, 101, 20):
        yy = y1 - (y1 - y0) * v / 100
        d.line([x0, yy, x1, yy], fill=(220, 224, 232), width=max(1, int(2 * u)))
        d.text((x0 - 10 * u, yy), str(v), font=_font(16 * u), fill=(70, 80, 100), anchor="rm")
    slot = (x1 - x0) / len(MONTHS)
    for i, (m, v) in enumerate(zip(MONTHS, f["bars"])):
        cx = x0 + slot * (i + 0.5)
        bw = slot * 0.58
        top = y1 - (y1 - y0) * v / 100
        d.rectangle([cx - bw / 2, top, cx + bw / 2, y1], fill=(50, 105, 190))
        d.text((cx, top - 6 * u), str(v), font=_font(16 * u, True), fill=(20, 30, 50), anchor="mb")
        d.text((cx, y1 + 10 * u), m, font=_font(17 * u), fill=(30, 40, 60), anchor="mt")
    # table, right half
    tx, ty = 620 * u, 150 * H / 800
    colw = (180 * u, 125 * u, 125 * u, 110 * u)
    rh = 52 * H / 800
    d.text((tx, ty - 34 * u), "Weekly depot totals", font=_font(20 * u, True), fill=(30, 40, 60))
    heads = ("Depot",) + COLS
    xx = tx
    for c, cw in zip(heads, colw):
        d.text((xx + 6 * u, ty + rh / 2), c, font=_font(16 * u, True), fill=(20, 30, 50), anchor="lm")
        xx += cw
    for r_i, r in enumerate(ROWS):
        yy = ty + rh * (r_i + 1)
        if r_i % 2 == 0:
            d.rectangle([tx, yy, tx + sum(colw), yy + rh], fill=(236, 240, 247))
        xx = tx
        for c_i, (val, cw) in enumerate(zip((r,) + tuple(f["table"][r]), colw)):
            d.text((xx + 6 * u, yy + rh / 2), str(val), font=_font(17 * u, c_i == 0), fill=(25, 30, 45), anchor="lm")
            xx += cw
    d.line([tx, ty + rh, tx + sum(colw), ty + rh], fill=(60, 70, 90), width=max(1, int(2 * u)))
    foot = f"Page {f['page']} of 8" + (f"   ref {tag}" if tag else "")
    d.text((40 * u, H - 40 * H / 800), foot, font=_font(15 * u), fill=(110, 115, 130))
    return img.resize((w, h), Image.LANCZOS)


def page_questions(seeds, n: int, rng_seed: int = 0, extra: bool = False):
    """n distinct (question, answer) pairs about the pages in `seeds`; answers are integers.
    Lookups only (21 per page), or with extra=True also differences and column totals (44 per page)."""
    rng = random.Random(rng_seed)
    pool = []
    for sd in seeds:
        f = page_facts(sd)
        p = f["page"]
        for m, v in zip(MONTHS, f["bars"]):
            pool.append((f"On page {p}, how many trucks were unloaded per day in {m}?", str(v)))
        for r in ROWS:
            for c, v in zip(COLS, f["table"][r]):
                pool.append((f"On page {p}, what is the '{c}' value for {r}?", str(v)))
        if extra:   # questions that need a little arithmetic on the page
            for r in ROWS:
                a, b, _ = f["table"][r]
                pool.append((f"On page {p}, how many more pallets went out than came in at {r}? "
                             "Give a negative number if fewer went out.", str(b - a)))
            for c_i, c in enumerate(COLS):
                pool.append((f"On page {p}, what is the total of the '{c}' column?",
                             str(sum(f["table"][r][c_i] for r in ROWS))))
            for i in range(len(MONTHS)):
                for j in range(i + 1, len(MONTHS)):
                    pool.append((f"On page {p}, how many more trucks per day were unloaded in {MONTHS[j]} than in "
                                 f"{MONTHS[i]}? Give a negative number if fewer.", str(f["bars"][j] - f["bars"][i])))
    rng.shuffle(pool)
    if n > len(pool):
        raise ValueError(f"only {len(pool)} questions available")
    return pool[:n]


def png_bytes(img: Image.Image, fmt="PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def image_block(img: Image.Image, cache: bool = False) -> dict:
    b = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.standard_b64encode(png_bytes(img, optimize=True)).decode()}}
    if cache:
        b["cache_control"] = {"type": "ephemeral"}
    return b


def stream_call(cl, ledger, part, model, content, system, max_tokens, note, est):
    """One streamed Messages call through the ledger client, thinking disabled.
    Returns ttft (first text delta), total time, usage dict, text and ledger cost."""
    ledger.guard(part, est)
    t0 = time.perf_counter()
    t_first, text = None, ""
    with cl.messages.stream(model=model, max_tokens=max_tokens, system=system,
                            thinking={"type": "disabled"},
                            messages=[{"role": "user", "content": content}]) as s:
        for dlt in s.text_stream:
            if t_first is None and dlt:
                t_first = time.perf_counter() - t0
            text += dlt
        msg = s.get_final_message()
    total = time.perf_counter() - t0
    cost = ledger.record(part, model, msg.usage, note=note)
    return {"ttft": t_first, "total": total, "usage": msg.usage.model_dump(), "text": text,
            "stop": msg.stop_reason, "cost": cost}


# ---- statistics -----------------------------------------------------------------------------------

def quantile_ci(x, q=0.5, n_boot=10000, seed=0):
    """q-quantile of x with a percentile-bootstrap 95% interval."""
    x = np.asarray([v for v in x if v is not None], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    boots = np.quantile(x[idx], q, axis=1)
    return float(np.quantile(x, q)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def median_ci(x, **kw):
    return quantile_ci(x, 0.5, **kw)


def diff_median_ci(a, b, n_boot=10000, seed=0):
    """median(b) - median(a) for two independent samples, bootstrap 95% interval."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    d = np.median(b[ib], axis=1) - np.median(a[ia], axis=1)
    return float(np.median(b) - np.median(a)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---- local GPU timing -----------------------------------------------------------------------------

def cuda_times(fn, warmup=3, reps=7):
    """Wall times in ms of fn(), torch.cuda.synchronize() before and after each run."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def med(ts):
    return float(np.median(ts))
