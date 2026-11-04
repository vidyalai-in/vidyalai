"""Shared helpers for Beyond Text Part 4 (vision-language models).

1. A synthetic, checkable image task set (`make_items`, `render`, `score`): four tasks whose
   answers are known exactly, rendered at any size from one 896 x 896 master drawing.
     count   "How many red shapes are there?"           integer, exact match
     read    a 5-character code printed in a label      string, exact match
     chart   the value of one bar in a bar chart        integer, correct if within 5
     absent  "Is there a blue triangle?" (POPE-style)   yes/no; half the items are "no"
     dense   "How many red dots?" with 12-40 small dots  integer, exact match (also scored within 10%)
2. Small-sample statistics (`wilson`, `paired_diff`).
3. Claude's documented image-token rule (`claude_tokens`, `claude_resized`).
4. Local VLM wrappers for SmolVLM and Qwen2-VL (`load_vlm`, `VLM.ask`, `VLM.time_split`) and an
   nvidia-smi sampler (`GpuSampler`).

Everything here is deterministic given the item seed, so later parts can reuse the same items.
"""
from __future__ import annotations

import base64
import io
import math
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

MASTER = 896                      # every image is drawn at 896 x 896, then resized
SIZES = (224, 448, 896)           # 8, 16 and 32 patches of 28 px per side
TASKS = ("count", "read", "chart", "absent", "dense")
COLORS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (30, 160, 70), "orange": (240, 150, 20)}
SHAPES = ("circle", "square", "triangle")
CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O or 1/I
_FONT_DIR = None


def _font(size, bold=False):
    global _FONT_DIR
    if _FONT_DIR is None:
        import matplotlib
        _FONT_DIR = matplotlib.get_data_path() + "/fonts/ttf/"
    return ImageFont.truetype(_FONT_DIR + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


# ---- Items -----------------------------------------------------------------------------------------

@dataclass
class Item:
    id: str
    task: str
    seed: int
    question: str
    answer: str
    meta: dict = field(default_factory=dict)


SUFFIX = {"count": "Answer with a single number only.",
          "dense": "Answer with a single number only.",
          "read": "Answer with the code only.",
          "chart": "Answer with a single number only.",
          "absent": "Answer with yes or no only."}


def make_items(n_per_task=150, seed=0):
    """n_per_task items for each of the four tasks; item i of a task always gets the same seed."""
    items = []
    for t_i, task in enumerate(TASKS):
        for i in range(n_per_task):
            s = seed * 1_000_003 + t_i * 10_007 + i
            rng = random.Random(s)
            items.append(_MAKERS[task](f"{task}-{i:03d}", s, rng))
    return items


def _place_shapes(rng, specs, r_lo=26, r_hi=38, margin=20):
    """Non-overlapping centres for a list of (color, shape); returns list of dicts."""
    out = []
    for color, shape in specs:
        for _ in range(2000):
            r = rng.randint(r_lo, r_hi)
            x = rng.randint(margin + r, MASTER - margin - r)
            y = rng.randint(margin + r, MASTER - margin - r)
            if all((x - o["x"]) ** 2 + (y - o["y"]) ** 2 > (r + o["r"] + 14) ** 2 for o in out):
                out.append(dict(color=color, shape=shape, x=x, y=y, r=r))
                break
        else:
            raise RuntimeError("could not place shapes")
    return out


def _make_count(iid, s, rng):
    target = rng.choice(list(COLORS))
    k = rng.randint(1, 12)
    others = [c for c in COLORS if c != target]
    n_other = rng.randint(4, 12)
    specs = [(target, rng.choice(SHAPES)) for _ in range(k)] + \
            [(rng.choice(others), rng.choice(SHAPES)) for _ in range(n_other)]
    rng.shuffle(specs)
    shapes = _place_shapes(rng, specs)
    return Item(iid, "count", s, f"How many {target} shapes are there in the image?", str(k),
                dict(shapes=shapes, target=target, n_total=len(specs)))


def _make_absent(iid, s, rng):
    present = int(iid[-3:]) % 2 == 0          # exactly half "yes"
    n = rng.randint(4, 8)
    specs = [(rng.choice(list(COLORS)), rng.choice(SHAPES)) for _ in range(n)]
    combos = set(specs)
    if present:
        q = rng.choice(specs)
    else:
        # Adversarial absent object: its colour and its shape both appear, but never together.
        cands = [(c, sh) for c in COLORS for sh in SHAPES if (c, sh) not in combos
                 and any(c == a for a, _ in specs) and any(sh == b for _, b in specs)]
        if not cands:
            cands = [(c, sh) for c in COLORS for sh in SHAPES if (c, sh) not in combos]
        q = rng.choice(cands)
    shapes = _place_shapes(rng, specs, r_lo=30, r_hi=44)
    return Item(iid, "absent", s, f"Is there {'an' if q[0][0] in 'aeiou' else 'a'} {q[0]} {q[1]} in the image?", "yes" if present else "no",
                dict(shapes=shapes, query=list(q)))


def _make_read(iid, s, rng):
    code = "".join(rng.choice(CODE_CHARS) for _ in range(5))
    # A small label somewhere on a page with grey "lines of text" as clutter.
    # Font size cycles over five values so every size condition has the same mix; the text's height
    # in the final image is font_px * size / 896 (e.g. 12 px type is 3 px tall at 224).
    font_px = (12, 16, 20, 26, 34)[int(iid[-3:]) % 5]
    x = rng.randint(40, MASTER - 300)
    y = rng.randint(60, MASTER - 120)
    return Item(iid, "read", s, "What is the 5-character code printed in the box?", code,
                dict(x=x, y=y, font_px=font_px))


def _make_chart(iid, s, rng):
    n = rng.randint(4, 6)
    labels = "ABCDEF"[:n]
    vals = [rng.randrange(5, 100, 5) for _ in range(n)]
    q = rng.randrange(n)
    return Item(iid, "chart", s, f"In the bar chart, what is the value of bar {labels[q]}?", str(vals[q]),
                dict(labels=labels, values=vals, query=labels[q]))


def _make_dense(iid, s, rng):
    target = rng.choice(list(COLORS))
    k = rng.randint(12, 40)
    others = [c for c in COLORS if c != target]
    specs = [(target, "circle") for _ in range(k)] + [(rng.choice(others), "circle") for _ in range(rng.randint(10, 20))]
    rng.shuffle(specs)
    shapes = _place_shapes(rng, specs, r_lo=9, r_hi=14, margin=16)
    return Item(iid, "dense", s, f"How many {target} dots are there in the image?", str(k),
                dict(shapes=shapes, target=target, n_total=len(specs)))


_MAKERS = {"dense": _make_dense, "count": _make_count, "absent": _make_absent, "read": _make_read, "chart": _make_chart}


# ---- Rendering -------------------------------------------------------------------------------------

def _draw_shape(d, sh):
    x, y, r, col = sh["x"], sh["y"], sh["r"], COLORS[sh["color"]]
    if sh["shape"] == "circle":
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    elif sh["shape"] == "square":
        a = r * 0.88
        d.rectangle([x - a, y - a, x + a, y + a], fill=col)
    else:
        d.polygon([(x, y - r), (x - r * 0.95, y + r * 0.75), (x + r * 0.95, y + r * 0.75)], fill=col)


def render_master(item: Item) -> Image.Image:
    img = Image.new("RGB", (MASTER, MASTER), (250, 250, 247))
    d = ImageDraw.Draw(img)
    m = item.meta
    if item.task in ("count", "absent", "dense"):
        for sh in m["shapes"]:
            _draw_shape(d, sh)
    elif item.task == "read":
        rng = random.Random(item.seed + 7)
        for row in range(24):                      # clutter: grey bars like lines of text
            yy = 30 + row * 36
            xx = 30
            while xx < MASTER - 60:
                w = rng.randint(30, 140)
                d.rectangle([xx, yy, min(xx + w, MASTER - 30), yy + 10], fill=(205, 205, 205))
                xx += w + rng.randint(12, 30)
        f = _font(m["font_px"], bold=True)
        tw = d.textlength(item.answer, font=f)
        x, y = m["x"], m["y"]
        pad = max(6, m["font_px"] // 2)
        d.rectangle([x - pad, y - pad, x + tw + pad, y + m["font_px"] * 1.15 + pad], fill="white", outline=(40, 40, 40), width=2)
        d.text((x, y), item.answer, font=f, fill=(20, 20, 20))
    elif item.task == "chart":
        x0, y0, x1, y1 = 120, 80, 840, 780          # plot area; y0 is the 100 line, y1 the 0 line
        ft = _font(26)
        for v in range(0, 101, 20):
            yy = y1 - (y1 - y0) * v / 100
            d.line([x0, yy, x1, yy], fill=(215, 215, 215), width=2)
            d.text((x0 - 16, yy), str(v), font=ft, fill=(60, 60, 60), anchor="rm")
        d.line([x0, y0, x0, y1], fill=(60, 60, 60), width=3)
        d.line([x0, y1, x1, y1], fill=(60, 60, 60), width=3)
        n = len(m["values"])
        slot = (x1 - x0) / n
        for i, (lab, v) in enumerate(zip(m["labels"], m["values"])):
            cx = x0 + slot * (i + 0.5)
            w = slot * 0.55
            d.rectangle([cx - w / 2, y1 - (y1 - y0) * v / 100, cx + w / 2, y1], fill=(60, 110, 190))
            d.text((cx, y1 + 18), lab, font=_font(30, bold=True), fill=(30, 30, 30), anchor="mt")
    return img


def render(item: Item, size: int) -> Image.Image:
    img = render_master(item)
    return img if size == MASTER else img.resize((size, size), Image.LANCZOS)


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode()


# ---- Scoring ---------------------------------------------------------------------------------------

def parse(task, text):
    """Answer extraction. Uses the last non-empty line, because a model that works out loud
    ("1. square at ..., That's 7.\n\n7") puts its final answer there."""
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    t = lines[-1].strip() if lines else ""
    if task in ("count", "chart", "dense"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", t)
        return nums[-1] if nums else None
    if task == "absent":
        w = re.findall(r"[a-z]+", t.lower())
        return w[0] if w and w[0] in ("yes", "no") else None
    if task == "read":
        # A short reply is the answer itself: compare case-insensitively ("vh3lq" counts).
        cands = re.findall(r"\b[A-Z0-9]{4,6}\b", t.replace(" ", "").upper() if len(t) <= 12 else t)
        return cands[0] if cands else None
    raise ValueError(task)


def score(item: Item, text: str) -> dict:
    p = parse(item.task, text)
    out_close = None
    ok = False
    exact = False
    if p is not None:
        if item.task == "chart":
            err = abs(float(p) - float(item.answer))
            ok, exact = err <= 5, err == 0
        elif item.task == "count":
            ok = exact = float(p) == float(item.answer)
        elif item.task == "dense":
            ok = exact = float(p) == float(item.answer)
            out_close = abs(float(p) - float(item.answer)) <= 0.1 * float(item.answer)
        else:
            ok = exact = p == item.answer
    r = dict(pred=p, ok=bool(ok), exact=bool(exact), parsed=p is not None)
    if item.task == "dense":
        r["within10"] = bool(out_close)
    return r


# ---- Statistics ------------------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def paired_diff(a, b, n_boot=10000, seed=0):
    """Mean of b - a over paired 0/1 outcomes, with a percentile bootstrap 95% CI."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---- Claude's image-token rule (platform.claude.com/docs/en/build-with-claude/vision-coordinates) --

def claude_tokens(w, h):
    return math.ceil(w / 28) * math.ceil(h / 28)


def claude_resized(w, h, max_edge=2576, max_tokens=4784):
    """Port of the docs' reference resize (high-resolution tier defaults)."""
    def fits(a, b):
        return math.ceil(a / 28) * 28 <= max_edge and math.ceil(b / 28) * 28 <= max_edge and claude_tokens(a, b) <= max_tokens
    if fits(w, h):
        return w, h
    if h > w:
        rh, rw = claude_resized(h, w, max_edge, max_tokens)
        return rw, rh
    ar = w / h
    lo, hi = 1, w
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / ar), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / ar), 1)


# ---- GPU sampling ----------------------------------------------------------------------------------

class GpuSampler:
    """Polls nvidia-smi every `period` seconds in a thread; .stop() returns median clock and power."""
    def __init__(self, period=0.2):
        self.period, self.rows, self._stop = period, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=clocks.sm,power.draw,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            try:
                c, p, u = (float(x) for x in out.strip().split(","))
                self.rows.append((c, p, u))
            except ValueError:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self.stop()

    def stop(self):
        self._stop.set()
        self._t.join()
        import statistics as st
        busy = [r for r in self.rows if r[2] > 0] or self.rows
        if not busy:
            return {}
        return dict(sm_clock_mhz=st.median(r[0] for r in busy), power_w=st.median(r[1] for r in busy),
                    samples=len(self.rows))


# ---- Local VLMs ------------------------------------------------------------------------------------

class VLM:
    """Wraps a Hugging Face VLM: .ask(img, question) -> text, token counts, and timing helpers."""

    def __init__(self, model_id, image_size=None):
        import torch
        from transformers import AutoProcessor
        self.id = model_id
        self.kind = "qwen2vl" if "Qwen2-VL" in model_id else "smolvlm"
        if self.kind == "qwen2vl":
            from transformers import Qwen2VLForConditionalGeneration as M
            # min_pixels low enough that 224 x 224 is not upscaled; max high enough that 896 is not shrunk.
            self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=56 * 56, max_pixels=1344 * 1344)
        else:
            from transformers import AutoModelForImageTextToText as M
            kw = {}
            if image_size is not None:     # longest edge the processor resizes to before tiling
                kw["size"] = {"longest_edge": image_size}
            self.proc = AutoProcessor.from_pretrained(model_id, **kw)
        self.model = M.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        self.torch = torch

    def inputs(self, img, question, device="cuda"):
        """Processor output; device=None leaves it on the CPU (for prefetching in a thread)."""
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        x = self.proc(text=[prompt], images=[img], return_tensors="pt")
        return x.to(device) if device else x

    def image_tokens(self, inputs):
        tok = self.proc.tokenizer
        name = "<|image_pad|>" if self.kind == "qwen2vl" else "<image>"
        tid = tok.convert_tokens_to_ids(name)
        return int((inputs["input_ids"] == tid).sum())

    def ask(self, img, question, max_new_tokens=8, prepped=None):
        x = prepped.to("cuda") if prepped is not None else self.inputs(img, question)
        with self.torch.inference_mode():
            out = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return text.strip(), dict(prompt_tokens=int(x["input_ids"].shape[1]), image_tokens=self.image_tokens(x))

    def vision_forward(self, x):
        m = self.model.model
        if self.kind == "qwen2vl":
            return m.visual(x["pixel_values"].to(self.torch.bfloat16), grid_thw=x["image_grid_thw"])
        return m.get_image_features(pixel_values=x["pixel_values"].to(self.torch.bfloat16),
                                    pixel_attention_mask=x.get("pixel_attention_mask"))

    def time_split(self, img, question, warmup=3, reps=10):
        """Median ms for (vision encoder + connector) and for the whole prefill forward pass."""
        torch = self.torch
        x = self.inputs(img, question)
        res = {}
        with torch.inference_mode():
            for name, fn in (("vision_ms", lambda: self.vision_forward(x)),
                             ("prefill_ms", lambda: self.model(**x, logits_to_keep=1))):
                for _ in range(warmup):
                    fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(reps):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn()
                    e.record()
                    torch.cuda.synchronize()
                    ts.append(s.elapsed_time(e))
                ts.sort()
                res[name] = ts[len(ts) // 2]
                res[name + "_all"] = [round(t, 3) for t in ts]
        res["prompt_tokens"] = int(x["input_ids"].shape[1])
        res["image_tokens"] = self.image_tokens(x)
        return res
