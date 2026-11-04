"""Exp 6: a real 4 x 4 CLIP similarity matrix on four rendered images, and the loss at several scales."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import clip_utils as cu
import math

def poly_star(cx, cy, r, n=5):
    return [(cx + (r if k % 2 == 0 else 0.45 * r) * math.sin(math.pi * k / n),
             cy - (r if k % 2 == 0 else 0.45 * r) * math.cos(math.pi * k / n)) for k in range(2 * n)]

def img(kind, col):
    im = Image.new("RGB", (224, 224), (235, 235, 235)); d = ImageDraw.Draw(im)
    if kind == "circle": d.ellipse([47, 47, 177, 177], fill=col)
    if kind == "square": d.rectangle([52, 52, 172, 172], fill=col)
    if kind == "triangle": d.polygon([(112, 40), (40, 180), (184, 180)], fill=col)
    if kind == "star": d.polygon(poly_star(112, 118, 80), fill=col)
    return im

items = [("circle", (210, 40, 40), "red"), ("square", (40, 80, 220), "blue"),
         ("triangle", (40, 170, 60), "green"), ("star", (235, 200, 30), "yellow")]
ims = [img(k, c) for k, c, _ in items]
caps = [f"a {n} {k}" for k, _, n in items]
w = Image.new("RGB", (4 * 224 + 3 * 16, 224), (255, 255, 255))
for i, im in enumerate(ims): w.paste(im, (i * 240, 0))
w.save("/home/amit/.cache/vidyalai-scratch/v03/matrix-images.png")
model, proc, kind = cu.load_encoder("openai/clip-vit-base-patch32")
I = cu.encode_images(model, proc, ims); T = cu.encode_texts(model, proc, caps, kind)
S = I @ T.t()
out = {"captions": caps, "cos": S.numpy().round(4).tolist(), "loss": {}}
for scale in (1, 10, 14.3, 30, 100):
    out["loss"][scale] = float(cu.clip_loss(I, T, scale))
    out.setdefault("rowprob", {})[scale] = torch.softmax(scale * S, 1).numpy().round(4).tolist()
print(json.dumps(out, indent=1))
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/v03/exp6.json", "w"), indent=1)
