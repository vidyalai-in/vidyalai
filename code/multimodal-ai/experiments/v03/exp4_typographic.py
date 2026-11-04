"""Exp 4: typographic attack. A white label with a wrong class name is pasted onto
(a) rendered shapes (own images, 224 px) and (b) CIFAR-10 test images upscaled to 224 px (measured only).
Usage: python exp4_typographic.py render | run <model-key>"""
import io, json, math, pickle, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import clip_utils as cu

OUT = "/home/amit/.cache/vidyalai-scratch/v03"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SHAPES = ["circle", "square", "triangle", "star", "heart", "hexagon"]
NEUTRAL = ["paper", "seven", "hello", "window", "music", "river"]  # not a class in either set


def poly_star(cx, cy, r, n=5):
    return [(cx + (r if k % 2 == 0 else 0.45 * r) * math.sin(math.pi * k / n),
             cy - (r if k % 2 == 0 else 0.45 * r) * math.cos(math.pi * k / n)) for k in range(2 * n)]


def draw_shape(kind, g):
    bg = tuple(int(v) for v in g.integers(200, 256, 3))
    fg = tuple(int(v) for v in g.integers(0, 200, 3))
    im = Image.new("RGB", (224, 224), bg); d = ImageDraw.Draw(im)
    r = g.uniform(55, 80); cx = g.uniform(r + 5, 224 - r - 5); cy = g.uniform(r + 5, 150 - r * 0.3)
    if kind == "circle": d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif kind == "square": d.rectangle([cx - r * .8, cy - r * .8, cx + r * .8, cy + r * .8], fill=fg)
    elif kind == "triangle": d.polygon([(cx, cy - r), (cx - r, cy + r * .75), (cx + r, cy + r * .75)], fill=fg)
    elif kind == "star": d.polygon(poly_star(cx, cy, r), fill=fg)
    elif kind == "hexagon": d.polygon([(cx + r * math.cos(math.pi * k / 3), cy + r * math.sin(math.pi * k / 3)) for k in range(6)], fill=fg)
    elif kind == "heart":
        pts = [(cx + r * 0.06 * 16 * math.sin(t) ** 3,
                cy - r * 0.06 * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)))
               for t in np.linspace(0, 2 * math.pi, 120)]
        d.polygon(pts, fill=fg)
    return im


def add_label(im, word, px, g):
    im = im.copy(); d = ImageDraw.Draw(im); f = ImageFont.truetype(FONT, px)
    l, t, r, b = d.textbbox((0, 0), word, font=f); w, h = r - l + 12, b - t + 10
    x = g.uniform(4, max(5, 220 - w)); y = g.uniform(224 - h - 40, 220 - h)
    d.rectangle([x, y, x + w, y + h], fill=(255, 255, 255), outline=(0, 0, 0))
    d.text((x + 6 - l, y + 5 - t), word, font=f, fill=(0, 0, 0))
    return im, (w * h) / 224 ** 2


def png(im):
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


def unpng(b): return Image.open(io.BytesIO(b)).convert("RGB")


def build():
    g = np.random.default_rng(0)
    sets = {}
    clean = []; ys = []
    for c, k in enumerate(SHAPES):
        for _ in range(100): clean.append(draw_shape(k, g)); ys.append(c)
    sets["shapes"] = (clean, np.array(ys), SHAPES)
    imgs, y = cu.cifar10("test")
    idx = np.concatenate([np.where(y == c)[0][:100] for c in range(10)])
    sets["cifar"] = ([imgs[i].resize((224, 224), Image.BICUBIC) for i in idx], y[idx], cu.CIFAR10_CLASSES)
    data = {}
    for name, (clean, ys, cls) in sets.items():
        wrong = [int((yy + g.integers(1, len(cls))) % len(cls)) for yy in ys]
        conds = {"clean": clean, "neutral32": [], "wrong16": [], "wrong32": []}
        area = {"neutral32": [], "wrong16": [], "wrong32": []}
        for im, w in zip(clean, wrong):
            for cname, word, px in [("neutral32", NEUTRAL[g.integers(len(NEUTRAL))], 32), ("wrong16", cls[w], 16), ("wrong32", cls[w], 32)]:
                a, ar = add_label(im, word, px, g); conds[cname].append(png(a)); area[cname].append(ar)
        conds["clean"] = [png(im) for im in clean]
        data[name] = {"conds": conds, "y": ys, "wrong": np.array(wrong), "classes": cls,
                      "area": {k: float(np.mean(v)) for k, v in area.items()}}
        print(name, len(ys), data[name]["area"])
    pickle.dump(data, open(f"{OUT}/typo.pkl", "wb"))
    for i in (0, 250):  # previews
        for k in data["shapes"]["conds"]: open(f"{OUT}/typo-shape{i}-{k}.png", "wb").write(data["shapes"]["conds"][k][i])


if sys.argv[1] == "render":
    build(); sys.exit()

key = sys.argv[2]
name = {"clip-b32": "openai/clip-vit-base-patch32", "siglip-b16": "google/siglip-base-patch16-224"}[key]
data = pickle.load(open(f"{OUT}/typo.pkl", "rb"))
model, proc, kind = cu.load_encoder(name)
TEMPL = {"shapes": ["a drawing of a {}.", "a {} shape.", "a picture of a {}."], "cifar": cu.CIFAR10_TEMPLATES}
res = {"model": name}
for ds, D in data.items():
    W = cu.zeroshot_classifier(model, proc, kind, D["classes"], TEMPL[ds])
    out = {"n": len(D["y"]), "area": D["area"]}
    corr = {}
    for c, ims in D["conds"].items():
        f = cu.encode_images(model, proc, [unpng(b) for b in ims], bs=100)
        pred = (f @ W.t()).argmax(1).numpy()
        corr[c] = (pred == D["y"]).astype(int)
        out[c] = {"acc": cu.wilson(int(corr[c].sum()), len(pred))}
        if c != "clean":
            hit = int((pred == D["wrong"]).sum())
            out[c]["pred_is_written_word"] = cu.wilson(hit, len(pred))
            # among items that were correct when clean
            m = corr["clean"] == 1
            out[c]["flip_given_clean_correct"] = cu.wilson(int(((pred == D["wrong"]) & m).sum()), int(m.sum()))
        print(key, ds, c, out[c], flush=True)
    out["paired"] = {c: cu.paired_diff_ci(corr["clean"], corr[c]) for c in corr if c != "clean"}
    if ds == "shapes":  # probabilities for the shown example (item 0)
        for c in D["conds"]:
            f = cu.encode_images(model, proc, [unpng(D["conds"][c][0])])
            z = float(model.logit_scale.exp()) * f @ W.t()
            p = (torch.sigmoid(z + float(model.logit_bias)) if kind == "siglip" else torch.softmax(z, -1))[0]
            out.setdefault("example0", {})[c] = {D["classes"][j]: round(float(p[j]), 4) for j in range(len(D["classes"]))}
    res[ds] = out
json.dump(res, open(f"{OUT}/exp4-{key}.json", "w"), indent=1, default=float)
