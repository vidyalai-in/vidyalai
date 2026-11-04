# Generates public/diagrams/v05-forward-fmnist.png: one Fashion-MNIST test image (index 0, an ankle boot;
# Xiao et al. 2017, MIT license, Zalando Research) noised with the closed form x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps
# under the linear schedule (top row) and the cosine noise schedule (bottom row), same noise eps (seed 0), T = 1000.
# Usage: python3 diagrams/v05-forward-fmnist.py /home/amit/.cache/vidyalai-scratch/mm/data/fmnist.npz public/diagrams/v05-forward-fmnist
import sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

npz, out = sys.argv[1], sys.argv[2]
x0 = np.load(npz)["test_x"][0].astype(np.float64) / 127.5 - 1
T = 1000
betas = np.linspace(1e-4, 0.02, T); ab_lin = np.cumprod(1 - betas)
f = lambda t: math.cos((t / T + 0.008) / 1.008 * math.pi / 2) ** 2
ab_cos = np.cumprod([1 - min(1 - f(t + 1) / f(t), 0.999) for t in range(T)])
eps = np.random.default_rng(0).standard_normal(x0.shape)
TS = [0, 100, 200, 300, 400, 500, 700, 999]
S, PAD, LW, TOP = 112, 10, 150, 46
font = bold = None
for p, b in (("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),):
    try: font, bold, small = ImageFont.truetype(p, 17), ImageFont.truetype(b, 17), ImageFont.truetype(p, 14)
    except OSError: font = bold = small = ImageFont.load_default()
W = LW + len(TS) * (S + PAD); H = TOP + 2 * (S + 44) + 6
img = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(img)
for j, t in enumerate(TS):
    d.text((LW + j * (S + PAD) + S // 2, 14), f"t = {t}", fill=(11, 26, 51), font=bold, anchor="mt")
for i, (name, ab) in enumerate((("Linear", ab_lin), ("Cosine", ab_cos))):
    y = TOP + i * (S + 44)
    d.text((8, y + S // 2 - 12), name, fill=(11, 26, 51), font=bold)
    d.text((8, y + S // 2 + 10), "schedule", fill=(74, 91, 120), font=font)
    for j, t in enumerate(TS):
        xt = math.sqrt(ab[t]) * x0 + math.sqrt(1 - ab[t]) * eps
        a = np.clip((xt + 1) * 127.5, 0, 255).astype(np.uint8)
        tile = Image.fromarray(a, "L").resize((S, S), Image.NEAREST).convert("RGB")
        X = LW + j * (S + PAD); img.paste(tile, (X, y))
        snr = ab[t] / (1 - ab[t])
        d.text((X + S // 2, y + S + 6), f"SNR {snr:.2g}" if snr < 100 else f"SNR {snr:.0f}", fill=(74, 91, 120), font=small, anchor="mt")
img.save(out + ".png", optimize=True)
print(img.size)
