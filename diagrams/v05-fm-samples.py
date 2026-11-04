# Generates public/diagrams/v05-fm-over-training.png and public/diagrams/v05-fm-samplers.png (data: v05-diffusion-math.md).
# Samples from the Fashion-MNIST U-Net DDPM (1.59M parameters, linear schedule, EMA weights) trained in scratch v05.
# over-training: 12 samples per row from the same starting noise (seed 2), DDIM 50 steps, at several training steps
#   (runs/fm_linear/grids.npz, written during training).
# samplers: the first 12 of the 2,000 samples per setting from the final model, same starting noise (seed 100):
#   DDIM and DDPM at 10, 50 and 250 or 1,000 steps (runs/samples_linear_<method><steps>_s0.npy).
# Usage: python3 diagrams/v05-fm-samples.py /home/amit/.cache/vidyalai-scratch/v05/runs public/diagrams
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

runs, out = sys.argv[1], sys.argv[2]
S, PAD, LW, TOP, N = 56, 4, 190, 8, 12
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()


def tile(x):
    a = np.clip((x.astype(np.float32) + 1) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(a, "L").resize((S, S), Image.NEAREST)


def sheet(rows, path):
    W = LW + N * (S + PAD); H = TOP + len(rows) * (S + PAD) + 4
    img = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(img)
    for i, (label, xs) in enumerate(rows):
        y = TOP + i * (S + PAD)
        d.text((8, y + S // 2), label, fill=(11, 26, 51), font=font, anchor="lm")
        for j in range(N):
            img.paste(tile(xs[j, 0]), (LW + j * (S + PAD), y))
    img.save(path, optimize=True); print(path, img.size)


g = np.load(f"{runs}/fm_linear/grids.npz")
steps = sorted(int(k) for k in g.keys())
want = [s for s in steps if s in (500, 1000, 1500, 2500, 5000, 7500, 10000, 12500)] + [steps[-1]]
want = sorted(set(want))
sheet([(f"step {s:,}", g[str(s)]) for s in want], f"{out}/v05-fm-over-training.png")
rows = []
for m, st in (("ddim", 10), ("ddpm", 10), ("ddim", 50), ("ddpm", 50), ("ddim", 250), ("ddpm", 1000)):
    rows.append((f"{m.upper()}, {st:,} steps", np.load(f"{runs}/samples_linear_{m}{st}_s0.npy")))
sheet(rows, f"{out}/v05-fm-samplers.png")
