"""Sample grid: rows = DDPM (DDIM), rectified flow (Euler), after reflow; columns = 1, 4, 16 and 100 (reflow: 32) steps.
Each cell: 4 x 4 samples from seed 0 (the same noise for every row and column). Writes public/diagrams/v07-samples.png."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = "/home/amit/.cache/vidyalai-scratch/v07/runs/"
rows = [("grid-ddpm.npz", "Diffusion, DDIM", [1, 4, 16, 100]), ("grid-rf.npz", "Rectified flow, Euler", [1, 4, 16, 100]),
        ("grid-reflow.npz", "After reflow, Euler", [1, 4, 16, 32])]
S = 2; C = 28 * S; PAD = 6; LW = 210; TH = 34
cell = 4 * C
W = LW + 4 * (cell + PAD) + PAD; H = TH + 3 * (cell + PAD) + PAD
img = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
except OSError:
    font = ImageFont.load_default()
for j in range(4):
    d.text((LW + PAD + j * (cell + PAD) + cell // 2, TH // 2), (f"{rows[0][2][j]} step" + ("s" if rows[0][2][j] > 1 else "")), fill=(11, 26, 51), font=font, anchor="mm")
for i, (f, label, steps) in enumerate(rows):
    z = np.load(R + f)
    y0 = TH + PAD + i * (cell + PAD)
    d.text((10, y0 + cell // 2), label, fill=(11, 26, 51), font=font, anchor="lm")
    for j, s in enumerate(steps):
        x = z[f"s{s}"].astype(np.float32)[:16, 0]
        g = ((x.clip(-1, 1) + 1) * 127.5).astype(np.uint8)
        g = g.reshape(4, 4, 28, 28).transpose(0, 2, 1, 3).reshape(112, 112)
        tile = Image.fromarray(g).resize((cell, cell), Image.NEAREST)
        img.paste(tile, (LW + PAD + j * (cell + PAD), y0))
    if steps[-1] != rows[0][2][-1]:
        d.text((10, y0 + cell // 2 + 24), f"(last column: {steps[-1]} steps)", fill=(74, 91, 120), font=font, anchor="lm")
img.save("/home/amit/AlpsCode/ai/vidyalai/public/diagrams/v07-samples.png", optimize=True)
print(img.size)
