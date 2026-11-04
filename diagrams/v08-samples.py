# Generates public/diagrams/v08-samples.png (+ .webp): Fashion-MNIST samples for v08-autoregressive-image-generation.md.
# Rows of 16: real training images (Fashion-MNIST, Zalando Research, MIT license), the VQ-VAE's reconstructions of
# them (7 x 7 codes, K = 512), and unconditional samples from the raster-order prior (temperature 1.0) and the
# masked prior at 8 steps (seed 0 of each run). Inputs: scratch v08/runs/grid_*.pt and v08/ckpt (written by
# v08/make_grid_inputs.py into v08/runs/grid_real.pt and grid_recon.pt).
# Usage: python diagrams/v08-samples.py <scratch v08/runs> public/diagrams/v08-samples
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
runs, out = sys.argv[1], sys.argv[2]
rows = [("Real images", "grid_real.pt"), ("VQ-VAE reconstructions", "grid_recon.pt"),
        ("Raster order, 49 passes", "grid_ar_t1.0.pt"), ("Masked, 8 passes", "grid_mask_T8.pt"),
        ("Masked, 1 pass", "grid_mask_T1.pt")]
S, N, PAD, LW = 56, 16, 4, 250
W = LW + N * (S + PAD) + PAD
H = PAD + len(rows) * (S + PAD)
fig = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(fig)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
except OSError:
    font = ImageFont.load_default()
for r, (label, f) in enumerate(rows):
    x = torch.load(f"{runs}/{f}").float()[:N]
    y0 = PAD + r * (S + PAD)
    d.text((10, y0 + S // 2 - 11), label, fill=(11, 26, 51), font=font)
    for i in range(N):
        a = ((x[i, 0].clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
        im = Image.fromarray(a).resize((S, S), Image.NEAREST)
        fig.paste(im, (LW + PAD + i * (S + PAD), y0))
fig.save(out + ".png", optimize=True)
fig.save(out + ".webp", quality=90)
print(fig.size)
