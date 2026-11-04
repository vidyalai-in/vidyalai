# Generates public/diagrams/v08-amused-grid.png (+ .webp): aMUSEd-256 (amused/amused-256, OpenRAIL++) samples from
# the v08 run, for v08-autoregressive-image-generation.md. Rows: prompts 0, 16, 33 and 10 of the v08 prompt set
# ("a red apple on a wooden table", "a snowy mountain peak under a blue sky", "a hot air balloon over green hills",
# "a red fox in the snow"); columns: 1, 2, 4 and 12 decoding steps. Seed 0, guidance scale 10, 256 x 256, fp16
# transformer and text encoder, fp32 VQ-GAN, default AmusedScheduler (cosine schedule, temperature 2 to 0).
# Usage: python diagrams/v08-amused-grid.py <scratch v08/amused> public/diagrams/v08-amused-grid
import sys
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]
P, STEPS, S, PAD, TOP = [0, 16, 33, 10], [1, 2, 4, 12], 180, 6, 34
W = len(STEPS) * (S + PAD) + PAD
H = TOP + len(P) * (S + PAD)
fig = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(fig)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
for c, s in enumerate(STEPS):
    x = PAD + c * (S + PAD)
    label = f"{s} step" + ("" if s == 1 else "s")
    w = d.textlength(label, font=font)
    d.text((x + (S - w) / 2, 6), label, fill=(11, 26, 51), font=font)
    for r, p in enumerate(P):
        im = Image.open(f"{src}/s{s:02d}_g10_seed0_p{p:02d}.png").convert("RGB").resize((S, S), Image.LANCZOS)
        fig.paste(im, (x, TOP + r * (S + PAD)))
fig.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(out + ".png", optimize=True)  # palette PNG keeps it under 300 KB
fig.save(out + ".webp", quality=85)
print(fig.size)
