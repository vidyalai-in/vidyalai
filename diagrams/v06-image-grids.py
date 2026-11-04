# Generates public/diagrams/v06-guidance-grid.png and v06-x0-steps.png from the v06 runs (scratch v06/runs/).
# Guidance grid: Stable Diffusion 1.5, DPM-Solver++ 2M, 20 steps, 512 x 512, one seed per row, six guidance scales.
# x0 figure: DDIM, 20 steps, guidance 7.5; top row the decoded noisy latent, bottom row the decoded predicted clean latent.
# Usage: python3 diagrams/v06-image-grids.py <runs dir> public/diagrams
import sys
from PIL import Image, ImageDraw, ImageFont
runs, out = sys.argv[1], sys.argv[2]
ROWS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 11
font = None
for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
    try:
        font = ImageFont.truetype(p, 14); small = ImageFont.truetype(p, 13); break
    except OSError:
        pass
INK = (11, 26, 51)
def sheet(cells, cols, labels_top, labels_left, S, left_w, path):
    pad, top = 6, 24
    rows = len(cells) // cols
    W = left_w + cols * (S + pad) + pad
    H = top + rows * (S + pad) + pad
    fig = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(fig)
    for c, t in enumerate(labels_top):
        d.text((left_w + pad + c * (S + pad) + S // 2, 4), t, fill=INK, font=font, anchor="ma")
    for r in range(rows):
        for line_i, line in enumerate(labels_left[r].split("\n")):
            d.text((6, top + pad + r * (S + pad) + 8 + line_i * 17), line, fill=INK, font=small)
        for c in range(cols):
            cell = cells[r * cols + c]
            if cell is None:  # the shipped safety checker flagged this decoded latent: not shown
                im = Image.new("RGB", (S, S), (200, 205, 214)); dd = ImageDraw.Draw(im)
                dd.text((S // 2, S // 2 - 10), "flagged by the", fill=INK, font=small, anchor="mm")
                dd.text((S // 2, S // 2 + 10), "safety checker", fill=INK, font=small, anchor="mm")
            else:
                im = Image.open(cell).convert("RGB").resize((S, S), Image.LANCZOS)
            fig.paste(im, (left_w + pad + c * (S + pad), top + pad + r * (S + pad)))
    fig.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(path + ".png", optimize=True)  # palette PNG keeps it small; fig.save(path + ".webp", quality=82)
    print(path, fig.size)
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/v06")
from prompts import PROMPTS
import textwrap
GS = ["1.0", "3.0", "5.0", "7.5", "12.0", "20.0"]
cells, left = [], []
for p in ROWS:
    left.append("\n".join(textwrap.wrap(PROMPTS[p], 17)))
    cells += [f"{runs}/img/guidance/p{p:02d}_s{SEED}_dpmpp2m_20_g{g}.png" for g in GS]
sheet(cells, 6, [f"scale {float(g):g}" for g in GS], left, 150, 150, f"{out}/v06-guidance-grid")
STEPS = [0, 2, 4, 7, 11, 19]
import json
flag = {r["step"]: r["nsfw"] for r in json.load(open(f"{runs}/exp3_x0.json")) if "step" in r and r["prompt"].startswith("a lighthouse")}
cells = [None if flag[i][0] else f"{runs}/x0/lighthouse_xt_{i:02d}.png" for i in STEPS] + [None if flag[i][1] else f"{runs}/x0/lighthouse_x0_{i:02d}.png" for i in STEPS]
sheet(cells, 6, [f"step {i+1} of 20" for i in STEPS], ["noisy latent\ndecoded", "predicted\nclean latent\ndecoded"], 150, 110, f"{out}/v06-x0-steps")
