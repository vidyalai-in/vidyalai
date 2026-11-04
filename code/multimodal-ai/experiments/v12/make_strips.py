"""Frame strips for the article. python make_strips.py tiny|animatediff OUT.png"""
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
def tiny(out, clips=(0, 1), scale=2):
    rows = [("real", "Real clips (test set)"), ("frames", "No temporal attention"),
            ("factorized", "Factorized attention"), ("full", "Full spacetime attention"),
            ("inflated", "Factorized, inflated from the frame-by-frame model")]
    S = 48 * scale; gap = 4; lab = 34
    W = 8 * (S + gap) - gap; H = len(rows) * (lab + len(clips) * (S + gap))
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im); y = 0
    for key, name in rows:
        x = np.load(f"runs/clips-{key}.npy")
        d.text((0, y + 9), name, fill=20, font=FB); y += lab
        for c in clips:
            for t in range(8):
                fr = Image.fromarray(x[c, t, 0]).resize((S, S), Image.NEAREST).convert("L")
                im.paste(fr, (t * (S + gap), y))
            y += S + gap
    im.save(out, optimize=True)
def animatediff(out, dirs, every=2, size=192):
    lab = 30; gap = 4
    n = len(range(0, 16, every)); W = n * (size + gap) - gap
    H = len(dirs) * (lab + size + gap)
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im); y = 0
    for path, name in dirs:
        d.text((0, y + 6), name, fill=(11, 26, 51), font=FB); y += lab
        for j, k in enumerate(range(0, 16, every)):
            fr = Image.open(f"{path}/{k:02d}.png").convert("RGB").resize((size, size), Image.LANCZOS)
            im.paste(fr, (j * (size + gap), y))
            d.text((j * (size + gap) + 5, y + size - 21), f"{k + 1}", fill="white", font=F, stroke_width=2, stroke_fill="black")
        y += size + gap
    im.save(out, optimize=True)
if __name__ == "__main__":
    if sys.argv[1] == "tiny":
        tiny(sys.argv[2])
    else:
        animatediff(sys.argv[2], [(p.split("=")[0], p.split("=")[1]) for p in sys.argv[3:]])
