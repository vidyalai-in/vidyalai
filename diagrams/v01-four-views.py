# Generates public/diagrams/v01-four-views.png (+ .webp): the article's synthetic test picture seen four ways.
# Inputs come from the v01 run (scratch runs/): orig_512.png, recon_vae.png, recon_vq.png, latent_vae.npy.
# Usage: python3 diagrams/v01-four-views.py <runs dir> public/diagrams/v01-four-views
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

runs, out = sys.argv[1], sys.argv[2]
S = 300                     # panel size in the figure
orig = Image.open(f"{runs}/orig_512.png").convert("RGB")
font = None
for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
    try:
        font = ImageFont.truetype(p, 15); break
    except OSError:
        pass
font = font or ImageFont.load_default()


def grid(img, cells, color=(20, 20, 20)):
    im = img.resize((S, S), Image.LANCZOS).copy()
    d = ImageDraw.Draw(im)
    for i in range(1, cells):
        v = round(i * S / cells)
        d.line([(v, 0), (v, S)], fill=color, width=1)
        d.line([(0, v), (S, v)], fill=color, width=1)
    return im


# 1. ViT-B/16 at 224 px: a 14 x 14 grid of 16-pixel patches
p1 = grid(orig.resize((224, 224), Image.LANCZOS), 14)
# 2. Claude at 512 px: ceil(512/28) = 19 -> 19 x 19 patches of 28 px (the last row and column padded)
padded = Image.new("RGB", (532, 532), (128, 128, 128))
padded.paste(orig, (0, 0))
p2 = grid(padded, 19)
# 3. SD VAE latent: first 3 of 4 channels as RGB, 64 x 64 cells, nearest-neighbour upscaled
z = np.load(f"{runs}/latent_vae.npy")[:3]
z = (z - z.min(axis=(1, 2), keepdims=True)) / (np.ptp(z, axis=(1, 2), keepdims=True) + 1e-8)
p3 = Image.fromarray((z.transpose(1, 2, 0) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
# 4. VQ-GAN decode of the 32 x 32 codes
p4 = Image.open(f"{runs}/recon_vq.png").convert("RGB").resize((S, S), Image.LANCZOS)

labels = ["ViT-B/16: 14 x 14 = 196 patches", "Claude: 19 x 19 = 361 (gray = padding)",
          "SD VAE latent: 64 x 64 x 4", "VQ-GAN: 32 x 32 codes, decoded"]
pad, top = 12, 30
W = 2 * S + 3 * pad
H = 2 * (S + top) + 3 * pad - pad
fig = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(fig)
for k, p in enumerate([p1, p2, p3, p4]):
    x = pad + (k % 2) * (S + pad)
    y = pad + (k // 2) * (S + top + pad)
    d.text((x, y), labels[k], fill=(11, 26, 51), font=font)
    fig.paste(p, (x, y + top - 6))
fig.save(out + ".png", optimize=True)
fig.save(out + ".webp", quality=85)
print(fig.size)
