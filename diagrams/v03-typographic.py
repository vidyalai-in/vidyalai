# Generates public/diagrams/v03-typographic.svg and v03-typographic-dark.svg (data: v03-clip-contrastive-learning.md).
# Usage: python3 diagrams/v03-typographic.py public/diagrams/v03-typographic, then render the PNG with sharp at density 144.
# Data: share of predictions equal to a wrong class name written on a white label (32 px bold text, about 8-9% of
# the pixels), with Wilson 95% intervals. Rendered shapes: 600 images, 6 classes. CIFAR-10: 1,000 test images
# (100 per class) upscaled to 224 px. CLIP = openai/clip-vit-base-patch32, SigLIP = google/siglip-base-patch16-224
# (zero-shot, prompt ensembles); Haiku = claude-haiku-5-5, thinking disabled, same images, label list in the prompt.
import sys
GROUPS = [
    ("Rendered shapes (600 images)", [("CLIP ViT-B/32", 0.940, 0.918, 0.956), ("SigLIP B/16", 0.637, 0.597, 0.674),
                                     ("Claude Haiku 5.5", 0.020, 0.012, 0.035)]),
    ("CIFAR-10 photos (1,000 images)", [("CLIP ViT-B/32", 0.847, 0.823, 0.868), ("SigLIP B/16", 0.744, 0.716, 0.770),
                                       ("Claude Haiku 5.5", 0.122, 0.103, 0.144)]),
]
W, H = 780, 400; L, R, T, B = 170, 40, 24, 58
BH, GAP = 30, 10
def sx(v): return L + v * (W - L - R)
def svg(th, acc):
    ink, mut, grid, surf = th
    halo = f'paint-order="stroke" stroke="{surf}" stroke-width="4" stroke-linejoin="round"'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (0, .25, .5, .75, 1):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v*100:.0f}%</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Predictions that match the wrong word on the label (95% interval)</text>')
    y = T + 8
    for g, bars in GROUPS:
        o.append(f'<text x="{L}" y="{y+12}" fill="{ink}" font-size="14" font-weight="bold">{g}</text>')
        y += 22
        for name, v, lo, hi in bars:
            o.append(f'<text x="{L-10}" y="{y+BH/2+5:.1f}" fill="{ink}" font-size="13" text-anchor="end">{name}</text>')
            o.append(f'<rect x="{L}" y="{y}" width="{max(sx(v)-L,1.5):.1f}" height="{BH}" fill="{acc}" rx="2"/>')
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y+BH/2}" y2="{y+BH/2}" stroke="{surf}" stroke-width="6"/>')
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y+BH/2}" y2="{y+BH/2}" stroke="{ink}" stroke-width="2"/>')
            for e in (lo, hi):
                o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{y+BH/2-6}" y2="{y+BH/2+6}" stroke="{ink}" stroke-width="2"/>')
            o.append(f'<text x="{sx(hi)+8:.1f}" y="{y+BH/2+5:.1f}" fill="{ink}" font-size="13" font-weight="bold" {halo}>{v*100:.1f}%</text>')
            y += BH + GAP
        y += 14
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff'); dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, '#0b5fd6')); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, '#38bdf8'))
