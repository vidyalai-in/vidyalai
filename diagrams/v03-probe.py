# Generates public/diagrams/v03-probe.svg and v03-probe-dark.svg (data: v03-clip-contrastive-learning.md).
# Usage: python3 diagrams/v03-probe.py public/diagrams/v03-probe, then render the PNG with sharp at density 144.
# Data: CIFAR-10 test accuracy (10,000 images) of a logistic-regression probe on frozen image embeddings, trained on
# k labelled images per class (10 random draws per k; mean and t-based 95% interval), versus zero-shot with the
# 18-template ensemble (Wilson 95% interval). Full probe: 40,000 labelled training images.
import math, sys
K = [1, 2, 4, 8, 16, 32, 64, 128]
SER = {
    "CLIP ViT-B/32": dict(probe=[(65.0, 62.2, 67.9), (75.9, 73.1, 78.6), (84.0, 82.4, 85.6), (87.8, 87.1, 88.4),
                                  (89.8, 89.5, 90.1), (91.2, 90.9, 91.5), (92.1, 91.9, 92.4), (92.8, 92.6, 92.9)],
                          zs=89.8, full=94.6),
    "SigLIP B/16": dict(probe=[(66.5, 64.2, 68.9), (81.2, 79.0, 83.4), (86.7, 85.4, 88.0), (90.4, 89.6, 91.1),
                                (92.5, 92.2, 92.8), (93.3, 93.0, 93.7), (94.1, 93.9, 94.3), (94.6, 94.5, 94.7)],
                         zs=92.4, full=96.0),
}
W, H = 780, 450; L, R, T, B = 64, 200, 24, 62
X0, X1 = 0.8, 160; Y0, Y1 = 60, 100
def sx(x): return L + (math.log(x) - math.log(X0)) / (math.log(X1) - math.log(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
def svg(th, accs):
    ink, mut, grid, surf = th
    halo = f'paint-order="stroke" stroke="{surf}" stroke-width="4" stroke-linejoin="round"'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for k in K:
        o.append(f'<line x1="{sx(k):.1f}" x2="{sx(k):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(k):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{k}</text>')
    for y in range(60, 101, 10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Labelled training images per class (log scale)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">CIFAR-10 test accuracy</text>')
    labels = []
    for (name, d), acc in zip(SER.items(), accs):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(d["zs"]):.1f}" y2="{sy(d["zs"]):.1f}" stroke="{acc}" stroke-width="2" stroke-dasharray="7 5"/>')
        pts = " ".join(f"{sx(k):.1f},{sy(m):.1f}" for k, (m, lo, hi) in zip(K, d["probe"]))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{acc}" stroke-width="2.5"/>')
        for k, (m, lo, hi) in zip(K, d["probe"]):
            o.append(f'<line x1="{sx(k):.1f}" x2="{sx(k):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{acc}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(k):.1f}" cy="{sy(m):.1f}" r="4.5" fill="{acc}" stroke="{surf}" stroke-width="1.5"/>')
        labels.append((name, d, acc))
    xr = W - R + 10
    # right-hand labels: zero-shot and probe end points, placed to avoid overlap
    ys = [("SigLIP B/16", "probe at 128: 94.6%", "zero-shot: 92.4%", accs[1]), ("CLIP ViT-B/32", "probe at 128: 92.8%", "zero-shot: 89.8%", accs[0])]
    y0 = sy(98)
    for i, (n, a, b, acc) in enumerate(ys):
        y = y0 + i * 78
        o.append(f'<text x="{xr}" y="{y:.1f}" fill="{acc}" font-size="14" font-weight="bold">{n}</text>')
        o.append(f'<text x="{xr}" y="{y+20:.1f}" fill="{ink}" font-size="13">{a}</text>')
        o.append(f'<text x="{xr}" y="{y+38:.1f}" fill="{ink}" font-size="13">{b}</text>')
    o.append(f'<text x="{xr}" y="{sy(70):.1f}" fill="{mut}" font-size="12">Solid: probe, mean of 10</text>')
    o.append(f'<text x="{xr}" y="{sy(70)+16:.1f}" fill="{mut}" font-size="12">draws, 95% interval</text>')
    o.append(f'<text x="{xr}" y="{sy(70)+36:.1f}" fill="{mut}" font-size="12">Dashed: zero-shot,</text>')
    o.append(f'<text x="{xr}" y="{sy(70)+52:.1f}" fill="{mut}" font-size="12">no labelled images</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff'); dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#0b5fd6', '#eb6834'))); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#38bdf8', '#d95926')))
