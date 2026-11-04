# Generates public/diagrams/v03-tiny-clip.svg and v03-tiny-clip-dark.svg (data: v03-clip-contrastive-learning.md).
# Usage: python3 diagrams/v03-tiny-clip.py public/diagrams/v03-tiny-clip, then render the PNG with sharp at density 144.
# Data: a 0.78M-parameter CLIP (4-layer CNN + 2-layer transformer, 64-d embeddings) trained from scratch on synthetic
# 32 x 32 shape scenes with 288 possible captions. Top-1 image-to-caption retrieval over all 288 captions on 2,000
# held-out images of seen shape-colour pairs; mean of 3 seeds, bars = min to max over seeds.
# Left: fixed temperature, batch 256, 250 steps (64,000 pairs); the learned run ends at scale 17.6 (tau = 0.057).
# Right: learned temperature; solid = 250 steps at every batch size, dashed = 64,000 pairs at every batch size.
import math, sys
TEMP = [(0.01, 96.9, 96.2, 97.5), (0.03, 99.0, 98.9, 99.1), (0.07, 99.2, 98.8, 99.4), (0.2, 97.2, 96.4, 98.0),
        (0.5, 67.4, 61.2, 72.0), (1.0, 55.4, 54.9, 56.0)]
LEARNED = (1 / 17.6, 99.4, 99.2, 99.5)
BS = [16, 64, 256, 1024]
STEPS = {"softmax": [(61.0, 57.0, 63.7), (97.2, 96.8, 97.9), (99.4, 99.2, 99.5), (99.6, 99.4, 99.8)],
         "sigmoid": [(38.8, 25.3, 55.3), (88.5, 82.2, 93.7), (97.2, 96.5, 98.2), (98.6, 98.0, 99.0)]}
DATA = {"softmax": [(98.4, 97.6, 99.1), (99.7, 99.5, 99.9), (99.4, 99.2, 99.5), (78.7, 71.6, 83.0)],
        "sigmoid": [(100.0, 100.0, 100.0), (99.8, 99.7, 99.8), (97.2, 96.5, 98.2), (2.5, 1.9, 3.3)]}
W, H = 780, 470; T, B = 46, 70
P1 = (64, 360); P2 = (440, 760)
def sy(y): return H - B - y / 100 * (H - T - B)
def sxa(x): return P1[0] + (math.log(x) - math.log(0.008)) / (math.log(1.25) - math.log(0.008)) * (P1[1] - P1[0])
def sxb(x): return P2[0] + (math.log2(x) - math.log2(11)) / (math.log2(1500) - math.log2(11)) * (P2[1] - P2[0])
def svg(th, a1, a2):
    ink, mut, grid, surf = th
    halo = f'paint-order="stroke" stroke="{surf}" stroke-width="4" stroke-linejoin="round"'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x0, x1 in (P1, P2):
        for y in range(0, 101, 20):
            o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
    for y in range(0, 101, 20):
        o.append(f'<text x="{P1[0]-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
        o.append(f'<text x="{P2[0]-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    o.append(f'<text transform="translate(16,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Top-1 retrieval, 288 captions</text>')
    o.append(f'<text x="{P1[0]}" y="{T-20}" fill="{ink}" font-size="14" font-weight="bold">Fixed temperature (batch 256)</text>')
    o.append(f'<text x="{P2[0]}" y="{T-20}" fill="{ink}" font-size="14" font-weight="bold">Batch size (learned temperature)</text>')
    for x in (0.01, 0.03, 0.1, 0.3, 1):
        o.append(f'<line x1="{sxa(x):.1f}" x2="{sxa(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sxa(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:g}</text>')
    o.append(f'<text x="{(P1[0]+P1[1])/2}" y="{H-B+44}" fill="{ink}" font-size="14" text-anchor="middle">Temperature τ (log scale)</text>')
    for b in BS:
        o.append(f'<line x1="{sxb(b):.1f}" x2="{sxb(b):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sxb(b):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{b}</text>')
    o.append(f'<text x="{(P2[0]+P2[1])/2}" y="{H-B+44}" fill="{ink}" font-size="14" text-anchor="middle">Pairs per batch (log scale)</text>')
    def series(pts, sx, col, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{sy(m):.1f}" for x, m, lo, hi in pts)}" fill="none" stroke="{col}" stroke-width="2.5"{d}/>')
        for x, m, lo, hi in pts:
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{col}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(m):.1f}" r="4.5" fill="{col}" stroke="{surf}" stroke-width="1.5"/>')
    series(TEMP, sxa, a1)
    x, m, lo, hi = LEARNED
    o.append(f'<rect x="{sxa(x)-5:.1f}" y="{sy(m)-5:.1f}" width="10" height="10" fill="{a2}" stroke="{surf}" stroke-width="1.5"/>')
    o.append(f'<rect x="{sxa(0.012)-5:.1f}" y="{sy(89)-9:.1f}" width="10" height="10" fill="{a2}"/>')
    o.append(f'<text x="{sxa(0.012)+10:.1f}" y="{sy(89):.1f}" fill="{ink}" font-size="12" {halo}>learned τ, ends at 0.057: 99.4%</text>')
    o.append(f'<text x="{sxa(1.0)-4:.1f}" y="{sy(55.4)+22:.1f}" fill="{ink}" font-size="12" text-anchor="end" {halo}>τ = 1: 55.4%</text>')
    for k, col in (("softmax", a1), ("sigmoid", a2)):
        series([(b, *v) for b, v in zip(BS, STEPS[k])], sxb, col)
        series([(b, *v) for b, v in zip(BS, DATA[k])], sxb, col, "6 5")
    lx, ly = P2[0] + 44, sy(24)
    for i, (t, col, dash) in enumerate((("softmax (CLIP), 250 steps", a1, ""), ("sigmoid (SigLIP), 250 steps", a2, ""),
                                        ("softmax, 64,000 pairs", a1, "6 5"), ("sigmoid, 64,000 pairs", a2, "6 5"))):
        y = ly + i * 20
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y-4:.1f}" y2="{y-4:.1f}" stroke="{col}" stroke-width="2.5"{d}/>')
        o.append(f'<text x="{lx+32}" y="{y:.1f}" fill="{ink}" font-size="12" {halo}>{t}</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff'); dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, '#0b5fd6', '#eb6834')); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, '#38bdf8', '#d95926'))
