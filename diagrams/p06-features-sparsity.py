# Generates public/diagrams/p06-features-sparsity.svg and -dark.svg (data: p06-superposition.md).
# Usage: python3 diagrams/p06-features-sparsity.py public/diagrams/p06-features-sparsity, then render the PNG with sharp at density 144.
# Data: toy models with n=20 features, m=5 hidden dimensions, importance 0.7^i, trained 10,000 steps (AdamW, lr 1e-3,
# batch 1024) at 13 feature densities from 1 to 0.001, 5 replicates each (lab/toy.py, p06 scripts/toy_runs.py sweep, sweep_linear).
# A feature counts as learned if ||W_i|| >= 0.5, and as in superposition if it is learned and
# sum over other features of (W_i_hat . W_j)^2 > 0.1. Lines: median over replicates; band: min to max.
import math, sys
DATA = {"dens":[1.0,0.5623413324356079,0.3162277638912201,0.17782793939113617,0.10000000149011612,0.05623413249850273,0.03162277489900589,0.017782794311642647,0.009999999776482582,0.005623413249850273,0.003162277629598975,0.0017782794311642647,0.0010000000474974513],"rep":[[5,5,7,7,9,10,10,10,11,12,11,13,14],[5,5,8,9,9,10,10,11,12,13,12,13,14],[5,5,8,9,10,10,11,11,12,13,13,14,15]],"sup":[[0,0,4,4,8,9,10,10,11,12,11,13,14],[0,0,6,8,8,10,10,11,12,13,12,13,14],[0,0,6,8,10,10,11,11,12,13,13,14,15]],"lin":[[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5,5,5,5]]}
W, H = 780, 450; L, R, T, B = 64, 24, 56, 62
X0, X1, Y0, Y1 = 1, 1000, 0, 20
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
def svg(th):
    ink, mut, grid, acc, surf, acc2 = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in (1, 10, 100, 1000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:g}</text>')
    for y in range(0, 21, 5):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">1 / (1 - S): one feature active in this many inputs (log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Features (of 20)</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(5):.1f}" y2="{sy(5):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="2 4"/>')
    o.append(f'<text x="{W-R-4}" y="{sy(5)+18:.1f}" fill="{mut}" font-size="12" text-anchor="end">5 hidden dimensions</text>')
    xs = [1 / d for d in DATA["dens"]]
    def line(key, col, dash=""):
        lo, md, hi = DATA[key]
        band = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, hi)) + " " + " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in reversed(list(zip(xs, lo))))
        o.append(f'<polygon points="{band}" fill="{col}" opacity="0.18"/>')
        pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, md))
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" {dash} points="{pts}"/>')
        for x, v in zip(xs, md):
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{col}"/>')
    line("lin", acc2)
    line("sup", acc, 'stroke-dasharray="7 5"')
    line("rep", acc)
    lg = [(acc, "", "ReLU output model: features learned"), (acc, 'stroke-dasharray="7 5"', "ReLU output model: learned, in superposition"), (acc2, "", "Linear model: features learned")]
    for i, (c, d, t) in enumerate(lg):
        y = T - 38 + (i % 2) * 18 if i < 2 else T - 38
        x = L + 4 if i < 2 else L + 400
        o.append(f'<line x1="{x}" x2="{x+28}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5" {d}/>')
        o.append(f'<text x="{x+36}" y="{y+4}" fill="{ink}" font-size="13">{t}</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff', '#eb6834')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d', '#d95926')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
