# Generates public/diagrams/p06-dimensionality.svg and -dark.svg (data: p06-superposition.md).
# Usage: python3 diagrams/p06-dimensionality.py public/diagrams/p06-dimensionality, then render the PNG with sharp at density 144.
# Data: ReLU output toy models with n=400 equally important features in m=30 dimensions, 30 densities from 1 to 0.001,
# 3 replicates each, 10,000 AdamW steps (lab/toy.py, p06 scripts/toy_runs.py geometry).
# Dots: each feature's dimensionality D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2 (rounded to 0.005 and de-duplicated per
# density). Line: dimensions per feature, m / ||W||_F^2, mean of the 3 replicates.
import math, sys
DATA = __DATA__
W, H = 780, 470; L, R, T, B = 64, 120, 24, 62
X0, X1, Y0, Y1 = 0.85, 1200, 0, 1.08
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
GUIDES = [(1, "1"), (3/4, "3/4  tetrahedron"), (2/3, "2/3  triangle"), (1/2, "1/2  antipodal pair"), (2/5, "2/5  pentagon"), (3/8, "3/8  square antiprism")]
def svg(th):
    ink, mut, grid, acc, surf, acc2 = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in (1, 10, 100, 1000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:g}</text>')
    for y in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:g}</text>')
    for v, lab in GUIDES:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{mut}" stroke-width="1" stroke-dasharray="3 4" opacity="0.7"/>')
    lab_y = {1: 0, 3/4: 0, 2/3: 0, 1/2: 0, 2/5: -3, 3/8: 9}
    for v, lab in GUIDES:
        o.append(f'<text x="{W-R+6}" y="{sy(v)+4+lab_y[v]:.1f}" fill="{mut}" font-size="11">{lab}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">1 / (1 - S), log scale</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Dimensions per feature</text>')
    for x, y in DATA["pts"]:
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4" fill="{acc}" opacity="0.6"/>')
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in DATA["dstar"])
    o.append(f'<polyline fill="none" stroke="{acc2}" stroke-width="2.5" points="{pts}"/>')
    lx, ly = sx(30), sy(0.9)
    o.append(f'<rect x="{lx-12}" y="{ly-16}" width="250" height="50" fill="{surf}"/>')
    o.append(f'<circle cx="{lx+10}" cy="{ly}" r="4" fill="{acc}" opacity="0.6"/><text x="{lx+24}" y="{ly+4}" fill="{ink}" font-size="13">one feature (its dimensionality)</text>')
    o.append(f'<line x1="{lx}" x2="{lx+20}" y1="{ly+22}" y2="{ly+22}" stroke="{acc2}" stroke-width="2.5"/><text x="{lx+24}" y="{ly+26}" fill="{ink}" font-size="13">model average, m / ||W||²</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff', '#eb6834')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d', '#d95926')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
