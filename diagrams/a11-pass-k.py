# Generates public/diagrams/a11-pass-k.svg and a11-pass-k-dark.svg (data: a11-agent-evals.md).
# pass@k (solid) and pass^k (dashed) against k for each configuration, unbiased estimators averaged
# over tasks, from n = 10 runs per task (Sonnet: 5). Data from scratch a11/analyze.py.
# Usage: python3 diagrams/a11-pass-k.py public/diagrams/a11-pass-k
import sys

W, H = 780, 470; L, R, T, B = 70, 30, 24, 62
X0, X1 = 1, 10; Y0, Y1 = 10, 100
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)

# (label, pass@k for k = 1.., pass^k for k = 1..)
CURVES = [
    ("Questions, thinking off (Haiku)",
     [77.22, 88.95, 93.10, 95.01, 96.05, 96.65, 96.99, 97.16, 97.22, 97.22],
     [77.22, 65.49, 57.92, 52.58, 48.61, 45.58, 43.24, 41.42, 40.00, 38.89]),
    ("Questions, thinking low (Haiku)",
     [86.94, 93.77, 95.49, 96.20, 96.59, 96.85, 97.04, 97.16, 97.22, 97.22],
     [86.94, 80.12, 75.02, 70.93, 67.50, 64.56, 61.97, 59.63, 57.50, 55.56]),
    ("Coding, title only (Haiku)",
     [85.00, 90.00, 92.26, 93.52, 94.32, 94.88, 95.33, 95.71, 96.07, 96.43],
     [85.00, 80.00, 77.26, 75.53, 74.33, 73.45, 72.77, 72.22, 71.79, 71.43]),
    ("Coding, title only (Sonnet)",
     [90.00, 92.50, 92.86, 92.86, 92.86],
     [90.00, 87.50, 85.36, 83.57, 82.14]),
]

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(Y0, Y1 + 1, 10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for x in range(X0, X1 + 1):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">k (number of runs of each task)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Share of tasks</text>')
    for (lab, at, hat), c in zip(CURVES, cols):
        for vals, dash in ((at, ""), (hat, ' stroke-dasharray="7 5"')):
            pts = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in enumerate(vals, 1))
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5"{dash} points="{pts}"/>')
            for k, v in enumerate(vals, 1):
                o.append(f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
    lx, ly = L + 16, H - B - 112
    o.append(f'<rect x="{lx-10}" y="{ly-18}" width="300" height="128" fill="{surf}" stroke="{grid}"/>')
    for i, ((lab, _, _), c) in enumerate(zip(CURVES, cols)):
        y = ly + i * 21
        o.append(f'<line x1="{lx}" x2="{lx+24}" y1="{y-4}" y2="{y-4}" stroke="{c}" stroke-width="3"/>')
        o.append(f'<text x="{lx+32}" y="{y+1}" fill="{ink}" font-size="13">{lab}</text>')
    y = ly + 4 * 21
    o.append(f'<line x1="{lx}" x2="{lx+24}" y1="{y-4}" y2="{y-4}" stroke="{mut}" stroke-width="2.5"/>')
    o.append(f'<text x="{lx+32}" y="{y+1}" fill="{ink}" font-size="13">pass@k</text>')
    o.append(f'<line x1="{lx+110}" x2="{lx+134}" y1="{y-4}" y2="{y-4}" stroke="{mut}" stroke-width="2.5" stroke-dasharray="7 5"/>')
    o.append(f'<text x="{lx+142}" y="{y+1}" fill="{ink}" font-size="13">pass^k</text>')
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6', '#1baf7a', '#8a5cd0')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5', '#199e70', '#9b6fe0')))
