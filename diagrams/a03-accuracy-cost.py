# Generates public/diagrams/a03-accuracy-cost.svg and a03-accuracy-cost-dark.svg (data: a03-react-planning.md).
# Accuracy against mean cost per run on the 36-task Saltmere set, one point per arm, with 95% Wilson intervals.
# Data from the lab ledger and runs/*.jsonl (scratch a03/analyze.py). Usage: python3 diagrams/a03-accuracy-cost.py public/diagrams/a03-accuracy-cost
import math
import sys

W, H = 780, 470; L, R, T, B = 78, 30, 24, 62
X0, X1 = 1.0, 40.0; Y0, Y1 = 60, 100
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)

# (label, USD per 1,000 runs, accuracy %, CI low, CI high, group, label dx, label dy)
G0, G1, G2 = "No thinking (Haiku 5.5)", "Adaptive thinking (Haiku 5.5)", "Adaptive thinking (Sonnet 5.5)"
ARMS = [
    ("plain loop", 1.31, 76.9, 68.1, 83.8, G0, -76, 5),
    ("ReAct", 1.52, 79.6, 71.1, 86.1, G0, -52, -6),
    ("plan-and-execute", 2.07, 77.8, 69.1, 84.6, G0, 10, 18),
    ("low", 1.88, 88.0, 80.5, 92.8, G1, -34, 5),
    ("medium", 2.14, 88.9, 81.6, 93.5, G1, 10, 18),
    ("high", 3.17, 91.7, 84.9, 95.6, G1, 10, 18),
    ("xhigh", 5.53, 96.3, 90.9, 98.6, G1, 10, 18),
    ("Sonnet medium", 21.2, 97.2, 85.8, 99.5, G2, -112, 5),
]
GROUPS = [G0, G1, G2]

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(Y0, Y1 + 1, 10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for x in (1, 2, 5, 10, 20, 40):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">${x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Cost per 1,000 runs (USD, log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Tasks solved (95% CI)</text>')
    for g, c in zip(GROUPS, cols):
        pts = [a for a in ARMS if a[5] == g]
        if len(pts) > 1 and g != GROUPS[0]:
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="' +
                     " ".join(f"{sx(a[1]):.1f},{sy(a[2]):.1f}" for a in pts) + '"/>')
        for lab, x, y, lo, hi, _, dx, dy in pts:
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="2"/>')
            o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{ink}" font-size="13">{lab}</text>')
    for k, (g, c) in enumerate(zip(GROUPS, cols)):
        y = H - B - 70 + k * 22
        o.append(f'<circle cx="{W-R-280}" cy="{y}" r="5" fill="{c}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{W-R-268}" y="{y+5}" fill="{ink}" font-size="13">{g}</text>')
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6', '#1baf7a')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5', '#199e70')))
