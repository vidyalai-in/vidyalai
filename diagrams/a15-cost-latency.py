# Generates public/diagrams/a15-cost-latency.svg and a15-cost-latency-dark.svg (data: a15-agent-cost-latency.md).
# Cost per completed task against median latency per task on the 36-task Saltmere set, one point per arm.
# Data from the lab ledger and runs/*.jsonl (scratch a15/analyze.py); batch-1h includes the 1-hour cache-write
# correction. Usage: python3 diagrams/a15-cost-latency.py public/diagrams/a15-cost-latency
import math
import sys

W, H = 780, 470; L, R, T, B = 78, 30, 24, 62
X0, X1 = 8.0, 3000.0; Y0, Y1 = 1.0, 30.0
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)

# (label, median seconds per task, USD per 1,000 completed tasks, group, label dx, label dy)
G0, G1, G2 = "Haiku 5.5, synchronous", "Sonnet 5.5 (alone or as lead)", "Haiku 5.5, Message Batches"
ARMS = [
    ("thinking off", 12.8, 1.73, G0, -96, 5),
    ("low", 16.9, 2.28, G0, -34, -4),
    ("medium", 18.8, 2.54, G0, 10, -4),
    ("high", 22.7, 3.27, G0, 10, 5),
    ("medium, no cache", 19.1, 4.04, G0, 10, -6),
    ("Sonnet medium", 13.6, 20.88, G1, 12, 5),
    ("Sonnet lead + Haiku workers", 25.2, 10.72, G1, 10, 5),
    ("batch, 5-min cache", 1109, 1.26, G2, -150, 18),
    ("batch, 1-hour cache", 924, 1.50, G2, -150, -8),
]
GROUPS = [G0, G1, G2]

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (1, 2, 5, 10, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">${y}</text>')
    for x, lab in ((10, "10 s"), (30, "30 s"), (100, "100 s"), (300, "5 min"), (1000, "17 min"), (3000, "50 min")):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Median time per task (log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Cost per 1,000 completed tasks (USD, log scale)</text>')
    for g, c in zip(GROUPS, cols):
        for lab, x, y, _, dx, dy in [a for a in ARMS if a[3] == g]:
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="6" fill="{c}" stroke="{surf}" stroke-width="2"/>')
            o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{ink}" font-size="13">{lab}</text>')
    for k, (g, c) in enumerate(zip(GROUPS, cols)):
        y = T + 20 + k * 22
        o.append(f'<circle cx="{W-R-250}" cy="{y}" r="6" fill="{c}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{W-R-238}" y="{y+5}" fill="{ink}" font-size="13">{g}</text>')
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#1baf7a', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#199e70', '#d95926')))
