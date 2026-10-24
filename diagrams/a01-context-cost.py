# Generates public/diagrams/a01-context-cost.svg and a01-context-cost-dark.svg (data: a01-what-is-an-ai-agent.md).
# Cumulative cost of one audit task (claude-haiku-5-5, effort medium) by agent step, mean of 3-5 runs per arm,
# from the lab ledger. Usage: python3 diagrams/a01-context-cost.py public/diagrams/a01-context-cost
import sys
W, H = 780, 450; L, R, T, B = 78, 150, 24, 62
X0, X1 = 1, 18; Y0, Y1 = 0, 6
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
# Cumulative USD per 1,000 runs after each step.
SERIES = [
    ("One tool call per step, no cache", [0.143, 0.289, 0.444, 0.615, 0.796, 0.998, 1.213, 1.448, 1.702, 1.974, 2.266, 2.577, 2.907, 3.259, 3.63, 4.025, 4.441, 5.46], "$5.46"),
    ("One tool call per step, cached", [0.04, 0.085, 0.132, 0.189, 0.237, 0.295, 0.357, 0.421, 0.486, 0.553, 0.624, 0.695, 0.767, 0.843, 0.92, 1.001, 1.084, 1.642], "$1.64"),
    ("Parallel tool calls, cached", [0.1, 0.241, 0.758, 1.598], "$1.60"),
]
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(0, 7):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">${y}</text>')
    for x in (1, 3, 6, 9, 12, 15, 18):
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="{mut}" stroke-width="1"/>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Agent step (model call)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Cumulative cost per 1,000 runs (USD)</text>')
    for (name, ys, lab), c in zip(SERIES, cols):
        pts = " ".join(f"{sx(i+1):.1f},{sy(v):.1f}" for i, v in enumerate(ys))
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"/>')
        for i, v in enumerate(ys):
            o.append(f'<circle cx="{sx(i+1):.1f}" cy="{sy(v):.1f}" r="4" fill="{c}" stroke="{surf}" stroke-width="2"/>')
        xe, ye = sx(len(ys)), sy(ys[-1])
        o.append(f'<text x="{xe+10:.1f}" y="{ye+5:.1f}" fill="{ink}" font-size="13" font-weight="bold">{lab}</text>')
    # legend
    for k, ((name, _, _), c) in enumerate(zip(SERIES, cols)):
        y = T + 14 + k * 22
        o.append(f'<line x1="{L+14}" x2="{L+38}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2"/>')
        o.append(f'<circle cx="{L+26}" cy="{y}" r="4" fill="{c}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{L+46}" y="{y+5}" fill="{ink}" font-size="13">{name}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6', '#1baf7a')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5', '#199e70')))
