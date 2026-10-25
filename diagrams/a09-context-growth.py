# Generates public/diagrams/a09-context-growth.svg and a09-context-growth-dark.svg (data: a09-computer-use-agents.md).
# Mean tokens the model read per step (input + cache reads + cache writes) on the 10 form-filling tasks,
# claude-haiku-5-5, effort medium, 50 runs per arm (5 repeats); a step's mean covers the runs that reached it.
# Data from scratch a09/results.jsonl. Usage: python3 diagrams/a09-context-growth.py public/diagrams/a09-context-growth
import sys

W, H = 780, 470; L, R, T, B = 86, 30, 24, 62
X0, X1 = 1, 10; Y0, Y1 = 0, 20000
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)

SERIES = [
    ("Pixels (computer toolset)", [4792, 6235, 7731, 9338, 10954, 12507, 13968, 15535, 17300, 19064]),
    ("Pixels + page (browser toolset)", [6413, 7960, 9174, 10143, 11808, 13372, 14917, 16128, 17663, 18712]),
    ("Page text only (browser toolset, no screenshots)", [6081, 6188, 6598, 7361, 8866, 9462, 10727, 11495, 12616, 13831]),
]

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(Y0, Y1 + 1, 5000):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:,}</text>')
    for x in range(1, 11):
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Agent step (model call)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Tokens read this step</text>')
    for k, ((name, ys), c) in enumerate(zip(SERIES, cols)):
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="' + " ".join(f"{sx(i+1):.1f},{sy(v):.1f}" for i, v in enumerate(ys)) + '"/>')
        for i, v in enumerate(ys):
            o.append(f'<circle cx="{sx(i+1):.1f}" cy="{sy(v):.1f}" r="4" fill="{c}" stroke="{surf}" stroke-width="2"/>')
        dy = 22 if k == 1 else -10  # keep the two top labels apart
        o.append(f'<text x="{sx(10)-8:.1f}" y="{sy(ys[-1])+dy:.1f}" fill="{ink}" font-size="13" text-anchor="end">{ys[-1]:,}</text>')
    for k, ((name, _), c) in enumerate(zip(SERIES, cols)):
        y = T + 24 + k * 24
        o.append(f'<circle cx="{L+24}" cy="{y}" r="5" fill="{c}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{L+36}" y="{y+5}" fill="{ink}" font-size="13">{name}</text>')
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#8a5cd6', '#2a78d6')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#9b72e0', '#3987e5')))
