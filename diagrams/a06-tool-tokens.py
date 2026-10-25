# Generates public/diagrams/a06-tool-tokens.svg and a06-tool-tokens-dark.svg (data: a06-model-context-protocol.md).
# Mean prompt tokens per request (input + cache reads + cache writes) against the number of tools available,
# claude-haiku-5-5, 79 requests: all tools loaded vs the BM25 tool search tool with every tool deferred.
# Data from scratch a06/analyze.py (runs/main.jsonl and runs/stress.jsonl). Usage: python3 diagrams/a06-tool-tokens.py public/diagrams/a06-tool-tokens
import sys

W, H = 780, 470; L, R, T, B = 86, 30, 24, 62
X0, X1 = 0, 500; Y0, Y1 = 0, 60000
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)

ALL = [(5, 1339), (20, 3284), (50, 7167), (100, 13683), (145, 19519), (300, 35810), (486, 55743)]
SEARCH = [(20, 2821), (50, 2875), (100, 2893), (145, 2893), (300, 2949), (486, 2957)]
SERIES = [("All tools loaded", ALL), ("Tool search (BM25), tools deferred", SEARCH)]

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(Y0, Y1 + 1, 10000):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:,}</text>')
    for x in range(0, 501, 100):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Tools available</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Prompt tokens per request</text>')
    for (name, pts), c in zip(SERIES, cols):
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="' + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts) + '"/>')
        for x, y in pts:
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="2"/>')
    x, y = ALL[-1]
    o.append(f'<text x="{sx(x)-10:.1f}" y="{sy(y)-12:.1f}" fill="{ink}" font-size="13" text-anchor="end">55,743</text>')
    x, y = SEARCH[-1]
    o.append(f'<text x="{sx(x)-10:.1f}" y="{sy(y)-12:.1f}" fill="{ink}" font-size="13" text-anchor="end">2,957</text>')
    for k, ((name, _), c) in enumerate(zip(SERIES, cols)):
        y = T + 24 + k * 24
        o.append(f'<circle cx="{L+24}" cy="{y}" r="5" fill="{c}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{L+36}" y="{y+5}" fill="{ink}" font-size="13">{name}</text>')
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5')))
