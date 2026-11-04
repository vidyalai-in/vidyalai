# Generates public/diagrams/e03-paired.svg and -dark.svg (data: e03-eval-statistics.md;
# numbers from results.json "compare_*.single": one sample per item, 1,140 MMLU items).
# Usage: python3 diagrams/e03-paired.py public/diagrams/e03-paired
import math, sys
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
SER_L = ('#eb6834', '#2a78d6', '#1baf7a', '#8a5cd0')
SER_D = ('#d95926', '#3987e5', '#199e70', '#9b6fe0')

DATA = [{"label": "Haiku: thinking off → adaptive, low", "diff": 0.1, "unpaired": [0.0708, 0.1292], "paired": [0.0789, 0.1211], "clustered": [0.0589, 0.1411]}, {"label": "Haiku 5.5 → Sonnet 5.5, both thinking off", "diff": 0.0693, "unpaired": [0.0387, 0.0999], "paired": [0.0497, 0.0889], "clustered": [0.0508, 0.0878]}, {"label": "Haiku 5.5 thinking low → Sonnet 5.5 thinking off", "diff": -0.0307, "unpaired": [-0.0572, -0.0042], "paired": [-0.0519, -0.0095], "clustered": [-0.0702, 0.0088]}]
W, H = 780, 520; L, R, T, B = 24, 24, 20, 62
X0, X1 = -8, 16; PW = 480
def sx(x): return L + 10 + (x - X0) / (X1 - X0) * (PW - 20)
KINDS = [('unpaired', 'Unpaired (as if different questions)', 0), ('paired', 'Paired by question', 1),
         ('clustered', 'Paired, clustered (subject)', 2)]
def svg(th, ser):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">',
         f'<rect width="{W}" height="{H}" fill="{surf}"/>']
    band = (H - B - T) / len(DATA)
    for x in range(-8, 17, 4):
        for i in range(len(DATA)):
            ya, yb = T + i * band + 34, T + (i + 1) * band - 8
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{ya:.1f}" y2="{yb:.1f}" stroke="{grid if x else mut}" stroke-width="{1 if x else 2}"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:+d}</text>' if x else f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">0</text>')
    o.append(f'<text x="{L+PW/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Accuracy difference (points), 95% interval</text>')
    for i, d in enumerate(DATA):
        y0 = T + i * band
        o.append(f'<text x="{L+PW+14}" y="{y0+22:.1f}" fill="{ink}" font-size="13">Difference: {100*d["diff"]:+.1f} points</text>')
        o.append(f'<text x="{L+10}" y="{y0+22:.1f}" fill="{ink}" font-size="14" font-weight="bold">{d["label"]}</text>')
        for j, (k, lab, c) in enumerate(KINDS):
            y = y0 + 48 + j * 26
            lo, hi = 100 * d[k][0], 100 * d[k][1]
            col = ser[c]
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="4" stroke-linecap="round"/>')
            o.append(f'<circle cx="{sx(100*d["diff"]):.1f}" cy="{y:.1f}" r="5.5" fill="{col}" stroke="{surf}" stroke-width="2"/>')
            o.append(f'<text x="{L+PW+14}" y="{y+4:.1f}" fill="{col}" font-size="13">{lab.split(" (")[0]}: {lo:+.1f} to {hi:+.1f}</text>')
    o.append('</svg>')
    return '\n'.join(o)
open(sys.argv[1] + '.svg', 'w').write(svg(light, SER_L)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, SER_D))
