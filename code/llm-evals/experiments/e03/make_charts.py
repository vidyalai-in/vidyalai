"""Writes the repo chart generators diagrams/e03-*.py with the logged numbers from results.json
frozen into them, so each generator runs standalone (python3 diagrams/e03-x.py public/diagrams/e03-x)."""

import json
from pathlib import Path

R = json.loads((Path(__file__).parent / "results.json").read_text())
REPO = Path("/home/amit/AlpsCode/ai/vidyalai/diagrams")

HEAD = '''import math, sys
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
SER_L = ('#eb6834', '#2a78d6', '#1baf7a', '#8a5cd0')
SER_D = ('#d95926', '#3987e5', '#199e70', '#9b6fe0')
'''

# ---------------------------------------------------------------- chart 1: CI width vs n --------
width = R["exp1"]["width_vs_n"]
c1 = f'''# Generates public/diagrams/e03-ci-width.svg and -dark.svg (data: e03-eval-statistics.md;
# numbers from the Part 3 analysis, results.json "exp1.width_vs_n": claude-haiku-5-5 on MMLU-1140,
# 300 random subsets per size). Usage: python3 diagrams/e03-ci-width.py public/diagrams/e03-ci-width
{HEAD}
DATA = {json.dumps(width)}
W, H = 780, 470; L, R, T, B = 70, 24, 24, 62
X0, X1 = 15, 1500; Y0, Y1 = 0, 18
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
SERIES = [('wilson_1', 'One run, items independent (Wilson)', 0),
          ('clustered_1', 'One run, clustered by subject', 0),
          ('clt_5', 'Mean of 5 runs per item', 1)]
def svg(th, ser):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {{W}} {{H}}" width="{{W}}" height="{{H}}" font-family="Arial,Helvetica,sans-serif" role="img">',
         f'<rect width="{{W}}" height="{{H}}" fill="{{surf}}"/>']
    for x in (20, 50, 100, 200, 400, 1000):
        o.append(f'<line x1="{{sx(x):.1f}}" x2="{{sx(x):.1f}}" y1="{{T}}" y2="{{H-B}}" stroke="{{grid}}"/>')
        o.append(f'<text x="{{sx(x):.1f}}" y="{{H-B+20}}" fill="{{mut}}" font-size="13" text-anchor="middle">{{x:,}}</text>')
    for y in range(0, 19, 3):
        o.append(f'<line x1="{{L}}" x2="{{W-R}}" y1="{{sy(y):.1f}}" y2="{{sy(y):.1f}}" stroke="{{grid}}"/>')
        o.append(f'<text x="{{L-8}}" y="{{sy(y)+4:.1f}}" fill="{{mut}}" font-size="13" text-anchor="end">±{{y}}</text>')
    o.append(f'<text x="{{(L+W-R)/2}}" y="{{H-14}}" fill="{{ink}}" font-size="14" text-anchor="middle">Questions in the eval (log scale)</text>')
    o.append(f'<text transform="translate(20,{{(T+H-B)/2}}) rotate(-90)" fill="{{ink}}" font-size="14" text-anchor="middle">95% interval half-width (points)</text>')
    cols = [ser[0], ser[1], ser[2]]
    dash = ['', '', '7 5']
    for j, (key, label, _) in enumerate(SERIES):
        pts = ' '.join(f'{{sx(d["n"]):.1f}},{{sy(100*d[key]):.1f}}' for d in DATA)
        o.append(f'<polyline fill="none" stroke="{{cols[j]}}" stroke-width="2.5" stroke-dasharray="{{dash[j]}}" points="{{pts}}"/>')
        for d in DATA:
            o.append(f'<circle cx="{{sx(d["n"]):.1f}}" cy="{{sy(100*d[key]):.1f}}" r="4" fill="{{cols[j]}}" stroke="{{surf}}" stroke-width="1.5"/>')
        ly = T + 14 + 22 * j
        o.append(f'<line x1="{{W-R-300}}" x2="{{W-R-272}}" y1="{{ly}}" y2="{{ly}}" stroke="{{cols[j]}}" stroke-width="3" stroke-dasharray="{{dash[j]}}"/>')
        o.append(f'<text x="{{W-R-264}}" y="{{ly+4}}" fill="{{ink}}" font-size="13">{{label}}</text>')
    last = DATA[-1]
    o.append(f'<text x="{{sx(last["n"])-6:.1f}}" y="{{sy(100*last["clustered_1"])-10:.1f}}" fill="{{ink}}" font-size="13" text-anchor="end">±{{100*last["clustered_1"]:.1f}}</text>')
    o.append(f'<text x="{{sx(last["n"])-6:.1f}}" y="{{sy(100*last["wilson_1"])+20:.1f}}" fill="{{ink}}" font-size="13" text-anchor="end">±{{100*last["wilson_1"]:.1f}}</text>')
    first = DATA[0]
    o.append(f'<text x="{{sx(first["n"])+8:.1f}}" y="{{sy(100*first["clustered_1"])-8:.1f}}" fill="{{ink}}" font-size="13">±{{100*first["wilson_1"]:.1f}} to ±{{100*first["clustered_1"]:.1f}}</text>')
    o.append('</svg>')
    return '\\n'.join(o)
open(sys.argv[1] + '.svg', 'w').write(svg(light, SER_L)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, SER_D))
'''
(REPO / "e03-ci-width.py").write_text(c1)

# ---------------------------------------------------------------- chart 2: paired vs unpaired ---
rows = []
for key, label in [("compare_off_vs_low", "Haiku: thinking off → adaptive, low"),
                   ("compare_haiku_vs_sonnet", "Haiku 5.5 → Sonnet 5.5, both thinking off"),
                   ("compare_low_vs_sonnet", "Haiku 5.5 thinking low → Sonnet 5.5 thinking off")]:
    s = R[key]["single"]
    rows.append(dict(label=label, diff=s["diff"], unpaired=s["ci_unpaired"], paired=s["ci_paired"],
                     clustered=s["ci_paired_clustered"]))
c2 = f'''# Generates public/diagrams/e03-paired.svg and -dark.svg (data: e03-eval-statistics.md;
# numbers from results.json "compare_*.single": one sample per item, 1,140 MMLU items).
# Usage: python3 diagrams/e03-paired.py public/diagrams/e03-paired
{HEAD}
DATA = {json.dumps(rows, ensure_ascii=False)}
W, H = 780, 520; L, R, T, B = 24, 24, 20, 62
X0, X1 = -8, 16; PW = 480
def sx(x): return L + 10 + (x - X0) / (X1 - X0) * (PW - 20)
KINDS = [('unpaired', 'Unpaired (as if different questions)', 0), ('paired', 'Paired by question', 1),
         ('clustered', 'Paired, clustered (subject)', 2)]
def svg(th, ser):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {{W}} {{H}}" width="{{W}}" height="{{H}}" font-family="Arial,Helvetica,sans-serif" role="img">',
         f'<rect width="{{W}}" height="{{H}}" fill="{{surf}}"/>']
    band = (H - B - T) / len(DATA)
    for x in range(-8, 17, 4):
        for i in range(len(DATA)):
            ya, yb = T + i * band + 34, T + (i + 1) * band - 8
            o.append(f'<line x1="{{sx(x):.1f}}" x2="{{sx(x):.1f}}" y1="{{ya:.1f}}" y2="{{yb:.1f}}" stroke="{{grid if x else mut}}" stroke-width="{{1 if x else 2}}"/>')
        o.append(f'<text x="{{sx(x):.1f}}" y="{{H-B+20}}" fill="{{mut}}" font-size="13" text-anchor="middle">{{x:+d}}</text>' if x else f'<text x="{{sx(x):.1f}}" y="{{H-B+20}}" fill="{{mut}}" font-size="13" text-anchor="middle">0</text>')
    o.append(f'<text x="{{L+PW/2}}" y="{{H-14}}" fill="{{ink}}" font-size="14" text-anchor="middle">Accuracy difference (points), 95% interval</text>')
    for i, d in enumerate(DATA):
        y0 = T + i * band
        o.append(f'<text x="{{L+PW+14}}" y="{{y0+22:.1f}}" fill="{{ink}}" font-size="13">Difference: {{100*d["diff"]:+.1f}} points</text>')
        o.append(f'<text x="{{L+10}}" y="{{y0+22:.1f}}" fill="{{ink}}" font-size="14" font-weight="bold">{{d["label"]}}</text>')
        for j, (k, lab, c) in enumerate(KINDS):
            y = y0 + 48 + j * 26
            lo, hi = 100 * d[k][0], 100 * d[k][1]
            col = ser[c]
            o.append(f'<line x1="{{sx(lo):.1f}}" x2="{{sx(hi):.1f}}" y1="{{y:.1f}}" y2="{{y:.1f}}" stroke="{{col}}" stroke-width="4" stroke-linecap="round"/>')
            o.append(f'<circle cx="{{sx(100*d["diff"]):.1f}}" cy="{{y:.1f}}" r="5.5" fill="{{col}}" stroke="{{surf}}" stroke-width="2"/>')
            o.append(f'<text x="{{L+PW+14}}" y="{{y+4:.1f}}" fill="{{col}}" font-size="13">{{lab.split(" (")[0]}}: {{lo:+.1f}} to {{hi:+.1f}}</text>')
    o.append('</svg>')
    return '\\n'.join(o)
open(sys.argv[1] + '.svg', 'w').write(svg(light, SER_L)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, SER_D))
'''
(REPO / "e03-paired.py").write_text(c2)

# ---------------------------------------------------------------- chart 3: power vs n -----------
pw = R["compare_low_vs_sonnet"]["power"]
c3 = f'''# Generates public/diagrams/e03-power.svg and -dark.svg (data: e03-eval-statistics.md;
# numbers from results.json "compare_low_vs_sonnet.power.curve": share of 1,500 simulated
# experiments per point, built by resampling the recorded answers, that find Haiku 5.5 (adaptive
# thinking, low) better than Sonnet 5.5 (thinking off) at the 5% level).
# Usage: python3 diagrams/e03-power.py public/diagrams/e03-power
{HEAD}
DATA = {json.dumps(pw["curve"])}
NSIM = 1500
DELTA = {pw["delta"]}
W, H = 780, 470; L, R, T, B = 70, 24, 24, 62
X0, X1 = 40, 5000
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - y / 100 * (H - T - B)
def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def svg(th, ser):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {{W}} {{H}}" width="{{W}}" height="{{H}}" font-family="Arial,Helvetica,sans-serif" role="img">',
         f'<rect width="{{W}}" height="{{H}}" fill="{{surf}}"/>']
    for x in (50, 100, 200, 500, 1000, 2000, 4000):
        o.append(f'<line x1="{{sx(x):.1f}}" x2="{{sx(x):.1f}}" y1="{{T}}" y2="{{H-B}}" stroke="{{grid}}"/>')
        o.append(f'<text x="{{sx(x):.1f}}" y="{{H-B+20}}" fill="{{mut}}" font-size="13" text-anchor="middle">{{x:,}}</text>')
    for y in range(0, 101, 20):
        o.append(f'<line x1="{{L}}" x2="{{W-R}}" y1="{{sy(y):.1f}}" y2="{{sy(y):.1f}}" stroke="{{grid}}"/>')
        o.append(f'<text x="{{L-8}}" y="{{sy(y)+4:.1f}}" fill="{{mut}}" font-size="13" text-anchor="end">{{y}}%</text>')
    o.append(f'<line x1="{{L}}" x2="{{W-R}}" y1="{{sy(80):.1f}}" y2="{{sy(80):.1f}}" stroke="{{mut}}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{{L+8}}" y="{{sy(80)-7:.1f}}" fill="{{mut}}" font-size="13">80% power</text>')
    o.append(f'<text x="{{(L+W-R)/2}}" y="{{H-14}}" fill="{{ink}}" font-size="14" text-anchor="middle">Questions per system (log scale), one answer each</text>')
    o.append(f'<text transform="translate(20,{{(T+H-B)/2}}) rotate(-90)" fill="{{ink}}" font-size="14" text-anchor="middle">Chance of detecting the gap</text>')
    for j, (key, label) in enumerate([('paired', 'Paired: both on the same questions'), ('unpaired', 'Unpaired: separate questions')]):
        col = ser[1 - j]
        pts = ' '.join(f'{{sx(d["n"]):.1f}},{{sy(100*d[key]):.1f}}' for d in DATA)
        o.append(f'<polyline fill="none" stroke="{{col}}" stroke-width="2.5" points="{{pts}}"/>')
        for d in DATA:
            lo, hi = wilson(d[key], NSIM)
            x = sx(d["n"])
            o.append(f'<line x1="{{x:.1f}}" x2="{{x:.1f}}" y1="{{sy(100*lo):.1f}}" y2="{{sy(100*hi):.1f}}" stroke="{{col}}" stroke-width="2"/>')
            o.append(f'<circle cx="{{x:.1f}}" cy="{{sy(100*d[key]):.1f}}" r="4" fill="{{col}}" stroke="{{surf}}" stroke-width="1.5"/>')
        ly = sy(32) + 22 * j
        o.append(f'<line x1="{{W-R-290}}" x2="{{W-R-262}}" y1="{{ly:.1f}}" y2="{{ly:.1f}}" stroke="{{col}}" stroke-width="3"/>')
        o.append(f'<text x="{{W-R-254}}" y="{{ly+4:.1f}}" fill="{{ink}}" font-size="13">{{label}}</text>')
    o.append(f'<text x="{{W-R-290}}" y="{{sy(32)-22:.1f}}" fill="{{mut}}" font-size="13">Gap: {{100*DELTA:.1f}} points</text>')
    o.append('</svg>')
    return '\\n'.join(o)
open(sys.argv[1] + '.svg', 'w').write(svg(light, SER_L)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, SER_D))
'''
(REPO / "e03-power.py").write_text(c3)
print("wrote 3 generators")
