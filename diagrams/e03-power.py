# Generates public/diagrams/e03-power.svg and -dark.svg (data: e03-eval-statistics.md;
# numbers from results.json "compare_low_vs_sonnet.power.curve": share of 1,500 simulated
# experiments per point, built by resampling the recorded answers, that find Haiku 5.5 (adaptive
# thinking, low) better than Sonnet 5.5 (thinking off) at the 5% level).
# Usage: python3 diagrams/e03-power.py public/diagrams/e03-power
import math, sys
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
SER_L = ('#eb6834', '#2a78d6', '#1baf7a', '#8a5cd0')
SER_D = ('#d95926', '#3987e5', '#199e70', '#9b6fe0')

DATA = [{"n": 50, "paired": 0.072, "unpaired": 0.0767}, {"n": 100, "paired": 0.1313, "unpaired": 0.086}, {"n": 200, "paired": 0.228, "unpaired": 0.148}, {"n": 400, "paired": 0.3853, "unpaired": 0.2553}, {"n": 800, "paired": 0.6393, "unpaired": 0.4793}, {"n": 1140, "paired": 0.8073, "unpaired": 0.5973}, {"n": 2000, "paired": 0.9613, "unpaired": 0.856}, {"n": 4000, "paired": 0.9993, "unpaired": 0.992}]
NSIM = 1500
DELTA = 0.03
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
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">',
         f'<rect width="{W}" height="{H}" fill="{surf}"/>']
    for x in (50, 100, 200, 500, 1000, 2000, 4000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:,}</text>')
    for y in range(0, 101, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(80):.1f}" y2="{sy(80):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{L+8}" y="{sy(80)-7:.1f}" fill="{mut}" font-size="13">80% power</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Questions per system (log scale), one answer each</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Chance of detecting the gap</text>')
    for j, (key, label) in enumerate([('paired', 'Paired: both on the same questions'), ('unpaired', 'Unpaired: separate questions')]):
        col = ser[1 - j]
        pts = ' '.join(f'{sx(d["n"]):.1f},{sy(100*d[key]):.1f}' for d in DATA)
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" points="{pts}"/>')
        for d in DATA:
            lo, hi = wilson(d[key], NSIM)
            x = sx(d["n"])
            o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(100*lo):.1f}" y2="{sy(100*hi):.1f}" stroke="{col}" stroke-width="2"/>')
            o.append(f'<circle cx="{x:.1f}" cy="{sy(100*d[key]):.1f}" r="4" fill="{col}" stroke="{surf}" stroke-width="1.5"/>')
        ly = sy(32) + 22 * j
        o.append(f'<line x1="{W-R-290}" x2="{W-R-262}" y1="{ly:.1f}" y2="{ly:.1f}" stroke="{col}" stroke-width="3"/>')
        o.append(f'<text x="{W-R-254}" y="{ly+4:.1f}" fill="{ink}" font-size="13">{label}</text>')
    o.append(f'<text x="{W-R-290}" y="{sy(32)-22:.1f}" fill="{mut}" font-size="13">Gap: {100*DELTA:.1f} points</text>')
    o.append('</svg>')
    return '\n'.join(o)
open(sys.argv[1] + '.svg', 'w').write(svg(light, SER_L)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, SER_D))
