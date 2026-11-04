# Generates public/diagrams/e03-ci-width.svg and -dark.svg (data: e03-eval-statistics.md;
# numbers from the Part 3 analysis, results.json "exp1.width_vs_n": claude-haiku-5-5 on MMLU-1140,
# 300 random subsets per size). Usage: python3 diagrams/e03-ci-width.py public/diagrams/e03-ci-width
import math, sys
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
SER_L = ('#eb6834', '#2a78d6', '#1baf7a', '#8a5cd0')
SER_D = ('#d95926', '#3987e5', '#199e70', '#9b6fe0')

DATA = [{"n": 20, "wilson_1": 0.1611, "clt_5": 0.1645, "clustered_1": 0.1693}, {"n": 50, "wilson_1": 0.1079, "clt_5": 0.108, "clustered_1": 0.1136}, {"n": 100, "wilson_1": 0.0768, "clt_5": 0.0759, "clustered_1": 0.0845}, {"n": 200, "wilson_1": 0.0547, "clt_5": 0.0537, "clustered_1": 0.065}, {"n": 400, "wilson_1": 0.0389, "clt_5": 0.0381, "clustered_1": 0.052}, {"n": 800, "wilson_1": 0.0276, "clt_5": 0.0269, "clustered_1": 0.0445}, {"n": 1140, "wilson_1": 0.0232, "clt_5": 0.0226, "clustered_1": 0.0416}]
W, H = 780, 470; L, R, T, B = 70, 24, 24, 62
X0, X1 = 15, 1500; Y0, Y1 = 0, 18
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
SERIES = [('wilson_1', 'One run, items independent (Wilson)', 0),
          ('clustered_1', 'One run, clustered by subject', 0),
          ('clt_5', 'Mean of 5 runs per item', 1)]
def svg(th, ser):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">',
         f'<rect width="{W}" height="{H}" fill="{surf}"/>']
    for x in (20, 50, 100, 200, 400, 1000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:,}</text>')
    for y in range(0, 19, 3):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">±{y}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Questions in the eval (log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">95% interval half-width (points)</text>')
    cols = [ser[0], ser[1], ser[2]]
    dash = ['', '', '7 5']
    for j, (key, label, _) in enumerate(SERIES):
        pts = ' '.join(f'{sx(d["n"]):.1f},{sy(100*d[key]):.1f}' for d in DATA)
        o.append(f'<polyline fill="none" stroke="{cols[j]}" stroke-width="2.5" stroke-dasharray="{dash[j]}" points="{pts}"/>')
        for d in DATA:
            o.append(f'<circle cx="{sx(d["n"]):.1f}" cy="{sy(100*d[key]):.1f}" r="4" fill="{cols[j]}" stroke="{surf}" stroke-width="1.5"/>')
        ly = T + 14 + 22 * j
        o.append(f'<line x1="{W-R-300}" x2="{W-R-272}" y1="{ly}" y2="{ly}" stroke="{cols[j]}" stroke-width="3" stroke-dasharray="{dash[j]}"/>')
        o.append(f'<text x="{W-R-264}" y="{ly+4}" fill="{ink}" font-size="13">{label}</text>')
    last = DATA[-1]
    o.append(f'<text x="{sx(last["n"])-6:.1f}" y="{sy(100*last["clustered_1"])-10:.1f}" fill="{ink}" font-size="13" text-anchor="end">±{100*last["clustered_1"]:.1f}</text>')
    o.append(f'<text x="{sx(last["n"])-6:.1f}" y="{sy(100*last["wilson_1"])+20:.1f}" fill="{ink}" font-size="13" text-anchor="end">±{100*last["wilson_1"]:.1f}</text>')
    first = DATA[0]
    o.append(f'<text x="{sx(first["n"])+8:.1f}" y="{sy(100*first["clustered_1"])-8:.1f}" fill="{ink}" font-size="13">±{100*first["wilson_1"]:.1f} to ±{100*first["clustered_1"]:.1f}</text>')
    o.append('</svg>')
    return '\n'.join(o)
open(sys.argv[1] + '.svg', 'w').write(svg(light, SER_L)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, SER_D))
