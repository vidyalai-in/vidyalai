# Generates public/diagrams/t03-edu-scores.svg and t03-edu-scores-dark.svg (data: t03-pretraining-data-pipeline.md).
# Usage: python3 diagrams/t03-edu-scores.py public/diagrams/t03-edu-scores, then render the PNG with sharp at density 144.
# Educational-quality scores (the `score` column of FineWeb-Edu sample-10BT) of the 2,424,000 documents downloaded in
# Part 3 (every 4th row group of all 14 files), counted in bins of 1/16 by stats.py. Seven documents above 5.1875 are omitted.
# FineWeb-Edu keeps documents whose score rounds to 3 or more, so nothing below 2.5 exists.
import sys
W, H = 780, 470; L, R, T, B = 78, 24, 24, 62
X0, X1 = 2.5, 5.0; Y0, Y1 = 0, 225
COUNTS = [176562, 219002, 200803, 185169, 171002, 156639, 144122, 132697, 121644, 111563, 101492, 92438, 84054, 75571, 67761, 60634, 53887, 47389, 41033, 35217, 30353, 25659, 20821, 17149, 11972, 10839, 8293, 6167, 4496, 3296, 2282, 1580, 971, 629, 356, 215, 134, 54, 29, 18, 2, 2, 2]
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
def svg(th):
    ink, mut, grid, acc, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0, 50, 100, 150, 200):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}K</text>')
    for i, c in enumerate(COUNTS):
        x = 2.5 + i / 16
        if x >= X1: break
        o.append(f'<rect x="{sx(x)+1:.1f}" y="{sy(c/1000):.1f}" width="{sx(x+1/16)-sx(x)-2:.1f}" height="{sy(0)-sy(c/1000):.1f}" fill="{acc}"/>')
    for x in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:.1f}</text>')
    for x in (3.5, 4.5):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{sx(3.0):.1f}" y="{sy(212):.1f}" fill="{ink}" font-size="14" text-anchor="middle">Rounds to 3: 86.7% of docs</text>')
    o.append(f'<text x="{sx(4.0):.1f}" y="{sy(120):.1f}" fill="{ink}" font-size="14" text-anchor="middle">Rounds to 4: 13.2%</text>')
    o.append(f'<text x="{sx(4.75):.1f}" y="{sy(60):.1f}" fill="{ink}" font-size="14" text-anchor="middle">5: 0.08%</text>')
    o.append(f'<text x="{sx(3.0):.1f}" y="{sy(192):.1f}" fill="{mut}" font-size="13" text-anchor="middle">median 2.91</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Educational-quality score from the FineWeb-Edu classifier</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Documents per 1/16 bin</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
