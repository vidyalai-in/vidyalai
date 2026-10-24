# Generates public/diagrams/t03-lsh-scurve.svg and t03-lsh-scurve-dark.svg (data: t03-pretraining-data-pipeline.md).
# Usage: python3 diagrams/t03-lsh-scurve.py public/diagrams/t03-lsh-scurve, then render the PNG with sharp at density 144.
# Measured by scurve.py: 600 held-out FineWeb-Edu documents, 20 edited copies each (12,000 pairs), exact 5-gram
# Jaccard similarity per pair, binned by 0.05; a pair counts as matched if all 8 MinHash values agree in any of 14 bands.
# Filled points: data.py's hash functions, mix64(shingle XOR seed). Hollow points: a first attempt with
# h(x) = (a*x + b) mod (2^61 - 1), a, b, x < 2^32, whose 112 "independent" hashes were correlated.
# Bars: Wilson 95% intervals. Line: 1 - (1 - s^8)^14.
import sys
W, H = 780, 470; L, R, T, B = 70, 24, 24, 62
X0, X1 = 0.3, 1.0; Y0, Y1 = 0.0, 1.0
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
good = [(0.333, 0.010, 0.003, 0.029), (0.377, 0.003, 0.001, 0.009), (0.425, 0.013, 0.008, 0.020), (0.474, 0.038, 0.029, 0.050),
        (0.525, 0.083, 0.068, 0.100), (0.575, 0.175, 0.153, 0.198), (0.625, 0.258, 0.231, 0.286), (0.675, 0.450, 0.418, 0.483),
        (0.724, 0.657, 0.623, 0.690), (0.775, 0.847, 0.817, 0.872), (0.825, 0.966, 0.949, 0.978), (0.875, 0.998, 0.990, 1.000),
        (0.925, 1.000, 0.993, 1.000), (0.971, 1.000, 0.991, 1.000)]
bad = [(0.333, 0.061), (0.377, 0.094), (0.425, 0.113), (0.474, 0.163), (0.525, 0.216), (0.575, 0.316), (0.625, 0.419),
       (0.675, 0.555), (0.724, 0.698), (0.775, 0.806), (0.825, 0.875), (0.875, 0.932), (0.925, 0.959), (0.971, 0.993)]
def theory(s): return 1 - (1 - s ** 8) ** 14
def svg(th):
    ink, mut, grid, acc, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for i in range(8):
        x = 0.3 + 0.1 * i
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:.1f}</text>')
    for y in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{int(y*100)}%</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Jaccard similarity of the two documents\' 5-word shingles</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Pairs flagged as duplicates</text>')
    pts = " ".join(f"{sx(X0 + (X1 - X0) * k / 200):.1f},{sy(theory(X0 + (X1 - X0) * k / 200)):.1f}" for k in range(201))
    o.append(f'<polyline fill="none" stroke="{mut}" stroke-width="2" stroke-dasharray="6 4" points="{pts}"/>')
    for x, y in bad:
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{surf}" stroke="{mut}" stroke-width="2"/>')
    for x, y, lo, hi in good:
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{acc}" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5.5" fill="{acc}" stroke="{surf}" stroke-width="2"/>')
    o.append(f'<text x="{L+14}" y="{T+22}" fill="{mut}" font-size="14">Dashed: 1 - (1 - s^8)^14, 14 bands of 8 hashes</text>')
    o.append(f'<text x="{L+14}" y="{T+44}" fill="{acc}" font-size="14" font-weight="bold">Filled: measured, data.py hash functions</text>')
    o.append(f'<text x="{L+14}" y="{T+66}" fill="{mut}" font-size="14">Hollow: first attempt, correlated hash functions</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
