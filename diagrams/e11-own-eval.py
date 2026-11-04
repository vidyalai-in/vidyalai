# Generates public/diagrams/e11-own-eval.svg and e11-own-eval-dark.svg (data: e11-reading-model-cards.md).
# Values come from the e11 run logs (results.json of scripts/analyze.py): accuracy with 95% intervals
# per configuration, and the paired difference adaptive-low minus thinking-off with its MDE.
# Usage: python3 diagrams/e11-own-eval.py public/diagrams/e11-own-eval, then render the PNG with sharp at density 144.
W, H = 780, 560
L, R = 300, 40
# Panel A: accuracy (%), 0-100
A_T, A_B = 58, 270
rows_a = [
    ("Closed book, thinking off", "175 answerable items, 1 run", 19.4, 14.2, 25.9),
    ("Open book, thinking off", "195 items, 3 runs", 98.5, 97.0, 99.9),
    ("Open book, adaptive, low effort", "195 items, 3 runs", 100.0, 98.1, 100.0),
    ("Open book, adaptive, medium effort", "193 items, 2 validation runs", 100.0, 98.0, 100.0),
]
# Panel B: paired difference in points, -4 to +6
B_T, B_B = 360, 488
X0b, X1b = -4.0, 6.0
MDE = 2.1
rows_b = [
    ("Low minus off, items independent", "", 1.5, 0.1, 3.0),
    ("Low minus off, clustered by article", "", 1.5, -0.3, 3.4),
]


def sxa(v): return L + v / 100 * (W - L - R)
def sxb(v): return L + (v - X0b) / (X1b - X0b) * (W - L - R)


def svg(th):
    ink, mut, grid, acc, surf, band = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">',
         f'<rect width="{W}" height="{H}" fill="{surf}"/>']
    # Panel A
    o.append(f'<text x="20" y="30" fill="{ink}" font-size="16" font-weight="bold">Accuracy on the 195-question eval (95% interval)</text>')
    for v in range(0, 101, 20):
        o.append(f'<line x1="{sxa(v):.1f}" x2="{sxa(v):.1f}" y1="{A_T}" y2="{A_B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sxa(v):.1f}" y="{A_B + 20}" fill="{mut}" font-size="13" text-anchor="middle">{v}%</text>')
    step = (A_B - A_T) / len(rows_a)
    for i, (lab, sub, m, lo, hi) in enumerate(rows_a):
        y = A_T + step * (i + 0.5)
        o.append(f'<text x="{L - 14}" y="{y - 3:.1f}" fill="{ink}" font-size="14" text-anchor="end">{lab}</text>')
        o.append(f'<text x="{L - 14}" y="{y + 14:.1f}" fill="{mut}" font-size="12" text-anchor="end">{sub}</text>')
        o.append(f'<line x1="{sxa(lo):.1f}" x2="{sxa(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{acc}" stroke-width="3"/>')
        o.append(f'<circle cx="{sxa(m):.1f}" cy="{y:.1f}" r="6" fill="{acc}" stroke="{surf}" stroke-width="2"/>')
        tx, anchor = (sxa(hi) + 10, "start") if m < 60 else (sxa(lo) - 10, "end")
        o.append(f'<text x="{tx:.1f}" y="{y - 9:.1f}" fill="{ink}" font-size="13" font-weight="bold" text-anchor="{anchor}">{m:.1f}%</text>')
    # Panel B
    o.append(f'<text x="20" y="{B_T - 26}" fill="{ink}" font-size="16" font-weight="bold">Paired difference, low effort minus thinking off (points)</text>')
    o.append(f'<rect x="{sxb(-MDE):.1f}" y="{B_T}" width="{sxb(MDE) - sxb(-MDE):.1f}" height="{B_B - B_T}" fill="{band}"/>')
    for v in range(-4, 7, 2):
        o.append(f'<line x1="{sxb(v):.1f}" x2="{sxb(v):.1f}" y1="{B_T}" y2="{B_B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sxb(v):.1f}" y="{B_B + 20}" fill="{mut}" font-size="13" text-anchor="middle">{v:+d}</text>')
    o.append(f'<line x1="{sxb(0):.1f}" x2="{sxb(0):.1f}" y1="{B_T}" y2="{B_B}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{sxb(-MDE) + 4:.1f}" y="{B_T + 15}" fill="{mut}" font-size="12">MDE ±{MDE}</text>')
    step = (B_B - B_T) / len(rows_b)
    for i, (lab, sub, m, lo, hi) in enumerate(rows_b):
        y = B_T + step * (i + 0.6)
        o.append(f'<text x="{L - 14}" y="{y + 5:.1f}" fill="{ink}" font-size="14" text-anchor="end">{lab}</text>')
        o.append(f'<line x1="{sxb(lo):.1f}" x2="{sxb(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{acc}" stroke-width="3"/>')
        o.append(f'<circle cx="{sxb(m):.1f}" cy="{y:.1f}" r="6" fill="{acc}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{sxb(hi) + 10:.1f}" y="{y + 5:.1f}" fill="{ink}" font-size="13">{m:+.1f} ({lo:+.1f} to {hi:+.1f})</text>')
    o.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 16}" fill="{mut}" font-size="13" text-anchor="middle">Claude Haiku 5.5, Message Batches API, November 2026</text>')
    o.append('</svg>')
    return '\n'.join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff', '#eef3fb')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d', '#13223f')
import sys
open(sys.argv[1] + '.svg', 'w').write(svg(light))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
