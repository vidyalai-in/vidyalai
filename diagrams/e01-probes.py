# Generates public/diagrams/e01-probes.svg and e01-probes-dark.svg (data: e01-why-llm-evals-are-hard.md).
# Claude Haiku 5.5 (thinking off) on the same items asked up to four ways (NOTA = the text "None of the other options"), accuracy with 95% Wilson intervals,
# and the chance rate (mean of 1 / number of options). Data from scratch e01/scripts/analyze.py.
# Usage: python3 diagrams/e01-probes.py public/diagrams/e01-probes, then render the PNG with sharp at density 144.
import sys

W, H = 780, 440; L, R, T, B = 270, 40, 24, 62
X0, X1 = 0, 100
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)

# (group, [(label, acc, lo, hi, kind)], chance); kind 0 = normal question, 1 = probe
GROUPS = [
    ("ARC-Challenge (1,172)", [("Question and options", 93.00, 91.40, 94.33, 0),
                               ("Options only", 47.27, 44.42, 50.13, 1)], 25.02),
    ("MMLU subset (1,140)", [("Question and options", 85.26, 83.09, 87.20, 0),
                             ("A wrong option swapped for NOTA", 81.05, 78.67, 83.22, 1),
                             ("Right option swapped for NOTA", 52.72, 49.82, 55.60, 1),
                             ("Options only", 47.72, 44.83, 50.62, 1)], 25.00),
    ("MMLU-Pro subset (840)", [("Question and options", 70.95, 67.79, 73.92, 0),
                               ("Options only", 35.48, 32.31, 38.77, 1)], 11.16),
]

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in range(0, 101, 20):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}%</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Accuracy, Claude Haiku 5.5 (95% interval)</text>')
    y = T + 8
    bh = 22
    for g, rows, chance in GROUPS:
        o.append(f'<text x="12" y="{y+14}" fill="{ink}" font-size="14" font-weight="bold">{g}</text>')
        y += 24
        top = y
        for lab, a, lo, hi, kind in rows:
            c = cols[kind]
            o.append(f'<text x="{L-10}" y="{y+bh/2+5:.1f}" fill="{ink}" font-size="13" text-anchor="end">{lab}</text>')
            o.append(f'<rect x="{sx(0):.1f}" y="{y:.1f}" width="{sx(a)-sx(0):.1f}" height="{bh}" fill="{c}" rx="2"/>')
            cy = y + bh / 2
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{ink}" stroke-width="2"/>')
            for e in (lo, hi):
                o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{cy-6:.1f}" y2="{cy+6:.1f}" stroke="{ink}" stroke-width="2"/>')
            o.append(f'<text x="{sx(hi)+8:.1f}" y="{cy+5:.1f}" fill="{ink}" font-size="13">{a:.1f}</text>')
            y += bh + 6
        o.append(f'<line x1="{sx(chance):.1f}" x2="{sx(chance):.1f}" y1="{top-4:.1f}" y2="{y-2:.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="5 4"/>')
        o.append(f'<text x="{sx(chance)+4:.1f}" y="{top-8:.1f}" fill="{mut}" font-size="12">chance {chance:.0f}%</text>')
        y += 22
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926')))
