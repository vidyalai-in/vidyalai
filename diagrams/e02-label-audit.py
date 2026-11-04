# Generates public/diagrams/e02-label-audit.svg and e02-label-audit-dark.svg (data: e02-how-benchmarks-are-built.md).
# Numbers come from the logged runs: e02/results/audit-haiku-k3.json (Haiku, 3 answers per item) and
# e02/results/sonnet.json (Sonnet, 1 answer per item on its stratified subset). 95% bootstrap CIs over items.
# Usage: python3 diagrams/e02-label-audit.py public/diagrams/e02-label-audit, then render the PNG with sharp at density 144.
import sys

# (label, n Haiku, Haiku (p, lo, hi), Sonnet (p, lo, hi) or None, n Sonnet)
ROWS = [
    ("ok", 5330, (84.12, 83.16, 85.08), (90.54, 87.57, 93.24), 370),
    ("bad options clarity", 25, (82.67, 68.00, 94.67), None, 0),
    ("no correct answer", 36, (72.22, 57.41, 86.11), None, 0),
    ("multiple correct answers", 39, (70.09, 55.56, 82.91), None, 0),
    ("expert", 32, (69.79, 54.17, 84.38), None, 0),
    ("bad question clarity", 132, (65.15, 57.07, 72.73), None, 0),
    ("wrong key: picks MMLU's key", 106, (18.87, 12.26, 26.10), (13.21, 7.55, 19.81), 106),
    ("wrong key: picks experts' answer", 101, (67.33, 58.75, 75.91), (76.24, 67.33, 84.16), 101),
]
W, H = 780, 520
L, R, T, B = 250, 30, 56, 62
RH = (H - T - B) / len(ROWS)


def sx(p):
    return L + p / 100 * (W - L - R)


def svg(th, c_h, c_s):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="Arial,Helvetica,sans-serif" role="img">', f'<rect width="{W}" height="{H}" fill="{surf}"/>']
    for p in range(0, 101, 20):
        o.append(f'<line x1="{sx(p):.1f}" x2="{sx(p):.1f}" y1="{T}" y2="{H - B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(p):.1f}" y="{H - B + 20}" fill="{mut}" font-size="13" text-anchor="middle">{p}%</text>')
    o.append(f'<line x1="{sx(25):.1f}" x2="{sx(25):.1f}" y1="{T}" y2="{H - B}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{sx(25) + 5:.1f}" y="{T - 6}" fill="{mut}" font-size="12">chance 25%</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 16}" fill="{ink}" font-size="14" text-anchor="middle">'
             'Share of answers that match (95% CI)</text>')
    o.append(f'<text x="12" y="{T - 6}" fill="{mut}" font-size="12">Agrees with MMLU\'s key, by annotation</text>')
    # legend
    o.append(f'<circle cx="{L + 6}" cy="22" r="6" fill="{c_h}"/>')
    o.append(f'<text x="{L + 18}" y="27" fill="{ink}" font-size="13">claude-haiku-5-5, 3 answers per item</text>')
    o.append(f'<rect x="{L + 290}" y="16" width="12" height="12" fill="{c_s}"/>')
    o.append(f'<text x="{L + 308}" y="27" fill="{ink}" font-size="13">claude-sonnet-5-5, 1 answer</text>')
    for k, (lab, n, hk, sn, ns) in enumerate(ROWS):
        y = T + RH * (k + 0.5)
        if k == 6:
            o.append(f'<line x1="12" x2="{W - R}" y1="{y - RH / 2:.1f}" y2="{y - RH / 2:.1f}" stroke="{mut}" stroke-width="1"/>')
        o.append(f'<text x="{L - 12}" y="{y - 2:.1f}" fill="{ink}" font-size="13" text-anchor="end">{lab}</text>')
        sub = f"n = {n:,}" + (f"; Sonnet n = {ns}" if sn else "")
        o.append(f'<text x="{L - 12}" y="{y + 14:.1f}" fill="{mut}" font-size="11" text-anchor="end">{sub}</text>')
        dy = -7 if sn else 0
        p, lo, hi = hk
        o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y + dy:.1f}" y2="{y + dy:.1f}" stroke="{c_h}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{sx(p):.1f}" cy="{y + dy:.1f}" r="6" fill="{c_h}" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<text x="{sx(hi) + 8:.1f}" y="{y + dy + 4:.1f}" fill="{ink}" font-size="12" stroke="{surf}" stroke-width="4" paint-order="stroke">{p:.1f}%</text>')
        if sn:
            p, lo, hi = sn
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y + 8:.1f}" y2="{y + 8:.1f}" stroke="{c_s}" stroke-width="2.5"/>')
            o.append(f'<rect x="{sx(p) - 5.5:.1f}" y="{y + 2.5:.1f}" width="11" height="11" fill="{c_s}" stroke="{surf}" stroke-width="1.5"/>')
            o.append(f'<text x="{sx(hi) + 8:.1f}" y="{y + 12:.1f}" fill="{ink}" font-size="12" stroke="{surf}" stroke-width="4" paint-order="stroke">{p:.1f}%</text>')
    o.append('</svg>')
    return '\n'.join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, '#0b5fd6', '#eb6834'))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, '#38bdf8', '#d95926'))
