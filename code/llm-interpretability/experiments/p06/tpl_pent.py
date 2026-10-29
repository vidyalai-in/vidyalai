# Generates public/diagrams/p06-pentagon.svg and -dark.svg (data: p06-superposition.md).
# Usage: python3 diagrams/p06-pentagon.py public/diagrams/p06-pentagon, then render the PNG with sharp at density 144.
# Data: ReLU output toy models with 5 equally important features in 2 dimensions, 50 runs per density, 10,000 AdamW
# steps (lab/toy.py, p06 scripts/toy_runs.py pentagon). Each panel draws the 5 columns of W of the lowest-loss run.
import math, sys
DATA = __DATA__
W, H = 780, 330
PW = 180; GAP = 10; L = 15; TOP = 46; R0 = 64
def svg(th):
    ink, mut, grid, acc, surf, acc2 = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for j, pan in enumerate(DATA):
        cx = L + j * (PW + GAP) + PW / 2; cy = TOP + 100
        o.append(f'<text x="{cx}" y="{TOP - 20}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{pan["title"]}</text>')
        o.append(f'<circle cx="{cx}" cy="{cy}" r="{R0}" fill="none" stroke="{grid}" stroke-width="1.5"/>')
        o.append(f'<line x1="{cx-R0-12}" x2="{cx+R0+12}" y1="{cy}" y2="{cy}" stroke="{grid}"/><line y1="{cy-R0-12}" y2="{cy+R0+12}" x1="{cx}" x2="{cx}" stroke="{grid}"/>')
        for (x, y) in pan["W"]:
            px, py = cx + x * R0, cy - y * R0
            o.append(f'<line x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}" stroke="{acc}" stroke-width="2.5"/>')
            o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{acc}"/>')
        for k, line in enumerate(pan["lines"]):
            o.append(f'<text x="{cx}" y="{cy + R0 + 40 + 18*k}" fill="{mut if k else ink}" font-size="13" text-anchor="middle">{line}</text>')
    o.append(f'<text x="{W/2}" y="{H-10}" fill="{mut}" font-size="12" text-anchor="middle">Circle: length 1. Each arrow is one feature\'s direction in the 2 hidden dimensions (lowest-loss run of 50).</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff', '#eb6834')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d', '#d95926')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
