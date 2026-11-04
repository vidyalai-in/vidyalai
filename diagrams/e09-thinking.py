# Generates public/diagrams/e09-thinking.svg and e09-thinking-dark.svg (data: e09-long-context-evals.md).
# Usage: python3 diagrams/e09-thinking.py public/diagrams/e09-thinking, then render the PNG with sharp at density 144.
# Data: Claude Haiku 5.5 on the same requests with thinking off and with adaptive thinking (default effort, medium),
# November 2026 (e09/scripts/grid.py stages small/large and tprimer/trest). 4 haystacks per length: 24 one-hop
# questions and 12 variable-tracking chains. One-hop graded strictly. Error bars: 95% Wilson intervals.
import math, sys
GROUPS = [  # label, (off k, n), (on k, n)
 ('8K', (1,24), (11,24)), ('32K', (2,24), (12,24)), ('96K', (2,24), (14,24)),
 ('8K', (2,12), (12,12)), ('32K', (0,12), (10,12)), ('96K', (0,12), (12,12)),
]
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)
W, H = 780, 470; L, R, T, B = 70, 24, 70, 92
def sy(y): return H - B - y * (H - T - B)
def svg(th):
    ink, mut, grid, acc, surf, acc2 = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(0, 101, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v/100):.1f}" y2="{sy(v/100):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(v/100)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v}%</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2:.1f}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Share answered correctly</text>')
    for i, (lab, c) in enumerate([('Thinking off', acc), ('Adaptive thinking', acc2)]):
        x = L + i * 170
        o.append(f'<rect x="{x}" y="{T-46}" width="16" height="16" fill="{c}"/>')
        o.append(f'<text x="{x+24}" y="{T-33}" fill="{ink}" font-size="14">{lab}</text>')
    o.append(f'<text x="{L+360}" y="{T-33}" fill="{mut}" font-size="12">Claude Haiku 5.5, same requests, 95% intervals</text>')
    gw = (W - L - R) / 6; bw = 34
    for g, (lab, off, on) in enumerate(GROUPS):
        cx = L + gw * (g + 0.5) + (12 if g >= 3 else -12)
        for j, ((k, n), c) in enumerate([(off, acc), (on, acc2)]):
            x = cx - bw - 2 + j * (bw + 4)
            r = k / n; lo, hi = wilson(k, n)
            o.append(f'<rect x="{x:.1f}" y="{sy(r):.1f}" width="{bw}" height="{max(1.5, sy(0)-sy(r)):.1f}" fill="{c}"/>')
            xm = x + bw / 2
            o.append(f'<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{ink}" stroke-width="1.5"/>')
            o.append(f'<line x1="{xm-5:.1f}" x2="{xm+5:.1f}" y1="{sy(hi):.1f}" y2="{sy(hi):.1f}" stroke="{ink}" stroke-width="1.5"/>')
            o.append(f'<text x="{xm:.1f}" y="{sy(hi)-6:.1f}" fill="{ink}" font-size="12" text-anchor="middle">{round(100*r)}%</text>')
        o.append(f'<text x="{cx:.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    for g0, lab in [(0, 'One-hop needle (24 per length)'), (3, 'Variable tracking (12 chains per length)')]:
        x0 = L + gw * g0 + (12 if g0 else -12) + 8; x1 = L + gw * (g0 + 3) + (12 if g0 else -12) - 8
        o.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{H-B+34}" y2="{H-B+34}" stroke="{mut}" stroke-width="1"/>')
        o.append(f'<text x="{(x0+x1)/2:.1f}" y="{H-B+54}" fill="{ink}" font-size="14" text-anchor="middle">{lab}</text>')
    o.append('</svg>')
    return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff', '#eb6834')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d', '#d95926')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
