# Generates public/diagrams/e09-length-curves.svg and e09-length-curves-dark.svg (data: e09-long-context-evals.md).
# Usage: python3 diagrams/e09-length-curves.py public/diagrams/e09-length-curves, then render the PNG with sharp at density 144.
# Data: Claude Haiku 5.5, thinking off, November 2026 (evals/lab/longctx.py, e09/scripts/grid.py and seq_run.py).
# Haystacks of Jane Austen text with 6 character needles and 3 variable chains each; 4 haystacks per length up to 96K,
# one haystack at 200K and at 500K. x = mean prompt tokens per length; error bars are 95% Wilson intervals.
import math, sys
X = [494, 3442, 7500, 15758, 31775, 62447, 96028, 199453, 505317]
SER = [  # label, key, (correct, n) per length
 ('Literal needle', 'lit', [(24,24),(22,24),(21,24),(21,24),(22,24),(21,24),(18,24),(6,6),(2,3)]),
 ('One-hop needle (strict)', 'hop', [(23,24),(11,24),(1,24),(2,24),(2,24),(2,24),(2,24),(0,6),(0,6)]),
 ('Variable tracking', 'vt', [(11,12),(1,12),(2,12),(0,12),(0,12),(2,12),(0,12),(0,3),(0,2)]),
]
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)
W, H = 780, 500; L, R, T, B = 70, 24, 92, 66
X0, X1 = 300, 800_000
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - y * (H - T - B)
def svg(th):
    ink, mut, grid, acc, surf, acc2 = th
    col = {'lit': acc, 'hop': acc2, 'vt': ink}
    shift = {'lit': 0.93, 'hop': 1.0, 'vt': 1.075}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(0, 101, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v/100):.1f}" y2="{sy(v/100):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(v/100)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v}%</text>')
    for x, lab in [(500, 'needles only'), (4000, '4K'), (16000, '16K'), (64000, '64K'), (256000, '256K')]:
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    # long-prompt price line
    o.append(f'<line x1="{sx(100000):.1f}" x2="{sx(100000):.1f}" y1="{T-16}" y2="{H-B}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 5"/>')
    o.append(f'<text x="{sx(100000)-6:.1f}" y="{T-6}" fill="{mut}" font-size="12" text-anchor="end">100K: long-prompt price starts (5x)</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Prompt length in tokens (log scale)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2:.1f}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Share answered correctly</text>')
    # legend
    lx = L
    for lab, key, _ in SER:
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{T-52}" y2="{T-52}" stroke="{col[key]}" stroke-width="3"/>')
        o.append(f'<circle cx="{lx+13}" cy="{T-52}" r="4.5" fill="{col[key]}"/>')
        o.append(f'<text x="{lx+34}" y="{T-47}" fill="{ink}" font-size="14">{lab}</text>')
        lx += 46 + 8.2 * len(lab)
    o.append(f'<text x="{L}" y="{T-22}" fill="{mut}" font-size="12">Claude Haiku 5.5, thinking off, 95% intervals. Up to 96K: 24 needles and 12 chains per length. Hollow: 2 to 6 items.</text>')
    for lab, key, pts in SER:
        xy = [(sx(x * shift[key]), sy(k / n)) for x, (k, n) in zip(X, pts)]
        o.append(f'<polyline fill="none" stroke="{col[key]}" stroke-width="2.5" points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in xy) + '"/>')
        for (x, (k, n)), (a, b) in zip(zip(X, pts), xy):
            lo, hi = wilson(k, n)
            o.append(f'<line x1="{a:.1f}" x2="{a:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{col[key]}" stroke-width="1.6" opacity="0.75"/>')
            filled = col[key] if n >= 12 else surf
            o.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" fill="{filled}" stroke="{col[key]}" stroke-width="2"/>')
    o.append('</svg>')
    return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff', '#eb6834')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d', '#d95926')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
