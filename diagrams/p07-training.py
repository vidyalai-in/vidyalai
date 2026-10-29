# Generates public/diagrams/p07-training.svg and p07-training-dark.svg (data: p07-sparse-autoencoders.md).
# Usage: python3 diagrams/p07-training.py public/diagrams/p07-training, then render the PNG with sharp at density 144.
# Palette and approach copied from diagrams/g04-roofline.py. Data: p07 runs/main.log.jsonl, GPT-2 small blocks.8.hook_resid_pre,
# fraction of variance explained on each 4,096-token training batch = 1 - MSE / 406 (406 = held-out variance in the
# SAEs' normalized units), every 500 steps.
import math, sys

LIGHT = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff')
DARK = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d')


def chart(spec, th):
    ink, mut, grid, acc, surf = th
    W, H = spec.get('W', 780), spec.get('H', 470)
    L, R, T, B = spec.get('margins', (78, 24, 24, 62))
    x0, x1 = spec['xlim']; y0, y1 = spec['ylim']; logy = spec.get('logy', False)
    def sx(x): return L + (x - x0) / (x1 - x0) * (W - L - R)
    def sy(y):
        if logy: return H - B - (math.log10(y) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) * (H - T - B)
        return H - B - (y - y0) / (y1 - y0) * (H - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in spec['yticks']:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{spec.get("yfmt", "{:g}").format(y)}</text>')
    for x, lab in spec['xticks']:
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">{spec["xlabel"]}</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">{spec["ylabel"]}</text>')
    styles = {'acc': (acc, 3, ''), 'ink': (ink, 2.5, '7 5'), 'mut': (mut, 2.5, '2 4'), 'mutsolid': (mut, 2.5, '')}
    for s in spec['series']:
        col, wdt, dash = styles[s['style']]
        pts = s['pts']
        if 'lo' in s:
            band = [(x, lo) for (x, _), lo in zip(pts, s['lo'])] + [(x, hi) for (x, _), hi in reversed(list(zip(pts, s['hi'])))]
            o.append(f'<polygon fill="{col}" fill-opacity="0.18" stroke="none" points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in band)}"/>')
        da = f' stroke-dasharray="{dash}"' if dash else ''
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="{wdt}"{da} points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)}"/>')
        for x, y in pts:
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{3.5 if s["style"]=="acc" else 3}" fill="{col}" stroke="{surf}" stroke-width="1.5"/>')
        lx, ly = s['label_at']; anchor = s.get('anchor', 'start')
        if s.get('swatch'):
            o.append(f'<line x1="{sx(lx)-40:.1f}" x2="{sx(lx)-8:.1f}" y1="{sy(ly)-5:.1f}" y2="{sy(ly)-5:.1f}" stroke="{col}" stroke-width="{wdt}"{da}/>')
        o.append(f'<text x="{sx(lx):.1f}" y="{sy(ly):.1f}" fill="{col}" font-size="14" font-weight="bold" text-anchor="{anchor}">{s["label"]}</text>')
    o.append('</svg>')
    return '\n'.join(o)


def write(spec, out):
    open(out + '.svg', 'w').write(chart(spec, LIGHT))
    open(out + '-dark.svg', 'w').write(chart(spec, DARK))


DATA = {'topk32_24k': [[0.41, 0.361], [2.05, 0.657], [4.1, 0.743], [6.14, 0.778], [8.19, 0.797], [10.24, 0.81], [12.29, 0.82], [14.34, 0.827], [16.38, 0.831], [18.43, 0.835], [20.48, 0.838], [22.53, 0.842], [24.58, 0.845], [26.62, 0.849], [28.67, 0.848]], 'relu_l1_15_24k': [[0.41, 0.491], [2.05, 0.338], [4.1, 0.38], [6.14, 0.402], [8.19, 0.413], [10.24, 0.43], [12.29, 0.444], [14.34, 0.449], [16.38, 0.453], [18.43, 0.456], [20.48, 0.459], [22.53, 0.464], [24.58, 0.471], [26.62, 0.477], [28.67, 0.475]], 'topk32_3k': [[0.41, 0.438], [2.05, 0.704], [4.1, 0.752], [6.14, 0.768], [8.19, 0.775], [10.24, 0.78], [12.29, 0.785], [14.34, 0.787], [16.38, 0.787], [18.43, 0.786], [20.48, 0.786], [22.53, 0.787], [24.58, 0.79], [26.62, 0.793], [28.67, 0.791]]}

SPEC = {
  'xlim': (0, 30), 'ylim': (0.3, 0.9),
  'yticks': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'yfmt': '{:.0%}',
  'xticks': [(x, str(x)) for x in range(0, 31, 5)],
  'xlabel': 'Training tokens (millions)',
  'ylabel': 'Variance explained (training batches)',
  'series': [
    {'style': 'acc', 'label': 'TopK, 24,576 latents (32 active)', 'pts': [tuple(p) for p in DATA['topk32_24k']], 'label_at': (29.5, 0.875), 'anchor': 'end'},
    {'style': 'mutsolid', 'label': 'TopK, 3,072 latents (32 active)', 'pts': [tuple(p) for p in DATA['topk32_3k']], 'label_at': (29.5, 0.745), 'anchor': 'end'},
    {'style': 'ink', 'label': 'ReLU + L1, 24,576 latents (L0 falls to 17)', 'pts': [tuple(p) for p in DATA['relu_l1_15_24k']], 'label_at': (29.5, 0.515), 'anchor': 'end'},
  ]}

write(SPEC, sys.argv[1])
