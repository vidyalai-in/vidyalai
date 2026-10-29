# Generates public/diagrams/p07-firing.svg and p07-firing-dark.svg (data: p07-sparse-autoencoders.md).
# Usage: python3 diagrams/p07-firing.py public/diagrams/p07-firing, then render the PNG with sharp at density 144.
# Palette and approach copied from diagrams/g04-roofline.py. Data: p07 runs/summary.json. How often each latent of three
# GPT-2 small layer-8 SAEs fires on 537,337 held-out FineWeb-Edu tokens, histogram of log10(frequency) in quarter-decade
# bins (latents that never fired are left out: 433, 34 and 5).
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


DATA = {'bins': [-6.5, -6.25, -6.0, -5.75, -5.5, -5.25, -5.0, -4.75, -4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0], 'topk32_24k': [0, 0, 0, 263, 267, 216, 331, 619, 1436, 2338, 3479, 3790, 3399, 2662, 1944, 1328, 936, 598, 346, 125, 43, 12, 4, 3, 4, 0], 'relu_l1_15_24k': [0, 0, 0, 3, 2, 9, 19, 67, 335, 1145, 2879, 4718, 4976, 4522, 3908, 1669, 275, 15, 0, 0, 0, 0, 0, 0, 0, 0], 'jb_24k': [0, 0, 0, 2, 3, 1, 13, 26, 96, 259, 816, 2445, 4247, 4140, 3472, 3001, 2568, 1901, 1047, 434, 82, 16, 2, 0, 0, 0]}
c = lambda b: [((DATA['bins'][i] + DATA['bins'][i + 1]) / 2, v) for i, v in enumerate(b) if -5.8 < DATA['bins'][i] < -0.2]

SPEC = {
  'xlim': (-5.75, -0.25), 'ylim': (0, 5000),
  'yticks': [0, 1000, 2000, 3000, 4000, 5000], 'yfmt': '{:,.0f}',
  'xticks': [(-5, '1 in 100,000'), (-4, '1 in 10,000'), (-3, '1 in 1,000'), (-2, '1 in 100'), (-1, '1 in 10')],
  'xlabel': 'How often a latent fires (share of tokens, log scale)',
  'ylabel': 'Number of latents',
  'margins': (78, 30, 24, 62),
  'series': [
    {'style': 'acc', 'label': 'Ours: TopK, 32 active', 'pts': c(DATA['topk32_24k']), 'label_at': (-4.45, 3800), 'anchor': 'end'},
    {'style': 'ink', 'label': 'Public ReLU (Bloom), L0 71', 'pts': c(DATA['jb_24k']), 'label_at': (-2.3, 3500), 'anchor': 'start'},
    {'style': 'mut', 'label': 'Ours: ReLU + L1, L0 17', 'pts': c(DATA['relu_l1_15_24k']), 'label_at': (-2.85, 4800), 'anchor': 'start'},
  ]}

write(SPEC, sys.argv[1])
