# Generates public/diagrams/p02-norm-growth.svg and p02-norm-growth-dark.svg. Three models, the same 304 held-out documents, first position excluded.
# Usage: python3 diagrams/p02-norm-growth.py public/diagrams/p02-norm-growth, then render the PNG with sharp at density 144.
# Palette and approach copied from diagrams/g04-roofline.py. Data: mean and 95% bootstrap CI over held-out documents.
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


DATA = {"gpt2": [4.62, 51.03, 57.49, 61.71, 66.99, 71.84, 79.02, 87.92, 102.18, 120.25, 150.53, 219.04, 447.97], "pythia-160m": [0.71, 9.02, 11.34, 13.55, 14.84, 15.62, 17.8, 19.2, 24.08, 25.03, 27.7, 31.04, 112.98], "qwen2.5-0.5b": [0.45, 5.55, 8.16, 9.84, 11.07, 12.26, 13.23, 14.59, 15.27, 15.78, 15.93, 16.42, 16.65, 17.59, 18.27, 20.15, 22.92, 27.35, 33.14, 37.65, 44.49, 57.11, 72.53, 78.16, 80.46]}

PY_LABEL = (0.08, 480)
QW_LABEL = (0.08, 322)
SPEC = {
  'xlim': (-0.02, 1.02), 'ylim': (0.3, 1000), 'logy': True,
  'yticks': [0.3, 1, 3, 10, 30, 100, 300, 1000],
  'xticks': [(0, 'embeddings'), (0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (1, 'last block')],
  'xlabel': 'Depth (fraction of the blocks run)',
  'ylabel': 'Mean residual-stream norm (log scale)',
  'series': [
    {'style': 'acc', 'label': 'GPT-2 small', 'pts': [(i / (len(DATA['gpt2']) - 1), v) for i, v in enumerate(DATA['gpt2'])], 'label_at': (0.08, 720), 'swatch': True},
    {'style': 'ink', 'label': 'Pythia-160M', 'pts': [(i / (len(DATA['pythia-160m']) - 1), v) for i, v in enumerate(DATA['pythia-160m'])], 'label_at': PY_LABEL, 'swatch': True},
    {'style': 'mut', 'label': 'Qwen2.5-0.5B', 'pts': [(i / (len(DATA['qwen2.5-0.5b']) - 1), v) for i, v in enumerate(DATA['qwen2.5-0.5b'])], 'label_at': QW_LABEL, 'swatch': True},
  ]}

write(SPEC, sys.argv[1])
