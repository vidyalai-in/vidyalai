# Generates public/diagrams/p02-model-compare.svg and p02-model-compare-dark.svg. Logit lens on three models, tuned lens on Pythia-160M; same 304 held-out documents.
# Usage: python3 diagrams/p02-model-compare.py public/diagrams/p02-model-compare, then render the PNG with sharp at density 144.
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


DATA = {"gpt2": [0.01, 7.41, 8.06, 8.26, 9.4, 11.07, 15.03, 18.9, 26.9, 34.14, 44.9, 59.24, 100.0], "pythia": [0.0, 1.15, 1.04, 1.68, 2.38, 4.16, 5.97, 8.89, 10.57, 20.11, 29.84, 38.53, 100.0], "qwen": [0.43, 0.3, 0.25, 0.23, 0.27, 0.27, 0.35, 0.43, 0.55, 0.61, 0.62, 0.82, 1.13, 1.55, 1.91, 3.76, 6.29, 11.16, 14.82, 18.85, 32.39, 41.3, 49.76, 62.42, 100.0], "pythia_tuned": [24.76, 29.21, 32.15, 34.82, 37.32, 42.94, 50.18, 53.39, 56.21, 62.37, 65.91, 72.09, 100.0]}

PT = (0.08, 93)
G2 = (0.08, 86)
PY = (0.08, 79)
QW = (0.08, 72)
def pts(k): v = DATA[k]; return [(i / (len(v) - 1), x) for i, x in enumerate(v)]
SPEC = {
  'xlim': (-0.02, 1.02), 'ylim': (0, 100),
  'yticks': [0, 20, 40, 60, 80, 100], 'yfmt': '{:g}%',
  'xticks': [(0, 'embeddings'), (0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (1, 'last block')],
  'xlabel': 'Depth (fraction of the blocks run)',
  'ylabel': 'Top guess matches final output',
  'series': [
    {'style': 'mut', 'label': 'Qwen2.5-0.5B, logit lens', 'pts': pts('qwen'), 'label_at': QW, 'swatch': True},
    {'style': 'mutsolid', 'label': 'Pythia-160M, logit lens', 'pts': pts('pythia'), 'label_at': PY, 'swatch': True},
    {'style': 'ink', 'label': 'GPT-2 small, logit lens', 'pts': pts('gpt2'), 'label_at': G2, 'swatch': True},
    {'style': 'acc', 'label': 'Pythia-160M, tuned lens', 'pts': pts('pythia_tuned'), 'label_at': PT, 'swatch': True},
  ]}

write(SPEC, sys.argv[1])
