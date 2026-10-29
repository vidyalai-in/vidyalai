# Generates public/diagrams/p02-lens-kl.svg and p02-lens-kl-dark.svg. GPT-2 small, 304 held-out FineWeb-Edu documents x 510 positions.
# Usage: python3 diagrams/p02-lens-kl.py public/diagrams/p02-lens-kl, then render the PNG with sharp at density 144.
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


DATA = {"logit": {"m": [72.337, 10.662, 9.063, 8.372, 7.876, 7.493, 6.805, 6.235, 5.355, 4.667, 3.691, 1.946], "lo": [71.968, 10.577, 9.005, 8.319, 7.827, 7.445, 6.763, 6.195, 5.313, 4.626, 3.656, 1.92], "hi": [72.718, 10.749, 9.121, 8.426, 7.927, 7.544, 6.849, 6.277, 5.398, 4.71, 3.726, 1.973]}, "bias": {"m": [10.426, 6.91, 6.0, 5.452, 5.018, 4.704, 4.184, 3.74, 3.175, 2.843, 2.355, 1.359], "lo": [10.33, 6.832, 5.94, 5.396, 4.966, 4.653, 4.148, 3.71, 3.151, 2.819, 2.332, 1.339], "hi": [10.524, 6.988, 6.06, 5.509, 5.072, 4.756, 4.222, 3.775, 3.201, 2.868, 2.379, 1.379]}, "muon": {"m": [4.691, 3.31, 3.024, 2.728, 2.485, 2.321, 1.934, 1.595, 1.207, 0.881, 0.493, 0.218], "lo": [4.636, 3.27, 2.986, 2.692, 2.45, 2.286, 1.911, 1.577, 1.192, 0.868, 0.485, 0.214], "hi": [4.747, 3.351, 3.063, 2.765, 2.522, 2.357, 1.959, 1.614, 1.224, 0.895, 0.501, 0.222]}}

SPEC = {
  'xlim': (-0.4, 11.4), 'ylim': (0.1, 100), 'logy': True,
  'yticks': [0.1, 0.3, 1, 3, 10, 30, 100],
  'xticks': [(0, 'emb')] + [(i, str(i)) for i in range(1, 12)],
  'xlabel': 'Read after block (emb = embeddings only)',
  'ylabel': 'KL to final output (bits, log scale)',
  'series': [
    {'style': 'ink', 'label': 'Logit lens', 'pts': list(enumerate(DATA['logit']['m'])), 'lo': DATA['logit']['lo'], 'hi': DATA['logit']['hi'], 'label_at': (4.2, 13.5)},
    {'style': 'mut', 'label': 'Bias only', 'pts': list(enumerate(DATA['bias']['m'])), 'lo': DATA['bias']['lo'], 'hi': DATA['bias']['hi'], 'label_at': (2.3, 4.1)},
    {'style': 'acc', 'label': 'Tuned lens', 'pts': list(enumerate(DATA['muon']['m'])), 'lo': DATA['muon']['lo'], 'hi': DATA['muon']['hi'], 'label_at': (1.6, 1.6)},
  ]}

write(SPEC, sys.argv[1])
