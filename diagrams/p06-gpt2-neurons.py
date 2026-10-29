# Generates public/diagrams/p06-gpt2-neurons.svg and -dark.svg (data: p06-superposition.md).
# Usage: python3 diagrams/p06-gpt2-neurons.py public/diagrams/p06-gpt2-neurons, then render the PNG with sharp at density 144.
# Data: GPT-2 small (fp32, TransformerLens 3.9.0, no weight processing), MLP neurons after GELU, 1,600 windows of 127 tokens:
# 400 each of English web text (FineWeb-Edu), German Wikipedia, Korean Wikipedia and Python standard-library code.
# A token fires a neuron if the activation exceeds half the neuron's maximum on this sample. Shares are of the neurons that
# fire on 20 or more tokens but under 1% of tokens (34,582 of 36,864). Band: 95% bootstrap interval over windows.
# p06 scripts/s6_gpt2_neurons.py, s6_analyze.py, s7_tokens.py, s7_analyze.py.
import sys
DATA = {"multi":[0.8693,0.8794,0.7651,0.7687,0.7628,0.7371,0.7285,0.7426,0.6992,0.6905,0.6672,0.6987],"mixed":[0.6365,0.7289,0.6335,0.6322,0.6287,0.5948,0.5821,0.5822,0.5405,0.5457,0.5519,0.5281],"tokspec":[0.1607,0.0945,0.075,0.0822,0.0845,0.1004,0.0858,0.1139,0.1044,0.1081,0.0769,0.1255],"multi_ci":[[0.8593295812606812,0.8680614829063416,0.7506536841392517,0.7527349591255188,0.7472147941589355,0.7252285480499268,0.7112298607826233,0.7301158905029297,0.6859208345413208,0.674494743347168,0.6534777283668518,0.6840962767601013],[0.8779206275939941,0.8879566192626953,0.7807402014732361,0.7821563482284546,0.7745048403739929,0.7587302923202515,0.7465478181838989,0.7616536617279053,0.7168956995010376,0.700825035572052,0.6819209456443787,0.713383674621582]]}
W, H = 780, 440; L, R, T, B = 64, 24, 70, 62
def sx(l): return L + l / 11 * (W - L - R)
def sy(v): return H - B - v * (H - T - B)
def svg(th):
    ink, mut, grid, acc, surf, acc2 = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{int(v*100)}%</text>')
    for l in range(12):
        o.append(f'<text x="{sx(l):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{l}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">GPT-2 small layer (MLP neurons)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Share of sparse neurons</text>')
    lo, hi = DATA["multi_ci"]
    band = " ".join(f"{sx(l):.1f},{sy(v):.1f}" for l, v in enumerate(hi)) + " " + " ".join(f"{sx(l):.1f},{sy(v):.1f}" for l, v in reversed(list(enumerate(lo))))
    o.append(f'<polygon points="{band}" fill="{acc}" opacity="0.2"/>')
    for key, col, dash in (("multi", acc, ""), ("mixed", acc, 'stroke-dasharray="7 5"'), ("tokspec", acc2, "")):
        pts = " ".join(f"{sx(l):.1f},{sy(v):.1f}" for l, v in enumerate(DATA[key]))
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" {dash} points="{pts}"/>')
        for l, v in enumerate(DATA[key]):
            o.append(f'<circle cx="{sx(l):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{col}"/>')
    lg = [(acc, "", "Fires in 2+ of the 4 kinds of text (10%+ of firings each)"),
          (acc, 'stroke-dasharray="7 5"', "...and on different top tokens in its two main kinds of text"),
          (acc2, "", "One token type makes up half or more of its firings")]
    for i, (c, d, t) in enumerate(lg):
        y = 20 + i * 18
        o.append(f'<line x1="{L+4}" x2="{L+32}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5" {d}/>')
        o.append(f'<text x="{L+40}" y="{y+4}" fill="{ink}" font-size="13">{t}</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff', '#eb6834')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d', '#d95926')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
