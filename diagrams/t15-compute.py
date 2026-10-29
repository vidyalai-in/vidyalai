# Generates public/diagrams/t15-compute.svg and t15-compute-dark.svg (data: t15-frontier-training-recipes.md).
# Usage: python3 diagrams/t15-compute.py public/diagrams/t15-compute, then render the PNG with sharp at density 144.
# Training compute, 6 x parameters x tokens unless a paper states its own total:
#   Part 8's 11M run: 6 x 10,938,960 x 218.8M; ours: 6 x 88,096,512 x 1,761,607,680 (Part 7);
#   SmolLM2-135M: 6 x 134.5M x 2T (paper, model card); SmolLM3-3B: 6 x 3B x 11.2T (Hugging Face blog);
#   OLMo 2 7B and 32B: 1.8 and 13.0 x 10^23 FLOPs (OLMo 2 paper, Table 6);
#   DeepSeek-V3: 6 x 37B activated x 14.8T (technical report); Llama 3 405B: 3.8 x 10^25 FLOPs (Llama 3 paper).
import math
import sys

OURS = 6 * 88_096_512 * 1_761_607_680
RUNS = [("Part 8's 11M run", 6 * 10_938_960 * 218.8e6, 'ours'),
        ('Our 88M model (Part 7)', OURS, 'ours'),
        ('SmolLM2-135M, 2T tokens', 6 * 134.5e6 * 2e12, 'pub'),
        ('OLMo 2 7B, 4.05T tokens', 1.8e23, 'pub'),
        ('SmolLM3-3B, 11.2T tokens', 6 * 3e9 * 11.2e12, 'pub'),
        ('OLMo 2 32B, 6.6T tokens', 13.0e23, 'pub'),
        ('DeepSeek-V3, 14.8T tokens', 6 * 37e9 * 14.8e12, 'pub'),
        ('Llama 3 405B, 15.6T tokens', 3.8e25, 'pub')]
W, H = 780, 430
L, R, T, B = 214, 92, 20, 58
X0, X1 = 1e16, 1e26
ROW = (H - T - B) / len(RUNS)


def sx(x):
    return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)


def ratio(c):
    r = c / OURS
    if r < 1:
        return f'{r:.3f}x ours' if r >= 0.001 else ''
    if r < 1.5:
        return 'ours'
    return f'{r:,.0f}x ours' if r < 1e6 else f'{r / 1e6:.1f} million x ours'


def svg(th, col):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="Arial,Helvetica,sans-serif" role="img">', f'<rect width="{W}" height="{H}" fill="{surf}"/>']
    for e in range(16, 27, 2):
        x = sx(10 ** e)
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{H - B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{H - B + 20}" fill="{mut}" font-size="13" text-anchor="middle">1e{e}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" fill="{ink}" font-size="14" text-anchor="middle">'
             f'Training compute (FLOPs, log scale)</text>')
    for i, (name, c, g) in enumerate(RUNS):
        y = T + i * ROW + ROW * 0.18
        h = ROW * 0.64
        o.append(f'<rect x="{L}" y="{y:.1f}" width="{sx(c) - L:.1f}" height="{h:.1f}" fill="{col[g]}" rx="2"/>')
        o.append(f'<text x="{L - 8}" y="{y + h / 2 + 5:.1f}" fill="{ink}" font-size="13" text-anchor="end">{name}</text>')
        o.append(f'<text x="{sx(c) + 6:.1f}" y="{y + h / 2 + 5:.1f}" fill="{mut}" font-size="12">{ratio(c)}</text>')
    o.append('</svg>')
    return '\n'.join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
lc = {'ours': '#eb6834', 'pub': '#0b5fd6'}
dc = {'ours': '#d95926', 'pub': '#38bdf8'}
open(sys.argv[1] + '.svg', 'w').write(svg(light, lc))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, dc))
