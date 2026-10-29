# Generates public/diagrams/t15-quant.svg and t15-quant-dark.svg (data: t15-frontier-training-recipes.md).
# Usage: python3 diagrams/t15-quant.py public/diagrams/t15-quant, then render the PNG with sharp at density 144.
# Data: t15/post/quant_summary.json. llama.cpp b11982, CPU build, AMD Ryzen 9 8945HS, 8 threads.
# Perplexity: llama-perplexity, n_ctx 1,024, 303 chunks of a held-out FineWeb-Edu slice (154,833 scored tokens).
# Speed: llama-bench tg128 (generating 128 tokens), mean of two runs of 5 repetitions each.
import math
import sys

# name, file size (MB), perplexity change vs F32 in % (chat model, base model), generation tokens/s (chat model)
ROWS = [('F32', 353.6, 0.000, 0.000, 134.2),
        ('F16', 177.4, 0.000, 0.000, 232.4),
        ('Q8_0', 94.8, -0.033, 0.031, 436.9),
        ('Q4_K_M', 59.6, 0.914, 1.019, 598.9)]
W, H = 780, 540
L, R, T, B = 78, 30, 46, 58
GAP = 46
PH = (H - T - B - GAP) / 2
X0, X1 = 45, 450


def sx(x):
    return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)


def sy_top(y):            # -0.2% .. 1.4%
    return T + PH - (y + 0.2) / 1.6 * PH


def sy_bot(y):            # 0 .. 700 tokens/s
    return T + PH + GAP + PH - y / 700 * PH


def svg(th, col):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="Arial,Helvetica,sans-serif" role="img">', f'<rect width="{W}" height="{H}" fill="{surf}"/>']
    for x in (50, 100, 200, 400):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{T + PH}" stroke="{grid}"/>')
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T + PH + GAP}" y2="{H - B}" stroke="{grid}"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H - B + 20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    for y in (0, 0.5, 1.0):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{sy_top(y):.1f}" y2="{sy_top(y):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{L - 8}" y="{sy_top(y) + 4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:+.1f}%</text>')
    for y in (0, 200, 400, 600):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{sy_bot(y):.1f}" y2="{sy_bot(y):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{L - 8}" y="{sy_bot(y) + 4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}</text>')
    o.append(f'<text x="{L}" y="{T - 16}" fill="{ink}" font-size="14" font-weight="bold">Perplexity vs F32 (chat / base)</text>')
    o.append(f'<text x="{L + 6}" y="{T + PH + GAP - 10}" fill="{ink}" font-size="14" font-weight="bold">'
             f'Generation speed, tokens per second (8 CPU threads)</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" fill="{ink}" font-size="14" text-anchor="middle">'
             f'GGUF file size (MB, log scale)</text>')
    for key, idx, dx in (('chat', 2, -7), ('base', 3, 7)):
        pts = ' '.join(f'{sx(r[1]):.1f},{sy_top(r[idx]):.1f}' for r in ROWS)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col[key]}" stroke-width="2"/>')
        for r in ROWS:
            o.append(f'<circle cx="{sx(r[1]):.1f}" cy="{sy_top(r[idx]):.1f}" r="5" fill="{col[key]}" stroke="{surf}" stroke-width="2"/>')
    for r in ROWS:
        if r[0] in ('F32', 'F16'):
            o.append(f'<text x="{sx(r[1]):.1f}" y="{sy_top(0) - 14:.1f}" fill="{ink}" font-size="13" '
                     f'text-anchor="middle">{r[0]}</text>')
        else:
            o.append(f'<text x="{sx(r[1]) + 10:.1f}" y="{sy_top(max(r[2], r[3])) - 10:.1f}" fill="{ink}" font-size="13">'
                     f'{r[0]}: {r[2]:+.2f}% / {r[3]:+.2f}%</text>')
    pts = ' '.join(f'{sx(r[1]):.1f},{sy_bot(r[4]):.1f}' for r in ROWS)
    o.append(f'<polyline points="{pts}" fill="none" stroke="{col["chat"]}" stroke-width="2"/>')
    for r in ROWS:
        o.append(f'<circle cx="{sx(r[1]):.1f}" cy="{sy_bot(r[4]):.1f}" r="5" fill="{col["chat"]}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{sx(r[1]) + 9:.1f}" y="{sy_bot(r[4]) - 8:.1f}" fill="{ink}" font-size="13">{r[0]}: {r[4]:.0f}</text>')
    lx, ly = W - R - 190, sy_top(1.3)
    for i, (k, t) in enumerate((('chat', 'Chat model (Part 11)'), ('base', 'Base model (Part 7)'))):
        o.append(f'<circle cx="{lx}" cy="{ly + 20 * i - 4}" r="5" fill="{col[k]}"/>'
                 f'<text x="{lx + 12}" y="{ly + 20 * i}" fill="{ink}" font-size="13">{t}</text>')
    o.append('</svg>')
    return '\n'.join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
lc = {'chat': '#0b5fd6', 'base': '#eb6834'}
dc = {'chat': '#38bdf8', 'base': '#d95926'}
open(sys.argv[1] + '.svg', 'w').write(svg(light, lc))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, dc))
