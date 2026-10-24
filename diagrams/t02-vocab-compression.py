# Generates public/diagrams/t02-vocab-compression.svg and t02-vocab-compression-dark.svg (data: t02-train-bpe-tokenizer.md).
# Usage: python3 diagrams/t02-vocab-compression.py public/diagrams/t02-vocab-compression, then render the PNG with sharp at density 144.
# Bytes per token on 26.1 MB of held-out FineWeb-Edu text (5,500 documents), logged by eval_compare.py.
# Filled points: byte-level BPE tokenizers trained on the 406 MB FineWeb-Edu sample at five vocabulary sizes.
# Hollow points: published tokenizers on the same text.
import math, sys
W, H = 780, 470; L, R, T, B = 70, 24, 24, 62
X0, X1 = 3000, 260000; Y0, Y1 = 3.3, 5.0
def sx(x): return L + (math.log2(x) - math.log2(X0)) / (math.log2(X1) - math.log2(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
ours = [(4096, 3.479), (8192, 3.931), (16384, 4.319), (32768, 4.610), (65536, 4.800)]
ref = [("Llama 2", 32000, 4.063, (10, 16)), ("SmolLM2", 49152, 4.605, (-6, 22)), ("GPT-2", 50257, 4.655, (10, 5)),
       ("GPT-4 cl100k", 100277, 4.791, (-44, 24)), ("Llama 3", 128256, 4.794, (-26, -12)),
       ("Qwen2.5", 151665, 4.702, (-20, 22)), ("GPT-4o o200k", 200019, 4.846, (-92, -12))]
def svg(th):
    ink, mut, grid, acc, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x, lab in [(4096, "4K"), (8192, "8K"), (16384, "16K"), (32768, "32K"), (65536, "64K"), (131072, "128K"), (262144, "256K")]:
        if x > X1: continue
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    for y in (3.5, 4.0, 4.5, 5.0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:.1f}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Vocabulary size (tokens, log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Bytes per token, held-out FineWeb-Edu</text>')
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ours)
    o.append(f'<polyline fill="none" stroke="{acc}" stroke-width="2.5" points="{pts}"/>')
    for x, y in ours:
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5.5" fill="{acc}" stroke="{surf}" stroke-width="2"/>')
        dx, dy = (-36, -10) if x == 65536 else (9, 18)
        o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{ink}" font-size="13">{y:.2f}</text>')
    o.append(f'<text x="{L+12}" y="{sy(4.85):.1f}" fill="{acc}" font-size="14" font-weight="bold">Filled: ours, trained on FineWeb-Edu</text>')
    o.append(f'<text x="{L+12}" y="{sy(4.85)+20:.1f}" fill="{mut}" font-size="14">Hollow: published tokenizers</text>')
    for name, x, y, (dx, dy) in ref:
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{surf}" stroke="{mut}" stroke-width="2"/>')
        o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{mut}" font-size="13">{name}</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
