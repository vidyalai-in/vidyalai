# Generates public/diagrams/v08-fd-passes.svg and -dark.svg (data: v08-autoregressive-image-generation.md).
# FD-FMNIST (lower is better) of 5,000 unconditional Fashion-MNIST samples against forward passes per image, for the
# masked prior (MaskGIT-style decoding, cosine schedule, Gumbel noise 4.5 annealed to 0) and the raster-order prior
# (temperature 1.0), both 5.0M-parameter transformers over the same 7 x 7 VQ-VAE codes (K = 512). Points are the mean
# of 3 sampling seeds, bars the min-max. Reference lines: the tokenizer's own reconstructions of 5,000 training images,
# and 5,000 real training images. Data: scratch v08/runs/prior_fd.jsonl.
# Usage: python3 diagrams/v08-fd-passes.py public/diagrams/v08-fd-passes
import math, sys
MASK = [(1, 304.26, 302.77, 305.46), (2, 186.38, 185.72, 187.24), (4, 65.08, 62.15, 67.40), (8, 37.25, 37.15, 37.32),
        (12, 40.99, 38.98, 42.19), (16, 44.54, 43.33, 45.16), (49, 41.54, 40.66, 42.18)]
AR = (49, 15.54, 15.16, 15.90)
RECON, REAL = 8.95, 0.64
W, H = 780, 460
L, R, T, B = 80, 30, 26, 66
X0, X1, Y0, Y1 = 0.8, 70, 0.4, 500
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    cm, ca = cols
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (1, 10, 100):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v}</text>')
    for x in (1, 2, 4, 8, 12, 16, 49):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Forward passes of the transformer per image (log scale)</text>')
    o.append(f'<text transform="translate(22,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">FD-FMNIST, lower is better (log)</text>')
    for y, lab in ((RECON, f"tokenizer reconstructions: {RECON}"), (REAL, f"real images: {REAL}")):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
        o.append(f'<text x="{sx(1.0)}" y="{sy(y)-7:.1f}" fill="{mut}" font-size="13">{lab}</text>')
    pts = " ".join(f"{sx(x):.1f},{sy(m):.1f}" for x, m, lo, hi in MASK)
    o.append(f'<polyline fill="none" stroke="{cm}" stroke-width="2.5" points="{pts}"/>')
    for x, m, lo, hi in MASK:
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{cm}" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(m):.1f}" r="5" fill="{cm}" stroke="{surf}" stroke-width="1.5"/>')
    o.append(f'<text x="{sx(1)+16:.1f}" y="{sy(304)+5:.1f}" fill="{cm}" font-size="13" font-weight="bold">masked prior</text>')
    o.append(f'<text x="{sx(8):.1f}" y="{sy(37.25)+22:.1f}" fill="{ink}" font-size="12" text-anchor="middle">37.3 at 8</text>')
    x, m, lo, hi = AR
    o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{ca}" stroke-width="2"/>')
    o.append(f'<rect x="{sx(x)-6:.1f}" y="{sy(m)-6:.1f}" width="12" height="12" fill="{ca}" stroke="{surf}" stroke-width="1.5"/>')
    o.append(f'<text x="{sx(x)-12:.1f}" y="{sy(m)+5:.1f}" fill="{ca}" font-size="13" font-weight="bold" text-anchor="end">raster order: 15.5</text>')
    o.append('</svg>')
    return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926')))
