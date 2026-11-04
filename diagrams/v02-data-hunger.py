# Generates public/diagrams/v02-data-hunger.svg and -dark.svg (data: v02-vision-transformers.md).
# A 3.2M-parameter ViT (4 x 4 patches, width 256, 6 blocks) and a 3.1M-parameter ResNet-style CNN trained from
# scratch on CIFAR-10 subsets of 5,000, 15,000 and 50,000 images: AdamW, 1,200 steps of batch 256, cosine decay,
# random crop + flip, bf16, RTX 4060 Laptop GPU. Solid: accuracy on the 10,000-image test set (mean of 2 seeds,
# bars span both seeds). Dashed: accuracy on (up to 5,000 of) the training images, without augmentation.
# Data from scratch v02/train.py -> results/train.jsonl. Usage: python3 diagrams/v02-data-hunger.py public/diagrams/v02-data-hunger
import sys, json, math
D = json.loads('{"vit": [[5000, 56.1, 55.9, 56.3, 94.7], [15000, 63.5, 62.8, 64.1, 73.2], [50000, 65.4, 64.8, 66.0, 68.1]], "cnn": [[5000, 81.5, 81.4, 81.7, 100.0], [15000, 86.7, 86.6, 86.9, 98.5], [50000, 89.2, 89.1, 89.2, 93.0]]}')   # model -> [[n_train, test mean %, test min, test max, train mean %], ...]
NAMES = {"vit": "Tiny ViT (3.2M parameters)", "cnn": "Small CNN (3.1M parameters)"}
W, H = 780, 470; L, R, T, B = 70, 30, 24, 62
X0, X1 = 4000, 62000; Y0, Y1 = 40, 100
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(40, 101, 10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for x in (5000, 15000, 50000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:,}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Training images (CIFAR-10 subset, log scale)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Accuracy</text>')
    for (m, pts), c in zip(D.items(), cols):
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="6 5" points="{" ".join(f"{sx(p[0]):.1f},{sy(p[4]):.1f}" for p in pts)}"/>')
        for p in pts:
            o.append(f'<rect x="{sx(p[0])-4:.1f}" y="{sy(p[4])-4:.1f}" width="8" height="8" fill="{surf}" stroke="{c}" stroke-width="2"/>')
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)}"/>')
        for n, mu, lo, hi, tr in pts:
            o.append(f'<line x1="{sx(n):.1f}" x2="{sx(n):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(n):.1f}" cy="{sy(mu):.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
            o.append(f'<text x="{sx(n)+9:.1f}" y="{sy(mu)+17:.1f}" fill="{c}" font-size="12">{mu:.1f}%</text>')
    lx, ly = sx(9500), sy(55)
    for k, (m, c) in enumerate(zip(D, cols)):
        y = ly + k * 22
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"/><text x="{lx+34}" y="{y+5}" fill="{ink}" font-size="13">{NAMES[m]}, test set</text>')
    y = ly + 2 * 22
    o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y}" y2="{y}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/><text x="{lx+34}" y="{y+5}" fill="{ink}" font-size="13">dashed: same model on its own training images</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926')))
