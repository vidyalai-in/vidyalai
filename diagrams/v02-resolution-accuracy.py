# Generates public/diagrams/v02-resolution-accuracy.svg and -dark.svg (data: v02-vision-transformers.md).
# ImageNet top-1 accuracy (1,000-way head) on the 2,386 Oxford-IIIT Pet test images whose breed has an exact
# ImageNet class (24 breeds), at each input resolution; error bars are 95% Wilson intervals. ViT position
# embeddings are interpolated bicubically to the new grid, with no fine-tuning. fp16, RTX 4060 Laptop GPU.
# Data from scratch v02/acc.py -> results/acc.json. Usage: python3 diagrams/v02-resolution-accuracy.py public/diagrams/v02-resolution-accuracy
import sys, json, math
ACC = json.loads('{"vit224": [[112, 48.3, 46.3, 50.3], [160, 82.4, 80.8, 83.9], [224, 89.4, 88.1, 90.6], [288, 90.6, 89.3, 91.7], [384, 90.6, 89.3, 91.7], [448, 89.3, 88.0, 90.5], [512, 87.3, 85.9, 88.5], [640, 82.6, 81.0, 84.1]], "vit384": [[112, 0.8, 0.5, 1.2], [160, 64.0, 62.0, 65.9], [224, 84.2, 82.7, 85.6], [288, 88.4, 87.1, 89.7], [384, 90.9, 89.7, 92.0], [448, 91.6, 90.4, 92.7], [512, 91.4, 90.2, 92.5], [640, 91.8, 90.6, 92.8]], "resnet50": [[112, 81.7, 80.1, 83.2], [160, 87.2, 85.8, 88.5], [224, 90.3, 89.0, 91.4], [288, 91.2, 90.0, 92.3], [384, 90.8, 89.6, 91.9], [448, 89.5, 88.2, 90.7], [512, 88.3, 87.0, 89.5], [640, 84.3, 82.8, 85.7]]}')    # model -> [[res, top1 %, lo, hi], ...]
NAMES = {"vit224": "ViT-B/16 trained at 224", "vit384": "ViT-B/16 trained at 384", "resnet50": "ResNet-50 trained at 224"}
W, H = 780, 480; L, R, T, B = 70, 24, 24, 62
X0, X1 = 100, 700; Y0, Y1 = 0, 100
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(0, 101, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for x in (112, 160, 224, 288, 384, 448, 512, 640):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Input side in pixels at test time (log scale)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Top-1 accuracy (95% CI)</text>')
    for (m, pts), c in zip(ACC.items(), cols):
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, lo, hi in pts)}"/>')
        for x, y, lo, hi in pts:
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
    for k, (m, c) in enumerate(zip(ACC, cols)):
        x, y = sx(300), sy(42) + k * 24
        o.append(f'<line x1="{x}" x2="{x+26}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"/>')
        o.append(f'<text x="{x+34}" y="{y+5}" fill="{ink}" font-size="13">{NAMES[m]}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#1baf7a', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#199e70', '#d95926')))
