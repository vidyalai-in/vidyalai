# Generates public/diagrams/v02-resolution-cost.svg and v02-resolution-cost-dark.svg (data: v02-vision-transformers.md).
# ViT-B/16 (google/vit-base-patch16-224, position embeddings interpolated) vs ResNet-50 (microsoft/resnet-50), fp16,
# RTX 4060 Laptop GPU, PyTorch 2.11. Top: images per second at batch 16 (median of 7 timed runs after 3 warm-up).
# Bottom: peak activation memory for one image (allocated beyond the weights), ViT with plain (eager) attention,
# ViT with fused SDPA attention, ResNet-50. Data from scratch v02/bench.py -> results/bench.json.
# Usage: python3 diagrams/v02-resolution-cost.py public/diagrams/v02-resolution-cost
import sys, math
RES = [224, 320, 448, 640, 896, 1024]
TOK = [197, 401, 785, 1601, 3137, 4097]
IPS = {"ViT-B/16": [466.1, 229.1, 114.2, 50.6, 21.9, 14.5], "ResNet-50": [886.2, 449.6, 240.5, 107.6, 54.1, 45.7]}
MEM = {"ViT-B/16, plain attention": [5.3, 18.2, 64.1, 249.4, 929.3, 1573.7],
       "ViT-B/16, fused attention": [4.0, 7.0, 13.8, 28.1, 55.1, 73.0],
       "ResNet-50": [5.9, 11.9, 23.0, 46.9, 91.9, 120.0]}
W, H = 780, 820
L, R = 84, 28
PANELS = [(30, 370, 10, 1000, "Images per second (batch 16, log scale)"),
          (450, 760, 2, 3000, "Peak activation memory, one image (MB, log scale)")]
X0, X1 = 200, 1100
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def mk_sy(top, bot, y0, y1):
    return lambda y: bot - (math.log10(y) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) * (bot - top)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for pi, (top, bot, y0, y1, ylab) in enumerate(PANELS):
        sy = mk_sy(top, bot, y0, y1)
        e = math.floor(math.log10(y0))
        while 10 ** e <= y1:
            for mlt in (1, 3):
                v = mlt * 10 ** e
                if y0 <= v <= y1:
                    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}" stroke-width="1"/>')
                    o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v:g}</text>')
            e += 1
        for x in RES:
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{top}" y2="{bot}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{sx(x):.1f}" y="{bot+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
        o.append(f'<text transform="translate(20,{(top+bot)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">{ylab}</text>')
        o.append(f'<text x="{(L+W-R)/2}" y="{bot+42}" fill="{ink}" font-size="14" text-anchor="middle">Input side in pixels (square image, log scale)</text>')
        data = IPS if pi == 0 else MEM
        cc = [cols[0], cols[1]] if pi == 0 else [cols[2], cols[0], cols[1]]
        for (name, ys), c in zip(data.items(), cc):
            pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(RES, ys))
            dash = ' stroke-dasharray="6 5"' if "plain" in name else ""
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5"{dash} points="{pts}"/>')
            for x, y in zip(RES, ys):
                o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        if pi == 0:
            for x, y, t in zip(RES, IPS["ViT-B/16"], TOK):
                o.append(f'<text x="{sx(x):.1f}" y="{sy(y)+22:.1f}" fill="{cols[0]}" font-size="12" text-anchor="middle">{t:,} tok</text>')
            leg = [(cols[0], "ViT-B/16 (86M parameters)", ""), (cols[1], "ResNet-50 (26M parameters)", "")]
            lx, ly = sx(560), top + 22
        else:
            leg = [(cols[2], "ViT-B/16, plain attention", ' stroke-dasharray="6 5"'), (cols[0], "ViT-B/16, fused attention (SDPA)", ""), (cols[1], "ResNet-50", "")]
            lx, ly = L + 20, top + 22
        for k, (c, lab, dash) in enumerate(leg):
            y = ly + k * 22
            o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"{dash}/>')
            o.append(f'<text x="{lx+34}" y="{y+5}" fill="{ink}" font-size="13">{lab}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834', '#8a5cd6')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926', '#9b72e0')))
