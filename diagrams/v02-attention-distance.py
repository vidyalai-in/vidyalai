# Generates public/diagrams/v02-attention-distance.svg and -dark.svg (data: v02-vision-transformers.md).
# Mean attention distance per head (dots) and its layer average (line) for google/vit-base-patch16-224 on 256
# Oxford-IIIT Pet test images at 224 x 224: for each patch query, the attention-weighted mean pixel distance to
# the other patches (weights renormalised over patch keys), averaged over queries and images.
# Data from scratch v02/acc.py -> results/attn_dist.json. Usage: python3 diagrams/v02-attention-distance.py public/diagrams/v02-attention-distance
import sys, json
AD = json.loads('[[20.309999465942383, 67.05999755859375, 65.79000091552734, 89.66999816894531, 0.11999999731779099, 54.77000045776367, 61.45000076293945, 83.88999938964844, 63.869998931884766, 107.41000366210938, 115.11000061035156, 113.75], [74.8499984741211, 35.380001068115234, 77.81999969482422, 19.760000228881836, 19.1299991607666, 51.93000030517578, 96.3499984741211, 40.95000076293945, 64.88999938964844, 50.689998626708984, 105.87999725341797, 19.479999542236328], [55.29999923706055, 63.060001373291016, 61.70000076293945, 26.670000076293945, 54.90999984741211, 77.2300033569336, 19.5, 114.0199966430664, 76.19999694824219, 31.700000762939453, 30.8799991607666, 95.20999908447266], [47.40999984741211, 78.6500015258789, 69.20999908447266, 41.83000183105469, 46.869998931884766, 58.86000061035156, 57.68000030517578, 82.70999908447266, 114.73999786376953, 37.779998779296875, 85.36000061035156, 98.95999908447266], [65.05000305175781, 50.68000030517578, 75.12999725341797, 55.189998626708984, 74.54000091552734, 75.87000274658203, 86.11000061035156, 88.47000122070312, 122.12000274658203, 53.130001068115234, 76.11000061035156, 54.52000045776367], [97.33000183105469, 68.93000030517578, 107.36000061035156, 95.5, 77.38999938964844, 78.19999694824219, 67.98999786376953, 83.6500015258789, 68.77999877929688, 100.41000366210938, 75.13999938964844, 64.70999908447266], [87.95999908447266, 84.37000274658203, 81.5199966430664, 85.55999755859375, 95.12999725341797, 98.18000030517578, 106.30999755859375, 102.70999908447266, 78.02999877929688, 98.2699966430664, 88.29000091552734, 123.1500015258789], [103.38999938964844, 100.87000274658203, 94.5199966430664, 97.66999816894531, 104.01000213623047, 90.16000366210938, 104.2300033569336, 100.80999755859375, 107.30999755859375, 114.2300033569336, 96.43000030517578, 99.69999694824219], [107.51000213623047, 100.81999969482422, 103.95999908447266, 99.33000183105469, 117.6500015258789, 96.8499984741211, 107.45999908447266, 101.45999908447266, 116.12000274658203, 106.9000015258789, 100.75, 103.23999786376953], [112.77999877929688, 121.47000122070312, 103.94999694824219, 113.5999984741211, 115.83000183105469, 116.5999984741211, 115.26000213623047, 112.58999633789062, 118.86000061035156, 108.62000274658203, 110.23999786376953, 109.79000091552734], [117.3499984741211, 119.41000366210938, 122.9800033569336, 109.6500015258789, 113.69000244140625, 118.30000305175781, 111.94000244140625, 121.23999786376953, 112.52999877929688, 109.62999725341797, 112.05999755859375, 115.55000305175781], [124.7699966430664, 114.05999755859375, 124.70999908447266, 122.2300033569336, 120.4000015258789, 121.22000122070312, 123.4800033569336, 120.18000030517578, 117.22000122070312, 118.97000122070312, 124.23999786376953, 119.95999908447266]]')   # 12 layers x 12 heads, pixels
W, H = 780, 460; L, R, T, B = 70, 24, 24, 62
def sx(l): return L + (l - 0.5) / 12 * (W - L - R)
def sy(y): return H - B - y / 140 * (H - T - B)
def svg(th, acc, acc2):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(0, 141, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}</text>')
    for l in range(1, 13):
        o.append(f'<text x="{sx(l):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{l}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Layer (1 = first block after the patch embedding)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Mean attention distance (pixels)</text>')
    # reference: average distance between two random patches on the 14 x 14 grid
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(116.5):.1f}" y2="{sy(116.5):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{sx(0.6):.1f}" y="{sy(116.5)-8:.1f}" fill="{mut}" font-size="12">uniform attention over the whole image: 116 px</text>')
    means = []
    for l, heads in enumerate(AD, 1):
        for h, v in enumerate(heads):
            o.append(f'<circle cx="{sx(l) + (h - 5.5) * 3.2:.1f}" cy="{sy(v):.1f}" r="3.5" fill="{acc}" fill-opacity="0.75"/>')
        means.append(sum(heads) / len(heads))
    o.append(f'<polyline fill="none" stroke="{acc2}" stroke-width="2.5" points="{" ".join(f"{sx(l):.1f},{sy(m):.1f}" for l, m in enumerate(means, 1))}"/>')
    lx, ly = sx(7.2), sy(30)
    o.append(f'<circle cx="{lx+13}" cy="{ly}" r="4" fill="{acc}"/><text x="{lx+30}" y="{ly+5}" fill="{ink}" font-size="13">one attention head (12 per layer)</text>')
    o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{ly+24}" y2="{ly+24}" stroke="{acc2}" stroke-width="2.5"/><text x="{lx+30}" y="{ly+29}" fill="{ink}" font-size="13">layer average</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, '#2a78d6', '#eb6834'))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, '#3987e5', '#d95926'))
