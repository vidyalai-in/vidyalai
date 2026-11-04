# Generates public/diagrams/v07-dit-vs-unet.svg and -dark.svg (data: v07-flow-matching-dit.md).
# FD-FMNIST (lower is better) of EMA snapshots every 500 training steps against training time, for a 4.10M-parameter U-Net
# and a 4.16M-parameter DiT (2x2 patches, 196 tokens, width 192, 6 blocks) trained with the rectified flow objective,
# plus the U-Net trained with the DDPM objective for comparison. Batch 128, AdamW 3e-4, bf16, RTX 4060 Laptop GPU.
# Each point: 2,000 samples, 16 sampling steps (Euler for flow, DDIM for DDPM), same noise for every snapshot.
# Data from scratch v07/evalgen.py snaps -> runs/snaps.jsonl. Usage: python3 diagrams/v07-dit-vs-unet.py public/diagrams/v07-dit-vs-unet
import sys, json, math
D = json.loads('{"rf-unet":[[0.732,61.006],[1.433,31.936],[2.132,20.415],[2.833,19.155],[3.542,15.608],[4.347,12.988],[5.177,14.825],[5.933,14.933]],"rf-dit":[[0.75,263.505],[1.487,129.162],[2.225,77.348],[2.96,57.765],[3.687,49.116],[4.412,43.625],[5.14,38.841],[5.872,37.693]],"ddpm-unet":[[0.712,179.168],[1.402,88.3],[2.093,48.443],[2.783,32.197],[3.482,22.735],[4.175,20.737],[4.868,19.748],[5.563,15.483]]}')
NAMES = {"rf-unet": "U-Net, rectified flow", "rf-dit": "DiT, rectified flow", "ddpm-unet": "U-Net, DDPM (DDIM sampler)"}
W, H = 780, 470; L, R, T, B = 70, 30, 24, 62
X0, X1 = 0, 6; Y0, Y1 = 8, 400
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (10, 20, 50, 100, 200):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:g}</text>')
    for x in range(0, int(X1) + 1):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Training time (minutes on the laptop GPU)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">FD-FMNIST at 16 steps (log scale)</text>')
    for (m, pts), c in zip(D.items(), cols):
        dash = ' stroke-dasharray="6 5"' if m == "ddpm-unet" else ""
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5"{dash} points="{" ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)}"/>')
        for x, y in pts:
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
    lx, ly = sx(X1 * 0.5), sy(300)
    for k, (m, c) in enumerate(zip(D, cols)):
        y = ly + k * 22
        dash = ' stroke-dasharray="6 5"' if m == "ddpm-unet" else ""
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"{dash}/><text x="{lx+34}" y="{y+5}" fill="{ink}" font-size="13">{NAMES[m]}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#8a4fd6', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#a271e8', '#d95926')))
