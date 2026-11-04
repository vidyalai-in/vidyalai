# Generates public/diagrams/v05-fm-steps.svg and -dark.svg (data: v05-diffusion-math.md).
# FD-FMNIST of 2,000 samples against sampling steps for the final linear-schedule U-Net (EMA weights), DDPM ancestral
# sampling (posterior variance) and DDIM (eta = 0) on the same evenly respaced grid; same starting noise (seed 100).
# Bars: range over 2 sampling seeds where measured. Right axis labels: samples per second at batch 1,000 from the
# measured forward time. Dashed: 2,000 real training images (1.05).
# Data from scratch v05/sweep_all.py -> runs/sweep.jsonl. Usage: python3 diagrams/v05-fm-steps.py public/diagrams/v05-fm-steps
import sys, json, math
D = json.loads('{"ddpm":[[10,25.44,25.44,25.44],[20,14.79,14.79,14.79],[50,10.26,10.12,10.4],[100,7.24,7.24,7.24],[250,7.17,7.15,7.18],[1000,6.61,6.61,6.61]],"ddim":[[10,14.83,14.64,15.02],[20,8.88,8.88,8.88],[50,7.61,6.77,8.45],[100,6.88,6.88,6.88],[250,7.03,7.03,7.03],[1000,7.17,7.17,7.17]]}')   # method -> [[steps, fd, lo, hi], ...]
FWD = 0.1085                # seconds per forward pass at batch 1,000
W, H = 780, 430; L, R, T, B = 70, 30, 34, 92
X0, X1, Y0, Y1 = 8, 1300, 1, 40
FLOOR = 1.05
NAMES = {"ddpm": "DDPM (random steps)", "ddim": "DDIM (deterministic)"}
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (1, 2, 5, 10, 20, 40):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}"/><text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="12.5" text-anchor="end">{y:g}</text>')
    for x in (10, 20, 50, 100, 250, 1000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/><text x="{sx(x):.1f}" y="{H-B+18}" fill="{mut}" font-size="12.5" text-anchor="middle">{x:,}</text>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+34}" fill="{mut}" font-size="11.5" text-anchor="middle">{1/(x*FWD/1000):,.0f}/s</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-B+56}" fill="{ink}" font-size="13.5" text-anchor="middle">Sampling steps (log scale) and samples per second at batch 1,000</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="13.5" text-anchor="middle">FD-FMNIST (log scale)</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(FLOOR):.1f}" y2="{sy(FLOOR):.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/>')
    for (m, c) in zip(("ddpm", "ddim"), cols):
        pts = D[m]
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)}"/>')
        for s, fd, lo, hi in pts:
            if hi > lo: o.append(f'<line x1="{sx(s):.1f}" x2="{sx(s):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(s):.1f}" cy="{sy(fd):.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
            dy = -10 if m == "ddpm" else 19
            o.append(f'<text x="{sx(s)+3:.1f}" y="{sy(fd)+dy:.1f}" fill="{c}" font-size="12.5" text-anchor="middle">{fd:.1f}</text>')
    y = H - 12; x = L
    for (m, c) in zip(("ddpm", "ddim"), cols):
        o.append(f'<line x1="{x}" x2="{x+26}" y1="{y-4}" y2="{y-4}" stroke="{c}" stroke-width="2.5"/><text x="{x+34}" y="{y}" fill="{ink}" font-size="13">{NAMES[m]}</text>'); x += 225
    o.append(f'<line x1="{x}" x2="{x+26}" y1="{y-4}" y2="{y-4}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/><text x="{x+34}" y="{y}" fill="{ink}" font-size="13">2,000 real images (1.05)</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5')))
