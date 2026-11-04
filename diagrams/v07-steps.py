# Generates public/diagrams/v07-steps.svg and -dark.svg (data: v07-flow-matching-dit.md).
# FD-FMNIST (Frechet distance on a small Fashion-MNIST classifier's features, not Inception; lower is better) against the
# number of sampling steps (network calls) for three 4.1M-parameter U-Nets trained on Fashion-MNIST for 4,000 steps of batch
# 128 (same network, same compute): DDPM noise prediction sampled with deterministic DDIM, rectified flow sampled with Euler
# steps, and the rectified flow after one reflow round (1,500 more steps on 20,000 of its own noise-sample pairs).
# Mean of 3 sampling seeds x 2,000 samples; bars span the 3 seeds. Dashed: 2,000 real training images (1.05).
# Data from scratch v07/evalgen.py -> runs/sweep.jsonl. Usage: python3 diagrams/v07-steps.py public/diagrams/v07-steps
import sys, json, math
D = json.loads('{"ddpm-unet":[[1,540.57,537.7,542.85],[2,539.0,536.16,541.27],[4,129.94,126.8,132.7],[8,35.19,34.44,35.57],[16,15.3,15.05,15.48],[32,9.31,9.07,9.5],[100,7.96,7.62,8.42]],"rf-unet":[[1,723.81,722.92,724.46],[2,459.78,457.03,462.87],[4,130.06,127.71,133.44],[8,32.72,31.17,34.22],[16,14.38,13.47,14.93],[32,9.09,8.31,10.05],[100,7.69,7.0,8.78]],"reflow-unet":[[1,46.54,45.37,47.56],[2,31.12,29.83,32.13],[4,23.79,22.49,24.91],[8,19.38,18.0,20.7],[16,16.78,15.35,18.02],[32,15.18,13.84,16.19]]}')
NAMES = {"ddpm-unet": "Diffusion (DDPM), DDIM sampler", "rf-unet": "Rectified flow, Euler sampler", "reflow-unet": "After one reflow, Euler"}
FLOOR = 1.05
W, H = 780, 480; L, R, T, B = 70, 30, 24, 62
X0, X1 = 0.8, 130; Y0, Y1 = 0.8, 800
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (1, 3, 10, 30, 100, 300):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:g}</text>')
    for x in (1, 2, 4, 8, 16, 32, 100):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Sampling steps (network calls per image, log scale)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">FD-FMNIST (log scale, lower is better)</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(FLOOR):.1f}" y2="{sy(FLOOR):.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{W-R-4}" y="{sy(FLOOR)-7:.1f}" fill="{mut}" font-size="13" text-anchor="end">2,000 real images: 1.05</text>')
    for (m, pts), c in zip(D.items(), cols):
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)}"/>')
        for s, mu, lo, hi in pts:
            o.append(f'<line x1="{sx(s):.1f}" x2="{sx(s):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(s):.1f}" cy="{sy(mu):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
    lx, ly = sx(12), sy(500)
    for k, (m, c) in enumerate(zip(D, cols)):
        y = ly + k * 22
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"/><text x="{lx+34}" y="{y+5}" fill="{ink}" font-size="13">{NAMES[m]}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6', '#1f9e74')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5', '#2bb38a')))
