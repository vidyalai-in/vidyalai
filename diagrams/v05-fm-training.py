# Generates public/diagrams/v05-fm-training.svg and -dark.svg (data: v05-diffusion-math.md).
# Fashion-MNIST U-Net DDPM (1.59M parameters, epsilon prediction, L_simple, AdamW 5e-4, batch 128, EMA 0.9995, bf16,
# torch.compile), RTX 4060 Laptop GPU. Left: training loss (mean of each 250-step window). Right: FD-FMNIST of 2,000
# EMA samples (DDIM, 50 steps, fixed seed) at the same training steps. Dashed: 2,000 real
# training images (1.05). Data from scratch v05/train_fm.py -> runs/fm_*/state.json.
# Usage: python3 diagrams/v05-fm-training.py public/diagrams/v05-fm-training
import sys, json, math
D = json.loads('{"linear":{"loss":[[250,0.1718],[500,0.066],[750,0.054],[1000,0.0431],[1250,0.0494],[1500,0.0437],[1750,0.0534],[2000,0.0485],[2250,0.0414],[2500,0.0381],[2750,0.0431],[3000,0.0431],[3250,0.0446],[3500,0.0463],[3750,0.0378],[4000,0.037],[4250,0.0422],[4500,0.0397],[4750,0.0426],[5000,0.0441],[5250,0.0385],[5500,0.0415],[5750,0.0428],[6000,0.0352],[6250,0.039],[6500,0.0381],[6750,0.0387],[7000,0.0384],[7250,0.0404],[7500,0.0363],[7750,0.0409],[8000,0.0356],[8250,0.0387],[8500,0.0406],[8750,0.0386],[9000,0.0387],[9250,0.0399],[9500,0.0411],[9750,0.0378],[10000,0.036],[10250,0.038],[10500,0.0367],[10750,0.0409],[11000,0.037],[11250,0.039],[11500,0.0348]],"fd":[[500,123.5,931.44],[1000,161.7,660.55],[1500,202.2,322.5],[2500,264.3,131.77],[5000,431.0,41.58],[7500,573.2,23.27],[10000,696.2,9.09],[11500,769.1,5.62]]},"cosine":{"loss":[[250,0.239],[500,0.1094],[750,0.0937],[1000,0.0797],[1250,0.0876],[1500,0.079],[1750,0.0897],[2000,0.0852],[2250,0.0741],[2500,0.0717],[2750,0.077],[3000,0.0769],[3250,0.0784],[3500,0.082],[3750,0.0706],[4000,0.0675],[4250,0.073],[4500,0.0709],[4750,0.0754],[5000,0.076],[5250,0.0685],[5500,0.0738],[5750,0.0748],[6000,0.0655],[6250,0.0708],[6500,0.07],[6750,0.0692],[7000,0.0689],[7250,0.0716],[7500,0.0669],[7750,0.0721],[8000,0.0651],[8250,0.0695],[8500,0.0709],[8750,0.0691],[9000,0.0704],[9250,0.0722],[9500,0.0721],[9750,0.0676],[10000,0.066],[10250,0.0697],[10500,0.067],[10750,0.0714],[11000,0.0669],[11250,0.0696],[11500,0.0629]],"fd":[[500,30.5,934.07],[1000,54.2,681.33],[1500,77.8,357.73],[2500,125.4,135.28],[5000,246.0,50.48],[7500,367.4,36.53],[10000,487.2,19.58],[11500,0,18.69]]}}')   # sched -> {"loss": [[step, loss], ...], "fd": [[step, train_s, fd], ...]}
NAMES = {"linear": "Linear schedule", "cosine": "Cosine noise schedule"}
W, H = 780, 430; PW = 290; L1, L2, T, B = 62, 62 + 290 + 104, 34, 92
FLOOR = 1.05
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    smax = max(p[0] for v in D.values() for p in v["loss"])
    XS = 12500; LY0, LY1 = 0.02, 0.12
    sx1 = lambda x: L1 + x / XS * PW
    sy1 = lambda y: H - B - (y - LY0) / (LY1 - LY0) * (H - T - B)
    sx2 = lambda x: L2 + x / XS * PW
    sy2 = lambda y: H - B - (math.log10(y) - 0) / (3 - 0) * (H - T - B)
    for y in (0.02, 0.04, 0.06, 0.08, 0.1, 0.12):
        o.append(f'<line x1="{L1}" x2="{L1+PW}" y1="{sy1(y):.1f}" y2="{sy1(y):.1f}" stroke="{grid}"/><text x="{L1-8}" y="{sy1(y)+4:.1f}" fill="{mut}" font-size="12.5" text-anchor="end">{y:g}</text>')
    for x in (0, 2500, 5000, 7500, 10000, 12500):
        o.append(f'<line x1="{sx1(x):.1f}" x2="{sx1(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/><text x="{sx1(x):.1f}" y="{H-B+18}" fill="{mut}" font-size="12.5" text-anchor="middle">{x:,}</text>')
    for y in (1, 10, 100, 1000):
        o.append(f'<line x1="{L2}" x2="{L2+PW}" y1="{sy2(y):.1f}" y2="{sy2(y):.1f}" stroke="{grid}"/><text x="{L2-8}" y="{sy2(y)+4:.1f}" fill="{mut}" font-size="12.5" text-anchor="end">{y:,}</text>')
    for x in (0, 2500, 5000, 7500, 10000, 12500):
        o.append(f'<line x1="{sx2(x):.1f}" x2="{sx2(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/><text x="{sx2(x):.1f}" y="{H-B+18}" fill="{mut}" font-size="12.5" text-anchor="middle">{x:,}</text>')
    o.append(f'<text x="{L1-4}" y="{T-14}" fill="{ink}" font-size="13.5" font-weight="bold">Training loss (noise MSE)</text>')
    o.append(f'<text x="{L2-4}" y="{T-14}" fill="{ink}" font-size="13.5" font-weight="bold">FD-FMNIST (log scale, lower is better)</text>')
    o.append(f'<text x="{L1+PW/2}" y="{H-B+40}" fill="{ink}" font-size="13.5" text-anchor="middle">Training steps (batch 128)</text>')
    o.append(f'<text x="{L2+PW/2}" y="{H-B+40}" fill="{ink}" font-size="13.5" text-anchor="middle">Training steps (batch 128)</text>')
    o.append(f'<line x1="{L2}" x2="{L2+PW}" y1="{sy2(FLOOR):.1f}" y2="{sy2(FLOOR):.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/>')
    for ki, ((k, v), c) in enumerate(zip(D.items(), cols)):
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{" ".join(f"{sx1(s):.1f},{sy1(min(max(l, LY0), LY1)):.1f}" for s, l in v["loss"])}"/>')
        pts = [(s, fd) for s, ts, fd in v["fd"] if fd < 1000]
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx2(x):.1f},{sy2(y):.1f}" for x, y in pts)}"/>')
        for x, y in pts:
            o.append(f'<circle cx="{sx2(x):.1f}" cy="{sy2(y):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        x, y = pts[-1]
        o.append(f'<text x="{sx2(x)+2:.1f}" y="{sy2(y)+(22 if ki == 0 else -10):.1f}" fill="{c}" font-size="12.5" text-anchor="middle">{y:.1f}</text>')
    y = H - 22; x = L1
    for (k, c) in zip(D, cols):
        o.append(f'<line x1="{x}" x2="{x+26}" y1="{y-4}" y2="{y-4}" stroke="{c}" stroke-width="2.5"/><text x="{x+34}" y="{y}" fill="{ink}" font-size="13">{NAMES[k]}</text>'); x += 220
    o.append(f'<line x1="{x}" x2="{x+26}" y1="{y-4}" y2="{y-4}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/><text x="{x+34}" y="{y}" fill="{ink}" font-size="13">2,000 real images (1.05)</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926')))
