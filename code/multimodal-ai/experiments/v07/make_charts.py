"""Writes diagrams/v07-steps.py and diagrams/v07-dit-vs-unet.py with data embedded from runs/results.json."""
import json
from pathlib import Path

R = json.load(open("/home/amit/.cache/vidyalai-scratch/v07/runs/results.json"))
REPO = Path("/home/amit/AlpsCode/ai/vidyalai/diagrams")

steps = {m: [[int(s), round(v["mean"], 2), round(v["min"], 2), round(v["max"], 2)] for s, v in R["sweep"][m].items()]
         for m in ("ddpm-unet", "rf-unet", "reflow-unet") if m in R["sweep"]}
STEPS = r'''# Generates public/diagrams/v07-steps.svg and -dark.svg (data: v07-flow-matching-dit.md).
# FD-FMNIST (Frechet distance on a small Fashion-MNIST classifier's features, not Inception; lower is better) against the
# number of sampling steps (network calls) for three 4.1M-parameter U-Nets trained on Fashion-MNIST for 4,000 steps of batch
# 128 (same network, same compute): DDPM noise prediction sampled with deterministic DDIM, rectified flow sampled with Euler
# steps, and the rectified flow after one reflow round (1,500 more steps on 20,000 of its own noise-sample pairs).
# Mean of 3 sampling seeds x 2,000 samples; bars span the 3 seeds. Dashed: 2,000 real training images (1.05).
# Data from scratch v07/evalgen.py -> runs/sweep.jsonl. Usage: python3 diagrams/v07-steps.py public/diagrams/v07-steps
import sys, json, math
D = json.loads('@@D')
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
'''
(REPO / "v07-steps.py").write_text(STEPS.replace("@@D", json.dumps(steps, separators=(",", ":"))))

if "snaps" in R and R["snaps"]:
    sn = {m: [[round(p["train_time_s"] / 60, 3), p["fd"]] for p in R["snaps"][m]] for m in ("rf-unet", "rf-dit", "ddpm-unet") if m in R["snaps"]}
    tm = R.get("timing", {})
    DIT = r'''# Generates public/diagrams/v07-dit-vs-unet.svg and -dark.svg (data: v07-flow-matching-dit.md).
# FD-FMNIST (lower is better) of EMA snapshots every 500 training steps against training time, for a 4.10M-parameter U-Net
# and a 4.16M-parameter DiT (2x2 patches, 196 tokens, width 192, 6 blocks) trained with the rectified flow objective,
# plus the U-Net trained with the DDPM objective for comparison. Batch 128, AdamW 3e-4, bf16, RTX 4060 Laptop GPU.
# Each point: 2,000 samples, 16 sampling steps (Euler for flow, DDIM for DDPM), same noise for every snapshot.
# Data from scratch v07/evalgen.py snaps -> runs/snaps.jsonl. Usage: python3 diagrams/v07-dit-vs-unet.py public/diagrams/v07-dit-vs-unet
import sys, json, math
D = json.loads('@@D')
NAMES = {"rf-unet": "U-Net, rectified flow", "rf-dit": "DiT, rectified flow", "ddpm-unet": "U-Net, DDPM (DDIM sampler)"}
W, H = 780, 470; L, R, T, B = 70, 30, 24, 62
X0, X1 = 0, @@XMAX; Y0, Y1 = 8, 400
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
'''
    xmax = max(p[0] for v in sn.values() for p in v)
    (REPO / "v07-dit-vs-unet.py").write_text(DIT.replace("@@D", json.dumps(sn, separators=(",", ":"))).replace("@@XMAX", str(int(xmax) + 1)))
print("ok")
