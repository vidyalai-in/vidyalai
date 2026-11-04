# Generates public/diagrams/v01-token-ledger.svg and v01-token-ledger-dark.svg (data: v01-multimodal-ai-overview.md).
# Every value comes from the v01 runs (scratch v01/runs/local_*.json, claude_count.json) or from the stated arithmetic.
# Usage: python3 diagrams/v01-token-ledger.py public/diagrams/v01-token-ledger, then render the PNG with sharp at density 144.
import math
W, H = 780, 640
L, R, T, B = 250, 24, 40, 56
X0, X1 = 1, 1e8
# (group, label, sequence positions, numbers stored or None)
rows = [
    ("Image: one 512 × 512 picture", None, None, None),
    (None, "Raw pixels (RGB)", 262144, 786432),
    (None, "ViT-B/16 (resized to 224)", 197, 197 * 768),
    (None, "SmolVLM-256M tokens", 1088, 1088 * 576),
    (None, "Claude visual tokens", 361, None),
    (None, "Qwen2-VL-2B tokens", 324, 324 * 1536),
    (None, "Stable Diffusion VAE latent", 64 * 64, 4 * 64 * 64),
    (None, "VQ-GAN codes (aMUSEd)", 1024, 1024),
    ("Audio: one second", None, None, None),
    (None, "Raw samples, 16 kHz", 16000, 16000),
    (None, "Whisper log-mel frames", 100, 8000),
    (None, "Whisper-tiny encoder, 1-s clip", 1500, 1500 * 384),
    (None, "Mimi codec, 8 codebooks", 100, 100),
    (None, "Text transcript (Moshi paper)", 3.5, None),
    ("Video: one second at 24 fps", None, None, None),
    (None, "Raw pixels, 512 × 512 frames", 24 * 262144, 24 * 786432),
    (None, "ViT-B/16 on every frame", 24 * 197, 24 * 197 * 768),
    (None, "SD VAE latent on every frame", 24 * 4096, 24 * 16384),
    (None, "Qwen2-VL-2B, 2 fps at 448", 256, 256 * 1536),
]
step = (H - T - B) / len(rows)


def sx(x):
    return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)


def svg(th):
    ink, mut, grid, acc, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for e in range(0, 9):
        x = 10 ** e
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H - B}" stroke="{grid}" stroke-width="1"/>')
        lab = {0: "1", 1: "10", 2: "100", 3: "1K", 4: "10K", 5: "100K", 6: "1M", 7: "10M", 8: "100M"}[e]
        o.append(f'<text x="{sx(x):.1f}" y="{H - B + 20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 12}" fill="{ink}" font-size="14" text-anchor="middle">Count (log scale)</text>')
    # legend
    o.append(f'<circle cx="{L + 8}" cy="{T - 22}" r="6" fill="{acc}"/>')
    o.append(f'<text x="{L + 20}" y="{T - 17}" fill="{ink}" font-size="13">sequence positions (vectors or tokens)</text>')
    o.append(f'<circle cx="{L + 300}" cy="{T - 22}" r="6" fill="none" stroke="{ink}" stroke-width="2"/>')
    o.append(f'<text x="{L + 312}" y="{T - 17}" fill="{ink}" font-size="13">numbers stored</text>')
    for i, (g, lab, pos, val) in enumerate(rows):
        y = T + step * (i + 0.5)
        if g:
            o.append(f'<text x="12" y="{y + 5:.1f}" fill="{ink}" font-size="14" font-weight="bold">{g}</text>')
            continue
        o.append(f'<text x="{L - 10}" y="{y + 4.5:.1f}" fill="{ink}" font-size="13" text-anchor="end">{lab}</text>')
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{grid}" stroke-width="1" stroke-dasharray="2 4"/>')
        if val:
            o.append(f'<line x1="{sx(pos):.1f}" x2="{sx(val):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{mut}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(val):.1f}" cy="{y:.1f}" r="6" fill="{surf}" stroke="{ink}" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(pos):.1f}" cy="{y:.1f}" r="6" fill="{acc}"/>')
    o.append('</svg>')
    return '\n'.join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d')
import sys
open(sys.argv[1] + '.svg', 'w').write(svg(light))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
