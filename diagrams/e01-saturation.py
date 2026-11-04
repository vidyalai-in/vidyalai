# Generates public/diagrams/e01-saturation.svg and e01-saturation-dark.svg (data: e01-why-llm-evals-are-hard.md).
# Zero-shot multiple-choice accuracy with 95% Wilson intervals on four item sets of rising difficulty.
# Claude Haiku 5.5 (thinking off): all items. Claude Sonnet 5.5 (thinking off): the same random 200 items
# of three sets. Open models (bf16, RTX 4060 Laptop GPU): highest log-probability option letter, all items.
# Data from scratch e01/scripts/analyze.py (results.json).
# Usage: python3 diagrams/e01-saturation.py public/diagrams/e01-saturation, then render the PNG with sharp at density 144.
import sys

W, H = 780, 500; L, R, T, B = 70, 30, 24, 96
SETS = ["ARC-Easy", "ARC-Challenge", "MMLU", "MMLU-Pro"]
CHANCE = [25.02, 25.02, 25.00, 11.16]
# label, [(acc, lo, hi) or None per set]
MODELS = [
    ("SmolLM2-135M", [(24.66, 22.97, 26.44), (24.15, 21.78, 26.68), (24.04, 21.64, 26.60), (11.19, 9.23, 13.50)]),
    ("Qwen2.5-0.5B", [(66.33, 64.40, 68.20), (50.94, 48.08, 53.79), (45.70, 42.83, 48.60), (15.60, 13.30, 18.20)]),
    ("Claude Haiku 5.5", [(98.23, 97.62, 98.69), (93.00, 91.40, 94.33), (85.26, 83.09, 87.20), (70.95, 67.79, 73.92)]),
    ("Claude Sonnet 5.5 (200 items)", [(100.0, 98.12, 100.0), None, (92.50, 88.00, 95.40), (88.00, 82.77, 91.80)]),
]
Y0, Y1 = 0, 100
step = (W - L - R) / len(SETS)
def cx(i): return L + step * (i + 0.5)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
OFF = [-36, -12, 12, 36]

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(0, 101, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Accuracy (95% interval)</text>')
    for i, (s, ch) in enumerate(zip(SETS, CHANCE)):
        o.append(f'<text x="{cx(i):.1f}" y="{H-B+22}" fill="{ink}" font-size="14" text-anchor="middle">{s}</text>')
        o.append(f'<line x1="{cx(i)-58:.1f}" x2="{cx(i)+58:.1f}" y1="{sy(ch):.1f}" y2="{sy(ch):.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{cx(0)-58:.1f}" y="{sy(25)-10:.1f}" fill="{mut}" font-size="12">chance</text>')
    for (lab, vals), c, dx in zip(MODELS, cols, OFF):
        pts = [(cx(i) + dx, sy(v[0])) for i, v in enumerate(vals) if v]
        if len(pts) > 1 and vals[1] is not None:
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.5" stroke-opacity="0.6" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        elif len(pts) > 1:
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.5" stroke-opacity="0.6" stroke-dasharray="3 4" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        for i, v in enumerate(vals):
            if not v:
                continue
            x = cx(i) + dx
            a, lo, hi = v
            o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
            for e in (lo, hi):
                o.append(f'<line x1="{x-4:.1f}" x2="{x+4:.1f}" y1="{sy(e):.1f}" y2="{sy(e):.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{x:.1f}" cy="{sy(a):.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
    # legend, two rows under the axis labels
    lx, ly = L + 10, H - 44
    for k, ((lab, _), c) in enumerate(zip(MODELS, cols)):
        x = lx + (k % 2) * 330
        y = ly + (k // 2) * 22
        o.append(f'<circle cx="{x+6}" cy="{y-4}" r="5" fill="{c}"/>')
        o.append(f'<text x="{x+18}" y="{y+1}" fill="{ink}" font-size="13">{lab}</text>')
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#8a5cd0', '#1baf7a', '#2a78d6', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#9b6fe0', '#199e70', '#3987e5', '#d95926')))
