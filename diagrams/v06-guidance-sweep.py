# Generates public/diagrams/v06-guidance-sweep.svg and -dark.svg (data: v06-latent-diffusion.md).
# Stable Diffusion 1.5 (fp16, DPM-Solver++ 2M, 20 steps, 512 x 512) on 24 own prompts x 3 seeds per guidance scale.
# Panels: CLIP score (openai/clip-vit-base-patch32, 100 x cosine), diversity (mean pairwise cosine distance of
# CLIP image embeddings across the 3 seeds of a prompt), and the share of images Claude Haiku 5.5 (thinking off)
# judged to show everything in the prompt. Bars: 95% cluster-bootstrap intervals over prompts.
# Data from scratch v06/analyze.py -> runs/results.json.
# Usage: python3 diagrams/v06-guidance-sweep.py public/diagrams/v06-guidance-sweep
import sys
DATA = {"1.0": {"clip": [29.3573, 28.2829, 30.4111], "diversity": [0.2354, 0.2145, 0.2561], "judge_yes": [0.1319, 0.0486, 0.2361]}, "3.0": {"clip": [33.5177, 32.6855, 34.4293], "diversity": [0.1341, 0.1122, 0.1566], "judge_yes": [0.4583, 0.2917, 0.625]}, "5.0": {"clip": [33.9803, 33.1583, 34.8708], "diversity": [0.1103, 0.0872, 0.1364], "judge_yes": [0.625, 0.4861, 0.7639]}, "7.5": {"clip": [34.2688, 33.4736, 35.1031], "diversity": [0.0872, 0.0712, 0.1051], "judge_yes": [0.6597, 0.5139, 0.8056]}, "12.0": {"clip": [34.2601, 33.453, 35.1622], "diversity": [0.0874, 0.0711, 0.1049], "judge_yes": [0.6528, 0.5, 0.7917]}, "20.0": {"clip": [34.383, 33.6155, 35.1981], "diversity": [0.0805, 0.0703, 0.0913], "judge_yes": [0.6528, 0.5, 0.7917]}}
GS = ["1.0", "3.0", "5.0", "7.5", "12.0", "20.0"]
PANELS = [("clip", "CLIP score", 1, "{:.0f}", (27, 36), [28, 30, 32, 34, 36]),
          ("diversity", "Diversity across seeds (x 100)", 100, "{:.0f}", (0, 30), [0, 10, 20, 30]),
          ("judge_yes", "Judged to match the prompt", 100, "{:.0f}%", (0, 100), [0, 25, 50, 75, 100])]
W, H = 780, 330
PW, PH, GX = 200, 200, 58
def svg(th, acc):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for k, (key, title, mul, fmt, (y0, y1), ticks) in enumerate(PANELS):
        ox, oy = 52 + k * (PW + GX), 52
        vals = [[v * mul for v in DATA[g][key]] for g in GS]
        sx = lambda i: ox + 14 + i * (PW - 28) / (len(GS) - 1)
        sy = lambda y: oy + PH - (y - y0) / (y1 - y0) * PH
        o.append(f'<text x="{ox}" y="{oy-20}" fill="{ink}" font-size="13.5" font-weight="bold">{title}</text>')
        for y in ticks:
            o.append(f'<line x1="{ox}" x2="{ox+PW}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{ox-6}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="11.5" text-anchor="end">{fmt.format(y)}</text>')
        for i, g in enumerate(GS):
            o.append(f'<text x="{sx(i):.1f}" y="{oy+PH+18}" fill="{mut}" font-size="11.5" text-anchor="middle">{float(g):g}</text>')
        o.append(f'<text x="{ox+PW/2}" y="{oy+PH+36}" fill="{mut}" font-size="12" text-anchor="middle">guidance scale</text>')
        o.append(f'<polyline fill="none" stroke="{acc}" stroke-width="2.2" points="{" ".join(f"{sx(i):.1f},{sy(v[0]):.1f}" for i, v in enumerate(vals))}"/>')
        for i, v in enumerate(vals):
            o.append(f'<line x1="{sx(i):.1f}" x2="{sx(i):.1f}" y1="{sy(v[1]):.1f}" y2="{sy(v[2]):.1f}" stroke="{acc}" stroke-width="1.6"/>')
            o.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v[0]):.1f}" r="4" fill="{acc}" stroke="{surf}" stroke-width="1.2"/>')
    o.append(f'<text x="52" y="{H-12}" fill="{mut}" font-size="12">70 to 72 images per scale (24 prompts x 3 seeds, minus 5 flagged); bars: 95% bootstrap intervals over prompts.</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, '#0b5fd6'))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, '#38bdf8'))
