# Generates public/diagrams/v04-accuracy-by-size.svg and -dark.svg (data: v04-vision-language-models.md).
# Accuracy (%) with 95% Wilson intervals on five synthetic tasks, 100 items per task and size, square images
# drawn at 896 px and resized. Claude Haiku 5.5 (thinking off) and Sonnet 5.5 via the Batch API; Qwen2-VL-2B-Instruct
# and SmolVLM-256M-Instruct in bf16 on an RTX 4060 Laptop GPU, greedy decoding. Data from scratch v04/analyze.py ->
# results.json (printed by make_chart_data.py).
# Usage: python3 diagrams/v04-accuracy-by-size.py public/diagrams/v04-accuracy-by-size
import sys, math
ACC = {"absent": {"Claude Haiku 5.5": [[224, 100.0, 96.3, 100.0], [448, 100.0, 96.3, 100.0], [896, 100.0, 96.3, 100.0], [1344, 100.0, 96.3, 100.0], [1792, 100.0, 96.3, 100.0]], "Qwen2-VL-2B": [[224, 96.0, 90.2, 98.4], [448, 98.0, 93.0, 99.5], [896, 99.0, 94.5, 99.8]], "SmolVLM-256M": [[224, 59.0, 49.2, 68.1], [448, 62.0, 52.2, 70.9], [896, 63.0, 53.2, 71.8]]}, "chart": {"Claude Haiku 5.5": [[224, 100.0, 96.3, 100.0], [448, 100.0, 96.3, 100.0], [896, 100.0, 96.3, 100.0], [1344, 100.0, 96.3, 100.0], [1792, 100.0, 96.3, 100.0]], "Qwen2-VL-2B": [[224, 66.0, 56.3, 74.5], [448, 100.0, 96.3, 100.0], [896, 94.0, 87.5, 97.2]], "SmolVLM-256M": [[224, 90.0, 82.6, 94.5], [448, 93.0, 86.2, 96.6], [896, 88.0, 80.2, 93.0]]}, "count": {"Claude Haiku 5.5": [[224, 98.0, 93.0, 99.5], [448, 100.0, 96.3, 100.0], [896, 99.0, 94.5, 99.8], [1344, 99.0, 94.5, 99.8], [1792, 99.0, 94.5, 99.8]], "Claude Sonnet 5.5": [[224, 100.0, 96.3, 100.0], [448, 100.0, 96.3, 100.0], [896, 100.0, 96.3, 100.0]], "Qwen2-VL-2B": [[224, 14.0, 8.5, 22.1], [448, 24.0, 16.7, 33.2], [896, 27.0, 19.3, 36.4]], "SmolVLM-256M": [[224, 21.0, 14.2, 30.0], [448, 24.0, 16.7, 33.2], [896, 21.0, 14.2, 30.0]]}, "dense": {"Claude Haiku 5.5": [[224, 44.0, 34.7, 53.8], [448, 97.0, 91.5, 99.0], [896, 100.0, 96.3, 100.0], [1344, 98.0, 93.0, 99.5], [1792, 94.0, 87.5, 97.2]], "Claude Sonnet 5.5": [[224, 61.0, 51.2, 70.0], [448, 100.0, 96.3, 100.0], [896, 100.0, 96.3, 100.0]], "Qwen2-VL-2B": [[224, 0.0, 0.0, 3.7], [448, 4.0, 1.6, 9.8], [896, 12.0, 7.0, 19.8]], "SmolVLM-256M": [[224, 1.0, 0.2, 5.5], [448, 0.0, 0.0, 3.7], [896, 0.0, 0.0, 3.7]]}, "read": {"Claude Haiku 5.5": [[224, 43.0, 33.7, 52.8], [448, 74.0, 64.6, 81.6], [896, 100.0, 96.3, 100.0], [1344, 100.0, 96.3, 100.0], [1792, 100.0, 96.3, 100.0]], "Claude Sonnet 5.5": [[224, 52.0, 42.3, 61.5], [448, 93.0, 86.2, 96.6], [896, 100.0, 96.3, 100.0]], "Qwen2-VL-2B": [[224, 32.0, 23.7, 41.7], [448, 80.0, 71.1, 86.7], [896, 98.0, 93.0, 99.5]], "SmolVLM-256M": [[224, 36.0, 27.3, 45.8], [448, 75.0, 65.7, 82.5], [896, 85.0, 76.7, 90.7]]}}
TASKS = [("count", "Count colored shapes (1-12)"), ("dense", "Count small dots (12-40)"),
         ("read", "Read a 5-character code"), ("chart", "Read a bar value (within 5)"),
         ("absent", "Is there a <color> <shape>? (half no)")]
NAMES = ["Claude Haiku 5.5", "Claude Sonnet 5.5", "Qwen2-VL-2B", "SmolVLM-256M"]
W, H = 780, 900
COLS, ROWS = 2, 3
PW, PH = 300, 210
GX, GY = 64, 40
X0, X1 = 190, 2100
def panel_origin(i):
    c, r = i % COLS, i // COLS
    return 66 + c * (PW + GX), 40 + r * (PH + 80)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for i, (task, title) in enumerate(TASKS):
        ox, oy = panel_origin(i)
        sx = lambda x: ox + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * PW
        sy = lambda y: oy + PH - y / 100 * PH
        o.append(f'<text x="{ox}" y="{oy-12}" fill="{ink}" font-size="14" font-weight="bold">{title.replace("<", "&lt;").replace(">", "&gt;")}</text>')
        for y in (0, 25, 50, 75, 100):
            o.append(f'<line x1="{ox}" x2="{ox+PW}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{ox-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{y}%</text>')
        for x in (224, 448, 896, 1792):
            o.append(f'<text x="{sx(x):.1f}" y="{oy+PH+18}" fill="{mut}" font-size="12" text-anchor="middle">{x}</text>')
        o.append(f'<text x="{ox+PW/2}" y="{oy+PH+36}" fill="{mut}" font-size="12" text-anchor="middle">image side in pixels (log scale)</text>')
        for k, name in enumerate(NAMES):
            pts = ACC.get(task, {}).get(name)
            if not pts:
                continue
            c = cols[k]
            off = (k - 1.5) * 4
            line = " ".join(f"{sx(x)+off:.1f},{sy(p):.1f}" for x, p, lo, hi in pts)
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" points="{line}"/>')
            for x, p, lo, hi in pts:
                X = sx(x) + off
                o.append(f'<line x1="{X:.1f}" x2="{X:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="1.5"/>')
                o.append(f'<circle cx="{X:.1f}" cy="{sy(p):.1f}" r="3.8" fill="{c}" stroke="{surf}" stroke-width="1.2"/>')
    # legend in the sixth slot
    ox, oy = panel_origin(5)
    o.append(f'<text x="{ox}" y="{oy+6}" fill="{ink}" font-size="14" font-weight="bold">Model</text>')
    for k, name in enumerate(NAMES):
        y = oy + 34 + k * 26
        o.append(f'<line x1="{ox}" x2="{ox+28}" y1="{y}" y2="{y}" stroke="{cols[k]}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{ox+14}" cy="{y}" r="4" fill="{cols[k]}"/>')
        o.append(f'<text x="{ox+38}" y="{y+5}" fill="{ink}" font-size="13">{name}</text>')
    o.append(f'<text x="{ox}" y="{oy+150}" fill="{mut}" font-size="12">100 items per point; bars are 95% intervals.</text>')
    o.append(f'<text x="{ox}" y="{oy+168}" fill="{mut}" font-size="12">Sonnet ran on the three counting and</text>')
    o.append(f'<text x="{ox}" y="{oy+184}" fill="{mut}" font-size="12">reading tasks only, at 224-896 px.</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#1f9e6e', '#eb6834', '#8a5cd6')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#2fb380', '#d95926', '#9b72e0')))
