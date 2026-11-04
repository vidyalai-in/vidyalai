# Generates public/diagrams/v04-text-height.svg and -dark.svg (data: v04-vision-language-models.md).
# Exact-match accuracy at reading a random 5-character code, against the font size of the code in the image the
# model received (font size at 896 px times size / 896), grouped into five bins (3-4, 5-6.5, 8-10, 12-20, 24-68 px)
# plotted at each bin's geometric middle. 40 to 220 items per point, 95% Wilson intervals.
# Data from scratch v04/analyze.py -> results.json (printed by make_chart_data.py).
# Usage: python3 diagrams/v04-text-height.py public/diagrams/v04-text-height
import sys, math
HEIGHT = {"Claude Haiku 5.5": [[3.46, 0.0, 0.0, 8.8, 40], [5.7, 41.7, 30.1, 54.3, 60], [8.94, 86.7, 75.8, 93.1, 60], [15.49, 100.0, 96.9, 100.0, 120], [40.4, 100.0, 98.3, 100.0, 220]], "Claude Sonnet 5.5": [[3.46, 15.0, 7.1, 29.1, 40], [5.7, 70.0, 57.5, 80.1, 60], [8.94, 95.0, 86.3, 98.3, 60], [15.49, 100.0, 96.3, 100.0, 100], [40.4, 100.0, 91.2, 100.0, 40]], "Qwen2-VL-2B": [[3.46, 0.0, 0.0, 8.8, 40], [5.7, 33.3, 22.7, 45.9, 60], [8.94, 86.7, 75.8, 93.1, 60], [15.49, 98.0, 93.0, 99.4, 100], [40.4, 100.0, 91.2, 100.0, 40]], "SmolVLM-256M": [[3.46, 2.5, 0.4, 12.9, 40], [5.7, 43.3, 31.6, 55.9, 60], [8.94, 80.0, 68.2, 88.2, 60], [15.49, 84.0, 75.6, 89.9, 100], [40.4, 92.5, 80.1, 97.4, 40]]}
NAMES = ["Claude Haiku 5.5", "Claude Sonnet 5.5", "Qwen2-VL-2B", "SmolVLM-256M"]
W, H = 780, 470
L, R, T, B = 78, 210, 30, 62
X0, X1 = 2.8, 55
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - y / 100 * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0, 25, 50, 75, 100):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for x, lab in ((3.46, "3-4"), (5.7, "5-6.5"), (8.94, "8-10"), (15.49, "12-20"), (40.4, "24-68")):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Font size of the code in the image the model sees (pixels, binned, log scale)</text>')
    o.append(f'<text transform="translate(22,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Codes read exactly</text>')
    for k, name in enumerate(NAMES):
        pts = HEIGHT.get(name)
        if not pts:
            continue
        c = cols[k]
        off = (k - 1.5) * 0.008
        line = " ".join(f"{sx(x * 10 ** off):.1f},{sy(p):.1f}" for x, p, lo, hi, n in pts)
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" points="{line}"/>')
        for x, p, lo, hi, n in pts:
            X = sx(x * 10 ** off)
            o.append(f'<line x1="{X:.1f}" x2="{X:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="1.4"/>')
            o.append(f'<circle cx="{X:.1f}" cy="{sy(p):.1f}" r="4" fill="{c}" stroke="{surf}" stroke-width="1.2"/>')
        y = T + 20 + k * 26
        o.append(f'<line x1="{W-R+22}" x2="{W-R+48}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"/>')
        o.append(f'<text x="{W-R+56}" y="{y+5}" fill="{ink}" font-size="13">{name}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#1f9e6e', '#eb6834', '#8a5cd6')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#2fb380', '#d95926', '#9b72e0')))
