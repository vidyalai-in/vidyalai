# Generates public/diagrams/a05-context-growth.svg and a05-context-growth-dark.svg (data: a05-context-engineering.md).
# Tokens the model read at each step of one 32-incident chain task (c10, repeat 0), claude-haiku-5-5, effort medium,
# one run per strategy, from the run transcripts. Usage: python3 diagrams/a05-context-growth.py public/diagrams/a05-context-growth
import sys
W, H = 780, 470; L, R, T, B = 78, 24, 24, 62
X0, X1 = 1, 44; Y0, Y1 = 0, 80
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
# Thousands of input tokens per model call (after any clearing or compaction).
SERIES = [
    ("Full results, full history", [0.77, 3.11, 5.29, 7.41, 9.7, 11.9, 14.16, 16.33, 18.52, 20.71, 22.98, 25.28, 27.48, 29.71, 31.87, 34.1, 36.34, 38.66, 40.98, 43.17, 45.31, 47.6, 49.8, 52.33, 54.6, 56.77, 59.01, 61.35, 63.54, 65.74, 67.93, 70.13, 72.64]),
    ("Compaction at 50K", [0.81, 3.15, 5.33, 7.45, 9.74, 11.94, 14.2, 16.37, 18.56, 20.75, 23.02, 25.32, 27.52, 29.75, 31.91, 34.15, 36.39, 38.7, 41.02, 43.21, 45.53, 47.81, 1.66, 4.14, 6.5, 8.75, 11.07, 13.49, 15.74, 18.02, 20.3, 22.58, 24.84]),
    ("Memory file, fresh window at 30K", [2.28, 2.4, 4.75, 6.93, 9.05, 11.6, 14.02, 16.5, 18.88, 21.29, 23.69, 26.17, 28.7, 31.11, 2.3, 2.43, 2.86, 5.09, 7.49, 9.94, 12.4, 14.93, 17.47, 19.88, 22.23, 24.73, 27.15, 29.57, 32.06, 2.3, 2.43, 3.13, 5.31, 7.79, 10.35, 12.73, 15.15, 17.56, 19.98, 22.37, 22.98]),
    ("Context editing at 30K", [0.77, 3.11, 5.29, 7.41, 9.7, 11.9, 14.16, 16.33, 18.52, 20.71, 22.98, 25.28, 27.56, 10.67, 12.82, 15.06, 17.3, 19.61, 21.94, 24.13, 29.5, 18.38, 16.4, 19.0, 21.35, 23.52, 25.84, 18.29, 20.51, 22.75, 24.96, 27.24, 19.1, 21.8, 24.38, 26.73, 28.78, 20.37, 22.58, 24.84, 27.08, 29.95, 21.38, 23.64]),
    ("Sized tool results", [1.0, 1.23, 1.46, 1.69, 1.92, 2.15, 2.38, 2.6, 2.82, 3.04, 3.27, 3.5, 3.73, 3.95, 4.17, 4.4, 4.63, 4.86, 5.1, 5.32, 5.54, 5.77, 5.99, 6.21, 6.44, 6.67, 6.89, 7.13, 7.36, 7.58, 7.81, 8.03, 8.25]),
]
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(0, 81, 10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}K</text>')
    for x in (1, 10, 20, 30, 40, 44):
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    for y, lab in ((30, "editing and memory threshold"), (50, "compaction trigger")):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{mut}" stroke-width="1" stroke-dasharray="5 4"/>')
        o.append(f'<text x="{W-R-4}" y="{sy(y)-5:.1f}" fill="{mut}" font-size="12" text-anchor="end">{lab}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Agent step (model call)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Tokens the model read (thousands)</text>')
    for (name, ys), c in zip(SERIES, cols):
        pts = " ".join(f"{sx(i+1):.1f},{sy(v):.1f}" for i, v in enumerate(ys))
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" points="{pts}"/>')
    for k, ((name, _), c) in enumerate(zip(SERIES, cols)):
        y = T + 14 + k * 21
        o.append(f'<line x1="{L+14}" x2="{L+38}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="3"/>')
        o.append(f'<text x="{L+46}" y="{y+5}" fill="{ink}" font-size="13">{name}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6', '#1baf7a', '#9b5de5', '#4a5b78')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5', '#199e70', '#b07cf0', '#9fb0cf')))
