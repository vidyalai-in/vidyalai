# Generates public/diagrams/v12-training-loss.svg and -dark.svg (data: v12-video-generation.md).
# Rectified-flow training loss (mean squared error of the predicted velocity, averaged over every 25th step in each window of 250)
# for the tiny spacetime-patch transformers trained on synthetic moving Fashion-MNIST clips (8 frames of 48 x 48, batch 32).
# Every batch is freshly generated, so the training loss is also a held-out loss. "Inflated" starts from the frame-by-frame
# model's weights with new temporal attention layers whose gates start at zero; its steps count from the start of fine-tuning.
# Data from scratch v12/train.py -> runs/train-*.jsonl. Usage: python3 diagrams/v12-training-loss.py public/diagrams/v12-training-loss
import sys, json
D = json.loads('{"frames": [[250, 0.83596], [500, 0.21575], [750, 0.18448], [1000, 0.15679], [1250, 0.15519], [1500, 0.14441], [1750, 0.14221], [2000, 0.13535], [2250, 0.12884], [2500, 0.12464], [2750, 0.12655], [3000, 0.12479], [3250, 0.11812], [3500, 0.11784], [3750, 0.12067], [4000, 0.11657]], "factorized": [[250, 0.84459], [500, 0.21029], [750, 0.172], [1000, 0.14296], [1250, 0.13678], [1500, 0.12644], [1750, 0.12499], [2000, 0.12087], [2250, 0.1121], [2500, 0.11062], [2750, 0.11042], [3000, 0.10997], [3250, 0.10391], [3500, 0.10801], [3750, 0.10663], [4000, 0.103]], "full": [[250, 0.83847], [500, 0.22312], [750, 0.17749], [1000, 0.14473], [1250, 0.13947], [1500, 0.12789], [1750, 0.12408], [2000, 0.11601], [2250, 0.10804], [2500, 0.10366], [2750, 0.10107], [3000, 0.10033], [3250, 0.09453], [3500, 0.09976], [3750, 0.10419], [4000, 0.09024]], "inflated": [[250, 0.1106], [500, 0.11678], [750, 0.10972], [1000, 0.09831], [1200, 0.10045]]}')
NAMES = {"frames": "Frame by frame (no temporal attention)", "factorized": "Factorized, from scratch",
         "full": "Full spacetime attention, from scratch", "inflated": "Factorized, inflated from the frame model"}
W, H = 780, 460; L, R, T, B = 70, 30, 24, 62
X0, X1 = 0, 4000; Y0, Y1 = 0.08, 0.24
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:.2f}</text>')
    for x in (0, 1000, 2000, 3000, 4000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:,}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Training steps (batches of 32 clips)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Flow-matching loss (lower is better)</text>')
    for (m, c) in zip(NAMES, cols):
        pts = [p for p in D[m] if Y0 <= p[1] <= Y1]
        dash = ' stroke-dasharray="7 4"' if m == "inflated" else ""
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5"{dash} points="{" ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)}"/>')
        e = D[m][-1]
        o.append(f'<circle cx="{sx(e[0]):.1f}" cy="{sy(e[1]):.1f}" r="4" fill="{c}"/>')
    lx, ly = sx(1700), sy(0.235)
    for k, (m, c) in enumerate(zip(NAMES, cols)):
        y = ly + k * 22
        dash = ' stroke-dasharray="7 4"' if m == "inflated" else ""
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"{dash}/><text x="{lx+34}" y="{y+5}" fill="{ink}" font-size="13">{NAMES[m]}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#1f9e74', '#2a78d6', '#8a5cd0')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#2bb38a', '#3987e5', '#a07ee0')))
