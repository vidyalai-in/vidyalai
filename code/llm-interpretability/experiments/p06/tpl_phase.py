# Generates public/diagrams/p06-phase-diagram.svg and -dark.svg (data: p06-superposition.md).
# Usage: python3 diagrams/p06-phase-diagram.py public/diagrams/p06-phase-diagram, then render the PNG with sharp at density 144.
# Data: ReLU output toy models. Left: 2 features in 1 dimension; right: 3 features in 2 dimensions. The last ("extra")
# feature has importance 0.1 to 10 (others 1); all features share one density, 1 to 0.01. 21 x 21 grid, 10 models per
# cell; each cell shows the outcome for the extra feature in the lowest-loss model of the 10:
# not learned (||W|| < 0.5), in superposition (learned, sum of squared overlaps with the others > 0.1), or its own dimension.
# lab/toy.py, p06 scripts/toy_runs.py phase2 and phase3; 10,000 AdamW steps, lr 1e-3, batch 1024.
import math, sys
DATA = __DATA__
W = 780; L = 70; TOP = 50; CELL = 14; GAP = 70; N = 21
PW = CELL * N
H = TOP + CELL * N + 118
def mix(a, b, t):
    a = [int(a[i:i+2], 16) for i in (1, 3, 5)]; b = [int(b[i:i+2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(a, b))
LEVEL = [0.0, 0.45, 1.0]
NAMES = ["Not learned", "In superposition", "Own dimension"]
def svg(th):
    ink, mut, grid, acc, surf, low = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for j, (title, key) in enumerate([("2 features in 1 dimension", "p2"), ("3 features in 2 dimensions", "p3")]):
        x0 = L + j * (PW + GAP)
        o.append(f'<text x="{x0 + PW/2}" y="{TOP - 16}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{title}</text>')
        maj = DATA[key]
        for r in range(N):          # r = density index, 0 = dense (1.0), drawn at the bottom
            for c in range(N):      # c = importance index, 0 = 0.1
                v = maj[r * N + c]
                y = TOP + (N - 1 - r) * CELL
                o.append(f'<rect x="{x0 + c*CELL}" y="{y}" width="{CELL}" height="{CELL}" fill="{mix(low, acc, LEVEL[v])}" stroke="{surf}" stroke-width="0.6"/>')
        for c, lab in ((0, "0.1"), (10, "1"), (20, "10")):
            o.append(f'<text x="{x0 + c*CELL + CELL/2}" y="{TOP + N*CELL + 16}" fill="{mut}" font-size="12" text-anchor="middle">{lab}</text>')
        o.append(f'<text x="{x0 + PW/2}" y="{TOP + N*CELL + 36}" fill="{ink}" font-size="13" text-anchor="middle">Importance of the extra feature (log)</text>')
        for r, lab in ((0, "1"), (10, "10"), (20, "100")):
            o.append(f'<text x="{x0 - 6}" y="{TOP + (N-1-r)*CELL + CELL/2 + 4}" fill="{mut}" font-size="12" text-anchor="end">{lab}</text>')
    o.append(f'<text transform="translate(18,{TOP + CELL*N/2}) rotate(-90)" fill="{ink}" font-size="13" text-anchor="middle">1 / (1 - S), log scale</text>')
    bx = L + 40; by = TOP + N*CELL + 62
    for i, nm in enumerate(NAMES):
        x = bx + i * 200
        o.append(f'<rect x="{x}" y="{by}" width="16" height="16" fill="{mix(low, acc, LEVEL[i])}" stroke="{mut}" stroke-width="0.6"/>')
        o.append(f'<text x="{x + 24}" y="{by + 13}" fill="{ink}" font-size="13">{nm}</text>')
    o.append(f'<text x="{W/2}" y="{by + 42}" fill="{mut}" font-size="12" text-anchor="middle">Each cell: outcome for the extra feature in the lowest-loss of 10 trained models</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff', '#eef2f9')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d', '#14233f')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
