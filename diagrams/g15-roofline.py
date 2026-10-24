# Generates public/diagrams/g15-roofline.svg and g15-roofline-dark.svg (data: g15-token-kernel-trace.md).
# Usage: python3 diagrams/g15-roofline.py public/diagrams/g15-roofline, then render the PNG with sharp at density 144.
# Each point is one kernel class of a Qwen2.5-1.5B decode step: x = FLOPs per byte (minimum DRAM bytes counted from
# the shapes), y = TFLOPS from torch.profiler kernel times summed over the step.
import math, sys
W, H = 780, 500; L, R, T, B = 78, 24, 24, 62
X0, X1 = 0.1, 1000; Y0, Y1 = 0.001, 40
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
BW = 0.2487; PK = 25.3
# (label, intensity, TFLOPS, hollow?)  hollow = not part of the batch-1 BF16 step
PTS = [('A', 1.0, 0.241, False), ('B', 1.0, 0.198, False), ('C', 6.0, 0.800, False), ('D', 6.0, 0.0759, False),
       ('E', 0.42, 0.0042, False), ('F', 0.83, 0.0413, False), ('G', 0.5, 0.0057, False),
       ('H', 3.75, 0.738, True), ('I', 31.2, 7.14, True), ('J', 115.5, 19.84, True), ('K', 6.0, 1.447, True)]
OFF = {'A': (-20, -4), 'B': (9, 14), 'C': (10, 5), 'D': (10, 5), 'E': (-20, 5), 'G': (10, 5), 'F': (10, 5),
       'H': (-22, -6), 'I': (10, 14), 'J': (-6, 22), 'K': (10, -2)}

def svg(th):
    ink, mut, grid, acc, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in (0.1, 1, 10, 100, 1000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:g}</text>')
    for y in (0.001, 0.01, 0.1, 1, 10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:g}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Arithmetic intensity (FLOPs per byte, log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">TFLOPS (log scale)</text>')
    xr = PK / BW
    o.append(f'<polyline fill="none" stroke="{ink}" stroke-width="2.5" points="{sx(X0):.1f},{sy(BW*X0):.1f} {sx(xr):.1f},{sy(PK):.1f} {sx(X1):.1f},{sy(PK):.1f}"/>')
    o.append(f'<text x="{sx(xr)-8:.1f}" y="{sy(PK)-8:.1f}" fill="{ink}" font-size="13" text-anchor="end">BF16 tensor cores: 25.3 TFLOPS</text>')
    ang = math.degrees(math.atan((sy(BW * 0.1) - sy(BW * 1)) / (sx(1) - sx(0.1))))
    o.append(f'<text transform="translate({sx(0.16):.1f},{sy(BW*0.16)-10:.1f}) rotate(-{ang:.1f})" fill="{ink}" font-size="13">DRAM: 248.7 GB/s</text>')
    o.append(f'<text x="{sx(xr)+6:.1f}" y="{sy(PK)-8:.1f}" fill="{mut}" font-size="12">ridge ≈ 102</text>')
    for lab, x, y, hollow in PTS:
        if hollow:
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{surf}" stroke="{acc}" stroke-width="2.5"/>')
        else:
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{acc}" stroke="{surf}" stroke-width="2"/>')
        dx, dy = OFF.get(lab, (9, 5))
        o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{ink}" font-size="14" font-weight="bold">{lab}</text>')
    o.append('</svg>'); return '\n'.join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
