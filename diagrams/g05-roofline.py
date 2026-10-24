# Generates public/diagrams/g05-roofline.svg and g05-roofline-dark.svg (data: g05-cuda-matmul-optimization.md).
# Usage: python3 diagrams/g05-roofline.py public/diagrams/g05-roofline, then render the PNG with sharp at density 144.
# x: FLOPs per byte of DRAM traffic (Nsight Compute dram__bytes read+write), y: TFLOPS measured with CUDA events.
import math, sys
W, H = 780, 470; L, R, T, B = 78, 24, 24, 62
X0, X1 = 2, 1000; Y0, Y1 = 0.05, 20
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
BW = 0.2487; F32 = 12.73; F32LOW = 8.755   # measured DRAM roof (TB/s), FP32 FMA roof at 2,159 MHz, FP32 peak at 1,425 MHz
# label, FLOPs per DRAM byte (ncu, n = 4,096), TFLOPS (CUDA events, median of 3 rounds), label offsets
PTS = [('1', 15.6, 0.100, 9, 5), ('2', 15.7, 0.667, 9, 5), ('3', 15.7, 0.977, 9, 5), ('4', 31.0, 2.553, 9, 5),
       ('5', 62.3, 3.885, 9, 14), ('6', 58.6, 3.526, -16, 10), ('6b', 76.5, 5.057, 9, 12), ('7', 86.1, 5.422, 9, -4),
       ('8', 59.9, 6.105, -18, -4), ('cuBLAS', 231.8, 5.571, 9, 5)]
def svg(th):
    ink, mut, grid, acc, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in (2, 10, 100, 1000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:g}</text>')
    for y in (0.1, 1, 10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:g}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">FLOPs per byte of DRAM traffic (log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">FP32 TFLOPS (log scale)</text>')
    xr = F32 / BW
    o.append(f'<polyline fill="none" stroke="{ink}" stroke-width="2.5" points="{sx(X0):.1f},{sy(BW*X0):.1f} {sx(xr):.1f},{sy(F32):.1f} {sx(X1):.1f},{sy(F32):.1f}"/>')
    o.append(f'<line x1="{sx(F32LOW/BW):.1f}" y1="{sy(F32LOW):.1f}" x2="{sx(X1):.1f}" y2="{sy(F32LOW):.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{W-R:.1f}" y="{sy(F32)-8:.1f}" fill="{ink}" font-size="13" text-anchor="end">FP32 roof at 2,159 MHz: 12.7 TFLOPS</text>')
    o.append(f'<text x="{W-R:.1f}" y="{sy(F32LOW)+18:.1f}" fill="{mut}" font-size="13" text-anchor="end">FP32 peak at 1,425 MHz: 8.8 TFLOPS</text>')
    ang = math.degrees(math.atan((sy(BW*3) - sy(BW*30)) / (sx(30) - sx(3))))
    o.append(f'<text transform="translate({sx(2.6):.1f},{sy(BW*2.6)-10:.1f}) rotate(-{ang:.1f})" fill="{ink}" font-size="13">DRAM: 248.7 GB/s</text>')
    for lab, x, y, dx, dy in PTS:
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{acc}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{ink}" font-size="13" font-weight="bold">{lab}</text>')
    o.append('</svg>'); return '\n'.join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light)); open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark))
