# Generates public/diagrams/p08-transcoder-fidelity.svg and -dark.svg (data: p08-circuit-tracing.md).
# Usage: python3 diagrams/p08-transcoder-fidelity.py public/diagrams/p08-transcoder-fidelity, then render the PNG with sharp at density 144.
# Data: Gemma Scope transcoders (google/gemma-scope-2b-pt-transcoders, width 16k, circuit-tracer's lowest-L0 set)
# on Gemma 2 2B, 64 held-out FineWeb-Edu documents x 128 tokens = 8,192 tokens (p08/post/fidelity.json).
# ROWS: (layer, fraction of the MLP output's variance left unexplained, mean active features per token).
ROWS=[(0, 0.083, 80.7), (1, 0.349, 64.4), (2, 0.315, 47.1), (3, 0.336, 53.1), (4, 0.299, 83.8), (5, 0.372, 92.2), (6, 0.354, 100.0), (7, 0.384, 76.3), (8, 0.396, 54.2), (9, 0.385, 77.0), (10, 0.425, 90.4), (11, 0.618, 5.5), (12, 0.682, 5.6), (13, 0.621, 8.1), (14, 0.584, 8.0), (15, 0.511, 8.8), (16, 0.524, 10.8), (17, 0.5, 12.6), (18, 0.424, 13.8), (19, 0.449, 12.7), (20, 0.47, 12.4), (21, 0.462, 14.6), (22, 0.479, 15.9), (23, 0.432, 26.7), (24, 0.305, 37.1), (25, 0.209, 41.8)]
import sys
W,H=780,560; L,R,T,B=74,24,30,60
GAP=56; PH=(H-T-B-GAP)/2
def sx(l): return L+(l+0.5)/26*(W-L-R)
def sy1(v): return T+PH-(v/0.8)*PH
def sy2(v): return T+PH+GAP+PH-(v/100)*PH
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (0,0.2,0.4,0.6,0.8):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy1(v):.1f}" y2="{sy1(v):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy1(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v:g}</text>')
    for v in (0,25,50,75,100):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy2(v):.1f}" y2="{sy2(v):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy2(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v:g}</text>')
    o.append(f'<text x="{L}" y="{T-10}" fill="{ink}" font-size="14" font-weight="bold">Variance of the MLP output the transcoder misses</text>')
    o.append(f'<text x="{L}" y="{T+PH+GAP-10}" fill="{ink}" font-size="14" font-weight="bold">Active features per token (L0)</text>')
    pts=' '.join(f'{sx(l):.1f},{sy1(v):.1f}' for l,v,_ in ROWS)
    o.append(f'<polyline fill="none" stroke="{acc}" stroke-width="2.5" points="{pts}"/>')
    for l,v,_ in ROWS: o.append(f'<circle cx="{sx(l):.1f}" cy="{sy1(v):.1f}" r="4" fill="{acc}" stroke="{surf}" stroke-width="1"/>')
    bw=(W-L-R)/26*0.62
    for l,_,z in ROWS:
        o.append(f'<rect x="{sx(l)-bw/2:.1f}" y="{sy2(z):.1f}" width="{bw:.1f}" height="{sy2(0)-sy2(z):.1f}" fill="{acc2}"/>')
    for l in range(0,26,5):
        o.append(f'<text x="{sx(l):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{l}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Gemma 2 2B layer</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
