# Generates public/diagrams/p05-causal-trace.svg and -dark.svg (data: p05-activation-patching.md).
# Usage: python3 diagrams/p05-causal-trace.py public/diagrams/p05-causal-trace, then render the PNG with sharp at density 144.
# Data: GPT-2 small, the 67 landmark prompts it knows (Part 1's set), causal tracing as in Meng et al. (2022):
# subject embeddings noised (3 x their standard deviation, 10 draws), one hidden state, or a window of 3 MLP or
# attention outputs, restored from the clean run. Average indirect effect on P(right city) (p05 scripts/trace.py).
import sys
D={"hidden": [[-0.0, -0.0001, 0.0001, 0.0002, 0.0002, 0.0003, 0.0004, 0.0004, 0.0005, 0.0004, 0.0003, 0.0], [0.0084, 0.0094, 0.0109, 0.0124, 0.0127, 0.0132, 0.0146, 0.0152, 0.0135, 0.0056, 0.002, 0.0], [0.0033, 0.0039, 0.0048, 0.0056, 0.0059, 0.0069, 0.0081, 0.0084, 0.0077, 0.0026, 0.0008, 0.0], [0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0003, 0.0003, 0.0004, 0.0004, 0.0003, 0.0002, 0.0], [0.0001, 0.0002, 0.0003, 0.0001, 0.0001, 0.0, 0.0002, 0.0002, 0.0003, 0.0005, 0.0003, 0.0], [0.0001, -0.0001, 0.0001, 0.0002, 0.0004, 0.0008, 0.0009, 0.001, 0.0071, 0.0237, 0.0376, 0.0746]], "mlp": [[0.0002, 0.0002, -0.0001, -0.0, -0.0001, 0.0, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0], [0.0087, 0.0092, 0.0009, 0.0005, 0.0, 0.0002, 0.0006, 0.0003, 0.0002, 0.0001, 0.0, -0.0], [0.0041, 0.0042, -0.0, -0.0001, 0.0, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0, 0.0], [-0.0001, -0.0003, -0.0002, 0.0, 0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0], [-0.0003, -0.0, -0.0, 0.0003, 0.0001, -0.0, -0.0001, -0.0, 0.0, 0.0, 0.0, 0.0], [0.0001, -0.0006, -0.0004, -0.0, -0.0005, -0.0005, -0.0002, -0.0001, 0.0038, 0.0052, 0.0008, -0.001]], "attn": [[-0.0002, -0.0001, 0.0001, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0002, 0.0003, 0.0002, 0.0001, 0.0001, 0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0], [0.0003, 0.0003, 0.0, 0.0, 0.0001, 0.0001, 0.0002, 0.0001, 0.0002, 0.0002, 0.0001, 0.0], [0.0001, 0.0001, -0.0001, -0.0001, -0.0, 0.0001, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0], [0.0002, 0.0003, 0.0001, -0.0001, -0.0002, -0.0001, -0.0, 0.0001, 0.0004, 0.0004, 0.0002, 0.0], [-0.0001, 0.0001, 0.0004, 0.0007, 0.001, 0.0009, 0.001, 0.0073, 0.026, 0.0405, 0.0512, 0.0164]]}
ROWS=["First subject token","Middle subject tokens","Last subject token","First token after","Further tokens","Last token"]
PANELS=[("hidden","Restore one hidden state (residual stream after the block)"),("mlp","Restore 3 MLP outputs (window centred on the block)"),("attn","Restore 3 attention outputs (window centred on the block)")]
W=780; L=196; CW=46; CH=22; PH=6*CH; TOP=40; GAP=58
H=TOP+3*PH+2*GAP+130
def mix(a,b,t):
    a=[int(a[i:i+2],16) for i in (1,3,5)]; b=[int(b[i:i+2],16) for i in (1,3,5)]
    return '#'+''.join(f'{round(x+(y-x)*t):02x}' for x,y in zip(a,b))
def svg(th):
    ink,mut,grid,acc,surf,low=th
    mx=max(v for k,_ in PANELS for r in D[k] for v in r if v is not None)
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for p,(k,title) in enumerate(PANELS):
        y0=TOP+p*(PH+GAP)
        o.append(f'<text x="{L}" y="{y0-10}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for j,rn in enumerate(ROWS):
            o.append(f'<text x="{L-8}" y="{y0+j*CH+CH/2+4}" fill="{mut}" font-size="12" text-anchor="end">{rn}</text>')
            for l in range(12):
                v=D[k][j][l]
                if v is None: continue
                t=max(0.0,min(1.0,v/mx))
                o.append(f'<rect x="{L+l*CW}" y="{y0+j*CH}" width="{CW}" height="{CH}" fill="{mix(low,acc,t**0.8)}" stroke="{surf}" stroke-width="1"/>')
                o.append(f'<text x="{L+l*CW+CW/2}" y="{y0+j*CH+CH/2+4}" fill="{surf if t>0.55 else mut}" font-size="10" text-anchor="middle">{abs(100*v) if abs(100*v)<0.05 else 100*v:.1f}</text>')
        for l in range(12):
            o.append(f'<text x="{L+l*CW+CW/2}" y="{y0+PH+15}" fill="{mut}" font-size="11" text-anchor="middle">{l}</text>')
    o.append(f'<text x="{L+6*CW}" y="{TOP+3*PH+2*GAP+36}" fill="{ink}" font-size="13" text-anchor="middle">Block</text>')
    by=TOP+3*PH+2*GAP+50; bx=L+6*CW-150
    for i in range(60):
        o.append(f'<rect x="{bx+i*5}" y="{by}" width="5.5" height="12" fill="{mix(low,acc,(i/59)**0.8)}"/>')
    for v in (0,0.5,1):
        o.append(f'<text x="{bx+v*300}" y="{by+28}" fill="{mut}" font-size="12" text-anchor="middle">{100*v*mx:.1f}</text>')
    o.append(f'<text x="{L+6*CW}" y="{by+50}" fill="{mut}" font-size="12" text-anchor="middle">Percentage points of P(right city) restored, mean over prompts (clean 7.9%, noised 0.4%)</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eef2f9')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#14233f')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
