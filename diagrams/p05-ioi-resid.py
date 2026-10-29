# Generates public/diagrams/p05-ioi-resid.svg and -dark.svg (data: p05-activation-patching.md).
# Usage: python3 diagrams/p05-ioi-resid.py public/diagrams/p05-ioi-resid, then render the PNG with sharp at density 144.
# Data: GPT-2 small, 500 IOI prompts (15 templates of Wang et al. 2022, BABA and ABBA), denoising with STR
# corruption (S2 replaced by IO). Share of the clean-corrupted logit-difference gap restored by patching
# the residual stream entering each block at each position group (p05 scripts/ioi.py, ioi_summary.py).
import sys
D=[[-0.0, -0.0, -0.0, 1.0, -0.0, -0.0], [-0.0, -0.0, -0.0, 1.0, -0.0, -0.0], [-0.0, -0.0, -0.0, 1.0, -0.0, -0.0], [-0.0, -0.0, -0.0, 1.0, -0.0, -0.0], [-0.0, -0.0, -0.0, 1.0, 0.0, -0.0], [-0.0, -0.0, -0.0, 0.98, 0.02, -0.0], [-0.0, -0.0, -0.0, 0.98, 0.02, -0.0], [-0.0, -0.0, -0.0, 0.94, 0.04, 0.01], [-0.0, -0.0, -0.0, 0.67, 0.02, 0.31], [-0.0, -0.0, -0.0, 0.06, 0.03, 0.91], [-0.0, -0.0, -0.0, -0.03, 0.02, 1.02], [-0.0, -0.0, -0.0, -0.01, 0.01, 0.99]]
COLS=["IO","S1","S1+1","S2","S2+1 to END-1","END"]
W=780; L=150; TOP=86; CW=96; CH=27; NL=12
H=TOP+CH*NL+96
def mix(a,b,t):
    a=[int(a[i:i+2],16) for i in (1,3,5)]; b=[int(b[i:i+2],16) for i in (1,3,5)]
    return '#'+''.join(f'{round(x+(y-x)*t):02x}' for x,y in zip(a,b))
def svg(th):
    ink,mut,grid,acc,surf,low=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    o.append(f'<text x="{L+CW*3}" y="24" fill="{ink}" font-size="15" font-weight="bold" text-anchor="middle">Share of the answer restored by patching one residual-stream slice</text>')
    for j,c in enumerate(COLS):
        o.append(f'<text x="{L+j*CW+CW/2}" y="{TOP-12}" fill="{ink}" font-size="13" text-anchor="middle">{c}</text>')
    o.append(f'<text x="{L+CW*3}" y="{TOP-34}" fill="{mut}" font-size="12" text-anchor="middle">Position in the prompt</text>')
    for l in range(NL):
        o.append(f'<text x="{L-10}" y="{TOP+l*CH+CH/2+5}" fill="{mut}" font-size="12" text-anchor="end">block {l}</text>')
        for j in range(len(COLS)):
            v=D[l][j]; t=max(0.0,min(1.0,v))
            fill=mix(low,acc,t**0.85)
            o.append(f'<rect x="{L+j*CW}" y="{TOP+l*CH}" width="{CW}" height="{CH}" fill="{fill}" stroke="{surf}" stroke-width="1.5"/>')
            txt=f'{v:.2f}' if abs(v)>=0.005 else '0'
            tc=surf if t>0.55 else ink
            o.append(f'<text x="{L+j*CW+CW/2}" y="{TOP+l*CH+CH/2+5}" fill="{tc}" font-size="12" text-anchor="middle">{txt}</text>')
    o.append(f'<text transform="translate(22,{TOP+CH*NL/2}) rotate(-90)" fill="{ink}" font-size="13" text-anchor="middle">Residual stream entering</text>')
    by=TOP+CH*NL+30; bx=L+CW*3-150
    for i in range(60):
        o.append(f'<rect x="{bx+i*5}" y="{by}" width="5.5" height="12" fill="{mix(low,acc,(i/59)**0.85)}"/>')
    for v in (0,0.5,1):
        o.append(f'<text x="{bx+v*300}" y="{by+28}" fill="{mut}" font-size="12" text-anchor="middle">{v:g}</text>')
    o.append(f'<text x="{L+CW*3}" y="{by+50}" fill="{mut}" font-size="12" text-anchor="middle">0 = the corrupted run, 1 = the clean run. Mean over 500 prompts.</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eef2f9')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#14233f')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
