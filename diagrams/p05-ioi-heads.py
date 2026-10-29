# Generates public/diagrams/p05-ioi-heads.svg and -dark.svg (data: p05-activation-patching.md).
# Usage: python3 diagrams/p05-ioi-heads.py public/diagrams/p05-ioi-heads, then render the PNG with sharp at density 144.
# Data: GPT-2 small, 500 IOI prompts. Left: activation patching of each head's output at all positions
# (denoising, STR corruption), share of the logit-difference gap restored. Middle and right: path patching
# (Wang et al. 2022; clean run, sender head from the ABC prompt), logit difference lost as a share of the clean
# value, for the path head -> logits and head -> name movers' queries (p05 scripts/ioi.py, ioi_summary.py).
import sys
D={"act": [[-0.0, 0.02, -0.0, 0.002, -0.0, 0.001, -0.003, -0.0, -0.0, -0.0, -0.001, -0.0], [-0.0, -0.0, 0.0, 0.001, -0.0, 0.005, -0.001, -0.001, 0.0, 0.0, -0.0, 0.003], [-0.0, -0.001, 0.002, -0.0, -0.0, -0.0, 0.0, 0.001, -0.0, -0.0, -0.0, 0.0], [0.124, -0.0, 0.0, -0.0, 0.022, -0.002, 0.006, -0.0, -0.0, -0.0, 0.0, -0.0], [-0.0, 0.0, -0.0, 0.023, -0.005, -0.001, 0.001, -0.004, 0.001, -0.0, 0.0, 0.001], [-0.002, 0.003, 0.001, 0.0, -0.001, 0.333, 0.0, -0.001, -0.015, 0.044, 0.007, -0.003], [0.036, 0.012, -0.0, -0.002, -0.005, -0.003, -0.019, 0.0, 0.0, 0.097, -0.003, 0.0], [-0.0, 0.015, 0.0, 0.12, -0.0, -0.011, 0.001, -0.002, -0.008, 0.213, 0.0, -0.001], [-0.002, 0.0, -0.01, 0.007, -0.0, -0.006, 0.293, 0.003, -0.0, 0.001, 0.267, 0.006], [0.004, -0.001, 0.009, 0.024, -0.024, 0.014, -0.066, 0.09, -0.001, 0.179, 0.001, -0.001], [0.125, 0.04, 0.017, 0.005, -0.0, 0.0, 0.059, -0.484, -0.0, 0.007, 0.056, 0.002], [0.003, -0.001, -0.107, 0.007, 0.0, -0.0, -0.014, 0.002, 0.006, 0.004, -0.264, 0.004]], "logits": [[-0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.001, -0.0, -0.0, -0.0, -0.0, -0.0], [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.001, 0.0, 0.0, -0.0, 0.0], [-0.0, -0.001, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.001, -0.0, 0.0], [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0], [0.0, -0.0, -0.0, -0.001, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0], [0.0, -0.0, 0.001, -0.0, -0.0, -0.0, -0.0, -0.001, 0.0, -0.0, -0.001, 0.001], [-0.003, 0.001, 0.001, -0.0, -0.001, -0.0, 0.0, -0.0, 0.0, -0.0, 0.001, 0.001], [0.0, -0.001, -0.0, 0.016, 0.0, -0.005, -0.001, 0.002, -0.003, 0.068, -0.001, 0.002], [0.001, -0.001, -0.009, 0.005, 0.0, -0.0, 0.067, 0.001, -0.005, -0.0, 0.067, -0.001], [0.025, 0.0, 0.01, 0.009, -0.006, 0.009, 0.322, 0.052, -0.008, 0.792, -0.0, -0.001], [0.221, 0.077, 0.015, 0.002, 0.001, 0.0, 0.075, -0.513, 0.001, 0.002, 0.131, 0.002], [0.005, -0.004, -0.11, 0.003, -0.001, 0.001, -0.017, 0.001, 0.006, 0.009, -0.263, 0.003]], "nmq": [[0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0], [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0], [0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0], [0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0], [-0.0, -0.0, 0.0, 0.001, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0], [0.001, 0.005, 0.0, 0.001, -0.0, -0.001, 0.0, -0.0, 0.0, -0.0, 0.001, -0.0], [-0.0, -0.0, -0.0, 0.051, 0.0, -0.003, 0.0, 0.001, -0.002, 0.09, -0.0, 0.002], [-0.001, 0.0, 0.001, 0.008, 0.0, 0.004, 0.164, 0.001, 0.001, 0.001, 0.109, -0.003], [-0.009, 0.0, -0.005, 0.009, 0.0, -0.001, -0.091, 0.016, -0.005, -0.048, -0.0, -0.0], [None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None]]}
PANELS=[('Patch head output','act','share restored'),('Path: head to logits','logits','share lost'),('Path: head to name movers','nmq','share lost')]
W=780; L=46; TOP=62; CELL=17; GAP=34; NL=12
PW=CELL*12
H=TOP+CELL*NL+128
def mix(a,b,t):
    a=[int(a[i:i+2],16) for i in (1,3,5)]; b=[int(b[i:i+2],16) for i in (1,3,5)]
    return '#'+''.join(f'{round(x+(y-x)*t):02x}' for x,y in zip(a,b))
def svg(th):
    ink,mut,grid,acc,surf,low=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for j,(title,k,unit) in enumerate(PANELS):
        x0=L+j*(PW+GAP); M=D[k]; mx=max(abs(v) for r in M for v in r if v is not None)
        o.append(f'<text x="{x0+PW/2}" y="{TOP-30}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{title}</text>')
        for l in range(NL):
            for h in range(12):
                v=M[l][h]
                if v is None:
                    o.append(f'<rect x="{x0+h*CELL}" y="{TOP+l*CELL}" width="{CELL}" height="{CELL}" fill="{surf}" stroke="{grid}" stroke-width="0.5"/>'); continue
                t=min(1.0,abs(v)/mx)
                o.append(f'<rect x="{x0+h*CELL}" y="{TOP+l*CELL}" width="{CELL}" height="{CELL}" fill="{mix(low,acc,t**0.8)}" stroke="{surf}" stroke-width="1"/>')
                if v<0 and t>0.12:
                    o.append(f'<line x1="{x0+h*CELL+4}" x2="{x0+h*CELL+CELL-4}" y1="{TOP+l*CELL+CELL/2}" y2="{TOP+l*CELL+CELL/2}" stroke="{surf if t>0.5 else ink}" stroke-width="2"/>')
        for h in range(12):
            o.append(f'<text x="{x0+h*CELL+CELL/2}" y="{TOP-8}" fill="{mut}" font-size="11" text-anchor="middle">{h}</text>')
        if j==0:
            for l in range(NL):
                o.append(f'<text x="{x0-6}" y="{TOP+l*CELL+CELL/2+4}" fill="{mut}" font-size="11" text-anchor="end">{l}</text>')
            o.append(f'<text transform="translate(14,{TOP+CELL*NL/2}) rotate(-90)" fill="{ink}" font-size="13" text-anchor="middle">Layer</text>')
        flat=sorted(((v,l,h) for l in range(NL) for h in range(12) for v in [M[l][h]] if v is not None),reverse=True)
        pos=', '.join(f'{l}.{h}' for v,l,h in flat[:3]); neg=', '.join(f'{l}.{h}' for v,l,h in flat[::-1][:2] if v<0)
        o.append(f'<text x="{x0+PW/2}" y="{TOP+CELL*NL+18}" fill="{mut}" font-size="12" text-anchor="middle">helps: {pos}</text>')
        if neg: o.append(f'<text x="{x0+PW/2}" y="{TOP+CELL*NL+34}" fill="{mut}" font-size="12" text-anchor="middle">hurts: {neg}</text>')
        o.append(f'<text x="{x0+PW/2}" y="{TOP+CELL*NL+50}" fill="{mut}" font-size="12" text-anchor="middle">max |effect| {mx:.2f} ({unit})</text>')
    o.append(f'<text x="{L+(3*PW+2*GAP)/2}" y="{TOP+CELL*NL+76}" fill="{ink}" font-size="13" text-anchor="middle">Head (columns), layer (rows). Stronger color = larger effect, scaled per panel. A bar marks a head that works against the answer.</text>')
    bx=L+(3*PW+2*GAP)/2-150; by=TOP+CELL*NL+90
    for i in range(60):
        o.append(f'<rect x="{bx+i*5}" y="{by}" width="5.5" height="12" fill="{mix(low,acc,(i/59)**0.8)}"/>')
    for v,lab in ((0,'0'),(1,'panel max')):
        o.append(f'<text x="{bx+v*300}" y="{by+27}" fill="{mut}" font-size="12" text-anchor="middle">{lab}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eef2f9')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#14233f')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
