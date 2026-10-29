# Generates public/diagrams/p04-head-scores.svg and -dark.svg (data: p04-induction-heads.md).
# Usage: python3 diagrams/p04-head-scores.py public/diagrams/p04-head-scores, then render the PNG with sharp at density 144.
# Data: GPT-2 small, mean attention score per layer x head. Previous-token: 200 WikiText-103 windows of 256 tokens.
# Duplicate and induction: 500 sequences of 50 random tokens repeated twice (lab/heads.py, p04 scripts/s1_scores.py).
import sys
D={"prev": [[0.042, 0.003, 0.032, 0.073, 0.14, 0.004, 0.047, 0.288, 0.044, 0.037, 0.034, 0.016], [0.297, 0.163, 0.052, 0.042, 0.068, 0.026, 0.038, 0.028, 0.019, 0.015, 0.037, 0.01], [0.171, 0.018, 0.546, 0.273, 0.278, 0.292, 0.083, 0.041, 0.272, 0.319, 0.112, 0.071], [0.007, 0.115, 0.372, 0.297, 0.013, 0.044, 0.281, 0.387, 0.281, 0.155, 0.075, 0.2], [0.195, 0.176, 0.027, 0.177, 0.042, 0.131, 0.099, 0.136, 0.023, 0.085, 0.013, 0.992], [0.015, 0.005, 0.129, 0.079, 0.138, 0.013, 0.203, 0.089, 0.032, 0.014, 0.052, 0.031], [0.16, 0.044, 0.018, 0.029, 0.033, 0.066, 0.024, 0.085, 0.262, 0.007, 0.012, 0.121], [0.238, 0.013, 0.005, 0.056, 0.043, 0.057, 0.016, 0.015, 0.125, 0.053, 0.007, 0.01], [0.022, 0.009, 0.022, 0.019, 0.076, 0.114, 0.03, 0.2, 0.048, 0.03, 0.037, 0.022], [0.032, 0.01, 0.022, 0.184, 0.011, 0.023, 0.008, 0.028, 0.02, 0.008, 0.092, 0.012], [0.016, 0.01, 0.013, 0.02, 0.028, 0.041, 0.013, 0.024, 0.012, 0.1, 0.01, 0.068], [0.026, 0.017, 0.012, 0.043, 0.017, 0.011, 0.009, 0.014, 0.087, 0.01, 0.046, 0.048]], "dup": [[0.009, 0.268, 0.014, 0.0, 0.002, 0.643, 0.011, 0.0, 0.008, 0.002, 0.06, 0.015], [0.0, 0.0, 0.002, 0.001, 0.003, 0.02, 0.033, 0.006, 0.008, 0.009, 0.006, 0.085], [0.0, 0.014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.001, 0.0], [0.666, 0.0, 0.0, 0.0, 0.005, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.001, 0.0, 0.014, 0.0, 0.0, 0.0, 0.008, 0.0, 0.008, 0.0], [0.011, 0.004, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.003, 0.055, 0.004], [0.0, 0.0, 0.006, 0.0, 0.009, 0.0, 0.022, 0.0, 0.0, 0.005, 0.009, 0.0], [0.0, 0.05, 0.039, 0.0, 0.0, 0.001, 0.006, 0.028, 0.0, 0.0, 0.014, 0.004], [0.004, 0.029, 0.002, 0.003, 0.0, 0.0, 0.01, 0.0, 0.002, 0.006, 0.003, 0.001], [0.024, 0.006, 0.017, 0.0, 0.009, 0.006, 0.012, 0.002, 0.011, 0.019, 0.0, 0.007], [0.015, 0.012, 0.005, 0.011, 0.008, 0.0, 0.014, 0.001, 0.007, 0.0, 0.011, 0.017], [0.011, 0.008, 0.007, 0.002, 0.01, 0.009, 0.01, 0.008, 0.006, 0.013, 0.002, 0.004]], "ind": [[0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.009, 0.0, 0.009, 0.009, 0.007, 0.014], [0.0, 0.0, 0.002, 0.001, 0.003, 0.009, 0.013, 0.013, 0.009, 0.012, 0.007, 0.0], [0.0, 0.014, 0.0, 0.0, 0.0, 0.0, 0.001, 0.005, 0.0, 0.0, 0.001, 0.0], [0.002, 0.0, 0.0, 0.0, 0.012, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.001, 0.0, 0.003, 0.0, 0.0, 0.0, 0.01, 0.0, 0.004, 0.0], [0.462, 0.909, 0.0, 0.0, 0.0, 0.933, 0.0, 0.0, 0.008, 0.018, 0.013, 0.002], [0.0, 0.0, 0.007, 0.0, 0.016, 0.0, 0.019, 0.0, 0.0, 0.918, 0.033, 0.0], [0.0, 0.229, 0.833, 0.001, 0.0, 0.002, 0.034, 0.111, 0.0, 0.0, 0.912, 0.044], [0.004, 0.362, 0.006, 0.03, 0.0, 0.0, 0.143, 0.0, 0.012, 0.015, 0.051, 0.006], [0.284, 0.142, 0.11, 0.0, 0.091, 0.012, 0.524, 0.012, 0.046, 0.502, 0.0, 0.039], [0.417, 0.524, 0.03, 0.144, 0.055, 0.0, 0.331, 0.468, 0.03, 0.0, 0.186, 0.297], [0.015, 0.049, 0.027, 0.003, 0.029, 0.086, 0.042, 0.058, 0.008, 0.351, 0.43, 0.014]]}
PANELS=[('Previous-token score','prev'),('Duplicate-token score','dup'),('Induction score','ind')]
W=780; L=46; TOP=54; CELL=17; GAP=34; NL=12
PW=CELL*12
H=TOP+CELL*NL+92
def mix(a,b,t):
    a=[int(a[i:i+2],16) for i in (1,3,5)]; b=[int(b[i:i+2],16) for i in (1,3,5)]
    return '#'+''.join(f'{round(x+(y-x)*t):02x}' for x,y in zip(a,b))
def svg(th):
    ink,mut,grid,acc,surf,low=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for j,(title,k) in enumerate(PANELS):
        x0=L+j*(PW+GAP)
        o.append(f'<text x="{x0+PW/2}" y="{TOP-28}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{title}</text>')
        for l in range(NL):
            for h in range(12):
                v=max(0.0,min(1.0,D[k][l][h]))
                o.append(f'<rect x="{x0+h*CELL}" y="{TOP+l*CELL}" width="{CELL}" height="{CELL}" fill="{mix(low,acc,v**0.8)}" stroke="{surf}" stroke-width="1"/>')
        for h in range(12):
                o.append(f'<text x="{x0+h*CELL+CELL/2}" y="{TOP-8}" fill="{mut}" font-size="11" text-anchor="middle">{h}</text>')
        if j==0:
            for l in range(NL):
                o.append(f'<text x="{x0-6}" y="{TOP+l*CELL+CELL/2+4}" fill="{mut}" font-size="11" text-anchor="end">{l}</text>')
            o.append(f'<text transform="translate(14,{TOP+CELL*NL/2}) rotate(-90)" fill="{ink}" font-size="13" text-anchor="middle">Layer</text>')
        # mark the top heads
        flat=sorted(((D[k][l][h],l,h) for l in range(NL) for h in range(12)),reverse=True)
        n={'prev':2,'dup':2,'ind':5}[k]
        for v,l,h in flat[:n]:
            o.append(f'<rect x="{x0+h*CELL+1.5}" y="{TOP+l*CELL+1.5}" width="{CELL-3}" height="{CELL-3}" fill="none" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<text x="{x0+PW/2}" y="{TOP+CELL*NL+18}" fill="{mut}" font-size="12" text-anchor="middle">top: '+', '.join(f'{l}.{h} ({v:.2f})' for v,l,h in flat[:2])+'</text>')
    o.append(f'<text x="{L+(3*PW+2*GAP)/2}" y="{TOP+CELL*NL+40}" fill="{ink}" font-size="13" text-anchor="middle">Head (columns), layer (rows). Stronger color = more attention. Outlined: highest-scoring heads.</text>')
    # color bar
    bx=L+(3*PW+2*GAP)/2-150; by=TOP+CELL*NL+54
    for i in range(60):
        o.append(f'<rect x="{bx+i*5}" y="{by}" width="5.5" height="12" fill="{mix(low,acc,(i/59)**0.8)}"/>')
    for v in (0,0.5,1):
        o.append(f'<text x="{bx+v*300}" y="{by+27}" fill="{mut}" font-size="12" text-anchor="middle">{v:g}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eef2f9')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#14233f')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
