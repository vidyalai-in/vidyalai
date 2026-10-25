# Generates public/diagrams/t05-lr-sweep.svg and t05-lr-sweep-dark.svg (data: t05-llm-training-loop.md).
# Usage: python3 diagrams/t05-lr-sweep.py public/diagrams/t05-lr-sweep, then render the PNG with sharp at density 144.
# Data: final validation loss (first 40 val micro-batches) of short fromscratch/train.py runs of the 22M model, each with
# its own complete schedule (10% warm-up, cosine to 10% of peak), BF16, seed 1337, RTX 4060 Laptop GPU.
# B524: 524,288 tokens per step (16 x 1,024 x 32), 50 steps, 26.2M tokens.  B65: 65,536 per step (16 x 1,024 x 4),
# 150 steps, 9.8M tokens.  REP: a second run of the B65 1.2e-2 setting, identical except for GPU nondeterminism.
import math, sys
B524=[(7.5e-4,7.1317),(1.5e-3,7.1701),(3e-3,7.2252),(6e-3,7.3106),(1.2e-2,7.5191)]
B65=[(3.75e-4,6.4959),(7.5e-4,6.3321),(1.5e-3,6.4491),(3e-3,6.554),(6e-3,6.8018),(1.2e-2,6.9846)]
REP=(1.2e-2,7.215)
W,H=780,440; L,R,T,B=70,24,24,62
X0,X1=3e-4,1.5e-2; Y0,Y1=6.2,7.7
def sx(x): return L+(math.log10(x)-math.log10(X0))/(math.log10(X1)-math.log10(X0))*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
def svg(th,cat):
    ink,mut,grid,surf=th; c1,c2=cat
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (6.2,6.4,6.6,6.8,7.0,7.2,7.4,7.6):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:.1f}</text>')
    for x,lab in ((3.75e-4,'0.375'),(7.5e-4,'0.75'),(1.5e-3,'1.5'),(3e-3,'3'),(6e-3,'6'),(1.2e-2,'12')):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Peak learning rate (× 10⁻³, log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Final validation loss</text>')
    for pts,c,lab in ((B524,c1,'524,288 tokens/step, 50 steps (26.2M tokens)'),(B65,c2,'65,536 tokens/step, 150 steps (9.8M tokens)')):
        o.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x,y in pts)}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for x,y in pts: o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{c}"/>')
    o.append(f'<circle cx="{sx(REP[0]):.1f}" cy="{sy(REP[1]):.1f}" r="5" fill="{surf}" stroke="{c2}" stroke-width="2.2"/>')
    o.append(f'<text x="{sx(REP[0])-10:.1f}" y="{sy(REP[1])+4:.1f}" fill="{ink}" font-size="13" text-anchor="end">rerun of the blue 12 × 10⁻³ run: 7.22</text>')
    x,y=B524[0]; o.append(f'<text x="{sx(x):.1f}" y="{sy(y)-12:.1f}" fill="{ink}" font-size="13" text-anchor="middle">7.13</text>')
    x,y=B65[1]; o.append(f'<text x="{sx(x):.1f}" y="{sy(y)+22:.1f}" fill="{ink}" font-size="13" text-anchor="middle">best: 6.33</text>')
    lx,ly=L+20,T+16
    for i,(c,lab) in enumerate(((c1,'524,288 tokens per step, 50 steps (26.2M tokens)'),(c2,'65,536 tokens per step, 150 steps (9.8M tokens)'))):
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{ly+i*22}" y2="{ly+i*22}" stroke="{c}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{lx+13}" cy="{ly+i*22}" r="4.5" fill="{c}"/>')
        o.append(f'<text x="{lx+34}" y="{ly+i*22+4}" fill="{ink}" font-size="13">{lab}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#eb6834','#2a78d6'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#d95926','#3987e5')))
