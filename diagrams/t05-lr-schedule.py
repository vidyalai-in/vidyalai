# Generates public/diagrams/t05-lr-schedule.svg and t05-lr-schedule-dark.svg (data: t05-llm-training-loop.md).
# Usage: python3 diagrams/t05-lr-schedule.py public/diagrams/t05-lr-schedule, then render the PNG with sharp at density 144.
# Data: the learning rate train.py logged at every step of the 22M run (fromscratch/train.py lr_at: peak 1.5e-3,
# 20 warm-up steps, cosine to 10% of peak at step 440). The logged values equal this formula exactly.
import math, sys
W,H=780,420; L,R,T,B=78,24,24,62
PEAK,WARM,STEPS,FLOOR=1.5e-3,20,440,0.1
def lr_at(s):
    if s<WARM: return PEAK*(s+1)/WARM
    p=min(1.0,(s-WARM)/(STEPS-WARM)); lo=PEAK*FLOOR
    return lo+0.5*(PEAK-lo)*(1+math.cos(math.pi*p))
LR=[lr_at(s) for s in range(STEPS)]
X1=440; Y1=1.6e-3
def sx(s): return L+s/X1*(W-L-R)
def sy(y): return H-B-y/Y1*(H-T-B)
def svg(th):
    ink,mut,grid,acc,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0,0.4e-3,0.8e-3,1.2e-3,1.6e-3):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y*1e3:.1f}</text>')
    for s in range(0,X1+1,40):
        o.append(f'<line x1="{sx(s):.1f}" x2="{sx(s):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(s):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{s}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Optimizer step (524,288 tokens each)</text>')
    o.append(f'<text transform="translate(22,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Learning rate (× 10⁻³)</text>')
    pts=' '.join(f'{sx(i):.1f},{sy(v):.1f}' for i,v in enumerate(LR))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{acc}" stroke-width="2.5"/>')
    o.append(f'<line x1="{sx(WARM):.1f}" x2="{sx(WARM):.1f}" y1="{T}" y2="{H-B}" stroke="{ink}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{sx(WARM)+8:.1f}" y="{sy(PEAK)-10:.1f}" fill="{ink}" font-size="13">step 19: peak 1.5 × 10⁻³ after 20 warm-up steps</text>')
    m=WARM+(STEPS-WARM)//2
    o.append(f'<circle cx="{sx(m):.1f}" cy="{sy(lr_at(m)):.1f}" r="4.5" fill="{acc}"/>')
    o.append(f'<text x="{sx(m)+10:.1f}" y="{sy(lr_at(m))-6:.1f}" fill="{ink}" font-size="13">step {m}: halfway, {lr_at(m)*1e3:.3f} × 10⁻³</text>')
    o.append(f'<circle cx="{sx(STEPS-1):.1f}" cy="{sy(LR[-1]):.1f}" r="4.5" fill="{acc}"/>')
    o.append(f'<text x="{sx(STEPS-1)-4:.1f}" y="{sy(LR[-1])+22:.1f}" fill="{ink}" font-size="13" text-anchor="end">floor: 10% of peak, 0.15 × 10⁻³</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
