# Generates public/diagrams/t09-benchmarks.svg and t09-benchmarks-dark.svg (data: t09-evaluate-base-model.md).
# Usage: python3 diagrams/t09-benchmarks.py public/diagrams/t09-benchmarks, then render the PNG with sharp at density 144.
# Data: zero-shot accuracy from fromscratch/evals.py (lm-eval-harness prompt formats), 95% binomial intervals.
# x: validation loss on train.py's 40-micro-batch subset (655,360 tokens): Part 7's log.jsonl for the 88M snapshots,
# Part 8's table for the sweep's final models. HellaSwag (10,042 items) is acc_norm; ARC-Easy (2,376) and PIQA (1,838) acc.
import sys, math
SNAP={  # step: (val loss, HellaSwag acc_norm, ARC-Easy acc, PIQA acc); None = not scored
 50:(6.9325,None,26.14,52.83),100:(6.0262,None,27.53,52.07),250:(4.8842,25.12,30.26,55.11),500:(4.1949,25.99,35.48,55.17),
 1000:(3.7602,26.63,42.34,58.00),1500:(3.5966,None,44.07,58.65),2000:(3.5029,27.76,45.92,60.07),3360:(3.3823,27.95,47.85,60.23)}
SWEEP={'11M':(4.1346,26.41,36.57,56.04),'15M':(3.9890,26.63,37.46,56.53),'22M':(3.8115,26.71,39.31,57.18),
 '33M':(3.6807,27.21,41.41,57.78),'47M':(3.5284,27.20,45.08,58.92)}
N=(10042,2376,1838); CH=(25.0,25.02,50.0)
PANELS=[('HellaSwag, acc_norm (10,042 items)',(24,30,1),0),('ARC-Easy, acc (2,376 items)',(20,52,5),1),('PIQA, acc (1,838 items)',(46,64,3),2)]
W,H=780,900; L,R=70,24; PH=230; GAP=50; TOP=56
X0,X1=7.1,3.3
def sx(x): return L+(X0-x)/(X0-X1)*(W-L-R)
def ci(p,n): return 196*math.sqrt(p/100*(1-p/100)/n)
def svg(th,acc,acc2):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    o.append(f'<line x1="{L}" x2="{L+26}" y1="20" y2="20" stroke="{acc2}" stroke-width="2.5"/><circle cx="{L+13}" cy="20" r="5" fill="{acc2}"/>')
    o.append(f'<text x="{L+34}" y="25" fill="{ink}" font-size="13">88M model over training (Part 7 snapshots, step labels)</text>')
    o.append(f'<rect x="{L+8}" y="34" width="10" height="10" fill="{acc}" transform="rotate(45 {L+13} 39)"/>')
    o.append(f'<text x="{L+34}" y="44" fill="{ink}" font-size="13">Part 8 sweep, final models, 20 tokens per parameter</text>')
    for k,(title,(y0,y1,dy),j) in enumerate(PANELS):
        t=TOP+k*(PH+GAP)+18; b=t+PH-18
        sy=lambda y:b-(y-y0)/(y1-y0)*(b-t)
        o.append(f'<text x="{L}" y="{t-8}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        y=y0
        while y<=y1+1e-9:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{y:g}%</text>'); y+=dy
        for x in (7,6,5,4):
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{t}" y2="{b}" stroke="{grid}" stroke-width="1"/>')
        if k==2:
            for x in (7,6,5,4,3.5):
                o.append(f'<text x="{sx(x):.1f}" y="{b+18}" fill="{mut}" font-size="13" text-anchor="middle">{x:g}</text>')
            o.append(f'<text x="{(L+W-R)/2}" y="{b+40}" fill="{ink}" font-size="14" text-anchor="middle">Validation loss, nats per token (training goes left to right)</text>')
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(CH[j]):.1f}" y2="{sy(CH[j]):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
        o.append(f'<text x="{sx(5.5):.1f}" y="{sy(CH[j])+16:.1f}" fill="{mut}" font-size="12" text-anchor="middle">chance {round(CH[j]):g}%</text>')
        pts=[(s,v[0],v[1+j]) for s,v in SNAP.items() if v[1+j] is not None]
        o.append(f'<polyline fill="none" stroke="{acc2}" stroke-width="2.5" points="{" ".join(f"{sx(x):.1f},{sy(p):.1f}" for _,x,p in pts)}"/>')
        for s,x,p in pts:
            e=ci(p,N[j])
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(p-e):.1f}" y2="{sy(p+e):.1f}" stroke="{acc2}" stroke-width="1.5"/>')
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(p):.1f}" r="5" fill="{acc2}" stroke="{surf}" stroke-width="1.5"/>')
            if k==1: o.append(f'<text x="{sx(x):.1f}" y="{sy(p+e)-6:.1f}" fill="{mut}" font-size="11" text-anchor="middle">{s}</text>')
        for name,v in SWEEP.items():
            x,p=v[0],v[1+j]; e=ci(p,N[j])
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(p-e):.1f}" y2="{sy(p+e):.1f}" stroke="{acc}" stroke-width="1.5"/>')
            o.append(f'<rect x="{sx(x)-5:.1f}" y="{sy(p)-5:.1f}" width="10" height="10" fill="{acc}" transform="rotate(45 {sx(x):.1f} {sy(p):.1f})"/>')
            if k==1: o.append(f'<text x="{sx(x):.1f}" y="{sy(p-e)+15:.1f}" fill="{mut}" font-size="11" text-anchor="middle">{name}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#38bdf8','#d95926'))
