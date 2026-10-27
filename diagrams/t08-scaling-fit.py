# Generates public/diagrams/t08-scaling-fit.svg and t08-scaling-fit-dark.svg (data: t08-scaling-laws-experiment.md).
# Usage: python3 diagrams/t08-scaling-fit.py public/diagrams/t08-scaling-fit, then render the PNG with sharp at density 144.
# Data: final loss of each run on the whole validation shard (14,991,360 tokens), from fromscratch/snapshots.py val_loss,
# against total and non-embedding parameters. Fits (t08 scripts/fit.py, all six points, least squares on the loss):
# total: L = 2.6928 + 525.33 / N^0.3645 and L = 20.011 * N^-0.09796; non-embedding: L = 1.8270 + 17.142 / N^0.1345 and
# L = 11.394 * N^-0.06818. Five sweep runs plus Part 7's 88M run, all at 20 tokens per total parameter.
import math, sys
NAMES=['11M','15M','22M','33M','47M','88M']
NT=[10938960,14749344,22025088,32939040,47195712,88096512]
NE=[3074640,5312160,9442176,17210400,28321344,62930688]
LW=[4.1201,3.9731,3.7973,3.6638,3.5135,3.3651]
FITS={'t':((2.6928,525.33,0.3645),(20.011,0.09796)),'e':((1.8270,17.142,0.1345),(11.394,0.06818))}
W,H=780,820; L,R=78,24
PANELS=[('t',NT,(40,360),2e6,1.5e8,'Total parameters (log scale), embedding table included'),
        ('e',NE,(450,770),2e6,1.5e8,'Non-embedding parameters (log scale)')]
Y0,Y1=3.25,4.3
def svg(th,acc,acc2):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for key,N,(t,b),X0,X1,xlab in PANELS:
        sx=lambda x:L+(math.log10(x)-math.log10(X0))/(math.log10(X1)-math.log10(X0))*(W-L-R)
        sy=lambda y:b-(y-Y0)/(Y1-Y0)*(b-t)
        for x in (2e6,5e6,1e7,2e7,5e7,1e8):
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{t}" y2="{b}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{sx(x):.1f}" y="{b+18}" fill="{mut}" font-size="13" text-anchor="middle">{x/1e6:g}M</text>')
        for y in (3.4,3.6,3.8,4.0,4.2):
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:.1f}</text>')
        o.append(f'<text x="{(L+W-R)/2}" y="{b+38}" fill="{ink}" font-size="14" text-anchor="middle">{xlab}</text>')
        o.append(f'<text transform="translate(20,{(t+b)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Final validation loss</text>')
        (E,A,a),(K,p)=FITS[key]
        xs=[X0*(X1/X0)**(i/200) for i in range(201)]
        pts=lambda f:" ".join(f"{sx(x):.1f},{sy(f(x)):.1f}" for x in xs if Y0<=f(x)<=Y1)
        o.append(f'<polyline points="{pts(lambda x:E+A*x**-a)}" fill="none" stroke="{ink}" stroke-width="2"/>')
        o.append(f'<polyline points="{pts(lambda x:K*x**-p)}" fill="none" stroke="{mut}" stroke-width="2" stroke-dasharray="7 5"/>')
        for i,(n,l) in enumerate(zip(N,LW)):
            c=acc2 if i==5 else acc
            o.append(f'<circle cx="{sx(n):.1f}" cy="{sy(l):.1f}" r="5.5" fill="{c}" stroke="{surf}" stroke-width="2"/>')
            o.append(f'<text x="{sx(n)+9:.1f}" y="{sy(l)-8:.1f}" fill="{ink}" font-size="12">{NAMES[i]}</text>')
        lx=L+14; ly=sy(3.45) if key=='t' else sy(3.45)
        o.append(f'<line x1="{lx}" x2="{lx+28}" y1="{ly-24}" y2="{ly-24}" stroke="{ink}" stroke-width="2"/>')
        sup=lambda v:f'<tspan dy="-6" font-size="10">−{v}</tspan>'
        ta,tp={'t':('0.365','0.098'),'e':('0.135','0.068')}[key]
        o.append(f'<text x="{lx+34}" y="{ly-20}" fill="{ink}" font-size="13">fit with a floor: L = {E:.2f} + {A:.1f} × N{sup(ta)}</text>')
        o.append(f'<line x1="{lx}" x2="{lx+28}" y1="{ly}" y2="{ly}" stroke="{mut}" stroke-width="2" stroke-dasharray="7 5"/>')
        o.append(f'<text x="{lx+34}" y="{ly+4}" fill="{ink}" font-size="13">pure power law: L = {K:.2f} × N{sup(tp)}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#38bdf8','#d95926'))
