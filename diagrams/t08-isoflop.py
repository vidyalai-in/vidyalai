# Generates public/diagrams/t08-isoflop.svg and t08-isoflop-dark.svg (data: t08-scaling-laws-experiment.md).
# Usage: python3 diagrams/t08-isoflop.py public/diagrams/t08-isoflop, then render the PNG with sharp at density 144.
# Data: three runs with the same training compute, 6 x N x D = 2.61e16 FLOPs (s15's main run): iso_s11, s15, iso_s22.
# Loss on the whole validation shard. Parabola in log N through the three points (t08 scripts/fit.py); published optima
# from Chinchilla eq. 10 (Hoffmann et al., 2022) and the Epoch refit (Besiroglu et al., 2024, eq. 3), extrapolated.
import math, sys
PTS=[(10938960,397737984,4.0037),(14749344,294977536,3.9731),(22025088,197525504,4.0828)]
X0,X1=8e6,3e7; Y0,Y1=3.94,4.12
W,H=780,430; L,R,T,B=70,24,24,62
def sx(x): return L+(math.log(x)-math.log(X0))/(math.log(X1)-math.log(X0))*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
xs=[math.log(p[0]) for p in PTS]; ys=[p[2] for p in PTS]
def parab(x):  # Lagrange form through the three points
    return sum(ys[i]*math.prod((x-xs[j])/(xs[i]-xs[j]) for j in range(3) if j!=i) for i in range(3))
c2=sum(ys[i]/math.prod(xs[i]-xs[j] for j in range(3) if j!=i) for i in range(3))
c1=-sum(ys[i]*sum(xs[j] for j in range(3) if j!=i)/math.prod(xs[i]-xs[j] for j in range(3) if j!=i) for i in range(3))
xo=-c1/(2*c2); NO=math.exp(xo); C=6*14749344*294977536
def svg(th,acc,acc2):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in (8e6,1e7,1.5e7,2e7,3e7):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x/1e6:g}M</text>')
    for y in (3.95,4.0,4.05,4.1):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:.2f}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Model size, total parameters (log scale); tokens = 2.61 × 10¹⁶ FLOPs / (6 × size)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Final validation loss</text>')
    for lab,n,dy in (('Chinchilla eq. 10: 15.5M',15.5e6,0),('Epoch refit: 12.4M',12.4e6,0)):
        o.append(f'<line x1="{sx(n):.1f}" x2="{sx(n):.1f}" y1="{T+20}" y2="{H-B}" stroke="{acc2}" stroke-width="1.8" stroke-dasharray="5 5"/>')
        o.append(f'<text x="{sx(n)+(5 if n>14e6 else -5):.1f}" y="{T+14}" fill="{ink}" font-size="12" text-anchor="{"start" if n>14e6 else "end"}">{lab}</text>')
    pts=" ".join(f"{sx(math.exp(x)):.1f},{sy(parab(x)):.1f}" for x in [math.log(X0)+(math.log(X1)-math.log(X0))*i/200 for i in range(201)] if Y0<=parab(x)<=Y1)
    o.append(f'<polyline points="{pts}" fill="none" stroke="{mut}" stroke-width="2"/>')
    o.append(f'<circle cx="{sx(NO):.1f}" cy="{sy(parab(xo)):.1f}" r="5" fill="{surf}" stroke="{ink}" stroke-width="2"/>')
    halo=f'paint-order="stroke" stroke="{surf}" stroke-width="4" stroke-linejoin="round"'
    o.append(f'<text x="{sx(NO)-10:.1f}" y="{sy(parab(xo))+28:.1f}" fill="{ink}" font-size="13" text-anchor="end" {halo}>parabola minimum: {NO/1e6:.1f}M, {C/6/NO/NO:.1f} tokens per parameter</text>')
    for (n,d,l),(dx,dy,anc) in zip(PTS,((10,-34,'start'),(12,24,'start'),(-12,-8,'end'))):
        o.append(f'<circle cx="{sx(n):.1f}" cy="{sy(l):.1f}" r="6" fill="{acc}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{sx(n)+dx:.1f}" y="{sy(l)+dy:.1f}" fill="{ink}" font-size="13" font-weight="bold" text-anchor="{anc}" {halo}>{n/1e6:.0f}M parameters × {d/1e6:.0f}M tokens</text>')
        o.append(f'<text x="{sx(n)+dx:.1f}" y="{sy(l)+dy+18:.1f}" fill="{ink}" font-size="13" text-anchor="{anc}" {halo}>{d/n:.1f} tokens per parameter: loss {l:.3f}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#38bdf8','#d95926'))
