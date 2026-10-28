import sys
W,H=780,760; L,R=70,24
P1=(44,224); P2=(290,470); P3=(536,700)        # panel y-ranges (top, bottom)
X0,X1=0,314
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
def mk(y0,y1,p):
    return lambda y: p[1]-(y-y0)/(y1-y0)*(p[1]-p[0])
sy1=mk(0.56,0.72,P1); sy2=mk(0.40,0.85,P2); sy3=mk(-6.0,1.0,P3)
def smooth(v,k=15):
    out=[]
    for i in range(len(v)):
        w=v[max(0,i-k//2):i+k//2+1]; out.append(sum(w)/len(w))
    return out
def line(o,pts,c,w=2.5,op=1.0,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    o.append(f'<polyline fill="none" stroke="{c}" stroke-width="{w}" opacity="{op}"{d} points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}"/>')
def dots(o,pts,c,surf):
    for x,y in pts:
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
def svg(th,acc,acc2):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    panels=((P1,sy1,(0.60,0.65,0.70),'DPO loss (faint: training, 15-step average; dots: 500 held-out pairs)','{:.2f}'),
            (P2,sy2,(0.4,0.5,0.6,0.7,0.8),'Preference accuracy: implicit reward of chosen above rejected','{:.0%}'),
            (P3,sy3,(-6,-4,-2,0),'Held-out pairs: change in log-probability per reply, in nats','{:g}'))
    for p,sy,ys,title,fmt in panels:
        o.append(f'<text x="{L}" y="{p[0]-14}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for y in ys:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{fmt.format(y)}</text>')
        for x in range(0,315,50):
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{p[0]}" y2="{p[1]}" stroke="{grid}" stroke-width="1"/>')
    for x in range(0,315,50):
        o.append(f'<text x="{sx(x):.1f}" y="{P3[1]+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Optimizer step (about 51 pairs each)</text>')
    # panel 1: loss
    import math
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy1(math.log(2)):.1f}" y2="{sy1(math.log(2)):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{W-R-4}" y="{sy1(math.log(2))-7:.1f}" fill="{mut}" font-size="12" text-anchor="end">ln 2 = 0.693: policy equals reference</text>')
    line(o,[(sx(i),sy1(v)) for i,v in enumerate(smooth(TRAIN_LOSS))],acc,2.0,0.45)
    line(o,[(sx(s),sy1(v)) for s,v in VAL_LOSS],acc)
    dots(o,[(sx(s),sy1(v)) for s,v in VAL_LOSS],acc,surf)
    s,v=VAL_LOSS[-1]
    o.append(f'<text x="{sx(s)-6:.1f}" y="{sy1(v)+26:.1f}" fill="{acc}" font-size="13" font-weight="bold" text-anchor="end">held-out {v:.3f}</text>')
    # panel 2: accuracy
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy2(0.5):.1f}" y2="{sy2(0.5):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{W-R-4}" y="{sy2(0.5)+16:.1f}" fill="{mut}" font-size="12" text-anchor="end">chance</text>')
    line(o,[(sx(i),sy2(v)) for i,v in enumerate(smooth(TRAIN_ACC))],acc,2.0,0.45)
    line(o,[(sx(s),sy2(v)) for s,v in VAL_ACC],acc)
    dots(o,[(sx(s),sy2(v)) for s,v in VAL_ACC],acc,surf)
    s,v=VAL_ACC[-1]
    o.append(f'<text x="{sx(s)-6:.1f}" y="{sy2(v)+22:.1f}" fill="{acc}" font-size="13" font-weight="bold" text-anchor="end">{v:.1%}</text>')
    # panel 3: log-prob changes
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy3(0):.1f}" y2="{sy3(0):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    for pts,c,lab in ((VAL_DC,acc,'chosen'),(VAL_DR,acc2,'rejected')):
        line(o,[(sx(s),sy3(v)) for s,v in pts],c); dots(o,[(sx(s),sy3(v)) for s,v in pts],c,surf)
        s,v=pts[-1]
        o.append(f'<text x="{sx(s)-6:.1f}" y="{sy3(v)+(-10 if lab=="chosen" else 20):.1f}" fill="{c}" font-size="13" font-weight="bold" text-anchor="end">{lab}: {v:+.1f}</text>')
    o.append(f'<text x="{sx(60):.1f}" y="{sy3(-3.0):.1f}" fill="{ink}" font-size="13">Both replies lose probability; the rejected one loses more.</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#38bdf8','#d95926'))
