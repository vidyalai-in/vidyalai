import sys
W,H=780,560; L,R=70,24
P1=(40,250); P2=(320,500)        # top and bottom panel y-ranges (top, bottom)
X0,X1=0,291
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
def mk(y0,y1,p):
    return lambda y: p[1]-(y-y0)/(y1-y0)*(p[1]-p[0])
sy1=mk(1.9,2.9,P1); sy2=mk(1.6,3.0,P2)
def smooth(v,k=9):
    out=[]
    for i in range(len(v)):
        w=v[max(0,i-k//2):i+k//2+1]; out.append(sum(w)/len(w))
    return out
def svg(th,acc,acc2):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for p,sy,ys,title in ((P1,sy1,(2.0,2.2,2.4,2.6,2.8),'Held-out loss on assistant tokens (500 validation conversations)'),
                          (P2,sy2,(1.6,2.0,2.4,2.8),'Training loss per step (9-step average; faint: every step)')):
        o.append(f'<text x="{L}" y="{p[0]-12}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for y in ys:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:.1f}</text>')
        for x in range(0,300,50):
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{p[0]}" y2="{p[1]}" stroke="{grid}" stroke-width="1"/>')
    for x in range(0,300,50):
        o.append(f'<text x="{sx(x):.1f}" y="{P2[1]+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Optimizer step (about 59,000 padded tokens each)</text>')
    # top panel: base line and validation curves
    b=VAL_A[0][1]
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy1(b):.1f}" y2="{sy1(b):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{W-R-4}" y="{sy1(b)-7:.1f}" fill="{mut}" font-size="12" text-anchor="end">base model: {b:.2f}</text>')
    for val,c in ((VAL_B,acc2),(VAL_A,acc)):
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(s):.1f},{sy1(v):.1f}" for s,v in val)}"/>')
        for s,v in val:
            o.append(f'<circle cx="{sx(s):.1f}" cy="{sy1(v):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        s,v=val[-1]
        o.append(f'<text x="{sx(s)-8:.1f}" y="{sy1(v)+(18 if c==acc else -10):.1f}" fill="{c}" font-size="13" font-weight="bold" text-anchor="end">{v:.3f}</text>')
    o.append(f'<text x="{sx(100):.1f}" y="{sy1(2.42):.1f}" fill="{ink}" font-size="13">The two runs overlap: they never differ by more than 0.006.</text>')
    o.append(f'<text x="{sx(100):.1f}" y="{sy1(2.42)+18:.1f}" fill="{mut}" font-size="12">Step 0 is the base model from Part 7.</text>')
    # bottom panel
    for tr,c in ((TRAIN_B,acc2),(TRAIN_A,acc)):
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="1" opacity="0.3" points="{" ".join(f"{sx(i):.1f},{sy2(v):.1f}" for i,v in enumerate(tr))}"/>')
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(i):.1f},{sy2(v):.1f}" for i,v in enumerate(smooth(tr)))}"/>')
    # legend
    lx=L+330
    for i,(c,t) in enumerate(((acc,'Loss on assistant tokens only'),(acc2,'Loss on every token (ablation)'))):
        y=P2[0]+18+20*i
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y-4}" y2="{y-4}" stroke="{c}" stroke-width="3"/><text x="{lx+34}" y="{y}" fill="{ink}" font-size="13">{t}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#38bdf8','#d95926'))
