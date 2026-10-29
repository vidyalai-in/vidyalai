import sys
W,H=780,640; L,R=70,24
P1=(84,330); P2=(392,590)
X1=max(len(r[1]) for r in RUNS)
X1=((X1+24)//25)*25
def sx(x): return L+x/X1*(W-L-R)
def mk(y0,y1,p): return lambda y: p[1]-(y-y0)/(y1-y0)*(p[1]-p[0])
ymax=max(max(v for _,v in r[2]) for r in RUNS)
YT=1.0 if ymax>0.75 else 0.8
sy1=mk(0,YT,P1); sy2=mk(0,YT,P2)
def smooth(v,k=9):
    return [sum(v[max(0,i-k//2):i+k//2+1])/len(v[max(0,i-k//2):i+k//2+1]) for i in range(len(v))]
def line(o,pts,c,w=2.5,op=1.0,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    o.append(f'<polyline fill="none" stroke="{c}" stroke-width="{w}" opacity="{op}"{d} points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}"/>')
def svg(th,cols):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    lx=L
    for k,(lab,_,_) in enumerate(RUNS):
        c=cols[k%len(cols)]
        o.append(f'<line x1="{lx}" x2="{lx+22}" y1="18" y2="18" stroke="{c}" stroke-width="{3.5 if k==0 else 2.5}"/>')
        o.append(f'<text x="{lx+28}" y="22" fill="{ink}" font-size="13"{" font-weight=\"bold\"" if k==0 else ""}>{lab}</text>')
        lx+=36+7.2*len(lab)
    for p,sy,title in ((P1,sy1,'Training reward: share of the 256 sampled replies the checker accepts (9-step average)'),
                       (P2,sy2,'Held-out accuracy: 500 problems, greedy, strict checker')):
        o.append(f'<text x="{L}" y="{p[0]-14}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for y in [i/10 for i in range(0,int(YT*10)+1,2)]:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:.0%}</text>')
        for x in range(0,X1+1,25):
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{p[0]}" y2="{p[1]}" stroke="{grid}" stroke-width="1"/>')
    for x in range(0,X1+1,25):
        o.append(f'<text x="{sx(x):.1f}" y="{P2[1]+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">GRPO step (32 problems, 8 replies each)</text>')
    for k in reversed(range(len(RUNS))):
        lab,rw,te=RUNS[k]; c=cols[k%len(cols)]; w=3.5 if k==0 else 2.2
        line(o,[(sx(i),sy1(v)) for i,v in enumerate(smooth(rw))],c,w)
        line(o,[(sx(s),sy2(v)) for s,v in te],c,w)
        for s,v in te:
            o.append(f'<circle cx="{sx(s):.1f}" cy="{sy2(v):.1f}" r="{4.5 if k==0 else 3.5}" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
    s,v=RUNS[0][2][-1]
    o.append(f'<text x="{sx(s)-6:.1f}" y="{sy2(v)-10:.1f}" fill="{cols[0]}" font-size="13" font-weight="bold" text-anchor="end">{v:.1%}</text>')
    s,v=RUNS[0][2][0]
    o.append(f'<text x="{sx(s)+8:.1f}" y="{sy2(v)-12:.1f}" fill="{ink}" font-size="13">warm start: {v:.1%}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,['#0b5fd6','#eb6834','#1a9e5c','#8b5cf6','#4a5b78','#d4a017']))
open(sys.argv[1]+'-dark.svg','w').write(svg(dark,['#38bdf8','#d95926','#34d399','#a78bfa','#9fb0cf','#e8b931']))
