# Generates public/diagrams/t13-kd-loss.svg and t13-kd-loss-dark.svg (data: t13-distillation-synthetic-data.md).
# Usage: python3 diagrams/t13-kd-loss.py public/diagrams/t13-kd-loss, then render the PNG with sharp at density 144.
# Data: held-out loss logged by train.py (plain CE) and fromscratch/distill.py (KD) for the S15 student (14.7M parameters),
# 1,000 steps of 65,536 tokens on the same token stream, seed 1337, peak lr 2.08e-3. Validation: the same 655,360
# tokens of the validation shard each time. Teacher: Part 7's 88M model. (step, loss); tokens = step * 65,536.
CE=[(0, 10.4425), (50, 7.0999), (100, 6.5187), (150, 6.1769), (200, 5.9386), (250, 5.7391), (300, 5.5871), (350, 5.4617), (400, 5.3434), (450, 5.2452), (500, 5.1535), (550, 5.1055), (600, 5.0206), (650, 4.9597), (700, 4.9149), (750, 4.8795), (800, 4.8494), (850, 4.8174), (900, 4.7836), (950, 4.7643), (1000, 4.7437)]
KD1=[(0, 10.4425), (50, 7.0617), (100, 6.4551), (150, 6.0484), (200, 5.7483), (250, 5.5357), (300, 5.3742), (350, 5.2362), (400, 5.1208), (450, 5.018), (500, 4.9184), (550, 4.8701), (600, 4.8052), (650, 4.7356), (700, 4.6928), (750, 4.6555), (800, 4.6296), (850, 4.6014), (900, 4.5697), (950, 4.5511), (1000, 4.53251)]
KD2=[(0, 10.4425), (100, 7.0608), (200, 6.3295), (300, 6.0004), (400, 5.6435), (500, 5.3912), (600, 5.2331), (700, 5.1119), (800, 5.038), (900, 4.979), (1000, 4.9408)]
import sys
W,H=780,600; L,R=70,62
P1=(44,300); P2=(372,530)
TOK=65536/1e6
X0,X1=0,1000*TOK
def sx(m): return L+(m-X0)/(X1-X0)*(W-L-R)
def mk(y0,y1,p): return lambda y: p[1]-(y-y0)/(y1-y0)*(p[1]-p[0])
sy1=mk(4.4,7.2,P1); sy2=mk(-0.6,0.3,P2)
ce=dict(CE)
def svg(th,cg,c1,c2):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for p,sy,ys,fmt,title in ((P1,sy1,(4.5,5.0,5.5,6.0,6.5,7.0),'{:.1f}','Held-out loss (nats per token) as the S15 student trains'),
                              (P2,sy2,(-0.6,-0.4,-0.2,0.0,0.2),'{:+.1f}','Advantage over plain CE at the same step (CE loss minus KD loss)')):
        o.append(f'<text x="{L}" y="{p[0]-14}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for y in ys:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{fmt.format(y).replace("+0.0","0.0")}</text>')
        for x in range(0,70,10):
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{p[0]}" y2="{p[1]}" stroke="{grid}" stroke-width="1"/>')
    for x in range(0,70,10):
        o.append(f'<text x="{sx(x):.1f}" y="{P2[1]+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}M</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Training tokens (the same stream for all three runs)</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy2(0):.1f}" y2="{sy2(0):.1f}" stroke="{cg}" stroke-width="2"/>')
    runs=((CE,cg,'Plain CE (hard labels)'),(KD2,c2,'KD, T = 2, alpha 0.9 (Hinton mix)'),(KD1,c1,'KD, T = 1, soft labels only'))
    for data,c,_ in runs:
        pts=[(s,v) for s,v in data if s>=50]
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(s*TOK):.1f},{sy1(v):.1f}" for s,v in pts)}"/>')
        for s,v in pts:
            o.append(f'<circle cx="{sx(s*TOK):.1f}" cy="{sy1(v):.1f}" r="3.5" fill="{c}" stroke="{surf}" stroke-width="1"/>')
        s,v=pts[-1]
        o.append(f'<text x="{sx(s*TOK)+7:.1f}" y="{sy1(v)+4:.1f}" fill="{c}" font-size="13" font-weight="bold">{v:.3f}</text>')
        if data is not CE:
            g=[(s,ce[s]-v) for s,v in data if s>=50 and s in ce]
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(s*TOK):.1f},{sy2(d):.1f}" for s,d in g)}"/>')
            for s,d in g:
                o.append(f'<circle cx="{sx(s*TOK):.1f}" cy="{sy2(d):.1f}" r="3.5" fill="{c}" stroke="{surf}" stroke-width="1"/>')
            s,d=g[-1]
            o.append(f'<text x="{sx(s*TOK)+7:.1f}" y="{sy2(d)+4:.1f}" fill="{c}" font-size="13" font-weight="bold">{d:+.3f}</text>')
    lx,ly=sx(30),P1[0]+22
    for i,(_,c,t) in enumerate(runs[::-1]):
        y=ly+20*i
        o.append(f'<line x1="{lx:.1f}" x2="{lx+26:.1f}" y1="{y-4}" y2="{y-4}" stroke="{c}" stroke-width="3"/><text x="{lx+34:.1f}" y="{y}" fill="{ink}" font-size="13">{t}</text>')
    o.append(f'<text x="{lx:.1f}" y="{ly+66}" fill="{mut}" font-size="12">Teacher (88M, Part 7): 3.38. Step 0 (10.44) is off the chart.</text>')
    o.append(f'<text x="{sx(40):.1f}" y="{sy2(0)-8:.1f}" fill="{mut}" font-size="12">0 = no better than plain CE</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#4a5b78','#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#9fb0cf','#38bdf8','#d95926'))
