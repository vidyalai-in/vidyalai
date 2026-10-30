# Generates public/diagrams/p09-strength-sweep.svg and -dark.svg (data: p09-activation-steering.md).
# Usage: python3 diagrams/p09-strength-sweep.py public/diagrams/p09-strength-sweep, then render the PNG with sharp at density 144.
# Data: Qwen2.5-1.5B-Instruct (bf16, transformers 5.21.1 hooks, RTX 4060 Laptop GPU). alpha x r added at layer 17, all positions,
# where r is the selected refusal direction (difference of means, norm 23.8). Refusal = refusal phrase in a 64-token greedy reply,
# on 100 held-out harmless (Alpaca) and 100 harmful (JailbreakBench) requests. Delta loss = mean next-token cross-entropy change on
# 100 held-out FineWeb-Edu documents of 256 tokens (no chat template), vs alpha = 0 (2.359 nats).
# Rows: (alpha, harmless refusal %, harmful refusal %, delta loss in nats)
import sys
S=[(-2, 1, 3, 0.958), (-1.5, 1, 2, 0.56), (-1, 1, 20, 0.258), (-0.5, 1, 65, 0.068), (0, 10, 99, 0.0), (0.5, 76, 99, 0.1), (1, 98, 89, 0.379), (1.5, 98, 90, 0.813), (2, 99, 96, 1.425), (3, 82, 82, 2.957), (4, 5, 2, 4.654), (6, 0, 0, 7.632)]
W,H=780,560; L,R,T,B=70,24,24,62
X0,X1=-2,6
P1=(T,T+250); P2=(T+300,H-B)      # panel pixel ranges (top, bottom)
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
def sy1(y): return P1[1]-y/100*(P1[1]-P1[0])
def sy2(y): return P2[1]-y/8*(P2[1]-P2[0])
def svg(th,cat):
    ink,mut,grid,surf=th; c1,c2=cat
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    o.append(f'<rect x="{sx(2.5):.1f}" y="{P1[0]}" width="{sx(X1)-sx(2.5):.1f}" height="{P2[1]-P1[0]}" fill="{grid}" fill-opacity="0.6"/>')
    o.append(f'<text x="{sx(4.25):.1f}" y="{P1[0]+16}" fill="{mut}" font-size="12" text-anchor="middle">text breaks down</text>')
    for y in (0,25,50,75,100):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy1(y):.1f}" y2="{sy1(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy1(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for y in (0,2,4,6,8):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy2(y):.1f}" y2="{sy2(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy2(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">+{y}</text>')
    for x in range(-2,7):
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:+d}</text>'.replace("+0<","0<"))
    o.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{P1[0]}" y2="{P2[1]}" stroke="{mut}" stroke-width="1.2" stroke-dasharray="2 3"/>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Strength: multiples of the refusal direction added at layer 17 (negative = subtracted)</text>')
    o.append(f'<text transform="translate(20,{(P1[0]+P1[1])/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Replies that refuse</text>')
    o.append(f'<text transform="translate(20,{(P2[0]+P2[1])/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Extra loss, nats</text>')
    for idx,c in ((2,c1),(1,c2)):
        pts=[(s[0],s[idx]) for s in S]
        o.append(f'<polyline points="{" ".join(f"{sx(a):.1f},{sy1(v):.1f}" for a,v in pts)}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for a,v in pts: o.append(f'<circle cx="{sx(a):.1f}" cy="{sy1(v):.1f}" r="3.5" fill="{c}"/>')
    pts=[(s[0],s[3]) for s in S]
    o.append(f'<polyline points="{" ".join(f"{sx(a):.1f},{sy2(v):.1f}" for a,v in pts)}" fill="none" stroke="{ink}" stroke-width="2.5"/>')
    for a,v in pts: o.append(f'<circle cx="{sx(a):.1f}" cy="{sy2(v):.1f}" r="3.5" fill="{ink}"/>')
    o.append(f'<text x="{sx(-1.9):.1f}" y="{sy1(86):.1f}" fill="{c1}" font-size="13" font-weight="bold">harmful requests</text>')
    o.append(f'<text x="{sx(0.35):.1f}" y="{sy1(52):.1f}" fill="{c2}" font-size="13" font-weight="bold">harmless requests</text>')
    o.append(f'<text x="{sx(-1.9):.1f}" y="{sy2(2.6):.1f}" fill="{ink}" font-size="13">loss on plain web text</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#2a78d6','#eb6834'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#3987e5','#d95926')))
