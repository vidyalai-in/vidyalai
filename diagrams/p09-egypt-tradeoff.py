# Generates public/diagrams/p09-egypt-tradeoff.svg and -dark.svg (data: p09-activation-steering.md).
# Usage: python3 diagrams/p09-egypt-tradeoff.py public/diagrams/p09-egypt-tradeoff, then render the PNG with sharp at density 144.
# Data: base Gemma 2 2B (bf16, transformers 5.21.1 hooks, RTX 4060 Laptop GPU). A unit direction times strength x 146.1 (the mean
# residual norm at that site on FineWeb-Edu) is added to the residual stream after layer 12 (the Gemma Scope layer_12 site), all positions.
# Directions: Gemma Scope 16k latent 7498's decoder row ("Egypt", Part 7); a difference of means (20 sentence templates naming Egypt
# minus the same templates naming 8 other countries); a random direction. y: share of 50 neutral prompts whose 40-token greedy
# continuation mentions Egypt, the Nile, a pharaoh, Cairo or a pyramid. x: extra next-token loss on 100 held-out FineWeb-Edu documents.
# Rows: (strength, extra loss nats, % mentioning Egypt)
import sys
MD=[(0.0, 0.0, 0.0), (0.1, 0.01, 0.0), (0.2, 0.064, 10.0), (0.3, 0.221, 24.0), (0.4, 0.485, 38.0), (0.5, 0.828, 56.0), (0.6, 1.229, 60.0), (0.8, 2.163, 62.0), (1.0, 3.143, 54.0)]
SA=[(0.0, 0.0, 0.0), (0.1, 0.007, 0.0), (0.2, 0.036, 0.0), (0.3, 0.093, 4.0), (0.4, 0.187, 10.0), (0.5, 0.331, 20.0), (0.6, 0.532, 32.0), (0.8, 1.09, 24.0), (1.0, 1.82, 8.0)]
RA=[(0.0, 0.0, 0.0), (0.1, 0.009, 0.0), (0.2, 0.033, 0.0), (0.3, 0.073, 0.0), (0.4, 0.131, 0.0), (0.5, 0.21, 0.0), (0.6, 0.311, 0.0), (0.8, 0.594, 0.0), (1.0, 1.001, 0.0)]
W,H=780,470; L,R,T,B=70,24,24,62
X0,X1=0,3.2; Y0,Y1=0,70
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
def svg(th,cat):
    ink,mut,grid,surf=th; c1,c2=cat
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0,10,20,30,40,50,60,70):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for x in (0,0.5,1,1.5,2,2.5,3):
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:g}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Side effect: extra loss on plain web text (nats)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Continuations that mention Egypt (of 50)</text>')
    for pts,c,lab,lx,ly in ((MD,c1,"difference of means",1.35,63.5),(SA,c2,"SAE latent 7498",0.62,36.5),(RA,mut,"random direction",1.25,3.2)):
        o.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for _,x,y in pts)}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for s,x,y in pts: o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{c}"/>')
        o.append(f'<text x="{sx(lx):.1f}" y="{sy(ly):.1f}" fill="{c}" font-size="13" font-weight="bold">{lab}</text>')
    for pts,c,keep,dx,dy in ((MD,c1,(0.2,0.4,0.8,1.0),-6,-9),(SA,c2,(0.4,0.6,1.0),8,-6)):
        for s,x,y in pts:
            if s in keep: o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{ink}" font-size="12" text-anchor="{"end" if dx<0 else "start"}">{s:g}</text>')
    o.append(f'<text x="{sx(3.15):.1f}" y="{sy(45):.1f}" fill="{mut}" font-size="12" text-anchor="end">numbers on points: strength,</text>')
    o.append(f'<text x="{sx(3.15):.1f}" y="{sy(45)+15:.1f}" fill="{mut}" font-size="12" text-anchor="end">in units of the residual norm</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#2a78d6','#eb6834'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#3987e5','#d95926')))
