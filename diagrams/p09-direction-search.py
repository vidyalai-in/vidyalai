# Generates public/diagrams/p09-direction-search.svg and -dark.svg (data: p09-activation-steering.md).
# Usage: python3 diagrams/p09-direction-search.py public/diagrams/p09-direction-search, then render the PNG with sharp at density 144.
# Data: Qwen2.5-1.5B-Instruct (Apache-2.0, bf16, Hugging Face transformers 5.21.1 forward hooks, RTX 4060 Laptop GPU).
# Candidate refusal directions = difference of mean residual vectors (start of each layer) between 128 harmful (AdvBench)
# and 128 harmless (Alpaca) instructions, here at the last template token (position -1, the newline after "assistant").
# Refusal score = log-odds that the first reply token is "I" or "As", averaged over 32 harmful / 32 harmless validation instructions.
# Bypass: the direction ablated at every layer, harmful prompts. Induce: the direction added at its own layer, harmless prompts.
# KL: how much ablation changes next-token distributions on harmless prompts (selection requires KL < 0.1, layer < 0.8 x 28).
import sys
BY=[(1, 2.5), (2, 1.35), (3, 1.72), (4, 1.49), (5, 1.55), (6, 1.84), (7, 1.65), (8, 1.35), (9, 1.58), (10, 1.45), (11, 0.12), (12, 1.35), (13, 0.36), (14, -4.39), (15, -4.48), (16, -4.81), (17, -2.02), (18, -1.44), (19, -1.97), (20, -6.87), (21, -3.82), (22, -4.28), (23, -5.37), (24, -4.29), (25, -2.22), (26, -4.69), (27, -3.08)]
IN=[(1, -4.71), (2, -4.91), (3, -4.83), (4, -4.69), (5, -5.13), (6, -5.14), (7, -5.08), (8, -4.88), (9, -4.77), (10, -4.29), (11, -4.1), (12, -2.28), (13, -1.98), (14, 1.18), (15, 1.59), (16, 1.41), (17, 1.8), (18, 1.82), (19, 2.2), (20, 1.63), (21, 1.62), (22, 1.71), (23, 1.73), (24, 1.57), (25, 1.44), (26, 1.39), (27, 1.05)]
KL=[(1, 0.316), (2, 0.035), (3, 0.213), (4, 0.024), (5, 0.131), (6, 0.255), (7, 0.02), (8, 0.041), (9, 0.029), (10, 0.272), (11, 0.477), (12, 0.129), (13, 0.165), (14, 0.827), (15, 1.33), (16, 0.751), (17, 0.068), (18, 0.058), (19, 0.084), (20, 1.284), (21, 0.107), (22, 0.104), (23, 0.162), (24, 0.134), (25, 0.114), (26, 0.141), (27, 0.149)]
BASE_H=1.66; BASE_S=-4.8; SEL=17
W,H=780,470; L,R,T,B=70,24,24,62
X0,X1=1,27; Y0,Y1=-8,4
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
def svg(th,cat):
    ink,mut,grid,surf=th; c1,c2=cat
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    o.append(f'<rect x="{sx(22.4):.1f}" y="{T}" width="{sx(X1)-sx(22.4):.1f}" height="{H-B-T}" fill="{grid}" fill-opacity="0.6"/>')
    o.append(f'<text x="{sx(24.7):.1f}" y="{sy(-6.6):.1f}" fill="{mut}" font-size="12" text-anchor="middle">too close to</text>')
    o.append(f'<text x="{sx(24.7):.1f}" y="{sy(-6.6)+15:.1f}" fill="{mut}" font-size="12" text-anchor="middle">the output</text>')
    for y in (-8,-6,-4,-2,0,2,4):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:+d}</text>')
    for x in (1,5,10,15,20,25):
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Layer the direction comes from (Qwen2.5-1.5B-Instruct has 28)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Refusal score (log-odds of "I" or "As" first)</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="{mut}" stroke-width="1.2"/>')
    for base,c,lab,lx,ly in ((BASE_H,c1,"harmful, unedited: +1.7",6.5,2.45),(BASE_S,c2,"harmless, unedited: -4.8",1.2,-5.75)):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(base):.1f}" y2="{sy(base):.1f}" stroke="{c}" stroke-width="1.8" stroke-dasharray="7 5"/>')
        o.append(f'<text x="{sx(lx):.1f}" y="{sy(ly):.1f}" fill="{ink}" font-size="12">{lab}</text>')
    for pts,c in ((BY,c1),(IN,c2)):
        o.append(f'<polyline points="{" ".join(f"{sx(l):.1f},{sy(v):.1f}" for l,v in pts)}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for l,v in pts: o.append(f'<circle cx="{sx(l):.1f}" cy="{sy(v):.1f}" r="{6 if l==SEL else 3.5}" fill="{c}" stroke="{surf}" stroke-width="{2 if l==SEL else 0}"/>')
    o.append(f'<line x1="{sx(SEL):.1f}" x2="{sx(SEL):.1f}" y1="{T+16}" y2="{H-B}" stroke="{ink}" stroke-width="1" stroke-dasharray="2 3"/>')
    o.append(f'<text x="{sx(SEL):.1f}" y="{T+10}" fill="{ink}" font-size="13" font-weight="bold" text-anchor="middle">chosen: layer 17</text>')
    o.append(f'<text x="{sx(4.0):.1f}" y="{sy(0.6):.1f}" fill="{ink}" font-size="13">ablated, harmful prompts</text>')
    o.append(f'<text x="{sx(4.5):.1f}" y="{sy(-3.3):.1f}" fill="{ink}" font-size="13">added, harmless prompts</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#2a78d6','#eb6834'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#3987e5','#d95926')))
