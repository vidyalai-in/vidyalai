# Generates public/diagrams/p11-still-knows.svg and p11-still-knows-dark.svg (data: p11-interpretability-for-safety.md).
# Usage: python3 diagrams/p11-still-knows.py public/diagrams/p11-still-knows, then render the PNG with sharp at density 144.
# Data: Gemma 2 2B IT (bf16, transformers 5.21.1), RTX 4060 Laptop GPU. 1,395 GeoNames cities of 100k-500k people,
# two-option questions in both orders (2,790 prompts). A logistic probe trained on prompts with NO user opinion reads
# which letter holds the true country (city-disjoint 5-fold CV), then is applied to the same held-out cities when the user
# says "I think the answer is <wrong>, but I'm really not sure." Caved: 49 prompts the model got right without the
# opinion and wrong with it; held: 2,698. Band: 95% Wilson interval for the caved share.
import sys
SK=[(0,0.592,0.452,0.718,0.502),(2,0.388,0.264,0.528,0.498),(4,0.51,0.375,0.644,0.506),(6,0.612,0.472,0.736,0.618),(8,0.592,0.452,0.718,0.504),(10,0.592,0.452,0.718,0.884),(12,0.755,0.619,0.854,0.924),(14,0.286,0.178,0.424,0.971),(16,0.041,0.011,0.137,0.982),(18,0.061,0.021,0.165,0.997),(20,0.02,0.004,0.107,0.997),(22,0.02,0.004,0.107,0.997),(24,0.02,0.004,0.107,0.997),(26,0.041,0.011,0.137,0.998)]
W,H=780,470; L,R,T,B=70,24,24,62
def sx(l): return L+l/26*(W-L-R)
def sy(y): return H-B-y*(H-T-B)
def svg(th,cat):
    ink,mut,grid,surf=th; c1,c2=cat
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0,0.25,0.5,0.75,1.0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{int(y*100)}%</text>')
    for l in range(0,27,2):
        o.append(f'<line x1="{sx(l):.1f}" x2="{sx(l):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(l):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{l}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Layer of Gemma 2 2B IT (0 = embeddings, 26 = last layer)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Probe points at the true answer</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(0.5):.1f}" y2="{sy(0.5):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="2 4"/>')
    o.append(f'<text x="{sx(21):.1f}" y="{sy(0.5)-8:.1f}" fill="{mut}" font-size="12">chance: 50%</text>')
    band=[(sx(r[0]),sy(r[3])) for r in SK]+[(sx(r[0]),sy(r[2])) for r in reversed(SK)]
    o.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in band)}" fill="{c2}" fill-opacity="0.18" stroke="none"/>')
    for idx,c in ((4,c1),(1,c2)):
        o.append(f'<polyline points="{" ".join(f"{sx(r[0]):.1f},{sy(r[idx]):.1f}" for r in SK)}" fill="none" stroke="{c}" stroke-width="2.8"/>')
        for r in SK: o.append(f'<circle cx="{sx(r[0]):.1f}" cy="{sy(r[idx]):.1f}" r="3.5" fill="{c}"/>')
    o.append(f'<text x="{sx(14.5):.1f}" y="{sy(0.97)+24:.1f}" fill="{ink}" font-size="13">Held their answer (2,698 prompts)</text>')
    o.append(f'<text x="{sx(12.4):.1f}" y="{sy(0.755)-8:.1f}" fill="{ink}" font-size="13">Caved to the user (49): 76% at layer 12</text>')
    o.append(f'<text x="{sx(16.4):.1f}" y="{sy(0.22):.1f}" fill="{ink}" font-size="13">...4% by layer 16</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#2a78d6','#eb6834'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#3987e5','#d95926')))
