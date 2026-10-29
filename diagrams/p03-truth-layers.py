# Generates public/diagrams/p03-truth-layers.svg and p03-truth-layers-dark.svg (data: p03-linear-probes.md).
# Usage: python3 diagrams/p03-truth-layers.py public/diagrams/p03-truth-layers, then render the PNG with sharp at density 144.
# Data: linear probe accuracy (mean and 95% bootstrap CI over cities) on 1,134 "The city of X is in Y." statements
# (567 GeoNames cities, one true and one false each), city-disjoint 5-fold cross-validation, read at the final "." token.
# Layer l = residual stream entering block l (0 = embeddings; last = output of the last block). GPT-2 small fp32,
# Qwen2.5-0.5B bf16, TransformerLens 3.9.0 without weight processing, RTX 4060 Laptop GPU.
# Dashed lines: the model's own output as a classifier (log-probability of the country, threshold fit in the same folds).
import sys
GPT2=[(0,0.5009,0.4832,0.5185),(1,0.6332,0.6102,0.6588),(2,0.6323,0.6076,0.6561),(3,0.6728,0.6490,0.6966),(4,0.6949,0.6693,0.7187),(5,0.7496,0.7266,0.7734),(6,0.7628,0.7399,0.7875),(7,0.7716,0.7496,0.7945),(8,0.7840,0.7628,0.8051),(9,0.7787,0.7566,0.8025),(10,0.7672,0.7443,0.7910),(11,0.7716,0.7504,0.7937),(12,0.7743,0.7504,0.7963)]
GPT2_BASE=0.6861
QWEN=[(0,0.5000,0.5000,0.5000),(1,0.5220,0.4956,0.5494),(2,0.5873,0.5617,0.6120),(3,0.6243,0.5970,0.6491),(4,0.6552,0.6296,0.6799),(5,0.7205,0.6966,0.7425),(6,0.7425,0.7196,0.7637),(7,0.8474,0.8263,0.8695),(8,0.9268,0.9109,0.9409),(9,0.9471,0.9330,0.9594),(10,0.9453,0.9303,0.9586),(11,0.9489,0.9339,0.9612),(12,0.9471,0.9312,0.9594),(13,0.9418,0.9268,0.9550),(14,0.9427,0.9277,0.9550),(15,0.9453,0.9303,0.9586),(16,0.9444,0.9303,0.9568),(17,0.9453,0.9303,0.9586),(18,0.9427,0.9277,0.9568),(19,0.9409,0.9259,0.9550),(20,0.9436,0.9286,0.9568),(21,0.9427,0.9277,0.9559),(22,0.9339,0.9180,0.9480),(23,0.9347,0.9189,0.9489),(24,0.9303,0.9145,0.9444)]
QWEN_BASE=0.9347
W,H=780,460; L,R,T,B=70,24,24,62
Y0,Y1=0.45,1.0
def sx(f): return L+f*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
def svg(th,cat):
    ink,mut,grid,surf=th; c1,c2=cat
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0.5,0.6,0.7,0.8,0.9,1.0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{int(y*100)}%</text>')
    for f in (0,0.25,0.5,0.75,1.0):
        o.append(f'<line x1="{sx(f):.1f}" x2="{sx(f):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(f):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{f:g}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Depth (layer ÷ number of blocks: GPT-2 small 12, Qwen2.5-0.5B 24)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Held-out accuracy, true vs false</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(0.5):.1f}" y2="{sy(0.5):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="2 4"/>')
    o.append(f'<text x="{W-R-4}" y="{sy(0.5)-6:.1f}" fill="{mut}" font-size="12" text-anchor="end">chance: 50%</text>')
    for pts,n,base,c in ((GPT2,12,GPT2_BASE,c1),(QWEN,24,QWEN_BASE,c2)):
        band=[(sx(l/n),sy(hi)) for l,_,_,hi in pts]+[(sx(l/n),sy(lo)) for l,_,lo,_ in reversed(pts)]
        o.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in band)}" fill="{c}" fill-opacity="0.18" stroke="none"/>')
        o.append(f'<polyline points="{" ".join(f"{sx(l/n):.1f},{sy(m):.1f}" for l,m,_,_ in pts)}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for l,m,_,_ in pts: o.append(f'<circle cx="{sx(l/n):.1f}" cy="{sy(m):.1f}" r="3.5" fill="{c}"/>')
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(base):.1f}" y2="{sy(base):.1f}" stroke="{c}" stroke-width="1.8" stroke-dasharray="7 5"/>')
    o.append(f'<text x="{sx(0.01):.1f}" y="{sy(QWEN_BASE)+18:.1f}" fill="{ink}" font-size="13" text-anchor="start">Qwen\'s own output: 93.5%</text>')
    o.append(f'<text x="{sx(0.46):.1f}" y="{sy(0.985):.1f}" fill="{ink}" font-size="13" text-anchor="start">Qwen2.5-0.5B probe: peak 94.9%</text>')
    o.append(f'<text x="{sx(0.62):.1f}" y="{sy(0.7840)-12:.1f}" fill="{ink}" font-size="13" text-anchor="start">GPT-2 small probe: peak 78.4%</text>')
    o.append(f'<text x="{sx(0.62):.1f}" y="{sy(GPT2_BASE)+18:.1f}" fill="{ink}" font-size="13" text-anchor="start">GPT-2\'s own output: 68.6%</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#2a78d6','#eb6834'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#3987e5','#d95926')))
