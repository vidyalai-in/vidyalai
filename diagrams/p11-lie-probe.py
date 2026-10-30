# Generates public/diagrams/p11-lie-probe.svg and p11-lie-probe-dark.svg (data: p11-interpretability-for-safety.md).
# Usage: python3 diagrams/p11-lie-probe.py public/diagrams/p11-lie-probe, then render the PNG with sharp at density 144.
# Data: Gemma 2 2B IT (bf16, Hugging Face transformers 5.21.1, no weight processing), RTX 4060 Laptop GPU.
# Logistic-regression probe on the residual stream at the last prompt token (hidden_states[l]), trained on
# "answer correctly" vs "answer incorrectly" prompts about 567 GeoNames cities (both option orders), city-disjoint
# 5-fold cross-validation. Threshold per fold = 99th percentile of held-out honest scores (1% false positives).
# Lines: share of held-out prompts flagged. ~1,130 prompts per condition.
import sys
LIE=[(0,0.0,0.0,0.0,0.0),(2,1.0,1.0,1.0,1.0),(4,1.0,0.2598,1.0,1.0),(6,1.0,0.8509,1.0,1.0),(8,1.0,1.0,1.0,1.0),(10,1.0,1.0,1.0,1.0),(12,1.0,1.0,1.0,1.0),(14,1.0,1.0,1.0,0.2064),(16,1.0,1.0,1.0,0.0159),(18,1.0,1.0,1.0,0.0195),(20,1.0,1.0,1.0,0.1125),(22,1.0,1.0,1.0,0.0664),(24,1.0,1.0,1.0,0.0354),(26,1.0,1.0,1.0,0.0585)]
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
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Prompts flagged as lies (1% false-positive threshold)</text>')
    series=[(1,c1,'',3.0),(2,c1,'8 5',2.2),(3,c1,'2 4',2.2),(4,c2,'',3.0)]
    for idx,c,dash,wd in series:
        pts=" ".join(f"{sx(r[0]):.1f},{sy(r[idx]):.1f}" for r in LIE)
        d=f' stroke-dasharray="{dash}"' if dash else ''
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="{wd}"{d}/>')
        for r in LIE: o.append(f'<circle cx="{sx(r[0]):.1f}" cy="{sy(r[idx]):.1f}" r="3.2" fill="{c}"/>')
    o.append(f'<text x="{sx(15):.1f}" y="{sy(1.0)+22:.1f}" fill="{ink}" font-size="13">Lies: instructed, paraphrased, told to hide</text>')
    o.append(f'<text x="{sx(15):.1f}" y="{sy(1.0)+40:.1f}" fill="{ink}" font-size="13">(three lines on top of each other)</text>')
    o.append(f'<text x="{sx(4.3):.1f}" y="{sy(0.26)+4:.1f}" fill="{mut}" font-size="12">paraphrased: 26%</text>')
    o.append(f'<text x="{sx(14.6):.1f}" y="{sy(0.47):.1f}" fill="{ink}" font-size="13">Honest answers that mention lying</text>')
    o.append(f'<text x="{sx(14.6):.1f}" y="{sy(0.47)+18:.1f}" fill="{ink}" font-size="13">(false alarms): 100% up to layer 12,</text>')
    o.append(f'<text x="{sx(14.6):.1f}" y="{sy(0.47)+36:.1f}" fill="{ink}" font-size="13">1.6% at layer 16</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#2a78d6','#eb6834'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#3987e5','#d95926')))
