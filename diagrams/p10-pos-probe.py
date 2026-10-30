# Generates public/diagrams/p10-pos-probe.svg and -dark.svg (data: p10-training-dynamics-interpretability.md).
# Usage: python3 diagrams/p10-pos-probe.py public/diagrams/p10-pos-probe, then render the PNG with sharp at density 144.
# Data: linear part-of-speech probes (lab/probes.py train_probe, 10 epochs) on the 88M model's residual stream at
# 12 snapshots; UD English EWT v2.18, 4,000 training sentences (64,014 words), 2,077 test sentences (24,386 words);
# control = Hewitt and Liang's random tag per word type, seed 0 (p10 scripts/probe_track.py).
import sys
D={"steps":[0,10,25,50,75,100,150,250,500,1000,2000,3360],"lookup":0.8405,"l8":[0.5231690314114656,0.5741819076519314,0.6277372262773723,0.684983187074551,0.7697449356187976,0.8165750840646272,0.865496596407775,0.8933404412367753,0.9089231526285574,0.9213893217419831,0.9285655704092513,0.9301238415484294],"l8u":[0.24820359281437127,0.2841317365269461,0.3458083832335329,0.4353293413173653,0.5553892215568862,0.5961077844311378,0.6916167664670658,0.7377245508982035,0.7706586826347306,0.7946107784431138,0.8302395209580838,0.8290419161676646],"l0":[0.7305831214631345,0.7315262855736898,0.7332075781185926,0.7606823587304191,0.8072664643647994,0.8268678750102518,0.8457311572213565,0.8598376117444435,0.8657016320839827,0.8681620601984745,0.8703354383662757,0.8705814811777249],"l8c":[0.5105798408923152,0.550643812023292,0.5685229229886,0.5513819404576397,0.578077585499877,0.6011646026408595,0.6312228327729025,0.6498810793077996,0.6469695727056508,0.6424177806938407,0.6382350528992045,0.6357746247847126]}
W=780; H=430; L=66; R=170; T=40; B=70
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    n=len(D['steps']); sx=lambda i: L+i/(n-1)*(W-L-R); y0,y1=0.2,1.0; sy=lambda v: H-B-(v-y0)/(y1-y0)*(H-T-B)
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (0.2,0.4,0.6,0.8,1.0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{round(v*100)}%</text>')
    for i,s in enumerate(D['steps']):
        o.append(f'<text x="{sx(i):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{s:,}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-B+42}" fill="{ink}" font-size="13" text-anchor="middle">Training step (snapshots, evenly spaced)</text>')
    o.append(f'<text x="{L}" y="{T-14}" fill="{ink}" font-size="14" font-weight="bold">Part-of-speech probe accuracy on held-out sentences</text>')
    yb=sy(D['lookup'])
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{yb:.1f}" y2="{yb:.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="2 4"/>')
    series=[('l8','layer 8, all words',acc,False),('l8u','layer 8, unseen words',acc,True),('l0','layer 0 (embeddings)',acc2,False),('l8c','layer 8, control tags',mut,True)]
    for k,name,col,dash in series:
        pts=' '.join(f'{sx(i):.1f},{sy(v):.1f}' for i,v in enumerate(D[k]))
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.2" {"stroke-dasharray=\"6 4\"" if dash else ""} points="{pts}"/>')
        for i,v in enumerate(D[k]): o.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{col}"/>')
    ends={k:D[k][-1] for k,_,_,_ in series}
    offs={'l8':4,'l0':2,'l8u':16,'l8c':4}
    for k,name,col,dash in series:
        o.append(f'<text x="{W-R+10}" y="{sy(ends[k])+offs[k]:.1f}" fill="{col}" font-size="12" font-weight="bold">{name}</text>')
    o.append(f'<text x="{L+8}" y="{yb-7:.1f}" fill="{mut}" font-size="12">word lookup, no model: 84.1%</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
