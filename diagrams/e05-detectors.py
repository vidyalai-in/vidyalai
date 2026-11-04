# Generates public/diagrams/e05-detectors.svg and e05-detectors-dark.svg (data: e05-benchmark-contamination.md).
# Usage: python3 diagrams/e05-detectors.py public/diagrams/e05-detectors, then render the PNG with sharp at density 144.
# Data: e05/runs/results-exp1.json. AUROC of each detector at telling a group's 234 items (positives) from the
# 234 never-seen items (negatives), scored on the original question + answer text by the contaminated
# SmolLM2-135M, with 95% percentile bootstrap intervals (2,000 resamples).
import sys
DETS=['Loss','zlib ratio','Min-K%','Min-K%++','Reference ratio']
PANELS=[  # title, [(auc, lo, hi) per detector]
 ('Verbatim, 1 copy',[(.565,.517,.617),(.583,.534,.635),(.575,.526,.626),(.564,.513,.617),(.669,.620,.717)]),
 ('Verbatim, 4 copies',[(.769,.727,.811),(.754,.711,.796),(.776,.732,.817),(.829,.791,.866),(.916,.889,.941)]),
 ('Verbatim, 16 copies',[(.971,.958,.983),(.968,.956,.980),(.971,.958,.982),(.984,.975,.991),(.994,.989,.999)]),
 ('Paraphrase, 16 copies',[(.737,.694,.781),(.705,.658,.752),(.717,.671,.764),(.729,.682,.776),(.877,.845,.906)]),
]
W,H=780,330; L,R,T,B=130,14,58,58
X0,X1=0.4,1.0
PW=(W-L-R)/len(PANELS); GAP=14
RH=(H-T-B)/len(DETS)
def sx(p,v): return L+p*PW+GAP/2+(v-X0)/(X1-X0)*(PW-GAP)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for p,(title,vals) in enumerate(PANELS):
        cx=(sx(p,X0)+sx(p,X1))/2
        o.append(f'<text x="{cx:.1f}" y="{T-24}" fill="{ink}" font-size="13" font-weight="bold" text-anchor="middle">{title}</text>')
        for v in (0.5,0.75,1.0):
            o.append(f'<line x1="{sx(p,v):.1f}" x2="{sx(p,v):.1f}" y1="{T-8}" y2="{H-B}" stroke="{mut if v==0.5 else grid}" stroke-width="{1.5 if v==0.5 else 1}"{" stroke-dasharray=\"5 4\"" if v==0.5 else ""}/>')
            o.append(f'<text x="{sx(p,v):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{v:g}</text>')
        for k,(a,lo,hi) in enumerate(vals):
            y=T+k*RH+RH/2
            col=acc2 if k==4 else acc
            o.append(f'<line x1="{sx(p,lo):.1f}" x2="{sx(p,hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2.5"/>')
            for e in (lo,hi): o.append(f'<line x1="{sx(p,e):.1f}" x2="{sx(p,e):.1f}" y1="{y-4:.1f}" y2="{y+4:.1f}" stroke="{col}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(p,a):.1f}" cy="{y:.1f}" r="5.5" fill="{col}" stroke="{surf}" stroke-width="1.5"/>')
    for k,d in enumerate(DETS):
        y=T+k*RH+RH/2
        o.append(f'<text x="{L-6}" y="{y+5:.1f}" fill="{ink}" font-size="14" text-anchor="end">{d}</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">AUROC against never-seen items (0.5 = coin flip, 1 = perfect)</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
