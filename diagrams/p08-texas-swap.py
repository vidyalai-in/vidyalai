# Generates public/diagrams/p08-texas-swap.svg and -dark.svg (data: p08-circuit-tracing.md).
# Usage: python3 diagrams/p08-texas-swap.py public/diagrams/p08-texas-swap, then render the PNG with sharp at density 144.
# Data: Gemma 2 2B, prompt "Fact: the capital of the state containing Dallas is", interventions on Gemma Scope
# transcoder features in the real model (attention free), p08/post/validate_summary.json.
# ROWS: (label, P(" Austin"), P(" Sacramento")).
ROWS=[("Unedited",0.4132,0.0052),("Texas features set to 0",0.1974,0.0236),("Texas features x -2",0.0000,0.0295),
      ("California features added",0.2775,0.0901),("Texas x -2, California added",0.0000,0.2676)]
import sys
W,H=780,430; L,R,T,B=250,30,50,56
def sx(p): return L+p/0.5*(W-L-R)
RH=(H-T-B)/len(ROWS)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (0,0.1,0.2,0.3,0.4,0.5):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{int(v*100)}%</text>')
    o.append(f'<rect x="{L}" y="14" width="12" height="12" fill="{acc}"/><text x="{L+18}" y="25" fill="{ink}" font-size="13">P(" Austin")</text>')
    o.append(f'<rect x="{L+130}" y="14" width="12" height="12" fill="{acc2}"/><text x="{L+148}" y="25" fill="{ink}" font-size="13">P(" Sacramento")</text>')
    for i,(lab,a,s) in enumerate(ROWS):
        y=T+i*RH
        o.append(f'<text x="{L-10}" y="{y+RH/2+5:.1f}" fill="{ink}" font-size="13" text-anchor="end">{lab}</text>')
        bh=RH*0.32
        for j,(p,c) in enumerate(((a,acc),(s,acc2))):
            yy=y+RH*0.16+j*bh
            o.append(f'<rect x="{L}" y="{yy:.1f}" width="{max(sx(p)-L,1):.1f}" height="{bh-2:.1f}" fill="{c}"/>')
            o.append(f'<text x="{sx(p)+6:.1f}" y="{yy+bh-6:.1f}" fill="{mut}" font-size="12">{p*100:.1f}%</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Probability of the next token</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
