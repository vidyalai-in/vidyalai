# Generates public/diagrams/e04-haiku-variants.svg and e04-haiku-variants-dark.svg (data: e04-eval-harness-sensitivity.md).
# Usage: python3 diagrams/e04-haiku-variants.py public/diagrams/e04-haiku-variants, then render the PNG with sharp at density 144.
# Data: Haiku 5.5 (thinking off) on 1,140 MMLU items, one sample per item and prompt, flex extractor (evals/lab/mcq.py).
# Each row: accuracy minus the baseline prompt (lm-eval MMLU layout plus "Reply with only the label"), paired over the
# same items, 95% interval clustered by subject.
import sys
BASE=78.8
ROWS=[  # label, group, diff, lo, hi, accuracy
 ('Same prompt, sent again','noise',-0.7,-1.6,0.2,78.1),
 ('"Question:" and (A) labels','format',0.8,-0.7,2.3,79.6),
 ('"Choices:" and A: labels','format',0.4,-1.1,1.8,79.1),
 ('Labels 1-4','format',-1.2,-2.9,0.4,77.5),
 ('Lowercase, labels a-d','format',0.5,-1.1,2.2,79.3),
 ('Options rotated 1 place','order',-0.6,-2.3,1.1,78.2),
 ('Options rotated 2 places','order',0.2,-1.8,2.1,78.9),
 ('Options rotated 3 places','order',0.4,-1.1,2.0,79.2),
 ('Five solved examples first','shots',-0.3,-2.6,2.1,78.5),
 ('No instruction: free reply','talk',9.3,5.3,13.3,88.1),
 ('No instruction, five examples','talk',8.9,4.7,13.0,87.6),
]
W,H=780,520; L,R,T,B=240,90,40,56
X0,X1=-5,15
RH=(H-T-B)/len(ROWS)
def sx(v): return L+(v-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(X0,X1+1,5):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid if v else mut}" stroke-width="{1 if v else 1.5}"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v:+d}</text>' if v else f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">0</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Accuracy change vs the baseline prompt ({BASE}%), points</text>')
    o.append(f'<text x="{W-R+45}" y="{T-14}" fill="{ink}" font-size="13" text-anchor="middle">Accuracy</text>')
    for k,(lab,g,d,lo,hi,a) in enumerate(ROWS):
        y=T+k*RH+RH/2
        if k and ROWS[k-1][1]!=g:
            o.append(f'<line x1="20" x2="{W-20}" y1="{T+k*RH:.1f}" y2="{T+k*RH:.1f}" stroke="{grid}" stroke-width="1" stroke-dasharray="4 4"/>')
        col=acc2 if g=='talk' else acc
        o.append(f'<text x="{L-12}" y="{y+5:.1f}" fill="{ink}" font-size="14" text-anchor="end">{lab}</text>')
        o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2.5"/>')
        for e in (lo,hi): o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{y-5:.1f}" y2="{y+5:.1f}" stroke="{col}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{sx(d):.1f}" cy="{y:.1f}" r="6" fill="{col}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{W-R+45}" y="{y+5:.1f}" fill="{ink}" font-size="14" text-anchor="middle">{a:.1f}%</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
