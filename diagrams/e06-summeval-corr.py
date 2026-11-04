# Generates public/diagrams/e06-summeval-corr.svg and e06-summeval-corr-dark.svg (data: e06-grading-generative-outputs.md).
# Usage: python3 diagrams/e06-summeval-corr.py public/diagrams/e06-summeval-corr, then render the PNG with sharp at density 144.
# Data: SummEval (100 CNN/DailyMail articles x 16 systems, 3 expert ratings per summary). Summary-level Kendall tau
# against the mean of two experts (averaged over the three choices of two), 95% bootstrap interval over articles.
# "One expert" is the third expert scored against the same target. Reference metrics use the original reference;
# the judge is Claude Haiku 5.5 (thinking off) with the anchored rubric, one run.
import sys
DIMS=[  # dimension, rows: (label, kind, tau, lo, hi)
 ('Coherence',[('ROUGE-1','metric',0.091,0.045,0.136),('BERTScore','metric',0.218,0.173,0.263),
               ('Haiku 5.5 rubric judge','judge',0.371,0.326,0.414),('One expert','human',0.590,0.560,0.620)]),
 ('Consistency',[('ROUGE-1','metric',0.137,0.088,0.184),('BERTScore','metric',0.081,0.033,0.129),
                 ('Haiku 5.5 rubric judge','judge',0.492,0.446,0.536),('One expert','human',0.792,0.755,0.826)]),
 ('Fluency',[('ROUGE-1','metric',0.090,0.041,0.137),('BERTScore','metric',0.127,0.078,0.174),
             ('Haiku 5.5 rubric judge','judge',0.273,0.231,0.313),('One expert','human',0.626,0.572,0.680)]),
 ('Relevance',[('ROUGE-1','metric',0.191,0.145,0.236),('BERTScore','metric',0.210,0.163,0.256),
               ('Haiku 5.5 rubric judge','judge',0.358,0.314,0.398),('One expert','human',0.413,0.371,0.454)]),
]
W=780; L,R,T,B=250,70,30,60; RH=24; GAP=30
H=T+B+sum(len(r)*RH for _,r in DIMS)+GAP*len(DIMS)
X0,X1=0.0,0.9
def sx(v): return L+(v-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    col={'metric':acc,'judge':acc2,'human':mut}
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for k in range(0,10):
        v=k/10
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid if k else mut}" stroke-width="{1 if k else 1.5}"/>')
        if k%2==0: o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v:.1f}</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Agreement with expert ratings (summary-level Kendall tau)</text>')
    y=T
    for dim,rows in DIMS:
        y+=GAP
        o.append(f'<text x="20" y="{y-9:.1f}" fill="{ink}" font-size="15" font-weight="bold">{dim}</text>')
        for lab,kind,t,lo,hi in rows:
            c=col[kind]; yc=y+RH/2
            o.append(f'<text x="{L-12}" y="{yc+5:.1f}" fill="{ink}" font-size="14" text-anchor="end">{lab}</text>')
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="{c}" stroke-width="2.5"/>')
            for e in (lo,hi): o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{yc-5:.1f}" y2="{yc+5:.1f}" stroke="{c}" stroke-width="2.5"/>')
            o.append(f'<circle cx="{sx(t):.1f}" cy="{yc:.1f}" r="6" fill="{c}" stroke="{surf}" stroke-width="2"/>')
            o.append(f'<text x="{sx(hi)+10:.1f}" y="{yc+5:.1f}" fill="{ink}" font-size="13">{t:.2f}</text>')
            y+=RH
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
