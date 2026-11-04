# Generates public/diagrams/e05-score-gain.svg and e05-score-gain-dark.svg (data: e05-benchmark-contamination.md).
# Usage: python3 diagrams/e05-score-gain.py public/diagrams/e05-score-gain, then render the PNG with sharp at density 144.
# Data: SmolLM2-135M continued on 7.7M tokens (FineWeb-Edu plus ARC-Challenge test items), scored by
# length-normalized option log-likelihood on 5 groups of 234 items (e05/runs/results-exp1.json).
# Each row: accuracy before and after training with 95% Wilson intervals; right column: gain over the
# never-seen group's drift (difference in differences) with a 95% bootstrap interval.
import sys
ROWS=[  # label, before, lo, hi, after, lo, hi, did, did_lo, did_hi
 ('Never in training',29.1,23.6,35.2,35.5,29.6,41.8,0.0,-7.3,7.3),
 ('Verbatim, 1 copy',30.8,25.2,37.0,42.3,36.2,48.7,5.1,-2.6,13.2),
 ('Verbatim, 4 copies',27.4,22.0,33.4,62.4,56.0,68.4,28.6,20.5,36.8),
 ('Verbatim, 16 copies',32.9,27.2,39.2,97.4,94.5,98.8,58.1,50.0,66.2),
 ('Paraphrase, 16 copies',27.4,22.0,33.4,82.9,77.6,87.2,49.1,41.0,57.3),
]
W,H=780,400; L,R,T,B=200,150,56,60
X0,X1=0,100
RH=(H-T-B)/len(ROWS)
def sx(v): return L+(v-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(X0,X1+1,25):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}%</text>')
    o.append(f'<line x1="{sx(25):.1f}" x2="{sx(25):.1f}" y1="{T}" y2="{H-B}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{sx(25)+5:.1f}" y="{T-8}" fill="{mut}" font-size="12">chance</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">ARC-Challenge accuracy on the group\'s 234 items</text>')
    o.append(f'<text x="{W-R+75}" y="{T-26}" fill="{ink}" font-size="13" text-anchor="middle">Gain vs never-</text>')
    o.append(f'<text x="{W-R+75}" y="{T-10}" fill="{ink}" font-size="13" text-anchor="middle">seen, points</text>')
    # legend
    o.append(f'<circle cx="{30}" cy="{T-26}" r="6" fill="{surf}" stroke="{mut}" stroke-width="2.5"/><text x="{42}" y="{T-21}" fill="{ink}" font-size="13">before</text>')
    o.append(f'<circle cx="{110}" cy="{T-26}" r="6" fill="{acc}" stroke="{surf}" stroke-width="2"/><text x="{122}" y="{T-21}" fill="{ink}" font-size="13">after training</text>')
    for k,(lab,b,bl,bh,a,al,ah,d,dl,dh) in enumerate(ROWS):
        y=T+k*RH+RH/2
        col=acc2 if lab.startswith('Para') else acc
        o.append(f'<text x="{L-12}" y="{y+5:.1f}" fill="{ink}" font-size="14" text-anchor="end">{lab}</text>')
        o.append(f'<line x1="{sx(b):.1f}" x2="{sx(a):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{grid}" stroke-width="5"/>')
        for lo,hi,yy,c in ((bl,bh,y-9,mut),(al,ah,y+9,col)):
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{c}" stroke-width="2"/>')
            for e in (lo,hi): o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{yy-4:.1f}" y2="{yy+4:.1f}" stroke="{c}" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(b):.1f}" cy="{y:.1f}" r="6" fill="{surf}" stroke="{mut}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{sx(a):.1f}" cy="{y:.1f}" r="7" fill="{col}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{W-R+75}" y="{y:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{d:+.1f}</text>')
        o.append(f'<text x="{W-R+75}" y="{y+17:.1f}" fill="{mut}" font-size="12" text-anchor="middle">[{dl:+.1f}, {dh:+.1f}]</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
