# Generates public/diagrams/e06-pairwise-agreement.svg and e06-pairwise-agreement-dark.svg (data: e06-grading-generative-outputs.md).
# Usage: python3 diagrams/e06-pairwise-agreement.py public/diagrams/e06-pairwise-agreement, then render the PNG with sharp at density 144.
# Data: MT-Bench turn-1 expert votes (lmsys/mt_bench_human_judgments, 910 answer pairs, 1,689 votes). Agreement as in
# Zheng et al. 2023: "with ties" counts a tie as its own verdict; "without ties" drops any vote pair where either side
# said tie. 95% bootstrap intervals resample the 80 questions. Human-human uses the 484 pairs with 2+ different experts.
# GPT-4 = the 2023 GPT-4 judgments released with the dataset (both orders; inconsistent = tie). Haiku 5.5: thinking off,
# MT-Bench pair-v2 prompt; "both orders" counts a win only if the verdict survives swapping the answers.
import sys
ROWS=[  # label, (with ties: est, lo, hi), (without ties: est, lo, hi)
 ('Expert vs another expert',(64.1,59.5,69.0),(82.0,77.3,86.7)),
 ('GPT-4 judge (2023 release)',(66.6,62.9,70.4),(85.9,82.7,89.0)),
 ('Haiku 5.5, scores each answer',(65.8,62.3,69.4),(85.4,82.6,88.1)),
 ('Haiku 5.5, pairwise, one order',(67.6,64.7,70.5),(84.0,81.2,86.9)),
 ('Haiku 5.5, pairwise, both orders',(68.6,65.7,71.6),(86.0,83.2,88.7)),
]
W,H=780,330; L,R,T,B=270,30,58,58
X0,X1=50,95
RH=(H-T-B)/len(ROWS)
def sx(v): return L+(v-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(X0,X1+1,5):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}%</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Agreement with an expert\'s vote on the same pair</text>')
    # legend
    o.append(f'<circle cx="{L+8}" cy="22" r="6" fill="{acc}"/><text x="{L+20}" y="27" fill="{ink}" font-size="14">with ties (chance 33%)</text>')
    o.append(f'<circle cx="{L+230}" cy="22" r="6" fill="{acc2}"/><text x="{L+242}" y="27" fill="{ink}" font-size="14">without ties (chance 50%)</text>')
    for k,(lab,s1,s2) in enumerate(ROWS):
        y=T+k*RH+RH/2
        o.append(f'<text x="{L-12}" y="{y+5:.1f}" fill="{ink}" font-size="14" text-anchor="end">{lab}</text>')
        for (e,lo,hi),c in ((s1,acc),(s2,acc2)):
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2.5"/>')
            for z in (lo,hi): o.append(f'<line x1="{sx(z):.1f}" x2="{sx(z):.1f}" y1="{y-5:.1f}" y2="{y+5:.1f}" stroke="{c}" stroke-width="2.5"/>')
            o.append(f'<circle cx="{sx(e):.1f}" cy="{y:.1f}" r="6" fill="{c}" stroke="{surf}" stroke-width="2"/>')
            o.append(f'<text x="{sx(e):.1f}" y="{y-11:.1f}" fill="{ink}" font-size="12" text-anchor="middle">{e:.1f}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
