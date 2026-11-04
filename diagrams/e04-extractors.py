# Generates public/diagrams/e04-extractors.svg and e04-extractors-dark.svg (data: e04-eval-harness-sensitivity.md).
# Usage: python3 diagrams/e04-extractors.py public/diagrams/e04-extractors, then render the PNG with sharp at density 144.
# Data: the same free-form replies (lm-eval MMLU prompt, no instruction, thinking off) of Haiku 5.5 and Sonnet 5.5
# on 285 MMLU items (5 per subject), scored by five answer extractors (evals/lab/mcq.py). Bars: accuracy with 95% Wilson
# intervals; right column: Sonnet minus Haiku, paired over items, 95% interval clustered by subject.
import sys
ROWS=[  # extractor, description, haiku (acc, lo, hi), sonnet (acc, lo, hi), gap (diff, lo, hi)
 ('strict','whole reply is one letter',(0.0,0.0,1.3),(0.0,0.0,1.3),(0.0,0.0,0.0)),
 ('first','reply starts with a letter',(54.4,48.6,60.1),(1.4,0.5,3.6),(-53.0,-59.9,-46.1)),
 ('answer is','"answer is X" or "Answer: X"',(33.0,27.8,38.6),(62.5,56.7,67.9),(29.5,16.6,42.3)),
 ('any','first lone A-D anywhere',(84.6,79.9,88.3),(81.1,76.1,85.2),(-3.5,-9.3,2.3)),
 ('flex','chain of rules, then option text',(86.7,82.2,90.1),(93.3,89.8,95.7),(6.7,2.5,10.9)),
]
W,H=780,470; L,R,T,B=210,150,58,56
RH=(H-T-B)/len(ROWS)
def sx(v): return L+v/100*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(0,101,20):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}%</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Accuracy on 285 MMLU items, same replies</text>')
    # legend
    o.append(f'<rect x="{L}" y="16" width="14" height="14" fill="{acc}"/><text x="{L+20}" y="28" fill="{ink}" font-size="13">Haiku 5.5</text>')
    o.append(f'<rect x="{L+110}" y="16" width="14" height="14" fill="{acc2}"/><text x="{L+130}" y="28" fill="{ink}" font-size="13">Sonnet 5.5</text>')
    o.append(f'<text x="{W-R+75}" y="28" fill="{ink}" font-size="13" text-anchor="middle">Sonnet minus</text>')
    o.append(f'<text x="{W-R+75}" y="44" fill="{ink}" font-size="13" text-anchor="middle">Haiku (points)</text>')
    for k,(name,desc,h,s,g) in enumerate(ROWS):
        y0=T+k*RH
        o.append(f'<text x="{L-10}" y="{y0+RH/2-4:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="end">{name}</text>')
        o.append(f'<text x="{L-10}" y="{y0+RH/2+13:.1f}" fill="{mut}" font-size="12" text-anchor="end">{desc}</text>')
        for j,(v,col) in enumerate(((h,acc),(s,acc2))):
            yb=y0+10+j*((RH-20)/2); bh=(RH-20)/2-6
            o.append(f'<rect x="{L}" y="{yb:.1f}" width="{max(sx(v[0])-L,1.5):.1f}" height="{bh:.1f}" fill="{col}"/>')
            yc=yb+bh/2
            o.append(f'<line x1="{sx(v[1]):.1f}" x2="{sx(v[2]):.1f}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="{ink}" stroke-width="1.5"/>')
            for e in (v[1],v[2]): o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{yc-4:.1f}" y2="{yc+4:.1f}" stroke="{ink}" stroke-width="1.5"/>')
            o.append(f'<text x="{sx(v[2])+6:.1f}" y="{yc+4:.1f}" fill="{ink}" font-size="12">{v[0]:.1f}</text>')
        gs=f'{g[0]:+.1f}' if g[0] else '0.0'
        o.append(f'<text x="{W-R+75}" y="{y0+RH/2:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{gs}</text>')
        o.append(f'<text x="{W-R+75}" y="{y0+RH/2+16:.1f}" fill="{mut}" font-size="12" text-anchor="middle">[{g[1]:+.1f}, {g[2]:+.1f}]</text>' if g[0] else '')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
