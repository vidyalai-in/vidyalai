# Generates public/diagrams/e08-math-graders.svg and e08-math-graders-dark.svg (data: e08-code-math-benchmarks.md).
# Usage: python3 diagrams/e08-math-graders.py public/diagrams/e08-math-graders, then render the PNG with sharp at density 144.
# Data: claude-haiku-5-5 on MATH-500 (500 problems, one reply each), thinking disabled vs adaptive thinking at effort
# high, the same extracted answer graded four ways (evals/lab/verify.py). Bars: accuracy with 95% Wilson intervals;
# right column: high minus off, paired over problems, 95% bootstrap interval.
import sys
ROWS=[  # grader, description, off (acc, lo, hi), high (acc, lo, hi), gap (diff, lo, hi)
 ('exact','same string as the key',(81.2,77.5,84.4),(73.4,69.4,77.1),(-7.8,-11.4,-4.2)),
 ('normalized','MATH repo string rules',(93.6,91.1,95.4),(86.0,82.7,88.8),(-7.6,-10.4,-4.8)),
 ('symbolic','math-verify (SymPy)',(97.4,95.6,98.5),(90.6,87.7,92.9),(-6.8,-9.6,-4.2)),
 ('read one by one','every disputed reply',(97.6,95.9,98.6),(98.6,97.1,99.3),(1.0,-0.4,2.6)),
]
W,H=780,430; L,R,T,B=210,150,58,56
RH=(H-T-B)/len(ROWS)
def sx(v): return L+v/100*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(0,101,20):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}%</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Accuracy on MATH-500, same replies, four graders</text>')
    o.append(f'<rect x="{L}" y="16" width="14" height="14" fill="{acc}"/><text x="{L+20}" y="28" fill="{ink}" font-size="13">Thinking off</text>')
    o.append(f'<rect x="{L+120}" y="16" width="14" height="14" fill="{acc2}"/><text x="{L+140}" y="28" fill="{ink}" font-size="13">Adaptive thinking, effort high</text>')
    o.append(f'<text x="{W-R+75}" y="28" fill="{ink}" font-size="13" text-anchor="middle">High minus</text>')
    o.append(f'<text x="{W-R+75}" y="44" fill="{ink}" font-size="13" text-anchor="middle">off (points)</text>')
    for k,(name,desc,a,b,g) in enumerate(ROWS):
        y0=T+k*RH
        o.append(f'<text x="{L-10}" y="{y0+RH/2-4:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="end">{name}</text>')
        o.append(f'<text x="{L-10}" y="{y0+RH/2+13:.1f}" fill="{mut}" font-size="12" text-anchor="end">{desc}</text>')
        for j,(v,col) in enumerate(((a,acc),(b,acc2))):
            yb=y0+10+j*((RH-20)/2); bh=(RH-20)/2-6
            o.append(f'<rect x="{L}" y="{yb:.1f}" width="{max(sx(v[0])-L,1.5):.1f}" height="{bh:.1f}" fill="{col}"/>')
            yc=yb+bh/2
            o.append(f'<line x1="{sx(v[1]):.1f}" x2="{sx(v[2]):.1f}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="{ink}" stroke-width="1.5"/>')
            for e in (v[1],v[2]): o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{yc-4:.1f}" y2="{yc+4:.1f}" stroke="{ink}" stroke-width="1.5"/>')
            o.append(f'<text x="{sx(v[2])+6:.1f}" y="{yc+4:.1f}" fill="{ink}" font-size="12">{v[0]:.1f}</text>')
        o.append(f'<text x="{W-R+75}" y="{y0+RH/2:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{g[0]:+.1f}</text>')
        o.append(f'<text x="{W-R+75}" y="{y0+RH/2+16:.1f}" fill="{mut}" font-size="12" text-anchor="middle">[{g[1]:+.1f}, {g[2]:+.1f}]</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
