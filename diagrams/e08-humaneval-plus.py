# Generates public/diagrams/e08-humaneval-plus.svg and e08-humaneval-plus-dark.svg (data: e08-code-math-benchmarks.md).
# Usage: python3 diagrams/e08-humaneval-plus.py public/diagrams/e08-humaneval-plus, then render the PNG with sharp at density 144.
# Data: HumanEval (164 problems) completions graded by the original tests and by HumanEval+ (evalplus/humanevalplus),
# run in the sandbox of evals/lab/verify.py. claude-haiku-5-5 thinking disabled, 10 samples per problem;
# claude-sonnet-5-5 "between_tools", 2 samples; Qwen2.5 Instruct models greedy, 1 sample. Bars: pass@1 (or pass@10)
# with 95% intervals (bootstrap over problems for Claude, Wilson for Qwen); right column: paired drop, bootstrap 95% CI.
import sys
ROWS=[  # model, setting, HumanEval (acc, lo, hi), HumanEval+ (acc, lo, hi), drop (diff, lo, hi)
 ('Haiku 5.5','pass@1, 10 samples',(96.4,93.6,98.7),(92.9,89.1,96.3),(3.5,1.2,6.3)),
 ('Haiku 5.5','pass@10',(98.8,97.0,100.0),(95.1,91.5,98.2),(3.7,1.2,6.7)),
 ('Sonnet 5.5','pass@1, 2 samples',(99.7,99.1,100.0),(94.8,91.2,97.9),(4.9,1.8,8.5)),
 ('Qwen2.5-1.5B','Instruct, greedy',(56.7,49.1,64.1),(48.2,40.7,55.8),(8.5,4.9,12.8)),
 ('Qwen2.5-0.5B','Instruct, greedy',(31.7,25.1,39.2),(27.4,21.2,34.7),(4.3,1.2,7.3)),
]
W,H=780,500; L,R,T,B=210,150,58,56
RH=(H-T-B)/len(ROWS)
def sx(v): return L+v/100*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(0,101,20):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}%</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Share of HumanEval problems passed, same code, two test suites</text>')
    o.append(f'<rect x="{L}" y="16" width="14" height="14" fill="{acc}"/><text x="{L+20}" y="28" fill="{ink}" font-size="13">HumanEval tests</text>')
    o.append(f'<rect x="{L+150}" y="16" width="14" height="14" fill="{acc2}"/><text x="{L+170}" y="28" fill="{ink}" font-size="13">HumanEval+ tests</text>')
    o.append(f'<text x="{W-R+75}" y="28" fill="{ink}" font-size="13" text-anchor="middle">Drop</text>')
    o.append(f'<text x="{W-R+75}" y="44" fill="{ink}" font-size="13" text-anchor="middle">(points)</text>')
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
        o.append(f'<text x="{W-R+75}" y="{y0+RH/2:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{g[0]:.1f}</text>')
        o.append(f'<text x="{W-R+75}" y="{y0+RH/2+16:.1f}" fill="{mut}" font-size="12" text-anchor="middle">[{g[1]:.1f}, {g[2]:.1f}]</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
