# Generates public/diagrams/t13-seq-distill.svg and t13-seq-distill-dark.svg (data: t13-distillation-synthetic-data.md).
# Usage: python3 diagrams/t13-seq-distill.py public/diagrams/t13-seq-distill, then render the PNG with sharp at density 144.
# Data: t13 scripts/eval_sft.py on four fine-tunes of Part 7's 88M model. Part 10 SFT: 26,000 smol-smoltalk conversations.
# The other three: the same 2,827 prompts with smol-smoltalk's original first reply, Qwen2.5-1.5B-Instruct's reply after
# filters, or the first 2,817 Qwen replies unfiltered; Part 10's lr 3e-4, 3 epochs. Replies: 200 held-out prompts, at most
# 256 tokens, greedy or sampled (T 0.8, top-k 50). ARC-Easy: Part 9's evals.py, zero-shot, 2,376 questions (+-2.0).
# (label, Part 10 SFT, original replies, Qwen filtered, Qwen unfiltered); None = not measured. Percent.
ROWS=[('Replies that end, greedy', 33.0, 37.5, 25.5, 27.0),
      ('Replies that end, sampled', 39.0, 60.5, 37.5, 32.5),
      ('Markdown in the reply, greedy', 8.5, 9.5, 48.0, 52.0),
      ('"As an AI" in the reply, greedy', 0.0, 0.0, 12.5, 13.5),
      ('ARC-Easy accuracy', 44.1, 42.1, 43.8, None)]
import sys
W,H=780,470; L,R,T,B=250,30,84,56
X0,X1=0,70
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
RH=(H-T-B)/len(ROWS)
def svg(th,cg,cb,co):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in range(0,71,10):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T-6}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}%</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Share of 200 held-out replies (benchmark: share of questions right)</text>')
    leg=[(cg,'filled','Part 10 SFT (26,000 conversations)'),(cb,'filled','Original replies, 2,827 prompts'),
         (co,'filled','Qwen replies, filtered'),(co,'hollow','Qwen replies, unfiltered')]
    for i,(c,kind,t) in enumerate(leg):
        x=24+(i%2)*380; y=24+(i//2)*24
        fill=c if kind=='filled' else surf
        o.append(f'<circle cx="{x+6}" cy="{y}" r="6" fill="{fill}" stroke="{c}" stroke-width="2"/><text x="{x+18}" y="{y+5}" fill="{ink}" font-size="13">{t}</text>')
    for i,(name,*vals) in enumerate(ROWS):
        y0=T+RH*i+RH/2
        o.append(f'<text x="{L-14}" y="{y0+4:.1f}" fill="{ink}" font-size="14" text-anchor="end">{name}</text>')
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{y0+RH/2:.1f}" y2="{y0+RH/2:.1f}" stroke="{grid}" stroke-width="1"/>') if i<len(ROWS)-1 else None
        for k,(v,(c,kind)) in enumerate(zip(vals,((cg,'filled'),(cb,'filled'),(co,'filled'),(co,'hollow')))):
            if v is None: continue
            y=y0+(k-1.5)*11
            fill=c if kind=='filled' else surf
            o.append(f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="5.5" fill="{fill}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<text x="{sx(v)+10:.1f}" y="{y+4:.1f}" fill="{mut}" font-size="11">{v:g}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#4a5b78','#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#9fb0cf','#38bdf8','#d95926'))
