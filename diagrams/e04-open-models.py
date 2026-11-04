# Generates public/diagrams/e04-open-models.svg and e04-open-models-dark.svg (data: e04-eval-harness-sensitivity.md).
# Usage: python3 diagrams/e04-open-models.py public/diagrams/e04-open-models, then render the PNG with sharp at density 144.
# Data: four open models on 1,140 MMLU items, bf16 on an RTX 4060 Laptop GPU (evals/lab/mcq.py, e04/scripts/run_open.py).
# Per model and scoring method: the band spans every prompt variant tried (letter: 5 formats x 0/5 shots plus 1- and
# 5-shot draws, 15 variants; cloze: 0/5 shots x sum/per-character, 4; generate: 0/5 shots, 2); the dot is the 5-shot
# lm-eval-style prompt (cloze: per-character), with its 95% Wilson interval.
import sys
MODELS=[  # name, {method: (band_lo, band_hi, dot, ci_lo, ci_hi)}
 ('SmolLM2-135M',{'letter':(23.0,26.7,25.2,22.7,27.8),'cloze':(30.4,32.2,32.2,29.5,35.0),'generate':(21.5,25.5,25.5,23.1,28.1)}),
 ('Qwen2.5-0.5B',{'letter':(40.1,50.2,48.3,45.4,51.2),'cloze':(33.4,36.6,36.6,33.8,39.4),'generate':(49.2,49.7,49.2,46.3,52.1)}),
 ('Qwen2.5-0.5B-Instruct',{'letter':(36.0,49.3,48.3,45.4,51.2),'cloze':(31.8,37.1,37.1,34.3,39.9),'generate':(46.4,48.3,48.3,45.4,51.2)}),
 ('Llama-3.2-1B',{'letter':(29.0,42.4,32.4,29.7,35.1),'cloze':(33.9,39.8,39.8,37.0,42.7),'generate':(31.9,40.5,31.9,29.3,34.7)}),
]
METH=['letter','cloze','generate']
W,H=780,500; L,R,T,B=190,24,56,56
X0,X1=15,55
LH=26; GAP=22
def sx(v): return L+(v-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    col={'letter':acc,'cloze':acc2,'generate':ink}
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(X0,X1+1,5):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}%</text>')
    o.append(f'<line x1="{sx(25):.1f}" x2="{sx(25):.1f}" y1="{T}" y2="{H-B}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{sx(25)+5:.1f}" y="{T-6}" fill="{mut}" font-size="12">chance 25%</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Accuracy on 1,140 MMLU items</text>')
    # legend
    lx=20
    for m in METH:
        o.append(f'<rect x="{lx}" y="14" width="26" height="10" rx="3" fill="{col[m]}" opacity="0.25"/><circle cx="{lx+13}" cy="19" r="5" fill="{col[m]}"/>')
        o.append(f'<text x="{lx+32}" y="24" fill="{ink}" font-size="13">{m}</text>'); lx+=110
    o.append(f'<text x="{lx+10}" y="24" fill="{mut}" font-size="12">band: all prompts tried; dot: 5-shot, 95% CI</text>')
    y=T+14
    for name,d in MODELS:
        o.append(f'<text x="{L-12}" y="{y+LH+5:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="end">{name}</text>')
        for m in METH:
            lo,hi,dot,cl,ch=d[m]
            c=col[m]
            o.append(f'<rect x="{sx(lo)-3:.1f}" y="{y-6:.1f}" width="{sx(hi)-sx(lo)+6:.1f}" height="12" rx="6" fill="{c}" opacity="0.25"/>')
            o.append(f'<line x1="{sx(cl):.1f}" x2="{sx(ch):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2"/>')
            for e in (cl,ch): o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{y-5:.1f}" y2="{y+5:.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(dot):.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
            o.append(f'<text x="{sx(max(hi,ch))+9:.1f}" y="{y+4:.1f}" fill="{mut}" font-size="12">{dot:.1f}</text>')
            y+=LH
        y+=GAP
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
