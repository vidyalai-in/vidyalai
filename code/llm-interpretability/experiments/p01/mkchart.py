"""Writes diagrams/p01-<name>.py generators with the logged data embedded."""
import json, sys
ROOT = "/home/amit/.cache/vidyalai-scratch/p01"
REPO = "/home/amit/AlpsCode/ai/vidyalai"

def rows(tag, L):
    r = json.load(open(f"{ROOT}/post/{tag}.json"))
    out = []
    for kind in ("attn", "mlp"):
        for l in range(L):
            e = r["components"][f"{kind}{l}"]
            d, m = e["dla"], e["delta_mean"]
            out.append((kind, l, round(d[0], 3), round(d[1], 3), round(d[2], 3), round(-m[0], 3), round(-m[2], 3), round(-m[1], 3)))
    return out

BARS = r'''import sys
W,H=780,624; L,R=64,20
PANELS=[('attn','Attention layers',62,282),('mlp','MLP layers',356,576)]
Y0,Y1=YRANGE
NL=max(r[1] for r in ROWS)+1
def sx(l,k): 
    cw=(W-L-R)/NL; return L+cw*l+cw*(0.18+0.34*k)
def bw(): return (W-L-R)/NL*0.30
def sy(y,top,bot): return bot-(y-Y0)/(Y1-Y0)*(bot-top)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    # legend
    o.append(f'<rect x="{L}" y="8" width="14" height="14" fill="{acc}"/><text x="{L+20}" y="20" fill="{ink}" font-size="13">Direct logit attribution (what it writes)</text>')
    o.append(f'<rect x="{L+330}" y="8" width="14" height="14" fill="{acc2}"/><text x="{L+350}" y="20" fill="{ink}" font-size="13">Logit difference lost when mean-ablated</text>')
    for kind,title,top,bot in PANELS:
        for y in TICKS:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y,top,bot):.1f}" y2="{sy(y,top,bot):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y,top,bot)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{y:g}</text>')
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(0,top,bot):.1f}" y2="{sy(0,top,bot):.1f}" stroke="{mut}" stroke-width="1.5"/>')
        o.append(f'<text x="{L}" y="{top-10}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for kd,l,d,dlo,dhi,m,mlo,mhi in ROWS:
            if kd!=kind: continue
            for k,(v,lo,hi,c) in enumerate(((d,dlo,dhi,acc),(m,mlo,mhi,acc2))):
                x=sx(l,k); y0=sy(0,top,bot); y1=sy(v,top,bot)
                o.append(f'<rect x="{x:.1f}" y="{min(y0,y1):.1f}" width="{bw():.1f}" height="{abs(y1-y0):.1f}" fill="{c}"/>')
                xc=x+bw()/2
                o.append(f'<line x1="{xc:.1f}" x2="{xc:.1f}" y1="{sy(lo,top,bot):.1f}" y2="{sy(hi,top,bot):.1f}" stroke="{ink}" stroke-width="1.2"/>')
            if NL<=12 or l%2==0:
                o.append(f'<text x="{L+(W-L-R)/NL*(l+0.5):.1f}" y="{bot+16}" fill="{mut}" font-size="12" text-anchor="middle">{l}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-8}" fill="{ink}" font-size="14" text-anchor="middle">Layer</text>')
    o.append(f'<text transform="translate(18,{H/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Effect on logit difference (logits)</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
'''

def write_bars(name, tag, L, yr, ticks, header):
    src = header + f"ROWS={rows(tag, L)!r}\nYRANGE={yr!r}\nTICKS={ticks!r}\n" + BARS
    open(f"{REPO}/diagrams/p01-{name}.py", "w").write(src)

which = sys.argv[1]
if which == "gpt2":
    write_bars("gpt2-layers", "gpt2", 12, (-0.4, 1.4), [-0.4, 0, 0.4, 0.8, 1.2],
        "# Generates public/diagrams/p01-gpt2-layers.svg and -dark.svg (data: p01-what-is-interpretability.md).\n"
        "# Usage: python3 diagrams/p01-gpt2-layers.py public/diagrams/p01-gpt2-layers, then render the PNG with sharp at density 144.\n"
        "# Data: GPT-2 small, 168 landmark prompts, mean over prompts with 95% bootstrap CIs (p01/post/gpt2.json).\n"
        "# ROWS: (kind, layer, DLA, lo, hi, logit diff lost under mean ablation, lo, hi).\n")
elif which == "qwen":
    write_bars("qwen-layers", "qwen", 24, (-1.0, 10.0), [0, 2, 4, 6, 8, 10],
        "# Generates public/diagrams/p01-qwen-layers.svg and -dark.svg (data: p01-what-is-interpretability.md).\n"
        "# Usage: python3 diagrams/p01-qwen-layers.py public/diagrams/p01-qwen-layers, then render the PNG with sharp at density 144.\n"
        "# Data: Qwen2.5-0.5B, same 168 prompts, mean over prompts with 95% bootstrap CIs (p01/post/qwen.json).\n"
        "# ROWS: (kind, layer, DLA, lo, hi, logit diff lost under mean ablation, lo, hi).\n")

SCAT = r'''import sys
W,H=780,520; L,R,T,B=70,24,24,62
X0,X1,Y0,Y1=-0.1,1.1,-0.1,1.1
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
LAB={(9,8):(-10,-10,'end'),(10,0):(10,4,'start'),(8,11):(10,-6,'start'),(11,2):(10,6,'start'),(0,7):(10,4,'start'),(6,4):(0,-11,'middle')}
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (0,0.2,0.4,0.6,0.8,1.0):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v:g}</text>')
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v:g}</text>')
    o.append(f'<line x1="{sx(-0.1):.1f}" y1="{sy(-0.1):.1f}" x2="{sx(1.1):.1f}" y2="{sy(1.1):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{sx(0.78):.1f}" y="{sy(0.72):.1f}" fill="{mut}" font-size="13" text-anchor="start">ablation effect = direct effect</text>')
    for l,h,d,m in HEADS:
        o.append(f'<circle cx="{sx(d):.1f}" cy="{sy(m):.1f}" r="4.5" fill="{acc}" fill-opacity="0.75" stroke="{surf}" stroke-width="1"/>')
        if (l,h) in LAB:
            dx,dy,an=LAB[(l,h)]
            o.append(f'<text x="{sx(d)+dx:.1f}" y="{sy(m)+dy:.1f}" fill="{ink}" font-size="13" font-weight="bold" text-anchor="{an}">{l}.{h}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Direct logit attribution of the head (logits)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Logit difference lost when mean-ablated</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
'''
if which == "heads":
    import numpy as np
    r = json.load(open(f"{ROOT}/post/gpt2.json"))
    dh, mh = np.array(r["heads"]["dla_mean"]), np.array(r["heads"]["dmean_mean"])
    heads = [(l, h, round(float(dh[l, h]), 3), round(float(-mh[l, h]), 3)) for l in range(12) for h in range(12)]
    hdr = ("# Generates public/diagrams/p01-gpt2-heads.svg and -dark.svg (data: p01-what-is-interpretability.md).\n"
           "# Usage: python3 diagrams/p01-gpt2-heads.py public/diagrams/p01-gpt2-heads, then render the PNG with sharp at density 144.\n"
           "# Data: GPT-2 small, all 144 heads, mean over 168 landmark prompts (p01/post/gpt2.json).\n"
           "# HEADS: (layer, head, DLA, logit diff lost when the head is mean-ablated).\n")
    open(f"{REPO}/diagrams/p01-gpt2-heads.py", "w").write(hdr + f"HEADS={heads!r}\n" + SCAT)
