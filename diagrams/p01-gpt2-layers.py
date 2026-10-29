# Generates public/diagrams/p01-gpt2-layers.svg and -dark.svg (data: p01-what-is-interpretability.md).
# Usage: python3 diagrams/p01-gpt2-layers.py public/diagrams/p01-gpt2-layers, then render the PNG with sharp at density 144.
# Data: GPT-2 small, 168 landmark prompts, mean over prompts with 95% bootstrap CIs (p01/post/gpt2.json).
# ROWS: (kind, layer, DLA, lo, hi, logit diff lost under mean ablation, lo, hi).
ROWS=[('attn', 0, 0.022, -0.003, 0.048, 0.265, 0.167, 0.367), ('attn', 1, 0.012, 0.003, 0.02, 0.017, -0.016, 0.049), ('attn', 2, 0.019, 0.004, 0.035, 0.025, -0.016, 0.066), ('attn', 3, 0.013, 0.0, 0.026, 0.007, -0.024, 0.038), ('attn', 4, 0.008, -0.004, 0.019, 0.019, -0.009, 0.047), ('attn', 5, 0.005, -0.006, 0.015, 0.005, -0.021, 0.031), ('attn', 6, 0.068, 0.053, 0.083, 0.139, 0.102, 0.177), ('attn', 7, 0.017, 0.008, 0.026, 0.013, -0.006, 0.032), ('attn', 8, 0.462, 0.396, 0.533, 0.421, 0.361, 0.481), ('attn', 9, 1.019, 0.881, 1.162, 0.997, 0.881, 1.115), ('attn', 10, 0.716, 0.59, 0.855, 0.458, 0.393, 0.527), ('attn', 11, 0.525, 0.462, 0.589, 0.384, 0.339, 0.43), ('mlp', 0, 0.012, -0.002, 0.026, 1.165, 0.924, 1.397), ('mlp', 1, -0.005, -0.012, 0.001, -0.052, -0.1, -0.005), ('mlp', 2, 0.004, -0.007, 0.014, 0.115, 0.053, 0.178), ('mlp', 3, 0.004, -0.014, 0.021, 0.143, 0.084, 0.205), ('mlp', 4, 0.001, -0.02, 0.023, 0.137, 0.079, 0.197), ('mlp', 5, -0.009, -0.034, 0.015, 0.062, 0.005, 0.118), ('mlp', 6, 0.002, -0.019, 0.023, 0.053, 0.001, 0.106), ('mlp', 7, 0.014, -0.006, 0.035, 0.112, 0.064, 0.161), ('mlp', 8, 0.05, 0.014, 0.086, 0.184, 0.129, 0.242), ('mlp', 9, 0.409, 0.335, 0.486, 0.434, 0.361, 0.507), ('mlp', 10, 0.346, 0.277, 0.417, 0.268, 0.223, 0.315), ('mlp', 11, -0.27, -0.352, -0.192, -0.263, -0.342, -0.188)]
YRANGE=(-0.4, 1.4)
TICKS=[-0.4, 0, 0.4, 0.8, 1.2]
import sys
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
