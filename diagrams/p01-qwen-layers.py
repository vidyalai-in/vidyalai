# Generates public/diagrams/p01-qwen-layers.svg and -dark.svg (data: p01-what-is-interpretability.md).
# Usage: python3 diagrams/p01-qwen-layers.py public/diagrams/p01-qwen-layers, then render the PNG with sharp at density 144.
# Data: Qwen2.5-0.5B, same 168 prompts, mean over prompts with 95% bootstrap CIs (p01/post/qwen.json).
# ROWS: (kind, layer, DLA, lo, hi, logit diff lost under mean ablation, lo, hi).
ROWS=[('attn', 0, -0.001, -0.003, 0.001, 3.639, 3.166, 4.139), ('attn', 1, 0.011, -0.004, 0.026, 3.117, 2.655, 3.58), ('attn', 2, -0.003, -0.017, 0.012, 3.748, 3.294, 4.228), ('attn', 3, 0.004, -0.005, 0.014, 0.182, 0.068, 0.307), ('attn', 4, 0.01, -0.004, 0.024, 0.591, 0.41, 0.774), ('attn', 5, 0.024, 0.012, 0.036, 0.308, 0.21, 0.408), ('attn', 6, -0.001, -0.01, 0.009, -0.019, -0.084, 0.049), ('attn', 7, -0.025, -0.043, -0.007, 0.017, -0.016, 0.051), ('attn', 8, -0.005, -0.035, 0.025, 0.376, 0.29, 0.464), ('attn', 9, -0.002, -0.015, 0.011, -0.02, -0.041, 0.002), ('attn', 10, -0.009, -0.025, 0.008, 0.154, 0.089, 0.219), ('attn', 11, -0.015, -0.025, -0.004, -0.041, -0.091, 0.006), ('attn', 12, 0.002, -0.016, 0.02, 0.1, 0.016, 0.186), ('attn', 13, 0.002, -0.01, 0.015, 0.05, 0.015, 0.085), ('attn', 14, 0.061, 0.038, 0.083, 0.141, 0.056, 0.229), ('attn', 15, -0.005, -0.027, 0.016, -0.023, -0.052, 0.007), ('attn', 16, 0.009, -0.001, 0.018, -0.049, -0.076, -0.023), ('attn', 17, 0.787, 0.725, 0.849, 1.743, 1.599, 1.888), ('attn', 18, 0.015, 0.005, 0.025, 0.031, 0.007, 0.055), ('attn', 19, 0.027, 0.01, 0.045, 0.049, 0.025, 0.074), ('attn', 20, 1.236, 1.11, 1.371, 0.726, 0.622, 0.838), ('attn', 21, 3.197, 2.892, 3.507, 1.797, 1.631, 1.969), ('attn', 22, 0.065, 0.047, 0.083, 0.018, 0.003, 0.032), ('attn', 23, 0.233, 0.168, 0.299, 0.099, 0.063, 0.137), ('mlp', 0, 0.0, -0.017, 0.017, 8.729, 8.158, 9.31), ('mlp', 1, 0.013, -0.019, 0.045, 3.959, 3.462, 4.461), ('mlp', 2, -0.009, -0.032, 0.014, 4.31, 3.785, 4.847), ('mlp', 3, 0.022, -0.012, 0.055, 3.668, 3.228, 4.127), ('mlp', 4, 0.011, -0.022, 0.042, 2.566, 2.212, 2.942), ('mlp', 5, -0.037, -0.072, -0.003, 1.27, 1.021, 1.526), ('mlp', 6, 0.022, -0.018, 0.063, 0.533, 0.38, 0.691), ('mlp', 7, 0.009, -0.03, 0.049, 0.326, 0.207, 0.448), ('mlp', 8, -0.035, -0.067, -0.002, 0.123, 0.034, 0.21), ('mlp', 9, 0.044, 0.015, 0.074, 0.142, 0.046, 0.237), ('mlp', 10, 0.009, -0.014, 0.033, -0.031, -0.106, 0.044), ('mlp', 11, -0.01, -0.043, 0.023, 0.161, 0.083, 0.238), ('mlp', 12, 0.02, -0.007, 0.047, 0.063, -0.005, 0.132), ('mlp', 13, 0.049, 0.02, 0.077, 0.007, -0.047, 0.059), ('mlp', 14, -0.005, -0.03, 0.019, 0.109, 0.04, 0.179), ('mlp', 15, 0.054, 0.02, 0.089, 0.073, -0.007, 0.154), ('mlp', 16, -0.033, -0.072, 0.007, 0.483, 0.385, 0.584), ('mlp', 17, 0.162, 0.112, 0.212, 0.314, 0.238, 0.389), ('mlp', 18, 0.239, 0.181, 0.296, 0.224, 0.154, 0.296), ('mlp', 19, 0.187, 0.138, 0.238, 0.264, 0.189, 0.341), ('mlp', 20, 1.159, 1.001, 1.322, 0.817, 0.663, 0.973), ('mlp', 21, 2.176, 1.916, 2.44, 1.097, 0.956, 1.235), ('mlp', 22, -0.136, -0.364, 0.096, 0.357, 0.192, 0.525), ('mlp', 23, -0.272, -0.542, -0.009, -0.401, -0.628, -0.176)]
YRANGE=(-1.0, 10.0)
TICKS=[0, 2, 4, 6, 8, 10]
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
