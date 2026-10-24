# Generates public/diagrams/t04-overfit.svg and t04-overfit-dark.svg (data: t04-write-gpt-model.md).
# Usage: python3 diagrams/t04-overfit.py public/diagrams/t04-overfit, then render the PNG with sharp at density 144.
# Data: per-step training loss of the 88M model (fromscratch/model.py) on one fixed batch of 4 x 1,024 held-out
# FineWeb-Edu tokens, AdamW lr 1e-3, BF16 autocast, seed 1337, RTX 4060 Laptop GPU; logged by t04 overfit.py.
import math, sys
W,H=780,440; L,R,T,B=70,24,24,62
LOSS=[10.5864,10.7445,9.4513,8.3743,7.6704,7.1150,6.7066,6.4639,6.3415,6.6711,6.3514,6.3198,6.2828,6.2028,6.1467,6.1090,6.1214,6.1382,6.0099,5.9597,5.8996,5.8427,5.8025,5.7313,5.6889,5.6468,5.6151,5.5580,5.4983,5.4668,5.4228,5.3556,5.4036,5.3270,5.2481,5.1949,5.1024,5.0285,4.9781,4.9003,4.9522,4.9861,4.8201,4.7715,4.7226,4.6377,4.5821,4.5050,4.4313,4.4003,4.2932,4.2255,4.1515,4.0986,3.9900,3.9239,3.8861,3.7747,3.7441,3.5872,3.5476,3.4002,3.3358,3.2807,3.1274,3.0792,2.9790,2.8688,2.7362,2.7345,2.6164,2.4964,2.5684,2.3794,2.4280,2.3210,2.1897,2.0987,1.9949,1.9567,1.8135,1.7121,1.6605,1.4793,1.4446,1.3452,1.2894,1.1419,1.0416,1.0462,0.9063,0.8783,0.7655,0.7149,0.6593,0.5758,0.5401,0.4905,0.4362,0.3763,0.3366,0.2942,0.3082,0.2302,0.2114,0.1638,0.1343,0.1168,0.0938,0.0807,0.0651,0.0571,0.0468,0.0390,0.0327,0.0268,0.0226,0.0189,0.0159,0.0134,0.0113,0.0097,0.0083,0.0072,0.0063,0.0055,0.0048]
X1=130; Y0,Y1=0.003,20
def sx(s): return L+s/X1*(W-L-R)
def sy(y): return H-B-(math.log10(y)-math.log10(Y0))/(math.log10(Y1)-math.log10(Y0))*(H-T-B)
def svg(th):
    ink,mut,grid,acc,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y,lab in ((0.01,'0.01'),(0.1,'0.1'),(1,'1'),(10,'10')):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{lab}</text>')
    for s in range(0,X1+1,20):
        o.append(f'<line x1="{sx(s):.1f}" x2="{sx(s):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(s):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{s}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Optimizer step (same batch every step)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Training loss (log scale)</text>')
    u=math.log(32768)
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(u):.1f}" y2="{sy(u):.1f}" stroke="{ink}" stroke-width="2" stroke-dasharray="7 5"/>')
    o.append(f'<text x="{W-R-6}" y="{sy(u)-8:.1f}" fill="{ink}" font-size="13" text-anchor="end">uniform guess: ln 32,768 = 10.40</text>')
    pts=' '.join(f'{sx(i):.1f},{sy(v):.1f}' for i,v in enumerate(LOSS))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{acc}" stroke-width="2.5"/>')
    n=len(LOSS)-1
    o.append(f'<circle cx="{sx(n):.1f}" cy="{sy(LOSS[-1]):.1f}" r="4.5" fill="{acc}"/>')
    o.append(f'<text x="{sx(n)-10:.1f}" y="{sy(LOSS[-1])+4:.1f}" fill="{ink}" font-size="13" text-anchor="end">step {n}: {LOSS[-1]:.4f}</text>')
    o.append(f'<text x="{sx(0)+8:.1f}" y="{sy(LOSS[0])-12:.1f}" fill="{ink}" font-size="13">step 0: {LOSS[0]:.2f}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
