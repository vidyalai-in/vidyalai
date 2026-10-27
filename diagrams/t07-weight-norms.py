# Generates public/diagrams/t07-weight-norms.svg and t07-weight-norms-dark.svg (data: t07-pretraining-run.md).
# Usage: python3 diagrams/t07-weight-norms.py public/diagrams/t07-weight-norms, then render the PNG with sharp at density 144.
# Data: the 88M main run's weight snapshots (runs/main88/model_*.pt). NORMS: (step, Frobenius norm of the tied embedding,
# of all other weight matrices together, of the 10 MLP down-projections together, mean gain of the final RMSNorm).
# The "decay alone" line multiplies (1 - lr x 0.1) over the run's schedule: what AdamW's weight decay would do with no gradient.
import math, sys
NORMS=[(0,100.174,130.19,17.74,1.0),(5,100.164,130.189,17.74,1.0),(10,100.152,130.188,17.744,1.0),(15,100.148,130.194,17.76,1.0001),(20,100.162,130.211,17.794,1.0004),(25,100.209,130.238,17.843,1.0009),(30,100.309,130.266,17.9,1.0014),(40,100.73,130.338,18.042,1.0029),(50,101.396,130.434,18.213,1.0047),(60,102.134,130.543,18.415,1.0065),(75,103.357,130.731,18.755,1.0096),(90,104.638,130.936,19.13,1.013),(100,105.535,131.089,19.415,1.0155),(125,108.218,131.57,20.328,1.0236),(150,111.692,132.346,21.674,1.0336),(200,119.842,134.104,24.623,1.0584),(250,127.364,135.765,27.309,1.0839),(300,133.856,137.541,29.97,1.1092),(400,144.402,140.94,34.673,1.1577),(500,152.766,143.934,38.635,1.2055),(600,159.467,146.809,42.183,1.2489),(700,164.683,149.583,45.407,1.2875),(800,168.765,152.165,48.275,1.321),(900,172.021,154.659,50.907,1.3496),(1000,174.707,156.951,53.252,1.3749),(1100,176.876,159.102,55.375,1.3953),(1200,178.69,161.039,57.255,1.4144),(1300,180.23,162.778,58.922,1.4324),(1400,181.531,164.326,60.383,1.449),(1500,182.618,165.701,61.669,1.4645),(1600,183.509,166.9,62.78,1.4788),(1700,184.25,167.899,63.72,1.4926),(1800,184.844,168.736,64.51,1.5049),(1900,185.314,169.407,65.159,1.5167),(2000,185.689,169.942,65.689,1.5275),(2100,186.0,170.353,66.113,1.5377),(2200,186.235,170.663,66.449,1.5473),(2300,186.411,170.887,66.708,1.556),(2400,186.545,171.036,66.903,1.5644),(2500,186.638,171.129,67.048,1.5717),(2600,186.7,171.165,67.143,1.5785),(2700,186.738,171.175,67.21,1.5846),(2800,186.764,171.161,67.254,1.5902),(2900,186.777,171.132,67.28,1.5953),(3000,186.781,171.094,67.293,1.5997),(3100,186.781,171.05,67.298,1.6037),(3200,186.777,171.003,67.299,1.6073),(3300,186.77,170.958,67.297,1.6107),(3360,186.767,170.931,67.295,1.6127)]
STEPS,WARM,PEAK,FLOOR=3360,150,6e-4,6e-5
def lr_at(s):
    if s<WARM: return PEAK*(s+1)/WARM
    return FLOOR+0.5*(PEAK-FLOOR)*(1+math.cos(math.pi*(s-WARM)/(STEPS-WARM)))
DECAY=[1.0]
for s in range(STEPS): DECAY.append(DECAY[-1]*(1-lr_at(s)*0.1))
W,H=780,460; L,R,T,B=70,24,24,62
Y0,Y1=0.8,2.0
def sx(s): return L+s/STEPS*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
def svg(th,cols):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0.8,1.0,1.2,1.4,1.6,1.8,2.0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:.1f}×</text>')
    for s in range(0,3361,500):
        o.append(f'<line x1="{sx(s):.1f}" x2="{sx(s):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(s):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{s:,}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Optimizer step</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Norm relative to step 0</text>')
    series=[(1,'tied embedding (100.2 → 186.8)',cols[0],''),(2,'all other weight matrices (130.2 → 170.9)',cols[1],''),(4,'final RMSNorm gain, mean (1.00 → 1.61)',cols[2],'')]
    for k,lab,c,_ in series:
        base=NORMS[0][k]
        o.append(f'<polyline points="{" ".join(f"{sx(n[0]):.1f},{sy(n[k]/base):.1f}" for n in NORMS)}" fill="none" stroke="{c}" stroke-width="2.4"/>')
        for n in NORMS: o.append(f'<circle cx="{sx(n[0]):.1f}" cy="{sy(n[k]/base):.1f}" r="2.6" fill="{c}"/>')
    o.append(f'<polyline points="{" ".join(f"{sx(s):.1f},{sy(DECAY[s]):.1f}" for s in range(0,STEPS+1,20))}" fill="none" stroke="{mut}" stroke-width="2" stroke-dasharray="7 5"/>')
    lx,ly=L+380,H-B-118
    for i,(k,lab,c,_) in enumerate(series):
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{ly+22*i}" y2="{ly+22*i}" stroke="{c}" stroke-width="2.6"/><text x="{lx+32}" y="{ly+22*i+4}" fill="{ink}" font-size="13">{lab}</text>')
    o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{ly+66}" y2="{ly+66}" stroke="{mut}" stroke-width="2" stroke-dasharray="7 5"/><text x="{lx+32}" y="{ly+70}" fill="{ink}" font-size="13">weight decay alone (→ 0.90)</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#eb6834','#2a78d6','#1baf7a'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#d95926','#3987e5','#199e70')))
