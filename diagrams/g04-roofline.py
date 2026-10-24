# Generates public/diagrams/g04-roofline.svg and g04-roofline-dark.svg (data: g04-gpu-profiling-roofline.md).
# Usage: python3 diagrams/g04-roofline.py public/diagrams/g04-roofline, then render the PNG with sharp at density 144.
import math
W,H=780,470; L,R,T,B=78,24,24,62
X0,X1=0.05,3000; Y0,Y1=0.01,40
def sx(x): return L+(math.log10(x)-math.log10(X0))/(math.log10(X1)-math.log10(X0))*(W-L-R)
def sy(y): return H-B-(math.log10(y)-math.log10(Y0))/(math.log10(Y1)-math.log10(Y0))*(H-T-B)
BW=0.2487; PK=25.3; F32=12.73
pts=[('A',0.083,0.0189),('B',1.0,0.2179),('B',1.0,0.2454),('B',1.0,0.1375),('C',1.25,0.1956),('C',1.25,0.2855),
('D',2.0,0.3988),('E',7.98,1.959),('F',62.7,12.9),('G',85,6.4),('H',341,22.0),('I',441,22.5),('J',683,6.1),('K',1024,23.1),('L',1365,24.9)]
off={'A':(8,4),'B':(-26,8),'C':(9,12),'D':(9,14),'E':(-16,0),'F':(-16,-6),'G':(9,4),'H':(-6,18),'I':(2,18),'J':(9,4),'K':(-10,18),'L':(6,18)}
def svg(th):
    ink,mut,grid,acc,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for e in range(-2,4):
        for m in (1,):
            x=10**e*m
            if X0<=x<=X1: o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:g}</text>') if X0<=x<=X1 else None
    for y in (0.01,0.1,1,10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:g}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Arithmetic intensity (FLOPs per byte, log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">TFLOPS (log scale)</text>')
    # roofs
    xr=PK/BW; xf=F32/BW
    o.append(f'<polyline fill="none" stroke="{ink}" stroke-width="2.5" points="{sx(X0):.1f},{sy(BW*X0):.1f} {sx(xr):.1f},{sy(PK):.1f} {sx(X1):.1f},{sy(PK):.1f}"/>')
    o.append(f'<line x1="{sx(xf):.1f}" y1="{sy(F32):.1f}" x2="{sx(X1):.1f}" y2="{sy(F32):.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{sx(xr)-8:.1f}" y="{sy(PK)-8:.1f}" fill="{ink}" font-size="13" text-anchor="end">BF16 tensor cores: 25.3 TFLOPS</text>')
    o.append(f'<text x="{W-R:.1f}" y="{sy(F32)+18:.1f}" fill="{mut}" font-size="13" text-anchor="end">FP32 CUDA cores: 12.7 TFLOPS</text>')
    o.append(f'<text transform="translate({sx(0.12):.1f},{sy(BW*0.12)-10:.1f}) rotate(-{math.degrees(math.atan((sy(BW*0.1)-sy(BW*1))/(sx(1)-sx(0.1)))):.1f})" fill="{ink}" font-size="13">DRAM: 248.7 GB/s</text>')
    o.append(f'<text x="{sx(xr)+6:.1f}" y="{sy(PK)-8:.1f}" fill="{mut}" font-size="12">ridge ≈ 102</text>')
    o.append(f'<line x1="{sx(xr):.1f}" y1="{sy(PK):.1f}" x2="{sx(xr):.1f}" y2="{sy(PK)+6:.1f}" stroke="{mut}" stroke-width="2"/>')
    done=set()
    for lab,x,y in pts:
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{acc}" stroke="{surf}" stroke-width="2"/>')
        if lab in done: continue
        done.add(lab); dx,dy=off[lab]
        o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{ink}" font-size="14" font-weight="bold">{lab}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d')
import sys
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
