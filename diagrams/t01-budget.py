# Generates public/diagrams/t01-budget.svg and t01-budget-dark.svg (data: t01-build-llm-from-scratch.md).
# Usage: python3 diagrams/t01-budget.py public/diagrams/t01-budget, then render the PNG with sharp at density 144.
# Data: measured training throughput on the RTX 4060 Laptop GPU (BF16, torch.compile, 524,288 tokens per step),
# from the run logs full_t20/t50/t85/t125.out; hours = 20 tokens per parameter / measured tokens per second.
import sys
W,H=780,470; L,R,T,B=70,24,30,92
YMAX=50
runs=[('22M',22_025_088,103_618,65.7),('47M',47_195_712,49_459,66.4),('88M',88_096_512,29_508,72.7),('126M',125_854_464,19_096,68.4)]
def sy(h): return H-B-h/YMAX*(H-T-B)
def svg(th):
    ink,mut,grid,acc,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for h in range(0,YMAX+1,10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(h):.1f}" y2="{sy(h):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(h)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{h}</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Hours to 20 tokens per parameter</text>')
    # budget lines
    for h,lab in ((8,'one night: 8 h'),(48,'one weekend: 48 h')):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(h):.1f}" y2="{sy(h):.1f}" stroke="{ink}" stroke-width="2" stroke-dasharray="7 5"/>')
        o.append(f'<text x="{L+6}" y="{sy(h)-7:.1f}" fill="{ink}" font-size="13">{lab}</text>')
    n=len(runs); slot=(W-L-R)/n; bw=slot*0.5
    for i,(lab,N,tps,mfu) in enumerate(runs):
        hrs=20*N/tps/3600
        cx=L+slot*(i+0.5)
        o.append(f'<rect x="{cx-bw/2:.1f}" y="{sy(hrs):.1f}" width="{bw:.1f}" height="{sy(0)-sy(hrs):.1f}" fill="{acc}"/>')
        if hrs>4: o.append(f'<text x="{cx:.1f}" y="{sy(hrs)+20:.1f}" fill="{surf}" font-size="14" font-weight="bold" text-anchor="middle">{hrs:.1f} h</text>')
        else: o.append(f'<text x="{cx:.1f}" y="{sy(hrs)-8:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{hrs:.1f} h</text>')
        o.append(f'<text x="{cx:.1f}" y="{H-B+22}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="middle">{lab} parameters</text>')
        o.append(f'<text x="{cx:.1f}" y="{H-B+42}" fill="{mut}" font-size="13" text-anchor="middle">{20*N/1e9:.2f}B tokens</text>')
        o.append(f'<text x="{cx:.1f}" y="{H-B+60}" fill="{mut}" font-size="13" text-anchor="middle">{tps/1000:.1f}k tokens/s</text>')
        o.append(f'<text x="{cx:.1f}" y="{H-B+78}" fill="{mut}" font-size="13" text-anchor="middle">MFU {mfu:.0f}%</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
