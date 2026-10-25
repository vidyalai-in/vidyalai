# Generates public/diagrams/t06-speed-ladder.svg and t06-speed-ladder-dark.svg (data: t06-fast-llm-training.md).
# Usage: python3 diagrams/t06-speed-ladder.py public/diagrams/t06-speed-ladder, then render the PNG with sharp at density 144.
# Data: median tokens/s of three full optimizer steps (524,288 tokens each) of the 88M model on the RTX 4060 Laptop GPU,
# from the benchmark logs L1_fp32, L2_tf32, L3_bf16, L4_compile_mb4, L5_compile_mb16 (Part 6 scratch folder).
# Hours = 1,761,607,680 tokens (Part 7's run) / tokens per second. MFU against the measured 25.3 TFLOPS BF16 peak.
import sys
W,H=780,430; L,R,T,B=250,96,24,58
XMAX=42000
runs=[('FP32, eager','micro-batch 4',5783),('+ TF32 matmuls','micro-batch 4',8545),('+ BF16 autocast','micro-batch 4',19808),
      ('+ torch.compile','micro-batch 4',29021),('+ micro-batch 16','compiled; later changes within noise',30078)]
FPT=622.95e6; PEAK=25.3e12; TOK=1_761_607_680
def sx(v): return L+v/XMAX*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(0,XMAX+1,10000):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v//1000}k</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Training tokens per second (88M model, full optimizer steps)</text>')
    ceil=PEAK/FPT
    o.append(f'<line x1="{sx(ceil):.1f}" x2="{sx(ceil):.1f}" y1="{T}" y2="{H-B}" stroke="{ink}" stroke-width="2" stroke-dasharray="7 5"/>')
    o.append(f'<text x="{sx(ceil)-6:.1f}" y="{T+14}" fill="{ink}" font-size="13" text-anchor="end">100% of 25.3 TFLOPS: 40.6k</text>')
    n=len(runs); slot=(H-T-B-20)/n; bh=slot*0.56
    for i,(lab,sub,tps) in enumerate(runs):
        cy=T+20+slot*(i+0.5)
        o.append(f'<rect x="{L}" y="{cy-bh/2:.1f}" width="{sx(tps)-L:.1f}" height="{bh:.1f}" fill="{acc}"/>')
        o.append(f'<text x="{L-10}" y="{cy-3:.1f}" fill="{ink}" font-size="14" font-weight="bold" text-anchor="end">{lab}</text>')
        o.append(f'<text x="{L-10}" y="{cy+14:.1f}" fill="{mut}" font-size="12" text-anchor="end">{sub}</text>')
        mfu=tps*FPT/PEAK*100; hrs=TOK/tps/3600
        o.append(f'<text x="{sx(tps)+8:.1f}" y="{cy-3:.1f}" fill="{ink}" font-size="13" font-weight="bold">{tps:,}</text>')
        o.append(f'<text x="{sx(tps)+8:.1f}" y="{cy+14:.1f}" fill="{mut}" font-size="12">MFU {mfu:.0f}%, {hrs:.1f} h</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
