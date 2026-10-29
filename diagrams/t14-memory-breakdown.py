# Generates public/diagrams/t14-memory-breakdown.svg and t14-memory-breakdown-dark.svg (data: t14-training-memory.md).
# Usage: python3 diagrams/t14-memory-breakdown.py public/diagrams/t14-memory-breakdown, then render the PNG with sharp at density 144.
# Data (GiB): the 88M model (Part 4's model.py), BF16 autocast, fused AdamW, 1,024-token sequences, RTX 4060 Laptop GPU.
# Predicted: fromscratch/memory.py (state_bytes, activation_bytes). Measured: torch.cuda.memory snapshots of one optimizer step
# (t14/scripts/bench.py + snap.py) at the step's peak; "backward temporaries" includes the few MiB the replay missed, and
# "allocator cache" is max_memory_reserved minus max_memory_allocated. Usable: 7,744 MiB allocated before WSL2 spilled to host memory.
CATS=['Weights (FP32)','AdamW state','Gradients (FP32)','BF16 weight copies','Activations, 10 blocks','Output layer and loss','Backward temporaries','Allocator cache']
ROWS=[('Eager, 4 sequences','predicted',[0.328,0.656,0.328,0.164,1.604,0.529,0,0]),
      ('Eager, 4 sequences','measured',[0.328,0.677,0.328,0.164,1.627,0.506,1.033,0.351]),
      ('Compiled, 16 sequences','predicted',[0.328,0.656,0.328,0.164,4.148,1.070,0,0]),
      ('Compiled, 16 sequences','measured',[0.328,0.672,0.328,0.164,4.126,1.070,0.062,0.157]),
      ('Eager, 16 sequences','predicted',[0.328,0.656,0.328,0.164,6.414,2.117,0,0])]
USABLE=7744/1024; CARD=8188/1024
import sys
W,H=780,440; L,R,T=196,40,30; BH=34; GAP=20
X1=10.5
def sx(v): return L+v/X1*(W-L-R)
def svg(th,cols):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    yb=T+10+len(ROWS)*(BH+GAP)
    for g in range(0,11,2):
        o.append(f'<line x1="{sx(g):.1f}" x2="{sx(g):.1f}" y1="{T}" y2="{yb}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(g):.1f}" y="{yb+18}" fill="{mut}" font-size="13" text-anchor="middle">{g}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{yb+40}" fill="{ink}" font-size="14" text-anchor="middle">GPU memory at the step\'s peak (GiB)</text>')
    for i,(name,kind,vals) in enumerate(ROWS):
        y=T+10+i*(BH+GAP)
        o.append(f'<text x="{L-10}" y="{y+14}" fill="{ink}" font-size="13" text-anchor="end">{name}</text>')
        o.append(f'<text x="{L-10}" y="{y+30}" fill="{mut}" font-size="12" text-anchor="end">{kind}</text>')
        x=0
        for v,c in zip(vals,cols):
            if v<=0: continue
            o.append(f'<rect x="{sx(x):.1f}" y="{y}" width="{sx(x+v)-sx(x):.1f}" height="{BH}" fill="{c}" stroke="{surf}" stroke-width="1"/>')
            x+=v
        o.append(f'<text x="{sx(x)+6:.1f}" y="{y+22}" fill="{ink}" font-size="13" font-weight="bold">{x:.2f}</text>')
    for v,t,dy in ((USABLE,'usable before spilling: 7.56',0),(CARD,'card: 8.0',14)):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T-6}" y2="{yb}" stroke="{ink}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{sx(USABLE)-4:.1f}" y="{T-10}" fill="{ink}" font-size="12" text-anchor="end">usable: 7.56</text>')
    o.append(f'<text x="{sx(CARD)+4:.1f}" y="{T-10}" fill="{ink}" font-size="12">card: 8.0</text>')
    ly=yb+62
    for j,(c,t) in enumerate(zip(cols,CATS)):
        lx=L-150+(j%3)*235; yy=ly+(j//3)*20
        o.append(f'<rect x="{lx}" y="{yy-11}" width="14" height="14" fill="{c}"/><text x="{lx+20}" y="{yy}" fill="{ink}" font-size="13">{t}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
lc=['#0b3d91','#0b5fd6','#5b8def','#a9c4f5','#eb6834','#f4a261','#9b59b6','#c8d0dc']
dc=['#1d4ed8','#38bdf8','#7dd3fc','#bae6fd','#d95926','#f4a261','#b07cc6','#4a5b78']
open(sys.argv[1]+'.svg','w').write(svg(light,lc)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,dc))
