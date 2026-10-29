# Generates public/diagrams/t14-techniques.svg and t14-techniques-dark.svg (data: t14-training-memory.md).
# Usage: python3 diagrams/t14-techniques.py public/diagrams/t14-techniques, then render the PNG with sharp at density 144.
# Data: t14/scripts/bench.py runs cmp_* (t14/post/runs_summary.json): the 88M model compiled, BF16 autocast, 16 sequences of
# 1,024 tokens per micro-batch, 4 micro-batches per step, median of 6 timed steps, RTX 4060 Laptop GPU.
# x: torch.cuda.max_memory_allocated (GiB). y: tokens/s divided by the run's median SM clock, relative to the baseline (16,039 per GHz).
PTS=[('Baseline','base',6.75,100.0,8,-8),
     ('Selective checkpointing','ckpt',5.11,96.0,-10,20),
     ('Checkpoint every other block','ckpt',4.83,87.1,-10,18),
     ('Checkpoint every block','ckpt',2.97,82.1,10,5),
     ('8-bit AdamW','state',6.28,99.2,10,22),
     ('Pure BF16','state',5.52,106.3,10,-8),
     ('CPU offload of AdamW','state',6.10,88.4,10,5),
     ('CPU offload, BF16 on GPU','state',5.19,89.7,-10,-10),
     ('LoRA, FP32 base','lora',5.73,98.0,-10,-10),
     ('LoRA, BF16 base','lora',4.93,104.6,-10,-8)]
import sys
W,H=780,470; L,R,T,B=72,30,24,62
X0,X1=2.5,7.25; Y0,Y1=78,110
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
def svg(th,col):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in (3,4,5,6,7):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    for y in (80,85,90,95,100,105,110):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(100):.1f}" y2="{sy(100):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Peak memory allocated (GiB), 16 sequences per micro-batch</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Speed per GHz vs baseline</text>')
    for name,g,x,y,dx,dy in PTS:
        c=col[g]
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="6" fill="{c}" stroke="{surf}" stroke-width="2"/>')
        anchor='start' if dx>0 else 'end'
        o.append(f'<text x="{sx(x)+dx:.1f}" y="{sy(y)+dy:.1f}" fill="{ink}" font-size="13" text-anchor="{anchor}">{name}</text>')
    lx,ly=L+14,T+18
    for i,(g,t) in enumerate((('ckpt','Activation checkpointing'),('state','Smaller or offloaded state'),('lora','LoRA (fine-tuning only)'))):
        o.append(f'<circle cx="{lx+6}" cy="{ly+20*i-4}" r="6" fill="{col[g]}"/><text x="{lx+18}" y="{ly+20*i}" fill="{ink}" font-size="13">{t}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
lc={'base':'#0b1a33','ckpt':'#eb6834','state':'#0b5fd6','lora':'#9b59b6'}
dc={'base':'#e8eefc','ckpt':'#d95926','state':'#38bdf8','lora':'#b07cc6'}
open(sys.argv[1]+'.svg','w').write(svg(light,lc)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,dc))
