# Generates public/diagrams/t09-bpb.svg and t09-bpb-dark.svg (data: t09-evaluate-base-model.md).
# Usage: python3 diagrams/t09-bpb.py public/diagrams/t09-bpb, then render the PNG with sharp at density 144.
# Data: fromscratch/evals.py bits_per_byte, each document scored on its own after the model's end-of-text token,
# 1,024-token sliding window with stride 512. FineWeb-Edu: the first 2,014 documents of Part 3's validation shard
# (9,143,418 bytes). WikiText-103 test, raw, 62 articles (1,290,527 bytes). Ours in BF16 autocast, the others in FP32.
import sys
ROWS=[('SmolLM2-135M','2T tokens',0.8526,0.9181,0),('Pythia-160M','300B tokens',0.9915,1.0397,0),
      ('GPT-2 small, 124M','WebText, tokens not published',0.9960,1.0251,0),('Pythia-70M','300B tokens',1.1170,1.2181,0),
      ('Ours, 88M','1.76B tokens',1.0447,1.2306,1),('Ours, 47M','0.94B tokens',1.0919,1.3031,1),
      ('Ours, 33M','0.66B tokens',1.1397,1.3585,1),('Ours, 22M','0.44B tokens',1.1821,1.4302,1),
      ('Ours, 15M','0.30B tokens',1.2382,1.5176,1),('Ours, 11M','0.22B tokens',1.2842,1.5650,1)]
W,H=780,600; L,R,T,B=230,30,64,62
X0,X1=0.8,1.6
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
RH=(H-T-B)/len(ROWS)
def sy(i): return T+RH*(i+0.5)
def svg(th,acc,acc2):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for k in range(9):
        x=X0+0.1*k
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:.1f}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Bits per byte (lower is better)</text>')
    # legend
    o.append(f'<circle cx="{L+6}" cy="24" r="6" fill="{ink}"/><text x="{L+18}" y="29" fill="{ink}" font-size="13">FineWeb-Edu validation slice (in-domain for ours)</text>')
    o.append(f'<circle cx="{L+6}" cy="46" r="5.5" fill="{surf}" stroke="{ink}" stroke-width="2"/><text x="{L+18}" y="51" fill="{ink}" font-size="13">WikiText-103 test, raw</text>')
    for i,(name,tok,fw,wt,ours) in enumerate(ROWS):
        c=acc2 if ours else acc; y=sy(i)
        o.append(f'<text x="{L-12}" y="{y-2:.1f}" fill="{ink}" font-size="13" text-anchor="end" font-weight="{"bold" if name=="Ours, 88M" else "normal"}">{name}</text>')
        o.append(f'<text x="{L-12}" y="{y+13:.1f}" fill="{mut}" font-size="11" text-anchor="end">{tok}</text>')
        o.append(f'<line x1="{sx(fw):.1f}" x2="{sx(wt):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="3" opacity="0.5"/>')
        o.append(f'<circle cx="{sx(fw):.1f}" cy="{y:.1f}" r="6.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<circle cx="{sx(wt):.1f}" cy="{y:.1f}" r="5.5" fill="{surf}" stroke="{c}" stroke-width="2.5"/>')
        o.append(f'<text x="{sx(fw)-10:.1f}" y="{y+4:.1f}" fill="{mut}" font-size="11" text-anchor="end">{fw:.3f}</text>')
        o.append(f'<text x="{sx(wt)+10:.1f}" y="{y+4:.1f}" fill="{mut}" font-size="11">{wt:.3f}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#38bdf8','#d95926'))
