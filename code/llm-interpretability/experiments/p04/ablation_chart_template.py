# Generates public/diagrams/p04-ablation.svg and -dark.svg (data: p04-induction-heads.md).
# Usage: python3 diagrams/p04-ablation.py public/diagrams/p04-ablation, then render the PNG with sharp at density 144.
# Data: GPT-2 small, mean ablation (p04 scripts/s2_ablate.py, s3_lite.py). Left: mean loss on tokens 2-50 of the second copy,
# 500 repeated random sequences; random = 50 draws of k heads outside the top 12. Right: WikiText-103, 976 windows of 512 tokens,
# mean loss per position in log-spaced bins; random 12 = average of 5 draws.
import sys, math
D=__DATA__
W,H=780,400; T=58; B=64
PL=(66,352); PR=(452,756)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    # ---- left panel
    x0,x1=PL; y0,y1=0,14
    sx=lambda k: x0+k/12*(x1-x0); sy=lambda v: H-B-(v-y0)/(y1-y0)*(H-T-B)
    o.append(f'<text x="{x0}" y="{T-30}" fill="{ink}" font-size="14" font-weight="bold">Repeated random tokens</text>')
    o.append(f'<text x="{x0}" y="{T-12}" fill="{mut}" font-size="12">loss on the second copy, nats</text>')
    for v in (0,2,4,6,8,10,12,14):
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{x0-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{v}</text>')
    for k in (0,1,2,3,4,5,8,12):
        o.append(f'<text x="{sx(k):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{k}</text>')
    o.append(f'<text x="{(x0+x1)/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">Heads ablated</text>')
    o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(13.16):.1f}" y2="{sy(13.16):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{x0+6}" y="{sy(13.16)-6:.1f}" fill="{mut}" font-size="12">first copy, nothing to copy: 13.2</text>')
    rk=[0]+sorted(int(k) for k in D['rand'])
    lo=[D['top'][0]]+[D['rand'][str(k)][0] for k in rk[1:]]; hi=[D['top'][0]]+[D['rand'][str(k)][2] for k in rk[1:]]
    med=[D['top'][0]]+[D['rand'][str(k)][1] for k in rk[1:]]
    band=' '.join(f'{sx(k):.1f},{sy(v):.1f}' for k,v in zip(rk,hi))+' '+' '.join(f'{sx(k):.1f},{sy(v):.1f}' for k,v in zip(rk[::-1],lo[::-1]))
    o.append(f'<polygon points="{band}" fill="{acc2}" fill-opacity="0.22" stroke="none"/>')
    o.append(f'<polyline fill="none" stroke="{acc2}" stroke-width="2.5" points="{" ".join(f"{sx(k):.1f},{sy(v):.1f}" for k,v in zip(rk,med))}"/>')
    pts=[(k,v) for k,v in zip(D['ks'],D['top']) if v is not None]
    o.append(f'<polyline fill="none" stroke="{acc}" stroke-width="2.5" points="{" ".join(f"{sx(k):.1f},{sy(v):.1f}" for k,v in pts)}"/>')
    for k,v in pts: o.append(f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="4" fill="{acc}" stroke="{surf}" stroke-width="1.5"/>')
    o.append(f'<text x="{sx(5)+8:.1f}" y="{sy(D["top"][5])+16:.1f}" fill="{acc}" font-size="13" font-weight="bold">top-k induction heads</text>')
    o.append(f'<text x="{sx(5.3):.1f}" y="{sy(2.5):.1f}" fill="{acc2}" font-size="13" font-weight="bold">random k heads</text>')
    o.append(f'<text x="{sx(5.3):.1f}" y="{sy(2.5)+15:.1f}" fill="{mut}" font-size="11">median, 5th-95th percentile</text>')
    # ---- right panel
    x0,x1=PR; y0,y1=3.2,5.0; lx0,lx1=math.log10(10),math.log10(512)
    sx=lambda p: x0+(math.log10(p)-lx0)/(lx1-lx0)*(x1-x0); sy=lambda v: H-B-(v-y0)/(y1-y0)*(H-T-B)
    o.append(f'<text x="{x0}" y="{T-30}" fill="{ink}" font-size="14" font-weight="bold">WikiText-103</text>')
    o.append(f'<text x="{x0}" y="{T-12}" fill="{mut}" font-size="12">loss per token, nats</text>')
    for v in (3.2,3.6,4.0,4.4,4.8):
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{x0-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{v:.1f}</text>')
    for p in (10,50,100,500):
        o.append(f'<line x1="{sx(p):.1f}" x2="{sx(p):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
        o.append(f'<text x="{sx(p):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{p}</text>')
    o.append(f'<text x="{(x0+x1)/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">Position in the window (log scale)</text>')
    for key,c,dash in (('rand12',acc2,''),('top12',acc,''),('clean',ink,'6 4')):
        pts=[(p,v) for p,v in D[key] if p>=10]
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" {"stroke-dasharray=%s" % chr(34)+dash+chr(34) if dash else ""} points="{" ".join(f"{sx(p):.1f},{sy(v):.1f}" for p,v in pts)}"/>')
    last={k:D[k][-1][1] for k in ('clean','top12','rand12')}
    o.append(f'<text x="{sx(115):.1f}" y="{sy(4.02):.1f}" fill="{acc}" font-size="13" font-weight="bold">top 12 induction</text>')
    o.append(f'<text x="{sx(115):.1f}" y="{sy(4.02)+16:.1f}" fill="{acc}" font-size="13" font-weight="bold">heads ablated</text>')
    o.append(f'<text x="{sx(60):.1f}" y="{sy(3.36):.1f}" fill="{ink}" font-size="13" font-weight="bold">intact</text>')
    o.append(f'<text x="{sx(60):.1f}" y="{sy(3.36)+16:.1f}" fill="{acc2}" font-size="13" font-weight="bold">random 12 ablated</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
