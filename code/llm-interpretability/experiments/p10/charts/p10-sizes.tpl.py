# Generates public/diagrams/p10-sizes.svg and -dark.svg (data: p10-training-dynamics-interpretability.md).
# Usage: python3 diagrams/p10-sizes.py public/diagrams/p10-sizes, then render the PNG with sharp at density 144.
# Data: highest induction score among all heads at each saved snapshot, 200 repeated random sequences
# (lab/dynamics.py for the From Scratch models: Part 8's sweep and Part 7's 88M run; lab/heads.py with
# TransformerLens for Pythia-70m's public checkpoints). Same score on both panels; only the x-axis differs.
import sys, math
D=__DATA__
W=780; H=440; T=58; B=126
PL=(62,360); PR=(452,756)
def svg(th):
    ink,mut,grid,acc,surf,acc2,ramp=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    def panel(px,title,key,lo,hi,ticks,fmt):
        x0,x1=px
        sx=lambda v: x0+(math.log10(v)-math.log10(lo))/(math.log10(hi)-math.log10(lo))*(x1-x0); sy=lambda v: H-B-v*(H-T-B)
        o.append(f'<text x="{x0}" y="{T-30}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        o.append(f'<text x="{x0}" y="{T-12}" fill="{mut}" font-size="12">strongest induction head</text>')
        for v in (0,0.25,0.5,0.75,1):
            o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
            o.append(f'<text x="{x0-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{v:g}</text>')
        for v in ticks:
            o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
            o.append(f'<text x="{sx(v):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{fmt(v)}</text>')
        for i,r in enumerate(D['runs']):
            col=acc2 if r['name'].startswith('Pythia') else ramp[i]
            pts=[(x,y) for x,y in zip(r[key],r['ind']) if lo<=x<=hi]
            o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.2" {"stroke-dasharray=\"6 4\"" if r["name"].startswith("Pythia") else ""} points="'+' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in pts)+'"/>')
            for x,y in pts: o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{col}"/>')
        return sx
    panel(PL,'Against tokens seen','tokens',1e7,3e11,(1e7,1e8,1e9,1e10,1e11),lambda v: {1e7:'10M',1e8:'100M',1e9:'1B',1e10:'10B',1e11:'100B'}[v])
    panel(PR,'Against optimizer steps','steps',100,150000,(100,1000,10000,100000),lambda v: f'{int(v):,}')
    o.append(f'<text x="{(PL[0]+PL[1])/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">Tokens seen (log scale)</text>')
    o.append(f'<text x="{(PR[0]+PR[1])/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">Optimizer steps (log scale)</text>')
    # legend
    ly=H-B+72
    for i,r in enumerate(D['runs']):
        col=acc2 if r['name'].startswith('Pythia') else ramp[i]
        x=62+(i%4)*180; y=ly+(i//4)*24
        o.append(f'<line x1="{x}" x2="{x+22}" y1="{y-4}" y2="{y-4}" stroke="{col}" stroke-width="3" {"stroke-dasharray=\"6 4\"" if r["name"].startswith("Pythia") else ""}/>')
        o.append(f'<text x="{x+28}" y="{y}" fill="{ink}" font-size="12">{r["name"]}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834',['#86b4f2','#5b97ea','#3479e0','#0b5fd6','#0848a6','#062f6e'])
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926',['#1e5a8a','#2a77b3','#3b94d6','#5cb2ee','#8fcdf7','#c4e6fc'])
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
