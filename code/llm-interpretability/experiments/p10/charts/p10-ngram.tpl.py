# Generates public/diagrams/p10-ngram.svg and -dark.svg (data: p10-training-dynamics-interpretability.md).
# Usage: python3 diagrams/p10-ngram.py public/diagrams/p10-ngram, then render the PNG with sharp at density 144.
# Data: the From Scratch 88M run's snapshots (steps 5 to 3,360), lab/dynamics.py NGram (p10 scripts/track.py).
# Unigram and bigram models: token counts over the first 100M training tokens. Left: KL(n-gram || model) in nats,
# averaged over 32,704 positions of 64 validation documents. Right: the model's loss on 64 sequences of 512 tokens
# sampled from each n-gram model; dashed lines are each sampler's own entropy (the best possible loss).
import sys, math
D=__DATA__
W=780; H=400; T=58; B=70
PL=(62,360); PR=(452,756)
X0,X1=math.log10(5),math.log10(3360)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    def panel(px,title,sub,y0,y1,ticks):
        x0,x1=px
        sx=lambda s: x0+(math.log10(s)-X0)/(X1-X0)*(x1-x0); sy=lambda v: H-B-(v-y0)/(y1-y0)*(H-T-B)
        o.append(f'<text x="{x0}" y="{T-30}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        o.append(f'<text x="{x0}" y="{T-12}" fill="{mut}" font-size="12">{sub}</text>')
        for v in ticks:
            o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
            o.append(f'<text x="{x0-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{v:g}</text>')
        for s in (10,100,1000):
            o.append(f'<line x1="{sx(s):.1f}" x2="{sx(s):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
            o.append(f'<text x="{sx(s):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{s:,}</text>')
        o.append(f'<text x="{(x0+x1)/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">Training step (log scale)</text>')
        return sx,sy
    def line(sx,sy,xs,ys,col,dash=False):
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.2" {"stroke-dasharray=\"6 4\"" if dash else ""} points="'+' '.join(f'{sx(s):.1f},{sy(v):.1f}' for s,v in zip(xs,ys))+'"/>')
    def mark_min(sx,sy,xs,ys,col,txt,dy,dx=0,anchor='middle'):
        i=min(range(len(ys)),key=lambda j: ys[j])
        o.append(f'<circle cx="{sx(xs[i]):.1f}" cy="{sy(ys[i]):.1f}" r="5" fill="{col}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{sx(xs[i])+dx:.1f}" y="{sy(ys[i])+dy:.1f}" fill="{col}" font-size="12" font-weight="bold" text-anchor="{anchor}">{txt} {ys[i]:.2f}, step {xs[i]}</text>')
    st=D['steps']
    sx,sy=panel(PL,'Distance from n-gram predictions','KL divergence on real text, nats',0,6,(0,1,2,3,4,5,6))
    line(sx,sy,st,D['kl_uni'],acc2); line(sx,sy,st,D['kl_bi'],acc)
    mark_min(sx,sy,st,D['kl_uni'],acc2,'unigram',22); mark_min(sx,sy,st,D['kl_bi'],acc,'bigram',22,dx=4,anchor='start')
    sx,sy=panel(PR,'Loss on text sampled from n-gram models','nats per token',4,11,(4,5,6,7,8,9,10,11))
    line(sx,sy,[5,3360],[D['H_uni']]*2,acc2,dash=True); line(sx,sy,[5,3360],[D['H_bi']]*2,acc,dash=True)
    line(sx,sy,st,D['uni'],acc2); line(sx,sy,st,D['bi'],acc)
    mark_min(sx,sy,st,D['uni'],acc2,'unigram text',12,dx=8,anchor='start'); mark_min(sx,sy,st,D['bi'],acc,'bigram text',22)
    o.append(f'<text x="{PR[1]}" y="{sy(D["H_uni"])+16:.1f}" fill="{mut}" font-size="11" text-anchor="end">unigram entropy {D["H_uni"]:.2f}</text>')
    o.append(f'<text x="{PR[1]}" y="{sy(D["H_bi"])+16:.1f}" fill="{mut}" font-size="11" text-anchor="end">bigram entropy {D["H_bi"]:.2f}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
