# Generates public/diagrams/p10-ngram.svg and -dark.svg (data: p10-training-dynamics-interpretability.md).
# Usage: python3 diagrams/p10-ngram.py public/diagrams/p10-ngram, then render the PNG with sharp at density 144.
# Data: the From Scratch 88M run's snapshots (steps 5 to 3,360), lab/dynamics.py NGram (p10 scripts/track.py).
# Unigram and bigram models: token counts over the first 100M training tokens. Left: KL(n-gram || model) in nats,
# averaged over 32,704 positions of 64 validation documents. Right: the model's loss on 64 sequences of 512 tokens
# sampled from each n-gram model; dashed lines are each sampler's own entropy (the best possible loss).
import sys, math
D={"steps":[5,10,15,20,25,30,40,50,60,75,90,100,125,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3360],"kl_uni":[2.2877,1.7067,1.5194,1.3355,1.1323,0.8958,0.5395,0.5351,0.7745,1.0818,1.3986,1.5476,1.7761,1.9227,2.3794,2.6378,2.8888,3.2475,3.5925,3.7672,3.9567,4.0802,4.2151,4.3139,4.3849,4.4206,4.4863,4.5823,4.5677,4.6425,4.6452,4.6907,4.6849,4.81,4.7896,4.7823,4.8586,4.8579,4.8756,4.9233,4.9354,4.9276,4.9522,4.9633,4.9577,4.9759,5.0099,4.9813],"kl_bi":[4.576,3.9889,3.7692,3.4123,3.0228,2.6666,2.0788,1.7273,1.5068,1.2985,1.2124,1.1932,1.1748,1.1555,1.2948,1.4424,1.5732,1.778,1.9905,2.1413,2.2496,2.3443,2.4308,2.47,2.5282,2.5736,2.6121,2.6764,2.7033,2.7267,2.7402,2.7844,2.7798,2.8692,2.8561,2.8423,2.9121,2.8976,2.9149,2.938,2.9576,2.9587,2.9727,2.9747,2.9778,2.9861,3.0042,2.9853],"uni":[9.8292,9.3057,9.1204,8.9421,8.7572,8.5165,8.1539,8.1545,8.3758,8.6722,8.9811,9.0926,9.2705,9.361,9.7158,9.8204,9.9539,9.9146,9.9215,9.7602,9.8312,9.6049,9.4134,9.4057,9.3498,9.172,9.1119,9.1247,8.9513,8.9454,8.8694,8.8107,8.8057,8.8188,8.7262,8.7371,8.7239,8.6915,8.6669,8.6747,8.6876,8.6717,8.6778,8.6405,8.6482,8.6667,8.648,8.6577],"bi":[9.8152,9.2712,9.0506,8.6688,8.2737,7.918,7.3363,6.9764,6.755,6.5474,6.4582,6.4327,6.3951,6.3574,6.4484,6.5407,6.6211,6.7072,6.7602,6.7744,6.8043,6.7963,6.7805,6.7724,6.7758,6.739,6.714,6.7458,6.6925,6.6926,6.6682,6.6637,6.6499,6.6904,6.6383,6.6191,6.6484,6.615,6.6089,6.607,6.6151,6.6072,6.599,6.5832,6.5824,6.594,6.5939,6.5863],"H_uni":7.6021809577941895,"H_bi":5.241074562072754}
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
