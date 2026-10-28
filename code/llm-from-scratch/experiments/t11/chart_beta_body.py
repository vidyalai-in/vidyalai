import sys
W,H=780,330; L=150; T=78; RH=52; GAP=40
PW=(W-L-20-2*GAP)/3                    # width of each of the three panels
def px(i): return L+i*(PW+GAP)
def mkx(i,x0,x1): return lambda x: px(i)+(x-x0)/(x1-x0)*PW
def svg(th,acc,acc2):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    titles=('KL from the SFT model','Change in log-prob','Reply length, temp. 1')
    subs=('nats per reply, 95% CI','held-out pairs, nats','mean tokens (cap 512)')
    axes=((0,10,(0,2,4,6,8,10)),(-6,0,(-6,-4,-2,0)),(200,400,(200,300,400)))
    ys=[T+RH*k+RH/2 for k in range(len(ROWS))]
    yb=T+RH*len(ROWS)
    for i,((x0,x1,ticks),t,s) in enumerate(zip(axes,titles,subs)):
        sx=mkx(i,x0,x1)
        o.append(f'<text x="{px(i)}" y="24" fill="{ink}" font-size="14" font-weight="bold">{t}</text>')
        o.append(f'<text x="{px(i)}" y="42" fill="{mut}" font-size="12">{s}</text>')
        for v in ticks:
            o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{yb}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{sx(v):.1f}" y="{yb+18}" fill="{mut}" font-size="12" text-anchor="middle">{v:g}</text>')
    for k,(name,sub,kl,ci,dc,dr,ln) in enumerate(ROWS):
        y=ys[k]
        o.append(f'<text x="{L-12}" y="{y-2:.1f}" fill="{ink}" font-size="13" text-anchor="end">{name}</text>')
        o.append(f'<text x="{L-12}" y="{y+14:.1f}" fill="{mut}" font-size="11" text-anchor="end">{sub}</text>')
        sx=mkx(0,0,10)
        o.append(f'<rect x="{sx(0):.1f}" y="{y-9:.1f}" width="{sx(kl)-sx(0):.1f}" height="18" fill="{acc}" rx="2"/>')
        o.append(f'<line x1="{sx(kl-ci):.1f}" x2="{sx(kl+ci):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{ink}" stroke-width="1.5"/>')
        o.append(f'<text x="{sx(kl+ci)+6:.1f}" y="{y+4:.1f}" fill="{ink}" font-size="12">{kl:.1f}</text>')
        sx=mkx(1,-6,0)
        o.append(f'<line x1="{sx(dr):.1f}" x2="{sx(dc):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{mut}" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(dc):.1f}" cy="{y:.1f}" r="6" fill="{acc}" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<circle cx="{sx(dr):.1f}" cy="{y:.1f}" r="6" fill="{acc2}" stroke="{surf}" stroke-width="1.5"/>')
        sx=mkx(2,200,400)
        o.append(f'<line x1="{sx(SFT_LEN):.1f}" x2="{sx(ln):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{mut}" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(ln):.1f}" cy="{y:.1f}" r="6" fill="{acc}" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<text x="{sx(ln)+10:.1f}" y="{y+4:.1f}" fill="{ink}" font-size="12">{ln:.0f}</text>')
    sx=mkx(2,200,400)
    o.append(f'<line x1="{sx(SFT_LEN):.1f}" x2="{sx(SFT_LEN):.1f}" y1="{T-6}" y2="{yb}" stroke="{ink}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{sx(SFT_LEN)+4:.1f}" y="{T-8}" fill="{ink}" font-size="11">SFT: {SFT_LEN:.0f}</text>')
    lx=px(1)
    o.append(f'<circle cx="{lx+6}" cy="{H-14}" r="6" fill="{acc}"/><text x="{lx+16}" y="{H-10}" fill="{ink}" font-size="12">chosen</text>')
    o.append(f'<circle cx="{lx+80}" cy="{H-14}" r="6" fill="{acc2}"/><text x="{lx+90}" y="{H-10}" fill="{ink}" font-size="12">rejected</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#38bdf8','#d95926'))
