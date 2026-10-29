import sys
W=780; L,R=70,24; PH=118; GAP=62; TOP=52
H=TOP+len(PANELS)*(PH+GAP)-GAP+62
X1=((max(len(s[1]) for p in PANELS for s in p[1])+24)//25)*25
def sx(x): return L+x/X1*(W-L-R)
def smooth(v,k=9):
    return [sum(v[max(0,i-k//2):i+k//2+1])/len(v[max(0,i-k//2):i+k//2+1]) for i in range(len(v))]
def svg(th,cols):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for j,(title,series,y0,y1,ticks,fmt) in enumerate(PANELS):
        top=TOP+j*(PH+GAP); bot=top+PH
        sy=lambda y: bot-(y-y0)/(y1-y0)*PH
        o.append(f'<text x="{L}" y="{top-12}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for y in ticks:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{fmt.format(y)}</text>')
        for x in range(0,X1+1,25):
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{top}" y2="{bot}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{sx(x):.1f}" y="{bot+17}" fill="{mut}" font-size="12" text-anchor="middle">{x}</text>')
        lx=W-R
        for k,(lab,vals) in reversed(list(enumerate(series))):
            c=cols[k]
            pts=[(sx(i),sy(v)) for i,v in enumerate(smooth(vals))]
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}"/>')
            if lab:
                o.append(f'<text x="{lx}" y="{top-12}" fill="{c}" font-size="13" font-weight="bold" text-anchor="end">{lab}</text>')
                lx-=16+7.2*len(lab)
    o.append(f'<text x="{(L+W-R)/2}" y="{H-8}" fill="{ink}" font-size="14" text-anchor="middle">GRPO step, main run (9-step average)</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,['#0b5fd6','#eb6834']))
open(sys.argv[1]+'-dark.svg','w').write(svg(dark,['#38bdf8','#d95926']))
