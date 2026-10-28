import sys
W,H=780,420; L,R,T,B=170,30,70,62
X0,X1=20,70
def sx(x): return L+(x-X0)/(X1-X0)*(W-L-R)
RH=(H-T-B)/len(ROWS)
def svg(th,cb,ca,cc):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in range(20,71,10):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}%</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Zero-shot accuracy, with 95% intervals (dashed tick: chance)</text>')
    leg=[(cb,'Base model (Part 7)'),(ca,'SFT, loss on assistant tokens'),(cc,'SFT, loss on every token')]
    for i,(c,t) in enumerate(leg):
        x=L+i*200
        o.append(f'<circle cx="{x+6}" cy="26" r="6" fill="{c}"/><text x="{x+18}" y="31" fill="{ink}" font-size="13">{t}</text>')
    for i,(name,b,a,c,ch) in enumerate(ROWS):
        y0=T+RH*i+RH/2
        o.append(f'<text x="{L-12}" y="{y0+4:.1f}" fill="{ink}" font-size="14" text-anchor="end">{name}</text>')
        o.append(f'<line x1="{sx(ch):.1f}" x2="{sx(ch):.1f}" y1="{y0-30:.1f}" y2="{y0+30:.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="4 4"/>')
        for k,((v,ci),col) in enumerate(((b,cb),(a,ca),(c,cc))):
            y=y0+(k-1)*18
            o.append(f'<line x1="{sx(v-ci):.1f}" x2="{sx(v+ci):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2.5" opacity="0.6"/>')
            o.append(f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="6" fill="{col}" stroke="{surf}" stroke-width="1.5"/>')
            o.append(f'<text x="{sx(v+ci)+8:.1f}" y="{y+4:.1f}" fill="{mut}" font-size="12">{v:.1f}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,'#4a5b78','#0b5fd6','#eb6834')); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,'#9fb0cf','#38bdf8','#d95926'))
