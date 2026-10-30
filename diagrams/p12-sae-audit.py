# Generates public/diagrams/p12-sae-audit.svg and -dark.svg (data: p12-interpretability-open-problems.md).
# Usage: python3 diagrams/p12-sae-audit.py public/diagrams/p12-sae-audit, then render the PNG with sharp at density 144.
# Palette and approach copied from diagrams/g04-roofline.py. Data: p12/runs/sae_gpt2.json. GPT-2 small layer 8: increase in
# next-token loss when each SAE's reconstruction replaces the activation, on five kinds of text, 95% bootstrap CIs.
# ROWS: (label, line 2, ours (v, lo, hi), OpenAI TopK 32k, Bloom ReLU 24k)
ROWS=[('FineWeb-Edu','(training text)',(0.1135,0.1104,0.1167),(0.1340,0.1304,0.1378),(0.1989,0.1929,0.2057)),
('WikiText','',(0.2172,0.2038,0.2315),(0.1872,0.1819,0.1931),(0.2837,0.2765,0.2914)),
('German','Wikipedia',(0.2303,0.2208,0.2397),(0.2077,0.1995,0.2159),(0.3301,0.3194,0.3408)),
('Python','code',(0.2611,0.2482,0.2741),(0.2730,0.2632,0.2834),(0.4181,0.4029,0.4339)),
('Korean','Wikipedia',(0.2345,0.2235,0.2449),(0.1397,0.1322,0.1466),(0.3221,0.3135,0.3301))]
import sys
W,H=780,470; L,R,T,B=78,24,52,70
Y0,Y1=0,0.45
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
def cx(i): return L+(W-L-R)/len(ROWS)*(i+0.5)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0,0.1,0.2,0.3,0.4):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:g}</text>')
    cols=[(acc,'Ours, TopK 24,576 (Part 7)'),(acc2,'OpenAI, TopK 32,768'),(mut,'Bloom, ReLU 24,576')]
    lx=L
    for j,(c,lab) in enumerate(cols):
        o.append(f'<circle cx="{lx+6}" cy="18" r="5" fill="{c}"/><text x="{lx+16}" y="22" fill="{ink}" font-size="13">{lab}</text>')
        lx+=[214,170,0][j]
    for i,(a,b,*vals) in enumerate(ROWS):
        x=cx(i)
        for j,((v,lo,hi),(c,_)) in enumerate(zip(vals,cols)):
            xx=x+(j-1)*20
            o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{ink}" stroke-width="1.5"/>')
            o.append(f'<circle cx="{xx:.1f}" cy="{sy(v):.1f}" r="{6 if j==0 else 5}" fill="{c}"/>')
        o.append(f'<text x="{x:.1f}" y="{H-B+20}" fill="{ink}" font-size="13" text-anchor="middle">{a}</text>')
        if b: o.append(f'<text x="{x:.1f}" y="{H-B+36}" fill="{mut}" font-size="12" text-anchor="middle">{b}</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Loss increase when spliced in (nats)</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
