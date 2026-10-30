# Generates public/diagrams/p12-induction-audit.svg and -dark.svg (data: p12-interpretability-open-problems.md).
# Usage: python3 diagrams/p12-induction-audit.py public/diagrams/p12-induction-audit, then render the PNG with sharp at density 144.
# Palette and approach copied from diagrams/g04-roofline.py. Data: p12/runs/induction_gpt2.json. GPT-2 small, mean loss on
# tokens 2-50 (2-200 for the long condition) of the second copy: intact, Part 4's top-5 induction heads mean-ablated (95% CI),
# and 20 random sets of 5 heads (median, 5th to 95th percentile); a dash marks the first-copy loss.
# ROWS: (label line 1, line 2, first copy, intact, top5, top5 lo, top5 hi, random median, random p5, random p95)
ROWS=[('Original','(Part 4 seeds)',13.16,0.218,5.48,5.43,5.54,0.249,0.206,0.383),
('New seeds','',13.15,0.233,5.46,5.40,5.51,0.262,0.220,0.389),
('Frequent','tokens',10.57,0.271,4.34,4.29,4.39,0.305,0.278,0.436),
('200-token','sequences',12.44,0.109,9.10,9.05,9.15,0.128,0.107,0.198),
('Repeated','WikiText',4.62,0.316,1.31,1.29,1.34,0.328,0.295,0.367)]
import math, sys
W,H=780,480; L,R,T,B=78,24,58,70
Y0,Y1=0.08,20
def sy(y): return H-B-(math.log10(y)-math.log10(Y0))/(math.log10(Y1)-math.log10(Y0))*(H-T-B)
def cx(i): return L+(W-L-R)/len(ROWS)*(i+0.5)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0.1,0.3,1,3,10):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:g}</text>')
    # legend
    lx=L; ly=18
    o.append(f'<circle cx="{lx+6}" cy="{ly}" r="5" fill="{acc2}"/><text x="{lx+16}" y="{ly+4}" fill="{ink}" font-size="13">Intact</text>')
    o.append(f'<circle cx="{lx+84}" cy="{ly}" r="5" fill="{mut}"/><text x="{lx+94}" y="{ly+4}" fill="{ink}" font-size="13">5 random heads ablated (median, 5-95%)</text>')
    o.append(f'<circle cx="{lx+372}" cy="{ly}" r="6" fill="{acc}"/><text x="{lx+382}" y="{ly+4}" fill="{ink}" font-size="13">Part 4\'s top 5 ablated</text>')
    o.append(f'<line x1="{lx+548}" x2="{lx+570}" y1="{ly}" y2="{ly}" stroke="{ink}" stroke-width="2.5"/><text x="{lx+576}" y="{ly+4}" fill="{ink}" font-size="13">First copy</text>')
    for i,(a,b,first,clean,t5,tlo,thi,rm,r5,r95) in enumerate(ROWS):
        x=cx(i)
        o.append(f'<line x1="{x-34:.1f}" x2="{x+34:.1f}" y1="{sy(first):.1f}" y2="{sy(first):.1f}" stroke="{ink}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{x-22:.1f}" cy="{sy(clean):.1f}" r="5" fill="{acc2}"/>')
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(r5):.1f}" y2="{sy(r95):.1f}" stroke="{mut}" stroke-width="2"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{sy(rm):.1f}" r="5" fill="{mut}"/>')
        o.append(f'<line x1="{x+22:.1f}" x2="{x+22:.1f}" y1="{sy(tlo):.1f}" y2="{sy(thi):.1f}" stroke="{ink}" stroke-width="1.5"/>')
        o.append(f'<circle cx="{x+22:.1f}" cy="{sy(t5):.1f}" r="6" fill="{acc}"/>')
        o.append(f'<text x="{x+32:.1f}" y="{sy(t5)+4:.1f}" fill="{ink}" font-size="12">{t5:.2f}</text>')
        o.append(f'<text x="{x:.1f}" y="{H-B+20}" fill="{ink}" font-size="13" text-anchor="middle">{a}</text>')
        if b: o.append(f'<text x="{x:.1f}" y="{H-B+36}" fill="{mut}" font-size="12" text-anchor="middle">{b}</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Loss on the second copy (nats, log scale)</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
