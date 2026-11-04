# Generates public/diagrams/e07-votes-needed.svg and e07-votes-needed-dark.svg (data: e07-chatbot-arena-elo.md).
# Usage: python3 diagrams/e07-votes-needed.py public/diagrams/e07-votes-needed, then render the PNG with sharp at density 144.
# Data: random subsamples (2%, 5%, 10%, 25%, 50%; 3 each) of 135,632 arena votes, refit with Bradley-Terry and 200
# bootstrap resamples; a pair counts as separated when the 95% interval of its rating difference excludes zero.
# Pairs grouped by their full-data gap. Points: measured share, Wilson 95% interval over pairs x 3 subsamples.
# Dashed lines: the formula prediction, power = Phi(gap / (296 * sqrt(1/n_i + 1/n_j)) - 1.96), averaged over pairs.
import sys, math
X=[112,280,576,1421,2836]
S=[  # label, measured, lo, hi, predicted (percent)
 ('Gap 40-80 points',[21.1,58.5,77.0,94.1,98.7],[18.9,55.7,74.5,92.7,97.9],[23.5,61.3,79.3,95.3,99.2],[26.7,52.3,74.5,92.7,97.6]),
 ('Gap 20-40 points',[7.8,21.0,40.0,64.8,80.1],[6.0,18.1,36.3,61.0,76.8],[10.1,24.3,43.9,68.4,83.0],[10.3,19.7,33.6,60.4,79.1]),
 ('Gap 10-20 points',[4.5,10.9,11.5,27.7,40.9],[2.8,8.1,8.6,23.3,35.9],[7.2,14.6,15.2,32.6,46.1],[5.4,8.2,12.5,24.8,42.1]),
]
W,H=780,470; L,R,T,B=70,190,24,62
X0,X1=80,4000
def sx(x): return L+(math.log10(x)-math.log10(X0))/(math.log10(X1)-math.log10(X0))*(W-L-R)
def sy(y): return H-B-y/100*(H-T-B)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    cols=[acc2,acc,mut]
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in (100,200,500,1000,2000,4000):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:,}</text>')
    for y in range(0,101,20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Votes per model (median, log scale)</text>')
    o.append(f'<text transform="translate(18,{(T+H-B)/2:.1f}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Pairs separated at 95%</text>')
    for k,(lab,m,lo,hi,pr) in enumerate(S):
        c=cols[k]
        o.append('<polyline fill="none" stroke="%s" stroke-width="2" stroke-dasharray="6 5" points="%s"/>'%(c,' '.join(f'{sx(x):.1f},{sy(p):.1f}' for x,p in zip(X,pr))))
        for x,v,a,b in zip(X,m,lo,hi):
            o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(a):.1f}" y2="{sy(b):.1f}" stroke="{c}" stroke-width="2.5"/>')
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        ly=T+30+k*26
        o.append(f'<circle cx="{W-R+22}" cy="{ly-4}" r="5" fill="{c}"/>')
        o.append(f'<text x="{W-R+34}" y="{ly}" fill="{ink}" font-size="13">{lab}</text>')
    ly=T+30+3*26
    o.append(f'<line x1="{W-R+12}" x2="{W-R+32}" y1="{ly-4}" y2="{ly-4}" stroke="{ink}" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{W-R+38}" y="{ly}" fill="{ink}" font-size="13">Formula</text>')
    o.append(f'<circle cx="{W-R+22}" cy="{ly+22}" r="5" fill="{ink}"/>')
    o.append(f'<text x="{W-R+34}" y="{ly+26}" fill="{ink}" font-size="13">Measured on</text>')
    o.append(f'<text x="{W-R+34}" y="{ly+42}" fill="{ink}" font-size="13">real subsamples</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
