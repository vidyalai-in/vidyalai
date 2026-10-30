# Generates public/diagrams/p10-phase-change.svg and -dark.svg (data: p10-training-dynamics-interpretability.md).
# Usage: python3 diagrams/p10-phase-change.py public/diagrams/p10-phase-change, then render the PNG with sharp at density 144.
# Data: the From Scratch 88M run's 48 trained snapshots (steps 5 to 3,360; 524,288 tokens per step), measured with
# lab/dynamics.py (p10 scripts/track.py). Slope: -(change in validation loss) / (change in ln step) between
# consecutive validation points (train.py's 40-batch subset, every 50 steps). Scores: mean attention over 200 repeated
# random sequences (induction, duplicate-token) or 200 validation windows of 256 tokens (previous-token, heads in
# layers 0 to 7). ICL score: loss over tokens 491-510 minus tokens 41-60, 1,000 validation documents, 95% CI band.
import sys, math
D=__DATA__
W=780; L=70; R=150; TOP=40; PH=118; GAP=58
NP=4; H=TOP+NP*PH+(NP-1)*GAP+64
X0,X1=math.log10(5),math.log10(3360)
def sx(s): return L+(math.log10(s)-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    def panel(i,title,unit,y0,y1,ticks):
        top=TOP+i*(PH+GAP); bot=top+PH
        sy=lambda v: bot-(v-y0)/(y1-y0)*PH
        o.append(f'<rect x="{sx(300):.1f}" y="{top}" width="{sx(600)-sx(300):.1f}" height="{PH}" fill="{acc}" fill-opacity="0.10"/>')
        o.append(f'<text x="{L}" y="{top-12}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        o.append(f'<text x="{W-R}" y="{top-12}" fill="{mut}" font-size="12" text-anchor="end">{unit}</text>')
        for v in ticks:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
            o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{v:g}</text>')
        for s in (10,30,100,300,1000,3000):
            o.append(f'<line x1="{sx(s):.1f}" x2="{sx(s):.1f}" y1="{top}" y2="{bot}" stroke="{grid}"/>')
            o.append(f'<text x="{sx(s):.1f}" y="{bot+16}" fill="{mut}" font-size="12" text-anchor="middle">{s:,}</text>')
        return sy
    def line(pts,sy,col,dash=False,w=2.2):
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="{w}" {"stroke-dasharray=\"6 4\"" if dash else ""} points="'+' '.join(f'{sx(s):.1f},{sy(v):.1f}' for s,v in pts)+'"/>')
    def label(s,v,sy,txt,col,dy=4,dx=8,anchor='start'):
        o.append(f'<text x="{sx(s)+dx:.1f}" y="{sy(v)+dy:.1f}" fill="{col}" font-size="12" font-weight="bold" text-anchor="{anchor}">{txt}</text>')
    st=D['steps']
    # 1: slope of the validation loss
    sy=panel(0,'How fast the loss falls per doubling of steps','nats per unit of ln(step)',0,1.6,(0,0.4,0.8,1.2,1.6))
    pts=[(s,v) for s,v in D['slope'] if 5<=s<=3360]
    line(pts,sy,ink)
    for s,v in pts: o.append(f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="3" fill="{ink}"/>')
    o.append(f'<text x="{sx(300)+(sx(600)-sx(300))/2:.1f}" y="{TOP+14}" fill="{acc}" font-size="12" font-weight="bold" text-anchor="middle">steps 300 to 600</text>')
    # 2: head scores
    sy=panel(1,'Attention scores of the strongest head','score, 0 to 1',0,1,(0,0.25,0.5,0.75,1))
    line(list(zip(st,D['prev'])),sy,acc2); line(list(zip(st,D['ind'])),sy,acc); line(list(zip(st,D['dup'])),sy,mut,dash=True)
    label(st[-1],D['ind'][-1],sy,'induction',acc,dy=8); label(st[-1],D['prev'][-1],sy,'previous-token',acc2,dy=18); label(st[-1],D['dup'][-1],sy,'duplicate-token',mut,dy=-8)
    # 3: second copy loss
    sy=panel(2,'Loss on a repeated random sequence','nats per token',0,12,(0,4,8,12))
    line(list(zip(st,D['first'])),sy,mut,dash=True); line(list(zip(st,D['second'])),sy,acc)
    label(st[-1],D['first'][-1],sy,'first copy',mut); label(st[-1],D['second'][-1],sy,'second copy',acc)
    # 4: ICL score
    sy=panel(3,'In-context learning score on real text','late minus early tokens, nats',-0.35,0.1,(-0.3,-0.2,-0.1,0,0.1))
    up=[(s,m+c) for s,(m,c) in zip(st,D['icl'])]; lo=[(s,m-c) for s,(m,c) in zip(st,D['icl'])]
    o.append('<polygon points="'+' '.join(f'{sx(s):.1f},{sy(v):.1f}' for s,v in up+lo[::-1])+f'" fill="{acc}" fill-opacity="0.18"/>')
    line([(s,m) for s,(m,c) in zip(st,D['icl'])],sy,acc)
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="13" text-anchor="middle">Training step (log scale; 524,288 tokens per step)</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
