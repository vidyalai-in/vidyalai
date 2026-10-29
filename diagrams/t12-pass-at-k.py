# Generates public/diagrams/t12-pass-at-k.svg and -dark.svg (data: t12-grpo-from-scratch.md).
# Usage: python3 diagrams/t12-pass-at-k.py public/diagrams/t12-pass-at-k, then render the PNG with sharp at density 144.
# Data: t12/scripts/evaluate.py. pass@k from n samples per problem at temperature 1 (n = 64 for the warm start and the
# GRPO model, 16 for the Part 10 SFT model, so its line stops at k = 16), unbiased estimator of Chen et al. (2021),
# strict checker. 'mixed' = warm start with 4,000 smol-smoltalk conversations added. Panels: held-out problems with 1 to 3 digits per number, and with exactly 4 and 5 digits.
MODELS=[('Part 10 SFT model', 'sft'), ('warm start', 'warm'), ('after GRPO', 'main'), ('mixed warm start', 'warm_mixed'), ('mixed + GRPO', 'grpo_mixed')]
PANELS=[('1 to 3 digits (200 held-out)', [{'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0}, {'1': 0.1138, '2': 0.1788, '4': 0.2598, '8': 0.3573, '16': 0.4698, '32': 0.5907, '64': 0.72}, {'1': 0.599, '2': 0.7225, '4': 0.8072, '8': 0.8636, '16': 0.9027, '32': 0.9265, '64': 0.94}, {'1': 0.0541, '2': 0.0908, '4': 0.1422, '8': 0.2129, '16': 0.3088, '32': 0.4297, '64': 0.57}, {'1': 0.2393, '2': 0.325, '4': 0.416, '8': 0.5096, '16': 0.603, '32': 0.6942, '64': 0.78}]), ('4 digits (100)', [{'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0}, {'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0, '32': 0.0, '64': 0.0}, {'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0, '32': 0.0, '64': 0.0}, {'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0, '32': 0.0, '64': 0.0}, {'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0, '32': 0.0, '64': 0.0}]), ('5 digits (100)', [{'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0}, {'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0, '32': 0.0, '64': 0.0}, {'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0, '32': 0.0, '64': 0.0}, {'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0, '32': 0.0, '64': 0.0}, {'1': 0.0, '2': 0.0, '4': 0.0, '8': 0.0, '16': 0.0, '32': 0.0, '64': 0.0}])]
import sys, math
W,H=780,360; L,R,T,B=64,20,70,64
NP=len(PANELS); GAP=30
PW=(W-L-R-GAP*(NP-1))/NP
KS=[1,2,4,8,16,32,64]
def px(i): return L+i*(PW+GAP)
def sx(i,k): return px(i)+math.log2(k)/math.log2(KS[-1])*PW
def sy(v): return H-B-v*(H-T-B)
def svg(th,cols):
    ink,mut,grid,surf=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    lx=L
    for m,(lab,_) in enumerate(MODELS):
        o.append(f'<line x1="{lx}" x2="{lx+22}" y1="18" y2="18" stroke="{cols[m]}" stroke-width="3"/>')
        o.append(f'<circle cx="{lx+11}" cy="18" r="4" fill="{cols[m]}"/>')
        o.append(f'<text x="{lx+28}" y="22" fill="{ink}" font-size="13">{lab}</text>')
        lx+=40+7.0*len(lab)
    for i,(title,series) in enumerate(PANELS):
        o.append(f'<text x="{px(i)}" y="{T-14}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for v in (0,0.25,0.5,0.75,1.0):
            o.append(f'<line x1="{px(i):.1f}" x2="{px(i)+PW:.1f}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}" stroke-width="1"/>')
            if i==0: o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v:.0%}</text>')
        for k in KS:
            o.append(f'<line x1="{sx(i,k):.1f}" x2="{sx(i,k):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{sx(i,k):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{k}</text>')
        if all(v==0 for vals in series for v in vals.values()):
            o.append(f'<text x="{px(i)+PW/2:.1f}" y="{sy(0.5):.1f}" fill="{ink}" font-size="13" text-anchor="middle">All five models: 0%</text>')
            o.append(f'<text x="{px(i)+PW/2:.1f}" y="{sy(0.5)+18:.1f}" fill="{mut}" font-size="12" text-anchor="middle">lines overlap at zero</text>')
        for m,vals in enumerate(series):
            pts=[(sx(i,k),sy(vals[str(k)])) for k in KS if str(k) in vals]
            o.append(f'<polyline fill="none" stroke="{cols[m]}" stroke-width="2.5" points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}"/>')
            for x,y in pts: o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{cols[m]}" stroke="{surf}" stroke-width="1"/>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-18}" fill="{ink}" font-size="14" text-anchor="middle">k, samples per problem (log scale): pass@k is the share of problems with at least one right answer</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,['#4a5b78','#eb6834','#0b5fd6','#d4a017','#1a9e5c']))
open(sys.argv[1]+'-dark.svg','w').write(svg(dark,['#9fb0cf','#d95926','#38bdf8','#e8b931','#34d399']))
