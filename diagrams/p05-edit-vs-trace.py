# Generates public/diagrams/p05-edit-vs-trace.svg and -dark.svg (data: p05-activation-patching.md).
# Usage: python3 diagrams/p05-edit-vs-trace.py public/diagrams/p05-edit-vs-trace, then render the PNG with sharp at density 144.
# Data: GPT-2 small, 67 landmark facts it knows. Per edited MLP layer: mean causal-tracing effect (window of 3 MLPs,
# max over tokens, share of P restored) and mean rewrite and paraphrase scores of a ROME-style rank-one edit to a
# new city, with 95% bootstrap CIs over facts (p05 scripts/rome.py, rome_summary.py).
import sys
D={"layers": [0, 1, 2, 3, 5, 7, 9, 11], "trace": [[0.229, 0.2772, 0.0484, 0.051, 0.0189, 0.0206, 0.1137, 0.0183], [0.1834, 0.2157, 0.0348, 0.0377, 0.0132, 0.0154, 0.0796, 0.0065], [0.2742, 0.3462, 0.0631, 0.0661, 0.0252, 0.0262, 0.1492, 0.0382]], "rewrite": [[0.9505, 0.9616, 0.9622, 0.9605, 0.9537, 0.9301, 0.9238, 0.0], [0.9425, 0.9566, 0.9567, 0.9511, 0.9372, 0.8893, 0.8764, -0.0], [0.9573, 0.966, 0.9671, 0.967, 0.9658, 0.9612, 0.9607, 0.0]], "para": [[0.8949, 0.9234, 0.9178, 0.9137, 0.9138, 0.8836, 0.7804, 0.0], [0.8782, 0.9118, 0.9043, 0.8964, 0.8938, 0.8399, 0.7322, -0.0], [0.9093, 0.9337, 0.9296, 0.9282, 0.9294, 0.9148, 0.8262, 0.0]]}
W,H=780,430; L,R,T,B=70,24,44,64
X=D["layers"]
def sx(l): return L+l/11*(W-L-R)
def sy(y): return H-B-y*(H-T-B)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0,0.25,0.5,0.75,1):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{y:g}</text>')
    for l in range(12):
        o.append(f'<text x="{sx(l):.1f}" y="{H-B+20}" fill="{mut}" font-size="12" text-anchor="middle">{l}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-18}" fill="{ink}" font-size="14" text-anchor="middle">MLP layer edited (or traced)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Score (0 to 1)</text>')
    series=[("rewrite",acc,"Rewrite score (edit prompt)",""),("para",acc2,"Paraphrase score (2 rephrasings)",""),("trace",mut,"Tracing effect (share restored)","6 4")]
    for k,c,lab,dash in series:
        m,lo,hi=D[k]
        band=' '.join(f'{sx(x):.1f},{sy(v):.1f}' for x,v in zip(X,hi))+' '+' '.join(f'{sx(x):.1f},{sy(v):.1f}' for x,v in zip(X[::-1],lo[::-1]))
        o.append(f'<polygon points="{band}" fill="{c}" opacity="0.18"/>')
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" stroke-dasharray="{dash}" points="'+' '.join(f'{sx(x):.1f},{sy(v):.1f}' for x,v in zip(X,m))+'"/>')
        for x,v in zip(X,m): o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{c}"/>')
    lx=L+10
    for i,(k,c,lab,dash) in enumerate(series):
        x0=lx+i*225
        o.append(f'<line x1="{x0}" x2="{x0+24}" y1="18" y2="18" stroke="{c}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
        o.append(f'<text x="{x0+30}" y="22" fill="{ink}" font-size="12">{lab}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
