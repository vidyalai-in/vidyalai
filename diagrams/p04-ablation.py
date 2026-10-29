# Generates public/diagrams/p04-ablation.svg and -dark.svg (data: p04-induction-heads.md).
# Usage: python3 diagrams/p04-ablation.py public/diagrams/p04-ablation, then render the PNG with sharp at density 144.
# Data: GPT-2 small, mean ablation (p04 scripts/s2_ablate.py, s3_lite.py). Left: mean loss on tokens 2-50 of the second copy,
# 500 repeated random sequences; random = 50 draws of k heads outside the top 12. Right: WikiText-103, 976 windows of 512 tokens,
# mean loss per position in log-spaced bins; random 12 = average of 5 draws.
import sys, math
D={"ks": [0, 1, 2, 3, 4, 5, 8, 12], "top": [0.21794701261818408, 0.26577490293979644, 0.35405476711690426, 0.4174870106726885, 3.3213285236358643, 5.48319153881073, 7.06215260887146, 8.122960153579712], "rand": {"1": [0.21419009268283845, 0.2237696498632431, 0.30698925256729126], "2": [0.2077278383076191, 0.22488151490688324, 0.2752163767814636], "3": [0.1999715507030487, 0.2372264340519905, 0.35731393843889225], "5": [0.2044184535741806, 0.2524406760931015, 1.082011124491689], "8": [0.20979641303420066, 0.28943274915218353, 0.5823222905397408], "12": [0.23082079440355302, 0.3878227025270462, 2.488289833068847]}, "clean": [[1.0, 10.603799819946289], [2.0, 7.767499923706055], [3.0, 6.5625], [4.47213595499958, 6.01324987411499], [6.48074069840786, 5.642249822616577], [8.94427190999916, 5.3831000328063965], [12.409673645990857, 4.72932493686676], [16.881943016134134, 4.542539882659912], [22.360679774997898, 4.3941333293914795], [29.732137494637012, 4.202666759490967], [40.55859958134649, 3.991030748073871], [54.99090833947008, 3.857924982905388], [73.7563556583431, 3.711827288974415], [99.44847912361456, 3.670443328221639], [134.08952233489387, 3.5845249831676482], [180.56577748842665, 3.5002944469451904], [243.35159748807897, 3.4083465811324447], [327.9329199699231, 3.346359187242936], [441.66955973895233, 3.311265902085738]], "top12": [[1.0, 10.463199615478516], [2.0, 7.677499771118164], [3.0, 6.515999794006348], [4.47213595499958, 5.948750019073486], [6.48074069840786, 5.587649822235107], [8.94427190999916, 5.359233379364014], [12.409673645990857, 4.772925019264221], [16.881943016134134, 4.595680141448975], [22.360679774997898, 4.441433350245158], [29.732137494637012, 4.260277801089817], [40.55859958134649, 4.0684230877802925], [54.99090833947008, 3.9566500186920166], [73.7563556583431, 3.824850017374212], [99.44847912361456, 3.8062233368555707], [134.08952233489387, 3.752037501335144], [180.56577748842665, 3.678255558013916], [243.35159748807897, 3.6133616395192605], [327.9329199699231, 3.582681633988205], [441.66955973895233, 3.570185601711273]], "rand12": [[1.0, 10.55526008605957], [2.0, 7.6237200736999515], [3.0, 6.477179908752442], [4.47213595499958, 5.935319995880127], [6.48074069840786, 5.603439998626708], [8.94427190999916, 5.372720019022624], [12.409673645990857, 4.748805046081543], [16.881943016134134, 4.567919998168945], [22.360679774997898, 4.4310800075531], [29.732137494637012, 4.254786703321668], [40.55859958134649, 4.046536933458769], [54.99090833947008, 3.9212637573480604], [73.7563556583431, 3.7850881858305496], [99.44847912361456, 3.7495806709925334], [134.08952233489387, 3.6654174971580504], [180.56577748842665, 3.582900004033689], [243.35159748807897, 3.4946076765452343], [327.9329199699231, 3.4372957156628976], [441.66955973895233, 3.4068657586068816]]}
W,H=780,400; T=58; B=64
PL=(66,352); PR=(452,756)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    # ---- left panel
    x0,x1=PL; y0,y1=0,14
    sx=lambda k: x0+k/12*(x1-x0); sy=lambda v: H-B-(v-y0)/(y1-y0)*(H-T-B)
    o.append(f'<text x="{x0}" y="{T-30}" fill="{ink}" font-size="14" font-weight="bold">Repeated random tokens</text>')
    o.append(f'<text x="{x0}" y="{T-12}" fill="{mut}" font-size="12">loss on the second copy, nats</text>')
    for v in (0,2,4,6,8,10,12,14):
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{x0-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{v}</text>')
    for k in (0,1,2,3,4,5,8,12):
        o.append(f'<text x="{sx(k):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{k}</text>')
    o.append(f'<text x="{(x0+x1)/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">Heads ablated</text>')
    o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(13.16):.1f}" y2="{sy(13.16):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{x0+6}" y="{sy(13.16)-6:.1f}" fill="{mut}" font-size="12">first copy, nothing to copy: 13.2</text>')
    rk=[0]+sorted(int(k) for k in D['rand'])
    lo=[D['top'][0]]+[D['rand'][str(k)][0] for k in rk[1:]]; hi=[D['top'][0]]+[D['rand'][str(k)][2] for k in rk[1:]]
    med=[D['top'][0]]+[D['rand'][str(k)][1] for k in rk[1:]]
    band=' '.join(f'{sx(k):.1f},{sy(v):.1f}' for k,v in zip(rk,hi))+' '+' '.join(f'{sx(k):.1f},{sy(v):.1f}' for k,v in zip(rk[::-1],lo[::-1]))
    o.append(f'<polygon points="{band}" fill="{acc2}" fill-opacity="0.22" stroke="none"/>')
    o.append(f'<polyline fill="none" stroke="{acc2}" stroke-width="2.5" points="{" ".join(f"{sx(k):.1f},{sy(v):.1f}" for k,v in zip(rk,med))}"/>')
    pts=[(k,v) for k,v in zip(D['ks'],D['top']) if v is not None]
    o.append(f'<polyline fill="none" stroke="{acc}" stroke-width="2.5" points="{" ".join(f"{sx(k):.1f},{sy(v):.1f}" for k,v in pts)}"/>')
    for k,v in pts: o.append(f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="4" fill="{acc}" stroke="{surf}" stroke-width="1.5"/>')
    o.append(f'<text x="{sx(5)+8:.1f}" y="{sy(D["top"][5])+16:.1f}" fill="{acc}" font-size="13" font-weight="bold">top-k induction heads</text>')
    o.append(f'<text x="{sx(5.3):.1f}" y="{sy(2.5):.1f}" fill="{acc2}" font-size="13" font-weight="bold">random k heads</text>')
    o.append(f'<text x="{sx(5.3):.1f}" y="{sy(2.5)+15:.1f}" fill="{mut}" font-size="11">median, 5th-95th percentile</text>')
    # ---- right panel
    x0,x1=PR; y0,y1=3.2,5.0; lx0,lx1=math.log10(10),math.log10(512)
    sx=lambda p: x0+(math.log10(p)-lx0)/(lx1-lx0)*(x1-x0); sy=lambda v: H-B-(v-y0)/(y1-y0)*(H-T-B)
    o.append(f'<text x="{x0}" y="{T-30}" fill="{ink}" font-size="14" font-weight="bold">WikiText-103</text>')
    o.append(f'<text x="{x0}" y="{T-12}" fill="{mut}" font-size="12">loss per token, nats</text>')
    for v in (3.2,3.6,4.0,4.4,4.8):
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/>')
        o.append(f'<text x="{x0-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{v:.1f}</text>')
    for p in (10,50,100,500):
        o.append(f'<line x1="{sx(p):.1f}" x2="{sx(p):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
        o.append(f'<text x="{sx(p):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{p}</text>')
    o.append(f'<text x="{(x0+x1)/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">Position in the window (log scale)</text>')
    for key,c,dash in (('rand12',acc2,''),('top12',acc,''),('clean',ink,'6 4')):
        pts=[(p,v) for p,v in D[key] if p>=10]
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" {"stroke-dasharray=%s" % chr(34)+dash+chr(34) if dash else ""} points="{" ".join(f"{sx(p):.1f},{sy(v):.1f}" for p,v in pts)}"/>')
    last={k:D[k][-1][1] for k in ('clean','top12','rand12')}
    o.append(f'<text x="{sx(115):.1f}" y="{sy(4.02):.1f}" fill="{acc}" font-size="13" font-weight="bold">top 12 induction</text>')
    o.append(f'<text x="{sx(115):.1f}" y="{sy(4.02)+16:.1f}" fill="{acc}" font-size="13" font-weight="bold">heads ablated</text>')
    o.append(f'<text x="{sx(60):.1f}" y="{sy(3.36):.1f}" fill="{ink}" font-size="13" font-weight="bold">intact</text>')
    o.append(f'<text x="{sx(60):.1f}" y="{sy(3.36)+16:.1f}" fill="{acc2}" font-size="13" font-weight="bold">random 12 ablated</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
