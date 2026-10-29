# Generates public/diagrams/p03-pos-control.svg and p03-pos-control-dark.svg (data: p03-linear-probes.md).
# Usage: python3 diagrams/p03-pos-control.py public/diagrams/p03-pos-control, then render the PNG with sharp at density 144.
# Data: GPT-2 small (fp32, TransformerLens 3.9.0, no weight processing), part-of-speech probes on UD English EWT v2.18
# (17 UPOS tags; trained on 64,014 words from 4,000 train sentences, tested on 24,386 words from all 2,077 test sentences),
# read at each word's last subword token. Real: accuracy on gold tags. Control: Hewitt & Liang (2019) control task,
# each word type mapped to a random tag (mean of 3 control seeds). Word identity: most frequent training tag per word.
# 95% bootstrap CIs over sentences are under +/-0.4 points for every real-tag point, so they are not drawn.
import sys
LIN=[(0,0.8839,0.7215),(1,0.9164,0.7026),(2,0.9250,0.6962),(3,0.9323,0.6815),(4,0.9368,0.6628),(5,0.9408,0.6442),(6,0.9425,0.6318),(7,0.9435,0.6198),(8,0.9440,0.6098),(9,0.9436,0.5992),(10,0.9407,0.5855),(11,0.9348,0.5659),(12,0.9268,0.5476)]
MLP=[(0,0.8859,0.8576),(1,0.9138,0.8604),(2,0.9269,0.8603),(3,0.9337,0.8538),(4,0.9387,0.8420),(5,0.9410,0.8301),(6,0.9426,0.8200),(7,0.9418,0.8099),(8,0.9430,0.7973),(9,0.9441,0.7796),(10,0.9421,0.7561),(11,0.9383,0.7271),(12,0.9345,0.7028)]
WID=0.8405
W,H=780,470; L,R,T,B=70,24,24,62
Y0,Y1=0.50,1.0
def sx(l): return L+l/12*(W-L-R)
def sy(y): return H-B-(y-Y0)/(Y1-Y0)*(H-T-B)
def line(pts,c,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    s=[f'<polyline points="{" ".join(f"{sx(l):.1f},{sy(v):.1f}" for l,v in pts)}" fill="none" stroke="{c}" stroke-width="2.5"{d}/>']
    s+= [f'<circle cx="{sx(l):.1f}" cy="{sy(v):.1f}" r="4" fill="{c}"/>' for l,v in pts]
    return s
def svg(th,cat):
    ink,mut,grid,surf=th; c1,c2=cat
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0.5,0.6,0.7,0.8,0.9,1.0):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{int(round(y*100))}%</text>')
    for l in range(0,13,2):
        o.append(f'<line x1="{sx(l):.1f}" x2="{sx(l):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(l):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{l}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">GPT-2 small layer (0 = embeddings, 12 = output of the last block)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Test accuracy (24,386 words)</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(WID):.1f}" y2="{sy(WID):.1f}" stroke="{mut}" stroke-width="1.8" stroke-dasharray="2 4"/>')
    o.append(f'<text x="{W-R-4}" y="{sy(WID)-7:.1f}" fill="{mut}" font-size="13" text-anchor="end">no model, just a lookup of each word\'s usual tag: 84.1%</text>')
    o+=line([(l,a) for l,a,_ in MLP],c2); o+=line([(l,a) for l,a,_ in LIN],c1)
    o+=line([(l,c) for l,_,c in MLP],c2,"7 5"); o+=line([(l,c) for l,_,c in LIN],c1,"7 5")
    o.append(f'<text x="{sx(6):.1f}" y="{sy(0.9425)-12:.1f}" fill="{ink}" font-size="13" text-anchor="middle">real tags: linear and MLP probes overlap (94.4% at layer 8)</text>')
    o.append(f'<text x="{sx(3.6):.1f}" y="{sy(0.755):.1f}" fill="{ink}" font-size="13" text-anchor="start">MLP probe, random control tags</text>')
    o.append(f'<text x="{sx(3.6):.1f}" y="{sy(0.695):.1f}" fill="{ink}" font-size="13" text-anchor="start">linear probe, random control tags</text>')
    o.append(f'<text x="{sx(12)-4:.1f}" y="{sy(0.7028)+22:.1f}" fill="{ink}" font-size="13" text-anchor="end">70.3%</text>')
    o.append(f'<text x="{sx(12)-4:.1f}" y="{sy(0.5476)+22:.1f}" fill="{ink}" font-size="13" text-anchor="end">54.8%</text>')
    lx,ly=L+290,H-B-46
    for i,(c,lab,dash) in enumerate(((c1,'linear probe',None),(c2,'MLP probe (1,024 hidden units)',None))):
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{ly+i*20}" y2="{ly+i*20}" stroke="{c}" stroke-width="2.5"/>')
        o.append(f'<text x="{lx+34}" y="{ly+i*20+4}" fill="{ink}" font-size="13">{lab}</text>')
    o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{ly+40}" y2="{ly+40}" stroke="{mut}" stroke-width="2.5" stroke-dasharray="7 5"/>')
    o.append(f'<text x="{lx+34}" y="{ly+44}" fill="{ink}" font-size="13">dashed: control task</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#ffffff'); dark=('#e8eefc','#9fb0cf','#22355a','#0b162d')
open(sys.argv[1]+'.svg','w').write(svg(light,('#2a78d6','#1baf7a'))); open(sys.argv[1]+'-dark.svg','w').write(svg(dark,('#3987e5','#199e70')))
