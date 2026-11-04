# Generates public/diagrams/e07-mini-arena.svg and e07-mini-arena-dark.svg (data: e07-chatbot-arena-elo.md).
# Usage: python3 diagrams/e07-mini-arena.py public/diagrams/e07-mini-arena, then render the PNG with sharp at density 144.
# Data: 60 English arena prompts, 5 Claude configurations, every ordered pair judged once by claude-haiku-5-5 (thinking
# off): 1,200 judgments. Bradley-Terry ratings (mean 1000), 95% intervals from 2,000 (plain) and 500 (style-controlled)
# bootstrap resamples of whole prompts. MMLU: accuracy on MMLU-1140 from Part 3 (first run); "n/a" = not measured.
import sys
ROWS=[  # label, bt, lo, hi, sc, lo, hi, mmlu
 ('Haiku, "detail + markdown" prompt',1195.1,1145.8,1251.7,867.8,729.6,953.9,'n/a'),
 ('Sonnet 5.5, thinking off',1144.4,1099.1,1198.7,1118.5,1077.7,1173.8,'86.7%'),
 ('Haiku 5.5, adaptive thinking, low',1042.7,997.0,1094.0,1081.6,1039.4,1134.2,'89.7%'),
 ('Haiku 5.5, thinking off',920.3,886.2,952.2,975.8,932.1,1030.4,'79.7%'),
 ('Haiku, "be concise" prompt',697.5,592.5,772.3,956.3,864.2,1061.1,'n/a'),
]
W,H=780,400; L,R,T,B=250,90,74,56
X0,X1=550,1300
RH=(H-T-B)/len(ROWS)
def sx(v): return L+(v-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(600,1301,100):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Rating from Haiku-judged pairs (Elo scale, mean 1000)</text>')
    o.append(f'<line x1="{L}" x2="{L+30}" y1="22" y2="22" stroke="{acc2}" stroke-width="4"/><circle cx="{L+15}" cy="22" r="5" fill="{acc2}"/>')
    o.append(f'<text x="{L+40}" y="27" fill="{ink}" font-size="13">Plain Bradley-Terry</text>')
    o.append(f'<line x1="{L+200}" x2="{L+230}" y1="22" y2="22" stroke="{acc}" stroke-width="4"/><circle cx="{L+215}" cy="22" r="5" fill="{acc}"/>')
    o.append(f'<text x="{L+240}" y="27" fill="{ink}" font-size="13">Style-controlled (95% intervals)</text>')
    o.append(f'<text x="{W-R+45}" y="{T-12}" fill="{ink}" font-size="13" text-anchor="middle">MMLU</text>')
    for k,(m,bt,blo,bhi,sc,lo,hi,mm) in enumerate(ROWS):
        y=T+k*RH+RH/2
        o.append(f'<text x="{L-10}" y="{y+5:.1f}" fill="{ink}" font-size="13" text-anchor="end">{m}</text>')
        o.append(f'<line x1="{sx(blo):.1f}" x2="{sx(bhi):.1f}" y1="{y-7:.1f}" y2="{y-7:.1f}" stroke="{acc2}" stroke-width="3.5"/>')
        o.append(f'<circle cx="{sx(bt):.1f}" cy="{y-7:.1f}" r="5" fill="{acc2}" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y+7:.1f}" y2="{y+7:.1f}" stroke="{acc}" stroke-width="3.5"/>')
        o.append(f'<circle cx="{sx(sc):.1f}" cy="{y+7:.1f}" r="5" fill="{acc}" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<text x="{W-R+45}" y="{y+5:.1f}" fill="{ink}" font-size="13" text-anchor="middle">{mm}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
