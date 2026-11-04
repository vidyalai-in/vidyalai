# Generates public/diagrams/e07-elo-order.svg and e07-elo-order-dark.svg (data: e07-chatbot-arena-elo.md).
# Usage: python3 diagrams/e07-elo-order.py public/diagrams/e07-elo-order, then render the PNG with sharp at density 144.
# Data: 135,632 votes of lmarena-ai/arena-human-preference-140k (April 17 to July 24, 2025), 52 models, ties as half a win.
# Per model (top 12 by Bradley-Terry): BT 95% bootstrap interval (1,000 resamples of votes), and the 2.5th-97.5th
# percentile of online Elo over 200 random vote orders at K=4 and at K=32 (evals/lab/arena.py, e07 runs/exp1.json).
import sys
ROWS=[  # model, bt, bt_lo, bt_hi, k4_lo, k4_hi, k32_lo, k32_hi
 ('gemini-2.5-pro',1124.1,1117.7,1130.6,1099.6,1154.8,1020.7,1224.4),
 ('gemini-2.5-pro-preview-03-25',1097.9,1082.5,1114.9,1072.6,1121.0,1019.9,1182.6),
 ('grok-4-0709',1093.4,1078.4,1107.8,1068.9,1120.3,1003.8,1171.4),
 ('o3-2025-04-16',1079.4,1073.0,1085.9,1044.1,1103.5,985.6,1169.7),
 ('chatgpt-4o-latest-20250326',1078.1,1071.1,1084.7,1047.5,1109.4,975.8,1161.0),
 ('gemini-2.5-pro-preview-05-06',1074.8,1064.1,1084.3,1045.4,1100.4,992.9,1158.1),
 ('deepseek-r1-0528',1074.5,1067.6,1082.1,1043.5,1100.7,986.5,1153.5),
 ('grok-3-preview-02-24',1071.3,1063.7,1078.8,1044.7,1108.1,994.0,1153.5),
 ('llama-4-maverick-03-26-experimental',1067.2,1059.2,1075.5,1039.4,1093.3,978.8,1158.5),
 ('gemini-2.5-flash',1061.3,1055.3,1067.3,1031.5,1093.5,985.0,1157.4),
 ('qwen3-235b-a22b-no-thinking',1041.3,1035.2,1047.3,1013.4,1072.2,952.3,1121.6),
 ('gemini-2.5-flash-preview-04-17',1034.8,1026.8,1043.1,1007.3,1067.8,943.8,1124.9),
]
W,H=780,640; L,R,T,B=262,24,78,56
X0,X1=940,1240
RH=(H-T-B)/len(ROWS)
def sx(v): return L+(v-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(950,1241,50):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Rating (Elo scale, mean of all 52 models = 1000)</text>')
    # legend
    lg=[(acc,'Bradley-Terry, 95% bootstrap interval',3.5),(acc2,'Online Elo, K = 4: range over 200 vote orders',2.5),(mut,'Online Elo, K = 32: range over 200 vote orders',1.5)]
    for k,(c,t,wd) in enumerate(lg):
        y=18+k*20
        o.append(f'<line x1="{L}" x2="{L+34}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="{wd+1}"/>')
        o.append(f'<text x="{L+44}" y="{y+5}" fill="{ink}" font-size="13">{t}</text>')
    for k,(m,bt,lo,hi,a,b,c,d) in enumerate(ROWS):
        y=T+k*RH+RH/2
        o.append(f'<text x="{L-10}" y="{y+5:.1f}" fill="{ink}" font-size="13" text-anchor="end">{m}</text>')
        o.append(f'<line x1="{sx(c):.1f}" x2="{sx(d):.1f}" y1="{y+9:.1f}" y2="{y+9:.1f}" stroke="{mut}" stroke-width="2.5"/>')
        o.append(f'<line x1="{sx(a):.1f}" x2="{sx(b):.1f}" y1="{y+2:.1f}" y2="{y+2:.1f}" stroke="{acc2}" stroke-width="3.5"/>')
        o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y-6:.1f}" y2="{y-6:.1f}" stroke="{acc}" stroke-width="4.5"/>')
        o.append(f'<circle cx="{sx(bt):.1f}" cy="{y-6:.1f}" r="4.5" fill="{acc}" stroke="{surf}" stroke-width="1.5"/>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
