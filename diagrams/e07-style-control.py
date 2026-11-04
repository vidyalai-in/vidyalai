# Generates public/diagrams/e07-style-control.svg and e07-style-control-dark.svg (data: e07-chatbot-arena-elo.md).
# Usage: python3 diagrams/e07-style-control.py public/diagrams/e07-style-control, then render the PNG with sharp at density 144.
# Data: 135,632 arena votes (lmarena-ai/arena-human-preference-140k). Hollow: plain Bradley-Terry (95% interval from 1,000
# vote resamples). Filled: Bradley-Terry with four style features (answer tokens, markdown headers, list items, bold),
# 95% interval from 100 vote resamples. Ranks among 52 models. Tokens: mean assistant tokens per vote, all turns.
import sys
ROWS=[  # model, bt, lo, hi, sc, lo, hi, rank_bt, rank_sc, tokens
 ('o3-2025-04-16',1079.4,1073.0,1085.9,1083.4,1076.4,1089.3,4,1,1203),
 ('gemini-2.5-pro',1124.1,1117.7,1130.6,1078.8,1072.1,1084.5,1,2,1874),
 ('kimi-k2-0711-preview',1031.0,1019.6,1042.1,1058.1,1048.0,1069.7,13,5,872),
 ('claude-opus-4-20250514',1005.1,999.7,1011.0,1044.0,1038.7,1049.7,28,9,746),
 ('o4-mini-2025-04-16',1023.1,1016.0,1030.5,1043.0,1036.0,1049.1,19,10,912),
 ('grok-3-preview-02-24',1071.3,1063.7,1078.8,1025.6,1018.8,1034.0,8,15,1952),
 ('claude-sonnet-4-20250514',974.0,967.5,980.2,1017.5,1012.1,1025.3,38,19,707),
 ('llama-4-maverick-03-26-experimental',1067.2,1059.2,1075.5,1011.0,1002.4,1021.2,9,22,2067),
 ('claude-3-7-sonnet-20250219-thinking-32k',936.3,929.5,943.8,1008.5,1001.4,1014.8,42,23,532),
 ('claude-3-5-sonnet-20241022',931.5,924.6,938.6,997.9,987.9,1006.3,45,30,473),
 ('qwen-max-2025-01-25',1024.6,1016.4,1032.7,997.3,989.3,1005.4,18,31,1366),
 ('llama-4-maverick-17b-128e-instruct',935.9,928.1,943.9,954.3,948.5,962.1,43,44,816),
]
W,H=780,600; L,R,T,B=280,150,70,56
X0,X1=900,1150
RH=(H-T-B)/len(ROWS)
def sx(v): return L+(v-X0)/(X1-X0)*(W-L-R)
def svg(th):
    ink,mut,grid,acc,surf,acc2=th
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in range(900,1151,50):
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{v}</text>')
    o.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-12}" fill="{ink}" font-size="14" text-anchor="middle">Rating (Elo scale, mean 1000)</text>')
    o.append(f'<circle cx="{L+6}" cy="22" r="6" fill="{surf}" stroke="{mut}" stroke-width="2.5"/>')
    o.append(f'<text x="{L+18}" y="27" fill="{ink}" font-size="13">Plain Bradley-Terry</text>')
    o.append(f'<circle cx="{L+160}" cy="22" r="6" fill="{acc}"/>')
    o.append(f'<text x="{L+172}" y="27" fill="{ink}" font-size="13">Style-controlled, rises</text>')
    o.append(f'<circle cx="{L+330}" cy="22" r="6" fill="{acc2}"/>')
    o.append(f'<text x="{L+342}" y="27" fill="{ink}" font-size="13">falls (bars: 95% interval)</text>')
    o.append(f'<text x="{W-R+50}" y="{T-14}" fill="{ink}" font-size="13" text-anchor="middle">Rank</text>')
    o.append(f'<text x="{W-R+118}" y="{T-14}" fill="{ink}" font-size="13" text-anchor="middle">Tokens</text>')
    for k,(m,bt,blo,bhi,sc,lo,hi,r0,r1,tok) in enumerate(ROWS):
        y=T+k*RH+RH/2
        up=sc>bt
        c=acc if up else acc2
        o.append(f'<text x="{L-10}" y="{y+5:.1f}" fill="{ink}" font-size="13" text-anchor="end">{m}</text>')
        o.append(f'<line x1="{sx(bt):.1f}" x2="{sx(sc):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="3 3"/>')
        o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="4"/>')
        o.append(f'<circle cx="{sx(bt):.1f}" cy="{y:.1f}" r="6" fill="{surf}" stroke="{mut}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{sx(sc):.1f}" cy="{y:.1f}" r="6" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<text x="{W-R+50}" y="{y+5:.1f}" fill="{ink}" font-size="13" text-anchor="middle">{r0} → {r1}</text>')
        o.append(f'<text x="{W-R+118}" y="{y+5:.1f}" fill="{ink}" font-size="13" text-anchor="middle">{tok:,}</text>')
    o.append('</svg>'); return '\n'.join(o)
light=('#0b1a33','#4a5b78','#dde5f2','#0b5fd6','#ffffff','#eb6834')
dark=('#e8eefc','#9fb0cf','#22355a','#38bdf8','#0b162d','#d95926')
open(sys.argv[1]+'.svg','w').write(svg(light)); open(sys.argv[1]+'-dark.svg','w').write(svg(dark))
