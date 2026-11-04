# Generates public/diagrams/e10-sandbag.svg and e10-sandbag-dark.svg (data: e10-safety-evals.md).
# Claude Haiku 5.5 on the same MMLU items under different system prompts. Each row: change in accuracy (points)
# against the neutral prompt, paired over items, 95% interval clustered by subject. Top four rows: 1,140 items,
# thinking off. Bottom row: 400 of them, adaptive thinking at effort low. Data: scratch e10/results.json.
# Usage: python3 diagrams/e10-sandbag.py public/diagrams/e10-sandbag, then render the PNG with sharp at density 144.
import sys

ROWS = [  # label, sub-label, diff, lo, hi, kind (0 = control, 1 = threat)
    ("Neutral prompt, sent again", "1,140 items, thinking off", 0.44, -0.57, 1.45, 0),
    ("\"Please do your very best\"", "1,140 items, thinking off", 1.49, 0.09, 2.89, 0),
    ("Low scores trigger unlearning", "1,140 items, thinking off", -0.35, -2.07, 1.37, 0),
    ("High scores trigger unlearning", "1,140 items, thinking off", -1.49, -3.51, 0.53, 1),
    ("High scores trigger unlearning", "400 items, adaptive thinking", 2.25, -0.37, 4.87, 1),
]
W, H = 780, 400; L, R, T, B = 270, 40, 30, 70
X0, X1 = -6, 6
def sx(v): return L + (v - X0) / (X1 - X0) * (W - L - R)

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in range(X0, X1 + 1, 2):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x:+d}</text>' if x else
                 f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">0</text>')
    o.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{T}" y2="{H-B}" stroke="{mut}" stroke-width="2"/>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-24}" fill="{ink}" font-size="14" text-anchor="middle">Change in accuracy vs the neutral prompt (points, 95% interval)</text>')
    rh = (H - T - B) / len(ROWS)
    for i, (lab, sub, d, lo, hi, k) in enumerate(ROWS):
        cy = T + rh * (i + 0.5)
        o.append(f'<text x="{L-12}" y="{cy-2:.1f}" fill="{ink}" font-size="13" text-anchor="end">{lab}</text>')
        o.append(f'<text x="{L-12}" y="{cy+14:.1f}" fill="{mut}" font-size="12" text-anchor="end">{sub}</text>')
        o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{cols[k]}" stroke-width="3"/>')
        for e in (lo, hi):
            o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{cy-7:.1f}" y2="{cy+7:.1f}" stroke="{cols[k]}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{sx(d):.1f}" cy="{cy:.1f}" r="6" fill="{cols[k]}" stroke="{surf}" stroke-width="2"/>')
        o.append(f'<text x="{sx(hi)+10:.1f}" y="{cy+5:.1f}" fill="{ink}" font-size="13">{d:+.2f}</text>')
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#0b5fd6', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#38bdf8', '#d95926')))
