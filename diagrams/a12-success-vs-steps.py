# Generates public/diagrams/a12-success-vs-steps.svg and a12-success-vs-steps-dark.svg (data: a12-agent-reliability.md).
# Share of chain tasks solved against the number of dependent steps, claude-haiku-5-5, thinking disabled, effort medium.
# Points: measured, with 95% Wilson intervals. Lines: p^n from the pooled per-step success rate, and the
# survival curve built from the measured hazard of the first error at each step (Kaplan-Meier).
# Data from scratch a12/analyze_chain.py. Usage: python3 diagrams/a12-success-vs-steps.py public/diagrams/a12-success-vs-steps
import sys

W, H = 780, 470; L, R, T, B = 78, 30, 24, 62
X0, X1 = 0, 80; Y0, Y1 = 0, 100
def sx(x): return L + (x - X0) / (X1 - X0) * (W - L - R)
def sy(y): return H - B - (y - Y0) / (Y1 - Y0) * (H - T - B)

P = 0.9734                     # pooled per-step success
MEASURED = [(5, 66.7, 48.8, 80.8), (10, 70.0, 52.1, 83.3), (20, 56.7, 39.2, 72.6), (40, 46.7, 30.2, 63.9), (80, 30.0, 14.5, 51.9)]       # (n, solved %, CI low, CI high)
SURVIVAL = [(1, 81.4), (2, 78.6), (3, 77.1), (4, 75.0), (5, 75.0), (6, 75.0), (7, 75.0), (8, 72.4), (9, 70.6), (10, 70.6), (11, 69.4), (12, 68.2), (13, 67.0), (14, 65.8), (15, 64.6), (16, 61.0), (17, 58.6), (18, 57.4), (19, 56.2), (20, 55.0), (21, 55.0), (22, 55.0), (23, 55.0), (24, 55.0), (25, 55.0), (26, 55.0), (27, 55.0), (28, 53.2), (29, 53.2), (30, 51.4), (31, 51.4), (32, 51.4), (33, 51.4), (34, 51.4), (35, 49.5), (36, 47.7), (37, 47.7), (38, 45.9), (39, 45.9), (40, 44.0), (41, 44.0), (42, 44.0), (43, 44.0), (44, 44.0), (45, 44.0), (46, 40.0), (47, 40.0), (48, 40.0), (49, 36.0), (50, 36.0), (51, 32.0), (52, 28.0), (53, 24.0), (54, 24.0), (55, 24.0), (56, 24.0), (57, 24.0), (58, 24.0), (59, 24.0), (60, 24.0), (61, 24.0), (62, 24.0), (63, 24.0), (64, 24.0), (65, 24.0), (66, 24.0), (67, 24.0), (68, 24.0), (69, 24.0), (70, 24.0), (71, 24.0), (72, 24.0), (73, 24.0), (74, 24.0), (75, 24.0), (76, 24.0), (77, 24.0), (78, 24.0), (79, 24.0), (80, 24.0)]       # (n, % with no error through step n), every step
EXTRA = [("Copy the record first (20 chains)", 40, 60.0, 38.7, 78.1)]             # (label, n, %, lo, hi) for the restate arm


def svg(th, cols):
    ink, mut, grid, surf = th
    c_meas, c_pn, c_km, c_extra = cols
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(Y0, Y1 + 1, 20):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for x in (0, 5, 10, 20, 40, 60, 80):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Dependent steps in the task</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Tasks solved (95% CI)</text>')
    pn = " ".join(f"{sx(n):.1f},{sy(100 * P ** n):.1f}" for n in range(0, X1 + 1))
    o.append(f'<polyline fill="none" stroke="{c_pn}" stroke-width="2" stroke-dasharray="6 5" points="{pn}"/>')
    km = " ".join(f"{sx(n):.1f},{sy(v):.1f}" for n, v in [(0, 100)] + SURVIVAL)
    o.append(f'<polyline fill="none" stroke="{c_km}" stroke-width="2.5" points="{km}"/>')
    for n, y, lo, hi in MEASURED:
        o.append(f'<line x1="{sx(n):.1f}" x2="{sx(n):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c_meas}" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(n):.1f}" cy="{sy(y):.1f}" r="5.5" fill="{c_meas}" stroke="{surf}" stroke-width="2"/>')
    for lab, n, y, lo, hi in EXTRA:
        x = sx(n) + 12
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c_extra}" stroke-width="2"/>')
        o.append(f'<rect x="{x-5:.1f}" y="{sy(y)-5:.1f}" width="10" height="10" fill="{c_extra}" stroke="{surf}" stroke-width="2"/>')
    leg = [("circle", c_meas, "Measured (Haiku 5.5)"), ("dash", c_pn, f"p^n with pooled p = {P:.3f}"),
           ("line", c_km, "No error yet (first-error hazard per step)")] + \
          [("rect", c_extra, lab) for lab, *_ in EXTRA]
    for k, (kind, c, lab) in enumerate(leg):
        x, y = L + 24, sy(30) + k * 22
        if kind == "circle":
            o.append(f'<circle cx="{x+12}" cy="{y}" r="5" fill="{c}"/>')
        elif kind == "rect":
            o.append(f'<rect x="{x+7}" y="{y-5}" width="10" height="10" fill="{c}"/>')
        else:
            dash = ' stroke-dasharray="6 5"' if kind == "dash" else ""
            o.append(f'<line x1="{x}" x2="{x+24}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"{dash}/>')
        o.append(f'<text x="{x+32}" y="{y+5}" fill="{ink}" font-size="13">{lab}</text>')
    o.append('</svg>'); return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834', '#1baf7a', '#8a5cd6')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926', '#199e70', '#9b72e0')))
