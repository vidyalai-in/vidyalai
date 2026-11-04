# Generates public/diagrams/e10-graders.svg and e10-graders-dark.svg (data: e10-safety-evals.md).
# XSTest replies from Claude Haiku 5.5 (thinking off, 2 runs), Qwen2.5-1.5B-Instruct and Gemma 2 2B IT (greedy),
# each scored by four graders. Left panel: share of the 250 safe prompts refused (over-refusal). Right panel: share of
# replies to the 200 unsafe prompts NOT refused (what a jailbreak paper would call attack success).
# Bars: rate with 95% interval (Wilson; Haiku's safe panel resamples prompts). Data: scratch e10/results.json.
# Usage: python3 diagrams/e10-graders.py public/diagrams/e10-graders, then render the PNG with sharp at density 144.
import sys

GRADERS = ["Prefix match", "Keyword anywhere", "XSTest judge", "Read by hand"]
SAFE = {  # model: [(rate, lo, hi)] in GRADERS order; "read" = every judge-flagged reply read, refusals confirmed
    "Claude Haiku 5.5": [(2.2, 0.8, 4.0), (5.8, 3.6, 8.2), (9.2, 6.4, 12.2), (0.8, 0.2, 2.9)],
    "Qwen2.5-1.5B-Instruct": [(48.4, 42.3, 54.6), (56.0, 49.8, 62.0), (47.6, 41.5, 53.8), (42.4, 36.4, 48.6)],
    "Gemma 2 2B IT": [(3.2, 1.6, 6.2), (12.0, 8.5, 16.6), (24.0, 19.1, 29.7), (11.6, 8.2, 16.2)],
}
UNSAFE_G = ["Prefix match", "Keyword anywhere", "XSTest judge", "StrongREJECT"]
UNSAFE = {  # share not refused (judge: full compliance; StrongREJECT: refusal item = 0)
    "Claude Haiku 5.5": [(67.0, 60.2, 73.1), (89.5, 84.5, 93.0), (4.5, 2.4, 8.3), (5.8, 3.3, 10.1)],
    "Qwen2.5-1.5B-Instruct": [(5.0, 2.7, 9.0), (2.5, 1.1, 5.7), (1.0, 0.3, 3.6), (1.5, 0.5, 4.3)],
    "Gemma 2 2B IT": [(53.5, 46.6, 60.3), (12.5, 8.6, 17.8), (3.5, 1.7, 7.0), (2.5, 1.1, 5.8)],
}
KINDS = {"safe": [0, 0, 1, 2], "unsafe": [0, 0, 1, 1]}  # substring rule, LLM judge, reading

W, H = 780, 462
PW = 330           # panel width for bars
LAB = 118          # label column inside each panel
PANELS = [(16, "Safe prompts: share refused", SAFE, GRADERS, "safe"), (406, "Unsafe prompts: share not refused", UNSAFE, UNSAFE_G, "unsafe")]
T = 44; BH = 15; GAP = 3; GROUP = 30

def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x0, title, data, names, kk in PANELS:
        L = x0 + LAB; R = x0 + PW + 10
        sx = lambda v: L + v / 100 * (R - L)
        o.append(f'<text x="{x0}" y="22" fill="{ink}" font-size="15" font-weight="bold">{title}</text>')
        y = T
        top = y
        at = len(o)
        for model, vals in data.items():
            o.append(f'<text x="{x0}" y="{y+12}" fill="{ink}" font-size="13" font-weight="bold">{model}</text>')
            y += 20
            for (v, lo, hi), name, k in zip(vals, names, KINDS[kk]):
                o.append(f'<text x="{L-8}" y="{y+BH/2+4:.1f}" fill="{ink}" font-size="12" text-anchor="end">{name}</text>')
                o.append(f'<rect x="{L:.1f}" y="{y:.1f}" width="{max(sx(v)-L, 1.5):.1f}" height="{BH}" fill="{cols[k]}" rx="2"/>')
                cy = y + BH / 2
                o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{ink}" stroke-width="1.6"/>')
                for e in (lo, hi):
                    o.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{cy-4:.1f}" y2="{cy+4:.1f}" stroke="{ink}" stroke-width="1.6"/>')
                o.append(f'<text x="{sx(hi)+5:.1f}" y="{cy+4:.1f}" fill="{ink}" font-size="12">{v:.1f}</text>')
                y += BH + GAP
            y += GROUP - GAP
        bottom = y - GROUP + 8
        for g in (0, 50, 100):
            o.insert(at, f'<line x1="{sx(g):.1f}" x2="{sx(g):.1f}" y1="{top-6}" y2="{bottom}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{sx(g):.1f}" y="{bottom+16}" fill="{mut}" font-size="12" text-anchor="middle">{g}%</text>')
    # legend
    ly = H - 22
    items = [("Substring rule", 0), ("Claude Haiku 5.5 as judge", 1), ("Read by hand", 2)]
    x = 16
    for name, k in items:
        o.append(f'<rect x="{x}" y="{ly-11}" width="14" height="14" fill="{cols[k]}" rx="2"/>')
        o.append(f'<text x="{x+20}" y="{ly}" fill="{ink}" font-size="13">{name}</text>')
        x += 20 + len(name) * 7.4 + 26
    o.append('</svg>'); return "\n".join(o)

light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#0b5fd6', '#4a5b78')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#38bdf8', '#9fb0cf')))
