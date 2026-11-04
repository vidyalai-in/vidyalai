# Generates public/diagrams/v13-cache-ttft.svg and -dark.svg (data: v13-multimodal-inference.md).
# Time to first token on claude-haiku-5-5 (thinking disabled, streamed) for one report page at 1200 x 800 (S),
# at 2352 x 1568 (L) and eight pages at 2352 x 1568 (XL), with no cache_control (plain), with a cache write
# on every call (write) and with a cache hit (hit). 40 calls per arm and size, interleaved in rounds.
# Filled dot: median, bar: its 95% bootstrap interval; hollow diamond: 90th percentile.
# Data from scratch v13/scripts/s04_analyze.py -> runs/results_cache.json.
# Usage: python3 diagrams/v13-cache-ttft.py public/diagrams/v13-cache-ttft
import sys

DATA = {"xmax": 4.5, "S": {"plain": {"median": [1.239, 1.156, 1.306], "p90": 1.552}, "write": {"median": [1.259, 1.15, 1.346], "p90": 1.658}, "hit": {"median": [1.127, 1.077, 1.186], "p90": 1.371}}, "L": {"plain": {"median": [1.381, 1.215, 1.518], "p90": 1.696}, "write": {"median": [1.474, 1.381, 1.596], "p90": 2.524}, "hit": {"median": [1.468, 1.356, 1.578], "p90": 1.811}}, "XL": {"plain": {"median": [2.682, 1.882, 3.037], "p90": 4.18}, "write": {"median": [2.761, 2.498, 2.975], "p90": 3.774}, "hit": {"median": [1.926, 1.85, 2.063], "p90": 3.103}}}
W, H = 780, 430
L, R, T, B = 250, 40, 30, 70
ARMS = [("plain", "no cache_control"), ("write", "cache write (miss)"), ("hit", "cache hit")]
SIZES = [("S", "1 page, 1200 × 800|1,274-token prefix"), ("L", "1 page, 2352 × 1568|4,731-token prefix"),
         ("XL", "8 pages, 2352 × 1568|37,680-token prefix")]


def svg(th, cols):
    ink, mut, grid, surf = th
    X1 = DATA["xmax"]
    sx = lambda x: L + x / X1 * (W - L - R)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    x = 0.0
    while x <= X1 + 1e-9:
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{x:g} s</text>')
        x += 0.5
    o.append(f'<text x="{(L+W-R)/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">time to first token, seconds (40 calls per row)</text>')
    rowh = (H - T - B) / 12
    y = T + rowh * 0.5
    for si, (key, label) in enumerate(SIZES):
        lines = label.split("|")
        y_mid = y + rowh
        for j, ln in enumerate(lines):
            o.append(f'<text x="12" y="{y_mid + 4 + (j - 0.5) * 16:.1f}" fill="{ink if j == 0 else mut}" font-size="13">{ln}</text>')
        for ai, (arm, alab) in enumerate(ARMS):
            d = DATA[key][arm]
            yy = y + rowh * ai
            o.append(f'<text x="{L-10}" y="{yy+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{alab}</text>')
            m, lo, hi = d["median"]
            o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{cols[ai]}" stroke-width="3"/>')
            o.append(f'<circle cx="{sx(m):.1f}" cy="{yy:.1f}" r="6" fill="{cols[ai]}" stroke="{surf}" stroke-width="1.5"/>')
            p = d["p90"]
            o.append(f'<polygon points="{sx(p):.1f},{yy-6:.1f} {sx(p)+6:.1f},{yy:.1f} {sx(p):.1f},{yy+6:.1f} {sx(p)-6:.1f},{yy:.1f}" fill="none" stroke="{cols[ai]}" stroke-width="2"/>')
            o.append(f'<text x="{sx(max(hi, p))+12:.1f}" y="{yy+4:.1f}" fill="{ink}" font-size="12">{m:.2f} s</text>')
        y += rowh * 4
        if si < 2:
            o.append(f'<line x1="12" x2="{W-R}" y1="{y - rowh*1.0:.1f}" y2="{y - rowh*1.0:.1f}" stroke="{grid}" stroke-width="1"/>')
    ly = H - 14
    o.append(f'<circle cx="{L}" cy="{ly-4}" r="6" fill="{ink}"/><text x="{L+12}" y="{ly}" fill="{ink}" font-size="12">median with 95% interval</text>')
    o.append(f'<polygon points="{L+190},{ly-10} {L+196},{ly-4} {L+190},{ly+2} {L+184},{ly-4}" fill="none" stroke="{ink}" stroke-width="2"/><text x="{L+202}" y="{ly}" fill="{ink}" font-size="12">90th percentile</text>')
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#9aa9c4', '#eb6834', '#2a78d6')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#7f93b8', '#d95926', '#3987e5')))
