# Generates public/diagrams/v13-vlm-stages.svg and -dark.svg (data: v13-multimodal-inference.md).
# Where the time goes in one local VLM request on an RTX 4060 Laptop GPU (bf16, transformers 5.21):
# processor (CPU, incl. copy to GPU), vision encoder + connector, language-model prefill given the image
# features, and 32 decoded tokens (decode ms per token x 32). Medians of 7 runs after 2 warm-ups.
# The same synthetic report page at three sizes for Qwen2-VL-2B; 1344 x 896 for SmolVLM-256M at its
# default processor setting and with longest_edge 512.
# Data from scratch v13/scripts/s05_local_summary.py -> runs/results_local.json ("stages").
# Usage: python3 diagrams/v13-vlm-stages.py public/diagrams/v13-vlm-stages
import sys

ROWS = [
    {"label": "Qwen2-VL-2B, 336 × 224|96 image tokens", "proc": 2.5, "enc": 46.3, "pre": 41.3, "dec": 24.4},
    {"label": "Qwen2-VL-2B, 672 × 448|384 image tokens", "proc": 4.2, "enc": 156.4, "pre": 77.1, "dec": 25.3},
    {"label": "Qwen2-VL-2B, 1344 × 896|1,536 image tokens", "proc": 12.7, "enc": 884.4, "pre": 245.5, "dec": 27.1},
    {"label": "SmolVLM-256M, default|832 image tokens", "proc": 301.0, "enc": 172.5, "pre": 25.1, "dec": 25.1},
    {"label": "SmolVLM-256M, edge 512|64 image tokens", "proc": 22.2, "enc": 24.9, "pre": 29.5, "dec": 25.5},
]
NDEC = 32
W, H = 780, 400
L, R, T, B = 230, 30, 30, 92
X1 = 2700
STAGES = [("proc", "processor (CPU)"), ("enc", "vision encoder"), ("pre", "LM prefill"), ("dec", "32 decoded tokens")]


def svg(th, cols):
    ink, mut, grid, surf = th
    sx = lambda x: L + x / X1 * (W - L - R)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for x in range(0, X1 + 1, 250):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        if x % 500 == 0:
            o.append(f'<text x="{sx(x):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{x/1000:g} s</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">seconds for one request: one image, one question, 32 output tokens</text>')
    bh = 30
    gap = (H - T - B - len(ROWS) * bh) / (len(ROWS) + 1)
    for i, r in enumerate(ROWS):
        y = T + gap * (i + 1) + bh * i
        lines = r["label"].split("|")
        for j, ln in enumerate(lines):
            o.append(f'<text x="{L-10}" y="{y + bh/2 + 4 + (j - (len(lines)-1)/2)*15:.1f}" fill="{ink if j == 0 else mut}" font-size="13" text-anchor="end">{ln}</text>')
        x0 = 0.0
        for k, (key, _) in enumerate(STAGES):
            v = r[key] * (NDEC if key == "dec" else 1)
            o.append(f'<rect x="{sx(x0):.1f}" y="{y:.1f}" width="{max(sx(x0+v)-sx(x0), 0.5):.1f}" height="{bh}" fill="{cols[k]}" stroke="{surf}" stroke-width="1"/>')
            x0 += v
            if key == "pre":
                ttft = x0
        o.append(f'<line x1="{sx(ttft):.1f}" x2="{sx(ttft):.1f}" y1="{y-4:.1f}" y2="{y+bh+4:.1f}" stroke="{ink}" stroke-width="2.5"/>')
        o.append(f'<text x="{sx(x0)+6:.1f}" y="{y+bh/2+4:.1f}" fill="{ink}" font-size="12">first token {ttft:,.0f} ms</text>')
    ly = H - 22
    x = 20
    for k, (_, label) in enumerate(STAGES):
        o.append(f'<rect x="{x}" y="{ly-11}" width="14" height="14" fill="{cols[k]}"/><text x="{x+20}" y="{ly}" fill="{ink}" font-size="13">{label}</text>')
        x += 40 + len(label) * 7.0
    o.append(f'<line x1="{x+4}" x2="{x+4}" y1="{ly-13}" y2="{ly+3}" stroke="{ink}" stroke-width="2.5"/><text x="{x+12}" y="{ly}" fill="{ink}" font-size="13">first token</text>')
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#9aa9c4', '#eb6834', '#2a78d6', '#1f9e74')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#5b6f94', '#d95926', '#3987e5', '#2bb38a')))
