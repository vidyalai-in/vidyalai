# Generates public/diagrams/v12-animatediff-cost.svg and -dark.svg (data: v12-video-generation.md).
# AnimateDiff (motion module mm_sd_v15_v2 on Stable Diffusion 1.5, fp16, 512 x 512) on an RTX 4060 Laptop GPU: time of one
# guided denoising step (U-Net on a batch of 2 latents: prompt and negative prompt) and peak allocated GPU memory, against the
# number of frames. Median of 5 timed steps after 2 warm-up; bars span min-max. 32 frames ran out of memory (95% cap of 8 GB).
# Data from scratch v12/exp3_animatediff.py --phase timing -> runs/exp3_timing_512.json.
# Usage: python3 diagrams/v12-animatediff-cost.py public/diagrams/v12-animatediff-cost
import sys, json
D = json.loads('[[1, 0.3157, 0.3148, 0.3201, 3.911], [4, 0.7694, 0.7551, 0.7756, 4.293], [8, 1.3539, 1.3365, 1.3748, 4.802], [16, 2.5152, 2.3982, 2.5836, 5.82], [24, 3.8522, 3.7429, 3.9017, 6.837]]')   # frames, median s, min s, max s, peak GiB
W, H = 780, 400; T, B = 40, 62
P = [(70, 370), (450, 750)]          # x ranges of the two panels
XMAX = 34
def sx(i, f): a, b = P[i]; return a + f / XMAX * (b - a)
def sy(v, vmax): return H - B - v / vmax * (H - T - B)
def svg(th, c):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for i, (title, vmax, ticks, unit) in enumerate((("Seconds per denoising step", 4.5, (0, 1, 2, 3, 4), "s"),
                                                    ("Peak GPU memory (GiB)", 8.5, (0, 2, 4, 6, 8), "GiB"))):
        a, b = P[i]
        o.append(f'<text x="{a}" y="{T-16}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        for y in ticks:
            o.append(f'<line x1="{a}" x2="{b}" y1="{sy(y, vmax):.1f}" y2="{sy(y, vmax):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{a-8}" y="{sy(y, vmax)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}</text>')
        for f in (1, 8, 16, 24, 32):
            o.append(f'<text x="{sx(i, f):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{f}</text>')
        o.append(f'<text x="{(a+b)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Frames per clip</text>')
        k = 1 if i == 0 else 4
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(i, r[0]):.1f},{sy(r[k], vmax):.1f}" for r in D)}"/>')
        for r in D:
            if i == 0:
                o.append(f'<line x1="{sx(i, r[0]):.1f}" x2="{sx(i, r[0]):.1f}" y1="{sy(r[2], vmax):.1f}" y2="{sy(r[3], vmax):.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(i, r[0]):.1f}" cy="{sy(r[k], vmax):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
            lab = f"{r[k]:.2f}" if i == 0 else f"{r[k]:.1f}"
            if i == 0:
                o.append(f'<text x="{sx(i, r[0])-6:.1f}" y="{sy(r[k], vmax)-10:.1f}" fill="{ink}" font-size="12" text-anchor="end">{lab}</text>')
            else:
                o.append(f'<text x="{sx(i, r[0])+4:.1f}" y="{sy(r[k], vmax)+20:.1f}" fill="{ink}" font-size="12">{lab}</text>')
        if i == 1:
            o.append(f'<line x1="{a}" x2="{b}" y1="{sy(8, vmax):.1f}" y2="{sy(8, vmax):.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/>')
            o.append(f'<text x="{a+4}" y="{sy(8, vmax)-7:.1f}" fill="{mut}" font-size="12">8 GB card</text>')
        o.append(f'<text x="{sx(i, 32):.1f}" y="{sy(0.6 if i == 0 else 1.2, vmax):.1f}" fill="{mut}" font-size="12" text-anchor="middle">32: out of</text>')
        o.append(f'<text x="{sx(i, 32):.1f}" y="{sy(0.6 if i == 0 else 1.2, vmax)+15:.1f}" fill="{mut}" font-size="12" text-anchor="middle">memory</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, '#2a78d6'))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, '#3987e5'))
