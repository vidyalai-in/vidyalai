# Generates public/diagrams/v08-codebook-usage.svg and -dark.svg (data: v08-autoregressive-image-generation.md).
# Codebook entries used on the 10,000 Fashion-MNIST test images by a small VQ-VAE (7 x 7 code grid, 3,000 steps,
# batch 256, Adam 1e-3, one seed each), with and without dead-code restarts every 200 steps.
# Data from scratch v08/runs/vq_sweep.jsonl.
# Usage: python3 diagrams/v08-codebook-usage.py public/diagrams/v08-codebook-usage
import math, sys
KS = [16, 64, 512, 4096]
USED = {"plain": [16, 64, 123, 132], "restart": [16, 64, 505, 4071]}
W, H = 780, 440
L, R, T, B = 86, 210, 30, 66
X0, X1, Y0, Y1 = 10, 6000, 10, 6000
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (10, 100, 1000):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v:,}</text>')
    for k in KS:
        o.append(f'<line x1="{sx(k):.1f}" x2="{sx(k):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(k):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{k:,}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Codebook size K (log scale)</text>')
    o.append(f'<text transform="translate(22,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Entries used on test images (log)</text>')
    # every entry used
    o.append(f'<line x1="{sx(12):.1f}" y1="{sy(12):.1f}" x2="{sx(5000):.1f}" y2="{sy(5000):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<line x1="{W-R+24}" x2="{W-R+52}" y1="{T+70}" y2="{T+70}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{W-R+60}" y="{T+75}" fill="{mut}" font-size="13">every entry used</text>')
    labels = {"plain": "plain VQ loss", "restart": "dead-code restarts"}
    for i, key in enumerate(("plain", "restart")):
        c = cols[i]
        pts = " ".join(f"{sx(k):.1f},{sy(u):.1f}" for k, u in zip(KS, USED[key]))
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{pts}"/>')
        for k, u in zip(KS, USED[key]):
            o.append(f'<circle cx="{sx(k):.1f}" cy="{sy(u):.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        ly = T + 20 + 25 * i
        o.append(f'<line x1="{W-R+24}" x2="{W-R+52}" y1="{ly-5}" y2="{ly-5}" stroke="{c}" stroke-width="2.5"/>')
        o.append(f'<text x="{W-R+60}" y="{ly:.1f}" fill="{ink}" font-size="13">{labels[key]}</text>')
    o.append(f'<text x="{sx(512)+8:.1f}" y="{sy(123)+18:.1f}" fill="{mut}" font-size="12">123 of 512</text>')
    o.append(f'<text x="{sx(4096)-8:.1f}" y="{sy(132)+20:.1f}" fill="{mut}" font-size="12" text-anchor="end">132 of 4,096</text>')
    o.append('</svg>')
    return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5')))
