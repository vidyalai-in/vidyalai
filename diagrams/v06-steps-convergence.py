# Generates public/diagrams/v06-steps-convergence.svg and -dark.svg (data: v06-latent-diffusion.md).
# Stable Diffusion 1.5 (fp16, guidance 7.5, 512 x 512), 16 own prompts x 1 seed. For each sampler and step count:
# PSNR of the image against the same prompt and seed sampled with 100 DPM-Solver++ steps (the reference).
# Bars: 95% bootstrap intervals over prompts. Data from scratch v06/analyze.py -> runs/results.json.
# Usage: python3 diagrams/v06-steps-convergence.py public/diagrams/v06-steps-convergence
import sys, math
DATA = {"ddim_4": {"sampler": "ddim", "steps": 4, "psnr": [12.21, 11.38, 13.01]}, "ddim_8": {"sampler": "ddim", "steps": 8, "psnr": [13.52, 12.42, 14.6]}, "ddim_15": {"sampler": "ddim", "steps": 15, "psnr": [14.84, 13.35, 16.33]}, "ddim_30": {"sampler": "ddim", "steps": 30, "psnr": [16.42, 14.45, 18.19]}, "euler_4": {"sampler": "euler", "steps": 4, "psnr": [12.49, 11.86, 13.17]}, "euler_8": {"sampler": "euler", "steps": 8, "psnr": [14.91, 13.83, 15.94]}, "euler_15": {"sampler": "euler", "steps": 15, "psnr": [17.98, 16.43, 19.36]}, "euler_30": {"sampler": "euler", "steps": 30, "psnr": [20.46, 18.36, 22.35]}, "dpmpp2m_4": {"sampler": "dpmpp2m", "steps": 4, "psnr": [13.48, 12.54, 14.42]}, "dpmpp2m_8": {"sampler": "dpmpp2m", "steps": 8, "psnr": [15.85, 14.38, 17.29]}, "dpmpp2m_15": {"sampler": "dpmpp2m", "steps": 15, "psnr": [18.79, 17.01, 20.41]}, "dpmpp2m_30": {"sampler": "dpmpp2m", "steps": 30, "psnr": [23.14, 20.7, 25.58]}}
NAMES = [("dpmpp2m", "DPM-Solver++ 2M"), ("ddim", "DDIM"), ("euler", "Euler")]
W, H = 780, 380
L, R, T, B = 70, 190, 30, 60
X0, X1 = 3.4, 36
def svg(th, cols):
    ink, mut, grid, surf = th
    ys = [v for k in DATA for v in DATA[k]["psnr"]]
    Y0, Y1 = math.floor(min(ys) / 5) * 5, math.ceil(max(ys) / 5) * 5
    sx = lambda x: L + (math.log(x) - math.log(X0)) / (math.log(X1) - math.log(X0)) * (W - L - R)
    sy = lambda y: T + (Y1 - y) / (Y1 - Y0) * (H - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(Y0, Y1 + 1, 5):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}</text>')
    for x in (4, 8, 15, 30):
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">sampling steps (log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">PSNR against the 100-step image (dB)</text>')
    for k, (smp, name) in enumerate(NAMES):
        pts = sorted((v["steps"], v["psnr"]) for v in DATA.values() if v["sampler"] == smp)
        off = (k - 1) * 4
        o.append(f'<polyline fill="none" stroke="{cols[k]}" stroke-width="2.4" points="{" ".join(f"{sx(s)+off:.1f},{sy(p[0]):.1f}" for s, p in pts)}"/>')
        for s, p in pts:
            X = sx(s) + off
            o.append(f'<line x1="{X:.1f}" x2="{X:.1f}" y1="{sy(p[1]):.1f}" y2="{sy(p[2]):.1f}" stroke="{cols[k]}" stroke-width="1.6"/>')
            o.append(f'<circle cx="{X:.1f}" cy="{sy(p[0]):.1f}" r="4.2" fill="{cols[k]}" stroke="{surf}" stroke-width="1.2"/>')
        y = T + 20 + k * 28
        o.append(f'<line x1="{W-R+22}" x2="{W-R+50}" y1="{y}" y2="{y}" stroke="{cols[k]}" stroke-width="2.6"/>')
        o.append(f'<circle cx="{W-R+36}" cy="{y}" r="4.2" fill="{cols[k]}"/>')
        o.append(f'<text x="{W-R+58}" y="{y+5}" fill="{ink}" font-size="13">{name}</text>')
    o.append(f'<text x="{W-R+22}" y="{T+120}" fill="{mut}" font-size="12">16 prompts, 1 seed each;</text>')
    o.append(f'<text x="{W-R+22}" y="{T+136}" fill="{mut}" font-size="12">guidance 7.5. Bars: 95%</text>')
    o.append(f'<text x="{W-R+22}" y="{T+152}" fill="{mut}" font-size="12">bootstrap over prompts.</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#1f9e6e', '#eb6834')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#2fb380', '#d95926')))
