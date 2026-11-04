# Generates public/diagrams/v12-attention-cost.svg and -dark.svg (data: v12-video-generation.md).
# Time of one self-attention call (8 heads x 64 dims, bf16, PyTorch 2.11 SDPA) on an RTX 4060 Laptop GPU against the number of
# video tokens, for full spacetime attention (one sequence of all tokens), factorized attention (spatial within each latent
# frame, then temporal across latent frames) and the textbook "math" kernel that stores the whole score matrix. Token counts
# assume a 4x8x8 causal VAE and 1x2x2 patches. Median of 7 timed calls after 3 warm-up (5 after 2 for math); bars span min-max.
# Data from scratch v12/exp1_attention.py -> runs/exp1_attention.json. Usage: python3 diagrams/v12-attention-cost.py public/diagrams/v12-attention-cost
import sys, json, math
D = json.loads('{"full": [[1280, 0.31, 0.276, 0.332], [9216, 9.784, 9.674, 9.86], [17550, 26.719, 26.545, 27.005], [32760, 97.969, 97.88, 100.23], [75600, 490.608, 485.978, 506.173], [118800, 1224.488, 1207.97, 1283.026]], "factorized": [[1280, 0.627, 0.612, 0.638], [9216, 3.237, 2.838, 3.333], [17550, 4.518, 4.478, 4.606], [32760, 8.901, 8.004, 9.111], [75600, 31.894, 31.561, 32.047], [118800, 49.338, 48.872, 50.164]], "full-math": [[1280, 3.018, 2.931, 3.085], [9216, 133.769, 132.563, 136.187], [17550, 4296.098, 4252.334, 4375.361]]}')
NAMES = {"full": "Full spacetime attention (fused SDPA kernel)", "factorized": "Factorized: space, then time",
         "full-math": "Full, textbook kernel (stores every score)"}
W, H = 780, 480; L, R, T, B = 78, 30, 24, 62
X0, X1 = 800, 200000; Y0, Y1 = 0.1, 10000
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (0.1, 1, 10, 100, 1000, 10000):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y:,g}</text>')
    for x, lab in ((1000, "1k"), (3000, "3k"), (10000, "10k"), (30000, "30k"), (100000, "100k")):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{lab}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">Video tokens in the clip (log scale)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Milliseconds per attention call (log scale)</text>')
    # quadratic reference through the full-attention point at 75,600 tokens
    n0, t0 = 75600, [p for p in D["full"] if p[0] == 75600][0][1]
    xs = [1000, 200000]
    o.append(f'<polyline fill="none" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5" points="{" ".join(f"{sx(x):.1f},{sy(t0*(x/n0)**2):.1f}" for x in xs)}"/>')
    o.append(f'<text x="{sx(2300):.1f}" y="{sy(t0*(2300/n0)**2)+20:.1f}" fill="{mut}" font-size="13">tokens squared</text>')
    for (m, c) in zip(("full", "factorized", "full-math"), cols):
        pts = D[m]
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)}"/>')
        for n, mu, lo, hi in pts:
            o.append(f'<line x1="{sx(n):.1f}" x2="{sx(n):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
            o.append(f'<circle cx="{sx(n):.1f}" cy="{sy(mu):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
    lab = {"full": (75600, -12, "491 ms"), "factorized": (75600, 22, "32 ms")}
    for m, (n, dy, txt) in lab.items():
        mu = [p for p in D[m] if p[0] == n][0][1]
        o.append(f'<text x="{sx(n)+8:.1f}" y="{sy(mu)+dy:.1f}" fill="{ink}" font-size="13">{txt}</text>')
    mm = D["full-math"][-1]
    o.append(f'<text x="{sx(mm[0])+10:.1f}" y="{sy(mm[1])+5:.1f}" fill="{ink}" font-size="13">4.3 s and 21 GiB: spilled past the 8 GB card</text>')
    lx, ly = sx(6500), sy(0.62)
    for k, (m, c) in enumerate(zip(("full", "factorized", "full-math"), cols)):
        y = ly + k * 22
        o.append(f'<line x1="{lx}" x2="{lx+26}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"/><text x="{lx+34}" y="{y+5}" fill="{ink}" font-size="13">{NAMES[m]}</text>')
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#1f9e74', '#eb6834')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#2bb38a', '#d95926')))
