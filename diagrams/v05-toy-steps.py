# Generates public/diagrams/v05-toy-steps.svg and -dark.svg (data: v05-diffusion-math.md).
# 2D spiral, MLP denoiser (4 layers of 256, 215k parameters), linear schedule T = 1000, 20,000 steps of batch 1,024,
# 3 training seeds. For each seed and sampler: 5,000 samples at 1-1,000 sampling steps (evenly respaced).
# Left: sliced Wasserstein distance to 5,000 fresh data points (500 random projections). Right: share of samples within
# 0.1 of the noise-free spiral curve. Dots: mean of 3 seeds; bars: min to max. Dashed: real data against real data.
# Data from scratch v05/toy.py -> runs/toy_results.json. Usage: python3 diagrams/v05-toy-steps.py public/diagrams/v05-toy-steps
import sys, json, math
D = json.loads('{"swd":{"ddpm":[[0.6376,0.596,0.678],[0.6217,0.5805,0.6599],[0.3223,0.3202,0.3242],[0.1744,0.1713,0.1766],[0.0884,0.0844,0.0925],[0.0367,0.0327,0.043],[0.0253,0.0222,0.0273],[0.0158,0.0151,0.0162],[0.0154,0.0125,0.0208]],"ddim":[[0.6376,0.596,0.678],[0.6206,0.5764,0.6619],[0.3078,0.3023,0.3106],[0.1268,0.1212,0.1359],[0.0582,0.0541,0.066],[0.0266,0.0218,0.0338],[0.0207,0.0157,0.028],[0.0182,0.0142,0.0244],[0.0178,0.0137,0.0235]]},"on":{"ddpm":[[0.1747,0.1724,0.1776],[0.1863,0.1826,0.1912],[0.2561,0.2548,0.258],[0.3458,0.3422,0.349],[0.7943,0.7916,0.7984],[0.9463,0.9394,0.951],[0.9673,0.9646,0.9706],[0.9677,0.9666,0.9698],[0.9577,0.9566,0.959]],"ddim":[[0.1747,0.1724,0.1776],[0.1869,0.1848,0.1894],[0.2722,0.2548,0.2882],[0.3579,0.3554,0.3598],[0.8053,0.799,0.8102],[0.9219,0.9204,0.9236],[0.9368,0.933,0.9394],[0.9373,0.937,0.9378],[0.9366,0.9344,0.9384]]},"floor":{"swd":0.017118249274790286,"on":0.9602400064468384}}')
W, H = 780, 440; PW = 296; L1, L2, T, B = 62, 62 + 296 + 98, 30, 92
S = [1, 2, 5, 10, 20, 50, 100, 250, 1000]
def sx(x, L): return L + (math.log10(x)) / 3 * PW
def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    panels = [("swd", L1, "Distance to the data (log scale, lower is better)", lambda y: H - B - (math.log10(y) + 2) / (math.log10(1) + 2) * (H - T - B), [0.01, 0.03, 0.1, 0.3, 1], lambda v: f"{v:g}"),
              ("on", L2, "Samples on the spiral (higher is better)", lambda y: H - B - y * (H - T - B), [0, 0.25, 0.5, 0.75, 1], lambda v: f"{v*100:.0f}%")]
    for key, L, title, sy, ticks, fmt in panels:
        for y in ticks:
            o.append(f'<line x1="{L}" x2="{L+PW}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="12.5" text-anchor="end">{fmt(y)}</text>')
        for x in (1, 10, 100, 1000):
            o.append(f'<line x1="{sx(x,L):.1f}" x2="{sx(x,L):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}"/>')
            o.append(f'<text x="{sx(x,L):.1f}" y="{H-B+18}" fill="{mut}" font-size="12.5" text-anchor="middle">{x:,}</text>')
        o.append(f'<text x="{L+PW/2}" y="{H-B+40}" fill="{ink}" font-size="13.5" text-anchor="middle">Sampling steps (log scale)</text>')
        o.append(f'<text x="{L-4}" y="{T-12}" fill="{ink}" font-size="13.5" font-weight="bold">{title}</text>')
        f = D["floor"][key]
        o.append(f'<line x1="{L}" x2="{L+PW}" y1="{sy(f):.1f}" y2="{sy(f):.1f}" stroke="{mut}" stroke-width="2" stroke-dasharray="6 5"/>')
        for (m, c) in zip(("ddpm", "ddim"), cols):
            pts = D[key][m]
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(f"{sx(s,L):.1f},{sy(p[0]):.1f}" for s, p in zip(S, pts))}"/>')
            for s, (mu, lo, hi) in zip(S, pts):
                o.append(f'<line x1="{sx(s,L):.1f}" x2="{sx(s,L):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
                o.append(f'<circle cx="{sx(s,L):.1f}" cy="{sy(mu):.1f}" r="4.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
    y = H - 22; x = L1
    for lab, c, dash in (("DDPM (random steps)", cols[0], ""), ("DDIM (deterministic)", cols[1], ""), ("real data vs real data", mut, ' stroke-dasharray="6 5"')):
        o.append(f'<line x1="{x}" x2="{x+26}" y1="{y-4}" y2="{y-4}" stroke="{c}" stroke-width="2.5"{dash}/><text x="{x+34}" y="{y}" fill="{ink}" font-size="13">{lab}</text>')
        x += 225
    o.append('</svg>'); return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#eb6834', '#2a78d6')))
open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#d95926', '#3987e5')))
