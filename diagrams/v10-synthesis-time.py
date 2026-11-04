# Generates public/diagrams/v10-synthesis-time.svg and -dark.svg (data: v10-text-to-speech.md).
# Kokoro-82M (fp32, voice af_heart, speed 1.0) wall time to synthesize passages of 5 to 320 words, against the
# length of the audio produced, on the RTX 4060 Laptop GPU and on the CPU (PyTorch, 8 threads). Each point is the
# median of 7 (GPU) or 5 (CPU) timed runs after 2 warm-up runs, including text-to-phoneme time; bars are min-max.
# The dashed diagonal is real time (compute time = audio length). Data: scratch v10/runs/timing_cuda.json, timing_cpu.json.
# Usage: python3 diagrams/v10-synthesis-time.py public/diagrams/v10-synthesis-time
import math, sys
GPU = [(5, 2.1, 0.0788, 0.0659, 0.0895), (10, 3.425, 0.0842, 0.0791, 0.0857), (20, 7.425, 0.1331, 0.128, 0.1448),
       (40, 14.75, 0.2792, 0.2734, 0.2853), (80, 26.675, 0.5104, 0.5, 0.531), (160, 51.8, 1.0127, 0.9809, 1.0323),
       (320, 101.9, 1.9844, 1.9561, 2.0214)]
CPU = [(5, 2.1, 0.548, 0.5058, 0.5864), (10, 3.425, 0.7762, 0.7726, 0.8149), (20, 7.425, 1.7878, 1.6338, 1.8073),
       (40, 14.75, 4.3804, 4.3137, 4.5042), (80, 26.675, 7.6755, 7.5596, 8.2129), (160, 51.8, 16.1576, 15.2515, 16.8716)]
W, H = 780, 470
L, R, T, B = 80, 30, 26, 66
X0, X1, Y0, Y1 = 1.5, 150, 0.03, 200
def sx(x): return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - L - R)
def sy(y): return H - B - (math.log10(y) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
def svg(th, cols):
    ink, mut, grid, surf = th
    cg, cc = cols
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for v in (0.1, 1, 10, 100):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{v:g} s</text>')
    for x in (2, 5, 10, 20, 50, 100):
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{x} s</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-16}" fill="{ink}" font-size="14" text-anchor="middle">Length of the speech produced (log scale)</text>')
    o.append(f'<text transform="translate(22,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">Time to synthesize it (log scale)</text>')
    # real-time diagonal
    o.append(f'<line x1="{sx(X0):.1f}" y1="{sy(X0):.1f}" x2="{sx(X1):.1f}" y2="{sy(X1):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{sx(60):.1f}" y="{sy(60)-10:.1f}" fill="{mut}" font-size="13" text-anchor="end">real time</text>')
    for data, c, lab, shape in ((CPU, cc, "CPU, 8 threads: 3 to 4 times faster than real time", "sq"), (GPU, cg, "RTX 4060 Laptop GPU: about 51 times faster", "ci")):
        pts = " ".join(f"{sx(a):.1f},{sy(t):.1f}" for _, a, t, lo, hi in data)
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{pts}"/>')
        for _, a, t, lo, hi in data:
            o.append(f'<line x1="{sx(a):.1f}" x2="{sx(a):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{c}" stroke-width="2"/>')
            if shape == "ci":
                o.append(f'<circle cx="{sx(a):.1f}" cy="{sy(t):.1f}" r="5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
            else:
                o.append(f'<rect x="{sx(a)-5:.1f}" y="{sy(t)-5:.1f}" width="10" height="10" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        ly = sy(110) if shape == "sq" else sy(60)
        if shape == "ci":
            o.append(f'<circle cx="{sx(1.85):.1f}" cy="{ly-4:.1f}" r="5" fill="{c}"/>')
        else:
            o.append(f'<rect x="{sx(1.85)-5:.1f}" y="{ly-9:.1f}" width="10" height="10" fill="{c}"/>')
        o.append(f'<text x="{sx(1.85)+12:.1f}" y="{ly:.1f}" fill="{c}" font-size="13" font-weight="bold">{lab}</text>')
    _, a, t, _, _ = GPU[-1]
    o.append(f'<text x="{sx(a)-10:.1f}" y="{sy(t)-12:.1f}" fill="{ink}" font-size="12" text-anchor="end">320 words: 101.9 s of speech in 1.98 s</text>')
    _, a, t, _, _ = GPU[0]
    o.append(f'<text x="{sx(a)+8:.1f}" y="{sy(t)+22:.1f}" fill="{ink}" font-size="12">5 words: 2.1 s in 79 ms</text>')
    o.append('</svg>')
    return "\n".join(o)
light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926')))
