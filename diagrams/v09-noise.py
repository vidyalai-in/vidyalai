# Generates public/diagrams/v09-noise.svg and -dark.svg (data: v09-speech-recognition-whisper.md).
# WER (Whisper English normalizer) on the first 200 utterances of the test-clean subset with white noise or 6-talker babble
# added at a fixed signal-to-noise ratio (powers over the whole clip). Same seeded noise for both models.
# Bars: 95% bootstrap intervals over utterances. Data from scratch v09/analyze.py -> runs/results.json.
# Usage: python3 diagrams/v09-noise.py public/diagrams/v09-noise
import sys
DATA = {"tiny": {"white": [["clean", 7.93, 6.68, 9.25], [20, 9.18, 7.99, 10.39], [10, 15.57, 13.52, 17.57], [5, 31.94, 25.12, 43.38], [0, 73.71, 55.08, 100.0]], "babble": [["clean", 7.93, 6.68, 9.25], [20, 9.13, 7.97, 10.45], [10, 15.27, 13.65, 16.85], [5, 38.95, 35.54, 42.4], [0, 157.5, 113.08, 211.41]]}, "turbo": {"white": [["clean", 1.96, 1.43, 2.48], [20, 2.1, 1.56, 2.62], [10, 2.95, 2.32, 3.64], [5, 4.61, 3.71, 5.62], [0, 11.58, 9.92, 13.44]], "babble": [["clean", 1.96, 1.43, 2.48], [20, 2.08, 1.58, 2.59], [10, 2.79, 2.18, 3.44], [5, 6.57, 5.49, 7.75], [0, 32.08, 23.35, 47.24]]}}
LABEL = {"tiny": "tiny", "base": "base", "small": "small", "turbo": "large-v3-turbo"}
XS = ["clean", 20, 10, 5, 0]
W, H = 780, 400
L, R, T, B = 70, 210, 30, 60


def svg(th, cols):
    ink, mut, grid, surf = th
    import math
    Y0, Y1 = 1, 300  # log scale: WER past 100% means more inserted words than reference words
    sy = lambda y: T + (math.log10(Y1) - math.log10(max(y, Y0))) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
    sx = lambda i: L + 20 + i * (W - L - R - 40) / (len(XS) - 1)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in (1, 2, 5, 10, 20, 50, 100, 200):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    for i, x in enumerate(XS):
        o.append(f'<text x="{sx(i):.1f}" y="{H-B+20}" fill="{mut}" font-size="13" text-anchor="middle">{"clean" if x == "clean" else f"{x} dB"}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-14}" fill="{ink}" font-size="14" text-anchor="middle">signal-to-noise ratio (louder noise to the right)</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">word error rate (log scale)</text>')
    k = 0
    legend = []
    for mi, m in enumerate(DATA):
        for ni, noise in enumerate(("white", "babble")):
            pts = DATA[m][noise]
            off = (k - 1.5) * 4
            dash = '' if noise == "babble" else ' stroke-dasharray="6 4"'
            o.append(f'<polyline fill="none" stroke="{cols[mi]}" stroke-width="2.4"{dash} points="{" ".join(f"{sx(XS.index(p[0]))+off:.1f},{sy(p[1]):.1f}" for p in pts)}"/>')
            for p in pts:
                X = sx(XS.index(p[0])) + off
                o.append(f'<line x1="{X:.1f}" x2="{X:.1f}" y1="{sy(p[2]):.1f}" y2="{sy(p[3]):.1f}" stroke="{cols[mi]}" stroke-width="1.6"/>')
                fill = cols[mi] if noise == "babble" else surf
                o.append(f'<circle cx="{X:.1f}" cy="{sy(p[1]):.1f}" r="4.2" fill="{fill}" stroke="{cols[mi]}" stroke-width="1.6"/>')
            legend.append((cols[mi], dash, noise == "babble", f"{LABEL[m]}, {noise}"))
            k += 1
    for j, (c, dash, filled, name) in enumerate(legend):
        y = T + 20 + j * 26
        o.append(f'<line x1="{W-R+22}" x2="{W-R+50}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.4"{dash}/>')
        o.append(f'<circle cx="{W-R+36}" cy="{y}" r="4.2" fill="{c if filled else surf}" stroke="{c}" stroke-width="1.6"/>')
        o.append(f'<text x="{W-R+58}" y="{y+5}" fill="{ink}" font-size="13">{name}</text>')
    for i, line in enumerate(("200 test-clean utterances;", "bars: 95% bootstrap", "over utterances.", "Babble: 6 talkers summed.", "Above 100%: more errors", "than words spoken.")):
        o.append(f'<text x="{W-R+22}" y="{T+140+i*16}" fill="{mut}" font-size="12">{line}</text>')
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834', '#1f9e6e')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926', '#2fb380')))
