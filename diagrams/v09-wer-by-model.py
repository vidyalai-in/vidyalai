# Generates public/diagrams/v09-wer-by-model.svg and -dark.svg (data: v09-speech-recognition-whisper.md).
# Whisper tiny, base, small and large-v3-turbo (fp16, greedy, English forced, batch 16) on 500 + 500 speaker-stratified
# LibriSpeech test-clean / test-other utterances, scored after Whisper's English text normalizer.
# Filled dots: this run, bars = 95% bootstrap intervals over utterances. Hollow dots: greedy WER the Whisper paper reports
# on the full test sets (Table 8). Speed label: real-time factor of the batched run (audio seconds per compute second).
# Data from scratch v09/analyze.py -> runs/results.json.
# Usage: python3 diagrams/v09-wer-by-model.py public/diagrams/v09-wer-by-model
import sys
DATA = {"tiny": {"rtf": 355.0, "wer": {"test-clean": [7.87, 6.98, 8.74], "test-other": [16.62, 15.21, 18.18]}}, "base": {"rtf": 281.3, "wer": {"test-clean": [5.28, 4.62, 5.95], "test-other": [11.82, 10.68, 12.98]}}, "small": {"rtf": 124.9, "wer": {"test-clean": [3.48, 2.92, 4.12], "test-other": [7.7, 6.77, 8.71]}}, "turbo": {"rtf": 49.8, "wer": {"test-clean": [2.16, 1.7, 2.7], "test-other": [3.71, 3.16, 4.28]}}}
PAPER = {"tiny": (7.6, 16.9), "base": (5.0, 12.4), "small": (3.4, 7.6)}
NAMES = [("tiny", "tiny, 39M"), ("base", "base, 74M"), ("small", "small, 244M"), ("turbo", "large-v3-turbo, 809M")]
W, H = 780, 420
L, R, T, B = 70, 200, 30, 86


def svg(th, cols):
    ink, mut, grid, surf = th
    Y0, Y1 = 0, max(20, max(v[2] for d in DATA.values() for v in d["wer"].values()) * 1.08)
    Y1 = int(-(-Y1 // 5) * 5)
    sy = lambda y: T + (Y1 - y) / (Y1 - Y0) * (H - T - B)
    cw = (W - L - R) / len(NAMES)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    for y in range(Y0, Y1 + 1, 5):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="13" text-anchor="end">{y}%</text>')
    o.append(f'<text transform="translate(20,{(T+H-B)/2}) rotate(-90)" fill="{ink}" font-size="14" text-anchor="middle">word error rate (normalized)</text>')
    for k, (m, label) in enumerate(NAMES):
        cx = L + cw * (k + 0.5)
        o.append(f'<text x="{cx:.1f}" y="{H-B+22}" fill="{ink}" font-size="13" text-anchor="middle">{label}</text>')
        o.append(f'<text x="{cx:.1f}" y="{H-B+40}" fill="{mut}" font-size="12" text-anchor="middle">{DATA[m]["rtf"]:.0f}x real time</text>')
        for j, sp in enumerate(("test-clean", "test-other")):
            x = cx + (j - 0.5) * 26
            w, lo, hi = DATA[m]["wer"][sp]
            o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{cols[j]}" stroke-width="2"/>')
            o.append(f'<circle cx="{x:.1f}" cy="{sy(w):.1f}" r="5" fill="{cols[j]}" stroke="{surf}" stroke-width="1.2"/>')
            o.append(f'<text x="{x + (-9 if j == 0 else 9):.1f}" y="{sy(w)+4:.1f}" fill="{ink}" font-size="12" text-anchor="{"end" if j == 0 else "start"}">{w:.1f}</text>')
            if m in PAPER:
                o.append(f'<circle cx="{x:.1f}" cy="{sy(PAPER[m][j]):.1f}" r="5" fill="none" stroke="{cols[j]}" stroke-width="1.6"/>')
    lx = W - R + 22
    for j, name in enumerate(("test-clean", "test-other")):
        y = T + 20 + j * 26
        o.append(f'<circle cx="{lx+8}" cy="{y}" r="5" fill="{cols[j]}"/>')
        o.append(f'<text x="{lx+22}" y="{y+5}" fill="{ink}" font-size="13">{name}, this run</text>')
    y = T + 20 + 2 * 26
    o.append(f'<circle cx="{lx+8}" cy="{y}" r="5" fill="none" stroke="{mut}" stroke-width="1.6"/>')
    o.append(f'<text x="{lx+22}" y="{y+5}" fill="{ink}" font-size="13">paper, full test set</text>')
    for i, line in enumerate(("500 utterances per set;", "bars: 95% bootstrap", "over utterances. Greedy", "decoding, batch 16, fp16,", "RTX 4060 Laptop GPU.")):
        o.append(f'<text x="{lx}" y="{T+120+i*16}" fill="{mut}" font-size="12">{line}</text>')
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926')))
