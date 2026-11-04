# Generates public/diagrams/v11-codec-quality.svg and -dark.svg (data: v11-voice-agents.md).
# Mimi (kyutai/mimi, 12.5 frames/s, 11-bit codes) and DAC 24 kHz (descript/dac_24khz, 75 frames/s, 10-bit codes)
# round trips at 1-32 codebooks on 128 Kokoro-82M stock-voice clips (32 sentences x 4 voices, 24 kHz).
# Left: Whisper large-v3-turbo word error rate after the round trip (normalized; bars = 95% bootstrap over clips).
# Right: STOI against the original (mean; bars = 95% bootstrap). Dashed: the original audio.
# Data from scratch v11/scripts/s09_analyze.py -> runs/results.json (codec_kokoro).
# Usage: python3 diagrams/v11-codec-quality.py public/diagrams/v11-codec-quality
import math
import sys

DATA = {"original_wer": 0.32, "mimi": [{"nq": 1, "kbps": 0.1375, "wer": [39.13, 34.69, 43.62], "stoi": [0.475, 0.47, 0.48]}, {"nq": 2, "kbps": 0.275, "wer": [2.81, 2.06, 3.6], "stoi": [0.783, 0.78, 0.786]}, {"nq": 4, "kbps": 0.55, "wer": [1.09, 0.57, 1.73], "stoi": [0.865, 0.861, 0.868]}, {"nq": 8, "kbps": 1.1, "wer": [0.45, 0.13, 0.82], "stoi": [0.919, 0.916, 0.921]}, {"nq": 16, "kbps": 2.2, "wer": [0.45, 0.13, 0.82], "stoi": [0.952, 0.951, 0.954]}, {"nq": 32, "kbps": 4.4, "wer": [0.45, 0.13, 0.82], "stoi": [0.972, 0.971, 0.973]}], "dac": [{"nq": 1, "kbps": 0.75, "wer": [6.71, 5.08, 8.64], "stoi": [0.773, 0.769, 0.778]}, {"nq": 2, "kbps": 1.5, "wer": [0.58, 0.2, 1.02], "stoi": [0.858, 0.854, 0.862]}, {"nq": 4, "kbps": 3.0, "wer": [0.45, 0.13, 0.82], "stoi": [0.919, 0.917, 0.922]}, {"nq": 8, "kbps": 6.0, "wer": [0.38, 0.13, 0.71], "stoi": [0.96, 0.959, 0.961]}, {"nq": 16, "kbps": 12.0, "wer": [0.38, 0.13, 0.71], "stoi": [0.984, 0.984, 0.985]}, {"nq": 32, "kbps": 24.0, "wer": [0.32, 0.06, 0.64], "stoi": [0.996, 0.996, 0.996]}]}

W, H = 790, 440
PW = 300          # panel width
L1, T, B = 64, 40, 64
L2 = L1 + PW + 92
X0, X1 = 0.1, 30


def sx(x, L):
    return L + (math.log10(x) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * PW


def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    labels = []
    panels = [(L1, "wer", "Word error rate (log scale)", 0.1, 50, [0.1, 0.3, 1, 3, 10, 30], lambda v: f"{v:g}%"),
              (L2, "stoi", "STOI (1 = same as the original)", 0.4, 1.0, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lambda v: f"{v:.1f}")]
    for L, key, title, Y0, Y1, ticks, fmt in panels:
        if key == "wer":
            sy = lambda y: T + (math.log10(Y1) - math.log10(max(y, Y0))) / (math.log10(Y1) - math.log10(Y0)) * (H - T - B)
        else:
            sy = lambda y, Y0=Y0, Y1=Y1: T + (Y1 - y) / (Y1 - Y0) * (H - T - B)
        for y in ticks:
            o.append(f'<line x1="{L}" x2="{L+PW}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{L-8}" y="{sy(y)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{fmt(y)}</text>')
        for x in (0.1, 0.3, 1, 3, 10, 30):
            o.append(f'<line x1="{sx(x, L):.1f}" x2="{sx(x, L):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
            o.append(f'<text x="{sx(x, L):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{x:g}</text>')
        o.append(f'<text x="{L+PW/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">bitrate, kbps (log scale)</text>')
        o.append(f'<text x="{L}" y="{T-14}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
        base = DATA["original_wer"] if key == "wer" else 1.0
        o.append(f'<line x1="{L}" x2="{L+PW}" y1="{sy(base):.1f}" y2="{sy(base):.1f}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="6 4"/>')
        if key == "wer":
            o.append(f'<text x="{L+4}" y="{sy(base)+16:.1f}" fill="{mut}" font-size="12">original audio: {base:.2f}%</text>')
        else:
            o.append(f'<text x="{L+4}" y="{sy(base)+16:.1f}" fill="{mut}" font-size="12">original audio</text>')
        for c, (codec, label) in enumerate((("mimi", "Mimi"), ("dac", "DAC 24 kHz"))):
            pts = DATA[codec]
            line = " ".join(f"{sx(p['kbps'], L):.1f},{sy(p[key][0]):.1f}" for p in pts)
            o.append(f'<polyline fill="none" stroke="{cols[c]}" stroke-width="2" points="{line}"/>')
            for p in pts:
                x = sx(p["kbps"], L)
                v, lo, hi = p[key]
                o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(min(hi, Y1)):.1f}" y2="{sy(max(lo, Y0)):.1f}" stroke="{cols[c]}" stroke-width="2"/>')
                o.append(f'<circle cx="{x:.1f}" cy="{sy(v):.1f}" r="4.5" fill="{cols[c]}" stroke="{surf}" stroke-width="1.2"/>')
                if key == "stoi" and c == 0:
                    dx, dy, anc = -8, -8, "end"
                else:
                    dx, dy, anc = 8, (-10 if c == 0 else 18), "start"
                labels.append(f'<text x="{x+dx:.1f}" y="{sy(v)+dy:.1f}" fill="{cols[c]}" font-size="12" font-weight="bold" text-anchor="{anc}" stroke="{surf}" stroke-width="3" paint-order="stroke">{p["nq"]}</text>')
    o += labels
    ly = H - 8
    for c, label in enumerate(("Mimi, 12.5 frames/s", "DAC 24 kHz, 75 frames/s")):
        x = L1 + 10 + c * 220
        o.append(f'<circle cx="{x}" cy="{ly-4}" r="5" fill="{cols[c]}"/><text x="{x+10}" y="{ly}" fill="{ink}" font-size="13">{label}</text>')
    o.append(f'<text x="{W-24}" y="{ly}" fill="{mut}" font-size="12" text-anchor="end">numbers: codebooks kept</text>')
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#2a78d6', '#eb6834')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#3987e5', '#d95926')))
