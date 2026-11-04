# Generates public/diagrams/v09-waveform-to-logmel.svg and -dark.svg (data: v09-speech-recognition-whisper.md).
# One Kokoro-82M stock-voice sentence (voice af_heart, speed 1.0, resampled 24 -> 16 kHz), seen three ways:
# the waveform, the short-time Fourier transform (400-sample Hann window = 25 ms, hop 160 = 10 ms, 201 bins)
# and Whisper's 80-bin log-mel features (WhisperFeatureExtractor of openai/whisper-tiny).
# Input: runs/fig_audio.npz from scratch v09/fig_audio.py. Heatmaps are embedded as small PNGs.
# Usage: python3 diagrams/v09-waveform-to-logmel.py <runs dir> public/diagrams/v09-waveform-to-logmel
import base64, io, math, sys
import numpy as np
from PIL import Image

runs, out = sys.argv[1], sys.argv[2]
d = np.load(f"{runs}/fig_audio.npz")
wave, stft_db, mel = d["wave"], d["stft_db"], d["mel80"]
SR = 16000
dur = len(wave) / SR
W = 780
L, R = 78, 24
PW = W - L - R
PH = [120, 150, 150]          # panel heights
GAP = 64
T0 = 34
H = T0 + sum(PH) + 2 * GAP + 54


def ramp(v, stops):
    """v in [0,1] -> RGB along a list of hex stops."""
    cols = np.array([[int(s[i:i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=float)
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (cols[i] * (1 - f) + cols[i + 1] * f).astype(np.uint8)


def heat_png(arr, stops, w, h, span):
    a = arr[::-1]  # low frequencies at the bottom
    hi = a.max(); lo = hi - span  # show the top `span` units (dB for the STFT, log-mel units for Whisper)
    v = (a - lo) / (hi - lo)
    img = Image.fromarray(ramp(v, stops), "RGB").resize((w, h), Image.BILINEAR)
    img = img.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buf = io.BytesIO(); img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def hz_to_mel(f):  # Slaney mel scale, as librosa.filters.mel (htk=False) used by Whisper
    f = np.asarray(f, float)
    lin = f / (200.0 / 3)
    log = 15 + np.log(np.maximum(f, 1e-9) / 1000) / (math.log(6.4) / 27)
    return np.where(f >= 1000, log, lin)


def svg(th, stops):
    ink, mut, grid, acc, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    sx = lambda t: L + t / dur * PW
    y = T0
    # panel 1: waveform
    h = PH[0]; mid = y + h / 2
    o.append(f'<text x="{L}" y="{y-12}" fill="{ink}" font-size="14" font-weight="bold">1. Waveform: 73,200 samples (16,000 per second)</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{mid}" y2="{mid}" stroke="{grid}"/>')
    step = len(wave) / PW / 2
    pts_hi, pts_lo = [], []
    peak = np.abs(wave).max()
    for k in range(PW * 2):
        seg = wave[int(k * step):int((k + 1) * step) + 1]
        xx = L + k / 2
        pts_hi.append(f"{xx:.1f},{mid - seg.max() / peak * (h / 2 - 4):.1f}")
        pts_lo.append(f"{xx:.1f},{mid - seg.min() / peak * (h / 2 - 4):.1f}")
    o.append(f'<polygon points="{" ".join(pts_hi + pts_lo[::-1])}" fill="{acc}" stroke="{acc}" stroke-width="0.6"/>')
    # one 25 ms window, drawn to scale
    t = 1.20
    o.append(f'<rect x="{sx(t):.1f}" y="{y}" width="{max(2.5, sx(t+0.025)-sx(t)):.1f}" height="{h}" fill="none" stroke="{ink}" stroke-width="1.6"/>')
    o.append(f'<text x="{sx(t)-4:.1f}" y="{y+h+16}" fill="{ink}" font-size="12">one 25 ms window (400 samples), drawn to scale; the next one starts 10 ms later</text>')
    y += h + GAP
    # panel 2: STFT
    h = PH[1]
    o.append(f'<text x="{L}" y="{y-12}" fill="{ink}" font-size="14" font-weight="bold">2. Short-time Fourier transform: 457 frames x 201 frequency bins (0 to 8,000 Hz)</text>')
    o.append(f'<image x="{L}" y="{y}" width="{PW}" height="{h}" preserveAspectRatio="none" href="data:image/png;base64,{heat_png(stft_db, stops, PW, h, 70.0)}"/>')
    for f in (0, 2000, 4000, 6000, 8000):
        yy = y + h - f / 8000 * h
        o.append(f'<text x="{L-8}" y="{yy+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{f:,}</text>')
    o.append(f'<text transform="translate(18,{y+h/2}) rotate(-90)" fill="{ink}" font-size="13" text-anchor="middle">Hz (linear)</text>')
    y += h + GAP
    # panel 3: log-mel
    h = PH[2]
    o.append(f'<text x="{L}" y="{y-12}" fill="{ink}" font-size="14" font-weight="bold">3. Whisper log-mel features: 80 mel bins per 10 ms frame</text>')
    o.append(f'<image x="{L}" y="{y}" width="{PW}" height="{h}" preserveAspectRatio="none" href="data:image/png;base64,{heat_png(mel, stops, PW, h, 1.75)}"/>')
    mmax = hz_to_mel(8000)
    for f in (0, 500, 1000, 2000, 4000, 8000):
        yy = y + h - float(hz_to_mel(f)) / float(mmax) * h
        o.append(f'<text x="{L-8}" y="{yy+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{f:,}</text>')
    o.append(f'<text transform="translate(18,{y+h/2}) rotate(-90)" fill="{ink}" font-size="13" text-anchor="middle">Hz (mel scale)</text>')
    for s in range(0, int(dur) + 1):
        o.append(f'<text x="{sx(s):.1f}" y="{y+h+18}" fill="{mut}" font-size="12" text-anchor="middle">{s}</text>')
    o.append(f'<text x="{L+PW/2}" y="{y+h+40}" fill="{ink}" font-size="13" text-anchor="middle">time (seconds)</text>')
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#0b5fd6', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#38bdf8', '#0b162d')
open(out + '.svg', 'w').write(svg(light, ['#ffffff', '#9cc3f5', '#0b5fd6', '#0b1a33']))
open(out + '-dark.svg', 'w').write(svg(dark, ['#0b162d', '#1d4f8a', '#38bdf8', '#f2f7ff']))
