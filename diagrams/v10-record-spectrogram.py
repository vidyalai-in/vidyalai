# Generates public/diagrams/v10-record-spectrogram.svg and -dark.svg (data: v10-text-to-speech.md).
# One Kokoro-82M sentence, "Please record the record for the archive." (stock voice af_heart, speed 1.0, 24 kHz),
# seen as the model built it: the pitch contour its prosody predictor chose (F0, captured from the model during
# synthesis), the log-mel spectrogram of the output audio (80 mel bins up to 12 kHz, 1,200-sample window = 50 ms,
# hop 300 = 12.5 ms, librosa), and each word's span from the predicted phoneme durations (25 ms frames), with
# the phonemes misaki chose. Input: scratch v10/runs/clip_record.wav and runs/fig_clips.json.
# Usage: python3 diagrams/v10-record-spectrogram.py <runs dir> public/diagrams/v10-record-spectrogram
# (needs numpy, pillow, soundfile and librosa: run with the scratch mm venv)
import base64, io, json, sys
import numpy as np
import soundfile as sf
import librosa
from PIL import Image

runs, out = sys.argv[1], sys.argv[2]
info = json.load(open(f"{runs}/fig_clips.json"))["record"]
wav, SR = sf.read(f"{runs}/clip_record.wav", dtype="float32")
dur = len(wav) / SR
mel = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=1200, hop_length=300, n_mels=80, fmax=12000)
mel = np.log10(np.maximum(mel, 1e-10))
mel = np.maximum(mel, mel.max() - 8.0)
mel = (mel - mel.min()) / (mel.max() - mel.min())
f0 = np.array(info["f0_pred"], float)
f0_t = (np.arange(len(f0)) + 0.5) * dur / len(f0)
# word spans straight from the predicted durations: pred_dur[0] is the start marker, then one duration
# (in 25 ms frames) per phoneme character, spaces included, then the end marker
ps = info["phonemes"]
dur_f = info["pred_dur"]
edges = np.cumsum([0] + dur_f) * 0.025          # frame boundaries in seconds
words, i, start = [], 0, None
for k, ch in enumerate(ps + " "):
    if ch == " " or k == len(ps):
        if start is not None:
            e = k
            while ps[e - 1] in ".,": e -= 1          # trailing punctuation is a pause, not part of the word
            words.append((ps[start:e], edges[1 + start], edges[1 + e]))
        start = None
    elif start is None:
        start = k
texts = [w.strip(".,") for w in info["text"].split()]
assert len(texts) == len(words), (texts, words)

W = 780
L, R = 70, 24
PW = W - L - R
T0 = 30
F0H, GAP1, MH, GAP2 = 110, 74, 190, 62
H = T0 + F0H + GAP1 + MH + GAP2 + 4
y_f0, y_mel = T0, T0 + F0H + GAP1
FMIN, FMAX = 100, 350


def sx(t): return L + t / dur * PW
def sf0(f): return y_f0 + F0H - (f - FMIN) / (FMAX - FMIN) * F0H


def ramp(v, stops):
    cols = np.array([[int(s[i:i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=float)
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (cols[i] * (1 - f) + cols[i + 1] * f).astype(np.uint8)


def heat_png(stops):
    rgb = ramp(mel[::-1], stops)
    im = Image.fromarray(rgb).resize((PW, MH), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def svg(th, stops, acc):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    # pitch panel
    o.append(f'<text x="{L}" y="{y_f0-10}" fill="{ink}" font-size="14" font-weight="bold">Pitch the prosody predictor chose (Hz)</text>')
    for f in (100, 200, 300):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sf0(f):.1f}" y2="{sf0(f):.1f}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{sf0(f)+4:.1f}" fill="{mut}" font-size="12" text-anchor="end">{f}</text>')
    seg, segs = [], []
    for t, f in zip(f0_t, f0):
        if f > 60:
            seg.append(f"{sx(t):.1f},{sf0(min(max(f, FMIN), FMAX)):.1f}")
        elif seg:
            segs.append(seg); seg = []
    if seg: segs.append(seg)
    for s in segs:
        if len(s) > 1:
            o.append(f'<polyline fill="none" stroke="{acc}" stroke-width="2.5" points="{" ".join(s)}"/>')
    # spectrogram
    o.append(f'<text x="{L}" y="{y_mel-30}" fill="{ink}" font-size="14" font-weight="bold">Log-mel spectrogram of the audio, with each word\'s predicted span</text>')
    o.append(f'<image x="{L}" y="{y_mel}" width="{PW}" height="{MH}" preserveAspectRatio="none" href="data:image/png;base64,{heat_png(stops)}"/>')
    o.append(f'<text x="{L-8}" y="{y_mel+10}" fill="{mut}" font-size="12" text-anchor="end">12k</text>')
    o.append(f'<text x="{L-8}" y="{y_mel+MH}" fill="{mut}" font-size="12" text-anchor="end">0 Hz</text>')
    o.append(f'<text transform="translate(22,{y_mel+MH/2}) rotate(-90)" fill="{mut}" font-size="12" text-anchor="middle">mel bins</text>')
    for j, (w, (ph, a, b)) in enumerate(zip(texts, words)):
        xa, xb = sx(a), sx(b)
        for x in ((xa, xb) if j == len(words) - 1 else (xa,)):
            for y1, y2 in ((y_f0, y_f0 + F0H), (y_mel, y_mel + MH)):
                o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y1}" y2="{y2}" stroke="{mut}" stroke-width="1" stroke-dasharray="3 3"/>')
        xm = (xa + xb) / 2
        bold = ' font-weight="bold"' if w == "record" else ""
        o.append(f'<text x="{xm:.1f}" y="{y_mel-8}" fill="{ink}" font-size="13" text-anchor="middle"{bold}>{w}</text>')
        ph = ph.rstrip(".")
        o.append(f'<text x="{xm:.1f}" y="{y_mel+MH+20}" fill="{acc if w == "record" else ink}" font-size="13" text-anchor="middle"{bold}>{ph}</text>')
    o.append(f'<text x="{L}" y="{y_mel+MH+44}" fill="{mut}" font-size="12">0 s</text>')
    o.append(f'<text x="{W-R}" y="{y_mel+MH+44}" fill="{mut}" font-size="12" text-anchor="end">{dur:.2f} s</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{y_mel+MH+44}" fill="{mut}" font-size="12" text-anchor="middle">phonemes from misaki; spans from the predicted durations (25 ms frames)</text>')
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(out + '.svg', 'w').write(svg(light, ['#ffffff', '#9cc3f2', '#2a78d6', '#0b1a33'], '#eb6834'))
    open(out + '-dark.svg', 'w').write(svg(dark, ['#0b162d', '#1f4f8f', '#3987e5', '#e8eefc'], '#d95926'))
