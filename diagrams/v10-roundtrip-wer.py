# Generates public/diagrams/v10-roundtrip-wer.svg and -dark.svg (data: v10-text-to-speech.md).
# Round-trip word error rate: Kokoro-82M (speed 1.0) speaks a sentence, Whisper large-v3-turbo (fp16, greedy)
# transcribes it, and WER is computed after Whisper's English text normalizer on both sides.
# Top panel: 200 public-domain sentences (100 each from The Time Machine and The Wonderful Wizard of Oz, Project
# Gutenberg) per stock voice (grade from Kokoro's VOICES.md in brackets), and the same Whisper on 200 random
# LibriSpeech test-clean utterances of real speakers (different text; measurement only).
# Bottom panel: the 60 hard sentences written for the article, 10 per category, all six voices pooled (60 clips).
# Points: corpus WER; bars: bootstrap 95% CIs over items (2,000 resamples). Data: scratch v10/runs/results.json.
# Usage: python3 diagrams/v10-roundtrip-wer.py public/diagrams/v10-roundtrip-wer
import sys
TOP = [("af_heart (A)", 0.55, 0.28, 0.87, "voice"), ("af_bella (A-)", 0.52, 0.24, 0.84, "voice"),
       ("bf_emma (B-)", 0.61, 0.32, 0.94, "voice"), ("am_michael (C+)", 0.66, 0.35, 1.02, "voice"),
       ("bm_george (C)", 0.52, 0.26, 0.81, "voice"), ("am_adam (F+)", 0.46, 0.20, 0.75, "voice"),
       ("Real speakers, LibriSpeech", 2.58, 1.61, 4.03, "ref")]
BOT = [("Heteronyms", 1.15, 0.38, 2.04), ("Numbers", 1.46, 0.0, 3.48), ("Rare words, names", 3.33, 1.65, 5.37),
       ("Currency", 3.49, 1.54, 6.0), ("Abbreviations", 7.14, 4.37, 10.23), ("Dates and times", 17.01, 13.06, 21.08)]
W = 780
L, R = 230, 60
ROW = 28
T1 = 40
H1 = ROW * len(TOP)
GAP = 96
T2 = T1 + H1 + GAP
H2 = ROW * len(BOT)
H = T2 + H2 + 52


def panel(o, th, rows, top, x1, ticks, title, cols):
    ink, mut, grid, surf = th
    def sx(x): return L + x / x1 * (W - L - R)
    o.append(f'<text x="{L}" y="{top-16}" fill="{ink}" font-size="14" font-weight="bold">{title}</text>')
    bot = top + ROW * len(rows)
    for v in ticks:
        o.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{bot}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{bot+18}" fill="{mut}" font-size="13" text-anchor="middle">{v:g}%</text>')
    for i, r in enumerate(rows):
        lab, w, lo, hi = r[:4]
        kind = r[4] if len(r) > 4 else "hard"
        c = cols[kind]
        y = top + ROW * i + ROW / 2
        o.append(f'<text x="{L-12}" y="{y+4:.1f}" fill="{ink}" font-size="13" text-anchor="end">{lab}</text>')
        o.append(f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2.5"/>')
        if kind == "ref":
            o.append(f'<rect x="{sx(w)-5:.1f}" y="{y-5:.1f}" width="10" height="10" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        else:
            o.append(f'<circle cx="{sx(w):.1f}" cy="{y:.1f}" r="5.5" fill="{c}" stroke="{surf}" stroke-width="1.5"/>')
        o.append(f'<text x="{sx(hi)+8:.1f}" y="{y+4:.1f}" fill="{mut}" font-size="12">{w:.2f}%</text>' if w < 2 else
                 f'<text x="{sx(hi)+8:.1f}" y="{y+4:.1f}" fill="{mut}" font-size="12">{w:.1f}%</text>')


def svg(th, cols):
    ink, mut, grid, surf = th
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    panel(o, th, TOP, T1, 4.5, (0, 1, 2, 3, 4), "200 public-domain sentences per voice, and real speech for reference", cols)
    panel(o, th, BOT, T2, 22, (0, 5, 10, 15, 20), "60 hard sentences by category, six voices pooled", cols)
    o.append(f'<text x="{(L+W-R)/2}" y="{H-10}" fill="{ink}" font-size="14" text-anchor="middle">Word error rate of Whisper large-v3-turbo on the audio, with 95% CI</text>')
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, {"voice": '#2a78d6', "ref": '#4a5b78', "hard": '#eb6834'}))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, {"voice": '#3987e5', "ref": '#9fb0cf', "hard": '#d95926'}))
