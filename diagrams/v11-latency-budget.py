# Generates public/diagrams/v11-latency-budget.svg and -dark.svg (data: v11-voice-agents.md).
# Time from the end of the user's speech to the first audio of the reply, for a cascaded voice agent on the laptop:
# Silero VAD endpoint (500 ms silence rule), Whisper large-v3-turbo, the LLM (streamed), Kokoro-82M on the GPU.
# Segments: per-stage medians over the calls (they don't add up exactly to the median total).
# Black tick and label: median of the per-call total; whisker to the 90th percentile of the per-call total.
# Dashed line: the 208 ms mean gap between turns in Stivers et al. (2009), 10 languages.
# Data from scratch v11/scripts/s09_analyze.py -> runs/results.json (budget_claude, budget_qwen).
# Usage: python3 diagrams/v11-latency-budget.py public/diagrams/v11-latency-budget
import sys

DATA = {"xmax": 4.0, "rows": [{"label": "Claude Haiku 5.5,|stream first sentence", "parts": {"vad": 0.604, "asr": 0.21, "llm": 1.03, "tts": 0.136}, "p50": 2.021, "p90": 2.393}, {"label": "Claude Haiku 5.5,|wait for the full reply", "parts": {"vad": 0.604, "asr": 0.21, "llm": 1.233, "tts": 0.335}, "p50": 2.405, "p90": 2.858}, {"label": "Qwen2.5-1.5B, local,|stream first sentence", "parts": {"vad": 0.605, "asr": 0.21, "llm": 0.511, "tts": 0.105}, "p50": 1.445, "p90": 1.829}, {"label": "Qwen2.5-1.5B, local,|wait for the full reply", "parts": {"vad": 0.605, "asr": 0.21, "llm": 0.818, "tts": 0.256}, "p50": 1.893, "p90": 3.195}]}

W, H = 780, 360
L, R, T, B = 210, 40, 40, 92
X1 = DATA["xmax"]
STAGES = [("vad", "endpoint wait (VAD)"), ("asr", "speech recognition"), ("llm", "LLM"), ("tts", "text-to-speech")]


def svg(th, cols):
    ink, mut, grid, surf = th
    sx = lambda x: L + x / X1 * (W - L - R)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif" role="img">']
    o.append(f'<rect width="{W}" height="{H}" fill="{surf}"/>')
    step = 0.5
    x = 0.0
    while x <= X1 + 1e-9:
        o.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H-B}" stroke="{grid}" stroke-width="1"/>')
        o.append(f'<text x="{sx(x):.1f}" y="{H-B+18}" fill="{mut}" font-size="12" text-anchor="middle">{x:g} s</text>')
        x += step
    o.append(f'<text x="{(L+W-R)/2}" y="{H-B+40}" fill="{ink}" font-size="13" text-anchor="middle">seconds from the end of the user\'s speech to the first sound of the reply</text>')
    rows = DATA["rows"]
    bh = 30
    gap = (H - T - B - len(rows) * bh) / (len(rows) + 1)
    for i, r in enumerate(rows):
        y = T + gap * (i + 1) + bh * i
        lines = r["label"].split("|")
        for j, ln in enumerate(lines):
            o.append(f'<text x="{L-10}" y="{y + bh/2 + 4 + (j - (len(lines)-1)/2)*15:.1f}" fill="{ink}" font-size="13" text-anchor="end">{ln}</text>')
        x0 = 0.0
        for k, (key, _) in enumerate(STAGES):
            v = r["parts"][key]
            o.append(f'<rect x="{sx(x0):.1f}" y="{y:.1f}" width="{sx(x0+v)-sx(x0):.1f}" height="{bh}" fill="{cols[k]}" stroke="{surf}" stroke-width="1"/>')
            x0 += v
        p50, p90 = r["p50"], r["p90"]
        o.append(f'<line x1="{sx(p50):.1f}" x2="{sx(p90):.1f}" y1="{y+bh/2:.1f}" y2="{y+bh/2:.1f}" stroke="{ink}" stroke-width="1.5"/>')
        o.append(f'<line x1="{sx(p90):.1f}" x2="{sx(p90):.1f}" y1="{y+bh/2-6:.1f}" y2="{y+bh/2+6:.1f}" stroke="{ink}" stroke-width="1.5"/>')
        o.append(f'<line x1="{sx(p50):.1f}" x2="{sx(p50):.1f}" y1="{y-4:.1f}" y2="{y+bh+4:.1f}" stroke="{ink}" stroke-width="2.5"/>')
        o.append(f'<text x="{sx(p90)+6:.1f}" y="{y+bh/2+4:.1f}" fill="{ink}" font-size="12">{p50:.2f} s (p90 {p90:.2f})</text>')
    hx = sx(0.208)
    o.append(f'<line x1="{hx:.1f}" x2="{hx:.1f}" y1="{T-8}" y2="{H-B}" stroke="{mut}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{hx+4:.1f}" y="{T-12}" fill="{mut}" font-size="12">human turn gap, 0.21 s</text>')
    ly = H - 22
    x = 20
    for k, (_, label) in enumerate(STAGES):
        o.append(f'<rect x="{x}" y="{ly-11}" width="14" height="14" fill="{cols[k]}"/><text x="{x+20}" y="{ly}" fill="{ink}" font-size="13">{label}</text>')
        x += 40 + len(label) * 7.2
    o.append('</svg>')
    return "\n".join(o)


light = ('#0b1a33', '#4a5b78', '#dde5f2', '#ffffff')
dark = ('#e8eefc', '#9fb0cf', '#22355a', '#0b162d')
if __name__ == "__main__":
    open(sys.argv[1] + '.svg', 'w').write(svg(light, ('#9aa9c4', '#1f9e74', '#2a78d6', '#eb6834')))
    open(sys.argv[1] + '-dark.svg', 'w').write(svg(dark, ('#5b6f94', '#2bb38a', '#3987e5', '#d95926')))
