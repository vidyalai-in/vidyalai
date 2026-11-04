"""Exp 2, TTS stage: Kokoro-82M on the GPU, time to first audio for a length-matched stand-in of each
reply's first sentence, and total synthesis time for a stand-in of the whole reply. Stand-ins are my own
sentences (tts_pool.txt); no model-generated text is synthesized. Median of 5 timed runs each after warm-up."""
import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch
import voice as V
D = "/home/amit/.cache/vidyalai-scratch/v11"
pool = [l.strip() for l in open(f"{D}/data/tts_pool.txt") if l.strip()]
src = sys.argv[1]  # runs/claude_stream.jsonl or runs/qwen_stream.jsonl
recs = [json.loads(l) for l in open(f"{D}/{src}") if '"ttft"' in l]

def nearest(n):
    return min(pool, key=lambda s: abs(len(s) - n))

def build(n):  # concatenate pool sentences until the length is close to n characters
    out, rng = "", np.random.default_rng(n)
    order = list(rng.permutation(len(pool)))
    while len(out) < n - 20 and order:
        s = pool[order.pop()]
        if len(out) + len(s) <= n + 40:
            out = (out + " " + s).strip()
    return out or nearest(n)

pipe = V.kokoro("cuda")
for _ in range(3):
    V.synth_first_audio(pipe, pool[15], "af_heart")
cache, rows = {}, []
with V.GpuSampler() as g:
    for r in recs:
        voice = V.STOCK_VOICES[r["turn"] % 4]
        a, b = nearest(r["first_chars"]), build(r["full_chars"])
        for key in (("first", a), ("full", b)):
            if key[1] not in cache:
                ts = [V.synth_first_audio(pipe, key[1], voice) for _ in range(6)][1:]
                cache[key[1]] = {"ttfa": float(np.median([t[0] for t in ts])), "total": float(np.median([t[1] for t in ts])),
                                 "audio_s": ts[0][2]}
        rows.append({"turn": r["turn"], "rep": r.get("rep", 0), "first_text_chars": len(a), "full_text_chars": len(b),
                     "first_ttfa": cache[a]["ttfa"], "first_audio_s": cache[a]["audio_s"],
                     "full_ttfa": cache[b]["ttfa"], "full_total": cache[b]["total"], "full_audio_s": cache[b]["audio_s"]})
out = {"rows": rows, "gpu": g.summary(), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20, "n_texts": len(cache)}
json.dump(out, open(f"{D}/runs/tts_{src.split('/')[-1].split('.')[0]}.json", "w"))
f = np.array([r["first_ttfa"] for r in rows]); t = np.array([r["full_total"] for r in rows])
print("first-sentence TTFA ms median", round(np.median(f) * 1e3), "full total ms median", round(np.median(t) * 1e3), out["gpu"], out["peak_vram_mb"])
