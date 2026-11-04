"""Exp 2, VAD stage: Silero VAD (streaming, 32 ms chunks, CPU, 1 thread) on the 40 Kokoro user turns.
Endpoint delay after the true end of speech, early cuts at mid-turn pauses, onset delay, compute per chunk."""
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, soundfile as sf, torch
import voice as V
from silero_vad import load_silero_vad, VADIterator
torch.set_num_threads(1)
D = "/home/amit/.cache/vidyalai-scratch/v11"
SR, CH = 16000, 512
meta = [m for m in json.load(open(f"{D}/runs/synth_meta.json")) if m["set"] == "turn"]
rng = np.random.default_rng(0)

def bounds(x, db=35.0, fr=160):
    r = np.sqrt(np.convolve(x ** 2, np.ones(fr) / fr, "same"))
    on = np.where(r > r.max() * 10 ** (-db / 20))[0]
    # internal pauses: runs below threshold longer than 100 ms between on and off
    act = r > r.max() * 10 ** (-db / 20)
    seg = act[on[0]:on[-1]]
    gaps, run = [], 0
    for a in seg:
        if not a: run += 1
        else:
            if run > 0.1 * SR: gaps.append(run / SR)
            run = 0
    return on[0] / SR, on[-1] / SR, gaps

model = load_silero_vad()
streams = []
for m in meta:
    x = V.resample(sf.read(f"{D}/audio/turn_{m['i']:02d}_{m['voice']}.wav", dtype="float32")[0], 24000, 16000)
    s0, s1, gaps = bounds(x)
    lead, tail = int(0.5 * SR), int(2.0 * SR)
    y = np.concatenate([np.zeros(lead, np.float32), x, np.zeros(tail, np.float32)])
    y = y + rng.normal(0, 0.001, len(y)).astype(np.float32)   # noise floor at -60 dBFS RMS
    streams.append({"turn": m["i"], "y": y, "start": s0 + 0.5, "end": s1 + 0.5, "gaps": gaps, "text": m["text"]})

res = {"items": [{"turn": s["turn"], "start": s["start"], "end": s["end"], "gaps": s["gaps"]} for s in streams], "runs": {}}
chunk_ms = []
for ms in [100, 200, 300, 500, 800]:
    rows = []
    for s in streams:
        it = VADIterator(model, threshold=0.5, sampling_rate=SR, min_silence_duration_ms=ms, speech_pad_ms=30)
        ev = []
        y = torch.from_numpy(s["y"])
        for k in range(0, len(y) - CH + 1, CH):
            t0 = time.perf_counter()
            e = it(y[k:k + CH])
            chunk_ms.append((time.perf_counter() - t0) * 1e3)
            if e:
                ev.append({**e, "at": (k + CH) / SR})
        starts = [e["at"] for e in ev if "start" in e]
        ends = [e for e in ev if "end" in e]
        early = [e for e in ends if e["end"] / SR < s["end"] - 0.05]
        final = [e for e in ends if e["end"] / SR >= s["end"] - 0.05]
        rows.append({"turn": s["turn"], "onset_delay": (starts[0] - s["start"]) if starts else None,
                     "n_early_cuts": len(early), "endpoint_delay": (final[0]["at"] - s["end"]) if final else None,
                     "events": ev})
    res["runs"][ms] = rows
    d = [r["endpoint_delay"] for r in rows if r["endpoint_delay"] is not None]
    k = sum(r["n_early_cuts"] > 0 for r in rows)
    print(ms, "median endpoint delay ms", round(np.median(d) * 1e3), "p90", round(np.percentile(d, 90) * 1e3),
          "turns cut early", k, "/", len(rows), "missing end", len(rows) - len(d), flush=True)
res["chunk_ms_median"] = float(np.median(chunk_ms)); res["chunk_ms_p90"] = float(np.percentile(chunk_ms, 90))
res["n_chunks"] = len(chunk_ms)
print("per-chunk ms median", res["chunk_ms_median"], "p90", res["chunk_ms_p90"])
gaps = [g for s in streams for g in s["gaps"]]
print("internal pauses > 100 ms:", len(gaps), "max", max(gaps) if gaps else 0, "turns with one", sum(bool(s["gaps"]) for s in streams))
json.dump(res, open(f"{D}/runs/vad.json", "w"))
