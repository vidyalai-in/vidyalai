"""Exp 2, ASR stage: Whisper (large-v3-turbo and tiny, fp16, greedy) on each user turn after the endpoint.
Time per turn = median of 5 timed runs after warm-up (feature extraction + generate), plus WER."""
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, soundfile as sf, torch
import voice as V
D = "/home/amit/.cache/vidyalai-scratch/v11"
meta = [m for m in json.load(open(f"{D}/runs/synth_meta.json")) if m["set"] == "turn"]
clips = [np.concatenate([V.resample(sf.read(f"{D}/audio/turn_{m['i']:02d}_{m['voice']}.wav", dtype="float32")[0], 24000, 16000),
                         np.zeros(9600, np.float32)]) for m in meta]
out = {}
for name in ["openai/whisper-large-v3-turbo", "openai/whisper-tiny"]:
    torch.cuda.reset_peak_memory_stats()
    m, p = V.whisper(name)
    norm = V.normalizer(p)
    for _ in range(3):
        V.transcribe(m, p, [clips[0]], batch=1)
    times, hyps = [], []
    with V.GpuSampler() as g:
        for c in clips:
            ts = []
            for _ in range(5):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                h = V.transcribe(m, p, [c], batch=1)[0]
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            times.append(float(np.median(ts))); hyps.append(h)
    rows = V.wer_counts([x["text"] for x in meta], hyps, norm)
    out[name] = {"times": times, "hyps": hyps, "err": rows.tolist(), "gpu": g.summary(),
                 "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}
    print(name, "median ms", round(np.median(times) * 1e3), "p90", round(np.percentile(times, 90) * 1e3),
          "WER %", round(V.boot_wer(rows)[0] * 100, 2), g.summary(), out[name]["peak_vram_mb"], flush=True)
    del m; torch.cuda.empty_cache()
json.dump(out, open(f"{D}/runs/asr_turns.json", "w"))
