"""Exp 3: does Whisper write text when nobody speaks? Usage: exp3_halluc.py <model>
Configs: greedy (plain), fallback (temperature 0..1 step 0.2, token compression ratio 1.35, avg logprob -1),
fallback+nospeech (adds no_speech_threshold 0.6), vad (Silero VAD gate, then greedy).
Writes runs/exp3_<model>.json."""
import sys, json, time, os
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab"); sys.path.insert(0, ".")
import numpy as np, torch
import asr, noise_clips
name = sys.argv[1]
clips = noise_clips.make_clips()
audios = [c["audio"] for c in clips]
proc, model = asr.load_whisper(name)
FB = dict(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35, logprob_threshold=-1.0)
configs = {"greedy": dict(do_sample=False),
           "fallback": dict(FB),
           "fallback+nospeech": dict(FB, no_speech_threshold=0.6)}
res = {"model": asr.MODELS[name], "clips": [{k: v for k, v in c.items() if k != "audio"} for c in clips], "out": {}}
for cname, g in configs.items():
    torch.manual_seed(0)
    t0 = time.perf_counter()
    # fallback configs keep per-token scores for every candidate (host RAM), so they run in batches of 4
    hyps, info = asr.transcribe(proc, model, audios, batch_size=16 if cname == "greedy" else int(os.environ.get("FB_BS", "4")), **g)
    res["out"][cname] = hyps
    res.setdefault("time_s", {})[cname] = time.perf_counter() - t0
    ne = sum(1 for h in hyps if asr.normalize(h, "basic"))
    import resource; rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(name, cname, f"maxrss {rss:.0f} MB", "non-empty", ne, "/", len(hyps), round(time.perf_counter() - t0, 1), "s", flush=True)
# Silero VAD gate (CPU)
from silero_vad import load_silero_vad, get_speech_timestamps
vad = load_silero_vad()
gate = []
for a in audios:
    ts = get_speech_timestamps(torch.from_numpy(a), vad, sampling_rate=16000)
    gate.append(len(ts) > 0)
res["vad_speech"] = gate
res["out"]["vad"] = [h if g else "" for h, g in zip(res["out"]["greedy"], gate)]
print(name, "vad says speech in", sum(gate), "clips; non-empty after gate",
      sum(1 for h in res["out"]["vad"] if asr.normalize(h, "basic")))
json.dump(res, open(f"runs/exp3_{name}.json", "w"), indent=0)
