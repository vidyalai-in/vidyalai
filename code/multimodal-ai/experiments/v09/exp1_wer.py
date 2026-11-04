"""Exp 1 / 4: transcribe the 500+500 LibriSpeech subsets with one Whisper model.
Usage: exp1_wer.py <model> [--beams N] [--limit K] [--bs B]
Writes runs/exp1_<model>_b<N>.jsonl (hyps) and runs/exp1_<model>_b<N>_meta.json (timing, memory, clocks)."""
import sys, json, time, argparse
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch
import asr
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--beams", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
sub = json.load(open("runs/subsets.json"))
utt = {}
for split in ("test-clean", "test-other"):
    for u in asr.librispeech(split):
        utt[u["id"]] = dict(u, split=split)
t_load = time.perf_counter()
proc, model = asr.load_whisper(a.model)
torch.cuda.synchronize(); t_load = time.perf_counter() - t_load
gen = dict(num_beams=a.beams, do_sample=False)
tag = f"{a.model}_b{a.beams}" + (f"_lim{a.limit}" if a.limit else "")
meta = {"model": asr.MODELS[a.model], "beams": a.beams, "batch_size": a.bs, "load_s": t_load,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
import transformers; meta["transformers"] = transformers.__version__
# warm-up on 2 batches
warm = [asr.load_flac(utt[i]["path"]) for i in sub["test-clean"][:a.bs]]
asr.transcribe(proc, model, warm, batch_size=a.bs, **gen)
rows = []
for split in ("test-clean", "test-other"):
    ids = sub[split][:a.limit] if a.limit else sub[split]
    audios = [asr.load_flac(utt[i]["path"]) for i in ids]
    torch.cuda.reset_peak_memory_stats()
    with asr.GpuMonitor() as mon:
        hyps, info = asr.transcribe(proc, model, audios, batch_size=a.bs, **gen)
    info.update(peak_alloc_mb=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
                clock_mhz=mon.clock, power_w=mon.power, temp_c=mon.temp, smi_samples=mon.n)
    info["rtf_x"] = info["audio_s"] / (info["feat_s"] + info["gen_s"])
    meta[split] = info
    print(split, {k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()}, flush=True)
    for i, h in zip(ids, hyps):
        rows.append({"id": i, "split": split, "ref": utt[i]["ref"], "hyp": h})
# fixed timing benchmarks: one batch of bs clips (median of 5 after 2 warm-ups), one ~10 s clip at batch 1
bench = [asr.load_flac(utt[i]["path"]) for i in sub["test-clean"][:a.bs]]
bsec = sum(len(x) for x in bench) / asr.SR
with asr.GpuMonitor() as mon:
    med, ts = asr.cuda_median(lambda: asr.transcribe(proc, model, bench, batch_size=a.bs, **gen), n=5, warmup=2)
meta["bench_batch"] = {"clips": len(bench), "audio_s": bsec, "median_s": med, "runs_s": ts, "rtf_x": bsec / med,
                       "clock_mhz": mon.clock, "power_w": mon.power}
ten = min(sub["test-clean"], key=lambda i: abs(len(asr.load_flac(utt[i]["path"])) / asr.SR - 10.0))
x10 = [asr.load_flac(utt[ten]["path"])]
with asr.GpuMonitor() as mon:
    med1, ts1 = asr.cuda_median(lambda: asr.transcribe(proc, model, x10, batch_size=1, **gen), n=5, warmup=2)
meta["bench_single"] = {"id": ten, "audio_s": len(x10[0]) / asr.SR, "median_s": med1, "runs_s": ts1,
                        "rtf_x": len(x10[0]) / asr.SR / med1, "clock_mhz": mon.clock, "power_w": mon.power}
# encoder-only time for one 30 s window (batch 1)
feat = proc.feature_extractor(x10, sampling_rate=asr.SR, return_tensors="pt").input_features.to("cuda", torch.float16)
with torch.inference_mode():
    mede, tse = asr.cuda_median(lambda: model.model.encoder(feat), n=10, warmup=3)
meta["bench_encoder_ms"] = mede * 1000
print("bench", json.dumps({k: meta[k] for k in ("bench_batch", "bench_single", "bench_encoder_ms")}, default=float)[:900])
with open(f"runs/exp1_{tag}.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
json.dump(meta, open(f"runs/exp1_{tag}_meta.json", "w"), indent=1, default=float)
