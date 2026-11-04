"""Exp 1: Kokoro-82M speed. RTF vs text length, peak VRAM, time to first audio
(sentence-by-sentence vs whole paragraph), determinism. Usage: exp_timing.py cuda|cpu"""
import sys, json, time, os, statistics
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch
import speech_tts as st

dev = sys.argv[1]
ONLY_TTFA = len(sys.argv) > 2 and sys.argv[2] == "ttfa"
N = 7 if dev == "cuda" else 5
torch.manual_seed(0)
model, pipes = st.load_kokoro(dev, langs=("a",))
pipe = pipes["a"]
VOICE = "af_heart"
pipe.load_voice(VOICE)

main = [json.loads(l)["text"] for l in open("texts/main.jsonl")]
words = " ".join(main).split()
LENGTHS = [5, 10, 20, 40, 80, 160, 320] if dev == "cuda" else [5, 10, 20, 40, 80, 160]
PARA = ("The kettle clicked off just as the rain began. Outside, the street lamps flickered on "
        "one by one. A bus hissed to a stop at the corner and pulled away again. Somewhere "
        "upstairs, a radio was playing an old song. By the time the tea was ready, the window "
        "had fogged over completely.")

sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)

def run(text, split=None):
    """Returns (t_first_audio, t_total, audio_seconds, g2p_seconds, chunks)."""
    sync(); t0 = time.perf_counter(); first = None; n = 0; chunks = 0
    for r in pipe(text, voice=VOICE, split_pattern=split):
        a = r.audio  # already on CPU (KModel.forward moves it)
        if first is None:
            first = time.perf_counter() - t0
        n += len(a); chunks += 1
    sync(); tot = time.perf_counter() - t0
    return first, tot, n / st.SR, chunks

def g2p_time(text):
    t0 = time.perf_counter(); st.phonemize(pipe, text); return time.perf_counter() - t0

res = dict(device=dev, voice=VOICE, torch=torch.__version__, threads=torch.get_num_threads(),
           n_timed=N, warmup=2, lengths=[], ttfa={})
if dev == "cuda":
    res["gpu"] = torch.cuda.get_device_name()

sampler = st.GpuSampler() if dev == "cuda" else None
if sampler: sampler.__enter__()
load0 = os.getloadavg()
for L in ([] if ONLY_TTFA else LENGTHS):
    text = " ".join(words[:L])
    if not text.endswith("."): text += "."
    nph = sum(len(p) for _, p in st.phonemize(pipe, text))
    for _ in range(2): run(text)
    if dev == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ts, aud = [], None
    for _ in range(N):
        f, t, aud, ch = run(text); ts.append(t)
    gts = [g2p_time(text) for _ in range(N)]
    row = dict(words=L, chars=len(text), phonemes=nph, chunks=ch, audio_s=aud,
               t_median=statistics.median(ts), t_min=min(ts), t_max=max(ts),
               g2p_median=statistics.median(gts),
               rtf=statistics.median(ts) / aud)
    if dev == "cuda":
        row["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
        row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    res["lengths"].append(row); print(row, flush=True)

# time to first audio: whole paragraph in one forward pass vs one sentence at a time
for name, split in [("whole", None), ("sentences", r"(?<=[.!?])\s+")]:
    for _ in range(2): run(PARA, split)
    fs, ts, loads = [], [], []
    for _ in range(N):
        f, t, aud, ch = run(PARA, split); fs.append(f); ts.append(t); loads.append(os.getloadavg()[0])
    res["ttfa"][name] = dict(first_median=statistics.median(fs), first_min=min(fs), first_max=max(fs),
                             total_median=statistics.median(ts), audio_s=aud, chunks=ch, first_all=fs, load1=loads)
    print(name, res["ttfa"][name], flush=True)
res["para"] = PARA
res["para_phonemes"] = sum(len(p) for _, p in st.phonemize(pipe, PARA))
res["loadavg_start"] = load0; res["loadavg_end"] = os.getloadavg()

# determinism: same text, same voice, twice
a1 = st.synth(pipes, main[0], VOICE)["audio"]; a2 = st.synth(pipes, main[0], VOICE)["audio"]
res["determinism"] = dict(len1=len(a1), len2=len(a2), max_abs_diff=float(np.abs(a1 - a2).max()) if len(a1) == len(a2) else None)
if sampler:
    sampler.__exit__(); res["gpu_stats"] = sampler.summary()
    res["model_mib"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
res["params"] = sum(p.numel() for p in model.parameters())
res["params_by_part"] = {k: sum(p.numel() for p in getattr(model, k).parameters())
                         for k in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]}
print(json.dumps({k: v for k, v in res.items() if k not in ("lengths",)}, indent=1))
json.dump(res, open(f"runs/timing_{dev}{'_ttfa' if ONLY_TTFA else ''}.json", "w"), indent=1)
