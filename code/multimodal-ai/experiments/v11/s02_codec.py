"""Exp 1: Mimi and DAC round trips on Kokoro speech: codes, timing, Whisper WER (STOI computed later on CPU)."""
import sys, json, time, glob, os
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, soundfile as sf, torch
import voice as V
D = "/home/amit/.cache/vidyalai-scratch/v11"
pilot = "--pilot" in sys.argv
kinds = [a for a in ("mimi", "dac") if a in sys.argv] or ["mimi", "dac"]
libri = "--libri" in sys.argv
tag = ("libri" if libri else "kokoro") + "_" + "_".join(kinds)
if libri:
    import random
    L = "/home/amit/.cache/vidyalai-scratch/hf/librispeech/LibriSpeech/test-clean"
    items = []
    for tf in sorted(glob.glob(f"{L}/*/*/*.trans.txt")):
        for line in open(tf):
            uid, txt = line.strip().split(" ", 1)
            items.append((uid, txt, tf.rsplit("/", 1)[0] + f"/{uid}.flac"))
    random.Random(0).shuffle(items)
    meta, wavs = [], []
    for uid, txt, p in items:
        a, sr = sf.read(p, dtype="float32")
        if 3.0 <= len(a) / sr <= 12.0:
            meta.append({"i": uid, "voice": "librispeech", "text": txt, "sec": len(a) / sr})
            wavs.append(V.resample(a, 16000, 24000))
        if len(meta) == 64:
            break
else:
    meta = [m for m in json.load(open(f"{D}/runs/synth_meta.json")) if m["set"] == "codec"]
    wavs = [sf.read(f"{D}/audio/codec_{m['i']:02d}_{m['voice']}.wav", dtype="float32")[0] for m in meta]
if pilot:
    meta, wavs = meta[:8], wavs[:8]
refs = [m["text"] for m in meta]
NQ = [1, 2, 4, 8, 16, 32]
models = {k: (V.mimi() if k == "mimi" else V.dac()) for k in kinds}
wm, wp = V.whisper()
norm = V.normalizer(wp)
os.makedirs(f"{D}/runs/codec_audio", exist_ok=True)
out = {"items": [{"i": m["i"], "voice": m["voice"], "sec": m["sec"]} for m in meta], "configs": {}}

def asr(clips24):
    return V.transcribe(wm, wp, [V.resample(c, 24000, 16000) for c in clips24])

t0 = time.time()
if kinds[0] == "mimi":
    hyp = asr(wavs)
    rows = V.wer_counts(refs, hyp, norm)
    out["configs"]["original"] = {"hyp": hyp, "err": rows.tolist()}
for kind, m in models.items():
    for nq in NQ:
        dec = []
        for w in wavs:
            y, shape = V.roundtrip(kind, m, w, nq)
            dec.append(y)
        hyp = asr(dec)
        rows = V.wer_counts(refs, hyp, norm)
        key = f"{kind}_{nq}"
        np.savez_compressed(f"{D}/runs/codec_audio/{tag.split('_')[0]}_{key}.npz", *[d.astype(np.float16) for d in dec])
        out["configs"][key] = {"hyp": hyp, "err": rows.tolist(), "code_shape_last": shape}
        print(key, shape, "WER", round(rows[:, 0].sum() / rows[:, 1].sum() * 100, 2), round(time.time() - t0, 1), flush=True)

# timing on a 10.0 s clip (two sentences joined), encode and decode separately
x10 = np.concatenate(wavs[:4])[: 10 * 24000]
timing = {}
if libri: models = {}
for kind, m in models.items():
    for nq in NQ:
        if kind == "mimi":
            enc = lambda: V.mimi_encode(m, x10, nq)
            codes = enc()
            dec = lambda: V.mimi_decode(m, codes)
        else:
            enc = lambda: V.dac_encode(m, x10, nq)
            codes = enc()
            dec = lambda: V.dac_decode(m, codes, nq)
        with V.GpuSampler() as g:
            te = V.time_cuda(enc, warmup=3, n=10)
            td = V.time_cuda(dec, warmup=3, n=10)
        timing[f"{kind}_{nq}"] = {"enc_ms": [t * 1e3 for t in te], "dec_ms": [t * 1e3 for t in td],
                                  "codes_shape": list(codes[:, :nq].shape), **g.summary()}
        print(kind, nq, "enc med", round(np.median(te) * 1e3, 1), "dec med", round(np.median(td) * 1e3, 1), g.summary(), flush=True)
out["timing_10s"] = timing
out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
out["versions"] = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
out["refs"] = refs
json.dump(out, open(f"{D}/runs/codec_{tag}{'_pilot' if pilot else ''}.json", "w"))
print("done", round(time.time() - t0, 1), "s, peak MB", out["peak_vram_mb"])
