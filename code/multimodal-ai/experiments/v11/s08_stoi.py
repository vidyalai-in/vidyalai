"""Exp 1: STOI of every codec round trip against its original (CPU). Kokoro clips at 24 kHz;
LibriSpeech clips compared at their native 16 kHz."""
import sys, json, glob
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, soundfile as sf
import voice as V
D = "/home/amit/.cache/vidyalai-scratch/v11"
which = sys.argv[1]  # kokoro or libri
res = {}
if which == "kokoro":
    meta = [m for m in json.load(open(f"{D}/runs/synth_meta.json")) if m["set"] == "codec"]
    refs = [sf.read(f"{D}/audio/codec_{m['i']:02d}_{m['voice']}.wav", dtype="float32")[0] for m in meta]
    fs = 24000
else:
    j = json.load(open(f"{D}/runs/codec_libri_mimi_dac.json"))
    L = "/home/amit/.cache/vidyalai-scratch/hf/librispeech/LibriSpeech/test-clean"
    refs = []
    for it in j["items"]:
        spk, ch, _ = it["i"].split("-")
        refs.append(sf.read(f"{L}/{spk}/{ch}/{it['i']}.flac", dtype="float32")[0])
    fs = 16000
for f in sorted(glob.glob(f"{D}/runs/codec_audio/{which}_*.npz")):
    key = f.split("/")[-1][len(which) + 1:-4]
    z = np.load(f)
    deg = [z[f"arr_{k}"].astype(np.float32) for k in range(len(z.files))]
    if fs != 24000:
        deg = [V.resample(d, 24000, fs)[: len(r)] for d, r in zip(deg, refs)]
    sc = [V.stoi_score(r, d[: len(r)], fs) for r, d in zip(refs, deg)]
    res[key] = {"stoi": [s for s, _ in sc], "lag": [l for _, l in sc]}
    print(key, round(float(np.mean(res[key]["stoi"])), 3), "lags", sorted(set(res[key]["lag"]))[:5], flush=True)
json.dump(res, open(f"{D}/runs/stoi_{which}.json", "w"))
