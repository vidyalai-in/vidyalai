"""Exp 2-4 synthesis: 200 public-domain sentences and 60 hard cases with 6 stock voices,
a speed sweep, and the spectrogram/audio-clip sentences. Saves 16 kHz int16 audio for ASR."""
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch, soundfile as sf
import speech_tts as st

VOICES = ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bm_george"]
SPEED_VOICES = ["af_heart", "am_michael"]
SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
main = [json.loads(l) for l in open("texts/main.jsonl")]
hard = [json.loads(l) for l in open("texts/hard.jsonl")]
model, pipes = st.load_kokoro("cuda")
meta, audio16 = [], {}

def do(set_name, items, voice, speed=1.0):
    for it in items:
        r = st.synth(pipes, it["text"], voice, speed=speed)
        key = f"{set_name}|{voice}|{speed}|{it['id']}"
        a16 = st.to16k(r["audio"])
        audio16[key] = (np.clip(a16, -1, 1) * 32767).astype(np.int16)
        meta.append(dict(key=key, set=set_name, voice=voice, speed=speed, id=it["id"],
                         text=it["text"], phonemes=r["phonemes"], n_phonemes=len(r["phonemes"]),
                         chunks=r["chunks"], dur_s=len(r["audio"]) / st.SR, n_frames=r["n_frames"]))

t0 = time.time()
for v in VOICES:
    do("main", main, v); do("hard", hard, v); print(v, len(meta), round(time.time() - t0, 1), flush=True)
for v in SPEED_VOICES:
    for s in SPEEDS:
        if s != 1.0:
            do("speed", main[:50], v, s)
    print("speed", v, len(meta), round(time.time() - t0, 1), flush=True)

# figure sentence and the two article clips (24 kHz wav, converted to MP3 later)
fig = {}
_f0 = {}
_orig = model.predictor.F0Ntrain
def _wrap(x, s):
    F0, N = _orig(x, s); _f0["F0"] = F0.squeeze().float().cpu().numpy().tolist(); return F0, N
model.predictor.F0Ntrain = _wrap
_pd = {}
_orig_fwt = model.forward_with_tokens
def _fwt(ids, ref_s, speed=1):
    a, d = _orig_fwt(ids, ref_s, speed); _pd["dur"] = d.cpu().numpy().tolist(); return a, d
model.forward_with_tokens = _fwt
for name, text, voice in [("record", "Please record the record for the archive.", "af_heart"),
                          ("abbrev", "Gen. Ortiz met Lt. Brown at 42 Oak St. on Tue., Nov. 18.", "am_michael")]:
    r = st.synth(pipes, text, voice)
    sf.write(f"runs/clip_{name}.wav", r["audio"], st.SR)
    fig[name] = dict(text=text, voice=voice, speed=1.0, phonemes=r["phonemes"], words=r["words"],
                     dur_s=len(r["audio"]) / st.SR, n_frames=r["n_frames"],
                     f0_pred=_f0.get("F0"), pred_dur=_pd.get("dur"))
    print(name, fig[name], flush=True)
json.dump(fig, open("runs/fig_clips.json", "w"), ensure_ascii=False, indent=1)
np.savez("runs/audio16k.npz", **{k.replace("|", "__"): v for k, v in audio16.items()})
with open("runs/synth_meta.jsonl", "w") as f:
    for m in meta: f.write(json.dumps(m, ensure_ascii=False) + "\n")
print("done", len(meta), "clips", round(time.time() - t0, 1), "s; peak MiB",
      torch.cuda.max_memory_allocated() / 2**20)
