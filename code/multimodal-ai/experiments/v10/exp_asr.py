"""Exp 2-3 ASR: transcribe every synthesized clip with Whisper large-v3-turbo (fp16, greedy),
plus 200 LibriSpeech test-clean utterances as a real-speech reference (measurement only)."""
import sys, json, time, glob, random, os
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch, soundfile as sf
import speech_tts as st

proc, model = st.load_whisper("openai/whisper-large-v3-turbo")
meta = [json.loads(l) for l in open("runs/synth_meta.jsonl")]
npz = np.load("runs/audio16k.npz")
done = set()
out_path = "runs/asr.jsonl"
if os.path.exists(out_path):
    done = {json.loads(l)["key"] for l in open(out_path)}
todo = [m for m in meta if m["key"] not in done]
todo.sort(key=lambda m: m["dur_s"])          # similar lengths per batch
t0 = time.time(); audio_s = 0.0
with open(out_path, "a") as f, st.GpuSampler() as gs:
    B = 16
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        auds = [npz[m["key"].replace("|", "__")].astype(np.float32) / 32767 for m in chunk]
        hyps = st.transcribe(proc, model, auds, batch=B)
        for m, h in zip(chunk, hyps):
            f.write(json.dumps(dict(key=m["key"], hyp=h), ensure_ascii=False) + "\n")
        audio_s += sum(len(a) for a in auds) / 16000
        if i % (B * 20) == 0:
            print(i, len(todo), round(time.time() - t0, 1), flush=True)
synth_stats = dict(clips=len(todo), audio_s=audio_s, wall_s=time.time() - t0, gpu=gs.summary())
print("synth clips", synth_stats, flush=True)

# LibriSpeech test-clean reference: 200 utterances drawn at random (seed 0), never shown or saved
root = "/home/amit/.cache/vidyalai-scratch/hf/librispeech/LibriSpeech/test-clean"
trans = {}
for tf in glob.glob(f"{root}/*/*/*.trans.txt"):
    for line in open(tf):
        k, t = line.strip().split(" ", 1); trans[k] = t
keys = sorted(trans); random.Random(0).shuffle(keys); keys = keys[:200]
auds = []
for k in keys:
    s, c, _ = k.split("-")
    a, sr = sf.read(f"{root}/{s}/{c}/{k}.flac", dtype="float32"); assert sr == 16000
    auds.append(a)
order = np.argsort([len(a) for a in auds])
hyps = [None] * len(auds)
t1 = time.time()
for i in range(0, len(order), 16):
    idx = order[i:i + 16]
    for j, h in zip(idx, st.transcribe(proc, model, [auds[j] for j in idx], batch=16)):
        hyps[j] = h
ls_stats = dict(n=len(keys), audio_s=sum(len(a) for a in auds) / 16000, wall_s=time.time() - t1,
                speakers=len({k.split('-')[0] for k in keys}))
# keep only scores, ids and per-utterance error counts; no transcripts are saved
rows = []
for k, h in zip(keys, hyps):
    e, n = st.wer_counts(trans[k], h, True); e_raw, n_raw = st.wer_counts(trans[k], h, False)
    rows.append(dict(id=k, err=e, words=n, err_raw=e_raw, words_raw=n_raw))
pairs = [(trans[k], h) for k, h in zip(keys, hyps)]
ls_stats["wer_norm"] = st.corpus_wer(pairs, True); ls_stats["wer_raw"] = st.corpus_wer(pairs, False)
print("librispeech", ls_stats, flush=True)
json.dump(dict(synth=synth_stats, librispeech=ls_stats, ls_rows=rows), open("runs/asr_stats.json", "w"), indent=1)
