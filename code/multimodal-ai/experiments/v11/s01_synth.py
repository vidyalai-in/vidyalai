"""Synthesize every Kokoro clip used in Part 11 (my own sentences, four stock voices)."""
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, soundfile as sf, torch
import voice as V
D = "/home/amit/.cache/vidyalai-scratch/v11"
pipe = V.kokoro("cuda")
meta = []
codec = [l.strip() for l in open(f"{D}/data/codec_sentences.txt") if l.strip()]
turns = [l.strip() for l in open(f"{D}/data/user_turns.txt") if l.strip()]
t0 = time.time()
for i, s in enumerate(codec):
    for v in V.STOCK_VOICES:
        a, ph = V.synth(pipe, s, v)
        sf.write(f"{D}/audio/codec_{i:02d}_{v}.wav", a, V.KOKORO_SR, subtype="FLOAT")
        meta.append({"set": "codec", "i": i, "voice": v, "text": s, "phonemes": ph, "sec": len(a) / V.KOKORO_SR})
for i, s in enumerate(turns):
    v = V.STOCK_VOICES[i % 4]
    a, ph = V.synth(pipe, s, v)
    sf.write(f"{D}/audio/turn_{i:02d}_{v}.wav", a, V.KOKORO_SR, subtype="FLOAT")
    meta.append({"set": "turn", "i": i, "voice": v, "text": s, "phonemes": ph, "sec": len(a) / V.KOKORO_SR})
json.dump(meta, open(f"{D}/runs/synth_meta.json", "w"), indent=1)
print("clips", len(meta), "audio s", round(sum(m["sec"] for m in meta), 1), "wall s", round(time.time() - t0, 1))
print("peak VRAM MB", torch.cuda.max_memory_allocated() / 2**20)
