"""Audio clip for the article: one Kokoro af_heart sentence, then its Mimi round trip with 8 codebooks,
then with 1 codebook. Speech bounds trimmed, 0.5 s gaps, peak-normalized together to -1 dBFS."""
import sys, json
import numpy as np, soundfile as sf
D = "/home/amit/.cache/vidyalai-scratch/v11"
k = 100  # codec item index: sentence 25, voice af_heart
orig = sf.read(f"{D}/audio/codec_25_af_heart.wav", dtype="float32")[0]
m8 = np.load(f"{D}/runs/codec_audio/kokoro_mimi_8.npz")[f"arr_{k}"].astype(np.float32)
m1 = np.load(f"{D}/runs/codec_audio/kokoro_mimi_1.npz")[f"arr_{k}"].astype(np.float32)
r = np.sqrt(np.convolve(orig ** 2, np.ones(240) / 240, "same"))
on = np.where(r > r.max() * 10 ** (-40 / 20))[0]
a, b = max(0, on[0] - 1200), min(len(orig), on[-1] + 2400)
gap = np.zeros(12000, np.float32)
y = np.concatenate([orig[a:b], gap, m8[a:b], gap, m1[a:b]])
y = y / np.abs(y).max() * 10 ** (-1 / 20)
sf.write(f"{D}/runs/v11-mimi-codebooks.wav", y, 24000)
print(len(y) / 24000, "s")
