"""Kokoro stock-voice sentence for the spectrogram figure (CPU). Saves runs/fig_audio.npz."""
import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch
import asr
TEXT = "Whisper listens to thirty seconds at a time and writes down what it heard."
VOICE = "af_heart"
a24, ph = asr.kokoro_say(TEXT, VOICE)
a16 = asr.resample(a24, 24000)
import soundfile as sf
sf.write("/home/amit/.cache/vidyalai-scratch/v09/runs/fig_sentence.wav", a16, 16000)
# STFT exactly as whisper/audio.py: n_fft 400 (25 ms), hop 160 (10 ms), Hann window
x = torch.from_numpy(a16)
st = torch.stft(x, 400, 160, window=torch.hann_window(400), return_complex=True)
mag = st[..., :-1].abs() ** 2  # (201, frames)
from transformers import WhisperFeatureExtractor
fe80 = WhisperFeatureExtractor.from_pretrained(asr.MODELS["tiny"])
fe128 = WhisperFeatureExtractor.from_pretrained(asr.MODELS["turbo"])
m80 = fe80(a16, sampling_rate=16000, return_tensors="np").input_features[0]
m128 = fe128(a16, sampling_rate=16000, return_tensors="np").input_features[0]
n = mag.shape[1]
print("samples", len(a16), "dur", len(a16)/16000, "stft", tuple(mag.shape), "mel80", m80.shape, "mel128", m128.shape, "frames speech", n)
print("phonemes:", ph)
print("feat range", float(m80.min()), float(m80.max()), "padded part value", float(m80[:, -1].mean()))
np.savez_compressed("/home/amit/.cache/vidyalai-scratch/v09/runs/fig_audio.npz", wave=a16, stft_db=10*np.log10(np.maximum(mag.numpy(), 1e-10)),
                    mel80=m80[:, :n], mel128=m128[:, :n], mel80_full=m80[:, ::10])
json.dump({"text": TEXT, "voice": VOICE, "phonemes": ph, "samples": len(a16), "seconds": len(a16)/16000,
           "stft_shape": list(mag.shape), "mel80_shape": list(m80.shape), "mel128_shape": list(m128.shape)},
          open("/home/amit/.cache/vidyalai-scratch/v09/runs/fig_audio.json", "w"), indent=1)
