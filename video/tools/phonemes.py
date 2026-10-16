"""IPA phonemes from speech, to check or match pronunciations (e.g. against an owner's recording).

  uv run --python 3.12 --with transformers --with torch --with torchaudio --with soundfile \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    python video/tools/phonemes.py recording.wav video/.tts-cache/<clip>.wav

Uses facebook/wav2vec2-lv-60-espeak-cv-ft with greedy CTC decoding (no espeak install needed).
Any sample rate or channel count is fine; MP3s need converting to WAV first (ffmpeg -i in.mp3 out.wav).
"""
import sys, json, torch, soundfile as sf, torchaudio
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
fe = Wav2Vec2FeatureExtractor.from_pretrained(name); model = Wav2Vec2ForCTC.from_pretrained(name).eval()
vocab = {v: k for k, v in json.load(open(hf_hub_download(name, "vocab.json"))).items()}
def phonemes(path):
    audio, sr = sf.read(path, dtype="float32")
    audio = torch.tensor(audio.mean(1) if audio.ndim > 1 else audio)
    if sr != 16000: audio = torchaudio.functional.resample(audio, sr, 16000)
    with torch.no_grad():
        ids = model(fe(audio.numpy(), sampling_rate=16000, return_tensors="pt").input_values).logits.argmax(-1)[0].tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and vocab[i] not in ("<pad>", "<s>", "</s>", "<unk>"): out.append(vocab[i])
        prev = i
    return " ".join(out).replace("|", "/")
for p in sys.argv[1:]: print(f"{p.split('/')[-1]:28} {phonemes(p)}")
