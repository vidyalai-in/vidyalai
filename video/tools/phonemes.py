"""IPA phonemes from speech, to check or match pronunciations (e.g. against an owner's recording).

  uv run --python 3.12 --with transformers --with torch --with torchaudio --with soundfile \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    python video/tools/phonemes.py recording.wav video/.tts-cache/<clip>.wav

Uses facebook/wav2vec2-lv-60-espeak-cv-ft with greedy CTC decoding (no espeak install needed).
Any sample rate or channel count is fine; MP3s need converting to WAV first (ffmpeg -i in.mp3 out.wav).
tts.py imports PhonemeRecognizer to verify brand-name pronunciations in cloned-voice takes.
"""

import difflib
import json
import sys

MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


class PhonemeRecognizer:
    def __init__(self, device: str = "cpu"):
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(device).eval()
        self.vocab = {v: k for k, v in json.load(open(hf_hub_download(MODEL, "vocab.json"))).items()}

    def __call__(self, audio, sr: int) -> list[str]:
        """Phonemes of a mono float32 signal."""
        import torch
        import torchaudio

        x = torch.as_tensor(audio, dtype=torch.float32)
        if sr != 16000:
            x = torchaudio.functional.resample(x, sr, 16000)
        inputs = self.fe(x.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            ids = self.model(inputs).logits.argmax(-1)[0].tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and self.vocab[i] not in ("<pad>", "<s>", "</s>", "<unk>", "|"):
                out.append(self.vocab[i])
            prev = i
        return out


def contains(heard: list[str], target: list[str]) -> float:
    """Best similarity (0-1) of `target` against any same-length window of `heard`."""
    n = len(target)
    windows = [heard[i : i + n] for i in range(max(1, len(heard) - n + 1))]
    return max(difflib.SequenceMatcher(None, w, target).ratio() for w in windows)


if __name__ == "__main__":
    import soundfile as sf

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rec = PhonemeRecognizer()
    for path in sys.argv[1:]:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        print(f"{path.split('/')[-1]:28} {' '.join(rec(audio, sr))}")
