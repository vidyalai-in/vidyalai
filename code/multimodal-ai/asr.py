"""Speech recognition helpers for Beyond Text Part 9 (v09, Whisper).

Models (Hugging Face, loaded in fp16 straight to the GPU):
  tiny  openai/whisper-tiny            39M   Apache-2.0 (HF card)
  base  openai/whisper-base            74M   Apache-2.0 (HF card)
  small openai/whisper-small          244M   Apache-2.0 (HF card)
  turbo openai/whisper-large-v3-turbo 809M   MIT (HF card)
Data: LibriSpeech test-clean / test-other (openslr.org/12, CC BY 4.0), used only to measure.
Voice shown in figures: Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0) stock voices only.

Contents
- librispeech(split): all utterances of a split as dicts (id, speaker, path, ref, n_words).
- pick_subset(utts, n, seed): speaker-stratified deterministic subset (round robin over speakers).
- load_flac(path): float32 mono 16 kHz.
- load_whisper(name): (processor, model) in fp16 on cuda, SDPA attention.
- transcribe(proc, model, audios, batch_size, **gen): texts plus timing split (features, generate).
- NORMALIZERS: "raw" (whitespace only), "lower" (lowercase only), "basic" (Whisper's BasicTextNormalizer: lowercase, drop
  punctuation and bracketed text), "english" (Whisper's EnglishTextNormalizer with its spelling map).
- word_errors(refs, hyps): per-utterance (S, D, I, N) from jiwer.
- wer(counts), boot_wer(counts): corpus WER and a bootstrap 95% CI over utterances.
- mix_at_snr(speech, noise, snr_db): add noise scaled to a target signal-to-noise ratio.
- white_noise(n, rng), babble(n, sources, rng, talkers): noise signals.
- GpuMonitor, cuda_median, wilson: measurement helpers.
- kokoro_say(text, voice): Kokoro stock-voice speech at 24 kHz (numpy float32).
"""
from __future__ import annotations

import glob
import math
import os
import statistics
import subprocess
import time

import numpy as np

os.environ.setdefault("HF_HOME", "/home/amit/.cache/vidyalai-scratch/hf")
LIBRI = "/home/amit/.cache/vidyalai-scratch/hf/librispeech/LibriSpeech"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
SR = 16000
MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "turbo": "openai/whisper-large-v3-turbo",
}
PARAMS_M = {"tiny": 39, "base": 74, "small": 244, "turbo": 809}


# ---------------------------------------------------------------- data
def librispeech(split: str):
    utts = []
    for trans in sorted(glob.glob(f"{LIBRI}/{split}/*/*/*.trans.txt")):
        d = os.path.dirname(trans)
        for line in open(trans):
            uid, text = line.strip().split(" ", 1)
            utts.append({"id": uid, "speaker": uid.split("-")[0], "path": f"{d}/{uid}.flac",
                         "ref": text, "n_words": len(text.split())})
    return utts


def pick_subset(utts, n: int, seed: int = 0, exclude=()):
    """Speaker-stratified subset: shuffle each speaker's utterances with a fixed seed, then take
    one per speaker in turn until n are chosen. Every speaker gets n/speakers utterances (+-1)."""
    rng = np.random.default_rng(seed)
    ex = set(exclude)
    by = {}
    for u in utts:
        if u["id"] not in ex:
            by.setdefault(u["speaker"], []).append(u)
    spk = sorted(by)
    for s in spk:
        idx = rng.permutation(len(by[s]))
        by[s] = [by[s][i] for i in idx]
    out, r = [], 0
    while len(out) < n:
        for s in spk:
            if r < len(by[s]) and len(out) < n:
                out.append(by[s][r])
        r += 1
    return out


def load_flac(path: str) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, sr
    return x


# ---------------------------------------------------------------- model
def load_whisper(name: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    rid = MODELS[name]
    proc = WhisperProcessor.from_pretrained(rid)
    model = WhisperForConditionalGeneration.from_pretrained(
        rid, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    return proc, model


def transcribe(proc, model, audios, batch_size: int = 16, **gen):
    """Short-form transcription (every clip <= 30 s). Clips are sorted by length inside the call so
    each batch decodes similar lengths; outputs come back in input order.
    Returns (texts, info) with info = {"feat_s", "gen_s", "audio_s", "n_tokens"}."""
    import torch
    gen.setdefault("language", "en")
    gen.setdefault("task", "transcribe")
    order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
    texts = [None] * len(audios)
    feat_s = gen_s = 0.0
    n_tok = 0
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        t0 = time.perf_counter()
        f = proc.feature_extractor([audios[i] for i in idx], sampling_rate=SR, return_tensors="pt")
        x = f.input_features.to("cuda", torch.float16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(x, **gen)
        seq = out if isinstance(out, torch.Tensor) else out["sequences"]
        seq = seq.long()  # transformers 5.21 returns float ids when a no-speech segment is skipped
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dec = proc.batch_decode(seq, skip_special_tokens=True)
        for j, i in enumerate(idx):
            texts[i] = dec[j].strip()
        n_tok += int((seq != proc.tokenizer.pad_token_id).sum())
        feat_s += t1 - t0
        gen_s += t2 - t1
    return texts, {"feat_s": feat_s, "gen_s": gen_s, "audio_s": sum(len(a) for a in audios) / SR,
                   "n_tokens": n_tok}


# ---------------------------------------------------------------- text and WER
def _english_normalizer():
    import json
    from huggingface_hub import hf_hub_download
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    spelling = json.load(open(hf_hub_download(MODELS["tiny"], "normalizer.json")))
    return EnglishTextNormalizer(spelling)


def _basic_normalizer():
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    return BasicTextNormalizer()


_NORM = {}


def normalize(text: str, kind: str) -> str:
    if kind == "raw":
        return " ".join(text.split())
    if kind == "lower":
        return " ".join(text.lower().split())
    if kind not in _NORM:
        _NORM[kind] = _english_normalizer() if kind == "english" else _basic_normalizer()
    return " ".join(_NORM[kind](text).split())


NORMALIZERS = ("raw", "lower", "basic", "english")


def word_errors(refs, hyps):
    """Per-utterance (substitutions, deletions, insertions, reference words). An empty reference
    counts every hypothesis word as an insertion."""
    import jiwer
    rows = []
    for r, h in zip(refs, hyps):
        n = len(r.split())
        if n == 0:
            rows.append((0, 0, len(h.split()), 0))
            continue
        if not h.strip():
            rows.append((0, n, 0, n))
            continue
        o = jiwer.process_words(r, h)
        rows.append((o.substitutions, o.deletions, o.insertions, n))
    return np.array(rows, dtype=np.int64)


def wer(c) -> float:
    c = np.asarray(c)
    return float(c[:, :3].sum() / c[:, 3].sum())


def boot_wer(c, B: int = 2000, seed: int = 0):
    """Corpus WER with a percentile bootstrap 95% CI, resampling utterances."""
    c = np.asarray(c)
    rng = np.random.default_rng(seed)
    e = c[:, :3].sum(1)
    n = c[:, 3]
    idx = rng.integers(0, len(c), size=(B, len(c)))
    w = e[idx].sum(1) / n[idx].sum(1)
    return wer(c), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))


def boot_diff(c1, c2, B: int = 2000, seed: int = 0):
    """Paired bootstrap of WER(c1) - WER(c2) over the same utterances."""
    c1, c2 = np.asarray(c1), np.asarray(c2)
    rng = np.random.default_rng(seed)
    e1, e2, n = c1[:, :3].sum(1), c2[:, :3].sum(1), c1[:, 3]
    idx = rng.integers(0, len(c1), size=(B, len(c1)))
    d = (e1[idx].sum(1) - e2[idx].sum(1)) / n[idx].sum(1)
    return wer(c1) - wer(c2), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- noise
def power(x) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def mix_at_snr(speech, noise, snr_db: float):
    """speech + noise scaled so 10*log10(P_speech / P_noise) = snr_db (powers over the whole clip,
    as in the Whisper paper's noise test)."""
    noise = noise[:len(speech)]
    g = math.sqrt(power(speech) / (power(noise) * 10 ** (snr_db / 10)))
    return (speech + g * noise).astype(np.float32)


def white_noise(n: int, rng) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def babble(n: int, sources, rng, talkers: int = 6) -> np.ndarray:
    """Sum of `talkers` speech clips (each scaled to unit power, random start, looped to length n)."""
    out = np.zeros(n, dtype=np.float64)
    for k in rng.choice(len(sources), size=talkers, replace=False):
        s = np.asarray(sources[k], dtype=np.float64)
        s = s / math.sqrt(power(s))
        start = int(rng.integers(0, len(s)))
        s = np.roll(s, -start)
        reps = int(math.ceil(n / len(s)))
        out += np.tile(s, reps)[:n]
    return out.astype(np.float32)


# ---------------------------------------------------------------- GPU measurement
class GpuMonitor:
    """Context manager: sample SM clock (MHz) and power (W) every 200 ms with nvidia-smi."""

    def __enter__(self):
        self.p = subprocess.Popen([SMI, "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                                   "--format=csv,noheader,nounits", "-lms", "200"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        time.sleep(0.3)
        return self

    def __exit__(self, *a):
        self.p.terminate()
        out, _ = self.p.communicate(timeout=5)
        rows = []
        for line in out.strip().splitlines():
            try:
                rows.append([float(v) for v in line.split(",")])
            except ValueError:
                pass
        rows = rows[2:] or rows
        arr = np.array(rows) if rows else np.zeros((1, 3))
        self.clock = float(np.median(arr[:, 0]))
        self.power = float(np.median(arr[:, 1]))
        self.temp = float(np.median(arr[:, 2]))
        self.n = len(rows)


def cuda_median(fn, n: int = 5, warmup: int = 2):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


# ---------------------------------------------------------------- Kokoro (stock voices only)
_KP = {}


def kokoro_say(text: str, voice: str = "af_heart", speed: float = 1.0, device: str = "cpu"):
    """Kokoro-82M stock voice, 24 kHz float32. Returns (audio, phonemes)."""
    from kokoro import KPipeline
    lang = voice[0]
    if lang not in _KP:
        _KP[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)
    chunks, phon = [], []
    for r in _KP[lang](text, voice=voice, speed=speed):
        chunks.append(np.asarray(r.audio, dtype=np.float32))
        phon.append(r.phonemes)
    return np.concatenate(chunks), " ".join(phon)


def resample(x, sr_in: int, sr_out: int = SR):
    import librosa
    return librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=sr_in, target_sr=sr_out)
