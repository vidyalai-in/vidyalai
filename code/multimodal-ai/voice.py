"""Shared helpers for Beyond Text Part 11 (audio tokens and voice agents).

Kokoro stock-voice synthesis, Whisper transcription and WER, Mimi and DAC codec round trips,
STOI, CUDA timing, an nvidia-smi sampler and bootstrap intervals. Everything runs on the
RTX 4060 Laptop GPU (8 GB) in fp16/bf16, or on the CPU where stated.
"""
import os
import re
import subprocess
import threading
import time

os.environ.setdefault("HF_HOME", "/home/amit/.cache/vidyalai-scratch/hf")
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib/wsl/lib")

import numpy as np
import torch

NVSMI = "/usr/lib/wsl/lib/nvidia-smi"
KOKORO_SR = 24_000
WHISPER_SR = 16_000
STOCK_VOICES = ["af_heart", "am_michael", "bf_emma", "bm_george"]  # Kokoro-82M stock voices


# ---------- resampling ----------
def resample(x, sr_in, sr_out):
    import torchaudio.functional as AF
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    return AF.resample(t, sr_in, sr_out).numpy()


# ---------- Kokoro ----------
def kokoro(device="cuda"):
    from kokoro import KPipeline
    return KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)


def synth(pipe, text, voice, speed=1.0):
    """Synthesize text with one stock voice. Returns (24 kHz float32 audio, phoneme strings)."""
    chunks, phon = [], []
    for res in pipe(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        chunks.append(res.audio.detach().float().cpu().numpy())
        phon.append(res.phonemes)
    return np.concatenate(chunks).astype(np.float32), phon


def synth_first_audio(pipe, text, voice, speed=1.0, sync=True):
    """Time to the first audio chunk and to the whole synthesis (seconds) for one call.
    The text is split on sentence ends so the generator yields one chunk per sentence."""
    t0 = time.perf_counter()
    first = None
    n = 0
    for res in pipe(text, voice=voice, speed=speed, split_pattern=r"(?<=[.!?])\s+"):
        a = res.audio
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = a.cpu()
        if first is None:
            first = time.perf_counter() - t0
        n += a.shape[-1]
    return first, time.perf_counter() - t0, n / KOKORO_SR


# ---------- Whisper ----------
def whisper(name="openai/whisper-large-v3-turbo", device="cuda"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float16).to(device).eval()
    return model, proc


@torch.no_grad()
def transcribe(model, proc, wavs16, batch=16, device="cuda"):
    """Greedy English transcription of a list of 16 kHz clips (each under 30 s)."""
    out = []
    for i in range(0, len(wavs16), batch):
        feats = proc.feature_extractor(wavs16[i:i + batch], sampling_rate=WHISPER_SR, return_tensors="pt")
        x = feats.input_features.to(device, torch.float16)
        ids = model.generate(x, language="en", task="transcribe", num_beams=1, do_sample=False)
        out += proc.batch_decode(ids, skip_special_tokens=True)
    return [o.strip() for o in out]


def normalizer(proc):
    return proc.tokenizer.normalize  # Whisper's English text normalizer


def wer_counts(refs, hyps, norm):
    """Per-item (errors, reference words) after normalization, for bootstrap over items."""
    import jiwer
    rows = []
    for r, h in zip(refs, hyps):
        r, h = norm(r), norm(h)
        o = jiwer.process_words(r, h if h.strip() else "")
        rows.append((o.substitutions + o.deletions + o.insertions, len(r.split())))
    return np.array(rows, dtype=float)


# ---------- codecs ----------
def mimi(device="cuda"):
    from transformers import MimiModel
    return MimiModel.from_pretrained("kyutai/mimi").to(device).eval()


def dac(device="cuda"):
    from transformers import DacModel
    return DacModel.from_pretrained("descript/dac_24khz").to(device).eval()


@torch.no_grad()
def mimi_encode(m, wav24, nq, device="cuda"):
    x = torch.as_tensor(wav24, device=device)[None, None]
    return m.encode(x, num_quantizers=nq).audio_codes  # (1, nq, frames)


@torch.no_grad()
def mimi_decode(m, codes):
    return m.decode(codes).audio_values[0, 0]


@torch.no_grad()
def dac_encode(m, wav24, nq, device="cuda"):
    x = torch.as_tensor(wav24, device=device)[None, None]
    return m.encode(x, n_quantizers=nq).audio_codes  # (1, nq_total, frames); only first nq used


@torch.no_grad()
def dac_decode(m, codes, nq):
    return m.decode(audio_codes=codes[:, :nq]).audio_values[0]


def roundtrip(kind, m, wav24, nq, device="cuda"):
    if kind == "mimi":
        c = mimi_encode(m, wav24, nq, device)
        y = mimi_decode(m, c)
    else:
        c = dac_encode(m, wav24, nq, device)
        y = dac_decode(m, c, nq)
        c = c[:, :nq]
    y = y.float().cpu().numpy().reshape(-1)[: len(wav24)]
    if len(y) < len(wav24):
        y = np.pad(y, (0, len(wav24) - len(y)))
    return y, tuple(c.shape)


def best_lag(ref, deg, max_lag=240):
    """Integer lag (samples) that best aligns deg to ref, by cross-correlation."""
    n = len(ref)
    f = np.fft.rfft(ref, 2 * n)
    g = np.fft.rfft(deg, 2 * n)
    cc = np.fft.irfft(f * np.conj(g))
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    return int(np.argmax(cc)) - max_lag


def stoi_score(ref, deg, fs=KOKORO_SR, align=False):
    """STOI of deg against ref. Reports the cross-correlation lag (within 10 ms) but only shifts by it if align."""
    from pystoi import stoi
    lag = best_lag(ref, deg)
    if not align:
        return float(stoi(ref, deg, fs, extended=False)), lag
    if lag > 0:
        deg = np.concatenate([np.zeros(lag, np.float32), deg[:-lag]])
    elif lag < 0:
        deg = np.concatenate([deg[-lag:], np.zeros(-lag, np.float32)])
    return float(stoi(ref, deg, fs, extended=False)), lag


# ---------- timing and GPU telemetry ----------
def time_cuda(fn, warmup=2, n=7):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


class GpuSampler:
    """Samples SM clock (MHz) and power (W) with nvidia-smi every 200 ms while active."""

    def __enter__(self):
        self.p = subprocess.Popen([NVSMI, "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits",
                                   "-lms", "200"], stdout=subprocess.PIPE, text=True)
        self.rows = []
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()
        return self

    def _read(self):
        for line in self.p.stdout:
            try:
                c, w = [float(v) for v in line.strip().split(",")]
                self.rows.append((c, w))
            except ValueError:
                pass

    def __exit__(self, *a):
        self.p.terminate()
        self.p.wait(timeout=5)
        self.t.join(timeout=2)

    def summary(self):
        if not self.rows:
            return {"clock_mhz": None, "power_w": None, "samples": 0}
        a = np.array(self.rows)
        return {"clock_mhz": float(np.median(a[:, 0])), "power_w": float(np.median(a[:, 1])), "samples": len(a)}


# ---------- statistics ----------
def boot_ci(x, stat=np.median, n=4000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    bs = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_wer(rows, n=4000, seed=0):
    """Corpus WER with a bootstrap 95% CI over items. rows: array of (errors, ref_words)."""
    rng = np.random.default_rng(seed)
    w = rows[:, 0].sum() / rows[:, 1].sum()
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(rows), len(rows))
        bs.append(rows[i, 0].sum() / rows[i, 1].sum())
    return float(w), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


SENT_END = re.compile(r"[.!?](?=\s|$)")
