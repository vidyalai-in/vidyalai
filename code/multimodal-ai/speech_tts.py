"""Shared helpers for Beyond Text Part 10 (text-to-speech).

Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0) with stock voices only, a few lines of
Whisper (openai/whisper-*, MIT) for ASR round trips, WER with bootstrap CIs, and a
background nvidia-smi sampler. Never used for cloning: voices are the stock packs.
"""
import os, re, json, time, threading, subprocess, statistics
import numpy as np

os.environ.setdefault("HF_HOME", "/home/amit/.cache/vidyalai-scratch/hf")
SR = 24000           # Kokoro output rate
FRAME_S = 600 / SR   # one predicted-duration frame = 25 ms
NVSMI = "/usr/lib/wsl/lib/nvidia-smi"
STOCK_VOICES = {     # name -> (lang_code, overall grade from VOICES.md)
    "af_heart": ("a", "A"), "af_bella": ("a", "A-"), "af_nicole": ("a", "B-"),
    "am_michael": ("a", "C+"), "am_fenrir": ("a", "C+"), "am_adam": ("a", "F+"),
    "bf_emma": ("b", "B-"), "bm_george": ("b", "C"),
}


# ---------------------------------------------------------------- Kokoro
def load_kokoro(device="cuda", langs=("a", "b")):
    """One KModel shared by one KPipeline per language."""
    import torch
    from kokoro import KModel, KPipeline
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    pipes = {l: KPipeline(lang_code=l, repo_id="hexgrad/Kokoro-82M", model=model) for l in langs}
    return model, pipes


def synth(pipes, text, voice, speed=1.0, split_pattern=None):
    """Synthesize text with a stock voice. Returns dict(audio float32 @24k, phonemes,
    chunks, words=[(word, start_s, end_s)], n_frames)."""
    assert voice in STOCK_VOICES, voice
    lang = STOCK_VOICES[voice][0]
    audios, phs, words, frames = [], [], [], 0
    offset = 0.0
    for r in pipes[lang](text, voice=voice, speed=speed, split_pattern=split_pattern):
        a = r.audio.numpy() if hasattr(r.audio, "numpy") else np.asarray(r.audio)
        audios.append(a); phs.append(r.phonemes)
        if r.pred_dur is not None:
            frames += int(r.pred_dur.sum())
        for t in (r.tokens or []):
            if getattr(t, "start_ts", None) is not None and t.end_ts is not None:
                words.append((t.text, offset + t.start_ts, offset + t.end_ts))
        offset += len(a) / SR
    audio = np.concatenate(audios) if audios else np.zeros(0, np.float32)
    return dict(audio=audio.astype(np.float32), phonemes=" | ".join(phs),
                chunks=len(audios), words=words, n_frames=frames)


def phonemize(pipe, text):
    """G2P only (no audio): list of (grapheme_chunk, phoneme_string)."""
    _, tokens = pipe.g2p(text)
    return [(gs, ps) for gs, ps, _ in pipe.en_tokenize(tokens)]


# ---------------------------------------------------------------- Whisper
def load_whisper(name="openai/whisper-large-v3-turbo", device="cuda"):
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(
        name, dtype=torch.float16, device_map=device).eval()
    return proc, model


def to16k(audio, sr=SR):
    import librosa
    return librosa.resample(np.asarray(audio, np.float32), orig_sr=sr, target_sr=16000)


def transcribe(proc, model, audios16k, batch=16, num_beams=1):
    """Greedy (or beam) English transcription of clips under 30 s, sampled at 16 kHz."""
    import torch
    out = []
    for i in range(0, len(audios16k), batch):
        feats = proc(audios16k[i:i + batch], sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(model.device, torch.float16)
        with torch.inference_mode():
            ids = model.generate(feats, language="en", task="transcribe",
                                 num_beams=num_beams, max_new_tokens=200)
        out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    return out


# ---------------------------------------------------------------- scoring
_NORM = None
def normalizer():
    """Whisper's English text normalizer (numbers, spelling, punctuation)."""
    global _NORM
    if _NORM is None:
        from transformers import WhisperTokenizer
        tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        _NORM = tok.normalize
    return _NORM


def basic_norm(t):
    """Lowercase and drop punctuation only (keeps digits and spellings as written)."""
    t = re.sub(r"[^\w\s']", " ", t.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", t).strip()


def wer_counts(ref, hyp, normalize=True):
    """(errors, ref_words) for one pair; errors = S + D + I.
    normalize=True: Whisper English normalizer; False: lowercase + strip punctuation."""
    import jiwer
    n = normalizer() if normalize else basic_norm
    ref, hyp = n(ref), n(hyp)
    ref = ref.strip() or "<empty>"
    o = jiwer.process_words(ref, hyp.strip() if hyp.strip() else "")
    return o.substitutions + o.deletions + o.insertions, len(ref.split())


def corpus_wer(pairs, normalize=True, n_boot=2000, seed=0):
    """Corpus WER = sum(errors)/sum(ref words) with a bootstrap 95% CI over items."""
    c = np.array([wer_counts(r, h, normalize) for r, h in pairs], float)
    e, n = c[:, 0], c[:, 1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    boots = e[idx].sum(1) / n[idx].sum(1)
    return dict(wer=float(e.sum() / n.sum()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), errors=int(e.sum()), words=int(n.sum()),
                items=len(c), items_with_error=int((e > 0).sum()))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


# ---------------------------------------------------------------- GPU sampler
class GpuSampler:
    """Samples SM clock (MHz), power (W) and memory used (MiB) every 0.2 s."""
    def __init__(self, period=0.2):
        self.period, self.rows, self._stop = period, [], threading.Event()
    def _run(self):
        while not self._stop.is_set():
            try:
                s = subprocess.run([NVSMI, "--query-gpu=clocks.sm,power.draw,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True,
                                   text=True, timeout=5).stdout.strip().split(",")
                self.rows.append([float(x) for x in s])
            except Exception:
                pass
            time.sleep(self.period)
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self.t.join()
    def summary(self):
        if not self.rows:
            return {}
        a = np.array(self.rows)
        return dict(sm_mhz_median=float(np.median(a[:, 0])), power_w_median=float(np.median(a[:, 1])),
                    power_w_max=float(a[:, 1].max()), mem_mib_max=float(a[:, 2].max()), samples=len(a))


def median_time(fn, n=7, warmup=2, cuda=True):
    """Median wall time of fn() over n runs after warm-up, synchronizing CUDA."""
    import torch
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if cuda: torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return dict(median=statistics.median(ts), min=min(ts), max=max(ts), n=n)
