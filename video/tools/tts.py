"""Narration audio for the article-to-video pipeline. Two engines:

  kokoro      Kokoro-82M stock voices (Apache 2.0). Deterministic, very fast.
  chatterbox  Chatterbox (Resemble AI, MIT) cloning a voice from a reference recording. Sampling is
              random, so each sentence is seeded, transcribed with Whisper, and retaken with a new
              seed when the words don't match the script (cloning models occasionally skip, repeat
              or garble words).

Commands (driven by JSON files that build.mjs writes):

  tts.py synth MANIFEST.json   Synthesize each sentence to its own WAV (cached by text + engine
                               settings) and write durations back into MANIFEST.json.
  tts.py mix TIMELINE.json     Place sentence WAVs at their start times into one WAV track.
  tts.py bed BED.json          Synthesize the music bed and the sound-effects track (see bed()).

Device: --device auto (default) uses whichever of CUDA and CPU is faster (Kokoro benchmarks both;
Chatterbox always prefers CUDA). build.mjs runs this through uv with the right dependencies.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

SAMPLE_RATE = 24000


def file_hash(path: str) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


# Settings that only decide which take is kept don't change the audio, so they stay out of the key.
SELECTION_ONLY = ("reference", "checks", "retakes", "max_wer", "asr")


def cache_key(engine: dict, text: str) -> str:
    settings = {k: v for k, v in engine.items() if k not in SELECTION_ONLY}
    if engine.get("reference"):
        settings["reference"] = file_hash(engine["reference"])
    return hashlib.sha1(f"{json.dumps(settings, sort_keys=True)}|{text}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------- Kokoro
class Kokoro:
    def __init__(self, engine: dict, device: str):
        import torch
        from kokoro import KModel, KPipeline

        self.voice, self.speed = engine["voice"], float(engine.get("speed", 1.0))
        lang = engine.get("lang", "a")

        def build(dev):
            model = KModel(repo_id="hexgrad/Kokoro-82M").to(dev).eval()
            return KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model)

        if device != "auto":
            self.pipe, self.device = build(device), device
        elif not torch.cuda.is_available():
            self.pipe, self.device = build("cpu"), "cpu"
        else:
            # Time one warmed-up sentence on each device and keep the faster.
            sample = "Treat a retrieval system as the foundation, and measure it before you tune the model."
            timings, pipes = {}, {}
            for dev in ("cuda", "cpu"):
                pipe = build(dev)
                for _ in pipe(sample, voice=self.voice):
                    pass
                start = time.perf_counter()
                for _ in pipe(sample, voice=self.voice):
                    pass
                timings[dev], pipes[dev] = time.perf_counter() - start, pipe
            self.device = min(timings, key=timings.get)
            self.pipe = pipes[self.device]
            print(f"tts: kokoro device timings {', '.join(f'{d}={t:.2f}s' for d, t in timings.items())}; using {self.device}", flush=True)

    def say(self, text: str, voice: str | None = None):
        """Audio plus per-word timings: [[token, start_s, end_s], ...] from Kokoro's duration model."""
        import numpy as np

        chunks, words, offset = [], [], 0.0
        for r in self.pipe(text, voice=voice or self.voice, speed=self.speed):
            if r.audio is None:
                continue
            audio = r.audio.cpu().numpy() if hasattr(r.audio, "cpu") else np.asarray(r.audio)
            for tk in r.tokens or []:
                if tk.start_ts is not None and tk.end_ts is not None and re.search(r"\w", tk.text):
                    words.append([tk.text, round(offset + tk.start_ts, 3), round(offset + tk.end_ts, 3)])
            offset += len(audio) / SAMPLE_RATE
            chunks.append(audio)
        return np.concatenate(chunks), {"words": words}


# ---------------------------------------------------------------- Chatterbox
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
LETTER_NAMES = {"bee": "b", "dee": "d", "eff": "f", "gee": "g", "aitch": "h", "jay": "j", "kay": "k", "em": "m",
                "pee": "p", "cue": "q", "ess": "s", "tee": "t", "vee": "v", "ex": "x", "zee": "z"}


def normalize_words(text: str) -> list[str]:
    """Lowercase words with numbers and common units spelled out, for comparing ASR to the script."""
    from num2words import num2words

    t = text.lower().replace("%", " percent ")
    t = re.sub(r"(\d)\s*gb\b", r"\1 gigabytes", t)
    t = re.sub(r"(\d)\s*ms\b", r"\1 milliseconds", t)
    t = NUM_RE.sub(lambda m: " " + num2words(float(m.group().replace(",", "")) if "." in m.group() else int(m.group().replace(",", ""))) + " ", t)
    t = re.sub(r"[^a-z' ]+", " ", t.replace("-", " "))
    # Spelled-out letters ("S S Dee") and acronyms ("SSD") compare equal: map unambiguous letter
    # names to letters, then merge runs of single letters into one token.
    merged, run = [], False
    for w in (LETTER_NAMES.get(w, w) for w in t.split()):
        single = len(w) == 1 and w not in ("a", "i")
        if single and run:
            merged[-1] += w
        else:
            merged.append(w)
        run = single
    return merged


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


class Chatterbox:
    def __init__(self, engine: dict, device: str):
        import numpy as np
        import torch

        self.torch = torch
        self.engine = engine
        self.device = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        self.turbo = engine.get("model") == "turbo"
        if self.turbo:
            import chatterbox.tts_turbo as turbo_mod

            # Turbo feeds librosa's float64 output straight into a float32 model; cast it.
            load, resample = turbo_mod.librosa.load, turbo_mod.librosa.resample
            turbo_mod.librosa.load = lambda *a, **k: (lambda r: (r[0].astype(np.float32), r[1]))(load(*a, **k))
            turbo_mod.librosa.resample = lambda *a, **k: resample(*a, **k).astype(np.float32)
            self.model = turbo_mod.ChatterboxTurboTTS.from_pretrained(device=self.device)
            self.model.prepare_conditionals(engine["reference"])
            self.kwargs = {k: float(engine[k]) for k in ("temperature",) if k in engine}
        else:
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.model.prepare_conditionals(engine["reference"], exaggeration=float(engine.get("exaggeration", 0.5)))
            self.kwargs = {k: float(engine[k]) for k in ("exaggeration", "cfg_weight", "temperature") if k in engine}
        assert self.model.sr == SAMPLE_RATE, f"Chatterbox sample rate {self.model.sr}"
        self.max_wer = float(engine.get("max_wer", 0.12))
        self.retakes = int(engine.get("retakes", 6))
        # Pronunciation checks: when a sentence contains `say`, its phonemes must contain `ipa`.
        self.checks = engine.get("checks", [])
        from transformers import pipeline

        self.asr = pipeline("automatic-speech-recognition", model=engine.get("asr", "openai/whisper-small.en"), device=0 if self.device == "cuda" else -1)
        self.phonemes = None
        if self.checks:
            sys.path.insert(0, str(Path(__file__).parent))
            from phonemes import PhonemeRecognizer, contains

            self.phonemes, self.contains = PhonemeRecognizer(self.device), contains
        print(f"tts: chatterbox {'turbo' if self.turbo else 'standard'} on {self.device}, up to {self.retakes} takes per sentence", flush=True)

    def score(self, text: str, audio) -> tuple[float, dict]:
        """0 is perfect. Word errors (Whisper) plus a penalty for each failed pronunciation check."""
        import difflib

        heard = self.asr({"raw": audio, "sampling_rate": SAMPLE_RATE})["text"]
        ref, hyp = normalize_words(text), normalize_words(heard)
        info = {"heard": heard.strip()}
        penalty = 0.0
        for check in self.checks:
            if check["say"] not in text:
                continue
            # The respelled brand name is verified by phonemes, so drop it from the word comparison:
            # remove its words from the script, and the closest-sounding heard word from the transcript.
            for w in normalize_words(check["say"]):
                if w in ref:
                    ref.remove(w)
            if hyp:
                target = "".join(normalize_words(check["say"]))
                hyp.remove(max(hyp, key=lambda h: difflib.SequenceMatcher(None, h, target).ratio()))
            similarity = self.contains(self.phonemes(audio, SAMPLE_RATE), check["ipa"].split())
            info.setdefault("pronunciation", {})[check["say"]] = round(similarity, 3)
            if similarity < float(check.get("min", 0.9)):
                penalty += 1.0
        info["wer"] = round(wer(ref, hyp), 3)
        return info["wer"] + penalty, info

    def say(self, text: str, voice: str | None = None):
        best = None
        for take in range(self.retakes):
            seed = int(hashlib.sha1(f"{text}|{take}".encode()).hexdigest()[:8], 16)
            self.torch.manual_seed(seed)
            audio = finish(self.model.generate(text, **self.kwargs).squeeze(0).cpu().numpy(), self.engine)
            score, info = self.score(text, audio)
            if best is None or score < best[0]:
                best = (score, audio, {**info, "take": take})
            if score <= self.max_wer:
                break
        score, audio, info = best
        if score > self.max_wer:
            print(f"tts: WARNING best of {self.retakes} takes scored {score:.2f}: {text!r} -> {info}", flush=True)
        return audio, info


def finish(audio, engine: dict):
    """Optional per-clip polish from the voice profile: trim dead air, then speed up without
    changing pitch (ffmpeg atempo, WSOLA). Applied to every take before it is scored."""
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf

    if engine.get("trim"):
        import librosa

        _, (start, end) = librosa.effects.trim(audio, top_db=40)
        pad = int(0.04 * SAMPLE_RATE)
        audio = audio[max(0, start - pad) : min(len(audio), end + pad)]
    tempo = float(engine.get("tempo", 1.0))
    if abs(tempo - 1.0) > 1e-3:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = f"{tmp}/in.wav", f"{tmp}/out.wav"
            sf.write(src, audio, SAMPLE_RATE)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", f"atempo={tempo}", dst], check=True)
            audio = sf.read(dst, dtype="float32")[0]
    return np.ascontiguousarray(audio, dtype=np.float32)


ENGINES = {"kokoro": Kokoro, "chatterbox": Chatterbox}


def synth(manifest_path: Path, device: str):
    import soundfile as sf

    manifest = json.loads(manifest_path.read_text())
    engine = manifest["engine"]
    cache = Path(manifest["cacheDir"])
    cache.mkdir(parents=True, exist_ok=True)

    def engine_for(item):
        # A step can be read by a second Kokoro voice (`speaker:` in script.yaml).
        return {**engine, "voice": item["voice"]} if item.get("voice") else engine

    todo = []
    for item in manifest["items"]:
        item["wav"] = str(cache / f"{cache_key(engine_for(item), item['tts'])}.wav")
        info = Path(item["wav"]).with_suffix(".json")
        stale = False
        if info.exists():
            saved = json.loads(info.read_text())
            scores = saved.get("pronunciation", {})
            stale = any(scores.get(c["say"], 1) < float(c.get("min", 0.9)) for c in engine.get("checks", []))
            stale = stale or (engine["name"] == "kokoro" and "words" not in saved)
        elif engine["name"] == "kokoro":
            stale = True  # clips cached before word timings existed
        if stale or not Path(item["wav"]).exists():
            todo.append(item)

    label = engine.get("voice") or Path(engine.get("reference", "?")).parent.name
    print(f"tts: {len(manifest['items'])} sentences, {len(todo)} to synthesize ({engine['name']}, {label})", flush=True)
    if todo:
        tts = ENGINES[engine["name"]](engine, device)
        start = time.perf_counter()
        for i, item in enumerate(todo, 1):
            audio, info = tts.say(item["tts"], item.get("voice"))
            info = {k: v for k, v in info.items() if v is not None}
            sf.write(item["wav"], audio, SAMPLE_RATE)
            if info:
                Path(item["wav"]).with_suffix(".json").write_text(json.dumps(info))
            if i % 10 == 0 or i == len(todo):
                print(f"tts: {i}/{len(todo)} ({time.perf_counter() - start:.0f}s)", flush=True)

    durations = {}
    for item in manifest["items"]:
        info = Path(item["wav"]).with_suffix(".json")
        durations[item["id"]] = {
            "wav": item["wav"],
            "seconds": sf.info(item["wav"]).duration,
            **(json.loads(info.read_text()) if info.exists() else {}),
        }
    manifest["durations"] = durations
    manifest_path.write_text(json.dumps(manifest, indent=1))


def mix(timeline_path: Path):
    import numpy as np
    import soundfile as sf

    timeline = json.loads(timeline_path.read_text())
    total = int(round(timeline["totalSeconds"] * SAMPLE_RATE))
    track = np.zeros(total, dtype=np.float32)
    for clip in timeline["clips"]:
        audio, sr = sf.read(clip["wav"], dtype="float32")
        assert sr == SAMPLE_RATE, f"{clip['wav']}: {sr} Hz"
        # 5 ms fades so clip edges can't click.
        fade = min(len(audio) // 2, int(0.005 * SAMPLE_RATE))
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            audio[:fade] *= ramp
            audio[-fade:] *= ramp[::-1]
        start = int(round(clip["start"] * SAMPLE_RATE))
        end = min(total, start + len(audio))
        track[start:end] += audio[: end - start]
    sf.write(timeline["out"], np.clip(track, -1, 1), SAMPLE_RATE)
    print(f"tts: mixed {len(timeline['clips'])} clips into {timeline['out']} ({timeline['totalSeconds']:.1f}s)", flush=True)


# ---------------------------------------------------------------- Music bed and sound effects
BED_RATE = 48000


def midi_hz(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def reverb(x, seconds: float, rng, decay: float = 2.2):
    """Convolve with exponentially decaying, softened noise (a plain room tail), by overlap-add."""
    import numpy as np

    n = int(seconds * BED_RATE)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / BED_RATE * (6.9 / decay))
    ir = np.convolve(ir, np.ones(24) / 24, mode="same")  # take the fizz off the tail
    ir /= np.sqrt(np.sum(ir**2))
    block = 1 << 16
    size = 1 << int(np.ceil(np.log2(block + n)))
    IR = np.fft.rfft(ir, size)
    out = np.zeros(len(x) + n)
    for i in range(0, len(x), block):
        seg = x[i : i + block]
        y = np.fft.irfft(np.fft.rfft(seg, size) * IR, size)[: len(seg) + n]
        out[i : i + len(y)] += y
    return out[: len(x)]


def synth_music(seconds: float, seed: int):
    """A calm ambient bed: a four-chord pad (Fmaj9, Am9, Cmaj9, G6/9) with a soft sub bass, a sparse
    plucked arpeggio and a reverb tail. Generated here so it is ours to use: no licence, no Content ID
    match. Returns stereo float32 at 48 kHz, peak about -6 dBFS."""
    import numpy as np

    rng = np.random.default_rng(seed)
    chords = [  # (bass, voicing) as MIDI notes
        (41, [57, 60, 64, 67, 69]),
        (45, [60, 64, 67, 71, 72]),
        (48, [59, 64, 67, 74, 76]),
        (43, [59, 62, 64, 69, 71]),
    ]
    bar = 6.0  # seconds per chord
    total = int(seconds * BED_RATE)
    pad = np.zeros(total + int(3 * BED_RATE))
    bass = np.zeros_like(pad)
    pluck = np.zeros_like(pad)
    seg = int((bar + 2.5) * BED_RATE)
    tt = np.arange(seg) / BED_RATE
    env = np.minimum(1, tt / 1.8) * np.clip((bar + 2.5 - tt) / 2.5, 0, 1)  # slow attack, crossfaded release
    for k in range(int(np.ceil(seconds / bar)) + 1):
        root, notes = chords[k % len(chords)]
        start = int(k * bar * BED_RATE)
        if start >= total:
            break
        wave = np.zeros(seg)
        for n in notes:
            for cents in (-7, 0, 7):  # three detuned voices per note: a gentle chorus
                f = midi_hz(n) * 2 ** (cents / 1200)
                ph = rng.uniform(0, 2 * np.pi)
                wave += np.sin(2 * np.pi * f * tt + ph) + 0.18 * np.sin(4 * np.pi * f * tt + ph)
        end = min(len(pad), start + seg)
        pad[start:end] += (wave * env)[: end - start] / (len(notes) * 3)
        fb = midi_hz(root)
        bass[start:end] += (np.sin(2 * np.pi * fb * tt) * env)[: end - start] * 0.5
        # Plucks: a chord tone an octave up on some beats (80 bpm), decaying quickly.
        beat = 60 / 80
        for b in range(int(bar / beat)):
            if rng.random() < 0.45:
                continue
            n = notes[rng.integers(len(notes))] + 12
            p0 = start + int((b * beat + rng.uniform(0, 0.03)) * BED_RATE)
            if p0 >= len(pluck):
                continue
            pl = np.arange(int(1.2 * BED_RATE)) / BED_RATE
            f = midi_hz(n)
            tone = (np.sin(2 * np.pi * f * pl) + 0.3 * np.sin(4 * np.pi * f * pl)) * np.exp(-pl / 0.28) * np.minimum(1, pl / 0.004)
            e = min(len(pluck), p0 + len(tone))
            pluck[p0:e] += tone[: e - p0] * rng.uniform(0.18, 0.3)
    lfo = 1 + 0.08 * np.sin(2 * np.pi * 0.07 * np.arange(len(pad)) / BED_RATE)
    dry = pad * lfo + bass + pluck
    dry = dry[:total]
    left = 0.65 * dry + 0.55 * reverb(dry, 2.4, rng)
    right = 0.65 * dry + 0.55 * reverb(dry, 2.4, rng)
    out = np.stack([left, right], axis=1)
    fade_in, fade_out = int(2.5 * BED_RATE), int(4 * BED_RATE)
    out[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    out[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None]
    return (out / np.max(np.abs(out)) * 0.5).astype(np.float32)


def sfx_sound(name: str, rng):
    """Short interface sounds, synthesized. Levels are absolute (dBFS peaks noted per sound)."""
    import numpy as np

    def t(sec):
        return np.arange(int(sec * BED_RATE)) / BED_RATE

    if name == "whoosh":  # filtered noise sweeping up then down, peak about -24 dBFS
        x = t(0.75)
        noise = rng.standard_normal(len(x))
        shape = np.sin(np.pi * np.clip(x / 0.75, 0, 1)) ** 2
        cutoff = 300 + 3500 * shape
        out, y = np.zeros_like(noise), 0.0
        for i, v in enumerate(noise):  # one-pole low-pass with a moving cutoff
            a = 1 - np.exp(-2 * np.pi * cutoff[i] / BED_RATE)
            y += a * (v - y)
            out[i] = y
        out *= shape
        return 0.063 * out / np.max(np.abs(out))
    if name == "pop":  # a soft blip, peak about -26 dBFS
        x = t(0.16)
        f = 520 + 420 * np.exp(-x / 0.02)
        out = np.sin(2 * np.pi * np.cumsum(f) / BED_RATE) * np.exp(-x / 0.05) * np.minimum(1, x / 0.002)
        return 0.05 * out
    if name == "tick":  # a clock tick, peak about -24 dBFS
        x = t(0.05)
        out = np.sin(2 * np.pi * 1900 * x) * np.exp(-x / 0.007)
        return 0.063 * out
    if name == "type":  # a key click, peak about -32 dBFS
        x = t(0.012)
        out = rng.standard_normal(len(x)) * np.exp(-x / 0.003)
        out = np.diff(out, prepend=0)  # crude high-pass
        return 0.025 * out / np.max(np.abs(out))
    if name == "ding":  # a bell for a quiz answer, peak about -20 dBFS
        x = t(1.8)
        out = sum(a * np.sin(2 * np.pi * 1046.5 * r * x) * np.exp(-x / (1.2 / (i + 1))) for i, (r, a) in enumerate([(1, 1), (2.0, 0.45), (3.01, 0.22), (4.2, 0.12)]))
        out *= np.minimum(1, x / 0.003)
        return 0.1 * out / np.max(np.abs(out))
    raise ValueError(f"unknown sound effect {name}")


def bed(bed_path: Path):
    """BED.json: {seconds, seed, music: "synth" | null, musicOut, sfx: [{name, t, db?}], sfxOut}."""
    import numpy as np
    import soundfile as sf

    spec = json.loads(bed_path.read_text())
    rng = np.random.default_rng(spec.get("seed", 0))
    if spec.get("music") == "synth":
        start = time.perf_counter()
        sf.write(spec["musicOut"], synth_music(spec["seconds"], spec.get("seed", 0)), BED_RATE)
        print(f"tts: music bed {spec['seconds']:.0f}s synthesized in {time.perf_counter() - start:.0f}s", flush=True)
    if spec.get("sfx") is not None:
        track = np.zeros(int((spec["seconds"] + 2) * BED_RATE))
        cache = {}
        for ev in spec["sfx"]:
            key = ev["name"]
            if key not in cache or key in ("whoosh", "type"):  # noise-based sounds vary a little each time
                cache[key] = sfx_sound(key, rng)
            snd = cache[key] * 10 ** (ev.get("db", 0) / 20)
            i = max(0, int(ev["t"] * BED_RATE))
            e = min(len(track), i + len(snd))
            track[i:e] += snd[: e - i]
        sf.write(spec["sfxOut"], track[: int(spec["seconds"] * BED_RATE)].astype(np.float32), BED_RATE)
        print(f"tts: {len(spec['sfx'])} sound effects placed", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    device = "auto"
    if "--device" in args:
        i = args.index("--device")
        device = args[i + 1]
        del args[i : i + 2]
    if len(args) != 2 or args[0] not in ("synth", "mix", "bed"):
        sys.exit(__doc__)
    if args[0] == "synth":
        synth(Path(args[1]), device)
    elif args[0] == "bed":
        bed(Path(args[1]))
    else:
        mix(Path(args[1]))
