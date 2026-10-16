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

    def say(self, text: str):
        import numpy as np

        chunks = [r.audio for r in self.pipe(text, voice=self.voice, speed=self.speed) if r.audio is not None]
        return np.concatenate([c.cpu().numpy() if hasattr(c, "cpu") else np.asarray(c) for c in chunks]), {}


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

    def say(self, text: str):
        best = None
        for take in range(self.retakes):
            seed = int(hashlib.sha1(f"{text}|{take}".encode()).hexdigest()[:8], 16)
            self.torch.manual_seed(seed)
            audio = self.model.generate(text, **self.kwargs).squeeze(0).cpu().numpy()
            score, info = self.score(text, audio)
            if best is None or score < best[0]:
                best = (score, audio, {**info, "take": take})
            if score <= self.max_wer:
                break
        score, audio, info = best
        if score > self.max_wer:
            print(f"tts: WARNING best of {self.retakes} takes scored {score:.2f}: {text!r} -> {info}", flush=True)
        return audio, info


ENGINES = {"kokoro": Kokoro, "chatterbox": Chatterbox}


def synth(manifest_path: Path, device: str):
    import soundfile as sf

    manifest = json.loads(manifest_path.read_text())
    engine = manifest["engine"]
    cache = Path(manifest["cacheDir"])
    cache.mkdir(parents=True, exist_ok=True)

    todo = []
    for item in manifest["items"]:
        item["wav"] = str(cache / f"{cache_key(engine, item['tts'])}.wav")
        info = Path(item["wav"]).with_suffix(".json")
        stale = False
        if info.exists():
            scores = json.loads(info.read_text()).get("pronunciation", {})
            stale = any(scores.get(c["say"], 1) < float(c.get("min", 0.9)) for c in engine.get("checks", []))
        if stale or not Path(item["wav"]).exists():
            todo.append(item)

    label = engine.get("voice") or Path(engine.get("reference", "?")).parent.name
    print(f"tts: {len(manifest['items'])} sentences, {len(todo)} to synthesize ({engine['name']}, {label})", flush=True)
    if todo:
        tts = ENGINES[engine["name"]](engine, device)
        start = time.perf_counter()
        for i, item in enumerate(todo, 1):
            audio, info = tts.say(item["tts"])
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


if __name__ == "__main__":
    args = sys.argv[1:]
    device = "auto"
    if "--device" in args:
        i = args.index("--device")
        device = args[i + 1]
        del args[i : i + 2]
    if len(args) != 2 or args[0] not in ("synth", "mix"):
        sys.exit(__doc__)
    if args[0] == "synth":
        synth(Path(args[1]), device)
    else:
        mix(Path(args[1]))
