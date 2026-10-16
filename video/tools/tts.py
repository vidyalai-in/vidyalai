"""Narration audio for the article-to-video pipeline, using Kokoro (open-weight TTS, Apache 2.0).

Two commands, both driven by JSON files that build.mjs writes:

  tts.py synth MANIFEST.json   Synthesize each sentence to its own WAV (cached by text+voice+speed)
                               and write the durations back into MANIFEST.json under "durations".
  tts.py mix TIMELINE.json     Place sentence WAVs at their start times into one WAV track.

Device: --device auto (default) times one sentence on CUDA and CPU and keeps the faster one.
Run it through uv so dependencies stay out of the repo, e.g.:
  uv run --python 3.12 --with "kokoro>=0.9.4" --with soundfile --with pip python tts.py synth m.json
"""

import hashlib
import json
import sys
import time
from pathlib import Path

SAMPLE_RATE = 24000


def load_pipeline(lang: str, device: str):
    import torch
    from kokoro import KModel, KPipeline

    def build(dev):
        model = KModel(repo_id="hexgrad/Kokoro-82M").to(dev).eval()
        return KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model)

    if device != "auto":
        return build(device), device
    if not torch.cuda.is_available():
        return build("cpu"), "cpu"

    # Pick whichever device synthesizes a sample sentence faster (after a warm-up run).
    sample = "Treat a retrieval system as the foundation, and measure it before you tune the model."
    timings = {}
    pipelines = {}
    for dev in ("cuda", "cpu"):
        pipe = build(dev)
        for _ in pipe(sample, voice="af_heart"):
            pass
        start = time.perf_counter()
        for _ in pipe(sample, voice="af_heart"):
            pass
        timings[dev] = time.perf_counter() - start
        pipelines[dev] = pipe
    best = min(timings, key=timings.get)
    print(f"tts: device timings {', '.join(f'{d}={t:.2f}s' for d, t in timings.items())}; using {best}", flush=True)
    return pipelines[best], best


def synth(manifest_path: Path, device: str):
    import numpy as np
    import soundfile as sf

    manifest = json.loads(manifest_path.read_text())
    cache = Path(manifest["cacheDir"])
    cache.mkdir(parents=True, exist_ok=True)
    voice, speed, lang = manifest["voice"], float(manifest.get("speed", 1.0)), manifest.get("lang", "a")

    todo = []
    for item in manifest["items"]:
        key = hashlib.sha1(f"{voice}|{speed}|{item['tts']}".encode()).hexdigest()[:16]
        item["wav"] = str(cache / f"{key}.wav")
        if not Path(item["wav"]).exists():
            todo.append(item)

    print(f"tts: {len(manifest['items'])} sentences, {len(todo)} to synthesize (voice {voice}, speed {speed})", flush=True)
    if todo:
        pipe, dev = load_pipeline(lang, device)
        start = time.perf_counter()
        for i, item in enumerate(todo, 1):
            chunks = [r.audio for r in pipe(item["tts"], voice=voice, speed=speed) if r.audio is not None]
            audio = np.concatenate([c.numpy() if hasattr(c, "numpy") else np.asarray(c) for c in chunks])
            sf.write(item["wav"], audio, SAMPLE_RATE)
            if i % 25 == 0 or i == len(todo):
                print(f"tts: {i}/{len(todo)} on {dev} ({time.perf_counter() - start:.0f}s)", flush=True)

    durations = {}
    for item in manifest["items"]:
        durations[item["id"]] = {"wav": item["wav"], "seconds": sf.info(item["wav"]).duration}
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
