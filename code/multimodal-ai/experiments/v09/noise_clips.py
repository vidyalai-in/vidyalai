"""Synthetic non-speech clips for the hallucination test (exp 3). Deterministic (seed 0)."""
import numpy as np
SR = 16000

def colored(n, rng, color):
    w = rng.standard_normal(n)
    if color == "white":
        x = w
    else:
        f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
        p = {"pink": 0.5, "brown": 1.0}[color]
        x = np.fft.irfft(np.fft.rfft(w) / f ** p, n)
    return x / np.sqrt(np.mean(x ** 2))

def tones(n, rng):
    """A sequence of random beeps, hums and chirps (sines 100-4000 Hz), unit RMS."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    pos = 0
    while pos < n:
        L = int(rng.uniform(0.2, 3.0) * SR); seg = slice(pos, min(n, pos + L)); tt = t[seg] - t[pos]
        kind = rng.integers(3)
        f0 = rng.uniform(100, 2000)
        if kind == 0:   # steady tone
            s = np.sin(2 * np.pi * f0 * tt)
        elif kind == 1: # chirp
            f1 = rng.uniform(100, 4000); s = np.sin(2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * max(tt[-1], 1e-3))))
        else:           # hum with harmonics
            s = sum(np.sin(2 * np.pi * f0 / 4 * k * tt) / k for k in range(1, 5))
        env = np.minimum(1, np.minimum(tt, tt[-1] - tt) / 0.01)
        x[seg] = s * env
        pos += L + int(rng.uniform(0, 0.8) * SR)
    return x / np.sqrt(np.mean(x ** 2) + 1e-12)

def make_clips(seed=0, per_class=50):
    rng = np.random.default_rng(seed)
    clips = []
    for cls in ("near-silence", "noise", "tones"):
        for k in range(per_class):
            n = int(rng.uniform(2, 30) * SR)
            if cls == "tones":
                x = tones(n, rng); db = -20.0; color = "tones"
            else:
                color = ("white", "pink", "brown")[k % 3]
                x = colored(n, rng, color)
                db = rng.uniform(-80, -60) if cls == "near-silence" else rng.uniform(-35, -15)
            clips.append({"cls": cls, "color": color, "dbfs": float(db), "seconds": n / SR,
                          "audio": (x * 10 ** (db / 20)).astype(np.float32)})
    clips.append({"cls": "digital-zero", "color": "none", "dbfs": float("-inf"), "seconds": 10.0,
                  "audio": np.zeros(10 * SR, dtype=np.float32)})
    return clips
