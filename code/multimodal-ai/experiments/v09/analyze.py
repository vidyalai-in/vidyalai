"""Post-process all runs into runs/results.json (small)."""
import sys, json, os, glob
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np
import asr
R = {"exp1": {}, "exp2": {}, "exp3": {}, "exp4": {}}
def load(p): return [json.loads(l) for l in open(p)]
def counts(rows, kind):
    return asr.word_errors([asr.normalize(r["ref"], kind) for r in rows], [asr.normalize(r["hyp"], kind) for r in rows])
for m in ("tiny", "base", "small", "turbo"):
    for b in (1, 5):
        p = f"runs/exp1_{m}_b{b}.jsonl"
        if not os.path.exists(p): continue
        rows, meta = load(p), json.load(open(p.replace(".jsonl", "_meta.json")))
        key = f"{m}_b{b}"
        R["exp1"][key] = {"meta": {k: v for k, v in meta.items()}}
        for sp in ("test-clean", "test-other"):
            rs = [r for r in rows if r["split"] == sp]
            d = {"n_utts": len(rs)}
            for kind in asr.NORMALIZERS:
                c = counts(rs, kind)
                w, lo, hi = asr.boot_wer(c)
                d[kind] = {"wer": w, "lo": lo, "hi": hi, "S": int(c[:, 0].sum()), "D": int(c[:, 1].sum()),
                           "I": int(c[:, 2].sum()), "N": int(c[:, 3].sum()),
                           "utts_with_error": int((c[:, :3].sum(1) > 0).sum())}
            R["exp1"][key][sp] = d
# exp4: beam 5 vs greedy, paired
for m in ("tiny", "base", "small", "turbo"):
    if os.path.exists(f"runs/exp1_{m}_b5.jsonl"):
        g, b = load(f"runs/exp1_{m}_b1.jsonl"), load(f"runs/exp1_{m}_b5.jsonl")
        for sp in ("test-clean", "test-other"):
            gs = [r for r in g if r["split"] == sp]; bs = [r for r in b if r["split"] == sp]
            assert [r["id"] for r in gs] == [r["id"] for r in bs]
            R["exp4"][f"{m}_{sp}"] = dict(zip(("diff", "lo", "hi"), asr.boot_diff(counts(bs, "english"), counts(gs, "english"))))
# exp2
for p in sorted(glob.glob("runs/exp2_*.jsonl")):
    m = p.split("exp2_")[1].split(".")[0]
    rows = load(p); out = {}
    for noise, snr in sorted({(r["noise"], r["snr"]) for r in rows}, key=lambda t: (t[0], -(t[1] or 99))):
        rs = [r for r in rows if r["noise"] == noise and r["snr"] == snr]
        c = counts(rs, "english")
        w, lo, hi = asr.boot_wer(c)
        out[f"{noise}_{snr}"] = {"noise": noise, "snr": snr, "n": len(rs), "wer": w, "lo": lo, "hi": hi,
                                 "S": int(c[:, 0].sum()), "D": int(c[:, 1].sum()), "I": int(c[:, 2].sum()), "N": int(c[:, 3].sum()),
                                 "empty": sum(1 for r in rs if not asr.normalize(r["hyp"], "basic"))}
    R["exp2"][m] = out
# exp3
for p in sorted(glob.glob("runs/exp3_*.json")):
    d = json.load(open(p)); m = p.split("exp3_")[1].split(".")[0]
    out = {}
    for cfg, hyps in d["out"].items():
        for cls in ("near-silence", "noise", "tones", "digital-zero"):
            idx = [i for i, c in enumerate(d["clips"]) if c["cls"] == cls]
            ne = [i for i in idx if asr.normalize(hyps[i], "basic")]
            words = [len(asr.normalize(hyps[i], "basic").split()) for i in ne]
            p_, lo, hi = asr.wilson(len(ne), len(idx))
            out[f"{cfg}|{cls}"] = {"n": len(idx), "nonempty": len(ne), "rate": p_, "lo": lo, "hi": hi,
                                   "median_words": float(np.median(words)) if words else 0.0}
    out["time_s"] = d.get("time_s"); out["vad_speech"] = int(sum(d["vad_speech"]))
    R["exp3"][m] = out
json.dump(R, open("runs/results.json", "w"), indent=1, default=float)
print(os.path.getsize("runs/results.json"), "bytes")
