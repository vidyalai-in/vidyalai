"""Score the round trips: WER per voice (main set), hard cases by category, speed sweep,
duration spread across voices. Writes runs/results.json (small, committable)."""
import sys, json, collections
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np
import speech_tts as st

meta = {json.loads(l)["key"]: json.loads(l) for l in open("runs/synth_meta.jsonl")}
hyp = {json.loads(l)["key"]: json.loads(l)["hyp"] for l in open("runs/asr.jsonl")}
asr_stats = json.load(open("runs/asr_stats.json"))
VOICES = ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bm_george"]
R = dict(voices={}, hard={}, speed={}, examples={})

def sel(**kw):
    return [m for m in meta.values() if all(m[k] == v for k, v in kw.items())]

# main set per voice
for v in VOICES:
    ms = sel(set="main", voice=v, speed=1.0)
    pairs = [(m["text"], hyp[m["key"]]) for m in ms]
    dur = sum(m["dur_s"] for m in ms); words = sum(len(m["text"].split()) for m in ms)
    R["voices"][v] = dict(grade=st.STOCK_VOICES[v][1], n=len(ms), audio_s=dur,
                          words_per_min=words / dur * 60,
                          wer_norm=st.corpus_wer(pairs, True), wer_basic=st.corpus_wer(pairs, False))
allpairs = [(m["text"], hyp[m["key"]]) for m in sel(set="main", speed=1.0)]
R["main_all"] = dict(wer_norm=st.corpus_wer(allpairs, True), wer_basic=st.corpus_wer(allpairs, False))

# the worst main-set items (to read the errors by eye)
errs = []
for m in sel(set="main", speed=1.0):
    e, n = st.wer_counts(m["text"], hyp[m["key"]], True)
    errs.append((e / n, e, m["voice"], m["id"], m["text"], hyp[m["key"]]))
errs.sort(reverse=True)
R["examples"]["worst_main"] = errs[:25]
# how many sentences have an error in every voice / in no voice
by_id = collections.defaultdict(list)
for x in errs: by_id[x[3]].append(x[1])
R["main_sentence_error_profile"] = dict(
    sentences=len(by_id), error_free_all_voices=sum(all(e == 0 for e in v) for v in by_id.values()),
    error_in_all_voices=sum(all(e > 0 for e in v) for v in by_id.values()))

# hard cases per category (all voices pooled, and af_heart alone)
cats = sorted({m["id"].split("_", 1)[1].rsplit("_", 1)[0] for m in sel(set="hard")})
for c in cats:
    ms = [m for m in sel(set="hard", speed=1.0) if m["id"].startswith(f"h_{c}_")]
    pairs = [(m["text"], hyp[m["key"]]) for m in ms]
    R["hard"][c] = dict(n=len(ms), wer_norm=st.corpus_wer(pairs, True), wer_basic=st.corpus_wer(pairs, False))
    R["examples"][f"hard_{c}"] = [(m["voice"], m["text"], hyp[m["key"]]) for m in ms if m["voice"] in ("af_heart", "am_michael")]

# speed sweep: duration ratio vs 1.0 and WER, per voice
for v in ["af_heart", "am_michael"]:
    base = {m["id"]: m for m in sel(set="main", voice=v, speed=1.0) if int(m["id"][1:]) < 50}
    for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        ms = [m for m in (sel(set="speed", voice=v, speed=s) if s != 1.0 else base.values())]
        ratios = np.array([m["dur_s"] / base[m["id"]]["dur_s"] for m in ms])
        pairs = [(m["text"], hyp[m["key"]]) for m in ms]
        words = sum(len(m["text"].split()) for m in ms); dur = sum(m["dur_s"] for m in ms)
        R["speed"][f"{v}@{s}"] = dict(voice=v, speed=s, n=len(ms), dur_ratio_median=float(np.median(ratios)),
                                     dur_ratio_p5=float(np.percentile(ratios, 5)), dur_ratio_p95=float(np.percentile(ratios, 95)),
                                     expected=1 / s, words_per_min=words / dur * 60,
                                     wer_norm=st.corpus_wer(pairs, True))

# one text, many renditions: duration spread across the 6 voices for each main sentence
d = collections.defaultdict(dict)
for m in sel(set="main", speed=1.0): d[m["id"]][m["voice"]] = m["dur_s"]
ratio = np.array([max(x.values()) / min(x.values()) for x in d.values()])
cv = np.array([np.std(list(x.values())) / np.mean(list(x.values())) for x in d.values()])
R["duration_spread"] = dict(sentences=len(d), max_over_min_median=float(np.median(ratio)),
                            max_over_min_p90=float(np.percentile(ratio, 90)), cv_median=float(np.median(cv)))
R["librispeech_ref"] = asr_stats["librispeech"]; R["asr_throughput"] = asr_stats["synth"]
json.dump(R, open("runs/results.json", "w"), indent=1, ensure_ascii=False)
for v, x in R["voices"].items():
    w = x["wer_norm"]; print(f"{v:11s} {x['grade']:3s} WER {w['wer']*100:5.2f} [{w['lo']*100:.2f},{w['hi']*100:.2f}] basic {x['wer_basic']['wer']*100:5.2f} wpm {x['words_per_min']:.0f}")
print("all", R["main_all"]["wer_norm"]["wer"], "LS", R["librispeech_ref"]["wer_norm"])
for c, x in R["hard"].items(): print(c, round(x["wer_norm"]["wer"] * 100, 2), x["wer_norm"]["lo"], x["wer_norm"]["hi"], "basic", round(x["wer_basic"]["wer"] * 100, 1))
for k, x in R["speed"].items(): print(k, round(x["dur_ratio_median"], 3), round(x["expected"], 3), round(x["wer_norm"]["wer"] * 100, 2), round(x["words_per_min"]))
print(R["duration_spread"], R["main_sentence_error_profile"])
