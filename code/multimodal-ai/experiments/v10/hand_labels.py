"""Hand labels of misaki's American-English phonemes for the 60 hard sentences (runs_g2p_hard.json), made by
reading each phoneme string: ok / acceptable / wrong, with the reason. Then compare with the ASR round trip."""
import sys, json, collections
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import speech_tts as st
LAB = {
 "h_numbers_9": ("wrong", "3/4 -> 'three four', 1/2 -> 'one two'"),
 "h_currency_7": ("wrong", "¥500 -> 'yen five hundred'"),
 "h_dates_times_0": ("acceptable", "March 3 -> 'March three' (cardinal)"),
 "h_dates_times_1": ("wrong", "12/05/2024 -> 'twelve zero five twenty twenty-four'"),
 "h_dates_times_4": ("wrong", "Tue., Nov. not expanded"),
 "h_dates_times_5": ("acceptable", "07:45 -> 'zero seven forty-five'"),
 "h_dates_times_9": ("wrong", "8:00 -> 'eight zero zero'"),
 "h_abbreviations_0": ("wrong", "Oak St. -> 'Saint'"),
 "h_abbreviations_3": ("wrong", "kg, cm spelled as letters"),
 "h_abbreviations_4": ("acceptable", "ASAP as a word, i.e. as letters"),
 "h_abbreviations_6": ("acceptable", "Prof. -> 'prof'"),
 "h_abbreviations_8": ("wrong", "tsp. -> 'T S P'"),
 "h_abbreviations_9": ("wrong", "Gen. -> 'Jen', Lt. -> 'L T'"),
 "h_heteronyms_6": ("wrong", "second bass read as the fish"),
 "h_rare_words_1": ("acceptable", "Llanfair, Aberystwyth anglicized"),
 "h_rare_words_4": ("wrong", "Gewurztraminer letter-by-letter guess"),
}
hard = [json.loads(l) for l in open("texts/hard.jsonl")]
meta = {json.loads(l)["key"]: json.loads(l) for l in open("runs/synth_meta.jsonl")}
hyp = {json.loads(l)["key"]: json.loads(l)["hyp"] for l in open("runs/asr.jsonl")}
labels = {h["id"]: LAB.get(h["id"], ("ok", ""))[0] for h in hard}
cnt = collections.Counter((h["id"].split("_", 1)[1].rsplit("_", 1)[0], labels[h["id"]]) for h in hard)
conf = collections.Counter(); per = {}
US = ["af_heart", "af_bella", "am_michael", "am_adam"]
for h in hard:
    flags = []
    for v in US:
        k = f"hard|{v}|1.0|{h['id']}"
        e, n = st.wer_counts(h["text"], hyp[k], True); flags.append(e > 0)
        conf[(labels[h["id"]], e > 0)] += 1
    per[h["id"]] = dict(label=labels[h["id"]], flagged_voices=sum(flags), hyp_af_heart=hyp[f"hard|af_heart|1.0|{h['id']}"])
out = dict(labels={k: v for k, v in LAB.items()}, counts={f"{a}|{b}": c for (a, b), c in cnt.items()},
           roundtrip_vs_hand={f"{a}|flagged={b}": c for (a, b), c in conf.items()}, per_sentence=per)
json.dump(out, open("runs/hand_labels.json", "w"), indent=1, ensure_ascii=False)
print(sorted(cnt.items())); print(sorted(conf.items()))
for k, v in per.items():
    if v["label"] == "wrong" or (v["label"] == "ok" and v["flagged_voices"] >= 2):
        print(k, v)
