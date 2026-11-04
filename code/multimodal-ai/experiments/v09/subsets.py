"""Build the speaker-stratified 500-utterance subsets of LibriSpeech test-clean and test-other (seed 0),
plus 60 babble-source utterances from test-other outside its subset. Writes runs/subsets.json."""
import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import asr
out = {}
for split in ("test-clean", "test-other"):
    u = asr.librispeech(split)
    import soundfile as sf
    long = [x["id"] for x in u if sf.info(x["path"]).duration > 30.0]
    print(split, "utterances over 30 s excluded:", len(long))
    s = asr.pick_subset(u, 500, seed=0, exclude=long)
    out[split] = [x["id"] for x in s]
    import soundfile as sf
    dur = [sf.info(x["path"]).duration for x in s]
    alld = [sf.info(x["path"]).duration for x in u]
    print(split, "all", len(u), "speakers", len({x['speaker'] for x in u}), "hours", round(sum(alld)/3600, 2),
          "| subset", len(s), "speakers", len({x['speaker'] for x in s}), "words", sum(x['n_words'] for x in s),
          "hours", round(sum(dur)/3600, 3), "max s", round(max(dur), 1), "min s", round(min(dur), 1))
    if split == "test-other":
        b = asr.pick_subset(u, 60, seed=1, exclude=out[split])
        out["babble_sources"] = [x["id"] for x in b]
        print("babble sources", len(b), "speakers", len({x['speaker'] for x in b}))
json.dump(out, open("runs/subsets.json", "w"))
