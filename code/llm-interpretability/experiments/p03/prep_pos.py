# Build the part-of-speech probing set from UD English EWT v2.18 (CC BY-SA 4.0).
# Words inside multiword tokens (e.g. "don't" = do + n't) are kept in the text but not probed,
# because they have no character span of their own.
import json, random
D = "/home/amit/.cache/vidyalai-scratch/interp/data/p03/ud"
OUT = "/home/amit/.cache/vidyalai-scratch/p03/runs/pos_data.json"


def read(path):
    sents, cur = [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# text = "):
            cur = {"text": line[9:], "words": [], "mwt_end": 0}
        elif line.startswith("#"):
            continue
        elif not line:
            if cur:
                sents.append(cur)
            cur = None
        else:
            f = line.split("\t")
            tid = f[0]
            if "." in tid:
                continue  # empty node
            if "-" in tid:
                a, b = map(int, tid.split("-"))
                cur["words"].append({"form": f[1], "upos": None, "mwt": True})
                cur["mwt_end"] = b
                continue
            if int(tid) <= cur["mwt_end"]:
                continue  # part of a multiword token
            cur["words"].append({"form": f[1], "upos": f[3], "mwt": False})
    return sents


def spans(s):
    out, cur = [], 0
    for w in s["words"]:
        i = s["text"].find(w["form"], cur)
        if i < 0:
            return None
        out.append((i, i + len(w["form"])))
        cur = i + len(w["form"])
    return out


random.seed(0)
data = {}
for split, n in [("train", 4000), ("dev", 500), ("test", None)]:
    ss = read(f"{D}/en_ewt-ud-{split}.conllu")
    ok = []
    for s in ss:
        sp = spans(s)
        if sp is None or len(s["text"]) > 600:
            continue
        keep = [(w["form"], w["upos"], a, b) for w, (a, b) in zip(s["words"], sp) if not w["mwt"]]
        ok.append({"text": s["text"], "words": keep})
    total = len(ss)
    if n:
        ok = random.sample(ok, n)
    data[split] = ok
    print(split, "sentences in file", total, "used", len(ok), "words", sum(len(s["words"]) for s in ok))
json.dump(data, open(OUT, "w"))
