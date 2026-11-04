"""Pick 200 sentences (100 per book) from two public-domain Gutenberg books."""
import re, json, random
out = []
for book, fn in [("The Time Machine (H. G. Wells, Gutenberg #35)", "dl/pg35.txt"),
                 ("The Wonderful Wizard of Oz (L. Frank Baum, Gutenberg #55)", "dl/pg55.txt")]:
    t = open(fn, encoding="utf-8").read()
    t = t.split("*** START OF")[1].split("***", 1)[1].split("*** END OF")[0]
    t = t.replace("\r", "").replace("_", "")
    t = re.sub(r"[“”\"]", "", t).replace("’", "'").replace("‘", "'")
    t = re.sub(r"\s+", " ", t)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", t)
    ok = [s.strip() for s in sents
          if 8 <= len(s.split()) <= 30 and not re.search(r"[0-9\[\]\(\)*/]|CHAPTER|Chapter|Mr\.|Mrs\.|\b[IVX]{2,}\b|--|—", s)
          and s.strip()[-1] in ".!?"]
    random.Random(0).shuffle(ok)
    for s in ok[:100]:
        out.append(dict(id=f"m{len(out):03d}", text=s, source=book))
    print(book, len(sents), "sentences,", len(ok), "eligible")
with open("texts/main.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
import statistics
print(len(out), "picked; words median", statistics.median(len(r["text"].split()) for r in out), "total", sum(len(r["text"].split()) for r in out))
