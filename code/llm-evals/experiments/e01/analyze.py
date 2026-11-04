"""Turn e01's raw runs into results.json: accuracies with 95% Wilson intervals, chance and
majority baselines, paired differences with 95% paired-bootstrap intervals."""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import validity as V
D = "/home/amit/.cache/vidyalai-scratch/evals/data"
R = "/home/amit/.cache/vidyalai-scratch/e01/runs"
SETS = {"arce": "e01-arc-easy.json", "arcc": "e01-arc-challenge.json", "mmlu": "mmlu-1140.json",
        "pro": "e01-mmlu-pro-840.json"}
KIND = {"f": "full", "co": "choices_only", "nc": "nota_correct", "nw": "nota_wrong"}
data = {k: V.load(f"{D}/{v}") for k, v in SETS.items()}

def ci(xs):
    p, lo, hi = V.wilson(int(sum(xs)), len(xs))
    return dict(n=len(xs), acc=round(100 * p, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2))

# correctness vectors: res[model][set-kind] = {idx: 0/1}; plus raw reply stats
res = collections.defaultdict(lambda: collections.defaultdict(dict))
stats = collections.defaultdict(collections.Counter)
first = collections.defaultdict(lambda: collections.defaultdict(dict))
def merged(name):
    """First-pass replies (max_tokens 8), with any reply cut off before naming a letter
    replaced by its re-ask at max_tokens 1024 (runs/<name>-rerun.jsonl)."""
    recs = {}
    for suffix in ("", "-rerun"):
        path = f"{R}/{name}{suffix}.jsonl"
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                r["rerun"] = bool(suffix)
                recs[r["id"]] = r
    return recs.values()

for name in ("haiku", "sonnet"):
    for r in merged(name):
        s, k, i = r["id"].split("-"); i = int(i)
        key = f"{s}-{KIND[k]}"
        it = data[s][i]
        if r["type"] != "succeeded":
            stats[f"{name}:{key}"]["failed_" + r["type"]] += 1
            res[name][key][i] = 0
            continue
        pred = V.extract_final(r["text"], len(it["choices"]))
        stats[f"{name}:{key}"]["unparsed" if pred is None else "parsed"] += 1
        stats[f"{name}:{key}"]["rerun"] += r["rerun"]
        if r.get("stop") == "refusal":
            stats[f"{name}:{key}"]["refusal"] += 1
        # nota variants keep the same answer index (the replaced or kept correct slot)
        res[name][key][i] = int(pred == it["answer"])
        if not r["rerun"]:
            first[name][key][i] = int(V.extract(r["text"], len(it["choices"])) == it["answer"])
        else:
            first[name][key][i] = 0  # strict first pass: a reply cut off at 8 tokens scores 0
for f in sorted(glob.glob(f"{R}/open-*.json")):
    o = json.load(open(f))
    name = o["model"].split("/")[-1]
    for key, preds in o["preds"].items():
        s = key.split("-")[0]
        res[name][key] = {i: int(p == data[s][i]["answer"]) for i, p in enumerate(preds)}

out = {"accuracy": {}, "baselines": {}, "paired": {}, "stats": {k: dict(v) for k, v in stats.items()}}
for s, items in data.items():
    chance = float(np.mean([1 / len(it["choices"]) for it in items]))
    letters = collections.Counter(it["answer"] for it in items)
    out["baselines"][s] = dict(n=len(items), chance=round(100 * chance, 2),
                               majority=round(100 * letters.most_common(1)[0][1] / len(items), 2),
                               majority_letter=V.LETTERS[letters.most_common(1)[0][0]])
for m, d in res.items():
    for key, v in d.items():
        out["accuracy"][f"{m}:{key}"] = ci(list(v.values()))

out["first_pass_strict"] = {f"{m}:{k}": ci(list(v.values())) for m, d in first.items() for k, v in d.items()}

def paired(m1, k1, m2, k2):
    a, b = res[m1][k1], res[m2][k2]
    idx = sorted(set(a) & set(b))
    mean, lo, hi = V.paired_diff([a[i] for i in idx], [b[i] for i in idx])
    return dict(n=len(idx), a=round(100 * np.mean([a[i] for i in idx]), 2), b=round(100 * np.mean([b[i] for i in idx]), 2),
                diff=round(100 * mean, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2))

if "sonnet" in res:
    for s in data:
        out["paired"][f"sonnet-haiku:{s}"] = paired("haiku", f"{s}-full", "sonnet", f"{s}-full")
if "haiku" in res:
    for k in ("nota_correct", "nota_wrong"):
        out["paired"][f"haiku:mmlu {k}-full"] = paired("haiku", "mmlu-full", "haiku", f"mmlu-{k}")
    # choices-only: of items answered right with the question, how many also right without it
    for s in ("arcc", "mmlu", "pro"):
        f, c = res["haiku"][f"{s}-full"], res["haiku"].get(f"{s}-choices_only", {})
        if c:
            both = sum(f[i] and c[i] for i in f)
            out["paired"][f"haiku:{s} choices_only-full"] = paired("haiku", f"{s}-full", "haiku", f"{s}-choices_only")
            out["paired"][f"haiku:{s} choices_only-full"]["right_both"] = both
            out["paired"][f"haiku:{s} choices_only-full"]["right_full"] = sum(f.values())
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/e01/results.json", "w"), indent=1)
for k, v in sorted(out["accuracy"].items()):
    print(f"{k:42s} {v}")
for k, v in out["baselines"].items():
    print(k, v)
for k, v in out["paired"].items():
    print(k, v)
for k, v in out["first_pass_strict"].items():
    print("first pass", k, v)
for k, v in out["stats"].items():
    print(k, v)

# NOTA: of items right in the normal format, how many go wrong when the right text is replaced
h = res["haiku"]
flip = sum(1 for i in h["mmlu-full"] if h["mmlu-full"][i] and not h["mmlu-nota_correct"][i])
out["nota_flips"] = dict(right_normal=sum(h["mmlu-full"].values()), wrong_after_swap=flip)
print("nota flips", out["nota_flips"])
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/e01/results.json", "w"), indent=1)
