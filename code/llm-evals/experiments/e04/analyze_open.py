"""Part 4, experiment 1: summarize the open models' runs (accuracy, spreads, rankings, letter bias)."""
import glob, json, sys
from collections import Counter
from itertools import combinations
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import mcq

ITEMS = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json"))
gold = [it["answer"] for it in ITEMS]
subj = [it["subject"] for it in ITEMS]
N = len(ITEMS)
runs = {json.load(open(p))["model"].split("/")[-1]: json.load(open(p))
        for p in sorted(glob.glob("/home/amit/.cache/vidyalai-scratch/e04/runs/open-*.json")) if "gpu_s" in json.load(open(p))}
argmax = lambda v: max(range(4), key=lambda c: v[c])

def per_item(run, name):
    """{variant: list of 0/1 per item, plus chosen option index (None if unanswered)}."""
    x = run["conds"][name]
    if name.startswith("letter"):
        ch = [argmax(v) for v in x]
        return {name: ch}
    if name.startswith("cloze"):
        return {name + ":sum": [argmax([o[0] for o in v]) for v in x],
                name + ":per_char": [argmax([o[0] / o[1] for o in v]) for v in x]}
    return {f"{name}:{e}": [mcq.EXTRACTORS[e](t, ITEMS[i]) for i, t in enumerate(x)] for e in mcq.EXTRACTORS}

summary = {"n_items": N, "models": {}}
choices = {}
for m, run in runs.items():
    ch = {}
    for name in run["conds"]:
        ch.update(per_item(run, name))
    choices[m] = ch
    d = {}
    for v, c in ch.items():
        k = sum(int(p == g) for p, g in zip(c, gold))
        d[v] = dict(acc=mcq.wilson(k, N), none=sum(p is None for p in c),
                    picks={mcq.LETTERS[j]: sum(p == j for p in c) / N for j in range(4)})
    summary["models"][m] = dict(conds=d, gpu_s=run["gpu_s"], peak_gb=run["max_mem_gb"], timing=run["timing"])

def corr(m, v):
    return [int(p == g) for p, g in zip(choices[m][v], gold)]

# spreads
fmts = list(mcq.FORMATS)
for m in runs:
    s = summary["models"][m]
    a = lambda v: s["conds"][v]["acc"][0]
    for k in (0, 5):
        vs = [f"letter-{f}-dev{k}" for f in fmts]
        best, worst = max(vs, key=a), min(vs, key=a)
        s[f"format_spread_{k}shot"] = dict(best=best, worst=worst, range=a(best) - a(worst),
                                           paired=mcq.paired(corr(m, best), corr(m, worst), subj))
    for k in (1, 5):
        vs = [f"letter-mmlu-{d}{k}" for d in ("dev", "s1", "s2")]
        best, worst = max(vs, key=a), min(vs, key=a)
        s[f"draw_spread_{k}shot"] = dict(best=best, worst=worst, range=a(best) - a(worst),
                                         paired=mcq.paired(corr(m, best), corr(m, worst), subj))
    s["shots_5_minus_0"] = mcq.paired(corr(m, "letter-mmlu-dev5"), corr(m, "letter-mmlu-dev0"), subj)
    s["letter_minus_cloze5"] = mcq.paired(corr(m, "letter-mmlu-dev5"), corr(m, "cloze-mmlu-dev5:per_char"), subj)
    s["cloze_norm_minus_sum5"] = mcq.paired(corr(m, "cloze-mmlu-dev5:per_char"), corr(m, "cloze-mmlu-dev5:sum"), subj)
    s["letter_minus_gen5flex"] = mcq.paired(corr(m, "letter-mmlu-dev5"), corr(m, "gen-mmlu-dev5:flex"), subj)
    allv = list(s["conds"])
    s["all_range"] = (min(a(v) for v in allv), max(a(v) for v in allv))

# rankings across every variant
models = list(runs)
order = {}
RANKED = [v for v in summary["models"][models[0]]["conds"]
          if v.startswith(("letter-", "cloze-")) or v.endswith(":flex")]  # 15 letter, 4 cloze, 2 generate
for m in models:
    a, b = choices[m]["letter-mmlu-dev5"], choices[m]["letternc-mmlu-dev5"]
    summary["models"][m]["bf16_cache_flips"] = sum(x != y for x, y in zip(a, b))
    summary["models"][m]["bf16_cache_diff"] = mcq.paired(corr(m, "letter-mmlu-dev5"), corr(m, "letternc-mmlu-dev5"), subj)
for v in RANKED:
    accs = {m: summary["models"][m]["conds"][v]["acc"][0] for m in models}
    order[v] = tuple(sorted(models, key=lambda m: -accs[m]))
summary["orderings"] = Counter(" > ".join(o) for o in order.values()).most_common()
summary["order_by_variant"] = {v: " > ".join(o) for v, o in order.items()}
# pairwise: for each pair of models, in how many variants does each lead, and is the lead significant?
pw = {}
for x, y in combinations(models, 2):
    lead = Counter()
    for v in RANKED:
        p = mcq.paired(corr(x, v), corr(y, v), subj)
        lead["sig " + x if p["low"] > 0 else "sig " + y if p["high"] < 0 else "tie"] += 1
    pw[f"{x} vs {y}"] = dict(lead)
summary["pairwise"] = pw
json.dump(summary, open("/home/amit/.cache/vidyalai-scratch/e04/runs/summary-open.json", "w"), indent=1)

for m in models:
    s = summary["models"][m]
    print(f"\n== {m}  gpu {s['gpu_s']}s peak {s['peak_gb']}GB  range {s['all_range'][0]:.3f}-{s['all_range'][1]:.3f}")
    for v, d in s["conds"].items():
        w = d["acc"]
        pk = " ".join(f"{L}{100*p:.0f}" for L, p in d["picks"].items())
        print(f"  {v:34s} {100*w[0]:5.1f} [{100*w[1]:4.1f},{100*w[2]:4.1f}] none {d['none']:4d}  picks {pk}")
    for key in ("format_spread_0shot", "format_spread_5shot", "draw_spread_1shot", "draw_spread_5shot"):
        p = s[key]["paired"]
        print(f"  {key}: {s[key]['best']} - {s[key]['worst']} = {100*p['diff']:+.1f} [{100*p['low']:+.1f},{100*p['high']:+.1f}]")
    for key in ("shots_5_minus_0", "letter_minus_cloze5", "cloze_norm_minus_sum5", "letter_minus_gen5flex"):
        p = s[key]
        print(f"  {key}: {100*p['diff']:+.1f} [{100*p['low']:+.1f},{100*p['high']:+.1f}]")
print("\norderings:", summary["orderings"])
print("bf16 cache flips:", {m: summary["models"][m]["bf16_cache_flips"] for m in models})
print("pairwise:", json.dumps(pw, indent=0))
