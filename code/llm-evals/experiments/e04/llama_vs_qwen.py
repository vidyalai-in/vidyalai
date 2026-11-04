"""Paired Llama-3.2-1B minus Qwen2.5-0.5B under each scoring method (same items, clustered by subject)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import mcq
I = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json"))
gold = [x["answer"] for x in I]; subj = [x["subject"] for x in I]
L = json.load(open("runs/open-Llama-3.2-1B.json"))["conds"]; Q = json.load(open("runs/open-Qwen2.5-0.5B.json"))["conds"]
am = lambda v: max(range(4), key=lambda c: v[c])
def corr(run, name, how):
    x = run[name]
    if how == "letter": p = [am(v) for v in x]
    elif how == "sum": p = [am([o[0] for o in v]) for v in x]
    elif how == "char": p = [am([o[0] / o[1] for o in v]) for v in x]
    else: p = [mcq.x_flex(t, I[i]) for i, t in enumerate(x)]
    return [int(a == g) for a, g in zip(p, gold)]
out = {}
for name, how in [("letter-mmlu-dev0", "letter"), ("letter-mmlu-dev5", "letter"), ("cloze-mmlu-dev0", "char"),
                  ("cloze-mmlu-dev5", "char"), ("cloze-mmlu-dev5", "sum"), ("gen-mmlu-dev5", "flex")]:
    a, b = corr(L, name, how), corr(Q, name, how)
    p = mcq.paired(a, b, subj)
    out[f"{name}:{how}"] = dict(llama=sum(a)/len(a), qwen=sum(b)/len(b), **p)
    print(f"{name}:{how:6s} llama {100*sum(a)/len(a):.1f} qwen {100*sum(b)/len(b):.1f} diff {100*p['diff']:+.1f} [{100*p['low']:+.1f},{100*p['high']:+.1f}]")
json.dump(out, open("runs/llama-vs-qwen.json", "w"), indent=1)
