"""Extra Haiku cuts: answered-only accuracy, the reasoning gain on directly answered items, letter shares."""
import json, sys
from collections import Counter
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e04/scripts")
import mcq
from haiku_common import ITEMS, CONDS
tag = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
by = {}
for l in open(f"runs/batch-{tag}.jsonl"):
    r = json.loads(l); by.setdefault(r["cond"], {})[r["i"]] = r
idx = sorted(by["mmlu"]); subj = [ITEMS[i]["subject"] for i in idx]
lab = lambda c: mcq.FORMATS[CONDS[c][0]].labels
pk = lambda c, i: mcq.x_flex(by[c][i]["text"], {"choices": by[c][i]["choices"]}, lab(c))
res = {}
for c in by:
    ans = [i for i in idx if pk(c, i) is not None]
    trunc = sum(by[c][i]["stop"] == "max_tokens" for i in idx)
    k = sum(pk(c, i) == by[c][i]["answer"] for i in ans)
    res[c] = dict(answered=len(ans), truncated=trunc, acc_answered=mcq.wilson(k, len(ans)))
    print(f"{c:6s} answered {len(ans)} truncated {trunc} acc|answered {100*k/len(ans):.1f}")
# items the answer-only prompt answered directly (end_turn): does visible reasoning still help?
direct = [i for i in idx if by["mmlu"][i]["stop"] == "end_turn" and pk("mmlu", i) is not None]
a = [int(pk("raw", i) == by["raw"][i]["answer"]) for i in direct]
b = [int(pk("mmlu", i) == by["mmlu"][i]["answer"]) for i in direct]
p = mcq.paired(a, b, [ITEMS[i]["subject"] for i in direct])
res["raw_minus_mmlu_direct"] = dict(n=len(direct), raw=sum(a)/len(a), mmlu=sum(b)/len(b), **p)
print("direct items", len(direct), "raw", sum(a)/len(a), "mmlu", sum(b)/len(b), p)
trunc_items = [i for i in idx if by["mmlu"][i]["stop"] == "max_tokens"]
if trunc_items:
    print("raw acc on the items mmlu truncated:", sum(pk("raw", i) == by["raw"][i]["answer"] for i in trunc_items), "/", len(trunc_items))
# reply length in raw
L = sorted(by["raw"][i]["usage"]["output_tokens"] for i in idx)
print("raw output tokens median", L[len(L)//2], "p90", L[int(.9*len(L))])
if "rot1" in by:
    sh = Counter(); n = 0
    for c in ("mmlu", "rot1", "rot2", "rot3"):
        for i in idx:
            q = pk(c, i)
            if q is not None:
                sh[q] += 1; n += 1
    res["shares_answered"] = {mcq.LETTERS[g]: mcq.wilson(sh[g], n) for g in range(4)}
    print("shares among answered", {k: round(100*v[0], 1) for k, v in res["shares_answered"].items()}, n)
json.dump(res, open(f"runs/extra-{tag}.json", "w"), indent=1)
