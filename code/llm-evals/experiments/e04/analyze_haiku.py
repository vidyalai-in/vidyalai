"""Part 4, experiments 2 and 3: Haiku 5.5 format variants, option rotations, and five extractors."""
import json, sys
from collections import Counter
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e04/scripts")
import mcq
from haiku_common import ITEMS, CONDS

tag = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
recs = [json.loads(l) for l in open(f"/home/amit/.cache/vidyalai-scratch/e04/runs/batch-{tag}.jsonl")]
by = {}
for r in recs:
    by.setdefault(r["cond"], {})[r["i"]] = r
conds = [c for c in CONDS if c in by]
idx = sorted(by[conds[0]])
subj = [ITEMS[i]["subject"] for i in idx]
out = {"n_items": len(idx), "failed": Counter(r["result"] for r in recs if r["result"] != "succeeded"),
       "stop": Counter(f"{r['cond']}:{r.get('stop')}" for r in recs if r.get("stop") != "end_turn"),
       "cost_usd": None, "conds": {}}

def labels(c):
    return mcq.FORMATS[CONDS[c][0]].labels

def pick(c, i, ex):
    r = by[c][i]
    if r["result"] != "succeeded":
        return None
    it = {"choices": r["choices"]}
    return mcq.EXTRACTORS[ex](r["text"], it, labels(c))

def correct(c, ex="flex"):
    return [int(pick(c, i, ex) == by[c][i]["answer"]) for i in idx]

for c in conds:
    d = {}
    for ex in mcq.EXTRACTORS:
        v = correct(c, ex)
        k = sum(v)
        none = sum(pick(c, i, ex) is None for i in idx)
        d[ex] = dict(acc=mcq.wilson(k, len(idx)), none=none)
    d["out_tokens_mean"] = sum(by[c][i]["usage"]["output_tokens"] for i in idx if "usage" in by[c][i]) / len(idx)
    d["vs_mmlu"] = mcq.paired(correct(c), correct("mmlu"), subj) if c != "mmlu" else None
    d["vs_mmlu_unclustered"] = mcq.paired(correct(c), correct("mmlu")) if c != "mmlu" else None
    out["conds"][c] = d

# run-to-run noise: identical requests
if "rep" in by:
    a, b = [pick("mmlu", i, "flex") for i in idx], [pick("rep", i, "flex") for i in idx]
    out["rep_agreement"] = sum(x == y for x, y in zip(a, b)) / len(idx)

# rotations: gold letter moves through A-D
rot = {0: "mmlu", 1: "rot1", 2: "rot2", 3: "rot3"}
if all(c in by for c in rot.values()):
    gold_pos = Counter(); gold_ok = Counter(); chosen = Counter(); total = 0
    content_same = 0; all_right = 0
    for i in idx:
        picks_content = []
        oks = []
        for r, c in rot.items():
            g = by[c][i]["answer"]; p = pick(c, i, "flex")
            gold_pos[g] += 1; gold_ok[g] += int(p == g); total += 1
            if p is not None:
                chosen[p] += 1
            picks_content.append(None if p is None else (p - r) % 4)  # back to the original option
            oks.append(p == g)
        content_same += len(set(picks_content)) == 1 and picks_content[0] is not None
        all_right += all(oks)
    out["rotation"] = dict(
        acc_by_gold={mcq.LETTERS[g]: mcq.wilson(gold_ok[g], gold_pos[g]) for g in range(4)},
        chosen_share={mcq.LETTERS[g]: mcq.wilson(chosen[g], total) for g in range(4)},
        mean_acc=sum(gold_ok.values()) / total,
        consistent=mcq.wilson(content_same, len(idx)), all_four_right=mcq.wilson(all_right, len(idx)),
        rstd=(sum((gold_ok[g] / gold_pos[g] - sum(gold_ok.values()) / total) ** 2 for g in range(4)) / 4) ** 0.5)
    # paired: gold at A vs gold at D (same items, different rotations)
    def at(gl):
        v = []
        for i in idx:
            for r, c in rot.items():
                if by[c][i]["answer"] == gl:
                    v.append(int(pick(c, i, "flex") == gl))
        return v
    out["rotation"]["A_minus_D"] = mcq.paired(at(0), at(3), subj)
    accs = {r: correct(c) for r, c in rot.items()}
    out["rotation"]["acc_by_rotation"] = {r: mcq.wilson(sum(v), len(v)) for r, v in accs.items()}

# extraction study on the free-form conditions: paired flex - strict etc.
for c in ("raw", "raw5"):
    if c in by:
        out["conds"][c]["flex_minus_strict"] = mcq.paired(correct(c, "flex"), correct(c, "strict"), subj)
        out["conds"][c]["flex_minus_any"] = mcq.paired(correct(c, "flex"), correct(c, "any"), subj)
        out["conds"][c]["flex_minus_answer_is"] = mcq.paired(correct(c, "flex"), correct(c, "answer_is"), subj)
        # how often each extractor disagrees with flex when both answer
        out["conds"][c]["any_wrong_letter_vs_flex"] = sum(
            pick(c, i, "any") is not None and pick(c, i, "flex") is not None and pick(c, i, "any") != pick(c, i, "flex") for i in idx)

json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/e04/runs/summary-{tag}.json", "w"), indent=1, default=str)

def f(w):
    return f"{100*w[0]:5.1f} [{100*w[1]:4.1f}, {100*w[2]:4.1f}]"
print("items", len(idx), "failed", dict(out["failed"]), "non-end_turn", dict(out["stop"]))
print(f"{'cond':6s} " + " ".join(f"{e:>22s}" for e in mcq.EXTRACTORS) + "   vs mmlu (flex, clustered)")
for c in conds:
    d = out["conds"][c]
    row = " ".join(f"{f(d[e]['acc'])} n{d[e]['none']:<4d}" for e in mcq.EXTRACTORS)
    v = d["vs_mmlu"]
    vs = "" if v is None else f"{100*v['diff']:+.1f} [{100*v['low']:+.1f}, {100*v['high']:+.1f}]"
    print(f"{c:6s} {row}  {vs}  out {d['out_tokens_mean']:.1f}")
print("rep agreement", out.get("rep_agreement"))
if "rotation" in out:
    print(json.dumps(out["rotation"], indent=0, default=str))
for c in ("raw", "raw5"):
    if c in out["conds"]:
        print(c, {k: v for k, v in out["conds"][c].items() if k.startswith(("flex_", "any_"))})
