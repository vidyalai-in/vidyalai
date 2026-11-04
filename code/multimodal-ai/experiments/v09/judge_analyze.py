import sys, json, numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab"); import asr
out = {json.loads(l)["cid"]: json.loads(l) for l in open("runs/judge_out.jsonl")}
items = json.load(open("runs/judge_items.json"))
hand = {k: v for k, v in json.load(open("runs/hand_labels.json")).items() if not k.startswith("_")}
def lab(o):
    t = (o.get("out") or "").strip().upper()
    return 1 if t.startswith("CHANGED") else 0 if t.startswith("SAME") else None
J = {k: lab(v) for k, v in out.items()}
unparsed = sum(1 for v in J.values() if v is None)
cut = sum(1 for v in out.values() if v.get("stop") == "max_tokens")
# agreement with hand labels
pairs = [(hand[k], J[k]) for k in hand]
agree = sum(a == b for a, b in pairs); n = len(pairs)
tp = sum(a == 1 and b == 1 for a, b in pairs); fp = sum(a == 0 and b == 1 for a, b in pairs); fn = sum(a == 1 and b == 0 for a, b in pairs); tn = sum(a == 0 and b == 0 for a, b in pairs)
po = agree / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n / n; kappa = (po - pe) / (1 - pe)
res = {"n_items": len(items), "unparsed": unparsed, "stop_max_tokens": cut, "hand_n": n, "agree": agree,
       "agree_ci": asr.wilson(agree, n), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "kappa": kappa, "by_model": {}}
for m in ("tiny", "base", "small", "turbo"):
    for sp in ("test-clean", "test-other"):
        its = [x for x in items if x["model"] == m and x["split"] == sp]
        ch = sum(1 for x in its if J[x["cid"]] == 1)
        res["by_model"][f"{m}|{sp}"] = {"err_utts": len(its), "changed": ch, "share": asr.wilson(ch, len(its)), "of_500": asr.wilson(ch, 500)}
    # meaning-changing errors per utterance with 1 error
    one = [x for x in items if x["model"] == m and x["errors"] == 1]
    res["by_model"][f"{m}|one_error"] = {"n": len(one), "changed": sum(1 for x in one if J[x["cid"]] == 1)}
json.dump(res, open("runs/judge_results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=lambda v: round(v, 3)))
