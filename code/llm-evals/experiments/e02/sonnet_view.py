"""Sonnet 5.5 (1 answer per item) on its stratified subset: all 370 flagged + 370 random ok items.
Unparsed replies were re-asked (structured output) for ok and wrong_groundtruth items only, so only
those strata are scored. Compared with Haiku's 3 answers on the same items. 95% bootstrap CIs over items."""
import json, sys, re
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import items
R = "/home/amit/.cache/vidyalai-scratch/e02"
data = items.load_redux(); N = len(data)
gold = np.array([d["answer"] for d in data]); et = np.array([d["error_type"] for d in data])
sugg = np.array([items.suggested_answer(d) if items.suggested_answer(d) is not None else -1 for d in data])
fix = {json.loads(l)["custom_id"]: json.loads(json.loads(l)["text"])["answer"] for l in open(f"{R}/runs/sonnet-k1-reask.jsonl")
       if json.loads(l)["result_type"] == "succeeded"}
S = {}; direct = {}
for l in open(f"{R}/runs/sonnet-k1.jsonl"):
    r = json.loads(l); i = int(r["custom_id"][1:5]); L = items.parse_letter(r["text"])
    direct[i] = L is not None
    L = L or fix.get(r["custom_id"])
    S[i] = items.LETTERS.index(L) if L else -1
H = np.load(f"{R}/results/picks-haiku-k3.npy")[:, 1:]
Hacc = (H == gold[:, None]).mean(1); Hsug = (H == sugg[:, None]).mean(1)
ok = np.array([i for i in S if et[i] == "ok"]); wg = np.array([i for i in S if et[i] == "wrong_groundtruth"])
wgs = np.array([i for i in wg if sugg[i] >= 0])
Sa = lambda ix: np.array([S[i] == gold[i] for i in ix], float)
Ss = lambda ix: np.array([S[i] == sugg[i] for i in ix], float)
assert all(S[i] >= 0 for i in np.r_[ok, wg]), "unscored items left"
out = {}
def b(f, n): return [round(float(x), 4) for x in items.boot_ci(f, n, 10000)]
out["sonnet_ok_acc"] = b(lambda s: Sa(ok)[s].mean(), len(ok))
out["haiku_same_ok_acc"] = b(lambda s: Hacc[ok][s].mean(), len(ok))
out["sonnet_minus_haiku_ok"] = b(lambda s: (Sa(ok) - Hacc[ok])[s].mean(), len(ok))
out["sonnet_wg_key"] = b(lambda s: Sa(wg)[s].mean(), len(wg))
out["sonnet_wg_annotator"] = b(lambda s: Ss(wgs)[s].mean(), len(wgs))
out["haiku_wg_key"] = b(lambda s: Hacc[wg][s].mean(), len(wg))
out["haiku_wg_annotator"] = b(lambda s: Hsug[wgs][s].mean(), len(wgs))
# Population of ok + wrong-key items (5,436): share of misses on wrong-key items. ok sample reweighted 5330/370.
w_ok = (et == "ok").sum() / len(ok)
rng = np.random.default_rng(0)
def boot_share(miss_ok, miss_wg, w=w_ok, reps=10000):
    pt = miss_wg.sum() / (miss_wg.sum() + w * miss_ok.sum())
    v = []
    for _ in range(reps):
        a = miss_ok[rng.integers(0, len(miss_ok), len(miss_ok))]; c = miss_wg[rng.integers(0, len(miss_wg), len(miss_wg))]
        v.append(c.sum() / (c.sum() + w * a.sum()))
    return [round(float(pt), 4)] + [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]
out["share_items_wg_in_pop"] = round(106 / (5330 + 106), 4)
out["sonnet_share_misses_on_wg"] = boot_share(1 - Sa(ok), 1 - Sa(wg))
allok = np.where(et == "ok")[0]
out["haiku_share_misses_on_wg"] = boot_share(1 - Hacc[allok], 1 - Hacc[wg], w=1.0)
out["haiku_share_misses_on_wg_same370"] = boot_share(1 - Hacc[ok], 1 - Hacc[wg])
for t in ["ok"] + items.FLAW_TYPES:
    ix = [i for i in S if et[i] == t]
    k = sum(direct[i] for i in ix)
    out[f"sonnet_direct_letter_{t}"] = [k, len(ix)] + [round(x, 4) for x in items.wilson(k, len(ix))]
fl = [i for i in S if et[i] != "ok"]; k = sum(direct[i] for i in fl)
out["sonnet_direct_letter_flagged"] = [k, len(fl)] + [round(x, 4) for x in items.wilson(k, len(fl))]
json.dump(out, open(f"{R}/results/sonnet.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
