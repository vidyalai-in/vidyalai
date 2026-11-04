"""Label-quality audit: Haiku (K samples per item) on MMLU-Redux 2.0, scored against the MMLU key.
Writes e02/results/audit.json. All intervals: 95%, bootstrap over items (10,000 resamples)."""
import json, sys, collections
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import items

R = "/home/amit/.cache/vidyalai-scratch/e02"
data = items.load_redux()
N = len(data)
gold = np.array([it["answer"] for it in data])
etype = np.array([it["error_type"] for it in data])
flag = etype != "ok"
sugg = np.array([items.suggested_answer(it) if items.suggested_answer(it) is not None else -1 for it in data])

def load(name, reask=True):
    """picks[i] = list of letter indices (-1 = unparsed) for item i. With reask, a reply that had no
    leading letter is replaced by the structured-output re-ask of the same request (runs/NAME-reask.jsonl)."""
    import os
    picks = collections.defaultdict(list)
    bad = []
    fix = {}
    if reask and os.path.exists(f"{R}/runs/{name}-reask.jsonl"):
        for line in open(f"{R}/runs/{name}-reask.jsonl"):
            r = json.loads(line)
            if r["result_type"] == "succeeded":
                fix[r["custom_id"]] = json.loads(r["text"])["answer"]
    for line in open(f"{R}/runs/{name}.jsonl"):
        r = json.loads(line)
        i = int(r["custom_id"][1:5])
        if r["result_type"] != "succeeded":
            bad.append(r["custom_id"]); continue
        L = items.parse_letter(r["text"])
        if L is None and r["custom_id"] in fix:
            L = fix[r["custom_id"]]
        if L is None:
            bad.append((r["custom_id"], r["text"], r.get("stop")))
        picks[i].append(items.LETTERS.index(L) if L else -1)
    return picks, bad

def summarize(name, reask=True):
    picks, bad = load(name, reask)
    idx = np.array(sorted(picks))
    P = np.array([picks[i] for i in idx])          # (items, K)
    K = P.shape[1]
    acc_key = (P == gold[idx, None]).mean(1)        # per-item accuracy vs MMLU key
    acc_sug = (P == sugg[idx, None]).mean(1)        # vs annotators' answer (wrong_groundtruth only)
    fl = flag[idx]; et = etype[idx]; wg = (et == "wrong_groundtruth") & (sugg[idx] >= 0)
    out = dict(model=name, items=len(idx), samples_per_item=K, unparsed=len(bad), unparsed_examples=bad[:5])
    # weights: for stratified subsets, reweight strata to the full 5,700-item population
    w = np.ones(len(idx))
    if len(idx) < N:
        for f in (True, False):
            w[fl == f] = (flag == f).sum() / (fl == f).sum()
    def wmean(x, ix): return np.sum(x[ix] * w[ix]) / np.sum(w[ix]) if len(ix) else np.nan
    def bci(fn, mask=None):
        base = np.where(mask)[0] if mask is not None else np.arange(len(idx))
        p, lo, hi = items.boot_ci(lambda s: fn(base[s]), len(base), reps=10000)
        return [round(float(p), 4), round(float(lo), 4), round(float(hi), 4), int(len(base))]
    by = {}
    for t in ["ok"] + items.FLAW_TYPES:
        m = et == t
        if m.sum():
            by[t] = bci(lambda s: acc_key[s].mean(), m)
    by["all_flagged"] = bci(lambda s: acc_key[s].mean(), fl)
    out["acc_by_type"] = by
    # wrong_groundtruth: how often the model picks the annotators' answer instead of the key
    out["wrong_gt_picks_annotator"] = bci(lambda s: acc_sug[s].mean(), wg)
    out["wrong_gt_picks_key"] = bci(lambda s: acc_key[s].mean(), wg)
    # headline scores
    allm = np.arange(len(idx))
    def headline(s):
        return wmean(acc_key, s)
    def dropped(s):
        s = s[~fl[s]]; return wmean(acc_key, s)
    def corrected(s):
        keep = s[(~fl[s]) | wg[s]]
        sc = np.where(wg[keep], acc_sug[keep], acc_key[keep])
        return np.sum(sc * w[keep]) / np.sum(w[keep])
    out["score_original"] = bci(headline)
    out["score_drop_flagged"] = bci(dropped)
    out["score_corrected"] = bci(corrected)
    out["delta_drop_minus_original"] = bci(lambda s: dropped(s) - headline(s))
    out["delta_corrected_minus_original"] = bci(lambda s: corrected(s) - headline(s))
    # share of the model's errors (vs key) that fall on flagged items
    err = 1 - acc_key
    out["share_errors_on_flagged"] = bci(lambda s: np.sum((err * w)[s][fl[s]]) / np.sum((err * w)[s]))
    out["share_items_flagged"] = round(float(flag.mean()), 4)
    # items the model gets wrong on every sample
    allwrong = acc_key == 0
    out["items_wrong_every_sample"] = int(allwrong.sum())
    out["share_flagged_among_always_wrong"] = bci(lambda s: fl[s][allwrong[s]].mean() if allwrong[s].any() else np.nan)
    # sample consistency: items where the K samples disagree
    out["items_with_disagreeing_samples"] = int((P.min(1) != P.max(1)).sum())
    return out, P, idx

if __name__ == "__main__":
    res = {}
    strict = "--strict" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name in names:
        o, P, idx = summarize(name, reask=not strict)
        res[name] = o
        np.save(f"{R}/results/picks-{name}{'-strict' if strict else ''}.npy", np.c_[idx, P])
        print(json.dumps(o, indent=1))
    json.dump(res, open(f"{R}/results/audit-{'_'.join(names)}{'-strict' if strict else ''}.json", "w"), indent=1)
