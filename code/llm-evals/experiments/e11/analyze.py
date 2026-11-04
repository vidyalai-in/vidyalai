"""Results for Part 11: accuracy per configuration, paired differences, MDE, closed-book check, costs."""
import sys, json, numpy as np, collections
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); import own_eval as oe, stats
items = [json.loads(l) for l in open("data/items.jsonl")]
ID = {it["id"]: it for it in items}
def load(tag, item_map=ID):
    out = collections.defaultdict(dict); meta = []
    for l in open(f"runs/{tag}.jsonl"):
        r = json.loads(l); rep, iid = r["id"].split("_", 1)
        if iid not in item_map: continue
        ans, rule = oe.final_answer(r.get("text"))
        out[iid][rep] = oe.grade(item_map[iid], ans)
        meta.append(dict(id=iid, rep=rep, stop=r.get("stop"), rule=rule, type=r["type"], out=(r.get("usage") or {}).get("output_tokens")))
    return out, meta
R = {}
res = {}
for tag in ("main_off", "main_low"):
    g, meta = load(tag); R[tag] = (g, meta)
    reps = sorted({m["rep"] for m in meta})
    X = np.array([[g[it["id"]].get(r, 0) for r in reps] for it in items], float)
    s = X.mean(1); clusters = [it["part"] for it in items]
    m, lo, hi = stats.clt_ci(s)
    cse = stats.clustered_se(s, clusters)
    kinds = {k: float(s[[it["kind"] == k for it in items]].mean()) for k in ("number", "text", "none")}
    res[tag] = dict(n=len(items), K=len(reps), acc=m, ci=(lo, hi), ci_clustered=(m - 1.96 * cse, m + 1.96 * cse),
                    per_rep=[float(X[:, j].mean()) for j in range(X.shape[1])], kinds=kinds,
                    cutoff=sum(x["stop"] == "max_tokens" for x in meta), no_answer_line=sum(x["rule"] != "answer_line" for x in meta),
                    errored=sum(x["type"] != "succeeded" for x in meta), n_replies=len(meta),
                    out_tokens_median=float(np.median([x["out"] or 0 for x in meta])), vs=(stats.variance_split(X) if X.var() > 0 else 'no variance'))
    res[tag]["X"] = X
XA, XB = res["main_off"].pop("X"), res["main_low"].pop("X")
pd = stats.paired_diff(XA.mean(1), XB.mean(1), clusters=[it["part"] for it in items])
def safe_vs(X):
    n, K = X.shape; p = X.mean(1)
    within = float(np.mean(p * (1 - p) * K / (K - 1))); between = float(p.var(ddof=1) - within / K)
    return within, between
wa, ba = safe_vs(XA); wb, bb = safe_vs(XB)
cov = float(np.cov(XA.mean(1), XB.mean(1), ddof=1)[0, 1])
pin = dict(omega2=ba + bb - 2 * cov, sigma2_a=wa, sigma2_b=wb)
res["low_minus_off"] = dict(pd, mcnemar_rep1=stats.mcnemar(XA[:, 0], XB[:, 0]), sign_flip_p=stats.sign_flip_p(XA.mean(1), XB.mean(1)),
                            power_inputs=pin, mde_k3=stats.mde(len(items), k_a=3, k_b=3, **pin), mde_k1=stats.mde(len(items), **pin),
                            n_for_observed=stats.sample_size(abs(pd["diff"]), **pin) if pd["diff"] else None,
                            items_differ=int((XA.mean(1) != XB.mean(1)).sum()))
# repeat consistency within a config
for tag, X in (("main_off", XA), ("main_low", XB)):
    res[tag]["items_inconsistent"] = int(((X.min(1) == 0) & (X.max(1) == 1)).sum())
# validation passes (medium effort, 2 reps) on items unchanged since validation
cand = {json.loads(l)["id"]: json.loads(l) for l in open("data/candidates_fixed.jsonl")}
same = [it for it in items if cand[it["id"]]["question"] == it["question"] and cand[it["id"]]["answer"] == it["answer"]]
gv, _ = load("val", cand)
XM = np.array([[gv[it["id"]][r] for r in ("r1", "r2")] for it in same], float)
idx = [items.index(it) for it in same]
res["medium_val"] = dict(n=len(same), acc=float(XM.mean()), ci=stats.clt_ci(XM.mean(1))[1:])
res["medium_minus_low"] = stats.paired_diff(XB[idx].mean(1), XM.mean(1))
# closed book
gc, metac = load("closed")
ans_items = [it for it in items if it["kind"] != "none"]
c = np.array([gc[it["id"]].get("r1", 0) for it in ans_items])
lo, hi = stats.wilson(int(c.sum()), len(c))
res["closed"] = dict(n=len(c), correct=int(c.sum()), acc=float(c.mean()), ci=(lo, hi),
                     by_kind={k: [int(c[[it["kind"] == k for it in ans_items]].sum()), sum(it["kind"] == k for it in ans_items)] for k in ("number", "text")},
                     guessable=[it["id"] for it, x in zip(ans_items, c) if x])
# open-book accuracy on items closed book got wrong
keep = [i for i, it in enumerate(items) if not (it["kind"] != "none" and gc[it["id"]].get("r1", 0))]
res["article_only"] = dict(n=len(keep), off=float(XA[keep].mean()), low=float(XB[keep].mean()),
                           diff=stats.paired_diff(XA[keep].mean(1), XB[keep].mean(1)))
# costs from the ledger
L = [json.loads(l) for l in open("/home/amit/.cache/vidyalai-scratch/agents/lab/ledger.jsonl")] if False else []
json.dump(res, open("results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:6000])
