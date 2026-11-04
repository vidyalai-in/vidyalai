"""All Part 9 numbers from the saved runs -> runs/e09-results.json (small) and printed tables."""
import json, sys, collections, glob
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import longctx as lc, stats
R = "/home/amit/.cache/vidyalai-scratch/e09/runs"


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return rows


def score(r):
    if r["task"] == "vt":
        c, rec = lc.grade_vt(r["reply"], r["gold"])
        return dict(correct=c, recall=rec, lenient=c)
    len_ = lc.grade_name(r["reply"], r["gold"])
    return dict(correct=int(len_ and not lc.hedged(r["reply"])), lenient=len_, hedged=int(lc.hedged(r["reply"])))


def cell(v):
    k, n = int(sum(v)), len(v)
    lo, hi = stats.wilson(k, n)
    return dict(k=k, n=n, rate=k / n if n else None, lo=lo, hi=hi)


out = {}
# ---------------- Haiku, thinking off: grid + long
grid = [r for r in load([f"{R}/grid-{s}.jsonl" for s in ("primer", "small", "large")]) if r["task"] != "know"]
know = [r for r in load([f"{R}/grid-primer.jsonl"]) if r["task"] == "know"]
out["haiku_know"] = cell([lc.grade_city(r["reply"], r["gold"]) for r in know])
longr = [r for r in load([f"{R}/long200k.jsonl", f"{R}/long500k.jsonl"]) if r.get("event") == "answer"]
for r in longr:
    r["L"] = 200000 if r["name"] == "long200k" else 500000
    r["n_tokens"] = r["usage"]["input_tokens"] + r["usage"]["cache_read_input_tokens"] + r["usage"]["cache_creation_input_tokens"]
    r["h"] = 0
allh = grid + longr
for r in allh:
    r.update(score(r))
tab = collections.defaultdict(lambda: collections.defaultdict(list))
toks = collections.defaultdict(list)
for r in allh:
    tab[r["task"]][r["L"]].append(r["correct"])
    if r["task"] == "onehop":
        tab["onehop_lenient"][r["L"]].append(r["lenient"])
    if r["task"] == "vt":
        tab["vt_recall"][r["L"]].append(r["recall"])
    toks[r["L"]].append(r["n_tokens"])
out["haiku"] = {t: {L: (cell(v) if t != "vt_recall" else dict(mean=float(np.mean(v)), n=len(v))) for L, v in sorted(d.items())}
                for t, d in tab.items()}
out["haiku_tokens"] = {L: dict(mean=float(np.mean(v)), min=int(min(v)), max=int(max(v))) for L, v in sorted(toks.items())}
cut = collections.defaultdict(list)
for r in allh:
    cut[r["task"]].append(int(r.get("stop") == "max_tokens"))
out["haiku_cutoff_share"] = {t: cell(v) for t, v in cut.items()}
# errors on literal: wrong inserted name vs denial
lit_wrong = [r for r in allh if r["task"] == "literal" and r["L"] > 0 and not r["correct"]]
out["literal_errors"] = dict(n=len(lit_wrong),
                             other_needle=sum(any(n in r["reply"] for n in lc.NAMES) and not lc.hedged(r["reply"]) for r in lit_wrong),
                             denial=sum(lc.hedged(r["reply"]) for r in lit_wrong))
vt_first = [r for r in allh if r["task"] == "vt" and r["L"] > 0 and not r["correct"]]
out["vt_errors"] = dict(n=len(vt_first), only_first=sum(set(__import__("re").findall(r"\b[A-Z]{5}\b", lc.final_line(r["reply"]))) == {r["gold"][0]} for r in vt_first))
# denial mentions "inserted"/"unrelated"/"out of place"
hop = [r for r in allh if r["task"] == "onehop" and r["L"] > 0]
import re
flag = re.compile(r"insert|unrelated|out[- ]of[- ]place|irrelevant|stray|odd|random|garbled", re.I)
out["onehop_flags_needle"] = cell([int(bool(flag.search(r["reply"]))) for r in hop])
out["onehop_hedged"] = cell([r["hedged"] for r in hop])
out["onehop_mentions_gold"] = cell([int(r["gold"] in r["reply"]) for r in hop])
# literal vs one-hop, paired by needle, L >= 4K (grid only)
pairs = {}
for r in grid:
    if r["L"] > 0 and r["task"] in ("literal", "onehop"):
        pairs.setdefault((r["L"], r["h"], r["gold"]), {})[r["task"]] = r["correct"]
a = [p["literal"] for p in pairs.values()]; b = [p["onehop"] for p in pairs.values()]
cl_ = [f"{k[0]}-{k[1]}" for k in pairs]
pdiff = stats.paired_diff(a, b, clusters=cl_)
out["lit_vs_hop"] = dict(n=len(a), literal=float(np.mean(a)), onehop=float(np.mean(b)), diff=pdiff["diff"],
                         ci=pdiff["ci_paired_clustered"], mcnemar=stats.mcnemar(a, b))
# depth: pooled 4K-96K, five bins
depth = collections.defaultdict(lambda: collections.defaultdict(list))
for r in grid:
    if r["L"] > 0 and r["task"] in ("literal", "onehop"):
        bn = min(4, int(r["depth"] * 5))
        depth[r["task"]][bn].append(r["correct"])
out["depth"] = {t: {b: cell(v) for b, v in sorted(d.items())} for t, d in depth.items()}
# design effect of haystack clustering on the pooled literal and one-hop scores
for t in ("literal", "onehop"):
    rs = [r for r in grid if r["L"] > 0 and r["task"] == t]
    out[f"deff_{t}"] = float(stats.design_effect([r["correct"] for r in rs], [f"{r['L']}-{r['h']}" for r in rs]))
# effective length: longest tested length whose score stays >= 85% of the needles-only score (NoLiMa's rule)
eff = {}
for t in ("literal", "onehop", "vt"):
    base = out["haiku"][t][0]["rate"]
    ok = [L for L, c in out["haiku"][t].items() if L > 0 and c["rate"] >= 0.85 * base]
    # "effective" = longest L such that every shorter tested L also passes
    Ls = sorted(L for L in out["haiku"][t] if L > 0)
    e = 0
    for L in Ls:
        if out["haiku"][t][L]["rate"] >= 0.85 * base:
            e = L
        else:
            break
    eff[t] = dict(base=base, threshold=0.85 * base, effective=e)
out["haiku_effective"] = eff
# ---------------- thinking arm, paired with thinking off on the same requests
th = load([f"{R}/grid-tprimer.jsonl", f"{R}/grid-trest.jsonl"])
off = {r["custom_id"]: r for r in grid}
th_out = {}
for t in ("onehop", "vt"):
    for L in (8000, 32000, 96000):
        on_ = [r for r in th if r["task"] == t and r["L"] == L]
        x_on = [score(r)["correct"] for r in on_]
        x_off = [off[r["custom_id"]]["correct"] for r in on_]
        pd = stats.paired_diff(x_off, x_on)
        th_out[f"{t}-{L}"] = dict(on=cell(x_on), off=cell(x_off), diff=pd["diff"], ci=pd["ci_paired"],
                                  mcnemar=stats.mcnemar(x_off, x_on),
                                  out_tokens=float(np.mean([r["usage"]["output_tokens"] for r in on_])))
out["thinking"] = th_out
# ---------------- open models
for tag in ("q05", "q15"):
    rows = load([f"{R}/open-{tag}.jsonl"])
    hs = [r for r in rows if r.get("event") == "haystack"]
    ans = [r for r in rows if not r.get("event") and r["task"] != "know"]
    kn = [r for r in rows if not r.get("event") and r["task"] == "know"]
    t2 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ans:
        r.update(score(r))
        t2[r["task"]][r["Lq"]].append(r["correct"])
        if r["task"] == "onehop":
            t2["onehop_lenient"][r["Lq"]].append(r["lenient"])
    out[f"open_{tag}"] = dict(
        know=cell([r["correct"] for r in kn]),
        tasks={t: {L: cell(v) for L, v in sorted(d.items())} for t, d in t2.items()},
        mem={r["Lq"]: dict(prefix_tokens=r["prefix_tokens"], kv_mib=r["kv_cache_bytes"] / 2**20,
                            peak_mib=r["peak_all_bytes"] / 2**20, prefill_s=r["prefill_s"]) for r in hs if r["h"] == 0},
        locked_s=None)
# ---------------- costs from the ledger
led = [json.loads(l) for l in open("/home/amit/.cache/vidyalai-scratch/agents/lab/ledger.jsonl")]
led = [e for e in led if e["part"] == "e09"]
cost = collections.defaultdict(float)
for e in led:
    n = e["note"]
    k = ("pilot" if n.startswith("pilot") else "grid-off" if n in ("grid primer", "grid small", "grid large")
         else "grid-thinking" if n.startswith("grid t") else "long200k" if n.startswith("long200k") else
         "long500k" if n.startswith("long500k") else n)
    cost[k] += e["cost"]
out["cost"] = dict(cost, total=sum(cost.values()))
json.dump(out, open(f"{R}/e09-results.json", "w"), indent=1, default=float)
print("saved")
# literal: edge depths (first and last fifth) vs the middle three fifths, haystack-cluster bootstrap
rs = [r for r in grid if r["L"] > 0 and r["task"] == "literal"]
edge = np.array([min(4, int(r["depth"] * 5)) in (0, 4) for r in rs]); cor = np.array([r["correct"] for r in rs])
hid = np.array([f"{r['L']}-{r['h']}" for r in rs]); uh = np.unique(hid)
rng = np.random.default_rng(0); diffs = []
for _ in range(10000):
    pick = rng.choice(uh, len(uh))
    idx = np.concatenate([np.where(hid == u)[0] for u in pick])
    e, m = cor[idx][edge[idx]], cor[idx][~edge[idx]]
    diffs.append(e.mean() - m.mean())
out["literal_edge_vs_middle"] = dict(edge=cell(cor[edge].tolist()), middle=cell(cor[~edge].tolist()),
                                     diff=float(cor[edge].mean() - cor[~edge].mean()),
                                     ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
json.dump(out, open(f"{R}/e09-results.json", "w"), indent=1, default=float)
print(out["literal_edge_vs_middle"])
