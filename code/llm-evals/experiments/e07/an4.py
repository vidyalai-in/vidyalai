"""Analyze the mini arena: judge parse rate, position bias, order consistency, BT with
prompt-clustered bootstrap, style-controlled BT, and comparison with MMLU-1140 accuracy."""
import sys, json, re, collections, numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import arena, stats
D = "/home/amit/.cache/vidyalai-scratch/e07"
CF = ["haiku-off", "haiku-off-markdown", "haiku-off-concise", "haiku-low", "sonnet-off"]
gens = {r["id"]: r for r in map(json.loads, open(f"{D}/runs/gen60.jsonl"))}
J = [json.loads(l) for l in open(f"{D}/runs/judge60.jsonl")]
def parse(t):
    t = (t or "").strip().strip("*.").upper()
    return t if t in ("A", "B", "TIE") else None
def feats(text):
    lines = text.splitlines()
    return [len(text), sum(bool(re.match(r"#{1,6}\s", l)) for l in lines),
            sum(bool(re.match(r"\s*([-*+]|\d+[.)])\s", l)) for l in lines), len(re.findall(r"\*\*[^*\n]+\*\*", text))]
rows, bad, ok = [], [], 0
for j in J:
    m = re.match(r"j(\d+)-(.+)-vs-(.+)$", j["id"]); i, x, y = int(m[1]), m[2], m[3]
    v = parse(j.get("text")) if j["type"] == "succeeded" else None
    if v is None: bad.append((j["id"], j.get("text"), j.get("stop"))); continue
    rows.append((i, x, y, v))
out = {"n_judgments": len(J), "unparsed": len(bad), "unparsed_examples": bad[:5]}
cnt = collections.Counter(v for *_, v in rows); out["verdicts"] = dict(cnt)
# position bias among decisive verdicts
dec = cnt["A"] + cnt["B"]; out["first_wins"] = [cnt["A"], dec, stats.wilson(cnt["A"], dec)]
# order consistency per prompt and unordered pair
res = {(i, x, y): v for i, x, y, v in rows}
cons = tot = 0
for (i, x, y), v in res.items():
    if x < y and (i, y, x) in res:
        w = res[(i, y, x)]; tot += 1
        cons += (v == "A" and w == "B") or (v == "B" and w == "A") or (v == "TIE" and w == "TIE")
out["order_consistent"] = [cons, tot, stats.wilson(cons, tot)]
idx = {c: k for k, c in enumerate(CF)}
a = np.array([idx[x] for i, x, y, v in rows]); b = np.array([idx[y] for i, x, y, v in rows])
yv = np.array([{"A": 1.0, "B": 0.0, "TIE": 0.5}[v] for *_, v in rows]); pr = np.array([i for i, *_ in rows])
bt = arena.fit_bt(a, b, yv, 5); bo = arena.bootstrap_bt(a, b, yv, 5, n_boot=2000, seed=0, clusters=pr)
out["bt"] = dict(zip(CF, bt.round(1).tolist()))
out["bt_ci"] = dict(zip(CF, np.percentile(bo, [2.5, 97.5], 0).T.round(1).tolist()))
# win rate of each config against all others
wr = {}
for c in CF:
    k = idx[c]; s = np.r_[yv[a == k], 1 - yv[b == k]]; wr[c] = [round(float(s.mean()) * 100, 1), len(s)]
out["score_vs_field"] = wr
# style control on our own features
F = {(i, c): feats(gens[f"g{i}-{c}"]["text"]) for i in range(60) for c in CF}
fa = np.array([F[(i, x)] for i, x, y, v in rows]); fb = np.array([F[(i, y)] for i, x, y, v in rows])
Z = arena.style_features(fa, fb)
sc, g = arena.fit_style_bt(a, b, yv, 5, Z)
cl, inv = np.unique(pr, return_inverse=True); rng = np.random.default_rng(0); S = []
for t in range(500):
    w = np.bincount(rng.integers(0, len(cl), len(cl)), minlength=len(cl))[inv]
    S.append(np.r_[arena.fit_style_bt(a, b, yv, 5, Z, w=w)[0], arena.fit_style_bt(a, b, yv, 5, Z, w=w)[1]])
S = np.array(S)
out["sc"] = dict(zip(CF, sc.round(1).tolist())); out["sc_ci"] = dict(zip(CF, np.percentile(S[:, :5], [2.5, 97.5], 0).T.round(1).tolist()))
out["sc_gamma"] = g.round(3).tolist(); out["sc_gamma_ci"] = np.percentile(S[:, 5:], [2.5, 97.5], 0).T.round(3).tolist()
out["mean_feats"] = {c: np.mean([F[(i, c)] for i in range(60)], 0).round(1).tolist() for c in CF}
out["stops"] = {c: collections.Counter(gens[f"g{i}-{c}"]["stop"] for i in range(60)) for c in CF}
# markdown vs plain Haiku head to head
hh = [v for i, x, y, v in rows if {x, y} == {"haiku-off", "haiku-off-markdown"}]
mdw = sum((v == "A") == (x == "haiku-off-markdown") and v != "TIE" for i, x, y, v in rows if {x, y} == {"haiku-off", "haiku-off-markdown"})
out["md_vs_plain"] = [mdw, len(hh), collections.Counter(hh)]
json.dump(out, open(f"{D}/runs/exp4.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
