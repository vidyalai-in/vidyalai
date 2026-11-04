"""Extra checks: (a) verbatim restatement self-duel; (b) the fluency anchor on SummEval's
lowercased summaries; (c) claude-v1 vs gpt-3.5-turbo head to head (a weak family check).
Writes results/extra.json."""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import grading as G
import stats as ST

D = "/home/amit/.cache/vidyalai-scratch/e06/"
FLIP = {"A": "B", "B": "A", "tie": "tie", None: None}
res = {}

# (a) restatement: in 'ab' the long copy is B; in 'ba' it is A
r = G.load_runs(D + "runs/restate.jsonl")
by = {}
for cid, x in r.items():
    i, _, order = cid.split("-")
    v = G.parse_verdict(x.get("text"))
    by.setdefault(i, {})[order] = v if order == "ab" else FLIP[v]  # now 'B' = long copy wins
long_w = [((d.get("ab") == "B") + (d.get("ba") == "B")) / 2 for d in by.values()]
orig_w = [((d.get("ab") == "A") + (d.get("ba") == "A")) / 2 for d in by.values()]
res["restate"] = dict(n=len(by), long_win=float(np.mean(long_w)), long_ci=G.boot_stat(lambda ix: np.mean(np.array(long_w)[ix]), len(by)),
                      orig_win=float(np.mean(orig_w)), orig_ci=G.boot_stat(lambda ix: np.mean(np.array(orig_w)[ix]), len(by)),
                      per_order={o: dict(Counter(d.get(o) for d in by.values())) for o in ("ab", "ba")},
                      cutoffs=sum(x.get("stop") == "max_tokens" for x in r.values()),
                      unparsed=sum(G.parse_verdict(x.get("text")) is None for x in r.values()))

# (b) fluency anchor
base = G.load_runs(D + "runs/rubric1.jsonl")
docs = json.load(open(D + "data/summeval.json"))
for run in ("fluency_truecase", "fluency_nocaps"):
    rr = G.load_runs(D + f"runs/{run}.jsonl")
    a, b, h = [], [], []
    for cid, x in rr.items():
        di, si = (int(t[1:]) for t in cid.split("-"))
        a.append(json.loads(base[cid]["text"])["fluency"])
        b.append(json.loads(x["text"])["fluency"])
        h.append(np.mean([e["fluency"] for e in docs[di]["expert"][si]]))
    a, b, h = (np.array(z, float) for z in (a, b, h))
    pdiff = ST.paired_diff(a, b)
    from scipy import stats as st
    res[run] = dict(n=len(a), mean_before=float(a.mean()), mean_after=float(b.mean()), human_mean=float(h.mean()),
                    diff=pdiff["diff"], ci=pdiff["ci_paired"],
                    share_le2_before=float((a <= 2).mean()), share_le2_after=float((b <= 2).mean()),
                    spearman_before=float(st.spearmanr(a, h)[0]), spearman_after=float(st.spearmanr(b, h)[0]))

# (c) claude-v1 vs gpt-3.5-turbo head to head
mt = json.load(open(D + "data/mtbench.json"))
pr = G.load_runs(D + "runs/pair.jsonl")
pr.update(G.load_runs(D + "runs/pair.retry.jsonl"))
hs, js, gs = [], [], []
for i, p in enumerate(mt):
    if {p["model_1"], p["model_2"]} != {"claude-v1", "gpt-3.5-turbo"}:
        continue
    a = G.parse_verdict(pr[f"p{i}-ab"]["text"]); b = FLIP[G.parse_verdict(pr[f"p{i}-ba"]["text"])]
    v = a if a == b else "tie"
    claude_is_1 = p["model_1"] == "claude-v1"
    sc = lambda x: {"A": 1.0, "B": 0.0, "tie": 0.5}[x] if claude_is_1 else {"A": 0.0, "B": 1.0, "tie": 0.5}[x]
    for vv in p["votes"]:
        hs.append(sc(vv["vote"])); js.append(sc(v)); gs.append(p["qid"])
hs, js = np.array(hs), np.array(js)
res["claude_vs_gpt35"] = dict(n_votes=len(hs), n_pairs=len(set(gs)), human=float(hs.mean()), haiku=float(js.mean()),
                              diff=float((js - hs).mean()),
                              ci=G.boot_stat(lambda ix: (js - hs)[ix].mean(), len(hs), groups=gs))
json.dump(res, open(D + "results/extra.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
