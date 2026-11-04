"""Reference-based metrics on SummEval's 1,600 summaries (CPU). Writes runs/metrics.json:
{metric: [[score per system] per doc]} for the original reference alone ("_1ref") and the
best over all 11 references ("_11ref")."""
import json
import sys

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import grading as G

D = "/home/amit/.cache/vidyalai-scratch/e06/"
docs = json.load(open(D + "data/summeval.json"))
out = {}
for tag, nref in (("1ref", 1), ("11ref", 11)):
    for name in ("rouge1", "rouge2", "rougeL", "bleu", "chrf", "token_f1"):
        out[f"{name}_{tag}"] = []
    for d in docs:
        refs = d["refs"][:nref]
        rows = {k: [] for k in ("rouge1", "rouge2", "rougeL", "bleu", "chrf", "token_f1")}
        for s in d["summaries"]:
            r = G.rouge(s, refs)
            for k in ("rouge1", "rouge2", "rougeL"):
                rows[k].append(r[k])
            rows["bleu"].append(G.bleu(s, refs))
            rows["chrf"].append(G.chrf(s, refs))
            rows["token_f1"].append(max(G.token_f1(s, x) for x in refs))
        for k, v in rows.items():
            out[f"{k}_{tag}"].append(v)
# a length baseline: words in the summary
out["length"] = [[len(s.split()) for s in d["summaries"]] for d in docs]
json.dump(out, open(D + "runs/metrics.json", "w"))
print({k: round(sum(map(sum, v)) / 1600, 3) for k, v in out.items()})
