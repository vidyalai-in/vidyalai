"""BERTScore F1 (roberta-large, layer 17, bf16 on the RTX 4060 Laptop GPU) for SummEval's 1,600
summaries, against the original reference and against all 11 (max). Cross-checks our
implementation against the bert-score package (0.3.13) on 64 pairs. Writes runs/bertscore.json."""
import json
import sys
import time

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import grading as G

D = "/home/amit/.cache/vidyalai-scratch/e06/"
docs = json.load(open(D + "data/summeval.json"))
t0 = time.time()
bs = G.BertScorer("FacebookAI/roberta-large", layer=17)
cands = [s for d in docs for s in d["summaries"]]
r1 = [[d["refs"][0]] for d in docs for _ in d["summaries"]]
r11 = [d["refs"] for d in docs for _ in d["summaries"]]
f1 = bs.f1(cands, r1)
f11 = bs.f1(cands, r11)
t1 = time.time()
out = {"bertscore_1ref": [f1[i * 16:(i + 1) * 16] for i in range(100)],
       "bertscore_11ref": [f11[i * 16:(i + 1) * 16] for i in range(100)],
       "seconds": round(t1 - t0, 1)}

# cross-check with the official package on 64 pairs (no idf, no rescaling)
try:
    sys.path.insert(0, D + "pylib")
    import bert_score
    import torch
    P, R, F = bert_score.score(cands[:64], [r[0] for r in r1[:64]], model_type="FacebookAI/roberta-large",
                               num_layers=17, idf=False, batch_size=32, device="cuda")
    diff = (torch.tensor(f1[:64]) - F).abs()
    out["crosscheck"] = dict(n=64, max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                             corr=float(torch.corrcoef(torch.stack([torch.tensor(f1[:64]), F]))[0, 1]))
except Exception as e:  # noqa: BLE001
    out["crosscheck"] = dict(error=repr(e)[:300])
json.dump(out, open(D + "runs/bertscore.json", "w"))
print(out["seconds"], out["crosscheck"], sum(f1) / len(f1), sum(f11) / len(f11))
