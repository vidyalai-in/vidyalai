"""Agreement between an LLM judge and the hidden tests on the coding runs."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from evals import wilson  # noqa: E402

RUNS = Path("/home/amit/.cache/vidyalai-scratch/a11/runs")
name = sys.argv[1] if len(sys.argv) > 1 else "haiku"
rows = [json.loads(l) for l in (RUNS / f"judge-{name}.jsonl").read_text().splitlines()]
print(f"judge {name}: {len(rows)} verdicts, missing verdicts {sum(r['verdict'] is None for r in rows)}, "
      f"cost ${sum(r['cost'] for r in rows):.4f}, per verdict ${sum(r['cost'] for r in rows)/len(rows):.5f}")


def stats(rs, label):
    tp = sum(r["tests"] and r["verdict"] == "PASS" for r in rs)
    fn = sum(r["tests"] and r["verdict"] != "PASS" for r in rs)
    fp = sum((not r["tests"]) and r["verdict"] == "PASS" for r in rs)
    tn = sum((not r["tests"]) and r["verdict"] != "PASS" for r in rs)
    n = len(rs)
    po = (tp + tn) / n
    pj, pt = (tp + fp) / n, (tp + fn) / n
    pe = pj * pt + (1 - pj) * (1 - pt)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    print(f"{label:12s} n={n} tests-pass {tp+fn} tests-fail {fp+tn} | TP {tp} FN {fn} FP {fp} TN {tn} | "
          f"agree {po:.3f} kappa {kappa:.3f} TPR {tpr:.3f} {tuple(round(x,3) for x in wilson(tp, tp+fn))} "
          f"TNR {tnr:.3f} {tuple(round(x,3) for x in wilson(tn, tn+fp))} | "
          f"score by tests {pt:.3f}, by judge {pj:.3f}")


stats(rows, "all")
for c in sorted({r["config"] for r in rows}):
    stats([r for r in rows if r["config"] == c], c)
by = defaultdict(Counter)
for r in rows:
    if (r["verdict"] == "PASS") != r["tests"]:
        by[r["task"]]["FP" if r["verdict"] == "PASS" else "FN"] += 1
print("disagreements by task:", dict(by))
