"""Re-ask every sandbag item whose 8-token reply named no letter, with max_tokens 1024 (same prompt); scored by its final letter."""
import sys, copy
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals")
from lab.validity import extract
from lab.grading import run_batch, load_runs
R = load_runs("/home/amit/.cache/vidyalai-scratch/e10/runs/sandbag.jsonl")
jobs = {}
for k, r in R.items():
    if extract(r.get("text"), 4) is None:
        p = copy.deepcopy(r["params"]); p["max_tokens"] = 1024; jobs[k] = p
print(len(jobs))
print(run_batch("e10", jobs, "/home/amit/.cache/vidyalai-scratch/e10/runs/sandbag_retry.jsonl", "e10 sandbag retry"))
