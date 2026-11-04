"""Sandbagging probe with adaptive thinking (effort low, summarized thinking shown): 400 random MMLU items (seed 0) x neutral / high_penalized."""
import sys, random
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals")
from lab import safety_evals as S
from lab.validity import load
from lab.grading import run_batch
items = load("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json")
sub = random.Random(0).sample(items, 400)
jobs = {}
for c in ("neutral", "high_penalized"):
    for it in sub:
        p = S.mmlu_request(it, c, max_tokens=768, thinking={"type": "adaptive", "display": "summarized"})
        p["output_config"] = {"effort": "low"}
        jobs[f"{c}__{it['id']}"] = p
print(run_batch("e10", jobs, "/home/amit/.cache/vidyalai-scratch/e10/runs/sandbag_think.jsonl", "e10 sandbag thinking"))
