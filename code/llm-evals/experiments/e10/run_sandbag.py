"""Sandbagging probe: MMLU 1,140 items under five system prompts (neutral twice), Haiku 5.5, thinking off, max_tokens 8."""
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals")
from lab import safety_evals as S
from lab.validity import load
from lab.grading import run_batch
items = load("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json")
conds = {"neutral": "neutral", "neutral2": "neutral", "best": "best", "high_penalized": "high_penalized", "low_penalized": "low_penalized"}
jobs = {f"{c}__{it['id']}": S.mmlu_request(it, sysname) for c, sysname in conds.items() for it in items}
print(run_batch("e10", jobs, "/home/amit/.cache/vidyalai-scratch/e10/runs/sandbag.jsonl", "e10 sandbag"))
