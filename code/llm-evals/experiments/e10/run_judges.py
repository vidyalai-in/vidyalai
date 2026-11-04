"""Judge XSTest replies with Claude Haiku 5.5: XSTest's 3-class prompt on every reply; the StrongREJECT rubric
on replies to the 200 unsafe prompts (first repeat only). Usage: run_judges.py tag [tag ...]"""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals")
from lab import safety_evals as S
from lab.grading import run_batch, load_runs
R = "/home/amit/.cache/vidyalai-scratch/e10/runs"
X = {r["id"]: r for r in S.load_xstest()}
jobs = {}
for tag in sys.argv[1:]:
    for rid, rec in load_runs(f"{R}/xstest_{tag}.jsonl").items():
        x = X[int(rid.split("-")[1])]
        jobs[f"x3-{tag}-{rid}"] = S.judge_request(x["prompt"], rec["text"])
        if x["label"] == "unsafe" and rid.startswith("r0-"):
            jobs[f"sr-{tag}-{rid}"] = S.strongreject_request(x["prompt"], rec["text"])
print(len(jobs))
print(run_batch("e10", jobs, f"{R}/judges.jsonl", "e10 judges " + ",".join(sys.argv[1:])))
