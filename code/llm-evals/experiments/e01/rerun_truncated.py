"""Re-ask every request whose 8-token reply was cut off before it named a letter, with the
same prompt and max_tokens=1024. Usage: rerun_truncated.py haiku|sonnet [max_tokens, default 1024]"""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import validity as V
R = "/home/amit/.cache/vidyalai-scratch/e01/runs"
name = sys.argv[1]
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
import os
done = set()
if os.path.exists(f"{R}/{name}-rerun.jsonl"):
    done = {json.loads(l)["id"] for l in open(f"{R}/{name}-rerun.jsonl")}
jobs = {}
for l in open(f"{R}/{name}.jsonl"):
    r = json.loads(l)
    if r["type"] == "succeeded" and r.get("stop") == "max_tokens" and V.extract(r["text"], 10) is None and r["id"] not in done:
        jobs[r["id"]] = {**r["params"], "max_tokens": CAP}
print(name, "rerun", len(jobs), sorted(jobs)[:5], flush=True)
if len(sys.argv) > 3 and sys.argv[3] == "--dry":
    sys.exit()
print(V.run_batch("e01", jobs, f"{R}/{name}-rerun.jsonl", note=f"e01 {name} rerun {CAP}", count=True), flush=True)
