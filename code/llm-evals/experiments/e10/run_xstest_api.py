"""XSTest replies from Claude Haiku 5.5: 450 prompts x 2 repeats, thinking off, no system prompt, max_tokens 256."""
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals")
from lab import safety_evals as S
from lab.grading import run_batch
model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
mt = int(sys.argv[3]) if len(sys.argv) > 3 else 256
tag = model.split("-")[1]
jobs = {f"r{k}-{r['id']}": S.reply_request(r["prompt"], model=model, max_tokens=mt) for k in range(reps) for r in S.load_xstest()}
print(run_batch("e10", jobs, f"/home/amit/.cache/vidyalai-scratch/e10/runs/xstest_{tag}.jsonl", f"e10 xstest {tag}"))
