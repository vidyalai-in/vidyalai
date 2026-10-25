"""What fills the window, turn by turn, in one naive run: python anatomy.py TAG TASK REPEAT"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402

tag, task, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ev = [json.loads(l) for l in (HERE / "runs" / "transcripts" / tag / f"{task}-r{k}.jsonl").read_text().splitlines()]
cl = ledger.client()
M = "claude-haiku-5-5"


def count(**kw):
    return cl.messages.count_tokens(model=M, **kw).input_tokens


start = ev[0]
base = count(messages=[{"role": "user", "content": "x"}])
sys_tools = count(system=start["system"], tools=start["tools"], messages=[{"role": "user", "content": "x"}]) - base
tools_only = count(tools=start["tools"], messages=[{"role": "user", "content": "x"}]) - base
task_tok = count(messages=[{"role": "user", "content": start["task"]}]) - base + 1
print(f"system prompt + tools: {sys_tools} (tools {tools_only}, system {sys_tools - tools_only}); task message: {task_tok}")

results_by_step = {e["n"]: e["results"] for e in ev if e["event"] == "tool_results"}
rows, res_cum, thk_cum, act_cum = [], 0, 0, 0
for e in ev:
    if e["event"] != "step":
        continue
    n = e["n"]
    other = e["context"] - sys_tools - task_tok - res_cum - thk_cum - act_cum
    rows.append((n, e["context"], sys_tools, task_tok, res_cum, act_cum, thk_cum, other))
    th = e["usage"]["output_tokens_details"]["thinking_tokens"]
    thk_cum += th
    act_cum += e["output_tokens"] - th
    for r in results_by_step.get(n, []):
        res_cum += count(messages=[{"role": "user", "content": r["content"]}]) - base + 1
print(f"{'step':>4s} {'context':>8s} {'sys+tools':>9s} {'task':>5s} {'results':>8s} {'calls+text':>10s} {'thinking':>8s} {'other':>6s}")
for r in rows:
    print(" ".join(f"{v:>8,}" for v in r))
json.dump(rows, open(HERE / f"anatomy-{Path(tag).name}-{task}-r{k}.json", "w"))
