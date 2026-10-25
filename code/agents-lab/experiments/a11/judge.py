"""Part 11: an LLM judge grades the coding runs' patches; compare its verdicts with the hidden tests.

usage: judge.py JUDGE_MODEL CONFIG [CONFIG ...] [--subset FILE]
The judge sees the full bug report and the agent's diff (never the tests or the test result) and ends
with VERDICT: PASS or VERDICT: FAIL. Results go to runs/judge-<model>.jsonl, resumable, 3 at a time.
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402

A11 = Path("/home/amit/.cache/vidyalai-scratch/a11")
RUNS = A11 / "runs"
LOCK = threading.Lock()
CL = ledger.client().with_options(max_retries=8)

SYSTEM = """You review patches to Python libraries. You get a bug report and a patch (a unified diff)
written by someone else. You cannot run code. Decide whether the patch fully fixes the bug the report
describes, for every case the report mentions, without breaking existing behaviour. Edits to test files
do not count either way. Explain briefly, then end with exactly one line: VERDICT: PASS or VERDICT: FAIL"""


def tasks():
    return {d.name: json.loads((d / "task.json").read_text())
            for d in (A11 / "coding" / "tasks").iterdir() if (d / "task.json").exists()}


def one(model, rec, task, out):
    tag = f"{rec['config']}-{rec['task']}-r{rec['rep']}"
    diff = (RUNS / "diffs" / f"{tag}.diff").read_text()
    msg = (f"Repository: {task['repo']}\n\n<bug_report>\n{task['issue'].strip()}\n</bug_report>\n\n"
           f"<patch>\n{diff.strip() or '(empty patch: no files changed)'}\n</patch>")
    resp = ledger.create(CL, "a11", note=f"a11 judge {tag}", model=model, max_tokens=8000,
                         system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                         thinking={"type": "adaptive"}, output_config={"effort": "medium"},
                         messages=[{"role": "user", "content": msg}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.findall(r"VERDICT:\s*(PASS|FAIL)", text)
    row = dict(tag=tag, config=rec["config"], task=rec["task"], rep=rec["rep"], judge=model,
               verdict=m[-1] if m else None, tests=rec["correct"], diff_lines=rec.get("diff_lines"),
               cost=ledger.cost_of(model, resp.usage), usage=resp.usage.model_dump(), text=text)
    with LOCK, out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{tag:45s} judge {row['verdict']} tests {'PASS' if rec['correct'] else 'FAIL'} ${row['cost']:.5f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--subset", default="")
    a = ap.parse_args()
    T = tasks()
    recs = [json.loads(l) for c in a.configs for l in (RUNS / f"{c}.jsonl").read_text().splitlines()]
    if a.subset:
        keep = set(Path(a.subset).read_text().split())
        recs = [r for r in recs if f"{r['config']}-{r['task']}-r{r['rep']}" in keep]
    out = RUNS / f"judge-{a.model.split('-')[1]}.jsonl"
    done = {json.loads(l)["tag"] for l in out.read_text().splitlines()} if out.exists() else set()
    recs = [r for r in recs if f"{r['config']}-{r['task']}-r{r['rep']}" not in done]
    print(f"{len(recs)} to judge; a11 spent ${ledger.spent('a11'):.4f}", flush=True)
    with ThreadPoolExecutor(3) as ex:
        futs = [ex.submit(one, a.model, r, T[r["task"]], out) for r in recs]
        for f in futs:
            try:
                f.result()
            except ledger.BudgetExceeded as e:
                print("BUDGET STOP:", e, flush=True)
                ex.shutdown(cancel_futures=True)
                break
    print(f"a11 spent: ${ledger.spent('a11'):.4f}", flush=True)


if __name__ == "__main__":
    main()
