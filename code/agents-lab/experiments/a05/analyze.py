"""Tables for Part 5 from runs/*.jsonl and the transcripts."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402
sys.path.insert(0, str(HERE))
from store import TASKS, grade  # noqa: E402
TASK = {t["id"]: t for t in TASKS}

ARMS = ["naive", "trunc", "sized", "edit", "compact", "memory"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def load(tag):
    """Run summaries, regraded against the current tasks.json (c07's expected answer was fixed)."""
    p = HERE / "runs" / f"{tag}.jsonl"
    rs = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    for r in rs:
        r["correct"] = r["answer"] is not None and grade(TASK[r["task"]], "ANSWER: " + r["answer"])
    return rs


def steps(tag, task, k):
    p = HERE / "runs" / "transcripts" / tag / f"{task}-r{k}.jsonl"
    return [e for e in map(json.loads, p.read_text().splitlines()) if e["event"] == "step"]


def nocache_cost(st):
    """The same tokens priced with no prompt caching: all input at the base rate."""
    c = 0
    for e in st:
        u = dict(e["usage"])
        u["input_tokens"] = e["context"]
        u["cache_read_input_tokens"] = u["cache_creation_input_tokens"] = 0
        c += ledger.cost_of("claude-haiku-5-5", u)
        for it in u.get("iterations") or []:
            if it.get("type") == "compaction":
                it = dict(it)
                it["input_tokens"] = it["input_tokens"] + (it.get("cache_read_input_tokens") or 0) + (it.get("cache_creation_input_tokens") or 0)
                it["cache_read_input_tokens"] = it["cache_creation_input_tokens"] = 0
                c += ledger.cost_of("claude-haiku-5-5", it)
    return c


def summary(tags=ARMS, kinds=None):
    rows = []
    for tag in tags:
        rs = [r for r in load(tag) if kinds is None or kind_of(r) in kinds]
        if not rs:
            continue
        n, k = len(rs), sum(r["correct"] for r in rs)
        p, lo, hi = wilson(k, n)
        peaks, nc, reads, total_in = [], 0, 0, 0
        for r in rs:
            st = steps(tag, r["task"], r["repeat"])
            peaks.append(max(max(e["context"], e["compaction_in"]) for e in st))
            nc += nocache_cost(st)
            reads += sum(e["cache_read"] + sum((it.get("cache_read_input_tokens") or 0) for it in (e["usage"].get("iterations") or []) if it.get("type") == "compaction") for e in st)
            total_in += sum(e["context"] + e["compaction_in"] for e in st)
        cost = sum(r["cost"] for r in rs)
        rows.append(dict(arm=tag, n=n, correct=k, acc=p, lo=lo, hi=hi,
                         steps=sum(r["steps"] for r in rs) / n, calls=sum(r["tool_calls"] for r in rs) / n,
                         input=total_in / n, output=sum(r["output"] for r in rs) / n,
                         peak=sum(peaks) / n, peak_max=max(peaks), hit=reads / total_in,
                         cost=cost / n, nocache=nc / n, per_correct=cost / k if k else float("nan"),
                         capped=sum(r["stop"] == "max_cost" for r in rs),
                         noans=sum(r["answer"] is None for r in rs)))
    return rows


def kind_of(r):
    return "long" if r["size"] >= 48 else r["kind"]


def show(rows, title):
    print(f"\n== {title}")
    print(f"{'arm':10s} {'n':>3s} {'correct':>16s} {'steps':>6s} {'calls':>6s} {'input':>9s} {'out':>6s} "
          f"{'peak':>7s} {'max':>7s} {'hit%':>5s} {'$/task':>8s} {'$nocache':>8s} {'$/correct':>9s} cap noans")
    for r in rows:
        print(f"{r['arm']:10s} {r['n']:3d} {r['correct']:3d} {100*r['acc']:5.1f} ({100*r['lo']:4.1f}-{100*r['hi']:5.1f}) "
              f"{r['steps']:6.1f} {r['calls']:6.1f} {r['input']:9,.0f} {r['output']:6,.0f} {r['peak']:7,.0f} "
              f"{r['peak_max']:7,.0f} {100*r['hit']:5.1f} {r['cost']:8.5f} {r['nocache']:8.5f} {r['per_correct']:9.5f} "
              f"{r['capped']:3d} {r['noans']:3d}")


if __name__ == "__main__":
    tags = sys.argv[1:] or ARMS
    show(summary(tags), "all tasks")
    for kinds in (["chain"], ["fanout"], ["long"]):
        show(summary(tags, kinds), kinds[0])
    print("\nper-task correct (all repeats):")
    for tag in tags:
        by = defaultdict(list)
        for r in load(tag):
            by[r["task"]].append(int(r["correct"]))
        print(f"{tag:10s}", " ".join(f"{t}:{sum(v)}/{len(v)}" for t, v in sorted(by.items())))
