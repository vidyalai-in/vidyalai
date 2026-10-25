"""Detail numbers for Part 5: clearing behaviour, compaction, memory, caching per step, the 100K line."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze import load, steps  # noqa: E402

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402


def events(tag, task, k):
    p = HERE / "runs" / "transcripts" / tag / f"{task}-r{k}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()]


def gets(st):
    return [c for e in st for c in e["tool_calls"] if c.startswith("get_incident")]


print("== context editing")
rows = load("edit")
notes_before, first_clear, refetch_runs, refetch_calls, total_get = 0, 0, 0, 0, 0
for r in rows:
    st = steps("edit", r["task"], r["repeat"])
    idx = next((i for i, e in enumerate(st) if e["cleared"] > 0), None)
    g = gets(st)
    dup = len(g) - len(set(g))
    total_get += len(g)
    refetch_calls += dup
    refetch_runs += dup > 0
    if idx is not None:
        first_clear += 1
        prev = st[idx - 1]
        if any(c["type"] == "text" and c.get("text") for c in prev["content"]):
            notes_before += 1
    writes = sum(e["cache_write"] for e in st)
    print(f"  {r['task']} r{r['repeat']} ok={r['correct']} stop={r['stop']} steps={r['steps']} gets={len(g)} dup={dup} "
          f"first_clear_step={idx + 1 if idx is not None else None} cache_write={writes:,} ${r['cost']:.4f}")
print(f"runs {len(rows)}; runs with clearing {first_clear}; text note in the step before first clearing {notes_before}; "
      f"runs that re-fetched {refetch_runs}; re-fetched calls {refetch_calls} of {total_get}")

print("\n== compaction")
for r in load("compact"):
    st = steps("compact", r["task"], r["repeat"])
    comp = [e for e in st if e["compacted"]]
    summ = [c["content"] for e in comp for c in e["content"] if c["type"] == "compaction"]
    ccost = sum(ledger.cost_of("claude-haiku-5-5", it) for e in comp for it in e["usage"]["iterations"]
                if it["type"] == "compaction")
    print(f"  {r['task']} r{r['repeat']} ok={r['correct']} compactions={len(comp)} at steps {[e['n'] for e in comp]} "
          f"summary chars {[len(s) for s in summ]} after={[e['context'] for e in comp]} compaction ${ccost:.5f} of ${r['cost']:.4f}")

print("\n== memory")
for r in load("memory"):
    ev = events("memory", r["task"], r["repeat"])
    st = [e for e in ev if e["event"] == "step"]
    mem = [c for e in st for c in e["tool_calls"] if c.startswith("memory")]
    cmds = Counter(json.loads(c[len("memory("):-1])["command"] for c in mem)
    g = gets(st)
    print(f"  {r['task']} r{r['repeat']} ok={r['correct']} windows={r['windows']} steps={r['steps']} memory_calls={len(mem)} "
          f"{dict(cmds)} gets={len(g)} dup={len(g) - len(set(g))} ${r['cost']:.4f}")

print("\n== naive: thinking, 100K line, fan-out caching")
th = out = th_steps = n_steps = 0
for r in load("naive"):
    st = steps("naive", r["task"], r["repeat"])
    for e in st:
        t = e["usage"]["output_tokens_details"]["thinking_tokens"]
        th += t
        out += e["output_tokens"]
        th_steps += t > 0
        n_steps += 1
    if r["size"] >= 48:
        over = [e for e in st if e["context"] > 100_000]
        print(f"  {r['task']}: steps over 100K {len(over)} of {len(st)}, their cost ${sum(e['cost'] for e in over):.4f} "
              f"of ${r['cost']:.4f}; peak {max(e['context'] for e in st):,}")
print(f"  thinking tokens {th:,} of {out:,} output tokens; steps with thinking {th_steps} of {n_steps}")

print("\n== per-step cache hit rate, naive c10 r0")
st = steps("naive", "c10", 0)
print("  " + " ".join(f"{e['n']}:{100 * e['cache_read'] / e['context']:.0f}" for e in st))
for tag in ("edit", "compact", "memory"):
    st = steps(tag, "c10", 0)
    spikes = [(e["n"], e["cache_write"], e["context"]) for e in st if e["cache_write"] > 5000]
    print(f"  {tag} c10 r0 cache-write spikes (step, written, context): {spikes}")

print("\n== chart data: context per step, c10 r0")
chart = {}
for tag in ("naive", "sized", "edit", "compact", "memory"):
    st = steps(tag, "c10", 0)
    chart[tag] = [[e["n"], e["context"], e["compaction_in"]] for e in st]
    print(f"  {tag}: {[c[1] for c in chart[tag]]}")
json.dump(chart, open(HERE / "chart-c10.json", "w"))
