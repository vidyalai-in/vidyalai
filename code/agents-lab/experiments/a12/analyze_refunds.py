"""Experiments 2-4 analysis: one row per arm."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k/n:.0%} ({lo:.0%}-{hi:.0%})"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(p * len(xs))) - 1)]


arms = sys.argv[1:]
for arm in arms:
    f = HERE / "runs" / f"refunds-{arm}.jsonl"
    if not f.exists():
        continue
    rs = [r for r in (json.loads(l) for l in open(f)) if r["seed"] < 900]
    n = len(rs)
    print(f"\n== {arm}: {n} runs, ${sum(r['cost'] for r in rs):.4f} total, ${sum(r['cost'] for r in rs)/n:.5f}/run")
    print("  all 8 exactly once:", pct(sum(r["all_ok"] for r in rs), n))
    print("  runs with a double refund:", pct(sum(r["extra_refunds"] > 0 for r in rs), n),
          " extra refunds total:", sum(r["extra_refunds"] for r in rs), "of", 8 * n, "orders")
    print("  runs with a missed order:", pct(sum(r["missed"] > 0 for r in rs), n), " wrong amount:", sum(r["wrong"] for r in rs))
    print("  report matches backend:", pct(sum(r["report_true"] for r in rs), n))
    print(f"  model calls mean {sum(r['calls'] for r in rs)/n:.1f}, p50 {q([r['calls'] for r in rs], .5)}, "
          f"p95 {q([r['calls'] for r in rs], .95)}, max {max(r['calls'] for r in rs)}; tool calls {sum(r['tool_calls'] for r in rs)/n:.1f}; "
          f"refund calls {sum(r['refund_calls'] for r in rs)/n:.1f}; get_order {sum(r['get_calls'] for r in rs)/n:.1f}")
    print("  stops:", Counter(r["stop"] for r in rs))
    if "fault_log" in rs[0]:
        c = Counter(k for r in rs for (_, _, k) in r["fault_log"])
        print("  faults injected:", dict(c), " per run", {k: round(v / n, 2) for k, v in c.items()})
    if "sim_sleep_s" in rs[0]:
        print(f"  simulated backoff per run: {sum(r['sim_sleep_s'] for r in rs)/n:.2f} s")
    if "crashed" in rs[0]:
        print("  crashed:", sum(r["crashed"] for r in rs), " resumed from step (mean):",
              round(sum(r["steps_resumed_from"] for r in rs) / n, 2),
              f" cost before crash ${sum(r['cost_before_crash'] for r in rs)/n:.5f}")
    if "stuck_id" in rs[0]:
        att = [r["stuck_attempts"] for r in rs]
        print("  stuck order refunded:", pct(sum(r["stuck_ok"] for r in rs), n),
              " attempts on it: mean", round(sum(att) / n, 1), "min", min(att), "max", max(att), Counter(att))
        print("  stuck reported failed:", pct(sum(r["stuck_reported_failed"] for r in rs), n),
              " other 7 all ok:", pct(sum(r["ok"] - (1 if r["stuck_ok"] else 0) == 7 for r in rs), n))
