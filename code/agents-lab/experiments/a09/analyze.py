"""Summaries for Part 9: success with Wilson CIs, steps, tokens, cost and latency per arm."""
import json
import math
import sys
from collections import defaultdict

IMG = 1337  # measured tokens per 1280x800 screenshot (1,334 visual tokens + 3)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(path, **filt):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if all(r.get(k, "" if k == "label" else None) == v for k, v in filt.items())]


def summary(rows, img=IMG):
    n, k = len(rows), sum(r["ok"] for r in rows)
    lo, hi = wilson(k, n)
    img_tok = sum(sum(r["images_per_step"]) for r in rows) * img / max(n, 1)
    inp = sum(r["input"] + r["cache_read"] + r["cache_write"] for r in rows) / max(n, 1)
    return dict(n=n, k=k, rate=100 * k / max(n, 1), lo=lo, hi=hi,
                steps=sum(r["steps"] for r in rows) / n, images=sum(r["images"] for r in rows) / n,
                input=inp, img_tok=img_tok, img_share=100 * img_tok / max(inp, 1),
                output=sum(r["output"] for r in rows) / n, cost=sum(r["cost"] for r in rows) / n,
                latency=sum(r["latency"] for r in rows) / n, wall=sum(r["wall"] for r in rows) / n,
                errors=sum(r["errors"] for r in rows) / n, last_ctx=sum(r["context"][-1] for r in rows) / n,
                total_cost=sum(r["cost"] for r in rows))


def fmt(s):
    return (f"{s['k']}/{s['n']} ({s['rate']:.1f}%, {s['lo']:.1f}-{s['hi']:.1f}) steps {s['steps']:.1f} imgs {s['images']:.1f} "
            f"in/run {s['input']:,.0f} (img {s['img_tok']:,.0f} = {s['img_share']:.0f}%) out {s['output']:,.0f} "
            f"last ctx {s['last_ctx']:,.0f} ${s['cost']:.5f}/run model {s['latency']:.1f}s wall {s['wall']:.1f}s err {s['errors']:.2f} "
            f"total ${s['total_cost']:.3f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/amit/.cache/vidyalai-scratch/a09/results.jsonl"
    rows = [json.loads(l) for l in open(path)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["arm"] + r.get("label", ""))].append(r)
    for g, rs in sorted(groups.items()):
        print(g, fmt(summary(rs)))
        for kind in ("info", "action"):
            sub = [r for r in rs if r["kind"] == kind]
            if sub:
                print("    ", kind, fmt(summary(sub)))
    # per-task table
    tasks = sorted({r["task"] for r in rows})
    arms = sorted(groups)
    print("\nper task:", " | ".join(f"{m.split('-')[1]}:{a}" for m, a in arms))
    for t in tasks:
        cells = []
        for g in arms:
            sub = [r for r in groups[g] if r["task"] == t]
            cells.append(f"{sum(r['ok'] for r in sub)}/{len(sub)}" if sub else "-")
        print(t, " | ".join(cells))
