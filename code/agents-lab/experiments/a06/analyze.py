"""Tables for the article from runs/main.jsonl (and runs/sonnet.jsonl if present)."""
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(tag):
    """Rows, re-scored with the current checker (requests_set.check)."""
    from requests_set import REQUESTS, check
    by_id = {r["id"]: r for r in REQUESTS}
    p = HERE / "runs" / f"{tag}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    for r in rows:
        r["right_tool"], r["right_args"] = check(by_id[r["id"]], r["called"], r["input"]) if r["called"] else (False, False)
    return rows


def table(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["arm"], r["n"])].append(r)
    print("| model | arm | N | trials | right tool | CI | tool+args | CI | prompt tok | out tok | $/trial | med lat s | target found | no call |")
    for (m, arm, n), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        k, ka, N = sum(r["right_tool"] for r in rs), sum(r["right_args"] for r in rs), len(rs)
        lo, hi = wilson(k, N)
        lo2, hi2 = wilson(ka, N)
        tok = statistics.mean(r["usage"]["input_tokens"] + r["usage"]["cache_read_input_tokens"]
                              + r["usage"]["cache_creation_input_tokens"] for r in rs)
        out = statistics.mean(r["usage"]["output_tokens"] for r in rs)
        cost = statistics.mean(r["usage"]["cost"] for r in rs)
        lat = statistics.median(r["latency"] for r in rs)
        found = f"{sum(r['target_found'] for r in rs)}/{N}" if arm.startswith("search") else (
            f"{sum(r['target'] in r['loaded'] for r in rs)}/{N}" if arm == "client_bm25" else "-")
        nocall = sum(r["called"] is None for r in rs)
        print(f"| {m[7:]} | {arm} | {n} | {N} | {100*k/N:.1f}% | {lo:.1f}-{hi:.1f} | {100*ka/N:.1f}% | {lo2:.1f}-{hi2:.1f} | "
              f"{tok:,.0f} | {out:.0f} | {cost:.5f} | {lat:.2f} | {found} | {nocall} |")
    return cells


def errors(cells, key):
    rs = cells.get(key, [])
    c = Counter((r["id"], r["target"], r["called"]) for r in rs if not r["right_tool"])
    print(f"\nwrong tool, {key}:")
    for (i, t, called), n in c.most_common():
        print(f"  {i:4s} {t:32s} -> {called}  x{n}")
    c = Counter((r["id"], json.dumps(r["input"])) for r in rs if r["right_tool"] and not r["right_args"])
    print(f"right tool wrong args, {key}:")
    for (i, inp), n in c.most_common():
        print(f"  {i:4s} {inp[:150]}  x{n}")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["main", "sonnet"]
    rows = [r for t in tags for r in load(t)]
    cells = table(rows)
    for key in sorted(cells):
        if key[2] >= 100 or "--all-errors" in sys.argv:
            errors(cells, key)
    cost = sum(r["usage"]["cost"] for r in rows)
    print(f"\ntotal cost of these rows ${cost:.4f}, {len(rows)} trials")
