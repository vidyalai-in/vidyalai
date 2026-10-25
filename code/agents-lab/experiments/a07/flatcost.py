"""Re-price every step of the full-record arms at Haiku 5.5's base rates (no over-100K premium)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from ledger import PRICES  # noqa: E402

HERE = Path(__file__).parent
for arm in sys.argv[1:] or ["single-full", "team-full"]:
    tot = defaultdict(lambda: [0.0, 0.0, 0])
    long_steps = steps = 0
    for d in sorted((HERE / "runs" / "transcripts" / arm).iterdir()):
        if d.name.endswith("r9"):
            continue
        kind = {"w": "wide", "d": "twostage", "c": "chain", "s": "wide12"}[d.name[0]]
        for f in d.glob("*.jsonl"):
            for l in f.read_text().splitlines():
                e = json.loads(l)
                if e.get("event") != "step":
                    continue
                m = e.get("model", "claude-haiku-5-5")
                p = PRICES[m]
                ctx = e["input_tokens"] + e["cache_read"] + e["cache_write"]
                flat = (e["input_tokens"] * p["in"] + e["cache_read"] * p["cache_read"] +
                        e["cache_write"] * p["cache_write"] + e["output_tokens"] * p["out"]) / 1e6
                tot[kind][0] += e["cost"]
                tot[kind][1] += flat
                steps += 1
                long_steps += m == "claude-haiku-5-5" and ctx > 100_000
        for k in tot:
            pass
    pre = {"wide": "w", "twostage": "d", "chain": "c", "wide12": "s"}
    n = {k: len([d for d in (HERE / "runs" / "transcripts" / arm).iterdir() if d.name[0] == pre[k] and not d.name.endswith("r9")]) for k in tot}
    print(arm, f"steps over 100K: {long_steps}/{steps}")
    for k, (a, f) in ((k, v[:2]) for k, v in tot.items()):
        print(f"  {k:9s} runs {n[k]:2d} actual ${a/n[k]:.4f}/task  flat-rate ${f/n[k]:.4f}/task")
