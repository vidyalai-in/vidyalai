"""Where does adaptive thinking happen? Thinking tokens by step position, from the transcripts.

Usage: python interleave.py ARM [ARM ...]
First step = the request carrying the task; middle = requests that only carry tool results and
end in another tool call; last = the request that writes the final answer.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent

for arm in sys.argv[1:]:
    pos = defaultdict(lambda: [0, 0, 0])   # steps, steps with thinking, thinking tokens
    kinds = defaultdict(lambda: [0, 0])
    for p in sorted((HERE / "runs" / "transcripts" / arm).glob("*.jsonl")):
        steps = [json.loads(l) for l in p.read_text().splitlines()]
        steps = [s for s in steps if s["event"] == "step"]
        for i, s in enumerate(steps):
            where = "first" if i == 0 else ("last" if s["stop_reason"] != "tool_use" else "middle")
            t = s.get("thinking_tokens", 0)
            pos[where][0] += 1
            pos[where][1] += bool(t)
            pos[where][2] += t
            kind = p.name.split("-")[0].rstrip("0123456789")
            kinds[kind][0] += t
            kinds[kind][1] += 1
    tot = sum(v[2] for v in pos.values())
    print(f"{arm}:")
    for w in ("first", "middle", "last"):
        n, k, t = pos[w]
        print(f"  {w:6s} steps {n:5d}  with thinking {k:5d} ({100*k/max(n,1):4.0f}%)  "
              f"thinking tokens {t:8d} ({100*t/max(tot,1):4.1f}%)  mean per step {t/max(n,1):6.0f}")
